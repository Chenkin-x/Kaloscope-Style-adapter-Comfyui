"""Shared feature workflow for ComfyUI, WebUI, standalone UI, CLI and API."""
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from feature_analysis import analyze_features, prepare_features
from model_loading import FEATURE_OUTPUTS


@torch.inference_mode()
def extract_tensor_batch(encoder, batch, output_type='default', layers='-1', intermediate_norm=True):
    if hasattr(encoder, 'extract_tensor'):
        return encoder.extract_tensor(batch, output_type, layers, intermediate_norm)
    if output_type in ('default', 'backbone'):
        return encoder(batch, return_features=True)
    raise ValueError(f'{output_type} requires a DINOv3 model; LSNet supports default/backbone')


@torch.inference_mode()
def extract_batch(images, bundle, output_type='default', layers='-1', intermediate_norm=True, batch_size=2):
    if output_type not in FEATURE_OUTPUTS:
        raise ValueError(f'Unknown feature output: {output_type}')
    if not images or len(images) > 512 or batch_size < 1:
        raise ValueError('Provide 1–512 images and a positive batch size')
    encoder = bundle['model']
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([bundle['transform'](image) for image in images[start:start + batch_size]]).to(bundle['device'])
        result = extract_tensor_batch(encoder, batch, output_type, layers, intermediate_norm)
        outputs.append(result.float().cpu().contiguous())
    return torch.cat(outputs)


def output_layout(output_type):
    if output_type == 'patch_map':
        return 'spatial'
    if output_type.startswith('intermediate_'):
        return 'layer_spatial' if output_type.endswith('patch_map') else (
            'layer_vectors' if output_type in ('intermediate_cls', 'intermediate_mean', 'intermediate_cls_mean') else 'layer_tokens')
    return 'auto'


def thumbnail_batch(images):
    return torch.stack([torch.from_numpy(np.array(ImageOps.pad(image.convert('RGB'), (240, 200), color='white'))).float() / 255
                        for image in images])


def cache_bytes(features, labels, output_type='default', layers='-1'):
    stream = io.BytesIO()
    np.savez_compressed(stream, features=features.detach().float().cpu().numpy(),
                        labels=np.asarray(labels, dtype=str), output_type=np.asarray(output_type), layers=np.asarray(layers))
    return stream.getvalue()


def read_cache(source):
    """Only numeric arrays and string metadata; never unpickle uploaded files."""
    with np.load(source, allow_pickle=False) as data:
        array = data['features']
        if array.dtype.kind not in 'fiu' or array.ndim < 2 or not 1 <= array.shape[0] <= 512 or not np.isfinite(array).all():
            raise ValueError('Invalid feature cache')
        labels = data['labels'].tolist()
        if not isinstance(labels, list) or len(labels) != array.shape[0]:
            raise ValueError('Cache labels must match features')
        output_type = str(data['output_type'])
        if output_type not in FEATURE_OUTPUTS:
            raise ValueError('Unknown cached feature output')
        return {'features': torch.from_numpy(array.astype(np.float32)), 'labels': labels,
                'output_type': output_type, 'layers': str(data['layers'])}


def analyze_cached(cached, chart_type='relationship_graph', images=None, **options):
    if options.get('tensor_layout', 'auto') == 'auto':
        options['tensor_layout'] = output_layout(cached['output_type'])
    if not options.get('labels'):
        options['labels'] = cached['labels']
    return analyze_features(cached['features'], chart_type=chart_type, images=images, **options)


def feature_tools(features, operation='common_features', reference_index=0, groups=None, **layout_options):
    vectors = prepare_features(features, **layout_options)[0]
    if operation == 'common_features':
        return {'common_features': vectors.mean(0).tolist(), 'sample_count': len(vectors)}
    if not 0 <= reference_index < len(vectors):
        raise ValueError('reference_index is outside the batch')
    targets = vectors if operation == 'similarity' else None
    if operation == 'compare_groups':
        if groups is None or len(groups) != len(vectors):
            raise ValueError('Provide one group name per image')
        names = sorted(set(str(group) for group in groups))
        targets = np.stack([vectors[np.asarray([str(group) == name for group in groups])].mean(0) for name in names])
    elif operation != 'similarity':
        raise ValueError(f'Unknown feature operation: {operation}')
    query = vectors[reference_index]
    norms = np.linalg.norm(targets, axis=1) * np.linalg.norm(query)
    if np.any(norms <= 1e-12):
        raise ValueError('Cosine similarity requires nonzero vectors')
    scores = (targets @ query / norms).clip(-1, 1)
    result = {'reference_index': reference_index, 'similarities': scores.tolist()}
    if operation == 'compare_groups':
        result.update(groups=names, best_group=names[int(np.argmax(scores))])
    return result


def save_analysis(directory, chart_type, image, report, distances):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png, json_path, csv_path = [directory / f'{chart_type}.{extension}' for extension in ('png', 'json', 'csv')]
    Image.fromarray((image[0].numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(png)
    json_path.write_text(report, encoding='utf-8')
    labels = json.loads(report)['labels']
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', *labels])
        for label, row in zip(labels, distances.tolist()):
            writer.writerow([label, *row])
    return [str(path) for path in (png, json_path, csv_path)]
