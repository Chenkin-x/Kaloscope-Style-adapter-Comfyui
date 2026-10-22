"""Standalone feature extraction, all charts and feature tools; no ComfyUI required."""
import argparse
import json
from pathlib import Path

from PIL import Image

from backend_lsnet.analysis import (extract_batch, cache_bytes, read_cache, thumbnail_batch,
                                   analyze_cached, save_analysis, feature_tools, output_layout)
from backend_lsnet.analysis_api import AnalysisOptions, values
from feature_analysis import CHART_TYPES, prepare_features
from model_loading import FEATURE_OUTPUTS, load_model_bundle


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='Image file or directory (extract once)')
    source.add_argument('--features', type=Path, help='features.npz cache (no model loaded)')
    result.add_argument('--model-dir', type=Path)
    result.add_argument('--device', default='cuda')
    result.add_argument('--output-type', choices=FEATURE_OUTPUTS, default='default')
    result.add_argument('--layers', default='-1')
    result.add_argument('--no-intermediate-norm', action='store_true')
    result.add_argument('--batch-size', type=int, default=2)
    result.add_argument('--output', type=Path, default=Path('outputs'))
    result.add_argument('--chart-type', choices=CHART_TYPES, default='relationship_graph')
    result.add_argument('--all-charts', action='store_true', help='Requires patch tokens/map for patch_energy')
    result.add_argument('--operation', choices=['charts', 'common_features', 'similarity', 'compare_groups'], default='charts')
    result.add_argument('--groups', type=Path, help='UTF-8 text, one group name per image')
    result.add_argument('--options-json', type=Path, help='JSON object with any analysis parameter')
    defaults = values(AnalysisOptions())
    for name, default in defaults.items():
        flag = '--' + name.replace('_', '-')
        if name == 'labels':
            result.add_argument(flag, default=argparse.SUPPRESS, help='One name per line or JSON array')
        elif isinstance(default, bool):
            result.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            result.add_argument(flag, type=type(default), default=argparse.SUPPRESS)
    return result


def main(args):
    options = values(AnalysisOptions())
    if args.options_json:
        supplied = json.loads(args.options_json.read_text(encoding='utf-8'))
        if not isinstance(supplied, dict) or set(supplied) - set(options):
            raise ValueError('options-json must be an object of known analysis options')
        options.update(supplied)
    options.update({name: value for name, value in vars(args).items() if name in options})
    images = None
    if args.features:
        cached = read_cache(args.features)
        print('Reusing cached features: no model loading or inference', flush=True)
    else:
        if not args.model_dir:
            raise ValueError('--input requires --model-dir')
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        paths = [args.input] if args.input.is_file() else sorted(p for p in args.input.rglob('*') if p.suffix.lower() in extensions)
        if not 1 <= len(paths) <= 512:
            raise ValueError('Input must contain 1–512 images')
        pictures = []
        for path in paths:
            with Image.open(path) as image:
                pictures.append(image.copy())
        bundle = load_model_bundle(args.model_dir, device=args.device)
        features = extract_batch(pictures, bundle, args.output_type, args.layers, not args.no_intermediate_norm, args.batch_size)
        cached = {'features': features, 'labels': [str(path.relative_to(args.input)) if args.input.is_dir() else path.name for path in paths],
                  'output_type': args.output_type, 'layers': args.layers}
        images = thumbnail_batch(pictures)
        del bundle
        print(f'Extracted {len(paths)} images once: {list(features.shape)}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'features.npz').write_bytes(cache_bytes(cached['features'], cached['labels'], cached['output_type'], cached['layers']))
    if args.operation != 'charts':
        layout = output_layout(cached['output_type']) if options['tensor_layout'] == 'auto' else options['tensor_layout']
        groups = args.groups.read_text(encoding='utf-8').splitlines() if args.groups else None
        report = feature_tools(cached['features'], args.operation, options['reference_index'], groups,
                               tensor_layout=layout, layer_index=options['layer_index'], layer_pooling=options['layer_pooling'], token_pooling=options['token_pooling'])
        (args.output / f'{args.operation}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return
    charts = CHART_TYPES if args.all_charts else [args.chart_type]
    if args.all_charts:
        layout = output_layout(cached['output_type']) if options['tensor_layout'] == 'auto' else options['tensor_layout']
        patches = prepare_features(cached['features'], layout, options['layer_index'], options['layer_pooling'], options['token_pooling'])[1]
        if patches is None or cached['output_type'] not in ('patch_tokens', 'patch_map', 'intermediate_patch_tokens', 'intermediate_patch_map'):
            raise ValueError('--all-charts includes patch_energy; extract patch_tokens or patch_map first')
    for chart in charts:
        image, report, distances = analyze_cached(cached, chart, images=images, **options)
        files = save_analysis(args.output, chart, image, report, distances)
        print(files[0], flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
