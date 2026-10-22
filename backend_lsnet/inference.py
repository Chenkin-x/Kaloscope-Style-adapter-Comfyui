"""Inference adapter shared by the standalone UI and API."""
from PIL import Image
from model_loading import load_model_bundle
from inference_artist import classify_image, resolved_mode
from backend_lsnet.analysis import extract_batch


def process_image_from_pil(image, model=None, checkpoint='', num_classes=None, feature_dim=None,
                           mode='auto', class_csv=None, device='cuda', top_k=5, threshold=0.0,
                           output_type='default', layers='-1', intermediate_norm=True):
    bundle = load_model_bundle(checkpoint=checkpoint, model_name=model, device=device, class_csv=class_csv)
    model_obj = bundle['model']
    mode = resolved_mode(model_obj, mode)
    tensor = bundle['transform'](image).unsqueeze(0)
    result = {}
    if mode in ('classify', 'both'):
        result['classification'] = classify_image(model_obj, tensor, device, bundle['class_mapping'], top_k, threshold)
    if mode in ('cluster', 'both'):
        result['features'] = extract_batch([image], bundle, output_type, layers, intermediate_norm)[0].tolist()
    return result


def process_image(image_path, **kwargs):
    with Image.open(image_path) as image:
        return process_image_from_pil(image, **kwargs)
