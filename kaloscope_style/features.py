"""Reference preprocessing exactly matching the Style full-data training crop."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def normalized(features):
    features = torch.as_tensor(features).detach().float().cpu()
    if features.ndim == 1:
        features = features[None]
    if features.ndim != 2 or features.shape[1] != 256 or not features.shape[0]:
        raise ValueError('Style requires [N,256] artist projector features, not backbone or patch tokens.')
    if not torch.isfinite(features).all() or (features.norm(dim=1) < 1e-8).any():
        raise ValueError('Style features must be finite and nonzero.')
    return F.normalize(features, dim=1)


def blend(features, weights=''):
    features = normalized(features)
    if weights.strip():
        try:
            values = torch.tensor([float(v.strip()) for v in weights.split(',')])
        except ValueError as exc:
            raise ValueError('Weights must be comma-separated numbers.') from exc
        if len(values) != len(features):
            raise ValueError('Supply one weight for each reference image/feature row.')
    else:
        values = torch.ones(len(features))
    if not torch.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, and have a positive sum.')
    return normalized((features * (values / values.sum())[:, None]).sum(0))


def preprocess(image):
    array = (image.detach().float().cpu().clamp(0, 1) * 255).round().byte().numpy()
    im = Image.fromarray(array, mode='RGB')
    w, h = im.size
    n = min(w, h)
    im = im.crop(((w-n)//2, (h-n)//2, (w+n)//2, (h+n)//2)).resize((512, 512), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
    return (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


@torch.inference_mode()
def encode(images, bundle):
    model = bundle['model']
    if not hasattr(model, 'extract_tensor') or getattr(model, 'projector', None) is None or model.projector[-1].out_features != 256:
        raise ValueError('Load the training-matched DINOv3 artist encoder with a 256-dimensional projector.')
    if images.ndim != 4 or images.shape[-1] != 3 or not len(images) or not torch.isfinite(images).all():
        raise ValueError('Reference images must be finite RGB IMAGE tensors [N,H,W,3].')
    if bundle.get('patcher') is not None:
        from comfy import model_management
        model_management.load_models_gpu([bundle['patcher']])
    device = next(model.parameters()).device
    result = []
    for image in images:
        pixels = preprocess(image)[None].to(device)
        # Training used BF16 autocast; keep the model's stored weights in FP32.
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)
        with context:
            result.append(model.extract_tensor(pixels, 'projector').float().cpu())
    return normalized(torch.cat(result))
