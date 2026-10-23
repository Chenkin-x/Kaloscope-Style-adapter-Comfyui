"""Offline artist-v1 projection only. Reuses comfyui-kaloscope's folders and files."""
import json
from pathlib import Path
import torch
from torch import nn

EXTENSIONS = ('.pt', '.pth', '.ckpt', '.safetensors', '.bin')


def normalized_key(key):
    while key.startswith(('module.', '_orig_mod.')):
        key = key.split('.', 1)[1]
    return key


def model_folders(models_dir):
    from .paths import encoder_paths
    return encoder_paths(models_dir)


def read_config(directory):
    p = Path(directory) / 'config.json'
    data = json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    if not isinstance(data, dict):
        raise ValueError('config.json must be an object.')
    return data


def find_checkpoint(directory):
    directory = Path(directory)
    config = read_config(directory)
    if config.get('checkpoint'):
        p = directory / config['checkpoint']
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for name in ('best.pt', 'best_checkpoint.pth', 'model.safetensors', 'pytorch_model.bin'):
        if (directory / name).is_file():
            return directory / name
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(files) != 1:
        raise ValueError('Set checkpoint in config.json to select one model file.')
    return files[0]


def state_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a tensor state dictionary.')
    for key in ('model', 'state_dict', 'model_ema', 'teacher', 'student'):
        if isinstance(payload.get(key), dict):
            return state_dict(payload[key])
    result = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = normalized_key(key)
        if key in result:
            raise ValueError(f'Duplicate tensor after key normalization: {key}')
        result[key] = value
    if not result:
        raise ValueError('No model tensors found.')
    return result


def build_backbone():
    # Exact artist-v1 ViT-B/16 factory settings. No hub, downloads or classifier code.
    from ._vendor.dinov3.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer(img_size=224, patch_size=16, in_chans=3,
        pos_embed_rope_base=100, pos_embed_rope_normalize_coords='separate',
        pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype='fp32',
        embed_dim=768, depth=12, num_heads=12, ffn_ratio=4,
        qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-5,
        norm_layer='layernormbf16', ffn_layer='mlp', ffn_bias=True,
        proj_bias=True, n_storage_tokens=4, mask_k_bias=True)


class StyleEncoder(nn.Module):
    def __init__(self, backbone, hidden=1536):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(nn.Linear(2 * backbone.embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 256))
        self.feature_dim, self.feature_source = 256, 'projector'
        self.has_classifier = False

    def extract_tensor(self, images, output_type='projector'):
        if output_type not in ('default', 'projector'):
            raise ValueError('The adapter encoder only exposes 256D style projections.')
        tokens = self.backbone.forward_features(images)
        pooled = torch.cat((tokens['x_norm_clstoken'], tokens['x_norm_patchtokens'].mean(1)), dim=1)
        return self.projector(pooled)

    def forward(self, images, return_features=True):
        return self.extract_tensor(images)

    def get_dtype(self):
        return next(self.parameters()).dtype


def load_encoder(directory):
    directory = Path(directory)
    config = read_config(directory)
    checkpoint = find_checkpoint(directory)
    if checkpoint.suffix.lower() == '.safetensors':
        from safetensors import safe_open
        # A classifier file may be shared; its large head need not enter memory.
        with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
            payload = {k: handle.get_tensor(k) for k in handle.keys()
                       if not normalized_key(k).startswith(('head.', 'linear_head.'))}
    else:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state_dict(payload)
    embedded = payload.get('model_config', {}) if isinstance(payload, dict) else {}
    embedded = embedded if isinstance(embedded, dict) else {}
    selection = config.get('model', embedded)
    spec = ({**embedded, **{k: config[k] for k in ('kwargs', 'pooling') if k in config}, 'name': selection}
            if isinstance(selection, str) else {**embedded, **dict(selection or {})})
    if embedded.get('name') and embedded['name'] != spec.get('name'):
        raise ValueError('config.json architecture differs from checkpoint model_config.')
    if spec.get('name') != 'dinov3_vitb16':
        raise ValueError('Style requires the matched Kaloscope artist-v1 DINOv3 ViT-B/16 with a 256D projector; LSNet and other backbones are not style encoders.')
    if spec.get('pooling', config.get('pooling', 'cls_mean')) != 'cls_mean' or spec.get('kwargs'):
        raise ValueError('Style requires the standard artist-v1 cls_mean architecture without backbone overrides.')
    if config.get('input_size', 512) != 512 or spec.get('projection_dim', 256) != 256:
        raise ValueError('Style artist-v1 requires input_size=512 and projection_dim=256.')
    backbone = build_backbone()
    backbone_state = {k.removeprefix('backbone.'): v for k, v in state.items() if k.startswith('backbone.')}
    if not backbone_state:
        backbone_state = {k: v for k, v in state.items()
                          if not k.startswith(('projector.', 'head.', 'linear_head.')) and k not in ('bias', 'log_temperature')}
    backbone.load_state_dict(backbone_state, strict=True)
    projection = {k.removeprefix('projector.'): v for k, v in state.items() if k.startswith('projector.')}
    if '0.weight' not in projection or '2.weight' not in projection or projection['2.weight'].shape[0] != 256:
        raise ValueError('This model has no compatible 256D projector. Choose the shared artist-v1 checkpoint, not a classifier-only file.')
    model = StyleEncoder(backbone, projection['0.weight'].shape[0])
    model.projector.load_state_dict(projection, strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite style encoder weights.')
    model.eval().requires_grad_(False)
    return {'model': model, 'device': torch.device('cpu'), 'feature_source': 'projector',
            'feature_dim': 256, 'has_classifier': False, 'checkpoint': str(checkpoint)}


class KaloscopeStyleEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {'model_folder': (list(model_folders(folder_paths.models_dir)),),
                             'device': (['auto', 'cpu', 'cuda'], {'default': 'auto'})}}

    RETURN_TYPES = ('KALOSCOPE_MODEL',)
    RETURN_NAMES = ('style_encoder',)
    FUNCTION = 'load'
    CATEGORY = 'Kaloscope Adapter'

    @classmethod
    def IS_CHANGED(cls, model_folder, device):
        import folder_paths
        directory = model_folders(folder_paths.models_dir)[model_folder]
        files = [find_checkpoint(directory), directory / 'config.json']
        return (device, *((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()))

    def load(self, model_folder, device='auto'):
        import folder_paths
        from comfy import model_management
        from comfy.model_patcher import ModelPatcher
        directory = model_folders(folder_paths.models_dir).get(model_folder)
        if directory is None:
            raise FileNotFoundError('Select the existing encoder folder under ComfyUI/models/kaloscope/.')
        bundle = load_encoder(directory)
        target = model_management.get_torch_device() if device == 'auto' else torch.device(device)
        bundle['patcher'] = ModelPatcher(bundle['model'], load_device=target, offload_device=torch.device('cpu'))
        bundle['device'] = target
        return (bundle,)
