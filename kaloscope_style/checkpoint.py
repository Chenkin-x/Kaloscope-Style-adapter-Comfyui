"""Safe tensor-only loading and portable export of intermediate Style checkpoints."""
import json
from pathlib import Path
import torch
from .adapter import StyleAdapter

from .compat import ADAPTER_FORMAT as FORMAT


def read_checkpoint(path):
    path = Path(path)
    if path.suffix.lower() == '.safetensors':
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            meta = handle.metadata() or {}
        if meta.get('format') != FORMAT:
            raise ValueError('Not a Kaloscope Style adapter safetensors file.')
        return load_file(str(path)), json.loads(meta['config']), int(meta.get('step', -1))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not {'trainable', 'config'} <= payload.keys():
        raise ValueError('Expected Style checkpoint with trainable/config; this is not a LoRA or DINO checkpoint.')
    return payload['trainable'], payload['config'], int(payload.get('step', -1))


def load_adapter(path, operations=torch.nn):
    state, config, step = read_checkpoint(path)
    config = dict(config)
    backend = 'moss' if config.get('model') in ('MOSS-Image-Base', 'MOSS-Image-SFT') else 'krea2'
    hidden, depth = (3072, 42) if backend == 'moss' else (6144, 28)
    required = dict(feature_dim=256, adapter_width=256, style_tokens=8)
    if config.get('lora_rank', 0 if backend == 'moss' else None) != 0:
        raise ValueError('Only Style checkpoints without LoRA are supported.')
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'Unsupported Style architecture: {key}={config.get(key)!r}, expected {expected}.')
    banks = config.get('adapter_banks')
    allowed_banks = (4, 8) if backend == 'moss' else (4,)
    if type(banks) is not int or banks not in allowed_banks:
        raise ValueError(f'Unsupported {backend} shared banks: {banks!r}; expected one of {allowed_banks}.')
    indices = config.get('injection_indices', list(range(depth)) if backend == 'moss' else None)
    slots = config.get('injection_slots', [i * banks // depth for i in range(depth)] if backend == 'moss' else None)
    if indices != list(range(depth)) or slots != [i * banks // depth for i in range(depth)]:
        raise ValueError(f'Expected Style injection into all {depth} {backend} blocks with {banks} shared banks.')
    config.update(injection_indices=indices, injection_slots=slots)
    if any(not key.startswith('adapter.') for key in state):
        raise ValueError('Checkpoint contains unsupported non-adapter trainable weights.')
    module = StyleAdapter(layers=banks, hidden=hidden, operations=operations)
    module.load_state_dict({key.removeprefix('adapter.'): value for key, value in state.items()}, strict=True)
    if not all(torch.isfinite(p).all() for p in module.parameters()):
        raise ValueError('Checkpoint contains nonfinite adapter weights.')
    return dict(adapter=module.eval().requires_grad_(False), config=config, backend=backend, step=step, path=str(path))


def export_checkpoint(source, destination):
    from safetensors.torch import save_file
    bundle = load_adapter(source)
    state = {'adapter.' + k: v.detach().cpu().contiguous() for k, v in bundle['adapter'].state_dict().items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    save_file(state, str(temporary), metadata={'format': FORMAT, 'config': json.dumps(bundle['config']),
                                              'step': str(bundle['step'])})
    temporary.replace(destination)
    return bundle['step']
