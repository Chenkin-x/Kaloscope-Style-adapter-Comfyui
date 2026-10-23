"""Public model choices backed by existing files; never copy or rename weights."""
from pathlib import Path
from .compat import LEGACY_ADAPTER_DIRECTORY, public_asset_name


def _choices(roots, folders=False):
    choices, ranks = {}, {}
    for priority, root in enumerate(roots):
        if not root.is_dir():
            continue
        candidates = sorted(root.iterdir()) if folders else sorted(root.rglob('*'))
        for path in candidates:
            if folders:
                if not path.is_dir(): continue
            elif not path.is_file() or path.suffix.lower() not in ('.pt', '.pth', '.safetensors'):
                continue
            relative = path.relative_to(root).as_posix()
            name = public_asset_name(relative)
            rank = (priority, name != relative)
            if name not in choices or rank < ranks[name]:
                choices[name], ranks[name] = path, rank
    return dict(sorted(choices.items()))


def adapter_paths(models_dir):
    root = Path(models_dir)
    return _choices([root / 'kaloscope_adapters', root / LEGACY_ADAPTER_DIRECTORY])


def bank_paths(models_dir):
    return _choices([Path(models_dir) / 'kaloscope_style_banks'])


def encoder_paths(models_dir):
    return _choices([Path(models_dir) / 'kaloscope'], folders=True)
