"""Model discovery shared by the independent UI and API."""
from pathlib import Path
import os

from model_loading import CHECKPOINT_EXTENSIONS, find_checkpoint, model_folders


def models_root():
    if os.environ.get('KALOSCOPE_MODELS_DIR'):
        return Path(os.environ['KALOSCOPE_MODELS_DIR'])
    try:
        from modules import paths
        return Path(paths.models_path)
    except (ImportError, AttributeError):
        return Path(__file__).resolve().parents[1] / 'models'


def get_available_models():
    return sorted(model_folders(models_root()))


def get_model_dir(model_name):
    folders = model_folders(models_root())
    if model_name not in folders:
        raise FileNotFoundError(f"Model folder not found: {model_name}")
    return folders[model_name]


def get_available_checkpoints(model_name):
    try:
        directory = get_model_dir(model_name)
    except FileNotFoundError:
        return []
    return [p.name for p in sorted(directory.iterdir()) if p.suffix.lower() in CHECKPOINT_EXTENSIONS]


def get_available_csv(model_name):
    return [p.name for p in sorted(get_model_dir(model_name).glob("*.csv"))]


def get_checkpoint_path(model_name, checkpoint_name=None):
    directory = get_model_dir(model_name)
    return str(directory / checkpoint_name) if checkpoint_name else str(find_checkpoint(directory))


def get_class_csv(model_name):
    path = get_model_dir(model_name) / "class_mapping.csv"
    return str(path) if path.is_file() else None
