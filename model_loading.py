"""Shared offline model loading for ComfyUI, CLI, WebUI and API."""
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn

LSNET_MODELS = (
    "lsnet_t_artist", "lsnet_s_artist", "lsnet_b_artist", "lsnet_l_artist",
    "lsnet_xl_artist", "lsnet_xl_artist_448",
)
CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt", ".safetensors")
FEATURE_OUTPUTS = (
    "default", "backbone", "cls", "mean", "cls_mean", "projector",
    "patch_tokens", "patch_map", "storage_tokens", "all_tokens", "prenorm",
    "intermediate_cls", "intermediate_mean", "intermediate_cls_mean",
    "intermediate_patch_tokens", "intermediate_patch_map", "intermediate_storage_tokens",
    "intermediate_all_tokens", "intermediate_prenorm",
)


def _selected_layers(text, count):
    try:
        requested = [int(item.strip()) for item in text.split(",")]
    except (ValueError, AttributeError):
        raise ValueError("layers must be comma-separated indices, e.g. -1 or 8,9,10,11") from None
    indices = [index + count if index < 0 else index for index in requested]
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f"Layer index out of range: model has {count} layers/stages")
    if len(set(indices)) != len(indices):
        raise ValueError("Layer indices must not repeat")
    return indices


def read_model_config(model_dir):
    path = Path(model_dir) / "config.json"
    if not path.exists():
        return {}
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return config


def find_checkpoint(model_dir):
    directory = Path(model_dir)
    config = read_model_config(directory)
    if config.get("checkpoint"):
        path = directory / config["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for name in ("best.pt", "best_checkpoint.pth", "model.safetensors", "pytorch_model.bin"):
        if (directory / name).is_file():
            return directory / name
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in CHECKPOINT_EXTENSIONS)
    if len(paths) != 1:
        raise ValueError(f"Expected one checkpoint in {directory}; set 'checkpoint' in config.json to select one")
    return paths[0]


def model_folders(models_dir):
    """Discover model subfolders under models/kaloscope."""
    root = Path(models_dir) / 'kaloscope'
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir()} if root.is_dir() else {}


def load_checkpoint_payload(path):
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    # Local training checkpoints also contain optimizer/RNG state (including numpy).
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_state_dict_keys(state):
    result = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        result[key] = value
    return result


def checkpoint_state(payload):
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "model_ema", "teacher", "student"):
        if isinstance(payload.get(key), dict):
            return checkpoint_state(payload[key])
    state = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    if not state:
        raise ValueError("Checkpoint contains no model tensors")
    return normalize_state_dict_keys(state)


def load_checkpoint_state(path):
    return checkpoint_state(load_checkpoint_payload(path))


def load_class_mapping(path):
    if not path:
        return None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"class_id", "class_name"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain class_id and class_name columns")
        mapping = {int(row["class_id"]): row["class_name"] for row in reader}
    if not mapping:
        raise ValueError("Class mapping is empty")
    return mapping


class DinoInferenceModel(nn.Module):
    """Expose the same return_features interface as LSNet without random heads."""
    def __init__(self, backbone, pooling, head=None, projector=None, feature_source="backbone",
                 classifier_input_normalization="none"):
        super().__init__()
        if classifier_input_normalization not in ("none", "l2_sqrt_dim"):
            raise ValueError("classifier_input_normalization must be none or l2_sqrt_dim")
        self.classifier_input_normalization = classifier_input_normalization
        self.backbone = backbone
        self.pooling = pooling
        self.head = head
        self.projector = projector
        self.feature_source = feature_source
        self.has_classifier = head is not None
        self.pooled_dim = backbone.embed_dim * (2 if pooling == "cls_mean" else 1)
        self.feature_dim = projector[-1].out_features if feature_source == "projector" else self.pooled_dim

    @staticmethod
    def _pool(cls, patches, pooling):
        if pooling == "cls":
            return cls
        if pooling == "mean":
            return patches.mean(1)
        return torch.cat((cls, patches.mean(1)), dim=1)

    def _intermediate(self, images, output_type, layers, norm):
        indices = _selected_layers(layers, self.backbone.n_blocks)
        kind = output_type.removeprefix("intermediate_")
        is_vit = hasattr(self.backbone, "blocks")
        if kind == "storage_tokens" and not self.backbone.n_storage_tokens:
            raise ValueError("This architecture has no storage/register tokens")
        kwargs = dict(n=sorted(indices), reshape=kind == "patch_map", return_class_token=True,
                      norm=False if kind == "prenorm" else norm)
        if is_vit:
            kwargs["return_extra_tokens"] = True
        outputs = self.backbone.get_intermediate_layers(images, **kwargs)
        selected = {}
        for index, result in zip(sorted(indices), outputs):
            patches, cls = result[:2]
            storage = result[2] if is_vit else cls.new_empty(cls.shape[0], 0, cls.shape[-1])
            if kind in ("cls", "mean", "cls_mean"):
                tensor = self._pool(cls, patches, kind)
            elif kind in ("patch_tokens", "patch_map"):
                tensor = patches
            elif kind == "storage_tokens":
                tensor = storage
            elif kind == "all_tokens" or (kind == "prenorm" and is_vit):
                tensor = torch.cat((cls.unsqueeze(1), storage, patches), dim=1)
            elif kind == "prenorm":
                tensor = patches
            else:
                raise ValueError(f"Unsupported intermediate output: {kind}")
            selected[index] = tensor
        if len({tuple(value.shape) for value in selected.values()}) != 1:
            raise ValueError("Selected stages have different tensor shapes; select one ConvNeXt stage at a time")
        return torch.stack([selected[index] for index in indices], dim=1)

    @torch.inference_mode()
    def extract_tensor(self, images, output_type="default", layers="-1", intermediate_norm=True):
        if output_type not in FEATURE_OUTPUTS:
            raise ValueError(f"Unsupported feature output: {output_type}")
        if output_type.startswith("intermediate_"):
            return self._intermediate(images, output_type, layers, intermediate_norm)
        if output_type == "patch_map":
            return self._intermediate(images, "intermediate_patch_map", "-1", True)[:, 0]
        tokens = self.backbone.forward_features(images)
        cls, patches = tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"]
        if output_type in ("cls", "mean", "cls_mean"):
            return self._pool(cls, patches, output_type)
        if output_type == "patch_tokens":
            return patches
        if output_type == "storage_tokens":
            if not self.backbone.n_storage_tokens:
                raise ValueError("This architecture has no storage/register tokens")
            return tokens["x_storage_tokens"]
        if output_type == "all_tokens":
            return torch.cat((cls.unsqueeze(1), tokens["x_storage_tokens"], patches), dim=1)
        if output_type == "prenorm":
            return tokens["x_prenorm"]
        pooled = self._pool(cls, patches, self.pooling)
        if output_type == "projector" or (output_type == "default" and self.feature_source == "projector"):
            if self.projector is None:
                raise ValueError("This checkpoint has no supported projector")
            return self.projector(pooled)
        return pooled

    def forward(self, images, return_features=False, return_both=False):
        tokens = self.backbone.forward_features(images)
        cls = tokens["x_norm_clstoken"]
        patches = tokens["x_norm_patchtokens"]
        features = self._pool(cls, patches, self.pooling)
        output = self.projector(features) if self.feature_source == "projector" else features
        if return_features:
            return output
        if self.head is None:
            raise ValueError("This checkpoint has no classification head; use feature extraction or similarity")
        head_input = features
        if self.classifier_input_normalization == "l2_sqrt_dim":
            # Match frozen-head training: normalize in fp32, then allow the
            # linear layer to follow the caller's autocast policy.
            head_input = torch.nn.functional.normalize(features.float(), dim=-1) * math.sqrt(self.pooled_dim)
            head_input = head_input.to(dtype=self.head.weight.dtype)
        logits = self.head(head_input)
        return (output, logits) if return_both else logits


def _load_dino(name, state, model_config, feature_source=None):
    from kaloscope_dinov3.architecture import build_backbone
    # Never load the training machine's weights path or download pretrained weights.
    backbone = build_backbone({**model_config, "name": name})
    prefixed = any(k.startswith("backbone.") for k in state)
    backbone_state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")} if prefixed else {
        k: v for k, v in state.items()
        if not k.startswith(("head.", "linear_head.", "projector.")) and k not in ("log_temperature", "bias")
    }
    backbone.load_state_dict(backbone_state, strict=True)
    head_state = None
    for prefix in ("head.", "linear_head."):
        if prefix + "weight" in state:
            head_state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
            break
    pooling = model_config.get("pooling")
    inferred_dim = head_state["weight"].shape[1] if head_state else (
        state["projector.0.weight"].shape[1] if "projector.0.weight" in state else backbone.embed_dim
    )
    pooling = pooling or ("cls_mean" if inferred_dim == 2 * backbone.embed_dim else "cls")
    if pooling not in ("cls", "mean", "cls_mean"):
        raise ValueError("pooling must be cls, mean or cls_mean")
    dimension = backbone.embed_dim * (2 if pooling == "cls_mean" else 1)
    head = None
    if head_state is not None:
        count, head_dim = head_state["weight"].shape
        if head_dim != dimension:
            raise ValueError("Classifier input dimension differs from pooling output")
        head = nn.Linear(dimension, count, bias="bias" in head_state)
        head.load_state_dict(head_state, strict=True)
    temporal = "log_temperature" in state and "bias" in state
    source = feature_source or model_config.get("feature_source") or ("projector" if temporal else "backbone")
    if source not in ("backbone", "projector"):
        raise ValueError("feature_source must be backbone or projector")
    projector = None
    if source == "projector" or "projector.0.weight" in state or "projector.2.weight" in state:
        if "projector.2.weight" not in state or "projector.0.weight" not in state:
            raise ValueError("Checkpoint has no supported projector")
        hidden, input_dim = state["projector.0.weight"].shape
        output_dim, output_hidden = state["projector.2.weight"].shape
        if input_dim != dimension or output_hidden != hidden:
            raise ValueError("Projector dimensions differ from pooling output")
        projector = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
        projector.load_state_dict({k.removeprefix("projector."): v for k, v in state.items() if k.startswith("projector.")}, strict=True)
    normalization = model_config.get("classifier_input_normalization", "none")
    return DinoInferenceModel(backbone, pooling, head, projector, source, normalization), temporal


def _load_lsnet(name, state):
    # DINOv3 does not import LSNet's Triton kernels or depend on timm registration.
    from lsnet_model import lsnet_artist
    from timm.models import create_model
    weight = state.get("head.l.weight")
    feature_dim = weight.shape[1] if weight is not None else None
    if "projection.0.l.weight" in state:
        feature_dim = state["projection.0.l.weight"].shape[0]
    model = create_model(name, pretrained=False, num_classes=weight.shape[0] if weight is not None else 0,
                         feature_dim=feature_dim, distillation="head_dist.l.weight" in state)
    model.load_state_dict(state, strict=True)
    model.has_classifier = weight is not None
    return model


def load_model_bundle(model_dir=None, device="cuda", checkpoint=None, model_name=None,
                      class_csv=None, input_size=None, feature_source=None):
    path = Path(checkpoint) if checkpoint else find_checkpoint(model_dir)
    config = read_model_config(model_dir or path.parent)
    selection = config.get("model", model_name)
    if selection is None:
        raise ValueError("Set the model architecture in config.json ('model') or pass an explicit model_name")
    payload = load_checkpoint_payload(path)
    state = checkpoint_state(payload)
    embedded = payload.get("model_config", {})
    embedded = embedded if isinstance(embedded, dict) else {}
    if isinstance(selection, dict):
        name = selection.get("name")
        options = {**embedded, **selection}
    else:
        name = selection
        options = {**embedded, **{k: config[k] for k in ("kwargs", "pooling", "feature_source", "classifier_input_normalization") if k in config}}
    if not isinstance(name, str):
        raise ValueError("config.json 'model' must be an architecture name or object with 'name'")
    if name.startswith("dinov3_") or name in ("custom_vit", "custom_convnext"):
        if embedded.get("name") and embedded["name"] != name:
            raise ValueError("config.json architecture differs from checkpoint model_config")
        model, temporal = _load_dino(name, state, options, feature_source)
        from kaloscope_dinov3.preprocessing import image_transform, prepare_rgb
        from torchvision import transforms
        training_config = payload.get("config", {})
        data = training_config.get("data", {}) if isinstance(training_config, dict) else {}
        data = {**data, **config.get("data", {})}
        size = config.get("input_size", data.get("image_size", input_size or (512 if temporal else 224)))
        custom_transform = data.get("custom_transform")
        if custom_transform and custom_transform.startswith("dinov3."):
            custom_transform = custom_transform.replace("dinov3.", "kaloscope_dinov3.", 1)
        if temporal and custom_transform is None:
            transform = transforms.Compose([prepare_rgb, transforms.Resize(size), transforms.CenterCrop(size),
                transforms.ToTensor(), transforms.Normalize(data.get("mean") or [0.485, 0.456, 0.406],
                                                           data.get("std") or [0.229, 0.224, 0.225])])
        else:
            transform = image_transform(size, data.get("mean"), data.get("std"), custom_transform=custom_transform)
    elif name in LSNET_MODELS:
        model = _load_lsnet(name, state)
        from kaloscope_dinov3.preprocessing import prepare_rgb
        from torchvision import transforms
        from lsnet_model.lsnet_artist import default_cfgs_artist
        from timm.data import resolve_data_config, create_transform
        size = config.get("input_size", input_size or default_cfgs_artist[name]["input_size"][1])
        transform = transforms.Compose([prepare_rgb, create_transform(
            **resolve_data_config({"input_size": (3, size, size)}, model=model))])
    else:
        raise ValueError(f"Unsupported model architecture in config.json: {name}")
    csv_path = Path(class_csv) if class_csv else path.parent / "class_mapping.csv"
    if class_csv and not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    mapping = load_class_mapping(csv_path) if csv_path.is_file() else None
    classes = config.get("classes", payload.get("classes"))
    if mapping is None and classes is not None:
        mapping = {i: str(label) for i, label in enumerate(classes)} if isinstance(classes, list) else {
            int(i): str(label) for i, label in classes.items()
        }
    count = model.head.out_features if isinstance(model.head, nn.Linear) else (
        model.head.l.out_features if model.has_classifier else 0)
    if model.has_classifier and mapping is not None and set(mapping) != set(range(count)):
        raise ValueError("Class mapping IDs must exactly match classifier outputs")
    model.to(device).eval()
    return {"model": model, "transform": transform, "class_mapping": mapping or {}, "device": device,
            "model_type": name, "has_classifier": model.has_classifier, "feature_dim": model.feature_dim,
            "feature_source": getattr(model, "feature_source", "backbone"), "input_size": size,
            "checkpoint": str(path)}
