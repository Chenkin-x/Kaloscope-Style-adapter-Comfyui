# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Construct DINOv3 inference backbones without downloading weights."""
import inspect

from .hub import backbones

MODEL_NAMES = tuple(name for name in vars(backbones) if name.startswith("dinov3_"))


def build_backbone(config):
    name = config.get("name", "dinov3_vits16")
    kwargs = config.get("kwargs", {})
    if name in ("custom_vit", "custom_convnext"):
        if name == "custom_vit":
            from .models.vision_transformer import DinoVisionTransformer
            factory = DinoVisionTransformer
        else:
            from .models.convnext import ConvNeXt
            factory = ConvNeXt
        unknown = set(kwargs) - set(inspect.signature(factory).parameters)
        if unknown:
            raise ValueError(f"Unknown custom architecture options: {sorted(unknown)}")
        backbone = factory(**kwargs)
        if name == "custom_vit":
            backbone.init_weights()
        return backbone
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name}; available: {MODEL_NAMES}, custom_vit, custom_convnext")
    return getattr(backbones, name)(pretrained=False, **kwargs)
