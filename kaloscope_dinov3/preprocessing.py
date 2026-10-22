# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Deterministic image preprocessing for inference; no datasets or augmentations."""
import importlib

import torch
from PIL import Image, ImageOps
from torchvision import transforms


def prepare_rgb(image):
    """Apply orientation and composite transparency before discarding alpha."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _import_callable(dotted_path):
    # Existing checkpoints can refer to the original preprocessing module.
    for prefix in ("dinov3.finetune.data.", "kaloscope_dinov3.finetune.data."):
        if dotted_path.startswith(prefix):
            dotted_path = "kaloscope_dinov3.preprocessing." + dotted_path[len(prefix):]
            break
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid transform path: {dotted_path!r} (must be 'module.attr')")
    return getattr(importlib.import_module(module_path), attr_name)


def image_transform(size, mean=None, std=None, custom_transform=None):
    if custom_transform is not None:
        factory = _import_callable(custom_transform)
        return transforms.Compose([prepare_rgb, factory(resize_size=size)])
    return transforms.Compose([
        prepare_rgb,
        transforms.Resize(round(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean or [0.485, 0.456, 0.406], std or [0.229, 0.224, 0.225]),
    ])


def lvd_transform(resize_size=256):
    from torchvision.transforms import v2
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def sat_transform(resize_size=256):
    from torchvision.transforms import v2
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    ])
