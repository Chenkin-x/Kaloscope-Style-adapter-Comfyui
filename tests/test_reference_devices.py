"""Exercise real mixed-device tensors with a minimal ComfyUI loader boundary."""
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from kaloscope_style import encoder, features


class DeviceBackbone(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        # Like DINO's CLS token: first parameter can reside on a different device.
        self.token = nn.Parameter(torch.ones(4))
        self.conv = nn.Conv2d(3, 4, kernel_size=2, stride=16)
        self.register_buffer('offset', torch.zeros(4))

    def forward_features(self, image):
        x = self.conv(image).mean((2, 3)) + self.token + self.offset
        return {'x_norm_clstoken': x, 'x_norm_patchtokens': x[:, None]}


class LoaderBoundary:
    def __init__(self, model, device, ignore_full_load=False):
        self.model, self.load_device = model, device
        self.ignore_full_load = ignore_full_load

    def load_models_gpu(self, models, force_full_load=False):
        # A cached partial load / --novram can ignore the top-level full-load flag.
        if force_full_load and not self.ignore_full_load:
            self.model.to(self.load_device)

    def patch_model(self, device_to=None, lowvram_model_memory=0):
        self.model.to(device_to)


class ReferenceDeviceTests(unittest.TestCase):
    def run_encoding(self, device, ignore_full_load=False, buffer_only=False):
        torch.manual_seed(42)
        model = encoder.StyleEncoder(DeviceBackbone(), 8).eval().requires_grad_(False).to(device)
        images = torch.rand(2, 16, 48, 3)
        expected = features.encode(images, {'model': model}, resize_mode='resize')
        if device.type == 'cuda':
            if not buffer_only:
                model.backbone.conv.cpu()
                model.projector.cpu()
            model.backbone.offset = model.backbone.offset.cpu()
        loader = LoaderBoundary(model, device, ignore_full_load)
        comfy = types.ModuleType('comfy')
        comfy.model_management = loader
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': loader}):
            actual = features.encode(images, {'model': model, 'patcher': loader}, resize_mode='resize')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(t.device == device for t in (*model.parameters(), *model.buffers())))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_partial_gpu_encoder_is_fully_loaded_before_forward(self):
        self.run_encoding(torch.device('cuda', torch.cuda.current_device()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cached_partial_load_is_repaired_when_manager_ignores_flag(self):
        self.run_encoding(torch.device('cuda', torch.cuda.current_device()), ignore_full_load=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_offloaded_buffers_are_checked_even_when_all_parameters_are_on_gpu(self):
        self.run_encoding(torch.device('cuda', torch.cuda.current_device()), ignore_full_load=True, buffer_only=True)

    def test_cpu_loader_still_encodes_without_cuda(self):
        self.run_encoding(torch.device('cpu'))
