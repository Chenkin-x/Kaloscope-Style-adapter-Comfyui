import unittest
import sys
import types
from unittest.mock import patch

import torch
from torch import nn

from kaloscope_style import features


class ColorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(3, 256, bias=False))
        with torch.no_grad():
            self.projector[0].weight.zero_()
            self.projector[0].weight[:3] = torch.eye(3)

    def extract_tensor(self, image, output_type):
        return self.projector(image.mean((2, 3)))


class ReferencePreprocessTests(unittest.TestCase):
    def setUp(self):
        # Red and blue borders are entirely outside the square center crop.
        self.image = torch.zeros(16, 48, 3)
        self.image[:, :16, 0] = 1
        self.image[:, 16:32, 1] = 1
        self.image[:, 32:, 2] = 1

    def test_resize_preserves_all_edges_in_landscape_and_portrait(self):
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        for image in (self.image, self.image.transpose(0, 1)):
            full = features.preprocess(image, 'resize') * std + mean
            crop = features.preprocess(image) * std + mean
            self.assertEqual(tuple(full.shape), (3, 512, 512))
            torch.testing.assert_close(full[:, 0, 0], torch.tensor([1., 0., 0.]))
            torch.testing.assert_close(full[:, -1, -1], torch.tensor([0., 0., 1.]))
            torch.testing.assert_close(crop, torch.tensor([0., 1., 0.])[:, None, None].expand_as(crop))

    def test_node_option_changes_features_and_old_calls_keep_default(self):
        with patch.dict(sys.modules, {'folder_paths': types.ModuleType('folder_paths')}):
            from kaloscope_style.nodes import KaloscopeStyleEncodeReference
        node = KaloscopeStyleEncodeReference()
        bundle = {'model': ColorEncoder()}
        images = torch.stack((self.image, self.image.flip(1)))
        old_call, = node.encode(images, bundle)
        crop, = node.encode(images, bundle, resize_mode='center_crop')
        full, = node.encode(images, bundle, resize_mode='resize')
        self.assertTrue(torch.equal(old_call, crop))
        self.assertFalse(torch.allclose(full, crop))
        self.assertEqual(tuple(full.shape), (2, 256))
        self.assertTrue(full.isfinite().all())
        torch.testing.assert_close(full.norm(dim=1), torch.ones(2))

    def test_invalid_resize_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'resize_mode'):
            features.preprocess(self.image, 'unknown')
