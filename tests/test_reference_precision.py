"""Reference encoding across stored dtypes and a failing AMP convolution backend."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from kaloscope_style import encoder, features


class ConvBackbone(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, kernel_size=2, stride=16)

    def forward_features(self, images):
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        return {'x_norm_clstoken': tokens.mean(1), 'x_norm_patchtokens': tokens}


class ReferencePrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(35)
        self.model = encoder.StyleEncoder(ConvBackbone(), 8).eval().requires_grad_(False)
        self.images = torch.rand(2, 20, 32, 3)

    def expected(self):
        parameter = next(self.model.parameters())
        with torch.inference_mode(), torch.autocast(parameter.device.type, enabled=False):
            return features.normalized(torch.cat([
                self.model.extract_tensor(features.preprocess(image)[None].to(parameter))
                for image in self.images]))

    def test_shared_low_precision_encoder_keeps_weights_and_aligns_input(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.model.to(dtype=dtype)
                before = {k: v.clone() for k, v in self.model.state_dict().items()}
                result = features.encode(self.images, {'model': self.model})
                torch.testing.assert_close(result, self.expected(), rtol=0, atol=0)
                self.assertEqual(result.dtype, torch.float32)
                for k, v in self.model.state_dict().items():
                    self.assertEqual(v.dtype, dtype)
                    self.assertTrue(torch.equal(v, before[k]))

    def test_cpu_encoding_isolated_from_outer_autocast(self):
        expected = self.expected()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = features.encode(self.images, {'model': self.model})
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_amp_convolution_mismatch_retries_without_changing_shared_weights(self):
        self.model.cuda()
        expected = self.expected()
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        conv = self.model.backbone.patch_embed
        original = conv._conv_forward
        failures = []

        def backend(input, weight, bias):
            # Inject the reported backend failure, using a real mismatched Conv2d.
            if torch.is_autocast_enabled('cuda'):
                failures.append(True)
                with torch.autocast('cuda', enabled=False):
                    return original(input.bfloat16(), weight, bias)
            return original(input, weight, bias)

        with patch.object(conv, '_conv_forward', side_effect=backend):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with self.assertLogs('kaloscope_style.features', level='WARNING') as logs:
                    result = features.encode(self.images, {'model': self.model})
                self.assertTrue(torch.is_autocast_enabled('cuda'))
        self.assertEqual(len(failures), 1)  # Remaining images use the successful precision.
        self.assertEqual(len(logs.output), 1)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for k, v in self.model.state_dict().items():
            self.assertEqual(v.dtype, torch.float32)
            self.assertTrue(torch.equal(v, before[k]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_without_native_bf16_uses_stored_precision(self):
        self.model.cuda()
        expected = self.expected()
        with patch.object(torch.cuda, 'is_bf16_supported', return_value=False):
            result = features.encode(self.images, {'model': self.model})
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_unrelated_cuda_errors_are_not_retried(self):
        self.model.cuda()
        for message in ('CUDA out of memory', 'invalid shape for input'):
            with self.subTest(message=message):
                with patch.object(self.model, 'extract_tensor', side_effect=RuntimeError(message)) as call:
                    with self.assertRaisesRegex(RuntimeError, message):
                        features.encode(self.images, {'model': self.model})
                    self.assertEqual(call.call_count, 1)
