"""Small numerical tests; python -m unittest discover -s tests -p test_z3.py."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaloscope_style.adapter import StyleAdapter, AfterBlock, StyleGuidance, guided_denoised
from kaloscope_style.features import blend, normalized, preprocess
from kaloscope_style.checkpoint import load_adapter, export_checkpoint


class Z3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_residual_matches_training_formula_and_leaves_text_and_negative_untouched(self):
        torch.manual_seed(12)
        adapter = StyleAdapter(dim=256, width=64, tokens=8, layers=1, hidden=32)
        torch.nn.init.normal_(adapter.out[0].weight, std=.01)
        ref = normalized(torch.randn(1, 256))
        x = torch.randn(4, 7, 32)
        original = lambda x, **kwargs: x + .2
        patcher = AfterBlock(original, adapter, ref, 0, 1.)
        result = patcher(x, transformer_options={'img_slice': [2, 7], 'cond_or_uncond': [0, 1]})
        base = x + .2
        st = adapter.encode(ref).expand(4, -1, -1)
        q = adapter.query[0](base[:, 2:]).reshape(4, -1, 1, 64).transpose(1, 2)
        k = adapter.key(st).reshape(4, -1, 1, 64).transpose(1, 2)
        v = adapter.value(st).reshape(4, -1, 1, 64).transpose(1, 2)
        expected = base.clone()
        delta = adapter.out[0](F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(4, -1, 64))
        expected[:2, 2:] += delta[:2]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(result[:, :2], base[:, :2], rtol=0, atol=0)
        off = patcher(x, transformer_options={'kaloscope_z3_disabled': True})
        torch.testing.assert_close(off, base, rtol=0, atol=0)

    def test_cfg_is_three_branch_velocity_equivalent_and_does_not_mutate_options(self):
        x, sigma = torch.randn(1, 4), .85
        v0, vt, vs = (torch.randn_like(x) for _ in range(3))
        d0, dt, ds = (x-sigma*v for v in (v0, vt, vs))
        options = {'transformer_options': {'existing': 'keep'}}
        calls = []
        def calc(model, cond, z, t, opts):
            calls.append(opts)
            self.assertTrue(opts['transformer_options']['kaloscope_z3_disabled'])
            return [dt]
        fake = types.ModuleType('comfy.samplers'); fake.calc_cond_batch = calc
        with patch.dict(sys.modules, {'comfy.samplers': fake}):
            output = StyleGuidance(5.5)(dict(input=x, sigma=torch.tensor([sigma]), model=None,
                model_options=options, input_cond=['positive'], uncond_denoised=d0, cond_denoised=ds, cond_scale=2.))
        torch.testing.assert_close(output / sigma, v0+2*(vt-v0)+5.5*(vs-vt))
        self.assertEqual(options, {'transformer_options': {'existing': 'keep'}})
        self.assertEqual(len(calls), 1)

    def test_blending_normalizes_each_reference_before_weighted_mean(self):
        x = torch.zeros(2, 256); x[0, 0] = 100; x[1, 1] = 1
        result = blend(x, '1,3')
        torch.testing.assert_close(result[0, :2], F.normalize(torch.tensor([1., 3.]), dim=0))
        with self.assertRaises(ValueError): blend(x, '-1,2')
        with self.assertRaises(ValueError): blend(x, '1')
        with self.assertRaises(ValueError): normalized(torch.zeros(1, 256))
        with self.assertRaises(ValueError): normalized(torch.randn(1, 1536))

    def test_preprocessing_is_center_crop_then_bicubic_resize(self):
        from PIL import Image
        import numpy as np
        pixels = torch.rand(41, 60, 3)
        array = (pixels*255).round().byte().numpy()
        im = Image.fromarray(array).crop((9, 0, 50, 41)).resize((512, 512), Image.Resampling.BICUBIC)
        expected = torch.from_numpy(np.array(im)).permute(2,0,1).float()/255
        expected = (expected-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        torch.testing.assert_close(preprocess(pixels), expected, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_architecture_rejection(self):
        config = dict(feature_dim=256, adapter_width=256, style_tokens=8, adapter_banks=4,
                      lora_rank=0, injection_indices=list(range(28)), injection_slots=[i//7 for i in range(28)])
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'step.pt', Path(temp)/'step.safetensors'
            state = {'adapter.'+k:v for k,v in StyleAdapter().state_dict().items()}
            torch.save(dict(trainable=state, config=config, step=12006), source)
            export_checkpoint(source, target)
            loaded = load_adapter(target)
            self.assertEqual(loaded['step'], 12006)
            for key, value in loaded['adapter'].state_dict().items():
                torch.testing.assert_close(value, state['adapter.'+key], rtol=0, atol=0)
            config['lora_rank'] = 8
            torch.save(dict(trainable=state, config=config), source)
            with self.assertRaises(ValueError): load_adapter(source)


class ResolutionAndArtistTests(unittest.TestCase):
    def test_turbo_uses_fixed_mu_and_eight_step_native_schedule(self):
        import math
        from kaloscope_style.schedule import sigma_schedule
        u = torch.linspace(1, 0, 9)
        expected = math.exp(1.15)*u / (1 + (math.exp(1.15)-1)*u)
        expected[0] = 1
        for size in (64,128,256):
            actual = sigma_schedule(8, 1, {'samples':torch.empty(1,16,1,size,size)}, 'turbo')
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sigma_preserves_512_and_uses_native_sequence_length(self):
        from kaloscope_style.schedule import sigma_schedule, resolution_mu
        self.assertAlmostEqual(resolution_mu(64, 64), .58125)
        self.assertAlmostEqual(resolution_mu(128, 128), .90625)
        self.assertAlmostEqual(resolution_mu(160, 160), 1.15)
        for start in (.85, 1.):
            old = sigma_schedule(28, start)
            new = sigma_schedule(28, start, {'samples': torch.empty(1,16,1,64,64)})
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            wide = sigma_schedule(28, start, {'samples': torch.empty(1,16,1,32,128)})
            torch.testing.assert_close(old, wide, rtol=0, atol=0)
            large = sigma_schedule(28, start, {'samples': torch.empty(1,16,1,128,128)})
            self.assertTrue((large[:-1] > large[1:]).all())
            self.assertTrue((large[1:-1] > old[1:-1]).all())
            self.assertAlmostEqual(float(large[0]), start, places=6)
            self.assertEqual(float(large[-1]), 0)

    def test_artist_lookup_matches_bank_and_rejects_unknown_or_wrong_encoder(self):
        from kaloscope_style.artists import artist_feature, ENCODER_SHA256
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bank.safetensors'
            z = normalized(torch.randn(2,256))
            meta = dict(format='kaloscope_z3_artist_bank_v1', encoder_sha256=ENCODER_SHA256,
                        names=json.dumps(['artist_one', 'artist_two']))
            save_file({'means':z, 'counts':torch.tensor([8,12])}, str(path), metadata=meta)
            got, info = artist_feature(path, 'artist two')
            torch.testing.assert_close(got, z[1:2])
            self.assertEqual(json.loads(info)['training_families'], 12)
            with self.assertRaisesRegex(ValueError, 'Unknown artist'): artist_feature(path, 'missing')
            meta['encoder_sha256'] = 'wrong'
            save_file({'means':z, 'counts':torch.tensor([8,12])}, str(path), metadata=meta)
            with self.assertRaisesRegex(ValueError, 'matching'): artist_feature(path, 'artist_one')


if __name__ == '__main__':
    unittest.main()
