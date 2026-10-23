import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file
from kaloscope_style import encoder, features


class TinyBackbone(nn.Module):
    embed_dim = 4
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
    def forward_features(self, image):
        x = self.linear(image.mean((2, 3)))
        return {'x_norm_clstoken': x, 'x_norm_patchtokens': x[:, None].expand(-1, 3, -1)}


class SharedEncoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'kaloscope' / 'shared_v1'
        self.folder.mkdir(parents=True)
        self.config = {'model': 'dinov3_vitb16', 'checkpoint': 'model.safetensors', 'pooling': 'cls_mean', 'input_size': 512}
        torch.manual_seed(21)
        self.oracle = encoder.StyleEncoder(TinyBackbone(), 8).eval()
        self.state = dict(self.oracle.state_dict())
        self.state.update({'head.weight': torch.ones(10, 8), 'head.bias': torch.ones(10)})
        self.save()
        self.mock = patch.object(encoder, 'build_backbone', TinyBackbone)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def save(self):
        (self.folder / 'config.json').write_text(json.dumps(self.config))
        save_file(self.state, str(self.folder / 'model.safetensors'))

    def test_shared_classifier_file_omits_head_and_preserves_features(self):
        before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        loaded = encoder.load_encoder(self.folder)
        self.assertFalse(loaded['has_classifier'])
        self.assertEqual(encoder.model_folders(self.root), {'shared_v1': self.folder})
        images = torch.rand(1, 24, 32, 3)
        # The original loader's bundle needs no patcher and no type conversion.
        original_bundle = {'model': self.oracle, 'device': torch.device('cpu'), 'has_classifier': True, 'extra': 'original'}
        self.assertTrue(torch.equal(features.encode(images, original_bundle), features.encode(images, loaded)))
        self.assertFalse(any(k.startswith('head.') for k in loaded['model'].state_dict()))
        self.assertEqual(before, {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.folder.iterdir()})

    def test_rejects_incompatible_models_and_invalid_weights(self):
        self.config['model'] = 'lsnet_t_artist'
        self.save()
        with self.assertRaisesRegex(ValueError, 'LSNet'):
            encoder.load_encoder(self.folder)
        self.config['model'] = 'dinov3_vitb16'
        self.state['projector.2.weight'] = torch.ones(128, 8)
        self.save()
        with self.assertRaisesRegex(ValueError, '256D'):
            encoder.load_encoder(self.folder)
        self.state = dict(self.oracle.state_dict())
        self.state['projector.2.bias'] = torch.full((256,), float('nan'))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            encoder.load_encoder(self.folder)

    def test_safe_nested_pt_and_training_prefixes(self):
        self.config['checkpoint'] = 'best.pt'
        self.config['model'] = {'name': 'dinov3_vitb16', 'pooling': 'cls_mean', 'projection_dim': 256}
        self.save()
        torch.save({'model': {'module._orig_mod.' + k: v for k, v in self.state.items()}}, self.folder / 'best.pt')
        loaded = encoder.load_encoder(self.folder)
        self.assertEqual(set(loaded['model'].state_dict()), set(self.oracle.state_dict()))
        for key, value in self.oracle.state_dict().items():
            self.assertTrue(torch.equal(value, loaded['model'].state_dict()[key]))

    def test_config_checkpoint_wins_over_default(self):
        (self.folder / 'best.pt').write_bytes(b'priority')
        self.assertEqual(encoder.find_checkpoint(self.folder).name, 'model.safetensors')
        self.config.pop('checkpoint')
        self.save()
        self.assertEqual(encoder.find_checkpoint(self.folder).name, 'best.pt')
