import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaloscope_style.artist_mix import parse_artist_mix, mix_artists
from kaloscope_style.artists import artist_feature, ENCODER_SHA256
from kaloscope_style.features import blend

class ArtistMixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'bank.safetensors'
        features = torch.zeros(4, 256)
        features[0, 0] = 1; features[1, 1] = 1; features[2, 2] = 1; features[3, 3] = 1
        save_file({'means': features, 'counts': torch.tensor([1, 2, 3, 4])}, str(self.path),
                  metadata={'format': 'kaloscope_z3_artist_bank_v1', 'encoder_sha256': ENCODER_SHA256,
                            'names': json.dumps(['rolua', 'wanke', 'nineo', 'foo_(bar)'])})

    def test_requested_syntax_matches_existing_manual_blend(self):
        got, info = mix_artists(self.path, 'rolua, (wanke:0.8), (nineo:0.5)')
        rows = torch.cat([artist_feature(self.path, n)[0] for n in ('rolua', 'wanke', 'nineo')])
        torch.testing.assert_close(got, blend(rows, '1,.8,.5'))
        torch.testing.assert_close(got[0,:3], torch.nn.functional.normalize(torch.tensor([1., .8, .5]), dim=0))
        self.assertAlmostEqual(sum(a['relative_weight'] for a in json.loads(info)['artists']), 1)

    def test_single_endpoint_zero_and_duplicates(self):
        single = artist_feature(self.path, 'rolua')[0]
        for text in ['rolua', '(rolua:7)', 'rolua, (wanke:0)', 'rolua, (rolua:2)']:
            torch.testing.assert_close(mix_artists(self.path, text)[0], single, rtol=0, atol=0)

    def test_literal_parentheses_and_escaping(self):
        for text in ['foo_(bar)', '(foo_(bar):0.8)', r'(foo_\(bar\):0.8)']:
            torch.testing.assert_close(mix_artists(self.path, text)[0], artist_feature(self.path, 'foo_(bar)')[0])
        self.assertEqual(parse_artist_mix('rolua，\n (wanke:8e-1), '), [('rolua',1.),('wanke',.8)])

    def test_global_scale_invariance_and_large_weights(self):
        a=mix_artists(self.path,'(rolua:1), (wanke:2)')[0]
        for t in ['(rolua:10), (wanke:20)','(rolua:1e307), (wanke:2e307)']:
            torch.testing.assert_close(a,mix_artists(self.path,t)[0])

    def test_bad_inputs_fail_clearly(self):
        for t in ['', '(rolua:0)', '(rolua:-1)', '(rolua:nan)', '(rolua:inf)', '(rolua:no)', '(rolua:1', 'rolua)', 'rolua\\']:
            with self.subTest(t=t), self.assertRaises(ValueError):mix_artists(self.path,t)
        with self.assertRaisesRegex(ValueError,'Unknown artist'):mix_artists(self.path,'rolua, missing')

if __name__=='__main__':unittest.main()
