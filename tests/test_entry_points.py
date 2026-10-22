"""Check real UI/API/CLI adapters, numerical parity and cache-only rendering."""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_model_loading import ReferenceInferenceModel, load_nodes, ROOT
from backend_lsnet.analysis import read_cache, cache_bytes, feature_tools
from backend_lsnet.analysis_api import AnalysisOptions, values
from backend_lsnet.analysis_ui import (extract_uploaded, import_uploaded_cache, plot_uploaded_cache,
                                      ANALYSIS_OPTION_NAMES)
from backend_lsnet.api import Api
from backend_lsnet.ui import create_ui
from feature_analysis import CHART_TYPES
from model_loading import FEATURE_OUTPUTS, load_model_bundle
from analysis_cli import parser, main


class EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.model_dir = cls.root / 'models' / 'kaloscope' / 'tiny'
        cls.model_dir.mkdir(parents=True)
        options = {'name': 'custom_vit', 'pooling': 'cls_mean', 'projection_dim': 8,
                   'kwargs': {'embed_dim': 24, 'depth': 2, 'num_heads': 3, 'patch_size': 16,
                              'n_storage_tokens': 2, 'pos_embed_rope_dtype': 'fp32'}}
        torch.manual_seed(7)
        model = ReferenceInferenceModel(options, 3).eval()
        torch.save({'model': model.state_dict(), 'model_config': options, 'classes': ['a', 'b', 'c']}, cls.model_dir / 'best.pt')
        (cls.model_dir / 'config.json').write_text(json.dumps({'model': 'custom_vit', 'input_size': 32}), encoding='utf-8')
        cls.input_dir = cls.root / 'images'
        cls.input_dir.mkdir()
        cls.images = [Image.new('RGB', (48, 40), color) for color in ((220,20,60),(160,10,20),(20,180,80),(10,110,50),(50,40,220),(30,20,150))]
        cls.files, cls.encoded = [], []
        for index, image in enumerate(cls.images):
            path = cls.input_dir / f'{index}.png'
            image.save(path)
            cls.files.append(str(path))
            cls.encoded.append(base64.b64encode(path.read_bytes()).decode())
        cls.bundle = load_model_bundle(cls.model_dir, device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.environment = patch.dict(os.environ, {'KALOSCOPE_MODELS_DIR': str(self.root / 'models')})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def test_all_feature_outputs_are_identical_in_comfy_ui_and_api(self):
        app = FastAPI()
        api = Api(app)
        nodes = load_nodes()
        images = torch.stack([torch.from_numpy(np.array(image.resize((48, 40)))).float() / 255 for image in self.images])
        try:
            with patch('backend_lsnet.analysis_ui.load_model_bundle', return_value=self.bundle), \
                 patch('backend_lsnet.analysis_api.load_model_bundle', return_value=self.bundle), TestClient(app) as client:
                for output_type in FEATURE_OUTPUTS:
                    with self.subTest(output_type=output_type):
                        layers = '-1,0' if output_type.startswith('intermediate_') else '-1'
                        expected = nodes.KaloscopeExtractFeaturesNode().extract(images, self.bundle, output_type, layers)[0]
                        cached, _, download = extract_uploaded(self.files, 'tiny', 'cpu', output_type, layers, True, 2)
                        torch.testing.assert_close(cached['features'], expected)
                        imported, _ = import_uploaded_cache(download)
                        torch.testing.assert_close(imported['features'], expected)
                        response = client.post('/kaloscope/v1/features', json={'input_images': self.encoded,
                                               'model_name': 'tiny', 'device': 'cpu', 'output_type': output_type, 'layers': layers})
                        self.assertEqual(response.status_code, 200, response.text)
                        cache = read_cache(io.BytesIO(base64.b64decode(response.json()['cache_base64'])))
                        torch.testing.assert_close(cache['features'], expected)
                        Path(download).unlink()
                        Path(download).parent.rmdir()
        finally:
            api.executor.shutdown()

    def test_all_charts_work_through_ui_and_api_without_loading_a_model(self):
        app = FastAPI()
        api = Api(app)
        from backend_lsnet.analysis import extract_batch
        features = extract_batch(self.images, self.bundle, 'patch_tokens')
        cached = {'features': features, 'labels': [f'Image {i}' for i in range(6)], 'output_type': 'patch_tokens', 'layers': '-1'}
        encoded = base64.b64encode(cache_bytes(features, cached['labels'], 'patch_tokens')).decode()
        options = values(AnalysisOptions(width=800, height=600, perplexity=3))
        try:
            with patch('backend_lsnet.analysis_api.load_model_bundle', side_effect=AssertionError('must not infer')), \
                 patch('backend_lsnet.analysis_ui.load_model_bundle', side_effect=AssertionError('must not infer')), TestClient(app) as client:
                for chart in CHART_TYPES:
                    with self.subTest(chart=chart):
                        image, report, files = plot_uploaded_cache(cached, chart, *[options[name] for name in ANALYSIS_OPTION_NAMES])
                        self.assertEqual(image.shape, (600, 800, 3))
                        response = client.post('/kaloscope/v1/analyze', json={'cache_base64': encoded, 'chart_type': chart, 'options': options})
                        self.assertEqual(response.status_code, 200, response.text)
                        result = response.json()
                        self.assertEqual(result['analysis']['chart_type'], chart)
                        np.testing.assert_allclose(result['distance_matrix'], json.loads(report)['distances'], atol=1e-7)
                        rendered = Image.open(io.BytesIO(base64.b64decode(result['image_base64'])))
                        self.assertEqual(rendered.size, (800, 600))
                        for file in files:
                            Path(file).unlink()
                        Path(files[0]).parent.rmdir()
        finally:
            api.executor.shutdown()

    def test_image_api_extracts_once_and_reuses_returned_cache(self):
        app = FastAPI()
        api = Api(app)
        try:
            with patch('backend_lsnet.analysis_api.load_model_bundle', return_value=self.bundle) as loading, \
                 patch.object(self.bundle['model'].backbone, 'forward_features', wraps=self.bundle['model'].backbone.forward_features) as forward, TestClient(app) as client:
                response = client.post('/kaloscope/v1/analyze', json={'image_batch': {'input_images': self.encoded,
                       'model_name': 'tiny', 'device': 'cpu', 'batch_size': 2}, 'options': {'width': 800, 'height': 600}})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(loading.call_count, 1)
                self.assertEqual(forward.call_count, 3)
                result = client.post('/kaloscope/v1/analyze', json={'cache_base64': response.json()['cache_base64'],
                         'chart_type': 'distance_heatmap', 'options': {'width': 800, 'height': 600}})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(forward.call_count, 3)
        finally:
            api.executor.shutdown()

    def test_feature_tools_known_values_and_http_input_errors(self):
        features = torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
        common = feature_tools(features)
        np.testing.assert_allclose(common['common_features'], [2/3, 2/3])
        self.assertEqual(feature_tools(features, 'compare_groups', 0, ['a','a','b'])['best_group'], 'a')
        app = FastAPI()
        api = Api(app)
        try:
            with TestClient(app) as client:
                result = client.post('/kaloscope/v1/feature-tools', json={'features': features.tolist(), 'operation': 'similarity'})
                self.assertEqual(result.status_code, 200)
                np.testing.assert_allclose(result.json()['similarities'], [1., 1/np.sqrt(2), 0.])
                result = client.post('/kaloscope/v1/analyze', json={'features': features.tolist(), 'cache_base64': 'invalid'})
                self.assertEqual(result.status_code, 400)
                result = client.post('/kaloscope/v1/analyze', json={'features': features.tolist(), 'chart_type': 'bad'})
                self.assertEqual(result.status_code, 400)
                self.assertEqual(len(client.get('/kaloscope/v1/models').json()['chart_types']), 18)
        finally:
            api.executor.shutdown()

    def test_cli_cache_and_real_image_path_match(self):
        output = self.root / 'cli'
        main(parser().parse_args(['--input', str(self.input_dir), '--model-dir', str(self.model_dir), '--device', 'cpu',
                                '--output-type', 'intermediate_patch_map', '--layers=-1,0', '--output', str(output),
                                '--chart-type', 'patch_energy', '--width', '800', '--height', '600']))
        cached = read_cache(output / 'features.npz')
        self.assertEqual(tuple(cached['features'].shape), (6, 2, 24, 2, 2))
        with patch('analysis_cli.load_model_bundle', side_effect=AssertionError('must not infer')):
            main(parser().parse_args(['--features', str(output / 'features.npz'), '--all-charts', '--output', str(output / 'redraw'),
                                     '--width', '800', '--height', '600', '--perplexity', '3']))
        self.assertEqual(len(list((output / 'redraw').glob('*.png'))), 18)

    def test_same_gradio_tabs_are_registered_by_webui_extension(self):
        ui = create_ui()
        callback_names = {callback.fn.__name__ for callback in ui.fns.values()}
        self.assertTrue({'infer', 'extract_uploaded', 'import_uploaded_cache', 'plot_uploaded_cache', 'tools_uploaded_cache'} <= callback_names)
        registrations = {}
        callbacks = types.SimpleNamespace(on_ui_tabs=lambda callback: registrations.update(tabs=callback),
                                          on_app_started=lambda callback: registrations.update(api=callback))
        modules = types.ModuleType('modules')
        modules.script_callbacks = callbacks
        modules.shared = types.SimpleNamespace()
        spec = importlib.util.spec_from_file_location('test_webui_extension', ROOT / 'scripts' / 'app.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'modules': modules}):
            spec.loader.exec_module(module)
            tabs = registrations['tabs']()
        self.assertTrue(module.IN_WEBUI)
        self.assertEqual(tabs[0][1:], ('Kaloscope', 'kaloscope_tab'))
        self.assertIsNotNone(registrations['api'])


if __name__ == '__main__':
    unittest.main()
