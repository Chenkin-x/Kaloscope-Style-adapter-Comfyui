import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from aiohttp import web
from kaloscope_style.workflows import migrate_workflow

ROOT = Path(__file__).resolve().parents[1]


class StandaloneTests(unittest.TestCase):
    def test_package_coexists_without_global_aliases_or_eager_dino_import(self):
        folder_paths = types.ModuleType('folder_paths')
        folder_paths.models_dir = '/tmp/kaloscope-test-no-models'
        routes = web.RouteTableDef()
        server = types.ModuleType('server')
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
        sentinels = {k: types.ModuleType(k) for k in ('model_loading', 'inference_artist', 'kaloscope_dinov3', 'lsnet_model', 'feature_analysis')}
        old_path = list(sys.path)
        name = 'generation_plugin_test'
        spec = importlib.util.spec_from_file_location(name, ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        try:
            with patch.dict(sys.modules, {**sentinels, 'folder_paths': folder_paths, 'server': server, name: module}):
                spec.loader.exec_module(module)
                for key, value in sentinels.items():
                    self.assertIs(sys.modules[key], value)
                self.assertEqual(sys.path, old_path)
                mapping = module.NODE_CLASS_MAPPINGS
                self.assertEqual(len(mapping), 10)
                self.assertNotRegex(str(mapping.keys()) + str(module.NODE_DISPLAY_NAME_MAPPINGS), r'(?i)z3')
                for cls in mapping.values():
                    self.assertNotRegex(str(cls.INPUT_TYPES()) + str(cls.RETURN_TYPES), r'(?i)z3')
                self.assertNotIn('KaloscopeModelLoader', mapping)
                self.assertEqual(mapping['KaloscopeStyleEncoderLoader'].RETURN_TYPES, ('KALOSCOPE_MODEL',))
                self.assertEqual(mapping['KaloscopeStyleEncodeReference'].INPUT_TYPES()['required']['model'][0], 'KALOSCOPE_MODEL')
                self.assertFalse(any(k.startswith(name + '.kaloscope_style._vendor') for k in sys.modules))
                self.assertTrue(all(r.path.startswith('/kaloscope-adapter/') for r in routes))
        finally:
            for key in list(sys.modules):
                if key == name or key.startswith(name + '.'):
                    sys.modules.pop(key)

    def test_migrate_style_only_api_preserves_mixed_analysis_graph(self):
        graph = {'1': {'class_type': 'KaloscopeModelLoader', 'inputs': {'model_folder': 'shared_v1', 'device': 'auto'}},
                 '2': {'class_type': 'KaloscopeStyleEncodeReference', 'inputs': {'model': ['1', 0]}}}
        migrated = migrate_workflow(graph)
        self.assertEqual(migrated['1']['class_type'], 'KaloscopeStyleEncoderLoader')
        self.assertEqual(migrated['1']['inputs'], graph['1']['inputs'])
        self.assertEqual(graph['1']['class_type'], 'KaloscopeModelLoader')
        graph['3'] = {'class_type': 'KaloscopeInference', 'inputs': {'model': ['1', 0]}}
        self.assertEqual(migrate_workflow(graph), graph)

    def test_release_examples_use_independent_loader_and_standard_vae(self):
        files = list((ROOT / 'examples' / 'release').glob('*.json'))
        self.assertEqual(len(files), 16)
        for path in files:
            text = path.read_text()
            self.assertNotIn('KaloscopeModelLoader', text, path.name)
            self.assertNotIn('krea2_z3_vae', text, path.name)
            json.loads(text)

    def test_runtime_has_no_analysis_dependency_or_path_mutation(self):
        forbidden = ('lsnet_model', 'backend_lsnet', 'scipy', 'sklearn', 'matplotlib', 'timm', 'triton', 'torchvision', 'gradio', 'fastapi')
        for path in [ROOT / '__init__.py', *(ROOT / 'kaloscope_style').rglob('*.py')]:
            text = path.read_text()
            self.assertNotIn('sys.path.', text, str(path))
            for line in text.splitlines():
                if line.strip().startswith(('import ', 'from ')):
                    self.assertFalse(any(term in line for term in forbidden), (path, line))
