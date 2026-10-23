import json
from pathlib import Path
import tempfile
import unittest
from kaloscope_style.compat import NODE_IDS, ASSET_NAMES, LEGACY_ADAPTER_DIRECTORY, LEGACY_ADAPTER_SOCKET
from kaloscope_style.paths import adapter_paths, bank_paths, encoder_paths
from kaloscope_style.workflows import migrate_workflow

ROOT = Path(__file__).resolve().parents[1]


class NamingAndMigrationTests(unittest.TestCase):
    def test_public_assets_resolve_to_same_legacy_files_without_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_adapters = root / LEGACY_ADAPTER_DIRECTORY
            old_adapters.mkdir()
            files = list(ASSET_NAMES.items())
            for old, _ in files[:3]: (old_adapters / old).write_bytes(b'original')
            banks = root / 'kaloscope_style_banks'; banks.mkdir()
            (banks / files[3][0]).write_bytes(b'bank')
            encoders = root / 'kaloscope'; encoders.mkdir()
            (encoders / files[4][0]).mkdir()
            before = sorted(str(p) for p in root.rglob('*'))
            choices = adapter_paths(root)
            for old, new in files[:3]: self.assertEqual(choices[new], old_adapters / old)
            self.assertEqual(bank_paths(root)[files[3][1]], banks / files[3][0])
            self.assertEqual(encoder_paths(root)[files[4][1]], encoders / files[4][0])
            self.assertEqual(before, sorted(str(p) for p in root.rglob('*')))
            newroot = root / 'kaloscope_adapters'; newroot.mkdir()
            (newroot / files[0][1]).write_bytes(b'preferred')
            self.assertEqual(adapter_paths(root)[files[0][1]], newroot / files[0][1])
            for old, new in files[5:]:
                (old_adapters / old).write_bytes(b'history')
                self.assertEqual(adapter_paths(root)[new], old_adapters / old)
            self.assertNotRegex(str(list(adapter_paths(root))), r'(?i)z3|stage[a-z]')
            self.assertEqual((old_adapters / files[0][0]).read_bytes(), b'original')

    def test_legacy_ui_ids_sockets_widgets_and_links_migrate_together(self):
        ids = {v: k for k, v in NODE_IDS.items()}
        old_file = next(iter(ASSET_NAMES))
        graph = {'nodes': [
            {'id': 1, 'type': ids['KaloscopeStyleAdapterLoader'], 'title': 'Kaloscope Z3 Checkpoint Loader',
             'widgets_values': [old_file], 'outputs': [{'name': LEGACY_ADAPTER_SOCKET, 'type': LEGACY_ADAPTER_SOCKET, 'links': [1]}]},
            {'id': 2, 'type': ids['KaloscopeStyleApply'], 'widgets_values': [3.5, .8],
             'inputs': [{'name': 'adapter', 'type': LEGACY_ADAPTER_SOCKET, 'link': 1}]},
            {'id': 3, 'type': 'CLIPTextEncode', 'widgets_values': ['A sign reading Z3, keep literal prompt']},
        ], 'links': [[1, 1, 0, 2, 0, LEGACY_ADAPTER_SOCKET]]}
        new = migrate_workflow(graph)
        self.assertEqual(new['nodes'][0]['type'], 'KaloscopeStyleAdapterLoader')
        self.assertEqual(new['nodes'][0]['widgets_values'], [ASSET_NAMES[old_file]])
        self.assertEqual(new['nodes'][1]['inputs'][0]['type'], 'KALOSCOPE_STYLE_ADAPTER')
        self.assertEqual(new['links'][0][5], 'KALOSCOPE_STYLE_ADAPTER')
        self.assertEqual(new['nodes'][2], graph['nodes'][2])
        self.assertEqual(new['nodes'][1]['widgets_values'], graph['nodes'][1]['widgets_values'])
        self.assertEqual(graph['nodes'][0]['type'], ids['KaloscopeStyleAdapterLoader'])
        self.assertEqual(migrate_workflow(new), new)

    def test_mixed_api_preserves_original_loader_directory_and_user_prompt(self):
        ids = {v: k for k, v in NODE_IDS.items()}
        old_folder = next(k for k, v in ASSET_NAMES.items() if v == 'artist-v1')
        graph = {'1': {'class_type': 'KaloscopeModelLoader', 'inputs': {'model_folder': old_folder, 'device': 'cpu'}},
                 '2': {'class_type': ids['KaloscopeStyleEncodeReference'], 'inputs': {'model': ['1', 0], 'reference_images': ['4', 0]}},
                 '3': {'class_type': 'KaloscopeArtistInference', 'inputs': {'model': ['1', 0]}},
                 '4': {'class_type': 'LoadImage', 'inputs': {'image': 'my_z3_reference.png'}}}
        new = migrate_workflow(graph)
        self.assertEqual(new['1'], graph['1'])
        self.assertEqual(new['4'], graph['4'])
        self.assertEqual(new['2']['class_type'], 'KaloscopeStyleEncodeReference')
        del graph['3']
        new = migrate_workflow(graph)
        self.assertEqual(new['1']['class_type'], 'KaloscopeStyleEncoderLoader')
        self.assertEqual(new['1']['inputs']['model_folder'], 'artist-v1')

    def test_public_documentation_examples_and_filenames_have_no_research_labels(self):
        paths = [ROOT / 'README.md', ROOT / 'README_MOSS_ZH.md', ROOT / 'pyproject.toml', ROOT / 'data/artists.json',
                 *(ROOT / 'docs').glob('*.md'), *(ROOT / 'examples').rglob('*')]
        for path in paths:
            if not path.is_file(): continue
            text = path.read_text()
            self.assertNotRegex(str(path.relative_to(ROOT)), r'(?i)z3|stagec|step12006')
            self.assertNotRegex(text, r'(?i)z3|stagec|step12006|222222222l')
