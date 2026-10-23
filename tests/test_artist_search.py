import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaloscope_style.catalog import ArtistIndex, artist_index, bundled_rows, ENCODER_SHA256
from kaloscope_style.artists import artist_feature
from kaloscope_style.routes import register_routes


def make_bank(path, names):
    values = torch.zeros(len(names), 256)
    for i in range(len(names)):
        values[i, i] = 1
    save_file({'means': values, 'counts': torch.arange(1, len(names)+1)}, str(path),
              metadata={'format': 'kaloscope_z3_artist_bank_v1', 'encoder_sha256': ENCODER_SHA256,
                        'names': json.dumps(names)})


class ArtistSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'bank.safetensors'
        make_bank(self.path, ['fkey', 'aaru_(one)', 'aaru_(two)'])

    def test_exact_case_spaces_and_typo_suggestions(self):
        index = artist_index(self.path)
        self.assertEqual(index.resolve(' FKEY '), 0)
        self.assertEqual(index.resolve('aaru (one)'), 1)
        self.assertEqual(index.search('fky')['results'][0]['name'], 'fkey')
        with self.assertRaisesRegex(ValueError, 'Unknown artist.*fkey'):
            index.resolve('fky')
        torch.testing.assert_close(artist_feature(self.path, 'FKEY')[0], artist_feature(self.path, 'fkey')[0], rtol=0, atol=0)

    def test_disambiguation_and_characters_are_not_auto_resolved(self):
        index = artist_index(self.path)
        self.assertEqual([r['name'] for r in index.search('aaru', 2)['results']], ['aaru_(one)', 'aaru_(two)'])
        for invalid in ['aaru', 'feiqizi_(fkey)']:
            with self.assertRaises(ValueError):
                index.resolve(invalid)

    def test_alias_resolution_and_conflicting_alias(self):
        rows=[dict(name='one', aliases=['alias'], families=1), dict(name='two', aliases=['shared'], families=1)]
        index=ArtistIndex(rows, 'test', True)
        self.assertEqual(index.resolve('ALIAS'), 0)
        rows[0]['aliases'].append('shared')
        with self.assertRaisesRegex(ValueError, 'Ambiguous artist'):
            ArtistIndex(rows, 'test', True).resolve('shared')

    def test_close_typo_outranks_incidental_substring_in_long_handle(self):
        rows=[dict(name=n, aliases=[], families=1) for n in ['fkey','hiyayakko_(jfky2528)']]
        self.assertEqual(ArtistIndex(rows, 'test', True).search('fky')['results'][0]['name'], 'fkey')

    def test_metadata_refresh_after_bank_replacement(self):
        self.assertEqual(artist_index(self.path).resolve('fkey'), 0)
        replacement = self.path.with_suffix('.new')
        make_bank(replacement, ['fkey', 'replacement'])
        replacement.replace(self.path)
        index = artist_index(self.path)
        self.assertEqual(len(index.rows), 2)
        self.assertEqual(index.resolve('replacement'), 1)

    def test_csv_count_aliases_and_actual_bank_filter(self):
        rows = bundled_rows()
        self.assertEqual(len(rows), 44129)
        self.assertEqual(len({r['name'] for r in rows}), len(rows))
        self.assertEqual(next(r for r in rows if r['name']=='fkey')['families'], 27)
        self.assertEqual(sum(len(r['aliases']) for r in rows), 105)
        self.assertEqual(artist_index(self.path).search('')['total_artists'], 3)
        self.assertFalse(artist_index().available)

    def test_malformed_bank_is_rejected(self):
        save_file({'means': torch.ones(1,256), 'counts': torch.ones(1)}, str(self.path),
                  metadata={'format':'kaloscope_z3_artist_bank_v1','encoder_sha256':ENCODER_SHA256,'names':'["one","two"]'})
        with self.assertRaisesRegex(ValueError, 'do not align'):
            artist_index(self.path)


class ArtistRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'bank.safetensors'
        make_bank(self.path, ['fkey', 'aaru_(one)', 'aaru_(two)'])
        routes = web.RouteTableDef()
        register_routes(routes, lambda: {'bank.safetensors': self.path})
        application = web.Application();application.add_routes(routes)
        self.client = TestClient(TestServer(application))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close();self.temp.cleanup()

    async def test_workflow_route_migrates_without_modifying_prompts(self):
        from kaloscope_style.compat import NODE_IDS
        old = next(k for k, v in NODE_IDS.items() if v == 'KaloscopeStyleArtistMix')
        graph = {'1': {'class_type': old, 'inputs': {'bank': 'bank.safetensors', 'artist_tags': 'fkey, (rolua:0.8)'}}}
        response = await self.client.post('/kaloscope-adapter/migrate-workflow', json=graph)
        self.assertEqual(response.status, 200)
        migrated = await response.json()
        self.assertEqual(migrated['1']['class_type'], 'KaloscopeStyleArtistMix')
        self.assertEqual(migrated['1']['inputs'], graph['1']['inputs'])
        response = await self.client.post('/kaloscope-adapter/migrate-workflow', json=[])
        self.assertEqual(response.status, 400)

    async def test_search_and_csv_download_use_selected_bank(self):
        response = await self.client.get('/kaloscope-adapter/artists', params={'bank':'bank.safetensors','q':'fky'})
        self.assertEqual(response.status, 200)
        data=await response.json();self.assertEqual(data['results'][0]['name'],'fkey');self.assertTrue(data['available'])
        response=await self.client.get('/kaloscope-adapter/artists/catalog.csv',params={'bank':'bank.safetensors'})
        content=(await response.read()).decode('utf-8-sig')
        self.assertEqual([r['name'] for r in csv.DictReader(io.StringIO(content))], ['fkey','aaru_(one)','aaru_(two)'])

    async def test_invalid_names_paths_and_limits_fail_without_file_access(self):
        for params in [{'bank':'../../secret'}, {'bank':str(self.path)}, {'limit':'abc'}, {'limit':'9999'}, {'q':'x'*161}]:
            response=await self.client.get('/kaloscope-adapter/artists',params=params)
            self.assertEqual(response.status,400,params)
            self.assertIn('error',await response.json())


if __name__ == '__main__':
    unittest.main()
