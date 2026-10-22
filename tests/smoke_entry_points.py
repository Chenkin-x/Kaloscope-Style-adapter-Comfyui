"""Check supplied images/checkpoint and a live standalone Gradio/API server."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from gradio_client import Client, handle_file

from backend_lsnet.analysis import cache_bytes, read_cache
from backend_lsnet.analysis_api import AnalysisOptions, values
from backend_lsnet.analysis_ui import extract_uploaded, ANALYSIS_OPTION_NAMES
from feature_analysis import CHART_TYPES
from model_loading import find_checkpoint
from analysis_cli import parser as cli_parser, main as cli_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('../test'))
    parser.add_argument('--model-dir', type=Path, default=Path('../model'))
    parser.add_argument('--output', type=Path, default=Path('../outputs/entry_points'))
    parser.add_argument('--patch-cache', type=Path, default=Path('../outputs/features.pt'))
    parser.add_argument('--server-url', default='http://127.0.0.1:7865')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'})
    # Test actual extraction with the provided checkpoint; replace only folder discovery.
    with patch('backend_lsnet.analysis_ui.get_checkpoint_path', return_value=str(find_checkpoint(args.model_dir))):
        cached, status, downloaded = extract_uploaded([str(path.resolve()) for path in paths], 'provided', args.device, 'default', '-1', True, 2)
    (args.output / 'features.npz').write_bytes(Path(downloaded).read_bytes())
    Path(downloaded).unlink()
    Path(downloaded).parent.rmdir()
    print(status, flush=True)
    original = torch.load(args.patch_cache, map_location='cpu', weights_only=True)
    torch.testing.assert_close(cached['features'], original['features'], rtol=1e-4, atol=1e-4)
    # Reuse previously inferred patches to test every chart without extra inference.
    patch_npz = args.output / 'patch_features.npz'
    patch_npz.write_bytes(cache_bytes(original['patch_tokens'], cached['labels'], 'patch_tokens'))
    payload = base64.b64encode(patch_npz.read_bytes()).decode()
    client = Client(args.server_url, verbose=False)
    imported = client.predict(handle_file(str(patch_npz.resolve())), api_name='/import_uploaded_cache')
    print(imported, flush=True)
    options = values(AnalysisOptions(width=1200, height=900, perplexity=3, top_k=3))
    options['labels'] = '\n'.join(original['labels'])
    api_options = {**options, 'labels': original['labels']}
    recorded = []
    for chart in CHART_TYPES:
        ui_dir = args.output / 'ui'
        ui_dir.mkdir(exist_ok=True)
        result = client.predict(chart, *[options[name] for name in ANALYSIS_OPTION_NAMES], api_name='/plot_uploaded_cache')
        shutil.copyfile(result[0], ui_dir / f'{chart}.png')
        (ui_dir / f'{chart}.json').write_text(result[1], encoding='utf-8')
        for source in result[2]:
            if str(source).endswith('.csv'):
                shutil.copyfile(source, ui_dir / f'{chart}.csv')
        request = urllib.request.Request(args.server_url + '/kaloscope/v1/analyze',
            data=json.dumps({'cache_base64': payload, 'chart_type': chart, 'options': api_options}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            api = json.load(response)
        api_dir = args.output / 'api'
        api_dir.mkdir(exist_ok=True)
        (api_dir / f'{chart}.png').write_bytes(base64.b64decode(api['image_base64']))
        (api_dir / f'{chart}.json').write_text(json.dumps(api['analysis'], indent=2, ensure_ascii=False), encoding='utf-8')
        np.testing.assert_allclose(api['distance_matrix'], json.loads(result[1])['distances'], atol=1e-7)
        for directory in (ui_dir, api_dir):
            with Image.open(directory / f'{chart}.png') as image:
                assert image.size == (1200, 900)
                image.verify()
        recorded.append(chart)
        print(f'Live UI and API cache rendering passed: {chart}', flush=True)
    (args.output / 'options.json').write_text(json.dumps(api_options), encoding='utf-8')
    cli_main(cli_parser().parse_args(['--features', str(patch_npz), '--all-charts', '--output', str(args.output / 'cli'),
                                    '--options-json', str(args.output / 'options.json')]))
    for chart in CHART_TYPES:
        cli = json.loads((args.output / 'cli' / f'{chart}.json').read_text(encoding='utf-8'))
        api = json.loads((args.output / 'api' / f'{chart}.json').read_text(encoding='utf-8'))
        np.testing.assert_allclose(cli['distances'], api['distances'], atol=1e-7)
    manifest = {'sample_count': len(paths), 'feature_shape': list(cached['features'].shape),
                'patch_shape': list(original['patch_tokens'].shape), 'charts_per_entry': recorded,
                'live_gradio_cache_session': True, 'analysis_model_forward_count': 0,
                'numeric_parity': 'UI / HTTP API / standalone CLI distances equal within 1e-7'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('All live entry-point checks passed', flush=True)


if __name__ == '__main__':
    main()
