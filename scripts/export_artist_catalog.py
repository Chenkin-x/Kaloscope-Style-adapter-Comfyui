"""Export a supported-name CSV from a Z3 bank; optionally include verified aliases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--aliases', type=Path, help='Verified JSON with a mapping of alias to canonical artist')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/artists.csv')
    args = parser.parse_args()
    alias_data = json.loads(args.aliases.read_text()) if args.aliases else {}
    aliases = {}
    for old, new in alias_data.get('mapping', {}).items():
        if old != new:
            aliases.setdefault(new, []).append(old)
    with safe_open(str(args.bank), framework='pt') as bank:
        meta = bank.metadata()
        assert meta['format'] == 'kaloscope_z3_artist_bank_v1'
        names = json.loads(meta['names'])
        counts = bank.get_tensor('counts').tolist()
        concentration = bank.get_tensor('concentration').tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out, lineterminator='\n')
        writer.writerow(['name', 'aliases', 'families', 'concentration'])
        for i, name in enumerate(names):
            writer.writerow([name, json.dumps(sorted(aliases.get(name, [])), ensure_ascii=False), counts[i], f'{concentration[i]:.8f}'])
    report = dict(artists=len(names), aliases=sum(len(aliases.get(n, [])) for n in names),
                  bank_name=args.bank.name, bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),
                  encoder_sha256=meta['encoder_sha256'], source=meta.get('source'),
                  alias_source=alias_data.get('source'), alias_source_sha256=alias_data.get('source_sha256'),
                  csv_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  families_definition='Unique works used in the recent artist prototype; not all training images.')
    args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
