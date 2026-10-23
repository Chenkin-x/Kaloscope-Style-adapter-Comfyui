from .compat import BANK_FORMAT, public_asset_name, public_text
"""Local artist names and search. No model weights are retained by this module."""
import csv
import difflib
import json
from functools import lru_cache
from pathlib import Path
from safetensors import safe_open

ENCODER_SHA256 = 'c370c49c7a7c1a68b2942b8b3aab080c25384a67c9d652d2005e15c8ab8860dd'
CATALOG = Path(__file__).resolve().parents[1] / 'data' / 'artists.csv'


def normalize_name(name):
    return name.strip().casefold().replace(' ', '_')


def file_version(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@lru_cache(maxsize=2)
def _csv_rows(version):
    with open(version[0], encoding='utf-8-sig', newline='') as stream:
        return tuple(dict(name=r['name'], aliases=json.loads(r['aliases']),
                          families=int(r['families']), concentration=float(r['concentration']))
                     for r in csv.DictReader(stream))


def bundled_rows():
    return _csv_rows(file_version(CATALOG))


class ArtistIndex:
    def __init__(self, rows, source, available):
        self.rows = tuple(rows)
        self.source, self.available = source, available
        self.by_name = {r['name']: i for i, r in enumerate(self.rows)}
        self.exact = {}
        for i, row in enumerate(self.rows):
            for spelling in [row['name'], *row['aliases']]:
                key = normalize_name(spelling)
                matches = self.exact.setdefault(key, [])
                if i not in matches:
                    matches.append(i)
        self.keys = tuple(self.exact)

    def resolve(self, name):
        canonical = name.strip().replace(' ', '_')
        if canonical in self.by_name:
            return self.by_name[canonical]
        matches = self.exact.get(normalize_name(name), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ', '.join(self.rows[i]['name'] for i in matches)
            raise ValueError(f'Ambiguous artist {name!r}; choose the full name: {choices}')
        choices = ', '.join(r['name'] for r in self.search(name, 5)['results'])
        hint = f' Suggestions: {choices}.' if choices else ''
        raise ValueError(f'Unknown artist {name!r}.{hint} Use artist search to select an exact name.')

    def search(self, query, limit=40):
        query = normalize_name(query)
        limit = max(1, min(int(limit), 100))
        if len(query) > 160:
            raise ValueError('Search text must be at most 160 characters.')
        hits = {}
        for key in self.keys:
            if not query:
                kind, score = 'browse', 0
            elif key == query:
                kind, score = 'exact', 1.0
            elif key.startswith(query):
                kind, score = 'prefix', .95
            elif query in key:
                kind, score = 'substring', .55 + .25 * len(query) / len(key)
            else:
                continue
            for i in self.exact[key]:
                if i not in hits or score > hits[i][0]:
                    hits[i] = (score, kind)
        exact_total = len(hits)
        if len(query) >= 3 and len(hits) < limit:
            for key in difflib.get_close_matches(query, self.keys, n=limit, cutoff=.65):
                score = .85 * difflib.SequenceMatcher(None, query, key).ratio()
                for i in self.exact[key]:
                    if i not in hits:
                        hits[i] = (score, 'fuzzy')
        ranked = sorted(hits, key=lambda i: (-hits[i][0], self.rows[i]['name']))[:limit]
        return dict(source=self.source, available=self.available, total_artists=len(self.rows),
                    exact_substring_matches=exact_total,
                    results=[dict(self.rows[i], match=hits[i][1], score=round(hits[i][0], 4)) for i in ranked])


@lru_cache(maxsize=4)
def _bank_index(version, csv_version):
    aliases = {r['name']: r['aliases'] for r in _csv_rows(csv_version)}
    with safe_open(version[0], framework='pt', device='cpu') as bank:
        meta = bank.metadata() or {}
        if meta.get('format') != BANK_FORMAT or meta.get('encoder_sha256') != ENCODER_SHA256:
            raise ValueError('Artist bank must use the matching Style artist-v1 encoder.')
        names = json.loads(meta['names'])
        if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError('Artist bank names must be unique, nonempty strings.')
        if bank.get_slice('means').get_shape() != [len(names), 256] or bank.get_slice('counts').get_shape() != [len(names)]:
            raise ValueError('Artist bank names, counts and 256-D features do not align.')
        counts = bank.get_tensor('counts').tolist()
        concentrations = bank.get_tensor('concentration').tolist() if 'concentration' in bank.keys() else [None] * len(names)
        rows = [dict(name=n, aliases=aliases.get(n, []), families=int(counts[i]), concentration=concentrations[i]) for i, n in enumerate(names)]
        index = ArtistIndex(rows, public_asset_name(Path(version[0]).name), True)
        index.bank_source = public_text(meta.get('source') or '')
        return index


@lru_cache(maxsize=2)
def _bundled_index(version):
    return ArtistIndex(_csv_rows(version), 'bundled CSV (no bank selected)', False)


def artist_index(path=None):
    version = file_version(CATALOG)
    return _bundled_index(version) if path is None else _bank_index(file_version(path), version)
