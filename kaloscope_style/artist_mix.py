"""Parse artist tags and mix bank prototypes through the existing blend path."""
import json
import math
import re
import torch
from .artists import artist_feature
from .features import blend


def parse_artist_mix(text):
    entries = []
    start = depth = 0
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                raise ValueError('Unmatched parentheses in artist tags.')
        elif char in ',，\n' and depth == 0:
            entries.append(text[start:i].strip())
            start = i + 1
    if depth or escaped:
        raise ValueError('Unclosed parentheses or trailing escape in artist tags.')
    entries.append(text[start:].strip())
    result = []
    for entry in entries:
        if not entry:
            continue
        name, weight = entry, 1.0
        if entry.startswith('(') and entry.endswith(')') and ':' in entry:
            name, number = entry[1:-1].rsplit(':', 1)
            try:
                weight = float(number.strip())
            except ValueError as exc:
                raise ValueError(f'Invalid artist weight in {entry!r}; use (artist:0.8).') from exc
        name = re.sub(r'\\([\\(),:])', r'\1', name.strip())
        if not name or not math.isfinite(weight) or weight < 0:
            raise ValueError('Artist names must be nonempty and weights finite and nonnegative.')
        result.append((name, weight))
    if not result or not any(w > 0 for _, w in result):
        raise ValueError('Supply at least one artist with a positive weight.')
    return result


def mix_artists(path, artist_tags):
    parsed = parse_artist_mix(artist_tags)
    scale = max(w for _, w in parsed)
    features, weights, details = [], [], []
    for name, weight in parsed:
        feature, info = artist_feature(path, name)
        features.append(feature)
        weights.append(weight / scale)
        details.append(dict(json.loads(info), weight=weight))
    total = sum(weights)
    for info, weight in zip(details, weights):
        info['relative_weight'] = weight / total
    mixed = blend(torch.cat(features), ','.join(str(w / total) for w in weights))
    return mixed, json.dumps(dict(artists=details, mixing='normalized weighted sum of bank prototypes'), ensure_ascii=False)
