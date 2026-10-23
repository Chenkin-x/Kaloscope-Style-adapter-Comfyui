"""Named artist prototypes in the frozen Style 256-D projector space."""
import json
from safetensors import safe_open
from .features import normalized
from .catalog import ENCODER_SHA256, artist_index, file_version


def artist_feature(path, artist_name):
    version = file_version(path)
    catalog = artist_index(path)
    index = catalog.resolve(artist_name)
    row = catalog.rows[index]
    with safe_open(str(path), framework='pt', device='cpu') as bank:
        feature = bank.get_slice('means')[index:index+1].clone()
        info = dict(artist=row['name'], training_families=row['families'], source=catalog.bank_source,
                    conditioning='named prototype; not native text-encoder knowledge')
    if file_version(path) != version:
        raise ValueError('Artist bank changed during lookup. Run the node again.')
    return normalized(feature), json.dumps(info, ensure_ascii=False)
