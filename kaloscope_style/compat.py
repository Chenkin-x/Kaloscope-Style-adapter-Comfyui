"""Read-only compatibility identifiers; never registered as public node names."""
import re

NODE_IDS = {'KaloscopeZ3EncoderLoader': 'KaloscopeStyleEncoderLoader', 'KaloscopeZ3ArtistStyle': 'KaloscopeStyleArtist', 'KaloscopeZ3ArtistMix': 'KaloscopeStyleArtistMix', 'KaloscopeZ3Loader': 'KaloscopeStyleAdapterLoader', 'KaloscopeZ3EncodeReference': 'KaloscopeStyleEncodeReference', 'KaloscopeZ3BlendFeatures': 'KaloscopeStyleBlendFeatures', 'KaloscopeZ3Apply': 'KaloscopeStyleApply', 'KaloscopeZ3Sigmas': 'KaloscopeStyleSigmas', 'KaloscopeMOSSZ3Sigmas': 'KaloscopeMOSSSigmas'}
ASSET_NAMES = {'z3_stageC_best_step63006.safetensors': 'Krea-Kaloscope-Raw.safetensors', 'z3_turbo_from63006_best_post00033000.safetensors': 'Krea-Kaloscope-Turbo.safetensors', 'moss_z3_step00008500.safetensors': 'Kaloscope-MOSS.safetensors', 'z3_artist_v1_recent.safetensors': 'artist-v1.safetensors', 'z3_artist_v1': 'artist-v1'}
ASSET_NAMES.update({
    'z3_stageA_final_step42006.safetensors': 'Krea-Kaloscope-Raw-step42006.safetensors',
    'z3_step12006.safetensors': 'Krea-Kaloscope-Raw-step12006.safetensors',
    'z3_turbo_from63006_best_post00011000.safetensors': 'Krea-Kaloscope-Turbo-step11000.safetensors',
    'z3_turbo_from63006_best_post00028000.safetensors': 'Krea-Kaloscope-Turbo-step28000.safetensors',
})

ADAPTER_FORMAT = 'kaloscope_z3_v1'
BANK_FORMAT = 'kaloscope_z3_artist_bank_v1'
LEGACY_ADAPTER_DIRECTORY = 'kaloscope_z3'
LEGACY_ADAPTER_SOCKET = 'KALOSCOPE_Z3_ADAPTER'

def public_asset_name(name):
    if '/' in name:
        return '/'.join(public_asset_name(part) for part in name.split('/'))
    if name in ASSET_NAMES:
        return ASSET_NAMES[name]
    # Preserve unknown checkpoints as separately selectable files without research labels.
    return re.sub(r'(?i)(?<![a-z0-9])z3(?=[_./\\ -]|$)', 'style', name)

def public_text(text):
    return re.sub(r'(?i)(?<![a-z0-9])z3(?=[_./\\ -]|$)', 'Style', text).replace('KaloscopeZ3', 'KaloscopeStyle')
