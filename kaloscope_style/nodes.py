"""Generation nodes with shared Kaloscope model-folder and socket compatibility."""
import json
import math
from pathlib import Path
import torch
import folder_paths
from .adapter import AfterBlock, StyleGuidance
from .checkpoint import load_adapter
from .features import blend, encode
from .schedule import sigma_schedule
from .artists import artist_feature
from .artist_mix import mix_artists
from .paths import adapter_paths, bank_paths


def artist_bank_paths():
    return bank_paths(folder_paths.models_dir)


class KaloscopeStyleArtist:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'bank': (list(artist_bank_paths()),),
                             'artist_name': ('STRING', {'default': 'chahan_(fried_rice0614)', 'dynamicPrompts': False,
                                                       'tooltip': 'Use Search artists for names, aliases and fuzzy suggestions. Full disambiguation suffixes are preserved.'})}}

    RETURN_TYPES = ('TENSOR', 'STRING')
    RETURN_NAMES = ('style_features', 'artist_info')
    FUNCTION = 'lookup'
    CATEGORY = 'Kaloscope Adapter'

    @classmethod
    def IS_CHANGED(cls, bank, artist_name):
        stat = artist_bank_paths()[bank].stat()
        return (stat.st_mtime_ns, stat.st_size, artist_name)

    def lookup(self, bank, artist_name):
        path = artist_bank_paths().get(bank)
        if path is None:
            raise FileNotFoundError('Place an artist bank under ComfyUI/models/kaloscope_style_banks/.')
        return artist_feature(path, artist_name)



class KaloscopeStyleArtistMix:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'bank': (list(artist_bank_paths()),),
                             'artist_tags': ('STRING', {'default': 'rolua, (wanke:0.8), (nineo:0.5)',
                                                        'multiline': True, 'dynamicPrompts': False})}}

    RETURN_TYPES = ('TENSOR', 'STRING')
    RETURN_NAMES = ('style_features', 'artist_info')
    FUNCTION = 'mix'
    CATEGORY = 'Kaloscope Adapter'

    @classmethod
    def IS_CHANGED(cls, bank, artist_tags):
        stat = artist_bank_paths()[bank].stat()
        return (stat.st_mtime_ns, stat.st_size, artist_tags)

    def mix(self, bank, artist_tags):
        path = artist_bank_paths().get(bank)
        if path is None:
            raise FileNotFoundError('Place an artist bank under ComfyUI/models/kaloscope_style_banks/.')
        return mix_artists(path, artist_tags)


def checkpoint_paths():
    return adapter_paths(folder_paths.models_dir)


class KaloscopeStyleAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'checkpoint': (list(checkpoint_paths()),)}}

    RETURN_TYPES = ('KALOSCOPE_STYLE_ADAPTER', 'STRING')
    RETURN_NAMES = ('adapter', 'checkpoint_info')
    FUNCTION = 'load'
    CATEGORY = 'Kaloscope Adapter'

    @classmethod
    def IS_CHANGED(cls, checkpoint):
        p = checkpoint_paths()[checkpoint]
        stat = p.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def load(self, checkpoint):
        from comfy.model_patcher import ModelPatcher
        from comfy import model_management, ops
        path = checkpoint_paths().get(checkpoint)
        if path is None:
            raise FileNotFoundError(f'Place {checkpoint} under ComfyUI/models/kaloscope_adapters/.')
        bundle = load_adapter(path, operations=ops.manual_cast)
        bundle['patcher'] = ModelPatcher(bundle['adapter'], load_device=model_management.get_torch_device(),
                                         offload_device=model_management.unet_offload_device())
        info = dict(step=bundle['step'], checkpoint=checkpoint, feature_dim=256, backend=bundle['backend'], injection_blocks=len(bundle['config']['injection_indices']),
                    banks=4, parameters=sum(p.numel() for p in bundle['adapter'].parameters()))
        return bundle, json.dumps(info, ensure_ascii=False)


class KaloscopeStyleEncodeReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference_images': ('IMAGE',), 'model': ('KALOSCOPE_MODEL',)}}

    RETURN_TYPES = ('TENSOR',)
    RETURN_NAMES = ('style_features',)
    FUNCTION = 'encode'
    CATEGORY = 'Kaloscope Adapter'

    def encode(self, reference_images, model):
        return (encode(reference_images, model),)


class KaloscopeStyleBlendFeatures:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'features': ('TENSOR',), 'weights': ('STRING', {'default': ''})},
                'optional': {'features_b': ('TENSOR',), 'features_c': ('TENSOR',)}}

    RETURN_TYPES = ('TENSOR',)
    RETURN_NAMES = ('blended_style',)
    FUNCTION = 'mix'
    CATEGORY = 'Kaloscope Adapter'

    def mix(self, features, weights='', features_b=None, features_c=None):
        from .features import normalized
        rows = [normalized(f) for f in (features, features_b, features_c) if f is not None]
        return (blend(torch.cat(rows), weights),)


class KaloscopeStyleApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'adapter': ('KALOSCOPE_STYLE_ADAPTER',),
                'style_features': ('TENSOR',),
                'style_guidance': ('FLOAT', {'default': 5.5, 'min': 0., 'max': 20., 'step': .1}),
                'adapter_strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 3., 'step': .05})}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'Kaloscope Adapter'

    def apply(self, model, adapter, style_features, style_guidance=5.5, adapter_strength=1.):
        if adapter['backend'] == 'moss':
            from .moss import apply_moss
            return (apply_moss(model, adapter, style_features, style_guidance, adapter_strength),)
        from comfy.ldm.krea2.model import SingleStreamDiT
        for value in (style_guidance, adapter_strength):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Guidance and strength must be finite and nonnegative.')
        base = model.get_model_object('diffusion_model')
        if not isinstance(base, SingleStreamDiT) or len(base.blocks) != 28 or base.first.out_features != 6144:
            raise ValueError('Style requires the native ComfyUI Krea2 28-block, width-6144 MODEL.')
        if model.model_options.get('kaloscope_style'):
            raise ValueError('Style already applied. Blend references before one Apply node, or branch from the original MODEL.')
        clone = model.clone()
        if style_guidance == 0 or adapter_strength == 0:
            return (clone,)
        if any(key in model.model_options for key in ('sampler_cfg_function', 'sampler_calc_cond_batch_function')):
            raise ValueError('Style needs independent text/style CFG; remove the other custom CFG patch first.')
        features = blend(style_features)
        for index, bank in zip(adapter['config']['injection_indices'], adapter['config']['injection_slots']):
            name = f'diffusion_model.blocks.{index}.forward'
            if name in model.object_patches:
                raise ValueError(f'Another node already replaces {name}.')
            original = model.get_model_object(name)
            clone.add_object_patch(name, AfterBlock(original, adapter['adapter'], features, bank, adapter_strength))
        clone.set_additional_models('kaloscope_style', [adapter['patcher']])
        clone.set_model_sampler_cfg_function(StyleGuidance(style_guidance), disable_cfg1_optimization=True)
        clone.model_options['kaloscope_style'] = {'step': adapter['step'], 'style_guidance': style_guidance,
                                             'adapter_strength': adapter_strength}
        return (clone,)


class KaloscopeStyleSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'steps': ('INT', {'default': 28, 'min': 1, 'max': 1000}),
                'start_sigma': ('FLOAT', {'default': .85, 'min': 0.001, 'max': 1., 'step': .01})},
                'optional': {'latent': ('LATENT',), 'model_variant': (['raw', 'turbo'], {'default': 'raw'})}}

    RETURN_TYPES = ('SIGMAS',)
    FUNCTION = 'schedule'
    CATEGORY = 'Kaloscope Adapter'

    def schedule(self, steps=28, start_sigma=.85, latent=None, model_variant='raw'):
        return (sigma_schedule(steps, start_sigma, latent, model_variant),)


from .encoder import KaloscopeStyleEncoderLoader

NODE_CLASS_MAPPINGS = {
    'KaloscopeStyleEncoderLoader': KaloscopeStyleEncoderLoader,
    'KaloscopeStyleArtist': KaloscopeStyleArtist,
    'KaloscopeStyleArtistMix': KaloscopeStyleArtistMix,
    'KaloscopeStyleAdapterLoader': KaloscopeStyleAdapterLoader,
    'KaloscopeStyleEncodeReference': KaloscopeStyleEncodeReference,
    'KaloscopeStyleBlendFeatures': KaloscopeStyleBlendFeatures,
    'KaloscopeStyleApply': KaloscopeStyleApply,
    'KaloscopeStyleSigmas': KaloscopeStyleSigmas,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'KaloscopeStyleEncoderLoader': 'Kaloscope Style Encoder Loader',
    'KaloscopeStyleArtist': 'Kaloscope Artist Style',
    'KaloscopeStyleArtistMix': 'Kaloscope Artist Mix',
    'KaloscopeStyleAdapterLoader': 'Kaloscope Adapter Loader',
    'KaloscopeStyleEncodeReference': 'Kaloscope Reference Encode',
    'KaloscopeStyleBlendFeatures': 'Kaloscope Blend Styles',
    'KaloscopeStyleApply': 'Kaloscope Apply Style',
    'KaloscopeStyleSigmas': 'Kaloscope Krea2 Sampling Schedule',
}

from .moss import KaloscopeMOSSBaseLoader, KaloscopeMOSSSigmas
NODE_CLASS_MAPPINGS['KaloscopeMOSSBaseLoader'] = KaloscopeMOSSBaseLoader
NODE_DISPLAY_NAME_MAPPINGS['KaloscopeMOSSBaseLoader'] = 'Kaloscope MOSS Base Loader'
NODE_CLASS_MAPPINGS['KaloscopeMOSSSigmas'] = KaloscopeMOSSSigmas
NODE_DISPLAY_NAME_MAPPINGS['KaloscopeMOSSSigmas'] = 'Kaloscope MOSS Sampling Schedule'
