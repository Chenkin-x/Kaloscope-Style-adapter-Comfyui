"""MOSS native ComfyUI block patches and independent text/style guidance."""
import math
from pathlib import Path
import torch
import folder_paths
from .adapter import guided_denoised, precision_context
from .features import blend


class KaloscopeMOSSBaseLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model_path': ('STRING', {'default': 'MOSS-Image-Base'}),
            'weight_dtype': (['default', 'bf16', 'fp32'], {'default': 'bf16'}),
            'text_device': (['default', 'cpu'], {'default': 'default'}),
        }}

    RETURN_TYPES = ('MODEL', 'CLIP', 'VAE')
    FUNCTION = 'load'
    CATEGORY = 'Kaloscope/Style Transfer'

    def load(self, model_path='MOSS-Image-Base', weight_dtype='bf16', text_device='default'):
        import nodes
        loader = nodes.NODE_CLASS_MAPPINGS.get('MOSSImageHFLoader')
        if loader is None:
            raise RuntimeError('Install the official MOSS-Image ComfyUI plugin (moss-image), then restart ComfyUI.')
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = Path(folder_paths.models_dir) / path
        for name in ('transformer', 'text_encoder', 'tokenizer', 'vae'):
            if not (path / name).is_dir():
                raise FileNotFoundError(f'MOSS model directory is missing {name}: {path}')
        return loader.execute(str(path), weight_dtype, text_device).result


class KaloscopeMOSSZ3Sigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'steps': ('INT', {'default': 50, 'min': 1, 'max': 1000}),
                             'shift': ('FLOAT', {'default': 5., 'min': .01, 'max': 100.})}}

    RETURN_TYPES = ('SIGMAS',)
    FUNCTION = 'schedule'
    CATEGORY = 'Kaloscope/Style Transfer'

    def schedule(self, steps=50, shift=5.):
        if not 1 <= steps <= 1000 or not math.isfinite(shift) or shift <= 0:
            raise ValueError('Require 1–1000 steps and a finite positive shift.')
        # MOSSImageScheduler: shifted inference grid snapped to training_shift=1.
        values = torch.linspace(1., shift / (1000 + shift - 1), steps)
        sigmas = shift * values / (1 + (shift - 1) * values)
        training = torch.linspace(1., .001, 1000).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=999)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas-training[lower]).abs() <= (training[upper]-sigmas).abs(), lower, upper)
        return (torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1))),)


class MOSSAfterBlock:
    def __init__(self, adapter, features, bank, strength):
        self.adapter, self.features, self.bank, self.strength = adapter, features, bank, strength

    def __call__(self, args, extra):
        result = extra['original_block'](args)
        options = args['transformer_options']
        if not options.get('kaloscope_z3_enabled', False):
            return result
        n = options['kaloscope_z3_image_tokens']
        x = result['img']
        with precision_context(x):
            style = self.adapter.encode(self.features.to(device=x.device, dtype=torch.float32))
            delta = self.adapter.residual(x[:, :n], style.expand(x.shape[0], -1, -1), self.bank)
        return {**result, 'img': torch.cat((x[:, :n] + (delta * self.strength).to(x.dtype), x[:, n:]), dim=1)}


def skip_block(args, extra):
    return args


def moss_cond_batch(args):
    """Native positive and native negative, skipping block 22 only for negative."""
    from comfy.samplers import calc_cond_batch
    from comfy.model_patcher import set_model_options_patch_replace
    model, conds, x, sigma = args['model'], args['conds'], args['input'], args['sigma']
    options = args['model_options']
    positive = calc_cond_batch(model, [conds[0], None], x, sigma, options)[0]
    negative_options = set_model_options_patch_replace(dict(options), skip_block, 'dit', 'single_block', 22)
    negative = calc_cond_batch(model, [None, conds[1]], x, sigma, negative_options)[1]
    return [positive, negative]


class MOSSStyleGuidance:
    def __init__(self, style_cfg):
        self.style_cfg = style_cfg

    def __call__(self, args):
        from comfy.samplers import calc_cond_batch
        native, negative = args['cond_denoised'], args['uncond_denoised']
        if self.style_cfg == 0:
            return args['input'] - (negative + args['cond_scale'] * (native - negative))
        x = args['input']
        options = dict(args['model_options'])
        options['transformer_options'] = dict(options.get('transformer_options', {}),
            kaloscope_z3_enabled=True, kaloscope_z3_image_tokens=x.shape[-2] * x.shape[-1])
        styled = calc_cond_batch(args['model'], [args['input_cond']], x, args['sigma'], options)[0]
        return x - guided_denoised(negative, native, styled, args['cond_scale'], self.style_cfg)


def apply_moss(model, bundle, features, style_guidance, adapter_strength):
    for value in (style_guidance, adapter_strength):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Guidance and strength must be finite and nonnegative.')
    base = model.get_model_object('diffusion_model')
    if (type(base).__name__ != 'MOSSImageTransformer' or len(base.layers) != 42
            or base.video_embedder.out_features != 3072 or base.latent_channels != 128):
        raise ValueError('MOSS Z3 requires the official MOSS-Image 42-layer, width-3072 MODEL; use MOSS Base Loader.')
    if model.model_options.get('kaloscope_z3'):
        raise ValueError('Z3 already applied; blend styles before one Apply node.')
    if any(key in model.model_options for key in ('sampler_cfg_function', 'sampler_calc_cond_batch_function')):
        raise ValueError('MOSS Z3 includes negative skip22 and independent CFG; remove other CFG/SkipLayerGuidance patches.')
    patches = model.model_options.get('transformer_options', {}).get('patches_replace', {}).get('dit', {})
    if any(('single_block', i) in patches for i in range(42)):
        raise ValueError('Another node already replaces a MOSS transformer block.')
    clone = model.clone()
    enabled = style_guidance > 0 and adapter_strength > 0
    if enabled:
        features = blend(features)
        for index, bank in zip(bundle['config']['injection_indices'], bundle['config']['injection_slots']):
            clone.set_model_patch_replace(MOSSAfterBlock(bundle['adapter'], features, bank, adapter_strength),
                                          'dit', 'single_block', index)
        clone.set_additional_models('kaloscope_z3', [bundle['patcher']])
    clone.set_model_sampler_calc_cond_batch_function(moss_cond_batch)
    clone.set_model_sampler_cfg_function(MOSSStyleGuidance(style_guidance if enabled else 0), disable_cfg1_optimization=True)
    clone.model_options['kaloscope_z3'] = dict(backend='moss', step=bundle['step'],
        style_guidance=style_guidance, adapter_strength=adapter_strength, uncond_skip_layers=[22])
    return clone
