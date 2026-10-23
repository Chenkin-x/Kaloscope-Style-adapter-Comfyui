"""Checkpoint-compatible Style side adapter; native Krea2 modulation is unchanged."""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F


class StyleAdapter(nn.Module):
    def __init__(self, dim=256, width=256, tokens=8, layers=4, hidden=6144, operations=nn):
        super().__init__()
        self.width, self.tokens, self.heads = width, tokens, width // 64
        self.projector = nn.Sequential(operations.LayerNorm(dim), operations.Linear(dim, width), nn.SiLU(),
                                       operations.Linear(width, tokens * width))
        self.norm = operations.LayerNorm(width)
        self.key, self.value = operations.Linear(width, width), operations.Linear(width, width)
        self.query = nn.ModuleList([operations.Linear(hidden, width) for _ in range(layers)])
        self.out = nn.ModuleList([operations.Linear(width, hidden, bias=False) for _ in range(layers)])

    def encode(self, style):
        return self.norm(self.projector(style).reshape(style.shape[0], self.tokens, self.width))

    def residual(self, image, style, bank):
        b = image.shape[0]
        q = self.query[bank](image).reshape(b, -1, self.heads, 64).transpose(1, 2)
        k = self.key(style).reshape(b, -1, self.heads, 64).transpose(1, 2)
        v = self.value(style).reshape(b, -1, self.heads, 64).transpose(1, 2)
        delta = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, self.width)
        return self.out[bank](delta)


def precision_context(x):
    if x.device.type in ('cuda', 'cpu') and x.dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(x.device.type, dtype=x.dtype)
    return contextlib.nullcontext()


class AfterBlock:
    """Object patch owned by a ModelPatcher clone, never a persistent forward hook."""
    def __init__(self, original, adapter, features, bank, strength):
        self.original, self.adapter, self.features = original, adapter, features
        self.bank, self.strength = bank, strength

    def __call__(self, *args, **kwargs):
        x = self.original(*args, **kwargs)
        options = kwargs.get('transformer_options', {})
        if options.get('kaloscope_style_disabled', False) or self.strength == 0:
            return x
        if options.get('reference_image_num_tokens'):
            raise ValueError('Style does not support Krea2 reference_latents/ICL in the same model call.')
        span = options.get('img_slice')
        if span is None:
            raise RuntimeError('Update ComfyUI: Krea2 must expose transformer_options[img_slice].')
        start, end = span
        flags = options.get('cond_or_uncond')
        if not flags or x.shape[0] % len(flags):
            raise ValueError('Style requires a standard ComfyUI CFG sampler with cond_or_uncond metadata.')
        active = torch.tensor([flag == 0 for flag in flags], device=x.device)
        active = active.repeat_interleave(x.shape[0] // len(flags))
        if not active.any():
            return x
        # Style has no independently trained AdaLN; inject AFTER each native block.
        with precision_context(x):
            style = self.adapter.encode(self.features.to(device=x.device, dtype=torch.float32))
            style = style.expand(x.shape[0], -1, -1)
            delta = self.adapter.residual(x[:, start:end], style, self.bank)
            delta = delta * active[:, None, None] * self.strength
        return torch.cat((x[:, :start], x[:, start:end] + delta.to(x.dtype), x[:, end:]), dim=1)


def guided_denoised(unconditional, text_only, styled, text_cfg, style_cfg):
    return unconditional + text_cfg * (text_only - unconditional) + style_cfg * (styled - text_only)


class StyleGuidance:
    def __init__(self, style_cfg):
        self.style_cfg = style_cfg

    def __call__(self, args):
        from comfy.samplers import calc_cond_batch
        options = dict(args['model_options'])
        options['transformer_options'] = dict(options.get('transformer_options', {}), kaloscope_style_disabled=True)
        text_only = calc_cond_batch(args['model'], [args['input_cond']], args['input'], args['sigma'], options)[0]
        result = guided_denoised(args['uncond_denoised'], text_only, args['cond_denoised'],
                                args['cond_scale'], self.style_cfg)
        # sampler_cfg_function returns input - denoised (not velocity or denoised).
        return args['input'] - result
