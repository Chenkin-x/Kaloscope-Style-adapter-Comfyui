"""Krea2 raw resolution-aware exponential time shift (f8 VAE, 2x2 patches)."""
import math
import torch


def resolution_mu(latent_height, latent_width):
    tokens = math.ceil(latent_height / 2) * math.ceil(latent_width / 2)
    return .5 + (tokens - 256) * (1.15 - .5) / (6400 - 256)


def sigma_schedule(steps=28, start_sigma=.85, latent=None, model_variant='raw'):
    if steps < 1 or not 0 < start_sigma <= 1:
        raise ValueError('steps >= 1 and 0 < start_sigma <= 1 required.')
    # No latent preserves the original 512-square workflow exactly.
    if model_variant not in ('raw', 'turbo'):
        raise ValueError('model_variant must be raw or turbo.')
    mu = (1.15 if model_variant == 'turbo' else
          .58125 if latent is None else resolution_mu(*latent['samples'].shape[-2:]))
    shift = math.exp(mu)
    u = torch.linspace(start_sigma / (shift * (1-start_sigma) + start_sigma), 0, steps+1)
    sigmas = shift * u / (1 + (shift-1) * u)
    sigmas[0] = start_sigma
    return sigmas
