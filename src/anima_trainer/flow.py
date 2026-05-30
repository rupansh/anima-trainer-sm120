"""Rectified-flow timestep sampling and target construction for Anima.

Mirrors sd-scripts: timestep_sampling="sigmoid" with sigmoid_scale=1.0.
The forward call expects timesteps in [0, 1000] (continuous), with the same
convention used by the DiT's t_embedder.
"""
from __future__ import annotations
import torch


def sample_sigmoid_timesteps(batch: int, *, scale: float = 1.0, device: torch.device) -> torch.Tensor:
    """Logit-normal timestep sampling: t = sigmoid(scale * z), z ~ N(0,1). Returns t in (0,1)."""
    z = torch.randn(batch, device=device)
    return torch.sigmoid(scale * z)


def noisy_input_and_target(
    latents: torch.Tensor,   # (B, C, H, W) in latent space
    *,
    sigmoid_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (noisy_input, target, timesteps, sigmas) for one batch.

    Rectified-flow target is `noise - latents`. The Anima DiT t_embedder
    expects timesteps in [0, 1] (sd-scripts scales `sigmas * 1000` down to
    [0, 1] before calling the DiT; we just pass sigmas directly).
    """
    b = latents.shape[0]
    device = latents.device
    noise = torch.randn_like(latents)
    sigmas = sample_sigmoid_timesteps(b, scale=sigmoid_scale, device=device)
    s = sigmas.view(b, *([1] * (latents.ndim - 1))).to(latents.dtype)
    noisy = (1.0 - s) * latents + s * noise
    target = noise - latents
    timesteps = sigmas  # [0, 1] range, matches Anima DiT t_embedder convention
    return noisy, target, timesteps, sigmas
