"""Anima model loading: DiT + Qwen3 text encoder + Qwen-image VAE.

We re-use sd-scripts loaders (load-bearing for correctness). All three are
returned frozen by default; the trainer thaws only the LoKr deltas it attaches
to the DiT.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .sdscripts_bridge import ensure_on_path
from .hashing import file_fingerprint


@dataclass
class LoadedModels:
    dit: torch.nn.Module
    text_encoder: torch.nn.Module
    tokenizer: object
    vae: torch.nn.Module
    vae_fp: str            # fingerprint of the VAE weights file
    te_fp: str             # fingerprint of the TE weights file
    dit_fp: str            # fingerprint of the DiT weights file


@dataclass(frozen=True)
class ModelFingerprints:
    dit: str
    vae: str
    text_encoder: str


def _weight_fp(path: str) -> str:
    _, _, h = file_fingerprint(path)
    return h


def fingerprint_model_files(
    *, dit_path: str, qwen3_path: str, vae_path: str
) -> ModelFingerprints:
    """Hash model inputs once for cache and resume compatibility checks."""
    return ModelFingerprints(
        dit=_weight_fp(dit_path),
        vae=_weight_fp(vae_path),
        text_encoder=_weight_fp(qwen3_path),
    )


def load_all(
    *,
    dit_path: str,
    qwen3_path: str,
    vae_path: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "torch",
    device: str = "cuda",
    loading_device: str = "cpu",
    fingerprints: ModelFingerprints | None = None,
) -> LoadedModels:
    """Load Anima DiT + Qwen3 TE + VAE. All frozen."""
    ensure_on_path()
    from library import anima_utils, qwen_image_autoencoder_kl  # type: ignore

    # Text encoder (frozen)
    te, tokenizer = anima_utils.load_qwen3_text_encoder(qwen3_path, dtype=dtype, device=loading_device)
    te.eval().requires_grad_(False)

    # VAE (frozen, no fp8)
    vae = qwen_image_autoencoder_kl.load_vae(
        vae_path,
        device=loading_device,
        disable_mmap=True,
        spatial_chunk_size=None,
        disable_cache=False,
    )
    vae.to(dtype).eval().requires_grad_(False)

    # DiT (trainable surface for LoKr deltas; base weights frozen)
    dit = anima_utils.load_anima_model(
        device=torch.device(device),
        dit_path=dit_path,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device=torch.device(loading_device),
        dit_weight_dtype=dtype,
        fp8_scaled=False,
    )
    dit.requires_grad_(False)  # only LoKr params are trainable; we set them grad-true on attach

    if fingerprints is None:
        fingerprints = fingerprint_model_files(
            dit_path=dit_path,
            qwen3_path=qwen3_path,
            vae_path=vae_path,
        )

    return LoadedModels(
        dit=dit,
        text_encoder=te,
        tokenizer=tokenizer,
        vae=vae,
        vae_fp=fingerprints.vae,
        te_fp=fingerprints.text_encoder,
        dit_fp=fingerprints.dit,
    )
