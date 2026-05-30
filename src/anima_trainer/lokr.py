"""Attach a LoKr adapter to the Anima DiT via lycoris.kohya.

lycoris was originally designed for SD/SDXL UNets but accepts arbitrary
nn.Modules under its `unet` argument — the term is anatomical, not literal.
We pass the Anima DiT here. Reference args mirror the baseline:
  algo=lokr, full_matrix=true, factor=8, preset=full

Network dim/alpha match the baseline (128/128). If a profile run shows
lycoris is the throughput bottleneck, we'll write a sm120-targeted LoKr; until
then we lean on the reference impl.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from .config import LokrCfg


_ANIMA_PRESETS_REGISTERED = False


def _register_anima_presets() -> None:
    """Register lycoris presets that target Anima's actual modules.

    Two custom presets:

    - `anima-full`: every Linear in every Block (cross+self attn + MLP +
      adaln_modulation), ~454 modules / ~30M trainable params. Wide capacity,
      slowest step time. This is what `factor=8` was tuned against.
    - `anima-cross-mlp`: cross-attention q/k/v/output + MLP layer1/layer2 only,
      ~168 modules / ~17M trainable params. Highest-leverage subset for
      multi-character/multi-style LoRAs: cross-attn carries text→image
      conditioning (character identity, style tokens), MLP carries channel
      mixing (style/texture). Self-attn (composition) is left to the base.

    Note: `unet_target_name` uses fnmatch globs because we flip
    `LycorisNetworkKohya.USE_FNMATCH = True` at attach time.
    """
    global _ANIMA_PRESETS_REGISTERED
    if _ANIMA_PRESETS_REGISTERED:
        return
    from lycoris.config import PRESET

    PRESET["anima-full"] = {
        "enable_conv": False,
        "unet_target_module": [
            "Block",                # 28 Anima transformer blocks (recursively wraps attn + MLP)
            "FinalLayer",           # the 3 already-wrapped Linears
        ],
        "unet_target_name": [],
        "text_encoder_target_module": [],
        "text_encoder_target_name": [],
    }

    PRESET["anima-cross-mlp"] = {
        "enable_conv": False,
        "unet_target_module": [],
        "unet_target_name": [
            # 28 × {q,k,v,output}_proj = 112 cross-attn Linears
            "blocks.*.cross_attn.q_proj",
            "blocks.*.cross_attn.k_proj",
            "blocks.*.cross_attn.v_proj",
            "blocks.*.cross_attn.output_proj",
            # 28 × {layer1, layer2} = 56 MLP Linears (GPT2FeedForward)
            "blocks.*.mlp.layer1",
            "blocks.*.mlp.layer2",
        ],
        "text_encoder_target_module": [],
        "text_encoder_target_name": [],
    }
    _ANIMA_PRESETS_REGISTERED = True


_ANIMA_PRESET_NAMES = {"anima-full", "anima-cross-mlp"}


def attach_lokr(
    dit: nn.Module,
    cfg: LokrCfg,
    *,
    network_dim: int = 128,
    network_alpha: float = 128.0,
    multiplier: float = 1.0,
    text_encoder: Optional[nn.Module] = None,
) -> nn.Module:
    """Wrap the DiT with a LyCORIS LoKr network. Returns the network module.

    The returned `network` carries the trainable parameters; you optimize over
    network.parameters() rather than dit.parameters().
    """
    # Patch lycoris.LokrModule.forward to do 1 mm per call instead of 2.
    from .lokr_patch import install as install_lokr_patch
    install_lokr_patch()

    from lycoris.kohya import create_network, LycorisNetworkKohya

    if cfg.preset in _ANIMA_PRESET_NAMES:
        _register_anima_presets()
    # `anima-cross-mlp` matches by fully-qualified module name with glob
    # patterns (e.g. `blocks.*.cross_attn.q_proj`); lycoris defaults to
    # `re.match` which wouldn't treat `.` and `*` as we want. fnmatch is the
    # right semantic. Setting this is idempotent and safe for `anima-full`
    # too (it doesn't use name patterns).
    LycorisNetworkKohya.USE_FNMATCH = True

    network = create_network(
        multiplier,
        network_dim,
        network_alpha,
        None,           # vae (unused for our purposes)
        text_encoder,   # may be None — we never train TE
        dit,            # the "unet" slot, in lycoris' parlance
        algo="lokr",
        full_matrix=cfg.full_matrix,
        factor=cfg.factor,
        preset=cfg.preset,
        train_norm=False,
        # bypass_mode=True (decomposed forward, doesn't materialize the full
        # Kronecker product) was tested empirically — same step time within
        # noise, ~17% MORE peak VRAM. The lycoris materialize-then-linear path
        # is already efficient on Anima's shapes.
    )
    network.apply_to(text_encoder, dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    return network


def trainable_param_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
