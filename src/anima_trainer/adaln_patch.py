"""Monkey-patch Anima's `Block._forward` to call custom Triton kernels for the
AdaLN modulation pattern `LayerNorm(x) * (1 + scale) + shift` and the
gated-residual pattern `x = x + gate * sublayer(...)`.

Why: the AdaLN modulation fires at 3 sites per Block × 28 blocks = 84 times
per forward. Stock PyTorch issues it as three separate kernels
(`layer_norm`, `mul`, `add`). Profiler showed `aten::layer_norm` alone was
~195 ms/step (7%), with the chained mul+add adding more. These ops are
launch-bound — fusing the chain collapses 3 launches into 1 per site.

The three gated-residual sites in this same forward are the matching
fusion target: each was `mul + add` = 2 kernels × 3 sites × 28 blocks =
168 launches. Replaced with one `FusedGatedAdd` Triton kernel per site → 84
launches dropped per forward.

Independently, the three `adaln_modulation_*` Sequentials per Block can be
collapsed via :mod:`adaln_merge` into one SiLU + one fused first Linear +
one batched `bmm` over the second Linears. The patched forward below
detects whether merge has been applied (`_adaln_modulation_merged` flag)
and dispatches accordingly.

Earlier this used `torch.compile` on a 3-op pure function. That worked but
paid an inductor warmup per new bucket resolution and added a runtime
dependency we couldn't poke. The current path uses hand-written Triton
kernels (forward + backward) in `adaln_kernel.py` — no warmup, explicit
backward, no inductor in the loop.
"""
from __future__ import annotations
import contextlib
from typing import Optional
import torch
from einops import rearrange

from .adaln_kernel import FusedAdaLN, FusedGatedAdd
from .adaln_merge import apply_merged_adaln


_PATCHED = False
_ORIG_FORWARD = None


def _patched_forward(
    self,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    attn_params,
    use_fp32: bool = False,
    rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    extra_per_block_pos_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replacement for `anima_models.Block._forward` (sd-scripts).

    Identical control flow to upstream; the only change is the AdaLN
    application — instead of the inline `_adaln_fn(...)` it calls our
    fused Triton kernel via `FusedAdaLN.apply(x, scale, shift, eps)`.
    """
    if use_fp32:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

    ctx = (
        torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32)
        if use_fp32
        else contextlib.nullcontext()
    )
    with ctx:
        if getattr(self, "_adaln_modulation_merged", False):
            # Merged path: 1 SiLU + 1 Linear + 1 bmm (lora) or 1 SiLU + 1
            # Linear (no-lora) → 9 modulation outputs in three triples.
            (shift_self, scale_self, gate_self), \
                (shift_cross, scale_cross, gate_cross), \
                (shift_mlp, scale_mlp, gate_mlp) = apply_merged_adaln(
                    self, emb_B_T_D, adaln_lora_B_T_3D if self.use_adaln_lora else None,
                )
        elif self.use_adaln_lora:
            shift_self, scale_self, gate_self = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(
                emb_B_T_D
            ).chunk(3, dim=-1)

    # (B, T, D) -> (B, T, 1, 1, D)
    shift_self_r = rearrange(shift_self, "b t d -> b t 1 1 d")
    scale_self_r = rearrange(scale_self, "b t d -> b t 1 1 d")
    gate_self_r = rearrange(gate_self, "b t d -> b t 1 1 d")
    shift_cross_r = rearrange(shift_cross, "b t d -> b t 1 1 d")
    scale_cross_r = rearrange(scale_cross, "b t d -> b t 1 1 d")
    gate_cross_r = rearrange(gate_cross, "b t d -> b t 1 1 d")
    shift_mlp_r = rearrange(shift_mlp, "b t d -> b t 1 1 d")
    scale_mlp_r = rearrange(scale_mlp, "b t d -> b t 1 1 d")
    gate_mlp_r = rearrange(gate_mlp, "b t d -> b t 1 1 d")

    B, T, H, W, D = x_B_T_H_W_D.shape

    # All three Anima LayerNorms have elementwise_affine=False so weight/bias
    # are None; only eps matters for parity. normalized_shape is implicit
    # in the kernel (last dim = D).
    eps_self = self.layer_norm_self_attn.eps
    eps_cross = self.layer_norm_cross_attn.eps
    eps_mlp = self.layer_norm_mlp.eps

    # 1. Self-attention
    normalized_x = FusedAdaLN.apply(x_B_T_H_W_D, scale_self_r, shift_self_r, eps_self)
    result = rearrange(
        self.self_attn(
            rearrange(normalized_x, "b t h w d -> b (t h w) d"),
            attn_params,
            None,
            rope_emb=rope_emb_L_1_1_D,
        ),
        "b (t h w) d -> b t h w d",
        t=T, h=H, w=W,
    )
    x_B_T_H_W_D = FusedGatedAdd.apply(x_B_T_H_W_D, gate_self_r, result)

    # 2. Cross-attention
    normalized_x = FusedAdaLN.apply(x_B_T_H_W_D, scale_cross_r, shift_cross_r, eps_cross)
    result = rearrange(
        self.cross_attn(
            rearrange(normalized_x, "b t h w d -> b (t h w) d"),
            attn_params,
            crossattn_emb,
            rope_emb=rope_emb_L_1_1_D,
        ),
        "b (t h w) d -> b t h w d",
        t=T, h=H, w=W,
    )
    x_B_T_H_W_D = FusedGatedAdd.apply(x_B_T_H_W_D, gate_cross_r, result)

    # 3. MLP
    normalized_x = FusedAdaLN.apply(x_B_T_H_W_D, scale_mlp_r, shift_mlp_r, eps_mlp)
    result = self.mlp(normalized_x)
    x_B_T_H_W_D = FusedGatedAdd.apply(x_B_T_H_W_D, gate_mlp_r, result)

    return x_B_T_H_W_D


def install() -> None:
    """Apply the patch globally. Idempotent. Must be called after sd-scripts is
    importable on sys.path (the trainer's `ensure_on_path()` handles that).
    """
    global _PATCHED, _ORIG_FORWARD
    if _PATCHED:
        return
    from library.anima_models import Block  # type: ignore
    _ORIG_FORWARD = Block._forward
    Block._forward = _patched_forward
    _PATCHED = True


def uninstall() -> None:
    """Restore the original forward. Useful for A/B benchmarks."""
    global _PATCHED, _ORIG_FORWARD
    if not _PATCHED:
        return
    from library.anima_models import Block  # type: ignore
    Block._forward = _ORIG_FORWARD
    _ORIG_FORWARD = None
    _PATCHED = False
