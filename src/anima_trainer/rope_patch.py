"""Monkey-patch Anima's `apply_rotary_pos_emb` to call TE's fused RoPE kernel.

sd-scripts' `_apply_rotary_pos_emb_base` (anima_models.py:147) is six Python-
level ops: `freqs[:cur_seq_len] → cos/sin → chunk → cat(-x2, x1) → mul → add`.
For self-attention this fires on q and k at every Block × 28 blocks =
~336 elementwise launches per forward, plus the cos/sin recomputation each call.

TE's `apply_rotary_pos_emb(..., fused=True)` runs one CUDA kernel per call
(precomputes cos/sin tables internally, fuses the rotate-half + multiply).
It works on sm_120: the gating in TE for sm_120 is on FP8 GEMMs, not RoPE.

This patch overrides `library.anima_models.apply_rotary_pos_emb` so every
call from `Attention.compute_qkv` flows through the fused path.
"""
from __future__ import annotations
from typing import Union
import torch

from transformer_engine.pytorch.attention.rope import (
    apply_rotary_pos_emb as te_apply_rotary_pos_emb,
)


_PATCHED = False
_ORIG_FN = None


def _patched_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
) -> torch.Tensor:
    """Same signature as `library.anima_models.apply_rotary_pos_emb` — but
    always routes to the TE fused kernel regardless of the caller's `fused`
    argument. The Anima call sites pass `fused=False` unconditionally
    (anima_models.py:351); flipping it to True is the whole point.

    TE expects `freqs` as float32 of shape `[s2, 1, 1, d2]`. Anima already
    constructs it that way (`VideoRopePosition3DEmb`), so no reshape needed.
    """
    # Anima never passes cp_rank; TE allows omitting it (defaults to 0).
    return te_apply_rotary_pos_emb(
        t,
        freqs,
        tensor_format=tensor_format,
        start_positions=start_positions,
        interleaved=interleaved,
        fused=True,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,
    )


def install() -> None:
    """Apply the patch globally. Idempotent. Must be called after
    `ensure_on_path()` so `library.anima_models` is importable."""
    global _PATCHED, _ORIG_FN
    if _PATCHED:
        return
    from library import anima_models  # type: ignore
    _ORIG_FN = anima_models.apply_rotary_pos_emb
    anima_models.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
    _PATCHED = True


def uninstall() -> None:
    """Restore the original RoPE for A/B benchmarks."""
    global _PATCHED, _ORIG_FN
    if not _PATCHED:
        return
    from library import anima_models  # type: ignore
    anima_models.apply_rotary_pos_emb = _ORIG_FN
    _ORIG_FN = None
    _PATCHED = False
