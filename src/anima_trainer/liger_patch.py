"""Replace Anima's `RMSNorm.forward` with a fused Triton kernel from Liger.

Anima's RMSNorm (sd-scripts/library/anima_models.py:223) does:

    with torch.autocast(device_type=x.device.type, dtype=torch.float32):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

i.e. it casts to fp32, computes `x * rsqrt(mean(x^2) + eps)` in fp32, casts back
to bf16, then multiplies by weight. There are 4 RMSNorms inside every Attention
block (`q_norm`, `k_norm` on self-attn and cross-attn), 28 blocks, plus
`t_embedding_norm` → 113 calls/forward. Each does multiple kernel launches
(pow, mean, rsqrt, mul, cast, mul) plus a fp32 round-trip.

LigerRMSNormFunction fuses all of that into a single Triton kernel and matches
the LLaMA RMSNorm semantics (fp32 stats, output in input dtype, then * weight).

Only RMSNorm is patched here. Anima's LayerNorms use `elementwise_affine=False`
(no learnable weight/bias) and Liger's layer-norm kernel needs both — not
worth the special case.
"""
from __future__ import annotations
import torch
from liger_kernel.ops.rms_norm import LigerRMSNormFunction


_PATCHED = False
_ORIG_FORWARD = None


def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Anima calls RMSNorm on shapes (B, T, H, D) for per-head q/k and (B, T, D)
    # for t_embedding_norm. Liger expects (B, T, H) (3D) or (BxT, H) (2D),
    # normalizing the last dim. Flatten everything leading to a single dim.
    orig_shape = x.shape
    D = orig_shape[-1]
    out = LigerRMSNormFunction.apply(
        x.reshape(-1, D).contiguous(),
        self.weight,
        self.eps,
        0.0,       # offset
        "llama",   # casting_mode: fp32 stats, bf16 multiply with weight
        False,     # in_place: keep False; q/k tensors come from rearrange and
                   # are shared with later residual paths in some Anima variants
        None,      # row_mode (auto)
    )
    return out.reshape(orig_shape)


def install() -> None:
    """Apply the patch globally. Idempotent. Must be called after sd-scripts is
    importable on sys.path (the trainer's ensure_on_path() handles that).
    """
    global _PATCHED, _ORIG_FORWARD
    if _PATCHED:
        return
    from library.anima_models import RMSNorm  # type: ignore
    _ORIG_FORWARD = RMSNorm.forward
    RMSNorm.forward = _patched_forward
    _PATCHED = True


def uninstall() -> None:
    """Restore the original forward. Useful for benchmarks / correctness tests."""
    global _PATCHED, _ORIG_FORWARD
    if not _PATCHED:
        return
    from library.anima_models import RMSNorm  # type: ignore
    RMSNorm.forward = _ORIG_FORWARD
    _ORIG_FORWARD = None
    _PATCHED = False
