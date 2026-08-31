"""Small fused Triton kernels used by the T-LoKr structured path.

The large rank projections are Transformer Engine FP8 GEMMs.  Between them,
LoKr needs an 8x8 Kronecker-factor projection and T-LoKr needs a per-example
rank mask.  Sending those through separate elementwise and batched-cuBLAS
kernels is launch-heavy, so these kernels fuse both operations.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rank_mix_kernel(
    hidden,
    weight,
    mask,
    out,
    n_per_batch: tl.constexpr,
    rank: tl.constexpr,
    total: tl.constexpr,
    HAS_MASK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total
    row = offsets // rank
    rank_idx = offsets - row * rank

    acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK,), dtype=tl.float32)

    for j in tl.static_range(0, 8):
        h = tl.load(hidden + (row * 8 + j) * rank + rank_idx, mask=valid)
        if TRANSPOSE:
            acc0 += h * tl.load(weight + j * 8 + 0)
            acc1 += h * tl.load(weight + j * 8 + 1)
            acc2 += h * tl.load(weight + j * 8 + 2)
            acc3 += h * tl.load(weight + j * 8 + 3)
            acc4 += h * tl.load(weight + j * 8 + 4)
            acc5 += h * tl.load(weight + j * 8 + 5)
            acc6 += h * tl.load(weight + j * 8 + 6)
            acc7 += h * tl.load(weight + j * 8 + 7)
        else:
            acc0 += h * tl.load(weight + 0 * 8 + j)
            acc1 += h * tl.load(weight + 1 * 8 + j)
            acc2 += h * tl.load(weight + 2 * 8 + j)
            acc3 += h * tl.load(weight + 3 * 8 + j)
            acc4 += h * tl.load(weight + 4 * 8 + j)
            acc5 += h * tl.load(weight + 5 * 8 + j)
            acc6 += h * tl.load(weight + 6 * 8 + j)
            acc7 += h * tl.load(weight + 7 * 8 + j)

    if HAS_MASK:
        batch_idx = row // n_per_batch
        scale = tl.load(mask + batch_idx * rank + rank_idx, mask=valid)
        acc0 *= scale
        acc1 *= scale
        acc2 *= scale
        acc3 *= scale
        acc4 *= scale
        acc5 *= scale
        acc6 *= scale
        acc7 *= scale

    tl.store(out + (row * 8 + 0) * rank + rank_idx, acc0, mask=valid)
    tl.store(out + (row * 8 + 1) * rank + rank_idx, acc1, mask=valid)
    tl.store(out + (row * 8 + 2) * rank + rank_idx, acc2, mask=valid)
    tl.store(out + (row * 8 + 3) * rank + rank_idx, acc3, mask=valid)
    tl.store(out + (row * 8 + 4) * rank + rank_idx, acc4, mask=valid)
    tl.store(out + (row * 8 + 5) * rank + rank_idx, acc5, mask=valid)
    tl.store(out + (row * 8 + 6) * rank + rank_idx, acc6, mask=valid)
    tl.store(out + (row * 8 + 7) * rank + rank_idx, acc7, mask=valid)


@triton.jit
def _rank_mix_wgrad_kernel(
    grad_mixed,
    hidden,
    mask,
    out,
    n_per_batch: tl.constexpr,
    rank: tl.constexpr,
    reduce_size: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    out_idx = pair // 8
    in_idx = pair - out_idx * 8
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < reduce_size
    row = offsets // rank
    rank_idx = offsets - row * rank
    grad = tl.load(
        grad_mixed + (row * 8 + out_idx) * rank + rank_idx,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        hidden + (row * 8 + in_idx) * rank + rank_idx,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    if HAS_MASK:
        batch_idx = row // n_per_batch
        value *= tl.load(
            mask + batch_idx * rank + rank_idx,
            mask=valid,
            other=0.0,
        )
    partial = tl.sum(grad * value, axis=0)
    tl.atomic_add(out + pair, partial)


def rank_mix(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    """Return ``weight[@T] @ (hidden * mask)`` for 8x8 ``weight``."""
    if hidden.shape[-2] != 8 or weight.shape != (8, 8):
        masked = hidden
        if mask is not None:
            view = (mask.shape[0],) + (1,) * (hidden.ndim - 2) + (mask.shape[1],)
            masked = hidden * mask.view(view)
        matrix = weight.transpose(0, 1) if transpose else weight
        return torch.matmul(matrix, masked)
    if not hidden.is_cuda:
        masked = hidden
        if mask is not None:
            view = (mask.shape[0],) + (1,) * (hidden.ndim - 2) + (mask.shape[1],)
            masked = hidden * mask.view(view)
        matrix = weight.transpose(0, 1) if transpose else weight
        return torch.matmul(matrix, masked)

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(hidden)
    batch, n_per_batch, _, rank = hidden.shape
    total = batch * n_per_batch * rank
    block = 256
    _rank_mix_kernel[(triton.cdiv(total, block),)](
        hidden,
        weight,
        mask if mask is not None else hidden,
        out,
        n_per_batch=n_per_batch,
        rank=rank,
        total=total,
        HAS_MASK=mask is not None,
        TRANSPOSE=transpose,
        BLOCK=block,
    )
    return out

def rank_mix_wgrad(
    grad_mixed: torch.Tensor,
    hidden: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Fused fp32 reduction for the 8x8 Kronecker-factor gradient."""
    if grad_mixed.shape[-2] != 8 or hidden.shape[-2] != 8:
        masked = hidden
        if mask is not None:
            view = (mask.shape[0],) + (1,) * (hidden.ndim - 2) + (mask.shape[1],)
            masked = hidden * mask.view(view)
        return torch.einsum("bnir,bnjr->ij", grad_mixed, masked)
    if not grad_mixed.is_cuda:
        masked = hidden
        if mask is not None:
            view = (mask.shape[0],) + (1,) * (hidden.ndim - 2) + (mask.shape[1],)
            masked = hidden * mask.view(view)
        return torch.einsum("bnir,bnjr->ij", grad_mixed, masked)

    grad_mixed = grad_mixed.contiguous()
    hidden = hidden.contiguous()
    out = torch.zeros((8, 8), device=hidden.device, dtype=torch.float32)
    batch, n_per_batch, _, rank = hidden.shape
    reduce_size = batch * n_per_batch * rank
    block = 1024
    _rank_mix_wgrad_kernel[(64, triton.cdiv(reduce_size, block))](
        grad_mixed,
        hidden,
        mask if mask is not None else hidden,
        out,
        n_per_batch=n_per_batch,
        rank=rank,
        reduce_size=reduce_size,
        HAS_MASK=mask is not None,
        BLOCK=block,
    )
    return out
