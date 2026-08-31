"""MXFP8 frozen-base quantization for LoKr training on sm_120 Blackwell.

# Architecture

LoKr already gives us a frozen base + trainable bf16 deltas. That maps
cleanly onto a quantization strategy:

  - **Base weight** (`LokrModule.org_module.weight`): bf16, FROZEN.
    Quantize **once at attach time** to MXFP8 (E4M3 with E8M0 32-element
    block scales). Keep both the bf16 and MXFP8 copies on device:

      * MXFP8 copy → forward GEMM via `tex.general_gemm` (~1.3–1.5× faster
        than bf16 at production shapes).
      * bf16 copy → backward dgrad (`grad_x = grad_y @ W_bf16`). Frozen,
        so no wgrad. This sidesteps the broken sm_120 MXFP8 dgrad path
        (see "Why bypass te.Linear" below).

    Memory: ~1.5× the original base weight size. On Anima DiT (~2.4 GB
    bf16) this is ~3.6 GB — comfortably under the 96 GB ceiling.

  - **LoKr delta** (`get_weight(...) → kron(W1, W2)`): bf16, TRAINABLE.
    Stays in bf16. The LoKr math is a small fraction of the total flops;
    quantizing the deltas would lose precision on the only signal we're
    actually learning.

  - **Saved adapter**: always bf16. `_save_lora` already casts to bf16
    before writing; quantization is forward-time only.

# Forward decomposition

Stock LoKr-patched forward at every wrapped Linear:

  merged_W = org.weight + α · diff_W              # bf16 add (4M–16M elem)
  y = F.linear(x, merged_W, org.bias)             # bf16 GEMM

MXFP8 path:

  y_base  = MXFP8FrozenLinear(x, W_mx, W_bf16)    # MXFP8 GEMM, frozen
  y_delta = F.linear(x, α · diff_W, None)         # bf16 GEMM, small (k=128 LoKr rank surface)
  y       = y_base + y_delta + org.bias?

This is mathematically the "bypass mode" decomposition (`x @ W.T = x @ (base + diff).T`
= `x @ base.T + x @ diff.T`), but the heavy half now runs in MXFP8 instead
of bf16. We previously rejected bypass mode for adding ~17% VRAM at no
speed gain — that calculus changes here: VRAM goes UP (extra MXFP8 copy)
but step time goes DOWN.

# Why bypass `te.Linear`

`te.Linear(...)` under `fp8_autocast(MXFP8BlockScaling())` runs forward
fine on sm_120, but the **backward** dgrad GEMM fails with
`CUBLAS_STATUS_NOT_SUPPORTED` for every Anima production shape — cuBLAS
13.5 (the latest) doesn't have a heuristic for the transposed-layout
MXFP8 GEMM on sm_120. That's exactly what the
`check_mxfp8_support()` "MXFP8 (for all gemm layouts) is not supported on
12.0+ architectures yet" gate is gating.

We don't need a backward MXFP8 GEMM. The base weight is frozen → no
wgrad. We only need a dgrad, and the bf16 dgrad against the cached bf16
weight is trivial (it's the same GEMM PyTorch would run for any frozen
Linear's backward). So we drop `te.Linear` and call the low-level
`tex.general_gemm` directly in forward, with a custom autograd.Function
that routes backward through bf16.

# Reproducibility notes

  - On sm_120, `_compute_mxfp8_support()` in
    `transformer_engine/pytorch/quantization.py` returns False with a
    "12.0+ architectures yet" message. The kernels themselves execute
    (forward path tested end-to-end here). We patch the cached gate
    result at module import time so our code paths that build
    `MXFP8Quantizer` instances don't trip the public check.
  - Some TE forward paths JIT-compile small helper kernels via NVRTC at
    runtime and need `cuda_runtime.h`. Set `NVTE_CUDA_INCLUDE_DIR` to
    point at the system or pip-installed CUDA headers; we set it at
    import to `/opt/cuda/include` if unset.
"""
from __future__ import annotations
import os
from typing import Iterable
import torch
import torch.nn as nn

# --- One-time TE setup ------------------------------------------------------

# Default NVTE_CUDA_INCLUDE_DIR to the most common path on Arch + NVIDIA
# toolkit installs. Users with a non-standard CUDA layout can pre-set the env
# var; we only fill it in when unset.
os.environ.setdefault("NVTE_CUDA_INCLUDE_DIR", "/opt/cuda/include")

import transformer_engine.pytorch as te                           # noqa: E402
import transformer_engine.pytorch.cpp_extensions as tex_ext       # noqa: E402
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor  # noqa: E402
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer  # noqa: E402
from transformer_engine.pytorch import quantization as teq        # noqa: E402
from transformer_engine.common.recipe import Float8BlockScaling   # noqa: E402
import transformer_engine_torch as tex                            # noqa: E402

# Bypass the conservative sm_120 gate. The gate's premise is "MXFP8 ... not
# supported on 12.0+ yet", but the forward layout works on sm_120 with the
# current cuBLAS; we only ever use that layout. The gate is patched here, not
# at the call site, so users of MXFP8Quantizer/general_gemm elsewhere in the
# process aren't surprised.
teq._MXFP8_SUPPORT = (True, "")


# Shared quantizers. MXFP8Quantizer is essentially a config object (fp8 dtype
# + rowwise/columnwise usage flags), not stateful per tensor — one instance
# can quantize many tensors.
_ROWWISE_QUANTIZER = MXFP8Quantizer(
    tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
)


# --- Autograd op -----------------------------------------------------------


class MXFP8FrozenLinear(torch.autograd.Function):
    """`y = x @ W.T (+ bias)` where `W` is frozen and pre-quantized to MXFP8.

    Forward:
      * Quantize `x` rowwise to MXFP8.
      * Call `tex.general_gemm(W_mx, x_mx, layout='TN')` which computes
        `out = x_mx @ W_mx.T` in MXFP8 → bf16 output.
      * Add bias in bf16 if provided.

    Backward:
      * `grad_x = grad_y @ W_bf16` — plain bf16 GEMM.
      * No `grad_W` (W is frozen, saved as a buffer not a Parameter).
      * `grad_bias = grad_y.sum(reduce-extra-dims)` if bias is provided
        (bias usually also frozen for our LoRA training, but autograd
        still flows through it correctly if it's trainable).

    The bf16 weight + bias are held by the caller (module buffers /
    attributes); we only stash references in ctx so backward can use them.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W_mx: MXFP8Tensor,
        W_bf16: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `general_gemm(A, B, layout='TN')` returns `B @ A.T`. With A=W and
        # B=x we get `x @ W.T`, which is exactly `F.linear(x, W)`.
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        x_mx = _ROWWISE_QUANTIZER.quantize(x_2d)
        out = tex_ext.general_gemm(W_mx, x_mx, out_dtype=x.dtype, layout="TN")
        y = out[0]
        y = y.reshape(*x.shape[:-1], W_bf16.shape[0])
        if bias is not None:
            y = y + bias
        # Save the bf16 weight for dgrad. We keep it as a non-tensor attribute
        # to dodge autograd's "saved tensor was mutated" tracking (W is frozen
        # but PyTorch's autograd is conservative about modifications).
        ctx.W_bf16 = W_bf16
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        W_bf16 = ctx.W_bf16
        # grad_x may come in as fp32 (autocast boundary, gradient checkpoint
        # recompute, or loss in float32). Cast both operands to the GEMM
        # dtype that matches the saved weight; cast the result back if
        # autograd's expected gradient dtype differs.
        target_dtype = W_bf16.dtype
        grad_y_2d = grad_y.reshape(-1, grad_y.shape[-1]).to(target_dtype)
        grad_x_2d = grad_y_2d @ W_bf16
        grad_x = grad_x_2d.reshape(*grad_y.shape[:-1], W_bf16.shape[1]).to(grad_y.dtype)
        grad_bias = None
        if ctx.has_bias:
            # Standard bias backward: sum across all dims except the last.
            grad_bias = grad_y.reshape(-1, grad_y.shape[-1]).sum(dim=0)
        return grad_x, None, None, grad_bias


def mxfp8_frozen_linear(
    x: torch.Tensor,
    W_mx: MXFP8Tensor,
    W_bf16: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional wrapper around :class:`MXFP8FrozenLinear`."""
    return MXFP8FrozenLinear.apply(x, W_mx, W_bf16, bias)


class MXFP8LoKrLinear(torch.autograd.Function):
    """`y = x @ merged_W.T (+ bias)` with MXFP8 forward GEMM + bf16 backward.

    Designed for LoKr-wrapped Linears where `merged_W = base_W + α · diff_W`
    is recomputed each step and trainable through `diff_W`'s autograd graph.

    Why MXFP8 forward + bf16 backward:
      - MXFP8 forward GEMM works on sm_120 (the TN layout has a cuBLAS
        heuristic).
      - MXFP8 backward (dgrad and wgrad) hits `CUBLAS_STATUS_NOT_SUPPORTED`
        on sm_120 — no heuristic for the transposed layouts. This is what
        `_compute_mxfp8_support()` is gating against.
      - We don't need MXFP8 backward: `merged_W` is held in bf16 (it lives
        in the autograd graph for W1/W2), so dgrad and wgrad are plain
        bf16 GEMMs against the saved bf16 tensor. Same speed as any other
        bf16 backward.

    The trick: we treat `merged_W` as "frozen for this GEMM kernel" — we
    quantize it just-in-time for the forward, but the bf16 reference is
    saved in `ctx` for backward. Autograd then flows `grad_merged_W` back
    through `merged_W = base_W + α · diff_W → kron(W1, W2)` via the normal
    bf16 graph, reaching `W1` and `W2`.

    Per-call cost:
      - 1× MXFP8 quantize on x (~50–200 μs depending on shape)
      - 1× MXFP8 quantize on merged_W (~50–200 μs)
      - 1× MXFP8 GEMM (forward, ~0.7× bf16 GEMM cost)
      - Backward: 2× bf16 GEMM (dgrad + wgrad) — same as bf16 path

    Net forward saving: ~25–30 % of the GEMM cost minus the quantize
    overhead. On Anima production shapes this nets a 5–12 % step time win
    when applied to the 168 LoKr-wrapped modules.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        merged_W: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Quantize x rowwise; merged_W rowwise (we never need columnwise
        # because backward is bf16).
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        x_mx = _ROWWISE_QUANTIZER.quantize(x_2d)
        W_mx = _ROWWISE_QUANTIZER.quantize(merged_W.contiguous())
        out = tex_ext.general_gemm(W_mx, x_mx, out_dtype=x.dtype, layout="TN")
        y = out[0]
        y = y.reshape(*x.shape[:-1], merged_W.shape[0])
        if bias is not None:
            y = y + bias
        # Save bf16 copies for backward. Saving merged_W is unavoidable
        # (we need it for dgrad), x is the LoKr-wrapped Linear's input.
        ctx.save_for_backward(x, merged_W)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, merged_W = ctx.saved_tensors
        target_dtype = merged_W.dtype
        gy = grad_y.to(target_dtype)
        # dgrad: grad_x = grad_y @ merged_W   shape: (..., D_out) @ (D_out, D_in)
        gy_flat = gy.reshape(-1, gy.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        grad_x_2d = gy_flat @ merged_W
        grad_x = grad_x_2d.reshape(*grad_y.shape[:-1], merged_W.shape[1]).to(grad_y.dtype)
        # wgrad: grad_merged_W = grad_y.T @ x  shape: (D_out, D_in)
        # Cast back to merged_W's dtype so autograd is happy with the upstream
        # graph (merged_W lives in bf16 via the LoKr add).
        grad_merged_W = gy_flat.t() @ x_flat.to(target_dtype)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.reshape(-1, grad_y.shape[-1]).sum(dim=0)
        return grad_x, grad_merged_W, grad_bias


def mxfp8_lokr_linear(
    x: torch.Tensor,
    merged_W: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional wrapper around :class:`MXFP8LoKrLinear`."""
    return MXFP8LoKrLinear.apply(x, merged_W, bias)


def is_quantized(mod: nn.Module) -> bool:
    """True if `mod` has had its base weight pre-quantized for MXFP8."""
    return getattr(mod, "_mxfp8_W", None) is not None


# --- Whole-DiT frozen-Linear quantization ----------------------------------


def _mxfp8_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Forward replacement for a pre-quantized frozen `nn.Linear`.

    Bound to the instance via `linear.forward = types.MethodType(...)`. Reads
    `self._mxfp8_W` and `self._mxfp8_W_bf16`. Falls back to bf16 `F.linear`
    if:
      - the quantized weight is missing (defensive)
      - the flattened activation's leading dim isn't divisible by 32
        (MXFP8 block constraint — e.g. timestep-embedding calls where
        the batch dim is the only leading dim and is small)

    The fallback path is rare on the production training shape (x typically
    flattens to (B*T*H*W, D) which is huge and aligned), so the per-call
    branch cost is negligible.
    """
    import torch.nn.functional as F
    W_mx = getattr(self, "_mxfp8_W", None)
    W_bf16 = getattr(self, "_mxfp8_W_bf16", None)
    if W_mx is None or W_bf16 is None:
        return F.linear(x, self.weight, self.bias)
    # MXFP8 requires both flattened dims % 32 == 0. The trailing dim was
    # checked at attach time (it's the Linear's in_features); check the
    # leading dim at call time.
    leading = x.numel() // x.shape[-1]
    if leading % 32 != 0:
        return F.linear(x, W_bf16, self.bias)
    return MXFP8FrozenLinear.apply(x, W_mx, W_bf16, self.bias)


def quantize_frozen_linears(
    dit: nn.Module,
    skip: Iterable[nn.Module] = (),
    min_size: int = 256 * 256,
    skip_name_substrings: tuple[str, ...] = ("adaln_modulation", "t_embedder", "FinalLayer"),
) -> int:
    """Pre-quantize every frozen `nn.Linear` in `dit` to MXFP8 and swap its
    forward to the MXFP8 path.

    Args:
      dit: the model (DiT). We walk `dit.modules()` and find every
        `nn.Linear` whose `weight.requires_grad` is False.
      skip: a collection of `nn.Module` instances to skip (e.g. the
        `org_module[0]` of each LokrModule — those are quantized via
        `quantize_lokr_base_weights` and accessed through the LoKr path,
        not via the standalone Linear's forward).
      min_size: skip Linears smaller than this `out_dim * in_dim`
        threshold. Tiny Linears (e.g. T-embedding internals) have so
        little FLOP that quantize-x-each-call overhead outweighs the
        GEMM savings. 256*256 = 65536 elements is a conservative cut.

    Quantization replaces `linear.forward` with a bound method that calls
    `MXFP8FrozenLinear`. The original `linear.weight` stays in place
    (still bf16) so backward dgrad has it; we also attach `_mxfp8_W`
    (MXFP8 copy) and `_mxfp8_W_bf16` (reference back to `linear.weight`)
    as non-buffer attributes so they don't appear in `state_dict`.

    Returns the number of Linears quantized. Idempotent.
    """
    import types

    # MXFP8 uses 32-element block scales (E8M0); both dims must be % 32 == 0.
    MXFP8_BLOCK = 32

    skip_ids = {id(m) for m in skip}
    # Build a name set for substring skip: modules whose qualified name
    # contains any of `skip_name_substrings` are bypassed (typically
    # adaln_modulation, timestep embedder, FinalLayer — called with small
    # leading dim so MXFP8 quantization-of-x falls back to bf16 anyway,
    # and we'd just pay the branch overhead).
    skip_name_ids: set[int] = set()
    for name, mod in dit.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in skip_name_substrings):
            skip_name_ids.add(id(mod))

    count = 0
    skipped_size = 0
    skipped_align = 0
    skipped_grad = 0
    skipped_other = 0
    skipped_name = 0
    for mod in dit.modules():
        if not isinstance(mod, nn.Linear):
            continue
        if id(mod) in skip_ids:
            skipped_other += 1
            continue
        if id(mod) in skip_name_ids:
            skipped_name += 1
            continue
        if mod.weight.requires_grad:
            # Don't quantize trainable weights. Anima DiT should have all
            # Linears frozen post-LoKr-attach (LoKr's own params are on the
            # LokrModule, not on this Linear).
            skipped_grad += 1
            continue
        if getattr(mod, "_mxfp8_W", None) is not None:
            continue
        out_dim, in_dim = mod.weight.shape
        if out_dim * in_dim < min_size:
            skipped_size += 1
            continue
        if (out_dim % MXFP8_BLOCK) or (in_dim % MXFP8_BLOCK):
            # E.g. Anima's FinalLayer projects 2048 → 68 (16 latent ch + 1 mask
            # ch, ×4 patch); 68 % 32 != 0. Stays in bf16; the FLOP cost is
            # negligible compared to the per-block GEMMs anyway.
            skipped_align += 1
            continue
        # MXFP8 quantize. Quantizer expects contiguous, 2D.
        W = mod.weight.detach().contiguous()
        W_mx = _ROWWISE_QUANTIZER.quantize(W)
        mod._mxfp8_W = W_mx
        mod._mxfp8_W_bf16 = W
        mod.forward = types.MethodType(_mxfp8_linear_forward, mod)
        count += 1
    print(
        f"  quantize_frozen_linears: {count} Linears quantized, "
        f"{skipped_grad} trainable skipped, {skipped_size} too small (<{min_size} elems), "
        f"{skipped_align} unaligned (dims not %{MXFP8_BLOCK}=0), "
        f"{skipped_other} explicitly skipped, "
        f"{skipped_name} small-input by name ({', '.join(skip_name_substrings)})"
    )
    return count


def collect_lokr_wrapped_linears(network: nn.Module) -> list[nn.Linear]:
    """Return the list of original `nn.Linear` instances that LoKr has
    wrapped. Used as the `skip` set for `quantize_frozen_linears` — the
    LoKr path handles those via `quantize_lokr_base_weights`, and
    re-quantizing the bare Linear would double-quantize.
    """
    from lycoris.modules.lokr import LokrModule
    wrapped: list[nn.Linear] = []
    for mod in network.modules():
        if isinstance(mod, LokrModule):
            inner = mod.org_module[0]
            if isinstance(inner, nn.Linear):
                wrapped.append(inner)
    return wrapped


# --- TE FP8 (Float8BlockScaling) -------------------------------------------
#
# Unlike MXFP8 which we had to bypass-mode (sm_120 cuBLAS lacks heuristics for
# the MXFP8 dgrad layout), `Float8BlockScaling` (128×128 weight blocks, 1×128
# activation blocks, the recipe DeepSeek-V3 trained with) has a proper sm_120
# heuristic for both forward and backward. We can use `te.Linear` directly and
# `te.fp8_autocast` to gate it, with no custom autograd needed.
#
# Microbench on a production GEMM shape (B=32768, D_in=D_out=2048):
#   bf16   fwd+bwd 2.376 ms
#   fp8-CS fwd+bwd 3.593 ms   (Float8CurrentScaling — *slower* than bf16, per
#                              tensor amax over 64 MB activations on the
#                              backward pass dominates the GEMM saving)
#   fp8-BS fwd+bwd 1.534 ms   (Float8BlockScaling — 1.55× bf16)
#
# So we use Float8BlockScaling exclusively for the `fp8` precision mode.


# Shared recipe singleton: stateless, safe to reuse.
_FP8_BLOCK_RECIPE = Float8BlockScaling()


# Float8BlockScaling-style quantizers, mirroring the recipe TE applies
# inside `te.fp8_autocast(Float8BlockScaling())`:
#   - activations: 1×128 block scaling (`block_scaling_dim=1`), E4M3
#   - weights:     128×128 block scaling (`block_scaling_dim=2`), E4M3
# Both quantizers produce a tensor with both rowwise and columnwise scale
# views, so the same quantizer instance feeds forward, dgrad, and wgrad.
_FP8_X_QUANTIZER = Float8BlockQuantizer(
    tex.DType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=1
)
_FP8_W_QUANTIZER = Float8BlockQuantizer(
    tex.DType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=2
)


class FP8LoKrLinear(torch.autograd.Function):
    """`y = x @ merged_W.T (+ bias)` with FP8 forward + FP8 dgrad + FP8 wgrad.

    Designed for LoKr-wrapped Linears where `merged_W = base_W + α·diff_W`
    is recomputed each step and trainable through `diff_W`'s autograd graph.

    Unlike `MXFP8LoKrLinear` (which has to do bf16 backward because sm_120
    cuBLAS lacks MXFP8 dgrad heuristics), Float8BlockScaling has working
    sm_120 heuristics for all three layouts (forward TN, dgrad NN,
    wgrad NT). So we can run the entire op in FP8 — three GEMMs that are
    each ~25-30% cheaper than bf16. Net at the MLP-layer1 shape
    (32768×2048×5440): bf16 6.39 ms → fp8 5.33 ms = 1.20× fwd+bwd.

    Gradient noise vs bf16: ~2.5-2.8% relative error on x.grad / W1.grad /
    W2.grad (the FP8 quantization noise floor at 128-block scaling). LoRA
    training is *very* tolerant of this — the per-step learning rate is much
    smaller than the noise, and Prodigy+SF adapts. Same gradient quality
    that DeepSeek-V3 trained successfully with on much larger models.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, merged_W: torch.Tensor, bias: torch.Tensor | None):
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        xq = _FP8_X_QUANTIZER.quantize(x_2d.contiguous())
        wq = _FP8_W_QUANTIZER.quantize(merged_W.contiguous())
        out = tex_ext.general_gemm(wq, xq, out_dtype=x.dtype, layout="TN")[0]
        y = out.reshape(*x.shape[:-1], merged_W.shape[0])
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x_2d, merged_W)
        ctx.has_bias = bias is not None
        ctx.orig_shape = x.shape
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x_2d, merged_W = ctx.saved_tensors
        gy = grad_y.reshape(-1, grad_y.shape[-1]).to(merged_W.dtype).contiguous()
        # Re-quantize gy + reuse merged_W/x quantizations. Re-quantizing is
        # cheap relative to the GEMMs themselves on the MLP shapes.
        gy_q = _FP8_X_QUANTIZER.quantize(gy)
        wq = _FP8_W_QUANTIZER.quantize(merged_W.contiguous())
        xq = _FP8_X_QUANTIZER.quantize(x_2d.contiguous())
        # dgrad: grad_x = grad_y @ merged_W   →  general_gemm(W, gy, layout='NN')
        grad_x_2d = tex_ext.general_gemm(wq, gy_q, out_dtype=grad_y.dtype, layout="NN")[0]
        grad_x = grad_x_2d.reshape(*grad_y.shape[:-1], merged_W.shape[1])
        # wgrad: grad_merged_W = grad_y.T @ x  →  general_gemm(x, gy, layout='NT')
        grad_merged_W = tex_ext.general_gemm(xq, gy_q, out_dtype=merged_W.dtype, layout="NT")[0]
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.reshape(-1, grad_y.shape[-1]).sum(dim=0)
        return grad_x, grad_merged_W, grad_bias


def fp8_lokr_linear(x: torch.Tensor, merged_W: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Functional wrapper around :class:`FP8LoKrLinear`."""
    return FP8LoKrLinear.apply(x, merged_W, bias)


class FP8FrozenLinear(torch.autograd.Function):
    """Frozen ``F.linear`` with a once-quantized FP8 block-scaled weight.

    T-LoKr computes its trainable contribution with structured bf16 GEMMs, so
    the base projection is genuinely frozen and never needs a wgrad.  Keeping
    the weight quantized across steps removes both the full merged-weight
    materialization and FP8 weight quantization performed by ordinary LoKr.
    Forward and dgrad use Transformer Engine's fused block-scaled GEMMs.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_q,
        weight_bf16: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        xq = _FP8_X_QUANTIZER.quantize(x_2d.contiguous())
        out = tex_ext.general_gemm(
            weight_q,
            xq,
            out_dtype=x.dtype,
            layout="TN",
        )[0]
        y = out.reshape(*x.shape[:-1], weight_bf16.shape[0])
        if bias is not None:
            y = y + bias
        ctx.weight_q = weight_q
        ctx.weight_bf16 = weight_bf16
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        weight = ctx.weight_bf16
        gy = grad_y.reshape(-1, grad_y.shape[-1]).to(weight.dtype).contiguous()
        gy_q = _FP8_X_QUANTIZER.quantize(gy)
        grad_x_2d = tex_ext.general_gemm(
            ctx.weight_q,
            gy_q,
            out_dtype=weight.dtype,
            layout="NN",
        )[0]
        grad_x = grad_x_2d.reshape(*grad_y.shape[:-1], weight.shape[1])
        if grad_x.dtype != grad_y.dtype:
            grad_x = grad_x.to(grad_y.dtype)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.reshape(-1, grad_y.shape[-1]).sum(dim=0)
        return grad_x, None, None, grad_bias


def fp8_frozen_linear(
    x: torch.Tensor,
    weight_q,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Functional wrapper for :class:`FP8FrozenLinear`."""
    return FP8FrozenLinear.apply(x, weight_q, weight_bf16, bias)


def quantize_tlokr_base_weights(network: nn.Module) -> int:
    """Prequantize every T-LoKr wrapped base Linear exactly once.

    Float8BlockScaling requires both weight dimensions to be divisible by
    128.  Anima's cross-attention and MLP targets all meet that contract; a
    mismatch is an error instead of a silent precision/performance fallback.
    """
    from lycoris.modules.lokr import LokrModule

    count = 0
    for module in network.modules():
        if not isinstance(module, LokrModule) or not getattr(
            module, "_tlokr_enabled", False
        ):
            continue
        if getattr(module, "_tlokr_fp8_base_weight", None) is not None:
            continue
        inner = module.org_module[0]
        if not isinstance(inner, nn.Linear):
            raise TypeError(f"{module.lora_name}: FP8 T-LoKr requires nn.Linear")
        out_dim, in_dim = inner.weight.shape
        if (out_dim % 128) or (in_dim % 128):
            raise ValueError(
                f"{module.lora_name}: FP8 T-LoKr base shape "
                f"{tuple(inner.weight.shape)} is not 128-aligned"
            )
        module._tlokr_fp8_base_weight = _FP8_W_QUANTIZER.quantize(
            inner.weight.detach().contiguous()
        )
        count += 1
    return count


def fp8_block_autocast():
    """Context manager that enables `te.fp8_autocast` with Float8BlockScaling.

    Apply around any forward through a DiT whose frozen Linears have been
    swapped via :func:`swap_frozen_linears_to_te` — otherwise `te.Linear`
    silently falls back to bf16.
    """
    return te.fp8_autocast(enabled=True, fp8_recipe=_FP8_BLOCK_RECIPE)


def _replace_child(parent: nn.Module, child_name: str, new: nn.Module) -> None:
    """Replace `getattr(parent, child_name)` (which may be a plain attribute
    or a registered submodule) with `new`. PyTorch's `__setattr__` handles
    the dispatch; we just need to be sure the old name is freed first to
    avoid lingering parameter registration."""
    setattr(parent, child_name, new)


def _build_te_linear(src: nn.Linear, device: torch.device) -> te.Linear:
    """Construct a `te.Linear` with the same shape + frozen weights as `src`.

    `src.weight` is copied into the new module under `no_grad`; both
    `weight` and `bias` (if present) have `requires_grad=False` to match
    the frozen-base contract.
    """
    out_dim, in_dim = src.weight.shape
    has_bias = src.bias is not None
    new = te.Linear(
        in_dim,
        out_dim,
        bias=has_bias,
        params_dtype=src.weight.dtype,
        device=device,
    )
    with torch.no_grad():
        new.weight.copy_(src.weight.to(device))
        if has_bias:
            new.bias.copy_(src.bias.to(device))
    new.weight.requires_grad_(False)
    if has_bias:
        new.bias.requires_grad_(False)
    return new


def patch_anima_checkpoint_for_fp8() -> None:
    """Swap `library.anima_models.torch_checkpoint` for `te.checkpoint`.

    Why: under `fp8_autocast`, `te.Linear` records FP8 metadata on the first
    forward and reuses it on subsequent calls. `torch.utils.checkpoint`
    recomputes the forward during backward without informing TE — the
    recompute creates *fresh* FP8 metadata, the saved-tensor counts diverge
    (forward: 88, recompute: 76 — seen in practice on Anima Block), and
    PyTorch raises CheckpointError.

    TE's `te.checkpoint` knows about the FP8 state and replays it during
    recompute. Drop-in replacement for `torch_checkpoint` in the Anima
    block's `_forward_wrap` path.

    Must be called *after* `ensure_on_path()` so `library.anima_models` is
    importable. Idempotent.
    """
    from library import anima_models  # type: ignore
    if getattr(anima_models, "_fp8_checkpoint_patched", False):
        return
    anima_models.torch_checkpoint = te.checkpoint
    anima_models._fp8_checkpoint_patched = True


def swap_frozen_linears_to_te(
    dit: nn.Module,
    skip: Iterable[nn.Module] = (),
    min_size: int = 256 * 256,
    skip_name_substrings: tuple[str, ...] = ("adaln_modulation", "t_embedder", "FinalLayer"),
) -> int:
    """Replace every frozen `nn.Linear` in `dit` with a `te.Linear` so the
    DiT forward runs in FP8 under `fp8_block_autocast()`.

    Same skip rules as :func:`quantize_frozen_linears`:
      - `skip`: LoKr-wrapped inner Linears (LoKr forward materializes
        `merged_W = base + α·diff_W` and runs bf16 `F.linear` against it;
        swapping the inner Linear doesn't help because LoKr never calls its
        forward).
      - `min_size`: skip tiny Linears where per-tensor quantize overhead
        eats the GEMM saving.
      - `skip_name_substrings`: small-input modules (timestep embedder,
        AdaLN modulation, FinalLayer) — same shape-alignment story as MXFP8.

    Float8BlockScaling requires both dims be divisible by 128 (block size);
    Linears that don't fit are left as bf16.

    Returns the number of Linears swapped. Idempotent.
    """
    FP8_BLOCK = 128

    skip_ids = {id(m) for m in skip}
    skip_name_ids: set[int] = set()
    for name, mod in dit.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in skip_name_substrings):
            skip_name_ids.add(id(mod))

    # First pass: collect (parent, child_name, child) tuples that need swap.
    # We can't mutate dit during named_modules() iteration safely.
    to_swap: list[tuple[nn.Module, str, nn.Linear]] = []
    skipped_size = 0
    skipped_align = 0
    skipped_grad = 0
    skipped_other = 0
    skipped_name = 0
    for parent in dit.modules():
        for child_name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if id(child) in skip_ids:
                skipped_other += 1
                continue
            if id(child) in skip_name_ids:
                skipped_name += 1
                continue
            if child.weight.requires_grad:
                skipped_grad += 1
                continue
            out_dim, in_dim = child.weight.shape
            if out_dim * in_dim < min_size:
                skipped_size += 1
                continue
            if (out_dim % FP8_BLOCK) or (in_dim % FP8_BLOCK):
                skipped_align += 1
                continue
            to_swap.append((parent, child_name, child))

    device = next(dit.parameters()).device
    count = 0
    for parent, child_name, child in to_swap:
        new = _build_te_linear(child, device)
        _replace_child(parent, child_name, new)
        count += 1

    print(
        f"  swap_frozen_linears_to_te: {count} Linears swapped to te.Linear, "
        f"{skipped_grad} trainable skipped, {skipped_size} too small (<{min_size} elems), "
        f"{skipped_align} unaligned (dims not %{FP8_BLOCK}=0), "
        f"{skipped_other} explicitly skipped, "
        f"{skipped_name} small-input by name ({', '.join(skip_name_substrings)})"
    )
    return count
