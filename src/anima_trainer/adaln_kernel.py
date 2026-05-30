"""Custom Triton kernels for Anima's AdaLN modulation.

The pattern at every Block site is

    LayerNorm(x, elementwise_affine=False, eps) * (1 + scale) + shift

where `x` is (B, T, H, W, D) and (scale, shift) are (B, T, 1, 1, D) — i.e.
broadcast across the (H, W) spatial dims for each (b, t) pair. There are 84
such sites per forward (3 per Block × 28 blocks). Stock PyTorch emits this
as three separate kernels (layer_norm → mul → add), which is launch-bound.

This file fuses each site into:

  - 1 Triton kernel on forward (LayerNorm stats + (1+s)*xhat + b in one pass)
  - 2 Triton kernels on backward (dx row-wise; dscale/dshift HW-reduction)

It supersedes the earlier `torch.compile`-based fusion: no inductor warmup
per resolution, no compile-cache state, and an explicit backward.

Numerical model: all stats and elementwise math run in fp32 internally. The
output, dx, dscale, and dshift are stored back in the input dtype (typically
bf16 — Anima's training dtype). This matches PyTorch's autocast-bf16
LayerNorm policy (fp32 stats, output cast to bf16); the trailing mul/add
that PyTorch would do in bf16 we instead keep in fp32 until the store, which
is *more* accurate, not less. Differences from eager-mode bf16 math are
strictly below the bf16 noise floor.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    X, S, B, Y, Mean, Rstd,
    HW, D, eps,
    BLOCK_D: tl.constexpr,
):
    """Forward: y = (x - mean) * rstd * (1 + scale) + shift, per (b,t,h,w) row.

    Each program handles one row of D elements. `bt = row // HW` selects which
    (b, t) pair's scale/shift to broadcast — this is the *only* place the
    AdaLN broadcasting is encoded.
    """
    row = tl.program_id(0)
    bt = row // HW

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    # Mask the centered values so the var sum doesn't pick up `(-mean)^2`
    # contributions from the padded BLOCK_D > D tail.
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd

    s = tl.load(S + bt * D + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + bt * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = xhat * (1.0 + s) + b

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    tl.store(Y + row * D + cols, y, mask=mask)


@triton.jit
def _bwd_dx_kernel(
    dY, X, S, Mean, Rstd, dX,
    HW, D,
    BLOCK_D: tl.constexpr,
):
    """Per-row dx using saved (mean, rstd). Standard LN backward with the
    effective LayerNorm weight set to `(1 + scale)`.

        c1 = mean(dxhat)
        c2 = mean(dxhat * xhat)
        dx = rstd * (dxhat - c1 - xhat * c2)
    """
    row = tl.program_id(0)
    bt = row // HW

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(dY + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S + bt * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    dxhat = tl.where(mask, dy * (1.0 + s), 0.0)

    c1 = tl.sum(dxhat, axis=0) / D
    c2 = tl.sum(dxhat * xhat, axis=0) / D
    dx = rstd * (dxhat - c1 - xhat * c2)

    tl.store(dX + row * D + cols, dx, mask=mask)


@triton.jit
def _bwd_dsdb_kernel(
    dY, X, Mean, Rstd, dS, dB,
    HW, D,
    BLOCK_D: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """Partial reduction of dscale, dshift over (H, W) for each (b, t):

        dscale[bt] += sum over BLOCK_HW rows of (dy * xhat)
        dshift[bt] += sum over BLOCK_HW rows of dy

    Parallelism: 3-D grid (bt, hw_block, d_block). Each program reduces
    BLOCK_HW rows, atomic-adds into fp32 destination buffers `dS, dB` of
    shape (BT, D). With B*T tiny (often 8), splitting HW *and* D gives us
    O(thousands) of programs — enough to saturate Blackwell's 100+ SMs
    instead of the serial-loop 64-program version this replaced.

    Destinations must be zero-initialized fp32; the wrapper casts back to
    the user dtype after the kernel.
    """
    bt = tl.program_id(0)
    hw_block = tl.program_id(1)
    d_block = tl.program_id(2)

    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = cols < D

    ds = tl.zeros((BLOCK_D,), dtype=tl.float32)
    db = tl.zeros((BLOCK_D,), dtype=tl.float32)

    hw_start = hw_block * BLOCK_HW
    base = bt * HW
    # Tail BLOCK_HW rows that fall past HW are guarded by `hw < HW`. For
    # buckets where BLOCK_HW divides HW (e.g. 4096/64=64 at 1024²) every
    # iteration is in-bounds and the branch is hoisted to a no-op.
    for i in range(BLOCK_HW):
        hw = hw_start + i
        if hw < HW:
            row = base + hw
            dy = tl.load(dY + row * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            x = tl.load(X + row * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)
            xhat = tl.where(mask_d, (x - mean) * rstd, 0.0)
            ds += dy * xhat
            db += dy

    tl.atomic_add(dS + bt * D + cols, ds, mask=mask_d)
    tl.atomic_add(dB + bt * D + cols, db, mask=mask_d)


class FusedAdaLN(torch.autograd.Function):
    """Custom autograd op for `LayerNorm(x, affine=False, eps) * (1+scale) + shift`.

    Inputs:
      x:     (B, T, H, W, D)   — the residual stream tensor
      scale: (B, T, 1, 1, D)   — broadcasts across H, W
      shift: (B, T, 1, 1, D)   — broadcasts across H, W
      eps:   float

    Returns: (B, T, H, W, D) in x.dtype.
    """

    @staticmethod
    def forward(ctx, x, scale, shift, eps):
        assert x.is_cuda, "FusedAdaLN requires CUDA tensors"
        B, T, H, W, D = x.shape
        # .contiguous() is a no-op if already contiguous. Required because
        # `view(-1, D)` errors on non-contiguous tensors (e.g. if upstream
        # produced a strided view).
        x_flat = x.contiguous().view(-1, D)
        scale_flat = scale.contiguous().view(-1, D)
        shift_flat = shift.contiguous().view(-1, D)

        N = x_flat.shape[0]       # B * T * H * W
        M = scale_flat.shape[0]   # B * T
        HW = H * W
        assert N == M * HW, f"shape mismatch: N={N}, M={M}, HW={HW}"

        y = torch.empty_like(x_flat)
        mean = torch.empty(N, dtype=torch.float32, device=x.device)
        rstd = torch.empty(N, dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        # Heuristic: for the typical Anima width (D=2048) 8 warps amortizes
        # the per-row reduction well. Narrower D gets fewer warps.
        num_warps = 8 if BLOCK_D >= 2048 else (4 if BLOCK_D >= 512 else 2)

        _fwd_kernel[(N,)](
            x_flat, scale_flat, shift_flat, y, mean, rstd,
            HW, D, eps,
            BLOCK_D=BLOCK_D, num_warps=num_warps,
        )

        ctx.save_for_backward(x_flat, scale_flat, mean, rstd)
        ctx.x_shape = (B, T, H, W, D)
        ctx.scale_shape = scale.shape
        ctx.HW = HW
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps

        return y.view(B, T, H, W, D)

    @staticmethod
    def backward(ctx, dy):
        x_flat, scale_flat, mean, rstd = ctx.saved_tensors
        B, T, H, W, D = ctx.x_shape
        HW = ctx.HW
        BLOCK_D = ctx.BLOCK_D
        num_warps = ctx.num_warps

        N = x_flat.shape[0]
        M = scale_flat.shape[0]

        dy_flat = dy.contiguous().view(-1, D)
        dx = torch.empty_like(x_flat)

        _bwd_dx_kernel[(N,)](
            dy_flat, x_flat, scale_flat, mean, rstd, dx,
            HW, D,
            BLOCK_D=BLOCK_D, num_warps=num_warps,
        )

        # dscale / dshift reduce HW positions per (b, t). The destinations
        # are tiny (BT × D ≈ 16K cells) so atomic_add into fp32 accumulators
        # is fine and lets us parallelize across HW. Pick BLOCK_HW small
        # enough that the grid saturates Blackwell's SMs; BLOCK_D=128 splits
        # D=2048 into 16 chunks, BLOCK_HW=64 splits HW=4096 into 64 chunks
        # → BT × 64 × 16 = ~8K programs at the production shape.
        dscale_fp32 = torch.zeros((M, D), dtype=torch.float32, device=dy.device)
        dshift_fp32 = torch.zeros((M, D), dtype=torch.float32, device=dy.device)

        DSDB_BLOCK_D = min(BLOCK_D, 128)
        DSDB_BLOCK_HW = min(HW, 64)
        n_d_blocks = triton.cdiv(D, DSDB_BLOCK_D)
        n_hw_blocks = triton.cdiv(HW, DSDB_BLOCK_HW)
        _bwd_dsdb_kernel[(M, n_hw_blocks, n_d_blocks)](
            dy_flat, x_flat, mean, rstd, dscale_fp32, dshift_fp32,
            HW, D,
            BLOCK_D=DSDB_BLOCK_D, BLOCK_HW=DSDB_BLOCK_HW, num_warps=4,
        )
        dscale = dscale_fp32.to(scale_flat.dtype)
        dshift = dshift_fp32.to(scale_flat.dtype)

        return (
            dx.view(B, T, H, W, D),
            dscale.view(ctx.scale_shape),
            dshift.view(ctx.scale_shape),
            None,
        )


def fused_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """Functional wrapper around :class:`FusedAdaLN` for convenience."""
    return FusedAdaLN.apply(x, scale, shift, eps)


# --- Fused gated-residual: out = x + gate * y ------------------------------
#
# At each Block site Anima does `x = x + gate * result` where x and result are
# (B, T, H, W, D) and gate is (B, T, 1, 1, D) — broadcast across (H, W). Stock
# PyTorch emits this as two kernels (mul, add). 84 sites per forward × 2 = 168
# launches we don't need. The fused kernel below collapses each into one Triton
# kernel: one load of x, one load of result, broadcast-load of gate, one store.


@triton.jit
def _gated_add_fwd_kernel(
    X, GATE, Y, OUT,
    HW, D,
    BLOCK_D: tl.constexpr,
):
    """out[row, d] = x[row, d] + gate[bt, d] * y[row, d]

    Each program handles one row of D elements. `bt = row // HW` selects which
    (b, t) pair's gate to broadcast — the only difference from a vanilla
    elementwise add.
    """
    row = tl.program_id(0)
    bt = row // HW

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0)
    y = tl.load(Y + row * D + cols, mask=mask, other=0.0)
    g = tl.load(GATE + bt * D + cols, mask=mask, other=0.0)

    out = x + g * y
    tl.store(OUT + row * D + cols, out, mask=mask)


@triton.jit
def _gated_add_bwd_dgate_kernel(
    dOUT, Y, dGATE,
    HW, D,
    BLOCK_D: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """dgate[bt] = sum over (h, w) of (dout * y).

    Mirror of the AdaLN dscale/dshift reduction: 3-D grid (bt, hw_block,
    d_block), atomic-add into an fp32 destination of shape (BT, D). The
    wrapper casts back to the user dtype.
    """
    bt = tl.program_id(0)
    hw_block = tl.program_id(1)
    d_block = tl.program_id(2)

    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = cols < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    hw_start = hw_block * BLOCK_HW
    base = bt * HW
    for i in range(BLOCK_HW):
        hw = hw_start + i
        if hw < HW:
            row = base + hw
            do = tl.load(dOUT + row * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            y = tl.load(Y + row * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            acc += do * y

    tl.atomic_add(dGATE + bt * D + cols, acc, mask=mask_d)


class FusedGatedAdd(torch.autograd.Function):
    """Custom autograd op for `out = x + gate * y`.

    Inputs:
      x:    (B, T, H, W, D)   — the residual stream tensor
      gate: (B, T, 1, 1, D)   — broadcasts across H, W
      y:    (B, T, H, W, D)   — the sublayer output

    Returns: (B, T, H, W, D) in x.dtype.

    Backward:
      dx = dout                     — identity
      dy = gate * dout              — pointwise (broadcast)
      dgate = sum_{h,w}(dout * y)   — HW reduction per (b, t)
    """

    @staticmethod
    def forward(ctx, x, gate, y):
        assert x.is_cuda, "FusedGatedAdd requires CUDA tensors"
        assert x.shape == y.shape, f"x {x.shape} != y {y.shape}"
        B, T, H, W, D = x.shape
        x_flat = x.contiguous().view(-1, D)
        y_flat = y.contiguous().view(-1, D)
        gate_flat = gate.contiguous().view(-1, D)

        N = x_flat.shape[0]
        M = gate_flat.shape[0]
        HW = H * W
        assert N == M * HW, f"shape mismatch: N={N}, M={M}, HW={HW}"

        out = torch.empty_like(x_flat)
        BLOCK_D = triton.next_power_of_2(D)
        num_warps = 8 if BLOCK_D >= 2048 else (4 if BLOCK_D >= 512 else 2)

        _gated_add_fwd_kernel[(N,)](
            x_flat, gate_flat, y_flat, out,
            HW, D,
            BLOCK_D=BLOCK_D, num_warps=num_warps,
        )

        # Save y and gate for backward — x is not needed (dx = dout).
        ctx.save_for_backward(y_flat, gate_flat)
        ctx.shape = (B, T, H, W, D)
        ctx.gate_shape = gate.shape
        ctx.HW = HW
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps

        return out.view(B, T, H, W, D)

    @staticmethod
    def backward(ctx, dout):
        y_flat, gate_flat = ctx.saved_tensors
        B, T, H, W, D = ctx.shape
        HW = ctx.HW
        BLOCK_D = ctx.BLOCK_D

        N = y_flat.shape[0]
        M = gate_flat.shape[0]

        dout_flat = dout.contiguous().view(-1, D)

        # dx = dout (identity) — return the input dout view, no kernel needed.
        # dy = gate * dout (broadcast mul). For now express as a plain torch op
        # — the kernel saving here is small (one mul), and writing a
        # broadcast-mul Triton kernel adds maintenance cost without a
        # measurable win on this shape.
        dy = (gate_flat.view(M, 1, D) * dout_flat.view(M, HW, D)).view(-1, D)

        # dgate reduces HW positions per (b, t). Same pattern as
        # _bwd_dsdb_kernel.
        dgate_fp32 = torch.zeros((M, D), dtype=torch.float32, device=dout.device)
        DG_BLOCK_D = min(BLOCK_D, 128)
        DG_BLOCK_HW = min(HW, 64)
        n_d_blocks = triton.cdiv(D, DG_BLOCK_D)
        n_hw_blocks = triton.cdiv(HW, DG_BLOCK_HW)
        _gated_add_bwd_dgate_kernel[(M, n_hw_blocks, n_d_blocks)](
            dout_flat, y_flat, dgate_fp32,
            HW, D,
            BLOCK_D=DG_BLOCK_D, BLOCK_HW=DG_BLOCK_HW, num_warps=4,
        )
        dgate = dgate_fp32.to(gate_flat.dtype)

        return (
            dout_flat.view(B, T, H, W, D),
            dgate.view(ctx.gate_shape),
            dy.view(B, T, H, W, D),
        )


def fused_gated_add(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Functional wrapper around :class:`FusedGatedAdd` — computes
    `x + gate * y` with broadcasting on the spatial dims.
    """
    return FusedGatedAdd.apply(x, gate, y)
