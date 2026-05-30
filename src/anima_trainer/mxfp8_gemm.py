"""MXFP8 GEMM kernels via Triton's `tl.dot_scaled`.

Why this exists:
  - sm_120 cuBLAS has a heuristic gap for MXFP8 backward (dgrad) GEMM
    layouts — `CUBLAS_STATUS_NOT_SUPPORTED` on every production shape we
    tried with cuBLAS 13.4 / 13.5. That's what TE's `check_mxfp8_support`
    gates against on compute ≥ 12.0.
  - Triton 3.7's `tl.dot_scaled` is the standard OCP-MX intrinsic
    (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
    It dispatches to native MXFP8 MMA where supported and bf16-emulates
    elsewhere — on sm_120 it fires the same hardware MMA the C++ CUTLASS
    example 79c uses, without going through cuBLAS at all.
  - That lets us run **MXFP8 in both directions** for LoKr-wrapped frozen
    Linears: forward GEMM (which cuBLAS handles fine via TE), AND
    backward dgrad (which cuBLAS doesn't — so this is the workaround).

Layout conventions:
  - For forward `y = x @ W.T`, the kernel computes `C = A @ B` with
    A = x (M×K), B = W.T (K×N). A is rowwise-quantized to FP8 + scales
    (M, K//32). B is the K×N view of W.T; its scales follow
    `tl.dot_scaled`'s required `rhs_scale` layout of (N, K//32) —
    i.e. the rowwise quantization of `W` itself (N is W's leading dim).
  - For backward dgrad `grad_x = grad_y @ W` with grad_y (M, N) and
    W (N, K), the kernel computes `C = A @ B` with A = grad_y (M, N),
    B = W (N, K). grad_y is rowwise-quantized; W as B needs `rhs_scale`
    layout (K, N//32) — which is the *columnwise* quantization of W
    (K is the trailing dim of W's transpose). We pre-quantize W in both
    layouts at attach time and store both.

E8M0 scales: 1 uint8 per 32 elements along the K dimension. The value
encodes `2^(stored_byte - 127)` (biased exponent). Quantizer follows the
OCP-MX spec: `exp = ceil(log2(amax / fp8_e4m3_max))`, scale = 2^exp.

Numerical floor: MXFP8 quantize+dequant noise is ~5% relative per element;
GEMM noise floor is ~4.5% relative on production shapes (matches TE's
quantize+dequant+bf16-mm reference).
"""
from __future__ import annotations
from typing import NamedTuple
import torch
import triton
import triton.language as tl


FP8_E4M3_MAX = 448.0


# --- Quantizers ------------------------------------------------------------


class MXFP8Block(NamedTuple):
    """An MXFP8-quantized tensor in Triton-friendly layout.

    Attributes:
      data: uint8 storage holding E4M3 FP8 values (same shape as the
        original bf16 tensor).
      scales: uint8 E8M0 scales. For a rowwise quant of an (M, K) tensor,
        scales have shape (M, K//32). For a columnwise quant of an (N, K)
        tensor (where N is the rowwise dim), the columnwise scales have
        shape (K, N//32) — this is the layout `tl.dot_scaled` expects
        when this tensor is used as `rhs` in a GEMM where K is the
        accumulation dim.
    """
    data: torch.Tensor    # uint8 (any shape)
    scales: torch.Tensor  # uint8 e8m0


@triton.jit
def _quantize_rowwise_kernel(
    X_ptr, D_ptr, S_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_dm, stride_dk,
    stride_sm, stride_sk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Quantize a 32-K-block group of rows to MXFP8 E4M3 + E8M0 scales.

    Each program handles BLOCK_M rows × BLOCK_K K-elements. BLOCK_K is a
    multiple of 32 — we have BLOCK_K // 32 scale groups per row in this tile.

    The math: for each group of 32 K-elements:
      amax = max(|x|)
      exp  = ceil(log2(amax / 448))       (E4M3 max = 448)
      scale = 2^exp
      out   = round_to_nearest_even(x / scale) clamped to E4M3 range
      e8m0_byte = exp + 127

    Stored byte = exp + 127, scale recovery = 2^(byte - 127). Standard
    OCP-MX layout, matches what `tl.dot_scaled` expects.
    """
    pid_m = tl.program_id(0)
    pid_k_group = tl.program_id(1)  # which 32-group along K
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ok_base = pid_k_group * BLOCK_K
    ok = ok_base + tl.arange(0, BLOCK_K)
    # Number of 32-element scale groups in this tile.
    GROUPS_PER_TILE: tl.constexpr = BLOCK_K // 32
    osk = pid_k_group * GROUPS_PER_TILE + tl.arange(0, GROUPS_PER_TILE)

    mask_m = om < M
    mask_k = ok < K

    # Load (BLOCK_M, BLOCK_K) tile.
    x = tl.load(
        X_ptr + om[:, None] * stride_xm + ok[None, :] * stride_xk,
        mask=mask_m[:, None] & mask_k[None, :], other=0.0,
    ).to(tl.float32)

    # Reshape to (BLOCK_M, GROUPS_PER_TILE, 32) for per-group amax.
    x_grouped = tl.reshape(x, (BLOCK_M, GROUPS_PER_TILE, 32))
    amax = tl.max(tl.abs(x_grouped), axis=2)        # (BLOCK_M, GROUPS_PER_TILE)
    amax = tl.maximum(amax, 1e-30)

    # exp = ceil(log2(amax / FP8_MAX)). log2 in Triton: tl.log2.
    # Note: ceil of log2 is well-defined here since amax > 0.
    log2_ratio = tl.log2(amax) - tl.log2(tl.full((), FP8_MAX, tl.float32))
    exp_f = tl.ceil(log2_ratio)
    exp_i = tl.where(exp_f > 127.0, 127.0, tl.where(exp_f < -127.0, -127.0, exp_f)).to(tl.int32)

    # Scale = 2^exp; broadcast across the 32 group elements.
    # `exp2` is `tl.exp2` in newer Triton, otherwise via libdevice.
    scale = tl.exp2(exp_i.to(tl.float32))           # (BLOCK_M, GROUPS_PER_TILE)
    scale_b = tl.reshape(scale, (BLOCK_M, GROUPS_PER_TILE, 1))

    # Quantize x / scale → e4m3.
    y = x_grouped / scale_b
    # E4M3 clamp (the .to(fp8_e4m3) does saturating round-to-nearest-even).
    y_fp8 = y.to(tl.float8e4nv)
    # Pack back to (BLOCK_M, BLOCK_K).
    y_packed = tl.reshape(y_fp8, (BLOCK_M, BLOCK_K))

    # Store fp8 data via uint8 alias.
    tl.store(
        D_ptr + om[:, None] * stride_dm + ok[None, :] * stride_dk,
        y_packed.to(tl.uint8, bitcast=True),
        mask=mask_m[:, None] & mask_k[None, :],
    )
    # Store e8m0 scales (exp + 127) as uint8.
    scale_bytes = (exp_i + 127).to(tl.uint8)
    mask_sk = osk < (K // 32)
    tl.store(
        S_ptr + om[:, None] * stride_sm + osk[None, :] * stride_sk,
        scale_bytes,
        mask=mask_m[:, None] & mask_sk[None, :],
    )


def quantize_rowwise(x: torch.Tensor) -> MXFP8Block:
    """Rowwise MXFP8 quantize of x[..., K]. Scales: (..., K//32).

    Fully fused Triton kernel — replaces the prior PyTorch-eager version
    that did 6+ ops with fp32 intermediates and was ~10× slower than
    necessary on K≥8192 shapes (we profiled).
    """
    assert x.shape[-1] % 32 == 0, f"K={x.shape[-1]} must be %32"
    shape = x.shape
    K = shape[-1]
    leading = x.numel() // K
    x_2d = x.reshape(leading, K).contiguous()
    data = torch.empty_like(x_2d, dtype=torch.uint8)
    scales = torch.empty((leading, K // 32), dtype=torch.uint8, device=x.device)
    # Tile shape: 16 rows × 128 K-elements (4 scale groups per tile) is a
    # decent default — small enough to fit in SMEM/registers, large enough
    # to amortize launch overhead.
    BLOCK_M = 16
    BLOCK_K = 128
    grid = (triton.cdiv(leading, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _quantize_rowwise_kernel[grid](
        x_2d, data, scales,
        leading, K,
        x_2d.stride(0), x_2d.stride(1),
        data.stride(0), data.stride(1),
        scales.stride(0), scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        FP8_MAX=FP8_E4M3_MAX,
        num_warps=4,
    )
    return MXFP8Block(
        data=data.reshape(*shape),
        scales=scales.reshape(*shape[:-1], K // 32),
    )


def quantize_weight_for_lhs(W: torch.Tensor) -> MXFP8Block:
    """Quantize W of shape (N, K) for use as the *forward rhs* in a GEMM
    `y = x @ W.T` (i.e. when W viewed as (K, N) via W.T is the rhs).

    Returns (data, scales) where:
      - data: (N, K) uint8 — stored row-major; the kernel uses transposed
        strides to access it as (K, N).
      - scales: (N, K//32) — matches `tl.dot_scaled`'s rhs_scale layout
        for an rhs of shape (K, N).

    This is exactly the same as `quantize_rowwise(W)`.
    """
    return quantize_rowwise(W)


def quantize_weight_for_dgrad(W: torch.Tensor) -> MXFP8Block:
    """Quantize W of shape (N, K) for use as the *backward dgrad rhs* in
    `grad_x = grad_y @ W`. In this GEMM, W is the rhs of shape (N, K)
    (no transpose — K is the trailing dim).

    `tl.dot_scaled` wants rhs_scale of shape (rhs_trailing, K_gemm//32).
    In dgrad, K_gemm = N (the accumulation dim), rhs_trailing = K.
    So scales must be shape (K, N//32) — i.e. *columnwise* MXFP8 quant
    of W (along the N dim, with 32-element N-blocks).

    Returns (data (N, K) uint8, scales (K, N//32) uint8).
    """
    N, K = W.shape
    assert N % 32 == 0, f"N={N} must be %32 for columnwise MXFP8 dgrad path"
    # Reshape so the dim to block-quantize is contiguous.
    # We want 32-blocks along N for each K position.
    W_kn = W.t().contiguous()  # (K, N), N-major (contiguous in N)
    block = quantize_rowwise(W_kn)  # data (K, N) uint8, scales (K, N//32)
    # The fp8 data needs to be returned in W's original (N, K) layout for
    # the kernel's rhs access pattern.
    data = block.data.t().contiguous().view(torch.uint8)  # (N, K) uint8
    return MXFP8Block(data=data, scales=block.scales)


def dequantize_rowwise(blk: MXFP8Block, original_shape: tuple) -> torch.Tensor:
    """Inverse — for validation / fallback. Returns bf16."""
    K = original_shape[-1]
    leading = blk.data.numel() // K
    data = blk.data.view(torch.float8_e4m3fn).reshape(leading, K // 32, 32).float()
    exp = blk.scales.reshape(leading, K // 32).to(torch.int32) - 127
    scale = (2.0 ** exp.float()).unsqueeze(-1)
    return (data * scale).reshape(original_shape).to(torch.bfloat16)


# --- Triton kernel ---------------------------------------------------------


_MXFP8_GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": gm},
                  num_stages=ns, num_warps=nw)
    # Trimmed based on prior sweep (BLOCK_K=64 won everywhere; BLOCK_M=128
    # and num_warps=8 / num_stages=3 dominated). Keep enough variety to let
    # the swizzle GROUP_SIZE_M and a few neighboring tile sizes still tune.
    for bm in (64, 128, 256)
    for bn in (128, 256)
    for bk in (64, 128)
    for ns in (3, 4)
    for nw in (4, 8)
    for gm in (4, 8)
    if (bm * bk + bn * bk) * 2 <= 200_000
]


@triton.autotune(configs=_MXFP8_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mxfp8_gemm_kernel(
    A, B, A_s, B_s, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute C[M,N] = A[M,K] @ B[K,N] in MXFP8.

    A_s: (M, K//32) e8m0 — rowwise scale of A.
    B_s: (N, K//32) e8m0 — `tl.dot_scaled`'s expected rhs_scale layout
    (note: N is the LEADING dim of B_s, not transposed).

    Tile-id swizzling: programs are issued in 1D and re-mapped to (pid_m,
    pid_n) so that GROUP_SIZE_M consecutive M-tiles process the same set of
    N-tiles together. This keeps B tiles hot in L2 across multiple M-tiles
    (which would otherwise be evicted between M-row sweeps under naïve
    row-major issue order). Typical 10–20% win on matmul.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)

    Ap  = A   + offs_m[:, None] * stride_am  + offs_k[None, :] * stride_ak
    Bp  = B   + offs_k[:, None] * stride_bk  + offs_n[None, :] * stride_bn
    ASp = A_s + offs_m[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    BSp = B_s + offs_n[:, None] * stride_bsn + offs_sk[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot_scaled(
            tl.load(Ap), tl.load(ASp), "e4m3",
            tl.load(Bp), tl.load(BSp), "e4m3",
            acc,
        )
        Ap  += BLOCK_K * stride_ak
        Bp  += BLOCK_K * stride_bk
        ASp += (BLOCK_K // 32) * stride_ask
        BSp += (BLOCK_K // 32) * stride_bsk

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16))


def mxfp8_matmul(A: MXFP8Block, B_data: torch.Tensor, B_scales: torch.Tensor,
                 M: int, N: int, K: int,
                 B_stride_k: int, B_stride_n: int) -> torch.Tensor:
    """C[M,N] = A.data[M,K] @ B_data[K,N] in MXFP8.

    Tile shapes are auto-tuned per (M, N, K) — first call to a new shape
    pays a sweep cost (a few seconds), then cached.
    """
    device = A.data.device
    C = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    # 1D grid for swizzled tile-id → (pid_m, pid_n) mapping inside the kernel.
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _mxfp8_gemm_kernel[grid](
        A.data, B_data, A.scales, B_scales, C,
        M, N, K,
        A.data.stride(0), A.data.stride(1),
        B_stride_k, B_stride_n,
        C.stride(0), C.stride(1),
        A.scales.stride(0), A.scales.stride(1),
        B_scales.stride(0), B_scales.stride(1),
    )
    return C


# --- Autograd op -----------------------------------------------------------


class MXFP8FrozenLinearTriton(torch.autograd.Function):
    """`y = x @ W.T` where W is frozen and pre-quantized to MXFP8.

    Forward and backward dgrad both run in MXFP8 via Triton's `tl.dot_scaled`.
    Skips cuBLAS entirely → no `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120 dgrad.

    Requires:
      - `W_fwd`: MXFP8Block from `quantize_weight_for_lhs(W)`.
        data is (D_out, D_in); used as rhs with transposed strides.
      - `W_bwd`: MXFP8Block from `quantize_weight_for_dgrad(W)`.
        data is (D_out, D_in); used as rhs in dgrad with original strides.
      - The bf16 `W` is NOT needed during backward (we use MXFP8 there too).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W_fwd_data: torch.Tensor, W_fwd_scales: torch.Tensor,
        W_bwd_data: torch.Tensor, W_bwd_scales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (..., D_in) → flatten leading dims to M = prod(...).
        D_in = x.shape[-1]
        D_out = W_fwd_data.shape[0]
        x_2d = x.reshape(-1, D_in)
        M = x_2d.shape[0]
        # Quantize x rowwise.
        x_blk = quantize_rowwise(x_2d)
        # Forward GEMM: y = x @ W.T. View W (D_out, D_in) as (D_in, D_out)
        # for the kernel — that's W.T with stride flip (data stays put).
        # B_stride_k corresponds to W's D_in dim → W.stride(1).
        # B_stride_n corresponds to W's D_out dim → W.stride(0).
        y = mxfp8_matmul(
            x_blk, W_fwd_data, W_fwd_scales,
            M=M, N=D_out, K=D_in,
            B_stride_k=W_fwd_data.stride(1),  # walking along D_in (the K of GEMM)
            B_stride_n=W_fwd_data.stride(0),  # walking along D_out (the N of GEMM)
        )
        y = y.reshape(*x.shape[:-1], D_out)
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x_blk.data, x_blk.scales,
                              W_bwd_data, W_bwd_scales)
        ctx.D_in = D_in
        ctx.D_out = D_out
        ctx.M = M
        ctx.has_bias = bias is not None
        ctx.x_orig_shape = x.shape
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x_data, x_scales, W_bwd_data, W_bwd_scales = ctx.saved_tensors
        D_in = ctx.D_in
        D_out = ctx.D_out
        M = ctx.M
        # grad_y arrives as bf16 with the original x's leading shape. View it
        # as 2D; quantize_rowwise calls .contiguous() internally as needed —
        # don't double up.
        gy_2d = grad_y.view(M, D_out) if grad_y.is_contiguous() else grad_y.reshape(M, D_out)
        if gy_2d.dtype != torch.bfloat16:
            gy_2d = gy_2d.to(torch.bfloat16)
        gy_blk = quantize_rowwise(gy_2d)
        # dgrad GEMM: grad_x = grad_y @ W. (M, D_out) @ (D_out, D_in) → (M, D_in).
        # B = W of shape (D_out, D_in). K_gemm = D_out, N_gemm = D_in.
        # B's K-axis = D_out = W's row → W.stride(0). B's N-axis = D_in = W's col → W.stride(1).
        grad_x_2d = mxfp8_matmul(
            gy_blk, W_bwd_data, W_bwd_scales,
            M=M, N=D_in, K=D_out,
            B_stride_k=W_bwd_data.stride(0),
            B_stride_n=W_bwd_data.stride(1),
        )
        grad_x = grad_x_2d.view(ctx.x_orig_shape) if grad_x_2d.is_contiguous() else grad_x_2d.reshape(ctx.x_orig_shape)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.reshape(-1, D_out).sum(dim=0)
        return grad_x, None, None, None, None, grad_bias


def mxfp8_frozen_linear_triton(
    x: torch.Tensor,
    W_fwd: MXFP8Block,
    W_bwd: MXFP8Block,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional wrapper for :class:`MXFP8FrozenLinearTriton`."""
    return MXFP8FrozenLinearTriton.apply(
        x, W_fwd.data, W_fwd.scales, W_bwd.data, W_bwd.scales, bias
    )
