"""Microbenchmark MXFP8 vs bf16 GEMM at Anima Linear shapes on sm_120.

Anima DiT Linear shapes per block:
  - q_proj, k_proj, v_proj, output_proj (self-attn): in=2048, out=2048
  - q_proj for cross-attn: in=2048, out=2048
  - k_proj, v_proj for cross-attn: in=1024, out=2048
  - mlp.fc1: in=2048, out=8192   (matches torchtune-style 4x expansion)
  - mlp.fc2: in=8192, out=2048

Self-attn at 1024² → seq=4096, batch=8 → M = 32768.
We bench forward GEMM only (frozen weights, no backward through them).

We compare:
  - bf16 @ bf16 -> bf16  (current path)
  - mxfp8 @ mxfp8 -> bf16 (frozen DiT + bf16 activations dynamically quantized)

Goal: see if mxfp8 actually delivers a runtime win on sm_120.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from torchao.prototype.mx_formats import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
    NVFP4WeightOnlyConfig,
)
from torchao.quantization import (
    quantize_,
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    PerRow,
    PerTensor,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


@dataclass
class Shape:
    name: str
    M: int
    K: int
    N: int


# Per-block shapes; we multiply by 28 in production (28 transformer blocks).
SHAPES = [
    Shape("attn_qkvo  (M=32768 K=2048 N=2048)", M=32768, K=2048, N=2048),
    Shape("attn_kv_x  (M=32768 K=1024 N=2048)", M=32768, K=1024, N=2048),
    Shape("mlp_fc1    (M=32768 K=2048 N=8192)", M=32768, K=2048, N=8192),
    Shape("mlp_fc2    (M=32768 K=8192 N=2048)", M=32768, K=8192, N=2048),
]


def make_linear(shape: Shape, dtype=torch.bfloat16):
    """Build a frozen bf16 nn.Linear with shape (K -> N)."""
    lin = nn.Linear(shape.K, shape.N, bias=False).to("cuda", dtype=dtype)
    lin.weight.requires_grad_(False)
    return lin


def _time(callable_, iters=30, warmup=10):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        callable_()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_bf16(shape: Shape, *, compile_: bool) -> float:
    lin = make_linear(shape)
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    return _time(lambda: lin(x))


def bench_mxfp8(shape: Shape, *, compile_: bool, kernel: KernelPreference = KernelPreference.AUTO) -> float | None:
    lin = make_linear(shape)
    cfg = MXDynamicActivationMXWeightConfig(
        block_size=32,
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=kernel,
    )
    try:
        quantize_(lin, cfg)
    except Exception as exc:
        print(f"  mxfp8/{kernel.name} quantize_ failed: {exc}")
        return None
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    try:
        out = lin(x)
    except Exception as exc:
        print(f"  mxfp8/{kernel.name} forward failed: {exc}")
        return None
    return _time(lambda: lin(x))


def bench_nvfp4(shape: Shape, *, compile_: bool, use_triton: bool = True) -> float | None:
    lin = make_linear(shape)
    cfg = NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=use_triton,
        use_dynamic_per_tensor_scale=True,
    )
    try:
        quantize_(lin, cfg)
    except Exception as exc:
        print(f"  nvfp4 quantize_ failed: {exc}")
        return None
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    try:
        out = lin(x)
    except Exception as exc:
        print(f"  nvfp4 forward failed: {exc}")
        return None
    return _time(lambda: lin(x))


def bench_fp8_rowwise(shape: Shape, *, compile_: bool) -> float | None:
    """Stable (non-MX) FP8 with per-row scaling — the torchtitan recipe."""
    lin = make_linear(shape)
    cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    try:
        quantize_(lin, cfg)
    except Exception as exc:
        print(f"  fp8-rowwise quantize_ failed: {exc}")
        return None
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    try:
        out = lin(x)
    except Exception as exc:
        print(f"  fp8-rowwise forward failed: {exc}")
        return None
    return _time(lambda: lin(x))


def bench_fp8_tensorwise(shape: Shape, *, compile_: bool) -> float | None:
    lin = make_linear(shape)
    cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    try:
        quantize_(lin, cfg)
    except Exception as exc:
        print(f"  fp8-tensorwise quantize_ failed: {exc}")
        return None
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    try:
        out = lin(x)
    except Exception as exc:
        print(f"  fp8-tensorwise forward failed: {exc}")
        return None
    return _time(lambda: lin(x))


def bench_int4_weight_only(shape: Shape, *, compile_: bool, group_size: int = 128) -> float | None:
    """BF16 acts × INT4 weights — the bf16+MSLK candidate."""
    lin = make_linear(shape)
    cfg = Int4WeightOnlyConfig(group_size=group_size)
    try:
        quantize_(lin, cfg)
    except Exception as exc:
        print(f"  int4 quantize_ failed: {exc}")
        return None
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    try:
        out = lin(x)
    except Exception as exc:
        print(f"  int4 forward failed: {exc}")
        return None
    return _time(lambda: lin(x))


def bench_nvfp4_weight_only(shape: Shape, *, compile_: bool) -> float | None:
    """NVFP4 weights, bf16 activations — the 'nvfp4 mixed' path."""
    lin = make_linear(shape)
    cfg = NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=True)
    try:
        quantize_(lin, cfg)
    except Exception as exc:
        print(f"  nvfp4-wonly quantize_ failed: {exc}")
        return None
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    try:
        out = lin(x)
    except Exception as exc:
        print(f"  nvfp4-wonly forward failed: {exc}")
        return None
    return _time(lambda: lin(x))


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability(0)}")
    print(f"torch: {torch.__version__}")
    import torchao
    print(f"torchao: {torchao.__version__}")
    print()
    # Focused on bf16+INT4 (MSLK weight-only path) vs current MXFP8 production.
    results: dict[str, dict] = {}
    for sh in SHAPES:
        torch._dynamo.reset();  bf      = bench_bf16(sh, compile_=True)
        torch._dynamo.reset();  bf_e    = bench_bf16(sh, compile_=False)
        torch._dynamo.reset();  mx_c    = bench_mxfp8(sh, compile_=True)
        torch._dynamo.reset();  i4_e    = bench_int4_weight_only(sh, compile_=False)
        torch._dynamo.reset();  i4_c    = bench_int4_weight_only(sh, compile_=True)
        results[sh.name] = dict(
            bf16_eager=bf_e, bf16_cmpl=bf,
            mxfp8_cmpl=mx_c,
            int4w_eager=i4_e, int4w_cmpl=i4_c,
        )

    cols = ["bf16_eager", "bf16_cmpl", "mxfp8_cmpl", "int4w_eager", "int4w_cmpl"]
    print()
    print("absolute ms/iter:")
    print(f"{'shape':<42} " + " ".join(f"{c:>13}" for c in cols))
    for sh in SHAPES:
        r = results[sh.name]
        cells = [f"{r[c]:>13.3f}" if r[c] is not None else f"{'fail':>13}" for c in cols]
        print(f"{sh.name:<42} " + " ".join(cells))
    print()
    bf16_key = "bf16_cmpl" if "bf16_cmpl" in cols else "bf16"
    print(f"speedup vs {bf16_key}:")
    print(f"{'shape':<42} " + " ".join(f"{c:>13}" for c in cols if c != bf16_key))
    for sh in SHAPES:
        r = results[sh.name]
        bf = r[bf16_key]
        cells = [f"{bf/r[c]:>12.2f}x" if (r[c] is not None and r[c] > 0) else f"{'fail':>13}"
                 for c in cols if c != bf16_key]
        print(f"{sh.name:<42} " + " ".join(cells))


if __name__ == "__main__":
    main()
