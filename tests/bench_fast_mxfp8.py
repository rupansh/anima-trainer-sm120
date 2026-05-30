"""Bench a hand-rolled FastMXFP8Linear vs torchao's MXFP8 eager and compiled.

Hypothesis: the slow eager torchao path is dominated by MXTensor's
__torch_dispatch__ machinery + per-call weight quantization. If we pre-quantize
the weight once and call torch._scaled_mm directly, we should approach the
compiled MXFP8 time without needing torch.compile.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


@dataclass
class Shape:
    name: str
    M: int
    K: int
    N: int


SHAPES = [
    Shape("attn_qkvo  (M=32768 K=2048 N=2048)", 32768, 2048, 2048),
    Shape("attn_kv_x  (M=32768 K=1024 N=2048)", 32768, 1024, 2048),
    Shape("mlp_fc1    (M=32768 K=2048 N=8192)", 32768, 2048, 8192),
    Shape("mlp_fc2    (M=32768 K=8192 N=2048)", 32768, 8192, 2048),
]


class FastMXFP8Linear(nn.Module):
    """Pre-quantizes weight once; calls torch._scaled_mm directly per forward.

    Mirrors the exact args torchao passes to `torch._scaled_mm` (traced):
      a:      (M, K)        e4m3  row-major contig (the activation qdata)
      b:      (N, K)        e4m3  col-major contig (the weight qdata, transposed-view)
      a_sc:   (M*Ks,)       e8m0  flat contig (activation swizzled scales)
      b_sc:   (Ns, K_block) e8m0  row-major contig (weight swizzled scales)

    where Ks = swizzled scale storage size for activation, etc. We rely on
    torchao's MXTensor.to_mx to produce these tensors (since the swizzle math
    is in torchao); we just reach in for `.qdata` and `.scale` so we can call
    `_scaled_mm` directly and skip the __torch_dispatch__ overhead per call.
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        assert base.bias is None
        w_mx = MXTensor.to_mx(
            base.weight.detach().contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=32,
            scaling_mode=ScaleCalculationMode.RCEIL,
            kernel_preference=KernelPreference.AUTO,
            is_swizzled_scales=True,
        )
        # weight qdata stored as (N, K) row-major; we'll do `.t()` per call (a
        # metadata-only stride flip → (K, N) col-major). The weight scale is
        # already laid out for direct consumption (no .t() needed when matched
        # to the col-major qdata view).
        self.register_buffer("w_q", w_mx.qdata)         # (N, K) e4m3 row-major
        self.register_buffer("w_scale", w_mx.scale)     # (Ns, Kb) e8m0
        self.out_features = base.out_features
        self.in_features = base.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        x_mx = MXTensor.to_mx(
            x2,
            elem_dtype=torch.float8_e4m3fn,
            block_size=32,
            scaling_mode=ScaleCalculationMode.RCEIL,
            kernel_preference=KernelPreference.AUTO,
            is_swizzled_scales=True,
        )
        # Activation scale gets flattened (matches what we traced from torchao).
        a_scale = x_mx.scale.reshape(-1)
        # Weight qdata: pass as (N, K) column-major view via .t() — _scaled_mm
        # interprets the second operand by its stride layout.
        out = torch._scaled_mm(
            x_mx.qdata,                # (M, K) row-major
            self.w_q.t(),              # (K, N) col-major view
            a_scale,                   # flat e8m0
            self.w_scale,              # already in the right layout
            bias=None,
            out_dtype=torch.bfloat16,
        )
        return out.view(*x.shape[:-1], self.out_features)


def make_linear(shape: Shape):
    lin = nn.Linear(shape.K, shape.N, bias=False).to("cuda", dtype=torch.bfloat16)
    lin.weight.requires_grad_(False)
    return lin


def _time(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_bf16(shape):
    lin = make_linear(shape)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    return _time(lambda: lin(x))


def bench_torchao_mxfp8(shape, *, compile_):
    lin = make_linear(shape)
    cfg = MXDynamicActivationMXWeightConfig(
        block_size=32,
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
    )
    quantize_(lin, cfg)
    if compile_:
        lin = torch.compile(lin, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    return _time(lambda: lin(x))


def bench_fast_mxfp8(shape, *, compile_):
    base = make_linear(shape)
    fast = FastMXFP8Linear(base)
    if compile_:
        fast = torch.compile(fast, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    return _time(lambda: fast(x))


class TritonMXFP8Linear(nn.Module):
    """Calls torchao's fused Triton quant kernel + Triton swizzle directly.

    Skips MXTensor entirely. Per forward: 2 Triton kernels (quant + swizzle)
    + 1 CUTLASS scaled_mm. Eager-friendly.
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        assert base.bias is None
        w = base.weight.detach().contiguous()
        w_q, w_scale_raw = torch.ops.torchao.triton_to_mxfp8_dim0(w, 32, "rceil")
        # Swizzle once at init
        w_scale_swiz = to_blocked(w_scale_raw.view(torch.float8_e8m0fnu), use_triton_kernel=True)
        self.register_buffer("w_q", w_q)
        self.register_buffer("w_scale_swiz", w_scale_swiz)
        N, K = w.shape
        self.N, self.K = N, K
        self.out_features = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        x_q, x_scale_raw = torch.ops.torchao.triton_to_mxfp8_dim0(x2, 32, "rceil")
        x_scale = to_blocked(x_scale_raw.view(torch.float8_e8m0fnu), use_triton_kernel=True)
        out = torch._scaled_mm(
            x_q,
            self.w_q.t(),
            x_scale,
            self.w_scale_swiz,
            bias=None,
            out_dtype=torch.bfloat16,
        )
        return out.view(*x.shape[:-1], self.N)


def bench_triton_mxfp8(shape, *, compile_):
    base = make_linear(shape)
    fast = TritonMXFP8Linear(base)
    if compile_:
        fast = torch.compile(fast, mode="default", dynamic=False)
    x = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.bfloat16)
    return _time(lambda: fast(x))


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability(0)}")
    print(f"torch: {torch.__version__}")
    print()
    cols = ["bf16", "torchao_eager", "torchao_cmpl", "fast_eager", "triton_eager", "triton_cmpl"]
    print(f"{'shape':<42} " + " ".join(f"{c:>13}" for c in cols))
    for sh in SHAPES:
        torch._dynamo.reset(); bf = bench_bf16(sh)
        torch._dynamo.reset(); ao_e = bench_torchao_mxfp8(sh, compile_=False)
        torch._dynamo.reset(); ao_c = bench_torchao_mxfp8(sh, compile_=True)
        torch._dynamo.reset(); fa_e = bench_fast_mxfp8(sh, compile_=False)
        torch._dynamo.reset(); tr_e = bench_triton_mxfp8(sh, compile_=False)
        torch._dynamo.reset(); tr_c = bench_triton_mxfp8(sh, compile_=True)
        print(f"{sh.name:<42} {bf:>13.3f} {ao_e:>13.3f} {ao_c:>13.3f} {fa_e:>13.3f} {tr_e:>13.3f} {tr_c:>13.3f}")
    print()
    print("speedup vs bf16 / vs torchao eager:")
    for sh in SHAPES:
        torch._dynamo.reset(); bf = bench_bf16(sh)
        torch._dynamo.reset(); ao_e = bench_torchao_mxfp8(sh, compile_=False)
        torch._dynamo.reset(); fa_e = bench_fast_mxfp8(sh, compile_=False)
        print(f"{sh.name:<42}  fast_eager: {bf/fa_e:.2f}x bf16, {ao_e/fa_e:.2f}x faster than torchao_eager")


if __name__ == "__main__":
    main()
