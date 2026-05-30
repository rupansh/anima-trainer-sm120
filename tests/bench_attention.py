"""Microbenchmark attention kernels at Anima's actual shapes on this GPU.

Anima geometry (from sd-scripts/library/anima_utils.py:32-90):
  - model_channels = 2048, num_heads = 16, head_dim = 128
  - patch_spatial = 2 → after VAE×8 + patch×2, a 1024² image has 64² = 4096 tokens
  - cross-attn KV: 512 tokens (Qwen3 max_length)

We benchmark the three torch SDPA backends in both forward-only and forward+backward
modes. FA3/FA4 are excluded (no sm_120 support). FA2 builds for sm_120 but only via
the SM89 fallback, not a Blackwell-optimized kernel. SageAttention is inference-only.

cuDNN is the only kernel on sm_120 with genuine Blackwell-specific optimizations
(2-CTA MMA, dQ matmul reordering, etc., per cuDNN release notes).
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch.nn.functional as F

B = 8
H = 16
HD = 128


@dataclass
class Shape:
    name: str
    s_q: int
    s_kv: int


SHAPES = [
    Shape("self_1024", s_q=4096, s_kv=4096),
    Shape("self_896x1152", s_q=4032, s_kv=4032),
    Shape("cross_1024", s_q=4096, s_kv=512),
    Shape("self_512", s_q=1024, s_kv=1024),
]


def _qkv(shape: Shape, *, requires_grad: bool, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(B, H, shape.s_q, HD, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, shape.s_kv, HD, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, H, shape.s_kv, HD, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


@contextmanager
def sdpa_backend(kind: str | None):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if kind is None:
        yield
        return
    table = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    with sdpa_kernel([table[kind]]):
        yield


def _time(fn, *, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def fwd_only(shape: Shape, kind: str | None) -> float | None:
    q, k, v = _qkv(shape, requires_grad=False)
    with sdpa_backend(kind):
        try:
            _ = F.scaled_dot_product_attention(q, k, v)
        except Exception:
            return None
        return _time(lambda: F.scaled_dot_product_attention(q, k, v))


def fwd_bwd(shape: Shape, kind: str | None) -> float | None:
    """Forward + backward on a scalar reduction of the output."""
    q, k, v = _qkv(shape, requires_grad=True)

    def step():
        with sdpa_backend(kind):
            out = F.scaled_dot_product_attention(q, k, v)
        loss = out.sum()
        # Clean grads each iter; we want pure forward+backward time.
        q.grad = k.grad = v.grad = None
        loss.backward()

    try:
        step()
    except Exception:
        return None
    return _time(step, warmup=3, iters=10)


def detect_default_backend(shape: Shape, *, tol_pct: float = 7.0) -> str:
    """Infer which backend torch picks by default by matching its time to the
    explicit-backend times. Whichever explicit time is closest (within `tol_pct`%)
    is reported as the default.
    """
    default_t = fwd_only(shape, None)
    if default_t is None:
        return "?"
    explicit = {k: fwd_only(shape, k) for k in ("cudnn", "flash", "mem_eff", "math")}
    explicit = {k: v for k, v in explicit.items() if v is not None}
    best = min(explicit, key=lambda k: abs(explicit[k] - default_t))
    if abs(explicit[best] - default_t) / default_t * 100.0 > tol_pct:
        return f"~{best}(±)"
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  capability: {torch.cuda.get_device_capability(0)}")
    print(f"torch: {torch.__version__}")
    print(f"B={B} H={H} D={HD} bf16  (ms/iter, lower is better)\n")

    print("default backend selection per shape:")
    for sh in SHAPES:
        print(f"  {sh.name:<14} -> {detect_default_backend(sh)}")
    print()

    cols = ["cudnn", "flash", "mem_eff", "math"]

    print("forward only")
    print(f"  {'shape':<14} " + " ".join(f"{c:>9}" for c in cols))
    for sh in SHAPES:
        row = [fwd_only(sh, c) for c in cols]
        cells = [f"{v:>9.3f}" if v is not None else f"{'--':>9}" for v in row]
        print(f"  {sh.name:<14} " + " ".join(cells))
    print()

    print("forward + backward")
    print(f"  {'shape':<14} " + " ".join(f"{c:>9}" for c in cols))
    for sh in SHAPES:
        row = [fwd_bwd(sh, c) for c in cols]
        cells = [f"{v:>9.3f}" if v is not None else f"{'--':>9}" for v in row]
        print(f"  {sh.name:<14} " + " ".join(cells))


if __name__ == "__main__":
    main()
