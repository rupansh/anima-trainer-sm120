"""Step-time microbench: custom Triton FusedAdaLN vs torch.compile vs eager.

Isolates the 3-op AdaLN modulation (84 calls/forward in Anima) by running
each path 84 times and measuring forward+backward. Production shape only.

Run: `python -m tests.bench_adaln_kernel`
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.adaln_kernel import FusedAdaLN


def eager(x, scale, shift, eps):
    n = F.layer_norm(x, [x.shape[-1]], None, None, eps)
    return n * (1 + scale) + shift


def _adaln_compiled(x, normalized_shape, eps, scale, shift):
    n = F.layer_norm(x, normalized_shape, None, None, eps)
    return n * (1 + scale) + shift


compiled_fn = torch.compile(_adaln_compiled, fullgraph=True, dynamic=False, mode="default")


def run_path(name, call, x, scale, shift, eps, iters=84, warmups=5):
    # warmup
    for _ in range(warmups):
        y = call(x, scale, shift, eps)
        y.sum().backward()
        x.grad = None
        scale.grad = None
        shift.grad = None
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y = call(x, scale, shift, eps)
        y.sum().backward()
        x.grad = None
        scale.grad = None
        shift.grad = None
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  {name}: {iters} iters in {dt*1000:.1f}ms ({dt/iters*1000:.3f}ms/iter)")
    return dt


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    # Production: 1024² / B=8 → H=W=64, D=2048
    B, T, H, W, D = 8, 1, 64, 64, 2048

    torch.manual_seed(0)
    x = torch.randn(B, T, H, W, D, dtype=dtype, device=device, requires_grad=True)
    scale = (torch.randn(B, T, 1, 1, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)
    shift = (torch.randn(B, T, 1, 1, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)

    print(f"shape: x={tuple(x.shape)} D={D} dtype={dtype}")

    print("\n--- Path: eager ---")
    t_eager = run_path("eager", eager, x, scale, shift, eps)

    print("\n--- Path: torch.compile ---")
    def call_compiled(x, scale, shift, eps):
        return compiled_fn(x, (D,), eps, scale, shift)
    t_compile = run_path("compile", call_compiled, x, scale, shift, eps)

    print("\n--- Path: custom Triton FusedAdaLN ---")
    def call_kernel(x, scale, shift, eps):
        return FusedAdaLN.apply(x, scale, shift, eps)
    t_kern = run_path("kernel", call_kernel, x, scale, shift, eps)

    print()
    print(f"speedup vs eager   : compile={t_eager/t_compile:.2f}×  kernel={t_eager/t_kern:.2f}×")
    print(f"kernel vs compile  : {t_compile/t_kern:.2f}× ({(t_compile-t_kern)/t_compile*100:+.1f}%)")


if __name__ == "__main__":
    main()
