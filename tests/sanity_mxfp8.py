"""Numerical-parity check for the MXFP8 frozen-Linear training path.

Builds a plain `nn.Linear` (frozen), wraps it with `quantize_frozen_linears`,
and compares forward/backward against the bf16 reference on Anima production
shapes. The LoKr-wrapped path stays bf16 in production — see
`fp8_quant.py:quantize_frozen_linears` for the rationale.

Run: `python -m tests.sanity_mxfp8`
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

os.environ.setdefault("NVTE_CUDA_INCLUDE_DIR", "/opt/cuda/include")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.fp8_quant import quantize_frozen_linears, is_quantized


def case(D_in: int, D_out: int, BS: int):
    print(f"\n--- D_in={D_in} D_out={D_out} BS={BS} ---")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn(BS, D_in, device=device, dtype=dtype)

    # Two copies of the same Linear, weights freeze for both.
    bf16_linear = nn.Linear(D_in, D_out, bias=False).to(device, dtype)
    for p in bf16_linear.parameters():
        p.requires_grad_(False)

    mx_linear = nn.Linear(D_in, D_out, bias=False).to(device, dtype)
    with torch.no_grad():
        mx_linear.weight.copy_(bf16_linear.weight)
    for p in mx_linear.parameters():
        p.requires_grad_(False)

    # Wrap mx_linear in a container so quantize_frozen_linears can walk it.
    container = nn.Module()
    container.linear = mx_linear
    n_q = quantize_frozen_linears(container, min_size=0)
    assert n_q == 1 and is_quantized(mx_linear)

    # Forward and backward (against the input, since weights are frozen)
    x_b = x.detach().clone().requires_grad_(True)
    y_b = bf16_linear(x_b)
    y_b.float().pow(2).mean().backward()

    x_m = x.detach().clone().requires_grad_(True)
    y_m = mx_linear(x_m)
    y_m.float().pow(2).mean().backward()

    def report(name, ref, got):
        ref32 = ref.float()
        err = (ref32 - got.float()).abs()
        scale = ref32.abs().max().item() + 1e-9
        rel = err.max().item() / scale
        ok = rel < 0.10  # fp8 noise floor at typical training scales
        print(f"  {'OK ' if ok else 'FAIL'} {name:6s}  max_err={err.max().item():.3e}  "
              f"mean_err={err.mean().item():.3e}  rel_max={rel:.3f}  "
              f"finite={torch.isfinite(got).all().item()}")
        return ok

    all_ok = True
    all_ok &= report("y",  y_b, y_m)
    all_ok &= report("dx", x_b.grad, x_m.grad)
    return all_ok


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  compute capability: {cc}")
    if cc < (12, 0):
        print(f"  WARNING: MXFP8 path was designed for sm_120; running on cc={cc}")

    ok = True
    # Anima production shapes for unwrapped frozen Linears
    ok &= case(D_in=2048, D_out=2048, BS=8 * 4096)   # self-attn q/k/v/output
    ok &= case(D_in=1024, D_out=2048, BS=8 * 4096)   # rare odd-shape projections
    ok &= case(D_in=2048, D_out=8192, BS=8 * 4096)   # mlp-shape (if not lokr-wrapped)
    ok &= case(D_in=8192, D_out=2048, BS=8 * 4096)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
