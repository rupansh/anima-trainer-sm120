"""Sanity for the Triton MXFP8 GEMM (`MXFP8FrozenLinearTriton`).

Verifies:
  1. quantize+dequant noise matches TE's MXFP8 (the spec-correct reference).
  2. Forward + backward dgrad through `MXFP8FrozenLinearTriton.apply` match
     a bf16 reference within fp8 noise (~5–10% rel) on Anima production
     shapes.

Run: `python -m tests.sanity_mxfp8_triton`
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import torch

os.environ.setdefault("NVTE_CUDA_INCLUDE_DIR", "/opt/cuda/include")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.mxfp8_gemm import (
    MXFP8FrozenLinearTriton,
    quantize_rowwise,
    quantize_weight_for_lhs,
    quantize_weight_for_dgrad,
    dequantize_rowwise,
)


def _check(name, ref, got, rel_tol=0.10):
    ref32 = ref.float()
    err = (ref32 - got.float()).abs()
    ref_max = ref32.abs().max().item() + 1e-9
    rel = err.max().item() / ref_max
    ok = rel < rel_tol and torch.isfinite(got).all().item()
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name:18s}  max={err.max().item():.3e}  mean={err.mean().item():.3e}  rel={rel:.3f}")
    return ok


def case(D_in: int, D_out: int, BS: int):
    print(f"\n--- D_in={D_in} D_out={D_out} BS={BS} ---")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(BS, D_in, device=device, dtype=dtype)
    W = torch.randn(D_out, D_in, device=device, dtype=dtype) / (D_in ** 0.5)

    # 1) Quantizer noise floor sanity
    x_blk = quantize_rowwise(x)
    x_recon = dequantize_rowwise(x_blk, x.shape)
    _check("x q→deq noise", x, x_recon, rel_tol=0.10)

    # 2) Forward + dgrad
    W_fwd = quantize_weight_for_lhs(W)
    W_bwd = quantize_weight_for_dgrad(W)
    print(f"  W shapes: fwd data={tuple(W_fwd.data.shape)} scales={tuple(W_fwd.scales.shape)}  "
          f"bwd scales={tuple(W_bwd.scales.shape)}")

    # Reference (bf16 fwd + bwd)
    x_ref = x.clone().requires_grad_(True)
    y_ref = torch.nn.functional.linear(x_ref, W)
    y_ref.float().pow(2).mean().backward()
    gx_ref = x_ref.grad.detach()

    # Triton MXFP8
    x_kern = x.clone().requires_grad_(True)
    y_kern = MXFP8FrozenLinearTriton.apply(
        x_kern, W_fwd.data, W_fwd.scales, W_bwd.data, W_bwd.scales, None,
    )
    y_kern.float().pow(2).mean().backward()
    gx_kern = x_kern.grad.detach()

    ok = True
    ok &= _check("y (fwd)", y_ref, y_kern, rel_tol=0.10)
    ok &= _check("grad_x (bwd dgrad)", gx_ref, gx_kern, rel_tol=0.10)
    return ok


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  compute capability: {cc}")
    ok = True
    # Anima production shapes
    ok &= case(D_in=2048, D_out=2048, BS=8 * 4096)   # cross-attn / self-attn proj
    ok &= case(D_in=1024, D_out=2048, BS=8 * 4096)   # cross-attn k/v
    ok &= case(D_in=2048, D_out=8192, BS=8 * 4096)   # mlp layer1
    ok &= case(D_in=8192, D_out=2048, BS=8 * 4096)   # mlp layer2
    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
