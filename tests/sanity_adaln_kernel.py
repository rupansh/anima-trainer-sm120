"""Numerical-parity check for the custom AdaLN Triton kernel.

Verifies `FusedAdaLN.apply(x, scale, shift, eps)` matches the reference
PyTorch expression `F.layer_norm(x, [D], None, None, eps) * (1+scale) + shift`
on both forward and backward (dx, dscale, dshift), at bf16 and fp32.

Run: `python -m tests.sanity_adaln_kernel`
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.adaln_kernel import FusedAdaLN


def _reference(x, scale, shift, eps):
    n = F.layer_norm(x, [x.shape[-1]], None, None, eps)
    return n * (1 + scale) + shift


def _check(name, ref, got, atol, rtol):
    diff = (ref.float() - got.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(ref.float(), got.float(), atol=atol, rtol=rtol)
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name}: max={max_diff:.3e}  mean={mean_diff:.3e}")
    return ok


def _case(B, T, H, W, D, dtype, atol, rtol):
    print(f"\n--- B={B} T={T} H={H} W={W} D={D} dtype={dtype} ---")
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-6

    x = torch.randn(B, T, H, W, D, dtype=dtype, device=device, requires_grad=True)
    scale = torch.randn(B, T, 1, 1, D, dtype=dtype, device=device, requires_grad=True) * 0.1
    shift = torch.randn(B, T, 1, 1, D, dtype=dtype, device=device, requires_grad=True) * 0.1

    # PyTorch-eager reference (same dtype as kernel input)
    x_r = x.detach().clone().requires_grad_(True)
    s_r = scale.detach().clone().requires_grad_(True)
    b_r = shift.detach().clone().requires_grad_(True)
    y_ref = _reference(x_r, s_r, b_r, eps)

    # Kernel path
    x_k = x.detach().clone().requires_grad_(True)
    s_k = scale.detach().clone().requires_grad_(True)
    b_k = shift.detach().clone().requires_grad_(True)
    y_kern = FusedAdaLN.apply(x_k, s_k, b_k, eps)

    # fp32 ground truth (all ops in fp32, then cast back to the kernel dtype
    # to bring rounding error onto the same surface — the kernel quantizes on
    # store, so any gap with fp32_truncated is true kernel error, while any
    # gap between eager and fp32_truncated is PyTorch's eager bf16 noise).
    x_g = x.detach().float().clone().requires_grad_(True)
    s_g = scale.detach().float().clone().requires_grad_(True)
    b_g = shift.detach().float().clone().requires_grad_(True)
    y_gt = _reference(x_g, s_g, b_g, eps)

    all_ok = True
    all_ok &= _check("forward y vs eager", y_ref, y_kern, atol, rtol)

    torch.manual_seed(1)
    dy_kern_dtype = torch.randn_like(y_ref)
    dy_fp32 = dy_kern_dtype.float()

    y_ref.backward(dy_kern_dtype)
    y_kern.backward(dy_kern_dtype)
    y_gt.backward(dy_fp32)

    all_ok &= _check("dx vs eager",         x_r.grad, x_k.grad, atol, rtol)
    all_ok &= _check("dshift vs eager",     b_r.grad, b_k.grad, atol, rtol)

    # For dscale in bf16, eager keeps n_ref in bf16 and accumulates
    # bf16(dy)*bf16(n_ref) over HW=O(1000s) positions; the kernel keeps xhat
    # in fp32 throughout, then casts the final ds to bf16. Compare against
    # an fp32 ground truth — the kernel should be at least as close as eager.
    err_eager  = (s_r.grad.float() - s_g.grad.float()).abs().max().item()
    err_kernel = (s_k.grad.float() - s_g.grad.float()).abs().max().item()
    if dtype == torch.float32:
        # Both eager and kernel are fp32; demand tight absolute parity.
        ok = err_kernel <= atol
    else:
        # bf16: kernel can be up to 1.5× as noisy as eager (typically tighter).
        # Use a small floor so the comparison isn't degenerate when eager is 0.
        ok = err_kernel <= max(err_eager * 1.5, 1e-3)
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] dscale vs fp32-gt: eager max={err_eager:.3e}  kernel max={err_kernel:.3e}")
    all_ok &= ok
    return all_ok


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    ok = True

    # Anima production shape: 1024px / B=8 → H=W=64, D=2048, T=1
    ok &= _case(8, 1, 64, 64, 2048, torch.bfloat16, atol=5e-2, rtol=5e-2)
    # 512px shape
    ok &= _case(8, 1, 32, 32, 2048, torch.bfloat16, atol=5e-2, rtol=5e-2)
    # Tight fp32 check — kernel should match reference to ~1e-5
    ok &= _case(4, 1, 16, 16, 1024, torch.float32, atol=1e-4, rtol=1e-4)
    # Non-power-of-2 D to exercise the mask path
    ok &= _case(2, 1, 8, 8, 768, torch.float32, atol=1e-4, rtol=1e-4)

    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
