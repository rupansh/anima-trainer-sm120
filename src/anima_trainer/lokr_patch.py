"""Monkey-patch `lycoris.modules.lokr.LokrModule.forward` to drop the redundant
materialize-then-subtract pattern and merge `base_W + diff_W` so each
LokrModule does ONE matmul per forward instead of two.

Stock lycoris (LokrModule.forward, non-bypass path):
    base        = self.org_forward(x)                         # 1 big mm
    base_W      = self._current_weight()
    diff_W      = self.get_weight(self.shape) * self.scalar
    new_W       = base_W + diff_W                              # materialize (N,K)
    delta_W     = new_W - base_W                               # == diff_W (waste)
    delta       = self.op(x, delta_W, None, **kw)              # 2nd big mm
    return base + delta

Patched (this file):
    diff_W   = self.get_weight(self.shape) * (scalar * mult)
    merged_W = self.org_module.weight + diff_W                 # 1 add, no waste
    return self.op(x, merged_W, self.org_module.bias, **kw)    # 1 big mm

Backward: autograd derives grad_merged_W = x.T @ grad_y, which flows to
grad_diff_W (base_W is frozen → no contribution) and through `get_weight`
back to lokr_w1 / lokr_w2 as usual. No custom autograd.Function needed.

Caveats:
  * Skips this fast path when `wd` (weight decomposition) is on — `wd` uses
    `apply_weight_decompose` which is a non-trivial transform on the merged
    weight, not a simple add. Falls back to lycoris's original forward.
  * Skips when `bypass_mode` is True — that path doesn't materialize the
    delta and is its own thing.
  * Skips during module_dropout (correctness).
"""
from __future__ import annotations
import torch
from lycoris.modules.lokr import LokrModule


_PATCHED = False
_ORIG_FORWARD = None
# Module-global FP8 toggle. Flipped on by `enable_fp8()` when precision="fp8".
# When True, `_patched_forward` swaps `F.linear(x, merged_W, bias)` for
# `FP8LoKrLinear` — single FP8 GEMM forward + FP8 dgrad + FP8 wgrad. Only
# valid on shapes the FP8 path supports (LoKr-wrapped Linears whose
# out_features and in_features are both divisible by 128).
_FP8_ACTIVE = False


def _patched_forward(self, x: torch.Tensor, *args, **kwargs):
    # Defer to the original forward for the cases we don't optimise.
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)
    if self.bypass_mode:
        return self.bypass_forward(x, self.multiplier)
    if self.wd:
        return _ORIG_FORWARD(self, x, *args, **kwargs)

    org = self.org_module[0]

    alpha = getattr(self, "_fused_alpha", None)
    if alpha is None:
        # Use a Python-float `alpha` cached at attach time. The stock lycoris
        # path `diff_weight * self.scalar` creates a MulBackward0 node per
        # LokrModule because `self.scalar` is a registered buffer (CUDA
        # tensor), even when `use_scalar=False` (its value is constant 1.0).
        # 454 of those per step was ~1.09 s of pure mul-backward cost in the
        # profile.
        alpha = float(self.scalar.item() if torch.is_tensor(self.scalar) else self.scalar) * float(self.multiplier)
        self._fused_alpha = alpha

    # --- merged-weight path ---
    diff_weight = self.get_weight(self.shape)
    if diff_weight.dtype != org.weight.dtype:
        diff_weight = diff_weight.to(org.weight.dtype)
    # When alpha == 1.0, plain `+` is the fast path (no autograd mul node).
    if alpha == 1.0:
        merged_weight = org.weight + diff_weight
    else:
        merged_weight = torch.add(org.weight, diff_weight, alpha=alpha)

    if _FP8_ACTIVE and getattr(self, "_fp8_ok", False):
        # JIT FP8: quantize merged_W + x once, do FP8 GEMM forward, FP8
        # dgrad + wgrad on backward. The same MM the bf16 path would do,
        # in FP8. ~1.2× faster fwd+bwd on the MLP shapes at the cost of
        # ~2.5% gradient noise (within LoRA training tolerance).
        from .fp8_quant import fp8_lokr_linear
        return fp8_lokr_linear(x, merged_weight, org.bias)

    return self.op(x, merged_weight, org.bias, **self.kw_dict)


def _mark_fp8_eligible(network) -> int:
    """Tag each LokrModule whose wrapped Linear has both dims % 128 == 0
    (Float8BlockScaling block size) — only those modules can take the FP8
    path. Returns the count of eligible modules."""
    count = 0
    for mod in network.modules():
        if not isinstance(mod, LokrModule):
            continue
        inner = mod.org_module[0]
        # Only LoKr-wrapped nn.Linear is in scope; LoKr also wraps Conv2d
        # in some presets but the anima preset is Linear-only.
        if not isinstance(inner, torch.nn.Linear):
            mod._fp8_ok = False
            continue
        out_dim, in_dim = inner.weight.shape
        ok = (out_dim % 128 == 0) and (in_dim % 128 == 0)
        mod._fp8_ok = ok
        if ok:
            count += 1
    return count


def enable_fp8(network) -> int:
    """Flip the LoKr forward path to FP8 for `network`. Returns the number
    of eligible LokrModules tagged. Must be called *after* `install()`."""
    global _FP8_ACTIVE
    n = _mark_fp8_eligible(network)
    _FP8_ACTIVE = True
    return n


def install() -> None:
    """Apply the patch globally. Idempotent."""
    global _PATCHED, _ORIG_FORWARD
    if _PATCHED:
        return
    _ORIG_FORWARD = LokrModule.forward
    LokrModule.forward = _patched_forward
    _PATCHED = True
