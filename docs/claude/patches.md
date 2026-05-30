# Patches (lokr_patch, liger_patch, adaln_patch, cross-mlp targeting)

The trainer applies three monkey-patches to the loaded sd-scripts Anima
modules, plus a custom LoKr targeting preset. End-to-end these account for
the 3.65 → 2.55 s/step gap on the wide-LoRA path (see `benchmarks.md`).

## LoKr forward merge (`src/anima_trainer/lokr_patch.py`)

Monkey-patches `lycoris.modules.lokr.LokrModule.forward` to drop a redundant
materialize-then-subtract pattern and merge `base_W + diff_W * scalar` so
each LokrModule does **one matmul** per forward instead of two:

  - stock: `base = x @ base_W.T`; `new_W = base_W + diff_W`;
           `delta_W = new_W - base_W` (== diff_W, wasted bandwidth);
           `delta = x @ delta_W.T`; `return base + delta`  → **2 big mms** + 2
           full-matrix temporaries
  - patched: `merged_W = base_W + diff_W * scalar`;
             `return x @ merged_W.T + bias`  → **1 big mm** + 1 add

Autograd derives `grad_merged_W = x.T @ grad_y` which flows naturally to
`grad_diff_W → grad_lokr_w1, grad_lokr_w2` (base_W frozen). No custom
autograd.Function required. The patch installs idempotently at LoKr-attach
time (see `lokr.py:attach_lokr`).

Verified loss values are bit-identical to the stock lycoris path across the
first 21 steps (same math, fewer ops). VRAM is *lower* than stock too — no
intermediate `new_weight` tensor materialized.

**Whole-DiT compile is not needed**: the merged-weight patch in eager mode
hits exactly the same step time (3.65 s, pre-AdaLN) as the
`compile_mode="default"` whole-DiT compile, without the 77 GB VRAM peak or
124 s warmup. Production keeps `compile_mode=""` and instead relies on the
custom AdaLN Triton kernel (`adaln_kernel.py` + `adaln_patch.py`) — see
`torch-compile.md` for the rejected whole-DiT compile scope.

**MXFP8 / NVFP4** — removed; see `quantization.md`.
Note that the rationale there was partly based on incorrect "no fp8 works
on sm_120" claims that have since been corrected (SGLang's CUTLASS kernel
does work). The actual decision to ship bf16-only stands: the engineering
cost of an autograd-aware fp8 path didn't pay back enough when mm was a
smaller fraction of step time. With current bottleneck distribution
(~50% mm), fp8 is worth revisiting — see "Open follow-ups" in
`project-layout.md`.

## AdaLN custom kernel (`src/anima_trainer/adaln_patch.py` + `adaln_kernel.py`)

Anima's `Block._forward` (sd-scripts/library/anima_models.py:867) defines an
inline helper:

    def _adaln_fn(_x, _norm_layer, _scale, _shift):
        return _norm_layer(_x) * (1 + _scale) + _shift

Called 3× per Block (self-attn, cross-attn, MLP) × 28 blocks = 84 sites per
forward. Each call issues 3 separate kernels (`layer_norm`, `mul`, `add`).
Profiler showed `aten::layer_norm` alone burned ~195 ms/step (7%); the chained
elementwise ops added more, and the launch-overhead-bound nature
(70% wall-clock-to-real-work ratio per kernel) made it the highest-leverage
remaining fusion target.

`adaln_kernel.py` implements a custom Triton kernel pair:

1. **Forward** (1 kernel): per-row LayerNorm stats in fp32, then
   `(1+scale)*xhat + shift` fused, store as bf16. Saves `mean, rstd` for
   backward. The (b, t) → (h, w) broadcast is encoded as `bt = row // HW`
   inside the kernel — no broadcast materialization.
2. **Backward dx** (1 kernel): row-wise LN backward using the saved
   `(mean, rstd)` and `(1 + scale)` as the effective LN weight.
3. **Backward dscale/dshift** (1 kernel): reduces `dy * xhat` and `dy`
   across `(h, w)` per `(b, t)` with a 3-D grid `(BT, HW_chunks, D_chunks)`
   and `atomic_add` into fp32 accumulators (cast to bf16 after). The atomics
   target a tiny `BT × D ≈ 16K-cell` buffer, so contention is negligible.

`adaln_patch.install()` monkey-patches `Block._forward` to call
`FusedAdaLN.apply(x, scale, shift, eps)` at the three sites, otherwise
identical control flow. Wired in `train.py` after `install_liger_patch()`.

**Why a hand-written kernel over `torch.compile`:** an earlier version used
`torch.compile` on a 3-op pure function and hit the same per-step win
(~10%). The compile path paid an extra ~700 ms inductor warmup per new
bucket resolution and added a tooling dependency that resists profiling and
debugging. The Triton kernel has no warmup, an explicit and readable
backward, and microbench parity with `torch.compile` at the production
shape (B=8, 1024², D=2048): forward 0.165 ms/iter (custom) vs 0.182 ms/iter
(compile); fwd+bwd 0.686 ms vs 0.659 ms — same ballpark, no compile state.

Numerical model: all stats and intermediate math in fp32, output cast to
bf16 on store. The kernel is strictly closer to fp32 ground truth than
PyTorch eager bf16 on `dscale` (eager's `(dy * n_ref)` mul happens in bf16;
the kernel keeps `xhat` in fp32 through the reduction). Parity test:
`tests/sanity_adaln_kernel.py` — forward / dx / dshift match eager to bf16
noise floor; dscale max-error against fp32 ground truth is ~50% of eager's
at 1024². fp32 inputs match to ~1e-5 absolute error.

End-to-end vs unpatched: same as the compile path (~10% step time win).
Loss bit-identical to the unpatched path on real (cached) training data.

## Liger RMSNorm patch (`src/anima_trainer/liger_patch.py`)

Anima's `RMSNorm.forward` (sd-scripts/library/anima_models.py:223) opens a
`torch.autocast(fp32)` block, casts the input to fp32, computes the norm
with `pow`/`mean`/`rsqrt`/`mul`, casts back to bf16, and multiplies by
weight. There are 4 of these per Block (`q_norm`, `k_norm` on self- and
cross-attn) × 28 blocks + `t_embedding_norm` = 113 calls/forward.

`liger_patch.install()` swaps the forward for `LigerRMSNormFunction` with
`casting_mode="llama"` (fp32 stats, bf16 weight multiply — matches Anima's
semantics). Numerical parity confirmed in `tests/sanity_cross_mlp_liger.py`:
max abs diff 7.8e-3, mean 4.96e-9 — bf16 noise floor. End-to-end
contribution measured by toggling the patch on/off with everything else
held constant: **3.27 → 2.85 s/step (≈13% faster)**. Loss is bit-identical
between patched and unpatched runs — pure kernel-scheduling win. Wired in
`train.py` after `ensure_on_path()`.

## Anima-aware LoKr targeting (`anima-cross-mlp` preset)

The `anima-full` preset wraps every Linear in every Block (454 modules,
30.6 M params). For multi-character / multi-style training the highest-
leverage subset is much smaller:

- **cross-attention** `{q,k,v,output}_proj` — where text tokens (character
  triggers, style triggers) map into image features. Wrap → identity and
  style absorption.
- **MLP** `{layer1, layer2}` (GPT2FeedForward) — channel mixing; carries
  texture / stylistic detail.
- **Skip self-attention** — composition/spatial structure already lives in
  the base model.
- **Skip adaln_modulation, FinalLayer, PatchEmbed** — infrastructure.

Result: 168 modules, 20.2 M trainable params. Step time was **2.85 s** with
cross-mlp + lokr_patch + Liger (pre-AdaLN); the AdaLN compile patch brings
it to **2.55 s** in current production. Match by fully-qualified module
name with fnmatch globs — `attach_lokr` flips
`LycorisNetworkKohya.USE_FNMATCH = True` because `unet_target_name` defaults
to `re.match` which would interpret `.` and `*` as regex tokens, not
path separators / wildcards.
