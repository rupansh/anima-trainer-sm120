# torch.compile

The trainer **does not use `torch.compile`** in production. This file
documents the two scopes that were investigated and rejected.

## Whole-DiT compile (`train.compile_mode = "default"` etc.) — **not used in production**

`train.compile_mode` selects `torch.compile(models.dit, mode=...)`. Results
with `lokr_patch` applied:

| mode             | warmup | step time (wide-LoRA) | peak VRAM | verdict |
|------------------|--------|------------------------|-----------|---------|
| `""` (eager)     | 1 s    | 3.65 s (pre-AdaLN)     | 16 GB     | production path |
| `"default"`      | 124 s  | 3.65 s                 | **77 GB** | matches eager speed but 4.7× VRAM blowup |
| `"max-autotune"` | 300 s+ | ≈ 3.65 s               | 77 GB+    | not worth it |

The 4.7× VRAM comes from inductor's activation-save heuristics applied
across the 454 LoKr-wrapped modules in the compiled graph. **Default to
eager** for the DiT-level compile. Compile is only worth that VRAM if you
have specific reasons (e.g. distributed inductor caching across runs).

Caveats from this investigation:

- Each new bucket shape triggers a recompile (~30 s with tiny LoRA; up to
  124 s with wide LoRA). For multi-bucket datasets that's a lot of warmup.
- Sampling pays ~70 s per unique sample-prompt resolution on first call
  (B=1 inference shape differs from B=8 training shape).
- Compile must happen AFTER LoKr is attached. Both swap forward functions
  on submodules; wrong order silently breaks the compiled graph.
- The sampling tick must not wrap calls in `cudnn_only()` — the VAE
  decoder has internal attention with shapes cuDNN doesn't have a kernel
  for. Let torch's default selector pick at sampling time.

## Targeted compile of the `_adaln_fused` helper — **replaced by a Triton kernel**

A previous iteration used `torch.compile` on a 3-op pure function
(`layer_norm → mul → add`), invoked from a monkey-patched
`Block._forward`. Because the compiled region was tiny and contained no
LoKr modules, inductor had nothing to over-save and none of the 77 GB /
124 s pathology applied. Warmup: ~700 ms per resolution. Win: 10% step
time.

That path was replaced by a hand-written Triton kernel in
`src/anima_trainer/adaln_kernel.py` — same speed at the production shape,
no inductor warmup per bucket, explicit backward. See `patches.md` →
"AdaLN custom kernel" for the kernel design and microbench numbers.
