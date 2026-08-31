# Baseline numbers (RTX PRO 6000 Blackwell, sm120, 96GB)

Captured 2026-05-27 against the unchanged `melted1` config (batch 8, 1024², bf16
mixed precision, gradient checkpointing on, 56 train images, LoKr 26,819 params).

**These baseline numbers are for the OLD lycoris-lora 3.4.0 from pypi**,
which only wrapped 3 modules (26K params). Once we switched to lycoris from
git (which actually supports Anima), the numbers below replace them. Kept for
historical reference; not the current production state.

| metric             | sd-scripts | bf16 eager | bf16+cmpl | mxfp8 eager       | mxfp8+cmpl |
|--------------------|------------|------------|-----------|-------------------|------------|
| step time          | 3.11 s     | 0.87 s     | 0.57 s    | 0.76 s            | 0.43 s     |
| epoch time         | ~22 s      | ~7.4 s     | 5.2 s     | 6.6 s             | 4.2 s      |
| peak VRAM          | 16,813 MiB | 7,665 MiB  | 7,001 MiB | 8,771 MiB         | 7,691 MiB  |
| loss range (early) | 0.05–0.08  | 0.04–0.09  | 0.04–0.09 | 0.04–0.09         | 0.04–0.09  |

## Current numbers (lycoris from git, eager)

`anima-full` is the original "every Linear in every Block" preset (454
modules / 30.6 M params). `anima-cross-mlp` is the focused subset
(`cross_attn.{q,k,v,output}_proj` + `mlp.{layer1,layer2}` per block, 168
modules / 20.2 M params) — the highest-leverage targeting for
multi-character / multi-style LoRAs.

| metric             | sd-scripts | full+lokr_patch | +cross-mlp | +Liger | **+AdaLN fusion** |
|--------------------|------------|-----------------|------------|--------|---------------------|
| step time (steady) | 5.32 s     | 3.65 s          | 3.27 s     | 2.85 s | **2.55 s** (2.09× vs sd-scripts) |
| epoch time (warm)  | ~38 s      | 26.7 s          | ~23.5 s    | ~21.5 s| **~19.8 s**         |
| trainable params   | 30.6 M     | 30.6 M          | 20.2 M     | 20.2 M | 20.2 M              |
| wrapped modules    | 454        | 454             | 168        | 168    | 168                 |

Decomposition of the wide-LoRA → production gap (3.65 → 2.55 s = 30%):
- Preset switch (anima-full → anima-cross-mlp): **~10%** (3.65 → 3.27 s).
  Fewer LokrModules wrapped → fewer big mms and fewer merged-weight adds
  per step. Self-attention's `{q,k,v,output}_proj` and the adaln-modulation
  Linears stop being trained.
- Liger fused RMSNorm: **~13%** (3.27 → 2.85 s). Replaces 113 autocast(fp32)
  round-trips per forward (4 RMSNorms × 28 blocks + t_embedding_norm) with
  single-kernel-launch Triton calls. Loss bit-identical.
- AdaLN fusion patch: **~10%** (2.85 → 2.55 s in the historical bf16 run).
  It replaces Anima's inline `_adaln_fn(x, norm, scale, shift) = norm(x) *
  (1+scale) + shift` with the explicit Triton forward/backward kernel in
  `adaln_kernel.py`. 84 sites × 3 ops collapse to 84 launches. Loss is
  bit-identical on cached training data, with no per-bucket Inductor warmup.

All three effects are independent and compose. Current production defaults
in `melted1.toml`. The `anima-full` preset is still available for runs that
need maximum capacity (single-style, single-character, very high-detail
training); pass `preset = "anima-full"` in the TOML for the wide-LoRA path.

The later precision A/B at the same batch-8 production shape measured
**~2.23 s/step with Transformer Engine block-scaled FP8 versus ~2.52 s/step
with bf16**. See `sm120-optimization-audit-2026-08.md` for the subsequent
upstream/research audit and rejected execution variants.

For per-patch implementation notes, see `patches.md`. For the
broader compile-scope discussion (whole-DiT vs targeted), see
`torch-compile.md`.

Don't claim improvements without re-running the same melted1 config and
recording all metrics above; cache state and the first epoch's warmup step
both bias single-run measurements.
