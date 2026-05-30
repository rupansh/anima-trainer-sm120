# Eval and convergence

Two CLI subcommands measure correctness/quality from different angles:

- `anima-train eval-compare A B` — pairwise sample comparison between two
  trainer outputs at matched (epoch, prompt_idx, seed) keys.
- `anima-train convergence <config> <lora>` — single-checkpoint metric;
  picks N training captions and measures latent cosine vs cached training
  latent.

## Comparison-eval results

All via `anima-train eval-compare A B --vae`.

### 15 epochs, current production (cross-mlp + Liger + AdaLN, d0=5e-5)

Captured 2026-05-27 against sd-scripts baseline (also 15 epochs, d0=5e-5).
Sampling once at epoch 15; 2 prompts (frieren, red-hat) compared.

| prompt                       | MSE    | PSNR  | SSIM  | LPIPS | latent_cos |
|------------------------------|--------|-------|-------|-------|------------|
| 00 (red hat, simple)         | 0.034  | 14.74 | 0.744 | 0.221 | **0.899**  |
| 01 (frieren, detail-heavy)   | 0.092  | 10.36 | 0.459 | 0.534 | **0.704**  |
| **mean**                     | 0.063  | 12.55 | 0.601 | 0.377 | **0.802**  |

Wall-clock the same run: sd-scripts 9:44 (5.25 s/step × 105), ours 5:30
(2.75 s/step on a warmer GPU — steady-state production benchmark is
2.55 s/step). **1.77× total wall-clock speedup at matched config.**

Interpretation: at 15 epochs neither trainer has groked the style, so this
measures mid-grokking trajectory agreement, not converged-LoRA agreement.
Simple-prompt cos already at 0.90; detail prompt drifts to 0.70 because
small bf16 ordering differences compound through 30 euler steps on
high-detail generations. Expected to ratchet up by 30 epochs (see below).

### 30 epochs (historical, pre-git-lycoris)

Captured earlier against the **broken-pypi LoKr** (3 modules, 26K params).
The comparison is between trainers, not between any production state.
Kept for the bf16-vs-sd-scripts agreement number, which is the relevant
"is our trainer numerically faithful" data point.

| pair                 | latent_cos | PSNR (dB) | SSIM     | LPIPS    | note |
|----------------------|------------|-----------|----------|----------|------|
| bf16 vs sd-scripts   | **0.977**  | **24.1**  | **0.90** | **0.11** | virtually identical |
| bf16 vs mxfp8        | 0.901      | 16.6      | 0.65     | 0.28     | historical (mxfp8 removed) |
| mxfp8 vs sd-scripts  | 0.893      | 16.0      | 0.62     | 0.30     | historical |

The 0.977 bf16-vs-sd-scripts number is from the tiny-LoRA era and is
load-bearing: it's the strongest evidence we have that our forward + flow-
matching + LoKr application reproduces sd-scripts to bf16 noise.

CLI:

```bash
anima-train eval-compare outputs/<A>/samples outputs/<B>/samples --vae
```

`--vae` adds the latent-cosine metric (loads Anima VAE; ~5 s setup).

## Convergence metric — and the grokking observation

`anima-train convergence <config> <lora>` picks N random training captions,
samples via the LoRA at the matching bucket resolution, and computes
cosine vs the cached training latent. Absolute value is not interpretable
in isolation (one caption matches many valid images); the **trend across
epochs** is the signal.

Recorded snapshots on melted1, seed=42:

| run                          | n  | mean | median | min  | max  | preset           |
|------------------------------|----|------|--------|------|------|------------------|
| 15ep, current production     | 8  | 0.51 | 0.52   | 0.22 | 0.69 | anima-cross-mlp  |
| 15ep, sd-scripts             | 8  | 0.42 | 0.48   | 0.06 | 0.58 | (full Anima)     |
| 30ep, tiny-LoRA bf16 (hist.) | 12 | 0.51 | 0.54   | 0.22 | 0.77 | pypi `full`      |
| 30ep, tiny-LoRA mxfp8 (hist.)| 12 | 0.53 | 0.54   | 0.35 | 0.67 | pypi `full` (removed) |

Mean cosine ~0.5 across all configurations is consistent with "the LoRA
produces semantically related outputs but has not memorized the training
distribution yet." Per the user, this matches their Anima experience —
**grokking is significant with Anima DiT**; the style typically doesn't
appear in samples until substantially more than 30-40 epochs of melted1.
The metric is a tool for measuring *when* it does appear, not a claim that
30 epochs should suffice.

Loading sd-scripts checkpoints into our convergence path: use a
config with `preset = "anima-full"` (matches the 451-module structure
sd-scripts ships) so the state_dict load doesn't mismatch.

Caution interpreting small spreads: the 0.51 (ours) vs 0.42 (sd-scripts)
gap at 15ep is within the per-prompt min/max noise (0.06 to 0.69 across
8 samples), not a claim that we converge faster.

Convergence levers if you want to chase grokking faster:
- Larger LoKr (`preset = "anima-full"`, 30.6 M trainable, ~1.5× our cross-mlp
  capacity).
- Longer training (60+ epochs).
- Caption hygiene — single consistent trigger token per character/style;
  see the multi-character recommendation in the project history.
- Do **not** bump d0 (adapts automatically).
