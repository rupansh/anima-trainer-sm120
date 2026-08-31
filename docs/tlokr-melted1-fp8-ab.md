# T-LoKr vs LoKr: melted1 FP8 A/B

Run date: 2026-09-01

This is a matched 30-epoch A/B on an NVIDIA RTX PRO 6000 Blackwell
Workstation Edition (driver 610.57.04). The environment was resolved only by
`uv` from `pyproject.toml`/`uv.lock`: PyTorch
2.15.0.dev20260830+cu130, Triton 3.8.0, and Transformer Engine 2.18.0.

## Method

Both arms used 1024px, batch size 4, seed 42, 30 epochs / 420 optimizer
steps, FP8 precision, factor 8, the `anima-cross-mlp` target set, Prodigy+SF
`d0=5e-5`, checkpoint/sample cadence 2, and the same 56-sample cache. A config
diff confirmed that only output identity and
`lokr.variant = "lokr" | "tlokr"` differed. T-LoKr used rank 128 with a 0.5
minimum-rank ratio.

This matches the user-facing training configuration, but not adapter capacity.
With `full_matrix=true`, ordinary LoKr stores W2 directly and its nominal
network rank is not a W2 constraint. T-LoKr must introduce a low-rank axis to
schedule, so it replaces W2 with a rank-128 `A @ B`. Of 168 targets, 112 have
a maximum W2 rank of 256 and are therefore capacity-constrained to half rank;
the other 56 have maximum rank 128. The original two-arm run is a practical
adapter comparison, not a schedule-only ablation.

The original `melted1-ds` source directory was unavailable. For this A/B only,
the already validated processed cache was copied to
`outputs/melted1-ab.cache.lance`; its exact cached JPEG crops and captions were
restored as source files and only the copy's source fingerprints were updated.
Cached latent/text bytes were unchanged. This makes the two arms directly
comparable to each other, but this run should be described as using the
recovered processed snapshot rather than the missing original source set.

## Result

| metric | LoKr | T-LoKr | T-LoKr delta |
|---|---:|---:|---:|
| trainable parameters | 20,195,840 | 15,608,320 | -22.72% |
| final adapter size | 40.45 MB | 31.31 MB | -22.59% |
| training-loop wall time | 7:07 | 7:49 | +9.84% |
| steady training rate | ~1.37 step/s | ~1.25 step/s | -8.96% |
| peak allocated VRAM | 55.97 GB | 56.10 GB | +0.13 GB |
| final displayed batch loss | 0.0427 | 0.0430 | +0.0003 |
| convergence mean, epoch 10 (n=8) | 0.5988 | 0.5795 | -0.0193 |
| convergence mean, epoch 30 (n=8) | 0.6458 | 0.6264 | -0.0194 |

The fixed-caption convergence probe used stable content-derived per-image
noise seeds. It measures cosine similarity between each generated latent and
the corresponding cached training-image latent. It therefore rewards exact
training-image reproduction/memorization; it is not a prompt alignment,
diversity, or general image-quality score.

To separate the topology change from the timestep schedule, a third arm used
the identical rank-128 factorized T-LoKr topology but set
`timestep_min_rank_ratio=1.0`, keeping every component active at every
timestep:

| arm | active rank | trainable parameters | epoch-10 latent cosine | epoch-30 latent cosine | final displayed loss |
|---|---:|---:|---:|---:|---:|
| direct full-matrix LoKr | direct W2 | 20,195,840 | 0.5988 | 0.6458 | 0.0427 |
| static factorized LoKr control | 128 | 15,608,320 | 0.5864 | 0.6409 | 0.0429 |
| scheduled T-LoKr | 64..128 | 15,608,320 | 0.5795 | 0.6264 | 0.0430 |

At epoch 30, only 0.0049 (25%) of the original 0.0194 gap comes from the
rank-128 factorization/capacity change. The timestep schedule accounts for
0.0145 (75%). At epoch 10 the split was 0.0124 (64%) factorization and 0.0069
(36%) scheduling, so the static factorization largely caught up while the
scheduled adapter continued to resist exact-image convergence. All three arms
ended at essentially the same displayed training loss.

This behavior is consistent with T-LoRA's stated purpose: suppressing
single/few-image overfitting while improving the balance between concept
fidelity, text alignment, and diversity. The paper also describes slightly
slower concept learning. Consequently, lower performance on this particular
memorization-sensitive probe is expected and does not by itself show a faulty
implementation.

The repository comparison tool matched all 30 generated pairs. Its aggregate
difference metrics were MSE 0.0294, PSNR 16.122, SSIM 0.6780, LPIPS 0.3201,
and Anima-VAE latent cosine 0.8870. These quantify divergence between the two
trajectories; they do not say which result is better. Direct inspection of the
two epoch-30 prompt pairs found both coherent and prompt-compliant, without a
defensible subjective winner.

No algebra or fused-kernel defect was found: the timestep direction and rank
formula match the official Vanilla T-LoRA implementation, and reference-vs-
fused CUDA tests cover forward output plus gradients for input, W1, A, and B.
This implementation is Vanilla T-LoKr, not the paper's SDXL orthogonal
parameterization; the official FLUX path likewise uses the Vanilla form.

If the objective is fastest reproduction of the training images, ordinary
LoKr is the better result here. If the objective is the T-LoRA tradeoff, this
convergence probe cannot choose a winner. That requires held-out compositional
prompts and separate concept-fidelity, text-alignment, and diversity metrics.
T-LoKr remains a smaller-adapter option, trading roughly 23% fewer parameters
and disk bytes for roughly 10% more wall time in this implementation.

Raw metric logs are in `outputs/tlokr-vs-lokr.eval.txt`,
`outputs/melted1-fp8/convergence*.txt`, and
`outputs/melted1-tlokr-fp8/convergence*.txt`. The schedule-only control is in
`outputs/melted1-tlokr-static-fp8/`, including `convergence-e30.txt` and
`train-30.log`.

## T-LoKr format and execution

The implementation applies Vanilla T-LoRA's linear timestep rank schedule to
LoKr's large Kronecker operand. It keeps the small operand full, factors the
large operand as `A @ B`, and evaluates the structured projection without
materializing either `A @ B` or the full Kronecker delta. Aligned large-factor
GEMMs use Transformer Engine block-scaled FP8; custom Triton kernels fuse the
per-example rank mask with the 8x8 small-factor forward/dgrad and fuse its fp32
weight-gradient reduction. Uniform-timestep sampling slices inactive rank
columns away.

T-LoKr uses a separate versioned adapter format within the `.safetensors`
container: `lokr_w1`, `lokr_w2_a`, `lokr_w2_b`, and
`tlokr_schedule=[1, max_rank, min_rank]`, plus
`anima_adapter_type="tlokr"` metadata. Strict loading rejects ordinary LoKr or
an incompatible T-LoKr schedule. Materializing `A @ B` could produce a static
LoKr export, but would discard timestep-dependent behavior.

Design reference: [T-LoRA: Timestep-Aware Low-Rank Adaptation for Diffusion
Models](https://arxiv.org/html/2507.05964v2).
