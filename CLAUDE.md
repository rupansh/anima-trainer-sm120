# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It's structured as: **load-bearing project context here, deep dives in `docs/claude/`**. The index at the bottom lists every topic file — consult it before answering questions about benchmarks, patches, eval/convergence, attention/quantization investigations, or compile scopes.

## Project intent

This repo is a greenfield **LoKr trainer for the Anima image model**, scoped to be **as fast as possible** on a single Blackwell GPU. It is benchmarked against `sd-scripts/anima_train_network.py` — that script is the reference for *correctness*, not for *performance*. sd-scripts is broken below fp32 for Anima, which is the slowness we're attacking.

`sd-scripts/` is vendored as the reference implementation and baseline. Build, don't port: re-use sd-scripts code only where the reference behavior is load-bearing (model loading, sampler math, lokr math) and write everything else from scratch with sm120 in mind. The trainer is built and shipping (`src/anima_trainer/`); see `docs/claude/project-layout.md` for what each module does.

## Hardware & performance targets

- **GPU architecture: NVIDIA Blackwell, sm120 specifically.** This is consumer/workstation Blackwell, not sm100: do not use or propose sm100-only `tcgen05`/TMEM kernels. Pick attention impls and GEMMs that actually dispatch on sm120 (TE/cuBLASLt, cuDNN SDPA, or measured sm120 Triton/CUTLASS paths) before falling back to generic CUDA.
- **VRAM ceiling:** must train **200+ images at 1024px, batch size 8, under 96GB**. >24GB is the *minimum* working VRAM. VRAM is not the optimization axis — wall-clock is.
- **Convergence parity is a later concern:** matching sd-scripts step-for-step is not required. If sd-scripts converges in 20 epochs and we do too, fine. Don't sacrifice step throughput for sample efficiency right now.

## Scope constraints (hard)

These narrow the search space deliberately — do not add knobs the user didn't ask for.

- **Production adapter:** LoKr. A non-LoKr adapter may be added only when a same-shape benchmark demonstrates a material wall-clock or time-to-quality win. Reference LoKr math: `https://github.com/KohakuBlueLeaf/LyCORIS` (`lycoris.kohya`, `algo=lokr`). **Install lycoris from git, not PyPI** — `pip install lycoris-lora==3.4.0` from PyPI doesn't include Anima support (its `full` preset only matches Anima's `FinalLayer`, wrapping just 3 modules / 26k params). Git HEAD (same `3.4.0` version string, confusingly) has the `Block`/`PatchEmbed`/`TimestepEmbedding`/`LLMAdapterTransformerBlock`/`Qwen3Attention`/`Qwen3MLP` patterns and wraps 454 modules / 30.6M params on Anima — ~1100× the capacity. Both `.venv/` and `sd-scripts-venv/` need the git version: `pip install git+https://github.com/KohakuBlueLeaf/LyCORIS.git`. Re-use the LyCORIS LoKr implementation if it's not the bottleneck; reimplement if it is.
- **Optimizer:** Prodigy+ Schedule Free only. Reference: `https://github.com/LoganBooker/prodigy-plus-schedule-free`. The **only** user-facing knob is `d0`. Three values worth knowing:
  - `1e-6`: the Prodigy library default. Prodigy+SF adapts d0 upward automatically, so this conservative start is safe.
  - `1e-4`: the historical default in `src/anima_trainer/optim.py:build()`.
  - **`5e-5`: the value that matches sd-scripts** (`melted1-anima-config.toml:19`); current `melted1.toml` uses this so trajectory comparisons against the baseline are direct. Reset to this value when running comparisons.

  Don't bump d0 manually for "faster initial movement" — that's an anti-pattern; the adaptation handles it. All other Prodigy params (`betas`, `weight_decay`, `use_bias_correction`, `weight_decay_by_lr`, `split_groups`, `factored`, `use_stableadamw`, `stochastic_rounding`, etc.) get tuned defaults baked into `optim.py` — see `melted1-anima-config.toml` `optimizer_args` for the canonical sd-scripts settings.
- **Precision modes:** `bf16` (reference), `mxfp8`, and `fp8` (fast production default; all via TransformerEngine, see `src/anima_trainer/fp8_quant.py`). The earlier just-in-time mxfp8/nvfp4 paths were investigated and removed (see `docs/claude/quantization.md` history section). The current `mxfp8` path pre-quantizes the frozen base weights once at attach time and keeps LoKr deltas in bf16. The current `fp8` path uses TE's `Float8BlockScaling` (128×128 weight blocks, 1×128 activation blocks — the recipe DeepSeek-V3 trained with): unwrapped frozen Linears are swapped for `te.Linear` and run forward+backward in FP8; LoKr-wrapped Linears use `FP8LoKrLinear` (custom autograd Function that JIT-quantizes `merged_W = base + α·diff_W` each step and does FP8 forward + FP8 dgrad + FP8 wgrad). On Anima production shape, `fp8` is **~11% faster than bf16** (2.23 vs 2.52 s/step at bs=8; the ratio holds across batch sizes — see `docs/claude/benchmarks.md`). Attention itself stays bf16 — cuDNN's FP8 SDPA kernels are not built for sm_120 (workstation Blackwell); see `docs/claude/attention-sm120.md`. Saved adapter is always bf16 regardless of training precision.
- **Resolutions:** 512px and 1024px **only**. Buckets are fixed — derive from `crop.py:12` for 1024 and downscale that list proportionally for 512. Cropping itself can be precomputed and cached.
- **What we train:** the DiT only. TE (Qwen3) and VAE stay frozen — never train them. Note: sd-scripts calls this `network_train_unet_only`, but Anima has no UNet — the flag is a legacy name meaning "freeze VAE + TE."

## Anima model anatomy

Anima is a Qwen-based DiT, not a UNet. Loading shape (from `sd-scripts/anima_train_network.py:82-99`):

- **Text encoder:** Qwen3 0.6B base → `./models/qwen_3_06b_base.safetensors`
- **VAE:** Qwen-image VAE → `./models/qwen_image_vae.safetensors`
- **DiT:** Anima base → `./models/anima-base-v1.0.safetensors`
- **Timestep sampling:** flow matching, `sigmoid` (see `melted1-anima-config.toml:72-73`)
- **VAE spatial downscale × patch size = 16** — bucket resolutions must be multiples of 16 (sd-scripts calls `verify_bucket_reso_steps(16)`).

Relevant sd-scripts library files to consult for behavior: `sd-scripts/library/anima_models.py`, `anima_utils.py`, `qwen_image_autoencoder_kl.py`, `strategy_anima.py`.

**Critical gotcha:** the Anima DiT `t_embedder` expects timesteps in `[0, 1]`, NOT `[0, 1000]`. sd-scripts builds `timesteps = sigmas * num_train_timesteps` then divides by 1000 right before the forward call (`anima_train_network.py:280`). Passing the unscaled timesteps inflates loss by ~25× and looks like a broken training run.

## Caching

- **Backend:** LanceDB (`https://www.lancedb.com`), local file. One DB per dataset is fine.
- **Cache aggressively** but **always validate**: source-file mtime + size + content hash. A stale cache silently corrupting training is the failure mode to prevent.
- Cached: cropped images per bucket, VAE latents (gated on a VAE-weights fingerprint), Qwen3 text-encoder outputs for training captions (gated on the TE-weights fingerprint + caption hash). See `cache.py`.
- **Do not cache sample prompts.** The user must be able to edit `sample_prompts` mid-run and have the next sampling tick pick up the new file. Re-read from disk every sample step.

## On-the-fly sampling

- Sampler: plain euler discrete for rectified flow (see `sample.py:euler_denoise`). The config's `sampler = "euler_a"` string is currently cosmetic — only one path is implemented.
- Triggered every N epochs (mirror sd-scripts' `sample_every_n_epochs`).
- Prompt file is hot-reloaded each tick — see prompt syntax in `melted1-ds-prompt.txt` (`--n` negative, `--w/--h`, `--l` CFG, `--s` steps, `--d` seed).

## Layout

```
models/                            # checkpoints (gitignored material)
  anima-base-v1.0.safetensors      # DiT
  qwen_3_06b_base.safetensors      # text encoder
  qwen_image_vae.safetensors       # VAE
melted1-ds/1_melted1/              # training dataset (56 image+caption pairs)
melted1-ds-prompt.txt              # sample prompts (editable during training)
melted1-anima-config.toml          # sd-scripts baseline config
crop.py                            # smartcrop preprocessor + canonical bucket list
sd-scripts/                        # vendored reference trainer + Anima libraries
sd-scripts-venv/                   # venv used to run the sd-scripts baseline
sd-scripts-outputs/                # baseline output dir (configured in the toml)
docs/claude/                       # split-out deep-dive docs (indexed below)
```

For the `src/anima_trainer/` source layout, see `docs/claude/project-layout.md`.

## Running the sd-scripts baseline

This is the comparison target. Run it from repo root:

```bash
source sd-scripts-venv/bin/activate
TOKENIZERS_PARALLELISM=true accelerate launch --num_cpu_threads_per_process 1 \
  sd-scripts/anima_train_network.py --config_file=./melted1-anima-config.toml
```

For fast iteration, edit `melted1-anima-config.toml`: drop `max_train_epochs` and lower `sample_every_n_epochs` to match. Keep `seed = 42` for reproducibility against our trainer.

**Always wipe `sd-scripts-outputs/` between baseline runs** (`rm -rf sd-scripts-outputs/* `). Leftover checkpoints and sample images from prior runs corrupt comparison metrics and waste disk. The comparison/eval script must read from a known-clean baseline directory.

## Anima-trainer venv

The trainer targets **Python 3.14** with the latest PyTorch nightly cu130 (separate from `sd-scripts-venv`, which is the existing Py3.12 baseline env and must not be touched). `.python-version` and `pyproject.toml` are pinned to 3.14.

**`pyproject.toml` is the single source of truth for every runtime dep.** Do not `uv pip install <pkg>` ad hoc — every dep belongs in the `[project.dependencies]` list, and any index that isn't pypi (e.g. the PyTorch nightly cu130 index for `torch` + `torchvision`) belongs in `[[tool.uv.index]]` + `[tool.uv.sources]`. `lycoris-lora` is pinned to a specific git commit because pypi's tagged 3.4.0 silently lacks Anima support — see the dep comment for why. Reproducibility is load-bearing: a fresh `git clone` + `uv sync` must produce the same working environment without secret install commands.

## Current production snapshot

At batch 8 / 1024², `anima-cross-mlp` preset (168 wrapped modules, 20.2 M trainable params), with `lokr_patch` + `liger_patch` + `adaln_patch` applied:

- **~2.23 s/step with the FP8 production default**; latest bf16 A/B was ~2.52 s/step.
- **~2.39× faster than sd-scripts' 5.32 s/step** at the same batch/resolution (adapter surfaces differ; this is a wall-clock reference, not a mathematical parity claim).
- Loss bit-identical to sd-scripts through the patch sequence; bf16-vs-sd-scripts sample latent_cos = **0.977** on the historical tiny-LoRA comparison.
- GUI is explicitly **not a priority**.

With the same production shape, Transformer Engine block-scaled FP8 is
**~2.23 s/step versus ~2.52 s/step for bf16**. The 2026-08-31 upstream and
research audit found no replacement that beats the merged FP8 LoKr block;
see `docs/claude/sm120-optimization-audit-2026-08.md` before proposing a new
compile, fusion, graph-capture, or adapter path. Gradient checkpointing was
deliberately not revisited in that audit.

See `docs/claude/benchmarks.md` for the full table and the wide-LoRA → production decomposition.

## Detailed documentation index (`docs/claude/`)

- **`project-layout.md`** — module-by-module description of `src/anima_trainer/`, tests, and outstanding optimization follow-ups (in rough leverage order).
- **`benchmarks.md`** — current and historical step-time numbers, and the wide-LoRA → production decomposition (cross-mlp + Liger + AdaLN).
- **`patches.md`** — implementation notes for `lokr_patch` (single-matmul merge), `liger_patch` (fused RMSNorm), `adaln_patch` + `adaln_kernel` (custom Triton AdaLN fusion), and the `anima-cross-mlp` LoKr preset.
- **`eval-and-convergence.md`** — `anima-train eval-compare` and `convergence` results, plus the grokking observation for Anima DiT.
- **`attention-sm120.md`** — settled attention backend investigation: shape benchmarks, rejected backends (FA2/3/4, xformers, Sage, custom Triton), and the corrected SGLang fp8 framing.
- **`throughput.md`** — per-step CUDA self-time decomposition; batch_size / gradient-checkpointing / I/O trade-offs.
- **`quantization.md`** — history of mxfp8 / nvfp4 investigations and why they were removed; conditions under which to revisit.
- **`torch-compile.md`** — why `torch.compile` is not used in production (both whole-DiT and the previous targeted-AdaLN scope).
- **`cutedsl-mxfp8.md`** — handoff doc for the CuTeDSL MXFP8 GEMM kernel for sm_120: which primitives to use (`MmaMXF8Op` from `cute.nvgpu.warp.mma`), env requirements (`CUTE_DSL_ARCH=sm_120a`, `nvidia-cublas>=13.4.1`), reference files (CUTLASS 79c + CuTeDSL `dense_gemm_sm120.py` + `nvfp4_gemm_0.py`), the scale-tensor (SFA/SFB) plumbing pattern, and acceptance criteria. Read this first if starting the kernel build.
- **`sm120-optimization-audit-2026-08.md`** — measured audit of `anima_lora`, current adapter research, Transformer Engine 2.18, regional compile, CUDA graphs, LoRA/LoKr execution variants, and the remaining sm120-specific opportunities.
