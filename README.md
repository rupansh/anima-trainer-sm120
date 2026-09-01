# anima-trainer

Fast LoKr trainer for the Anima image model on NVIDIA Blackwell workstation GPUs (sm120 — RTX PRO 6000 / RTX 5090). Targets bf16 + cuDNN attention + custom kernel patches; ~2× faster per step than `sd-scripts/anima_train_network.py` at numerically comparable output.

PS: I didn't write a single line of code for this project and plan to keep it that way.

## Requirements

- NVIDIA Blackwell **workstation** GPU (sm120); ≥24 GB VRAM (96 GB recommended for batch 8 / 1024²).
- Python **3.14** (`.python-version`-pinned).
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Linux x86_64 (the nightly PyTorch cu130 wheels we use only ship for this platform).

## Install

```bash
git clone --recurse-submodules https://github.com/rupansh/anima-trainer-sm120
cd anima-trainer
uv sync
```

`pyproject.toml` is the single source of truth — `uv sync` pulls torch nightly cu130, pins lycoris to the git commit that supports Anima, and installs everything else from pypi. No side commands.

Then drop the model files into `models/`:

```
models/
  anima-base-v1.0.safetensors    # DiT
  qwen_3_06b_base.safetensors    # text encoder
  qwen_image_vae.safetensors     # VAE
```

## Dataset layout

```
<dataset-name>/
  <repeats>_<name>/
    img1.png
    img1.txt          # caption matching img1.png
    img2.png
    img2.txt
    ...
```

Captions live in `.txt` files next to each image. `<repeats>` is an integer prefix (e.g. `1_melted1` = 1× repeats); kept for sd-scripts compatibility.

## Train

Edit `melted1.toml` (or copy and edit) — the user-facing knobs are:

| key | default | meaning |
|---|---|---|
| `train.resolution` | 1024 | 512 or 1024 |
| `train.batch_size` | 8 | |
| `train.max_train_epochs` | 15 | |
| `train.save_every_n_epochs` | 2 | adapter + resumable-state checkpoint cadence |
| `train.precision` | `"fp8"` | Fast production path; use `"bf16"` as the reference mode |
| `train.seed` | 42 | |
| `train.num_workers` | 8 | persistent cache-reader processes; automatically falls back to 0 after an abrupt worker death |
| `optim.d0` | 5e-05 | Prodigy+SF starting estimate; adapts upward automatically — leave alone |
| `lokr.preset` | `"anima-cross-mlp"` | `"anima-cross-mlp"` (168 modules, fast) or `"anima-full"` (454 modules, higher capacity) |
| `lokr.factor` | 8 | LoKr decomposition factor |
| `lokr.variant` | `"lokr"` | `"lokr"` or timestep-aware `"tlokr"` |
| `lokr.timestep_min_rank_ratio` | 0.5 | T-LoKr rank fraction used at maximum noise; ignored by ordinary LoKr |
| `sample.every_n_epochs` | 15 | sampling cadence |
| `sample.prompts_file` | `./melted1-ds-prompt.txt` | hot-reloaded each tick |
| `paths.*` | — | DiT/Qwen3/VAE/dataset/output/cache locations |

Then:

```bash
# 1) one-time cache build (cropped images + VAE latents + Qwen3 text embeds)
uv run anima-train precompute melted1.toml

# 2) train
uv run anima-train train melted1.toml
```

Checkpoints land in `paths.output_dir/`. Samples (if `sample.every_n_epochs` is set) go in `paths.output_dir/samples/`.

To edit sample prompts mid-run: just save `sample.prompts_file` — the next sampling tick reads from disk and picks up the changes.

### Timestep-aware LoKr

Set `lokr.variant = "tlokr"` to apply the vanilla T-LoRA rank schedule to
LoKr's large Kronecker operand. At noisy flow timesteps only a prefix of the
factor rank is active; the rank increases linearly toward the clean endpoint.
`full_matrix = true` is required. The default 0.5 minimum-rank ratio follows
the paper's recommended schedule.

T-LoKr does not materialize the large LoKr delta. It uses structured factor
GEMMs, Transformer Engine block-scaled FP8 on aligned production shapes, and
fused Triton kernels for timestep masking, the small Kronecker projection, and
its weight gradient. Sampling slices inactive factor columns out entirely.
Saved adapters include a format marker and schedule parameters, so strict loads
cannot silently interpret T-LoKr weights as ordinary LoKr.

Calling the attached DiT normally as `dit(x, timesteps, ...)` automatically
scopes the T-LoKr rank schedule to that forward. Low-level calls to an adapter
module still require `set_timestep()`/`clear_timestep()` explicitly; training
with activation checkpointing captures the same context for block
recomputation during backward.

### Crash recovery

Every adapter checkpoint also publishes a full resumable training state under
`paths.cache_db/_training_state/`. The adapter is written first, then the
optimizer, LoKr weights, epoch/global-step position, and Python/NumPy/CPU/CUDA
RNG states are atomically published. Sampling happens afterward, so a sampling
failure cannot destroy the checkpoint boundary.

When a compatible state exists, an interactive training run asks whether to
resume it (default: yes). Automation must make the choice explicit:

```bash
# Resume without prompting.
uv run anima-train train melted1.toml --resume

# Deliberately start over and quarantine the previous state.
uv run anima-train train melted1.toml --no-resume
```

Compatibility covers model hashes, exact cached latents/text embeddings and
sample order, training-math settings, LoKr/optimizer settings, and runtime
versions. Worker count and checkpoint/sample cadence may be changed freely;
the target epoch may be extended or changed to any value not earlier than the
cached epoch. Corrupt or incompatible state is
quarantined and a normal run starts fresh; explicit `--resume` fails instead of
silently changing trajectory. The active state is removed only after the final
adapter has been written successfully.

Portable `.safetensors` files contain adapter weights only. A checkpoint made
before resumable-state support can be used as weights, but cannot reconstruct
the old optimizer or RNG trajectory exactly.

DataLoader workers persist across epochs. If a worker exits abruptly, training
reconstructs that epoch's deterministic batch plan, skips batches already
completed, and permanently continues with `num_workers=0`. Dataset/cache
exceptions still propagate rather than being mistaken for a dead worker.

## Prompt syntax

`melted1-ds-prompt.txt` — one prompt per non-empty/non-comment line. Per-line flags:

- `--n <text>` — negative prompt
- `--w <int>` / `--h <int>` — output resolution
- `--l <float>` — CFG scale
- `--s <int>` — sampling steps
- `--d <int>` — seed

Example:

```
a girl in a red hat --n bad anatomy --w 1024 --h 1024 --l 5 --s 30 --d 344114142
```

## Evaluation

Compare two sample directories (e.g. yours vs sd-scripts) pairwise:

```bash
uv run anima-train eval-compare outputs/A/samples outputs/B/samples --vae
```

Reports MSE / PSNR / SSIM / LPIPS / VAE-latent cosine per matched (epoch, prompt_idx, seed). `--vae` adds the latent metric (slower — loads the Anima VAE).

See [`docs/tlokr-melted1-fp8-ab.md`](docs/tlokr-melted1-fp8-ab.md) for the
matched 30-epoch T-LoKr/LoKr run and its throughput, memory, size, convergence,
and paired-image results.

Convergence metric on a single LoRA checkpoint:

```bash
uv run anima-train convergence melted1.toml outputs/melted1/melted1-anima.safetensors -n 8
```

Samples N random training captions and reports the cosine between predicted and cached training latents. Absolute value is not interpretable in isolation; the **trend across epochs** is the signal.

## Running the sd-scripts baseline (for comparison)

There's a separate `sd-scripts-venv/` (Python 3.12) used to run the upstream reference. Don't touch it from `.venv`. To run:

```bash
source sd-scripts-venv/bin/activate
TOKENIZERS_PARALLELISM=true accelerate launch --num_cpu_threads_per_process 1 \
  sd-scripts/anima_train_network.py --config_file=./melted1-anima-config.toml
```

Wipe `sd-scripts-outputs/` between runs (`rm -rf sd-scripts-outputs/*`) so stale checkpoints/samples don't confuse comparisons.

## Notes

- Resolutions are restricted to **512 or 1024**; bucket sizes are fixed.
- `train.gradient_checkpointing = true` is the default — saves VRAM for the cross-mlp preset with modest step-time cost.
- The first training step at each new bucket resolution pays a one-time ~700 ms inductor compile warmup (AdaLN fused kernel). Subsequent steps at the same bucket are at full speed.
- See `CLAUDE.md` for the deep architecture / optimization story and the historical rationale behind each scope decision.
