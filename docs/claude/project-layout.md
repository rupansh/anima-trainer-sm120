# Project layout (current)

Built:

- `src/anima_trainer/` — the trainer package, installed editable.
  - `config.py` — TOML loader; the user-facing knobs and their defaults live here.
  - `buckets.py` — bucket tables for 1024 and 512 derived from `crop.py:12`.
  - `hashing.py` — xxhash3 fingerprints + mtime/size revalidation.
  - `cache.py` — LanceDB cache (crops / latents / text_embeds), with VAE- and TE-weights fingerprint gating on latents and text embeddings.
  - `sdscripts_bridge.py` — adds vendored `sd-scripts/` to sys.path on demand.
  - `model.py` — loads DiT + Qwen3 + VAE via sd-scripts; returns a `LoadedModels`.
  - `lokr.py` — wraps the DiT with `lycoris.kohya`. Registers the `anima-full` and `anima-cross-mlp` presets; flips `LycorisNetworkKohya.USE_FNMATCH = True` for fnmatch globs.
  - `lokr_patch.py` — monkey-patches `LokrModule.forward` to merge `base_W + alpha*diff_W` into a single matmul per LokrModule.
  - `liger_patch.py` — monkey-patches Anima's `RMSNorm.forward` to use `LigerRMSNormFunction` (fp32 stats, bf16 weight multiply, single fused Triton kernel).
  - `adaln_patch.py` — monkey-patches `Block._forward` to call the custom Triton kernel from `adaln_kernel.py` (LayerNorm + (1+scale)*x + shift) instead of the inline lambda. Replaces the older `torch.compile` path.
  - `adaln_kernel.py` — custom Triton kernels (forward + dx + dscale/dshift) and `FusedAdaLN(torch.autograd.Function)` wrapper for the AdaLN fusion.
  - `optim.py` — Prodigy+SF with locked defaults; only `d0` exposed.
  - `precision.py` — autocast wrapper; only `bf16` is implemented. `quantize_dit_in_place` is a no-op kept for call-site compatibility (mxfp8/nvfp4 removed).
  - `flow.py` — sigmoid timestep sampling + rectified-flow target.
  - `preprocess.py` — smartcrop+bucket; writes JPEGs into `crops` table.
  - `encode.py` — VAE / Qwen3 encoding with cache fronting.
  - `dataset.py` — `CachedAnimaDataset`, bucket-aware sampler, collate.
  - `sample.py` — euler discrete + hot-reloading prompt parser/file.
  - `attention_ctx.py` — SDPA backend pinner (`{CUDNN_ATTENTION, FLASH_ATTENTION}`); see `attention-sm120.md`.
  - `precompute.py` — one-shot fill of the LanceDB cache.
  - `train.py` — the training loop (bf16, flow matching, Prodigy+SF, sampling hook). Wires `install_liger_patch()` and `install_adaln_patch()` after `ensure_on_path()`.
  - `cli.py` — `anima-train precompute|train|eval-compare|convergence` entrypoint.
- `eval_compare.py` — pairwise sample comparison (`anima-train eval-compare A B`).
  Reports pixel MSE/PSNR, simple SSIM, LPIPS (alex), and optional VAE-latent
  cosine. Handles both our sample filename style and sd-scripts'. Use with
  `--vae` flag for the cosine-in-Anima-latent metric (slower).
- `convergence.py` — single-checkpoint convergence metric
  (`anima-train convergence <config> <lora.safetensors>`). Picks N random
  training captions, samples via the trained LoRA at the matching bucket
  resolution, computes cosine between predicted latent and cached training
  latent. Absolute number is informative but not interpretable; the trend
  across epochs is the signal.
- `melted1.toml` — production trainer config (cross-mlp preset, d0=5e-5).
- `tests/smoke_load.py` — model load + LoKr attach smoke check.
- `tests/sanity_cross_mlp_liger.py` — Liger RMSNorm numerical parity + cross-mlp preset module-count check.
- `tests/sanity_adaln_patch.py` — single-process A/B for the AdaLN patch (loss/step-time before vs after install). NOTE: the loss-diff number this prints is confounded by training dynamics across the warmup-then-measure structure; rely on the real `train.py` log for parity (steps 1-4 are bit-identical on cached data).
- `tests/sanity_adaln_kernel.py` — pure numerical parity for `FusedAdaLN.apply` (forward + dx + dscale + dshift) vs PyTorch eager and fp32 ground truth, in both bf16 and fp32.
- `tests/bench_adaln_kernel.py` — microbench of the AdaLN modulation at the production shape; compares eager / torch.compile / custom kernel for fwd+bwd.
- `tests/profile_step.py` — torch.profiler kernel-level breakdown (bucketed by op type).
- `tests/ncu_bench.py` — minimal one-step bench for nsight-compute (NVTX-bounded). Requires `NVreg_RestrictProfilingToAdminUsers=0` or sudo to actually collect counters.

Open follow-ups (in rough leverage order):

- **`aten::copy_` ~184 ms/step lead** — the profile shows ~7% of step time in dtype/layout copies. Best guess: fp32→bf16 cast on the LokrModule diff weight + autograd saved-tensor materialization for backward. Cheapest probe: `stochastic_rounding=True` already on in Prodigy+SF; check if optimizer state can stay bf16 via that path. Estimate: ~5% if the cast turns out to be eliminable.
- **fp8 GEMMs via SGLang's CUTLASS sm120 kernel** (`sgl-kernel`, `torch.ops.sgl_kernel.fp8_scaled_mm.default`, PR #9969 merged 2025-09-07). Inference-only blockwise fp8 (both inputs fp8). For training: wrap in autograd.Function + decide full-fp8 vs weight-only. Realistic projection: 15-20% on top of current 2.55 s (~50% of step is `aten::mm`; ~1.5× speedup on that portion is typical real-world fp8). 1-2 days of engineering + a numerical-parity convergence gate. **Not the dead end the earlier "Quantization libraries surveyed" section implied — see the SGLang note in `attention-sm120.md` for the corrected framing.**
- **CUDA graphs per bucket** — ~3000 kernel launches/step, many in small launch-bound elementwise kernels at ~20% HBM-BW utilization. Capture once per bucket → ~5-10% on steady-state. Requires static input tensor addresses and pre-allocated bucket buffers. High complexity.
- **Block-level fusion** — AdaLN is now a custom Triton kernel; the next-wider fusion target is `Block._forward` itself. Options: (a) hand-written Triton over the whole block (high effort), or (b) `torch.compile(Block._forward)` which gives one compiled graph per block — should sidestep the 77 GB / 124 s pathology since the scope is tiny. ~5-15% unverified.
- **`torch.backends.cudnn.benchmark = True`** — one line; lets cuDNN autotune kernel variants per shape, fixed buckets make this safe. Possibly 0-3%.
- **Convergence parity** — at 30 epochs the LoKr's mean latent-cosine is ~0.51 (style hasn't visibly appeared in samples for either sd-scripts or us). Per user, grokking is significant with Anima DiT — typically needs 30-40+ epochs. Levers: larger LoKr (`preset = "anima-full"`), longer training, better caption hygiene for multi-character setups.

Dead ends (don't re-investigate without new evidence): see `attention-sm120.md` and `quantization.md`.

GUI is explicitly **not a priority**.
