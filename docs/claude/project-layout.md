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
  - `precision.py` — autocast wrapper for `bf16`, `mxfp8`, and `fp8`; the quantized execution implementations live in `fp8_quant.py` and the merged adapted-linear route is selected through `lokr_patch.py`.
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

- **Profile the remaining copy/quantize traffic in the full FP8 step.** The current merged adapted-linear path is already the fastest tested route. Attribute copies to exact call sites before attempting another kernel; the old profile's blanket `aten::copy_` estimate predates production FP8.
- **Grouped self-QKV with one activation quantization.** A low-level TE grouped-GEMM probe was 5.339 → 5.085 ms for QKV forward+dgrad with exact output/dgrad parity, but projects to only ~0.3% of a 2.23 s step and did not clear a full-block acceptance run. Do not enable it without a stable full-block and full-step win.
- **Time-to-quality initialization experiment.** EVA/PiSSA/LoRA-GA/IPA may reduce steps-to-quality for ordinary LoRA, but ordinary two-branch LoRA was ~8% slower per block here. A comparable LoKr initializer needs a fixed-seed convergence gate; initialization claims are not throughput claims.
- **Convergence parity.** At 30 epochs the LoKr's mean latent-cosine is ~0.51 (style hasn't visibly appeared in samples for either sd-scripts or us). Per user, grokking is significant with Anima DiT — typically needs 30-40+ epochs. Levers: larger LoKr (`preset = "anima-full"`), longer training, better caption hygiene for multi-character setups.

Measured rejects as of 2026-08-31: per-block regional `torch.compile`, CUDA
graph replay, upstream wide self-QKV fusion, ordinary two-branch LoRA,
structured LoKr base/delta bypass, reusing saved FP8 representations, and a
Transformer Engine 2.18 upgrade for GEMM speed. See
`sm120-optimization-audit-2026-08.md` for numbers and scope.

Dead ends (don't re-investigate without new evidence): see `attention-sm120.md` and `quantization.md`.

GUI is explicitly **not a priority**.
