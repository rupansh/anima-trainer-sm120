# Attention on sm_120 (settled)

Empirically benchmarked at Anima shapes (B=8, H=16, D=128, bf16, fwd+bwd):

|       shape       | cuDNN  | Flash  | mem_eff | math    |
|-------------------|--------|--------|---------|---------|
| self_1024 (4096²) | 13.28  | 13.87  | 58.31   | 128.5   |
| self_896×1152     | 13.03  | 13.50  | 56.74   | 127.4   |
| cross_1024 (Q×512)| 2.71   | **2.53** | 7.58   | 18.1    |
| self_512  (1024²) | 1.01   | **0.87** | 3.71   | 8.80    |

Torch's default SDPA selector already picks cuDNN for self-attention and Flash
for cross-attention — i.e. the per-shape winner in both cases. Pinning
`{CUDNN_ATTENTION, FLASH_ATTENTION}` (`src/anima_trainer/attention_ctx.py`)
matches default wall-clock to within measurement noise (0.86–0.88 s/step),
but locks out a future torch version silently regressing to mem_eff/math.

Rejected backends on sm_120 (do not re-investigate without new evidence):

- **FA3 / FA4** — target sm_90a (H100) and sm_100 (Blackwell datacenter).
  sm_120 is the *workstation* Blackwell die; it lacks TMEM and the async
  tensor pipelines those kernels assume. Per Dao-AILab/flash-attention#1810,
  #2307: not supported and not on the roadmap.
- **FA2** (`flash-attn` PyPI package) — hardcoded arch whitelist excludes
  sm_120. Force-builds fall back to SM89 (Ada) kernels with no Blackwell-
  specific optimization, so they're strictly worse than cuDNN's Blackwell
  path. Don't waste time compiling it.
- **SageAttention** — inference-only. The production package has no backward
  pass; `SageBwd` is a research paper, not a usable library. Useless for us.

The real remaining wins on sm_120 are **fp8** (the GEMMs in q/k/v/output
projections + MLP, not the attention math itself; see the corrected
section below and `quantization.md` — SGLang's CUTLASS sm120 fp8 kernel
works, contrary to our earlier blanket dismissal) and possibly the
launch-bound elementwise fusions / CUDA graphs (see "Open follow-ups" in
`project-layout.md`). Attention itself is not the bottleneck in bf16.

Also rejected (May 2026):

- **xformers** on sm_120: `cutlassF-blackwell` / `cutlassB-blackwell` kernels
  show "unavailable", ops are gated to `capability <= (9, 0)`. Tracking issue:
  facebookresearch/xformers#1395. Not viable today.
- **cuDNN FP8 SDPA via `te.DotProductAttention`** (May 2026, *corrected
  twice*): the **actual** truth, after digging through the cuDNN error
  path rather than relying on TE's silent fallback to bf16:

  1. Calling `te.DotProductAttention` under `te.fp8_autocast(...)` without
     `fp8_dpa=True` on the recipe **silently runs bf16 attention**. The
     recipe defaults to `fp8_dpa=False`. Any "FP8 SDPA timing" measured
     this way is bf16. (Symptom: profile shows
     `cudnn_generated_fort_native_sdpa_sm120_flash_fprop` / bf16-style
     kernel names, not `fused_attn_fp8`-prefixed ones.)
  2. Setting `fp8_dpa=True` on `Float8CurrentScaling` / `DelayedScaling`
     trips a hard sm_120 gate in
     `transformer_engine/pytorch/attention/dot_product_attention/utils.py:608`
     ("Disabling Flash/FusedAttention as FP8 is not supported for compute
     capability = sm120") → backend selection returns `NoBackend` →
     immediate error.
  3. Patching past that gate exposes the real root cause: cuDNN 9.20
     returns `CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH` for `smVersion=1200`
     on the FP8 fused-attn kernel. cuDNN ships FP8 SDPA kernels for sm_90
     (H100) and sm_100 (datacenter Blackwell), **not for sm_120 workstation
     Blackwell**. The TE gate is enforcing a real cuDNN limitation, not
     being conservative.
  4. `Float8BlockScaling` and `NVFP4` are *also* unconditionally disabled
     for FusedAttention (`utils.py:604`) regardless of compute capability.

  Forward path if you want to revisit:
  - **cuDNN 9.22 also fails** (tested 2026-05-28 by upgrading from 9.20
    via `nvidia-cudnn-cu13==9.22.0.52`). Errors are *more explicit*:
      - `MXFP8BlockScaling(fp8_dpa=True)`: cuDNN raises
        `"MXFP8 SDPA is only supported on Blackwell Data Center
        architectures."` → sm_100 (B100/B200) only, **not** sm_120 (GB202
        workstation Blackwell / RTX 50).
      - `Float8CurrentScaling(fp8_dpa=True)`:
        `CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH` for `smVersion=1200`,
        engine ID 4.
    The TE 9.21 version gate at `utils.py:598` is the *minimum* — cuDNN
    builds the kernels selectively per arch and sm_120 isn't in the list.
    The 9.21+ ladder unlocks sm_100, not sm_120. Don't bump cuDNN to
    chase this; the kernel binary simply doesn't exist for our arch.
  - Flash-attn v2 (`flash-attn` PyPI) has **no FP8 path** — branch 2.x
    is bf16/fp16 only; FP8 attention came in FA3 (sm_90-only) and FA4
    (sm_90 + sm_100). None target sm_120.
  - Until NVIDIA ships sm_120 FP8 attention kernels in cuDNN (no public
    timeline as of 2026-05), **attention stays bf16**. Currently ~28%
    of step time at the production shape; this is the real ceiling for
    FP8 training on sm_120 right now. The remaining gap closes only via
    a hand-rolled Triton sm_120 FP8 attention kernel (deferred per
    `project-layout.md` follow-ups).
- **Custom Triton attention kernel**: deferred. cuDNN already runs self_1024
  bf16 fwd at ~368 TFLOPS, well above the on-paper 250 TFLOPS bf16 dense
  ceiling on this card — i.e. it's already exploiting mixed-precision tensor-
  core paths. A hand-rolled kernel might capture 5-15% but the same effort
  spent on fp8/fp4 GEMMs captures much more. Revisit only after fp8 is in.

**Important correction to earlier framing**: an earlier version of this
section concluded that "no off-the-shelf fp8 path works on sm_120." That was
wrong, and the correction matters because the conclusion drove the
decision to remove mxfp8/nvfp4 entirely. Specifically:

- **SGLang's CUTLASS sm120 fp8 blockwise GEMM works** (PR
  sgl-project/sglang#9969, merged 2025-09-07). Exposed as
  `torch.ops.sgl_kernel.fp8_scaled_mm.default()` in the `sgl-kernel` pip
  package. Inference-only, full fp8 (both inputs fp8 with per-block scales),
  not weight-only. Adapting to training requires `autograd.Function` glue
  plus a decision about full-fp8 vs weight-only forward. **Realistic
  projection at our current bottleneck distribution: 15-20% on top of
  2.55 s/step**, given `aten::mm` is ~50% of the step now (was a much
  smaller fraction back when we tried mxfp8). The earlier "only ~4%
  improvement" data point is from before the AdaLN/Liger/cross-mlp wins
  changed the bottleneck mix.

Quantization libraries / paths we did try and where they actually fail
(keep these notes so the same wall isn't hit twice):

- **TransformerEngine** — **correction**: an earlier note here claimed TE
  required system `cudnn-dev`/`nccl-dev` and didn't cover sm_120. Both are
  wrong as of v2.15 (May 2026). TE on sm_120 today:

  - Install: `pip install --no-build-isolation transformer-engine[pytorch]`
    against PyTorch nightly cu13x. Compiles from source; pulls
    `nvidia-cudnn-frontend` from pip, no system headers required.
  - **`check_fp8_support()`** → True on sm_120 (compute ≥ 9.0 gate).
    Both `DelayedScaling` and `Float8CurrentScaling` recipes run.
  - **`check_nvfp4_support()`** → True on sm_120.
  - **`check_fp8_block_scaling_support()`** → True on sm_120 with CUDA ≥
    12.9 (matches our cu13 nightly).
  - **`check_mxfp8_support()`** → currently **False** on sm_120 — the gate
    in `transformer_engine/pytorch/quantization.py:_compute_mxfp8_support`
    returns "MXFP8 (for all gemm layouts) is not supported on 12.0+
    architectures yet." This is conservative, not a hard kernel limit:
    open PR #2833 ("[Pyt][Common] Enabling/Guarding sm120 support
    (non-attention)", last touched 2026-05-23) is enabling/guarding the
    individual MXFP8 paths for sm_120, and its description confirms the
    MXFP8 cast kernels execute on sm_120 today (the PR includes an
    arch-agnostic CAST_DBIAS race fix found while testing on sm_120).
    Patching the gate locally or running PR #2833's branch lets MXFP8
    forward + backward run.
  - sm_120 has been actively supported in TE since at least Oct 2025
    (PR #2279 reorganized arch-specific compilation; #2320 fixed
    attention on sm_120; #2482 fixed sm_120 build with CUDA 12; #2693
    enabled cuDNN ≥ 9.18.1 fused attention on sm_120).
- **torchao stable paths** (`Float8DynamicActivationFloat8WeightConfig`,
  `Int4WeightOnlyConfig`) — fail with `RuntimeError: cutlass cannot
  initialize` on sm_120. The CUTLASS tinygemm dispatch they use was built
  for sm_100 datacenter Blackwell; the workstation die isn't in the table.
  Includes the production `PerRow` recipe torchtitan uses for the
  2K-cluster 1.5× pre-training speedup — *that recipe doesn't run on our
  card*. This is a torchao-dispatch issue, not a CUTLASS-the-library issue
  (see SGLang above).
- **NVIDIA Model-Optimizer** — inference-focused (PTQ + QAT for
  TensorRT-LLM); no frozen-base + bf16-adapter LoRA workflow comparable to
  torchao's. Useful for deployment quantization, not training.
- **microsoft/microxcaling** — pure PyTorch emulation library (computes in
  fp32 then quantizes the result). Useful for studying MX numerics; useless
  for performance.
- **bf16 + MSLK** — all MSLK `bf16bf16bf16_*` ops are grouped (MoE) variants;
  no dense `bf16 × bf16` GEMM. `bf16x9_gemm` is misnamed (expects float32
  inputs). The `bf16i4bf16_rowwise` op (BF16 acts × INT4 weights) is the
  only candidate for weight-only quantization, but torchao's Int4 config
  doesn't dispatch to it — it goes to the same broken CUTLASS path.

**Corrected bottom line:** the off-the-shelf fp8/mxfp8 paths that
*dispatch correctly on sm_120* are:

1. **TransformerEngine** (`transformer_engine.pytorch`) — official
   support for `DelayedScaling`/`Float8CurrentScaling` on sm_120;
   MXFP8 runs once the conservative `check_mxfp8_support` gate is
   bypassed (PR #2833 in flight). Autograd-aware, so it's a real
   training path (not inference-only).
2. **`sgl-kernel`'s CUTLASS blockwise fp8 GEMM**
   (`torch.ops.sgl_kernel.fp8_scaled_mm`) — inference-only as shipped;
   would need autograd glue. Less attractive now that TE covers
   sm_120.
3. **`torchao.prototype.mx_formats`** with `KernelPreference.AUTO` and
   `torch.compile` — also dispatches, but slower and less mature than
   TE for our use case.

For this trainer, **TransformerEngine MXFP8 is the path** — frozen
base weights pre-quantized once at attach time, LoKr delta stays in
bf16, saved adapter stays in bf16. See `src/anima_trainer/fp8_quant.py`
when it lands.
