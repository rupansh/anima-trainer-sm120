# Quantization — history and the current TE-MXFP8 attempt

The original brief called for `bf16`, `mxfp8`, and `nvfp4` precision modes.
The first round (custom Triton/cublasLt MXFP8 + NVFP4) was removed for the
reasons below. The current attempt is **TransformerEngine MXFP8 with
frozen-base pre-quantization** — see `src/anima_trainer/fp8_quant.py` and
the "Current attempt" section at the end of this file.

## History — what we tried first and why it was removed

**MXFP8** got two implementations, both shipped briefly then removed:

1. *Tiny-LoRA era* (pre-git-lycoris-fix, 3 modules wrapped): a
   `TritonMXFP8Linear` that pre-quantized the frozen Linears and called
   `torch._scaled_mm` directly, matched compile's speed without warmup. Won
   ~1.3-1.65× microbench, 0.43 s/step end-to-end.
2. *Wide-LoRA era* (post-git-lycoris-fix, 454 modules wrapped): pre-quantize
   stops being viable because `merged_W = base_W + alpha * diff_W` is
   recomputed each step (diff_W trainable). Just-in-time quantization
   inside an `autograd.Function` gave only ~4% step-time improvement
   (3.65 → 3.5 s/step) because mm is no longer the dominant cost — the
   per-step time is now dominated by the 28-block transformer's internal
   ops we can't restructure without breaking the LoKr wrapping.

The full mxfp8 backward attempt hit cublasLt scale-layout issues
(`CUBLAS_STATUS_NOT_SUPPORTED`) that matched torchao's MXTensor dispatch
pattern would have needed extra debugging, and the theoretical ceiling was
~1.17× — not worth the maintenance burden for an algorithm we'd then have
to keep updated with torchao changes.

**NVFP4** was empirically tested with the tiny-LoRA setup and produced
**broken sample images** — pure noise speckles by epoch 2. The 4-bit
quantization compounds through 30 euler denoising steps; the sampler
never converges. CUTLASS path on sm_120 actually works for dense GEMM
(CUTLASS #3096 affected only MoE grouped-GEMM), and MSLK Triton is
available via `pip install mslk --index-url https://download.pytorch.org/whl/nightly/cu130`,
but neither rescues sample quality.

## Current attempt — TransformerEngine MXFP8 (frozen base)

The conditions for "quantization is worth revisiting" listed in the old
section are *now satisfied*:

1. The wrapped LoKr trainable surface is small (168 modules, ~20 M params).
   The bulk of every GEMM input is the *frozen* base weight — exactly the
   case where one-time quantization pays.
2. mm is now ~52 % of step time (was a smaller fraction back when the
   tiny-LoRA mxfp8 number was taken); fp8/mxfp8 has real headroom.
3. TransformerEngine on sm_120 is genuinely supported — see the corrected
   notes in `attention-sm120.md`. `Float8CurrentScaling` and
   `Float8BlockScaling` work out of the box; MXFP8 runs once the
   conservative `check_mxfp8_support` gate is patched out (open PR #2833
   demonstrates the kernels execute on sm_120).

Architecture in this attempt:

- Each `LokrModule.org_module.weight` is quantized **once at attach time**
  with TE's MXFP8 cast (E4M3 with E8M0 per-32-element block scales). The
  quantized tensor + scale buffer replace the bf16 weight as a frozen
  buffer.
- LoKr's trainable `W1`, `W2`, and the derived `diff_W` stay in bf16.
- Forward: instead of `merged_W = base_W + α·diff_W; F.linear(x, merged_W)`,
  we do two GEMMs and sum: `F.linear_mxfp8(x_q, base_q) + F.linear(x, diff_W)`.
  This keeps the LoKr math in bf16 (cheap, small) and the heavy base GEMM
  in MXFP8.
- Backward through the base GEMM produces only `grad_x` (base is frozen);
  through the LoKr GEMM produces `grad_x + grad_diff_W` as usual.
- Saved adapter is **always bf16** — quantization is forward-time only.

This is the "bypass mode" decomposition we previously rejected on VRAM
grounds, except the heavy half now runs in MXFP8 instead of bf16, which
recovers (and beats) the merged-weight bf16 path.

See `src/anima_trainer/fp8_quant.py` for the implementation, and
`tests/sanity_mxfp8.py` for the parity check.
