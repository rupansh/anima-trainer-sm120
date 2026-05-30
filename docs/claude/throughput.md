# Throughput characterization

Current production at batch_size=8 / 1024² / bf16 (cross-mlp + lokr_patch +
Liger + AdaLN-compile, 168 wrapped modules / 20.2 M params):

- **2.55 s/step steady-state, ~20 s/epoch, ~16 GB peak VRAM**.
- First step of a bucket pays ~700 ms compile warmup (AdaLN inductor
  compile); cached on subsequent steps at that bucket.

Per-step CUDA self-time decomposition from `tests/profile_step.py`:

| component                          | per-step  | % of step |
|------------------------------------|-----------|-----------|
| `aten::mm` (LokrModule + base mms) | ~1.26 s   | ~44%      |
| Flash attention (fwd + bwd)        | ~0.57 s   | ~20%      |
| Elementwise (mul + add + copy)     | ~0.52 s   | ~18%      |
| LayerNorm + Liger RMSNorm          | ~0.25 s   | ~9%       |
| Optimizer + grad clip + misc       | ~0.25 s   | ~9%       |

Big GEMMs hit **~478 TFLOPS** measured (95% of the ~503 TFLOPS bf16 dense
spec on RTX PRO 6000 Blackwell). The remaining "fat" relative to a
silicon-floor estimate (~1.5-1.8 s/step) lives in launch-bound elementwise
(~150 ms reachable via fusion / CUDA graphs) and dtype copies (~184 ms in
`aten::copy_`, partly recoverable). Roofline-wise we are within ~50% of
hardware floor for the LoKr + bf16 + sm120 path.

Wider/deeper trade-offs:

- **Doubling batch_size to 16: 2× step time, same samples/sec.** Compute-
  bound on bf16 tensor cores. Memory tricks (CPU offload, lower-precision
  storage) cost speed without buying anything we need.
- **Gradient checkpointing**: currently `true` in `melted1.toml`. With
  cross-mlp (168 trainable LoKrModules touching activations across 28
  blocks) it does meaningfully reduce VRAM. (Old note about it being a
  no-op was from the broken-pypi 3-module LoRA era.) Cost in step time is
  small enough we keep it on by default.
- **I/O is not visible per step.** Dataloader runs in 4 worker processes;
  12 MB host→device transfers with `non_blocking=True` overlap with kernel
  execution; LanceDB reads are off the critical path. The 184 ms `aten::copy_`
  observed in profiling is on-GPU dtype churn, not I/O.
