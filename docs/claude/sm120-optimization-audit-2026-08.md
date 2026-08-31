# SM120 optimization audit — 2026-08-31

## Bottom line

The current merged-weight, block-scaled-FP8 LoKr implementation remains the
fastest tested production path. No code from the audited upstream techniques
or adapter alternatives passed the real-block speed gate, so none was enabled.
Gradient checkpointing was explicitly out of scope and was not retested.

Host stack for these measurements: RTX PRO 6000 Blackwell Workstation
(compute capability 12.0, 96 GB), PyTorch `2.13.0.dev20260521+cu130`, CUDA
13.0 runtime, cuDNN 9.20, Triton 3.7, and Transformer Engine 2.15. SM120 is
not SM100 and has no usable `tcgen05`/TMEM programming path; SM100 kernels
are not candidates for this project.

Production reference: batch 8, 1024-class latent shape, `anima-cross-mlp`,
168 adapted modules / 20,195,840 trainable parameters, Prodigy+Schedule-Free.
Measured full training is about **2.23 s/step in FP8** and **2.52 s/step in
bf16**.

## Same-shape execution results

Unless marked as a microbenchmark, block results use an actual Anima block,
production tensor shapes, backward, Liger RMSNorm, custom AdaLN/residual
patches, SDPA, and the current FP8 LoKr implementation. The local training
dataset was unavailable, so candidate gates used synthetic inputs at those
exact shapes; the 2.23/2.52 s full-step figures above are the existing
production A/B rather than a new dataset run.

| Candidate | Reference | Candidate | Result |
|---|---:|---:|---|
| Regional `torch.compile` | eager 58.595 ms | 58.777 ms | flat/slower; reject |
| CUDA graph replay | eager 58.923 ms | 58.943 ms | flat; keep disabled |
| Ordinary LoRA rank 16 | LoKr 58.923 ms | 63.488 ms | 7.7% slower |
| Ordinary LoRA rank 32 | LoKr 58.923 ms | 63.639 ms | 8.0% slower |
| Merged-FP8 LoRA rank 16 | LoKr 58.923 ms | 58.317 ms | ~1% noise with less capacity |
| Structured base + LoKr bypass | merged LoKr 58.923 ms | 78.883 ms | 34% slower |
| TE 2.18 dominant merged op | TE 2.15 8.716 ms | 8.721 ms | flat; no upgrade for speed |
| Wide fused self-QKV microbench | three linears 3.737 ms | 4.289 ms | 15% slower |
| Alternative LoKr `bmm` mixing microbench | flat 7.924 ms | 9.060 ms | slower |
| Alternative LoKr `einsum` mixing microbench | flat 7.924 ms | 8.276 ms | slower |

The structured bypass is the decisive LoKr result: although it avoids
re-quantizing a full merged weight, the extra dense base GEMM plus structured
adapter work loses badly to one dense FP8 tensor-core GEMM. Saving FP8
representations for backward was also neutral to slower (`save_for_backward`
was 5.5% slower; a context-attribute variant was within 0.8% noise).

The frozen six-layer LLM adapter is only about 7.26 ms, roughly 0.33% of a
2.23 s step, so caching/reworking it cannot materially change training time.

## Upstream `anima_lora` review

The reviewed upstream snapshot was commit `d188b975` from 2026-08-31. Its
headline rank-32 ordinary-LoRA result is not comparable to this trainer's
20.2M-parameter LoKr configuration. Relevant techniques were evaluated as
follows:

- Regional compile: tested locally and flat because TE low-level FP8 calls
  split the graph.
- Wide self-QKV and cross-KV fusion: the dominant self-QKV shape was 15%
  slower locally; preserving three GEMM shapes gives better algorithms.
- SVD-down: an initialization/time-to-quality technique, not a per-step
  throughput optimization. It remains interesting only behind a convergence
  experiment.
- T-LoRA: masks the completed low-rank projection, so it does not remove its
  GEMMs and is not a raw throughput win.
- Channel scaling: changes optimization/capacity and lacks a sufficient
  quality-parity case for this project; upstream's turbo comparison was not a
  speed-and-quality win.
- REPA: adds a representation-alignment loss and encoder work. It is a DiT
  pretraining convergence technique, not a free adapter-training speedup.
- FA4: upstream itself removed the route; its consumer-SM120 fork was slower
  than FA2. Attention remains bf16 SDPA here.

Source: [upstream README](https://github.com/sorryhyun/anima_lora/blob/main/README.md),
[upstream FA4 notes](https://github.com/sorryhyun/anima_lora/blob/main/docs/optimizations/fa4.md).

## Research and library audit

Recent adapter initialization work—[EVA](https://arxiv.org/abs/2410.07170),
[PiSSA](https://arxiv.org/abs/2404.02948),
[LoRA-GA](https://arxiv.org/abs/2407.05000), and
[IPA](https://openreview.net/forum?id=aLmQeZx2pR)—targets convergence or rank
allocation. Those methods may reduce steps-to-quality, but they do not make
the adapter's steady-state GEMMs cheaper. Because the locally tested ordinary
LoRA forward/backward is about 8% slower per block than merged FP8 LoKr, any
such arm must beat LoKr on fixed-seed wall-clock-to-quality, not paper step
counts. A LoKr analogue of these initializers would be new experimental work,
not an assumed transfer.

[T-LoRA](https://arxiv.org/abs/2507.05964) and
[REPA](https://arxiv.org/abs/2410.06940) were also reviewed. Their mechanisms
do not remove the dominant dense training GEMMs in this implementation.

Transformer Engine 2.18 adds grouped/block-scaling functionality and
additional quantization caching, but a separately built 2.18 wheel was flat
against 2.15 on the dominant merged LoKr operation. Do not upgrade solely for
performance without a full-step A/B. See the
[Transformer Engine release notes](https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/).

CUTLASS releases now expose more SM120 scaled-FP8 grouped/pointer-array
building blocks, but headline SM100/Rubin `tcgen05`/TMEM paths are irrelevant
to this GPU. Architecture eligibility must be checked before porting a kernel.
See the [CUTLASS changelog](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md?plain=1).

cuDNN's FP8 attention backward path is documented with a causal-mask
restriction. Anima training uses non-causal attention, so it is not a valid
drop-in training backend. See
[cuDNN attention limitations](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html).

## Remaining opportunities

1. Re-profile one full production FP8 step and attribute copy/quantization
   traffic by call site. Old bf16 `aten::copy_` totals are not actionable.
2. Revisit low-level grouped self-QKV only if a short, reproducible full-block
   gate confirms it. Quantizing the shared activation once improved the
   isolated QKV forward+dgrad slice from 5.339 to 5.085 ms with exact parity,
   but the projected whole-step win is only about 0.3%.
3. If minimizing wall-clock-to-quality becomes more important than raw step
   speed, benchmark one initializer arm against merged FP8 LoKr with identical
   data order, seed, parameter budget, and quality checkpoints.

Do not spend another optimization cycle on regional compile, CUDA graphs,
wide QKV fusion, two-branch ordinary LoRA, structured LoKr bypass, TE 2.18,
or gradient checkpointing without new upstream evidence that changes the
measured execution model.
