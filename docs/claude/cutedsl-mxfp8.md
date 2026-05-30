# CuTeDSL MXFP8 GEMM for sm_120 — handoff doc

This doc consolidates everything learned across multiple sessions about
implementing a CuTeDSL MXFP8 GEMM kernel on sm_120 (RTX PRO 6000 Blackwell
workstation). It's structured so a fresh conversation can pick up without
re-deriving any of it.

## Why this kernel exists

We hit a wall doing MXFP8 training on sm_120 via TransformerEngine:

- TE's `Float8CurrentScaling` and `Float8BlockScaling` work fwd+bwd on
  sm_120. ✅
- TE's `MXFP8BlockScaling` forward works on sm_120 via the low-level
  `tex.general_gemm`. ✅
- TE's MXFP8 **backward** dgrad GEMM fails with `CUBLAS_STATUS_NOT_SUPPORTED`
  on every shape we tried, with cuBLAS up to **13.5.1.27** (the latest on
  pypi.nvidia.com). ❌ — this is exactly what
  `check_mxfp8_support()` is gating against on compute ≥ 12.0.
- We have a working Triton `tl.dot_scaled` MXFP8 kernel in
  `src/anima_trainer/mxfp8_gemm.py`. PTX confirms native
  `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0`
  fires (zero bf16 emulation), but the surrounding kernel infrastructure
  (TMA, persistent scheduling, multi-stage pipeline, multi-CTA L2 reuse)
  is rudimentary compared to cuBLAS, so per-shape benches show
  **0.95–1.33×** vs bf16 cuBLAS depending on shape.

The CuTeDSL kernel would get us CUTLASS-quality infrastructure (TMA,
persistent, pipeline, swizzled L2) on top of the same hardware MMA the
Triton kernel already fires correctly. Realistic expected gain: **12–18%
step time** at production shapes (mm is 52 % of step, FP8 has ~2 ×
theoretical tensor-core throughput vs bf16, discount to ~1.5 × after
quantize overhead and BW constraints).

## Hardware + software context (verified)

| | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Compute capability | (12, 0) — **sm_120 workstation**, not sm_100 datacenter |
| Has TMEM? | **No** — sm_120 has no tensor memory; use warp-level MMA, not `tcgen05` |
| Has TMA multicast? | **No** — cluster shape must be `(1, 1, 1)` |
| CUDA toolkit | 13.4.0.1 system, 13.1 in venv (pip) |
| PyTorch | nightly cu130 (Python 3.14, cp314 wheels) |
| `nvidia-cublas` pin | **`>=13.4.1`** required — TE wheel was built against 13.4 and uses `cublasLtGroupedMatrixLayoutInit_internal` which doesn't exist in 13.1. PyTorch nightly default is 13.1, so we pin in pyproject.toml. |
| `nvidia-cutlass-dsl[cu13]` pin | **`>=4.5.2`** — 4.4.1 only has cp310-cp313 wheels (no cp314). 4.5.2 ships cp314. |
| TE | `transformer-engine[pytorch]>=2.15` (prebuilt wheel works once cublas is upgraded) |
| Env at runtime | `NVTE_CUDA_INCLUDE_DIR=/opt/cuda/include` (set in `fp8_quant.py`); `CUTE_DSL_ARCH=sm_120a` (must be set BEFORE `import cutlass`) |

## The CuTeDSL primitive that makes this possible

`cutlass.cute.nvgpu.warp.mma.MmaMXF8Op` is the warp-level MXFP8 MMA Op.
Admissible archs: `[sm_120a, sm_120f, sm_121a, sm_121f]`. The `f`
variants raise an "sm_120f currently not supported" error — **set
`CUTE_DSL_ARCH=sm_120a`**.

```python
from cutlass.cute.nvgpu.warp.mma import MmaMXF8Op
import cutlass

# This works once CUTE_DSL_ARCH is set to sm_120a in env.
op = MmaMXF8Op(
    cutlass.Float8E4M3FN,   # ab_dtype  (both A and B are FP8 E4M3)
    cutlass.Float32,        # acc_dtype (always fp32 for these MMAs)
    cutlass.Float8E8M0FNU,  # sf_type   (E8M0 = 8-bit unsigned biased exponent)
)
# Internal defaults the Op imposes:
#   shape_mnk    = (16, 8, 32)   ← hardware MMA tile shape for FP8 on sm_120
#   sf_vec_size  = 32            ← one E8M0 scale per 32 K-elements (MXFP8 spec)
```

PTX that this dispatches to (verified):

```
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
```

This is the *native* sm_120 MXFP8 hardware instruction — NOT cuBLAS, NOT
bf16 emulation. The kernel bypasses cuBLAS entirely.

### MMA trait & scale pointer setting

The Op's `_make_trait()` returns a `MmaMXF8Trait` (subclass of
`MmaBlockScaledTrait`). The trait has `Field.SFA` and `Field.SFB`
admissible fields. **In the kernel mainloop, set these fields with
pointers to the SFA/SFB tiles before each `cute.gemm` call** — that's
how the warp MMA learns where to load the E8M0 scales from:

```python
# Pseudo-code, exact partition/iterator setup varies
trait.set(cute.MmaAtomField.SFA, sfa_ptr)
trait.set(cute.MmaAtomField.SFB, sfb_ptr)
cute.gemm(tiled_mma, acc, tCrA[k_block], tCrB[k_block], acc)
```

### Same-dtype vs mixed-dtype

| op | A dtype | B dtype | use case |
|---|---|---|---|
| **`MmaMXF8Op`** | FP8 E4M3 | FP8 E4M3 (same) | **What we want for the trainer** |
| `MmaMXF8F6F4Op` | FP4/FP8 mixed | FP8/FP4 mixed | not us |
| `MmaMXF4Op` / `MmaMXF4NVF4Op` | FP4 | FP4 | NVFP4 path, not us |

The same-width mixed-FP8 (E4M3 × E5M2) is explicitly **not supported** —
use `MmaMXF8Op` for same-dtype.

## Reference kernels to crib from

Four reference files are downloaded to `/tmp/cutlass_refs/` (regenerate
with `gh api ...` if a fresh session needs them; URLs in the comments
below):

| file | source | what it gives us |
|---|---|---|
| `79c.cu` (549 lines) | [CUTLASS examples/79_blackwell_geforce_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm) | sm_120 MXFP8 GEMM in C++ — confirms `OpClassBlockScaledTensorOp`, `cutlass::arch::Sm120`, `ThreadBlockShape = Shape<_128,_128,_128>`, `ClusterShape = Shape<_1,_1,_1>`. Uses `cutlass::mx_float8_t<float_e4m3_t>` types. The setup section also shows SFA/SFB layout via `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`. |
| `80a.cu` (557 lines) | [examples/80_blackwell_geforce_sparse_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/80_blackwell_geforce_sparse_gemm) | sm_120 MXFP8 sparse — illustrative for layout details. |
| `dense_gemm_sm120.py` (1325 lines) | [examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm) | The **scaffold** to fork. sm_120 FP8 GEMM in CuTeDSL with full warp-specialized persistent kernel, TMA, multi-stage pipeline. **Does not have block-scaled MMA** — we add SFA/SFB plumbing. The MMA op used here is `MmaF16BF16Op`/`MmaFP8Op` — we swap for `MmaMXF8Op`. Has the sm_120-specific bits (no TMEM, no multicast, cluster=1,1,1) wired correctly. |
| `nvfp4_gemm_0.py` (778 lines) | [examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm) | sm_100 NVFP4 block-scaled — has the **SFA/SFB plumbing pattern** to copy. `blockscaled_utils.tile_atom_to_shape_SF`, `make_smem_layout_sfa`, `make_smem_layout_sfb`, plus the TMA atoms for SFA/SFB. Uses `tcgen05.MmaMXF4NVF4Op` (sm_100 TMEM) which we replace with `warp.MmaMXF8Op` (sm_120). |

**The synthesis**: take `dense_gemm_sm120.py` as the structural scaffold,
graft in the SFA/SFB handling from `nvfp4_gemm_0.py`, swap the MMA Op to
`MmaMXF8Op` from `cute.nvgpu.warp.mma`, and adjust dtypes/dims for MXFP8
(FP8 E4M3 instead of FP4 E2M1; sf_vec_size=32 instead of 16; tile (16,8,32)
instead of (16,8,64)).

## CuTeDSL utilities for block-scaled GEMM

```python
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import warp, cpasync
```

Key helpers used by `nvfp4_gemm_0.py`:

- `blockscaled_utils.tile_atom_to_shape_SF(shape, sf_vec_size)` — produces
  the SFA/SFB tensor layout. The shape is `((Atom_M, Rest_M), (Atom_K, Rest_K), RestL)`
  for SFA and `((Atom_N, Rest_N), (Atom_K, Rest_K), RestL)` for SFB.
- `blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec_size, num_stage)` — staged SMEM layout for SFA.
- `blockscaled_utils.make_smem_layout_sfb(...)` — same for SFB.
- `cute.nvgpu.make_tiled_tma_atom_A(...)` / `..._B(...)` — TMA copy atoms.
  For SFA/SFB tensors, pass `internal_type=cutlass.Int16` (per
  `nvfp4_gemm_0.py` line 235).

Note: the sm_100 example uses `sm100_utils.make_smem_layout_a` — for
sm_120 we use the helpers from `sm90_utils` (Hopper-style — sm_120 lacks
TMEM but has TMA, like Hopper). `dense_gemm_sm120.py` already imports as
`sm90_utils` for the same reason.

## Quantize layout — what scales look like on disk

Two key facts about the scale layout that CuTeDSL/CUTLASS expects:

1. **E8M0 scale encoding**: `byte = clamp(exp + 127, 0, 255)` where
   `exp = ceil(log2(amax_per_block / fp8_e4m3_max))` and `fp8_e4m3_max = 448`.
   Recovery: `scale = 2^(byte - 127)`. **`ceil`, not `round`**, otherwise
   amax can overflow FP8's representable range on edge cases.

2. **Layout: TE's MXFP8Quantizer output is NOT directly usable**.
   `MXFP8Quantizer.quantize(x)` returns an `MXFP8Tensor` whose
   `_rowwise_scale_inv` is a PADDED, SWIZZLED layout: minimum
   `(128, 4)` per tile even when the tensor is `(M, K) = (64, 128)`. The
   padding is for TMA-aligned access in the sm_100 datacenter MMA path.
   `tl.dot_scaled` AND CuTeDSL's `MmaMXF8Op` want different (raw / atom-
   tiled) layouts. **Write your own quantizer** that produces the layout
   the consumer wants — our `src/anima_trainer/mxfp8_gemm.py` already does
   this for Triton; for CuTeDSL, produce the layout
   `blockscaled_utils.tile_atom_to_shape_SF` describes.

## sm_120-specific constraints (read off the CUTLASS 79c file)

```cpp
// From 79c.cu:
using ArchTag             = cutlass::arch::Sm120;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape    = Shape<_128,_128,_128>;
using ClusterShape        = Shape<_1,_1,_1>;
```

Quoted from the file:
> Note that GeForce RTX 50 series GPUs do not support:
> 1. Multicast feature of TMA load. Cluster shape has to be 1x1x1.
> 2. Dynamic datatypes.

Mapped to CuTeDSL conventions:
- `cluster_shape_mnk = (1, 1, 1)` (no multicast)
- `tile_shape_mnk = (128, 128, 128)` as a reasonable default (autotune may favor different)
- `mma_inst_mnk = (16, 8, 32)` (hardware MMA tile)
- `atom_layout = (2, 2, 1)` (per `dense_gemm_sm120.py`, 2×2 atom tile)

## What the kernel does (high-level pseudocode)

```python
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import warp, cpasync

AB_DTYPE = cutlass.Float8E4M3FN
SF_DTYPE = cutlass.Float8E8M0FNU
ACC_DTYPE = cutlass.Float32
C_DTYPE = cutlass.BFloat16
SF_VEC_SIZE = 32

class Sm120MXFP8GemmKernel:
    def __init__(self, tile_shape_mnk=(128, 128, 128)):
        self.cluster_shape_mnk = (1, 1, 1)  # forced — no multicast on sm_120
        self.tile_shape_mnk = tile_shape_mnk
        self.atom_layout = (2, 2, 1)
        self.num_mma_warps = 4
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp  # +1 DMA warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.ab_stage = 4  # multi-stage pipeline depth
        # ... (mirror dense_gemm_sm120.py constructor)

    def _setup_attributes(self):
        # Build the MMA Op
        op = warp.MmaMXF8Op(AB_DTYPE, ACC_DTYPE, SF_DTYPE)
        # (16, 8, 32) MMA × atom_layout = effective per-warp tile
        permutation_mnk = (
            self.atom_layout[0] * 16,
            self.atom_layout[1] * 8 * 2,   # ldmatrix.x4 expansion (per dense_gemm comment)
            self.atom_layout[2] * 32,
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout),
            permutation_mnk=permutation_mnk,
        )

        # SFA/SFB SMEM layouts via blockscaled_utils
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.tile_shape_mnk, SF_VEC_SIZE, self.ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.tile_shape_mnk, SF_VEC_SIZE, self.ab_stage,
        )
        # A/B SMEM layouts via sm90_utils (Hopper-style, fits sm_120 since
        # both have TMA + no TMEM)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(...)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(...)
        # ... (rest mirrors dense_gemm_sm120.py)

    @cute.jit
    def __call__(self, a, b, sfa, sfb, c, stream):
        # 1. Set up tensors + TMA atoms for A, B, SFA, SFB, C
        # 2. Compute grid
        # 3. Launch self.kernel
        ...

    @cute.kernel
    def kernel(self, tma_atom_a, tma_tensor_a,
                     tma_atom_b, tma_tensor_b,
                     tma_atom_sfa, tma_tensor_sfa,
                     tma_atom_sfb, tma_tensor_sfb,
                     tma_atom_c, tma_tensor_c,
                     ...):
        # Body mirrors dense_gemm_sm120.py kernel(), with these additions:
        # - Allocate sSFA, sSFB in shared memory
        # - TMA load SFA, SFB tiles alongside A, B
        # - In mainloop, before each cute.gemm:
        #     thr_mma.partition_A(sSFA[stage]) → tCsSFA
        #     thr_mma.partition_B(sSFB[stage]) → tCsSFB
        #     mma_trait.set(Field.SFA, tCsSFA_ptr)
        #     mma_trait.set(Field.SFB, tCsSFB_ptr)
        #     cute.gemm(tiled_mma, acc, tCrA[k], tCrB[k], acc)
        ...
```

## Verification path

Once the kernel compiles + runs:

1. **Numerical parity**: compare output against
   `src/anima_trainer/mxfp8_gemm.py:MXFP8FrozenLinearTriton` for the same
   input. Expect max rel err ~5 % (MXFP8 noise floor) — match exactly to
   Triton's output, not to bf16.

2. **PTX verification**: dump the compiled kernel's PTX and confirm
   `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X` lines
   are present (no `bf16` mma fallback).

3. **Bench**: per-shape vs `torch.nn.functional.linear` bf16 reference at
   our production shapes (`B*T*H*W = 32768`, D ∈ {1024, 2048, 8192}). Expect
   wins of 1.3–1.7× per GEMM (closing the gap that Triton has vs cuBLAS).

4. **End-to-end**: wire into `src/anima_trainer/fp8_quant.py` replacing
   the `MXFP8FrozenLinear` Triton op. Run a few training steps; loss
   should track within 1 % of bf16 step-for-step (we verified MXFP8 quality
   in `outputs/melted1-mxfp8/` — see `bf16-cmp.log` / `mxfp8-cmp.log`).

## Estimated end-to-end gain

At our production config (B=8, 1024², bf16 baseline 2.49 s/step):
- mm bucket is 52 % of step (1.25 s out of 2.49 s)
- MXFP8 has ~2 × theoretical tensor-core throughput vs bf16
- Realistic after quantize overhead + BW: ~1.5 × per GEMM
- Forward GEMMs (174 unwrapped + 168 LoKr-wrapped frozen base) and
  backward dgrad both benefit
- Expected: 0.5 × × 1.25 s = ~0.4 s saved → step time 2.49 → 2.1 s →
  **~15 % win**

## What's already implemented in the repo (NOT throwaway)

These should be reused, not re-built:

- **`src/anima_trainer/mxfp8_gemm.py`** — Triton kernel, autograd op, and
  E4M3+E8M0 quantizer kernel. Quantizer is shape-agnostic; the GEMM is
  the part that needs replacing. The quantize+kernel split is the right
  decomposition.
- **`src/anima_trainer/fp8_quant.py`** — frozen-Linear walker
  (`quantize_frozen_linears`), the `MXFP8FrozenLinear` autograd op (TE
  backend), gate patch for `check_mxfp8_support`. The Frame stays the
  same; just plug in the CuTeDSL kernel as a faster backend for
  `mxfp8_matmul`.
- **`tests/sanity_mxfp8_triton.py`** — parity test rig that already
  understands the Triton output. Reuse for CuTeDSL parity (compare
  CuTeDSL output to Triton output, expect ULP-level match).
- **`outputs/melted1-bf16-cmp/`** + **`outputs/melted1-mxfp8/`** —
  10-epoch sample comparison. MXFP8 quality acceptable (LPIPS
  0.07–0.40 vs bf16, training loss within 1 %).
- **`docs/claude/attention-sm120.md`** — corrected TE-on-sm_120 notes
  (FP8/NVFP4 work; MXFP8 backward broken on cuBLAS; PR #2833 is just
  guards, not workarounds).

## Things we ruled out (don't re-litigate)

- **`tcgen05.BlockScaledMmaOp`** — sm_100 only (`admissible_archs =
  [sm_100a, sm_103a]`). Uses TMEM which sm_120 doesn't have. Verified.
- **TE PR #2833 workaround for MXFP8 backward** — reading the full PR
  diff: it disables/guards broken paths (grouped GEMM on sm_120, NVFP4
  SR), does **not** add a backward MXFP8 dgrad workaround. The regular
  `cublaslt_gemm.cu` MXFP8 dgrad call is untouched.
- **`tl.dot_scaled` autotune** for closing the gap — we tried, 96-config
  sweep picked `BLOCK_K=64, num_warps=8, num_stages=3`. The remaining
  gap to cuBLAS is infrastructure (TMA, persistent, L2 swizzle), not
  tile sizing.
- **Upgrading TE to dev branch** — PR #2833 is still open as of
  2026-05-23. v2.15 is the latest release. Building from PR branch is
  ~2 hrs work and gives us... the same MXFP8 dgrad gate we already have.
- **bf16 LoKr params + skip Prodigy upcast** — tested, ~40 ms / step
  slower. Prodigy's `y = p.float()` is a no-op view when p is fp32; allocates
  for bf16. Don't change.
- **CuTeDSL on tcgen05** — different namespace, sm_100 only.

## Open questions for the new conversation

1. **How does CuTeDSL handle the SFA/SFB pointer-set on the
   `MmaBlockScaledTrait`?** The `set(Field.SFA, ptr)` API is shown in the
   trait base class but the exact integration with `cute.gemm` in
   warp-level (vs tcgen05) needs to be worked out. The
   `nvfp4_gemm_0.py` example uses `tcgen05.MmaMXF4NVF4Op` so its
   trait-setting code is via UMMA, not warp MMA. We need the equivalent
   for warp MMA — possibly via the partition mechanism (similar to how
   `partition_A`, `partition_B` work for the data tensors).

2. **What's the right `permutation_mnk` for sm_120 MXFP8?** The
   `dense_gemm_sm120.py` FP8 path uses
   `(atom_layout[0]*16, atom_layout[1]*8*2, atom_layout[2]*16)` with
   instruction shape `(16, 8, 16)`. For MXFP8 the instruction shape is
   `(16, 8, 32)` so the K-dim term doubles. The N-dim `*2` is for
   `ldmatrix.x4` retiling — verify this still applies for FP8 vs FP16.

3. **Pipeline tx_count for the SFA/SFB tiles** — these are small (one
   uint8 per 32 K-elements) but need to count against the TMA bytes-
   committed. See `nvfp4_gemm_0.py:252-259` for the pattern (4-operand
   total `tx_count`).

4. **Compile cache key** — `@cute.jit` caches by argument shape/dtype.
   For our 4 production shapes (D combinations), we'd get 4 compile
   warmups (each ~10-30s). Should we expose a way to pre-warm them at
   trainer init?

## How to bootstrap the new conversation

1. Read **CLAUDE.md** + **docs/claude/attention-sm120.md** for project context.
2. Read **this file** for the CuTeDSL specifics.
3. Re-fetch the reference files (they're under 3000 lines total):
   ```bash
   mkdir -p /tmp/cutlass_refs && cd /tmp/cutlass_refs
   gh api repos/NVIDIA/cutlass/contents/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu --jq .content | base64 -d > 79c.cu
   gh api repos/NVIDIA/cutlass/contents/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py --jq .content | base64 -d > dense_gemm_sm120.py
   gh api repos/NVIDIA/cutlass/contents/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py --jq .content | base64 -d > nvfp4_gemm_0.py
   ```
4. Confirm env: `CUTE_DSL_ARCH=sm_120a python -c "from cutlass.cute.nvgpu.warp.mma import MmaMXF8Op; import cutlass; print(MmaMXF8Op(cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU).admissible_archs)"`. Should print `[Arch.sm_120a, Arch.sm_120f, Arch.sm_121a, Arch.sm_121f]`.
5. New file target: `src/anima_trainer/mxfp8_cutedsl.py`. Start by copying
   `dense_gemm_sm120.py` verbatim and stripping the CLI/argparse code (save ~80 lines),
   then add SFA/SFB tensors at each layer (constructor → setup_attributes → __call__ →
   kernel body). Use `nvfp4_gemm_0.py` lines 188-260 (SFA/SFB SMEM + TMA atom setup)
   as the template. Replace `MmaF16BF16Op` with `MmaMXF8Op`.
6. First verification target: compile + run on a small shape (M=N=K=128) and check
   output is finite. Then expand to parity vs Triton.

## Time estimate (calibrated to this session's pace)

- Scaffold the file (copy dense_gemm + strip + add SFA/SFB declarations): 1 hr
- Get it to compile (MLIR errors are the painful part): 1-2 hr
- Numerical parity vs Triton ground truth: 30 min
- Bench + tune tile sizes: 30 min
- Wire into PyTorch autograd + sanity test: 30 min
- **Total realistic: 3-5 hours of focused work**

If the new session has unrestricted Bash and can iterate on compile
errors quickly, lower end. If MLIR errors are particularly opaque, upper
end.

## Acceptance criteria

The kernel is "done" when:
1. `sanity_mxfp8_cutedsl.py` passes (parity vs Triton ~5 % rel err)
2. Bench shows per-shape ≥ 1.3 × vs bf16 on the heavy shapes (mlp layer1/2)
3. PTX contains the native `kind::mxf8f6f4.block_scale` MMA, zero bf16 fallback
4. End-to-end training run hits ≥ 12 % step time improvement vs bf16
5. 10-epoch sample comparison shows similar quality to the Triton-MXFP8 run
   (LPIPS 0.07-0.40 vs bf16)

That last criterion can use the existing
`outputs/melted1-bf16-cmp/samples/` as the comparison baseline.
