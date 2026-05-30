"""Precision mode for the DiT forward.

Supported modes:

  - **bf16** (default): the trainer's storage + math dtype. All weights and
    activations are bf16; LoKr deltas are bf16; optimizer state is fp32
    via Prodigy+SF's upcast (cannot be skipped — see
    `docs/claude/throughput.md`).
  - **mxfp8**: bf16 storage, but each unwrapped frozen Linear's weight is
    pre-quantized once to MXFP8 (E4M3 with E8M0 32-element block scales)
    at attach time. Forward GEMMs against the base run in MXFP8 via a
    custom `tex.general_gemm` path; backward dgrad stays bf16 because sm_120
    cuBLAS lacks the heuristic for MXFP8 dgrad layouts.
  - **fp8**: bf16 storage, but each unwrapped frozen `nn.Linear` is swapped
    for a `te.Linear` and the DiT forward runs under
    `te.fp8_autocast(Float8BlockScaling())`. Unlike MXFP8, sm_120 cuBLAS
    *does* have heuristics for `Float8BlockScaling` dgrad, so both forward
    and backward GEMMs against frozen weights run in FP8 (128×128 block
    scales). On microbench (B=32768, D=2048): bf16 fwd+bwd 2.38ms, fp8
    fwd+bwd 1.53ms = 1.55×. LoKr deltas stay bf16 (LoKr-wrapped Linears
    are not swapped — their forward materializes `base + α·diff_W` which
    can't be cleanly expressed as a `te.Linear` call).

The earlier just-in-time MXFP8 on the merged weight failed because LoKr's
`merged_W = base_W + α · diff_W` changes every step — re-quantizing every
step ate the win. Both `mxfp8` and `fp8` only quantize the FROZEN half,
which is precisely what makes them work.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext
from typing import Literal

Mode = Literal["bf16", "mxfp8", "fp8"]


def torch_dtype(mode: Mode) -> torch.dtype:
    """Storage dtype for activations + LoKr deltas + base weight pre-quant."""
    if mode in ("bf16", "mxfp8", "fp8"):
        return torch.bfloat16
    raise NotImplementedError(
        f"precision={mode!r}: only 'bf16', 'mxfp8', or 'fp8' is supported"
    )


@contextmanager
def autocast_for(mode: Mode, device_type: str = "cuda"):
    """Autocast context. All three modes use bf16 autocast. MXFP8 GEMMs are
    invoked explicitly via `mxfp8_frozen_linear`; FP8 GEMMs are gated by
    `fp8_autocast` (see `fp8_autocast_for`) which is applied separately."""
    if mode not in ("bf16", "mxfp8", "fp8"):
        raise NotImplementedError(mode)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        yield


def fp8_autocast_for(mode: Mode):
    """Context manager that enables TE FP8 autocast for `precision="fp8"`,
    and a no-op for any other mode. Use around any forward through the DiT
    (train + sample) so that the `te.Linear` swaps actually run in FP8."""
    if mode != "fp8":
        return nullcontext()
    # Lazy import: TE init touches CUDA + writes NVTE_CUDA_INCLUDE_DIR.
    from .fp8_quant import fp8_block_autocast
    return fp8_block_autocast()


def quantize_dit_in_place(dit: nn.Module, mode: Mode, **_kwargs) -> None:
    """No-op for all modes — MXFP8 / FP8 quantization is wired up in
    train.py after `attach_lokr` (see `fp8_quant.quantize_frozen_linears`
    and `fp8_quant.swap_frozen_linears_to_te`)."""
    if mode not in ("bf16", "mxfp8", "fp8"):
        raise NotImplementedError(
            f"precision={mode!r}: only 'bf16', 'mxfp8', or 'fp8' is supported"
        )
