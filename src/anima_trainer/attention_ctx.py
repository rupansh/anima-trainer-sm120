"""Centralised SDPA backend control for sm_120 (RTX PRO 6000 Blackwell).

Why this exists:
  - FA3/FA4 do not run on sm_120 (no TMEM, no Blackwell-datacenter async pipes).
  - FA2 in stock flash-attn doesn't whitelist sm_120 and falls back to SM89.
  - SageAttention is inference-only (no backward).
  - cuDNN is the only kernel on sm_120 with genuine Blackwell-specific
    optimizations (per cuDNN release notes: 2-CTA MMA, dQ matmul reordering).

Empirical microbench on this GPU (B=8, H=16, D=128, bf16, fwd+bwd):
                  cuDNN    Flash   meff    math
    self_1024     13.28    13.87   58.31   128.5      <- cuDNN wins
    self_896×1152 13.03    13.50   56.74   127.4      <- cuDNN wins
    cross_1024     2.71     2.53    7.58    18.1      <- Flash slightly faster
    self_512       1.01     0.87    3.71     8.80     <- Flash slightly faster

Torch's default selector on sm_120 already picks cuDNN for the heavy self_1024
shape and Flash for cross/small shapes — i.e. it's already near-optimal. We
expose a context manager that pins both as allowed and excludes the slow paths
(mem_eff, math) so a future torch version can't silently regress to them.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch


@contextmanager
def sm120_sdpa():
    """Restrict SDPA to {cuDNN, Flash}. torch picks the faster of the two per shape."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        yield


@contextmanager
def cudnn_only():
    """For sampling/inference, cuDNN wins forward-only on every shape we use."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        yield
