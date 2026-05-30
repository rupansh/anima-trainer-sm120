"""Per-bucket CUDA-graph capture for the Anima training step.

Buckets are fixed (`crop.py` enumerates them; `BucketBatchSampler` only emits
single-bucket batches), so every (bucket_idx, batch_size) pair will be seen
many times during training. After a few warmup steps to let cuDNN and Triton
autotune settle, we capture one CUDA graph per pair and replay it for every
subsequent step in that bucket. Hot-loop cost drops to:

   batch h2d copy  +  graph.replay()  +  optimizer.step()

— no Python op dispatch, no per-kernel CPU launch overhead.

What's IN the graph
-------------------
Forward through the DiT, the MSE loss, and `loss.backward()`. Gradients
accumulate into static `.grad` buffers on the LoKr parameters, which
**must not be `None`** when capture starts (see `_warm_grads`).

What's OUT of the graph
-----------------------
`optimizer.step()` and `optimizer.zero_grad(set_to_none=False)`. Prodigy+SF
has python control flow and dynamic state — capturing it would be brittle.
Optimizer step is a small fraction of step time anyway.

Compat with other patches
-------------------------
  * LoKr-patched forward: captured fine; the patch computes `merged_W =
    base + α·diff_W` from frozen `base` and learnable `diff_W` each step
    — the merge tensor lives in the graph's private allocator.
  * Liger / FusedAdaLN / FusedGatedAdd / RoPE Triton kernels: all
    capturable.
  * Selective activation checkpointing (SAC): `torch.utils.checkpoint`
    with `use_reentrant=False` is capturable on PyTorch 2.4+.
  * FP8: te.Linear under `fp8_autocast` is capturable when the metadata
    cache has been warmed (3 warmup steps cover this).

VRAM
----
Captured graphs allocate from a private pool. We pass a **shared pool**
across all buckets via `graph_pool_handle()` so multiple captured graphs
do not multiply VRAM. Net VRAM growth vs. eager mode is small (a few
hundred MB for the graph object + static buffers).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch


@dataclass
class _BucketGraph:
    """One captured graph + the static tensors that feed it."""
    graph: torch.cuda.CUDAGraph
    # Inputs (we copy real data into these every step).
    latent: torch.Tensor
    prompt_embeds: torch.Tensor
    qwen3_mask: torch.Tensor
    t5_ids: torch.Tensor
    t5_mask: torch.Tensor
    noisy: torch.Tensor      # noisy5 (B, C, 1, H, W)
    target: torch.Tensor
    timesteps: torch.Tensor
    padding_mask: torch.Tensor
    # Output the graph wrote into.
    loss: torch.Tensor


class CUDAGraphRunner:
    """Maintain one captured forward+backward graph per (bucket, batch_size).

    Usage
    -----
    >>> runner = CUDAGraphRunner(forward_and_loss_fn, warmup_steps=3)
    >>> # inside training loop:
    >>> loss = runner.step(bucket_key, batch_dict)
    >>> # caller handles: clip_grad_norm, optimizer.step, zero_grad(set_to_none=False)

    `forward_and_loss_fn(static_batch) -> loss_tensor` is the user's
    captured region — it consumes the static input tensors and returns a
    scalar `loss`. The runner calls `loss.backward()` for you (so .grad
    accumulates on parameters that were `param.grad`-initialized before
    capture began).
    """

    def __init__(
        self,
        forward_and_loss_fn: Callable[[dict], torch.Tensor],
        *,
        warmup_steps: int = 3,
    ) -> None:
        self._forward = forward_and_loss_fn
        self._warmup_steps = warmup_steps
        # Shared memory pool: all per-bucket graphs allocate from this so
        # we don't pay 28× the per-graph workspace.
        self._pool = torch.cuda.graph_pool_handle()
        # Dedicated side stream for warmup forward+backward and capture.
        # Captured graphs replay onto whatever stream calls graph.replay()
        # — typically the default stream where `optim.step` lives.
        # Running warmup on the same stream we'll capture on avoids
        # autograd's "stale reference to default stream" capture error.
        self._capture_stream = torch.cuda.Stream()
        # Map (bucket_key) -> warmup-count or _BucketGraph
        self._state: dict = {}

    def _warm_grads(self, params: list[torch.nn.Parameter]) -> None:
        """Ensure every trainable parameter has a `.grad` tensor allocated.
        CUDA-graph capture cannot allocate gradient tensors during
        replay — `param.grad` must already be a real tensor and stay
        the same tensor across iterations.
        """
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    def _allocate_static(self, batch: dict) -> _BucketGraph:
        """Allocate static input tensors matching the shapes/dtypes/devices
        of the first warmup batch. We deep-clone everything so subsequent
        warmup steps can write into them via `.copy_()` (which is also what
        the captured replay path uses).
        """
        def static_like(t: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(t.detach())

        return _BucketGraph(
            graph=torch.cuda.CUDAGraph(),
            latent=static_like(batch["latent"]),
            prompt_embeds=static_like(batch["prompt_embeds"]),
            qwen3_mask=static_like(batch["qwen3_mask"]),
            t5_ids=static_like(batch["t5_ids"]),
            t5_mask=static_like(batch["t5_mask"]),
            noisy=static_like(batch["noisy"]),
            target=static_like(batch["target"]),
            timesteps=static_like(batch["timesteps"]),
            padding_mask=static_like(batch["padding_mask"]),
            # Loss placeholder — gets overwritten during capture.
            loss=torch.empty((), device=batch["latent"].device, dtype=torch.float32),
        )

    def _copy_in(self, bg: _BucketGraph, batch: dict) -> None:
        """Copy real-batch tensors into the static buffers (non-blocking).
        Must be called before every replay AND before warmup forward calls
        (so the warmup path behaves like the replay path).
        """
        bg.latent.copy_(batch["latent"], non_blocking=True)
        bg.prompt_embeds.copy_(batch["prompt_embeds"], non_blocking=True)
        bg.qwen3_mask.copy_(batch["qwen3_mask"], non_blocking=True)
        bg.t5_ids.copy_(batch["t5_ids"], non_blocking=True)
        bg.t5_mask.copy_(batch["t5_mask"], non_blocking=True)
        bg.noisy.copy_(batch["noisy"], non_blocking=True)
        bg.target.copy_(batch["target"], non_blocking=True)
        bg.timesteps.copy_(batch["timesteps"], non_blocking=True)
        bg.padding_mask.copy_(batch["padding_mask"], non_blocking=True)

    def step(
        self,
        bucket_key: tuple,
        batch: dict,
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """Run one training step for `bucket_key`.

        Returns the loss tensor (already-detached scalar on GPU). Caller is
        responsible for `optim.step()`, `zero_grad(set_to_none=False)`, and
        any grad clipping — all of which happen outside the captured graph.
        """
        st = self._state.get(bucket_key)
        default_stream = torch.cuda.current_stream()

        # --- Path A: captured replay ---
        if isinstance(st, _BucketGraph):
            self._copy_in(st, batch)
            st.graph.replay()
            return st.loss

        # --- Path B/C: warmup or capture — both run on the side stream so
        # that autograd's saved stream references stay consistent across
        # warmup -> capture.
        self._capture_stream.wait_stream(default_stream)

        if st is None:
            self._state[bucket_key] = 1
            warmup_count = 1
        else:
            self._state[bucket_key] = st + 1
            warmup_count = st + 1

        # Path B: warmup forward+backward on the side stream.
        if warmup_count < self._warmup_steps:
            with torch.cuda.stream(self._capture_stream):
                loss = self._forward(batch)
                loss.backward()
            default_stream.wait_stream(self._capture_stream)
            return loss.detach()

        # Path C: final warmup pass, then capture. The bg.* buffers live in
        # the captured graph's memory pool — allocate them on the side
        # stream too so the pool ownership is consistent.
        with torch.cuda.stream(self._capture_stream):
            # One last eager pass to populate `.grad` buffers + settle
            # any first-call lazy state (Triton autotune, cuDNN heuristic).
            loss = self._forward(batch)
            loss.backward()
            self._warm_grads(params)
            bg = self._allocate_static(batch)
            self._copy_in(bg, batch)
            torch.cuda.current_stream().synchronize()

            # The captured region: forward+backward reading from bg.* and
            # accumulating into params' .grad buffers. The override flag
            # tells autograd to redirect any AccumulateGrad nodes whose
            # saved stream reference came from a different (warmup) stream
            # to the capture stream — without it, capture errors out.
            torch.autograd.graph.set_override_stale_capture_stream(True)
            try:
                with torch.cuda.graph(bg.graph, pool=self._pool):
                    static_loss = self._forward({
                        "latent": bg.latent,
                        "prompt_embeds": bg.prompt_embeds,
                        "qwen3_mask": bg.qwen3_mask,
                        "t5_ids": bg.t5_ids,
                        "t5_mask": bg.t5_mask,
                        "noisy": bg.noisy,
                        "target": bg.target,
                        "timesteps": bg.timesteps,
                        "padding_mask": bg.padding_mask,
                    })
                    # Write loss into the persistent bg.loss buffer.
                    bg.loss.copy_(static_loss.detach())
                    static_loss.backward()
            finally:
                torch.autograd.graph.set_override_stale_capture_stream(False)
        default_stream.wait_stream(self._capture_stream)
        self._state[bucket_key] = bg
        return loss.detach()


def make_bucket_key(latent: torch.Tensor, t5_ids: torch.Tensor) -> tuple:
    """Stable hashable key for a captured graph. Shape covers the latent
    buckets; the t5_ids shape pins the (batch_size, max_token_len).
    """
    return (tuple(latent.shape), tuple(t5_ids.shape))
