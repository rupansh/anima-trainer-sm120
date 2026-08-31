"""Persistent DataLoader workers with deterministic single-process failover."""
from __future__ import annotations

import sys
from typing import Callable, Iterator

import torch
from torch.utils.data import DataLoader, Sampler


_WORKER_FAILURE_MARKERS = (
    "exited unexpectedly",
    "killed by signal",
    "is killed by signal",
)


def is_worker_process_failure(exc: BaseException) -> bool:
    """Recognize abrupt worker death without swallowing dataset exceptions."""
    message = str(exc)
    if "DataLoader worker" in message and any(
        marker in message for marker in _WORKER_FAILURE_MARKERS
    ):
        return True
    # A forkserver/spawn worker can disappear while DataLoader is still
    # establishing its control channel, before PyTorch has a PID to include in
    # the usual RuntimeError. Do not mistake the same exception raised *by the
    # dataset* (which PyTorch labels as caught in a worker) for transport loss.
    caught_dataset_error = "Caught " in message and "DataLoader worker process" in message
    return not caught_dataset_error and isinstance(
        exc, (ConnectionError, EOFError, BrokenPipeError)
    )


class _RemainingBatchSampler(Sampler[list[int]]):
    """Replay a deterministic batch plan starting after completed batches."""

    def __init__(self, batch_sampler, completed: int):
        self.batch_sampler = batch_sampler
        self.completed = completed

    def __iter__(self):
        iterator = iter(self.batch_sampler)
        for _ in range(self.completed):
            try:
                next(iterator)
            except StopIteration:
                return
        yield from iterator

    def __len__(self) -> int:
        return max(0, len(self.batch_sampler) - self.completed)


class ResilientDataLoader:
    """Keep workers across epochs and fall back after abrupt worker death.

    A failed multiprocessing iterator may have prefetched beyond the last
    batch consumed by the training loop.  Recovery therefore reconstructs the
    epoch's deterministic batch plan and skips exactly the number of batches
    that successfully completed, instead of trusting the failed iterator's
    internal sampler position.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_sampler,
        collate_fn,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        loader_factory: Callable = DataLoader,
    ) -> None:
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.configured_workers = num_workers
        self.active_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_factory = loader_factory
        self.worker_seed_generator = torch.Generator().manual_seed(seed)
        self._loader = self._make_loader(batch_sampler, num_workers)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    @property
    def using_fallback(self) -> bool:
        return self.active_workers == 0 and self.configured_workers > 0

    def _make_loader(self, batch_sampler, num_workers: int):
        return self.loader_factory(
            self.dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0,
            generator=self.worker_seed_generator,
        )

    def iter_epoch(self) -> Iterator:
        completed = 0

        if self.configured_workers == 0:
            yield from self._loader
            return

        if self.active_workers > 0:
            try:
                iterator = iter(self._loader)
                while True:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        return
                    yield batch
                    completed += 1
            except Exception as exc:
                if not is_worker_process_failure(exc):
                    raise
                print(
                    "[data] worker process died after "
                    f"{completed}/{len(self)} batches; switching permanently to "
                    "num_workers=0 and replaying the remaining deterministic batch plan",
                    file=sys.stderr,
                    flush=True,
                )
                # Drop references to the failed iterator/loader so PyTorch can
                # tear down its pin-memory thread and worker bookkeeping.
                iterator = None
                self.active_workers = 0
                # Keep a complete single-process loader for later epochs.
                self._loader = self._make_loader(self.batch_sampler, 0)

        remaining = _RemainingBatchSampler(self.batch_sampler, completed)
        fallback = self._loader if completed == 0 else self._make_loader(remaining, 0)
        yield from fallback
