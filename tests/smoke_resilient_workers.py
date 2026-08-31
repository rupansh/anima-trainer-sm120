"""Real-process smoke for persistent DataLoader workers on Python 3.14."""
from __future__ import annotations

import os
import torch
from torch.utils.data import (
    BatchSampler,
    Dataset,
    SequentialSampler,
    TensorDataset,
    get_worker_info,
)
from torch.utils.data._utils.collate import default_collate

from anima_trainer.resilient_loader import ResilientDataLoader


class _ExitInWorkerDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        if index == 0 and get_worker_info() is not None:
            os._exit(91)
        return index


def main() -> None:
    dataset = TensorDataset(torch.arange(12))
    sampler = BatchSampler(
        SequentialSampler(dataset), batch_size=3, drop_last=False
    )
    loader = ResilientDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=default_collate,
        num_workers=2,
        pin_memory=False,
        seed=42,
    )

    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    first = [batch[0].tolist() for batch in loader.iter_epoch()]
    first_pids = [worker.pid for worker in loader._loader._iterator._workers]
    second = [batch[0].tolist() for batch in loader.iter_epoch()]
    second_pids = [worker.pid for worker in loader._loader._iterator._workers]

    assert first == expected
    assert second == expected
    assert first_pids == second_pids
    assert not loader.using_fallback
    print(f"persistent worker smoke passed: pids={first_pids}")

    failing_dataset = _ExitInWorkerDataset()
    failing_sampler = BatchSampler(
        SequentialSampler(failing_dataset), batch_size=2, drop_last=False
    )
    recovering_loader = ResilientDataLoader(
        failing_dataset,
        batch_sampler=failing_sampler,
        collate_fn=default_collate,
        num_workers=1,
        pin_memory=False,
        seed=42,
    )
    recovered = [batch.tolist() for batch in recovering_loader.iter_epoch()]
    assert recovered == [[0, 1], [2, 3]]
    assert recovering_loader.using_fallback
    print("abrupt worker-death recovery smoke passed")


if __name__ == "__main__":
    main()
