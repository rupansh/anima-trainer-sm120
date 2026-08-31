from __future__ import annotations

import unittest

from anima_trainer.resilient_loader import (
    ResilientDataLoader,
    is_worker_process_failure,
)


class _FakeLoader:
    def __init__(self, batch_sampler, *, fail_with: RuntimeError | None = None):
        self.batch_sampler = batch_sampler
        self.fail_with = fail_with

    def __iter__(self):
        for index, batch in enumerate(self.batch_sampler):
            if index == 1 and self.fail_with is not None:
                raise self.fail_with
            yield tuple(batch)


class ResilientDataLoaderTests(unittest.TestCase):
    def test_worker_death_replays_only_unfinished_batches(self) -> None:
        calls: list[dict] = []

        def factory(_dataset, **kwargs):
            calls.append(kwargs)
            failure = None
            if kwargs["num_workers"]:
                failure = RuntimeError(
                    "DataLoader worker (pid(s) 101, 102) exited unexpectedly"
                )
            return _FakeLoader(kwargs["batch_sampler"], fail_with=failure)

        batches = [[0, 1], [2, 3], [4, 5]]
        loader = ResilientDataLoader(
            list(range(6)),
            batch_sampler=batches,
            collate_fn=lambda batch: batch,
            num_workers=2,
            pin_memory=True,
            seed=7,
            loader_factory=factory,
        )

        self.assertEqual(list(loader.iter_epoch()), [(0, 1), (2, 3), (4, 5)])
        self.assertTrue(loader.using_fallback)
        self.assertTrue(calls[0]["persistent_workers"])
        self.assertTrue(all(not call["persistent_workers"] for call in calls[1:]))
        self.assertTrue(all(call["num_workers"] == 0 for call in calls[1:]))

        # Once multiprocessing has failed, later epochs stay in the main
        # process and still start from the complete deterministic plan.
        self.assertEqual(list(loader.iter_epoch()), [(0, 1), (2, 3), (4, 5)])

    def test_dataset_exception_is_not_hidden_as_worker_death(self) -> None:
        calls = 0

        class DatasetErrorLoader:
            def __iter__(self):
                raise RuntimeError(
                    "Caught RuntimeError in DataLoader worker process 0.\n"
                    "Original Traceback: bad cached tensor"
                )
                yield  # pragma: no cover

        def factory(_dataset, **_kwargs):
            nonlocal calls
            calls += 1
            return DatasetErrorLoader()

        loader = ResilientDataLoader(
            [0],
            batch_sampler=[[0]],
            collate_fn=lambda batch: batch,
            num_workers=1,
            pin_memory=False,
            seed=1,
            loader_factory=factory,
        )
        with self.assertRaisesRegex(RuntimeError, "bad cached tensor"):
            list(loader.iter_epoch())
        self.assertEqual(calls, 1)
        self.assertFalse(loader.using_fallback)

    def test_configured_single_process_loader_is_reused(self) -> None:
        calls = 0

        def factory(_dataset, **kwargs):
            nonlocal calls
            calls += 1
            return _FakeLoader(kwargs["batch_sampler"])

        loader = ResilientDataLoader(
            [0, 1],
            batch_sampler=[[0], [1]],
            collate_fn=lambda batch: batch,
            num_workers=0,
            pin_memory=False,
            seed=1,
            loader_factory=factory,
        )
        self.assertEqual(list(loader.iter_epoch()), [(0,), (1,)])
        self.assertEqual(list(loader.iter_epoch()), [(0,), (1,)])
        self.assertEqual(calls, 1)

    def test_failure_classifier_is_narrow(self) -> None:
        self.assertTrue(
            is_worker_process_failure(
                RuntimeError("DataLoader worker (pid 10) is killed by signal: Illegal instruction")
            )
        )
        self.assertFalse(
            is_worker_process_failure(
                RuntimeError("Caught ValueError in DataLoader worker process 0")
            )
        )
        self.assertTrue(is_worker_process_failure(ConnectionResetError(104, "reset")))
        self.assertFalse(
            is_worker_process_failure(
                ConnectionResetError(
                    104,
                    "Caught ConnectionResetError in DataLoader worker process 0",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
