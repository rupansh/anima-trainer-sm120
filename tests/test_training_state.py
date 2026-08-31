from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from anima_trainer.cache import Cache
from anima_trainer.config import Config
from anima_trainer.optim import build as build_optimizer
from anima_trainer.training_state import (
    TrainingStateCache,
    TrainingStateError,
    build_compatibility,
    choose_resume,
    compatibility_mismatches,
    progress_mismatches,
    restore_optimizer_state,
    restore_rng_state,
)


def _assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)


class TrainingStateCacheTests(unittest.TestCase):
    def _trained_pair(self):
        network = torch.nn.Linear(3, 2)
        optimizer = build_optimizer(network.parameters(), d0=1e-6)
        optimizer.train()
        loss = network(torch.tensor([[1.0, 2.0, 3.0]])).square().mean()
        loss.backward()
        optimizer.step()
        return network, optimizer

    def test_round_trip_includes_model_optimizer_progress_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = TrainingStateCache(
                Path(temp_dir) / "cache.lance",
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            network, optimizer = self._trained_pair()
            expected_network = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }
            expected_optimizer = optimizer.state_dict()

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=6,
                global_step=42,
                max_train_epochs=20,
                compatibility={"contract": 1},
            )
            expected_rng_values = (
                random.random(),
                float(np.random.random()),
                torch.rand(4),
            )

            manifest = state.read_manifest()
            self.assertIsNotNone(manifest)
            payload = state.load_payload(manifest)
            self.assertEqual(payload["next_epoch"], 6)
            self.assertEqual(payload["global_step"], 42)
            _assert_nested_equal(self, payload["network"], expected_network)
            _assert_nested_equal(self, payload["optimizer"], expected_optimizer)

            restored_network, restored_optimizer = self._trained_pair()
            restored_network.load_state_dict(payload["network"], strict=True)
            restore_optimizer_state(restored_optimizer, payload["optimizer"])
            _assert_nested_equal(self, restored_network.state_dict(), expected_network)
            _assert_nested_equal(
                self, restored_optimizer.state_dict(), expected_optimizer
            )

            # Disturb all three RNGs, then prove restoration resumes at the
            # values immediately following the checkpoint boundary.
            random.random()
            np.random.random()
            torch.rand(10)
            restore_rng_state(payload["rng"])
            actual_rng_values = (
                random.random(),
                float(np.random.random()),
                torch.rand(4),
            )
            self.assertEqual(actual_rng_values[0], expected_rng_values[0])
            self.assertEqual(actual_rng_values[1], expected_rng_values[1])
            torch.testing.assert_close(
                actual_rng_values[2], expected_rng_values[2], rtol=0, atol=0
            )

    def test_new_generation_atomically_replaces_old_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = TrainingStateCache(
                Path(temp_dir) / "cache.lance",
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            network, optimizer = self._trained_pair()
            first = state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=1,
                global_step=3,
                max_train_epochs=5,
                compatibility={"contract": 1},
            )
            second = state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=2,
                global_step=6,
                max_train_epochs=5,
                compatibility={"contract": 1},
            )
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(list(state.run_dir.glob("state-*.pt")), [second])
            self.assertEqual(state.read_manifest()["next_epoch"], 2)

    def test_payload_digest_detects_same_size_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = TrainingStateCache(
                Path(temp_dir) / "cache.lance",
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            network, optimizer = self._trained_pair()
            payload_path = state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=1,
                global_step=3,
                max_train_epochs=5,
                compatibility={"contract": 1},
            )
            with payload_path.open("r+b") as handle:
                handle.seek(payload_path.stat().st_size // 2)
                byte = handle.read(1)
                handle.seek(-1, 1)
                handle.write(bytes([byte[0] ^ 0x01]))
            with self.assertRaisesRegex(TrainingStateError, "digest check failed"):
                state.load_payload(state.read_manifest())

    def test_non_object_manifest_is_rejected_and_can_be_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = TrainingStateCache(
                Path(temp_dir) / "cache.lance",
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            state.run_dir.mkdir(parents=True)
            state.manifest_path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(TrainingStateError, "not a JSON object"):
                state.read_manifest()
            quarantined = state.quarantine("bad-manifest")
            self.assertTrue(quarantined.is_file())
            self.assertFalse(state.manifest_path.exists())

    def test_manifest_rejects_unsafe_payload_name_and_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = TrainingStateCache(
                Path(temp_dir) / "cache.lance",
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            network, optimizer = self._trained_pair()
            state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=1,
                global_step=3,
                max_train_epochs=5,
                compatibility={"contract": 1},
            )
            manifest = state.read_manifest()

            for field, value, expected_error in (
                ("payload_file", "../state-escape.pt", "unsafe payload path"),
                ("payload_size", False, "payload size is invalid"),
                ("saved_at_ns", -1, "save timestamp is invalid"),
            ):
                with self.subTest(field=field):
                    changed = {**manifest, field: value}
                    state.manifest_path.write_text(
                        json.dumps(changed), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(TrainingStateError, expected_error):
                        state.read_manifest()

    def test_clear_removes_current_resumable_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = TrainingStateCache(
                Path(temp_dir) / "cache.lance",
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            network, optimizer = self._trained_pair()
            payload = state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=1,
                global_step=3,
                max_train_epochs=1,
                compatibility={"contract": 1},
            )
            state.clear()
            self.assertFalse(payload.exists())
            self.assertIsNone(state.read_manifest())

    def test_state_directory_coexists_with_lancedb_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache.lance"
            cache = Cache(cache_path)
            state = TrainingStateCache(
                cache_path,
                output_dir=Path(temp_dir) / "outputs",
                output_name="run",
            )
            network, optimizer = self._trained_pair()
            state.save(
                network=network,
                optimizer=optimizer,
                next_epoch=1,
                global_step=3,
                max_train_epochs=5,
                compatibility={"contract": 1},
            )

            reopened = Cache(cache_path)
            self.assertEqual(
                set(reopened.db.list_tables().tables),
                {"crops", "latents", "text_embeds"},
            )


class ResumePolicyTests(unittest.TestCase):
    manifest = {"next_epoch": 4, "global_step": 12}

    def test_cli_override_wins(self) -> None:
        self.assertTrue(choose_resume(self.manifest, requested=True))
        self.assertFalse(choose_resume(self.manifest, requested=False))

    def test_interactive_prompt_defaults_to_resume(self) -> None:
        self.assertTrue(
            choose_resume(
                self.manifest,
                requested=None,
                input_fn=lambda _prompt: "",
                is_interactive=True,
            )
        )
        self.assertFalse(
            choose_resume(
                self.manifest,
                requested=None,
                input_fn=lambda _prompt: "n",
                is_interactive=True,
            )
        )

    def test_noninteractive_run_requires_an_explicit_choice(self) -> None:
        with self.assertRaisesRegex(TrainingStateError, "--resume or --no-resume"):
            choose_resume(self.manifest, requested=None, is_interactive=False)

    def test_compatibility_excludes_operational_knobs(self) -> None:
        base = Config()
        operational_change = replace(
            base,
            train=replace(
                base.train,
                num_workers=99,
                max_train_epochs=500,
                save_every_n_epochs=17,
            ),
            sample=replace(base.sample, every_n_epochs=0),
        )
        kwargs = {
            "dit_fp": "dit",
            "vae_fp": "vae",
            "te_fp": "te",
            "data_cache_fp": "data",
        }
        self.assertEqual(
            build_compatibility(base, **kwargs),
            build_compatibility(operational_change, **kwargs),
        )

        semantic_change = replace(base, train=replace(base.train, seed=999))
        mismatches = compatibility_mismatches(
            build_compatibility(base, **kwargs),
            build_compatibility(semantic_change, **kwargs),
        )
        self.assertEqual(mismatches, ["train.seed changed"])

    def test_progress_must_be_an_epoch_boundary_within_the_target(self) -> None:
        manifest = {"next_epoch": 4, "global_step": 12}
        self.assertEqual(
            progress_mismatches(
                manifest, max_train_epochs=10, steps_per_epoch=3
            ),
            [],
        )
        self.assertEqual(
            progress_mismatches(
                manifest, max_train_epochs=3, steps_per_epoch=4
            ),
            [
                "target epoch 3 precedes cached epoch 4",
                "cached global step 12 does not match epoch boundary 4 x 4 = 16",
            ],
        )


if __name__ == "__main__":
    unittest.main()
