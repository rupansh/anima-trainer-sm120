from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from anima_trainer.config import load
from anima_trainer.dataset import scan_dataset
from anima_trainer.model import fingerprint_model_files


class ConfigValidationTests(unittest.TestCase):
    def _load_with(self, section: str, setting: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            path.write_text(
                textwrap.dedent(
                    f"""
                    [{section}]
                    {setting}
                    """
                ),
                encoding="utf-8",
            )
            return load(path)

    def test_zero_checkpoint_cadence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "save_every_n_epochs must be positive"):
            self._load_with("train", "save_every_n_epochs = 0")

    def test_negative_workers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_workers cannot be negative"):
            self._load_with("train", "num_workers = -1")

    def test_zero_sampling_cadence_disables_sampling(self) -> None:
        cfg = self._load_with("sample", "every_n_epochs = 0")
        self.assertEqual(cfg.sample.every_n_epochs, 0)

    def test_tlokr_variant_and_rank_floor_parse(self) -> None:
        cfg = self._load_with(
            "lokr",
            'variant = "tlokr"\ntimestep_min_rank_ratio = 0.5',
        )
        self.assertEqual(cfg.lokr.variant, "tlokr")
        self.assertEqual(cfg.lokr.timestep_min_rank_ratio, 0.5)

    def test_invalid_timestep_rank_floor_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be in \\(0, 1\\]"):
            self._load_with("lokr", "timestep_min_rank_ratio = 0.0")


class DatasetOrderingTests(unittest.TestCase):
    def test_scan_order_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            folder = root / "2_subject"
            folder.mkdir()
            for stem in ("z", "a"):
                (folder / f"{stem}.png").write_bytes(b"")
                (folder / f"{stem}.txt").write_text(stem, encoding="utf-8")

            samples = scan_dataset(root)
            self.assertEqual(
                [(sample.src_path, sample.caption) for sample in samples],
                [
                    ("2_subject/a.png", "a"),
                    ("2_subject/a.png", "a"),
                    ("2_subject/z.png", "z"),
                    ("2_subject/z.png", "z"),
                ],
            )


class ModelFingerprintTests(unittest.TestCase):
    def test_model_files_are_hashed_once_for_preflight_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dit = root / "dit.safetensors"
            qwen = root / "qwen.safetensors"
            vae = root / "vae.safetensors"
            dit.write_bytes(b"dit")
            qwen.write_bytes(b"qwen")
            vae.write_bytes(b"vae")

            first = fingerprint_model_files(
                dit_path=str(dit), qwen3_path=str(qwen), vae_path=str(vae)
            )
            qwen.write_bytes(b"changed")
            second = fingerprint_model_files(
                dit_path=str(dit), qwen3_path=str(qwen), vae_path=str(vae)
            )
            self.assertEqual(first.dit, second.dit)
            self.assertEqual(first.vae, second.vae)
            self.assertNotEqual(first.text_encoder, second.text_encoder)


if __name__ == "__main__":
    unittest.main()
