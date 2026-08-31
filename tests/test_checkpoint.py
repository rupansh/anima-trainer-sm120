from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file
import torch

from anima_trainer.train import _save_lora


class AdapterCheckpointTests(unittest.TestCase):
    def test_adapter_checkpoint_is_replaceable_and_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            network = torch.nn.Linear(3, 2)
            _save_lora(network, path)
            first = load_file(path)
            self.assertTrue(first)
            self.assertTrue(all(tensor.dtype == torch.bfloat16 for tensor in first.values()))

            with torch.no_grad():
                network.weight.fill_(7)
            _save_lora(network, path)
            second = load_file(path)
            torch.testing.assert_close(
                second["weight"],
                torch.full_like(second["weight"], 7),
                rtol=0,
                atol=0,
            )
            self.assertEqual(list(path.parent.glob(".adapter.safetensors.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
