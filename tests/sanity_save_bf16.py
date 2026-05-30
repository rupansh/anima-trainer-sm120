"""Verify saved adapter safetensors are pure bf16, even after MXFP8 quantization.

Builds a LokrModule, quantizes its base to MXFP8, saves via _save_lora, then
reloads and asserts:
  - every tensor in the file is bf16
  - no `_mxfp8_W*` keys leaked into the file
  - keys cover the expected LoKr params (lokr_w1, lokr_w2, alpha)

Run: `python -m tests.sanity_save_bf16`
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn

os.environ.setdefault("NVTE_CUDA_INCLUDE_DIR", "/opt/cuda/include")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.sdscripts_bridge import ensure_on_path
ensure_on_path()
from anima_trainer.lokr_patch import install as install_lokr_patch
from anima_trainer.fp8_quant import quantize_lokr_base_weights
install_lokr_patch()

from lycoris.modules.lokr import LokrModule
from safetensors.torch import save_file, safe_open


def _save_lora_like_train_py(network, path):
    """Mirror of train.py:_save_lora — keep this in sync if train.py changes."""
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in network.state_dict().items()}
    save_file(sd, str(path))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = "cuda"
    dtype = torch.bfloat16

    # Build a tiny LoKr-wrapped network — one Linear in a container.
    base = nn.Linear(2048, 2048, bias=False).to(device, dtype)
    network = nn.Module()
    mod = LokrModule(
        "test", base,
        multiplier=1.0, lora_dim=128, alpha=128, factor=8, full_matrix=True,
    ).to(device)
    mod.apply_to()
    network.lokr_test = mod

    # Pre-quantize for MXFP8 (this attaches _mxfp8_W as a non-buffer attribute)
    n = quantize_lokr_base_weights(network)
    print(f"quantized modules: {n}")
    assert hasattr(mod, "_mxfp8_W") and mod._mxfp8_W is not None

    # Save
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "adapter.safetensors"
        _save_lora_like_train_py(network, out)
        print(f"saved {out} ({out.stat().st_size} bytes)")

        # Inspect: every tensor bf16, no _mxfp8 keys
        with safe_open(out, framework="pt") as f:
            keys = list(f.keys())
            print(f"keys ({len(keys)}):")
            for k in sorted(keys):
                t = f.get_tensor(k)
                ok = t.dtype == torch.bfloat16
                flag = "OK" if ok else "FAIL"
                print(f"  [{flag}] {k:50s}  shape={tuple(t.shape)}  dtype={t.dtype}")
                assert ok, f"non-bf16 tensor in save: {k} is {t.dtype}"
                assert "_mxfp8" not in k, f"MXFP8 internal attr leaked into save: {k}"
            # Production sanity: should contain the LoKr params
            expected = {"lokr_test.alpha", "lokr_test.lokr_w1", "lokr_test.lokr_w2"}
            missing = expected - set(keys)
            assert not missing, f"expected keys missing from save: {missing}"
            print(f"  expected LoKr keys present: {sorted(expected)}")

    print("\nALL CHECKS PASSED — saved adapter is pure bf16, no MXFP8 leakage")


if __name__ == "__main__":
    main()
