"""Add the vendored sd-scripts repo to sys.path so we can import its Anima libs.

We re-use sd-scripts' loaders for DiT / Qwen3 / VAE since they encode the
exact model config the Anima checkpoints assume. We do not run any training
code from sd-scripts — only the parts that are load-bearing for correctness.
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_SD_SCRIPTS = _HERE.parents[2] / "sd-scripts"

_added = False


def ensure_on_path() -> None:
    global _added
    if _added:
        return
    if not _SD_SCRIPTS.is_dir():
        raise FileNotFoundError(f"vendored sd-scripts not found at {_SD_SCRIPTS}")
    p = str(_SD_SCRIPTS)
    if p not in sys.path:
        sys.path.insert(0, p)
    _added = True
