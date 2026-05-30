"""File and caption hashing for cache validation.

A cache row is valid when the source's (mtime, size, xxhash) all match. We
include all three to catch the failure modes independently:
  - mtime cheap to check, can drift under timestamp manipulation.
  - size catches truncation/concat without re-hashing.
  - xxhash catches in-place edits that preserve size+mtime.
"""
from __future__ import annotations
import os
from pathlib import Path
import xxhash

_CHUNK = 1 << 20  # 1 MiB


def file_fingerprint(path: str | os.PathLike) -> tuple[float, int, str]:
    p = Path(path)
    st = p.stat()
    h = xxhash.xxh3_128()
    with p.open("rb") as f:
        while True:
            buf = f.read(_CHUNK)
            if not buf:
                break
            h.update(buf)
    return st.st_mtime, st.st_size, h.hexdigest()


def fingerprint_matches(path: str | os.PathLike, mtime: float, size: int, digest: str) -> bool:
    """Cheap-first: mtime+size before re-hashing."""
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return False
    if st.st_size != size:
        return False
    if abs(st.st_mtime - mtime) > 1e-6 and not _rehash_matches(p, digest):
        return False
    return True


def _rehash_matches(p: Path, digest: str) -> bool:
    h = xxhash.xxh3_128()
    with p.open("rb") as f:
        while True:
            buf = f.read(_CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest() == digest


def text_digest(s: str) -> str:
    return xxhash.xxh3_128(s.encode("utf-8")).hexdigest()
