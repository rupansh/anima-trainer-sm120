"""LanceDB cache for cropped images, VAE latents and Qwen3 text embeddings.

Validation rules (read-side):
  - source-image rows: revalidate against (mtime, size, xxhash) of the on-disk image.
  - latent rows: also gated on a VAE-weights fingerprint, since changing the VAE
    invalidates every cached latent.
  - text-embed rows: gated on (caption hash, TE-weights fingerprint).

Anything that fails revalidation is treated as a miss and recomputed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pyarrow as pa
import lancedb

from .hashing import file_fingerprint, fingerprint_matches, text_digest


# ------- arrow schemas -------------------------------------------------------

CROPS_SCHEMA = pa.schema([
    ("src_path", pa.string()),         # key: canonical path relative to dataset root
    ("src_mtime", pa.float64()),
    ("src_size", pa.int64()),
    ("src_xxhash", pa.string()),
    ("bucket_idx", pa.int32()),
    ("bucket_w", pa.int32()),
    ("bucket_h", pa.int32()),
    ("crop_jpeg", pa.binary()),        # JPEG-encoded crop at bucket resolution
])

LATENTS_SCHEMA = pa.schema([
    ("src_path", pa.string()),         # key
    ("src_xxhash", pa.string()),
    ("vae_fp", pa.string()),           # VAE weights fingerprint at encode time
    ("bucket_idx", pa.int32()),
    ("dtype", pa.string()),            # e.g. "bfloat16"
    ("shape", pa.list_(pa.int32())),
    ("data", pa.binary()),             # raw bytes; reshape with shape+dtype
])

TEXT_EMBEDS_SCHEMA = pa.schema([
    ("caption_xxhash", pa.string()),   # key
    ("caption", pa.string()),
    ("src_path", pa.string()),
    ("te_fp", pa.string()),
    ("dtype", pa.string()),
    ("shape", pa.list_(pa.int32())),
    ("data", pa.binary()),
    ("mask_shape", pa.list_(pa.int32())),
    ("mask_data", pa.binary()),
])


@dataclass(frozen=True)
class CropRow:
    src_path: str
    src_mtime: float
    src_size: int
    src_xxhash: str
    bucket_idx: int
    bucket_w: int
    bucket_h: int
    crop_jpeg: bytes


@dataclass(frozen=True)
class LatentRow:
    src_path: str
    src_xxhash: str
    vae_fp: str
    bucket_idx: int
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True)
class TextEmbedRow:
    caption_xxhash: str
    caption: str
    src_path: str
    te_fp: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    mask_shape: tuple[int, ...]
    mask_data: bytes


# ------- store ---------------------------------------------------------------


class Cache:
    """Thin wrapper over a single LanceDB database with three tables.

    Tables are created lazily. Lookups are by primary-key string; writes use
    delete-then-insert to keep semantics simple. LanceDB transactions are
    per-table, which is fine here — we never need cross-table atomicity.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = lancedb.connect(self.path)
        self._t_crops = self._open_or_create("crops", CROPS_SCHEMA)
        self._t_latents = self._open_or_create("latents", LATENTS_SCHEMA)
        self._t_text = self._open_or_create("text_embeds", TEXT_EMBEDS_SCHEMA)

    def _open_or_create(self, name: str, schema: pa.Schema):
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    # ---- crops --------------------------------------------------------------

    def get_crop(self, src_path: str, *, source_file: Optional[str] = None) -> Optional[CropRow]:
        rows = (
            self._t_crops.search()
            .where(f"src_path = '{_q(src_path)}'", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        if source_file is not None and not fingerprint_matches(
            source_file, r["src_mtime"], r["src_size"], r["src_xxhash"]
        ):
            return None
        return CropRow(
            src_path=r["src_path"],
            src_mtime=r["src_mtime"],
            src_size=r["src_size"],
            src_xxhash=r["src_xxhash"],
            bucket_idx=r["bucket_idx"],
            bucket_w=r["bucket_w"],
            bucket_h=r["bucket_h"],
            crop_jpeg=bytes(r["crop_jpeg"]),
        )

    def put_crop(self, row: CropRow) -> None:
        self._t_crops.delete(f"src_path = '{_q(row.src_path)}'")
        self._t_crops.add([_crop_to_dict(row)])

    # ---- latents ------------------------------------------------------------

    def get_latent(self, src_path: str, *, vae_fp: str, source_file: Optional[str] = None) -> Optional[LatentRow]:
        rows = (
            self._t_latents.search()
            .where(f"src_path = '{_q(src_path)}'", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        if r["vae_fp"] != vae_fp:
            return None
        if source_file is not None:
            cur_hash = r["src_xxhash"]
            try:
                _, _, h = file_fingerprint(source_file)
            except FileNotFoundError:
                return None
            if h != cur_hash:
                return None
        return LatentRow(
            src_path=r["src_path"],
            src_xxhash=r["src_xxhash"],
            vae_fp=r["vae_fp"],
            bucket_idx=r["bucket_idx"],
            dtype=r["dtype"],
            shape=tuple(r["shape"]),
            data=bytes(r["data"]),
        )

    def put_latent(self, row: LatentRow) -> None:
        self._t_latents.delete(f"src_path = '{_q(row.src_path)}'")
        self._t_latents.add([_latent_to_dict(row)])

    # ---- text embeds --------------------------------------------------------

    def get_text(self, caption: str, *, te_fp: str) -> Optional[TextEmbedRow]:
        key = text_digest(caption)
        rows = (
            self._t_text.search()
            .where(f"caption_xxhash = '{_q(key)}'", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        if r["te_fp"] != te_fp:
            return None
        return TextEmbedRow(
            caption_xxhash=r["caption_xxhash"],
            caption=r["caption"],
            src_path=r["src_path"],
            te_fp=r["te_fp"],
            dtype=r["dtype"],
            shape=tuple(r["shape"]),
            data=bytes(r["data"]),
            mask_shape=tuple(r["mask_shape"]),
            mask_data=bytes(r["mask_data"]),
        )

    def put_text(self, row: TextEmbedRow) -> None:
        self._t_text.delete(f"caption_xxhash = '{_q(row.caption_xxhash)}'")
        self._t_text.add([_text_to_dict(row)])


# ------- helpers -------------------------------------------------------------


def _q(s: str) -> str:
    return s.replace("'", "''")


def _crop_to_dict(r: CropRow) -> dict:
    return {
        "src_path": r.src_path,
        "src_mtime": r.src_mtime,
        "src_size": r.src_size,
        "src_xxhash": r.src_xxhash,
        "bucket_idx": r.bucket_idx,
        "bucket_w": r.bucket_w,
        "bucket_h": r.bucket_h,
        "crop_jpeg": r.crop_jpeg,
    }


def _latent_to_dict(r: LatentRow) -> dict:
    return {
        "src_path": r.src_path,
        "src_xxhash": r.src_xxhash,
        "vae_fp": r.vae_fp,
        "bucket_idx": r.bucket_idx,
        "dtype": r.dtype,
        "shape": list(r.shape),
        "data": r.data,
    }


def _text_to_dict(r: TextEmbedRow) -> dict:
    return {
        "caption_xxhash": r.caption_xxhash,
        "caption": r.caption,
        "src_path": r.src_path,
        "te_fp": r.te_fp,
        "dtype": r.dtype,
        "shape": list(r.shape),
        "data": r.data,
        "mask_shape": list(r.mask_shape),
        "mask_data": r.mask_data,
    }


# ------- (de)serialization for numpy/torch tensors as raw bytes -------------


def tensor_to_blob(arr) -> tuple[str, tuple[int, ...], bytes]:
    """Serialize a torch tensor or numpy array as (dtype_str, shape, raw_bytes)."""
    if hasattr(arr, "detach"):  # torch.Tensor
        arr = arr.detach().contiguous().cpu().numpy()
    arr = np.ascontiguousarray(arr)
    return str(arr.dtype), tuple(arr.shape), arr.tobytes()


def blob_to_numpy(dtype: str, shape: tuple[int, ...], data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
