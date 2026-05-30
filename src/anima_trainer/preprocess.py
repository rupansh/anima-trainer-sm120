"""Image preprocessing: smartcrop into a fixed bucket and re-encode as JPEG.

Equivalent to crop.py but operating in-memory and writing to our LanceDB cache
instead of overwriting on disk. Bucket selection mirrors crop.py:closest().
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from PIL import Image
import smartcrop

from .buckets import BucketChoice, buckets_for, pick_bucket
from .hashing import file_fingerprint
from .cache import Cache, CropRow


_CROPPER = smartcrop.SmartCrop()


def _resize_then_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize keeping aspect ratio, then smart-crop to (target_w, target_h)."""
    src_ar = img.size[0] / img.size[1]
    tgt_ar = target_w / target_h
    if src_ar > tgt_ar:
        img = img.resize((target_w, int(target_w / src_ar)), Image.LANCZOS)
        # Wait — we need the shorter side to match. Re-do per crop.py logic:
    # Mirror crop.py:16 exactly:
    if tgt_ar > src_ar:
        img = img.resize((target_w, int(target_w * img.size[1] / img.size[0])), Image.LANCZOS)
    elif tgt_ar < src_ar:
        img = img.resize((int(target_h * img.size[0] / img.size[1]), target_h), Image.LANCZOS)
    else:
        return img.resize((target_w, target_h), Image.LANCZOS)
    res = _CROPPER.crop(img, width=target_w, height=target_h)
    box = (
        res["top_crop"]["x"],
        res["top_crop"]["y"],
        res["top_crop"]["x"] + res["top_crop"]["width"],
        res["top_crop"]["y"] + res["top_crop"]["height"],
    )
    return img.crop(box)


def crop_to_bucket(image_path: str | Path, resolution: int) -> tuple[BucketChoice, Image.Image]:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    table = buckets_for(resolution)
    choice = pick_bucket(img.size[0], img.size[1], table)
    cropped = _resize_then_crop(img, choice.w, choice.h)
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    return choice, cropped


def to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def precompute_crop(cache: Cache, *, src_path: str, dataset_root: str | Path, resolution: int) -> CropRow:
    """Smartcrop a source image and store it in the crops table.

    Re-uses an existing cached row if the file fingerprint matches.
    """
    src_abs = str(Path(dataset_root) / src_path) if not Path(src_path).is_absolute() else src_path
    existing = cache.get_crop(src_path, source_file=src_abs)
    if existing is not None:
        return existing
    choice, img = crop_to_bucket(src_abs, resolution)
    jpeg = to_jpeg_bytes(img)
    mtime, size, digest = file_fingerprint(src_abs)
    row = CropRow(
        src_path=src_path,
        src_mtime=mtime,
        src_size=size,
        src_xxhash=digest,
        bucket_idx=choice.idx,
        bucket_w=choice.w,
        bucket_h=choice.h,
        crop_jpeg=jpeg,
    )
    cache.put_crop(row)
    return row
