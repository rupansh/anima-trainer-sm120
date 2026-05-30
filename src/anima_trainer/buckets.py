"""Fixed bucket table for Anima training.

Derived from crop.py:12 (the 1024px set). 512px buckets are obtained by
halving each side and snapping to a multiple of 16 (the Anima VAE×patch step).
We never up-scale source images: bucket_no_upscale=True in the baseline config.
"""
from __future__ import annotations
from dataclasses import dataclass

BUCKETS_1024: list[tuple[int, int]] = [
    (1024, 1024),
    (896, 1152),
    (832, 1216),
    (768, 1344),
    (640, 1536),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
]

STEP = 16  # Anima VAE spatial downscale (8) × patch size (2)


def _snap(x: int) -> int:
    return max(STEP, (x // STEP) * STEP)


def buckets_for(resolution: int) -> list[tuple[int, int]]:
    """Return the bucket table for a target resolution (512 or 1024).

    For 1024 we return the canonical list. For 512 we scale each (w,h) by 0.5
    and snap to STEP so the aspect ratio is preserved as closely as possible.
    """
    if resolution == 1024:
        return list(BUCKETS_1024)
    if resolution == 512:
        scale = 512 / 1024
        return [(_snap(int(w * scale)), _snap(int(h * scale))) for w, h in BUCKETS_1024]
    raise ValueError(f"unsupported resolution {resolution!r}; only 512 and 1024 are allowed")


@dataclass(frozen=True)
class BucketChoice:
    idx: int
    w: int
    h: int

    @property
    def aspect(self) -> float:
        return self.w / self.h


def pick_bucket(src_w: int, src_h: int, table: list[tuple[int, int]]) -> BucketChoice:
    """Choose the bucket whose aspect ratio is closest to the source.

    Mirrors crop.py:35 `closest()`.
    """
    src_ar = src_w / src_h
    best = min(range(len(table)), key=lambda i: abs(table[i][0] / table[i][1] - src_ar))
    w, h = table[best]
    return BucketChoice(idx=best, w=w, h=h)
