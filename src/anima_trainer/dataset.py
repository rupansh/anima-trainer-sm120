"""Dataset and bucket-aware sampler that reads from our LanceDB cache.

The cache is assumed to be precomputed (see cli.py:precompute). Training reads
only cached latents + text embeds — no image decode, no Qwen3 forward, no VAE
forward in the hot loop. Buckets are honored by the sampler: every batch is
single-bucket so the latents stack without padding.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterator, List
import torch
from torch.utils.data import Dataset, Sampler

from .cache import Cache
from .encode import latent_to_tensor, text_to_tensors


@dataclass(frozen=True)
class Sample:
    src_path: str
    bucket_idx: int
    caption: str


class CachedAnimaDataset(Dataset):
    """Maps an index -> precomputed (latent, qwen3_embeds, qwen3_mask, caption).

    Stores only the LanceDB *path*, not a live Cache handle — LanceDB
    connections wrap an asyncio runtime which doesn't survive pickling into
    DataLoader workers. Each worker (and the main process) opens its own Cache
    on first access.
    """

    def __init__(self, samples: List[Sample], cache_db_path: str, vae_fp: str, te_fp: str):
        self.samples = samples
        self._cache_db_path = cache_db_path
        self._cache: Cache | None = None
        self.vae_fp = vae_fp
        self.te_fp = te_fp

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            self._cache = Cache(self._cache_db_path)
        return self._cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        lrow = self.cache.get_latent(s.src_path, vae_fp=self.vae_fp)
        if lrow is None:
            raise RuntimeError(f"missing cached latent for {s.src_path}")
        trow = self.cache.get_text(s.caption, te_fp=self.te_fp)
        if trow is None:
            raise RuntimeError(f"missing cached text embed for caption[{s.src_path}]")
        latent = latent_to_tensor(lrow)
        embeds, mask = text_to_tensors(trow)
        return {
            "latent": latent,
            "prompt_embeds": embeds,
            "qwen3_attn_mask": mask.to(torch.bool),
            "caption": s.caption,
            "src_path": s.src_path,
            "bucket_idx": s.bucket_idx,
        }


class BucketBatchSampler(Sampler[List[int]]):
    """Yields batches of dataset indices that share a bucket.

    Buckets are filled left-to-right; the leftover (partial) batch at the end of
    each bucket is included unless `drop_last=True`. Order is shuffled per epoch.
    """

    def __init__(self, samples: List[Sample], batch_size: int, *, drop_last: bool = False, seed: int = 0):
        self.samples = samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_to_idxs: dict[int, list[int]] = {}
        for i, s in enumerate(samples):
            self.bucket_to_idxs.setdefault(s.bucket_idx, []).append(i)
        self._epoch = 0

    def set_epoch(self, e: int) -> None:
        self._epoch = e

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        all_batches: list[list[int]] = []
        for idxs in self.bucket_to_idxs.values():
            order = idxs[:]
            rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batch = order[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        n = 0
        for idxs in self.bucket_to_idxs.values():
            if self.drop_last:
                n += len(idxs) // self.batch_size
            else:
                n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n


def collate(batch: list[dict]) -> dict:
    """Stack same-bucket samples. Assumes batch is pre-filtered to one bucket."""
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "prompt_embeds": torch.stack([b["prompt_embeds"] for b in batch]),
        "qwen3_attn_mask": torch.stack([b["qwen3_attn_mask"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "src_path": [b["src_path"] for b in batch],
        "bucket_idx": batch[0]["bucket_idx"],
    }


def scan_dataset(root: str | Path) -> List[Sample]:
    """Walk an sd-scripts-style dataset directory.

    Expects subfolders named `<repeats>_<name>/` containing `<stem>.{png,jpg,jpeg}`
    and matching `<stem>.txt` caption files.
    """
    samples: list[Sample] = []
    root = Path(root)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        try:
            repeats_s, _ = sub.name.split("_", 1)
            repeats = int(repeats_s)
        except ValueError:
            continue
        # Stable file ordering is part of the resumable batch-plan contract.
        # Filesystem iteration order can change after directory edits or a
        # reboot even when the dataset contents are identical.
        for img in sorted(sub.iterdir()):
            if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            cap_path = img.with_suffix(".txt")
            if not cap_path.exists():
                continue
            caption = cap_path.read_text(encoding="utf-8").strip()
            rel = str(img.relative_to(root))
            for _ in range(repeats):
                samples.append(Sample(src_path=rel, bucket_idx=-1, caption=caption))
    return samples
