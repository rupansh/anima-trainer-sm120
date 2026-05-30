"""Cache-fronted VAE and Qwen3 encoding.

Both encoders are expensive and entirely deterministic in eval mode given fixed
weights, so we cache aggressively. Each cached row carries the weight-file
fingerprint of the encoder used — changing the VAE or Qwen3 weights invalidates
the corresponding cache entries automatically.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np
import torch

from .cache import Cache, LatentRow, TextEmbedRow, tensor_to_blob, blob_to_numpy
from .hashing import text_digest
from .sdscripts_bridge import ensure_on_path


def _img_bytes_to_tensor(jpeg: bytes, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(BytesIO(jpeg)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return t  # (1, 3, H, W)


@torch.no_grad()
def encode_latents(
    cache: Cache,
    vae,
    *,
    vae_fp: str,
    src_path: str,
    crop_jpeg: bytes,
    src_xxhash: str,
    bucket_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> LatentRow:
    """VAE-encode a single image and store in the latents table."""
    existing = cache.get_latent(src_path, vae_fp=vae_fp)
    if existing is not None and existing.src_xxhash == src_xxhash:
        return existing

    img = _img_bytes_to_tensor(crop_jpeg, device, dtype)
    # encode_pixels_to_latents takes 4D (B, C, H, W) in [-1, 1] and applies mean/std.
    latent = vae.encode_pixels_to_latents(img).squeeze(0)  # (C, H, W)
    dtype_str, shape, data = tensor_to_blob(latent.to(torch.float16))   # fp16 storage is plenty
    row = LatentRow(
        src_path=src_path,
        src_xxhash=src_xxhash,
        vae_fp=vae_fp,
        bucket_idx=bucket_idx,
        dtype=dtype_str,
        shape=shape,
        data=data,
    )
    cache.put_latent(row)
    return row


def latent_to_tensor(row: LatentRow, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.from_numpy(blob_to_numpy(row.dtype, row.shape, row.data).copy()).to(dtype)


# ---------- Qwen3 text encoding ---------------------------------------------


@torch.no_grad()
def encode_text(
    cache: Cache,
    *,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    te_fp: str,
    src_path: str,
    caption: str,
) -> TextEmbedRow:
    """Run Qwen3 over a caption (and store T5 token IDs alongside)."""
    existing = cache.get_text(caption, te_fp=te_fp)
    if existing is not None:
        return existing

    tokens = tokenize_strategy.tokenize(caption)
    out = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
    prompt_embeds, attn_mask, t5_ids, t5_attn = out

    # Pack prompt_embeds + attn_mask into the 'data' / 'mask_data' slots, and
    # tuck the T5 fields into 'shape'/'mask_shape' suffixes by appending them as
    # extra trailing dims is too cute — instead we serialize four blobs by
    # concatenating with length prefixes is also ugly. Simpler: use the row's
    # mask_* slot for qwen3 mask only, and recompute T5 tokens at use time
    # since T5 tokenization is cheap (just a tokenizer call).
    embeds_dtype, embeds_shape, embeds_data = tensor_to_blob(prompt_embeds.squeeze(0).to(torch.float16))
    mask_dtype, mask_shape, mask_data = tensor_to_blob(attn_mask.squeeze(0).to(torch.int8))
    assert mask_dtype == "int8"

    row = TextEmbedRow(
        caption_xxhash=text_digest(caption),
        caption=caption,
        src_path=src_path,
        te_fp=te_fp,
        dtype=embeds_dtype,
        shape=embeds_shape,
        data=embeds_data,
        mask_shape=mask_shape,
        mask_data=mask_data,
    )
    cache.put_text(row)
    return row


def text_to_tensors(row: TextEmbedRow) -> tuple[torch.Tensor, torch.Tensor]:
    embeds = torch.from_numpy(blob_to_numpy(row.dtype, row.shape, row.data).copy())
    mask = torch.from_numpy(blob_to_numpy("int8", row.mask_shape, row.mask_data).copy())
    return embeds, mask
