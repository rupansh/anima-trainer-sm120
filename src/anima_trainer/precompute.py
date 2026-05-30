"""One-shot precompute: crop, VAE-encode, and Qwen3-encode the dataset into the cache.

Idempotent: every step checks the cache first and skips if the source fingerprint
and encoder fingerprint both still match. Safe to re-run after editing captions
or replacing images.
"""
from __future__ import annotations
import time
from pathlib import Path
import torch

from .config import Config
from .model import load_all
from .preprocess import precompute_crop
from .encode import encode_latents, encode_text
from .cache import Cache
from .dataset import scan_dataset
from .sdscripts_bridge import ensure_on_path


def run(cfg: Config) -> None:
    ensure_on_path()
    from library import strategy_anima  # type: ignore

    device = "cuda"
    dtype = torch.bfloat16

    cache = Cache(cfg.paths.cache_db)
    samples = scan_dataset(cfg.paths.train_data_dir)
    # Deduplicate (the same image appears `repeats` times in the training list)
    unique_paths = {s.src_path: s for s in samples}
    print(f"scanning {cfg.paths.train_data_dir}: {len(samples)} samples ({len(unique_paths)} unique)")

    # Phase 1: smartcrop every image into its bucket
    t0 = time.time()
    for s in unique_paths.values():
        precompute_crop(cache, src_path=s.src_path, dataset_root=cfg.paths.train_data_dir, resolution=cfg.train.resolution)
    print(f"  crops done in {time.time()-t0:.1f}s")

    # Phase 2: VAE-encode each crop into latents
    print("loading VAE for latent encoding ...")
    models = load_all(
        dit_path=cfg.paths.dit,        # still loaded once, but we drop it immediately
        qwen3_path=cfg.paths.qwen3,
        vae_path=cfg.paths.vae,
        dtype=dtype,
        attn_mode="torch",
        device=device,
        loading_device="cpu",
    )
    models.dit.cpu()
    del models.dit
    torch.cuda.empty_cache()

    models.vae.to(device)
    t0 = time.time()
    for s in unique_paths.values():
        crow = cache.get_crop(s.src_path)
        if crow is None:
            raise RuntimeError(f"missing crop for {s.src_path}; this shouldn't happen")
        encode_latents(
            cache,
            models.vae,
            vae_fp=models.vae_fp,
            src_path=s.src_path,
            crop_jpeg=crow.crop_jpeg,
            src_xxhash=crow.src_xxhash,
            bucket_idx=crow.bucket_idx,
            device=torch.device(device),
            dtype=dtype,
        )
    print(f"  latents done in {time.time()-t0:.1f}s")
    models.vae.cpu()
    del models.vae
    torch.cuda.empty_cache()

    # Phase 3: Qwen3-encode every caption
    models.text_encoder.to(device)
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=models.tokenizer,
        t5_tokenizer=None,           # will load default from sd-scripts/configs
        qwen3_path=cfg.paths.qwen3,
        t5_tokenizer_path=None,
    )
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    t0 = time.time()
    for s in unique_paths.values():
        encode_text(
            cache,
            tokenize_strategy=tokenize_strategy,
            encoding_strategy=encoding_strategy,
            text_encoder=models.text_encoder,
            te_fp=models.te_fp,
            src_path=s.src_path,
            caption=s.caption,
        )
    print(f"  text embeds done in {time.time()-t0:.1f}s")
    print("precompute complete.")
