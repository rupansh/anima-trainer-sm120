"""Convergence metric: how much has the LoRA learned the training distribution?

Procedure:
  1. Pick N random samples from the training set (caption + cached latent).
  2. For each, sample via euler_a with the trained LoRA at the bucket
     resolution of that training image's latent.
  3. Compute cosine similarity between the generated latent (z_pred) and the
     cached training latent (z_target) in the Anima VAE latent space.

The absolute number is informative but not directly interpretable — captions
underdetermine the image (one caption can match many valid images). What we
care about is the **trend across epochs**: a LoRA that's learning the dataset
distribution should show rising cosine over training.

Run as:
    anima-train convergence <config.toml> <lora.safetensors> [--n 8] [--seed 12345]
"""
from __future__ import annotations
from pathlib import Path
import random
from typing import Optional
import torch

from .config import Config
from .model import load_all
from .lokr import attach_lokr
from .cache import Cache
from .dataset import scan_dataset
from .encode import latent_to_tensor
from .sample import euler_denoise, _encode_via_strategy, _apply_llm_adapter
from .precision import torch_dtype, quantize_dit_in_place
from .attention_ctx import sm120_sdpa
from .sdscripts_bridge import ensure_on_path


def _bucket_dims_to_pixels(latent_shape: tuple[int, ...]) -> tuple[int, int]:
    """(C=16, H_lat, W_lat) -> (W_pix, H_pix). VAE spatial downscale = 8."""
    _, h, w = latent_shape
    return w * 8, h * 8


@torch.no_grad()
def measure(
    cfg: Config,
    lora_path: Path,
    *,
    n_samples: int = 8,
    seed: int = 12345,
    steps: int = 30,
    flow_shift: float = 3.0,
    cfg_scale: float = 1.0,
) -> dict:
    """Run the convergence metric on a single LoRA checkpoint.

    Returns a dict with per-sample cosines and summary stats.
    """
    ensure_on_path()
    from library import strategy_anima  # type: ignore

    device = torch.device("cuda")
    dtype = torch_dtype(cfg.train.precision)

    # Load models (frozen DiT + LoKr applied below, TE + VAE move to GPU briefly).
    models = load_all(
        dit_path=cfg.paths.dit,
        qwen3_path=cfg.paths.qwen3,
        vae_path=cfg.paths.vae,
        dtype=dtype,
        attn_mode="torch",
        device="cuda",
        loading_device="cpu",
    )
    models.dit.to(device)
    models.text_encoder.to(device)
    # VAE not needed here — we compare in cached latent space directly.

    network = attach_lokr(models.dit, cfg.lokr, network_dim=128, network_alpha=128.0).to(device)
    # Load the trained LoRA weights into the network.
    from safetensors.torch import load_file
    sd = load_file(str(lora_path))
    # Convert any bf16 keys to float and load
    network.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    network.eval()

    # Quantize frozen DiT linears if configured (must happen after LoKr attach).
    quantize_dit_in_place(models.dit, cfg.train.precision)
    models.dit.eval()

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=models.tokenizer,
        t5_tokenizer=None,
        qwen3_path=cfg.paths.qwen3,
        t5_tokenizer_path=None,
    )
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    # Pull random training samples (unique captions, with cached latent).
    cache = Cache(cfg.paths.cache_db)
    samples = scan_dataset(cfg.paths.train_data_dir)
    unique = {s.src_path: s for s in samples}
    rng = random.Random(seed)
    chosen = rng.sample(list(unique.values()), k=min(n_samples, len(unique)))

    rows: list[tuple[str, float, tuple[int, int]]] = []
    with sm120_sdpa():
        for s in chosen:
            lrow = cache.get_latent(s.src_path, vae_fp=models.vae_fp)
            if lrow is None:
                raise RuntimeError(f"no cached latent for {s.src_path}; run precompute first")
            target = latent_to_tensor(lrow).to(device=device, dtype=torch.float32)  # (16, H, W)

            # Encode the caption + LLM adapter.
            embeds, attn, t5_ids, t5_mask = _encode_via_strategy(
                s.caption,
                tokenize_strategy=tokenize_strategy,
                encoding_strategy=encoding_strategy,
                text_encoder=models.text_encoder,
                device=device,
                dtype=dtype,
            )
            cross = _apply_llm_adapter(models.dit, embeds, attn, t5_ids, t5_mask)

            # Sample at the bucket's pixel resolution. Use a fixed seed per
            # sample so noise is reproducible but differs across samples.
            w_pix, h_pix = _bucket_dims_to_pixels(target.shape)
            latents = euler_denoise(
                models.dit,
                cross,
                width=w_pix,
                height=h_pix,
                steps=steps,
                flow_shift=flow_shift,
                cfg=cfg_scale,
                neg_crossattn_emb=None,
                seed=seed + hash(s.src_path) % (1 << 30),
                device=device,
                dtype=dtype,
            )
            # latents: (1, 16, 1, H, W) -> (16, H, W)
            pred = latents.squeeze(0).squeeze(1).to(torch.float32)

            cos = torch.nn.functional.cosine_similarity(
                pred.flatten(), target.flatten(), dim=0
            ).item()
            rows.append((s.src_path, float(cos), (h_pix, w_pix)))

    cos_vals = [c for _, c, _ in rows]
    return {
        "per_sample": rows,
        "n": len(rows),
        "mean": sum(cos_vals) / len(cos_vals),
        "min": min(cos_vals),
        "max": max(cos_vals),
        "median": sorted(cos_vals)[len(cos_vals) // 2],
    }


def run_cli(cfg: Config, lora_path: Path, *, n_samples: int, seed: int) -> None:
    print(f"loading {lora_path.name} ...")
    out = measure(cfg, lora_path, n_samples=n_samples, seed=seed)
    print()
    print(f"{'src_path':<60} {'res':>10} {'lat_cos':>9}")
    for path, cos, (h, w) in out["per_sample"]:
        short = path if len(path) <= 60 else "..." + path[-57:]
        print(f"{short:<60} {h:>4}x{w:<4} {cos:>9.4f}")
    print()
    print(
        f"convergence summary (n={out['n']}):  "
        f"mean={out['mean']:.4f}  median={out['median']:.4f}  "
        f"min={out['min']:.4f}  max={out['max']:.4f}"
    )
