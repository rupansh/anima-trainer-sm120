"""Quantitative sample-to-sample comparison between two trainer output directories.

Use this to verify that two runs (e.g. ours vs sd-scripts) produce
numerically-close images at matched (epoch, prompt, seed).
Reports four metrics per pair:

  - pixel MSE / PSNR — coarse pixel agreement
  - SSIM — structural similarity (channel-averaged)
  - LPIPS (alex) — perceptual distance using a pretrained network
  - VAE latent cosine — semantic agreement in the Anima VAE's latent space

Sample-file naming conventions handled:
  - sd-scripts: `<output_name>_e<epoch>_<idx>_<timestamp>_<seed>.png`
  - ours:       `e<epoch>_<idx>_<seed>.png`
"""
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from PIL import Image


# ---------- filename parsing ------------------------------------------------


@dataclass(frozen=True)
class SampleKey:
    """Identifies a sample by (epoch, prompt_index, seed) — the natural pairing key."""
    epoch: int
    idx: int
    seed: str

    def __str__(self) -> str:
        return f"e{self.epoch:06d}_{self.idx:02d}_{self.seed}"


# sd-scripts:  melted1-bench_e000001_00_20260527002751_344114142.png
_SD_RE = re.compile(r"_e(\d+)_(\d+)_\d+_([^.]+)\.png$")
# ours:        e000001_00_344114142.png
_OURS_RE = re.compile(r"^e(\d+)_(\d+)_([^.]+)\.png$")


def parse_key(path: Path) -> SampleKey | None:
    name = path.name
    m = _OURS_RE.match(name) or _SD_RE.search(name)
    if m is None:
        return None
    return SampleKey(epoch=int(m.group(1)), idx=int(m.group(2)), seed=m.group(3))


def index_dir(d: Path) -> dict[SampleKey, Path]:
    out: dict[SampleKey, Path] = {}
    for p in d.glob("*.png"):
        k = parse_key(p)
        if k is not None:
            out[k] = p
    return out


# ---------- metrics ---------------------------------------------------------


def _to_tensor_01(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Image -> (1, 3, H, W) float32 in [0, 1] on device."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pixel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


def pixel_psnr(mse: float, peak: float = 1.0) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM averaged across channels using a simple 11×11 box filter.

    Not as accurate as a Gaussian-kernel SSIM but dependency-free; close
    enough for ranking matched samples against each other.
    """
    K1, K2, L = 0.01, 0.03, 1.0
    c1, c2 = (K1 * L) ** 2, (K2 * L) ** 2
    kernel = torch.ones((1, 1, 11, 11), device=a.device) / 121.0
    # apply per-channel
    def conv(x):
        b, c, h, w = x.shape
        return torch.nn.functional.conv2d(x.reshape(b * c, 1, h, w), kernel, padding=5).reshape(b, c, h, w)

    mu_a, mu_b = conv(a), conv(b)
    mu_aa, mu_bb = conv(a * a), conv(b * b)
    mu_ab = conv(a * b)
    sigma_a2 = mu_aa - mu_a * mu_a
    sigma_b2 = mu_bb - mu_b * mu_b
    sigma_ab = mu_ab - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a2 + sigma_b2 + c2)
    return float((num / den).mean().item())


@torch.no_grad()
def lpips_metric(a01: torch.Tensor, b01: torch.Tensor, model) -> float:
    """LPIPS expects inputs in [-1, 1]."""
    a, b = a01 * 2 - 1, b01 * 2 - 1
    return float(model(a, b).item())


@torch.no_grad()
def vae_latent_cosine(a01: torch.Tensor, b01: torch.Tensor, vae) -> float:
    """Encode both images through the Anima VAE and compare in latent space.

    Anima VAE expects inputs in [-1, 1] (encode_pixels_to_latents takes 4D
    (B, C, H, W) in [-1, 1]).
    """
    a = (a01 * 2 - 1).to(next(vae.parameters()).dtype)
    b = (b01 * 2 - 1).to(next(vae.parameters()).dtype)
    za = vae.encode_pixels_to_latents(a).flatten().float()
    zb = vae.encode_pixels_to_latents(b).flatten().float()
    cos = torch.nn.functional.cosine_similarity(za, zb, dim=0)
    return float(cos.item())


# ---------- main ------------------------------------------------------------


def _print_pair_table(pair_rows: list[tuple[SampleKey, dict]]) -> None:
    print(f"{'key':<28} {'mse':>8} {'psnr':>7} {'ssim':>6} {'lpips':>7} {'lat_cos':>8}")
    for k, m in pair_rows:
        print(
            f"{str(k):<28} {m['mse']:>8.5f} {m['psnr']:>7.2f} "
            f"{m['ssim']:>6.3f} {m['lpips']:>7.4f} {m['latent_cos']:>8.4f}"
        )


def _summary(pair_rows: list[tuple[SampleKey, dict]]) -> None:
    if not pair_rows:
        return
    keys = ["mse", "psnr", "ssim", "lpips", "latent_cos"]
    print()
    print(f"{'metric':<14} {'mean':>10} {'median':>10} {'min':>10} {'max':>10}")
    for k in keys:
        vals = np.array([m[k] for _, m in pair_rows if np.isfinite(m[k])])
        if not len(vals):
            continue
        print(
            f"{k:<14} {vals.mean():>10.4f} {float(np.median(vals)):>10.4f} "
            f"{vals.min():>10.4f} {vals.max():>10.4f}"
        )


def run(dir_a: Path, dir_b: Path, *, use_lpips: bool = True, use_vae: bool = False) -> None:
    a_idx = index_dir(dir_a)
    b_idx = index_dir(dir_b)
    common = sorted(set(a_idx) & set(b_idx), key=lambda k: (k.epoch, k.idx))
    print(f"{dir_a} : {len(a_idx)} samples")
    print(f"{dir_b} : {len(b_idx)} samples")
    print(f"matched: {len(common)} pairs")
    if not common:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = None
    if use_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    vae = None
    if use_vae:
        from .sdscripts_bridge import ensure_on_path
        ensure_on_path()
        from library import qwen_image_autoencoder_kl  # type: ignore
        vae = qwen_image_autoencoder_kl.load_vae(
            "./models/qwen_image_vae.safetensors",
            device="cuda",
            disable_mmap=True,
            spatial_chunk_size=None,
            disable_cache=False,
        )
        vae.to(torch.bfloat16).eval()

    rows: list[tuple[SampleKey, dict]] = []
    for k in common:
        img_a = Image.open(a_idx[k])
        img_b = Image.open(b_idx[k])
        # Normalize sizes if they differ (only happens if buckets are different).
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size, Image.LANCZOS)
        a = _to_tensor_01(img_a, device)
        b = _to_tensor_01(img_b, device)
        mse = pixel_mse(a, b)
        m = {
            "mse": mse,
            "psnr": pixel_psnr(mse),
            "ssim": ssim_simple(a, b),
            "lpips": lpips_metric(a, b, lpips_model) if lpips_model is not None else float("nan"),
            "latent_cos": vae_latent_cosine(a, b, vae) if vae is not None else float("nan"),
        }
        rows.append((k, m))

    _print_pair_table(rows)
    _summary(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two sample directories pairwise.")
    p.add_argument("dir_a", type=Path)
    p.add_argument("dir_b", type=Path)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--vae", action="store_true",
                   help="Also compute VAE-latent cosine (slow; needs the Anima VAE)")
    args = p.parse_args(argv)
    run(args.dir_a, args.dir_b, use_lpips=not args.no_lpips, use_vae=args.vae)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
