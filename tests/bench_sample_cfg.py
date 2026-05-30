"""Bench batched-vs-sequential CFG inside the Euler sampler.

Loads the production DiT (+ liger/adaln patches), generates dummy cross-attn
embeds matching the real shapes, and times both code paths under the same
torch.no_grad / fp8_autocast context that sample.euler_denoise uses.

Run:
    python tests/bench_sample_cfg.py
"""
from __future__ import annotations
import time
import torch

from anima_trainer.sdscripts_bridge import ensure_on_path
ensure_on_path()

from anima_trainer.model import load_all
from anima_trainer.attention_ctx import sm120_sdpa
from anima_trainer.liger_patch import install as install_liger_patch
from anima_trainer.adaln_patch import install as install_adaln_patch
from anima_trainer.precision import fp8_autocast_for


PRECISION = "bf16"
DTYPE = torch.bfloat16
WIDTH = 512
HEIGHT = 512
STEPS = 20
CFG = 5.0
FLOW_SHIFT = 3.0
WARMUP = 2
ITERS = 4


@torch.no_grad()
def _euler_sequential(dit, ce_pos, ce_neg, *, padding_mask, sigmas, cfg, dtype):
    """Reference: two batch-1 forwards per step (the pre-change behaviour)."""
    latent_h, latent_w = padding_mask.shape[-2:]
    x = torch.randn((1, 16, 1, latent_h, latent_w), dtype=dtype, device=padding_mask.device)
    for i in range(STEPS):
        t = sigmas[i].unsqueeze(0)
        with fp8_autocast_for(PRECISION):
            pos = dit(x, t, ce_pos, padding_mask=padding_mask).float()
            neg = dit(x, t, ce_neg, padding_mask=padding_mask).float()
        model_out = neg + cfg * (pos - neg)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + (model_out * dt).to(dtype)
    return x


@torch.no_grad()
def _euler_batched(dit, ce_pos, ce_neg, *, padding_mask, sigmas, cfg, dtype):
    """Current: one batch-2 forward per step."""
    latent_h, latent_w = padding_mask.shape[-2:]
    x = torch.randn((1, 16, 1, latent_h, latent_w), dtype=dtype, device=padding_mask.device)
    ce_cat = torch.cat([ce_pos, ce_neg], dim=0)
    pad_cat = padding_mask.expand(2, -1, -1, -1)
    for i in range(STEPS):
        with fp8_autocast_for(PRECISION):
            x_in = torch.cat([x, x], dim=0)
            t_in = sigmas[i].expand(2)
            out = dit(x_in, t_in, ce_cat, padding_mask=pad_cat).float()
        pos, neg = out[0:1], out[1:2]
        model_out = neg + cfg * (pos - neg)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + (model_out * dt).to(dtype)
    return x


def _time(fn, *args, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args, **kw)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    install_liger_patch()
    install_adaln_patch()

    device = torch.device("cuda")
    models = load_all(
        dit_path="./models/anima-base-v1.0.safetensors",
        qwen3_path="./models/qwen_3_06b_base.safetensors",
        vae_path="./models/qwen_image_vae.safetensors",
        dtype=DTYPE,
        attn_mode="torch",
        device="cuda",
        loading_device="cpu",
    )
    dit = models.dit.to(device).eval()

    # Probe cross-attn dim from the first block to build dummy embeds.
    first_block = dit.blocks[0] if hasattr(dit, "blocks") else dit.transformer_blocks[0]
    ctx_dim = first_block.cross_attn.k_proj.in_features
    seq_len = 512  # AnimaTokenizeStrategy.t5_max_length

    ce_pos = torch.randn((1, seq_len, ctx_dim), dtype=DTYPE, device=device)
    ce_neg = torch.randn((1, seq_len, ctx_dim), dtype=DTYPE, device=device)
    lat_h, lat_w = HEIGHT // 8, WIDTH // 8
    padding_mask = torch.zeros((1, 1, lat_h, lat_w), dtype=DTYPE, device=device)

    sigmas = torch.linspace(1.0, 0.0, STEPS + 1, device=device, dtype=DTYPE)
    if FLOW_SHIFT != 1.0:
        sigmas = (sigmas * FLOW_SHIFT) / (1 + (FLOW_SHIFT - 1) * sigmas)

    print(f"resolution={WIDTH}x{HEIGHT}  steps={STEPS}  cfg={CFG}  precision={PRECISION}")
    print(f"crossattn dim={ctx_dim}  seq_len={seq_len}")

    with sm120_sdpa():
        # warmup each path
        for _ in range(WARMUP):
            _euler_sequential(dit, ce_pos, ce_neg, padding_mask=padding_mask, sigmas=sigmas, cfg=CFG, dtype=DTYPE)
            _euler_batched(dit, ce_pos, ce_neg, padding_mask=padding_mask, sigmas=sigmas, cfg=CFG, dtype=DTYPE)

        seq_times = [
            _time(_euler_sequential, dit, ce_pos, ce_neg, padding_mask=padding_mask, sigmas=sigmas, cfg=CFG, dtype=DTYPE)
            for _ in range(ITERS)
        ]
        bat_times = [
            _time(_euler_batched, dit, ce_pos, ce_neg, padding_mask=padding_mask, sigmas=sigmas, cfg=CFG, dtype=DTYPE)
            for _ in range(ITERS)
        ]

    seq_mean = sum(seq_times) / len(seq_times)
    bat_mean = sum(bat_times) / len(bat_times)
    print()
    print(f"sequential CFG ({STEPS} steps): {seq_mean:.3f} s  (per step {seq_mean / STEPS * 1000:.1f} ms)  raw={[f'{t:.3f}' for t in seq_times]}")
    print(f"batched    CFG ({STEPS} steps): {bat_mean:.3f} s  (per step {bat_mean / STEPS * 1000:.1f} ms)  raw={[f'{t:.3f}' for t in bat_times]}")
    print(f"speedup: {seq_mean / bat_mean:.2f}x  ({(1 - bat_mean / seq_mean) * 100:.1f}% wall-clock reduction)")


if __name__ == "__main__":
    main()
