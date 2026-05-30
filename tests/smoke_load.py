"""End-to-end smoke test: load Anima DiT + Qwen3 + VAE, attach LoKr, count params.

Usage:
    .venv/bin/python tests/smoke_load.py
"""
from __future__ import annotations
import time
import torch

from anima_trainer.model import load_all
from anima_trainer.lokr import attach_lokr, trainable_param_count
from anima_trainer.config import LokrCfg


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def main():
    t0 = time.time()
    print("loading anima stack ...")
    m = load_all(
        dit_path="./models/anima-base-v1.0.safetensors",
        qwen3_path="./models/qwen_3_06b_base.safetensors",
        vae_path="./models/qwen_image_vae.safetensors",
        dtype=torch.bfloat16,
        attn_mode="torch",
        device="cuda",
        loading_device="cpu",
    )
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  dit_fp={m.dit_fp[:12]}..  te_fp={m.te_fp[:12]}..  vae_fp={m.vae_fp[:12]}..")

    n_dit = sum(p.numel() for p in m.dit.parameters())
    n_te = sum(p.numel() for p in m.text_encoder.parameters())
    n_vae = sum(p.numel() for p in m.vae.parameters())
    print(f"  DiT params: {n_dit:,}")
    print(f"  TE  params: {n_te:,}")
    print(f"  VAE params: {n_vae:,}")

    print("attaching LoKr ...")
    net = attach_lokr(m.dit, LokrCfg(), network_dim=128, network_alpha=128.0)
    n_train = trainable_param_count(net)
    print(f"  trainable LoKr params: {n_train:,}")

    # Push DiT to GPU + dry run one forward
    print("moving DiT to cuda ...")
    m.dit.to("cuda")
    net.to("cuda")
    torch.cuda.empty_cache()
    print(f"  cuda alloc: {fmt_bytes(torch.cuda.memory_allocated())}")
    print(f"  cuda reserved: {fmt_bytes(torch.cuda.memory_reserved())}")
    print("smoke OK.")


if __name__ == "__main__":
    main()
