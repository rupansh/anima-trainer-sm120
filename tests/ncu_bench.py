"""Minimal production-shape step for nsight-compute profiling.

Runs a few warmup steps, then a single timed step with no instrumentation, so
ncu can profile clean kernel launches without torch.profiler overhead.

Use synthetic random tensors at production shapes so we don't need the cache
or the dataloader. Production shapes for melted1: B=8, latent (8,16,128,128),
prompt_embeds (8, 512, 1024), bf16. Same modules wrapped as production
(anima-cross-mlp + lokr_patch + Liger RMSNorm).

Run:
    source .venv/bin/activate
    ncu --set basic --launch-skip 50 --launch-count 30 \\
        --kernel-name 'regex:cutlass|flash|rms_norm|elementwise|layer_norm' \\
        --target-processes all --export /tmp/anima_ncu \\
        python -m tests.ncu_bench

The --launch-skip 50 skips Python startup/init kernels and lands inside the
timed step. --launch-count 30 keeps wall-clock manageable (ncu adds ~50×
slowdown per profiled kernel).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.config import load
from anima_trainer.model import load_all
from anima_trainer.lokr import attach_lokr, trainable_param_count
from anima_trainer.optim import build as build_optimizer
from anima_trainer.flow import noisy_input_and_target
from anima_trainer.precision import torch_dtype, autocast_for
from anima_trainer.attention_ctx import sm120_sdpa
from anima_trainer.sdscripts_bridge import ensure_on_path


def main():
    cfg = load(Path("melted1.toml"))
    from dataclasses import replace
    cfg = replace(cfg, train=replace(cfg.train, compile_mode=""))

    ensure_on_path()
    from anima_trainer.liger_patch import install as install_liger
    install_liger()
    from library import anima_utils  # noqa

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch_dtype(cfg.train.precision)

    models = load_all(
        dit_path=cfg.paths.dit, qwen3_path=cfg.paths.qwen3, vae_path=cfg.paths.vae,
        dtype=dtype, attn_mode="torch", device=device, loading_device="cpu",
    )
    models.dit.to(device)

    network = attach_lokr(models.dit, cfg.lokr, network_dim=128, network_alpha=128.0).to(device)
    print(f"trainable LoKr params: {trainable_param_count(network):,}")
    if cfg.train.gradient_checkpointing and hasattr(models.dit, "enable_gradient_checkpointing"):
        models.dit.enable_gradient_checkpointing()
    optim = build_optimizer(network.parameters(), d0=cfg.optim.d0)
    optim.train()
    t5_tok = anima_utils.load_t5_tokenizer(None)

    # Synthetic production-shape batch.
    B = cfg.train.batch_size  # 8
    H = W = 128                # 1024/8 (VAE downscale + patch step)
    latents = torch.randn(B, 16, H, W, device=device, dtype=dtype)
    prompt_embeds = torch.randn(B, 512, 1024, device=device, dtype=dtype)
    qwen3_mask = torch.ones(B, 512, device=device, dtype=torch.long)
    captions = ["a placeholder caption"] * B
    enc = t5_tok(captions, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    t5_ids = enc["input_ids"].to(device, dtype=torch.long)
    t5_mask = enc["attention_mask"].to(device)

    def step():
        noisy, target, timesteps, _ = noisy_input_and_target(latents)
        noisy5 = noisy.unsqueeze(2)
        bs, _, h, w = latents.shape
        padding_mask = torch.zeros(bs, 1, h, w, dtype=dtype, device=device)
        with autocast_for(cfg.train.precision), sm120_sdpa():
            pred = models.dit(noisy5, timesteps, prompt_embeds, padding_mask=padding_mask,
                              target_input_ids=t5_ids, target_attention_mask=t5_mask,
                              source_attention_mask=qwen3_mask)
        pred = pred.squeeze(2)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optim.step()
        return float(loss.item())

    # 3 warmup steps for allocator + Prodigy state.
    for i in range(3):
        l = step()
        torch.cuda.synchronize()
        print(f"warmup step {i}: loss={l:.4f}")

    # NVTX range so ncu can target this region with --nvtx-include "anima_step/".
    # Outside the range, kernels run uninstrumented (full speed).
    torch.cuda.synchronize()
    t0 = time.time()
    torch.cuda.nvtx.range_push("anima_step")
    l = step()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"timed step: loss={l:.4f}  dt={time.time()-t0:.3f}s")


if __name__ == "__main__":
    main()
