"""Numerical parity check + step-time bench for the AdaLN compile patch.

Single process (don't churn the allocator and risk Xid 175 again):
  1. Load DiT once.
  2. Attach LoKr (cross-mlp) + install Liger RMSNorm.
  3. Run 3 warmup + 3 timed steps WITHOUT adaln_patch — record losses + time.
  4. install_adaln_patch().
  5. Run 1 warmup (lets inductor compile) + 3 timed steps — record losses + time.
  6. Compare.

If the patched losses match unpatched losses to bf16 noise, the patch is
correct. Then compare median step times.

Run: `python -m tests.sanity_adaln_patch`
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.config import load
from anima_trainer.model import load_all
from anima_trainer.lokr import attach_lokr
from anima_trainer.optim import build as build_optimizer
from anima_trainer.flow import noisy_input_and_target
from anima_trainer.precision import torch_dtype, autocast_for
from anima_trainer.attention_ctx import sm120_sdpa
from anima_trainer.sdscripts_bridge import ensure_on_path


def main() -> None:
    cfg = load(Path("melted1.toml"))
    from dataclasses import replace
    cfg = replace(cfg, train=replace(cfg.train, compile_mode=""))

    ensure_on_path()
    # Install Liger first (production path) so we measure the additional adaln
    # contribution on top of it.
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
    if cfg.train.gradient_checkpointing and hasattr(models.dit, "enable_gradient_checkpointing"):
        models.dit.enable_gradient_checkpointing()
    optim = build_optimizer(network.parameters(), d0=cfg.optim.d0)
    optim.train()
    t5_tok = anima_utils.load_t5_tokenizer(None)

    B = cfg.train.batch_size  # 8
    H = W = 128
    # IMPORTANT: keep the same RNG state across the two configurations so the
    # losses are directly comparable, not just close in distribution.
    torch.manual_seed(0)
    latents_ref = torch.randn(B, 16, H, W, device=device, dtype=dtype)
    prompt_embeds = torch.randn(B, 512, 1024, device=device, dtype=dtype)
    qwen3_mask = torch.ones(B, 512, device=device, dtype=torch.long)
    captions = ["a placeholder caption"] * B
    enc = t5_tok(captions, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    t5_ids = enc["input_ids"].to(device, dtype=torch.long)
    t5_mask = enc["attention_mask"].to(device)
    padding_mask_ref = torch.zeros(B, 1, H, W, dtype=dtype, device=device)

    def step(seed: int):
        # Re-derive noisy/target/timesteps deterministically per seed so the
        # before/after losses are head-to-head comparable.
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latents_ref.shape, generator=gen, device=device, dtype=dtype)
        # noisy_input_and_target reads from torch.* RNG; bypass it for parity.
        sigmas = torch.full((B,), 0.5, device=device, dtype=dtype)
        sigmas_b = sigmas[:, None, None, None]
        noisy = (1 - sigmas_b) * latents_ref + sigmas_b * noise
        target = noise - latents_ref
        timesteps = sigmas.float()
        noisy5 = noisy.unsqueeze(2)
        with autocast_for(cfg.train.precision), sm120_sdpa():
            pred = models.dit(noisy5, timesteps, prompt_embeds, padding_mask=padding_mask_ref,
                              target_input_ids=t5_ids, target_attention_mask=t5_mask,
                              source_attention_mask=qwen3_mask)
        pred = pred.squeeze(2)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optim.step()
        return float(loss.item())

    print("=== WITHOUT adaln_patch (baseline: cross-mlp + lokr_patch + Liger) ===")
    # warmup
    for i in range(3):
        l = step(seed=100 + i)
        torch.cuda.synchronize()
    # timed
    losses_a = []
    times_a = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        l = step(seed=200 + i)
        torch.cuda.synchronize()
        dt = time.time() - t0
        losses_a.append(l)
        times_a.append(dt)
        print(f"  step {i}: loss={l:.6f}  dt={dt:.3f}s")

    print("\n=== WITH adaln_patch ===")
    from anima_trainer.adaln_patch import install as install_adaln
    install_adaln()
    # First call with the patch will trigger inductor compile on this shape;
    # don't time it.
    print("  (compile warmup — this will be slow)")
    for i in range(2):
        l = step(seed=300 + i)
        torch.cuda.synchronize()
    losses_b = []
    times_b = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        l = step(seed=200 + i)  # same seed as losses_a → directly comparable
        torch.cuda.synchronize()
        dt = time.time() - t0
        losses_b.append(l)
        times_b.append(dt)
        print(f"  step {i}: loss={l:.6f}  dt={dt:.3f}s")

    print("\n=== Comparison ===")
    for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
        print(f"  seed={200+i}: baseline loss={la:.6f}  patched={lb:.6f}  diff={abs(la-lb):.2e}")
    ma, mb = sorted(times_a)[len(times_a)//2], sorted(times_b)[len(times_b)//2]
    print(f"\n  median step time: baseline={ma:.3f}s  patched={mb:.3f}s  speedup={ma/mb:.3f}×")
    print(f"  max loss diff: {max(abs(a-b) for a,b in zip(losses_a, losses_b)):.2e}")


if __name__ == "__main__":
    main()
