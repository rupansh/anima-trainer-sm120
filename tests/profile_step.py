"""Profile a single training step to find the real hot path with the wide LoRA.

We've been guessing where the 5.29 s/step time goes. This script captures one
full forward+backward+optimizer step under `torch.profiler` and prints the top
kernels + Python frames by CUDA time.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.profiler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.config import load
from anima_trainer.model import load_all
from anima_trainer.lokr import attach_lokr, trainable_param_count
from anima_trainer.optim import build as build_optimizer
from anima_trainer.cache import Cache
from anima_trainer.dataset import CachedAnimaDataset, BucketBatchSampler, collate, scan_dataset
from anima_trainer.flow import noisy_input_and_target
from anima_trainer.precision import torch_dtype, autocast_for
from anima_trainer.attention_ctx import sm120_sdpa
from anima_trainer.sdscripts_bridge import ensure_on_path


def main():
    cfg = load(Path("melted1.toml"))
    # Force eager (no compile) for clean profiling.
    from dataclasses import replace
    cfg = replace(cfg, train=replace(cfg.train, compile_mode=""))

    ensure_on_path()
    # Install Liger RMSNorm + custom AdaLN kernel patches — production
    # train() does this, so the profile must too or the numbers will be
    # misleading.
    from anima_trainer.liger_patch import install as install_liger
    from anima_trainer.adaln_patch import install as install_adaln
    install_liger()
    install_adaln()
    from library import anima_utils, strategy_anima  # noqa

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

    cache = Cache(cfg.paths.cache_db)
    samples = scan_dataset(cfg.paths.train_data_dir)
    enriched = []
    for s in samples:
        crow = cache.get_crop(s.src_path)
        enriched.append(s.__class__(src_path=s.src_path, bucket_idx=crow.bucket_idx, caption=s.caption))
    ds = CachedAnimaDataset(enriched, cache_db_path=cfg.paths.cache_db,
                            vae_fp=models.vae_fp, te_fp=models.te_fp)
    sampler = BucketBatchSampler(enriched, cfg.train.batch_size, drop_last=False, seed=cfg.train.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)

    # Warm up + do one step to get past first-step JIT
    batch = next(iter(loader))
    latents = batch["latent"].to(device, dtype=dtype)
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype)
    qwen3_mask = batch["qwen3_attn_mask"].to(device)
    enc = t5_tok(batch["caption"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    t5_ids = enc["input_ids"].to(device, dtype=torch.long)
    t5_mask = enc["attention_mask"].to(device)

    for _ in range(2):
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
    torch.cuda.synchronize()

    # Profile two steps to amortize one-time costs
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(2):
            noisy, target, timesteps, _ = noisy_input_and_target(latents)
            noisy5 = noisy.unsqueeze(2)
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
        torch.cuda.synchronize()

    # Two profiled steps → divide each row by 2 to get per-step time.
    print("\n=== Top ops by CUDA total time (over 2 profiled steps) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=80))

    # Bucket the totals by op category so we can see where the time actually goes.
    rows = prof.key_averages()
    # torch renamed cuda_time_total → device_time_total; keep both for safety.
    def _self_us(r):
        return getattr(r, "self_device_time_total", None) or getattr(r, "self_cuda_time_total", 0)
    total_cuda_us = sum(_self_us(r) for r in rows)
    buckets = {
        "mm / addmm / scaled_mm":   ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::_scaled_mm"),
        "attention (SDPA + bwd)":    ("aten::scaled_dot_product_attention", "aten::_scaled_dot_product_", "aten::_flash_attention", "aten::_efficient_attention", "aten::_cudnn_attention"),
        "Liger RMSNorm (fwd+bwd)":   ("LigerRMSNormFunction", "_rms_norm_forward", "_rms_norm_backward", "rms_norm"),
        "layer_norm / native_layer_norm": ("aten::layer_norm", "aten::native_layer_norm"),
        "LoKr kron (fwd+bwd)":       ("aten::kron", "KronBackward", "_kron"),
        "elementwise (mul/add/copy)": ("aten::mul", "aten::add", "aten::div", "aten::copy_", "aten::clone", "aten::to_copy", "aten::_to_copy"),
        "reshape / permute / view":  ("aten::view", "aten::reshape", "aten::permute", "aten::transpose", "aten::contiguous", "aten::expand", "aten::squeeze", "aten::unsqueeze"),
        "optimizer (Prodigy step)":  ("aten::lerp", "aten::sqrt", "aten::addcmul", "aten::addcdiv", "ProdigyPlus"),
        "grad clip / norm":          ("aten::clip_grad_norm", "aten::_foreach_norm", "aten::stack", "aten::linalg_vector_norm"),
        "loss (mse + bwd)":          ("aten::mse_loss", "aten::pow"),
    }
    bucketed_us = {k: 0 for k in buckets}
    unbucketed = []
    for r in rows:
        n = r.key
        us = _self_us(r)
        assigned = False
        for cat, prefixes in buckets.items():
            if any(n.startswith(p) or p in n for p in prefixes):
                bucketed_us[cat] += us
                assigned = True
                break
        if not assigned and us > 0:
            unbucketed.append((n, us))

    print("\n=== Per-step time breakdown (over 2 profiled steps) ===")
    step_us = total_cuda_us / 2
    print(f"total CUDA self time / step: {step_us/1e3:.1f} ms")
    for cat, us in sorted(bucketed_us.items(), key=lambda kv: -kv[1]):
        pct = 100 * us / total_cuda_us if total_cuda_us else 0
        print(f"  {cat:<36} {us/2/1e3:>8.1f} ms / step  ({pct:5.1f}%)")
    other_us = sum(u for _, u in unbucketed)
    print(f"  {'other (unbucketed)':<36} {other_us/2/1e3:>8.1f} ms / step  ({100*other_us/total_cuda_us:5.1f}%)")
    print("\nTop 10 unbucketed ops by time:")
    for n, u in sorted(unbucketed, key=lambda kv: -kv[1])[:10]:
        print(f"  {n[:70]:<72} {u/2/1e3:>7.1f} ms / step")


if __name__ == "__main__":
    main()
