"""Bench the new patch stack (RoPE / FusedGatedAdd / merged AdaLN /
optional CUDA graphs).

Usage:
  python tests/bench_new_patches.py [--precision fp8|bf16|mxfp8]
                                    [--cuda-graphs]
                                    [--warmup N] [--measure M]

Loads the production training setup, runs `warmup` step iterations to
settle algorithms / CUDA-graph capture / FP8 metadata caches, then times
`measure` more steps. Reports median ms/step and peak VRAM.
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_trainer.config import load
from anima_trainer.model import load_all
from anima_trainer.lokr import attach_lokr, trainable_param_count
from anima_trainer.optim import build as build_optimizer
from anima_trainer.cache import Cache
from anima_trainer.dataset import CachedAnimaDataset, BucketBatchSampler, collate, scan_dataset
from anima_trainer.flow import noisy_input_and_target
from anima_trainer.precision import torch_dtype, autocast_for, fp8_autocast_for, quantize_dit_in_place
from anima_trainer.attention_ctx import sm120_sdpa
from anima_trainer.sdscripts_bridge import ensure_on_path
from anima_trainer.liger_patch import install as install_liger_patch
from anima_trainer.adaln_patch import install as install_adaln_patch
from anima_trainer.rope_patch import install as install_rope_patch
from anima_trainer.adaln_merge import merge_adaln_modulation
from anima_trainer.cuda_graphs import CUDAGraphRunner, make_bucket_key
from anima_trainer.tlokr import clear_timestep as clear_tlokr_timestep
from anima_trainer.tlokr import set_timestep as set_tlokr_timestep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("melted1-bench.toml"))
    ap.add_argument("--precision", choices=["bf16", "fp8", "mxfp8"], default=None)
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--no-gc", action="store_true",
                    help="disable gradient checkpointing (eats VRAM, saves recompute)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override config batch_size (useful when testing --no-gc at smaller batch)")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--measure", type=int, default=12)
    args = ap.parse_args()

    cfg = load(args.config)
    from dataclasses import replace
    train_cfg = cfg.train
    if args.precision:
        train_cfg = replace(train_cfg, precision=args.precision)
    train_cfg = replace(
        train_cfg,
        compile_mode="",
        cuda_graphs=args.cuda_graphs,
        cuda_graph_warmup_steps=3,
        gradient_checkpointing=(not args.no_gc),
        batch_size=(args.batch_size if args.batch_size else train_cfg.batch_size),
    )
    cfg = replace(cfg, train=train_cfg)

    print(f"\n=== BENCH config ===")
    print(f"  precision         : {cfg.train.precision}")
    print(f"  CUDA graphs       : {args.cuda_graphs}")
    print(f"  gradient checkpt  : {cfg.train.gradient_checkpointing}")
    print(f"  warmup steps      : {args.warmup}")
    print(f"  measured steps    : {args.measure}")

    ensure_on_path()
    install_liger_patch()
    install_adaln_patch()
    install_rope_patch()
    from library import anima_utils  # type: ignore

    torch.manual_seed(cfg.train.seed)
    device = "cuda"
    dtype = torch_dtype(cfg.train.precision)

    models = load_all(
        dit_path=cfg.paths.dit, qwen3_path=cfg.paths.qwen3, vae_path=cfg.paths.vae,
        dtype=dtype, attn_mode="torch", device=device, loading_device="cpu",
    )
    models.dit.to(device)
    n_merged = merge_adaln_modulation(models.dit)
    print(f"  merged AdaLN-mod : {n_merged} blocks")

    network = attach_lokr(models.dit, cfg.lokr, network_dim=128, network_alpha=128.0).to(device)
    print(f"  {cfg.lokr.variant} params     : {trainable_param_count(network):,}")
    if cfg.train.gradient_checkpointing and hasattr(models.dit, "enable_gradient_checkpointing"):
        models.dit.enable_gradient_checkpointing()

    if cfg.train.precision == "mxfp8":
        from anima_trainer.fp8_quant import quantize_frozen_linears, collect_lokr_wrapped_linears
        skip_set = collect_lokr_wrapped_linears(network)
        quantize_frozen_linears(models.dit, skip=skip_set)
    elif cfg.train.precision == "fp8":
        from anima_trainer.fp8_quant import (
            swap_frozen_linears_to_te,
            collect_lokr_wrapped_linears,
            patch_anima_checkpoint_for_fp8,
            quantize_tlokr_base_weights,
        )
        skip_set = collect_lokr_wrapped_linears(network)
        swap_frozen_linears_to_te(models.dit, skip=skip_set)
        if cfg.lokr.variant == "tlokr":
            quantize_tlokr_base_weights(network)
        else:
            from anima_trainer.lokr_patch import enable_fp8 as enable_lokr_fp8
            enable_lokr_fp8(network)
        if cfg.train.gradient_checkpointing:
            patch_anima_checkpoint_for_fp8()
    quantize_dit_in_place(models.dit, cfg.train.precision)

    optim = build_optimizer(network.parameters(), d0=cfg.optim.d0)
    optim.train()
    t5_tok = anima_utils.load_t5_tokenizer(None)

    cache = Cache(cfg.paths.cache_db)
    samples = scan_dataset(cfg.paths.train_data_dir)
    enriched = []
    for s in samples:
        crow = cache.get_crop(s.src_path)
        enriched.append(s.__class__(src_path=s.src_path, bucket_idx=crow.bucket_idx, caption=s.caption))
    ds = CachedAnimaDataset(enriched, cache_db_path=cfg.paths.cache_db, vae_fp=models.vae_fp, te_fp=models.te_fp)
    sampler = BucketBatchSampler(enriched, cfg.train.batch_size, drop_last=True, seed=cfg.train.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=2)

    def _t5(caps):
        enc = t5_tok(caps, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        return enc["input_ids"].to(device, dtype=torch.long), enc["attention_mask"].to(device)

    def _forward_and_loss(static_batch):
        if cfg.lokr.variant == "tlokr":
            set_tlokr_timestep(static_batch["timesteps"])
        with autocast_for(cfg.train.precision), sm120_sdpa(), fp8_autocast_for(cfg.train.precision):
            pred = models.dit(
                static_batch["noisy"], static_batch["timesteps"], static_batch["prompt_embeds"],
                padding_mask=static_batch["padding_mask"],
                target_input_ids=static_batch["t5_ids"],
                target_attention_mask=static_batch["t5_mask"],
                source_attention_mask=static_batch["qwen3_mask"],
            )
        pred = pred.squeeze(2)
        return torch.nn.functional.mse_loss(pred.float(), static_batch["target"].float())

    runner = None
    if cfg.train.cuda_graphs:
        runner = CUDAGraphRunner(_forward_and_loss, warmup_steps=cfg.train.cuda_graph_warmup_steps)
    use_set_to_none = runner is None

    torch.cuda.reset_peak_memory_stats()
    times_ms = []
    step = 0

    # Loop over the dataloader; collect step times. We need enough batches
    # for warmup + measure. Loop the loader if necessary.
    iterator = iter(loader)
    while step < args.warmup + args.measure:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        latents = batch["latent"].to(device, dtype=dtype, non_blocking=True)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype, non_blocking=True)
        qwen3_mask = batch["qwen3_attn_mask"].to(device, non_blocking=True)
        t5_ids, t5_mask = _t5(batch["caption"])
        noisy, target, timesteps, _ = noisy_input_and_target(latents)
        noisy5 = noisy.unsqueeze(2)
        bs, _, h, w = latents.shape
        padding_mask = torch.zeros(bs, 1, h, w, dtype=dtype, device=device)
        static_batch = dict(
            latent=latents, prompt_embeds=prompt_embeds, qwen3_mask=qwen3_mask,
            t5_ids=t5_ids, t5_mask=t5_mask, noisy=noisy5, target=target,
            timesteps=timesteps, padding_mask=padding_mask,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optim.zero_grad(set_to_none=use_set_to_none)
        try:
            if runner is not None:
                key = make_bucket_key(latents, t5_ids)
                loss = runner.step(key, static_batch, list(network.parameters()))
            else:
                loss = _forward_and_loss(static_batch)
                loss.backward()
        finally:
            if cfg.lokr.variant == "tlokr":
                clear_tlokr_timestep()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optim.step()
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3

        step += 1
        if step <= args.warmup:
            print(f"  warmup {step}/{args.warmup}: {dt_ms:.1f} ms  loss={loss.item():.4f}")
        else:
            times_ms.append(dt_ms)
            print(f"  measure {step-args.warmup}/{args.measure}: {dt_ms:.1f} ms  loss={loss.item():.4f}")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    print("\n=== RESULTS ===")
    print(f"  median step       : {statistics.median(times_ms):.1f} ms")
    print(f"  mean step         : {statistics.mean(times_ms):.1f} ms")
    print(f"  min / max         : {min(times_ms):.1f} / {max(times_ms):.1f} ms")
    print(f"  peak allocated    : {peak_gb:.2f} GB")
    print(f"  peak reserved     : {reserved_gb:.2f} GB")


if __name__ == "__main__":
    main()
