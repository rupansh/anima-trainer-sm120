"""Training loop: bf16 rectified-flow over the Anima DiT with LoKr deltas + Prodigy+SF.

Lifecycle:
  1. Load models (DiT + Qwen3 + VAE); freeze all base weights.
  2. Attach LoKr; only adapter params are trainable.
  3. Iterate cached (latent, prompt_embeds, qwen3_mask) batches from LanceDB.
  4. Forward / backward / step. Save adapter weights every N epochs.
  5. Sample every N epochs by reloading the hot prompt file.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config
from .model import load_all
from .lokr import attach_lokr, trainable_param_count
from .optim import build as build_optimizer
from .cache import Cache
from .dataset import CachedAnimaDataset, BucketBatchSampler, collate, scan_dataset
from .flow import noisy_input_and_target
from .precision import torch_dtype, autocast_for, fp8_autocast_for, quantize_dit_in_place
from .sample import sample_all_prompts
from .attention_ctx import sm120_sdpa
from .sdscripts_bridge import ensure_on_path
from .liger_patch import install as install_liger_patch
from .adaln_patch import install as install_adaln_patch
from .rope_patch import install as install_rope_patch
from .adaln_merge import merge_adaln_modulation
from .cuda_graphs import CUDAGraphRunner, make_bucket_key


def _t5_tokens_for(captions: list[str], t5_tokenizer, max_len: int = 512, device: str = "cuda"):
    enc = t5_tokenizer(captions, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    return enc["input_ids"].to(device, dtype=torch.long), enc["attention_mask"].to(device)


def train(cfg: Config) -> None:
    ensure_on_path()
    # Liger fused RMSNorm kernel — replaces Anima's autocast(fp32) round-trip
    # for all 113 RMSNorm calls per forward. Must happen AFTER ensure_on_path()
    # so `library.anima_models` is importable.
    install_liger_patch()
    # Custom Triton AdaLN kernel + fused gated-residual — replaces the
    # 3-op (layer_norm → mul → add) chain at all 84 AdaLN sites and the
    # 2-op (mul → add) chain at all 84 gated-residual sites with single
    # Triton kernels (forward + custom backward).
    install_adaln_patch()
    # TE fused RoPE kernel — replaces sd-scripts' python-level rotate-half
    # chain at all 56 self-attn q/k sites per forward with one CUDA kernel
    # per call.
    install_rope_patch()
    from library import anima_utils  # type: ignore

    torch.manual_seed(cfg.train.seed)
    device = "cuda"
    dtype = torch_dtype(cfg.train.precision)

    models = load_all(
        dit_path=cfg.paths.dit,
        qwen3_path=cfg.paths.qwen3,
        vae_path=cfg.paths.vae,
        dtype=dtype,
        attn_mode="torch",
        device=device,
        loading_device="cpu",
    )
    # TE + VAE stay on CPU during training (training reads cached embeds/latents).
    # They are moved to GPU for the duration of each sampling tick, then back.
    models.dit.to(device)

    # Merge each Block's three `adaln_modulation_*` Sequentials into one
    # SiLU + one fused first Linear + one bmm. Saves ~168 kernel launches per
    # forward on tiny (B*T) GEMMs that were launch-bound. Must happen after
    # model is on device but is safe before or after LoKr/FP8 — the merged
    # Linear's name still starts with `adaln_modulation_` so existing FP8
    # skip rules apply.
    n_merged = merge_adaln_modulation(models.dit)
    print(f"merged AdaLN-modulation triplets in {n_merged} Blocks")

    # Sampling strategies (lazy-built; only used by sample_all_prompts).
    from library import strategy_anima  # type: ignore
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=models.tokenizer,
        t5_tokenizer=None,
        qwen3_path=cfg.paths.qwen3,
        t5_tokenizer_path=None,
    )
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    network = attach_lokr(models.dit, cfg.lokr, network_dim=128, network_alpha=128.0).to(device)
    n_train = trainable_param_count(network)
    print(f"trainable LoKr params: {n_train:,}")

    if cfg.train.gradient_checkpointing and hasattr(models.dit, "enable_gradient_checkpointing"):
        models.dit.enable_gradient_checkpointing()

    # MXFP8 frozen-base quantization. For `precision="mxfp8"` we pre-quantize
    # each LokrModule's frozen base weight to MXFP8 once (forward GEMMs run
    # ~1.3× faster on sm_120; backward dgrad stays bf16 against the cached
    # original). See src/anima_trainer/fp8_quant.py for the rationale and
    # the `te.Linear`-bypass design.
    if cfg.train.precision == "mxfp8":
        from .fp8_quant import (
            quantize_frozen_linears,
            collect_lokr_wrapped_linears,
        )
        # Quantize every frozen Linear in the DiT (self-attn q/k/v/output,
        # adaln_modulation, FinalLayer, ...) — skipping LoKr-wrapped ones,
        # which stay bf16. LoKr-wrapped Linears need a 2-GEMM decomposition
        # to mix mxfp8 base with bf16 delta, which net-loses against the
        # merged-weight bf16 path; better to leave them alone and reap the
        # mxfp8 win on the unwrapped half.
        skip_set = collect_lokr_wrapped_linears(network)
        n_quant_dit = quantize_frozen_linears(models.dit, skip=skip_set)
        print(f"MXFP8: quantized {n_quant_dit} unwrapped frozen Linears (LoKr-wrapped stay bf16)")
    elif cfg.train.precision == "fp8":
        # FP8 pass coverage:
        #   * Unwrapped frozen Linears (self-attn q/k/v/o, LLM adapter,
        #     etc.): swap to te.Linear → forward + dgrad in FP8 via
        #     fp8_autocast(Float8BlockScaling).
        #   * LoKr-wrapped Linears (cross-attn + MLP per the
        #     anima-cross-mlp preset): keep the wrapper but route the
        #     merged-weight GEMM through FP8LoKrLinear, which JIT-
        #     quantizes both x and merged_W and runs forward + dgrad +
        #     wgrad in FP8. ~1.2× faster fwd+bwd on the MLP shape.
        from .fp8_quant import (
            swap_frozen_linears_to_te,
            collect_lokr_wrapped_linears,
            patch_anima_checkpoint_for_fp8,
        )
        from .lokr_patch import enable_fp8 as enable_lokr_fp8
        skip_set = collect_lokr_wrapped_linears(network)
        n_swapped = swap_frozen_linears_to_te(models.dit, skip=skip_set)
        n_lokr_fp8 = enable_lokr_fp8(network)
        print(
            f"FP8: swapped {n_swapped} unwrapped frozen Linears to te.Linear; "
            f"{n_lokr_fp8} LoKr-wrapped Linears routed through FP8LoKrLinear"
        )
        # te.Linear under fp8_autocast caches FP8 metadata across the forward;
        # torch.utils.checkpoint recomputes without that metadata, producing
        # a mismatched saved-tensor count. Swap to te.checkpoint.
        if cfg.train.gradient_checkpointing:
            patch_anima_checkpoint_for_fp8()
    # quantize_dit_in_place is a no-op for bf16/mxfp8/fp8.
    quantize_dit_in_place(models.dit, cfg.train.precision)

    # torch.compile must happen AFTER LoKr + (optional) quantize — they replace
    # the weight tensor and forward function on submodules respectively.
    if cfg.train.compile_mode:
        print(f"compiling DiT with mode={cfg.train.compile_mode!r} (first step will be slow)...")
        models.dit = torch.compile(models.dit, mode=cfg.train.compile_mode, dynamic=False)

    optim = build_optimizer(network.parameters(), d0=cfg.optim.d0)
    optim.train()

    # T5 tokenizer for the LLM adapter target_input_ids
    t5_tok = anima_utils.load_t5_tokenizer(None)

    # Data
    cache = Cache(cfg.paths.cache_db)
    samples = scan_dataset(cfg.paths.train_data_dir)
    # Look up bucket_idx from cached crops
    enriched = []
    for s in samples:
        crow = cache.get_crop(s.src_path)
        if crow is None:
            raise RuntimeError(f"no cached crop for {s.src_path}; run precompute first")
        enriched.append(s.__class__(src_path=s.src_path, bucket_idx=crow.bucket_idx, caption=s.caption))
    samples = enriched
    print(f"dataset: {len(samples)} samples across {len({s.bucket_idx for s in samples})} buckets")

    ds = CachedAnimaDataset(samples, cache_db_path=cfg.paths.cache_db, vae_fp=models.vae_fp, te_fp=models.te_fp)
    sampler = BucketBatchSampler(samples, cfg.train.batch_size, drop_last=False, seed=cfg.train.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.train.num_workers, pin_memory=True)

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CUDA graphs: optional opt-in (cfg.train.cuda_graphs). The captured
    # region is forward+backward; optimizer.step + clip stay outside the
    # graph because Prodigy+SF has python control flow that's not capture-
    # safe. One graph per (bucket_shape, batch_size); shared memory pool.
    def _forward_and_loss(static_batch: dict) -> torch.Tensor:
        """Single training step body — forward through DiT + MSE loss.
        Called by both the eager and CUDA-graph paths so behavior is
        bit-identical between them. Returns a fp32 scalar loss.
        """
        with autocast_for(cfg.train.precision), sm120_sdpa(), fp8_autocast_for(cfg.train.precision):
            pred = models.dit(
                static_batch["noisy"],
                static_batch["timesteps"],
                static_batch["prompt_embeds"],
                padding_mask=static_batch["padding_mask"],
                target_input_ids=static_batch["t5_ids"],
                target_attention_mask=static_batch["t5_mask"],
                source_attention_mask=static_batch["qwen3_mask"],
            )
        pred = pred.squeeze(2)
        return torch.nn.functional.mse_loss(pred.float(), static_batch["target"].float())

    graph_runner: CUDAGraphRunner | None = None
    if cfg.train.cuda_graphs:
        graph_runner = CUDAGraphRunner(
            _forward_and_loss,
            warmup_steps=cfg.train.cuda_graph_warmup_steps,
        )
        print(f"CUDA graphs enabled (warmup={cfg.train.cuda_graph_warmup_steps} steps per bucket)")

    # CUDA-graph capture cannot allocate gradients during replay — `.grad`
    # must be a real tensor that persists across iterations. Use
    # set_to_none=False so zeroing keeps the buffer alive.
    use_set_to_none = graph_runner is None

    global_step = 0
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.train.max_train_epochs
    train_pbar = tqdm(total=total_steps, desc="training", unit="step", position=0, dynamic_ncols=True)
    for epoch in range(cfg.train.max_train_epochs):
        sampler.set_epoch(epoch)
        epoch_pbar = tqdm(
            total=steps_per_epoch,
            desc=f"epoch {epoch+1}/{cfg.train.max_train_epochs}",
            unit="step",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )
        for batch in loader:
            latents = batch["latent"].to(device, dtype=dtype, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype, non_blocking=True)
            qwen3_mask = batch["qwen3_attn_mask"].to(device, non_blocking=True)
            t5_ids, t5_mask = _t5_tokens_for(batch["caption"], t5_tok, device=device)

            noisy, target, timesteps, sigmas = noisy_input_and_target(latents)
            # 4D → 5D for DiT (T=1)
            noisy5 = noisy.unsqueeze(2)
            bs, _, h, w = latents.shape
            padding_mask = torch.zeros(bs, 1, h, w, dtype=dtype, device=device)

            static_batch = {
                "latent": latents,
                "prompt_embeds": prompt_embeds,
                "qwen3_mask": qwen3_mask,
                "t5_ids": t5_ids,
                "t5_mask": t5_mask,
                "noisy": noisy5,
                "target": target,
                "timesteps": timesteps,
                "padding_mask": padding_mask,
            }

            optim.zero_grad(set_to_none=use_set_to_none)

            if graph_runner is not None:
                # Skip captured path when the final partial batch has a
                # different shape than the captured one — drop_last=False in
                # the sampler so the last batch of a bucket may shrink.
                key = make_bucket_key(latents, t5_ids)
                loss = graph_runner.step(key, static_batch, list(network.parameters()))
            else:
                loss = _forward_and_loss(static_batch)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optim.step()
            global_step += 1
            postfix = {"loss": f"{loss.item():.4f}", "step": global_step}
            epoch_pbar.set_postfix(postfix, refresh=False)
            epoch_pbar.update(1)
            train_pbar.set_postfix(postfix, refresh=False)
            train_pbar.update(1)

        epoch_pbar.close()
        if (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            ckpt = out_dir / f"{cfg.train.output_name}_e{epoch+1:06d}.safetensors"
            _save_lora(network, ckpt)
        if cfg.sample.every_n_epochs and (epoch + 1) % cfg.sample.every_n_epochs == 0:
            optim.eval()                           # Prodigy+SF requires .eval() before inference
            models.dit.eval()
            try:
                # No SDPA restriction here: the VAE decoder's attention hits
                # shapes that cuDNN refuses; let torch pick freely.
                sample_all_prompts(
                    dit=models.dit,
                    text_encoder=models.text_encoder,
                    vae=models.vae,
                    tokenize_strategy=tokenize_strategy,
                    encoding_strategy=encoding_strategy,
                    prompts_file=cfg.sample.prompts_file,
                    out_dir=out_dir / "samples",
                    epoch=epoch + 1,
                    device=torch.device(device),
                    dtype=dtype,
                    precision=cfg.train.precision,
                )
            finally:
                models.dit.train()
                optim.train()

    train_pbar.close()
    final = out_dir / f"{cfg.train.output_name}.safetensors"
    _save_lora(network, final)
    print(f"done. saved final adapter to {final}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


def _save_lora(network: torch.nn.Module, path: Path) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in network.state_dict().items()}
    save_file(sd, str(path))
    print(f"saved {path} ({sum(t.numel() for t in sd.values()):,} params)")
