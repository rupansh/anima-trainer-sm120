"""On-the-fly Euler sampler for the Anima DiT with a hot-reloading prompt file.

The prompt file is **re-read every sampling tick** — never cached. The user may
edit it freely while training runs and the next tick will pick up the new file.

Prompt syntax (one prompt per line, matches sd-scripts):
    <positive prompt> --n <negative> --w 1024 --h 1024 --l 5 --s 30 --d 12345

Flags:
    --n  negative prompt (default "")
    --w  width  (default 1024)
    --h  height (default 1024)
    --l  CFG scale (default 5.0)
    --s  steps (default 30)
    --d  seed (default random)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import shlex
import time
from typing import Optional
import numpy as np
from PIL import Image
import torch

from .sdscripts_bridge import ensure_on_path
from .precision import fp8_autocast_for
from .tlokr import clear_timestep as clear_tlokr_timestep
from .tlokr import set_timestep as set_tlokr_timestep


# ---- prompt parsing --------------------------------------------------------


@dataclass
class Prompt:
    text: str
    negative: str = ""
    width: int = 1024
    height: int = 1024
    cfg: float = 5.0
    steps: int = 30
    seed: Optional[int] = None
    flow_shift: float = 3.0


_FLAGS = {"--n", "--w", "--h", "--l", "--s", "--d", "--f"}


def parse_prompt_line(line: str) -> Optional[Prompt]:
    """Split into the positive text + a dict of flag → value, then build Prompt.

    The positive prompt and `--n` value can both span multiple whitespace-separated
    tokens (everything up to the next `--<flag>`). The remaining flags take a
    single token.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    toks = line.split()
    # Walk left-to-right and partition into [prefix_tokens] + segments keyed by flag.
    segments: dict[str | None, list[str]] = {None: []}
    current: str | None = None
    for tok in toks:
        if tok in _FLAGS:
            current = tok
            segments[current] = []
        else:
            segments[current].append(tok)
    p = Prompt(text=" ".join(segments.get(None, [])))
    for flag, words in segments.items():
        if flag is None or not words:
            continue
        if flag == "--n":
            p.negative = " ".join(words)
        elif flag == "--w":
            p.width = int(words[0])
        elif flag == "--h":
            p.height = int(words[0])
        elif flag == "--l":
            p.cfg = float(words[0])
        elif flag == "--s":
            p.steps = int(words[0])
        elif flag == "--d":
            p.seed = int(words[0])
        elif flag == "--f":
            p.flow_shift = float(words[0])
    return p


def read_prompts(path: str | Path) -> list[Prompt]:
    p = Path(path)
    if not p.is_file():
        return []
    out: list[Prompt] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        pp = parse_prompt_line(line)
        if pp is not None:
            out.append(pp)
    return out


# ---- prompt encoding -------------------------------------------------------


def _encode_via_strategy(
    prompt: str,
    *,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
):
    tokens = tokenize_strategy.tokenize(prompt)
    embeds, attn_mask, t5_ids, t5_mask = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
    return (
        embeds.to(device=device, dtype=dtype),
        attn_mask.to(device=device),
        t5_ids.to(device=device, dtype=torch.long),
        t5_mask.to(device=device),
    )


def _apply_llm_adapter(dit, prompt_embeds, attn_mask, t5_ids, t5_mask) -> torch.Tensor:
    crossattn_emb = dit.llm_adapter(
        source_hidden_states=prompt_embeds,
        target_input_ids=t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=attn_mask,
    )
    crossattn_emb[~t5_mask.bool()] = 0
    return crossattn_emb


# ---- euler integration -----------------------------------------------------


@torch.no_grad()
def euler_denoise(
    dit,
    crossattn_emb: torch.Tensor,
    *,
    width: int,
    height: int,
    steps: int,
    flow_shift: float,
    cfg: float,
    neg_crossattn_emb: Optional[torch.Tensor],
    seed: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    precision: str = "bf16",
) -> torch.Tensor:
    """Plain euler discrete for rectified flow. Returns latents in (1, 16, 1, H/8, W/8).

    `precision` gates the TE FP8 autocast — pass the training precision through
    so samples are generated with the same forward path the LoRA was trained
    against (the saved adapter is bf16, but the *base* the adapter is added to
    is FP8-quantized when precision="fp8" and we want sample-time parity).
    """
    latent_h, latent_w = height // 8, width // 8
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    else:
        gen = None
    x = torch.randn((1, 16, 1, latent_h, latent_w), generator=gen, dtype=torch.float32).to(device).to(dtype)

    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)
    use_cfg = cfg > 1.0 and neg_crossattn_emb is not None

    # NOTE: batched CFG (one batch-2 forward instead of two batch-1) was tried and
    # rejected — at 1024² the Anima DiT is already SM-saturated at B=1, so batch-2
    # is ~3% slower. It only helps at ≤512² (see tests/bench_sample_cfg.py). Since
    # user prompts are all 1024-scale, we keep the sequential path.
    for i in range(steps):
        t = sigmas[i].unsqueeze(0)
        # The Euler schedule is uniform before flow shifting. Supplying the
        # equivalent Python scalar lets T-LoKr slice its factors at batch=1
        # without synchronizing on ``t.item()`` or computing masked ranks.
        t_scalar = 1.0 - (i / steps)
        if flow_shift != 1.0:
            t_scalar = (t_scalar * flow_shift) / (
                1.0 + (flow_shift - 1.0) * t_scalar
            )
        set_tlokr_timestep(t, uniform_timestep=t_scalar)
        try:
            with fp8_autocast_for(precision):
                if use_cfg:
                    pos = dit(x, t, crossattn_emb, padding_mask=padding_mask).float()
                    neg = dit(x, t, neg_crossattn_emb, padding_mask=padding_mask).float()
                    model_out = neg + cfg * (pos - neg)
                else:
                    model_out = dit(x, t, crossattn_emb, padding_mask=padding_mask).float()
        finally:
            clear_tlokr_timestep()
        dt = sigmas[i + 1] - sigmas[i]
        x = x + (model_out * dt).to(dtype)
    return x


# ---- decode + save ---------------------------------------------------------


@torch.no_grad()
def decode_latents_to_image(vae, latents: torch.Tensor) -> Image.Image:
    """Decode 5D latents to a single PIL.Image (assumes batch=1)."""
    pixels = vae.decode_to_pixels(latents)  # returns 4D or 5D depending on input
    if pixels.ndim == 5:
        pixels = pixels.squeeze(2)
    pixels = pixels.squeeze(0).clamp(-1, 1)
    pixels = ((pixels.float() + 1.0) * 127.5).round().to(torch.uint8)
    arr = pixels.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ---- top-level "do one tick" -----------------------------------------------


def sample_all_prompts(
    *,
    dit,
    text_encoder,
    vae,
    tokenize_strategy,
    encoding_strategy,
    prompts_file: str | Path,
    out_dir: str | Path,
    epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    precision: str = "bf16",
) -> None:
    """One sampling tick: read prompts file, generate every prompt, save PNGs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(prompts_file)
    if not prompts:
        print(f"[sample] no prompts in {prompts_file}; skipping")
        return

    # Move TE+VAE to GPU for this tick (cheap relative to running the DiT).
    te_was_on = next(text_encoder.parameters()).device
    vae_was_on = next(vae.parameters()).device
    text_encoder.to(device)
    vae.to(device)
    try:
        for j, pp in enumerate(prompts):
            t0 = time.time()
            embeds, attn, t5_ids, t5_mask = _encode_via_strategy(
                pp.text,
                tokenize_strategy=tokenize_strategy,
                encoding_strategy=encoding_strategy,
                text_encoder=text_encoder,
                device=device,
                dtype=dtype,
            )
            with fp8_autocast_for(precision):
                crossattn_emb = _apply_llm_adapter(dit, embeds, attn, t5_ids, t5_mask)
            neg_crossattn_emb = None
            if pp.cfg > 1.0:
                n_embeds, n_attn, n_t5_ids, n_t5_mask = _encode_via_strategy(
                    pp.negative,
                    tokenize_strategy=tokenize_strategy,
                    encoding_strategy=encoding_strategy,
                    text_encoder=text_encoder,
                    device=device,
                    dtype=dtype,
                )
                with fp8_autocast_for(precision):
                    neg_crossattn_emb = _apply_llm_adapter(dit, n_embeds, n_attn, n_t5_ids, n_t5_mask)

            latents = euler_denoise(
                dit,
                crossattn_emb,
                width=pp.width,
                height=pp.height,
                steps=pp.steps,
                flow_shift=pp.flow_shift,
                cfg=pp.cfg,
                neg_crossattn_emb=neg_crossattn_emb,
                seed=pp.seed,
                device=device,
                dtype=dtype,
                precision=precision,
            )
            img = decode_latents_to_image(vae, latents)
            name = f"e{epoch:06d}_{j:02d}_{pp.seed if pp.seed is not None else 'rng'}.png"
            img.save(out / name)
            print(f"[sample] e{epoch} {j}: '{pp.text[:60]}...' -> {name}  ({time.time()-t0:.1f}s)")
    finally:
        text_encoder.to(te_was_on)
        vae.to(vae_was_on)
        torch.cuda.empty_cache()
