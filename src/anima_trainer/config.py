"""User-facing TOML config for the anima-trainer.

Knobs are kept deliberately few. Everything else is a tuned constant. If you
think you need to add a knob here, you almost certainly don't — the constraints
in CLAUDE.md spell out which dials we expose and which we don't.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import tomllib
from typing import Literal

Precision = Literal["bf16", "mxfp8", "fp8"]


@dataclass(frozen=True)
class LokrCfg:
    factor: int = 8           # `factor` in lycoris.kohya algo=lokr
    full_matrix: bool = True
    preset: str = "full"


@dataclass(frozen=True)
class OptimCfg:
    """Prodigy+SF. d0 is the only user-tweakable knob.

    Default 1e-6 = the Prodigy library default. Prodigy+SF adapts d0 upward
    automatically as it learns the loss landscape, so starting conservative
    is fine.
    """
    d0: float = 1e-6


@dataclass(frozen=True)
class SampleCfg:
    every_n_epochs: int = 5
    sampler: str = "euler_a"
    prompts_file: str = "./melted1-ds-prompt.txt"   # hot-reloaded; never cached


@dataclass(frozen=True)
class PathsCfg:
    dit: str = "./models/anima-base-v1.0.safetensors"
    qwen3: str = "./models/qwen_3_06b_base.safetensors"
    vae: str = "./models/qwen_image_vae.safetensors"
    train_data_dir: str = "./melted1-ds"
    output_dir: str = "./outputs"
    cache_db: str = "./cache.lance"


@dataclass(frozen=True)
class TrainCfg:
    output_name: str = "anima"              # prefix for saved adapter safetensors
    resolution: int = 1024                  # 512 or 1024
    batch_size: int = 8
    max_train_epochs: int = 30
    save_every_n_epochs: int = 2
    precision: Precision = "fp8"
    seed: int = 42
    gradient_checkpointing: bool = True
    num_workers: int = 8
    # Experimental: capture one CUDA graph per (bucket_shape, batch_size).
    # The production FP8/LoKr block benchmark was flat (58.923 vs 58.943 ms),
    # so keep this disabled unless a changed stack wins a full-step A/B.
    cuda_graphs: bool = False
    # Number of warm-up steps per bucket before capture. 3 is enough for
    # cuDNN to pick a final algorithm and for Triton autotune to settle.
    cuda_graph_warmup_steps: int = 3
    # torch.compile mode for the DiT: None disables it. Values: "default",
    # "reduce-overhead", "max-autotune". Caveat: each new bucket shape triggers
    # a recompile, so multi-bucket training pays a one-time cost per bucket.
    compile_mode: str | None = None


@dataclass(frozen=True)
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    lokr: LokrCfg = field(default_factory=LokrCfg)
    sample: SampleCfg = field(default_factory=SampleCfg)


def _build(klass, data: dict):
    """Construct a frozen dataclass from a dict, ignoring extras-by-design."""
    fields = {f for f in klass.__dataclass_fields__}
    extras = set(data) - fields
    if extras:
        raise ValueError(f"{klass.__name__}: unknown keys {sorted(extras)}")
    return klass(**{k: v for k, v in data.items() if k in fields})


def load(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = Config(
        paths=_build(PathsCfg, raw.get("paths", {})),
        train=_build(TrainCfg, raw.get("train", {})),
        optim=_build(OptimCfg, raw.get("optim", {})),
        lokr=_build(LokrCfg, raw.get("lokr", {})),
        sample=_build(SampleCfg, raw.get("sample", {})),
    )
    if cfg.train.resolution not in (512, 1024):
        raise ValueError(f"resolution must be 512 or 1024; got {cfg.train.resolution}")
    return cfg


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
