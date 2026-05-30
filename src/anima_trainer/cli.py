"""anima-trainer CLI.

Commands:
  precompute <config.toml>   — fill the LanceDB cache (crops + latents + text).
  train      <config.toml>   — run training (assumes precompute has been run).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .config import load as load_cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("anima-train")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_pre = sub.add_parser("precompute", help="cache crops/latents/text embeds")
    s_pre.add_argument("config", type=Path)
    s_train = sub.add_parser("train", help="run training")
    s_train.add_argument("config", type=Path)
    s_eval = sub.add_parser("eval-compare", help="pairwise compare two sample dirs")
    s_eval.add_argument("dir_a", type=Path)
    s_eval.add_argument("dir_b", type=Path)
    s_eval.add_argument("--no-lpips", action="store_true")
    s_eval.add_argument("--vae", action="store_true", help="also compute VAE-latent cosine")
    s_conv = sub.add_parser("convergence", help="latent-cosine convergence metric")
    s_conv.add_argument("config", type=Path)
    s_conv.add_argument("lora", type=Path)
    s_conv.add_argument("-n", "--n-samples", type=int, default=8)
    s_conv.add_argument("--seed", type=int, default=12345)

    args = p.parse_args(argv)

    if args.cmd == "eval-compare":
        from .eval_compare import run as eval_run
        eval_run(args.dir_a, args.dir_b, use_lpips=not args.no_lpips, use_vae=args.vae)
        return 0

    cfg = load_cfg(args.config)

    if args.cmd == "convergence":
        from .convergence import run_cli
        run_cli(cfg, args.lora, n_samples=args.n_samples, seed=args.seed)
        return 0

    if args.cmd == "precompute":
        from .precompute import run
        run(cfg)
    elif args.cmd == "train":
        from .train import train
        train(cfg)
    else:
        raise SystemExit(f"unknown command {args.cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
