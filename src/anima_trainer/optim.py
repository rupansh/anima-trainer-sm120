"""Prodigy+ Schedule Free with our locked-in performance defaults.

User exposes only `d0`. Every other knob is hard-tuned. If you need to change
one of these, do it deliberately by editing this file — not from config.

Defaults are the same as the sd-scripts baseline `optimizer_args` (see
melted1-anima-config.toml:19), except `d0` is bumped to 1e-4 per the project
brief.
"""
from __future__ import annotations
import torch
from typing import Iterable


def build(params: Iterable[torch.nn.Parameter], *, d0: float = 1e-4):
    """Construct Prodigy+SF with our defaults. Returns (optimizer, train_call, eval_call).

    Schedule-free needs explicit `train()` / `eval()` calls when switching
    between training and evaluation (sampling). We expose those as callables so
    the rest of the loop is symmetric.
    """
    from prodigyplus import ProdigyPlusScheduleFree

    opt = ProdigyPlusScheduleFree(
        list(params),
        lr=1.0,
        betas=(0.95, 0.99),
        weight_decay=1e-4,
        weight_decay_by_lr=True,
        d0=d0,
        d_coef=1.0,
        prodigy_steps=0,
        eps=1e-8,
        split_groups=True,
        split_groups_mean=True,
        factored=True,
        use_stableadamw=True,
        use_bias_correction=True,
        use_cautious=False,
        stochastic_rounding=True,
        use_schedulefree=True,
    )
    return opt
