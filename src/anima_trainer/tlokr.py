"""Timestep-aware LoKr (T-LoKr) for Anima's flow-matching DiT.

The T-LoRA paper's vanilla schedule activates a prefix of the adapter rank:

    r(t) = floor((r_max - r_min) * (1 - t) + r_min),  t in [0, 1]

Anima uses the same convention as rectified-flow sigma: ``t=1`` is the
noisiest point and ``t=0`` is the clean endpoint.  LoKr does not normally
have one global low-rank axis when ``full_matrix=true``, so T-LoKr factors
the *large* Kronecker operand as ``W2 = A @ B`` and schedules that inner
rank.  The small Kronecker operand ``W1`` remains full.

For a linear input, the adapter is evaluated without materializing either
``W2`` or ``kron(W1, W2)``.  It is three structured GEMMs:

    x -> B -> timestep mask -> A -> W1

This avoids the very large full-delta wgrad in ordinary materialized LoKr.
All modules in a forward share one lazily-created mask tensor.  Batch-1
sampling additionally slices A/B to the active rank, eliminating the mask
kernel and the inactive-rank FLOPs entirely.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tlokr_kernels import rank_mix, rank_mix_wgrad


_FORMAT_VERSION = 1


def active_rank_for_timestep(
    timestep: float,
    *,
    max_rank: int,
    min_rank: int,
) -> int:
    """Return the vanilla T-LoRA prefix rank for a scalar flow timestep."""
    if max_rank <= 0:
        raise ValueError(f"max_rank must be positive; got {max_rank}")
    if not 1 <= min_rank <= max_rank:
        raise ValueError(
            f"min_rank must be in [1, {max_rank}]; got {min_rank}"
        )
    t = min(1.0, max(0.0, float(timestep)))
    return max(
        min_rank,
        min(max_rank, math.floor((max_rank - min_rank) * (1.0 - t) + min_rank)),
    )


@dataclass
class _TimestepState:
    timesteps: torch.Tensor
    # When the caller already knows the whole batch has one timestep (Euler
    # sampling), retain it as a Python float.  The adapter can slice its
    # factors without a GPU -> CPU synchronization.
    uniform_timestep: float | None = None
    masks: dict[tuple[int, int, torch.dtype, torch.device], torch.Tensor] = field(
        default_factory=dict
    )

    def mask(
        self,
        *,
        max_rank: int,
        min_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = (max_rank, min_rank, dtype, device)
        cached = self.masks.get(key)
        if cached is not None:
            return cached

        t = self.timesteps.detach().to(device=device, dtype=torch.float32).reshape(-1)
        active = torch.floor(
            (max_rank - min_rank) * (1.0 - t.clamp(0.0, 1.0)) + min_rank
        ).to(torch.int64)
        active.clamp_(min=min_rank, max=max_rank)
        ranks = torch.arange(max_rank, device=device).unsqueeze(0)
        mask = (ranks < active.unsqueeze(1)).to(dtype=dtype)
        self.masks[key] = mask
        return mask


_TIMESTEP_STATE: ContextVar[_TimestepState | None] = ContextVar(
    "anima_tlokr_timestep_state", default=None
)


def set_timestep(
    timesteps: torch.Tensor,
    *,
    uniform_timestep: float | None = None,
) -> None:
    """Set the timestep context used by every T-LoKr module in one pass.

    The context intentionally remains live through backward: activation
    checkpointing recomputes adapter forwards while the backward is running.
    Call :func:`clear_timestep` in a ``finally`` block after backward.
    """
    if not torch.is_tensor(timesteps):
        raise TypeError("T-LoKr timesteps must be a torch.Tensor")
    _TIMESTEP_STATE.set(
        _TimestepState(timesteps=timesteps, uniform_timestep=uniform_timestep)
    )


def clear_timestep() -> None:
    """Clear the current T-LoKr timestep context."""
    _TIMESTEP_STATE.set(None)


def _current_state() -> _TimestepState:
    state = _TIMESTEP_STATE.get()
    if state is None:
        raise RuntimeError(
            "T-LoKr forward has no timestep context; call "
            "anima_trainer.tlokr.set_timestep() around the DiT forward"
        )
    return state


def _validate_schedule_on_load(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    _local_metadata: dict,
    _strict: bool,
    _missing_keys: list[str],
    _unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Reject checkpoints whose T-LoKr topology differs from this module."""
    key = prefix + "tlokr_schedule"
    incoming = state_dict.get(key)
    if incoming is None:
        # strict=True reports the missing marker through the normal loader.
        return
    expected = module.tlokr_schedule.detach().to(device="cpu", dtype=torch.int64)
    actual = incoming.detach().to(device="cpu", dtype=torch.int64)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        error_msgs.append(
            f"{key} is incompatible: checkpoint has {actual.tolist()}, "
            f"model expects {expected.tolist()}"
        )


def convert_network(
    network: nn.Module,
    *,
    rank: int,
    min_rank_ratio: float,
) -> int:
    """Convert a freshly-created full-matrix LyCORIS LoKr network in place.

    LyCORIS still owns targeting, wrapper application, state-dict naming, and
    compatibility with the rest of the trainer.  Only each wrapped Linear's
    large factor and forward algebra change here.

    Returns the number of converted modules.
    """
    from lycoris.modules.lokr import LokrModule

    if rank <= 0:
        raise ValueError(f"T-LoKr rank must be positive; got {rank}")
    if not 0.0 < min_rank_ratio <= 1.0:
        raise ValueError(
            f"T-LoKr min_rank_ratio must be in (0, 1]; got {min_rank_ratio}"
        )
    min_rank = max(1, min(rank, math.ceil(rank * min_rank_ratio)))

    # LycorisNetworkKohya keeps newly-created adapters in plain lists until
    # ``apply_to`` registers them as submodules. Conversion deliberately runs
    # before apply_to, so inspect those lists when present.
    pending = list(getattr(network, "unet_loras", ())) + list(
        getattr(network, "text_encoder_loras", ())
    )
    candidates = pending if pending else list(network.modules())

    converted = 0
    for module in candidates:
        if not isinstance(module, LokrModule):
            continue
        if module.module_type != "linear":
            raise TypeError(
                f"T-LoKr currently supports Linear targets only; "
                f"{module.lora_name} wraps {module.module_type}"
            )
        if not module.use_w1 or not module.use_w2:
            raise ValueError(
                f"{module.lora_name} was not created in full-matrix LoKr mode; "
                "T-LoKr conversion requires full_matrix=true"
            )
        if module.wd or module.bypass_mode:
            raise ValueError(
                f"{module.lora_name}: T-LoKr does not support weight "
                "decomposition or LyCORIS bypass mode"
            )
        if module.rank_dropout or module.dropout:
            raise ValueError(
                f"{module.lora_name}: T-LoKr uses the timestep rank schedule "
                "and does not support LyCORIS rank/dropout"
            )

        old_w2 = module.lokr_w2
        out_factor, in_factor = old_w2.shape
        if rank > min(out_factor, in_factor):
            raise ValueError(
                f"T-LoKr rank {rank} exceeds the large Kronecker factor "
                f"capacity {tuple(old_w2.shape)} in {module.lora_name}"
            )

        # Remove the full W2 parameter before registering the factor pair.
        del module.lokr_w2
        w2_a = nn.Parameter(
            torch.empty(
                out_factor,
                rank,
                device=old_w2.device,
                dtype=old_w2.dtype,
            )
        )
        w2_b = nn.Parameter(
            torch.empty(
                rank,
                in_factor,
                device=old_w2.device,
                dtype=old_w2.dtype,
            )
        )
        nn.init.kaiming_uniform_(w2_a, a=math.sqrt(5))
        nn.init.zeros_(w2_b)
        module.register_parameter("lokr_w2_a", w2_a)
        module.register_parameter("lokr_w2_b", w2_b)

        module.use_w2 = False
        module.full_matrix = False
        module.lora_dim = rank
        module._tlokr_enabled = True
        module._tlokr_rank = rank
        module._tlokr_min_rank = min_rank
        if not isinstance(module.scalar, nn.Parameter):
            module._tlokr_alpha = (
                float(module.scale)
                * float(
                    module.scalar.item()
                    if torch.is_tensor(module.scalar)
                    else module.scalar
                )
                * float(module.multiplier)
            )
        # Persistent marker makes a T-LoKr checkpoint fail loudly if it is
        # accidentally loaded into an ordinary LoKr topology with strict=True.
        module.register_buffer(
            "tlokr_schedule",
            torch.tensor([_FORMAT_VERSION, rank, min_rank], dtype=torch.int32),
        )
        module.register_load_state_dict_pre_hook(_validate_schedule_on_load)
        converted += 1

    if converted == 0:
        raise RuntimeError("T-LoKr conversion found no LyCORIS LoKr modules")
    network._tlokr_enabled = True
    network._tlokr_rank = rank
    network._tlokr_min_rank = min_rank
    return converted


class _StructuredLoKr(torch.autograd.Function):
    """Memory-bounded structured Kronecker projection with manual backward.

    Let ``W2 = A @ B`` and ``W = kron(C, W2)``.  Autograd over the naïve
    three-GEMM expression retains both rank and expanded-factor activations at
    every one of Anima's 168 adapter sites.  The expanded activations dominate
    memory at 1024².  This function saves only the adapter input and the small
    factors, then recomputes its intermediates one module at a time during
    backward.  The six gradients are ordinary GEMMs; no full delta or full
    weight gradient is ever formed.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2_a: torch.Tensor,
        w2_b: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        compute_dtype = x.dtype
        c = w1.to(dtype=compute_dtype)
        a = w2_a.to(dtype=compute_dtype)
        b = w2_b.to(dtype=compute_dtype)

        uq = c.shape[1]
        grouped = x.reshape(x.shape[0], -1, uq, x.shape[-1] // uq)
        hidden = torch.matmul(grouped, b.transpose(0, 1))
        # Associate the Kronecker projection as C @ H @ A.T. Applying the
        # tiny C=(8x8) before A keeps its contracted surface at rank=128
        # instead of the much wider vp=256/1024 output.
        mixed = rank_mix(hidden, c, mask)
        out = torch.matmul(mixed, a.transpose(0, 1))

        tensors = (x, w1, w2_a, w2_b)
        if mask is not None:
            tensors += (mask,)
        ctx.save_for_backward(*tensors)
        ctx.has_mask = mask is not None
        ctx.input_shape = x.shape
        return out.reshape(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        x, w1, w2_a, w2_b = saved[:4]
        mask = saved[4] if ctx.has_mask else None
        compute_dtype = x.dtype
        c = w1.to(dtype=compute_dtype)
        a = w2_a.to(dtype=compute_dtype)
        b = w2_b.to(dtype=compute_dtype)

        batch = x.shape[0]
        uq = c.shape[1]
        up = c.shape[0]
        vp = a.shape[0]
        grouped = x.reshape(batch, -1, uq, x.shape[-1] // uq)

        # Recompute the rank-space forward intermediates locally. They are released
        # as soon as this adapter's backward returns rather than retained for
        # all transformer blocks at once.
        hidden = torch.matmul(grouped, b.transpose(0, 1))
        mixed = rank_mix(hidden, c, mask)

        grad_blocks = grad_output.to(compute_dtype).reshape(
            batch, -1, up, vp
        )
        grad_w2_a = (
            grad_blocks.reshape(-1, vp).transpose(0, 1)
            @ mixed.reshape(-1, mixed.shape[-1])
        )
        grad_mixed = torch.matmul(grad_blocks, a)
        grad_w1 = rank_mix_wgrad(grad_mixed, hidden, mask)
        grad_hidden = rank_mix(grad_mixed, c, mask, transpose=True)

        grad_w2_b = (
            grad_hidden.reshape(-1, grad_hidden.shape[-1]).transpose(0, 1)
            @ grouped.reshape(-1, grouped.shape[-1])
        )
        grad_grouped = torch.matmul(grad_hidden, b)
        grad_x = grad_grouped.reshape(ctx.input_shape)

        return (
            grad_x.to(x.dtype),
            grad_w1.to(w1.dtype),
            grad_w2_a.to(w2_a.dtype),
            grad_w2_b.to(w2_b.dtype),
            None,
        )


class _FP8StructuredLoKr(torch.autograd.Function):
    """Block-scaled FP8 specialization of :class:`_StructuredLoKr`.

    The large B/A projections and their dgrad/wgrad GEMMs have 128-aligned
    dimensions on every Anima target.  Transformer Engine handles those in
    FP8.  The tiny W1 projection is only 8x8, so it stays bf16: quantization
    overhead would dominate its arithmetic.  Quantized A/B weights from the
    forward are retained as opaque, non-state objects and reused by backward.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2_a: torch.Tensor,
        w2_b: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        from .fp8_quant import (
            _FP8_W_QUANTIZER,
            _FP8_X_QUANTIZER,
            tex_ext,
        )

        compute_dtype = x.dtype
        c = w1.to(dtype=compute_dtype)
        a = w2_a.to(dtype=compute_dtype).contiguous()
        b = w2_b.to(dtype=compute_dtype).contiguous()
        aq = _FP8_W_QUANTIZER.quantize(a)
        bq = _FP8_W_QUANTIZER.quantize(b)

        uq = c.shape[1]
        grouped = x.reshape(x.shape[0], -1, uq, x.shape[-1] // uq)
        grouped_2d = grouped.reshape(-1, grouped.shape[-1]).contiguous()
        grouped_q = _FP8_X_QUANTIZER.quantize(grouped_2d)
        hidden_2d = tex_ext.general_gemm(
            bq, grouped_q, out_dtype=compute_dtype, layout="TN"
        )[0]
        hidden = hidden_2d.reshape(*grouped.shape[:-1], b.shape[0])
        mixed = rank_mix(hidden, c, mask)
        mixed_q = _FP8_X_QUANTIZER.quantize(
            mixed.reshape(-1, mixed.shape[-1]).contiguous()
        )
        out_2d = tex_ext.general_gemm(
            aq, mixed_q, out_dtype=compute_dtype, layout="TN"
        )[0]
        out = out_2d.reshape(*mixed.shape[:-1], a.shape[0])

        tensors = (x, w1, w2_a, w2_b)
        if mask is not None:
            tensors += (mask,)
        ctx.save_for_backward(*tensors)
        ctx.aq = aq
        ctx.bq = bq
        ctx.has_mask = mask is not None
        ctx.input_shape = x.shape
        return out.reshape(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from .fp8_quant import _FP8_X_QUANTIZER, tex_ext

        saved = ctx.saved_tensors
        x, w1, w2_a, w2_b = saved[:4]
        mask = saved[4] if ctx.has_mask else None
        compute_dtype = x.dtype
        c = w1.to(dtype=compute_dtype)
        batch = x.shape[0]
        uq = c.shape[1]
        up = c.shape[0]
        rank = w2_b.shape[0]
        vp = w2_a.shape[0]

        grouped = x.reshape(batch, -1, uq, x.shape[-1] // uq)
        grouped_2d = grouped.reshape(-1, grouped.shape[-1]).contiguous()
        grouped_q = _FP8_X_QUANTIZER.quantize(grouped_2d)
        hidden_2d = tex_ext.general_gemm(
            ctx.bq, grouped_q, out_dtype=compute_dtype, layout="TN"
        )[0]
        hidden = hidden_2d.reshape(*grouped.shape[:-1], rank)
        mixed = rank_mix(hidden, c, mask)
        mixed_q = _FP8_X_QUANTIZER.quantize(
            mixed.reshape(-1, rank).contiguous()
        )

        grad_blocks = grad_output.to(compute_dtype).reshape(
            batch, -1, up, vp
        )
        grad_blocks_2d = grad_blocks.reshape(-1, vp).contiguous()
        grad_blocks_q = _FP8_X_QUANTIZER.quantize(grad_blocks_2d)

        # A wgrad: grad_output.T @ mixed.
        grad_w2_a = tex_ext.general_gemm(
            mixed_q,
            grad_blocks_q,
            out_dtype=compute_dtype,
            layout="NT",
        )[0]
        # A dgrad: grad_output @ A.
        grad_mixed_2d = tex_ext.general_gemm(
            ctx.aq,
            grad_blocks_q,
            out_dtype=compute_dtype,
            layout="NN",
        )[0]
        grad_mixed = grad_mixed_2d.reshape(*mixed.shape)
        grad_w1 = rank_mix_wgrad(grad_mixed, hidden, mask)
        grad_hidden = rank_mix(grad_mixed, c, mask, transpose=True)
        grad_hidden_2d = grad_hidden.reshape(-1, rank).contiguous()
        grad_hidden_q = _FP8_X_QUANTIZER.quantize(grad_hidden_2d)

        # B wgrad and dgrad share the same quantized grouped/grad-hidden pair.
        grad_w2_b = tex_ext.general_gemm(
            grouped_q,
            grad_hidden_q,
            out_dtype=compute_dtype,
            layout="NT",
        )[0]
        grad_grouped_2d = tex_ext.general_gemm(
            ctx.bq,
            grad_hidden_q,
            out_dtype=compute_dtype,
            layout="NN",
        )[0]
        grad_x = grad_grouped_2d.reshape(ctx.input_shape)

        return (
            grad_x.to(x.dtype),
            grad_w1.to(w1.dtype),
            grad_w2_a.to(w2_a.dtype),
            grad_w2_b.to(w2_b.dtype),
            None,
        )


def _structured_delta(module, x: torch.Tensor, *, fp8: bool) -> torch.Tensor:
    """Evaluate ``x @ kron(W1, A @ B).T`` without building either matrix."""
    state = _current_state()
    rank = module._tlokr_rank
    min_rank = module._tlokr_min_rank
    w1 = module.lokr_w1

    # W1 is (up, uq).  The input's final dimension is uq*vq.
    uq = w1.shape[1]
    if x.shape[-1] % uq:
        raise RuntimeError(
            f"{module.lora_name}: input width {x.shape[-1]} is not divisible "
            f"by the W1 input factor {uq}"
        )
    if state.uniform_timestep is not None:
        # Batch-1/single-timestep inference: slicing skips inactive-rank work
        # in both factor GEMMs and removes the mask kernel.
        active = active_rank_for_timestep(
            state.uniform_timestep,
            max_rank=rank,
            min_rank=min_rank,
        )
        # Block-scaled FP8 needs a 128-aligned contracted rank. Most sampled
        # timesteps use 64..127 active ranks, where the smaller bf16 sliced
        # GEMMs are both valid and cheaper than padding back to 128.
        function = (
            _FP8StructuredLoKr
            if fp8 and active % 128 == 0
            else _StructuredLoKr
        )
        return function.apply(
            x,
            w1,
            module.lokr_w2_a[:, :active],
            module.lokr_w2_b[:active],
            None,
        )
    else:
        mask = state.mask(
            max_rank=rank,
            min_rank=min_rank,
            dtype=x.dtype,
            device=x.device,
        )
        if x.shape[0] != mask.shape[0]:
            raise RuntimeError(
                f"{module.lora_name}: activation batch {x.shape[0]} does not "
                f"match timestep batch {mask.shape[0]}"
            )
        function = _FP8StructuredLoKr if fp8 else _StructuredLoKr
        return function.apply(
            x,
            w1,
            module.lokr_w2_a,
            module.lokr_w2_b,
            mask,
        )


def forward(module, x: torch.Tensor) -> torch.Tensor:
    """Optimized T-LoKr forward called by :mod:`lokr_patch`."""
    org = module.org_module[0]
    base_weight_q = getattr(module, "_tlokr_fp8_base_weight", None)
    if base_weight_q is not None:
        from .fp8_quant import fp8_frozen_linear

        base = fp8_frozen_linear(x, base_weight_q, org.weight, org.bias)
    else:
        base = F.linear(x, org.weight, org.bias)

    delta = _structured_delta(module, x, fp8=base_weight_q is not None)
    scalar = module.scalar
    if isinstance(scalar, nn.Parameter):
        # Not used by the trainer, but retain correct differentiability if a
        # caller constructs such a module manually.
        delta = delta * scalar
        alpha = float(module.scale) * float(module.multiplier)
    else:
        alpha = module._tlokr_alpha
    # torch.add(alpha=...) fuses adapter scaling and residual addition.
    return torch.add(base, delta, alpha=alpha)
