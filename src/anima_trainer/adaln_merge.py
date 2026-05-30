"""Merge Anima's three per-Block `adaln_modulation_*` Sequentials into one
shared SiLU + one fused first Linear + one batched second matmul.

Original per-Block layout (use_adaln_lora=True path, the production config):

    adaln_modulation_self_attn  = Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3D))
    adaln_modulation_cross_attn = Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3D))
    adaln_modulation_mlp        = Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3D))

Each is called on the same input `emb_B_T_D`. That's 9 kernel launches per
Block (3 SiLU + 3 first Linear + 3 second Linear) × 28 Blocks = 252 launches
per forward, all on tiny `(B, T, *)` shapes (B*T ≈ 8 in production) where the
launch cost dwarfs the matmul.

Merged layout (this file):

    adaln_modulation_silu    = SiLU                                   # shared, called once
    adaln_modulation_l1      = Linear(D, 3 * lora_dim)                # concat of 3 first Linears
    adaln_modulation_l2_w    = (3, lora_dim, 3D) float buffer         # stack of 3 second Linears

Forward becomes 3 launches per Block (1 SiLU + 1 Linear + 1 bmm) → 84
launches across 28 Blocks. Numerically identical to the un-merged path:

    h          = SiLU(emb)                                   # (B, T, D)
    h1         = h @ W1_concat.T                             # (B, T, 3*lora_dim)
    h1_3split  = h1.view(B, T, 3, lora_dim).permute(2, 0, 1, 3).reshape(3, B*T, lora_dim)
    out_3split = bmm(h1_3split, W2_stack)                    # (3, B*T, 3D)
    [self_o, cross_o, mlp_o] = out_3split.unbind(0)          # each (B*T, 3D)

The non-lora path (`use_adaln_lora=False`) only has a single Linear inside
each Sequential. We handle it too: 3 of `Linear(D, 3D)` collapse into 1
`Linear(D, 9D)`, then `chunk(9)` for output. No bmm needed.

Skip rule for FP8 / MXFP8: the merged Linear's qualified name starts with
`adaln_modulation_` so the existing skip_name_substrings tuple in
`fp8_quant.py` continues to leave it bf16. The stacked-weight buffer
isn't an `nn.Linear` at all → it's invisible to those passes.
"""
from __future__ import annotations
import torch
import torch.nn as nn


_TAG = "_adaln_modulation_merged"


def _merge_block_lora(block: nn.Module) -> bool:
    """Merge for `use_adaln_lora=True`. Returns True if merged, False if skipped."""
    if getattr(block, _TAG, False):
        return False
    if not getattr(block, "use_adaln_lora", False):
        return False

    sa = block.adaln_modulation_self_attn
    ca = block.adaln_modulation_cross_attn
    mp = block.adaln_modulation_mlp

    # Layout check: Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3D))
    if not (len(sa) == len(ca) == len(mp) == 3):
        return False
    if not all(isinstance(m[0], nn.SiLU) for m in (sa, ca, mp)):
        return False
    if not all(isinstance(m[1], nn.Linear) and isinstance(m[2], nn.Linear)
               for m in (sa, ca, mp)):
        return False

    l1_self, l1_cross, l1_mlp = sa[1], ca[1], mp[1]
    l2_self, l2_cross, l2_mlp = sa[2], ca[2], mp[2]

    D = l1_self.in_features
    lora_dim = l1_self.out_features
    three_D = l2_self.out_features
    for lin in (l1_cross, l1_mlp):
        assert lin.in_features == D and lin.out_features == lora_dim
    for lin in (l2_cross, l2_mlp):
        assert lin.in_features == lora_dim and lin.out_features == three_D

    device = l1_self.weight.device
    dtype = l1_self.weight.dtype

    # First Linear: concat along output dim. (lora_dim, D) × 3 → (3*lora_dim, D).
    w1 = torch.cat(
        [l1_self.weight.detach(), l1_cross.weight.detach(), l1_mlp.weight.detach()],
        dim=0,
    ).contiguous()
    merged_l1 = nn.Linear(D, 3 * lora_dim, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        merged_l1.weight.copy_(w1)
    merged_l1.weight.requires_grad_(False)

    # Second Linear: stack as a (3, lora_dim, 3D) buffer for `bmm`. Each
    # original is (3D, lora_dim) — we transpose to (lora_dim, 3D) for the
    # right-hand operand of bmm.
    w2_stack = torch.stack(
        [
            l2_self.weight.detach().t().contiguous(),
            l2_cross.weight.detach().t().contiguous(),
            l2_mlp.weight.detach().t().contiguous(),
        ],
        dim=0,
    ).contiguous()  # (3, lora_dim, 3D)

    # Register the buffer (not a Parameter — frozen, no autograd).
    # Using register_buffer so it's tracked in `to(device)` / `state_dict`
    # could be confusing on re-save; persistent=False keeps it out of the
    # saved adapter (and the saved DiT wasn't ours to begin with).
    block.register_buffer("adaln_modulation_l2_stack", w2_stack, persistent=False)

    # Replace the three Sequentials with a SiLU + merged first Linear. Keep
    # names that start with `adaln_modulation_` so the FP8 skip rules in
    # `fp8_quant.py` continue to apply.
    block.adaln_modulation_silu = nn.SiLU()
    block.adaln_modulation_l1 = merged_l1
    # Free the originals — they're frozen and now redundant. Setting to
    # None unregisters the submodule (modules() iteration won't see them).
    del block.adaln_modulation_self_attn
    del block.adaln_modulation_cross_attn
    del block.adaln_modulation_mlp

    block._adaln_merged_dims = (D, lora_dim, three_D)
    setattr(block, _TAG, True)
    return True


def _merge_block_no_lora(block: nn.Module) -> bool:
    """Merge for `use_adaln_lora=False`. Returns True if merged."""
    if getattr(block, _TAG, False):
        return False
    if getattr(block, "use_adaln_lora", False):
        return False

    sa = block.adaln_modulation_self_attn
    ca = block.adaln_modulation_cross_attn
    mp = block.adaln_modulation_mlp

    if not (len(sa) == len(ca) == len(mp) == 2):
        return False
    if not all(isinstance(m[0], nn.SiLU) and isinstance(m[1], nn.Linear)
               for m in (sa, ca, mp)):
        return False

    l_self, l_cross, l_mlp = sa[1], ca[1], mp[1]
    D = l_self.in_features
    three_D = l_self.out_features
    for lin in (l_cross, l_mlp):
        assert lin.in_features == D and lin.out_features == three_D

    device = l_self.weight.device
    dtype = l_self.weight.dtype

    # (3*3D, D) — produces all 9 of {shift,scale,gate} × {self,cross,mlp} in
    # one matmul; chunk(9) downstream.
    w_concat = torch.cat(
        [l_self.weight.detach(), l_cross.weight.detach(), l_mlp.weight.detach()],
        dim=0,
    ).contiguous()
    merged = nn.Linear(D, 3 * three_D, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        merged.weight.copy_(w_concat)
    merged.weight.requires_grad_(False)

    block.adaln_modulation_silu = nn.SiLU()
    block.adaln_modulation_l1 = merged
    del block.adaln_modulation_self_attn
    del block.adaln_modulation_cross_attn
    del block.adaln_modulation_mlp

    block._adaln_merged_dims = (D, None, three_D)
    setattr(block, _TAG, True)
    return True


def merge_adaln_modulation(dit: nn.Module) -> int:
    """Walk `dit` and merge every Anima Block's AdaLN-modulation triplet.

    Idempotent. Returns the number of Blocks merged. Safe to call before or
    after `attach_lokr` (LoKr's `anima-*` presets don't wrap
    `adaln_modulation_*` modules) and before or after the FP8 / MXFP8 swap
    passes (the merged Linear's name still matches the skip substring
    `adaln_modulation`).
    """
    try:
        from library.anima_models import Block  # type: ignore
    except ImportError:
        return 0

    count = 0
    for mod in dit.modules():
        if not isinstance(mod, Block):
            continue
        if getattr(mod, _TAG, False):
            continue
        if getattr(mod, "use_adaln_lora", False):
            if _merge_block_lora(mod):
                count += 1
        else:
            if _merge_block_no_lora(mod):
                count += 1
    return count


def apply_merged_adaln(
    block: nn.Module,
    emb_B_T_D: torch.Tensor,
    adaln_lora_B_T_3D: torch.Tensor | None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # shift_self, scale_self, gate_self
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # shift_cross, scale_cross, gate_cross
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # shift_mlp, scale_mlp, gate_mlp
]:
    """Compute all 9 of `{shift, scale, gate} × {self, cross, mlp}` from the
    merged AdaLN modules. Returns three triples in the same order the patched
    Block forward consumes them.

    Inputs:
      block:            a Block already merged via :func:`merge_adaln_modulation`.
      emb_B_T_D:        (B, T, D) timestep embedding.
      adaln_lora_B_T_3D: optional (B, T, 3D) tensor added to each modulation
                        output (the per-step adaln-lora gate). May be None.
    """
    D, lora_dim, three_D = block._adaln_merged_dims
    B, T, _ = emb_B_T_D.shape
    h = block.adaln_modulation_silu(emb_B_T_D)

    if lora_dim is None:
        # No-lora path. merged Linear is (D → 9D), chunk into 9.
        out = block.adaln_modulation_l1(h)
        if adaln_lora_B_T_3D is not None:
            out = out + adaln_lora_B_T_3D.repeat(1, 1, 3)
        ssg = out.chunk(9, dim=-1)
        return (ssg[0], ssg[1], ssg[2]), (ssg[3], ssg[4], ssg[5]), (ssg[6], ssg[7], ssg[8])

    # Lora path.
    h1 = block.adaln_modulation_l1(h)                       # (B, T, 3*lora_dim)
    # Reshape (B, T, 3, lora_dim) → (3, B*T, lora_dim) for bmm.
    h1 = h1.view(B, T, 3, lora_dim).permute(2, 0, 1, 3).reshape(3, B * T, lora_dim)
    # bmm: (3, B*T, lora_dim) @ (3, lora_dim, 3D) → (3, B*T, 3D)
    out = torch.bmm(h1, block.adaln_modulation_l2_stack)
    out = out.view(3, B, T, three_D)
    if adaln_lora_B_T_3D is not None:
        # Broadcast (B, T, 3D) across the leading "3" axis.
        out = out + adaln_lora_B_T_3D.unsqueeze(0)
    ssg_self, ssg_cross, ssg_mlp = out.unbind(0)
    return (
        tuple(ssg_self.chunk(3, dim=-1)),
        tuple(ssg_cross.chunk(3, dim=-1)),
        tuple(ssg_mlp.chunk(3, dim=-1)),
    )
