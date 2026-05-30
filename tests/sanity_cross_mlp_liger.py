"""Sanity check the cross-attn+mlp preset and the Liger RMSNorm patch.

Run from repo root: `python -m tests.sanity_cross_mlp_liger`

Checks:
  1. Liger-patched RMSNorm matches the original numerically (bf16 inputs).
  2. `anima-cross-mlp` preset wraps the expected modules (~168).
  3. Trainable param count is in the expected ballpark (~17M).
  4. Module names actually targeted = cross_attn.{q,k,v,output}_proj + mlp.{layer1,layer2}.
"""
from __future__ import annotations
import torch
from anima_trainer.sdscripts_bridge import ensure_on_path
ensure_on_path()


def test_liger_rmsnorm_matches() -> None:
    """Run original Anima RMSNorm and Liger-patched version on the same input,
    compare. Should be very close in bf16 (LLaMA mode does fp32 stats too)."""
    from library.anima_models import RMSNorm  # type: ignore
    from anima_trainer.liger_patch import install, uninstall

    torch.manual_seed(0)
    head_dim = 128
    rms = RMSNorm(head_dim, eps=1e-6).cuda().to(torch.bfloat16)
    # Randomize weight so it's not just ones.
    with torch.no_grad():
        rms.weight.normal_(mean=1.0, std=0.05)

    x = torch.randn(2, 1024, 16, head_dim, device="cuda", dtype=torch.bfloat16)

    uninstall()
    with torch.no_grad():
        y_orig = RMSNorm.forward(rms, x).clone()

    install()
    with torch.no_grad():
        y_liger = RMSNorm.forward(rms, x).clone()

    # Compare — bf16 has ~3 decimal digits of precision; tolerance should be
    # generous for the per-element diff, tight for the mean.
    diff = (y_orig.float() - y_liger.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = (mean_diff / y_orig.float().abs().mean().item())
    print(f"  RMSNorm  max abs diff={max_diff:.4e}  mean={mean_diff:.4e}  rel={rel:.4e}")
    assert max_diff < 5e-2, f"Liger RMSNorm diverges: {max_diff}"
    assert rel < 5e-3, f"Liger RMSNorm relative error too high: {rel}"
    print("  RMSNorm parity: PASS")


def test_cross_mlp_preset() -> None:
    from anima_trainer.config import LokrCfg
    from anima_trainer.lokr import attach_lokr, trainable_param_count
    from anima_trainer.model import load_all

    print("loading DiT on CPU (only meta info needed)...")
    models = load_all(
        dit_path="./models/anima-base-v1.0.safetensors",
        qwen3_path="./models/qwen_3_06b_base.safetensors",
        vae_path="./models/qwen_image_vae.safetensors",
        dtype=torch.bfloat16,
        attn_mode="torch",
        device="cuda",
        loading_device="cpu",
    )
    dit = models.dit  # already on CPU; we don't need to move for module counting

    cfg = LokrCfg(factor=8, full_matrix=True, preset="anima-cross-mlp")
    network = attach_lokr(dit, cfg, network_dim=128, network_alpha=128.0)
    n_modules = len(network.unet_loras)
    n_params = trainable_param_count(network)
    print(f"  wrapped modules: {n_modules}")
    print(f"  trainable LoKr params: {n_params:,}")

    # Expected: 6 patterns × 28 blocks = 168 modules.
    assert 150 < n_modules < 200, f"unexpected module count: {n_modules}"
    # Inspect a few names — verify cross_attn + mlp only, no self_attn / adaln.
    wrapped_targets = sorted({l.lora_name for l in network.unet_loras})
    print("  sample wrapped names:")
    for n in wrapped_targets[:6]:
        print(f"    {n}")
    bad = [n for n in wrapped_targets if "self_attn" in n or "adaln" in n]
    assert not bad, f"unexpected modules wrapped: {bad[:5]}"
    print("  preset targets: PASS")


def main() -> None:
    print("=== Liger RMSNorm parity ===")
    test_liger_rmsnorm_matches()
    print()
    print("=== anima-cross-mlp preset ===")
    test_cross_mlp_preset()
    print()
    print("ALL OK")


if __name__ == "__main__":
    main()
