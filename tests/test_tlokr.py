from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from anima_trainer.convergence import _sample_seed
from anima_trainer.lokr_patch import install
from anima_trainer.tlokr import (
    _capture_timestep_callable,
    _current_state,
    _install_block_checkpoint_patch,
    active_rank_for_timestep,
    clear_timestep,
    convert_network,
    install_root_timestep_hooks,
    set_timestep,
)
from anima_trainer.tlokr_kernels import rank_mix, rank_mix_wgrad
from lycoris.modules.lokr import LokrModule


def _make_module() -> tuple[torch.nn.Module, LokrModule, torch.nn.Linear]:
    torch.manual_seed(7)
    original = torch.nn.Linear(16, 16, bias=True)
    original.requires_grad_(False)
    adapter = LokrModule(
        "test_lokr",
        original,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        factor=2,
        full_matrix=True,
    )
    network = torch.nn.Module()
    network.add_module("test_lokr", adapter)
    install()
    convert_network(network, rank=4, min_rank_ratio=0.5)
    return network, adapter, original


class ConvergenceSeedTests(unittest.TestCase):
    def test_sample_seed_is_stable_and_source_specific(self) -> None:
        self.assertEqual(_sample_seed(42, "/data/a.png"), 910253215)
        self.assertEqual(
            _sample_seed(42, "/data/a.png"),
            _sample_seed(42, "/data/a.png"),
        )
        self.assertNotEqual(
            _sample_seed(42, "/data/a.png"),
            _sample_seed(42, "/data/b.png"),
        )


class TLoKrScheduleTests(unittest.TestCase):
    def test_schedule_endpoints_and_midpoint(self) -> None:
        self.assertEqual(
            active_rank_for_timestep(1.0, max_rank=128, min_rank=64), 64
        )
        self.assertEqual(
            active_rank_for_timestep(0.5, max_rank=128, min_rank=64), 96
        )
        self.assertEqual(
            active_rank_for_timestep(0.0, max_rank=128, min_rank=64), 128
        )

    def test_schedule_clamps_out_of_range_flow_timesteps(self) -> None:
        self.assertEqual(
            active_rank_for_timestep(2.0, max_rank=8, min_rank=4), 4
        )
        self.assertEqual(
            active_rank_for_timestep(-1.0, max_rank=8, min_rank=4), 8
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton parity")
class TLoKrKernelCudaTests(unittest.TestCase):
    def test_fused_rank_mix_and_wgrad_match_torch(self) -> None:
        torch.manual_seed(11)
        hidden = torch.randn(
            2, 32, 8, 128, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)
        mask[0, :73] = 1
        mask[1, :111] = 1
        masked = hidden * mask[:, None, None, :]

        actual = rank_mix(hidden, weight, mask)
        expected = torch.matmul(weight, masked)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=6.25e-2)

        grad = torch.randn_like(actual)
        actual_t = rank_mix(grad, weight, mask, transpose=True)
        expected_t = torch.matmul(weight.T, grad * mask[:, None, None, :])
        torch.testing.assert_close(actual_t, expected_t, rtol=2e-2, atol=6.25e-2)

        actual_wgrad = rank_mix_wgrad(grad, hidden, mask)
        expected_wgrad = torch.einsum(
            "bnir,bnjr->ij", grad.float(), masked.float()
        )
        torch.testing.assert_close(
            actual_wgrad,
            expected_wgrad,
            rtol=1e-2,
            atol=2e-1,
        )


class TLoKrForwardTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_timestep()

    def test_zero_initialization_is_exact_base_output_at_every_timestep(self) -> None:
        _, adapter, original = _make_module()
        x = torch.randn(2, 3, 16)
        expected = original._lycoris_original_forward(x) if hasattr(
            original, "_lycoris_original_forward"
        ) else F.linear(x, original.weight, original.bias)
        set_timestep(torch.tensor([0.0, 1.0]))
        actual = adapter(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_structured_batch_forward_matches_explicit_kronecker_weights(self) -> None:
        _, adapter, original = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)

        x = torch.randn(2, 5, 16)
        timesteps = torch.tensor([0.0, 1.0])
        set_timestep(timesteps)
        actual = adapter(x)

        expected_rows = []
        for batch_idx, timestep in enumerate(timesteps.tolist()):
            active = active_rank_for_timestep(
                timestep,
                max_rank=adapter._tlokr_rank,
                min_rank=adapter._tlokr_min_rank,
            )
            w2 = (
                adapter.lokr_w2_a[:, :active]
                @ adapter.lokr_w2_b[:active, :]
            )
            delta_weight = torch.kron(adapter.lokr_w1, w2) * adapter.scale
            expected_rows.append(
                F.linear(
                    x[batch_idx],
                    original.weight + delta_weight,
                    original.bias,
                )
            )
        expected = torch.stack(expected_rows)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_uniform_sliced_path_matches_masked_path(self) -> None:
        _, adapter, _ = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)
        x = torch.randn(1, 4, 16)
        t = torch.tensor([0.375])

        set_timestep(t)
        masked = adapter(x)
        set_timestep(t, uniform_timestep=0.375)
        sliced = adapter(x)
        torch.testing.assert_close(sliced, masked, rtol=1e-5, atol=1e-6)

    def test_manual_backward_matches_explicit_kronecker_autograd(self) -> None:
        _, adapter, original = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)

        timesteps = torch.tensor([0.2, 0.9])
        x = torch.randn(2, 3, 16, requires_grad=True)
        probe = torch.randn(2, 3, 16)
        set_timestep(timesteps)
        (adapter(x) * probe).sum().backward()
        actual = (
            x.grad.detach().clone(),
            adapter.lokr_w1.grad.detach().clone(),
            adapter.lokr_w2_a.grad.detach().clone(),
            adapter.lokr_w2_b.grad.detach().clone(),
        )

        x_ref = x.detach().clone().requires_grad_(True)
        w1 = adapter.lokr_w1.detach().clone().requires_grad_(True)
        w2_a = adapter.lokr_w2_a.detach().clone().requires_grad_(True)
        w2_b = adapter.lokr_w2_b.detach().clone().requires_grad_(True)
        rows = []
        for batch_idx, timestep in enumerate(timesteps.tolist()):
            active = active_rank_for_timestep(
                timestep,
                max_rank=adapter._tlokr_rank,
                min_rank=adapter._tlokr_min_rank,
            )
            w2 = w2_a[:, :active] @ w2_b[:active]
            weight = original.weight + torch.kron(w1, w2) * adapter.scale
            rows.append(F.linear(x_ref[batch_idx], weight, original.bias))
        (torch.stack(rows) * probe).sum().backward()
        expected = (x_ref.grad, w1.grad, w2_a.grad, w2_b.grad)

        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_missing_timestep_fails_loudly(self) -> None:
        _, adapter, _ = _make_module()
        with self.assertRaisesRegex(RuntimeError, "no timestep context"):
            adapter(torch.randn(1, 16))

    def test_root_hook_derives_positional_timestep_and_cleans_up(self) -> None:
        _, adapter, _ = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)

        class ToyDiT(torch.nn.Module):
            def __init__(self, layer: torch.nn.Module) -> None:
                super().__init__()
                self.layer = layer

            def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
                return self.layer(x)

        dit = ToyDiT(adapter)
        install_root_timestep_hooks(dit)
        x = torch.randn(2, 3, 16)
        timesteps = torch.tensor([0.25, 0.75])

        set_timestep(timesteps)
        expected = adapter(x)
        clear_timestep()
        actual = dit(x, timesteps)
        torch.testing.assert_close(actual, expected)

        with self.assertRaisesRegex(RuntimeError, "no timestep context"):
            adapter(x)

    def test_root_hook_accepts_keyword_and_preserves_explicit_context(self) -> None:
        _, adapter, _ = _make_module()

        class ToyDiT(torch.nn.Module):
            def __init__(self, layer: torch.nn.Module) -> None:
                super().__init__()
                self.layer = layer

            def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
                return self.layer(x)

        dit = ToyDiT(adapter)
        install_root_timestep_hooks(dit)
        install_root_timestep_hooks(dit)  # idempotent
        x = torch.randn(1, 16)
        explicit = torch.tensor([0.0])
        set_timestep(explicit)
        dit(x=x, timesteps=torch.tensor([1.0]))

        # The hook must not clear the explicit training context.
        adapter(x)

    def test_root_hook_cleans_up_after_failed_forward(self) -> None:
        _, adapter, _ = _make_module()

        class FailingDiT(torch.nn.Module):
            def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
                adapter(x)
                raise ValueError("probe")

        dit = FailingDiT()
        install_root_timestep_hooks(dit)
        with self.assertRaisesRegex(ValueError, "probe"):
            dit(torch.randn(1, 16), torch.tensor([0.5]))
        with self.assertRaisesRegex(RuntimeError, "no timestep context"):
            adapter(torch.randn(1, 16))

    def test_captured_context_survives_checkpoint_recomputation(self) -> None:
        _, adapter, _ = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)
        x = torch.randn(2, 3, 16, requires_grad=True)
        timesteps = torch.tensor([0.2, 0.8])

        set_timestep(timesteps)
        contextual_adapter = _capture_timestep_callable(
            adapter,
            _current_state(),
        )
        output = checkpoint(contextual_adapter, x, use_reentrant=False)
        # Root execution has ended before non-reentrant checkpointing reruns
        # the adapter in backward.
        clear_timestep()
        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_anima_block_checkpoint_retains_timestep_context(self) -> None:
        from anima_trainer.sdscripts_bridge import ensure_on_path

        ensure_on_path()
        from library.anima_models import Block

        _install_block_checkpoint_patch()
        _, adapter, _ = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)

        # Build the smallest possible Block shell around a T-LoKr adapter. Its
        # real forward owns checkpoint selection; the instance _forward keeps
        # this regression independent of attention/AdaLN compute.
        block = Block.__new__(Block)
        torch.nn.Module.__init__(block)
        block.adapter = adapter
        block.gradient_checkpointing = True
        block.unsloth_offload_checkpointing = False
        block.cpu_offload_checkpointing = False
        block.train()

        def adapter_forward(x: torch.Tensor, *_args):
            return block.adapter(x)

        object.__setattr__(block, "_forward", adapter_forward)
        x = torch.randn(2, 3, 16, requires_grad=True)
        set_timestep(torch.tensor([0.3, 0.7]))
        output = block(x, None, None, None, False, None, None, None)
        clear_timestep()
        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_anima_execution_boundary_derives_and_cleans_context(self) -> None:
        _, adapter, _ = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_(mean=0.0, std=0.2)

        class ToyAnima(torch.nn.Module):
            def __init__(self, layer: torch.nn.Module) -> None:
                super().__init__()
                self.layer = layer

            def forward_mini_train_dit(
                self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
            ):
                return self.layer(x)

            def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
                return self.forward_mini_train_dit(x, timesteps)

        dit = ToyAnima(adapter)
        install_root_timestep_hooks(dit)
        x = torch.randn(2, 3, 16)
        timesteps = torch.tensor([0.1, 0.9])

        set_timestep(timesteps)
        expected = adapter(x)
        clear_timestep()
        torch.testing.assert_close(dit(x, timesteps), expected)
        torch.testing.assert_close(
            dit.forward_mini_train_dit(x, timesteps),
            expected,
        )
        with self.assertRaisesRegex(RuntimeError, "no timestep context"):
            adapter(x)

    def test_state_dict_round_trip_preserves_tlokr_topology(self) -> None:
        network, adapter, _ = _make_module()
        with torch.no_grad():
            adapter.lokr_w2_b.normal_()
        state = network.state_dict()
        self.assertIn("test_lokr.lokr_w2_a", state)
        self.assertIn("test_lokr.lokr_w2_b", state)
        self.assertIn("test_lokr.tlokr_schedule", state)
        self.assertNotIn("test_lokr.lokr_w2", state)

        restored, _, _ = _make_module()
        restored.load_state_dict(state, strict=True)
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_state_dict_rejects_a_different_timestep_schedule(self) -> None:
        network, _, _ = _make_module()
        state = network.state_dict()
        state["test_lokr.tlokr_schedule"] = torch.tensor([1, 4, 3])

        restored, _, _ = _make_module()
        with self.assertRaisesRegex(RuntimeError, "is incompatible"):
            restored.load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
