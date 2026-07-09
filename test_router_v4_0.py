"""Tests for the v4.0 ReMoE router.

Verifies:
1. Output shapes match v3.0 contract (so it can drop into HSPMNBlock).
2. Gradient flows through the gate (v3.0's hard-threshold path did not).
3. Adaptive L1 coefficient moves toward target sparsity over multiple steps.
4. ALF-LB bias moves in the correct direction (away from imbalance).
5. Z-loss is finite and proportional to logit magnitudes.
6. Static-shape selection: indices.shape is independent of input data.
"""
import unittest
import torch
from router_v4_0 import ReMoERouter
from utils_v3_0 import seed_everything


def _train(m):
    return m.train()


def _infer(m):
    return m.train(False)


class TestReMoERouter(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.dim = 64
        self.B = 2
        self.S = 128

    def test_output_shapes(self):
        router = ReMoERouter(self.dim, target_sparsity=0.25)
        x = torch.randn(self.B, self.S, self.dim)
        out = router(x)
        k = int(self.S * 0.25)
        self.assertEqual(out.gate.shape, (self.B, self.S))
        self.assertEqual(out.indices.shape, (self.B, k))
        self.assertEqual(out.kv_indices.shape, (self.B, k + router.local_window))
        self.assertTrue(torch.isfinite(out.aux_loss))
        self.assertGreaterEqual(float(out.active_fraction), 0.0)
        self.assertLessEqual(float(out.active_fraction), 1.0)

    def test_gate_nonneg_and_differentiable(self):
        router = ReMoERouter(self.dim, target_sparsity=0.5)
        x = torch.randn(self.B, self.S, self.dim, requires_grad=True)
        out = router(x)
        self.assertGreaterEqual(float(out.gate.min()), 0.0)
        loss = out.gate.sum() + out.aux_loss
        loss.backward()
        self.assertIsNotNone(router.gate_proj.weight.grad)
        self.assertGreater(float(router.gate_proj.weight.grad.abs().sum()), 0.0)

    def test_static_indices_shape(self):
        router = ReMoERouter(self.dim, target_sparsity=0.2)
        x1 = torch.randn(self.B, self.S, self.dim)
        x2 = torch.randn(self.B, self.S, self.dim) * 100
        o1 = router(x1)
        o2 = router(x2)
        self.assertEqual(o1.indices.shape, o2.indices.shape)
        self.assertEqual(o1.kv_indices.shape, o2.kv_indices.shape)

    def test_local_window_covered(self):
        W = 16
        router = ReMoERouter(self.dim, target_sparsity=0.1, local_window=W)
        x = torch.randn(1, 64, self.dim)
        out = router(x)
        last_w = set(range(64 - W, 64))
        chosen = set(out.kv_indices[0].tolist())
        self.assertTrue(last_w.issubset(chosen),
                        f"Local window not fully covered: missing {last_w - chosen}")

    def test_alf_lb_bias_moves_toward_target(self):
        target = 0.5
        router = _train(ReMoERouter(self.dim, target_sparsity=target, bias_update_rate=0.05))
        with torch.no_grad():
            router.gate_proj.bias.fill_(-5.0)
        x = torch.randn(self.B, self.S, self.dim)

        bias_before = float(router.route_bias)
        for _ in range(20):
            _ = router(x)
        bias_after = float(router.route_bias)
        self.assertGreater(bias_after, bias_before,
                           f"ALF-LB bias did not move up: {bias_before} -> {bias_after}")

    def test_l1_coef_shrinks_when_too_sparse(self):
        target = 0.5
        router = _train(ReMoERouter(self.dim, target_sparsity=target,
                                    l1_coef_init=1.0, l1_adapt_rate=0.1))
        with torch.no_grad():
            router.gate_proj.bias.fill_(-5.0)
        x = torch.randn(self.B, self.S, self.dim)

        l1_before = float(router.l1_coef)
        for _ in range(10):
            _ = router(x)
        l1_after = float(router.l1_coef)
        self.assertLess(l1_after, l1_before,
                        f"L1 coef did not shrink under low activity: {l1_before} -> {l1_after}")

    def test_no_updates_in_inference(self):
        router = _infer(ReMoERouter(self.dim, target_sparsity=0.5))
        x = torch.randn(self.B, self.S, self.dim)
        bias_before = router.route_bias.clone()
        l1_before = router.l1_coef.clone()
        for _ in range(5):
            _ = router(x)
        self.assertTrue(torch.equal(bias_before, router.route_bias))
        self.assertTrue(torch.equal(l1_before, router.l1_coef))

    def test_z_loss_grows_with_logit_magnitude(self):
        router = ReMoERouter(self.dim, target_sparsity=0.5,
                             l1_coef_init=0.0, z_loss_coef=1.0)
        with torch.no_grad():
            router.gate_proj.weight.mul_(50.0)
        x = torch.randn(self.B, self.S, self.dim)
        out_big = router(x)

        with torch.no_grad():
            router.gate_proj.weight.div_(500.0)
        out_small = router(x)

        self.assertGreater(float(out_big.aux_loss), float(out_small.aux_loss))


if __name__ == "__main__":
    unittest.main()
