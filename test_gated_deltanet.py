"""Tests for Gated DeltaNet reflexive stream."""
import unittest
import torch
from gated_deltanet import GatedDeltaNetReflexive, _chunked_gated_delta
from utils_v3_0 import seed_everything


class TestGatedDeltaNet(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.dim = 64
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = self.dim // self.num_heads

    def test_recurrence_at_alpha_1_beta_0_is_identity(self):
        """α=1, β=0 → state never changes, output q·0 = 0 (state starts at 0)."""
        B, S, H, D = 1, 4, 2, 8
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        alpha = torch.ones(B, S, H)
        beta = torch.zeros(B, S, H)
        out = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=2)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-5))

    def test_recurrence_writes_when_beta_positive(self):
        """β>0 means state should accumulate v·k^T contributions."""
        B, S, H, D = 1, 8, 2, 4
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        alpha = torch.ones(B, S, H)
        beta = torch.ones(B, S, H) * 0.5
        out = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=4)
        # After multiple writes, output magnitude should be nonzero.
        self.assertGreater(float(out.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(out).all())

    def test_chunkwise_matches_unchunked(self):
        """Chunking must not change the math (chunk=1 vs chunk=S vs chunk=4)."""
        B, S, H, D = 1, 16, 2, 8
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        alpha = torch.sigmoid(torch.randn(B, S, H))
        beta = torch.sigmoid(torch.randn(B, S, H))

        a = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=1)
        b = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=4)
        c = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=S)
        self.assertTrue(torch.allclose(a, b, atol=1e-4))
        self.assertTrue(torch.allclose(b, c, atol=1e-4))

    def test_reflexive_module_shape(self):
        m = GatedDeltaNetReflexive(self.dim, self.num_heads, self.num_kv_heads, self.head_dim, use_fla=False)
        kv_dim = self.num_kv_heads * self.head_dim
        q = torch.randn(2, 32, self.dim)
        k = torch.randn(2, 32, kv_dim)
        v = torch.randn(2, 32, kv_dim)
        out = m(q, k, v)
        self.assertEqual(out.shape, (2, 32, self.dim))
        self.assertTrue(torch.isfinite(out).all())

    def test_reflexive_module_gradients(self):
        m = GatedDeltaNetReflexive(self.dim, self.num_heads, self.num_kv_heads, self.head_dim, use_fla=False).train()
        kv_dim = self.num_kv_heads * self.head_dim
        q = torch.randn(1, 16, self.dim, requires_grad=True)
        k = torch.randn(1, 16, kv_dim, requires_grad=True)
        v = torch.randn(1, 16, kv_dim, requires_grad=True)
        out = m(q, k, v)
        out.sum().backward()
        # Both gates and SwiGLU weights must receive gradient.
        self.assertGreater(float(m.gate_alpha.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(m.gate_beta.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(m.gate_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(m.down_proj.weight.grad.abs().sum()), 0.0)

    def test_no_nan_on_long_sequence(self):
        """Stress: 256-token recurrence should not blow up."""
        m = GatedDeltaNetReflexive(self.dim, self.num_heads, self.num_kv_heads, self.head_dim,
                                   chunk_size=32, use_fla=False).train(False)
        kv_dim = self.num_kv_heads * self.head_dim
        q = torch.randn(1, 256, self.dim) * 2.0
        k = torch.randn(1, 256, kv_dim) * 2.0
        v = torch.randn(1, 256, kv_dim) * 2.0
        out = m(q, k, v)
        self.assertTrue(torch.isfinite(out).all(),
                        "Gated DeltaNet diverged on long sequence")


if __name__ == "__main__":
    unittest.main()
