"""Tests for NSA triple-branch attention."""
import unittest
import torch
from nsa_attention import NSAAttention, _compress_branch, _select_branch, _sliding_window_attention
from utils_v3_0 import seed_everything


class TestNSA(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.B, self.H, self.S, self.D = 2, 4, 256, 32

    def test_compress_branch_shape_and_finite(self):
        q = torch.randn(self.B, self.H, self.S, self.D)
        k = torch.randn(self.B, self.H, self.S, self.D)
        v = torch.randn(self.B, self.H, self.S, self.D)
        out = _compress_branch(q, k, v, block_size=32, stride=16, scale=1.0)
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_compress_branch_short_sequence(self):
        """S < block_size → returns zero (no compression possible)."""
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        v = torch.randn(1, 2, 16, 8)
        out = _compress_branch(q, k, v, block_size=32, stride=16, scale=1.0)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_select_branch_zero_when_indices_none(self):
        q = torch.randn(self.B, self.H, self.S, self.D)
        k = torch.randn(self.B, self.H, self.S, self.D)
        v = torch.randn(self.B, self.H, self.S, self.D)
        out = _select_branch(q, k, v, None, scale=1.0)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_select_branch_with_indices(self):
        q = torch.randn(self.B, self.H, self.S, self.D)
        k = torch.randn(self.B, self.H, self.S, self.D)
        v = torch.randn(self.B, self.H, self.S, self.D)
        K = 16
        idx = torch.stack([torch.sort(torch.randperm(self.S)[:K])[0] for _ in range(self.B)])
        out = _select_branch(q, k, v, idx, scale=1.0)
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_window_branch_causal(self):
        """A token at position s should only attend to positions [s-W+1, s]."""
        q = torch.randn(1, 1, 8, 4)
        k = torch.randn(1, 1, 8, 4)
        v = torch.randn(1, 1, 8, 4)
        out = _sliding_window_attention(q, k, v, window_size=4, scale=0.5)
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_full_nsa_module(self):
        nsa = NSAAttention(self.H, self.D, compress_block_size=32, compress_stride=16, window_size=64)
        q = torch.randn(self.B, self.H, self.S, self.D)
        k = torch.randn(self.B, self.H, self.S, self.D)
        v = torch.randn(self.B, self.H, self.S, self.D)
        K = 32
        idx = torch.stack([torch.sort(torch.randperm(self.S)[:K])[0] for _ in range(self.B)])
        result = nsa(q, k, v, select_indices=idx)
        self.assertEqual(result.out.shape, (self.B, self.H, self.S, self.D))
        self.assertTrue(torch.isfinite(result.out).all())
        # Gates are sigmoid → in (0, 1).
        self.assertTrue((result.gate_compress >= 0).all())
        self.assertTrue((result.gate_compress <= 1).all())
        self.assertTrue((result.gate_window >= 0).all())
        self.assertTrue((result.gate_window <= 1).all())

    def test_nsa_gradients(self):
        nsa = NSAAttention(self.H, self.D, compress_block_size=32, compress_stride=16, window_size=64).train()
        q = torch.randn(self.B, self.H, self.S, self.D, requires_grad=True)
        k = torch.randn(self.B, self.H, self.S, self.D, requires_grad=True)
        v = torch.randn(self.B, self.H, self.S, self.D, requires_grad=True)
        K = 32
        idx = torch.stack([torch.sort(torch.randperm(self.S)[:K])[0] for _ in range(self.B)])
        result = nsa(q, k, v, select_indices=idx)
        result.out.sum().backward()
        # Gate weights, q, k, v all must receive gradient.
        self.assertGreater(float(nsa.gate_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(q.grad.abs().sum()), 0.0)
        self.assertGreater(float(k.grad.abs().sum()), 0.0)
        self.assertGreater(float(v.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
