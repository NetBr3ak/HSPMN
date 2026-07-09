"""Tests for HSPMNBlockV4 - decoupled QKV + ReMoE router integration."""

import unittest
import warnings
import torch

# Patch flex_attention for CPU tests (FlexAttention requires CUDA + compile).
import hspmn_v4_0
from hspmn_v4_0 import HSPMNBlockV4
from utils_v3_0 import HSPMNConfig, seed_everything

_orig_flex = hspmn_v4_0.flex_attention
_orig_mask = hspmn_v4_0.create_block_mask


def _flex_stub(q, k, v, block_mask=None, enable_gqa=False):
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)


hspmn_v4_0.flex_attention = _flex_stub
hspmn_v4_0.create_block_mask = lambda *a, **kw: None

warnings.filterwarnings("ignore")


class TestHSPMNBlockV4(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.cfg = HSPMNConfig(dim=64, num_heads=4, num_kv_heads=2, sparsity_k=0.5)

    def test_forward_shape(self):
        m = HSPMNBlockV4(self.cfg, num_sink_tokens=4)
        x = torch.randn(2, 32, self.cfg.dim)
        out, aux, kv = m(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(aux))
        self.assertEqual(len(kv), 3)

    def test_decoupled_qkv_have_separate_grads(self):
        """The whole point of Phase 2a: refl-Q and ctx-Q must have INDEPENDENT gradients."""
        m = HSPMNBlockV4(self.cfg, num_sink_tokens=4).train()
        x = torch.randn(1, 16, self.cfg.dim)
        out, aux, _ = m(x)
        loss = out.sum() + aux
        loss.backward()
        # Both projections must receive gradient (proves both streams ran).
        self.assertIsNotNone(m.q_proj_refl.weight.grad)
        self.assertIsNotNone(m.q_proj_ctx.weight.grad)
        self.assertGreater(float(m.q_proj_refl.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(m.q_proj_ctx.weight.grad.abs().sum()), 0.0)
        # Gradients must not be identical (would indicate a parameter-tying bug).
        self.assertFalse(
            torch.allclose(m.q_proj_refl.weight.grad, m.q_proj_ctx.weight.grad)
        )

    def test_router_grad_via_gate(self):
        """Gradient must flow through router gate weights (absorption defense)."""
        m = HSPMNBlockV4(self.cfg, num_sink_tokens=4).train()
        x = torch.randn(1, 16, self.cfg.dim)
        out, aux, _ = m(x)
        (out.sum() + aux).backward()
        self.assertIsNotNone(m.router.gate_proj.weight.grad)
        self.assertGreater(float(m.router.gate_proj.weight.grad.abs().sum()), 0.0)

    def test_kv_cache_grows_correctly(self):
        m = HSPMNBlockV4(self.cfg, num_sink_tokens=4).train(False)
        x1 = torch.randn(1, 8, self.cfg.dim)
        out1, _, kv1 = m(x1)
        x2 = torch.randn(1, 1, self.cfg.dim)
        out2, _, kv2 = m(x2, past_key_values=kv1)
        self.assertEqual(out2.shape, (1, 1, self.cfg.dim))
        # KV cache grew by 1 token's sparse selection (but never beyond k+W).
        self.assertGreaterEqual(kv2[0].shape[2], kv1[0].shape[2])

    def test_sink_count_respected(self):
        m = HSPMNBlockV4(self.cfg, num_sink_tokens=8)
        self.assertEqual(m.sink_tokens.shape, (1, 8, self.cfg.dim))

    def test_random_gate_baseline_runs(self):
        """Absorption diagnostic: replace ReMoE gate with random; block must still run."""
        m = HSPMNBlockV4(self.cfg, num_sink_tokens=4).train(False)
        with torch.no_grad():
            m.router.gate_proj.weight.normal_(0, 1.0)
            m.router.gate_proj.bias.fill_(0.0)
        x = torch.randn(1, 32, self.cfg.dim)
        out, _, _ = m(x)
        self.assertTrue(torch.isfinite(out).all())


def _restore():
    hspmn_v4_0.flex_attention = _orig_flex
    hspmn_v4_0.create_block_mask = _orig_mask


if __name__ == "__main__":
    try:
        unittest.main(exit=False)
    finally:
        _restore()
