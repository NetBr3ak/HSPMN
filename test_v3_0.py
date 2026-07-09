"""
Proprietary / All Rights Reserved - Non-Commercial Use Only
Source-available for portfolio viewing only. Commercial use, unauthorized modification, reproduction, or distribution is strictly prohibited. All rights reserved.
"""
import unittest
import warnings
import torch
from hspmn_v3_0 import HSPMNBlock, TopKRouter
from utils_v3_0 import HSPMNConfig, setup_logging, seed_everything

# Suppress external library warnings
warnings.filterwarnings("ignore", message=".*HF_HUB_ENABLE_HF_TRANSFER.*")

logger = setup_logging(__name__)


class TestHSPMNv3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        logger.info("Running on CPU for isolated code verification")

        # Patch flex_attention to avoid torch.compile hangs on CPU.
        import hspmn_v3_0
        cls._original_flex = hspmn_v3_0.flex_attention
        cls._original_mask = hspmn_v3_0.create_block_mask

        def _flex_stub(q, k, v, block_mask=None, enable_gqa=False):
            if q.shape[1] != k.shape[1]:
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

        hspmn_v3_0.flex_attention = _flex_stub
        hspmn_v3_0.create_block_mask = lambda *args, **kwargs: None

    @classmethod
    def tearDownClass(cls):
        import hspmn_v3_0
        hspmn_v3_0.flex_attention = cls._original_flex
        hspmn_v3_0.create_block_mask = cls._original_mask

    def setUp(self):
        seed_everything(42)
        self.config = HSPMNConfig(dim=64, num_heads=4, num_kv_heads=2, sparsity_k=0.5)

    def test_router_entropy_loss(self):
        router = TopKRouter(self.config.dim, target_sparsity=0.5).to(self.device)
        x = torch.randn(2, 128, self.config.dim, device=self.device)
        out = router(x)
        self.assertIsNotNone(out.aux_loss)
        self.assertTrue(out.aux_loss.isfinite())
        self.assertTrue(out.aux_loss != 0)
        self.assertEqual(out.mask.shape, (2, 128))
        out.aux_loss.backward()
        self.assertIsNotNone(router.gate.weight.grad)

    def test_aux_loss_bounds(self):
        model = HSPMNBlock(self.config).to(self.device).train()
        x = torch.randn(2, 64, self.config.dim, device=self.device)
        _, aux_loss, _ = model(x)
        self.assertTrue(torch.isfinite(aux_loss))
        self.assertTrue(abs(aux_loss.item()) > 0.0)

    def test_reflexive_stream_correctness(self):
        model = HSPMNBlock(self.config).to(self.device)
        x = torch.randn(2, 64, self.config.dim, device=self.device)
        x_norm = model.norm(x)
        q = model.q_proj(x_norm)
        k = model.k_proj(x_norm)
        v = model.v_proj(x_norm)
        out = model.reflexive(q, k, v)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_gqa_shapes_with_kv_cache(self):
        cfg = HSPMNConfig(dim=64, num_heads=4, num_kv_heads=1)
        model = HSPMNBlock(cfg).to(self.device).eval()
        self.assertEqual(model.num_kv_heads, 1)

        x = torch.randn(1, 16, cfg.dim, device=self.device)
        out1, _, past_kv = model(x)
        self.assertEqual(out1.shape, (1, 16, 64))

        x2 = torch.randn(1, 1, cfg.dim, device=self.device)
        out2, _, new_kv = model(x2, past_key_values=past_kv)
        self.assertEqual(out2.shape, (1, 1, 64))
        self.assertEqual(new_kv[0].shape[2], past_kv[0].shape[2] + 1)

    def test_incremental_decode_runs(self):
        # NOTE: NOT a parity test. Patched flex_attention on CPU is non-causal,
        # so full-vs-incremental values cannot match. Real parity requires
        # CUDA + FlexAttention with the causal block_mask. This only checks
        # shape + finiteness of incremental decoding.
        model = HSPMNBlock(self.config).to(self.device).eval()
        x_full = torch.randn(1, 5, self.config.dim, device=self.device)
        out_full, _, _ = model(x_full)

        past_kv = None
        out_inc_list = []
        for i in range(5):
            out_step, _, past_kv = model(x_full[:, i:i + 1, :], past_key_values=past_kv)
            out_inc_list.append(out_step)
        out_inc = torch.cat(out_inc_list, dim=1)

        self.assertEqual(out_full.shape, out_inc.shape)
        self.assertTrue(torch.isfinite(out_inc).all())

    def test_forward_pass_integration(self):
        model = HSPMNBlock(self.config).to(self.device)
        x = torch.randn(1, 128, self.config.dim, device=self.device)
        out, aux, cache = model(x)
        self.assertEqual(out.shape, (1, 128, self.config.dim))
        self.assertTrue(torch.is_tensor(aux))
        self.assertEqual(len(cache), 3)

    def test_hf_wrapper_e2e(self):
        from hspmn_hf_wrapper import HSPMNWrapperConfig, HSPMNWrapperModel
        hf_config = HSPMNWrapperConfig(vocab_size=512, dim=64, num_heads=4, num_kv_heads=2)
        model = HSPMNWrapperModel(hf_config)
        input_ids = torch.randint(0, hf_config.vocab_size, (1, 16))
        labels = torch.randint(0, hf_config.vocab_size, (1, 16))
        out = model(input_ids=input_ids, labels=labels, use_cache=True)
        self.assertEqual(out.logits.shape, (1, 16, hf_config.vocab_size))
        self.assertTrue(out.loss.isfinite())
        self.assertTrue(hasattr(out, "past_key_values"))


if __name__ == "__main__":
    unittest.main()
