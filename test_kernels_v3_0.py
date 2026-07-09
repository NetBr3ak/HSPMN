"""
Proprietary / All Rights Reserved - Non-Commercial Use Only
Source-available for portfolio viewing only. Commercial use, unauthorized modification, reproduction, or distribution is strictly prohibited. All rights reserved.
"""
import unittest
import torch
import torch.nn.functional as F
from kernels_v3_0 import sparse_query_sparse_key_attention
from utils_v3_0 import setup_logging

logger = setup_logging(__name__)


class TestTritonKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_sqsk_attention_correctness(self):
        """Verifies SQSK Triton kernel against a masked PyTorch reference."""
        B, H, S, D = 2, 4, 128, 64
        K_active_q, K_active_k = 32, 64

        q = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)

        q_indices = torch.stack([torch.randperm(S, device=self.device)[:K_active_q] for _ in range(B)])
        q_indices = torch.sort(q_indices, dim=1)[0]
        k_indices = torch.stack([torch.randperm(S, device=self.device)[:K_active_k] for _ in range(B)])
        k_indices = torch.sort(k_indices, dim=1)[0]

        # Reference: gather, compute SDPA with causal mask
        qi = q_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        ki = k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        q_sel = torch.gather(q, 1, qi).permute(0, 2, 1, 3)
        k_sel = torch.gather(k, 1, ki).permute(0, 2, 1, 3)
        v_sel = torch.gather(v, 1, ki).permute(0, 2, 1, 3)

        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q_sel, k_sel.transpose(-2, -1)) * scale
        causal = k_indices.unsqueeze(1).unsqueeze(2) <= q_indices.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~causal, float("-inf"))
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_sel).permute(0, 2, 1, 3)

        # Triton kernel
        out_triton = sparse_query_sparse_key_attention(
            torch.gather(q, 1, qi).contiguous(),
            torch.gather(k, 1, ki).contiguous(),
            torch.gather(v, 1, ki).contiguous(),
            q_indices.contiguous(),
            k_indices.contiguous(),
        )

        max_diff = (out_triton - out_ref).abs().max().item()
        logger.info(f"Max diff Triton vs PyTorch: {max_diff}")
        self.assertTrue(torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2),
                        f"Triton output mismatch. Max diff: {max_diff}")

    def test_sqsk_handles_unaligned_q_count(self):
        """Reproduces the empty-tail-block NaN: K_active_q not divisible by BLOCK_M."""
        B, H, S, D = 1, 4, 256, 64
        K_active_q, K_active_k = 70, 64  # 70 % 64 != 0 and 70 % 128 != 0

        q = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)

        q_indices = torch.stack([torch.randperm(S, device=self.device)[:K_active_q] for _ in range(B)])
        q_indices = torch.sort(q_indices, dim=1)[0]
        k_indices = torch.stack([torch.randperm(S, device=self.device)[:K_active_k] for _ in range(B)])
        k_indices = torch.sort(k_indices, dim=1)[0]

        qi = q_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        ki = k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)

        out_triton = sparse_query_sparse_key_attention(
            torch.gather(q, 1, qi).contiguous(),
            torch.gather(k, 1, ki).contiguous(),
            torch.gather(v, 1, ki).contiguous(),
            q_indices.contiguous(),
            k_indices.contiguous(),
        )
        self.assertFalse(torch.isnan(out_triton).any(),
                         "Kernel produced NaN on unaligned tail block")
        self.assertTrue(torch.isfinite(out_triton).all())


if __name__ == "__main__":
    unittest.main()
