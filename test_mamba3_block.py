"""CPU smoke tests for Mamba-3 SISO reflexive port.

Verifies:
1. Forward shape contract matches GatedDeltaNetReflexive.
2. Gradients flow through learnable parameters.
3. Output is finite (no NaN / Inf) on long sequences.
4. Drop-in works inside HSPMNBlockV4 with reflexive='mamba3'.
"""
import torch

from mamba3_block import Mamba3SISOReflexive


def test_forward_shape():
    B, S, dim, H, KV, D = 2, 32, 128, 4, 2, 32
    block = Mamba3SISOReflexive(dim=dim, num_heads=H, num_kv_heads=KV,
                                head_dim=D, mlp_ratio=2, n_state=8,
                                use_fla=False)
    q = torch.randn(B, S, dim)
    k = torch.randn(B, S, KV * D)
    v = torch.randn(B, S, KV * D)
    out = block(q, k, v)
    assert out.shape == (B, S, dim), f"shape mismatch: {out.shape}"
    print("[1/4] forward shape OK", out.shape)


def test_gradient_flow():
    B, S, dim, H, KV, D = 1, 16, 64, 2, 1, 32
    block = Mamba3SISOReflexive(dim=dim, num_heads=H, num_kv_heads=KV,
                                head_dim=D, mlp_ratio=2, n_state=4,
                                use_fla=False)
    q = torch.randn(B, S, dim, requires_grad=True)
    k = torch.randn(B, S, KV * D, requires_grad=True)
    v = torch.randn(B, S, KV * D, requires_grad=True)
    out = block(q, k, v)
    out.sum().backward()
    for name in ["a_re", "a_im", "B_proj", "C_proj", "D_skip"]:
        param = getattr(block, name)
        if hasattr(param, "weight"):
            param = param.weight
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
    print("[2/4] gradient flow OK")


def test_long_sequence_stability():
    B, S, dim, H, KV, D = 1, 512, 64, 2, 1, 32
    block = Mamba3SISOReflexive(dim=dim, num_heads=H, num_kv_heads=KV,
                                head_dim=D, mlp_ratio=2, n_state=4,
                                use_fla=False)
    block.train(False)
    q = torch.randn(B, S, dim)
    k = torch.randn(B, S, KV * D)
    v = torch.randn(B, S, KV * D)
    with torch.no_grad():
        out = block(q, k, v)
    assert torch.isfinite(out).all(), "non-finite output on S=512"
    assert out.abs().max() < 1e3, f"output magnitude blew up: {out.abs().max()}"
    print("[3/4] long-seq stability OK", out.abs().max().item())


def test_drop_in_v4_block():
    from hspmn_v4_0 import HSPMNBlockV4
    from utils_v3_0 import HSPMNConfig

    cfg = HSPMNConfig(dim=64, num_heads=4, num_kv_heads=2,
                      mlp_ratio=2, max_seq_len=64, sparsity_k=0.25)
    block = HSPMNBlockV4(cfg, num_sink_tokens=4, reflexive="mamba3", attention="sqsk")
    block.train(False)
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        out, aux, _ = block(x)
    assert out.shape == (1, 16, 64), f"v4 block shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "non-finite v4 output with mamba3"
    print("[4/4] HSPMNBlockV4 + reflexive='mamba3' OK", out.shape)


if __name__ == "__main__":
    test_forward_shape()
    test_gradient_flow()
    test_long_sequence_stability()
    test_drop_in_v4_block()
    print("\nAll Mamba-3 tests pass (4/4).")
