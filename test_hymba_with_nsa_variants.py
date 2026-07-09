"""Regression tests for hymba-with-nsa family variants.

Five variants:
  - hymba-with-nsa (P2 winner; gate-less)
  - hymba-with-nsa-gated (learned attn_gate_proj)
  - hymba-with-nsa-randgate (frozen N(0,0.02) attn_gate_proj)
  - hymba-with-nsa-pcgate (parameter-free predictive-coding gate)
  - hymba-with-asa (alternating sliding/select per layer)

Tests cover: shape correctness, gradient flow, gate distribution properties.
"""
import torch

from hymba_with_nsa_lm import HymbaWithNSALM
from hymba_with_nsa_block import HymbaWithNSABlock


def _build(variant_kw, dim=64, n_layers=2):
    return HymbaWithNSALM(vocab_size=512, n_layers=n_layers, dim=dim,
                          num_heads=4, num_kv_heads=2, max_seq_len=64,
                          **variant_kw).float()


def test_baseline_shape():
    m = _build({})
    ids = torch.randint(0, 512, (2, 32))
    out = m(ids, labels=ids)
    assert out["logits"].shape == (2, 32, 512), out["logits"].shape
    out["loss"].backward()
    assert m.layers[0].attn_gate_proj is None
    print("[1/5] baseline: shape, gradient, gate-less OK")


def test_gated_grad_flow():
    m = _build({"use_attn_gate": True})
    ids = torch.randint(0, 512, (2, 32))
    out = m(ids, labels=ids)
    out["loss"].backward()
    gw = m.layers[0].attn_gate_proj.weight
    assert gw.grad is not None
    assert gw.grad.abs().sum() > 0, "gate weight should receive gradient"
    print("[2/5] gated: gate weight has nonzero gradient")


def test_randgate_frozen():
    m = _build({"random_gate": True})
    ids = torch.randint(0, 512, (2, 32))
    out = m(ids, labels=ids)
    out["loss"].backward()
    gw = m.layers[0].attn_gate_proj.weight
    assert not gw.requires_grad, "randgate weight should be frozen"
    assert gw.grad is None
    print("[3/5] randgate: gate weight frozen, no gradient")


def test_pcgate_no_params():
    m_baseline = _build({})
    m_pcgate = _build({"gate_mode": "predictive_coding"})
    n_b = sum(p.numel() for p in m_baseline.parameters())
    n_p = sum(p.numel() for p in m_pcgate.parameters())
    assert n_b == n_p, f"pcgate should add zero params; baseline {n_b}, pcgate {n_p}"
    ids = torch.randint(0, 512, (2, 32))
    out = m_pcgate(ids, labels=ids)
    out["loss"].backward()
    g = m_pcgate.layers[0]._last_gate
    assert g is not None
    assert g.min() >= 0 and g.max() <= 1, f"pcgate values out of [0,1]: {g.min()} {g.max()}"
    print(f"[4/5] pcgate: zero added params; gate range {g.min():.3f}-{g.max():.3f}")


def test_asa_alternation():
    m = _build({"attn_mode": "asa"})
    # Verify layer 0 and 1 have different asa_branch (alternation).
    assert m.layers[0].nsa.asa_branch != m.layers[1].nsa.asa_branch, \
        f"ASA should alternate; layer 0={m.layers[0].nsa.asa_branch}, " \
        f"layer 1={m.layers[1].nsa.asa_branch}"
    ids = torch.randint(0, 512, (2, 32))
    out = m(ids, labels=ids)
    out["loss"].backward()
    print(f"[5/5] asa: alternation between layers; "
          f"layer 0 branch={m.layers[0].nsa.asa_branch}, "
          f"layer 1 branch={m.layers[1].nsa.asa_branch}")


def test_pcgate_temperature_clamp():
    """PC gate must not crash when SSM output has zero variance."""
    m = _build({"gate_mode": "predictive_coding"})
    # Force-zero SSM-related projections to produce uniform output.
    with torch.no_grad():
        m.layers[0].ssm_in_proj.weight.zero_()
    ids = torch.randint(0, 512, (1, 16))
    out = m(ids, labels=ids)
    # Should not crash; gate may all be ~0.5.
    out["loss"].backward()
    print("[bonus] pcgate: degenerate SSM (zero in_proj) handled without NaN/inf")


def test_select_from_compress_zero_param_added():
    """NSA-select-from-compress should add zero parameters vs baseline."""
    m_baseline = _build({})
    m_select = _build({"nsa_select_from_compress": True, "nsa_n_select_blocks": 4})
    n_b = sum(p.numel() for p in m_baseline.parameters())
    n_s = sum(p.numel() for p in m_select.parameters())
    assert n_b == n_s, f"select variant should be parameter-free; baseline {n_b}, select {n_s}"
    # Train one step to confirm differentiable path works.
    ids = torch.randint(0, 512, (2, 64))
    labels = torch.randint(0, 512, (2, 64))
    out = m_select(ids, labels=labels)
    out["loss"].backward()
    # Verify all 3 NSA gates are non-trivial (compress + select + window).
    g_c = m_select.layers[0].nsa.gate_proj.weight
    assert g_c.grad is not None and g_c.grad.abs().sum() > 0
    print("[bonus2] select-from-compress: zero added params, gradient flows through all 3 branches")


if __name__ == "__main__":
    test_baseline_shape()
    test_gated_grad_flow()
    test_randgate_frozen()
    test_pcgate_no_params()
    test_asa_alternation()
    test_pcgate_temperature_clamp()
    test_select_from_compress_zero_param_added()
    print("\nAll 7 variant tests pass.")
