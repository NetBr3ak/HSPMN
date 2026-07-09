"""Unit tests for the Phase-1 synthetic state-tracking benchmarks."""

import torch

from synthetic_mqar import build_mqar_batch
from synthetic_dyck import build_dyck2_batch, sample_dyck2, VOCAB_DYCK2
from synthetic_parity import build_parity_batch, VOCAB_PARITY


def test_mqar_shapes():
    """ids/labels have shape [B, 2N+2M]; labels are -100 except after queries."""
    B, N, M = 4, 8, 3
    ids, labels = build_mqar_batch(B, N, M, vocab=64, seed=0)
    S = 2 * N + 2 * M
    assert ids.shape == (B, S), f"ids shape: {ids.shape}"
    assert labels.shape == (B, S), f"labels shape: {labels.shape}"
    # Labels should be != -100 only at positions 2N+1, 2N+3, ..., S-1.
    expected = set(2 * N + 2 * j + 1 for j in range(M))
    actual = set((labels[0] != -100).nonzero(as_tuple=True)[0].tolist())
    assert actual == expected, f"label positions mismatch: {actual} vs {expected}"
    print("[1/3] MQAR layout OK")


def test_dyck2_balanced():
    """All sampled Dyck-2 strings must be balanced."""
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        L = int(torch.randint(2, 64, (1,), generator=g).item()) * 2  # even
        s = sample_dyck2(L, max_depth=8, generator=g).tolist()
        # Verify balance.
        stack = []
        ok = True
        for tok in s:
            if tok in (0, 2):
                stack.append(tok)
            elif tok in (1, 3):
                if not stack or stack[-1] != tok - 1:
                    ok = False
                    break
                stack.pop()
        assert ok and not stack, f"unbalanced or invalid: {s}"
    # Batch builder also works.
    ids, labels = build_dyck2_batch(B=2, length=32, max_depth=8, seed=0)
    assert ids.shape == (2, 32)
    assert labels.shape == (2, 32)
    assert (ids < VOCAB_DYCK2).all(), "out-of-vocab tokens"
    print("[2/3] Dyck-2 balance + shape OK")


def test_parity_labels():
    """Parity at cue is XOR of preceding non-cue bits."""
    ids, labels = build_parity_batch(B=4, length=32, p_query=0.3, seed=0)
    assert ids.shape == (4, 32)
    # Manually verify a few cue positions.
    for b in range(4):
        running_parity = 0
        for t in range(32):
            tok = int(ids[b, t].item())
            lab = int(labels[b, t].item())
            if tok == 2:
                # Cue: label should match running_parity.
                assert lab == running_parity, (
                    f"row {b} pos {t}: label={lab} but running={running_parity}"
                )
            else:
                if lab != -100:
                    assert False, f"non-cue position has label {lab}"
                running_parity = (running_parity + tok) % 2
    assert (ids < VOCAB_PARITY).all()
    print("[3/3] Parity labels match XOR-of-prefix OK")


if __name__ == "__main__":
    test_mqar_shapes()
    test_dyck2_balanced()
    test_parity_labels()
    print("\nAll synthetic-benchmark tests pass (3/3).")
