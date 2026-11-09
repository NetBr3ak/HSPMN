"""Small regression checks for training and autoregressive inference contracts."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gated_deltanet import _chunked_gated_delta
from train_v4 import DenseLM, get_batch


def test_batch_labels_shift_only_inside_lm():
    tokens = np.arange(8, dtype=np.int32)
    ids, labels = get_batch(tokens, 1, len(tokens), torch.device("cpu"))
    torch.testing.assert_close(ids, labels)
    model = DenseLM(8, 1, 32, 4, 2, 8)
    out = model(ids, labels=labels)
    expected = F.cross_entropy(out["logits"][0, :-1], ids[0, 1:])
    torch.testing.assert_close(out["loss"], expected)
    out["loss"].backward()
    assert model.layers[0].q_proj.weight.grad is not None


@pytest.mark.parametrize("length,seq_len", [(1, 1), (3, 4)])
def test_batch_rejects_unusable_stream(length, seq_len):
    with pytest.raises(ValueError, match="sequence length"):
        get_batch(np.arange(length), 1, seq_len, torch.device("cpu"))


def test_delta_rule_recovers_written_value():
    key = torch.tensor([[[[1.0, 0.0]]]])
    value = torch.tensor([[[[0.0, 1.0]]]])
    gate = torch.ones(1, 1, 1)
    out = _chunked_gated_delta(key, key, value, gate, gate)
    torch.testing.assert_close(out, value / math.sqrt(2))


def test_delta_rule_matches_independent_matrix_recurrence():
    generator = torch.Generator().manual_seed(17)
    q, k, v = [torch.randn(1, 5, 1, 3, generator=generator) for _ in range(3)]
    k = F.normalize(k, dim=-1)
    alpha = torch.rand(1, 5, 1, generator=generator)
    beta = torch.rand(1, 5, 1, generator=generator)
    state = torch.zeros(3, 3)
    expected = []
    for step in range(5):
        key = k[0, step, 0]
        transition = torch.eye(3) - beta[0, step, 0] * torch.outer(key, key)
        state = alpha[0, step, 0] * transition @ state
        state += beta[0, step, 0] * torch.outer(key, v[0, step, 0])
        expected.append(q[0, step, 0] @ state / math.sqrt(3))
    actual = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=2)
    torch.testing.assert_close(actual[0, :, 0], torch.stack(expected))
