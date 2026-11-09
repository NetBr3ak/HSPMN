"""Small regression checks for training and autoregressive inference contracts."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

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
