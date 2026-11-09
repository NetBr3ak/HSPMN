"""Regression checks for validation labels and completion boundaries."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from measure_theorem_constants import _eval_loss_on_fixed_batches, estimate_L_star


class _NextTokenOracle(torch.nn.Module):
    def forward(self, ids, labels=None):
        logits = torch.full((*ids.shape, 64), -30.0)
        logits[:, :-1].scatter_(-1, ids[:, 1:, None], 30.0)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1)
            )
        return {"logits": logits, "loss": loss}


def test_theorem_losses_use_next_token_labels():
    tokens = np.random.default_rng(0).integers(0, 64, 5000)
    model = _NextTokenOracle()
    assert estimate_L_star(model, tokens, 2, 2, 32, "cpu") == pytest.approx(0.0)
    fixed_starts = [np.array([0, 33]), np.array([66, 99])]
    loss = _eval_loss_on_fixed_batches(model, tokens, fixed_starts, 2, 32, "cpu")
    assert loss == pytest.approx(0.0)
