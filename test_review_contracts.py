"""Regression checks for validation labels and completion boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eval_hellaswag import score_completion
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


class _Tokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[int(token) for token in text.split()])


@pytest.mark.parametrize("ending", ["4", "4 5"])
def test_completion_scores_every_ending_token(ending):
    model = _NextTokenOracle()
    assert score_completion(model, _Tokenizer(), "1 2 3", ending, "cpu") == 0.0


def test_completion_includes_first_ending_error():
    class WrongFirstToken(_NextTokenOracle):
        def forward(self, ids, labels=None):
            out = super().forward(ids, labels)
            out["logits"][:, 2, 4] = -30.0
            out["logits"][:, 2, 6] = 30.0
            return out

    loss = score_completion(WrongFirstToken(), _Tokenizer(), "1 2 3", "4 5", "cpu")
    assert loss == pytest.approx(30.0)


def test_completion_rejects_empty_context():
    with pytest.raises(ValueError, match="nonempty tokenized context"):
        score_completion(_NextTokenOracle(), _Tokenizer(), "", "4", "cpu")


def test_completion_without_ending_has_no_score():
    loss = score_completion(_NextTokenOracle(), _Tokenizer(), "1 2 3", "", "cpu")
    assert loss == float("inf")
