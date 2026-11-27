import math

import pytest
import torch

import synthetic_dyck
import synthetic_mqar
import synthetic_parity


class BatchLoss(torch.nn.Module):
    def forward(self, ids, labels=None):
        return {
            "logits": torch.zeros(*ids.shape, 16),
            "loss": ids[0, 0].float(),
        }


@pytest.mark.parametrize(
    ("module", "evaluate", "kwargs"),
    [
        (synthetic_dyck, "evaluate_dyck2", {"length": 4}),
        (synthetic_mqar, "evaluate_mqar", {"N": 2, "M": 0, "vocab": 16}),
        (synthetic_parity, "evaluate_parity", {"length": 4, "p_query": 0}),
    ],
)
def test_empty_observations_raise(module, evaluate, kwargs):
    with pytest.raises(ValueError, match="label"):
        getattr(module, evaluate)(BatchLoss(), n_examples=0, device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("evaluate", "kwargs"),
    [
        (synthetic_mqar.evaluate_mqar, {"N": 2, "M": 0, "vocab": 16}),
        (synthetic_parity.evaluate_parity, {"length": 4, "p_query": 0}),
    ],
)
def test_no_query_labels_raise(evaluate, kwargs):
    with pytest.raises(ValueError, match="label"):
        evaluate(BatchLoss(), n_examples=2, device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("module", "builder", "evaluate", "kwargs"),
    [
        (synthetic_dyck, "build_dyck2_batch", "evaluate_dyck2", {"length": 4}),
        (synthetic_mqar, "build_mqar_batch", "evaluate_mqar", {"N": 2, "M": 1}),
        (synthetic_parity, "build_parity_batch", "evaluate_parity", {"length": 4}),
    ],
)
def test_loss_weighted_by_shifted_labels_with_empty_middle_batch(
    monkeypatch, module, builder, evaluate, kwargs
):
    batches = iter(
        [
            (
                torch.tensor([[2, 0, 0, 0], [2, 0, 0, 0]]),
                torch.tensor([[-100, 0, 0, -100], [-100, 0, -100, -100]]),
            ),
            (torch.tensor([[9, 0], [9, 0]]), torch.full((2, 2), -100)),
            (torch.tensor([[4, 0, 0]]), torch.tensor([[-100, 0, -100]])),
        ]
    )
    monkeypatch.setattr(module, builder, lambda *args, **kw: next(batches))
    result = getattr(module, evaluate)(
        BatchLoss(), n_examples=5, batch_size=2, device="cpu", **kwargs
    )
    assert result["loss"] == pytest.approx(2.5)
    if module is synthetic_dyck:
        assert result["ppl"] == pytest.approx(16)
    else:
        assert result["n_total"] == 4
        assert result["accuracy"] == 1.0


def test_normal_dyck_batch_uses_next_token_loss():
    class Uniform(torch.nn.Module):
        def forward(self, ids, labels=None):
            return {
                "logits": torch.zeros(*ids.shape, 4),
                "loss": torch.tensor(math.log(4)),
            }

    result = synthetic_dyck.evaluate_dyck2(
        Uniform(), length=4, n_examples=3, batch_size=2, device="cpu"
    )
    assert result["loss"] == pytest.approx(math.log(4))
    assert result["ppl"] == pytest.approx(4)


@pytest.mark.parametrize(
    ("evaluate", "kwargs", "vocab"),
    [
        (synthetic_mqar.evaluate_mqar, {"N": 2, "M": 2, "vocab": 16}, 16),
        (synthetic_parity.evaluate_parity, {"length": 8, "p_query": 1}, 3),
    ],
)
def test_normal_queries_use_model_cross_entropy(evaluate, kwargs, vocab):
    from train_v4 import DenseLM

    model = DenseLM(vocab, 1, 16, 2, 1, 8)
    for parameter in model.parameters():
        parameter.data.zero_()
    result = evaluate(model, n_examples=3, batch_size=2, device="cpu", **kwargs)
    assert result["loss"] == pytest.approx(math.log(vocab))
    assert result["n_total"] > 0


@pytest.mark.parametrize("penalty", [0.0, 2.0])
def test_dyck_perplexity_excludes_model_penalty(penalty):
    class Uniform(torch.nn.Module):
        def forward(self, ids, labels=None):
            return {
                "logits": torch.zeros(*ids.shape, 4),
                "loss": torch.tensor(math.log(4) + penalty),
            }

    result = synthetic_dyck.evaluate_dyck2(
        Uniform(), length=4, n_examples=3, batch_size=2, device="cpu"
    )
    assert result["loss"] == pytest.approx(math.log(4) + penalty)
    assert result["ppl"] == pytest.approx(4)


def test_dyck_perplexity_weights_nonuniform_predictions(monkeypatch):
    batches = iter(
        [
            (
                torch.zeros((2, 4), dtype=torch.long),
                torch.tensor([[-100, 1, 1, -100], [-100, 1, -100, -100]]),
            ),
            (torch.zeros((1, 2), dtype=torch.long), torch.tensor([[-100, 0]])),
        ]
    )
    monkeypatch.setattr(
        synthetic_dyck, "build_dyck2_batch", lambda *args, **kwargs: next(batches)
    )

    class Nonuniform(torch.nn.Module):
        def forward(self, ids, labels=None):
            logits = torch.tensor([0.0, math.log(3), 0.0, 0.0]).expand(*ids.shape, 4)
            return {"logits": logits, "loss": torch.tensor(7.0)}

    result = synthetic_dyck.evaluate_dyck2(
        Nonuniform(), length=4, n_examples=3, batch_size=2, device="cpu"
    )
    assert result["loss"] == pytest.approx(7)
    assert result["ppl"] == pytest.approx(math.exp((3 * math.log(2) + math.log(6)) / 4))
