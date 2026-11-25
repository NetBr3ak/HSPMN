import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from train_v4 import TrainConfig, get_batch, run_validation


class ConstantLoss(torch.nn.Module):
    def forward(self, tokens, labels=None):
        return {"loss": tokens.float().mean()}


def sample_training_batches(every, isolated):
    np.random.seed(42)
    rng = np.random.default_rng(42) if isolated else None
    cfg = TrainConfig(batch_size=2, seq_len=4, eval_iters=2)
    tokens = np.arange(1000)
    model = ConstantLoss()
    batches = []
    for step in range(1, 7):
        batch, _ = get_batch(tokens, cfg.batch_size, cfg.seq_len, "cpu")
        batches.append(batch.numpy().copy())
        if step % every == 0:
            run_validation(model, tokens, cfg, "cpu", rng=rng)
    return np.stack(batches)


def test_isolated_validation_does_not_advance_training_samples():
    frequent = sample_training_batches(every=1, isolated=True)
    sparse = sample_training_batches(every=3, isolated=True)
    np.testing.assert_array_equal(frequent, sparse)
    np.testing.assert_array_equal(frequent, sample_training_batches(1, isolated=True))


def test_default_validation_preserves_legacy_shared_random_stream():
    frequent = sample_training_batches(every=1, isolated=False)
    sparse = sample_training_batches(every=3, isolated=False)
    assert not np.array_equal(frequent, sparse)
    np.testing.assert_array_equal(frequent, sample_training_batches(1, isolated=False))

    legacy = np.random.RandomState(42)
    expected = []
    tokens = np.arange(1000)
    for _ in range(6):
        starts = legacy.randint(0, len(tokens) - 4 + 1, size=(2,))
        expected.append(np.stack([tokens[start : start + 4] for start in starts]))
        for _ in range(2):
            legacy.randint(0, len(tokens) - 4 + 1, size=(2,))
    np.testing.assert_array_equal(frequent, np.stack(expected))


@pytest.mark.parametrize(
    ("script", "checkpoint"),
    [("train_v4.py", "dense_final.pt"), ("train_p4_350m.py", "dense_p4_final.pt")],
)
def test_trainer_keeps_weights_when_validation_cadence_changes(
    tmp_path, script, checkpoint
):
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.arange(128, dtype=np.int64) % 64
    np.save(data / "train_tokens.npy", tokens)
    np.save(data / "valid_tokens.npy", tokens[::-1])
    models = []
    for cadence in (1, 2):
        output = tmp_path / str(cadence)
        command = [
            sys.executable,
            script,
            "--variant",
            "dense",
            "--n_layers",
            "1",
            "--dim",
            "32",
            "--num_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--seq_len",
            "4",
            "--batch_size",
            "1",
            "--grad_accum",
            "1",
            "--steps",
            "3",
            "--warmup_steps",
            "1",
            "--eval_iters",
            "1",
            "--eval_every",
            str(cadence),
            "--log_every",
            "100",
            "--data_dir",
            str(data),
            "--save_dir",
            str(output),
            "--isolate_validation_rng",
        ]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run(command, check=True, text=True, env=env)
        state = torch.load(output / checkpoint, map_location="cpu", weights_only=True)
        models.append(state["model"])
    assert models[0].keys() == models[1].keys()
    for name in models[0]:
        assert torch.equal(models[0][name], models[1][name]), name
