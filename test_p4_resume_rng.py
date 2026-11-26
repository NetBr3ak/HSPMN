import os
import subprocess
import sys

import numpy as np
import pytest
import torch

INTERRUPT = """
import sys
from types import SimpleNamespace
import train_p4_350m as trainer

mode = sys.argv.pop(1)
if mode == "periodic":
    log = trainer.logger.info
    def interrupt_after_save(message, *args, **kwargs):
        log(message, *args, **kwargs)
        if "ckpt saved" in str(message):
            raise SystemExit(73)
    trainer.logger.info = interrupt_after_save
else:
    step = trainer.torch.optim.AdamW.step
    completed = [0]
    def count_step(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        completed[0] += 1
        return result
    trainer.torch.optim.AdamW.step = count_step
    trainer.time = SimpleNamespace(time=lambda: 10000.0 if completed[0] >= 2 else 0.0)
trainer.main()
"""


def command(data, output, isolated, mode):
    arguments = [
        "--variant",
        "dense",
        "--n_layers",
        "1",
        "--dim",
        "32",
        "--num_heads",
        "4",
        "--num_kv_heads",
        "2",
        "--seq_len",
        "8",
        "--batch_size",
        "2",
        "--grad_accum",
        "1",
        "--steps",
        "4",
        "--warmup_steps",
        "1",
        "--eval_every",
        "1",
        "--eval_iters",
        "2",
        "--log_every",
        "100",
        "--ckpt_every",
        "2" if mode == "periodic" else "100",
        "--kill_clock_h",
        "1",
        "--data_dir",
        str(data),
        "--save_dir",
        str(output),
    ]
    if isolated:
        arguments.append("--isolate_validation_rng")
    return arguments


def run(arguments, mode=None, check=True):
    command_line = [sys.executable]
    if mode:
        command_line += ["-c", INTERRUPT, mode]
    else:
        command_line += ["train_p4_350m.py"]
    return subprocess.run(
        command_line + arguments,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture
def token_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.random.default_rng(19).integers(0, 17, 256, dtype=np.int64)
    np.save(data / "train_tokens.npy", tokens)
    np.save(data / "valid_tokens.npy", tokens[::-1])
    return data


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("mode", ["periodic", "clock"])
def test_resume_matches_continuous_training(token_files, tmp_path, isolated, mode):
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    original = command(token_files, continuous, isolated, mode)
    interrupted = command(token_files, resumed, isolated, mode)
    run(original)
    result = run(interrupted, mode=mode, check=False)
    assert result.returncode == (73 if mode == "periodic" else 1), result.stderr
    checkpoint = torch.load(resumed / "dense_p4_state.pt", map_location="cpu")
    assert checkpoint["step"] == 2
    assert checkpoint["tokens_seen"] == 32
    run(interrupted)
    expected = torch.load(
        continuous / "dense_p4_final.pt", map_location="cpu", weights_only=True
    )
    observed = torch.load(
        resumed / "dense_p4_final.pt", map_location="cpu", weights_only=True
    )
    assert observed["model"].keys() == expected["model"].keys()
    for name in expected["model"]:
        assert torch.equal(observed["model"][name], expected["model"][name]), name
    assert observed["opt"]["param_groups"] == expected["opt"]["param_groups"]
    for parameter, expected_values in expected["opt"]["state"].items():
        for name, value in expected_values.items():
            assert torch.equal(observed["opt"]["state"][parameter][name], value)
    assert observed["final_valid_ce"] == expected["final_valid_ce"]
    assert observed["tokens_seen"] == expected["tokens_seen"] == 64


def test_resume_legacy_checkpoint(token_files, tmp_path):
    output = tmp_path / "legacy"
    arguments = command(token_files, output, False, "periodic")
    result = run(arguments, mode="periodic", check=False)
    assert result.returncode == 73, result.stderr
    path = output / "dense_p4_state.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint.pop("rng_state", None)
    checkpoint.pop("tokens_seen", None)
    torch.save(checkpoint, path)
    arguments[arguments.index("--batch_size") + 1] = "1"
    resumed = run(arguments)
    assert "no random state" in resumed.stderr
    final = torch.load(
        output / "dense_p4_final.pt", map_location="cpu", weights_only=True
    )
    assert final["step"] == 4
    assert final["tokens_seen"] == 48
