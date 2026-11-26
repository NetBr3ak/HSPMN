import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

FAIL_SAVE = """
import hashlib
import resource
import signal
import sys
from pathlib import Path
import train_p4_350m as trainer

marker = Path(sys.argv.pop(1))
save = trainer.torch.save
def constrained_save(value, path, *args, **kwargs):
    if value["step"] == 2:
        old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024, old_limit[1]))
        try:
            return save(value, path, *args, **kwargs)
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, old_limit)
    result = save(value, path, *args, **kwargs)
    if value["step"] == 1:
        marker.write_text(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return result
trainer.torch.save = constrained_save
trainer.main()
"""


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux file-size limits")
def test_failed_checkpoint_write_preserves_last_good_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.random.default_rng(19).integers(0, 17, 256, dtype=np.int64)
    np.save(data / "train_tokens.npy", tokens)
    np.save(data / "valid_tokens.npy", tokens[::-1])
    output = tmp_path / "output"
    marker = tmp_path / "first-checkpoint.sha256"
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
        "1",
        "--log_every",
        "100",
        "--ckpt_every",
        "1",
        "--data_dir",
        str(data),
        "--save_dir",
        str(output),
    ]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    interrupted = subprocess.run(
        [sys.executable, "-c", FAIL_SAVE, str(marker), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert interrupted.returncode != 0
    assert marker.exists(), interrupted.stderr
    assert "iostream error" in interrupted.stderr, interrupted.stderr
    checkpoint_path = output / "dense_p4_state.pt"
    assert (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == marker.read_text()
    )
    assert not list(output.glob("checkpoint-*.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 1
    subprocess.run(
        [sys.executable, "train_p4_350m.py", *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    final = torch.load(
        output / "dense_p4_final.pt", map_location="cpu", weights_only=True
    )
    recovered_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    assert recovered_checkpoint["step"] == 4
    assert final["step"] == 4
    assert final["tokens_seen"] == 64
