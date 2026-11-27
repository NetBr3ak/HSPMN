import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import train_v4


@pytest.fixture
def final_path(tmp_path, monkeypatch):
    for split in ("train", "valid"):
        np.save(tmp_path / f"{split}_tokens.npy", np.arange(16, dtype=np.int64))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_v4.py",
            "--variant",
            "dense",
            "--steps",
            "0",
            "--dim",
            "32",
            "--n_layers",
            "1",
            "--num_heads",
            "4",
            "--num_kv_heads",
            "2",
            "--save_dir",
            str(tmp_path),
            "--data_dir",
            str(tmp_path),
        ],
    )
    return tmp_path / "dense_final.pt"


def test_interrupted_final_save_keeps_previous_model(final_path, monkeypatch):
    checkpoint = final_path
    torch.save({"model": {}, "step": 17}, checkpoint)
    previous = checkpoint.read_bytes()

    def interrupted_save(payload, destination):
        Path(destination).write_bytes(b"incomplete")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="simulated write failure"):
        train_v4.main()
    assert checkpoint.read_bytes() == previous
    assert not list(checkpoint.parent.glob(".checkpoint-*"))


def test_final_save_preserves_symlink_and_archive_name(final_path):
    target = final_path.parent / "stored.pt"
    torch.save({"step": 17}, target)
    target.chmod(0o640)
    final_path.symlink_to(target.name)

    train_v4.main()

    assert final_path.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o640
    assert torch.load(target, map_location="cpu", weights_only=True)["step"] == 0
    with zipfile.ZipFile(target) as archive:
        assert {member.split("/")[0] for member in archive.namelist()} == {
            final_path.stem
        }
