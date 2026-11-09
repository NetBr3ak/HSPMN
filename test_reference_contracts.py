"""Small regression checks for training and autoregressive inference contracts."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import aggregate_v6
import run_v5_queue2
import run_v6_queue
import train_p4_350m
from gated_deltanet import _chunked_gated_delta
from hymba_with_nsa_lm import HymbaWithNSALM
from nsa_attention import NSAAttention
from train_v4 import DenseLM, get_batch


class _EchoRecurrent(torch.nn.Module):
    def forward(self, q, k, v):
        return q


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


def test_predictive_gate_does_not_read_suffix():
    torch.manual_seed(17)
    model = HymbaWithNSALM(
        vocab_size=32,
        n_layers=1,
        dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        gate_mode="predictive_coding",
    ).eval()
    # Isolate gate causality from the recurrent kernel and its initialization scale.
    block = model.layers[0]
    block.ssm = _EchoRecurrent()
    ids = torch.randint(0, 32, (1, 12))
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 32
    with torch.no_grad():
        prefix = model(ids)["logits"][:, :6]
        prefix_gate = block._last_gate[:, :, :6].clone()
        altered_prefix = model(changed)["logits"][:, :6]
        altered_gate = block._last_gate[:, :, :6].clone()
        short_prefix = model(ids[:, :6])["logits"]
    torch.testing.assert_close(prefix, altered_prefix)
    torch.testing.assert_close(prefix, short_prefix)
    torch.testing.assert_close(prefix_gate, altered_gate)
    torch.testing.assert_close(prefix_gate, block._last_gate)


def test_compress_selection_does_not_read_suffix():
    torch.manual_seed(17)
    attention = NSAAttention(
        1,
        4,
        compress_block_size=2,
        compress_stride=2,
        window_size=2,
        select_from_compress=True,
        n_select_blocks=1,
    ).eval()
    q = torch.zeros(1, 1, 12, 4)
    q[..., 0] = 1
    k = torch.zeros_like(q)
    k[:, :, 2:4, 0] = 1
    v = torch.arange(12).float().view(1, 1, 12, 1).expand_as(q)
    changed_q = q.clone()
    # Dominate the old pooled -1e9 mask sentinel without changing any prefix query.
    changed_q[:, :, 6:, 0] = 1e12
    with torch.no_grad():
        prefix = attention(q, k, v).out[:, :, :6]
        altered_prefix = attention(changed_q, k, v).out[:, :, :6]
        short_prefix = attention(q[:, :, :6], k[:, :, :6], v[:, :, :6]).out
    torch.testing.assert_close(prefix, altered_prefix)
    torch.testing.assert_close(prefix, short_prefix)


def test_aggregate_preserves_summary_without_run_logs(tmp_path, monkeypatch):
    output = tmp_path / "summary.md"
    output.write_text("recorded result\n")
    monkeypatch.setattr(aggregate_v6, "CKPT", str(tmp_path / "missing"))
    monkeypatch.setattr(aggregate_v6, "OUT", str(output))
    with pytest.raises(SystemExit, match="No run logs"):
        aggregate_v6.main()
    assert output.read_text() == "recorded result\n"


@pytest.mark.parametrize("queue", [run_v5_queue2, run_v6_queue])
def test_queue_rejects_failed_process_with_artifact(queue, tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "log", lambda message: None)
    states = iter([False, True])
    monkeypatch.setattr(queue, "done", lambda job: next(states))
    monkeypatch.setattr(queue.subprocess, "call", lambda *args, **kwargs: 1)
    if queue is run_v6_queue:
        monkeypatch.setattr(queue, "wait_for_gpu", lambda name: None)
    assert not queue.run_job({"name": "run", "artifact": "run.pt", "cmd": "false"})


def test_kill_clock_saves_resume_not_final(tmp_path, monkeypatch):
    tokens = np.tile(np.arange(8, dtype=np.int32), 4)
    np.save(tmp_path / "train_tokens.npy", tokens)
    np.save(tmp_path / "valid_tokens.npy", tokens)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_p4_350m.py",
            "--variant",
            "dense",
            "--data_dir",
            str(tmp_path),
            "--save_dir",
            str(tmp_path),
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
            "--steps",
            "1",
            "--kill_clock_h",
            "-1",
        ],
    )
    monkeypatch.setattr(train_p4_350m, "get_device", lambda: torch.device("cpu"))
    with pytest.raises(SystemExit, match="Training interrupted"):
        train_p4_350m.main()
    state = torch.load(tmp_path / "dense_p4_state.pt", weights_only=True)
    assert state["step"] == 0
    assert not (tmp_path / "dense_p4_final.pt").exists()
