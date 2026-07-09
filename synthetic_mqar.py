"""Multi-Query Associative Recall (MQAR) - Arora et al. 2024 (Zoology).

Standard sub-quadratic LM benchmark. Sequence is a random permutation of
key-value pairs followed by query keys; model must produce the value seen
earlier for each query key.

Format:
  [k_1 v_1 k_2 v_2 ... k_N v_N Q_1 Q_2 ... Q_M]
Loss: cross-entropy on the positions immediately AFTER each query key - those
are where the model must produce the recalled value.

For HSPMN v5 we use:
  N (KV pairs) ∈ {32, 64, 128, 256}
  Sequence length S ∈ {2k, 4k, 8k}
  Vocabulary V = 8192 (large enough for unique keys + values)
  Train 50k examples / 5 epochs at 1.7M params.

Reference target: RWKV-7 reaches ≥99% at S=8k, KV=256 (Peng et al. Table 7).
"""
from typing import Tuple

import torch


def build_mqar_batch(B: int, N: int, M: int, vocab: int = 8192,
                     device: str = "cpu", seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build B examples of MQAR with N KV pairs and M queries.

    Returns:
        ids:    [B, S]    sequence of token ids
        labels: [B, S]    label at position t = next-token target if t is
                          immediately after a query key, else -100 (ignore)
    Layout per example:
      [k_1 v_1 k_2 v_2 ... k_N v_N Q_1 X Q_2 X ... Q_M X]
    Total length S = 2N + 2M  (Q_i followed by a placeholder slot for the value).
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    S = 2 * N + 2 * M
    ids = torch.zeros(B, S, dtype=torch.long, device=device)
    labels = torch.full((B, S), -100, dtype=torch.long, device=device)

    # Reserve key vocab in [1, vocab//2) and value vocab in [vocab//2, vocab).
    key_lo, key_hi = 1, vocab // 2
    val_lo, val_hi = vocab // 2, vocab

    for b in range(B):
        # Sample N unique keys.
        if gen is not None:
            perm = torch.randperm(key_hi - key_lo, generator=gen, device=device)[:N] + key_lo
            vals = torch.randint(val_lo, val_hi, (N,), generator=gen, device=device)
        else:
            perm = torch.randperm(key_hi - key_lo, device=device)[:N] + key_lo
            vals = torch.randint(val_lo, val_hi, (N,), device=device)
        keys = perm

        # Lay out KV pairs.
        for i in range(N):
            ids[b, 2 * i] = keys[i]
            ids[b, 2 * i + 1] = vals[i]

        # Sample M query indices (with replacement from the N keys).
        if gen is not None:
            q_idx = torch.randint(0, N, (M,), generator=gen, device=device)
        else:
            q_idx = torch.randint(0, N, (M,), device=device)
        for j in range(M):
            pos = 2 * N + 2 * j
            ids[b, pos] = keys[q_idx[j]]
            ids[b, pos + 1] = 0  # placeholder; loss is on this position
            labels[b, pos + 1] = vals[q_idx[j]]

    return ids, labels


@torch.no_grad()
def evaluate_mqar(model, N: int, M: int, vocab: int = 8192,
                  n_examples: int = 256, device: str = "cuda",
                  batch_size: int = 32) -> dict:
    """Return {'accuracy': float, 'loss': float} for a model on MQAR.

    Model must support model(ids, labels=labels) → dict with 'logits' and 'loss'.
    """
    model.train(False)
    correct = 0
    total = 0
    losses = []
    for start in range(0, n_examples, batch_size):
        bs = min(batch_size, n_examples - start)
        ids, labels = build_mqar_batch(bs, N, M, vocab=vocab,
                                       device=device, seed=42 + start)
        out = model(ids, labels=labels)
        logits = out["logits"]
        # Predict on positions where labels != -100. Note next-token shift:
        # logits[..., t-1] predicts ids at position t. Labels at position t.
        pred = logits[:, :-1, :].argmax(dim=-1)
        target = labels[:, 1:]
        mask = target != -100
        correct += ((pred == target) & mask).sum().item()
        total += mask.sum().item()
        if "loss" in out and out["loss"] is not None:
            losses.append(float(out["loss"].item()))
    return {
        "accuracy": correct / max(1, total),
        "loss": sum(losses) / max(1, len(losses)),
        "n_correct": correct, "n_total": total,
    }


if __name__ == "__main__":
    # Quick CPU sanity check.
    ids, labels = build_mqar_batch(B=2, N=4, M=2, vocab=64, device="cpu", seed=0)
    print("Sample sequence (first row):")
    print(" ids   :", ids[0].tolist())
    print(" labels:", labels[0].tolist())
    print("Note: label != -100 only at positions immediately after a query key.")
