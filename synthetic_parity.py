"""Parity - the canonical state-tracking benchmark.

Vocabulary: {0, 1, 2}.  0 and 1 are bits.  2 is the "predict-parity-now" cue.

Task: at each position labeled 2, model must predict the parity of the bits
seen so far. Pure Transformers fail this beyond ~50 tokens (TC^0 result of
Merrill & Sabharwal 2023). Mamba-3 / RWKV-7 (regular-language recognizers)
should pass it indefinitely.

Format per example:
  [b_1 b_2 b_3 ... b_S]
At a fraction `p_query` of positions chosen at random, b_t is replaced by 2
(the cue) and the label at that position becomes XOR of all preceding bits.
"""

from typing import Tuple

import torch


VOCAB_PARITY = 3  # 0, 1, 2 (cue)


def build_parity_batch(
    B: int, length: int, p_query: float = 0.1, device: str = "cpu", seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (ids, labels). At cue positions (token=2), label is parity of
    all bit positions strictly before this one. Other positions: label = -100.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (B, length), generator=gen)  # 0 or 1
    cue_mask = torch.rand(B, length, generator=gen) < p_query
    cue_mask[:, 0] = False  # no cue at t=0 (no preceding bits)

    # Compute running parity over the BIT positions only (cues don't count).
    # Cues override their original bit, so they contribute 0 to parity.
    bits_for_parity = bits.clone()
    bits_for_parity[cue_mask] = 0
    # Parity at position t = XOR of bits_for_parity[:, :t]
    cum = bits_for_parity.cumsum(dim=1) % 2  # parity AT position t INCLUSIVE
    # Parity STRICTLY BEFORE position t:
    parity_before = torch.zeros(B, length, dtype=torch.long)
    parity_before[:, 1:] = cum[:, :-1]

    # Build ids: cue replaces bit at cue position.
    ids = bits.clone()
    ids[cue_mask] = 2

    # Labels: at cue positions, target = parity_before. Elsewhere: -100.
    labels = torch.full((B, length), -100, dtype=torch.long)
    labels[cue_mask] = parity_before[cue_mask]

    return ids.to(device), labels.to(device)


@torch.no_grad()
def evaluate_parity(
    model,
    length: int,
    p_query: float = 0.1,
    n_examples: int = 256,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Return {'accuracy': float, 'loss': float} on parity cue positions."""
    model.train(False)
    correct = 0
    total = 0
    losses = []
    for start in range(0, n_examples, batch_size):
        bs = min(batch_size, n_examples - start)
        ids, labels = build_parity_batch(
            bs, length, p_query=p_query, device=device, seed=2000 + start
        )
        out = model(ids, labels=labels)
        logits = out["logits"]
        # Predict at the cue positions themselves (label is at that position;
        # logits[..., t-1] predicts position t, but we labeled position t with
        # parity_before, so prediction at position t comes from logits[..., t-1]
        # but only the parity classes (0 or 1) matter.
        # Use shift form: logits[:, :-1] predicts ids[:, 1:].
        # Labels[:, 1:] != -100 marks cue positions. So target uses labels[:, 1:].
        pred = logits[:, :-1, :2].argmax(dim=-1)  # restrict to {0, 1} for accuracy
        target = labels[:, 1:]
        mask = target != -100
        correct += ((pred == target) & mask).sum().item()
        total += mask.sum().item()
        if "loss" in out and out["loss"] is not None:
            losses.append(float(out["loss"].item()))
    return {
        "accuracy": correct / max(1, total),
        "loss": sum(losses) / max(1, len(losses)),
        "n_correct": correct,
        "n_total": total,
    }


if __name__ == "__main__":
    ids, labels = build_parity_batch(B=2, length=16, p_query=0.3, seed=0)
    syms = ["0", "1", "?"]  # ? = cue
    for b in range(2):
        seq = "".join(syms[t] for t in ids[b].tolist())
        labs = labels[b].tolist()
        print(f"row {b}: {seq}")
        cues = [(t, labs[t]) for t in range(len(labs)) if labs[t] != -100]
        print(f"  cues @ (pos, parity): {cues}")
