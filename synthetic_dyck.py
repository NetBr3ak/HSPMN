"""Dyck-2 balanced parentheses - state-tracking benchmark.

Vocabulary:
  0 = '('   1 = ')'
  2 = '['   3 = ']'
  4 = end / pad
  (5 = optional EOS placeholder)

Task: given a Dyck-2 prefix, predict whether the *next* token is a valid
extension (i.e., either an opening bracket of either type, or the matching
closing bracket of the most recent unmatched opening). We frame this as
language modeling: train on valid Dyck-2 strings, evaluate held-out
perplexity. Pure Transformers fail Dyck-2 at long depth (TC^0 limitation,
Merrill & Sabharwal 2023). Mamba-3 / RWKV-7 should pass.

Reference: standard Dyck-k benchmark from the formal-language-recognition
literature (Hewitt et al., Bhattamishra et al.).
"""
from typing import Tuple

import torch


VOCAB_DYCK2 = 5  # 0='(' 1=')' 2='[' 3=']' 4=PAD


def sample_dyck2(length: int, max_depth: int, generator: torch.Generator = None) -> torch.Tensor:
    """Generate a single valid Dyck-2 string of *exactly* `length` tokens.

    Random walk over (open, close, open, close, ...) decisions, biased so the
    string ends balanced. Each opening is a uniform 50/50 choice between
    () and []. Closing always matches the most recent opening.
    """
    if generator is None:
        generator = torch.Generator()
    seq = torch.zeros(length, dtype=torch.long)
    stack = []
    for t in range(length):
        # Decide open vs close.
        depth = len(stack)
        remaining = length - t
        # Force-open if stack empty; force-close if depth would exceed remaining.
        if depth == 0 or (depth >= max_depth):
            open_ok, close_ok = (depth == 0), (depth > 0 and depth >= max_depth)
            if open_ok and close_ok:
                p_open = 0.5
            elif open_ok:
                p_open = 1.0
            else:
                p_open = 0.0
        else:
            # Bias so depth tends to drain by end.
            p_open = max(0.0, min(1.0, 0.5 + 0.3 * (1.0 - depth / max(1, max_depth))))
        # Force balance: must close all by end.
        if depth >= remaining:
            p_open = 0.0
        u = torch.rand(1, generator=generator).item()
        if u < p_open:
            # Open: pick ( or [
            kind = int(torch.rand(1, generator=generator).item() < 0.5)  # 0 or 1
            tok = 2 * kind  # 0=( 2=[
            seq[t] = tok
            stack.append(kind)
        else:
            kind = stack.pop()
            tok = 2 * kind + 1  # 1=) 3=]
            seq[t] = tok
    # If stack is non-empty (string not balanced), force-close remaining.
    # Already constrained above; assert for safety.
    assert len(stack) == 0, f"unbalanced Dyck-2 of length {length}, stack={stack}"
    return seq


def build_dyck2_batch(B: int, length: int, max_depth: int = 16,
                      device: str = "cpu", seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (ids, labels) for next-token training on Dyck-2."""
    gen = torch.Generator().manual_seed(seed)
    seqs = torch.stack([sample_dyck2(length, max_depth, gen) for _ in range(B)])
    ids = seqs.to(device)
    labels = ids.clone()
    return ids, labels


@torch.no_grad()
def evaluate_dyck2(model, length: int, max_depth: int = 16,
                   n_examples: int = 256, device: str = "cuda",
                   batch_size: int = 32) -> dict:
    """Return next-token PPL on held-out Dyck-2 strings."""
    model.train(False)
    losses = []
    n_tokens = 0
    for start in range(0, n_examples, batch_size):
        bs = min(batch_size, n_examples - start)
        ids, labels = build_dyck2_batch(bs, length, max_depth, device=device, seed=1000 + start)
        out = model(ids, labels=labels)
        loss = float(out["loss"].item())
        losses.append(loss * bs)
        n_tokens += bs
    mean_loss = sum(losses) / max(1, n_tokens)
    import math
    return {"loss": mean_loss, "ppl": math.exp(mean_loss), "n_examples": n_examples}


if __name__ == "__main__":
    # Sanity: print a few random Dyck-2 strings.
    gen = torch.Generator().manual_seed(0)
    syms = ["(", ")", "[", "]", "."]
    for _ in range(3):
        s = sample_dyck2(20, max_depth=8, generator=gen)
        print("".join(syms[t] for t in s.tolist()))
