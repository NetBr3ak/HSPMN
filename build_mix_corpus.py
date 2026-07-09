"""Build mixture corpora for the v6 routability phase diagram.

Motivation (gate-ceiling finding, 2026-06): per-token routing between the
attention and SSM streams fails on uniform prose because the routing target
s_t = 1[attention lowers the loss at t] is nearly information-free
(optimal-probe ceiling ~0.03 bits vs the ~0.72-bit full-benefit threshold of
the gate-channel bound). The theory's constructive direction: raise the
ceiling by making the data heterogeneous, so that "which stream helps" becomes
predictable from context.

This script interleaves wikitext-103 prose chunks with synthetic key-value
recall segments (MQAR-style, rendered directly in GPT-2 token-id space) at a
controlled token fraction pi. Inside a recall segment, answering a query
requires exact associative recall across ~100 tokens: the attention stream is
decisive there, while the fixed-state SSM stream saturates. On prose both
streams are known to be substitutable. Prediction P1 (pre-registered): the
optimal-probe ceiling I_max(pi) rises monotonically with pi.

Recall segment layout (SEG_LEN=256 tokens):
    SEP  k1 v1 k2 v2 ... kN vN  [ Q k_j A v_j ] x n_queries  SEP-pad
Keys/values are drawn per-segment from a fixed 512-id alphabet (disjoint key
and value halves), so the mapping is unique within a segment and random across
segments: the ONLY way to predict v_j at an answer position is in-context
recall, never memorisation.

Outputs, per pi in --pis:
    data/mix_pi{P}/train_tokens.npy        (uint16, --train_tokens)
    data/mix_pi{P}/valid_tokens.npy        (uint16, --valid_tokens)
    data/mix_pi{P}/valid_answer_mask.npy   (uint8, 1 = answer position)
    data/mix_pi{P}/meta.json

Usage:
    python3 build_mix_corpus.py --pis 0.0 0.1 0.25 0.5
"""
import argparse
import json
import os

import numpy as np

DATA_DIR = "/opt/docker/LLM/HSPMN/data"
SEG_LEN = 256
N_PAIRS = 48
N_QUERIES = 32
ALPHABET = 512          # ids per role bank (keys | values)
ID_LOW, ID_HIGH = 1000, 45000   # safe interior of the GPT-2 vocab
SEP, QMARK, AMARK = 50256, 198, 25    # <|endoftext|>, "\n", ":"


def make_banks(seed: int):
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(ID_LOW, ID_HIGH), size=2 * ALPHABET,
                     replace=False)
    return ids[:ALPHABET], ids[ALPHABET:]


def recall_segment(rng, key_bank, val_bank):
    """One SEG_LEN recall segment + its answer-position mask."""
    keys = rng.choice(key_bank, size=N_PAIRS, replace=False)
    vals = rng.choice(val_bank, size=N_PAIRS, replace=True)
    toks = [SEP]
    for k, v in zip(keys, vals):
        toks.extend((int(k), int(v)))
    mask = [0] * len(toks)
    qidx = rng.choice(N_PAIRS, size=N_QUERIES, replace=False)
    for j in qidx:
        toks.extend((QMARK, int(keys[j]), AMARK, int(vals[j])))
        mask.extend((0, 0, 0, 1))
    # pad to SEG_LEN with SEP
    pad = SEG_LEN - len(toks)
    toks.extend([SEP] * pad)
    mask.extend([0] * pad)
    return np.array(toks[:SEG_LEN], dtype=np.uint16), \
        np.array(mask[:SEG_LEN], dtype=np.uint8)


def build_stream(src, n_tokens, pi, seed, key_bank, val_bank):
    """Interleave prose chunks and recall segments to n_tokens total."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_tokens, dtype=np.uint16)
    mask = np.zeros(n_tokens, dtype=np.uint8)
    cursor = int(rng.integers(0, max(1, len(src) - n_tokens - SEG_LEN)))
    pos = 0
    while pos < n_tokens:
        take = min(SEG_LEN, n_tokens - pos)
        if rng.random() < pi:
            seg, m = recall_segment(rng, key_bank, val_bank)
            out[pos:pos + take] = seg[:take]
            mask[pos:pos + take] = m[:take]
        else:
            if cursor + take > len(src):
                cursor = 0
            out[pos:pos + take] = src[cursor:cursor + take]
            cursor += take
        pos += take
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pis", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--train_tokens", type=int, default=60_000_000)
    ap.add_argument("--valid_tokens", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    train_src = np.load(f"{DATA_DIR}/train_tokens.npy", mmap_mode="r")
    valid_src = np.load(f"{DATA_DIR}/valid_tokens.npy", mmap_mode="r")
    key_bank, val_bank = make_banks(args.seed)

    for pi in args.pis:
        tag = f"{int(round(pi * 100)):02d}"
        out_dir = f"{DATA_DIR}/mix_pi{tag}"
        meta_p = f"{out_dir}/meta.json"
        if os.path.exists(meta_p):
            print(f"skip pi={pi}: {meta_p} exists")
            continue
        os.makedirs(out_dir, exist_ok=True)
        tr, _ = build_stream(train_src, args.train_tokens, pi,
                             args.seed + 1, key_bank, val_bank)
        va, va_mask = build_stream(valid_src, args.valid_tokens, pi,
                                   args.seed + 2, key_bank, val_bank)
        np.save(f"{out_dir}/train_tokens.npy", tr)
        np.save(f"{out_dir}/valid_tokens.npy", va)
        np.save(f"{out_dir}/valid_answer_mask.npy", va_mask)
        realized = float(va_mask.mean())
        with open(meta_p, "w") as f:
            json.dump({"pi": pi, "seg_len": SEG_LEN, "n_pairs": N_PAIRS,
                       "n_queries": N_QUERIES, "alphabet": ALPHABET,
                       "train_tokens": len(tr), "valid_tokens": len(va),
                       "valid_answer_frac": realized,
                       "seed": args.seed}, f, indent=2)
        print(f"built pi={pi} -> {out_dir}  answer_frac={realized:.4f}")


if __name__ == "__main__":
    main()
