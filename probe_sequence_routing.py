"""Sequence-level routing probe - is the only open forward direction viable?

The gate-ceiling probes killed PER-TOKEN routing (NSA-helps is ~0.03-bit
decodable). The theorem leaves one forward path: route at the SEQUENCE/segment
level instead of per token. That can only help if the NSA benefit actually
VARIES across sequences (some sequences strongly need attention, others do not)
AND that variation is predictable from a cheap sequence summary.

This probe, on a trained checkpoint, computes for each validation sequence the
total contextual benefit Δ_seq = Σ_t (CE_no-NSA − CE_full), then reports:
  - the between-sequence spread of Δ_seq (is there anything to route on?);
  - the within/between variance split of the per-token benefit (is "NSA helps"
    a sequence property or just per-token noise?);
  - the held-out MI of a logistic predicting sign(Δ_seq) from the sequence-mean
    residual (can a cheap router decide it?).
High between-sequence structure + predictability => sequence-level routing has
headroom (a concrete breakthrough direction). Low => even sequence routing is
information-starved on this data, and the real lever is task/data diversity.

Usage:
    python3 probe_sequence_routing.py --variant hymba-with-nsa-gated \
        --ckpt checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/...final.pt \
        --n_seq 256 --seq_len 512 --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device
from measure_gate_channel_mi import mutual_information_bits, per_token_ce
from probe_gate_ceiling import binary_entropy_bits, fit_logistic

DATA_DIR = "./data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_seq", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = TrainConfig(
        variant=args.variant,
        n_layers=args.n_layers,
        dim=args.dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        batch_size=args.batch,
        grad_accum=1,
        steps=1,
        lr=1e-3,
        warmup_steps=1,
        nsa_window=256,
    )
    valid_tok = np.load(f"{DATA_DIR}/valid_tokens.npy", mmap_mode="r")
    train_tok = np.load(f"{DATA_DIR}/train_tokens.npy", mmap_mode="r")
    vocab = int(max(train_tok.max(), valid_tok.max())) + 1
    model = build_model(cfg, vocab, device, dtype)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.train(False)
    print(f"loaded {args.ckpt}")

    # Ablate ALL NSA streams at once -> whole-model contextual benefit per token.
    def ablate_all():
        return [
            layer.nsa.register_forward_hook(
                lambda m, i, o: o._replace(out=torch.zeros_like(o.out))
            )
            for layer in model.layers
            if getattr(layer, "nsa", None) is not None
        ]

    rng = np.random.default_rng(0)
    seq_delta = []  # Δ_seq total benefit per sequence
    seq_meanpos = []  # fraction of tokens where NSA helps, per sequence
    within_var = []  # within-sequence variance of per-token benefit
    feats = []  # sequence-mean residual (input embedding mean) for routing
    nb = math.ceil(args.n_seq / args.batch)
    with torch.no_grad():
        for _ in range(nb):
            starts = rng.integers(
                0, len(valid_tok) - args.seq_len - 1, size=(args.batch,)
            )
            x = np.stack([valid_tok[a : a + args.seq_len] for a in starts])
            x_t = torch.from_numpy(x).long().to(device)
            ce_on = per_token_ce(model, x_t, device)  # [B,S-1]
            hs = ablate_all()
            try:
                ce_off = per_token_ce(model, x_t, device)
            finally:
                for h in hs:
                    h.remove()
            delta = ce_off - ce_on  # [B,S-1] benefit
            seq_delta.extend(delta.sum(1).float().cpu().tolist())
            seq_meanpos.extend((delta > 0).float().mean(1).cpu().tolist())
            within_var.extend(delta.float().var(1).cpu().tolist())
            feats.append(model.embed(x_t).float().mean(1).cpu())  # [B,d] seq summary

    seq_delta = np.array(seq_delta)
    seq_meanpos = np.array(seq_meanpos)
    feats = torch.cat(feats)
    pos = (seq_delta > 0).astype(int)

    # variance decomposition of per-token benefit: between-seq vs within-seq
    per_tok_mean = seq_delta / (args.seq_len - 1)
    between_var = float(np.var(per_tok_mean))
    within_var_mean = float(np.mean(within_var))
    frac_between = between_var / (between_var + within_var_mean + 1e-12)

    # can a cheap sequence summary predict sign(Δ_seq)?
    n = feats.size(0)
    ntr = int(n * 0.7)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    feats, posT = feats[perm], torch.tensor(pos)[perm]
    pva = fit_logistic(feats[:ntr], posT[:ntr].float(), feats[ntr:], device="cpu")
    yva = posT[ntr:].numpy()
    acc = float(((pva > 0.5).astype(int) == yva).mean())
    mi = mutual_information_bits(pva, yva, n_bins=10)
    p = float(pos.mean())
    thr = binary_entropy_bits(p) - 0.5

    print(
        f"\nSequence-level NSA benefit, {len(seq_delta)} sequences, variant={args.variant}"
    )
    print(
        f"  Δ_seq (total CE saved by NSA): mean={seq_delta.mean():.3f} std={seq_delta.std():.3f} "
        f"min={seq_delta.min():.2f} max={seq_delta.max():.2f}"
    )
    print(f"  fraction of sequences where NSA net-helps: {p:.3f}")
    print(
        f"  per-token benefit variance: between-seq={between_var:.3e} within-seq={within_var_mean:.3e} "
        f"-> between fraction = {frac_between:.3f}"
    )
    print(
        f"  predict sign(Δ_seq) from seq-mean embedding (held-out): acc={acc:.3f} "
        f"I(pred;sign)={mi:.4f} bits | threshold={thr:.3f}"
    )
    verdict = (
        "SEQUENCE-LEVEL SIGNAL EXISTS -> routing whole sequences/segments to "
        "attention vs SSM has headroom; viable breakthrough direction."
        if (frac_between > 0.15 and mi > thr)
        else "WEAK sequence-level signal too -> on homogeneous text the contextual "
        "benefit is neither concentrated in particular sequences nor predictable; "
        "the lever is task/data diversity (retrieval-heavy corpora), not routing "
        "granularity. Even sequence routing is information-starved here."
    )
    print(f"VERDICT: {verdict}")

    out = Path(f"./results/sequence_routing_{args.variant}_2026-06.md")
    out.write_text(
        f"# Sequence-level routing probe - {args.variant}\n\n"
        f"Checkpoint `{args.ckpt}`, {len(seq_delta)} sequences x {args.seq_len} tokens.\n\n"
        f"- Δ_seq (total CE saved by NSA): mean {seq_delta.mean():.3f}, std {seq_delta.std():.3f}, "
        f"range [{seq_delta.min():.2f}, {seq_delta.max():.2f}].\n"
        f"- Fraction of sequences NSA net-helps: {p:.3f}.\n"
        f"- Per-token benefit variance: between-seq {between_var:.3e} / within-seq "
        f"{within_var_mean:.3e} -> **between-sequence fraction {frac_between:.3f}**.\n"
        f"- Predict sign(Δ_seq) from sequence-mean embedding (held-out): acc {acc:.3f}, "
        f"I(pred;sign) {mi:.4f} bits vs threshold {thr:.3f}.\n\n**VERDICT:** {verdict}\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
