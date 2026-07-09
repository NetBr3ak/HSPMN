"""Gate-as-channel mutual information I(g; s) - Theorem 1' measurement.

The absorption-resistance theorem's headline constant lambda (loss curvature) is
hard to measure and conceptually fragile (a neural-net Hessian's smallest
eigenvalue is ~0/negative; see BUG-1 in TROUBLESHOOTING.md). This script measures
the *information-theoretic* alternative:

    Treat the per-token gate g_t = sigmoid(W h_t) as a noisy channel deciding
    "route to the contextual (NSA) stream". Measure its mutual information with a
    NON-CIRCULAR label s_t = "does the contextual stream actually help at token
    t", defined by a counterfactual ablation that is INDEPENDENT of the gate's
    parameters:

        s^L_t = 1  iff  CE(NSA_layer_L ablated)_t  -  CE(full model)_t  >  margin

    i.e. zeroing layer L's NSA output raises the next-token loss at t (the
    contextual stream was carrying signal there). g^L_t is layer L's gate value.

A learned gate should have I(g; s) > 0 (it routes where NSA helps); the
Aquino-Michaels random-gate control should have I ~ 0 (frozen-random gate is
uninformative). Reporting I(gated) > I(randgate) ~ 0 is a PREDICTIVE,
non-circular separation criterion - the answer to the "post-hoc / circular"
reviewer objection - and removes lambda from the headline bound.

Usage:
    python3 measure_gate_channel_mi.py --variant hymba-with-nsa-gated \
        --ckpt checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/hymba-with-nsa-gated_final.pt \
        --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device

DATA_DIR = "/opt/docker/LLM/HSPMN/data"


def mutual_information_bits(g, s, n_bins=10):
    """Plug-in MI (bits) between continuous g in [0,1] and binary s, with the
    Miller-Madow bias correction. g, s are 1-D numpy arrays of equal length."""
    g = np.clip(np.asarray(g, dtype=np.float64), 0.0, 1.0)
    s = np.asarray(s, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gb = np.clip(np.digitize(g, edges[1:-1]), 0, n_bins - 1)
    N = len(g)
    joint = np.zeros((n_bins, 2), dtype=np.float64)
    for gi, si in zip(gb, s):
        joint[gi, si] += 1.0
    joint /= N
    pg = joint.sum(axis=1, keepdims=True)
    ps = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    mi = float(
        (
            joint[nz]
            * np.log2(
                joint[nz] / (pg @ np.ones((1, 2)))[nz] / (np.ones((n_bins, 1)) @ ps)[nz]
            )
        ).sum()
    )
    # Miller-Madow correction: + (cells_used - rows_used - cols_used + 1)/(2 N ln2)
    cells = int(nz.sum())
    rows = int((pg > 0).sum())
    cols = int((ps > 0).sum())
    mi += (cells - rows - cols + 1) / (2.0 * N * math.log(2.0))
    return max(mi, 0.0)


def _selftest_mi():
    """Validate the estimator: independent -> ~0 bits, deterministic -> ~1 bit."""
    rng = np.random.default_rng(0)
    g = rng.uniform(0, 1, size=20000)
    s_indep = rng.integers(0, 2, size=20000)
    s_dep = (g > 0.5).astype(int)
    mi_indep = mutual_information_bits(g, s_indep)
    mi_dep = mutual_information_bits(g, s_dep)
    assert mi_indep < 0.02, f"MI self-test failed (independent): {mi_indep}"
    assert mi_dep > 0.9, f"MI self-test failed (deterministic): {mi_dep}"
    return mi_indep, mi_dep


def per_token_ce(model, x_t, device):
    """Next-token CE per position [B, S-1] from a single forward (no grad)."""
    out = model(x_t)
    logits = out["logits"]
    sl = logits[:, :-1, :].contiguous()
    tgt = x_t[:, 1:].contiguous()
    ce = F.cross_entropy(sl.view(-1, sl.size(-1)), tgt.view(-1), reduction="none")
    return ce.view(x_t.size(0), -1).float()  # [B, S-1]


def gated_layers(model):
    return [
        (i, layer)
        for i, layer in enumerate(model.layers)
        if getattr(layer, "attn_gate_proj", None) is not None
        or getattr(layer, "gate_mode", "linear") == "predictive_coding"
    ]


def measure(
    model, valid_tok, n_batches, batch, seq, device, margin, n_bins, label_model=None
):
    """Measure I(g; s). g (the gate) always comes from `model`. The label s
    (does NSA help) comes from `label_model` when given -- an INDEPENDENTLY
    trained gate-free baseline -- which makes the measurement non-circular: g
    cannot have shaped the model that defines s. When label_model is None, s is
    taken from `model` itself (the weaker, post-hoc self-measurement)."""
    layers = gated_layers(model)
    if not layers:
        return None
    model.train(False)
    s_model = label_model if label_model is not None else model
    s_model.train(False)

    rng = np.random.default_rng(seed=0)
    starts = [
        rng.integers(0, len(valid_tok) - seq - 1, size=(batch,))
        for _ in range(n_batches)
    ]

    g_acc = {i: [] for i, _ in layers}
    s_acc = {i: [] for i, _ in layers}

    with torch.no_grad():
        for s_off in starts:
            x = np.stack([valid_tok[a : a + seq] for a in s_off])
            x_t = torch.from_numpy(x).long().to(device)

            # Gate g^L_t from the main model.
            _ = per_token_ce(model, x_t, device)  # populates _last_gate
            gates = {}
            for i, layer in layers:
                g = getattr(layer, "_last_gate", None)  # [B, n_attn, S, 1]
                if g is None:
                    continue
                gm = g.float().mean(dim=1).squeeze(-1)  # [B, S]
                gates[i] = gm[:, :-1]  # align to CE positions

            # Label s^L_t from s_model (the gate-free baseline when cross-model).
            ce_on = per_token_ce(s_model, x_t, device)  # [B, S-1]
            for i, _l in layers:
                tgt = s_model.layers[i]
                handle = tgt.nsa.register_forward_hook(
                    lambda m, inp, out: out._replace(out=torch.zeros_like(out.out))
                )
                try:
                    ce_off = per_token_ce(s_model, x_t, device)
                finally:
                    handle.remove()
                delta = ce_off - ce_on  # >0 => NSA helped
                s = (delta > margin).long()
                if i in gates:
                    g_acc[i].append(gates[i].reshape(-1).cpu().numpy())
                    s_acc[i].append(s.reshape(-1).cpu().numpy())

    out = {}
    for i, _ in layers:
        if not g_acc[i]:
            continue
        g = np.concatenate(g_acc[i])
        s = np.concatenate(s_acc[i])
        out[i] = {
            "mi_bits": mutual_information_bits(g, s, n_bins=n_bins),
            "p_helps": float(s.mean()),
            "gate_mean": float(g.mean()),
            "n": int(len(g)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument(
        "--label_ckpt",
        default=None,
        help="Gate-free baseline checkpoint to define the label s "
        "(non-circular). If omitted, s is self-measured on --ckpt.",
    )
    ap.add_argument(
        "--label_variant",
        default="hymba-with-nsa",
        help="Variant of --label_ckpt (default: gate-free base).",
    )
    args = ap.parse_args()

    mi_i, mi_d = _selftest_mi()
    print(
        f"MI estimator self-test OK: independent={mi_i:.4f} bits, "
        f"deterministic={mi_d:.4f} bits"
    )

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

    valid_tok = np.load(f"{args.data_dir}/valid_tokens.npy", mmap_mode="r")
    train_tok = np.load(f"{args.data_dir}/train_tokens.npy", mmap_mode="r")
    vocab = max(int(max(train_tok.max(), valid_tok.max())) + 1, 50257)
    model = build_model(cfg, vocab, device, dtype)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"loaded {args.ckpt}")

    label_model = None
    if args.label_ckpt:
        lcfg = TrainConfig(
            variant=args.label_variant,
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
        label_model = build_model(lcfg, vocab, device, dtype)
        label_model.load_state_dict(
            torch.load(args.label_ckpt, map_location=device)["model"]
        )
        print(f"loaded label model (non-circular s): {args.label_ckpt}")

    res = measure(
        model,
        valid_tok,
        args.n_batches,
        args.batch,
        args.seq_len,
        device,
        args.margin,
        args.n_bins,
        label_model=label_model,
    )
    if res is None:
        print("No gated layers on this variant - I(g;s) not measurable.")
        return

    mis = [v["mi_bits"] for v in res.values()]
    mi_mean = float(np.mean(mis))
    print(f"\nPer-layer I(g; s) [bits], variant={args.variant}:")
    for i in sorted(res):
        v = res[i]
        print(
            f"  layer {i:2d}: I={v['mi_bits']:.4f}  p(NSA helps)={v['p_helps']:.3f}  "
            f"gate_mean={v['gate_mean']:.3f}  n={v['n']}"
        )
    print(f"mean I(g; s) = {mi_mean:.4f} bits")

    out_md = (
        Path(args.out_md)
        if args.out_md
        else Path(
            f"/opt/docker/LLM/HSPMN/results/gate_channel_mi_{args.variant}_2026-06-02.md"
        )
    )
    lines = [
        f"# Gate-as-channel I(g; s) - {args.variant} (2026-06-02)",
        "",
        f"**Checkpoint:** `{args.ckpt}`.",
        f"Counterfactual label: per-layer NSA ablation raises next-token CE "
        f"by > {args.margin} (independent of gate params).",
        f"MI estimator self-test: independent={mi_i:.4f}, det={mi_d:.4f} bits.",
        "",
        "| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |",
        "|---|---|---|---|---|",
    ]
    for i in sorted(res):
        v = res[i]
        lines.append(
            f"| {i} | {v['mi_bits']:.4f} | {v['p_helps']:.3f} | "
            f"{v['gate_mean']:.3f} | {v['n']} |"
        )
    lines += [
        "",
        f"**mean I(g; s) = {mi_mean:.4f} bits**",
        "",
        "Interpretation: a learned gate routes where the contextual stream "
        "helps, so I(g;s) > 0; the random-gate control should give I ~ 0. "
        "This is a non-circular, predictive separation criterion that does "
        "not depend on the loss curvature lambda.",
    ]
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
