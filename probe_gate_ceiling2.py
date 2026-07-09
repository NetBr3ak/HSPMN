"""Gate-ceiling probe #2 - the STRONGEST realistic gate: nonlinear MLP on rich
stream features. Answers: is the absorption-resistance null structural for the
whole architecture class, or escapable by a richer gate?

probe #1 showed a LINEAR gate on the residual input h_t decodes the NSA-helps
label s at only ~0.03 bits (<< 0.49-bit threshold). Here we give an MLP probe
the richest realistic per-token features a gate could use without a second
forward: h_t, plus per-head norms of the contextual output C_t (pre-gate) and
the reflexive output R_t, plus their difference. If even this stays << 0.49,
the threshold is unreachable for ANY gate of this architecture (structural null,
a strong theorem result). If it crosses, building such a gate is the
breakthrough.

Usage:
    python3 probe_gate_ceiling2.py --variant hymba-with-nsa-gated \
        --ckpt checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/hymba-with-nsa-gated_final.pt \
        --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device
from measure_gate_channel_mi import mutual_information_bits, per_token_ce, gated_layers

DATA_DIR = "/opt/docker/LLM/HSPMN/data"


def binary_entropy_bits(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def fit_mlp(Xtr, ytr, Xva, yva, hidden=64, epochs=400, lr=3e-3, wd=1e-3, device="cpu"):
    Xtr, ytr, Xva = Xtr.to(device), ytr.to(device), Xva.to(device)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    net = nn.Sequential(
        nn.Linear(Xtr.size(1), hidden), nn.GELU(), nn.Linear(hidden, 1)
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(Xtr.size(0), device=device)
        for j in range(0, Xtr.size(0), bs):
            idx = perm[j : j + bs]
            opt.zero_grad()
            logit = net(Xtr[idx]).squeeze(-1)
            F.binary_cross_entropy_with_logits(logit, ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        pva = torch.sigmoid(net(Xva).squeeze(-1)).cpu().numpy()
    return pva


def collect(model, valid_tok, n_batches, batch, seq, device):
    layers = gated_layers(model)
    if not layers:
        return None
    model.train(False)
    capH, capC, capR = {}, {}, {}

    def pre_h(i, layer):
        def f(mod, inp):
            capH[i] = inp[0].detach()

        return layer.attn_gate_proj.register_forward_pre_hook(f)

    def post_c(i, layer):
        def f(mod, inp, out):
            capC[i] = out.out.detach()  # NSAOutput.out [B,nH,S,hd]

        return layer.nsa.register_forward_hook(f)

    def post_r(i, layer):
        def f(mod, inp, out):
            capR[i] = out.detach()  # ssm out [B,S,ssm_dim]

        return layer.ssm.register_forward_hook(f)

    handles = []
    for i, layer in layers:
        if getattr(layer, "attn_gate_proj", None) is None:
            continue
        handles.append(pre_h(i, layer))
        handles.append(post_c(i, layer))
        if getattr(layer, "ssm", None) is not None:
            handles.append(post_r(i, layer))
    if not handles:
        return None

    rng = np.random.default_rng(0)
    starts = [
        rng.integers(0, len(valid_tok) - seq - 1, size=(batch,))
        for _ in range(n_batches)
    ]
    Fs = {i: [] for i, _ in layers}
    Ss = {i: [] for i, _ in layers}
    with torch.no_grad():
        for s_off in starts:
            x = np.stack([valid_tok[a : a + seq] for a in s_off])
            x_t = torch.from_numpy(x).long().to(device)
            capH.clear()
            capC.clear()
            capR.clear()
            ce_on = per_token_ce(model, x_t, device)
            B = x_t.size(0)
            feat = {}
            for i, layer in layers:
                if i not in capH:
                    continue
                h = capH[i][:, :-1, :].float()  # [B,S-1,d]
                c = (
                    capC[i].float().norm(dim=-1).transpose(1, 2)[:, :-1, :]
                )  # [B,S-1,nH]
                parts = [h, c]
                if i in capR:
                    r = capR[i].float()
                    nh = layer.n_ssm_heads
                    r = r.view(B, r.size(1), nh, -1).norm(dim=-1)[
                        :, :-1, :
                    ]  # [B,S-1,nH_ssm]
                    parts.append(r)
                    cmean = c.mean(-1, keepdim=True)
                    rmean = r.mean(-1, keepdim=True)
                    parts.append(cmean - rmean)
                feat[i] = torch.cat(
                    [p.reshape(-1, p.size(-1)) for p in parts], dim=-1
                ).cpu()
            for i, layer in layers:
                if i not in feat:
                    continue
                handle = layer.nsa.register_forward_hook(
                    lambda m, inp, out: out._replace(out=torch.zeros_like(out.out))
                )
                try:
                    ce_off = per_token_ce(model, x_t, device)
                finally:
                    handle.remove()
                s = (ce_off - ce_on > 0).long().reshape(-1).cpu()
                Fs[i].append(feat[i])
                Ss[i].append(s)
    for h in handles:
        h.remove()
    return {i: (torch.cat(Fs[i]), torch.cat(Ss[i])) for i in Fs if Fs[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="MLP hidden width (capacity check: try 128/256)",
    )
    ap.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Train/val split permutation seed (CI over splits)",
    )
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
    print(f"loaded {args.ckpt}")

    data = collect(model, valid_tok, args.n_batches, args.batch, args.seq_len, device)
    if not data:
        print("No probeable gate.")
        return
    pdev = "cuda" if device.type == "cuda" else "cpu"
    rows = []
    for i in sorted(data):
        Fz, S = data[i]
        n = Fz.size(0)
        g = torch.Generator().manual_seed(args.split_seed)
        perm = torch.randperm(n, generator=g)
        Fz, S = Fz[perm], S[perm]
        ntr = int(n * 0.7)
        pva = fit_mlp(
            Fz[:ntr],
            S[:ntr].float(),
            Fz[ntr:],
            S[ntr:].float(),
            hidden=args.hidden,
            device=pdev,
        )
        yva = S[ntr:].numpy()
        acc = float(((pva > 0.5).astype(int) == yva).mean())
        mi = mutual_information_bits(pva, yva, n_bins=12)
        rows.append((i, float(S.float().mean()), Fz.size(1), acc, mi))

    p_avg = float(np.mean([r[1] for r in rows]))
    thr = binary_entropy_bits(p_avg) - 0.5
    mi_avg = float(np.mean([r[4] for r in rows]))
    print("\nMLP probe on [h_t, ||C||/head, ||R||/head, ||C||-||R||] -> NSA-helps")
    print(
        f"{'layer':>5} {'p(s=1)':>7} {'featdim':>7} {'val_acc':>8} {'I(pred;s) bits':>14}"
    )
    for i, p, fd, acc, mi in rows:
        print(f"{i:>5} {p:>7.3f} {fd:>7d} {acc:>8.3f} {mi:>14.4f}")
    print(f"\nmean held-out I(pred;s) = {mi_avg:.4f} bits | threshold = {thr:.3f} bits")
    verdict = (
        "ABOVE THRESHOLD -> a rich nonlinear gate could resist absorption; "
        "BUILD IT (breakthrough)."
        if mi_avg > thr
        else "BELOW THRESHOLD even for a nonlinear gate on both streams -> the "
        "NSA-helps signal is fundamentally low-information in this "
        "architecture. The absorption-resistance null is STRUCTURAL and "
        "class-wide, not a gate-design failure. Strong theorem corollary."
    )
    print(f"VERDICT: {verdict}")
    out = Path(f"/opt/docker/LLM/HSPMN/results/gate_ceiling2_{args.variant}_2026-06.md")
    lines = [
        f"# Gate-ceiling probe #2 (MLP + stream features) - {args.variant}",
        "",
        f"Checkpoint `{args.ckpt}`.",
        "",
        "| layer | p(s=1) | feat dim | val_acc | I(pred;s) bits |",
        "|---|---|---|---|---|",
    ]
    for i, p, fd, acc, mi in rows:
        lines.append(f"| {i} | {p:.3f} | {fd} | {acc:.3f} | {mi:.4f} |")
    lines += [
        "",
        f"**mean held-out I(pred;s) = {mi_avg:.4f} bits** vs threshold {thr:.3f} bits.",
        "",
        f"**VERDICT:** {verdict}",
    ]
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
