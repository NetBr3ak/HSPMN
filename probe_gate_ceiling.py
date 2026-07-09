"""Gate-ceiling probe - can ANY linear gate of this family cross the threshold?

The gate-channel theorem says a gate resists absorption only if I(g;s) exceeds
H(s) - 1/2 log2 (~0.49 bits when p(s)~1/2), where s = 1[the contextual (NSA)
stream lowers the next-token loss at t]. Trained gates carry only ~0.004 bits.
Two explanations: (A) the signal s is NOT linearly decodable from the gate input
h_t at all (no gate of this form can ever cross the threshold -> the supervised-
gate idea is doomed; the theorem's null is structural), or (B) the signal IS in
h_t but the LM-loss-trained gate never learned it because the backbone absorbs
the gradient (a SUPERVISED gate could cross -> worth training).

This probe decides A vs B WITHOUT any training-loop change: on a trained
checkpoint, collect (h_t, s_t) per gated layer, fit a regularized logistic
probe h_t -> s_t on a TRAIN split, and report the mutual information I(pred; s)
and accuracy on a held-out VAL split. The held-out I is the *ceiling* an
optimally-trained linear gate could reach. Compare it to the ~0.49-bit threshold.

Usage:
    python3 probe_gate_ceiling.py --variant hymba-with-nsa-gated \
        --ckpt checkpoints_p4_gated_s42/hymba-with-nsa-gated_p4_final.pt \
        --n_layers 24 --dim 896 --num_heads 14 --num_kv_heads 2
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device
from measure_gate_channel_mi import mutual_information_bits, per_token_ce, gated_layers

DATA_DIR = "/opt/docker/LLM/HSPMN/data"


def binary_entropy_bits(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def fit_logistic(Xtr, ytr, Xva, epochs=300, lr=0.05, wd=1e-3, device="cpu"):
    """Tiny regularized logistic regression (fp32). Returns val probabilities."""
    Xtr = Xtr.to(device); ytr = ytr.to(device); Xva = Xva.to(device)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    w = torch.zeros(Xtr.size(1), 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        logit = Xtr @ w + b
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pva = torch.sigmoid(Xva @ w + b).squeeze(-1)
    return pva.cpu().numpy()


def collect(model, valid_tok, n_batches, batch, seq, device):
    layers = gated_layers(model)
    if not layers:
        return None
    model.train(False)
    # Capture each gate's input h_t via a pre-hook on attn_gate_proj.
    feats = {i: [] for i, _ in layers}
    cap = {}

    def mk(i, l):
        def pre(mod, inp):
            cap[i] = inp[0].detach()
        return l.attn_gate_proj.register_forward_pre_hook(pre)
    handles = [mk(i, l) for i, l in layers if getattr(l, "attn_gate_proj", None) is not None]
    if not handles:
        return None  # pcgate etc. has no attn_gate_proj input to probe

    rng = np.random.default_rng(0)
    starts = [rng.integers(0, len(valid_tok) - seq - 1, size=(batch,)) for _ in range(n_batches)]
    Hs = {i: [] for i, _ in layers}; Ss = {i: [] for i, _ in layers}
    with torch.no_grad():
        for s_off in starts:
            x = np.stack([valid_tok[a:a + seq] for a in s_off])
            x_t = torch.from_numpy(x).long().to(device)
            cap.clear()
            ce_on = per_token_ce(model, x_t, device)            # [B,S-1]; fills cap
            hcap = {i: cap[i][:, :-1, :].reshape(-1, cap[i].size(-1)).float().cpu()
                    for i in cap}                               # align to CE positions
            for i, l in layers:
                if i not in hcap:
                    continue
                handle = l.nsa.register_forward_hook(
                    lambda m, inp, out: out._replace(out=torch.zeros_like(out.out)))
                try:
                    ce_off = per_token_ce(model, x_t, device)
                finally:
                    handle.remove()
                s = (ce_off - ce_on > 0).long().reshape(-1).cpu()
                Hs[i].append(hcap[i]); Ss[i].append(s)
    for h in handles:
        h.remove()
    return {i: (torch.cat(Hs[i]), torch.cat(Ss[i])) for i in Hs if Hs[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--data_dir", default=DATA_DIR)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = TrainConfig(variant=args.variant, n_layers=args.n_layers, dim=args.dim,
                      num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
                      seq_len=args.seq_len, batch_size=args.batch, grad_accum=1,
                      steps=1, lr=1e-3, warmup_steps=1, nsa_window=256)
    valid_tok = np.load(f"{args.data_dir}/valid_tokens.npy", mmap_mode="r")
    train_tok = np.load(f"{args.data_dir}/train_tokens.npy", mmap_mode="r")
    vocab = max(int(max(train_tok.max(), valid_tok.max())) + 1, 50257)
    model = build_model(cfg, vocab, device, dtype)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    print(f"loaded {args.ckpt}")

    data = collect(model, valid_tok, args.n_batches, args.batch, args.seq_len, device)
    if not data:
        print("No probeable gate (need attn_gate_proj). Use a -gated variant.")
        return

    rows = []
    for i in sorted(data):
        H, S = data[i]
        n = H.size(0); ntr = int(n * 0.7)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        H, S = H[perm], S[perm]
        Xtr, ytr, Xva, yva = H[:ntr], S[:ntr].float(), H[ntr:], S[ntr:]
        p = float(S.float().mean())
        pva = fit_logistic(Xtr, ytr, Xva, device="cpu")
        acc = float(((pva > 0.5).astype(int) == yva.numpy()).mean())
        mi = mutual_information_bits(pva, yva.numpy(), n_bins=12)
        rows.append((i, p, binary_entropy_bits(p), acc, mi))

    p_avg = float(np.mean([r[1] for r in rows]))
    # Full-benefit threshold, Pinsker form (corrected 2026-06-10):
    # I >= 1/2 + H(s) - log2 [nats]  ->  bits: 0.5*log2(e) + H_bits(s) - 1.
    thr = 0.5 * math.log2(math.e) + binary_entropy_bits(p_avg) - 1.0
    mi_avg = float(np.mean([r[4] for r in rows]))
    print(f"\nGate-input -> NSA-helps decodability (held-out), variant={args.variant}")
    print(f"{'layer':>5} {'p(s=1)':>7} {'H(s)':>6} {'val_acc':>8} {'I(pred;s) bits':>14}")
    for i, p, hs, acc, mi in rows:
        print(f"{i:>5} {p:>7.3f} {hs:>6.3f} {acc:>8.3f} {mi:>14.4f}")
    print(f"\nmean held-out I(pred;s) = {mi_avg:.4f} bits  | full-benefit threshold = {thr:.3f} bits")
    verdict = ("CEILING ABOVE THRESHOLD -> signal exists in h_t; a supervised gate "
               "could cross -> WORTH training." if mi_avg > thr else
               "CEILING BELOW THRESHOLD -> the NSA-helps signal is not linearly "
               "decodable from the gate input; NO gate of this form can cross the "
               "threshold. The theorem's null is structural, not a training failure.")
    print(f"VERDICT: {verdict}")

    out = Path(args.out_md) if args.out_md else Path(
        f"/opt/docker/LLM/HSPMN/results/gate_ceiling_{args.variant}_2026-06.md")
    lines = [f"# Gate-ceiling probe - {args.variant}", "",
             f"Checkpoint `{args.ckpt}`. Held-out logistic probe h_t -> s_t "
             f"(s = NSA-ablation raises CE).", "",
             "| layer | p(s=1) | H(s) | val_acc | I(pred;s) bits |",
             "|---|---|---|---|---|"]
    for i, p, hs, acc, mi in rows:
        lines.append(f"| {i} | {p:.3f} | {hs:.3f} | {acc:.3f} | {mi:.4f} |")
    lines += ["", f"**mean held-out I(pred;s) = {mi_avg:.4f} bits** vs "
              f"full-benefit threshold = {thr:.3f} bits (Pinsker form).",
              "", f"**VERDICT:** {verdict}"]
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
