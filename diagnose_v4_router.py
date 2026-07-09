"""H1 diagnostic - probe ReMoE gate distribution on a trained v4 checkpoint.

Hypothesis H1 (Phase 2 assessment): v4 block-fusion's +19% PPL gap vs head-fusion
is partly explained by ReMoE gate collapse - most tokens have gate ≈ 0 → ctx stream
silently skipped for 75% of tokens.

Test: load `v4-gdn-nsa_lr1e-3_seed42` checkpoint, forward a held-out validation
batch, log per-layer gate statistics:
    - active_fraction (gate > 0)
    - gate.mean(), gate.std()
    - gate.mean() conditional on gate>0
    - L1 of gate distribution

Per-checkpoint output feeds the aggregate in
results/router_diagnostics_summary_2026-05-14.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device

# Module-level defaults; override via --data_dir and --ckpt_root args
_DEFAULT_DATA_DIR = "./data"
_DEFAULT_CKPT_ROOT = "./checkpoints_p2screen"


def gather_router_stats(model, valid_tok, n_batches, batch, seq, device):
    """Forward batches, hook every ReMoE router, return per-layer stats."""
    stats = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            g = output.gate
            stats.setdefault(layer_idx, []).append(
                {
                    "gate_mean": float(g.mean().item()),
                    "gate_std": float(g.std().item()),
                    "active_fraction": float((g > 0).float().mean().item()),
                    "gate_conditional_mean": float(
                        g[g > 0].mean().item() if (g > 0).any() else 0.0
                    ),
                    "gate_l1": float(g.abs().mean().item()),
                }
            )

        return hook

    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "router"):
            handles.append(layer.router.register_forward_hook(make_hook(idx)))

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            starts = np.random.randint(0, len(valid_tok) - seq - 1, size=(batch,))
            x = np.stack([valid_tok[s : s + seq] for s in starts])
            x_t = torch.from_numpy(x).long().to(device)
            model(x_t)

    for h in handles:
        h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", default="v4-gdn-nsa", choices=["v4-gdn-nsa", "v4-rwkv7-nsa"]
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_tag", default="1e-3")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument(
        "--data_dir",
        default=_DEFAULT_DATA_DIR,
        help="Path to data directory with tokenized validation/train splits",
    )
    ap.add_argument(
        "--ckpt_root",
        default=_DEFAULT_CKPT_ROOT,
        help="Root directory of checkpoint checkpoints",
    )
    ap.add_argument(
        "--out_dir",
        default="./results",
        help="Output directory for markdown and JSON results",
    )
    ap.add_argument(
        "--date_tag", default="2026-05-12", help="Date tag for output filename"
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

    valid_tok = np.load(f"{args.data_dir}/valid_tokens.npy", mmap_mode="r")
    train_tok = np.load(f"{args.data_dir}/train_tokens.npy", mmap_mode="r")
    vocab = int(max(train_tok.max(), valid_tok.max())) + 1
    print(f"vocab={vocab} valid_tokens={len(valid_tok):,}")

    model = build_model(cfg, vocab, device, dtype)
    ckpt_dir = Path(args.ckpt_root) / f"{args.variant}_lr{args.lr_tag}_seed{args.seed}"
    ckpt_path = ckpt_dir / f"{args.variant}_final.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"loaded ckpt: {ckpt_path}")

    stats = gather_router_stats(
        model, valid_tok, args.n_batches, args.batch, args.seq_len, device
    )
    if not stats:
        print("WARNING: no router stats recorded - model has no `router` attribute.")
        return

    rows = []
    for layer_idx in sorted(stats):
        recs = stats[layer_idx]
        ms = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        ms["std_active_fraction"] = float(np.std([r["active_fraction"] for r in recs]))
        ms["layer"] = layer_idx
        rows.append(ms)

    af = [r["active_fraction"] for r in rows]
    cm = [r["gate_conditional_mean"] for r in rows]
    h_route_by_layer = [4.0 * p * (1.0 - p) for p in af]
    overall = {
        "variant": args.variant,
        "lr_tag": args.lr_tag,
        "seed": args.seed,
        "n_layers": len(rows),
        "n_batches": args.n_batches,
        "mean_active_fraction": float(np.mean(af)),
        "min_active_fraction": float(np.min(af)),
        "max_active_fraction": float(np.max(af)),
        "h_route": float(np.mean(h_route_by_layer)),
        "dead_layers": int(np.sum(np.array(af) <= 1e-6)),
        "near_dead_layers": int(np.sum(np.array(af) < 0.01)),
        "saturated_layers": int(np.sum(np.array(af) > 0.95)),
        "mean_gate_conditional_mean": float(np.mean(cm)),
        "rows": rows,
    }
    for r, h in zip(rows, h_route_by_layer):
        r["h_route_layer"] = float(h)

    out_dir = Path(args.out_dir)
    out_stem = f"router_diagnostic_{args.variant}_lr{args.lr_tag}_seed{args.seed}"
    out_md = out_dir / f"{out_stem}.md"
    lines = [
        f"# H1 Router-Collapse Diagnostic on {args.variant}",
        "",
        f"**Date:** {args.date_tag}. **Checkpoint:** `{ckpt_path}`.",
        f"**Validation batches:** {args.n_batches} × {args.batch} × {args.seq_len}.",
        "",
        "## Headline numbers",
        "",
        f"- mean active_fraction across layers: {overall['mean_active_fraction']:.4f}",
        "- target_sparsity (config): 0.25 (set by ALF-LB bias)",
        f"- min/max active_fraction: {overall['min_active_fraction']:.4f} / {overall['max_active_fraction']:.4f}",
        f"- layerwise routing-health H_route: {overall['h_route']:.4f}",
        f"- dead / near-dead / saturated layers: {overall['dead_layers']} / {overall['near_dead_layers']} / {overall['saturated_layers']}",
        f"- mean gate.mean()|active: {overall['mean_gate_conditional_mean']:.4f}",
        "",
        "## Per-layer",
        "",
        "| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['active_fraction']:.4f} | "
            f"{r['h_route_layer']:.4f} | {r['std_active_fraction']:.4f} | {r['gate_mean']:.4f} | "
            f"{r['gate_std']:.4f} | {r['gate_conditional_mean']:.4f} | "
            f"{r['gate_l1']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    mean_af = overall["mean_active_fraction"]
    if mean_af < 0.05:
        lines.append(
            f"**H1 CONFIRMED:** mean active_fraction = {mean_af:.4f} far below "
            "target_sparsity=0.25; ReMoE gate collapsed near-off. Contextual stream "
            "effectively skipped on ~95% of tokens."
        )
    elif abs(mean_af - 0.25) < 0.05:
        lines.append(
            f"**H1 NOT confirmed:** mean active_fraction = {mean_af:.4f} matches "
            "target_sparsity=0.25 - ALF-LB held the gate. Loss does not come from "
            "gate-side collapse."
        )
    else:
        lines.append(
            f"**H1 partial:** mean active_fraction = {mean_af:.4f} drifted from "
            "target=0.25. Gate is not fully off, but ALF-LB did not hold target - "
            "partial absorption likely."
        )
    lines.append("")
    out_md.write_text("\n".join(lines))
    print(f"\n→ {out_md}")
    print(f"summary: {overall}")

    (out_dir / f"{out_stem}.json").write_text(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
