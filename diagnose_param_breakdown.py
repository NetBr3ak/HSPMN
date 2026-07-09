"""Per-block parameter accounting for the 5 P2 variants at 140M config.

H2 (Q/K/V decoupling cost) and H3 (structural fusion cost) need concrete numbers
beyond the total-param column. This script breaks down the parameter budget per
named submodule (projections, norms, MLP, etc.) for one block of each variant.

Output: results/phase2_param_breakdown_2026-05-12.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from train_v4 import TrainConfig, build_model


def block_breakdown(model):
    """Get one block of the model + named-parameter breakdown."""
    # All variants have a `.layers` ModuleList of blocks.
    blk = model.layers[0]
    rows = []
    seen_total = 0
    for name, p in blk.named_parameters():
        n = p.numel()
        rows.append((name, n, list(p.shape)))
        seen_total += n
    return rows, seen_total


def total_breakdown(model):
    """Total breakdown: embed, blocks total, norm, lm_head."""
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n.lower())
    head = sum(p.numel() for n, p in model.named_parameters()
               if "lm_head" in n.lower())
    layers = sum(p.numel() for n, p in model.named_parameters() if ".layers." in n)
    norm_top = sum(p.numel() for n, p in model.named_parameters()
                   if n.startswith("norm.") or n == "norm.weight")
    total = sum(p.numel() for p in model.parameters())
    return {
        "embed": embed,
        "layers": layers,
        "norm_top": norm_top,
        "lm_head": head,
        "total": total,
    }


def aggregate_block_rows(rows):
    """Aggregate by prefix (q_proj, k_proj, v_proj, o_proj, etc.) plus 'misc'."""
    buckets = defaultdict(int)
    for name, n, _ in rows:
        # Use first path component as the bucket.
        bucket = name.split(".")[0]
        buckets[bucket] += n
    return dict(buckets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/opt/docker/LLM/HSPMN/results",
                    help="Output directory for markdown and JSON")
    ap.add_argument("--date_tag", default="2026-05-12",
                    help="Date tag for output filename")
    args = ap.parse_args()

    VARIANTS = ["dense", "hymba", "hymba-with-nsa", "v4-gdn-nsa", "v4-rwkv7-nsa"]
    rows = []
    for v in VARIANTS:
        cfg = TrainConfig(
            variant=v, n_layers=12, dim=768, num_heads=12, num_kv_heads=4,
            seq_len=1024, batch_size=1, grad_accum=1, steps=1, lr=1e-4,
            warmup_steps=1, nsa_window=256,
        )
        m = build_model(cfg, vocab=50257, device=torch.device("cpu"),
                        dtype=torch.float32)
        per_block, _ = block_breakdown(m)
        block_buckets = aggregate_block_rows(per_block)
        block_total = sum(block_buckets.values())
        top = total_breakdown(m)
        rows.append({
            "variant": v,
            "block_buckets": block_buckets,
            "block_total": block_total,
            "n_layers": 12,
            "embed": top["embed"],
            "lm_head": top["lm_head"],
            "total": top["total"],
        })

    # Write a markdown summary.
    out = Path(args.out_dir) / f"phase2_param_breakdown_{args.date_tag}.md"
    lines = [f"# Per-Block Parameter Accounting - {args.date_tag}",
             "",
             "Config: 12 layers / dim 768 / heads 12 / kv-heads 4 / seq 1024 (P2 config).",
             "Vocab assumed 50,257 (GPT-2 BPE; matches wikitext-103 setup).",
             ""]
    lines.append("## Per-block parameter buckets (one block, params)")
    lines.append("")
    all_buckets = set()
    for r in rows:
        all_buckets.update(r["block_buckets"].keys())
    bucket_order = sorted(all_buckets, key=lambda b: (b not in {"q_proj", "k_proj", "v_proj", "o_proj"}, b))
    header_cells = ["Variant", "Block total", *bucket_order]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for r in rows:
        cells = [r["variant"], f"{r['block_total']:,}"]
        for b in bucket_order:
            cells.append(f"{r['block_buckets'].get(b, 0):,}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Whole-model totals")
    lines.append("")
    lines.append("| Variant | Embed | Layers (12 blocks) | LM head | Total |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['variant']} | {r['embed']:,} | {r['n_layers']*r['block_total']:,} | "
                     f"{r['lm_head']:,} | {r['total']:,} ({r['total']/1e6:.2f}M) |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    # Compute hypothetical H2 cost: v4 has q_proj_refl + q_proj_ctx + similar; hymba shares.
    proj_share = lambda r: sum(r["block_buckets"].get(k, 0)
                               for k in r["block_buckets"]
                               if "proj" in k and any(t in k for t in ("q_", "k_", "v_", "o_")))
    for r in rows:
        proj = proj_share(r)
        lines.append(f"- **{r['variant']}** total Q/K/V/O projection params per block: {proj:,}. "
                     f"×12 layers = {proj*12:,} ({proj*12/1e6:.2f}M).")
    lines.append("")
    lines.append("If v4-* projection share is much larger than hymba-with-nsa, H2 (Q/K/V decoupling cost) "
                 "is part of the gap; if comparable, the +19.47% PPL gap is structural (H3) and not H2.")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

    # Also dump JSON.
    out_json = out.with_suffix(".json")
    out_json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
