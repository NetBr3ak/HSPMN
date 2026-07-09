"""Aggregate Phase-2 synthetics JSON into a comparison markdown.

Reads phase1_synthetics.json (existing variants: dense, v4-gdn-nsa,
v4-mamba3-nsa, v4-rwkv7-nsa, hymba) + phase2_synthetics_2026-05-12.json
(new variants: hymba-with-nsa, hymba-with-nsa-gated, hymba-with-nsa-randgate,
hymba-with-asa) → unified table per task, mark winners + losers.

Output: results/phase2_synthetics_table_2026-05-12.md
"""
import argparse
import json
from pathlib import Path

P1 = Path("/opt/docker/LLM/HSPMN/results/phase1_synthetics.json")
P2 = Path("/opt/docker/LLM/HSPMN/results/phase2_synthetics_2026-05-12.json")
OUT = Path("/opt/docker/LLM/HSPMN/results/phase2_synthetics_table_2026-05-12.md")


def load(p):
    if not p.is_file():
        return []
    with open(p) as f:
        return json.load(f)


def fmt(x, k):
    if x.get("error"):
        return "ERR"
    if k == "accuracy":
        return f"{x['accuracy']*100:.2f}%"
    if k == "ppl":
        return f"{x['ppl']:.3f}" if "ppl" in x else "n/a"
    if k == "loss":
        return f"{x['loss']:.3f}"
    return str(x.get(k, "n/a"))


def main(p1: Path = P1, p2: Path = P2, out: Path = OUT):
    rows = load(p1) + load(p2)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    lines = ["# Phase 2 Synthetics Table - 2026-05-12",
             "",
             "Comparison of all variants (Phase 1 + Phase 2 additions) on the small-scale "
             "synthetic state-tracking battery (1.7M-2.8M params, 4 layers, dim 128, 2000 steps).",
             "",
             "Reviewer-C expressivity probe. State-tracking primitives (Mamba-3, RWKV-7, GDN) "
             "are expected to win on parity / Dyck-2; pure-attention models (dense) are expected "
             "to fail at parity (TC^0 limit).",
             ""]

    for task, recs in by_task.items():
        lines.append(f"## {task}")
        lines.append("")
        if task == "mqar":
            header = "| Variant | Params (M) | Accuracy | Final CE |"
            sep = "|---|---|---|---|"
            keys = ["accuracy", "final_train_ce"]
            metrics = lambda r: [fmt(r, "accuracy"),
                                 f"{r['final_train_ce']:.3f}" if "final_train_ce" in r else "n/a"]
        elif task == "dyck2":
            header = "| Variant | Params (M) | PPL | CE |"
            sep = "|---|---|---|---|"
            metrics = lambda r: [fmt(r, "ppl"), fmt(r, "loss")]
        elif task == "parity":
            header = "| Variant | Params (M) | Accuracy | CE |"
            sep = "|---|---|---|---|"
            metrics = lambda r: [fmt(r, "accuracy"), fmt(r, "loss")]
        else:
            continue
        lines.append(header)
        lines.append(sep)

        # Sort: dense first, then v4-*, then hymba-* family.
        def sort_key(r):
            v = r["variant"]
            return (0 if v == "dense" else 1 if v.startswith("v4-") else 2, v)
        recs_sorted = sorted(recs, key=sort_key)

        # Find best per task.
        best_idx = None
        best_val = None
        for i, r in enumerate(recs_sorted):
            if r.get("error"):
                continue
            if task == "mqar" and "accuracy" in r:
                val = r["accuracy"]
                better = val > (best_val or 0)
            elif task == "dyck2" and "ppl" in r:
                val = r["ppl"]
                better = best_val is None or val < best_val
            elif task == "parity" and "accuracy" in r:
                val = r["accuracy"]
                better = val > (best_val or 0)
            else:
                continue
            if better:
                best_val = val
                best_idx = i

        for i, r in enumerate(recs_sorted):
            if r.get("error"):
                lines.append(f"| {r['variant']} | n/a | ERR | {r['error']} |")
                continue
            params = f"{r.get('params_M', 0):.3f}" if r.get('params_M') else "n/a"
            mvals = metrics(r)
            star = " ★" if i == best_idx else ""
            lines.append(f"| {r['variant']}{star} | {params} | " + " | ".join(mvals) + " |")
        lines.append("")

    # Summary line.
    lines.append("## Headline takeaways")
    lines.append("")
    lines.append("- Best per task marked with ★.")
    lines.append("- Random-gate (Aquino-Michaels baseline) is compared with the gated learned "
                 "variant on hymba-with-nsa - gap between them measures how much routing actually "
                 "carries (per arXiv:2603.02227).")
    lines.append("- v4-* family already shown to lose on wikitext (P2 screen 2026-05-09); these "
                 "synthetic numbers test whether the loss is universal or task-specific.")
    lines.append("")

    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=Path, default=P1,
                        help="Phase 1 synthetics JSON (default: %(default)s)")
    parser.add_argument("--p2", type=Path, default=P2,
                        help="Phase 2 synthetics JSON (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Output markdown file (default: %(default)s)")
    args = parser.parse_args()
    main(p1=args.p1, p2=args.p2, out=args.out)
