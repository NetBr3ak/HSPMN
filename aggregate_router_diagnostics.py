"""Aggregate per-seed router diagnostics into tables for the diagnostic paper.

Reads JSON files emitted by diagnose_v4_router.py:
  results/router_diagnostic_<variant>_lr<lr_tag>_seed<seed>.json

Writes:
  results/router_diagnostics_summary_2026-05-14.md
  results/router_diagnostics_layer_health_2026-05-14.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

RESULTS = Path("/opt/docker/LLM/HSPMN/results")
OUT_MD = RESULTS / "router_diagnostics_summary_2026-05-14.md"
OUT_CSV = RESULTS / "router_diagnostics_layer_health_2026-05-14.csv"


def fmt_mean_sd(xs):
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{mean(xs):.4f} ± {stdev(xs):.4f}"


def main(results_dir: Path = RESULTS, out_md: Path = OUT_MD, out_csv: Path = OUT_CSV):
    paths = sorted(results_dir.glob("router_diagnostic_*_lr*_seed*.json"))
    if not paths:
        raise SystemExit("No router_diagnostic_*.json files found. Run diagnose_v4_router.py first.")

    records = []
    layer_rows = []
    for path in paths:
        rec = json.loads(path.read_text())
        records.append(rec)
        for row in rec["rows"]:
            layer_rows.append({
                "variant": rec["variant"],
                "lr_tag": rec["lr_tag"],
                "seed": rec["seed"],
                "layer": row["layer"],
                "active_fraction": row["active_fraction"],
                "h_route_layer": row.get("h_route_layer", 4 * row["active_fraction"] * (1 - row["active_fraction"])),
                "gate_mean": row["gate_mean"],
                "gate_std": row["gate_std"],
                "gate_conditional_mean": row["gate_conditional_mean"],
                "gate_l1": row["gate_l1"],
            })

    by_key = {}
    for rec in records:
        key = (rec["variant"], rec["lr_tag"])
        by_key.setdefault(key, []).append(rec)

    lines = [
        "# Router diagnostics summary - 2026-05-14",
        "",
        "Aggregates per-seed router diagnostics emitted by `diagnose_v4_router.py`.",
        "",
        "## Variant-level summary",
        "",
        "| Variant | LR | Seeds | mean active_fraction | H_route | dead layers | near-dead layers | saturated layers |",
        "|---|---|---:|---:|---:|---:|---:|---:|",  # noqa: E501
    ]
    for (variant, lr_tag), recs in sorted(by_key.items()):
        seeds = ",".join(str(r["seed"]) for r in sorted(recs, key=lambda x: x["seed"]))
        lines.append(
            f"| {variant} | {lr_tag} | {seeds} | "
            f"{fmt_mean_sd([r['mean_active_fraction'] for r in recs])} | "
            f"{fmt_mean_sd([r['h_route'] for r in recs])} | "
            f"{fmt_mean_sd([float(r['dead_layers']) for r in recs])} | "
            f"{fmt_mean_sd([float(r['near_dead_layers']) for r in recs])} | "
            f"{fmt_mean_sd([float(r['saturated_layers']) for r in recs])} |"
        )

    lines += [
        "",
        "## Interpretation guide",
        "",
        "- `mean active_fraction` is the aggregate number that can look healthy.",
        "- `H_route = mean_l 4 p_l (1-p_l)` is layerwise routing health.",
        "- Low `H_route` with plausible aggregate active fraction is the headline failure mode.",
        "- `near-dead layers` uses active fraction < 0.01; `saturated layers` uses active fraction > 0.95.",
        "",
        "## Layer-level CSV",
        "",
        f"Layer-level data written to `{out_csv}`.",
        "",
    ]
    out_md.write_text("\n".join(lines))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="Results directory (default: %(default)s)")
    parser.add_argument("--out-md", type=Path, default=OUT_MD,
                        help="Output summary markdown file (default: %(default)s)")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV,
                        help="Output layer-level CSV file (default: %(default)s)")
    args = parser.parse_args()
    main(results_dir=args.results_dir, out_md=args.out_md, out_csv=args.out_csv)
