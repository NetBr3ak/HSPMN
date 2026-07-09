"""Plot router active-fraction heatmaps from aggregate_router_diagnostics.py CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main(csv_path: Path, out_png: Path, out_pdf: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib unavailable: {exc}")

    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["layer"] = int(row["layer"])
            row["active_fraction"] = float(row["active_fraction"])
            rows.append(row)

    variants = sorted({r["variant"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    layers = list(range(12))

    fig, axes = plt.subplots(len(variants), 1, figsize=(9.0, 3.2 * len(variants)), constrained_layout=True)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        mat = np.full((len(seeds), len(layers)), np.nan)
        for r in rows:
            if r["variant"] == variant:
                mat[seeds.index(r["seed"]), r["layer"]] = r["active_fraction"]
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(f"{variant}: per-layer active fraction")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Seed")
        ax.set_xticks(layers)
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([str(s) for s in seeds])
        for i, seed in enumerate(seeds):
            for j, layer in enumerate(layers):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if v < 0.35 or v > 0.75 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.9)
    cbar.set_label("active fraction")
    fig.suptitle("Batch-level balance masks layerwise router collapse", fontsize=13)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/opt/docker/LLM/HSPMN"),
        help="Project root (default: /opt/docker/LLM/HSPMN)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Router diagnostics CSV (default: <root>/results/router_diagnostics_layer_health_<date>.csv)",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG path (default: <root>/paper/router_active_fraction_heatmap.png)",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Output PDF path (default: <root>/paper/router_active_fraction_heatmap.pdf)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2026-05-14",
        help="Date string for CSV filename (default: 2026-05-14)",
    )
    args = parser.parse_args()

    results = args.root / "results"
    paper = args.root / "paper"
    csv_path = args.csv or (results / f"router_diagnostics_layer_health_{args.date}.csv")
    out_png = args.out_png or (paper / "router_active_fraction_heatmap.png")
    out_pdf = args.out_pdf or (paper / "router_active_fraction_heatmap.pdf")

    main(csv_path, out_png, out_pdf)
