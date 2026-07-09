"""Aggregate v6 routability-phase-diagram artifacts into one summary.

Reads: checkpoints_v6/*/*_log.csv (final valid CE), results/v6_ceiling_*.md
(probe ceilings), results/v6_mi_*.md (gate MI), results/v6_split_*.md
(answer vs prose CE). Writes results/v6_routability_summary.md with the
P1-P4 verdict table. Resilient to missing artifacts (prints gaps).
"""

import glob
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = f"{ROOT}/checkpoints_v6"
OUT = f"{ROOT}/results/v6_routability_summary.md"


def last_valid_ce(csv_path):
    ce = None
    with open(csv_path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > 2 and parts[2]:
                ce = float(parts[2])
    return ce


def collect_ppl():
    """{(variant, pi, seed): ppl}. Skips decor-coef sweep dirs (suffix _c*),
    which would otherwise collide with the main-grid coef-0.1 runs."""
    out = {}
    pat = re.compile(r"(hymba-with-nsa[\w-]*)_pi(\d+)_s(\d+)$")
    for d in sorted(glob.glob(f"{CKPT}/*")):
        m = pat.search(os.path.basename(d))
        if not m:
            continue
        variant, pi, seed = m.group(1), m.group(2), int(m.group(3))
        csvs = glob.glob(f"{d}/*_log.csv")
        if not csvs:
            continue
        ce = last_valid_ce(csvs[0])
        if ce is not None:
            out[(variant, pi, seed)] = math.exp(ce)
    return out


def grep_float(path, pattern):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def main():
    ppl = collect_ppl()
    pis = sorted({k[1] for k in ppl})
    lines = ["# v6 routability phase diagram - summary", ""]

    # ---- PPL grid -----------------------------------------------------
    lines += [
        "## Validation PPL by variant x pi (mean +/- std over seeds)",
        "",
        "| variant | " + " | ".join(f"pi=0.{p}" for p in pis) + " |",
        "|---|" + "---|" * len(pis),
    ]
    variants = sorted({k[0] for k in ppl})
    for v in variants:
        row = [v]
        for p in pis:
            vals = [ppl[k] for k in ppl if k[0] == v and k[1] == p]
            row.append(
                f"{np.mean(vals):.2f} +/- {np.std(vals):.2f} (n={len(vals)})"
                if vals
                else "-"
            )
        lines.append("| " + " | ".join(row) + " |")

    # ---- P1: ceiling vs pi ---------------------------------------------
    lines += ["", "## P1 - probe ceiling I_max(pi) [bits]", ""]
    ceilings = {}
    for p in pis:
        val = grep_float(
            f"{ROOT}/results/v6_ceiling_gated_pi{p}.md",
            r"mean held-out I\(pred;s\) = ([\d.]+) bits",
        )
        if val is None:
            val = grep_float(
                f"{ROOT}/results/v6_ceiling_gated_pi{p}.md", r"= ([\d.]+) bits"
            )
        ceilings[p] = val
        lines.append(f"- pi=0.{p}: I_max = {val if val is not None else 'MISSING'}")
    vals = [ceilings[p] for p in pis if ceilings.get(p) is not None]
    if len(vals) >= 2:
        mono = all(b >= a for a, b in zip(vals, vals[1:]))
        lines.append(
            f"\n**P1 verdict: {'CONFIRMED (monotone rise)' if mono else 'FALSIFIED (not monotone)'}**"
        )

    # ---- P2: gated vs randgate per pi -----------------------------------
    lines += ["", "## P2 - learned vs random gate by pi", ""]
    for p in pis:
        g = [ppl[k] for k in ppl if k[0] == "hymba-with-nsa-gated" and k[1] == p]
        r = [ppl[k] for k in ppl if k[0] == "hymba-with-nsa-randgate" and k[1] == p]
        if g and r:
            sep = np.mean(g) + np.std(g) < np.mean(r) - np.std(r)
            lines.append(
                f"- pi=0.{p}: gated {np.mean(g):.2f}+/-{np.std(g):.2f} "
                f"vs randgate {np.mean(r):.2f}+/-{np.std(r):.2f} "
                f"-> {'SEPARATED' if sep else 'tie'}"
            )

    # ---- P3: split eval ---------------------------------------------------
    lines += ["", "## P3 - answer-position vs background CE", ""]
    for f in sorted(glob.glob(f"{ROOT}/results/v6_split_*.md")):
        a = grep_float(f, r"answer positions: n=\d+  CE=([\d.]+)")
        b = grep_float(f, r"background positions: n=\d+  CE=([\d.]+)")
        if a is not None and b is not None:
            lines.append(
                f"- {os.path.basename(f)[9:-3]}: answer CE {a:.3f}, "
                f"background CE {b:.3f}"
            )

    # ---- P4: decor ceiling -----------------------------------------------
    lines += ["", "## P4 - decorrelation effect on ceiling (pi=0.25)", ""]
    dec = grep_float(
        f"{ROOT}/results/v6_ceiling_decor_gated_pi25.md",
        r"mean held-out I\(pred;s\) = ([\d.]+) bits",
    )
    base = ceilings.get("25")
    if dec is not None and base is not None:
        lines.append(
            f"- gated: {base:.4f} bits, decor-gated: {dec:.4f} bits "
            f"-> {'RAISED' if dec > base else 'not raised'}"
        )

    # ---- decor coefficient sweep (pi=0.25, seed 42) -------------------------
    lines += ["", "## Decor coefficient sweep (pi=0.25, s42)", ""]
    for d in sorted(glob.glob(f"{CKPT}/hymba-with-nsa-decor_pi25_s42_c*")):
        coef = os.path.basename(d).rsplit("_c", 1)[-1]
        csvs = glob.glob(f"{d}/*_log.csv")
        ce = last_valid_ce(csvs[0]) if csvs else None
        if ce is not None:
            lines.append(f"- coef {coef}: PPL {math.exp(ce):.2f}")

    # ---- MI table ----------------------------------------------------------
    lines += ["", "## Gate MI (non-circular) by pi", ""]
    for p in pis:
        for tag in ("gated", "randgate"):
            v = grep_float(
                f"{ROOT}/results/v6_mi_{tag}_pi{p}.md",
                r"mean I\(g; ?s\)[^=]*= ?([\d.]+)",
            )
            lines.append(f"- pi=0.{p} {tag}: {v if v is not None else 'MISSING'}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(ppl)} runs aggregated)")


if __name__ == "__main__":
    main()
