"""Generate the v5 paper figures (matplotlib -> PDF+PNG in paper/figures/).

All numbers are the honest 3-seed results recorded in
results/phase4_350m_analysis_2026-06-05.md and results/theorem_empirics_2026-06-02.md.
No TeX needed; figures are embedded by build_v5_pdf.py.
"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

C_HEAD, C_DENSE, C_GATED, C_RAND = "#2563eb", "#6b7280", "#16a34a", "#dc2626"


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf/.png")


# --- Fig 1: head-fusion advantage vs dense decays with scale ---------------
def fig_scaling():
    scales = ["140M", "350M"]
    hymba = np.array([139.14, 85.32])
    hymba_sd = np.array([8.43, 2.64])
    dense = np.array([144.01, 83.75])
    dense_sd = np.array([5.64, 2.73])
    delta_pct = (hymba - dense) / dense * 100
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(2)
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(x, delta_pct, "o-", color=C_HEAD, lw=2, ms=8)
    # propagate seed std into the delta% as a rough band
    band = np.sqrt(hymba_sd**2 + dense_sd**2) / dense * 100
    ax.fill_between(x, delta_pct - band, delta_pct + band, color=C_HEAD, alpha=0.15)
    for xi, d in zip(x, delta_pct):
        ax.annotate(
            f"{d:+.2f}%",
            (xi, d),
            textcoords="offset points",
            xytext=(0, 12 if d < 0 else -16),
            ha="center",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel("hymba-with-nsa PPL\nΔ vs dense (%)")
    ax.set_xlabel("model scale")
    ax.set_title(
        "Head-fusion advantage decays with scale\n(−3.39% at 140M → tie at 350M)"
    )
    ax.set_ylim(-7, 5)
    savefig(fig, "fig_scaling_advantage")


# --- Fig 2: gated vs randgate at 350M (the null), with base + dense --------
def fig_gate_null():
    labels = ["hymba\n(base)", "dense", "gated\n(learned)", "randgate\n(A–M)"]
    means = [85.32, 83.75, 82.35, 83.79]
    sds = [2.64, 2.73, 4.78, 1.37]
    cols = [C_HEAD, C_DENSE, C_GATED, C_RAND]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(4)
    ax.bar(x, means, yerr=sds, capsize=5, color=cols, alpha=0.85, error_kw=dict(lw=1.2))
    for xi, m, s in zip(x, means, sds):
        ax.annotate(f"{m:.1f}±{s:.1f}", (xi, m + s + 0.3), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("350M valid PPL (3 seeds)")
    ax.set_ylim(75, 92)
    ax.set_title(
        "Learned gate ties random gate at 350M\n"
        "Δ(gated−randgate)=−1.4 PPL, p≈0.7 (n.s.)"
    )
    savefig(fig, "fig_gate_null_350m")


# --- Fig 3: gate mutual information vs the resistance threshold -------------
def fig_mi_threshold():
    groups = ["140M", "350M"]
    gated = [0.0059, 0.0040]
    gated_sd = [0.0004, 0.0003]
    rand = [0.0029, 0.0021]
    rand_sd = [0.0006, 0.0]
    thr = 0.72
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    x = np.arange(2)
    w = 0.36
    ax.bar(
        x - w / 2,
        gated,
        w,
        yerr=gated_sd,
        capsize=4,
        color=C_GATED,
        label="learned gate",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        rand,
        w,
        yerr=rand_sd,
        capsize=4,
        color=C_RAND,
        label="random gate (A–M)",
        alpha=0.85,
    )
    ax.axhline(thr, color="black", ls="--", lw=1.2)
    ax.annotate(
        f"full-benefit threshold ≈ {thr} bits\n(½+H(s)−log2 nats)",
        (1.25, thr),
        xytext=(0, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.5,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("mean I(g; s)  [bits, log]")
    ax.set_xlabel("model scale")
    ax.set_title(
        "Gate carries ~2× random's information,\nbut stays 2 decades below the full-benefit threshold"
    )
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    savefig(fig, "fig_mi_threshold")


# --- Fig 5: gate-ceiling probe - the structural null --------------------
def fig_gate_ceiling():
    layers = list(range(12))
    lin = [
        0.0146,
        0.0180,
        0.0107,
        0.0231,
        0.0234,
        0.0206,
        0.0307,
        0.0308,
        0.0553,
        0.0271,
        0.0428,
        0.0810,
    ]
    mlp = [
        0.0144,
        0.0124,
        0.0174,
        0.0203,
        0.0259,
        0.0357,
        0.0377,
        0.0417,
        0.0473,
        0.0481,
        0.0395,
        0.0686,
    ]
    thr = 0.72
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(12)
    w = 0.4
    ax.bar(x - w / 2, lin, w, color=C_HEAD, alpha=0.85, label="linear probe on $h_t$")
    ax.bar(
        x + w / 2,
        mlp,
        w,
        color=C_GATED,
        alpha=0.85,
        label="MLP on $[h_t,\\|C\\|,\\|R\\|]$",
    )
    ax.axhline(thr, color="black", ls="--", lw=1.2)
    ax.annotate(
        f"full-benefit threshold ≈ {thr} bits",
        (5.5, thr),
        xytext=(0, -3),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.annotate(
        "ceiling ≈ 20× below threshold",
        (5.5, 0.10),
        ha="center",
        fontsize=9,
        color="#b91c1c",
        fontweight="bold",
    )
    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out I(pred; s)  [bits]")
    ax.set_title(
        "No gate can reach the full-benefit threshold: the NSA-helps signal\n"
        "is nearly information-free from token-local features"
    )
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    savefig(fig, "fig_gate_ceiling")


# --- Fig 6: v6 routability phase diagram (executed programme) --------------
def fig_v6_phase():
    """Data: results/v6_routability_summary.md (137-job queue, 2026-07-09)."""
    pis = np.array([0.0, 0.10, 0.25, 0.50])
    ceiling = np.array([0.0177, 0.0582, 0.1076, 0.1392])
    mi_gated = np.array([0.0034, 0.0043, 0.0410, 0.0331])
    mi_rand = np.array([0.0031, 0.0026, 0.0152, 0.0222])
    thr = 0.72
    ppl = {  # variant: (means, stds) over seeds {42,1337,2026}
        "gate-less": (
            [111.90, 109.63, 103.15, 88.73],
            [3.56, 1.64, 3.13, 2.94],
            C_HEAD,
            "o-",
        ),
        "PC gate": (
            [114.71, 110.59, 101.09, 87.26],
            [4.68, 0.98, 1.78, 1.04],
            "#9333ea",
            "s-",
        ),
        "learned gate": (
            [115.99, 110.31, 104.13, 92.61],
            [2.53, 2.32, 3.05, 2.47],
            C_GATED,
            "^-",
        ),
        "random gate": (
            [115.60, 111.75, 105.04, 90.69],
            [3.46, 1.67, 0.96, 4.40],
            C_RAND,
            "v-",
        ),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))

    ax1.plot(
        pis,
        ceiling,
        "o-",
        color=C_HEAD,
        lw=2,
        ms=7,
        label="probe ceiling $I_{\\max}(\\pi)$",
    )
    ax1.plot(
        pis, mi_gated, "s--", color=C_GATED, lw=1.5, ms=5, label="learned-gate $I(g;s)$"
    )
    ax1.plot(
        pis, mi_rand, "v--", color=C_RAND, lw=1.5, ms=5, label="random-gate $I(g;s)$"
    )
    ax1.axhline(thr, color="black", ls="--", lw=1.2)
    ax1.annotate(
        f"full-benefit threshold = {thr} bits",
        (0.25, thr),
        xytext=(0, -4),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    ax1.set_yscale("log")
    ax1.set_ylim(2e-3, 1.5)
    ax1.set_xlabel("recall fraction $\\pi$")
    ax1.set_ylabel("information  [bits, log]")
    ax1.set_title(
        "P1: the ceiling rises 7.9$\\times$ with $\\pi$,\n"
        "yet stays 5$\\times$ below threshold"
    )
    ax1.legend(loc="center right", fontsize=8, frameon=False)

    for name, (m, s, c, style) in ppl.items():
        m, s = np.array(m), np.array(s)
        ax2.errorbar(
            pis,
            m,
            yerr=s,
            fmt=style,
            color=c,
            lw=1.6,
            ms=5,
            capsize=3,
            label=name,
            alpha=0.9,
        )
    ax2.set_xlabel("recall fraction $\\pi$")
    ax2.set_ylabel("valid PPL (3 seeds)")
    ax2.set_title(
        "P2: learned ties random at every $\\pi$\n"
        "(Welch $p \\geq 0.52$), as the bound requires"
    )
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    savefig(fig, "fig_v6_phase")


if __name__ == "__main__":
    fig_scaling()
    fig_gate_null()
    fig_mi_threshold()
    fig_gate_ceiling()
    fig_v6_phase()
    print("done")
