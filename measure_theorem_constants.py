"""Measure the absorption-resistance theorem constants on a trained model.

Per `results/phase3_theorem_draft_2026-05-12.md`:

  rho >= 1 + (lambda * sigma_C^2 * p * (1-p) * gamma^2)
              / (2 * L(theta_*)) * (1 - O(delta, sigma_g))

Constants to measure:
  - sigma_C: RMS of NSA contribution per token.
  - p: fraction of tokens where gate > median.
  - gamma: gate saturation margin: 1 - 2*delta.
  - lambda: curvature of the loss along the gate-action axis (fp32 scalar
            gate-gain finite difference; see measure_lambda_gate_gain).
  - L(theta_*): the trained model's empirical loss.

Output: per-layer table + summary.

Usage:
    python3 measure_theorem_constants.py --variant hymba-with-nsa --ckpt PATH
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device

DATA_DIR = "/opt/docker/LLM/HSPMN/data"


def measure_sigma_C(model, valid_tok, n_batches, batch, seq, device):
    """Forward batches, collect NSA-branch output magnitudes per layer."""
    norms_by_layer = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            o = output.out
            n = o.float().norm(dim=-1)
            norms_by_layer.setdefault(layer_idx, []).append(float(n.mean().item()))
        return hook

    for idx, layer in enumerate(model.layers):
        nsa = getattr(layer, "nsa", None)
        if nsa is not None:
            handles.append(nsa.register_forward_hook(make_hook(idx)))

    model.train(False)
    with torch.no_grad():
        for _ in range(n_batches):
            starts = np.random.randint(0, len(valid_tok) - seq - 1, size=(batch,))
            x = np.stack([valid_tok[s:s + seq] for s in starts])
            x_t = torch.from_numpy(x).long().to(device)
            model(x_t)

    for h in handles:
        h.remove()
    return {k: float(np.mean(v)) for k, v in norms_by_layer.items()}


def measure_gate_stats(model, valid_tok, n_batches, batch, seq, device):
    """Per-layer (p, gamma) from cached `_last_gate` in HymbaWithNSABlock."""
    stats_by_layer = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            g = getattr(module, "_last_gate", None)
            if g is None:
                return
            gf = g.float()
            p_active = float((gf > 0.5).float().mean().item())
            delta = float(torch.minimum(gf, 1 - gf).mean().item())
            stats_by_layer.setdefault(layer_idx, []).append({
                "p": p_active, "delta": delta,
                "gate_mean": float(gf.mean().item()),
                "gate_std": float(gf.std().item()),
            })
        return hook

    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "attn_gate_proj") or getattr(layer, "gate_mode", "linear") == "predictive_coding":
            handles.append(layer.register_forward_hook(make_hook(idx)))

    model.train(False)
    with torch.no_grad():
        for _ in range(n_batches):
            starts = np.random.randint(0, len(valid_tok) - seq - 1, size=(batch,))
            x = np.stack([valid_tok[s:s + seq] for s in starts])
            x_t = torch.from_numpy(x).long().to(device)
            model(x_t)

    for h in handles:
        h.remove()

    aggregated = {}
    for k, recs in stats_by_layer.items():
        if recs:
            aggregated[k] = {kk: float(np.mean([r[kk] for r in recs])) for kk in recs[0]}
            aggregated[k]["gamma"] = 1.0 - 2.0 * aggregated[k]["delta"]
    return aggregated


def estimate_L_star(model, valid_tok, n_batches, batch, seq, device):
    """Empirical loss = mean CE on validation."""
    model.train(False)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            starts = np.random.randint(0, len(valid_tok) - seq - 1, size=(batch,))
            x = np.stack([valid_tok[s:s + seq] for s in starts])
            y = np.stack([valid_tok[s + 1:s + seq + 1] for s in starts])
            x_t = torch.from_numpy(x).long().to(device)
            y_t = torch.from_numpy(y).long().to(device)
            out = model(x_t, labels=y_t)
            losses.append(float(out["loss"].item()))
    return float(np.mean(losses))


def _eval_loss_on_fixed_batches(model, valid_tok, fixed_starts, batch, seq, device):
    """Eval mean loss on a fixed set of validation windows (reproducible).

    Loss is accumulated in fp32 to avoid bf16 rounding noise in finite-difference.
    """
    model.train(False)
    loss_acc = 0.0  # fp32 accumulator
    count = 0
    with torch.no_grad():
        for s_off in fixed_starts:
            x = np.stack([valid_tok[s:s + seq] for s in s_off])
            y = np.stack([valid_tok[s + 1:s + seq + 1] for s in s_off])
            x_t = torch.from_numpy(x).long().to(device)
            y_t = torch.from_numpy(y).long().to(device)
            out = model(x_t, labels=y_t)
            # Convert to fp32 for accumulation to avoid bf16 quantization noise
            loss_acc += float(out["loss"].float().item())
            count += 1
    return loss_acc / max(count, 1)


def _selftest_fd_curvature():
    """Validate the 5-point finite-difference curvature estimator on f(x)=x^2
    (true second derivative 2.0) before trusting it on the model. Guards against
    the kind of silent numerical failure that produced BUG-1 (lambda == 0)."""
    f = lambda a: a * a
    a0, h = 1.0, 1e-2
    kappa = (-f(a0 + 2 * h) + 16 * f(a0 + h) - 30 * f(a0)
             + 16 * f(a0 - h) - f(a0 + -2 * h)) / (12 * h * h)
    assert abs(kappa - 2.0) < 1e-3, f"FD self-test failed: kappa={kappa} (expected 2.0)"
    return float(kappa)


def measure_lambda_gate_gain(model, valid_tok, n_batches, batch, seq, device,
                             eps=1e-2):
    """Curvature of the loss w.r.t. a scalar gate-gain alpha, measured in fp32.

    Every ``attn_gate_proj`` output is scaled by a single scalar ``alpha``
    (alpha == 1 reproduces the trained model); we measure d^2 L / d alpha^2 at
    alpha == 1 with a 5-point finite-difference stencil.

    Why this fixes BUG-1: the original estimator perturbed a random unit-norm
    direction across *all* gate weights, so a 1e-2 step diluted to ~1e-4 per
    weight -- below the bf16 ULP at loss ~5, making L+ == L0 == L- and kappa == 0.
    A single scalar alpha concentrates the whole 1e-2 step into one degree of
    freedom, and evaluating in fp32 removes bf16 rounding from the difference, so
    the curvature is well resolved.

    Interpretation: this is the curvature of the loss along the gate-action axis
    -- exactly the quantity the absorption-resistance bound's ``lambda`` refers
    to -- NOT the smallest global Hessian eigenvalue (which for a neural net is
    ~0 or negative and is not what the bound needs). An honest near-zero result
    here is a real finding (gate confidence barely curves the loss), no longer a
    numerical artifact.
    """
    _selftest_fd_curvature()

    gate_params = []
    for l in model.layers:
        agp = getattr(l, "attn_gate_proj", None)
        if agp is not None:
            gate_params.append(agp.weight)
            if agp.bias is not None:
                gate_params.append(agp.bias)
    if not gate_params:
        return None  # No learned gate on this variant.

    # Measurement runs in fp32 to keep the finite difference above rounding noise.
    if next(model.parameters()).dtype != torch.float32:
        model.float()
    orig = [p.data.clone() for p in gate_params]

    rng = np.random.default_rng(seed=0)
    fixed_starts = [rng.integers(0, len(valid_tok) - seq - 1, size=(batch,))
                    for _ in range(n_batches)]

    def L_at(alpha):
        for p, o in zip(gate_params, orig):
            p.data.copy_(alpha * o)
        return _eval_loss_on_fixed_batches(model, valid_tok, fixed_starts,
                                           batch, seq, device)

    a, h = 1.0, eps
    Lm2, Lm1, L0, Lp1, Lp2 = (L_at(a - 2 * h), L_at(a - h), L_at(a),
                              L_at(a + h), L_at(a + 2 * h))
    for p, o in zip(gate_params, orig):  # restore exactly
        p.data.copy_(o)

    kappa = (-Lp2 + 16 * Lp1 - 30 * L0 + 16 * Lm1 - Lm2) / (12 * h * h)
    print(f"  gate-gain curvature: L(1)={L0:.6f} "
          f"stencil=[{Lm2:.6f},{Lm1:.6f},{L0:.6f},{Lp1:.6f},{Lp2:.6f}] "
          f"lambda={kappa:.6e}")
    return {
        "L0": float(L0),
        "lambda_gate_gain": float(kappa),
        "lambda_min_estimate": float(kappa),  # back-compat key for downstream
        "lambda_mean": float(kappa),
        "lambda_max_estimate": float(kappa),
        "stencil": [float(x) for x in (Lm2, Lm1, L0, Lp1, Lp2)],
        "eps": eps,
        "method": "fp32 5-point FD on scalar gate-gain alpha",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--lambda_eps", type=float, default=1e-2,
                    help="Finite-difference step size for lambda (default 1e-2 to escape bf16 noise).")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg = TrainConfig(
        variant=args.variant, n_layers=args.n_layers, dim=args.dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len, batch_size=args.batch, grad_accum=1,
        steps=1, lr=1e-3, warmup_steps=1, nsa_window=256,
    )

    valid_tok = np.load(f"{DATA_DIR}/valid_tokens.npy", mmap_mode="r")
    train_tok = np.load(f"{DATA_DIR}/train_tokens.npy", mmap_mode="r")
    vocab = int(max(train_tok.max(), valid_tok.max())) + 1
    model = build_model(cfg, vocab, device, dtype)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"loaded {args.ckpt}")

    L_star = estimate_L_star(model, valid_tok, args.n_batches, args.batch,
                             args.seq_len, device)
    print(f"L(theta_*) = {L_star:.4f}  ppl = {math.exp(L_star):.2f}")

    sigma_C = measure_sigma_C(model, valid_tok, args.n_batches, args.batch,
                              args.seq_len, device)
    print(f"sigma_C per layer: {sigma_C}")

    gate = measure_gate_stats(model, valid_tok, args.n_batches, args.batch,
                              args.seq_len, device)
    print(f"gate stats per layer: {gate}")

    print("\nMeasuring lambda via fp32 gate-gain curvature (5-point FD)...")
    lambda_data = measure_lambda_gate_gain(
        model, valid_tok, args.n_batches, args.batch, args.seq_len, device,
        eps=getattr(args, "lambda_eps", 1e-2),
    )
    if lambda_data is None:
        print("No attn_gate_proj layers - lambda not measurable on this variant.")
    else:
        print(f"lambda_gate_gain = {lambda_data['lambda_gate_gain']:.6e}")

    if not args.out_md:
        out_md = Path(f"/opt/docker/LLM/HSPMN/results/phase3_constants_"
                      f"{args.variant}_2026-05-12.md")
    else:
        out_md = Path(args.out_md)
    lines = [f"# Phase 3 Empirical Constants - {args.variant} (2026-05-12)",
             "",
             f"**Checkpoint:** `{args.ckpt}`.",
             f"**Validation batches:** {args.n_batches} x {args.batch} x {args.seq_len}.",
             "",
             f"## Loss",
             "",
             f"- L(theta_*) = {L_star:.4f}  (PPL {math.exp(L_star):.2f})",
             "",
             "## Per-layer constants",
             "",
             "| Layer | sigma_C (mean ||C||) | p | gamma | gate.mean | gate.std |",
             "|---|---|---|---|---|---|"]
    layers = sorted(set(sigma_C) | set(gate))
    for layer_idx in layers:
        s = sigma_C.get(layer_idx, "-")
        s_str = f"{s:.3f}" if isinstance(s, float) else "-"
        g = gate.get(layer_idx)
        if g:
            p = g["p"]; gm = g["gate_mean"]; gs = g["gate_std"]; gma = g["gamma"]
            lines.append(f"| {layer_idx} | {s_str} | {p:.3f} | {gma:.3f} | {gm:.3f} | {gs:.3f} |")
        else:
            lines.append(f"| {layer_idx} | {s_str} | - | - | - | - |")
    lines.append("")
    if sigma_C and gate:
        sigma_C_avg = float(np.mean(list(sigma_C.values())))
        p_avg = float(np.mean([g["p"] for g in gate.values()]))
        gamma_avg = float(np.mean([g["gamma"] for g in gate.values()]))
        if lambda_data is not None:
            lambda_used = lambda_data["lambda_gate_gain"]
            lambda_src = (f"gate-gain curvature, {lambda_data['method']} "
                          f"(eps={lambda_data['eps']})")
        else:
            lambda_used = 0.1
            lambda_src = "placeholder (no gate; gate-gain curvature disabled)"
        rho_lb = 1.0 + (lambda_used * sigma_C_avg**2
                        * p_avg * (1 - p_avg) * gamma_avg**2
                        / max(2.0 * L_star, 1e-6))
        lines.append(f"## Bound estimate")
        lines.append("")
        lines.append(f"- sigma_C (avg) = {sigma_C_avg:.4f}")
        lines.append(f"- p (avg) = {p_avg:.4f}")
        lines.append(f"- gamma (avg) = {gamma_avg:.4f}")
        lines.append(f"- lambda = {lambda_used:.6e} ({lambda_src})")
        lines.append(f"- L(theta_*) = {L_star:.4f}")
        lines.append(f"- **rho_LB - 1 ~ {rho_lb - 1:.6e}**")
        lines.append("")
        if lambda_data is not None:
            lines.append("## Lambda gate-gain curvature details")
            lines.append("")
            lines.append(f"- method: {lambda_data['method']} (eps={lambda_data['eps']})")
            sten = lambda_data["stencil"]
            lines.append(f"- 5-point stencil L(1-2h..1+2h): "
                         f"{', '.join(f'{x:.6f}' for x in sten)}")
            lines.append(f"- lambda_gate_gain = {lambda_data['lambda_gate_gain']:.6e}")
            lines.append("")
            lines.append("**Interpretation:** lambda is the curvature of the loss "
                         "along the gate-action axis (uniform scaling of the gate "
                         "logit by a scalar alpha at alpha=1), measured in fp32 so "
                         "the difference is above bf16 rounding. This is the "
                         "directional curvature the absorption-resistance bound "
                         "refers to, not the smallest global Hessian eigenvalue. The "
                         "estimator is self-tested on f(x)=x^2 (kappa=2.0) before use.")
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
