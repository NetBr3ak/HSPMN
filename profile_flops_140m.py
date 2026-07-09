"""FLOPs profile for the P2 + P2b + P2c + P2d variants at the 140M config.

Uses torch.utils.flop_counter.FlopCounterMode to count multiply-add operations
during a single forward pass. Cross-checks with the per-token ratio
6 * num_params (Kaplan et al. 2020) and reports both.

Output: results/phase2_flops_140m_2026-05-12.md
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from train_v4 import TrainConfig, build_model


VARIANTS = [
    "dense", "hymba", "hymba-with-nsa",
    "hymba-with-nsa-gated", "hymba-with-nsa-randgate",
    "hymba-with-nsa-pcgate", "hymba-with-nsa-select",
    "hymba-with-asa",
    "v4-gdn-nsa", "v4-rwkv7-nsa",
]


def profile_one(variant, n_layers=12, dim=768, num_heads=12, num_kv_heads=4,
                seq_len=1024, batch=1, vocab=50257):
    cfg = TrainConfig(
        variant=variant, n_layers=n_layers, dim=dim,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        seq_len=seq_len, batch_size=batch, grad_accum=1,
        steps=1, lr=1e-3, warmup_steps=1, nsa_window=256,
    )
    device = torch.device("cpu")
    dtype = torch.float32
    model = build_model(cfg, vocab, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, vocab, (batch, seq_len))
    flop_counter = FlopCounterMode(display=False, depth=0)
    with flop_counter:
        model(ids)
    flops = flop_counter.get_total_flops()
    return {
        "variant": variant,
        "n_params": n_params,
        "n_params_M": n_params / 1e6,
        "flops_fwd": flops,
        "flops_per_token_fwd": flops / (batch * seq_len),
        "kaplan_6p_per_token": 6 * n_params,   # forward + backward, Kaplan approx
        "kaplan_2p_per_token_fwd": 2 * n_params,  # forward only, Kaplan approx
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--output", type=str, default="/opt/docker/LLM/HSPMN/results/phase2_flops_140m_2026-05-12.md")
    args = ap.parse_args()

    results = []
    for v in args.variants:
        print(f"Profiling {v}...")
        try:
            r = profile_one(v, seq_len=args.seq_len)
            results.append(r)
            print(f"  {v}: {r['n_params_M']:.2f}M params, "
                  f"{r['flops_fwd']/1e9:.2f}G FLOPs fwd, "
                  f"{r['flops_per_token_fwd']/1e6:.2f}M FLOPs/token")
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}): {e}")
            results.append({"variant": v, "error": f"{type(e).__name__}: {e}"})

    out = Path(args.output)
    lines = ["# FLOPs Profile at 140M Config - 2026-05-12",
             "",
             "Forward-pass FLOPs counted via `torch.utils.flop_counter.FlopCounterMode` "
             f"at seq_len={args.seq_len}, batch=1, vocab=50257, 12L/768d/12-4 GQA.",
             "",
             "Kaplan approx: training FLOPs/token ≈ 6 × params (forward 2x params, "
             "backward 4x params). The measured forward FLOPs/token can be larger "
             "than 2P due to attention's $O(S^2)$ component (compress + sliding-window + "
             "select branches in NSA).",
             "",
             "| Variant | Params (M) | Fwd FLOPs (G) | Fwd FLOPs/tok (M) | 2P approx (M) | Δ vs 2P |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['variant']} | ERR | {r['error']} | | | |")
            continue
        fpt = r["flops_per_token_fwd"] / 1e6
        twop = r["kaplan_2p_per_token_fwd"] / 1e6
        delta = (fpt - twop) / twop * 100 if twop > 0 else 0
        lines.append(f"| `{r['variant']}` | {r['n_params_M']:.2f} | "
                     f"{r['flops_fwd']/1e9:.2f} | "
                     f"{fpt:.2f} | {twop:.2f} | {delta:+.1f}% |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Variants with NSA's compress + sliding-window branches add O(S^2) "
                 "compute on top of the 2P forward base.")
    lines.append("- ASA mode drops the compress branch → FLOPs/token should be lower than "
                 "NSA at the same parameter count.")
    lines.append("- PC-gate adds a small per-token reduction (norm + sigmoid) - negligible.")
    lines.append("- Block-fusion variants (v4-*) have a separate full-attention contextual "
                 "stream and a separate reflexive stream; total FLOPs are higher than "
                 "head-fusion at matched params.")
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")
    (out.with_suffix(".json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
