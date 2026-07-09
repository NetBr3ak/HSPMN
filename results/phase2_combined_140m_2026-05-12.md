# 140M Combined Ablation Table - lr=1e-3 / 3 seeds / 3000 steps

**Date:** 2026-05-12 session. Compiled from P2 + P2b + P2c results.
**Recipe:** 12L / 768d / 12-4 GQA / S=1024 / B=4 ga=8 / bf16 / RTX 5090.

## Combined per-variant results

| Variant | n | Valid CE (mean ± std) | PPL (mean ± std) | tok/s (med) | Δ% vs dense |
|---|---|---|---|---|---|
| `dense` | 3 | 4.9694 ± 0.0397 | 144.01 ± 5.64 | 83,154 | - |
| `hymba` | 3 | 4.9642 ± 0.0683 | 143.42 ± 9.98 | 66,071 | -0.41% |
| `v4-gdn-nsa` | 3 | 5.1107 ± 0.0897 | 166.23 ± 15.12 | 35,578 | +15.43% |
| `v4-rwkv7-nsa` | 3 | 5.1309 ± 0.0834 | 169.55 ± 14.03 | 32,551 | +17.73% |
| `hymba-with-nsa` | 3 | 4.9343 ± 0.0598 | 139.14 ± 8.43 | 50,582 | -3.39% |
| `hymba-with-nsa-gated` | 3 | 4.9675 ± 0.0635 | 143.85 ± 9.27 | 49,720 | -0.11% |
| `hymba-with-nsa-randgate` | 3 | 4.9844 ± 0.0512 | 146.24 ± 7.56 | 50,029 | +1.55% |
| `hymba-with-asa` | 3 | 5.0046 ± 0.0621 | 149.29 ± 9.42 | 61,795 | +3.66% |
| `hymba-with-nsa-pcgate` | 3 | 4.9414 ± 0.0508 | 140.09 ± 7.21 | 49,204 | -2.73% |
| `hymba-with-nsa-select` | 3 | 4.9473 ± 0.0679 | 141.01 ± 9.75 | 45,764 | -2.09% |

## Routing-ablation contrasts at 140M

| Pair (A → B) | Description | A PPL | B PPL | Δ (B − A) |
|---|---|---|---|---|
| `hymba-with-nsa` → `hymba-with-nsa-gated` | Adding a learned per-head sigmoid gate | 139.14 | 143.85 | +4.72 (+3.39%) |
| `hymba-with-nsa-gated` → `hymba-with-nsa-randgate` | Aquino-Michaels: learned vs frozen-random gate parameters | 143.85 | 146.24 | +2.39 (+1.66%) |
| `hymba-with-nsa` → `hymba-with-nsa-pcgate` | Predictive-coding (parameter-free) gate vs gate-less baseline | 139.14 | 140.09 | +0.95 (+0.68%) |
| `hymba-with-nsa` → `hymba-with-asa` | ASA alternation (drop compress; alternate sliding/select per layer) vs NSA-no-select | 139.14 | 149.29 | +10.15 (+7.29%) |
| `hymba-with-nsa` → `hymba-with-nsa-select` | Parameter-free NSA-select-from-compress vs NSA-no-select | 139.14 | 141.01 | +1.87 (+1.34%) |
| `hymba-with-nsa` → `v4-gdn-nsa` | Head-fusion vs block-fusion (same ingredients) - the headline empirical defeat | 139.14 | 166.23 | +27.09 (+19.47%) |

## What we conclude

- **Architectural finding:** head-fusion beats block-fusion at 140M.
- **Routing finding (if applicable):** gated/randgate/pcgate deltas tell us whether the gate apparatus adds anything beyond the gate-less baseline at this scale.
- **350M follow-up:** the winning combination should be re-tested at 350M; Phase 4a runs hymba-with-nsa @ 348.8M for ~1B tokens.