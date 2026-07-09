# Gate-as-channel I(g; s) - hymba-with-nsa-randgate (2026-06-02)

**Checkpoint:** `checkpoints_p2b/hymba-with-nsa-randgate_lr1e-3_seed42/hymba-with-nsa-randgate_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0029 | 0.429 | 0.402 | 4088 |
| 1 | 0.0033 | 0.504 | 0.466 | 4088 |
| 2 | 0.0059 | 0.424 | 0.413 | 4088 |
| 3 | 0.0023 | 0.412 | 0.454 | 4088 |
| 4 | 0.0014 | 0.413 | 0.484 | 4088 |
| 5 | 0.0011 | 0.458 | 0.528 | 4088 |
| 6 | 0.0032 | 0.526 | 0.515 | 4088 |
| 7 | 0.0021 | 0.620 | 0.471 | 4088 |
| 8 | 0.0016 | 0.587 | 0.490 | 4088 |
| 9 | 0.0012 | 0.409 | 0.535 | 4088 |
| 10 | 0.0028 | 0.647 | 0.503 | 4088 |
| 11 | 0.0061 | 0.407 | 0.539 | 4088 |

**mean I(g; s) = 0.0028 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.