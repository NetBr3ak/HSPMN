# Gate-as-channel I(g; s) - hymba-with-nsa-randgate (2026-06-02)

**Checkpoint:** `checkpoints_v6/hymba-with-nsa-randgate_pi25_s42/hymba-with-nsa-randgate_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0022 | 0.440 | 0.424 | 4088 |
| 1 | 0.0079 | 0.406 | 0.431 | 4088 |
| 2 | 0.0081 | 0.415 | 0.433 | 4088 |
| 3 | 0.0034 | 0.429 | 0.452 | 4088 |
| 4 | 0.0072 | 0.431 | 0.472 | 4088 |
| 5 | 0.0024 | 0.415 | 0.505 | 4088 |
| 6 | 0.0186 | 0.396 | 0.498 | 4088 |
| 7 | 0.0064 | 0.375 | 0.488 | 4088 |
| 8 | 0.0319 | 0.405 | 0.533 | 4088 |
| 9 | 0.0310 | 0.472 | 0.526 | 4088 |
| 10 | 0.0502 | 0.496 | 0.507 | 4088 |
| 11 | 0.0129 | 0.549 | 0.614 | 4088 |

**mean I(g; s) = 0.0152 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.