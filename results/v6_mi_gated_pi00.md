# Gate-as-channel I(g; s) - hymba-with-nsa-gated (2026-06-02)

**Checkpoint:** `checkpoints_v6/hymba-with-nsa-gated_pi00_s42/hymba-with-nsa-gated_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0018 | 0.425 | 0.264 | 4088 |
| 1 | 0.0014 | 0.453 | 0.362 | 4088 |
| 2 | 0.0023 | 0.422 | 0.493 | 4088 |
| 3 | 0.0023 | 0.428 | 0.453 | 4088 |
| 4 | 0.0030 | 0.421 | 0.542 | 4088 |
| 5 | 0.0023 | 0.421 | 0.436 | 4088 |
| 6 | 0.0089 | 0.400 | 0.703 | 4088 |
| 7 | 0.0019 | 0.453 | 0.398 | 4088 |
| 8 | 0.0052 | 0.508 | 0.442 | 4088 |
| 9 | 0.0044 | 0.510 | 0.507 | 4088 |
| 10 | 0.0035 | 0.663 | 0.593 | 4088 |
| 11 | 0.0035 | 0.659 | 0.478 | 4088 |

**mean I(g; s) = 0.0034 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.