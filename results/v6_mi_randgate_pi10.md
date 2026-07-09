# Gate-as-channel I(g; s) - hymba-with-nsa-randgate (2026-06-02)

**Checkpoint:** `checkpoints_v6/hymba-with-nsa-randgate_pi10_s42/hymba-with-nsa-randgate_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0029 | 0.432 | 0.405 | 4088 |
| 1 | 0.0036 | 0.428 | 0.392 | 4088 |
| 2 | 0.0033 | 0.451 | 0.420 | 4088 |
| 3 | 0.0012 | 0.448 | 0.426 | 4088 |
| 4 | 0.0013 | 0.474 | 0.431 | 4088 |
| 5 | 0.0020 | 0.449 | 0.475 | 4088 |
| 6 | 0.0027 | 0.404 | 0.479 | 4088 |
| 7 | 0.0026 | 0.414 | 0.481 | 4088 |
| 8 | 0.0067 | 0.429 | 0.509 | 4088 |
| 9 | 0.0020 | 0.464 | 0.492 | 4088 |
| 10 | 0.0019 | 0.640 | 0.516 | 4088 |
| 11 | 0.0017 | 0.656 | 0.525 | 4088 |

**mean I(g; s) = 0.0026 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.