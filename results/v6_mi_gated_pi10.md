# Gate-as-channel I(g; s) - hymba-with-nsa-gated (2026-06-02)

**Checkpoint:** `checkpoints_v6/hymba-with-nsa-gated_pi10_s42/hymba-with-nsa-gated_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0056 | 0.432 | 0.205 | 4088 |
| 1 | 0.0025 | 0.428 | 0.334 | 4088 |
| 2 | 0.0030 | 0.451 | 0.415 | 4088 |
| 3 | 0.0029 | 0.448 | 0.388 | 4088 |
| 4 | 0.0029 | 0.474 | 0.507 | 4088 |
| 5 | 0.0025 | 0.449 | 0.473 | 4088 |
| 6 | 0.0039 | 0.404 | 0.489 | 4088 |
| 7 | 0.0037 | 0.414 | 0.465 | 4088 |
| 8 | 0.0065 | 0.429 | 0.572 | 4088 |
| 9 | 0.0029 | 0.464 | 0.411 | 4088 |
| 10 | 0.0038 | 0.640 | 0.323 | 4088 |
| 11 | 0.0111 | 0.656 | 0.564 | 4088 |

**mean I(g; s) = 0.0043 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.