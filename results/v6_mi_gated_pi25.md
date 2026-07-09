# Gate-as-channel I(g; s) - hymba-with-nsa-gated (2026-06-02)

**Checkpoint:** `checkpoints_v6/hymba-with-nsa-gated_pi25_s42/hymba-with-nsa-gated_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0041 | 0.440 | 0.244 | 4088 |
| 1 | 0.0241 | 0.406 | 0.318 | 4088 |
| 2 | 0.0184 | 0.415 | 0.333 | 4088 |
| 3 | 0.0289 | 0.429 | 0.481 | 4088 |
| 4 | 0.0144 | 0.431 | 0.546 | 4088 |
| 5 | 0.0074 | 0.415 | 0.500 | 4088 |
| 6 | 0.0062 | 0.396 | 0.478 | 4088 |
| 7 | 0.0901 | 0.375 | 0.595 | 4088 |
| 8 | 0.0768 | 0.405 | 0.597 | 4088 |
| 9 | 0.0726 | 0.472 | 0.536 | 4088 |
| 10 | 0.0542 | 0.496 | 0.250 | 4088 |
| 11 | 0.0941 | 0.549 | 0.325 | 4088 |

**mean I(g; s) = 0.0410 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.