# Gate-as-channel I(g; s) - hymba-with-nsa-gated (2026-06-02)

**Checkpoint:** `checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/hymba-with-nsa-gated_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0035 | 0.429 | 0.285 | 4088 |
| 1 | 0.0049 | 0.504 | 0.283 | 4088 |
| 2 | 0.0038 | 0.424 | 0.515 | 4088 |
| 3 | 0.0069 | 0.412 | 0.566 | 4088 |
| 4 | 0.0056 | 0.413 | 0.559 | 4088 |
| 5 | 0.0063 | 0.458 | 0.466 | 4088 |
| 6 | 0.0028 | 0.526 | 0.651 | 4088 |
| 7 | 0.0054 | 0.620 | 0.526 | 4088 |
| 8 | 0.0036 | 0.587 | 0.482 | 4088 |
| 9 | 0.0025 | 0.409 | 0.403 | 4088 |
| 10 | 0.0089 | 0.647 | 0.599 | 4088 |
| 11 | 0.0116 | 0.407 | 0.173 | 4088 |

**mean I(g; s) = 0.0055 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.