# Gate-as-channel I(g; s) - hymba-with-nsa-gated (2026-06-02)

**Checkpoint:** `checkpoints_p4_gated_s42/hymba-with-nsa-gated_p4_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0082 | 0.384 | 0.425 | 4088 |
| 1 | 0.0004 | 0.382 | 0.072 | 4088 |
| 2 | 0.0041 | 0.448 | 0.339 | 4088 |
| 3 | 0.0011 | 0.414 | 0.568 | 4088 |
| 4 | 0.0036 | 0.472 | 0.634 | 4088 |
| 5 | 0.0040 | 0.430 | 0.488 | 4088 |
| 6 | 0.0030 | 0.420 | 0.541 | 4088 |
| 7 | 0.0031 | 0.413 | 0.454 | 4088 |
| 8 | 0.0023 | 0.409 | 0.538 | 4088 |
| 9 | 0.0026 | 0.435 | 0.451 | 4088 |
| 10 | 0.0030 | 0.423 | 0.373 | 4088 |
| 11 | 0.0023 | 0.420 | 0.418 | 4088 |
| 12 | 0.0020 | 0.409 | 0.462 | 4088 |
| 13 | 0.0042 | 0.384 | 0.513 | 4088 |
| 14 | 0.0025 | 0.344 | 0.534 | 4088 |
| 15 | 0.0043 | 0.387 | 0.600 | 4088 |
| 16 | 0.0032 | 0.431 | 0.465 | 4088 |
| 17 | 0.0025 | 0.432 | 0.558 | 4088 |
| 18 | 0.0034 | 0.512 | 0.489 | 4088 |
| 19 | 0.0029 | 0.353 | 0.367 | 4088 |
| 20 | 0.0041 | 0.605 | 0.412 | 4088 |
| 21 | 0.0159 | 0.364 | 0.515 | 4088 |
| 22 | 0.0010 | 0.207 | 0.032 | 4088 |
| 23 | 0.0044 | 0.437 | 0.266 | 4088 |

**mean I(g; s) = 0.0037 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.