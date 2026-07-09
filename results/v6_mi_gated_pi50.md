# Gate-as-channel I(g; s) - hymba-with-nsa-gated (2026-06-02)

**Checkpoint:** `checkpoints_v6/hymba-with-nsa-gated_pi50_s42/hymba-with-nsa-gated_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0049 | 0.415 | 0.217 | 4088 |
| 1 | 0.0079 | 0.383 | 0.316 | 4088 |
| 2 | 0.0135 | 0.367 | 0.399 | 4088 |
| 3 | 0.0370 | 0.454 | 0.541 | 4088 |
| 4 | 0.0346 | 0.459 | 0.365 | 4088 |
| 5 | 0.0533 | 0.441 | 0.346 | 4088 |
| 6 | 0.0467 | 0.378 | 0.146 | 4088 |
| 7 | 0.0132 | 0.251 | 0.413 | 4088 |
| 8 | 0.0131 | 0.433 | 0.448 | 4088 |
| 9 | 0.0786 | 0.305 | 0.553 | 4088 |
| 10 | 0.0147 | 0.359 | 0.532 | 4088 |
| 11 | 0.0793 | 0.416 | 0.651 | 4088 |

**mean I(g; s) = 0.0331 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.