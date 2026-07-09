# Gate-as-channel I(g; s) - hymba-with-nsa-randgate (2026-06-02)

**Checkpoint:** `checkpoints_p4_randgate_s42/hymba-with-nsa-randgate_p4_final.pt`.
Counterfactual label: per-layer NSA ablation raises next-token CE by > 0.0 (independent of gate params).
MI estimator self-test: independent=0.0006, det=0.9999 bits.

| Layer | I(g;s) bits | p(NSA helps) | gate.mean | n |
|---|---|---|---|---|
| 0 | 0.0089 | 0.384 | 0.395 | 4088 |
| 1 | 0.0006 | 0.382 | 0.367 | 4088 |
| 2 | 0.0013 | 0.448 | 0.434 | 4088 |
| 3 | 0.0010 | 0.414 | 0.540 | 4088 |
| 4 | 0.0014 | 0.472 | 0.482 | 4088 |
| 5 | 0.0030 | 0.430 | 0.399 | 4088 |
| 6 | 0.0027 | 0.420 | 0.498 | 4088 |
| 7 | 0.0016 | 0.413 | 0.444 | 4088 |
| 8 | 0.0021 | 0.409 | 0.449 | 4088 |
| 9 | 0.0027 | 0.435 | 0.477 | 4088 |
| 10 | 0.0015 | 0.423 | 0.560 | 4088 |
| 11 | 0.0021 | 0.420 | 0.514 | 4088 |
| 12 | 0.0011 | 0.409 | 0.486 | 4088 |
| 13 | 0.0019 | 0.384 | 0.489 | 4088 |
| 14 | 0.0016 | 0.344 | 0.496 | 4088 |
| 15 | 0.0014 | 0.387 | 0.512 | 4088 |
| 16 | 0.0017 | 0.431 | 0.519 | 4088 |
| 17 | 0.0034 | 0.432 | 0.553 | 4088 |
| 18 | 0.0019 | 0.512 | 0.588 | 4088 |
| 19 | 0.0018 | 0.353 | 0.598 | 4088 |
| 20 | 0.0007 | 0.605 | 0.514 | 4088 |
| 21 | 0.0051 | 0.364 | 0.496 | 4088 |
| 22 | 0.0011 | 0.207 | 0.519 | 4088 |
| 23 | 0.0008 | 0.437 | 0.519 | 4088 |

**mean I(g; s) = 0.0021 bits**

Interpretation: a learned gate routes where the contextual stream helps, so I(g;s) > 0; the random-gate control should give I ~ 0. This is a non-circular, predictive separation criterion that does not depend on the loss curvature lambda.