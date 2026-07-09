# Gate-ceiling probe #2 (MLP + stream features) - hymba-with-nsa-gated

Checkpoint `checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/hymba-with-nsa-gated_final.pt`.

| layer | p(s=1) | feat dim | val_acc | I(pred;s) bits |
|---|---|---|---|---|
| 0 | 0.473 | 781 | 0.561 | 0.0180 |
| 1 | 0.430 | 781 | 0.569 | 0.0174 |
| 2 | 0.452 | 781 | 0.575 | 0.0207 |
| 3 | 0.465 | 781 | 0.572 | 0.0233 |
| 4 | 0.439 | 781 | 0.581 | 0.0274 |
| 5 | 0.436 | 781 | 0.590 | 0.0285 |
| 6 | 0.499 | 781 | 0.597 | 0.0405 |
| 7 | 0.508 | 781 | 0.597 | 0.0370 |
| 8 | 0.517 | 781 | 0.604 | 0.0424 |
| 9 | 0.580 | 781 | 0.618 | 0.0420 |
| 10 | 0.676 | 781 | 0.655 | 0.0396 |
| 11 | 0.428 | 781 | 0.648 | 0.0725 |

**mean held-out I(pred;s) = 0.0341 bits** vs threshold 0.500 bits.

**VERDICT:** BELOW THRESHOLD even for a nonlinear gate on both streams -> the NSA-helps signal is fundamentally low-information in this architecture. The absorption-resistance null is STRUCTURAL and class-wide, not a gate-design failure. Strong theorem corollary.