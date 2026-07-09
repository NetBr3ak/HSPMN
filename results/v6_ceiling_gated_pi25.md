# Gate-ceiling probe - hymba-with-nsa-gated

Checkpoint `checkpoints_v6/hymba-with-nsa-gated_pi25_s42/hymba-with-nsa-gated_final.pt`. Held-out logistic probe h_t -> s_t (s = NSA-ablation raises CE).

| layer | p(s=1) | H(s) | val_acc | I(pred;s) bits |
|---|---|---|---|---|
| 0 | 0.406 | 0.974 | 0.609 | 0.0472 |
| 1 | 0.366 | 0.947 | 0.634 | 0.0785 |
| 2 | 0.385 | 0.962 | 0.630 | 0.0952 |
| 3 | 0.418 | 0.980 | 0.587 | 0.0437 |
| 4 | 0.371 | 0.951 | 0.624 | 0.0732 |
| 5 | 0.419 | 0.981 | 0.637 | 0.0907 |
| 6 | 0.378 | 0.957 | 0.646 | 0.0925 |
| 7 | 0.432 | 0.987 | 0.629 | 0.1302 |
| 8 | 0.480 | 0.999 | 0.663 | 0.1741 |
| 9 | 0.551 | 0.992 | 0.697 | 0.2037 |
| 10 | 0.388 | 0.964 | 0.659 | 0.1226 |
| 11 | 0.441 | 0.990 | 0.628 | 0.1401 |

**mean held-out I(pred;s) = 0.1076 bits** vs full-benefit threshold = 0.703 bits (Pinsker form).

**VERDICT:** CEILING BELOW THRESHOLD -> the NSA-helps signal is not linearly decodable from the gate input; NO gate of this form can cross the threshold. The theorem's null is structural, not a training failure.