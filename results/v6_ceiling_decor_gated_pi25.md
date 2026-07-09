# Gate-ceiling probe - hymba-with-nsa-decor-gated

Checkpoint `checkpoints_v6/hymba-with-nsa-decor-gated_pi25_s42/hymba-with-nsa-decor-gated_final.pt`. Held-out logistic probe h_t -> s_t (s = NSA-ablation raises CE).

| layer | p(s=1) | H(s) | val_acc | I(pred;s) bits |
|---|---|---|---|---|
| 0 | 0.436 | 0.988 | 0.585 | 0.0432 |
| 1 | 0.444 | 0.991 | 0.615 | 0.1239 |
| 2 | 0.375 | 0.954 | 0.622 | 0.0550 |
| 3 | 0.402 | 0.972 | 0.635 | 0.1016 |
| 4 | 0.427 | 0.985 | 0.600 | 0.0455 |
| 5 | 0.395 | 0.968 | 0.639 | 0.0668 |
| 6 | 0.367 | 0.948 | 0.652 | 0.0571 |
| 7 | 0.428 | 0.985 | 0.621 | 0.0970 |
| 8 | 0.449 | 0.993 | 0.637 | 0.1221 |
| 9 | 0.550 | 0.993 | 0.700 | 0.2113 |
| 10 | 0.273 | 0.845 | 0.743 | 0.1120 |
| 11 | 0.467 | 0.997 | 0.642 | 0.0950 |

**mean held-out I(pred;s) = 0.0942 bits** vs full-benefit threshold = 0.702 bits (Pinsker form).

**VERDICT:** CEILING BELOW THRESHOLD -> the NSA-helps signal is not linearly decodable from the gate input; NO gate of this form can cross the threshold. The theorem's null is structural, not a training failure.