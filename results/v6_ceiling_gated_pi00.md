# Gate-ceiling probe - hymba-with-nsa-gated

Checkpoint `checkpoints_v6/hymba-with-nsa-gated_pi00_s42/hymba-with-nsa-gated_final.pt`. Held-out logistic probe h_t -> s_t (s = NSA-ablation raises CE).

| layer | p(s=1) | H(s) | val_acc | I(pred;s) bits |
|---|---|---|---|---|
| 0 | 0.453 | 0.993 | 0.552 | 0.0118 |
| 1 | 0.414 | 0.978 | 0.564 | 0.0117 |
| 2 | 0.428 | 0.985 | 0.550 | 0.0122 |
| 3 | 0.415 | 0.979 | 0.557 | 0.0119 |
| 4 | 0.441 | 0.990 | 0.540 | 0.0072 |
| 5 | 0.425 | 0.984 | 0.565 | 0.0123 |
| 6 | 0.494 | 1.000 | 0.553 | 0.0166 |
| 7 | 0.477 | 0.999 | 0.550 | 0.0205 |
| 8 | 0.406 | 0.974 | 0.593 | 0.0265 |
| 9 | 0.473 | 0.998 | 0.546 | 0.0172 |
| 10 | 0.653 | 0.931 | 0.651 | 0.0368 |
| 11 | 0.576 | 0.983 | 0.585 | 0.0272 |

**mean held-out I(pred;s) = 0.0177 bits** vs full-benefit threshold = 0.719 bits (Pinsker form).

**VERDICT:** CEILING BELOW THRESHOLD -> the NSA-helps signal is not linearly decodable from the gate input; NO gate of this form can cross the threshold. The theorem's null is structural, not a training failure.