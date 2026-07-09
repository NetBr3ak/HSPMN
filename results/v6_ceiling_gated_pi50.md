# Gate-ceiling probe - hymba-with-nsa-gated

Checkpoint `checkpoints_v6/hymba-with-nsa-gated_pi50_s42/hymba-with-nsa-gated_final.pt`. Held-out logistic probe h_t -> s_t (s = NSA-ablation raises CE).

| layer | p(s=1) | H(s) | val_acc | I(pred;s) bits |
|---|---|---|---|---|
| 0 | 0.360 | 0.943 | 0.643 | 0.0606 |
| 1 | 0.359 | 0.942 | 0.635 | 0.0825 |
| 2 | 0.327 | 0.912 | 0.722 | 0.1522 |
| 3 | 0.332 | 0.917 | 0.748 | 0.2473 |
| 4 | 0.316 | 0.900 | 0.691 | 0.0776 |
| 5 | 0.298 | 0.879 | 0.682 | 0.0675 |
| 6 | 0.238 | 0.792 | 0.778 | 0.1167 |
| 7 | 0.276 | 0.850 | 0.744 | 0.1462 |
| 8 | 0.348 | 0.932 | 0.681 | 0.1555 |
| 9 | 0.247 | 0.806 | 0.769 | 0.1609 |
| 10 | 0.289 | 0.867 | 0.725 | 0.1487 |
| 11 | 0.381 | 0.959 | 0.754 | 0.2551 |

**mean held-out I(pred;s) = 0.1392 bits** vs full-benefit threshold = 0.620 bits (Pinsker form).

**VERDICT:** CEILING BELOW THRESHOLD -> the NSA-helps signal is not linearly decodable from the gate input; NO gate of this form can cross the threshold. The theorem's null is structural, not a training failure.