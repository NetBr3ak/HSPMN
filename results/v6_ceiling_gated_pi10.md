# Gate-ceiling probe - hymba-with-nsa-gated

Checkpoint `checkpoints_v6/hymba-with-nsa-gated_pi10_s42/hymba-with-nsa-gated_final.pt`. Held-out logistic probe h_t -> s_t (s = NSA-ablation raises CE).

| layer | p(s=1) | H(s) | val_acc | I(pred;s) bits |
|---|---|---|---|---|
| 0 | 0.413 | 0.978 | 0.607 | 0.0357 |
| 1 | 0.429 | 0.986 | 0.583 | 0.0369 |
| 2 | 0.416 | 0.979 | 0.587 | 0.0388 |
| 3 | 0.416 | 0.980 | 0.579 | 0.0324 |
| 4 | 0.416 | 0.979 | 0.573 | 0.0365 |
| 5 | 0.438 | 0.989 | 0.575 | 0.0230 |
| 6 | 0.421 | 0.982 | 0.587 | 0.0494 |
| 7 | 0.458 | 0.995 | 0.591 | 0.0616 |
| 8 | 0.494 | 1.000 | 0.608 | 0.0913 |
| 9 | 0.424 | 0.983 | 0.620 | 0.0816 |
| 10 | 0.449 | 0.992 | 0.601 | 0.0640 |
| 11 | 0.516 | 0.999 | 0.668 | 0.1466 |

**mean held-out I(pred;s) = 0.0582 bits** vs full-benefit threshold = 0.711 bits (Pinsker form).

**VERDICT:** CEILING BELOW THRESHOLD -> the NSA-helps signal is not linearly decodable from the gate input; NO gate of this form can cross the threshold. The theorem's null is structural, not a training failure.