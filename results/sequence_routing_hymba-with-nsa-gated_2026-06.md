# Sequence-level routing probe - hymba-with-nsa-gated

Checkpoint `checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/hymba-with-nsa-gated_final.pt`, 256 sequences x 512 tokens.

- Δ_seq (total CE saved by NSA): mean 405.426, std 111.427, range [107.34, 830.23].
- Fraction of sequences NSA net-helps: 1.000.
- Per-token benefit variance: between-seq 4.755e-02 / within-seq 4.430e+00 -> **between-sequence fraction 0.011**.
- Predict sign(Δ_seq) from sequence-mean embedding (held-out): acc 1.000, I(pred;sign) 0.0000 bits vs threshold -0.500.

**VERDICT:** WEAK sequence-level signal too -> on homogeneous text the contextual benefit is neither concentrated in particular sequences nor predictable; the lever is task/data diversity (retrieval-heavy corpora), not routing granularity. Even sequence routing is information-starved here.
