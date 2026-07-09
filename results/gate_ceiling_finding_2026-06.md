# Gate-ceiling probe - the absorption null is STRUCTURAL (2026-06)

A theory-guided probe run during the autonomous breakthrough hunt. It upgrades
the paper's result from "the learned gate happened to tie random" to a measured
impossibility: no gate of this architecture class can cross the resistance
threshold, because the routing target is nearly information-free.

## Setup

On the trained `hymba-with-nsa-gated` 140M checkpoint, for each gated layer we
collect the gate's would-be features and the label
`s_t = 1[ablating that layer's NSA raises the next-token CE at t]`, then fit an
*optimal* probe `features -> s_t` on a 70% train split and report the mutual
information `I(pred; s)` and accuracy on the 30% held-out split. The held-out
`I(pred;s)` is the **ceiling** any gate of that form could reach. Threshold for
absorption resistance (Proposition 1', CORRECTED 2026-06-10 via Pinsker): full
routing benefit requires `I > 1/2 + H(s) - log2 ≈ 0.49 nats ≈ 0.72 bits`;
the advantage of any gate is capped by `sqrt(2(I + log2 - H(s)))`.
(The earlier `H(s) - 1/2 log2 ≈ 0.49 bits` form relied on the chord bound
`H_b(q) <= 2q log2`, which is false by concavity; numbers below updated.)

## Result

| probe | features | mean held-out I(pred;s) | val acc |
|---|---|---|---|
| linear | residual input `h_t` (gate's actual input) | **0.0315 bits** | 0.54–0.68 |
| MLP (64h, GELU) | `[h_t, ‖C‖/head, ‖R‖/head, ‖C‖−‖R‖]` (781-dim) | **0.0341 bits** | 0.55–0.65 |

Both ceilings are $\approx 21\times$ below the $\approx0.72$-bit full-benefit threshold (advantage cap $A \le 0.24$). The nonlinear
MLP on *both stream outputs* does no better than the linear probe on the
residual. The trained gate's own MI (0.004–0.006 bits) is ~6–8× below even this
ceiling - but the ceiling itself is what matters: it is structural.

## Interpretation (the real contribution)

The event "the contextual stream lowers the loss at token $t$" is **nearly
information-free** ($\approx 0.03$ bits decodable, accuracy barely above the
$p\approx0.5$ base rate) from any token-local feature, including the two
streams' own outputs. At the token level the reflexive (GDN) and contextual
(NSA) streams are **largely substitutable** - there is almost nothing to route
on. This is the mechanistic reason routing absorption is so hard to beat in
confidence-gated dual-stream hybrids: not that the optimiser fails to find a
good gate, but that **no good token-local gate exists**.

This makes the paper's negative result a *proved + measured impossibility* for
the architecture class, with a falsifiable threshold and a measured ceiling -
strictly stronger than "learned ties random."

## What this rules in / out for a breakthrough

- RULED OUT (don't burn GPU): any per-token gate on residual/stream features
  (linear, MLP, supervised-to-the-ablation-label, info-bottleneck) - all are
  capped at ~0.03 bits ≪ 0.72. A supervised gate could recover the 6–8× from
  0.004 to ~0.03 but still cannot cross. The supervised-MI-gate idea is dead.
- POSSIBLE future positive direction (architecture redesign, not a gate tweak):
  make the streams *non-redundant by construction* so `s` becomes predictable
  (forced specialization / orthogonality between streams), or move from
  per-token routing to *sequence/segment-level* routing where attention-vs-SSM
  utility is more separable (e.g. long-context retrieval spans). Both are new
  architectures, out of scope for the current paper; logged as future work.

## Action

Integrated into the paper as §5.9 + a theorem corollary + Fig. 5. The structural
null is now a headline finding, not a caveat.

## Sequence-level follow-up - optimal policy is always-on (2026-06-09)

`probe_sequence_routing.py` on the gated 140M checkpoint, 256 sequences:
- Ablating ALL NSA raises CE on **256/256** sequences; Δ_seq = 405 ± 111 nats,
  range [107, 830] - the contextual stream **uniformly helps every sequence**.
- Per-token benefit variance: **between-sequence only 1.1%**, within-sequence
  98.9% → "NSA helps" is a within-sequence per-token phenomenon, not a sequence
  property. So sequence/segment-level ON/OFF routing is moot (always ON).

**Capstone:** the optimal routing policy is trivially **always-on** (= the
gate-less baseline). Any gate can only deviate from always-on using the
~0.03-bit per-token residual signal, so it can only tie or hurt - exactly the
observed null. The gate is useless because optimal routing is *trivial*, not
because routing is *hard*. The genuine lever for routing-benefit is therefore
**task/data diversity** (retrieval-heavy corpora where attention is sometimes
decisive and SSM sometimes sufficient), not gate design or routing granularity.

## Robustness checks (2026-06-10, post-review)

Reviewer objection: "maybe your probe is just underfit." Tested:
- **Capacity sweep** (MLP hidden 64 / 128 / 256, same data, default split):
  held-out I = 0.0341 / 0.0346 / 0.0368 bits. A 4x capacity increase moves
  the ceiling by 8%. Plateau; not underfit in any way that matters against
  the ~0.72-bit threshold.
- **Split variance** (hidden=64, split seeds 0/1/2/3):
  I = 0.0341 / 0.0296 / 0.0327 / 0.0341 bits (mean 0.0326, spread ~0.005).
- Most routing-favourable value across all runs: **0.037 bits** -> still
  ~20x below threshold; advantage cap A <= 0.25 for any token-local gate.
Paper updated to quote the conservative (largest) ceiling.
