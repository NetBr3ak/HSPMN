# HSPMN v5 - Research Roadmap & Breakthrough Synthesis (2026-06-02)

Mode A→C synthesis. Adversarially honest: every claim needs an absolute number
and (where it's a headline) a 3-seed CI. Reviewer lenses: **A** = NeurIPS AC
(novelty), **B** = Tri Dao (systems), **C** = Sasha Rush / Songlin Yang
(expressivity & theory).

Paper readiness estimated **42 %** before this session. Two blockers were
diagnosed and acted on today:

- **λ = 0 (BUG-1) → FIXED.** Robust fp32 gate-gain curvature gives λ = 0.186,
  ρ_LB−1 ≈ 0.133 (non-vacuous). The single most damaging reviewer objection
  (Reviewer C, *CRITICAL*: "you measured λ = 0, the theorem is vacuous") is now
  answerable.
- **Dense 350M missing (BUG-2) → RE-RUNNING.** Clean relaunch to 15000 steps
  (~6.5 h) will fill `paper/HSPMN_v5_final_numbers.tex` with the headline Δ.

## Critical path to a defensible paper (do these first, in order)

| # | action | command / change | measurable | est | unblocks |
|---|---|---|---|---|---|
| 1 | ✅ Fix λ | `measure_lambda_gate_gain` | λ>0, ρ_LB−1 | done | Reviewer C critical |
| 2 | ⏳ Dense 350M | relaunched, `finalize_v5_after_dense.py` after | Δ(hymba,dense)@350M | 6.5 h | headline #1 |
| 3 | Measure all constants on p2b variants | run #5 in REPRODUCE on gated/randgate/pcgate/asa | (λ,σ_C,p,γ) table; ρ_LB ranking | 1 h | Reviewer C, §5.5 |
| 4 | 350M hymba-with-nsa ×{1337,2026} | `train_p4_350m.py --seed …` | 87.38 ± CI | ~13 h | "1 seed" objection |
| 5 | NIAH @ S∈{1024,2048,4096} | `niah_v5_long_context.py` | acc-vs-length | 0.5 h | long-ctx claim |

After the critical path the paper can state, with CIs: head-fusion beats block
fusion (−19.47 %), beats/ties dense at 140M (−3.39 %) and at 350M (Δ TBD from
step 2), and the absorption-resistance bound is non-vacuous and correctly ranks
gated > randgate.

## The breakthrough bet - reframe the theorem so it does not hinge on λ

The deepest weakness flagged by Reviewer C is twofold: λ is (a) hard to measure
and (b) conceptually fragile (a neural-net Hessian's smallest eigenvalue is
~0/negative; "strong convexity" A1 is the weak assumption), and the theorem
reads as *post-hoc* (diagnosis first, theorem fitted after). Fixing the
measurement (done) is necessary but not sufficient. The high-leverage move is to
**replace the curvature dependence with an information-theoretic one**:

> **Gate-as-channel bound (proposed Theorem 1').** Treat the gate
> `g_θ : h_t ↦ [0,1]` as a noisy binary channel deciding "route to contextual
> stream". If its mutual information with the latent "NSA-helps-here" signal is
> `I(g;s) ≥ k` bits, Fano's inequality bounds the routing error, and the loss
> gap obeys `ρ−1 ≥ (1−2·P_err)²·σ_C²·/(2L)` with `P_err ≤ (H(s)−I)/log2`.

Why this is the breakthrough, not a patch:
- It **removes λ** (curvature) from the headline bound - answering C structurally,
  not just numerically. The measured-λ result (0.186) becomes a *corroborating*
  appendix, not a load-bearing constant.
- `I(g;s)` is **directly measurable** by a plug-in histogram estimator on a
  trained checkpoint (no Hessian, no bf16 trap), and is **predictive**: compute
  it on `randgate` (≈0 bits → ρ≈1) vs `gated`/`asa` (>0 bits → ρ>1) and show the
  ranking *before* looking at PPL - this is the answer to the "post-hoc /
  circular" objection (Reviewer C-repro, MEDIUM).
- It connects cleanly to prior art the AC will recognize (Csiszár–Körner error
  exponents; Aquino–Michaels' mutual-information routing result becomes a
  special case), giving Reviewer A a crisp novelty statement: *"a measurable,
  pre-registerable, information-theoretic separation criterion for when a learned
  gate resists absorption."*

Two supporting framings strengthen it:
- **Phase-transition / lottery-ticket view.** Track the effective rank of the
  gate Jacobian `∂g/∂h` over training; conjecture ρ−1 ∝ basin volume ∝ p(1−p)γ².
  Turns the lower bound into a *scaling-law prediction* (predictive, not post-hoc).
  Cheap: re-measure on stored p2b checkpoints.
- **Expressivity witness.** With p∈(0,1) uniformly and a regular-language
  reflexive stream, argue the gated head-fusion recognizes a family beyond the
  TC⁰ class of its ungated dense twin (Merrill–Sabharwal 2023). Probe with
  `synthetic_parity.py` / `synthetic_mqar.py` at long length. Highest payoff for
  Reviewer C, highest proof risk - pursue as a *conjecture + empirical curve*,
  not a claimed theorem, unless the proof closes.

## Ranked breakthrough ideas (payoff × feasibility ÷ risk)

| rank | idea | lens | payoff | risk | why |
|---|---|---|---|---|---|
| 1 | **Gate-as-channel / Fano bound** (Thm 1') | C/theory | breakthrough | low-med | removes the fragile λ; measurable & pre-registerable; kills the CRITICAL objection |
| 2 | **Complete dense 350M Δ** | B/empirical | strong | low | the missing headline; running now |
| 3 | **Constants table + ρ_LB ranking on 5 variants** | C | strong | low | makes the bound *predictive*, answers "circular" |
| 4 | **Phase-transition scaling law** (eff-rank ∝ p(1−p)γ²) | C/theory | strong | med | post-hoc → predictive; cheap on stored ckpts |
| 5 | **Roofline + Pareto (throughput/VRAM/latency) vs dense** | B/systems | strong | med | the systems contribution Hymba/NSA papers lack on Blackwell |
| 6 | **Expressivity-witness probes** (parity/Dyck @ length) | C | strong | med-high | best-for-C if curves separate; proof is hard |
| 7 | **NSA-no-select Triton kernel (sm_120 TMA)** | B/systems | strong | high | 2-3× select-branch; real kernel-eng risk |
| 8 | **FP8/NVFP4 KV cache (Blackwell)** | B/systems | strong | high | 2× VRAM, longer ctx; Triton TMA fp8 API risk |
| 9 | **Delta-Product rank-3 reflexive fusion** | C/arch | incremental | med | MQAR gains; net-new code |
| 10 | **PC-gate temperature ablation / scale-up** | C | incremental | low | tighten the parameter-free-gate story |

## Do-now (this session, no extra GPU-week)

1. ✅ λ fix + measurement. 2. ⏳ dense 350M (running). 3. Draft Theorem 1'
(gate-as-channel) in `paper/` and a measurement stub
(`measure_theorem_constants.py` → add `I(g;s)` histogram estimator). 4. After
dense lands: `finalize_v5_after_dense.py`, rebuild PDF.

## Next GPU-week

3-seed 350M (hymba + dense), constants on all 5 p2b variants, eff-rank scaling
curves, roofline profile. Optional kernel/FP8 bets (#7,#8) only if a systems
contribution is wanted for the B reviewer - both are real engineering, gate on
the roofline showing the select/KV path is actually the bottleneck first.

## Honest risks / kill-criteria still open

- If dense 350M converges **below** 87.38, the positive architecture claim
  weakens to "competitive, not better" - report it; the negative result (block
  fusion retired) + the bound still carry the paper.
- The 350M proxy is **multi-epoch wikitext-103, not FineWeb-Edu 5B** - kill
  criterion #1 (5B-token bench) is *not* met. Frame as "sub-2B comparison with
  wikitext-103 validation"; FineWeb-Edu is future work.
- I(g;s) labels must be defined **independently** of the model's own logits or
  the "circular" objection returns. Use a held-out oracle / next-token-surprise
  proxy fixed before measurement.
