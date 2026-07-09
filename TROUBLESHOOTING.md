# Troubleshooting

## BUG-1 - `lambda = 0`, absorption-resistance bound vacuous  ✅ FIXED 2026-06-02

**Symptom.** `measure_theorem_constants.py` reported
`lambda_min = lambda_mean = lambda_max = 0.000000e+00`; all FD directions gave
identical `L+ == L0 == L- == 5.066406`. The bound
`ρ−1 ≥ λ·σ_C²·p(1−p)·γ²/(2L)` collapsed to 0.

**Root cause.** The old estimator perturbed a **random unit-norm direction
across all `attn_gate_proj` weights**. A 1e-2 step therefore diluted to ~1e-4
per weight (`eps/√N`, N≈12·768) - below the **bf16 ULP at loss ≈ 5** - so the
loss did not move and curvature rounded to exactly 0. fp32 loss accumulation
alone did not help, because the per-batch loss is computed from bf16 logits.

**Fix.** `measure_lambda_gate_gain()` measures the curvature of the loss w.r.t.
a **single scalar gate-gain α** (α=1 = trained model; scales every gate logit),
via a 5-point FD in **fp32**. A scalar concentrates the whole 1e-2 step into one
degree of freedom and fp32 removes rounding, so the difference is well resolved.
This is the *gate-action directional curvature* the bound refers to - not the
near-zero/negative global minimum Hessian eigenvalue. The estimator self-tests
on `f(x)=x²` (κ=2.0) before use.

**Result.** λ = **0.186** on `hymba-with-nsa-gated` 140M seed42; ρ_LB−1 ≈
**0.133** (non-vacuous, correctly signed vs the +1.66 % empirical A–M delta).
See `results/phase3_constants_hymba-with-nsa-gated_2026-06-02.md`.

## BUG-2 - dense 350M baseline never finished  ⏳ RELAUNCHED 2026-06-02

**Symptom.** `checkpoints_p4b_dense/` had only a log + csv, no
`dense_p4_final.pt`; `finalize_v5_after_dense.py` crashed with
`FileNotFoundError`; the v5 paper's headline Δ-vs-dense was blank.

**Root cause.** The original run was **killed externally at step 600/15000**
(~16 min). `train_p4_350m.py` only writes `*_final.pt` after the loop completes
(or on the internal `kill_clock` break) - an external SIGTERM hits neither path,
so no final checkpoint. The orchestrator chain logged "Dense done" anyway
(a silent failure), then finalize ran against a missing file.

**Fix.** Relaunched cleanly to full 15000 steps with `--kill_clock_h 10.0` and
`nohup` (survives the session); writes intermediate `dense_p4_state.pt` every
1000 steps (resumable) and `dense_p4_final.pt` on completion. ETA ≈ 6.5 h at
42k tok/s. When it lands, run `finalize_v5_after_dense.py` to populate
`paper/HSPMN_v5_final_numbers.tex`. **Hardening TODO:** add a SIGTERM handler in
`train_p4_350m.py` that saves `*_final.pt` on signal, and make
`orchestrate_v5_chain.sh` verify `*_final.pt` exists before logging "done"
instead of trusting exit status.

## Common issues

- **`get_device_capability() == (12, 0)`** → Blackwell sm_120. FlashAttention-3/4
  are unavailable; attention falls back to SDPA / Triton. Do not assume FA kernels.
- **FLA kernels missing** → reflexive blocks (`gated_deltanet`, `rwkv7_block`,
  `mamba3_block`) fall back to a pure-PyTorch O(S) loop. Correct but slow; fine
  for diagnostics, not for throughput numbers.
- **NaN in NSA `_compress_branch`** → fully-masked rows softmax to NaN; guarded
  by `row_visible`. If you change masking, keep that guard.
- **Checkpoint unpickling** is safe across file moves: checkpoints store only
  `OrderedDict` state_dicts + plain-dict configs (no custom classes pickled).
- **OOM at long S** → `nsa_attention._sliding_window_attention` materializes an
  S×S score matrix (O(S²)); infeasible past ~8k without a chunked kernel.
- **`torch.load` weights_only warning** → checkpoints are trusted local files;
  `weights_only=False` is intentional here.
