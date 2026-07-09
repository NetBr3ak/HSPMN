# HSPMN - Hybrid State-space / Mixed-attention Networks

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10%2B%2Bcu128-orange)
![Hardware](https://img.shields.io/badge/GPU-RTX_5090_(sm__120)-green)

**Author**: Szymon Jędryczko - Tenzan Logic / Independent Researcher, Kraków

Research code for sub-2B hybrid language models (a linear-recurrent *reflexive*
stream + a sparse-attention *contextual* stream) trained on a single RTX 5090
(Blackwell, sm_120, 32 GB, PyTorch 2.10.0+cu128).
Versions: v3 → v4 → **v5 (this paper)**. The v6 routability programme is
executed and reported *inside* this v5 paper as Contribution 5 / §5, not a
separate manuscript.

**Paper**: *Nothing to Route On: A Measured Information Ceiling Explains
Routing Absorption in Hybrid Mamba-Attention Language Models*
(`paper/HSPMN_v5_draft.tex`, PDF preview `paper/HSPMN_v5_draft.pdf`).

## Headline (v5 final)

1. **Block fusion retired, with a mechanism.** Head fusion beats block fusion
   by **+19.5 % PPL** at matched ingredients and parameters (140M, 3 seeds);
   the block-fusion ReMoE router modally collapses (6/12 layers dead, 2 more
   nearly dead, 2 saturated) while batch-level load balance looks healthy.
2. **The gate null, replicated at two scales.** A learned per-token gate ties
   a frozen random gate at 140M (143.85 ± 9.27 vs 146.24 ± 7.56) and at 350M
   (82.35 ± 4.78 vs 83.79 ± 1.37, 3 seeds each). Gate-less beats both. Head
   fusion's own −3.4 % edge over dense at 140M decays to a tie at 350M.
3. **A gate-channel bound that predicts the null.** Fano + Pinsker give an
   advantage cap `sqrt(2(I + log2 − H(s)))`; full routing benefit needs
   ~0.72 bits. Trained gates carry 0.004–0.006 bits.
4. **The null is structural.** Optimal probes trained directly on the oracle
   label top out at ~0.037 bits (20× below threshold): the routing target is
   nearly information-free. Ablating attention hurts on 256/256 sequences and
   98.9 % of benefit variance is within-sequence. The optimal policy is
   always-on, which is the gate-less baseline. There is nothing to route on.
5. **A pre-registered programme, executed (v6).** Routability phase diagram
   over mixture corpora with recall fraction π ∈ {0, .1, .25, .5}: the
   oracle-label ceiling rises monotonically 0.018 → 0.139 bits (P1 confirmed)
   yet stays 5× below threshold; the learned gate ties random at every π
   exactly as the bound requires (P2). Stream decorrelation fails to raise the
   ceiling (P4 falsified). Bonus: the parameter-free PC gate beats the learned
   gate pooled across the diagram (−2.35 PPL, p = 0.041) and ties gate-less.

> v4 manuscript (separate, arXiv-ready): decoupled-QKV absorption-ratio 1.26× at
> the 31M-equivalent diagnostic scale vs the >2.0× Aquino–Michaels threshold;
> the v3.0 "3.32× speedup" claim is retracted (re-measured 0.95× of a compiled
> cuDNN dense baseline). Reported for context only; the v4 manuscript itself
> is a separate paper and its `HSPMN_v4_0.{tex,pdf}` sources are not part of
> this release.

## Status

| Item | State |
|---|---|
| 140M headline sweep (10 variants × 3 seeds) | ✅ done (`results/phase2_combined_140m_2026-05-12.md`) |
| 350M: hymba / dense / gated / randgate × 3 seeds | ✅ done (`results/phase4_350m_analysis_2026-06-05.md`) |
| Gate-channel theorem + MI + ceiling + sequence probes | ✅ done (paper §5–7) |
| v5 paper | ✅ submission-ready (`paper/HSPMN_v5_draft.tex`) |
| v6 routability programme (137 jobs) | ✅ done (`results/v6_routability_summary.md`) |

## Quick start

```bash
python3 -m pytest test_*.py -q          # 56 tests, ~5 s - green before anything
python3 train_v4.py --variant hymba-with-nsa --lr 1e-3 --seed 42   # a 140M run
```

Full commands → **`REPRODUCE.md`**. Architecture & module map → **`ARCHITECTURE.md`**.
Known pitfalls → **`TROUBLESHOOTING.md`**. Research trail → the dated reports in
`results/`.

## Layout

```
*.py            live code: blocks · LMs · trainers · analysis · diagnostics · evals · tests
paper/          LaTeX source (HSPMN_v5_draft.tex, final) + build scripts + figures
results/        dated logs, tables, theorem drafts, v6 routability artifacts
```

`data/` (tokenized corpora) and `checkpoints_*/` (trained weights) are not
shipped in this release: regenerate with `tokenize_corpus.py` /
`train_v4.py` / `train_p4_350m.py` per `REPRODUCE.md`. Every number in the
paper traces to a static file already present under `results/`.

Critical v5 path: `train_v4.py` → `hymba_with_nsa_lm` → `hymba_with_nsa_block`
→ {`gated_deltanet`, `nsa_attention`, `hspmn_v3_0`}; `utils_v3_0` is shared core.

## Hardware

RTX 5090, sm_120, 32 GB. No FlashAttention-3/4 - attention is SDPA or the Triton
kernel in `kernels_v3_0.py`. Dense 350M trains at ~42k tok/s using 13.9 GB.

## License & citation

See `LICENSE` and `CITATION.cff`.
