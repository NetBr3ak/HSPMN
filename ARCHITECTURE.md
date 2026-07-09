# HSPMN - Architecture & Module Map

Hybrid State-space / Predictive-coding / Mixed-attention Networks. Research code
for sub-2B hybrid language models on a single RTX 5090 (Blackwell, sm_120, 32 GB,
PyTorch 2.10.0+cu128; **no FlashAttention-3/4** on this arch - all attention is
SDPA or hand-written Triton/PyTorch).

## v5 architecture in one paragraph

A v5 block runs two streams **in parallel at the head level** (Hymba-style head
fusion), then concatenates: a **reflexive** linear-recurrent stream (gated
DeltaNet by default) and a **contextual** sparse-attention stream (NSA without
the learned-selection branch - compression + sliding window only). An optional
per-token gate modulates the contextual stream. This *head fusion* replaced the
v4 *block fusion* (two full sub-layers merged by a ReMoE router), which was
empirically retired (see below).

```
                 ┌─────────────── HymbaWithNSABlock ───────────────┐
 x ──► RMSNorm ──┤  reflexive heads:  GatedDeltaNet (linear RNN)    │
                 │  contextual heads: NSA(compress + window)  ×gate │──► concat ──► o_proj ──► +x
                 └──────────────────────────────────────────────────┘
                                    │
                                 SwiGLU MLP ──► +
```

## Critical path (v5 - what actually trains the winner)

```
train_v4.py            universal trainer; build_model() dispatches by variant string
  └─ build_model ─────► hymba_with_nsa_lm.py        LM head + embedding + N blocks
                          └─ hymba_with_nsa_block.py  the v5 block (head fusion)
                               ├─ gated_deltanet.py    reflexive stream  (ICLR'25, 2412.06464)
                               ├─ nsa_attention.py      contextual stream (DeepSeek, 2502.11089)
                               └─ hspmn_v3_0.py         shared RMSNorm / RoPE / SwiGLU / sinks
utils_v3_0.py           universal: logging, seeding, device, TrainConfig helpers
```

`train_p4_350m.py` is a thin 350M-scale wrapper around `train_v4.py`'s
`build_model / get_batch / run_validation / cosine_lr / TrainConfig`.

## Module dependency graph (in-degree = #local importers)

| in-deg | module | role | v5 status |
|---|---|---|---|
| 28 | `utils_v3_0` | logging, seed, device, config | **shared core** |
| 10 | `hspmn_v3_0` | RMSNorm, RoPE, SwiGLU, sinks, TopKRouter | **shared core** |
| 9  | `train_v4` | universal trainer + `build_model` dispatch | **core** |
| 7  | `hspmn_v4_0` | v4 block-fusion model | baseline (retired arch) |
| 4  | `gated_deltanet` | reflexive linear-recurrent stream | **core (v5)** |
| 3  | `nsa_attention` | NSA / ASA contextual stream | **core (v5)** |
| 3  | `router_v4_0` | ReMoE router (ALF-LB, z-loss) | baseline (v4 only) |
| 3  | `kernels_v3_0` | Triton SQSK sparse attention | v3/v4 path |
| 3  | `hymba_with_nsa_lm` | v5 LM wrapper | **core (v5)** |
| 3  | `hspmn_v4_lm` | v4 LM wrapper | baseline |
| 2  | `rwkv7_block`, `mamba3_block` | alt reflexive primitives | bench / ablation |
| 2  | `hymba_with_nsa_block`, `hymba_lm` | v5 block, Hymba LM | **core (v5)** / baseline |

## Variants `train_v4.py --variant` accepts

`dense`, `hymba`, **`hymba-with-nsa`** (P2 winner), `hymba-with-nsa-gated`
(learned sigmoid gate), `hymba-with-nsa-randgate` (Aquino–Michaels random-gate
control), `hymba-with-nsa-pcgate` (parameter-free predictive-coding gate),
`hymba-with-nsa-select` (parameter-free top-K select from compress scores),
`hymba-with-asa` (alternating window/select per layer), and `v4-*` block-fusion
families (`v4-gdn-nsa`, `v4-rwkv7-nsa`, …) kept as comparison baselines.

## Reflexive primitives (interchangeable)

- `gated_deltanet.py` - Gated DeltaNet, chunkwise-parallel, FLA kernel + pure
  PyTorch fallback. **Default reflexive stream.**
- `rwkv7_block.py` - RWKV-7 generalized delta (rank-2 state, in-context lr).
- `mamba3_block.py` - Mamba-3 SISO, complex-valued 2×2 block-diagonal state.

## Contextual primitive

- `nsa_attention.py` - Native Sparse Attention: compress + select + window
  branches with per-token sigmoid gates. v5 uses **no-select** (compress+window).
  ASA mode alternates window/select per layer. `kernels_v3_0.py` holds a Triton
  sparse-query/sparse-key kernel used by the v3/v4 path.

## Why block fusion was retired (the v5 pivot)

At 140M on wikitext-103 (45 runs: 5 variants × 3 LR × 3 seeds, 3000 steps):

| variant | best PPL ± std (3 seeds) | Δ vs dense |
|---|---|---|
| dense | 144.01 ± 5.64 | - |
| hymba | 143.42 ± 9.98 | −0.41 % |
| **hymba-with-nsa** | **139.14 ± 8.43** | **−3.39 %** |
| v4-gdn-nsa (block fusion) | 166.23 ± 15.12 | +15.43 % |
| v4-rwkv7-nsa (block fusion) | 169.55 ± 14.03 | +17.73 % |

Same-ingredient head→block fusion swap: **+19.47 % PPL**, larger than any
LR/seed variance. Cause: the v4 ReMoE router **modally collapsed** - 7 of 12
layers had fully dead gates, 2 saturated; ALF-LB held the *batch-averaged*
active fraction near target while per-layer routing was destroyed. The v5
absorption-resistance bound predicts exactly this: `ρ−1 ∝ p(1−p)γ²` → 0 when
per-layer `p ∈ {0,1}`. See `results/router_diagnostics_summary_2026-05-14.md`.

## Hardware / sm_120 notes

- Blackwell sm_120, 32 GB. Dense 350M trains at ~42–43k tok/s, 13.9 GB VRAM at
  S=1024/B=8/ga=8 - large headroom for bigger batch or longer context.
- No FlashAttention-3/4. Attention is `F.scaled_dot_product_attention` or the
  Triton kernel in `kernels_v3_0.py`. FP8/NVFP4 TMA paths are **not yet wired**
  (future work).

## Repository layout

```
*.py (root)        live code - blocks, LMs, trainers, analysis, diagnostics, evals, tests
paper/             LaTeX source (HSPMN_v5_draft.tex, final) + build scripts + figures
results/           dated experiment logs, tables, theorem drafts, v6 routability artifacts
```

`data/` (tokenized wikitext-103, tokenizer, hellaswag_val.jsonl, mix_pi* corpora)
and `checkpoints_*/` (trained weights per variant/seed) are regenerated
locally, not shipped in this release - see `REPRODUCE.md`.

See `REPRODUCE.md` for commands, `TROUBLESHOOTING.md` for known bugs.
