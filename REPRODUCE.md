# Reproducing HSPMN v5

All commands run from the repo root on a single RTX 5090 (sm_120), PyTorch
2.10.0+cu128. Pinned deps in `requirements.txt`. Data lives in `data/`
(`train_tokens.npy` = wikitext-103, 119.1M tokens; `valid_tokens.npy` = 250k).

## 0. Environment + sanity

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# expect: 2.10.0+cu128 (12, 0)
python3 -m pytest test_*.py -q          # 61 tests, ~5s, must be green before any run
```

## 1. Data prep (only if data/*.npy missing)

```bash
python3 tokenize_corpus.py              # wikitext-103 → data/train_tokens.npy, valid_tokens.npy
# FineWeb-Edu (optional, large; download was bandwidth-limited historically):
bash download_fineweb_edu.sh && python3 tokenize_fineweb_edu.py
```

## 2. Synthetic expressivity battery (≈30 min)

```bash
python3 run_p2_synthetics.py            # MQAR / Dyck-2 / parity across variants
python3 aggregate_p2_synthetics.py      # → results/phase2_synthetics_table_*.md
```

## 3. 140M headline sweep (the −3.39 % result; ≈30 GPU-h for the full 45 runs)

Variants × LR × seed at S=1024, B=4, ga=8, 3000 steps. Single run:

```bash
python3 train_v4.py --variant hymba-with-nsa --lr 1e-3 --seed 42 \
  --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4 \
  --seq_len 1024 --batch_size 4 --grad_accum 8 --steps 3000 \
  --save_dir checkpoints_p2screen/hymba-with-nsa_lr1e-3_seed42
python3 aggregate_p2screen.py           # → 140M table (dense/hymba/hymba-with-nsa/v4-*)
```

## 4. 350M proxy runs

**hymba-with-nsa (winner, already trained → checkpoints_p4_350m/, PPL 87.38):**

```bash
python3 train_p4_350m.py --variant hymba-with-nsa --seed 42 \
  --n_layers 24 --dim 896 --num_heads 14 --num_kv_heads 2 \
  --seq_len 1024 --batch_size 8 --grad_accum 8 --steps 15000 --warmup_steps 500 \
  --lr 1e-3 --save_dir checkpoints_p4_350m            # 348.8M params, ~6.5h
```

**dense 350M baseline (BUG-2 fix - relaunched 2026-06-02; ~6.5h):**

```bash
python3 train_p4_350m.py --variant dense --seed 42 \
  --n_layers 24 --dim 1024 --num_heads 16 --num_kv_heads 4 \
  --seq_len 1024 --batch_size 8 --grad_accum 8 --steps 15000 --warmup_steps 500 \
  --lr 1e-3 --weight_decay 0.1 --eval_every 500 --ckpt_every 1000 \
  --kill_clock_h 10.0 --save_dir checkpoints_p4b_dense   # 368.79M params
# Resumes automatically from checkpoints_p4b_dense/dense_p4_state.pt if interrupted.
```

When `checkpoints_p4b_dense/dense_p4_final.pt` exists, fill the paper headline:

```bash
python3 finalize_v5_after_dense.py      # → paper/HSPMN_v5_final_numbers.tex, results/phase4b_results_summary.md
```

## 5. Theorem constants (BUG-1 fixed)

```bash
python3 measure_theorem_constants.py --variant hymba-with-nsa-gated \
  --ckpt checkpoints_p2b/hymba-with-nsa-gated_lr1e-3_seed42/hymba-with-nsa-gated_final.pt \
  --n_batches 4 --batch 2 --seq_len 512 \
  --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4 \
  --out_md results/phase3_constants_hymba-with-nsa-gated_$(date +%F).md
# Reports σ_C, p, γ, λ (gate-gain curvature, fp32) and ρ_LB-1.
# Measured 2026-06-02: λ=0.186, ρ_LB-1 ≈ 0.133 (non-vacuous).
```

## 6. Downstream + long-context evals

```bash
python3 eval_hellaswag.py --ckpt checkpoints_p4_350m/hymba-with-nsa_p4_final.pt
python3 niah_v5_long_context.py --ckpt checkpoints_p4_350m/hymba-with-nsa_p4_final.pt \
  --seqs 1024 2048 4096          # needle-in-a-haystack beyond trained S
python3 profile_flops_140m.py    # FLOPs / arithmetic-intensity table
```

## 7. v4 router-collapse diagnostic (the retirement evidence)

```bash
python3 diagnose_v4_router.py --ckpt checkpoints_p2screen/v4-gdn-nsa_lr1e-3_seed42/...final.pt
python3 aggregate_router_diagnostics.py && python3 plot_router_diagnostics.py
```

## 8. Paper

```bash
python3 paper/build_v5_pdf.py    # → paper/HSPMN_v5_draft.pdf
python3 paper/build_pdf.py       # v4 (arXiv-ready)
```

## 9. v6 routability phase diagram (pre-registered; ~2 GPU-days)

Predictions P1-P4 are frozen in the header of `run_v6_queue.py` (2026-07-07),
before any v6 run. The queue is idempotent (skip-if-artifact-exists) and waits
for a free GPU.

```bash
nohup python3 run_v6_queue.py > results/v6_queue.out 2>&1 &
python3 run_v6_queue.py --list          # progress
# pieces, runnable standalone:
python3 build_mix_corpus.py --pis 0.0 0.1 0.25 0.5   # mixture corpora
python3 eval_mix_split.py --variant ... --ckpt ... --data_dir data/mix_pi25
python3 aggregate_v6.py                 # → results/v6_routability_summary.md
```

## Recipe constants (keep matched for fair Δ)

AdamW (β=0.9,0.95, wd=0.1, fused), lr=1e-3 cosine + 500 warmup, bf16, grad-clip
1.0, seed 42. 140M = 12L/768d/12H/4KV; 350M hymba = 24L/896d/14H/2KV (348.8M);
350M dense = 24L/1024d/16H/4KV (368.79M, GQA-locked).
