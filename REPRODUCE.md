# Running HSPMN from a fresh clone

Run commands from the repository root. CPU checks need no corpus or checkpoint.
The full training recipes need a CUDA GPU and generated token arrays.

## What can be reproduced

| Task | Hardware | Inputs and limits |
|---|---|---|
| Lint and unit tests | CPU | Source checkout; no dataset |
| Tokenization and mixture generation | CPU | External corpus and tokenizer downloads |
| Small dense training check | CPU or CUDA | Generated local token arrays |
| Full model training | CUDA recommended | Linux GPU environment and token arrays |
| Gate probes and downstream evaluation | CUDA recommended | Locally trained checkpoints; none are shipped |
| Manuscript compilation | CPU | TeX Live; independent of model training |

The committed measurements are historical aggregates, not a verified run of the
corrected reference code. They come from runs that trained and validated on a
t+2 target, not on next-token prediction
([evidence](README.md#reproduction-status)). Raw training logs, checkpoints, exact corpus hashes and
the custom Triton build are absent. Several former aggregators are absent too.
The code defects and their possible impact are listed in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Do not overwrite the recorded reports
with new results.

## CPU environment and checks

The project declares its Python requirement in [pyproject.toml](pyproject.toml).
The following versions match the CPU CI environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
# Linux; on macOS omit --index-url.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.3 pytest ruff==0.17.3
ruff check .
ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests that require CUDA or Triton skip without those requirements. Keep those
tests in GPU validation. CI does not run full training or validate the paper.
Measured on 2026-11-09 for version 5.1.0: 71 passed and one Triton skip in
2.62 s, as reported by pytest. Method: Python 3.13.9, torch 2.10.0+cpu,
NumPy 2.4.3 and `OMP_NUM_THREADS=1 python -m pytest -q -rs` on Linux with an
Intel Core i5-8350U laptop CPU. One run, not an average.

This repository runs as scripts from its root. These commands do not require
installing the checkout as a Python package.

## GPU environment

Use a separate environment from the CPU setup. The historical dependency record
is [requirements.txt](requirements.txt). It records a custom Triton build with
only a nearest release pin, so it is not a complete environment lock.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install pytest ruff==0.17.3
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
python -m pytest -q
```

The wheel command follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v210).
Check the installed FLA and Triton kernels on the target GPU before launching a
sweep. Falling back to a Python recurrence changes performance and may change
numerics. A CUDA wheel cannot add GPU support to macOS.
Also check automatic-selection queries before the first compression block is
complete. Their empty attention rows must give finite zero outputs on each
target SDPA backend. A CPU pass does not establish that CUDA behavior.

The [recorded experiment](results/phase4_350m_analysis_2026-06-05.md) used an
RTX 5090. Other devices have not been benchmarked here.

## Data preparation (CPU)

The commands below download a pinned WikiText raw corpus revision and GPT-2
tokenizer for new runs. The historical token arrays were not retained, so
byte-for-byte equality with the original training input is not established.

| Input | Source | Licence |
|---|---|---|
| WikiText-103 raw corpus | [Salesforce dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) | Card lists CC BY-SA and GFDL; retain its attribution and terms when redistributing data |
| GPT-2 tokenizer | [Model card](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/README.md) | MIT |

WikiText attribution: Stephen Merity, Caiming Xiong, James Bradbury and Richard
Socher, *Pointer Sentinel Mixture Models*, using Wikipedia articles. The download
is external data, not part of this repository's MIT code licence.

```bash
python -m pip install tokenizers==0.22.2 pyarrow==24.0.0
mkdir -p data/tokenizer
corpus_url=https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1
curl --fail --location "$corpus_url/train-00000-of-00002.parquet" -o data/wikitext_train_0.parquet
curl --fail --location "$corpus_url/train-00001-of-00002.parquet" -o data/wikitext_train_1.parquet
curl --fail --location "$corpus_url/validation-00000-of-00001.parquet" -o data/wikitext_valid.parquet
curl --fail --location https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json -o data/tokenizer/tokenizer.json
python tokenize_corpus.py
python -c "import numpy as np; print({s: np.load('data/' + s + '_tokens.npy', mmap_mode='r').shape for s in ('train', 'valid')})"
```

Tokenization skips blank rows and appends an end-of-text token after each
remaining row. It holds the token list in memory. Use a machine with memory
headroom; preparation time and peak memory are not recorded. Existing output
arrays are skipped, so use a new data directory when changing inputs.

### Small training check

This checks data loading, model construction, backward propagation and
checkpoint writing. The small dimensions and step count are test settings,
not a published experiment recipe.

```bash
python train_v4.py --variant dense --n_layers 1 --dim 32 \
  --num_heads 4 --num_kv_heads 2 --seq_len 16 --batch_size 1 \
  --grad_accum 1 --steps 2 --warmup_steps 1 --eval_every 1 \
  --eval_iters 1 --log_every 1 --save_dir checkpoints_smoke/dense
```

Require a zero exit code and `checkpoints_smoke/dense/dense_final.pt`.
A smoke test does not check scientific convergence.

## Smaller-scale training recipe (CUDA)

Architecture and training settings come from the
[combined ablation report](results/phase2_combined_140m_2026-05-12.md) and
[parameter accounting](results/phase2_param_breakdown_2026-05-12.md).
All flags below are explicit; trainer defaults describe a different run.
The reports do not record the historical warmup. The warmup below is a new-run
choice, not a recovered experiment setting.

```bash
python train_v4.py --variant hymba-with-nsa --lr 1e-3 --seed 42 \
  --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4 \
  --seq_len 1024 --batch_size 4 --grad_accum 8 --steps 3000 \
  --warmup_steps 500 --nsa_window 256 \
  --save_dir checkpoints_corrected_140m/hymba-with-nsa_seed42
```

For the learned and random-gate controls, use
`--variant hymba-with-nsa-gated` and
`--variant hymba-with-nsa-randgate`. Repeat with seeds 1337 and 2026,
using distinct save directories. These seed values are recorded in the
[theorem-empirics report](results/theorem_empirics_2026-06-02.md).
The historical aggregate does not preserve every trainer flag or the original
learning-rate screen; this is an explicit new-run recipe, not an exact replay.

Time estimate: `3000 * 1024 * 4 * 8 / 50582 / 3600 = 0.54 GPU-hours` for
the training-token budget at the old recorded hybrid throughput.
[Source: token rate and recipe](results/phase2_combined_140m_2026-05-12.md).
This excludes evaluation, setup and checkpoint I/O. It is a planning estimate,
not a measured runtime for the corrected code.

## Larger-scale training recipe (CUDA)

The [scale report](results/phase4_350m_analysis_2026-06-05.md) records the
architecture, training budget and evaluation settings. The hybrid configuration
is defined in the historical queue, while the dense configuration uses its own
width and head counts.

```bash
python train_p4_350m.py --variant hymba-with-nsa --seed 42 \
  --train_npy data/train_tokens.npy --valid_npy data/valid_tokens.npy \
  --n_layers 24 --dim 896 --num_heads 14 --num_kv_heads 2 \
  --seq_len 1024 --batch_size 8 --grad_accum 8 --steps 15000 \
  --warmup_steps 500 --lr 1e-3 --weight_decay 0.1 --nsa_window 512 \
  --eval_every 500 --eval_iters 32 --ckpt_every 1000 --kill_clock_h 10 \
  --save_dir checkpoints_corrected_350m/hymba-with-nsa_seed42

python train_p4_350m.py --variant dense --seed 42 \
  --train_npy data/train_tokens.npy --valid_npy data/valid_tokens.npy \
  --n_layers 24 --dim 1024 --num_heads 16 --num_kv_heads 4 \
  --seq_len 1024 --batch_size 8 --grad_accum 8 --steps 15000 \
  --warmup_steps 500 --lr 1e-3 --weight_decay 0.1 --nsa_window 512 \
  --eval_every 500 --eval_iters 32 --ckpt_every 1000 --kill_clock_h 10 \
  --save_dir checkpoints_corrected_350m/dense_seed42
```

Repeat the hybrid recipe for the learned and frozen-random variants and for
the other recorded seeds, always with a separate save directory.

The trainer resumes from `<variant>_p4_state.pt` in its save directory.
On its time limit it saves resumable state and exits unsuccessfully without
creating a final artifact. Rerun the same command to continue.
An external termination only preserves the last periodic checkpoint.
The resume state does not preserve random-generator state, so a resumed run
is not guaranteed to match an uninterrupted run.

The [larger-scale report](results/phase4_350m_analysis_2026-06-05.md#narrative-decision-pending-queue-2)
records a Queue-2 estimate of about 30 hours. This is a historical planning
estimate, not a measured runtime or an estimate for the corrected code.
Measure the first run on the target machine before budgeting the whole sweep.
The time-limit flag controls interruptions; it is not a runtime estimate.

## Synthetic state-tracking tasks

The synthetic trainer generates MQAR, Dyck-2 and parity inputs locally.
It does not need WikiText or a tokenizer download.

```bash
python train_synthetic.py --all
```

This uses the task and model settings in [train_synthetic.py](train_synthetic.py).
Compare new outputs with the archived
[JSON](results/phase2_synthetics_2026-05-12.json) and
[table](results/phase2_synthetics_table_2026-05-12.md), without overwriting them.
The original Phase-2 battery runner and table aggregator are not included;
`--all` runs the variants currently listed in the trainer, not an exact replay
of that table. Runtime has not been recorded; measure it on the target host.

## Mixtures, probes and evaluators

`run_v5_queue2.py` is a historical follow-up queue, not a fresh-clone
end-to-end runner. It expects a completed ungated larger-scale checkpoint and
does not train every missing baseline.

Inspect the mixture queue without starting GPU work:

```bash
python run_v6_queue.py --list
python run_v5_queue2.py --list
```

After preparing the token arrays, the mixture corpus can be generated on CPU:

```bash
python build_mix_corpus.py --pis 0.0 0.1 0.25 0.5
```

The fractions above match the
[archived mixture study](results/v6_routability_summary.md).
The queue contains its own training, probe and evaluation commands.
Run it only in a separate checkout for new experiments: its output paths
overlap the committed historical reports, and its completion checks use artifact
existence, not a full configuration comparison.

For an independently trained learned-gate checkpoint, a gate-information probe is:

```bash
python measure_gate_channel_mi.py --variant hymba-with-nsa-gated \
  --ckpt checkpoints_corrected_140m/hymba-with-nsa-gated_seed42/hymba-with-nsa-gated_final.pt \
  --label_ckpt checkpoints_corrected_140m/hymba-with-nsa_seed42/hymba-with-nsa_final.pt \
  --n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4 \
  --out_md results/new_gate_mi.md
```

This command depends on both training runs. It uses the architecture recorded in
the [smaller-scale report](results/phase2_combined_140m_2026-05-12.md).
The original linear/nonlinear probes and sequence probes remain available,
but their legacy threshold formulas need review; do not use their verdict text
as a validated theorem check.

`eval_hellaswag.py` requires both `--variant` and `--ckpt`, a tokenizer
and an external HellaSwag JSONL file. No evaluation data or result is supplied.
Verify the source's licence and schema before using an external copy.

`profile_flops_140m.py` builds full models, even on CPU. It is not a quick
sanity test. Pass a new `--output` path because its default overwrites a
recorded report. `aggregate_v6.py` requires new per-run CSVs and probe artifacts;
it cannot regenerate the original measurements from the shipped summary.

## Paper

The repository contains an unpublished TeX manuscript and a separate ReportLab
preview generator. The old generated preview was removed because its header
misstated its source. The manuscript has not been updated to reflect the code
audit: it presents the t+2 perplexities and probes as next-token
language-model results.

For a TeX build in a separate checkout:

```bash
cd paper
pdflatex HSPMN_v5_draft.tex
pdflatex HSPMN_v5_draft.tex
```

Do not run the preview or figure generators as a test: they overwrite tracked
PDF/PNG artifacts. The preview script contains separate text and tables, not a
TeX renderer. See [paper/00README.txt](paper/00README.txt) for dependencies and
the missing heatmap-renderer limitation.
