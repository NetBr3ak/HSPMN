# HSPMN: hybrid recurrent-attention language models

[![CI](https://github.com/NetBr3ak/HSPMN/actions/workflows/ci.yml/badge.svg)](https://github.com/NetBr3ak/HSPMN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Research code for an unpublished manuscript on routing between a Gated DeltaNet
recurrent stream and a Native Sparse Attention (NSA) stream. Includes dense
baselines, training scripts, CPU tests and recorded WikiText-103 experiments.

Author: Szymon Jędryczko.
[Manuscript source](paper/HSPMN_v5_draft.tex) |
[Architecture](ARCHITECTURE.md) |
[Reproduction](REPRODUCE.md)

## Recorded findings

All recorded runs trained and validated on a t+2 target: each position learned
to predict the token two places ahead, not the next token. Every perplexity
below is a t+2 perplexity, not a next-token result. The cause and the evidence
are under [Reproduction status](#reproduction-status).

The experiment labels are 140M and 350M. These are scale names, not exact
parameter counts. The [parameter audit](results/phase2_param_breakdown_2026-05-12.md)
and [larger-scale report](results/phase4_350m_analysis_2026-06-05.md)
give the counts.

At the smaller scale, head fusion recorded 139.14 ± 8.43 validation perplexity
(PPL, lower is better), versus 166.23 ± 15.12 for block fusion with the same
stream types. The relative increase is 19.47%, computed as
`100 * (166.23 / 139.14 - 1)`. These are means and reported standard deviations
over seeds, not confidence intervals.
[Source: combined ablation table](results/phase2_combined_140m_2026-05-12.md).

A learned per-token gate ties
a frozen random gate at 140M (143.85 ± 9.27 vs 146.24 ± 7.56) and at 350M
(82.35 ± 4.78 vs 83.79 ± 1.37, 3 seeds each). Gate-less beats both at
140M (139.14 ± 8.43); at 350M all three are within seed noise (gate-less
85.32 ± 2.64). Head fusion's own −3.4 % edge over dense at 140M decays to
a tie at 350M.
Sources: [smaller-scale table](results/phase2_combined_140m_2026-05-12.md)
and [larger-scale report, decisive result](results/phase4_350m_analysis_2026-06-05.md#decisive-result-2026-06-08---gate-vs-random-gate-at-350m-null).

These sentences describe the archived reports. They are not an independent
replication or a claim of statistical equivalence. The reports also cover
[gate-information probes](results/gate_ceiling_finding_2026-06.md),
[router diagnostics](results/router_diagnostics_summary_2026-05-14.md)
and a [mixture-corpus experiment](results/v6_routability_summary.md).
The larger-scale ± values are population standard deviations (`ddof=0`), not
sample estimates. For example, applying `statistics.pstdev` to the learned-gate
seeds in the larger-scale report gives the displayed 4.78.

## Reproduction status

The audit found a double label shift in every trainer revision in this
repository, from the first import `7c25e92` to `30001dc`. `train_v4.get_batch`
shifted the labels by one token, and every language model shifted them again
inside its loss. Training and validation therefore used a t+2 target.
The audit also found an incorrect Gated DeltaNet CPU recurrence and future-token
access in the predictive-coding gate and automatic NSA block selection.
Software version 5.1.0 fixes these in the reference code; the fixes have not
yet been run at training scale. The recorded measurements predate 5.1.0.
The measurements and manuscript have not been changed.

The [archived v6 report](results/v6_routability_summary.md) shows the same
effect in the recorded runs:

| Line in the report | Recorded value | Meaning |
|---|---|---|
| 7 | Ungated zero-mixture validation PPL 111.90 from the trainer, mean of 3 seeds | CE `ln(111.90) = 4.718` nats on the t+2 target |
| 80 to 82 | Background CE 7.312, 7.364 and 7.329 nats for the same three seeds | The same models scored on next-token prediction by `eval_mix_split.py` |
| 32 to 91 | Answer-position CE 21.325 to 27.558 nats in the mixture runs | Worse than a uniform guess, `ln(50257) = 10.82` nats |

Raw logs and checkpoints are not in the repository. The gate-information
probes, probe ceilings and v6 verdicts P1 to P4 score next-token CE
(`measure_gate_channel_mi.per_token_ce`) on these t+2 models, so they measure a
task the models did not learn. None of the archived PPL or probe numbers is a
next-token finding. Use new output directories for corrected runs.
See [known limitations](TROUBLESHOOTING.md).

## CPU quick start

Use Python compatible with [pyproject.toml](pyproject.toml).
No dataset, checkpoint or GPU is needed for the test suite.

```bash
git clone https://github.com/NetBr3ak/HSPMN.git
cd HSPMN
python3 -m venv .venv
. .venv/bin/activate
# Linux CPU wheel; on macOS omit the --index-url option.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.3 pytest ruff==0.17.3
ruff check .
ruff format --check .
python -m pytest -q
```

CI runs lint, format checks and CPU tests. CUDA/Triton tests remain in the suite
and skip when their hardware or runtime is absent. A CPU pass does not validate
GPU kernels, training throughput or the manuscript results.

For dataset downloads, a small training check and the full GPU recipes, see
[REPRODUCE.md](REPRODUCE.md). The recorded hardware and training settings are in
the [experiment report](results/phase4_350m_analysis_2026-06-05.md).

## Repository map

| Path | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Streams, variants and code entry points |
| [REPRODUCE.md](REPRODUCE.md) | Environment, data preparation and run commands |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Correctness findings and remaining limits |
| Root Python files | Models, trainers, probes, evaluators and tests |
| [results/](results/) | Recorded aggregates, parameter counts and diagnostics |
| [paper/](paper/) | Unpublished manuscript, figures and preview generator |

Token arrays and trained weights are generated locally and ignored by Git.
The historical version suffixes in filenames are retained because the scripts
import one another by those names.

## Licence, assets and citation

Code, manuscript and project-generated figures use the [MIT licence](LICENSE).
Bundled DejaVu fonts retain their separate [font licence](paper/fonts/LICENSE);
see the [DejaVu project](https://dejavu-fonts.github.io/License.html).
Figure sources are [paper/make_figures.py](paper/make_figures.py) and the
[router CSV](results/router_diagnostics_layer_health_2026-05-14.csv).
The original heatmap renderer is not included.

WikiText and the GPT-2 tokenizer are downloaded separately. Their sources and
licences are listed in [REPRODUCE.md](REPRODUCE.md#data-preparation-cpu).
The repository's MIT licence does not relicense external datasets.

Use [CITATION.cff](CITATION.cff) to cite the software. The manuscript has no
publication identifier in this repository.
