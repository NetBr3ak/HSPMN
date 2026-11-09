# Known limitations and correctness notes

## Recorded results versus corrected code

The aggregate files in `results/` and the manuscript are preserved as recorded.
The training-time source revision, raw logs and checkpoints are not supplied.
Every trainer revision in this repository applied the double label shift, so the
archived runs trained and validated on a t+2 target. The evidence lines from
`results/v6_routability_summary.md` are listed in [README.md](README.md#reproduction-status).
New results need new output paths and a recorded source revision.

| Finding | Reference-code change | Consequence for old measurements |
|---|---|---|
| Token batches shifted labels before the LM shifted them again | Labels match input IDs in `train_v4.py` and both loss paths in `measure_theorem_constants.py`; the LM alone shifts them | The old code predicts the token after the next one (a t+2 target). Archived PPL is t+2 PPL, not next-token PPL |
| Gate MI, probe ceilings and v6 P1 to P4 depend on the trained objective | No historical probe number or verdict is changed | `measure_gate_channel_mi.per_token_ce` scores next tokens and supplies the ceiling and sequence probes. On the t+2 models these probes score a task the models did not learn. Rerun training and probes before interpreting the archived verdicts |
| HellaSwag omitted the first ending token | Scores from the last context logit and includes every ending token | Single-token endings now have a finite score; no HellaSwag result is archived here |
| Dense block imported a missing benchmark module | Uses the shared rotary embedding implementation | Historical dense numerical equivalence is unknown |
| PyTorch Gated DeltaNet wrote a value-first state but read it key-first | Uses key-first writes, decays before the delta update and scales the query | CPU reference output changed; accelerated FLA is a separate path |
| CPU and FLA paths used different Q/K transforms | CPU now normalizes Q and K instead of applying SiLU only to K | GPU parity still needs a kernel comparison on target hardware |
| Predictive-coding gate normalized over the entire sequence | Uses prefix moments | Future tokens no longer determine an earlier gate; old PC-gate results need causal reruns |
| Automatic NSA selection pooled future queries | Selects completed blocks per query with a reference attention mask | Old selection-variant results need causal reruns |
| Training CE was multiplied by gradient accumulation again | Logs the already averaged accumulation loss | This logging defect does not change backward loss or validation loss |
| Larger trainer saved a final artifact after a time-limit break | Saves resume state and exits unsuccessfully | Existing partial final artifacts cannot be trusted merely because their filename says final |
| Completed-state resume left the final step undefined | Tracks completed steps before entering the loop | A completed checkpoint can reach final evaluation without an undefined loop variable |
| Queues ignored failed process exit codes | Require successful exit and an artifact; report failures through exit status | Existing artifacts are still skipped without content validation |
| Aggregation without raw logs overwrote the summary with an empty table | Fails before writing when no run logs exist | Partial grids are still possible; inspect completeness before interpreting an aggregate |

The Gated DeltaNet recurrence follows the state convention and decay order in the
[FLA reference implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py).
The regression suite checks a written key-value pair and an independent matrix
form. CPU checks alone do not establish parity with Triton/FLA kernels.

## Scientific interpretation

A failure to detect a difference is not proof of equivalence. The archived tables
report seed variation; their standard deviations are not confidence intervals.

The manuscript uses information-theoretic bounds. Mutual information measured
with a finite trained probe is evidence about that probe, not a general upper
bound on all information in the hidden state. Read the theoretical assumptions
separately from the empirical probe result.

The legacy probe scripts still print older threshold formulas and broad verdicts.
Their outputs must not be treated as a checked implementation of the corrected
manuscript bound. Their numeric estimates and theorem interpretations need a
separate scientific review before further publication.

The nominal model-scale names are not exact parameter counts or exact parameter
matching. Use [the parameter audit](results/phase2_param_breakdown_2026-05-12.md)
and [scale report](results/phase4_350m_analysis_2026-06-05.md).

## Runtime and data

| Symptom | Check |
|---|---|
| CUDA or Triton tests skip | CPU CI deliberately does not validate GPU kernels. Run the same suite on the target CUDA host |
| Early automatic-selection rows have no completed compression block | Check finite zero outputs and causal behavior on each target CUDA SDPA backend; CPU behavior does not validate CUDA |
| Training is unexpectedly slow | Check whether FLA loaded. The PyTorch recurrence loops over tokens |
| Tokenizer or Parquet file missing | Follow the download commands in REPRODUCE.md before tokenizing |
| Existing token output is silently reused | Tokenization skips existing output arrays. Use a new output directory when changing source inputs |
| A larger-scale run reads a different corpus | Pass explicit `--train_npy` and `--valid_npy`; automatic discovery prefers FineWeb files independently |
| Out of memory at long context | Window attention and automatic selection use quadratic reference tensors |
| Queue says DONE on a fresh checkout | Some summary artifacts are committed. DONE means an artifact exists, not that the underlying jobs ran locally |
| Resumed run differs from uninterrupted run | Resume checkpoints do not restore random-generator state or validate every configuration field |
| Final `tokens_seen` looks too small after resume | The larger trainer counts tokens only since the latest process start; this field is not a lifetime training-token count |
| A locally generated preview differs from TeX | ReportLab uses separately maintained text. The missing heatmap PNG is omitted from that preview |

Older block-fusion routers also select positions using whole-sequence top-k.
Treat those modules as diagnostic comparisons, not as a verified causal
language-model implementation. The ASA selection-only branch has no external
selection indices in the head-fusion path and contributes zero attention there.
These limitations remain for a scientific follow-up.

## Checkpoint trust

Load only checkpoints that you produced or whose origin you trust.
The HellaSwag evaluator now explicitly uses restricted `weights_only=True`
loading for the state dictionaries and plain configuration data written by the
trainers. Do not switch to unrestricted pickle loading to accept an unknown file.
Restricted loading does not make arbitrary checkpoint contents trustworthy.
The repository distributes no checkpoints.

The former curvature-estimation and interrupted-dense-run notes remain in the
[theorem report](results/theorem_empirics_2026-06-02.md) and
[larger-scale report](results/phase4_350m_analysis_2026-06-05.md).
Those dated reports include preliminary interpretations later superseded within
the same files. Prefer the final sections and keep the audit limits above.
