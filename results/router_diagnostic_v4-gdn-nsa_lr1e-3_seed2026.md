# H1 Router-Collapse Diagnostic on v4-gdn-nsa

**Date:** 2026-05-12. **Checkpoint:** `/opt/docker/LLM/HSPMN/checkpoints_p2screen/v4-gdn-nsa_lr1e-3_seed2026/v4-gdn-nsa_final.pt`.
**Validation batches:** 8 × 4 × 1024.

## Headline numbers

- mean active_fraction across layers: 0.2405
- target_sparsity (config): 0.25 (set by ALF-LB bias)
- min/max active_fraction: 0.0000 / 0.9926
- layerwise routing-health H_route: 0.0368
- dead / near-dead / saturated layers: 5 / 9 / 2
- mean gate.mean()|active: 1.0006

## Per-layer

| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.9102 | 0.3271 | 0.0116 | 0.0371 | 0.0256 | 0.0408 | 0.0371 |
| 1 | 0.9926 | 0.0292 | 0.0031 | 1.8406 | 0.6674 | 1.8542 | 1.8406 |
| 2 | 0.0009 | 0.0035 | 0.0007 | 0.0004 | 0.0124 | 0.3194 | 0.0004 |
| 3 | 0.0001 | 0.0002 | 0.0002 | 0.0000 | 0.0015 | 0.0578 | 0.0000 |
| 4 | 0.0001 | 0.0004 | 0.0002 | 0.0002 | 0.0088 | 0.3796 | 0.0002 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | 0.9807 | 0.0759 | 0.0065 | 8.1454 | 3.4589 | 8.3051 | 8.1454 |
| 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | 0.0014 | 0.0056 | 0.0014 | 0.0025 | 0.0602 | 1.0508 | 0.0025 |

## Interpretation

**H1 NOT confirmed:** mean active_fraction = 0.2405 matches target_sparsity=0.25 - ALF-LB held the gate. Loss does not come from gate-side collapse.
