# H1 Router-Collapse Diagnostic on v4-gdn-nsa

**Date:** 2026-05-12. **Checkpoint:** `/opt/docker/LLM/HSPMN/checkpoints_p2screen/v4-gdn-nsa_lr1e-3_seed42/v4-gdn-nsa_final.pt`.
**Validation batches:** 8 × 4 × 1024.

## Headline numbers

- mean active_fraction across layers: 0.2262
- target_sparsity (config): 0.25 (set by ALF-LB bias)
- min/max active_fraction: 0.0000 / 0.9800
- layerwise routing-health H_route: 0.0844
- dead / near-dead / saturated layers: 6 / 8 / 2
- mean gate.mean()|active: 7.2282

## Per-layer

| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.7636 | 0.7221 | 0.0231 | 0.0298 | 0.0289 | 0.0390 | 0.0298 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.9800 | 0.0782 | 0.0038 | 3.4232 | 1.0292 | 3.4928 | 3.4232 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 0.0125 | 0.0493 | 0.0047 | 0.4206 | 3.6898 | 33.3563 | 0.4206 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | 0.0002 | 0.0006 | 0.0002 | 0.0001 | 0.0062 | 0.2822 | 0.0001 |
| 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | 0.9577 | 0.1619 | 0.0221 | 47.2180 | 316.1693 | 49.5311 | 47.2180 |
| 10 | 0.0000 | 0.0001 | 0.0001 | 0.0000 | 0.0006 | 0.0369 | 0.0000 |
| 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

**H1 NOT confirmed:** mean active_fraction = 0.2262 matches target_sparsity=0.25 - ALF-LB held the gate. Loss does not come from gate-side collapse.
