# H1 Router-Collapse Diagnostic on v4-rwkv7-nsa

**Date:** 2026-05-12. **Checkpoint:** `/opt/docker/LLM/HSPMN/checkpoints_p2screen/v4-rwkv7-nsa_lr1e-3_seed2026/v4-rwkv7-nsa_final.pt`.
**Validation batches:** 8 × 4 × 1024.

## Headline numbers

- mean active_fraction across layers: 0.1440
- target_sparsity (config): 0.25 (set by ALF-LB bias)
- min/max active_fraction: 0.0000 / 0.9651
- layerwise routing-health H_route: 0.0716
- dead / near-dead / saturated layers: 10 / 10 / 1
- mean gate.mean()|active: 0.1779

## Per-layer

| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.7623 | 0.7248 | 0.0160 | 0.0495 | 0.0465 | 0.0649 | 0.0495 |
| 1 | 0.9651 | 0.1347 | 0.0074 | 1.9973 | 0.7068 | 2.0696 | 1.9973 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

**H1 partial:** mean active_fraction = 0.1440 drifted from target=0.25. Gate is not fully off, but ALF-LB did not hold target - partial absorption likely.
