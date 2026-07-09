# H1 Router-Collapse Diagnostic on v4-gdn-nsa

**Date:** 2026-05-12. **Checkpoint:** `/opt/docker/LLM/HSPMN/checkpoints_p2screen/v4-gdn-nsa_lr1e-3_seed1337/v4-gdn-nsa_final.pt`.
**Validation batches:** 8 × 4 × 1024.

## Headline numbers

- mean active_fraction across layers: 0.0759
- target_sparsity (config): 0.25 (set by ALF-LB bias)
- min/max active_fraction: 0.0000 / 0.8944
- layerwise routing-health H_route: 0.0368
- dead / near-dead / saturated layers: 5 / 10 / 0
- mean gate.mean()|active: 0.2251

## Per-layer

| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.8944 | 0.3779 | 0.0130 | 0.0387 | 0.0272 | 0.0433 | 0.0387 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0001 | 0.0001 | 0.0000 | 0.0004 | 0.0264 | 0.0000 |
| 4 | 0.0046 | 0.0182 | 0.0036 | 0.0054 | 0.0931 | 1.2202 | 0.0054 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | 0.0000 | 0.0001 | 0.0001 | 0.0000 | 0.0003 | 0.0214 | 0.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | 0.0114 | 0.0449 | 0.0089 | 0.0167 | 0.1802 | 1.3037 | 0.0167 |
| 10 | 0.0001 | 0.0004 | 0.0002 | 0.0000 | 0.0018 | 0.0763 | 0.0000 |
| 11 | 0.0000 | 0.0001 | 0.0001 | 0.0000 | 0.0002 | 0.0103 | 0.0000 |

## Interpretation

**H1 partial:** mean active_fraction = 0.0759 drifted from target=0.25. Gate is not fully off, but ALF-LB did not hold target - partial absorption likely.
