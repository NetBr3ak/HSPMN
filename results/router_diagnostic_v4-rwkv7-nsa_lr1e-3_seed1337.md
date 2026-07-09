# H1 Router-Collapse Diagnostic on v4-rwkv7-nsa

**Date:** 2026-05-12. **Checkpoint:** `/opt/docker/LLM/HSPMN/checkpoints_p2screen/v4-rwkv7-nsa_lr1e-3_seed1337/v4-rwkv7-nsa_final.pt`.
**Validation batches:** 8 × 4 × 1024.

## Headline numbers

- mean active_fraction across layers: 0.1560
- target_sparsity (config): 0.25 (set by ALF-LB bias)
- min/max active_fraction: 0.0000 / 0.9905
- layerwise routing-health H_route: 0.0481
- dead / near-dead / saturated layers: 7 / 10 / 1
- mean gate.mean()|active: 1.4999

## Per-layer

| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.8637 | 0.4708 | 0.0150 | 0.0513 | 0.0406 | 0.0594 | 0.0513 |
| 1 | 0.0097 | 0.0384 | 0.0023 | 0.0621 | 0.6305 | 6.3949 | 0.0621 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.0002 | 0.0010 | 0.0003 | 0.0001 | 0.0057 | 0.2288 | 0.0001 |
| 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | 0.9905 | 0.0376 | 0.0022 | 9.9120 | 9.0160 | 10.0073 | 9.9120 |
| 9 | 0.0073 | 0.0290 | 0.0087 | 0.0116 | 0.1480 | 1.3081 | 0.0116 |
| 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

**H1 partial:** mean active_fraction = 0.1560 drifted from target=0.25. Gate is not fully off, but ALF-LB did not hold target - partial absorption likely.
