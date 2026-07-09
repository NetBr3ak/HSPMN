# H1 Router-Collapse Diagnostic on v4-rwkv7-nsa

**Date:** 2026-05-12. **Checkpoint:** `/opt/docker/LLM/HSPMN/checkpoints_p2screen/v4-rwkv7-nsa_lr1e-3_seed42/v4-rwkv7-nsa_final.pt`.
**Validation batches:** 8 × 4 × 1024.

## Headline numbers

- mean active_fraction across layers: 0.3529
- target_sparsity (config): 0.25 (set by ALF-LB bias)
- min/max active_fraction: 0.0000 / 0.9772
- layerwise routing-health H_route: 0.1680
- dead / near-dead / saturated layers: 3 / 7 / 2
- mean gate.mean()|active: 2.6265

## Per-layer

| Layer | active_fraction | H_route_layer | std | gate.mean() | gate.std() | gate.mean()|active | L1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.8419 | 0.5324 | 0.0148 | 0.0301 | 0.0240 | 0.0357 | 0.0301 |
| 1 | 0.9772 | 0.0890 | 0.0040 | 2.6267 | 0.7295 | 2.6878 | 2.6267 |
| 2 | 0.5178 | 0.9987 | 0.0254 | 2.0425 | 2.6838 | 3.9374 | 2.0425 |
| 3 | 0.9705 | 0.1144 | 0.0045 | 14.2339 | 5.8806 | 14.6638 | 14.2339 |
| 4 | 0.0001 | 0.0004 | 0.0002 | 0.0001 | 0.0059 | 0.1893 | 0.0001 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | 0.9259 | 0.2745 | 0.0204 | 4.0953 | 3.0413 | 4.4196 | 4.0953 |
| 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | 0.0013 | 0.0051 | 0.0011 | 0.0060 | 0.1534 | 3.0103 | 0.0060 |
| 10 | 0.0001 | 0.0002 | 0.0001 | 0.0006 | 0.0363 | 2.3250 | 0.0006 |
| 11 | 0.0002 | 0.0010 | 0.0004 | 0.0002 | 0.0075 | 0.2493 | 0.0002 |

## Interpretation

**H1 partial:** mean active_fraction = 0.3529 drifted from target=0.25. Gate is not fully off, but ALF-LB did not hold target - partial absorption likely.
