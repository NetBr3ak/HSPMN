# Router diagnostics summary - 2026-05-14

Aggregates per-seed router diagnostics emitted by `diagnose_v4_router.py`.

## Variant-level summary

| Variant | LR | Seeds | mean active_fraction | H_route | dead layers | near-dead layers | saturated layers |
|---|---|---:|---:|---:|---:|---:|---:|
| v4-gdn-nsa | 1e-3 | 42,1337,2026 | 0.1808 ± 0.0912 | 0.0527 ± 0.0274 | 5.3333 ± 0.5774 | 9.0000 ± 1.0000 | 1.3333 ± 1.1547 |
| v4-rwkv7-nsa | 1e-3 | 42,1337,2026 | 0.2176 ± 0.1173 | 0.0959 ± 0.0635 | 6.6667 ± 3.5119 | 9.0000 ± 1.7321 | 1.3333 ± 0.5774 |

## Interpretation guide

- `mean active_fraction` is the aggregate number that can look healthy.
- `H_route = mean_l 4 p_l (1-p_l)` is layerwise routing health.
- Low `H_route` with plausible aggregate active fraction is the headline failure mode.
- `near-dead layers` uses active fraction < 0.01; `saturated layers` uses active fraction > 0.95.

## Layer-level CSV

Layer-level data written to `/opt/docker/LLM/HSPMN/results/router_diagnostics_layer_health_2026-05-14.csv`.
