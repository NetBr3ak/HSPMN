# Phase 2 Synthetics Table - 2026-05-12

Comparison of all variants (Phase 1 + Phase 2 additions) on the small-scale synthetic state-tracking battery (1.7M-2.8M params, 4 layers, dim 128, 2000 steps).

Reviewer-C expressivity probe. State-tracking primitives (Mamba-3, RWKV-7, GDN) are expected to win on parity / Dyck-2; pure-attention models (dense) are expected to fail at parity (TC^0 limit).

## mqar

| Variant | Params (M) | Accuracy | Final CE |
|---|---|---|---|
| dense | 2.786 | 0.02% | 8.334 |
| v4-gdn-nsa | 2.177 | 0.72% | 7.823 |
| v4-mamba3-nsa | 2.243 | 0.71% | 7.872 |
| v4-rwkv7-nsa ★ | 2.308 | 1.03% | 7.776 |
| hymba | 2.181 | 0.12% | 8.303 |
| hymba-with-asa | 2.182 | 0.06% | 8.354 |
| hymba-with-nsa | 2.183 | 0.37% | 8.131 |
| hymba-with-nsa-gated | 2.184 | 0.23% | 8.160 |
| hymba-with-nsa-randgate | 2.184 | 0.02% | 8.356 |

## dyck2

| Variant | Params (M) | PPL | CE |
|---|---|---|---|
| dense | 0.691 | 2.737 | 1.007 |
| v4-gdn-nsa | 1.129 | 2.590 | 0.952 |
| v4-mamba3-nsa | 1.195 | 2.646 | 0.973 |
| v4-rwkv7-nsa ★ | 1.260 | 2.565 | 0.942 |
| hymba | 1.133 | 2.693 | 0.990 |
| hymba-with-asa | 1.134 | 2.679 | 0.986 |
| hymba-with-nsa | 1.135 | 2.677 | 0.985 |
| hymba-with-nsa-gated | 1.136 | 2.635 | 0.969 |
| hymba-with-nsa-randgate | 1.136 | 2.656 | 0.977 |

## parity

| Variant | Params (M) | Accuracy | CE |
|---|---|---|---|
| dense | 0.690 | 50.86% | 0.685 |
| v4-gdn-nsa ★ | 1.129 | 52.41% | 0.661 |
| v4-mamba3-nsa | 1.195 | 50.58% | 0.672 |
| v4-rwkv7-nsa | 1.260 | 52.15% | 0.658 |
| hymba | 1.133 | 50.73% | 0.683 |
| hymba-with-asa | 1.133 | 51.09% | 0.681 |
| hymba-with-nsa | 1.134 | 50.59% | 0.684 |
| hymba-with-nsa-gated | 1.135 | 50.90% | 0.678 |
| hymba-with-nsa-randgate | 1.135 | 50.68% | 0.686 |

## Headline takeaways

- Best per task marked with ★.
- Random-gate (Aquino-Michaels baseline) is compared with the gated learned variant on hymba-with-nsa - gap between them measures how much routing actually carries (per arXiv:2603.02227).
- v4-* family already shown to lose on wikitext (P2 screen 2026-05-09); these synthetic numbers test whether the loss is universal or task-specific.
