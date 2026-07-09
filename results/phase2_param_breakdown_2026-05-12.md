# Per-Block Parameter Accounting - 2026-05-12

Config: 12 layers / dim 768 / heads 12 / kv-heads 4 / seq 1024 (P2 config).
Vocab assumed 50,257 (GPT-2 BPE; matches wikitext-103 setup).

## Per-block parameter buckets (one block, params)

| Variant | Block total | k_proj | o_proj | q_proj | v_proj | down | down_proj | gate | gate_proj | k_proj_ctx | k_proj_refl | meta_tokens | norm | norm1 | norm2 | nsa | q_proj_ctx | q_proj_refl | reflexive | router | sink_tokens | ssm | ssm_in_proj | ssm_k_proj | ssm_v_proj | up | up_proj | v_proj_ctx | v_proj_refl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 6,292,992 | 196,608 | 589,824 | 589,824 | 196,608 | 1,572,864 | 0 | 1,572,864 | 0 | 0 | 0 | 0 | 0 | 768 | 768 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,572,864 | 0 | 0 | 0 |
| hymba | 9,637,260 | 196,608 | 589,824 | 589,824 | 196,608 | 0 | 2,359,296 | 0 | 2,359,296 | 0 | 0 | 49,152 | 768 | 0 | 768 | 0 | 0 | 0 | 0 | 0 | 0 | 444,300 | 294,912 | 98,304 | 98,304 | 0 | 2,359,296 | 0 | 0 |
| hymba-with-nsa | 9,644,190 | 196,608 | 589,824 | 589,824 | 196,608 | 0 | 2,359,296 | 0 | 2,359,296 | 0 | 0 | 49,152 | 768 | 0 | 768 | 6,930 | 0 | 0 | 0 | 0 | 0 | 444,300 | 294,912 | 98,304 | 98,304 | 0 | 2,359,296 | 0 | 0 |
| v4-gdn-nsa | 9,676,093 | 0 | 589,824 | 0 | 0 | 0 | 0 | 0 | 0 | 196,608 | 196,608 | 0 | 768 | 0 | 0 | 27,684 | 589,824 | 589,824 | 7,084,824 | 769 | 6,144 | 0 | 0 | 0 | 0 | 0 | 0 | 196,608 | 196,608 |
| v4-rwkv7-nsa | 10,859,569 | 0 | 589,824 | 0 | 0 | 0 | 0 | 0 | 0 | 196,608 | 196,608 | 0 | 768 | 0 | 0 | 27,684 | 589,824 | 589,824 | 8,268,300 | 769 | 6,144 | 0 | 0 | 0 | 0 | 0 | 0 | 196,608 | 196,608 |

## Whole-model totals

| Variant | Embed | Layers (12 blocks) | LM head | Total |
|---|---|---|---|---|
| dense | 38,597,376 | 75,515,904 | 38,597,376 | 152,711,424 (152.71M) |
| hymba | 38,597,376 | 115,647,120 | 0 | 154,245,264 (154.25M) |
| hymba-with-nsa | 38,597,376 | 115,730,280 | 0 | 154,328,424 (154.33M) |
| v4-gdn-nsa | 38,597,376 | 116,113,116 | 0 | 154,711,260 (154.71M) |
| v4-rwkv7-nsa | 38,597,376 | 130,314,828 | 0 | 168,912,972 (168.91M) |

## Interpretation

- **dense** total Q/K/V/O projection params per block: 1,572,864. ×12 layers = 18,874,368 (18.87M).
- **hymba** total Q/K/V/O projection params per block: 1,769,472. ×12 layers = 21,233,664 (21.23M).
- **hymba-with-nsa** total Q/K/V/O projection params per block: 1,769,472. ×12 layers = 21,233,664 (21.23M).
- **v4-gdn-nsa** total Q/K/V/O projection params per block: 2,555,904. ×12 layers = 30,670,848 (30.67M).
- **v4-rwkv7-nsa** total Q/K/V/O projection params per block: 2,555,904. ×12 layers = 30,670,848 (30.67M).

If v4-* projection share is much larger than hymba-with-nsa, H2 (Q/K/V decoupling cost) is part of the gap; if comparable, the +19.47% PPL gap is structural (H3) and not H2.