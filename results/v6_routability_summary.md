# v6 routability phase diagram - summary

## Validation PPL by variant x pi (mean +/- std over seeds)

| variant | pi=0.00 | pi=0.10 | pi=0.25 | pi=0.50 |
|---|---|---|---|---|
| hymba-with-nsa | 111.90 +/- 3.56 (n=3) | 109.63 +/- 1.64 (n=3) | 103.15 +/- 3.13 (n=3) | 88.73 +/- 2.94 (n=3) |
| hymba-with-nsa-decor | 111.59 +/- 4.07 (n=3) | - | 103.10 +/- 2.31 (n=3) | - |
| hymba-with-nsa-decor-gated | 114.37 +/- 4.54 (n=3) | - | 103.42 +/- 1.86 (n=3) | - |
| hymba-with-nsa-gated | 115.99 +/- 2.53 (n=3) | 110.31 +/- 2.32 (n=3) | 104.13 +/- 3.05 (n=3) | 92.61 +/- 2.47 (n=3) |
| hymba-with-nsa-pcgate | 114.71 +/- 4.68 (n=3) | 110.59 +/- 0.98 (n=3) | 101.09 +/- 1.78 (n=3) | 87.26 +/- 1.04 (n=3) |
| hymba-with-nsa-randgate | 115.60 +/- 3.46 (n=3) | 111.75 +/- 1.67 (n=3) | 105.04 +/- 0.96 (n=3) | 90.69 +/- 4.40 (n=3) |

## P1 - probe ceiling I_max(pi) [bits]

- pi=0.00: I_max = 0.0177
- pi=0.10: I_max = 0.0582
- pi=0.25: I_max = 0.1076
- pi=0.50: I_max = 0.1392

**P1 verdict: CONFIRMED (monotone rise)**

## P2 - learned vs random gate by pi

- pi=0.00: gated 115.99+/-2.53 vs randgate 115.60+/-3.46 -> tie
- pi=0.10: gated 110.31+/-2.32 vs randgate 111.75+/-1.67 -> tie
- pi=0.25: gated 104.13+/-3.05 vs randgate 105.04+/-0.96 -> tie
- pi=0.50: gated 92.61+/-2.47 vs randgate 90.69+/-4.40 -> tie

## P3 - answer-position vs background CE

- hymba-with-nsa-decor-gated_pi00_s1337: answer CE 0.000, background CE 7.393
- hymba-with-nsa-decor-gated_pi00_s2026: answer CE 0.000, background CE 7.322
- hymba-with-nsa-decor-gated_pi00_s42: answer CE 0.000, background CE 7.433
- hymba-with-nsa-decor-gated_pi25_s1337: answer CE 24.561, background CE 8.790
- hymba-with-nsa-decor-gated_pi25_s2026: answer CE 21.864, background CE 8.784
- hymba-with-nsa-decor-gated_pi25_s42: answer CE 25.137, background CE 8.731
- hymba-with-nsa-decor_pi00_s1337: answer CE 0.000, background CE 7.330
- hymba-with-nsa-decor_pi00_s2026: answer CE 0.000, background CE 7.353
- hymba-with-nsa-decor_pi00_s42: answer CE 0.000, background CE 7.371
- hymba-with-nsa-decor_pi25_s1337: answer CE 23.515, background CE 8.842
- hymba-with-nsa-decor_pi25_s2026: answer CE 23.290, background CE 8.813
- hymba-with-nsa-decor_pi25_s42: answer CE 23.231, background CE 8.772
- hymba-with-nsa-gated_pi00_s1337: answer CE 0.000, background CE 7.337
- hymba-with-nsa-gated_pi00_s2026: answer CE 0.000, background CE 7.383
- hymba-with-nsa-gated_pi00_s42: answer CE 0.000, background CE 7.384
- hymba-with-nsa-gated_pi10_s1337: answer CE 22.436, background CE 7.802
- hymba-with-nsa-gated_pi10_s2026: answer CE 22.334, background CE 7.776
- hymba-with-nsa-gated_pi10_s42: answer CE 21.325, background CE 7.845
- hymba-with-nsa-gated_pi25_s1337: answer CE 26.302, background CE 8.723
- hymba-with-nsa-gated_pi25_s2026: answer CE 22.849, background CE 8.697
- hymba-with-nsa-gated_pi25_s42: answer CE 23.523, background CE 8.768
- hymba-with-nsa-gated_pi50_s1337: answer CE 25.448, background CE 10.405
- hymba-with-nsa-gated_pi50_s2026: answer CE 21.816, background CE 10.371
- hymba-with-nsa-gated_pi50_s42: answer CE 22.075, background CE 10.268
- hymba-with-nsa-pcgate_pi00_s1337: answer CE 0.000, background CE 7.317
- hymba-with-nsa-pcgate_pi00_s2026: answer CE 0.000, background CE 7.374
- hymba-with-nsa-pcgate_pi00_s42: answer CE 0.000, background CE 7.330
- hymba-with-nsa-pcgate_pi10_s1337: answer CE 23.312, background CE 7.870
- hymba-with-nsa-pcgate_pi10_s2026: answer CE 23.133, background CE 7.865
- hymba-with-nsa-pcgate_pi10_s42: answer CE 24.925, background CE 7.784
- hymba-with-nsa-pcgate_pi25_s1337: answer CE 22.406, background CE 8.714
- hymba-with-nsa-pcgate_pi25_s2026: answer CE 22.598, background CE 8.819
- hymba-with-nsa-pcgate_pi25_s42: answer CE 23.061, background CE 8.879
- hymba-with-nsa-pcgate_pi50_s1337: answer CE 27.558, background CE 10.951
- hymba-with-nsa-pcgate_pi50_s2026: answer CE 26.343, background CE 10.509
- hymba-with-nsa-pcgate_pi50_s42: answer CE 24.257, background CE 10.689
- hymba-with-nsa-randgate_pi00_s1337: answer CE 0.000, background CE 7.314
- hymba-with-nsa-randgate_pi00_s2026: answer CE 0.000, background CE 7.410
- hymba-with-nsa-randgate_pi00_s42: answer CE 0.000, background CE 7.301
- hymba-with-nsa-randgate_pi10_s1337: answer CE 23.642, background CE 7.862
- hymba-with-nsa-randgate_pi10_s2026: answer CE 21.538, background CE 7.766
- hymba-with-nsa-randgate_pi10_s42: answer CE 21.519, background CE 7.789
- hymba-with-nsa-randgate_pi25_s1337: answer CE 24.670, background CE 8.798
- hymba-with-nsa-randgate_pi25_s2026: answer CE 21.818, background CE 8.756
- hymba-with-nsa-randgate_pi25_s42: answer CE 21.832, background CE 8.810
- hymba-with-nsa-randgate_pi50_s1337: answer CE 25.730, background CE 10.415
- hymba-with-nsa-randgate_pi50_s2026: answer CE 23.550, background CE 10.604
- hymba-with-nsa-randgate_pi50_s42: answer CE 25.438, background CE 10.397
- hymba-with-nsa_pi00_s1337: answer CE 0.000, background CE 7.312
- hymba-with-nsa_pi00_s2026: answer CE 0.000, background CE 7.364
- hymba-with-nsa_pi00_s42: answer CE 0.000, background CE 7.329
- hymba-with-nsa_pi10_s1337: answer CE 23.021, background CE 7.790
- hymba-with-nsa_pi10_s2026: answer CE 21.794, background CE 7.820
- hymba-with-nsa_pi10_s42: answer CE 25.008, background CE 7.794
- hymba-with-nsa_pi25_s1337: answer CE 26.939, background CE 8.773
- hymba-with-nsa_pi25_s2026: answer CE 23.200, background CE 8.756
- hymba-with-nsa_pi25_s42: answer CE 22.867, background CE 8.723
- hymba-with-nsa_pi50_s1337: answer CE 22.260, background CE 10.944
- hymba-with-nsa_pi50_s2026: answer CE 25.802, background CE 10.502
- hymba-with-nsa_pi50_s42: answer CE 21.936, background CE 10.519

## P4 - decorrelation effect on ceiling (pi=0.25)

- gated: 0.1076 bits, decor-gated: 0.0942 bits -> not raised

## Decor coefficient sweep (pi=0.25, s42)

- coef 0.03: PPL 103.31
- coef 0.3: PPL 103.50

## Gate MI (non-circular) by pi

- pi=0.00 gated: 0.0034
- pi=0.00 randgate: 0.0031
- pi=0.10 gated: 0.0043
- pi=0.10 randgate: 0.0026
- pi=0.25 gated: 0.041
- pi=0.25 randgate: 0.0152
- pi=0.50 gated: 0.0331
- pi=0.50 randgate: 0.0222
