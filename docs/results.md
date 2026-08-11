# Verified results

All model selection used frozen validation splits. Each final TEST split was
evaluated once and was not used for subsequent tuning.

## Closed-loop simulation

| Track | Estimator | Steps | Ordered gates | Mean gate error |
|:--|:--|--:|--:|--:|
| Circle | raw NPE | 368 | 8/8 | 14.80 cm |
| Circle | EKF | 355 | 8/8 | 8.25 cm |
| Lemniscate | raw NPE | 468 | 8/8 | 8.30 cm |
| Lemniscate | EKF | 459 | 8/8 | 5.64 cm |
| U-turn | raw NPE | 362 | 8/8 | 10.15 cm |
| U-turn | EKF | 347 | 8/8 | 5.16 cm |

## Published-video comparison

The project teaser reports Lemniscate NPE mean error 7.9 cm and EKF mean error
7.2 cm over its displayed 500-frame run. Matching the historical code's metric
window gives:

| Source | Mean | Std | Max | Jitter |
|:--|--:|--:|--:|--:|
| Reproduced NPE | 6.19 cm | 3.84 cm | 34.59 cm | 3.75 cm |
| Reproduced EKF | 4.83 cm | 2.36 cm | 9.76 cm | 1.23 cm |
| Reproduced dynamics | 8.31 cm | 2.63 cm | 14.35 cm | 3.48 cm |

EKF reduces reproduced NPE mean error by 21.98% and jitter by 67.27%. The NPE
maximum contains one retained outlier; EKF suppresses it and the strict gate
sequence still passes. The locked comparison is
`results/baselines/teaser_lemniscate_comparison.json`.

Exact checkpoints, dataset fingerprints, frozen TEST metrics, and artifact
SHA-256 values are recorded in `configs/assets/manifest.json` and the full
[reproduction log](reproduction.md).
