# Datasets and external assets

Large binary assets are not committed to Git. Code and small manifests live in
the repository; generated data lives under one external root.

## Canonical local layout

```text
/home/chi/UAV/quadpilot-data/
├── gsplat_sources/
│   ├── lemniscate/
│   └── uturn/
├── gsplat_outputs/
│   ├── lemniscate/{preflight,smoke-1,smoke-101,train-30k}/
│   └── uturn/{preflight,smoke-1,smoke-101,train-30k}/
├── npe_datasets/
│   ├── circle/{uniform_seed42_10000_v1,gate_seed4242_4000_v1}/
│   ├── lemniscate/{uniform_seed42_10000_v1,gate_seed4242_4000_v1,launch_corridor_seed31415_2000_v1}/
│   └── uturn/{uniform_seed42_10000_v1,gate_seed4242_4000_v1}/
├── npe_models/{circle,lemniscate,uturn}/
├── closed_loop/
└── ns114_cache/
```

The recovered Circle GSplat checkpoint is stored alongside the other external
assets at
`/home/chi/UAV/quadpilot-data/gsplat_outputs/circle/splatfacto/2025-05-09_144210`.
It came from the original untouched archive and is ignored by Git.

## Verified dataset inventory

| Track | Uniform | Gate focused | Additional | Render failures |
|:--|--:|--:|--:|--:|
| Circle | 10,000 | 4,000 | — | 0 |
| Lemniscate | 10,000 | 4,000 | 2,000 launch-corridor | 0 |
| U-turn | 10,000 | 4,000 | — | 0 |

Every completed dataset contains `metadata.json`, `samples.jsonl`,
`progress.json`, sequential RGB PNGs, and per-image SHA-256 receipts. Dataset
fingerprints and source-asset hashes are recorded in
`configs/assets/manifest.json`.

## Commands

```bash
quadpilot data download --help
quadpilot data generate uniform --help
quadpilot data generate gate --help
quadpilot verify dataset --help
```

Generation is fail-closed. A resume request must match the original sampler,
seed, bounds, intrinsics, target count, renderer checkpoint, transform, and code
identity. Orphaned or partially committed files are rejected instead of being
silently repaired.
