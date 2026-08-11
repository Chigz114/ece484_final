# QuadPilot ECE484

Reproducible vision-based quadrotor racing with Gaussian Splatting, a ResNet50
Neural Pose Estimator (NPE), EKF fusion, and closed-loop gate tracking.

The recovered simulation pipeline is complete for Circle, Lemniscate, and
U-turn. Each track passes two ordered laps (8/8 gates) with locked datasets,
checkpoints, results, and SHA-256 provenance. Physical Crazyflie deployment is
kept behind an offline, fail-closed readiness check until calibration and
prop-off safety evidence are supplied.

## Project layout

```text
ece484_final/
├── src/quadpilot/          # Installable control, perception, simulation, and CLI package
├── configs/                # Asset manifest, environment pins, and hardware templates
├── scripts/                # Environment bootstrap and pinned GSplat wrapper
├── tests/                  # Unit, integration, and locked regression tests
├── docs/                   # Architecture, data, results, and safety documentation
├── results/baselines/      # Small committed reference results (never large model files)
├── pyproject.toml          # Package metadata and dependencies
└── README.md
```

Large datasets, GSplat runs, NPE checkpoints, and closed-loop outputs stay
outside Git. The verified local data root is `/home/chi/UAV/quadpilot-data` in
WSL; see [datasets](docs/datasets.md) for the canonical layout.

## Quick start

Python 3.10 is the reference version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
quadpilot --help
quadpilot simulate oracle --track all --max-steps 1200
```

For NPE training, install the ML extra in the CUDA-enabled WSL environment:

```bash
python -m pip install -e '.[ml]'
quadpilot train npe --help
quadpilot evaluate npe --help
quadpilot simulate closed-loop --help
```

Nerfstudio/gsplat uses the separately pinned environment described in
`configs/environments/`; it is intentionally not mixed into the NPE environment.

## Main commands

```text
quadpilot data download ...
quadpilot data generate uniform ...
quadpilot data generate gate ...
quadpilot train npe ...
quadpilot evaluate npe ...
quadpilot simulate closed-loop ...
quadpilot verify assets ...
quadpilot verify dataset ...
quadpilot verify closed-loop ...
quadpilot hardware calibrate ...
quadpilot hardware preflight ...
```

Every subcommand supports `--help`. Expensive or hardware-facing work is never
started by the package simply because it is installed.

## Verified simulation results

| Track | NPE training | Frozen evaluation | Closed loop |
|:--|:--|:--|:--|
| Circle | base + gate fine-tune | PASS | raw 8/8, EKF 8/8 |
| Lemniscate | base + gate + launch-corridor fine-tune | PASS | raw 8/8, EKF 8/8 |
| U-turn | base + gate fine-tune | PASS | raw 8/8, EKF 8/8 |

The locked Lemniscate teaser comparison gives NPE mean error 6.19 cm and EKF
mean error 4.83 cm over the matching metric window. EKF reduces NPE mean error
by 21.98% and jitter by 67.27%; this is within the intended reproduction target
of the published project video.

Detailed commands, hashes, caveats, and frozen TEST policy are in the
[reproduction log](docs/reproduction.md). A compact numerical summary is in
[results](docs/results.md).

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
ruff check src tests
```

Current locked CPU suite: **151 passed, 1 opt-in Docker diagnostic skipped**.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproduction procedure and full evidence](docs/reproduction.md)
- [Dataset and external-asset layout](docs/datasets.md)
- [Coordinate frames](docs/coordinate_frames.md)
- [Verified results](docs/results.md)
- [Hardware safety boundary](docs/hardware_safety.md)

The original ECE484 submission remains recoverable from Git history at commit
`f0232f67`; it is not mixed into the maintained runtime tree.
