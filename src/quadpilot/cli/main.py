"""Single, discoverable command-line entry point for the project."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

COMMANDS: dict[tuple[str, ...], tuple[str, str]] = {
    ("data", "download"): (
        "quadpilot.cli.data_download",
        "download verified GSplat source data",
    ),
    ("data", "generate", "uniform"): (
        "quadpilot.cli.data_generate_uniform",
        "render a deterministic uniform NPE dataset",
    ),
    ("data", "generate", "gate"): (
        "quadpilot.cli.data_generate_gate",
        "render a gate-focused NPE dataset",
    ),
    ("train", "npe"): (
        "quadpilot.cli.train_npe",
        "train or fine-tune the neural pose estimator",
    ),
    ("evaluate", "npe"): ("quadpilot.cli.evaluate_npe", "evaluate a frozen NPE split"),
    ("simulate", "closed-loop"): (
        "quadpilot.cli.simulate_closed_loop",
        "run the GSplat/NPE visual closed loop",
    ),
    ("simulate", "oracle"): (
        "quadpilot.cli.simulate_oracle",
        "run the deterministic oracle baseline",
    ),
    ("simulate", "synthetic"): (
        "quadpilot.cli.simulate_synthetic",
        "run the synthetic pose-noise baseline",
    ),
    ("render", "smoke"): (
        "quadpilot.cli.render_smoke",
        "smoke-test a recovered renderer",
    ),
    ("verify", "assets"): (
        "quadpilot.cli.verify_assets",
        "verify renderer assets against the manifest",
    ),
    ("verify", "dataset"): (
        "quadpilot.cli.verify_dataset",
        "verify an NPE dataset and its receipts",
    ),
    ("verify", "gsplat"): (
        "quadpilot.cli.verify_gsplat",
        "verify a GSplat training run",
    ),
    ("verify", "closed-loop"): (
        "quadpilot.cli.verify_closed_loop",
        "verify locked closed-loop artifacts",
    ),
    ("hardware", "calibrate"): (
        "quadpilot.cli.hardware_calibrate",
        "fit the Vicon-to-NeRF calibration from offline evidence",
    ),
    ("hardware", "preflight"): (
        "quadpilot.cli.hardware_preflight",
        "run the offline, fail-closed hardware readiness check",
    ),
    ("compare", "teaser"): (
        "quadpilot.cli.compare_teaser",
        "compare a Lemniscate run with the published teaser metrics",
    ),
}


def _usage() -> str:
    rows = [
        "QuadPilot reproducible visual-control pipeline",
        "",
        "Usage: quadpilot GROUP COMMAND [ARGS]",
        "",
        "Commands:",
    ]
    width = max(len(" ".join(command)) for command in COMMANDS)
    for command, (_, description) in sorted(COMMANDS.items()):
        rows.append(f"  {' '.join(command):<{width}}  {description}")
    rows.extend(
        ["", "Run `quadpilot GROUP COMMAND --help` for command-specific options."]
    )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["-h"], ["--help"], ["help"]):
        print(_usage())
        return 0

    for command in sorted(COMMANDS, key=len, reverse=True):
        if tuple(arguments[: len(command)]) != command:
            continue
        module_name, _ = COMMANDS[command]
        module = importlib.import_module(module_name)
        return int(module.main(arguments[len(command) :]))

    print(f"Unknown command: {' '.join(arguments)}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
