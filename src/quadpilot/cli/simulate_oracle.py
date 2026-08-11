#!/usr/bin/env python3
"""Run the visual planner/controller without requiring NeRF or NPE assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quadpilot.simulation.runner import run_oracle_simulation
from quadpilot.simulation.tracks import TRACKS

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asset-independent Quad Pilots visual-control baseline"
    )
    parser.add_argument("--track", choices=[*TRACKS, "all"], default="all")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--yaw-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--legacy-pass-radius",
        action="store_true",
        help="Use the submitted controller's permissive 1.5x gate threshold",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "repro_outputs" / "oracle"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tracks = TRACKS if args.track == "all" else {args.track: TRACKS[args.track]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_complete = True

    for name in tracks:
        result = run_oracle_simulation(
            name,
            max_steps=args.max_steps,
            pass_radius_multiplier=1.5 if args.legacy_pass_radius else 1.0,
            position_noise_std=args.position_noise_std,
            yaw_noise_std=args.yaw_noise_std,
            seed=args.seed,
        )
        metrics = result.evaluation.to_dict()
        metrics.update(
            {
                "controller_completed": result.completed_by_controller,
                "steps": result.steps,
                "duration_s": result.duration_s,
                "observation_position_noise_std_m": args.position_noise_std,
                "observation_yaw_noise_std_rad": args.yaw_noise_std,
                "seed": args.seed,
            }
        )
        np.savetxt(
            args.output_dir / f"{name}_trajectory.txt",
            result.states[:, [0, 1, 2, 6]],
            fmt="%.9f",
        )
        with (args.output_dir / f"{name}_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)

        mge = result.evaluation.mean_gate_error_m
        print(
            f"{name:11s} controller={result.completed_by_controller!s:5s} "
            f"evaluator={result.evaluation.completed!s:5s} "
            f"steps={result.steps:4d} "
            f"SR={100.0 * result.evaluation.success_rate:6.2f}% "
            f"MGE={(100.0 * mge if mge is not None else float('nan')):6.2f}cm"
        )
        all_complete &= result.completed_by_controller and result.evaluation.completed

    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
