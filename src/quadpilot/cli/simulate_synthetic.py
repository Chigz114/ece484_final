#!/usr/bin/env python3
"""Stress-test the recovered control loop with synthetic NPE-like errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quadpilot.simulation.runner import SimulationResult, run_pose_simulation
from quadpilot.simulation.tracks import TRACKS

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic pose-noise and EKF Monte Carlo reproduction"
    )
    parser.add_argument("--track", choices=[*TRACKS, "all"], default="all")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--position-noise-std", type=float, default=0.05)
    parser.add_argument("--yaw-noise-std-deg", type=float, default=1.0)
    parser.add_argument("--crossing-hysteresis", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "repro_outputs" / "synthetic_pose" / "summary.json",
    )
    return parser.parse_args(argv)


def run_metrics(result: SimulationResult) -> dict[str, float | bool | int | None]:
    sample_count = min(len(result.states), len(result.estimated_states))
    truth = result.states[:sample_count, :3]
    estimate = result.estimated_states[:sample_count, :3]
    errors = np.linalg.norm(estimate - truth, axis=1)
    estimated_steps = np.linalg.norm(np.diff(estimate, axis=0), axis=1)
    residual_steps = np.linalg.norm(np.diff(estimate - truth, axis=0), axis=1)
    control_changes = np.linalg.norm(np.diff(result.controls, axis=0), axis=1)
    return {
        "completed": result.evaluation.completed,
        "controller_completed": result.completed_by_controller,
        "successful_crossings": result.evaluation.successful_crossings,
        "success_rate": result.evaluation.success_rate,
        "mean_gate_error_m": result.evaluation.mean_gate_error_m,
        "steps": result.steps,
        "pose_mean_euclidean_error_m": float(np.mean(errors)),
        "pose_p90_euclidean_error_m": float(np.percentile(errors, 90)),
        "legacy_motion_jitter_m": float(np.std(estimated_steps)),
        "residual_step_jitter_m": float(np.std(residual_steps)),
        "mean_control_change": (
            float(np.mean(control_changes)) if len(control_changes) else 0.0
        ),
    }


def aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = (
        "success_rate",
        "mean_gate_error_m",
        "steps",
        "pose_mean_euclidean_error_m",
        "pose_p90_euclidean_error_m",
        "legacy_motion_jitter_m",
        "residual_step_jitter_m",
        "mean_control_change",
    )
    summary: dict[str, object] = {
        "completed_runs": sum(bool(run["completed"]) for run in runs),
        "total_runs": len(runs),
        "completion_rate": sum(bool(run["completed"]) for run in runs) / len(runs),
    }
    for key in numeric_keys:
        values = [float(run[key]) for run in runs if run[key] is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.crossing_hysteresis < 0:
        raise ValueError("--crossing-hysteresis cannot be negative")

    tracks = list(TRACKS) if args.track == "all" else [args.track]
    yaw_noise_std = np.deg2rad(args.yaw_noise_std_deg)
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile": "synthetic independent Gaussian pose errors",
        "warning": (
            "This isolates estimator/control robustness; it is not a substitute "
            "for evaluating the recovered neural checkpoint on rendered images."
        ),
        "settings": {
            "seed_start": 0,
            "seed_count": args.seeds,
            "position_noise_std_per_axis_m": args.position_noise_std,
            "yaw_noise_std_deg": args.yaw_noise_std_deg,
            "crossing_hysteresis_m": args.crossing_hysteresis,
            "gate_radius_m": 0.38,
            "laps": 2,
            "dt_s": 0.05,
            "max_steps": args.max_steps,
        },
        "results": {},
    }

    all_ekf_complete = True
    for track in tracks:
        track_results: dict[str, object] = {}
        for estimator in ("raw", "ekf"):
            runs: list[dict[str, object]] = []
            for seed in range(args.seeds):
                result = run_pose_simulation(
                    track,
                    max_steps=args.max_steps,
                    estimator=estimator,
                    position_noise_std=args.position_noise_std,
                    yaw_noise_std=yaw_noise_std,
                    crossing_hysteresis_m=args.crossing_hysteresis,
                    seed=seed,
                )
                metrics = run_metrics(result)
                metrics["seed"] = seed
                runs.append(metrics)
            summary = aggregate(runs)
            track_results[estimator] = {"summary": summary, "runs": runs}
            print(
                f"{track:11s} {estimator:3s} "
                f"complete={summary['completed_runs']:>3}/{args.seeds:<3} "
                f"mean_SR={100 * float(summary['mean_success_rate']):6.2f}% "
                f"pose={100 * float(summary['mean_pose_mean_euclidean_error_m']):5.2f}cm "
                f"MGE={100 * float(summary['mean_mean_gate_error_m']):5.2f}cm"
            )
            if estimator == "ekf":
                all_ekf_complete &= summary["completed_runs"] == args.seeds
        payload["results"][track] = track_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}")
    return 0 if all_ekf_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
