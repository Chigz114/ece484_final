"""Metric-compatible comparison with the Quad Pilots teaser summary slate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..control.dynamics import step_dynamics

TEASER_REFERENCE: dict[str, Any] = {
    "track": "lemniscate",
    "displayed_total_frames": 500,
    "sources": {
        "NPE": {"mean_cm": 7.9, "std_cm": 4.0, "max_cm": 25.3, "jitter_cm": 3.87},
        "EKF": {"mean_cm": 7.2, "std_cm": 3.7, "max_cm": 19.2, "jitter_cm": 1.37},
        "DYN": {"mean_cm": 8.3, "std_cm": 2.5, "max_cm": 14.2, "jitter_cm": 3.34},
        "GT": {"jitter_cm": 0.95},
    },
}


def _finite_array(name: str, value: np.ndarray, *, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def position_jitter_cm(poses: np.ndarray) -> float:
    """Legacy teaser jitter: std of consecutive 3-D displacement magnitudes."""

    xyz = _finite_array("poses", poses, columns=poses.shape[1])[:, :3]
    if len(xyz) < 2:
        return 0.0
    displacements_cm = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 100.0
    return float(np.std(displacements_cm))


def position_statistics(poses: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pose_array = np.asarray(poses, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if pose_array.ndim != 2 or truth_array.ndim != 2:
        raise ValueError("poses and truth must be two-dimensional")
    if pose_array.shape[0] != truth_array.shape[0] or pose_array.shape[1] < 3:
        raise ValueError(
            "poses and truth must have aligned rows and at least xyz columns"
        )
    if truth_array.shape[1] < 3:
        raise ValueError("truth must have at least xyz columns")
    if not np.isfinite(pose_array).all() or not np.isfinite(truth_array).all():
        raise ValueError("poses and truth must be finite")

    errors_cm = np.linalg.norm(pose_array[:, :3] - truth_array[:, :3], axis=1) * 100.0
    return {
        "mean_cm": float(np.mean(errors_cm)),
        "std_cm": float(np.std(errors_cm)),
        "max_cm": float(np.max(errors_cm)),
        "jitter_cm": position_jitter_cm(pose_array),
    }


def legacy_dyn_poses(
    states: np.ndarray,
    controls: np.ndarray,
    *,
    dt: float,
    seed: int,
    position_noise_m: float = 0.05,
    yaw_noise_rad: float = 0.05,
) -> np.ndarray:
    """Reproduce the teaser's historical one-step DYN construction.

    The historical script combined the current GT position/yaw with the
    previous GT velocity, advanced it using the previous control, then added
    uniform noise.  This is retained here as a comparison contract; it is not
    presented as an accumulated inertial dead-reckoning estimator.
    """

    state_array = _finite_array("states", states, columns=7)
    control_array = _finite_array("controls", controls, columns=4)
    if len(state_array) != len(control_array) + 1:
        raise ValueError("states must contain exactly one more row than controls")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if position_noise_m < 0 or yaw_noise_rad < 0:
        raise ValueError("noise bounds must be non-negative")

    rng = np.random.RandomState(seed)
    poses = [state_array[0, [0, 1, 2, 6]].copy()]
    for index in range(1, len(state_array)):
        historical_state = np.concatenate(
            (
                state_array[index, :3],
                state_array[index - 1, 3:6],
                state_array[index, 6:7],
            )
        )
        prediction = step_dynamics(historical_state, control_array[index - 1], dt=dt)
        position_noise = rng.uniform(-position_noise_m, position_noise_m, 3)
        yaw_noise = float(rng.uniform(-yaw_noise_rad, yaw_noise_rad))
        poses.append(
            np.asarray(
                [
                    *(prediction[:3] + position_noise),
                    prediction[6] + yaw_noise,
                ],
                dtype=np.float64,
            )
        )
    return np.asarray(poses, dtype=np.float64)


def compare_teaser_metrics(
    *,
    states: np.ndarray,
    observations: np.ndarray,
    estimated_states: np.ndarray,
    controls: np.ndarray,
    controller_pass_steps: Sequence[int],
    dt: float,
    dyn_seed: int,
) -> dict[str, Any]:
    """Compute the teaser-compatible clipped NPE/EKF/DYN/GT statistics."""

    state_array = _finite_array("states", states, columns=7)
    observation_array = _finite_array("observations", observations, columns=4)
    estimate_array = _finite_array("estimated_states", estimated_states, columns=7)
    control_array = _finite_array("controls", controls, columns=4)
    if len(observation_array) != len(state_array) or len(estimate_array) != len(
        state_array
    ):
        raise ValueError("states, observations, and estimates must have equal length")
    if len(control_array) + 1 != len(state_array):
        raise ValueError("controls must have one fewer row than states")

    pass_steps = [int(step) for step in controller_pass_steps]
    if len(pass_steps) < 2 or pass_steps != sorted(pass_steps):
        raise ValueError(
            "controller pass steps must contain at least two ordered values"
        )
    start, end = pass_steps[0], pass_steps[-1]
    if start < 0 or end > len(state_array) or start >= end:
        raise ValueError(
            "controller pass metric window is outside the saved trajectory"
        )

    dyn_poses = legacy_dyn_poses(
        state_array,
        control_array,
        dt=dt,
        seed=dyn_seed,
    )
    metric_slice = slice(start, end)
    truth_window = state_array[metric_slice]
    reproduced = {
        "NPE": position_statistics(observation_array[metric_slice], truth_window),
        "EKF": position_statistics(estimate_array[metric_slice], truth_window),
        "DYN": position_statistics(dyn_poses[metric_slice], truth_window),
        "GT": {"jitter_cm": position_jitter_cm(truth_window)},
    }

    deltas_percent: dict[str, dict[str, float]] = {}
    for source, metrics in reproduced.items():
        reference_metrics = TEASER_REFERENCE["sources"][source]
        deltas_percent[source] = {
            metric: 100.0
            * (value - reference_metrics[metric])
            / reference_metrics[metric]
            for metric, value in metrics.items()
        }

    npe_mean = reproduced["NPE"]["mean_cm"]
    ekf_mean = reproduced["EKF"]["mean_cm"]
    npe_jitter = reproduced["NPE"]["jitter_cm"]
    ekf_jitter = reproduced["EKF"]["jitter_cm"]
    return {
        "contract": {
            "metric_window": "Python range(first controller pass, last controller pass)",
            "jitter": "std of consecutive xyz displacement magnitudes in centimeters",
            "dyn": "historical current-position/previous-velocity one-step prediction with uniform noise",
            "dyn_seed": dyn_seed,
            "dt": dt,
        },
        "metric_window": {
            "start_step_inclusive": start,
            "end_step_exclusive": end,
            "samples": end - start,
            "saved_observations": len(state_array),
            "teaser_displayed_total_frames": TEASER_REFERENCE["displayed_total_frames"],
        },
        "reference": TEASER_REFERENCE["sources"],
        "reproduced": reproduced,
        "delta_percent_vs_teaser": deltas_percent,
        "ekf_vs_npe": {
            "position_mean_reduction_percent": 100.0 * (npe_mean - ekf_mean) / npe_mean,
            "jitter_reduction_percent": 100.0 * (npe_jitter - ekf_jitter) / npe_jitter,
        },
    }
