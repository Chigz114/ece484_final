"""Deterministic simulation harness with an injectable pose observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..control.controller import LegacyVisionControlCore
from ..control.dynamics import step_dynamics
from ..estimation.ekf import PoseEKF
from .evaluation import EvaluationResult, evaluate_ordered_gates
from .tracks import TrackConfig, get_track


@dataclass(frozen=True)
class SimulationResult:
    track: str
    estimator: str
    dt: float
    states: np.ndarray
    observations: np.ndarray
    estimated_states: np.ndarray
    controls: np.ndarray
    completed_by_controller: bool
    controller_passes: tuple[tuple[int, str, float], ...]
    evaluation: EvaluationResult

    @property
    def steps(self) -> int:
        return len(self.controls)

    @property
    def duration_s(self) -> float:
        return self.steps * self.dt


def run_pose_simulation(
    track: str | TrackConfig,
    *,
    max_steps: int = 1200,
    dt: float = 0.05,
    gate_radius: float = 0.38,
    pass_radius_multiplier: float = 1.0,
    crossing_hysteresis_m: float = 0.0,
    position_noise_std: float = 0.0,
    yaw_noise_std: float = 0.0,
    seed: int = 0,
    estimator: str = "raw",
    ekf_outlier_threshold: float | None = 4.0,
    ekf_process_acceleration_std: float = 0.5,
    ekf_process_yaw_rate_std: float = 0.15,
) -> SimulationResult:
    """Run the visual core with a raw or EKF-filtered pose observation."""

    config = get_track(track) if isinstance(track, str) else track
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if position_noise_std < 0 or yaw_noise_std < 0:
        raise ValueError("noise standard deviations cannot be negative")
    if estimator not in {"raw", "ekf"}:
        raise ValueError("estimator must be 'raw' or 'ekf'")

    controller = LegacyVisionControlCore(
        config,
        dt=dt,
        gate_radius=gate_radius,
        pass_radius_multiplier=pass_radius_multiplier,
        crossing_hysteresis_m=crossing_hysteresis_m,
    )
    rng = np.random.default_rng(seed)
    state = np.asarray(config.initial_state, dtype=np.float64)
    states = [state.copy()]
    observations: list[np.ndarray] = []
    estimated_states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    previous_raw_position: np.ndarray | None = None
    pose_ekf = (
        PoseEKF(
            observation_position_std=max(position_noise_std, 1e-6),
            observation_yaw_std=max(yaw_noise_std, 1e-6),
            process_acceleration_std=ekf_process_acceleration_std,
            process_yaw_rate_std=ekf_process_yaw_rate_std,
        )
        if estimator == "ekf"
        else None
    )

    for _ in range(max_steps):
        observation = state[[0, 1, 2, 6]].copy()
        if position_noise_std:
            observation[:3] += rng.normal(0.0, position_noise_std, 3)
        if yaw_noise_std:
            observation[3] += rng.normal(0.0, yaw_noise_std)
            observation[3] = (observation[3] + np.pi) % (2.0 * np.pi) - np.pi

        observations.append(observation)

        if pose_ekf is not None:
            if pose_ekf.initialized:
                previous_control = controls[-1] if controls else np.zeros(4)
                pose_ekf.predict(previous_control, dt)
            update = pose_ekf.update(
                observation, outlier_threshold=ekf_outlier_threshold
            )
            estimate = update.state
            controller_observation = estimate[[0, 1, 2, 6]]
            command = controller.step(
                controller_observation,
                velocity_estimate=estimate[3:6],
            )
        else:
            if previous_raw_position is None:
                estimated_velocity = state[3:6].copy()
            else:
                estimated_velocity = (observation[:3] - previous_raw_position) / dt
            previous_raw_position = observation[:3].copy()
            estimate = np.concatenate(
                [observation[:3], estimated_velocity, observation[3:4]]
            )
            command = controller.step(observation, velocity_hint=state[3:6])
        estimated_states.append(estimate.copy())

        if command.completed:
            break
        controls.append(command.control)
        state = step_dynamics(state, command.control, dt=dt)
        states.append(state.copy())

    state_array = np.asarray(states)
    evaluation = evaluate_ordered_gates(
        state_array, config, dt=dt, laps=2, gate_radius=gate_radius
    )
    return SimulationResult(
        track=config.name,
        estimator=estimator,
        dt=dt,
        states=state_array,
        observations=np.asarray(observations),
        estimated_states=np.asarray(estimated_states),
        controls=np.asarray(controls),
        completed_by_controller=controller.completed,
        controller_passes=tuple(controller.pass_events),
        evaluation=evaluation,
    )


def run_oracle_simulation(
    track: str | TrackConfig,
    **kwargs: object,
) -> SimulationResult:
    """Backward-compatible exact-pose baseline."""

    if "estimator" in kwargs:
        raise TypeError("run_oracle_simulation always uses the raw estimator")
    return run_pose_simulation(track, estimator="raw", **kwargs)
