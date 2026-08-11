"""Auditable image-to-pose closed-loop simulation for Quad Pilots.

The truth state is used only to render the camera image and advance the plant.
The controller receives either raw NPE poses (with velocity obtained solely
from consecutive NPE positions) or a :class:`PoseEKF` state estimate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from PIL import Image

from .controller import LegacyVisionControlCore
from .data_generation import normalize_rgb, pose_to_camera_matrix
from .dynamics import step_dynamics
from .estimation import PoseEKF
from .evaluation import EvaluationResult, evaluate_ordered_gates
from .npe import atomic_json_save, predict_poses
from .tracks import TrackConfig, get_track


class RGBRenderer(Protocol):
    """Minimum renderer contract used by the closed loop."""

    def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray: ...


DecodedPosePredictor = Callable[[np.ndarray], np.ndarray]
ObservationProvider = Callable[[np.ndarray, int], np.ndarray]


def true_state_to_camera_matrix(state: np.ndarray) -> np.ndarray:
    """Convert truth ``[x,y,z,vx,vy,vz,yaw]`` to the legacy camera pose.

    The simulated vehicle carries a level camera, so roll and pitch remain
    zero.  The resulting 4x4 matrix is in the original NeRF world frame; the
    render adapter performs the subsequent Nerfstudio normalization.
    """

    state = np.asarray(state, dtype=np.float64)
    if state.shape != (7,):
        raise ValueError(f"state must have shape (7,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("truth state contains NaN or infinity")
    pose = np.array(
        [state[0], state[1], state[2], 0.0, 0.0, state[6]],
        dtype=np.float64,
    )
    return pose_to_camera_matrix(pose)


def _rgb_u8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"renderer returned shape {array.shape}; expected HxWx3 RGB"
        )
    return normalize_rgb(
        array,
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        minimum_dynamic_range=0,
    )


def render_true_state_rgb(
    renderer: RGBRenderer, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Render one truth state and return ``(RGB uint8, camera_to_world)``."""

    camera_to_world = true_state_to_camera_matrix(state)
    render_u8 = getattr(renderer, "render_rgb_u8", None)
    image = (
        render_u8(camera_to_world)
        if callable(render_u8)
        else renderer.render_rgb(camera_to_world)
    )
    return _rgb_u8(image), camera_to_world


class TorchNPEPredictor:
    """Single-image adapter around the canonical reproducible NPE inference."""

    def __init__(
        self,
        model: Any,
        normalizer: Any,
        image_transform: Callable[[Image.Image], Any],
        *,
        device: Any,
        amp_enabled: bool = False,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.image_transform = image_transform
        self.device = device
        self.amp_enabled = bool(amp_enabled)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(_rgb_u8(rgb), mode="RGB")
        tensor = self.image_transform(image)
        if getattr(tensor, "ndim", None) != 3:
            raise ValueError("NPE image transform must return a CHW tensor")
        prediction = predict_poses(
            self.model,
            tensor.unsqueeze(0),
            self.normalizer,
            device=self.device,
            amp_enabled=self.amp_enabled,
        )
        decoded = prediction.xyz_yaw.detach().cpu().numpy()
        if decoded.shape != (1, 4):
            raise ValueError(
                f"decoded NPE pose must have shape (1,4), got {decoded.shape}"
            )
        # predict_poses/decode_predictions guarantees these are physical NeRF
        # xyz coordinates, not normalized training targets.
        return decoded[0].astype(np.float64, copy=True)


def oracle_pose_observation(state: np.ndarray, _step_index: int) -> np.ndarray:
    """Injectable exact observation used only for CPU regression tests."""

    state = np.asarray(state, dtype=np.float64)
    return state[[0, 1, 2, 6]].copy()


def _validated_observation(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    observation = np.asarray(value, dtype=np.float64)
    if observation.shape == (1, 4):
        observation = observation[0]
    if observation.shape != (4,):
        raise ValueError(
            f"decoded NPE observation must have shape (4,), got {observation.shape}"
        )
    if not np.isfinite(observation).all():
        raise ValueError("decoded NPE observation contains NaN or infinity")
    observation = observation.copy()
    observation[3] = (observation[3] + np.pi) % (2.0 * np.pi) - np.pi
    return observation


def _rows(values: list[np.ndarray], width: int) -> np.ndarray:
    if not values:
        return np.empty((0, width), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape((-1, width))


def _camera_rows(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0, 4, 4), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape((-1, 4, 4))


def _summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "std": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
    }


def _pose_error_metrics(
    poses: np.ndarray, truth_states: np.ndarray
) -> dict[str, Any]:
    if len(poses) == 0:
        return {
            "samples": 0,
            "position_error_cm": _summary(np.empty(0)),
            "yaw_error_deg": _summary(np.empty(0)),
            "position_step_jitter_cm": None,
        }
    positions = poses[:, :3]
    truth_positions = truth_states[: len(poses), :3]
    position_error_cm = np.linalg.norm(positions - truth_positions, axis=1) * 100.0
    yaw = poses[:, -1]
    truth_yaw = truth_states[: len(poses), 6]
    yaw_delta = np.arctan2(np.sin(yaw - truth_yaw), np.cos(yaw - truth_yaw))
    yaw_error_deg = np.abs(yaw_delta) * (180.0 / np.pi)
    jitter = (
        float(np.std(np.linalg.norm(np.diff(positions, axis=0), axis=1)) * 100.0)
        if len(positions) >= 2
        else 0.0
    )
    return {
        "samples": int(len(poses)),
        "position_error_cm": _summary(position_error_cm),
        "yaw_error_deg": _summary(yaw_error_deg),
        "position_step_jitter_cm": jitter,
    }


@dataclass(frozen=True)
class VisualLoopResult:
    track: str
    estimator: str
    dt: float
    max_steps: int
    gate_radius: float
    crossing_hysteresis_m: float
    states: np.ndarray
    observations: np.ndarray
    estimated_states: np.ndarray
    controls: np.ndarray
    camera_to_world: np.ndarray
    completed_by_controller: bool
    controller_passes: tuple[tuple[int, str, float], ...]
    evaluation: EvaluationResult | None
    termination_reason: str
    failure_reason: str | None
    ekf_update_accepted: tuple[bool, ...]
    ekf_mahalanobis: tuple[float | None, ...]
    snapshot_paths: tuple[str, ...]

    @property
    def steps(self) -> int:
        """Number of controls actually applied to the truth plant."""

        return int(len(self.controls))

    @property
    def duration_s(self) -> float:
        return self.steps * self.dt

    @property
    def succeeded(self) -> bool:
        return bool(
            self.failure_reason is None
            and self.completed_by_controller
            and self.evaluation is not None
            and self.evaluation.completed
        )

    def validate_alignment(self) -> None:
        """Validate the discrete-time contract of every stored array.

        ``states[k]`` is :math:`s_k`, ``observations[k]`` and
        ``estimated_states[k]`` are measurements/estimates at the same time,
        and ``controls[k]`` is the already-validated command applied over
        :math:`[k\,dt,(k+1)\,dt]` to produce ``states[k+1]``.
        """

        expected_shapes = {
            "states": (7,),
            "observations": (4,),
            "estimated_states": (7,),
            "controls": (4,),
            "camera_to_world": (4, 4),
        }
        for name, tail_shape in expected_shapes.items():
            array = np.asarray(getattr(self, name))
            if array.ndim != len(tail_shape) + 1 or array.shape[1:] != tail_shape:
                raise ValueError(
                    f"{name} must have shape (N,{','.join(map(str, tail_shape))})"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if len(self.states) != len(self.controls) + 1:
            raise ValueError("states must contain exactly one more row than controls")
        if len(self.estimated_states) > len(self.observations):
            raise ValueError("an estimate cannot exist without its observation")
        if len(self.observations) > len(self.states):
            raise ValueError("an observation cannot be later than the final truth state")
        if len(self.camera_to_world) not in {0, len(self.observations)}:
            raise ValueError(
                "camera poses must be absent for injected observations or align one-to-one"
            )
        if len(self.ekf_update_accepted) != len(self.ekf_mahalanobis):
            raise ValueError("EKF acceptance and Mahalanobis arrays must align")
        if self.estimator == "raw" and self.ekf_update_accepted:
            raise ValueError("raw estimator results cannot contain EKF updates")
        if self.estimator == "ekf" and len(self.ekf_update_accepted) != len(
            self.estimated_states
        ):
            raise ValueError("EKF diagnostics must align one-to-one with estimates")

        control_count = len(self.controls)
        state_count = len(self.states)
        observation_count = len(self.observations)
        estimate_count = len(self.estimated_states)
        expected_counts = {
            "max_steps": (control_count, control_count),
            "observation_failure": (control_count, control_count),
            "estimation_failure": (state_count, control_count),
            "controller_failure": (state_count, state_count),
            "dynamics_failure": (state_count, state_count),
            "controller_complete": (state_count, state_count),
        }
        if self.termination_reason not in expected_counts:
            raise ValueError(f"unknown termination reason: {self.termination_reason}")
        expected_observations, expected_estimates = expected_counts[
            self.termination_reason
        ]
        if observation_count != expected_observations:
            raise ValueError(
                f"{self.termination_reason} requires {expected_observations} "
                f"observations, got {observation_count}"
            )
        if estimate_count != expected_estimates:
            raise ValueError(
                f"{self.termination_reason} requires {expected_estimates} "
                f"estimates, got {estimate_count}"
            )

    def sample_alignment_dict(self) -> dict[str, Any]:
        """Return explicit step/time axes so terminal samples are unambiguous."""

        self.validate_alignment()

        def axis(count: int) -> dict[str, Any]:
            steps = np.arange(count, dtype=np.int64)
            return {
                "step_indices": steps.tolist(),
                "times_s": (steps.astype(np.float64) * self.dt).tolist(),
            }

        return {
            "contract": (
                "z[k] and xhat[k] observe s[k]; validated u[k] advances "
                "s[k] to s[k+1]"
            ),
            "states": axis(len(self.states)),
            "observations": axis(len(self.observations)),
            "estimated_states": axis(len(self.estimated_states)),
            "controls": axis(len(self.controls)),
            "camera_to_world": axis(len(self.camera_to_world)),
        }

    def metrics_dict(self) -> dict[str, Any]:
        self.validate_alignment()
        truth_at_observation = self.states[: len(self.observations)]
        estimate_poses = (
            self.estimated_states[:, [0, 1, 2, 6]]
            if len(self.estimated_states)
            else np.empty((0, 4), dtype=np.float64)
        )
        truth_jitter = (
            float(
                np.std(
                    np.linalg.norm(
                        np.diff(truth_at_observation[:, :3], axis=0), axis=1
                    )
                )
                * 100.0
            )
            if len(truth_at_observation) >= 2
            else 0.0
        )
        return {
            "track": self.track,
            "estimator": self.estimator,
            "succeeded": self.succeeded,
            "termination_reason": self.termination_reason,
            "failure_reason": self.failure_reason,
            "steps": self.steps,
            "duration_s": self.duration_s,
            "dt": self.dt,
            "max_steps": self.max_steps,
            "gate_radius_m": self.gate_radius,
            "crossing_hysteresis_m": self.crossing_hysteresis_m,
            "controller_completed": self.completed_by_controller,
            "controller_passes": [
                {"step": int(step), "gate": gate, "radial_error_m": float(error)}
                for step, gate, error in self.controller_passes
            ],
            "strict_evaluation": (
                self.evaluation.to_dict() if self.evaluation is not None else None
            ),
            "raw_npe": _pose_error_metrics(
                self.observations, truth_at_observation
            ),
            "controller_estimate": _pose_error_metrics(
                estimate_poses, truth_at_observation
            ),
            "truth_position_step_jitter_cm": truth_jitter,
            "ekf_updates_accepted": int(sum(self.ekf_update_accepted)),
            "ekf_updates_rejected": int(
                len(self.ekf_update_accepted) - sum(self.ekf_update_accepted)
            ),
            "rendered_observations": int(len(self.camera_to_world)),
            "snapshots_written": int(len(self.snapshot_paths)),
            "sample_counts": {
                "states": int(len(self.states)),
                "observations": int(len(self.observations)),
                "estimated_states": int(len(self.estimated_states)),
                "controls": int(len(self.controls)),
                "camera_to_world": int(len(self.camera_to_world)),
            },
        }

    def to_json_dict(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "metadata": dict(metadata or {}),
            "metrics": self.metrics_dict(),
            "sample_alignment": self.sample_alignment_dict(),
            "states": self.states.tolist(),
            "observations": self.observations.tolist(),
            "estimated_states": self.estimated_states.tolist(),
            "controls": self.controls.tolist(),
            "camera_to_world": self.camera_to_world.tolist(),
            "ekf_update_accepted": list(self.ekf_update_accepted),
            "ekf_mahalanobis": list(self.ekf_mahalanobis),
            "snapshot_paths": list(self.snapshot_paths),
        }


def run_visual_closed_loop(
    track: str | TrackConfig,
    *,
    renderer: RGBRenderer | None = None,
    predictor: DecodedPosePredictor | None = None,
    observation_provider: ObservationProvider | None = None,
    estimator: str = "raw",
    max_steps: int = 1200,
    dt: float = 0.05,
    gate_radius: float = 0.38,
    pass_radius_multiplier: float = 1.0,
    crossing_hysteresis_m: float = 0.05,
    laps: int = 2,
    ekf: PoseEKF | None = None,
    ekf_outlier_threshold: float | None = 4.0,
    ekf_observation_position_std: float = 0.05,
    ekf_observation_yaw_std: float = np.deg2rad(1.0),
    ekf_process_acceleration_std: float = 0.5,
    ekf_process_yaw_rate_std: float = 0.15,
    snapshot_every: int = 0,
    snapshot_dir: str | Path | None = None,
) -> VisualLoopResult:
    """Run one independent raw-NPE or EKF-NPE feedback experiment.

    Supplying ``observation_provider`` bypasses rendering/inference and is
    intended for deterministic tests.  Otherwise both ``renderer`` and
    ``predictor`` are required.  Observation, estimation, controller, and
    plant failures stop before another control is applied.
    """

    config = get_track(track) if isinstance(track, str) else track
    if estimator not in {"raw", "ekf"}:
        raise ValueError("estimator must be 'raw' or 'ekf'")
    if max_steps <= 0 or dt <= 0 or gate_radius <= 0 or laps <= 0:
        raise ValueError("max_steps, dt, gate_radius, and laps must be positive")
    if crossing_hysteresis_m < 0:
        raise ValueError("crossing_hysteresis_m cannot be negative")
    if snapshot_every < 0:
        raise ValueError("snapshot_every cannot be negative")
    if snapshot_every and snapshot_dir is None:
        raise ValueError("snapshot_dir is required when snapshot_every is nonzero")
    if observation_provider is None and (renderer is None or predictor is None):
        raise ValueError(
            "renderer and predictor are required without an observation_provider"
        )

    snapshots = Path(snapshot_dir).resolve() if snapshot_dir is not None else None
    if snapshot_every and snapshots is not None:
        snapshots.mkdir(parents=True, exist_ok=True)

    controller = LegacyVisionControlCore(
        config,
        dt=dt,
        gate_radius=gate_radius,
        pass_radius_multiplier=pass_radius_multiplier,
        crossing_hysteresis_m=crossing_hysteresis_m,
        total_laps=laps,
    )
    pose_ekf = ekf
    if estimator == "ekf" and pose_ekf is None:
        pose_ekf = PoseEKF(
            observation_position_std=ekf_observation_position_std,
            observation_yaw_std=ekf_observation_yaw_std,
            process_acceleration_std=ekf_process_acceleration_std,
            process_yaw_rate_std=ekf_process_yaw_rate_std,
        )

    state = np.asarray(config.initial_state, dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("track initial_state must contain seven finite values")
    states = [state.copy()]
    observations: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    camera_matrices: list[np.ndarray] = []
    accepted_updates: list[bool] = []
    mahalanobis_values: list[float | None] = []
    saved_snapshots: list[str] = []
    previous_raw_position: np.ndarray | None = None
    termination_reason = "max_steps"
    failure_reason: str | None = None

    for step_index in range(max_steps):
        rgb: np.ndarray | None = None
        camera_to_world: np.ndarray | None = None
        try:
            if observation_provider is not None:
                candidate = observation_provider(state.copy(), step_index)
            else:
                assert renderer is not None and predictor is not None
                rgb, camera_to_world = render_true_state_rgb(renderer, state)
                candidate = predictor(rgb)
            observation = _validated_observation(candidate)
            if snapshot_every and step_index % snapshot_every == 0:
                if rgb is None or snapshots is None:
                    raise RuntimeError(
                        "snapshots require the real renderer/predictor observation path"
                    )
                snapshot_path = snapshots / f"frame_{step_index:06d}.png"
                Image.fromarray(rgb, mode="RGB").save(snapshot_path)
                saved_snapshots.append(str(snapshot_path))
        except Exception as exc:  # Stop before control: fail closed.
            termination_reason = "observation_failure"
            failure_reason = f"{type(exc).__name__}: {exc}"
            break

        observations.append(observation.copy())
        if camera_to_world is not None:
            camera_matrices.append(camera_to_world.copy())

        ekf_accepted: bool | None = None
        ekf_mahalanobis: float | None = None
        try:
            if estimator == "raw":
                # Never use state[3:6].  Even the first raw estimate starts at
                # zero velocity because an image alone contains no velocity.
                raw_velocity = (
                    np.zeros(3, dtype=np.float64)
                    if previous_raw_position is None
                    else (observation[:3] - previous_raw_position) / dt
                )
                previous_raw_position = observation[:3].copy()
                estimate = np.concatenate(
                    [observation[:3], raw_velocity, observation[3:4]]
                )
            else:
                assert pose_ekf is not None
                # Control u[k-1] advances the estimator from k-1 to k before
                # observation z[k] is fused.  No current/future command leaks in.
                if controls:
                    pose_ekf.predict(controls[-1], dt)
                update = pose_ekf.update(
                    observation, outlier_threshold=ekf_outlier_threshold
                )
                estimate = np.asarray(update.state, dtype=np.float64)
                ekf_accepted = bool(update.accepted)
                ekf_mahalanobis = (
                    None
                    if update.mahalanobis_distance is None
                    else float(update.mahalanobis_distance)
                )
                if ekf_mahalanobis is not None and not np.isfinite(
                    ekf_mahalanobis
                ):
                    raise ValueError("EKF returned a non-finite Mahalanobis distance")
            if estimate.shape != (7,) or not np.isfinite(estimate).all():
                raise ValueError("estimator returned a non-finite or malformed state")
        except Exception as exc:
            termination_reason = "estimation_failure"
            failure_reason = f"{type(exc).__name__}: {exc}"
            break

        estimates.append(estimate.copy())
        if estimator == "ekf":
            assert ekf_accepted is not None
            accepted_updates.append(ekf_accepted)
            mahalanobis_values.append(ekf_mahalanobis)
        try:
            command = controller.step(
                estimate[[0, 1, 2, 6]],
                velocity_estimate=estimate[3:6],
            )
            if command.completed:
                termination_reason = "controller_complete"
                break
            control = np.asarray(command.control, dtype=np.float64)
            if control.shape != (4,) or not np.isfinite(control).all():
                raise ValueError("controller returned a non-finite or malformed control")
            tolerance = 1e-12
            if (
                np.any(np.abs(control[:3]) > controller.max_acceleration + tolerance)
                or abs(float(control[3])) > controller.max_yaw_rate + tolerance
            ):
                raise ValueError("controller returned a command outside configured limits")
        except Exception as exc:
            termination_reason = "controller_failure"
            failure_reason = f"{type(exc).__name__}: {exc}"
            break

        try:
            next_state = step_dynamics(state, control, dt=dt)
            if next_state.shape != (7,) or not np.isfinite(next_state).all():
                raise ValueError("dynamics returned a non-finite or malformed state")
        except Exception as exc:
            termination_reason = "dynamics_failure"
            failure_reason = f"{type(exc).__name__}: {exc}"
            break
        controls.append(control.copy())
        state = next_state
        states.append(state.copy())

    state_array = _rows(states, 7)
    evaluation = (
        evaluate_ordered_gates(
            state_array,
            config,
            dt=dt,
            laps=laps,
            gate_radius=gate_radius,
        )
        if len(state_array) >= 2
        else None
    )
    result = VisualLoopResult(
        track=config.name,
        estimator=estimator,
        dt=float(dt),
        max_steps=int(max_steps),
        gate_radius=float(gate_radius),
        crossing_hysteresis_m=float(crossing_hysteresis_m),
        states=state_array,
        observations=_rows(observations, 4),
        estimated_states=_rows(estimates, 7),
        controls=_rows(controls, 4),
        camera_to_world=_camera_rows(camera_matrices),
        completed_by_controller=controller.completed,
        controller_passes=tuple(controller.pass_events),
        evaluation=evaluation,
        termination_reason=termination_reason,
        failure_reason=failure_reason,
        ekf_update_accepted=tuple(accepted_updates),
        ekf_mahalanobis=tuple(mahalanobis_values),
        snapshot_paths=tuple(saved_snapshots),
    )
    result.validate_alignment()
    return result


def save_visual_loop_result(
    result: VisualLoopResult,
    output_dir: str | Path,
    *,
    run_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Atomically save full JSON plus numeric compressed NPZ artifacts."""

    result.validate_alignment()
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = run_name or f"{result.track}_{result.estimator}"
    json_path = directory / f"{stem}.json"
    npz_path = directory / f"{stem}.npz"
    temporary_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    mahalanobis = np.asarray(
        [np.nan if value is None else value for value in result.ekf_mahalanobis],
        dtype=np.float64,
    )
    state_steps = np.arange(len(result.states), dtype=np.int64)
    observation_steps = np.arange(len(result.observations), dtype=np.int64)
    estimate_steps = np.arange(len(result.estimated_states), dtype=np.int64)
    control_steps = np.arange(len(result.controls), dtype=np.int64)
    camera_steps = np.arange(len(result.camera_to_world), dtype=np.int64)
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            states=result.states,
            observations=result.observations,
            estimated_states=result.estimated_states,
            controls=result.controls,
            camera_to_world=result.camera_to_world,
            ekf_update_accepted=np.asarray(
                result.ekf_update_accepted, dtype=np.bool_
            ),
            ekf_mahalanobis=mahalanobis,
            state_step_indices=state_steps,
            state_times_s=state_steps.astype(np.float64) * result.dt,
            observation_step_indices=observation_steps,
            observation_times_s=observation_steps.astype(np.float64) * result.dt,
            estimate_step_indices=estimate_steps,
            estimate_times_s=estimate_steps.astype(np.float64) * result.dt,
            control_step_indices=control_steps,
            control_times_s=control_steps.astype(np.float64) * result.dt,
            camera_step_indices=camera_steps,
            camera_times_s=camera_steps.astype(np.float64) * result.dt,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_npz, npz_path)
    payload = result.to_json_dict(metadata)
    payload["artifacts"] = {"npz": str(npz_path), "json": str(json_path)}
    atomic_json_save(json_path, payload)
    return json_path, npz_path
