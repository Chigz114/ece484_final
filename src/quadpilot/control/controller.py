"""Asset-independent reproduction of the December visual control core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation.tracks import TrackConfig, get_track
from .trajectory import generate_gate_transition


@dataclass(frozen=True)
class ControlStep:
    control: np.ndarray
    completed: bool
    event: str | None
    target_gate: str
    lap_count: int
    crossing_error: float | None = None


class LegacyVisionControlCore:
    """Legacy planner/controller with pose estimation injected as an input.

    The original script coupled Torch model loading, gate mission state and
    trajectory control in one global function.  This class preserves the
    December 2025 control law while allowing oracle/noisy/NPE observations to
    be tested independently.
    """

    def __init__(
        self,
        track: str | TrackConfig,
        *,
        dt: float = 0.05,
        gate_radius: float = 0.38,
        pass_radius_multiplier: float = 1.0,
        crossing_hysteresis_m: float = 0.0,
        target_speed: float = 1.5,
        lookahead_distance: float = 0.6,
        yaw_lookahead_distance: float = 2.0,
        total_laps: int = 2,
    ) -> None:
        self.track = get_track(track) if isinstance(track, str) else track
        self.dt = float(dt)
        self.gate_radius = float(gate_radius)
        self.pass_radius_multiplier = float(pass_radius_multiplier)
        self.crossing_hysteresis_m = float(crossing_hysteresis_m)
        self.target_speed = float(target_speed)
        self.lookahead_distance = float(lookahead_distance)
        self.yaw_lookahead_distance = float(yaw_lookahead_distance)
        self.total_laps = int(total_laps)

        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.gate_radius <= 0 or self.pass_radius_multiplier <= 0:
            raise ValueError("gate radii must be positive")
        if self.crossing_hysteresis_m < 0:
            raise ValueError("crossing_hysteresis_m cannot be negative")
        if self.total_laps <= 0:
            raise ValueError("total_laps must be positive")

        self.kp_velocity = np.array([4.0, 4.0, 4.0], dtype=np.float64)
        self.kp_yaw = 2.5
        self.kd_yaw = 0.8
        self.max_acceleration = 5.0
        self.max_yaw_rate = 2.0
        self.yaw_filter_alpha = 0.3

        self._ordered_gates = self.track.ordered_gates()
        self._gate_normals = tuple(gate.normal for gate in self._ordered_gates)
        self._initial_gate_sides = [
            float(side) for side in self.track.incoming_gate_sides
        ]
        self.reset()

    def reset(self) -> None:
        self.target_gate_index = 0
        self.lap_count = 0
        self.completed = False
        self._gate_plane_sides = list(self._initial_gate_sides)
        self._current_trajectory: np.ndarray | None = None
        self._previous_observed_position: np.ndarray | None = None
        self._previous_yaw = 0.0
        self._filtered_yaw_rate = 0.0
        self._step_index = 0
        self.pass_events: list[tuple[int, str, float]] = []

    @property
    def current_trajectory(self) -> np.ndarray | None:
        if self._current_trajectory is None:
            return None
        return self._current_trajectory.copy()

    def _initial_trajectory(self, position: np.ndarray, yaw: float) -> np.ndarray:
        gate = self._ordered_gates[self.target_gate_index]
        vector = np.asarray(gate.center) - position
        distance = np.linalg.norm(vector)
        if distance > 0.1:
            start_direction = vector / distance
        else:
            start_direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return generate_gate_transition(
            position,
            start_direction,
            np.asarray(gate.center),
            self._gate_normals[self.target_gate_index],
            straight_dist=0.6,
            track=self.track.name,
        )

    def _transition_after_pass(self, previous_gate_index: int) -> np.ndarray:
        next_index = self.target_gate_index
        previous_gate = self._ordered_gates[previous_gate_index]
        next_gate = self._ordered_gates[next_index]
        is_lap_transition = (
            previous_gate_index == len(self._ordered_gates) - 1 and next_index == 0
        )
        return generate_gate_transition(
            np.asarray(previous_gate.center),
            self._gate_normals[previous_gate_index],
            np.asarray(next_gate.center),
            self._gate_normals[next_index],
            straight_dist=0.8,
            is_lap_transition=is_lap_transition,
            track=self.track.name,
        )

    def _estimate_velocity(
        self,
        position: np.ndarray,
        velocity_hint: np.ndarray | None,
        velocity_estimate: np.ndarray | None,
    ) -> np.ndarray:
        if velocity_estimate is not None:
            velocity = np.asarray(velocity_estimate, dtype=np.float64)
            if velocity.shape != (3,):
                raise ValueError("velocity_estimate must have shape (3,)")
        elif self._previous_observed_position is None:
            velocity = (
                np.zeros(3, dtype=np.float64)
                if velocity_hint is None
                else np.asarray(velocity_hint, dtype=np.float64)
            )
        else:
            velocity = (position - self._previous_observed_position) / self.dt
        self._previous_observed_position = position.copy()
        return velocity

    def _check_gate_crossing(
        self, position: np.ndarray
    ) -> tuple[str | None, float | None]:
        target_index = self.target_gate_index
        gate = self._ordered_gates[target_index]
        normal = self._gate_normals[target_index]
        previous_side = self._gate_plane_sides[target_index]

        relative = position - np.asarray(gate.center)
        distance_to_plane = float(np.dot(relative, normal))
        current_side = (
            previous_side
            if abs(distance_to_plane) < self.crossing_hysteresis_m
            else float(np.sign(distance_to_plane))
        )
        radial_error = float(np.linalg.norm(relative - distance_to_plane * normal))

        event: str | None = None
        crossing_error: float | None = None
        if current_side != previous_side and previous_side != 0.0:
            crossing_error = radial_error
            if radial_error <= self.gate_radius * self.pass_radius_multiplier:
                event = "pass"
                self.pass_events.append((self._step_index, gate.name, radial_error))
                previous_gate_index = target_index
                self._gate_plane_sides[target_index] = current_side
                self.target_gate_index += 1
                if self.target_gate_index >= len(self._ordered_gates):
                    self.target_gate_index = 0
                    self.lap_count += 1
                    if self.lap_count >= self.total_laps:
                        self.completed = True
                        return event, crossing_error
                    self._gate_plane_sides = list(self._initial_gate_sides)
                self._gate_plane_sides[self.target_gate_index] = (
                    self._initial_gate_sides[self.target_gate_index]
                )
                self._current_trajectory = self._transition_after_pass(
                    previous_gate_index
                )
            else:
                event = "miss"
                # Preserve the submitted controller's plane-side update.  A
                # miss therefore has to loop back before it can be re-armed.
                self._gate_plane_sides[target_index] = current_side
        elif current_side != 0.0:
            self._gate_plane_sides[target_index] = current_side
        return event, crossing_error

    def _trajectory_control(
        self, position: np.ndarray, velocity: np.ndarray, yaw: float
    ) -> np.ndarray:
        assert self._current_trajectory is not None
        distances = np.linalg.norm(self._current_trajectory - position, axis=1)
        closest_index = int(np.argmin(distances))

        carrot_position = self._current_trajectory[-1]
        for index in range(closest_index, len(self._current_trajectory)):
            chord = np.linalg.norm(
                self._current_trajectory[index]
                - self._current_trajectory[closest_index]
            )
            if chord >= self.lookahead_distance:
                carrot_position = self._current_trajectory[index]
                break

        direction = carrot_position - position
        distance = np.linalg.norm(direction)
        if distance > 0.01:
            direction /= distance
        else:
            direction[:] = 0.0
        target_velocity = direction * self.target_speed
        acceleration_world = self.kp_velocity * (target_velocity - velocity)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        acceleration_body = np.array(
            [
                acceleration_world[0] * cos_yaw + acceleration_world[1] * sin_yaw,
                -acceleration_world[0] * sin_yaw + acceleration_world[1] * cos_yaw,
                acceleration_world[2],
            ]
        )
        acceleration_body = np.clip(
            acceleration_body, -self.max_acceleration, self.max_acceleration
        )

        yaw_index = len(self._current_trajectory) - 1
        for index in range(closest_index, len(self._current_trajectory)):
            chord = np.linalg.norm(
                self._current_trajectory[index]
                - self._current_trajectory[closest_index]
            )
            if chord >= self.yaw_lookahead_distance:
                yaw_index = index
                break
        if yaw_index < len(self._current_trajectory) - 1:
            tangent = (
                self._current_trajectory[yaw_index + 1]
                - self._current_trajectory[yaw_index]
            )
        else:
            tangent = self._current_trajectory[-1] - self._current_trajectory[-2]

        desired_yaw = float(np.arctan2(tangent[1], tangent[0]))
        yaw_error = (desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
        yaw_rate_estimate = (yaw - self._previous_yaw) / self.dt
        raw_yaw_rate = self.kp_yaw * yaw_error - self.kd_yaw * yaw_rate_estimate
        self._filtered_yaw_rate = (
            self.yaw_filter_alpha * raw_yaw_rate
            + (1.0 - self.yaw_filter_alpha) * self._filtered_yaw_rate
        )
        yaw_rate = float(
            np.clip(self._filtered_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        )
        self._previous_yaw = yaw
        return np.concatenate([acceleration_body, np.array([yaw_rate])])

    def step(
        self,
        observation: np.ndarray,
        *,
        velocity_hint: np.ndarray | None = None,
        velocity_estimate: np.ndarray | None = None,
    ) -> ControlStep:
        """Compute one body-frame command from ``[x, y, z, yaw]``."""

        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (4,):
            raise ValueError(
                f"observation must have shape (4,), got {observation.shape}"
            )
        if self.completed:
            target = self._ordered_gates[self.target_gate_index].name
            return ControlStep(np.zeros(4), True, None, target, self.lap_count)

        position = observation[:3]
        yaw = float(observation[3])
        velocity = self._estimate_velocity(position, velocity_hint, velocity_estimate)
        if self._current_trajectory is None:
            self._current_trajectory = self._initial_trajectory(position, yaw)

        event, crossing_error = self._check_gate_crossing(position)
        if self.completed:
            target = self._ordered_gates[-1].name
            self._step_index += 1
            return ControlStep(
                np.zeros(4), True, event, target, self.lap_count, crossing_error
            )

        control = self._trajectory_control(position, velocity, yaw)
        target = self._ordered_gates[self.target_gate_index].name
        self._step_index += 1
        return ControlStep(
            control, False, event, target, self.lap_count, crossing_error
        )
