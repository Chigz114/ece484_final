"""Deterministic pose EKF with the project's explicit frame contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import step_dynamics


@dataclass(frozen=True)
class EKFUpdate:
    state: np.ndarray
    accepted: bool
    mahalanobis_distance: float | None


class PoseEKF:
    """Fuse `[x, y, z, yaw]` observations with body-frame controls.

    This is a corrected, testable form of the submitted `DroneEKF`.  The plant
    and filter both use the same semi-implicit integrator, the Jacobian is
    evaluated at the prior state, covariance is scaled by `dt`, and simulated
    disturbances are kept outside the estimator.
    """

    def __init__(
        self,
        *,
        observation_position_std: float = 0.05,
        observation_yaw_std: float = np.deg2rad(1.0),
        process_acceleration_std: float = 0.5,
        process_yaw_rate_std: float = 0.15,
        initial_position_std: float = 0.10,
        initial_velocity_std: float = 0.50,
        initial_yaw_std: float = np.deg2rad(5.0),
    ) -> None:
        values = (
            observation_position_std,
            observation_yaw_std,
            process_acceleration_std,
            process_yaw_rate_std,
            initial_position_std,
            initial_velocity_std,
            initial_yaw_std,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all EKF standard deviations must be positive")

        self.observation_position_std = float(observation_position_std)
        self.observation_yaw_std = float(observation_yaw_std)
        self.process_acceleration_std = float(process_acceleration_std)
        self.process_yaw_rate_std = float(process_yaw_rate_std)
        self.initial_covariance = np.diag(
            [
                initial_position_std**2,
                initial_position_std**2,
                initial_position_std**2,
                initial_velocity_std**2,
                initial_velocity_std**2,
                initial_velocity_std**2,
                initial_yaw_std**2,
            ]
        )
        self.observation_matrix = np.zeros((4, 7), dtype=np.float64)
        self.observation_matrix[0, 0] = 1.0
        self.observation_matrix[1, 1] = 1.0
        self.observation_matrix[2, 2] = 1.0
        self.observation_matrix[3, 6] = 1.0
        self.observation_covariance = np.diag(
            [
                observation_position_std**2,
                observation_position_std**2,
                observation_position_std**2,
                observation_yaw_std**2,
            ]
        )
        self.state = np.zeros(7, dtype=np.float64)
        self.covariance = self.initial_covariance.copy()
        self.initialized = False

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def initialize(
        self, observation: np.ndarray, velocity: np.ndarray | None = None
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (4,):
            raise ValueError("observation must have shape (4,)")
        initial_velocity = (
            np.zeros(3, dtype=np.float64)
            if velocity is None
            else np.asarray(velocity, dtype=np.float64)
        )
        if initial_velocity.shape != (3,):
            raise ValueError("velocity must have shape (3,)")
        self.state = np.concatenate(
            [observation[:3], initial_velocity, [self.wrap_angle(observation[3])]]
        )
        self.covariance = self.initial_covariance.copy()
        self.initialized = True
        return self.state.copy()

    @staticmethod
    def _transition_jacobian(
        state: np.ndarray, control: np.ndarray, dt: float
    ) -> np.ndarray:
        yaw = float(state[6])
        ax_body, ay_body = np.asarray(control, dtype=np.float64)[:2]
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        dax_dyaw = -ax_body * sin_yaw - ay_body * cos_yaw
        day_dyaw = ax_body * cos_yaw - ay_body * sin_yaw

        jacobian = np.eye(7, dtype=np.float64)
        jacobian[0, 3] = dt
        jacobian[1, 4] = dt
        jacobian[2, 5] = dt
        jacobian[0, 6] = dax_dyaw * dt**2
        jacobian[1, 6] = day_dyaw * dt**2
        jacobian[3, 6] = dax_dyaw * dt
        jacobian[4, 6] = day_dyaw * dt
        return jacobian

    def _process_covariance(self, yaw: float, dt: float) -> np.ndarray:
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        body_to_world = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        noise_map = np.zeros((7, 4), dtype=np.float64)
        noise_map[:3, :3] = body_to_world * dt**2
        noise_map[3:6, :3] = body_to_world * dt
        noise_map[6, 3] = dt
        spectral = np.diag(
            [
                self.process_acceleration_std**2,
                self.process_acceleration_std**2,
                self.process_acceleration_std**2,
                self.process_yaw_rate_std**2,
            ]
        )
        return noise_map @ spectral @ noise_map.T

    def predict(self, control: np.ndarray, dt: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("EKF must be initialized before predict")
        control = np.asarray(control, dtype=np.float64)
        if control.shape != (4,):
            raise ValueError("control must have shape (4,)")
        if dt <= 0:
            raise ValueError("dt must be positive")

        prior_state = self.state.copy()
        transition = self._transition_jacobian(prior_state, control, dt)
        process = self._process_covariance(prior_state[6], dt)
        self.state = step_dynamics(prior_state, control, dt)
        self.covariance = (
            transition @ self.covariance @ transition.T + process
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return self.state.copy()

    def update(
        self,
        observation: np.ndarray,
        *,
        outlier_threshold: float | None = 4.0,
    ) -> EKFUpdate:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (4,):
            raise ValueError("observation must have shape (4,)")
        if outlier_threshold is not None and outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive or None")
        if not self.initialized:
            state = self.initialize(observation)
            return EKFUpdate(state=state, accepted=True, mahalanobis_distance=None)

        innovation = observation - self.observation_matrix @ self.state
        innovation[3] = self.wrap_angle(innovation[3])
        innovation_covariance = (
            self.observation_matrix
            @ self.covariance
            @ self.observation_matrix.T
            + self.observation_covariance
        )
        try:
            solved_innovation = np.linalg.solve(
                innovation_covariance, innovation
            )
            gain = np.linalg.solve(
                innovation_covariance,
                self.observation_matrix @ self.covariance,
            ).T
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(innovation_covariance)
            solved_innovation = inverse @ innovation
            gain = self.covariance @ self.observation_matrix.T @ inverse
        mahalanobis = float(
            np.sqrt(max(0.0, innovation @ solved_innovation))
        )
        if outlier_threshold is not None and mahalanobis > outlier_threshold:
            return EKFUpdate(
                state=self.state.copy(),
                accepted=False,
                mahalanobis_distance=mahalanobis,
            )

        self.state = self.state + gain @ innovation
        self.state[6] = self.wrap_angle(self.state[6])
        identity_minus_gain_h = (
            np.eye(7) - gain @ self.observation_matrix
        )
        self.covariance = (
            identity_minus_gain_h
            @ self.covariance
            @ identity_minus_gain_h.T
            + gain @ self.observation_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return EKFUpdate(
            state=self.state.copy(),
            accepted=True,
            mahalanobis_distance=mahalanobis,
        )
