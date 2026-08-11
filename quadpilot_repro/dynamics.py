"""Small deterministic dynamics model shared by simulation and EKF tests."""

from __future__ import annotations

import numpy as np


def body_acceleration_to_world(acceleration: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a three-axis acceleration command from body to world frame."""

    ax_body, ay_body, az_body = np.asarray(acceleration, dtype=np.float64)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [
            ax_body * cos_yaw - ay_body * sin_yaw,
            ax_body * sin_yaw + ay_body * cos_yaw,
            az_body,
        ],
        dtype=np.float64,
    )


def step_dynamics(
    state: np.ndarray, control: np.ndarray, dt: float = 0.05
) -> np.ndarray:
    """Advance the project double-integrator by one semi-implicit Euler step.

    The contract is explicit: state is in the NeRF world frame and acceleration
    control is in the drone body frame.  This is the transform that was missing
    from the public FalconGym template but present in the project repository.
    """

    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if state.shape != (7,):
        raise ValueError(f"state must have shape (7,), got {state.shape}")
    if control.shape != (4,):
        raise ValueError(f"control must have shape (4,), got {control.shape}")
    if dt <= 0:
        raise ValueError("dt must be positive")

    position = state[:3].copy()
    velocity = state[3:6].copy()
    yaw = float(state[6])

    acceleration_world = body_acceleration_to_world(control[:3], yaw)
    velocity += acceleration_world * dt
    position += velocity * dt
    yaw = (yaw + float(control[3]) * dt + np.pi) % (2.0 * np.pi) - np.pi

    return np.concatenate([position, velocity, np.array([yaw])])
