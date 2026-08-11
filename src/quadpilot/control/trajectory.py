"""Trajectory primitives from the December 2025 visual controller."""

from __future__ import annotations

import numpy as np


def generate_hermite_spline(
    p0: np.ndarray,
    p1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    *,
    num_points: int = 50,
    scale_factor: float = 0.5,
) -> np.ndarray:
    """Generate a cubic Hermite connection between two 3-D points."""

    t = np.linspace(0.0, 1.0, num_points)
    scale = np.linalg.norm(p1 - p0) * scale_factor
    m0_scaled = m0 * scale
    m1_scaled = m1 * scale
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00[:, None] * p0
        + h10[:, None] * m0_scaled
        + h01[:, None] * p1
        + h11[:, None] * m1_scaled
    )


def generate_arc(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    z: float,
    *,
    num_points: int = 30,
) -> np.ndarray:
    angles = np.linspace(start_angle, end_angle, num_points)
    points = np.zeros((num_points, 3), dtype=np.float64)
    points[:, 0] = center[0] + radius * np.cos(angles)
    points[:, 1] = center[1] + radius * np.sin(angles)
    points[:, 2] = z
    return points


def generate_gate_transition(
    start_pos: np.ndarray,
    start_normal: np.ndarray,
    end_pos: np.ndarray,
    end_normal: np.ndarray,
    *,
    straight_dist: float = 0.3,
    num_straight: int = 10,
    num_curve: int = 40,
    is_lap_transition: bool = False,
    track: str = "circle",
) -> np.ndarray:
    """Reproduce the final project's gate-to-gate path primitive.

    The U-turn D-to-A transition intentionally retains the hand-designed
    arc/straight/arc maneuver used in the submitted controller.
    """

    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_normal = np.asarray(start_normal, dtype=np.float64)
    end_pos = np.asarray(end_pos, dtype=np.float64)
    end_normal = np.asarray(end_normal, dtype=np.float64)

    vec_to_next = end_pos - start_pos
    dist_to_next = np.linalg.norm(vec_to_next)
    dir_to_next = vec_to_next / dist_to_next if dist_to_next > 0.1 else start_normal

    if is_lap_transition and track == "uturn" and dist_to_next > 4.0:
        arc1_radius = 1.5
        arc1_center = start_pos + np.array([arc1_radius, 0.0, 0.0])
        arc1 = generate_arc(
            arc1_center,
            arc1_radius,
            np.pi / 2.0,
            3.0 * np.pi / 2.0,
            start_pos[2],
            num_points=50,
        )

        arc2_radius = 0.8
        arc2_center = end_pos + np.array([0.0, arc2_radius, 0.0])
        arc2_start = arc2_center + np.array([arc2_radius, 0.0, 0.0])
        straight = np.linspace(arc1[-1], arc2_start, 30)
        arc2 = generate_arc(
            arc2_center,
            arc2_radius,
            0.0,
            -np.pi / 4.0,
            end_pos[2],
            num_points=20,
        )
        final_approach = np.linspace(arc2[-1], end_pos, 15)
        return np.vstack([arc1, straight[1:], arc2[1:], final_approach[1:]])

    exit_dir = dir_to_next / np.linalg.norm(dir_to_next)
    exit_point = start_pos + straight_dist * exit_dir
    approach_point = end_pos - straight_dist * end_normal
    straight1 = np.linspace(start_pos, exit_point, num_straight)
    curve = generate_hermite_spline(
        exit_point,
        approach_point,
        exit_dir,
        end_normal,
        num_points=num_curve,
        scale_factor=0.6,
    )
    straight2 = np.linspace(approach_point, end_pos, num_straight)
    return np.vstack([straight1, curve[1:], straight2[1:]])
