"""Transparent gate-focused pose sampling for NPE fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from ..simulation.tracks import TrackConfig, get_track
from .generation import (
    CameraIntrinsics,
    PoseBounds,
    PoseSample,
    pose_to_camera_matrix,
)


@dataclass(frozen=True)
class GateFocusConfig:
    min_approach_distance_m: float = 0.35
    max_approach_distance_m: float = 2.0
    max_lateral_offset_m: float = 0.55
    max_vertical_offset_m: float = 0.32
    max_yaw_jitter_deg: float = 25.0
    image_margin_px: float = 32.0
    maximum_rejections: int = 100

    def __post_init__(self) -> None:
        continuous_names = (
            "min_approach_distance_m",
            "max_approach_distance_m",
            "max_lateral_offset_m",
            "max_vertical_offset_m",
            "max_yaw_jitter_deg",
            "image_margin_px",
        )
        continuous_values = tuple(getattr(self, name) for name in continuous_names)
        for name, value in zip(continuous_names, continuous_values):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
        if not np.isfinite(continuous_values).all():
            raise ValueError("all gate-focus configuration values must be finite")
        if self.min_approach_distance_m <= 0:
            raise ValueError("minimum approach distance must be positive")
        if self.max_approach_distance_m <= self.min_approach_distance_m:
            raise ValueError("maximum approach distance must exceed the minimum")
        if min(self.max_lateral_offset_m, self.max_vertical_offset_m) < 0:
            raise ValueError("lateral and vertical offsets cannot be negative")
        if not 0 <= self.max_yaw_jitter_deg <= 90:
            raise ValueError("yaw jitter must lie in [0,90] degrees")
        if not np.isfinite(self.image_margin_px) or self.image_margin_px < 0:
            raise ValueError("image margin must be a finite non-negative value")
        if (
            isinstance(self.maximum_rejections, (bool, np.bool_))
            or not isinstance(self.maximum_rejections, Integral)
            or self.maximum_rejections <= 0
        ):
            raise ValueError("maximum_rejections must be a positive integer")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GateCenterProjection:
    """One world point expressed in the legacy OpenCV image convention."""

    u_px: float
    v_px: float
    depth_m: float

    def is_visible(
        self, intrinsics: CameraIntrinsics, *, margin_px: float = 0.0
    ) -> bool:
        _validate_intrinsics(intrinsics)
        if not np.isfinite(margin_px) or margin_px < 0:
            raise ValueError("image margin must be a finite non-negative value")
        if 2.0 * margin_px >= min(intrinsics.width, intrinsics.height):
            raise ValueError("image margin leaves no usable image area")
        return bool(
            self.depth_m > 0.0
            and np.isfinite([self.u_px, self.v_px, self.depth_m]).all()
            and margin_px <= self.u_px <= intrinsics.width - 1.0 - margin_px
            and margin_px <= self.v_px <= intrinsics.height - 1.0 - margin_px
        )


def _validate_intrinsics(intrinsics: CameraIntrinsics) -> None:
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        raise ValueError("camera dimensions must be positive")
    values = np.array(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or min(intrinsics.fx, intrinsics.fy) <= 0:
        raise ValueError("camera focal lengths/principal point must be finite")
    if not (0 <= intrinsics.cx < intrinsics.width):
        raise ValueError("camera cx must lie inside the image")
    if not (0 <= intrinsics.cy < intrinsics.height):
        raise ValueError("camera cy must lie inside the image")


def project_world_point_to_image(
    pose: np.ndarray,
    world_point: np.ndarray,
    intrinsics: CameraIntrinsics = CameraIntrinsics(),
) -> GateCenterProjection:
    """Project a NeRF-world point using the renderer's body-camera contract.

    ``pose_to_camera_matrix`` stores body +X forward, +Y left, and +Z up.
    The recovered renderer maps OpenCV +Z to body +X, OpenCV +X to body -Y,
    and OpenCV +Y to body -Z.  This function applies that exact convention
    before the pinhole projection.
    """

    _validate_intrinsics(intrinsics)
    pose = np.asarray(pose, dtype=np.float64)
    point = np.asarray(world_point, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose must contain six finite values")
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("world_point must contain three finite values")

    body_to_world = pose_to_camera_matrix(pose)
    point_body = body_to_world[:3, :3].T @ (point - body_to_world[:3, 3])
    camera_x = -float(point_body[1])  # right
    camera_y = -float(point_body[2])  # down
    camera_z = float(point_body[0])  # optical axis / forward
    if camera_z <= 0.0:
        return GateCenterProjection(float("nan"), float("nan"), camera_z)
    return GateCenterProjection(
        u_px=float(intrinsics.fx * camera_x / camera_z + intrinsics.cx),
        v_px=float(intrinsics.fy * camera_y / camera_z + intrinsics.cy),
        depth_m=camera_z,
    )


class GateFocusedPoseSampler:
    """Sample pre-gate views that resemble the controller's approach phase.

    A point is drawn on the incoming side of one uniformly selected gate,
    offset laterally/vertically, and aimed along the gate's forward normal with
    bounded yaw jitter.  Bounds and a margin-safe gate-center projection are
    enforced by rejection rather than clipping, so the recorded distribution
    has no artificial mass on box or image boundaries.
    """

    def __init__(
        self,
        track: str | TrackConfig,
        bounds: PoseBounds,
        config: GateFocusConfig = GateFocusConfig(),
        intrinsics: CameraIntrinsics = CameraIntrinsics(),
    ) -> None:
        self.track = get_track(track) if isinstance(track, str) else track
        self.bounds = bounds
        self.config = config
        _validate_intrinsics(intrinsics)
        if 2.0 * config.image_margin_px >= min(intrinsics.width, intrinsics.height):
            raise ValueError("image margin leaves no usable image area")
        self.intrinsics = intrinsics
        self.gates = self.track.ordered_gates()

    @staticmethod
    def _within(value: float, limits: tuple[float, float]) -> bool:
        return limits[0] <= value <= limits[1]

    def sample(self, rng: np.random.Generator) -> PoseSample:
        # Select the gate once so visibility/bounds rejection cannot silently
        # skew the requested uniform gate distribution.
        gate_index = int(rng.integers(0, len(self.gates)))
        gate = self.gates[gate_index]
        for rejection_count in range(self.config.maximum_rejections):
            normal = gate.normal
            tangent = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
            approach = float(
                rng.uniform(
                    self.config.min_approach_distance_m,
                    self.config.max_approach_distance_m,
                )
            )
            lateral = float(
                rng.uniform(
                    -self.config.max_lateral_offset_m,
                    self.config.max_lateral_offset_m,
                )
            )
            vertical = float(
                rng.uniform(
                    -self.config.max_vertical_offset_m,
                    self.config.max_vertical_offset_m,
                )
            )
            position = (
                np.asarray(gate.center, dtype=np.float64)
                - approach * normal
                + lateral * tangent
                + np.array([0.0, 0.0, vertical])
            )
            if not (
                self._within(float(position[0]), self.bounds.x)
                and self._within(float(position[1]), self.bounds.y)
                and self._within(float(position[2]), self.bounds.z)
            ):
                continue
            yaw_jitter = float(
                rng.uniform(
                    -np.deg2rad(self.config.max_yaw_jitter_deg),
                    np.deg2rad(self.config.max_yaw_jitter_deg),
                )
            )
            gate_yaw = float(np.arctan2(normal[1], normal[0]))
            yaw = float((gate_yaw + yaw_jitter + np.pi) % (2 * np.pi) - np.pi)
            pose = np.array(
                [position[0], position[1], position[2], 0.0, 0.0, yaw],
                dtype=np.float64,
            )
            projection = project_world_point_to_image(
                pose, np.asarray(gate.center), self.intrinsics
            )
            if not projection.is_visible(
                self.intrinsics, margin_px=self.config.image_margin_px
            ):
                continue
            return PoseSample(
                pose=pose,
                annotations={
                    "focus_gate": gate.name,
                    "approach_distance_m": approach,
                    "lateral_offset_m": lateral,
                    "vertical_offset_m": vertical,
                    "yaw_jitter_rad": yaw_jitter,
                    "gate_center_u_px": projection.u_px,
                    "gate_center_v_px": projection.v_px,
                    "gate_center_depth_m": projection.depth_m,
                    "image_margin_px": self.config.image_margin_px,
                    "rejections_before_acceptance": rejection_count,
                },
            )
        raise RuntimeError(
            "gate-focused sampler could not place "
            f"{gate.name} inside both pose bounds and the margin-safe camera FOV"
        )
