"""Offline hardware-calibration and safety-readiness contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _points(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3:
        raise ValueError(f"{name} must contain at least three xyz points")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    centered = array - np.mean(array, axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-9) < 2:
        raise ValueError(f"{name} points are collinear or coincident")
    return array


@dataclass(frozen=True)
class SimilarityTransform:
    source_frame: str
    target_frame: str
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not self.source_frame or not self.target_frame:
            raise ValueError("source and target frame names are required")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be positive and finite")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation/translation shapes must be (3,3)/(3,)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("rotation and translation must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("rotation determinant must be +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: Any) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.shape[-1:] != (3,) or not np.isfinite(array).all():
            raise ValueError("points must be finite xyz rows")
        return self.scale * (array @ self.rotation.T) + self.translation

    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def estimate_similarity_transform(
    source_points: Any,
    target_points: Any,
    *,
    source_frame: str,
    target_frame: str,
    estimate_scale: bool = True,
) -> tuple[SimilarityTransform, dict[str, float]]:
    """Estimate target <- source with the Umeyama least-squares solution."""

    source = _points("source_points", source_points)
    target = _points("target_points", target_points)
    if source.shape != target.shape:
        raise ValueError("source and target point arrays must have identical shape")

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    if estimate_scale:
        variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        if variance <= 1e-12:
            raise ValueError("source point variance is too small")
        scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    transform = SimilarityTransform(
        source_frame=source_frame,
        target_frame=target_frame,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    residuals = np.linalg.norm(transform.apply(source) - target, axis=1)
    report = {
        "correspondences": int(len(source)),
        "rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "mean_error_m": float(np.mean(residuals)),
        "max_error_m": float(np.max(residuals)),
    }
    return transform, report


def similarity_payload(
    transform: SimilarityTransform,
    report: dict[str, float],
    *,
    input_sha256: str,
    accepted_rmse_m: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "similarity_transform",
        "source_frame": transform.source_frame,
        "target_frame": transform.target_frame,
        "input_sha256": input_sha256,
        "estimate": {
            "scale": transform.scale,
            "rotation": transform.rotation.tolist(),
            "translation_m": transform.translation.tolist(),
            "matrix_target_from_source": transform.matrix().tolist(),
        },
        "residuals": report,
        "acceptance": {
            "maximum_rmse_m": accepted_rmse_m,
            "passed": report["rmse_m"] <= accepted_rmse_m,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def check_hardware_readiness(config_path: Path) -> dict[str, Any]:
    """Fail-closed, read-only readiness check before any hardware command."""

    resolved = config_path.expanduser().resolve()
    config = _load_json(resolved)
    base = resolved.parent
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record(
        "schema_version",
        config.get("schema_version") == 1,
        str(config.get("schema_version")),
    )
    stage = config.get("stage")
    record(
        "stage",
        stage == "bench_prop_off",
        f"stage={stage!r}; only bench_prop_off is enabled",
    )

    evidence = config.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    loaded_evidence: dict[str, dict[str, Any]] = {}
    for name in ("vicon_from_nerf", "camera_intrinsics", "body_from_camera"):
        spec = evidence.get(name, {})
        raw_path = spec.get("path") if isinstance(spec, dict) else None
        expected_sha = spec.get("sha256") if isinstance(spec, dict) else None
        if not isinstance(raw_path, str) or not raw_path:
            record(f"evidence.{name}", False, "path is not configured")
            continue
        path = (
            (base / raw_path).resolve()
            if not Path(raw_path).is_absolute()
            else Path(raw_path).resolve()
        )
        if not path.is_file():
            record(f"evidence.{name}", False, f"missing file: {path}")
            continue
        actual_sha = sha256_file(path)
        hash_ok = isinstance(expected_sha, str) and actual_sha == expected_sha.lower()
        record(f"evidence.{name}.sha256", hash_ok, f"actual={actual_sha}")
        try:
            loaded_evidence[name] = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            record(f"evidence.{name}.json", False, f"{type(exc).__name__}: {exc}")

    calibration = loaded_evidence.get("vicon_from_nerf")
    if calibration is not None:
        acceptance = calibration.get("acceptance", {})
        frames_ok = (
            calibration.get("source_frame") == "nerf_world"
            and calibration.get("target_frame") == "vicon_world"
        )
        record("calibration.frames", frames_ok, "required nerf_world -> vicon_world")
        record(
            "calibration.accepted", acceptance.get("passed") is True, str(acceptance)
        )

    intrinsics = loaded_evidence.get("camera_intrinsics")
    if intrinsics is not None:
        values = [
            intrinsics.get(key) for key in ("fx", "fy", "cx", "cy", "width", "height")
        ]
        numeric = all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in values
        )
        positive = numeric and all(float(value) > 0 for value in values)
        record("camera_intrinsics.values", positive, str(values))
        rms = intrinsics.get("calibration_rms_px")
        record(
            "camera_intrinsics.rms",
            isinstance(rms, (int, float)) and 0 <= float(rms) <= 1.0,
            f"calibration_rms_px={rms!r}",
        )

    extrinsics = loaded_evidence.get("body_from_camera")
    if extrinsics is not None:
        try:
            matrix = np.asarray(extrinsics["matrix_body_from_camera"], dtype=np.float64)
            rotation = matrix[:3, :3]
            rigid = (
                matrix.shape == (4, 4)
                and np.isfinite(matrix).all()
                and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
            )
        except (KeyError, TypeError, ValueError):
            rigid = False
        record("body_from_camera.rigid", rigid, "requires a finite rigid 4x4 matrix")

    topics = config.get("topics", {})
    required_topics = ("pose", "odom", "image", "setpoint", "command", "estop")
    topic_values = (
        [topics.get(name) for name in required_topics]
        if isinstance(topics, dict)
        else []
    )
    topics_ok = (
        len(topic_values) == len(required_topics)
        and all(
            isinstance(value, str) and value.startswith("/") for value in topic_values
        )
        and len(set(topic_values)) == len(topic_values)
    )
    record(
        "topics", topics_ok, str(dict(zip(required_topics, topic_values, strict=False)))
    )

    frames = config.get("frames", {})
    frame_contract = {
        "npe": "nerf_world",
        "pose": "vicon_world",
        "command": "world_accel_yaw_rate",
    }
    record(
        "frames",
        frames == frame_contract,
        f"required={frame_contract}; actual={frames}",
    )

    timeouts = config.get("timeouts_s", {})
    timeout_limits = {"pose": 0.1, "image": 0.2, "setpoint": 0.2, "command": 0.1}
    timeout_ok = isinstance(timeouts, dict) and all(
        isinstance(timeouts.get(name), (int, float))
        and 0 < float(timeouts[name]) <= limit
        for name, limit in timeout_limits.items()
    )
    record("timeouts", timeout_ok, f"limits={timeout_limits}; actual={timeouts}")

    geofence = config.get("geofence_m", {})
    try:
        minimum = np.asarray(geofence["min"], dtype=np.float64)
        maximum = np.asarray(geofence["max"], dtype=np.float64)
        geofence_ok = (
            minimum.shape == (3,)
            and maximum.shape == (3,)
            and np.isfinite(minimum).all()
            and np.isfinite(maximum).all()
            and np.all(minimum < maximum)
            and np.all(maximum - minimum <= np.asarray([12.0, 12.0, 4.0]))
        )
    except (KeyError, TypeError, ValueError):
        geofence_ok = False
    record("geofence", geofence_ok, str(geofence))

    limits = config.get("control_limits", {})
    limits_ok = (
        isinstance(limits, dict)
        and isinstance(limits.get("max_acceleration_mps2"), (int, float))
        and 0 < float(limits["max_acceleration_mps2"]) <= 3.0
        and isinstance(limits.get("max_yaw_rate_rad_s"), (int, float))
        and 0 < float(limits["max_yaw_rate_rad_s"]) <= 2.0
    )
    record("control_limits", limits_ok, str(limits))

    safety = config.get("manual_safety_checks", {})
    required_safety = (
        "propellers_removed",
        "operator_present",
        "physical_estop_tested",
        "radio_kill_tested",
        "vicon_occlusion_tested",
        "command_sign_tested",
        "geofence_tested",
        "no_people_in_test_volume",
        "battery_secured",
    )
    safety_ok = isinstance(safety, dict) and all(
        safety.get(name) is True for name in required_safety
    )
    record(
        "manual_safety_checks",
        safety_ok,
        str({name: safety.get(name) for name in required_safety}),
    )

    blockers = [check["name"] for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "config": str(resolved),
        "status": "READY_FOR_PROP_OFF_BENCH" if not blockers else "BLOCKED",
        "checks": checks,
        "blockers": blockers,
        "hardware_commands_executed": False,
    }
