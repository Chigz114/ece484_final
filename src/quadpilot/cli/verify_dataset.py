#!/usr/bin/env python3
"""CPU-only, fail-closed verification for one generated NPE dataset.

The verifier is read-only.  It delegates the canonical content fingerprint to
``quadpilot.perception.npe.build_dataset_index(..., fingerprint_mode="full")`` and
then checks the stricter generation contract that the trainer does not need to
enforce itself: one explicit source/track/seed, continuous label-safe sample
IDs, fully decodable 640x480 RGB PNGs, zero renderer failures, and exact
metadata/JSONL alignment.  Optional renderer provenance gates can pin the
checkpoint step, Gaussian count, checkpoint SHA-256, and dataparser-transform
SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
POSE_FORMAT = ("x", "y", "z", "roll", "pitch", "yaw")
POSE_UNITS = ("m", "m", "m", "rad", "rad", "rad")
TRACKS = ("circle", "lemniscate", "uturn")
GENERATOR_SAMPLERS = {
    "scripts/generate_repro_npe_dataset.py": "uniform_bounds",
    "scripts/generate_repro_gate_dataset.py": (
        "quadpilot.datasets.gate_sampling.GateFocusedPoseSampler"
    ),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GATE_ANNOTATION_KEYS = frozenset(
    {
        "focus_gate",
        "approach_distance_m",
        "lateral_offset_m",
        "vertical_offset_m",
        "yaw_jitter_rad",
        "gate_center_u_px",
        "gate_center_v_px",
        "gate_center_depth_m",
        "image_margin_px",
        "rejections_before_acceptance",
    }
)
GATE_GEOMETRY_ABS_TOLERANCE = 1e-9
GATE_4000_MINIMUM_PER_GATE = 850
GATE_4000_MAXIMUM_PER_GATE = 1150
GATE_SAMPLING_DESCRIPTION = (
    "uniform gate, incoming-side axial/lateral/vertical offsets, "
    "gate center inside margin-safe camera FOV"
)
TEMPORARY_NAME_PATTERN = re.compile(
    r"(?:^|[._-])(?:tmp|part|partial)(?:$|[._-])|\.crdownload$",
    flags=re.IGNORECASE,
)


class VerificationError(RuntimeError):
    """The dataset violated the frozen NPE-generation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _reject_constant(value: str) -> None:
    raise VerificationError(f"JSON contains non-standard non-finite value {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_json_text(text: str, source: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"cannot parse {source}: {exc}") from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON file: {path}")
    _require(path.stat().st_size > 0, f"empty JSON file: {path}")
    payload = _parse_json_text(path.read_text(encoding="utf-8"), str(path))
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _strict_integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    return result


def _pose(value: Any, name: str) -> tuple[float, ...]:
    _require(
        isinstance(value, list) and len(value) == 6, f"{name} must have six values"
    )
    result: list[float] = []
    for index, item in enumerate(value):
        _require(
            isinstance(item, (int, float)) and not isinstance(item, bool),
            f"{name}[{index}] must be numeric",
        )
        number = float(item)
        _require(math.isfinite(number), f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing file for SHA-256: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int, int], ...]:
    """Capture mutation-sensitive metadata without changing the dataset."""
    rows: list[tuple[str, int, int, int]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        rows.append(
            (relative, stat.S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns)
        )
    return tuple(rows)


def _reject_symlinks_and_temporary_files(root: Path) -> None:
    _require(not root.is_symlink(), "dataset root must not be a symlink")
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        _require(not path.is_symlink(), f"symlink is forbidden in dataset: {relative}")
        _require(
            TEMPORARY_NAME_PATTERN.search(path.name) is None,
            f"temporary/partial artifact is forbidden: {relative}",
        )


def _safe_declared_artifact(value: Any, name: str) -> Path:
    _require(isinstance(value, str) and value, f"metadata provenance {name} is missing")
    path = Path(value)
    _require(path.is_absolute(), f"metadata provenance {name} must be absolute")
    _require(not path.is_symlink(), f"metadata provenance {name} must not be a symlink")
    _require(path.is_file(), f"metadata provenance {name} does not exist: {path}")
    return path.resolve(strict=True)


def _default_index_builder(data_dirs: Sequence[Path], *, fingerprint_mode: str) -> Any:
    # Import lazily so lightweight verifier tests do not require a local Torch
    # installation.  Clearing visibility before the import makes the CLI's
    # CPU-only boundary explicit even if it is launched on a GPU host.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from quadpilot.perception.npe import build_dataset_index

    return build_dataset_index(data_dirs, fingerprint_mode=fingerprint_mode)


IndexBuilder = Callable[..., Any]


def _verify_index(
    root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    track: str,
    expected_frames: int,
    index_builder: IndexBuilder,
) -> tuple[Any, dict[str, Any]]:
    try:
        dataset = index_builder([root], fingerprint_mode="full")
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            f"build_dataset_index(full) rejected the dataset: {exc}"
        ) from exc
    _require(
        getattr(dataset, "fingerprint_mode", None) == "full",
        "dataset index is not full-content mode",
    )
    fingerprint = getattr(dataset, "fingerprint", None)
    _require(
        isinstance(fingerprint, str) and SHA256_PATTERN.fullmatch(fingerprint),
        "dataset index fingerprint is not a SHA-256",
    )
    sources = getattr(dataset, "sources", None)
    records = getattr(dataset, "records", None)
    _require(
        isinstance(sources, tuple) and len(sources) == 1,
        "dataset index must contain one source",
    )
    _require(
        isinstance(records, tuple) and len(records) == expected_frames,
        "dataset index frame count changed",
    )
    source = sources[0]
    _require(isinstance(source, dict), "dataset index source must be an object")
    _require(source.get("track") == track, "dataset index source track changed")
    _require(source.get("schema_version") == 2, "dataset index source schema changed")
    _require(
        source.get("sample_count") == expected_frames,
        "dataset index source count changed",
    )
    try:
        source_path = Path(source["path"]).resolve(strict=True)
    except Exception as exc:
        raise VerificationError("dataset index source path is invalid") from exc
    _require(
        source_path == root, "dataset index source path differs from requested root"
    )
    source_id = source.get("source_id")
    _require(
        isinstance(source_id, str) and source_id, "dataset index source_id is invalid"
    )
    metadata_digest = _sha256(root / "metadata.json")
    _require(
        source.get("metadata_sha256") == metadata_digest,
        "dataset index metadata SHA-256 differs from metadata.json",
    )

    image_bytes = 0
    for index, (record, row) in enumerate(zip(records, manifest_rows)):
        expected_relative = row["image"]
        _require(
            getattr(record, "source_id", None) == source_id,
            f"record {index} source_id changed",
        )
        try:
            record_root = Path(record.source_root).resolve(strict=True)
        except Exception as exc:
            raise VerificationError(f"record {index} source_root is invalid") from exc
        _require(
            record_root == root, f"record {index} escapes the single dataset source"
        )
        _require(
            getattr(record, "relative_image", None) == expected_relative,
            f"dataset index image {index} disagrees with samples.jsonl",
        )
        _require(
            getattr(record, "key", None) == f"{source_id}/{expected_relative}",
            f"dataset index key {index} changed",
        )
        try:
            record_pose = tuple(float(item) for item in record.pose)
        except Exception as exc:
            raise VerificationError(f"dataset index pose {index} is invalid") from exc
        _require(
            record_pose == tuple(row["pose"]),
            f"dataset index pose {index} disagrees with samples.jsonl",
        )
        _require(
            getattr(record, "width", None) == EXPECTED_WIDTH
            and getattr(record, "height", None) == EXPECTED_HEIGHT,
            f"dataset index dimensions changed for sample {index}",
        )
        digest = getattr(record, "image_sha256", None)
        _require(
            isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest),
            f"dataset index image {index} lacks a full SHA-256",
        )
        image_path = root / Path(*PurePosixPath(expected_relative).parts)
        image_bytes += image_path.stat().st_size
    return dataset, {
        "fingerprint": fingerprint,
        "fingerprint_mode": "full",
        "single_source": True,
        "source_id": source_id,
        "source_metadata_sha256": metadata_digest,
        "image_bytes": image_bytes,
    }


def _verify_renderer_provenance(
    metadata: Mapping[str, Any],
    *,
    expected_checkpoint_step: int | None,
    expected_gaussians: int | None,
    expected_checkpoint_sha256: str | None,
    expected_transform_sha256: str | None,
) -> dict[str, Any]:
    provenance = metadata.get("provenance")
    _require(isinstance(provenance, dict), "metadata provenance must be an object")
    generator = provenance.get("generator")
    _require(generator in GENERATOR_SAMPLERS, "metadata provenance generator changed")
    checkpoint_step = _strict_integer(
        provenance.get("checkpoint_step"), "metadata checkpoint_step", minimum=0
    )
    gaussian_count = _strict_integer(
        provenance.get("gaussian_count"), "metadata gaussian_count", minimum=1
    )
    if expected_checkpoint_step is not None:
        _require(
            checkpoint_step == expected_checkpoint_step,
            "checkpoint step differs from expectation",
        )
    if expected_gaussians is not None:
        _require(
            gaussian_count == expected_gaussians,
            "Gaussian count differs from expectation",
        )

    report: dict[str, Any] = {
        "generator": generator,
        "checkpoint_step": checkpoint_step,
        "gaussian_count": gaussian_count,
        "checkpoint_sha256_checked": expected_checkpoint_sha256 is not None,
        "transform_sha256_checked": expected_transform_sha256 is not None,
    }
    if expected_checkpoint_sha256 is not None:
        checkpoint = _safe_declared_artifact(provenance.get("checkpoint"), "checkpoint")
        actual = _sha256(checkpoint)
        _require(
            actual == expected_checkpoint_sha256,
            "checkpoint SHA-256 differs from expectation",
        )
        if expected_checkpoint_step is not None:
            expected_name = f"step-{expected_checkpoint_step:09d}.ckpt"
            _require(
                checkpoint.name == expected_name, "checkpoint filename/step mismatch"
            )
        report.update(
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": actual,
            }
        )
    if expected_transform_sha256 is not None:
        transform = _safe_declared_artifact(
            provenance.get("dataparser_transform"), "dataparser_transform"
        )
        actual = _sha256(transform)
        _require(
            actual == expected_transform_sha256,
            "dataparser transform SHA-256 differs from expectation",
        )
        report.update(
            {
                "dataparser_transform_path": str(transform),
                "dataparser_transform_size_bytes": transform.stat().st_size,
                "dataparser_transform_sha256": actual,
            }
        )
    return report


def _verify_progress_schema(
    progress: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    record_integrity: str,
    expected_frames: int,
    attempts: int,
    failures: int,
) -> dict[str, Any]:
    legacy_keys = {"attempts", "successes", "failures", "rng_state"}
    resume_keys = legacy_keys | {
        "schema_version",
        "target_samples",
        "maximum_failures",
        "generation_contract",
    }
    actual_keys = set(progress)
    if actual_keys == legacy_keys:
        _require(
            record_integrity == "legacy_unhashed",
            "mixed dataset schema: legacy progress requires unhashed records",
        )
        _require(
            "target_samples" not in metadata and "maximum_failures" not in metadata,
            "mixed dataset schema: legacy progress has resume metadata fields",
        )
        progress_mode = "legacy"
        maximum_failures: int | None = None
        contract_verified = False
    elif actual_keys == resume_keys:
        _require(
            record_integrity == "verified_per_record",
            "mixed dataset schema: resume progress requires hashed records",
        )
        _require(
            progress.get("schema_version") == 1,
            "resume progress schema_version must be 1",
        )
        target = _strict_integer(
            progress.get("target_samples"), "progress target_samples", minimum=1
        )
        _require(
            target == expected_frames,
            "progress target_samples differs from expected frames",
        )
        maximum_failures = _strict_integer(
            progress.get("maximum_failures"), "progress maximum_failures", minimum=0
        )
        _require(
            _strict_integer(
                metadata.get("target_samples"), "metadata target_samples", minimum=1
            )
            == target,
            "metadata/progress target_samples differ",
        )
        _require(
            _strict_integer(
                metadata.get("maximum_failures"), "metadata maximum_failures", minimum=0
            )
            == maximum_failures,
            "metadata/progress maximum_failures differ",
        )
        provenance = metadata.get("provenance")
        _require(isinstance(provenance, dict), "metadata provenance must be an object")
        generator = provenance.get("generator")
        _require(
            generator in GENERATOR_SAMPLERS, "metadata provenance generator changed"
        )
        expected_contract = {
            "track": metadata.get("track"),
            "seed": metadata.get("seed"),
            "bounds": metadata.get("bounds"),
            "intrinsics": metadata.get("intrinsics"),
            "provenance": provenance,
            "pose_sampler": GENERATOR_SAMPLERS[generator],
        }
        _require(
            progress.get("generation_contract") == expected_contract,
            "progress generation_contract differs from completed metadata",
        )
        progress_mode = "resume_v1"
        contract_verified = True
    else:
        raise VerificationError(
            "progress.json fields are neither the frozen legacy schema nor resume schema v1"
        )

    progress_attempts = _strict_integer(
        progress.get("attempts"), "progress attempts", minimum=0
    )
    progress_successes = _strict_integer(
        progress.get("successes"), "progress successes", minimum=0
    )
    progress_failures = _strict_integer(
        progress.get("failures"), "progress failures", minimum=0
    )
    _require(progress_attempts == attempts, "metadata/progress attempts differ")
    _require(
        progress_successes == expected_frames, "progress successes differs from frames"
    )
    _require(progress_failures == failures, "metadata/progress failures differ")
    _require(
        progress_attempts == progress_successes + progress_failures,
        "progress attempts must equal successes plus failures",
    )
    _require(
        isinstance(progress.get("rng_state"), dict),
        "progress rng_state must be an object",
    )
    return {
        "mode": progress_mode,
        "schema_version": 1 if progress_mode == "resume_v1" else "legacy",
        "target_samples": expected_frames,
        "maximum_failures": maximum_failures,
        "generation_contract_verified": contract_verified,
        "attempts": progress_attempts,
        "successes": progress_successes,
        "failures": progress_failures,
    }


def _strict_json_float(value: Any, name: str) -> float:
    _require(type(value) is float, f"{name} must be a JSON float")
    _require(math.isfinite(value), f"{name} must be finite")
    return value


def _require_close(actual: float, expected: float, name: str) -> None:
    _require(
        math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=GATE_GEOMETRY_ABS_TOLERANCE,
        ),
        f"{name} differs from reconstructed gate geometry",
    )


def _verify_gate_counts(
    counts: Mapping[str, int], *, expected_frames: int
) -> dict[str, Any]:
    required_gate_names = ("Gate A", "Gate B", "Gate C", "Gate D")
    _require(set(counts) == set(required_gate_names), "gate count keys must be A/B/C/D")
    _require(
        sum(counts.values()) == expected_frames,
        "gate counts do not sum to the expected frame count",
    )
    balance_applied = expected_frames == 4000
    if balance_applied:
        for gate_name, count in counts.items():
            _require(
                GATE_4000_MINIMUM_PER_GATE <= count <= GATE_4000_MAXIMUM_PER_GATE,
                f"4000-frame gate count for {gate_name} is outside [850,1150]: {count}",
            )
    return {
        "applied": balance_applied,
        "expected_frames": expected_frames,
        "minimum_per_gate": GATE_4000_MINIMUM_PER_GATE if balance_applied else None,
        "maximum_per_gate": GATE_4000_MAXIMUM_PER_GATE if balance_applied else None,
        "passed": True,
    }


def _verify_gate_focused_contract(
    metadata: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    track: str,
    expected_frames: int,
) -> dict[str, Any]:
    """Verify the exact default gate sampler and every recorded geometry label."""
    import numpy as np

    from quadpilot.datasets.gate_sampling import (
        GateFocusConfig,
        GateFocusedPoseSampler,
        project_world_point_to_image,
    )
    from quadpilot.datasets.generation import (
        BASE_DATASET_BOUNDS,
        CameraIntrinsics,
    )
    from quadpilot.simulation.tracks import get_track

    provenance = metadata.get("provenance")
    _require(isinstance(provenance, dict), "metadata provenance must be an object")
    _require(
        provenance.get("generator") == "scripts/generate_repro_gate_dataset.py",
        "gate-focused expectation requires the gate dataset generator",
    )
    _require(
        provenance.get("sampling") == GATE_SAMPLING_DESCRIPTION,
        "gate-focused provenance sampling description changed",
    )

    expected_config = GateFocusConfig().to_dict()
    actual_config = provenance.get("gate_focus_config")
    _require(
        isinstance(actual_config, dict) and set(actual_config) == set(expected_config),
        "metadata gate_focus_config fields differ from current defaults",
    )
    for key, expected in expected_config.items():
        actual = actual_config[key]
        _require(
            type(actual) is type(expected) and actual == expected,
            f"metadata gate_focus_config {key} differs from current default",
        )

    intrinsics_payload = metadata.get("intrinsics")
    expected_intrinsic_keys = {"width", "height", "fx", "fy", "cx", "cy"}
    _require(
        isinstance(intrinsics_payload, dict)
        and set(intrinsics_payload) == expected_intrinsic_keys,
        "gate-focused metadata intrinsics must contain the exact camera fields",
    )
    try:
        intrinsics = CameraIntrinsics(**intrinsics_payload)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            f"invalid gate-focused camera intrinsics: {exc}"
        ) from exc
    _require(
        intrinsics.width == EXPECTED_WIDTH and intrinsics.height == EXPECTED_HEIGHT,
        "gate-focused camera dimensions must be 640x480",
    )
    default_intrinsics = CameraIntrinsics()
    for key in expected_intrinsic_keys:
        actual = intrinsics_payload[key]
        expected = getattr(default_intrinsics, key)
        _require(
            type(actual) is type(expected) and actual == expected,
            f"gate-focused metadata intrinsics {key} differs from the current default",
        )

    expected_bounds = {
        key: list(getattr(BASE_DATASET_BOUNDS[track], key))
        for key in ("x", "y", "z", "yaw")
    }
    bounds_payload = metadata.get("bounds")
    _require(
        isinstance(bounds_payload, dict)
        and set(bounds_payload) == set(expected_bounds),
        "gate-focused metadata bounds fields differ from the canonical track bounds",
    )
    for key, expected in expected_bounds.items():
        actual = bounds_payload[key]
        _require(
            isinstance(actual, list)
            and len(actual) == 2
            and all(type(value) is float for value in actual)
            and actual == expected,
            f"gate-focused metadata bounds {key} differs from the canonical track bounds",
        )

    track_config = get_track(track)
    required_gate_names = ("Gate A", "Gate B", "Gate C", "Gate D")
    _require(
        set(track_config.gates) == set(required_gate_names),
        "gate-focused verifier requires the authoritative A/B/C/D gates",
    )
    counts = {name: 0 for name in required_gate_names}
    rejection_counts: list[int] = []
    maximum_yaw_jitter = math.radians(expected_config["max_yaw_jitter_deg"])
    margin = expected_config["image_margin_px"]
    replay_sampler = GateFocusedPoseSampler(
        track_config,
        BASE_DATASET_BOUNDS[track],
        GateFocusConfig(),
        intrinsics,
    )
    replay_rng = np.random.default_rng(metadata["seed"])

    for index, row in enumerate(manifest_rows):
        annotations = row.get("annotations")
        _require(
            isinstance(annotations, dict) and set(annotations) == GATE_ANNOTATION_KEYS,
            f"sample {index} gate annotations must have the exact required fields",
        )
        focus_gate = annotations["focus_gate"]
        _require(
            type(focus_gate) is str and focus_gate in track_config.gates,
            f"sample {index} focus_gate is not an authoritative {track} gate",
        )
        counts[focus_gate] += 1

        approach = _strict_json_float(
            annotations["approach_distance_m"],
            f"sample {index} approach_distance_m",
        )
        lateral = _strict_json_float(
            annotations["lateral_offset_m"],
            f"sample {index} lateral_offset_m",
        )
        vertical = _strict_json_float(
            annotations["vertical_offset_m"],
            f"sample {index} vertical_offset_m",
        )
        yaw_jitter = _strict_json_float(
            annotations["yaw_jitter_rad"],
            f"sample {index} yaw_jitter_rad",
        )
        annotated_u = _strict_json_float(
            annotations["gate_center_u_px"],
            f"sample {index} gate_center_u_px",
        )
        annotated_v = _strict_json_float(
            annotations["gate_center_v_px"],
            f"sample {index} gate_center_v_px",
        )
        annotated_depth = _strict_json_float(
            annotations["gate_center_depth_m"],
            f"sample {index} gate_center_depth_m",
        )
        annotated_margin = _strict_json_float(
            annotations["image_margin_px"],
            f"sample {index} image_margin_px",
        )
        rejections = _strict_integer(
            annotations["rejections_before_acceptance"],
            f"sample {index} rejections_before_acceptance",
            minimum=0,
        )

        _require(
            expected_config["min_approach_distance_m"]
            <= approach
            <= expected_config["max_approach_distance_m"],
            f"sample {index} approach_distance_m is outside the configured range",
        )
        _require(
            abs(lateral) <= expected_config["max_lateral_offset_m"],
            f"sample {index} lateral_offset_m is outside the configured range",
        )
        _require(
            abs(vertical) <= expected_config["max_vertical_offset_m"],
            f"sample {index} vertical_offset_m is outside the configured range",
        )
        _require(
            abs(yaw_jitter) <= maximum_yaw_jitter,
            f"sample {index} yaw_jitter_rad is outside the configured range",
        )
        _require(
            annotated_margin == margin,
            f"sample {index} image_margin_px differs from the configured margin",
        )
        _require(
            rejections < expected_config["maximum_rejections"],
            f"sample {index} rejections_before_acceptance is outside the configured range",
        )
        _require(annotated_depth > 0.0, f"sample {index} gate depth must be positive")
        _require(
            margin <= annotated_u <= intrinsics.width - 1.0 - margin
            and margin <= annotated_v <= intrinsics.height - 1.0 - margin,
            f"sample {index} annotated gate center is outside the margin-safe FOV",
        )

        gate = track_config.gates[focus_gate]
        normal = gate.normal
        tangent_x = -float(normal[1])
        tangent_y = float(normal[0])
        expected_x = float(gate.center[0] - approach * normal[0] + lateral * tangent_x)
        expected_y = float(gate.center[1] - approach * normal[1] + lateral * tangent_y)
        expected_z = float(gate.center[2] + vertical)
        gate_yaw = math.atan2(float(normal[1]), float(normal[0]))
        expected_yaw = (gate_yaw + yaw_jitter + math.pi) % (2.0 * math.pi) - math.pi
        pose = row["pose"]
        _require_close(pose[0], expected_x, f"sample {index} x")
        _require_close(pose[1], expected_y, f"sample {index} y")
        _require_close(pose[2], expected_z, f"sample {index} z")
        _require_close(pose[3], 0.0, f"sample {index} roll")
        _require_close(pose[4], 0.0, f"sample {index} pitch")
        yaw_error = math.atan2(
            math.sin(pose[5] - expected_yaw),
            math.cos(pose[5] - expected_yaw),
        )
        _require(
            abs(yaw_error) <= GATE_GEOMETRY_ABS_TOLERANCE,
            f"sample {index} yaw differs from reconstructed gate geometry",
        )

        reconstructed_pose = [
            expected_x,
            expected_y,
            expected_z,
            0.0,
            0.0,
            expected_yaw,
        ]
        projection = project_world_point_to_image(
            reconstructed_pose,
            gate.center,
            intrinsics,
        )
        _require(
            projection.is_visible(intrinsics, margin_px=margin),
            f"sample {index} reconstructed gate center is outside the margin-safe FOV",
        )
        _require_close(annotated_u, projection.u_px, f"sample {index} gate_center_u_px")
        _require_close(annotated_v, projection.v_px, f"sample {index} gate_center_v_px")
        _require_close(
            annotated_depth,
            projection.depth_m,
            f"sample {index} gate_center_depth_m",
        )
        rejection_counts.append(rejections)

        replayed = replay_sampler.sample(replay_rng)
        replayed_pose = tuple(float(value) for value in replayed.pose)
        _require(
            tuple(pose) == replayed_pose,
            f"sample {index} pose differs from deterministic gate sampler replay",
        )
        _require(
            annotations == dict(replayed.annotations or {}),
            f"sample {index} annotations differ from deterministic gate sampler replay",
        )

    _require(len(rejection_counts) == expected_frames, "gate annotation count changed")
    count_gate = _verify_gate_counts(counts, expected_frames=expected_frames)

    total_rejections = sum(rejection_counts)
    return {
        "enabled": True,
        "generator": provenance["generator"],
        "gate_focus_config": expected_config,
        "counts": {
            gate_name.removeprefix("Gate "): counts[gate_name]
            for gate_name in required_gate_names
        },
        "geometry_verified": expected_frames,
        "projection_verified": expected_frames,
        "seed_replay_verified": expected_frames,
        "rejection_stats": {
            "minimum": min(rejection_counts),
            "maximum": max(rejection_counts),
            "total": total_rejections,
            "mean": total_rejections / expected_frames,
            "samples_with_rejections": sum(value > 0 for value in rejection_counts),
        },
        "count_gate": count_gate,
    }


def verify_dataset(
    dataset_dir: Path,
    *,
    track: str,
    seed: int,
    expected_frames: int,
    expected_checkpoint_step: int | None = None,
    expected_gaussians: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_transform_sha256: str | None = None,
    expect_gate_focused: bool = False,
    index_builder: IndexBuilder | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready audit report without changing the dataset."""
    _require(track in TRACKS, f"unsupported track: {track}")
    seed = _strict_integer(seed, "expected seed")
    expected_frames = _strict_integer(expected_frames, "expected frames", minimum=1)
    _require(
        isinstance(expect_gate_focused, bool),
        "expect_gate_focused must be a boolean",
    )
    if expected_checkpoint_step is not None:
        expected_checkpoint_step = _strict_integer(
            expected_checkpoint_step, "expected checkpoint step", minimum=0
        )
    if expected_gaussians is not None:
        expected_gaussians = _strict_integer(
            expected_gaussians, "expected Gaussian count", minimum=1
        )
    for value, name in (
        (expected_checkpoint_sha256, "expected checkpoint SHA-256"),
        (expected_transform_sha256, "expected transform SHA-256"),
    ):
        if value is not None:
            _require(
                isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
                f"{name} must be 64 lowercase hexadecimal characters",
            )

    _require(dataset_dir.is_dir(), f"dataset directory does not exist: {dataset_dir}")
    _require(not dataset_dir.is_symlink(), "dataset directory must not be a symlink")
    root = dataset_dir.resolve(strict=True)
    if index_builder is None:
        index_builder = _default_index_builder
    snapshot_before = _tree_snapshot(root)
    _reject_symlinks_and_temporary_files(root)

    metadata_path = root / "metadata.json"
    manifest_path = root / "samples.jsonl"
    progress_path = root / "progress.json"
    metadata = _load_json_object(metadata_path)
    progress = _load_json_object(progress_path)

    _require(metadata.get("schema_version") == 2, "metadata schema_version must be 2")
    _require(
        metadata.get("track") == track, "metadata track differs from expected track"
    )
    _require(
        _strict_integer(metadata.get("seed"), "metadata seed") == seed,
        "metadata seed differs from expected seed",
    )
    n_frames = _strict_integer(metadata.get("n_frames"), "metadata n_frames", minimum=1)
    _require(
        n_frames == expected_frames, "metadata n_frames differs from expected frames"
    )
    attempts = _strict_integer(metadata.get("attempts"), "metadata attempts", minimum=0)
    failures = _strict_integer(
        metadata.get("render_failures"), "metadata render_failures", minimum=0
    )
    _require(failures == 0, "metadata reports one or more render failures")
    _require(
        attempts == expected_frames, "zero-failure dataset must have attempts == frames"
    )
    _require(
        metadata.get("samples_manifest") == "samples.jsonl",
        "metadata samples_manifest changed",
    )
    _require(
        metadata.get("image_format") == "RGB uint8 PNG", "metadata image_format changed"
    )
    _require(
        metadata.get("image_size") == [EXPECTED_WIDTH, EXPECTED_HEIGHT],
        "metadata image_size changed",
    )
    _require(
        metadata.get("pose_format") == list(POSE_FORMAT), "metadata pose_format changed"
    )
    _require(
        metadata.get("pose_units") == list(POSE_UNITS), "metadata pose_units changed"
    )
    _require(
        metadata.get("pose_coordinate_frame") == "original_nerf_world",
        "metadata pose coordinate frame changed",
    )

    poses_payload = metadata.get("poses")
    _require(isinstance(poses_payload, list), "metadata poses must be a list")
    _require(
        len(poses_payload) == expected_frames, "metadata pose count differs from frames"
    )
    metadata_poses = [
        _pose(value, f"metadata pose {index}")
        for index, value in enumerate(poses_payload)
    ]

    _require(manifest_path.is_file(), "samples.jsonl is missing")
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    _require(
        len(lines) == expected_frames,
        "samples.jsonl physical line count differs from frames",
    )
    _require(all(line.strip() for line in lines), "samples.jsonl contains a blank line")
    manifest_rows: list[dict[str, Any]] = []
    record_hash_presence: list[bool] = []
    for index, line in enumerate(lines):
        record = _parse_json_text(line, f"{manifest_path}:{index + 1}")
        _require(isinstance(record, dict), f"sample {index} must be an object")
        required_fields = {"sample_id", "image", "pose", "attempt"}
        _require(
            required_fields <= set(record), f"sample {index} is missing required fields"
        )
        _require(
            set(record) <= required_fields | {"annotations", "image_sha256"},
            f"sample {index} has an unexpected field",
        )
        if "annotations" in record:
            _require(
                isinstance(record["annotations"], dict),
                f"sample {index} annotations must be an object",
            )
        sample_id = _strict_integer(
            record.get("sample_id"), f"sample {index} sample_id", minimum=0
        )
        _require(
            sample_id == index,
            "samples.jsonl sample IDs must be continuous and line-ordered from zero",
        )
        attempt = _strict_integer(
            record.get("attempt"), f"sample {index} attempt", minimum=1
        )
        _require(
            attempt == index + 1,
            "zero-failure dataset attempts must be continuous from one",
        )
        expected_image = f"images/frame_{index:05d}.png"
        _require(
            record.get("image") == expected_image,
            f"sample {index} image path is not canonical",
        )
        sample_pose = _pose(record.get("pose"), f"sample {index} pose")
        _require(
            sample_pose == metadata_poses[index],
            f"metadata/sample pose {index} differs",
        )
        has_image_hash = "image_sha256" in record
        record_hash_presence.append(has_image_hash)
        image_sha256 = record.get("image_sha256")
        if has_image_hash:
            _require(
                isinstance(image_sha256, str)
                and SHA256_PATTERN.fullmatch(image_sha256) is not None,
                f"sample {index} image_sha256 must be 64 lowercase hexadecimal characters",
            )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "image": expected_image,
                "pose": sample_pose,
                "attempt": attempt,
                "image_sha256": image_sha256,
                "annotations": record.get("annotations"),
            }
        )

    _require(
        all(record_hash_presence) or not any(record_hash_presence),
        "mixed samples schema: image_sha256 must be present on every record or none",
    )
    record_integrity = (
        "verified_per_record" if all(record_hash_presence) else "legacy_unhashed"
    )

    images_dir = root / "images"
    _require(images_dir.is_dir(), "images directory is missing")
    expected_names = {f"frame_{index:05d}.png" for index in range(expected_frames)}
    actual_entries = list(images_dir.iterdir())
    _require(
        all(path.is_file() for path in actual_entries),
        "images directory contains a non-file entry",
    )
    actual_names = {path.name for path in actual_entries}
    _require(
        actual_names == expected_names,
        "images directory does not exactly match samples.jsonl",
    )
    for index in range(expected_frames):
        path = images_dir / f"frame_{index:05d}.png"
        try:
            with Image.open(path) as image:
                _require(image.format == "PNG", f"sample {index} is not encoded as PNG")
                _require(image.mode == "RGB", f"sample {index} is not RGB")
                _require(
                    image.size == (EXPECTED_WIDTH, EXPECTED_HEIGHT),
                    f"sample {index} dimensions are not 640x480",
                )
                _require(
                    getattr(image, "n_frames", 1) == 1,
                    f"sample {index} is animated/multiframe",
                )
                image.load()
                _require(
                    image.getbands() == ("R", "G", "B"),
                    f"sample {index} RGB bands changed",
                )
        except VerificationError:
            raise
        except Exception as exc:
            raise VerificationError(
                f"sample {index} PNG cannot be fully decoded: {exc}"
            ) from exc
        if record_integrity == "verified_per_record":
            actual_image_hash = _sha256(path)
            _require(
                actual_image_hash == manifest_rows[index]["image_sha256"],
                f"sample {index} image SHA-256 differs from samples.jsonl",
            )

    progress_report = _verify_progress_schema(
        progress,
        metadata,
        record_integrity=record_integrity,
        expected_frames=expected_frames,
        attempts=attempts,
        failures=failures,
    )
    gate_focused_report = (
        _verify_gate_focused_contract(
            metadata,
            manifest_rows,
            track=track,
            expected_frames=expected_frames,
        )
        if expect_gate_focused
        else None
    )

    dataset, index_report = _verify_index(
        root,
        manifest_rows,
        track=track,
        expected_frames=expected_frames,
        index_builder=index_builder,
    )
    del dataset
    provenance_report = _verify_renderer_provenance(
        metadata,
        expected_checkpoint_step=expected_checkpoint_step,
        expected_gaussians=expected_gaussians,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_transform_sha256=expected_transform_sha256,
    )

    snapshot_after = _tree_snapshot(root)
    _require(
        snapshot_after == snapshot_before, "dataset tree changed during verification"
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "cpu_only": True,
        "dataset_modified": False,
        "dataset": str(root),
        "track": track,
        "seed": seed,
        "frames": expected_frames,
        "render_failures": 0,
        "attempts": attempts,
        "images": {
            "count": expected_frames,
            "format": "PNG",
            "mode": "RGB",
            "width": EXPECTED_WIDTH,
            "height": EXPECTED_HEIGHT,
            "fully_decoded": expected_frames,
            "total_bytes": index_report.pop("image_bytes"),
        },
        "samples_manifest": {
            "path": "samples.jsonl",
            "continuous_sample_ids": True,
            "metadata_pose_image_aligned": True,
            "per_record_image_integrity": record_integrity,
            "sha256": _sha256(manifest_path),
        },
        "progress_schema": progress_report,
        "metadata_sha256": _sha256(metadata_path),
        "progress_sha256": _sha256(progress_path),
        "dataset_index": index_report,
        "renderer_provenance": provenance_report,
        "temporary_or_partial_files": 0,
        "dataset_tree_unchanged": True,
    }
    if gate_focused_report is not None:
        report["gate_focused"] = gate_focused_report
    return report


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _sha256_arg(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be 64 lowercase hexadecimal characters")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-frames", required=True, type=_positive_int)
    parser.add_argument("--expected-checkpoint-step", type=_nonnegative_int)
    parser.add_argument("--expected-gaussians", type=_positive_int)
    parser.add_argument("--expected-checkpoint-sha256", type=_sha256_arg)
    parser.add_argument("--expected-transform-sha256", type=_sha256_arg)
    parser.add_argument(
        "--expect-gate-focused",
        action="store_true",
        help=(
            "require the current default gate-focused generator, annotations, "
            "geometry, projection, and the 4000-frame per-gate count gate"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = verify_dataset(
            args.dataset_dir,
            track=args.track,
            seed=args.seed,
            expected_frames=args.expected_frames,
            expected_checkpoint_step=args.expected_checkpoint_step,
            expected_gaussians=args.expected_gaussians,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_transform_sha256=args.expected_transform_sha256,
            expect_gate_focused=args.expect_gate_focused,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except VerificationError as exc:
        print(f"VERIFY_REPRO_NPE_DATASET_FAILED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"VERIFY_REPRO_NPE_DATASET_FAILED: unexpected {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
