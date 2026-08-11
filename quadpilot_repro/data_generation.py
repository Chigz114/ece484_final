"""Crash-aware, label-safe NPE dataset generation primitives."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image


PROGRESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 546.84164912
    fy: float = 547.57957461
    cx: float = 349.18316327
    cy: float = 215.54486004

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"camera {name} must be a positive integer")
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"camera {name} must be a finite real number")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not 0 <= self.cx < self.width or not 0 <= self.cy < self.height:
            raise ValueError("camera principal point must lie inside the image")


@dataclass(frozen=True)
class PoseBounds:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    yaw: tuple[float, float] = (-np.pi, np.pi)

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "yaw"):
            low, high = getattr(self, name)
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError(f"invalid {name} bounds: {(low, high)}")


BASE_DATASET_BOUNDS: dict[str, PoseBounds] = {
    "circle": PoseBounds(x=(-4.7, 0.0), y=(-6.7, -1.1), z=(-0.8, 0.4)),
    "uturn": PoseBounds(x=(-5.0, 1.0), y=(-7.2, -1.0), z=(-0.8, 0.5)),
    "lemniscate": PoseBounds(x=(-4.5, 0.0), y=(-7.0, -0.5), z=(-1.0, 0.2)),
}

# The Lemniscate controller starts below the original axis-aligned training
# box and flies a short, nearly straight launch segment before entering it.
# Keep that recovery region separate from the historical base distribution so
# its provenance and source balance remain auditable.
LAUNCH_CORRIDOR_BOUNDS: dict[str, PoseBounds] = {
    "lemniscate": PoseBounds(
        x=(-3.8, -3.0),
        y=(-8.8, -6.5),
        z=(-0.75, -0.05),
        yaw=(0.9, 1.85),
    ),
}


def resolve_dataset_bounds(track: str, region: str = "base") -> PoseBounds:
    if track not in BASE_DATASET_BOUNDS:
        raise ValueError(f"unknown track: {track!r}")
    if region == "base":
        return BASE_DATASET_BOUNDS[track]
    if region == "launch-corridor" and track in LAUNCH_CORRIDOR_BOUNDS:
        return LAUNCH_CORRIDOR_BOUNDS[track]
    raise ValueError(f"region {region!r} is not defined for track {track!r}")


class RGBRenderer(Protocol):
    def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
        """Return an HxWx3 RGB image as uint8 or float in [0, 1]."""


@dataclass(frozen=True)
class PoseSample:
    """A renderer pose plus optional JSON-safe sampling annotations."""

    pose: np.ndarray
    annotations: Mapping[str, Any] | None = None


class PoseSampler(Protocol):
    def sample(self, rng: np.random.Generator) -> PoseSample:
        """Draw one labelled pose from ``rng``."""


def pose_to_camera_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert `[x,y,z,roll,pitch,yaw]` to the legacy camera-to-world matrix."""

    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("pose must have shape (6,)")
    if not np.isfinite(pose).all():
        raise ValueError("pose contains NaN or infinity")
    x, y, z, roll, pitch, yaw = pose
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation_z @ rotation_y @ rotation_x
    camera_to_world[:3, 3] = [x, y, z]
    return camera_to_world


def normalize_rgb(
    image: Any,
    *,
    width: int,
    height: int,
    minimum_dynamic_range: int = 5,
) -> np.ndarray:
    """Validate and normalize a renderer image to HxWx3 RGB uint8."""

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    expected_shape = (height, width, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"renderer returned shape {array.shape}; expected {expected_shape} RGB"
        )
    if not np.isfinite(array).all():
        raise ValueError("renderer returned NaN or infinity")

    if np.issubdtype(array.dtype, np.floating):
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < -1e-5 or maximum > 1.00001:
            raise ValueError(
                f"float renderer output must be in [0,1], got [{minimum},{maximum}]"
            )
        normalized = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < 0 or maximum > 255:
            raise ValueError(
                f"integer renderer output must be in [0,255], got [{minimum},{maximum}]"
            )
        normalized = array.astype(np.uint8)
    else:
        raise TypeError(f"unsupported renderer dtype: {array.dtype}")

    dynamic_range = int(normalized.max()) - int(normalized.min())
    if dynamic_range < minimum_dynamic_range:
        raise ValueError(
            f"renderer output dynamic range {dynamic_range} is below "
            f"minimum {minimum_dynamic_range}"
        )
    return np.ascontiguousarray(normalized)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_normalized(value: Any) -> Any:
    """Return the exact JSON-domain representation or reject unsafe values."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


class ReproDatasetGenerator:
    """Generate exactly N successful samples with continuous label-safe IDs."""

    def __init__(
        self,
        renderer: RGBRenderer,
        output_dir: Path,
        *,
        track: str,
        bounds: PoseBounds,
        intrinsics: CameraIntrinsics = CameraIntrinsics(),
        seed: int = 42,
        provenance: dict[str, Any] | None = None,
        pose_sampler: PoseSampler | None = None,
    ) -> None:
        self.renderer = renderer
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.track = track
        self.bounds = bounds
        self.intrinsics = intrinsics
        self.seed = int(seed)
        self.provenance = provenance or {}
        self.pose_sampler = pose_sampler
        self.rng = np.random.default_rng(self.seed)

    @property
    def records_path(self) -> Path:
        return self.output_dir / "samples.jsonl"

    @property
    def progress_path(self) -> Path:
        return self.output_dir / "progress.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    def _generation_contract(self) -> dict[str, Any]:
        sampler_name = (
            "uniform_bounds"
            if self.pose_sampler is None
            else (
                f"{type(self.pose_sampler).__module__}."
                f"{type(self.pose_sampler).__qualname__}"
            )
        )
        return _json_normalized(
            {
                "track": self.track,
                "seed": self.seed,
                "bounds": asdict(self.bounds),
                "intrinsics": asdict(self.intrinsics),
                "provenance": self.provenance,
                "pose_sampler": sampler_name,
            }
        )

    def _progress_payload(
        self,
        *,
        target_samples: int,
        maximum_failures: int,
        attempts: int,
        successes: int,
        failures: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "target_samples": target_samples,
            "maximum_failures": maximum_failures,
            "generation_contract": self._generation_contract(),
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "rng_state": _json_normalized(self.rng.bit_generator.state),
        }

    def _initialize_output(
        self, *, target_samples: int, maximum_failures: int
    ) -> None:
        if self.output_dir.is_symlink():
            raise ValueError(f"output directory cannot be a symlink: {self.output_dir}")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(self.output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to mix with non-empty dataset directory: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("x", encoding="utf-8", newline="\n"):
            pass
        self._save_progress(
            target_samples=target_samples,
            maximum_failures=maximum_failures,
            attempts=0,
            successes=0,
            failures=0,
        )

    def sample_pose(self) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(*self.bounds.x),
                self.rng.uniform(*self.bounds.y),
                self.rng.uniform(*self.bounds.z),
                0.0,
                0.0,
                self.rng.uniform(*self.bounds.yaw),
            ],
            dtype=np.float64,
        )

    def sample(self) -> PoseSample:
        sample = (
            PoseSample(self.sample_pose())
            if self.pose_sampler is None
            else self.pose_sampler.sample(self.rng)
        )
        pose = np.asarray(sample.pose, dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("pose sampler must return six finite pose values")
        annotations = dict(sample.annotations or {})
        # Fail before rendering if annotations cannot be serialized exactly.
        json.dumps(annotations, allow_nan=False)
        return PoseSample(pose=pose, annotations=annotations)

    @staticmethod
    def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            json.dumps(payload, allow_nan=False)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid {description}: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{description} must contain a JSON object: {path}")
        return payload

    @staticmethod
    def _strict_nonnegative_int(value: Any, *, field: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{field} must be a non-negative integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return result

    def _validate_output_topology(self) -> bool:
        if self.output_dir.is_symlink() or not self.output_dir.is_dir():
            raise ValueError(f"resume output must be a real directory: {self.output_dir}")
        allowed_root = {
            self.images_dir.name,
            self.records_path.name,
            self.progress_path.name,
            self.metadata_path.name,
        }
        root_entries = {entry.name: entry for entry in self.output_dir.iterdir()}
        unexpected_root = sorted(set(root_entries) - allowed_root)
        if unexpected_root:
            raise ValueError(
                f"unexpected files in resume output: {unexpected_root}"
            )
        required_root = {
            self.images_dir.name,
            self.records_path.name,
            self.progress_path.name,
        }
        missing_root = sorted(required_root - set(root_entries))
        if missing_root:
            raise ValueError(f"resume output is missing required files: {missing_root}")
        for path in (self.records_path, self.progress_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"resume file must be a regular file: {path}")
        if self.images_dir.is_symlink() or not self.images_dir.is_dir():
            raise ValueError(f"resume images path must be a real directory: {self.images_dir}")
        if self.metadata_path.name in root_entries and (
            self.metadata_path.is_symlink() or not self.metadata_path.is_file()
        ):
            raise ValueError(f"metadata must be a regular file: {self.metadata_path}")
        return self.metadata_path.name in root_entries

    def _load_records(self) -> list[dict[str, Any]]:
        raw = self.records_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("samples.jsonl ends with an incomplete record")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("samples.jsonl is not valid UTF-8") from exc
        if "\r" in text:
            raise ValueError("samples.jsonl must use canonical LF line endings")

        records: list[dict[str, Any]] = []
        previous_attempt = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise ValueError(f"empty samples.jsonl line {line_number}")
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                json.dumps(record, allow_nan=False)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid samples.jsonl record on line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"samples.jsonl line {line_number} is not an object")
            required_keys = {
                "sample_id",
                "image",
                "image_sha256",
                "pose",
                "attempt",
            }
            allowed_keys = required_keys | {"annotations"}
            if set(record) - allowed_keys or not required_keys.issubset(record):
                raise ValueError(
                    f"unexpected or missing fields in samples.jsonl line {line_number}"
                )

            sample_id = self._strict_nonnegative_int(
                record["sample_id"], field="sample_id"
            )
            if sample_id != len(records):
                raise ValueError(
                    f"sample_id must be continuous: got {sample_id}, "
                    f"expected {len(records)}"
                )
            expected_image = f"images/frame_{sample_id:05d}.png"
            if record["image"] != expected_image:
                raise ValueError(
                    f"sample {sample_id} image path must be {expected_image!r}"
                )
            digest = record["image_sha256"]
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and digest == digest.lower()
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"sample {sample_id} has an invalid image SHA-256")

            pose = record["pose"]
            if not (
                isinstance(pose, list)
                and len(pose) == 6
                and all(type(value) is float for value in pose)
                and np.isfinite(pose).all()
            ):
                raise ValueError(f"sample {sample_id} pose must contain six finite floats")
            attempt = self._strict_nonnegative_int(
                record["attempt"], field="attempt"
            )
            if attempt <= previous_attempt:
                raise ValueError("sample attempts must be positive and strictly increasing")
            previous_attempt = attempt
            if "annotations" in record:
                if not isinstance(record["annotations"], dict) or not record["annotations"]:
                    raise ValueError(
                        f"sample {sample_id} annotations must be a non-empty object"
                    )
            records.append(record)
        return records

    def _validate_images(self, records: list[dict[str, Any]]) -> None:
        image_entries = list(self.images_dir.iterdir())
        for entry in image_entries:
            if entry.is_symlink() or not entry.is_file():
                raise ValueError(f"unexpected non-file image entry: {entry}")
        expected_names = {Path(record["image"]).name for record in records}
        actual_names = {entry.name for entry in image_entries}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            orphaned = sorted(actual_names - expected_names)
            raise ValueError(
                f"resume image set mismatch: missing={missing}, orphaned={orphaned}"
            )

        expected_size = (self.intrinsics.width, self.intrinsics.height)
        for record in records:
            sample_id = record["sample_id"]
            image_path = self.output_dir / record["image"]
            actual_hash = _sha256_file(image_path)
            if actual_hash != record["image_sha256"]:
                raise ValueError(f"sample {sample_id} image SHA-256 mismatch")
            try:
                with Image.open(image_path) as image:
                    if image.format != "PNG":
                        raise ValueError("image is not PNG")
                    if image.mode != "RGB":
                        raise ValueError("image is not RGB")
                    if image.size != expected_size:
                        raise ValueError(
                            f"image size {image.size} does not match {expected_size}"
                        )
                    image.verify()
            except Exception as exc:
                raise ValueError(
                    f"sample {sample_id} image failed PNG/RGB/size validation"
                ) from exc

    def _validate_progress(
        self,
        *,
        target_samples: int,
        maximum_failures: int,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], int, int]:
        progress = self._read_json_object(
            self.progress_path, description="progress receipt"
        )
        expected_keys = {
            "schema_version",
            "target_samples",
            "maximum_failures",
            "generation_contract",
            "attempts",
            "successes",
            "failures",
            "rng_state",
        }
        if set(progress) != expected_keys:
            raise ValueError("progress receipt has unexpected or missing fields")
        if progress["schema_version"] != PROGRESS_SCHEMA_VERSION:
            raise ValueError("unsupported progress receipt schema")
        recorded_target = self._strict_nonnegative_int(
            progress["target_samples"], field="target_samples"
        )
        if recorded_target != target_samples:
            raise ValueError(
                f"resume target cannot change: {recorded_target} != {target_samples}"
            )
        recorded_maximum = self._strict_nonnegative_int(
            progress["maximum_failures"], field="maximum_failures"
        )
        if recorded_maximum != maximum_failures:
            raise ValueError(
                "resume maximum_failures cannot change: "
                f"{recorded_maximum} != {maximum_failures}"
            )
        if progress["generation_contract"] != self._generation_contract():
            raise ValueError("resume generation contract does not match the original")

        attempts = self._strict_nonnegative_int(
            progress["attempts"], field="attempts"
        )
        successes = self._strict_nonnegative_int(
            progress["successes"], field="successes"
        )
        failures = self._strict_nonnegative_int(
            progress["failures"], field="failures"
        )
        if successes != len(records):
            raise ValueError("progress successes do not match samples.jsonl")
        if attempts != successes + failures:
            raise ValueError("progress attempts must equal successes plus failures")
        if successes > target_samples:
            raise ValueError("progress contains more successes than the target")
        if failures > maximum_failures:
            raise ValueError("progress already exceeds the fixed failure budget")
        if records and records[-1]["attempt"] > attempts:
            raise ValueError("sample attempt exceeds the progress attempt count")
        if successes == target_samples and records[-1]["attempt"] != attempts:
            raise ValueError("a completed dataset cannot end with failed attempts")
        if not isinstance(progress["rng_state"], dict):
            raise ValueError("progress rng_state must be an object")
        return progress, attempts, failures

    def _replay_and_verify(
        self,
        records: list[dict[str, Any]],
        *,
        attempts: int,
        recorded_rng_state: dict[str, Any],
    ) -> None:
        records_by_attempt = {record["attempt"]: record for record in records}
        for attempt in range(1, attempts + 1):
            sample = self.sample()
            record = records_by_attempt.get(attempt)
            if record is None:
                continue
            expected_pose = [float(value) for value in sample.pose]
            if record["pose"] != expected_pose:
                raise ValueError(
                    f"sample {record['sample_id']} pose disagrees with seed replay"
                )
            expected_annotations = _json_normalized(dict(sample.annotations or {}))
            actual_annotations = record.get("annotations", {})
            if actual_annotations != expected_annotations:
                raise ValueError(
                    f"sample {record['sample_id']} annotations disagree with seed replay"
                )
            if bool(expected_annotations) != ("annotations" in record):
                raise ValueError(
                    f"sample {record['sample_id']} annotation presence disagrees with replay"
                )
        actual_rng_state = _json_normalized(self.rng.bit_generator.state)
        if actual_rng_state != recorded_rng_state:
            raise ValueError("progress rng_state disagrees with seed replay")

    def _save_image(self, sample_id: int, rgb: np.ndarray) -> tuple[str, str]:
        filename = f"frame_{sample_id:05d}.png"
        final_path = self.images_dir / filename
        temporary = final_path.with_suffix(".png.tmp")
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite image: {final_path}")
        with temporary.open("xb") as handle:
            Image.fromarray(rgb, mode="RGB").save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
        with Image.open(final_path) as saved:
            roundtrip = np.asarray(saved.convert("RGB"))
        if not np.array_equal(roundtrip, rgb):
            raise IOError(f"saved PNG round-trip mismatch: {final_path}")
        return f"images/{filename}", _sha256_file(final_path)

    def _save_progress(
        self,
        *,
        target_samples: int,
        maximum_failures: int,
        attempts: int,
        successes: int,
        failures: int,
    ) -> None:
        _atomic_json(
            self.progress_path,
            self._progress_payload(
                target_samples=target_samples,
                maximum_failures=maximum_failures,
                attempts=attempts,
                successes=successes,
                failures=failures,
            ),
        )

    def _build_metadata(
        self,
        records: list[dict[str, Any]],
        *,
        target_samples: int,
        maximum_failures: int,
        attempts: int,
        failures: int,
    ) -> dict[str, Any]:
        return _json_normalized(
            {
                "schema_version": 2,
                "n_frames": len(records),
                "target_samples": target_samples,
                "maximum_failures": maximum_failures,
                "track": self.track,
                "seed": self.seed,
                "attempts": attempts,
                "render_failures": failures,
                "bounds": asdict(self.bounds),
                "pose_format": ["x", "y", "z", "roll", "pitch", "yaw"],
                "pose_units": ["m", "m", "m", "rad", "rad", "rad"],
                "pose_coordinate_frame": "original_nerf_world",
                "body_axis_convention": "+X forward, +Y left, +Z up",
                "camera_axis_convention": (
                    "OpenCV +X right, +Y down, +Z forward; optical +Z equals body +X"
                ),
                "yaw_convention": (
                    "radians about original-world +Z; positive rotates body +X toward +Y"
                ),
                "image_format": "RGB uint8 PNG",
                "image_size": [self.intrinsics.width, self.intrinsics.height],
                "intrinsics": asdict(self.intrinsics),
                "poses": [record["pose"] for record in records],
                "samples_manifest": self.records_path.name,
                "provenance": self.provenance,
            }
        )

    def _load_resume_state(
        self, *, target_samples: int, maximum_failures: int
    ) -> tuple[list[dict[str, Any]], int, int, dict[str, Any] | None]:
        metadata_exists = self._validate_output_topology()
        records = self._load_records()
        progress, attempts, failures = self._validate_progress(
            target_samples=target_samples,
            maximum_failures=maximum_failures,
            records=records,
        )
        self._validate_images(records)
        self._replay_and_verify(
            records,
            attempts=attempts,
            recorded_rng_state=progress["rng_state"],
        )

        if metadata_exists and len(records) != target_samples:
            raise ValueError("partial resume output cannot contain metadata.json")
        if len(records) != target_samples:
            return records, attempts, failures, None

        expected_metadata = self._build_metadata(
            records,
            target_samples=target_samples,
            maximum_failures=maximum_failures,
            attempts=attempts,
            failures=failures,
        )
        if metadata_exists:
            actual_metadata = self._read_json_object(
                self.metadata_path, description="dataset metadata"
            )
            if actual_metadata != expected_metadata:
                raise ValueError("completed dataset metadata is inconsistent")
        else:
            _atomic_json(self.metadata_path, expected_metadata)
        return records, attempts, failures, expected_metadata

    def generate(
        self,
        successful_samples: int,
        *,
        maximum_failures: int | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        if (
            isinstance(successful_samples, (bool, np.bool_))
            or not isinstance(successful_samples, Integral)
            or successful_samples <= 0
        ):
            raise ValueError("successful_samples must be positive")
        successful_samples = int(successful_samples)
        maximum_failures = (
            max(10, successful_samples // 20)
            if maximum_failures is None
            else maximum_failures
        )
        if (
            isinstance(maximum_failures, (bool, np.bool_))
            or not isinstance(maximum_failures, Integral)
            or maximum_failures < 0
        ):
            raise ValueError("maximum_failures must be a non-negative integer")
        maximum_failures = int(maximum_failures)
        self.rng = np.random.default_rng(self.seed)
        # Validate the immutable contract before creating any output.
        self._generation_contract()

        records: list[dict[str, Any]] = []
        attempts = 0
        failures = 0
        output_is_empty = (
            not self.output_dir.exists()
            or (
                self.output_dir.is_dir()
                and not self.output_dir.is_symlink()
                and not any(self.output_dir.iterdir())
            )
        )
        if not resume or output_is_empty:
            self._initialize_output(
                target_samples=successful_samples,
                maximum_failures=maximum_failures,
            )
        else:
            records, attempts, failures, completed_metadata = self._load_resume_state(
                target_samples=successful_samples,
                maximum_failures=maximum_failures,
            )
            if completed_metadata is not None:
                return completed_metadata

        with self.records_path.open("a", encoding="utf-8", newline="\n") as record_file:
            while len(records) < successful_samples:
                sample = self.sample()
                pose = sample.pose
                attempts += 1
                try:
                    rendered = self.renderer.render_rgb(
                        pose_to_camera_matrix(pose)
                    )
                    rgb = normalize_rgb(
                        rendered,
                        width=self.intrinsics.width,
                        height=self.intrinsics.height,
                    )
                except Exception:
                    failures += 1
                    self._save_progress(
                        target_samples=successful_samples,
                        maximum_failures=maximum_failures,
                        attempts=attempts,
                        successes=len(records),
                        failures=failures,
                    )
                    if failures > maximum_failures:
                        raise RuntimeError(
                            f"renderer exceeded failure budget: {failures} > "
                            f"{maximum_failures}"
                        )
                    continue

                sample_id = len(records)
                relative_image, image_sha256 = self._save_image(sample_id, rgb)
                record = {
                    "sample_id": sample_id,
                    "image": relative_image,
                    "image_sha256": image_sha256,
                    "pose": [float(value) for value in pose],
                    "attempt": attempts,
                }
                if sample.annotations:
                    record["annotations"] = _json_normalized(
                        dict(sample.annotations)
                    )
                record_file.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                record_file.flush()
                os.fsync(record_file.fileno())
                records.append(record)
                self._save_progress(
                    target_samples=successful_samples,
                    maximum_failures=maximum_failures,
                    attempts=attempts,
                    successes=len(records),
                    failures=failures,
                )

        metadata = self._build_metadata(
            records,
            target_samples=successful_samples,
            maximum_failures=maximum_failures,
            attempts=attempts,
            failures=failures,
        )
        _atomic_json(self.metadata_path, metadata)
        return metadata
