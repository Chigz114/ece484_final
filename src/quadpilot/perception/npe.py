"""Reproducible Neural Pose Estimator (NPE) training primitives.

The historical Quad Pilots model consumes an RGB image and regresses a pose
label stored as ``[x, y, z, roll, pitch, yaw]``.  Roll and pitch are fixed to
zero in the generated data, so the network head intentionally remains the
original five values: normalized ``x/y/z`` plus ``sin(yaw)/cos(yaw)``.

This module is deliberately independent of the renderer.  It accepts both the
label-safe ``samples.jsonl`` datasets produced by :mod:`data_generation` and
the legacy ``metadata.json`` layout, but rejects legacy datasets whose frame
numbers and pose indices do not match exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CHECKPOINT_SCHEMA_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
MODEL_OUTPUT_FORMAT = (
    "x_normalized",
    "y_normalized",
    "z_normalized",
    "sin_yaw",
    "cos_yaw",
)
MODEL_OUTPUT_SPACE = "normalized_xyz_sincos"
POSE_FORMAT = ("x", "y", "z", "roll", "pitch", "yaw")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def atomic_json_save(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_torch_checkpoint(
    path: str | Path, map_location: Any = "cpu"
) -> dict[str, Any]:
    """Load a trusted local checkpoint across PyTorch 2.1 and newer APIs."""

    try:
        checkpoint = torch.load(
            Path(path), map_location=map_location, weights_only=False
        )
    except TypeError:  # PyTorch before ``weights_only`` was added.
        checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return checkpoint


@dataclass(frozen=True)
class DatasetRecord:
    """One verified image/pose pair."""

    key: str
    source_id: str
    source_root: Path
    relative_image: str
    pose: tuple[float, float, float, float, float, float]
    image_sha256: str
    width: int
    height: int

    @property
    def image_path(self) -> Path:
        return self.source_root / Path(self.relative_image)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "pose": list(self.pose),
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class DatasetIndex:
    records: tuple[DatasetRecord, ...]
    fingerprint: str
    fingerprint_mode: str
    sources: tuple[dict[str, Any], ...]

    def by_key(self) -> dict[str, DatasetRecord]:
        return {record.key: record for record in self.records}


def _safe_relative_image(root: Path, value: Any) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid image path in dataset manifest: {value!r}")
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image path escapes dataset root: {value!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path escapes dataset root: {value!r}") from exc
    return relative.as_posix(), candidate


def _validated_pose(
    value: Any, *, context: str
) -> tuple[float, float, float, float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{context} must be six finite pose values in {POSE_FORMAT}")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _read_source_samples(
    root: Path, metadata: Mapping[str, Any]
) -> list[tuple[str, tuple[float, ...]]]:
    poses = metadata.get("poses")
    manifest_name = metadata.get("samples_manifest")
    samples: list[tuple[str, tuple[float, ...]]] = []

    if manifest_name:
        manifest_relative, manifest_path = _safe_relative_image(root, manifest_name)
        del manifest_relative
        if not manifest_path.is_file():
            raise FileNotFoundError(f"samples manifest does not exist: {manifest_path}")
        seen_ids: set[int] = set()
        seen_images: set[str] = set()
        id_to_entry: dict[int, tuple[str, tuple[float, ...]]] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {manifest_path}:{line_number}"
                    ) from exc
                sample_id = record.get("sample_id")
                if (
                    not isinstance(sample_id, int)
                    or sample_id < 0
                    or sample_id in seen_ids
                ):
                    raise ValueError(
                        f"invalid or duplicate sample_id at {manifest_path}:{line_number}"
                    )
                relative, _ = _safe_relative_image(root, record.get("image"))
                if relative in seen_images:
                    raise ValueError(f"duplicate image in samples manifest: {relative}")
                pose = _validated_pose(
                    record.get("pose"), context=f"{manifest_path}:{line_number} pose"
                )
                seen_ids.add(sample_id)
                seen_images.add(relative)
                id_to_entry[sample_id] = (relative, pose)

        if sorted(seen_ids) != list(range(len(id_to_entry))):
            raise ValueError("samples.jsonl sample IDs must be continuous from zero")
        # Sample IDs, rather than line order or a filename convention, are the
        # authoritative pose/image index.
        samples = [id_to_entry[index] for index in range(len(id_to_entry))]
        if poses is not None:
            if len(poses) != len(samples):
                raise ValueError(
                    "metadata poses and samples manifest have different lengths"
                )
            for index, (_, pose) in enumerate(samples):
                metadata_pose = _validated_pose(
                    poses[index], context=f"metadata pose {index}"
                )
                if not np.allclose(metadata_pose, pose, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"metadata pose {index} disagrees with samples manifest"
                    )
    else:
        if not isinstance(poses, list) or not poses:
            raise ValueError(
                f"legacy dataset has no non-empty poses list: {root / 'metadata.json'}"
            )
        expected = {f"frame_{index:05d}.png" for index in range(len(poses))}
        images_dir = root / "images"
        actual = (
            {path.name for path in images_dir.glob("frame_*.png")}
            if images_dir.is_dir()
            else set()
        )
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(
                "legacy frame/pose indices are not one-to-one; "
                f"missing={missing}, extra={extra}. Regenerate with the label-safe generator."
            )
        samples = [
            (
                f"images/frame_{index:05d}.png",
                _validated_pose(pose, context=f"metadata pose {index}"),
            )
            for index, pose in enumerate(poses)
        ]

    declared = metadata.get("n_frames")
    if declared is not None and int(declared) != len(samples):
        raise ValueError(
            f"metadata n_frames={declared} but discovered {len(samples)} samples"
        )
    if not samples:
        raise ValueError(f"dataset contains no samples: {root}")
    return samples


def build_dataset_index(
    data_dirs: Sequence[str | Path],
    *,
    fingerprint_mode: str = "full",
    verify_images: bool = True,
) -> DatasetIndex:
    """Validate datasets and calculate a deterministic content fingerprint.

    ``fingerprint_mode='full'`` hashes every image byte and is the default.
    The explicit ``'stat'`` mode is faster but weaker; it records file size and
    nanosecond modification time and is visibly marked in all provenance.
    """

    if fingerprint_mode not in {"full", "stat"}:
        raise ValueError("fingerprint_mode must be 'full' or 'stat'")
    if not data_dirs:
        raise ValueError("at least one data directory is required")

    records: list[DatasetRecord] = []
    sources: list[dict[str, Any]] = []
    seen_roots: set[Path] = set()
    for source_number, directory in enumerate(data_dirs):
        root = Path(directory).expanduser().resolve()
        if root in seen_roots:
            raise ValueError(f"duplicate dataset directory: {root}")
        seen_roots.add(root)
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"dataset metadata does not exist: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata root must be an object: {metadata_path}")

        source_id = f"{source_number}:{root.name}"
        samples = _read_source_samples(root, metadata)
        expected_size = metadata.get("image_size")
        if expected_size is not None:
            if not isinstance(expected_size, list) or len(expected_size) != 2:
                raise ValueError(f"invalid image_size in {metadata_path}")
            expected_width, expected_height = map(int, expected_size)
        else:
            expected_width = expected_height = None

        source_records: list[DatasetRecord] = []
        for sample_number, (relative_image, pose) in enumerate(samples):
            relative_image, image_path = _safe_relative_image(root, relative_image)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"image does not exist for sample {sample_number}: {image_path}"
                )
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    if verify_images:
                        image.verify()
            except Exception as exc:
                raise ValueError(f"invalid image file: {image_path}") from exc
            if expected_width is not None and (width, height) != (
                expected_width,
                expected_height,
            ):
                raise ValueError(
                    f"image size mismatch for {image_path}: {(width, height)} != "
                    f"{(expected_width, expected_height)}"
                )
            if fingerprint_mode == "full":
                image_digest = sha256_file(image_path)
            else:
                stat = image_path.stat()
                image_digest = f"stat:{stat.st_size}:{stat.st_mtime_ns}"
            key = f"{source_id}/{relative_image}"
            source_records.append(
                DatasetRecord(
                    key=key,
                    source_id=source_id,
                    source_root=root,
                    relative_image=relative_image,
                    pose=pose,  # type: ignore[arg-type]
                    image_sha256=image_digest,
                    width=width,
                    height=height,
                )
            )
        records.extend(source_records)
        sources.append(
            {
                "source_id": source_id,
                "path": str(root),
                "track": metadata.get("track"),
                "schema_version": metadata.get("schema_version", "legacy"),
                "metadata_sha256": sha256_file(metadata_path),
                "sample_count": len(source_records),
            }
        )

    if len({record.key for record in records}) != len(records):
        raise ValueError("dataset record keys are not unique")
    fingerprint_payload = {
        "fingerprint_mode": fingerprint_mode,
        "sources": [
            {
                "source_id": source["source_id"],
                "track": source["track"],
                "schema_version": source["schema_version"],
                "metadata_sha256": source["metadata_sha256"],
                "sample_count": source["sample_count"],
            }
            for source in sources
        ],
        "records": [record.fingerprint_payload() for record in records],
    }
    return DatasetIndex(
        records=tuple(records),
        fingerprint=sha256_payload(fingerprint_payload),
        fingerprint_mode=fingerprint_mode,
        sources=tuple(sources),
    )


def _split_counts(n_samples: int, ratios: Sequence[float]) -> list[int]:
    if n_samples <= 0:
        raise ValueError("cannot split an empty dataset source")
    values = np.asarray(ratios, dtype=np.float64)
    if (
        values.shape != (3,)
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, atol=1e-12)
    ):
        raise ValueError("train/val/test ratios must be non-negative and sum to one")
    raw = values * n_samples
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts), kind="stable")[
        : n_samples - int(counts.sum())
    ]:
        counts[index] += 1
    positive = np.flatnonzero(values > 0)
    if n_samples >= len(positive):
        for empty in [index for index in positive if counts[index] == 0]:
            donors = [index for index in positive if counts[index] > 1]
            if not donors:
                break
            donor = max(
                donors, key=lambda index: (counts[index], values[index], -index)
            )
            counts[donor] -= 1
            counts[empty] += 1
    if int(counts.sum()) != n_samples:
        raise AssertionError("split allocation did not preserve sample count")
    return counts.tolist()


def create_or_load_split_manifest(
    path: str | Path,
    dataset: DatasetIndex,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    create: bool = True,
) -> dict[str, Any]:
    """Create a deterministic, source-balanced split or validate an existing one."""

    destination = Path(path)
    if destination.is_file():
        with destination.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_split_manifest(manifest, dataset)
        return manifest
    if not create:
        raise FileNotFoundError(f"split manifest does not exist: {destination}")

    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    _split_counts(1, ratios)  # ratio validation
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    source_ids = sorted({record.source_id for record in dataset.records})
    for source_id in source_ids:
        source_records = [
            record for record in dataset.records if record.source_id == source_id
        ]
        ranked = sorted(
            source_records,
            key=lambda record: hashlib.sha256(
                f"{int(seed)}:{record.key}".encode("utf-8")
            ).digest(),
        )
        train_count, val_count, test_count = _split_counts(len(ranked), ratios)
        splits["train"].extend(record.key for record in ranked[:train_count])
        splits["val"].extend(
            record.key for record in ranked[train_count : train_count + val_count]
        )
        splits["test"].extend(
            record.key for record in ranked[train_count + val_count :]
        )
        if len(ranked) != train_count + val_count + test_count:
            raise AssertionError("split slicing error")

    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "fingerprint_mode": dataset.fingerprint_mode,
        "seed": int(seed),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "counts": {name: len(keys) for name, keys in splits.items()},
        "splits": splits,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    validate_split_manifest(manifest, dataset)
    atomic_json_save(destination, manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any], dataset: DatasetIndex) -> None:
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported split schema: {manifest.get('schema_version')}")
    if manifest.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError(
            "split manifest dataset fingerprint does not match current dataset"
        )
    if manifest.get("fingerprint_mode") != dataset.fingerprint_mode:
        raise ValueError(
            "split manifest fingerprint mode does not match current dataset"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("split manifest must contain exactly train/val/test")
    all_keys: list[str] = []
    for name in ("train", "val", "test"):
        keys = splits[name]
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"split {name} is not a list of record keys")
        all_keys.extend(keys)
    expected = {record.key for record in dataset.records}
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("split manifest contains duplicate/leaked records")
    if set(all_keys) != expected:
        missing = sorted(expected - set(all_keys))[:5]
        extra = sorted(set(all_keys) - expected)[:5]
        raise ValueError(
            f"split manifest does not cover dataset exactly; missing={missing}, extra={extra}"
        )
    stored_hash = manifest.get("manifest_sha256")
    if stored_hash:
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256", None)
        if stored_hash != sha256_payload(unhashed):
            raise ValueError("split manifest checksum is invalid")


def records_for_split(
    dataset: DatasetIndex,
    manifest: Mapping[str, Any],
    split: str,
) -> tuple[DatasetRecord, ...]:
    validate_split_manifest(manifest, dataset)
    if split == "all":
        return dataset.records
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, test, or all")
    by_key = dataset.by_key()
    return tuple(by_key[key] for key in manifest["splits"][split])


def filter_records_by_source_ids(
    records: Sequence[DatasetRecord],
    dataset: DatasetIndex,
    source_ids: Sequence[str],
) -> tuple[DatasetRecord, ...]:
    """Filter an already-frozen split by exact dataset source identifiers.

    The caller must obtain ``records`` from :func:`records_for_split` before
    applying this helper.  Filtering never creates or reshuffles a split, so a
    gate-focused evaluation cannot accidentally include that source's training
    or validation records.
    """

    requested = tuple(str(source_id) for source_id in source_ids)
    if not requested:
        return tuple(records)
    if len(requested) != len(set(requested)):
        raise ValueError("source filter contains duplicate source_id values")

    available = {
        str(source["source_id"])
        for source in dataset.sources
        if isinstance(source, Mapping) and "source_id" in source
    }
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(
            f"source filter contains unknown source_id values: {unknown}; "
            f"available={sorted(available)}"
        )

    selected = set(requested)
    filtered = tuple(record for record in records if record.source_id in selected)
    if not filtered:
        raise ValueError(
            "source filter selected zero records from the frozen split; "
            f"requested={list(requested)}"
        )
    return filtered


@dataclass(frozen=True)
class PoseNormalizer:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    epsilon: float = 1e-6

    @classmethod
    def fit(
        cls, records: Sequence[DatasetRecord], epsilon: float = 1e-6
    ) -> "PoseNormalizer":
        if not records:
            raise ValueError("cannot fit a pose normalizer without training records")
        positions = np.asarray(
            [record.pose[:3] for record in records], dtype=np.float64
        )
        mean = positions.mean(axis=0)
        std = positions.std(axis=0, ddof=0)
        std = np.maximum(std, float(epsilon))
        return cls(tuple(mean.tolist()), tuple(std.tolist()), float(epsilon))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseNormalizer":
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),  # type: ignore[arg-type]
            std=tuple(float(value) for value in payload["std"]),  # type: ignore[arg-type]
            epsilon=float(payload.get("epsilon", 1e-6)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def encode_pose(self, pose: Sequence[float]) -> torch.Tensor:
        values = _validated_pose(pose, context="pose")
        position = (np.asarray(values[:3]) - np.asarray(self.mean)) / np.asarray(
            self.std
        )
        yaw = values[5]
        return torch.tensor(
            [position[0], position[1], position[2], math.sin(yaw), math.cos(yaw)],
            dtype=torch.float32,
        )

    def decode_positions(self, normalized_positions: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.mean,
            dtype=normalized_positions.dtype,
            device=normalized_positions.device,
        )
        std = torch.as_tensor(
            self.std,
            dtype=normalized_positions.dtype,
            device=normalized_positions.device,
        )
        return normalized_positions * std + mean

    def decode_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 2 or outputs.shape[1] != 5:
            raise ValueError("model outputs must have shape [N,5]")
        positions = self.decode_positions(outputs[:, :3])
        yaw = torch.atan2(outputs[:, 3], outputs[:, 4]).unsqueeze(1)
        return torch.cat([positions, yaw], dim=1)


@dataclass(frozen=True)
class PreprocessConfig:
    width: int = 224
    height: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    brightness: float = 0.05
    contrast: float = 0.05
    saturation: float = 0.02
    interpolation: str = "bilinear"
    antialias: bool = True
    color_space: str = "RGB"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreprocessConfig":
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            mean=tuple(float(value) for value in payload["mean"]),  # type: ignore[arg-type]
            std=tuple(float(value) for value in payload["std"]),  # type: ignore[arg-type]
            brightness=float(payload.get("brightness", 0.0)),
            contrast=float(payload.get("contrast", 0.0)),
            saturation=float(payload.get("saturation", 0.0)),
            interpolation=str(payload.get("interpolation", "bilinear")),
            antialias=bool(payload.get("antialias", True)),
            color_space=str(payload.get("color_space", "RGB")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_image_transform(config: PreprocessConfig, *, training: bool) -> Any:
    """Build torchvision preprocessing lazily to keep indexing lightweight."""

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if config.interpolation != "bilinear":
        raise ValueError(f"unsupported resize interpolation: {config.interpolation}")
    if config.color_space != "RGB":
        raise ValueError(f"unsupported image color space: {config.color_space}")

    operations: list[Any] = [
        transforms.Resize(
            (config.height, config.width),
            interpolation=InterpolationMode.BILINEAR,
            antialias=config.antialias,
        ),
    ]
    if training and any((config.brightness, config.contrast, config.saturation)):
        operations.append(
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
    return transforms.Compose(operations)


class NPEImageDataset(Dataset):
    def __init__(
        self,
        records: Sequence[DatasetRecord],
        normalizer: PoseNormalizer,
        transform: Any,
    ) -> None:
        if not records:
            raise ValueError("NPEImageDataset cannot be empty")
        self.records = tuple(records)
        self.normalizer = normalizer
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb) if self.transform is not None else rgb.copy()
        return tensor, self.normalizer.encode_pose(record.pose), record.key


def _torchvision_model(backbone: str, weights: str) -> tuple[nn.Module, int]:
    from torchvision import models

    choices = {
        "resnet18": (models.resnet18, models.ResNet18_Weights, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights, 2048),
    }
    if backbone not in choices:
        raise ValueError(
            f"unsupported backbone {backbone!r}; expected {sorted(choices)}"
        )
    constructor, enum_class, feature_dim = choices[backbone]
    normalized_weights = weights.lower()
    if normalized_weights == "none":
        resolved_weights = None
    elif normalized_weights == "imagenet1k_v1":
        # This may download weights.  It can only happen following this explicit
        # option; the default path above never performs network access.
        resolved_weights = enum_class.IMAGENET1K_V1
    else:
        raise ValueError("weights must be 'none' or the explicit 'imagenet1k_v1'")
    return constructor(weights=resolved_weights), feature_dim


def torchvision_weight_provenance(backbone: str, weights: str) -> dict[str, Any]:
    """Describe and hash an explicitly selected torchvision weight artifact."""

    normalized_weights = weights.lower()
    if normalized_weights == "none":
        return {"identifier": "none", "network_download_requested": False}
    from torchvision import models

    enums = {
        "resnet18": models.ResNet18_Weights,
        "resnet34": models.ResNet34_Weights,
        "resnet50": models.ResNet50_Weights,
    }
    if backbone not in enums or normalized_weights != "imagenet1k_v1":
        raise ValueError("unsupported torchvision weight provenance request")
    selected = enums[backbone].IMAGENET1K_V1
    filename = Path(urllib.parse.urlparse(selected.url).path).name
    cached_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached_path.is_file():
        raise FileNotFoundError(
            "torchvision constructed with ImageNet weights but its cache artifact "
            f"cannot be found for hashing: {cached_path}"
        )
    return {
        "identifier": f"{backbone}.IMAGENET1K_V1",
        "network_download_requested": True,
        "source_url": selected.url,
        "cache_path": str(cached_path.resolve()),
        "sha256": sha256_file(cached_path),
        "bytes": cached_path.stat().st_size,
    }


class NPEModel(nn.Module):
    """Historical ResNet NPE architecture with a five-value pose head."""

    def __init__(self, *, backbone: str = "resnet50", weights: str = "none") -> None:
        super().__init__()
        base, feature_dim = _torchvision_model(backbone, weights)
        self.backbone_name = backbone
        self.initial_weights = weights.lower()
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "architecture": "resnet_npe_v1",
            "backbone": self.backbone_name,
            "output_format": list(MODEL_OUTPUT_FORMAT),
            "output_space": MODEL_OUTPUT_SPACE,
            "initial_weights": self.initial_weights,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(images))


def convert_legacy_state_to_normalized_outputs(
    state_dict: Mapping[str, torch.Tensor],
    normalizer: PoseNormalizer,
) -> dict[str, torch.Tensor]:
    """Convert a historical raw-XYZ NPE head to normalized XYZ outputs.

    The operation is exactly affine: for the last layer rows that once emitted
    ``x``, ``y`` and ``z``, it substitutes ``(value - mean) / std``.  This lets
    an explicitly supplied legacy checkpoint seed the new fine-tuning chain
    without silently changing its physical predictions.
    """

    converted = {name: value.detach().clone() for name, value in state_dict.items()}
    weight_candidates = [
        name for name in converted if name.endswith("regressor.7.weight")
    ]
    bias_candidates = [name for name in converted if name.endswith("regressor.7.bias")]
    if len(weight_candidates) != 1 or len(bias_candidates) != 1:
        raise ValueError(
            "legacy state does not contain one unambiguous regressor.7 output layer"
        )
    weight = converted[weight_candidates[0]]
    bias = converted[bias_candidates[0]]
    if weight.ndim != 2 or weight.shape[0] != 5 or bias.shape != (5,):
        raise ValueError("legacy NPE output layer must have five outputs")
    mean = torch.as_tensor(normalizer.mean, dtype=weight.dtype, device=weight.device)
    std = torch.as_tensor(normalizer.std, dtype=weight.dtype, device=weight.device)
    weight[:3] = weight[:3] / std[:, None]
    bias[:3] = (bias[:3] - mean) / std
    return converted


def convert_state_dict_to_legacy_raw_xyz(
    state_dict: Mapping[str, torch.Tensor],
    normalizer: PoseNormalizer,
) -> dict[str, torch.Tensor]:
    """Return an explicit export state whose first three outputs are raw XYZ.

    This is the inverse of :func:`convert_legacy_state_to_normalized_outputs`.
    It is intentionally never embedded in every repro checkpoint; callers must
    opt into a separate legacy-controller export and label it accordingly.
    """

    converted = {name: value.detach().clone() for name, value in state_dict.items()}
    weight_candidates = [
        name for name in converted if name.endswith("regressor.7.weight")
    ]
    bias_candidates = [name for name in converted if name.endswith("regressor.7.bias")]
    if len(weight_candidates) != 1 or len(bias_candidates) != 1:
        raise ValueError(
            "normalized state does not contain one unambiguous regressor.7 output layer"
        )
    weight = converted[weight_candidates[0]]
    bias = converted[bias_candidates[0]]
    if weight.ndim != 2 or weight.shape[0] != 5 or bias.shape != (5,):
        raise ValueError("normalized NPE output layer must have five outputs")
    mean = torch.as_tensor(normalizer.mean, dtype=weight.dtype, device=weight.device)
    std = torch.as_tensor(normalizer.std, dtype=weight.dtype, device=weight.device)
    weight[:3] = weight[:3] * std[:, None]
    bias[:3] = bias[:3] * std + mean
    return converted


@dataclass(frozen=True)
class PoseLossConfig:
    position_weight: float = 1.0
    orientation_weight: float = 0.5
    orientation_norm_weight: float = 0.01

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseLossConfig":
        return cls(**{key: float(value) for key, value in payload.items()})


def pose_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: PoseLossConfig = PoseLossConfig(),
) -> dict[str, torch.Tensor]:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 5
    ):
        raise ValueError("prediction and target must both have shape [N,5]")
    position = torch.mean((prediction[:, :3] - target[:, :3]) ** 2, dim=1)
    orientation = torch.mean((prediction[:, 3:] - target[:, 3:]) ** 2, dim=1)
    orientation_norm = (torch.linalg.vector_norm(prediction[:, 3:], dim=1) - 1.0) ** 2
    total = (
        config.position_weight * position
        + config.orientation_weight * orientation
        + config.orientation_norm_weight * orientation_norm
    )
    return {
        "total": total,
        "position": position,
        "orientation": orientation,
        "orientation_norm": orientation_norm,
    }


def wrapped_angle_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def pose_error_vectors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalizer: PoseNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_pose = normalizer.decode_outputs(prediction)
    target_pose = normalizer.decode_outputs(target)
    position_cm = (
        torch.linalg.vector_norm(predicted_pose[:, :3] - target_pose[:, :3], dim=1)
        * 100.0
    )
    yaw_deg = torch.abs(
        wrapped_angle_difference(predicted_pose[:, 3], target_pose[:, 3])
    ) * (180.0 / math.pi)
    return position_cm, yaw_deg


def decode_predictions(
    outputs: torch.Tensor,
    normalizer: PoseNormalizer,
    *,
    minimum_orientation_norm: float = 1e-8,
) -> torch.Tensor:
    """Decode normalized network outputs to ``[x, y, z, yaw]``.

    A near-zero sine/cosine vector has no defined angle and is rejected rather
    than silently turning into yaw zero in a closed-loop controller.
    """

    if outputs.ndim != 2 or outputs.shape[1] != 5:
        raise ValueError("NPE outputs must have shape [N,5]")
    if not torch.isfinite(outputs).all():
        raise ValueError("NPE outputs contain NaN or infinity")
    orientation_norm = torch.linalg.vector_norm(outputs[:, 3:], dim=1)
    if torch.any(orientation_norm < minimum_orientation_norm):
        raise ValueError("NPE emitted a degenerate sin/cos yaw vector")
    return normalizer.decode_outputs(outputs)


@dataclass(frozen=True)
class PredictionBatch:
    normalized_output: torch.Tensor
    xyz_yaw: torch.Tensor


@torch.inference_mode()
def predict_poses(
    model: nn.Module,
    images: torch.Tensor,
    normalizer: PoseNormalizer,
    *,
    device: torch.device | str,
    amp_enabled: bool = False,
) -> PredictionBatch:
    """Run offline-safe NPE inference and decode physical NeRF coordinates."""

    device = torch.device(device)
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP pose prediction requires CUDA")
    model.eval()
    images = images.to(device, non_blocking=True)
    with _autocast(amp_enabled):
        output = model(images)
    decoded = decode_predictions(output, normalizer)
    return PredictionBatch(normalized_output=output, xyz_yaw=decoded)


class MetricAccumulator:
    """Accumulate exact sample-weighted losses and pose-error distributions."""

    def __init__(self) -> None:
        self.count = 0
        self.loss_sums = {
            name: 0.0
            for name in ("total", "position", "orientation", "orientation_norm")
        }
        self.position_errors_cm: list[float] = []
        self.yaw_errors_deg: list[float] = []

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        normalizer: PoseNormalizer,
        loss_config: PoseLossConfig = PoseLossConfig(),
    ) -> None:
        components = pose_loss_components(
            prediction.detach(), target.detach(), loss_config
        )
        position, yaw = pose_error_vectors(
            prediction.detach(), target.detach(), normalizer
        )
        batch_size = int(prediction.shape[0])
        self.count += batch_size
        for name, values in components.items():
            self.loss_sums[name] += float(values.double().sum().cpu())
        self.position_errors_cm.extend(
            float(value) for value in position.double().cpu()
        )
        self.yaw_errors_deg.extend(float(value) for value in yaw.double().cpu())

    @staticmethod
    def _distribution(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
            "rmse": float(np.sqrt(np.mean(array**2))),
        }

    def compute(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("cannot compute metrics without samples")
        return {
            "sample_count": self.count,
            "loss": {
                name: value / self.count for name, value in self.loss_sums.items()
            },
            "position_error_cm": self._distribution(self.position_errors_cm),
            "yaw_error_deg": self._distribution(self.yaw_errors_deg),
        }


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
        persistent_workers=False,
    )


def _autocast(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    normalizer: PoseNormalizer,
    loss_config: PoseLossConfig,
    accumulation_steps: int = 1,
    amp_enabled: bool = False,
    scaler: Any | None = None,
    max_grad_norm: float | None = None,
) -> tuple[dict[str, Any], int]:
    """Train one epoch with sample-exact gradient accumulation."""

    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP is only supported on CUDA in this reproduction chain")
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulator = MetricAccumulator()
    accumulated_samples = 0
    optimizer_steps = 0

    for batch_index, (images, targets, _keys) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(amp_enabled):
            prediction = model(images)
            components = pose_loss_components(prediction, targets, loss_config)
            summed_loss = components["total"].sum()
        scaler.scale(summed_loss).backward()
        accumulated_samples += int(images.shape[0])
        accumulator.update(prediction, targets, normalizer, loss_config)

        is_boundary = (batch_index + 1) % accumulation_steps == 0
        is_last = batch_index + 1 == len(loader)
        if is_boundary or is_last:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(accumulated_samples))
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
            optimizer_steps += 1

    return accumulator.compute(), optimizer_steps


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    normalizer: PoseNormalizer,
    loss_config: PoseLossConfig,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP is only supported on CUDA in this reproduction chain")
    model.eval()
    accumulator = MetricAccumulator()
    for images, targets, _keys in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(amp_enabled):
            prediction = model(images)
        accumulator.update(prediction, targets, normalizer, loss_config)
    return accumulator.compute()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _rng_state_on_cpu(value: Any, *, label: str) -> torch.Tensor:
    """Normalize a serialized RNG state to the CPU ByteTensor API contract.

    ``torch.load(..., map_location="cuda")`` moves every tensor in a
    checkpoint, including the CPU and CUDA generator states, onto CUDA.
    PyTorch's RNG restoration APIs nevertheless require CPU ByteTensors.
    Keeping this conversion explicit makes GPU resume behave like CPU resume
    without silently accepting malformed checkpoint state.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} RNG state must be a torch.Tensor")
    normalized = value.detach().cpu()
    if normalized.dtype != torch.uint8 or normalized.ndim != 1:
        raise ValueError(
            f"{label} RNG state must be a one-dimensional torch.uint8 tensor"
        )
    return normalized.contiguous()


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_rng_state_on_cpu(state["torch_cpu"], label="CPU"))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("CUDA RNG states must be a list or tuple")
        torch.cuda.set_rng_state_all(
            [
                _rng_state_on_cpu(item, label=f"CUDA[{index}]")
                for index, item in enumerate(cuda_states)
            ]
        )


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": Image.__version__,
    }
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        versions["torchvision"] = f"unavailable: {type(exc).__name__}"
    return versions


def validate_repro_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported NPE checkpoint schema: {checkpoint.get('schema_version')}"
        )
    required = {
        "model_config",
        "model_state_dict",
        "normalizer",
        "preprocess",
        "loss_config",
        "dataset",
        "split_manifest",
        "provenance",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"NPE checkpoint is missing required provenance/state: {missing}"
        )
    if checkpoint["model_config"].get("output_format") != list(MODEL_OUTPUT_FORMAT):
        raise ValueError("checkpoint model output format is incompatible")
    if checkpoint["model_config"].get("output_space") != MODEL_OUTPUT_SPACE:
        raise ValueError(
            "checkpoint output_space is not normalized_xyz_sincos; refusing ambiguous XYZ decoding"
        )


def model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[NPEModel, PoseNormalizer, PreprocessConfig, PoseLossConfig]:
    validate_repro_checkpoint(checkpoint)
    model_config = checkpoint["model_config"]
    # Never use the historical initialization weights here: all trained
    # parameters are present in the checkpoint and loading must stay offline.
    model = NPEModel(backbone=model_config["backbone"], weights="none")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    normalizer = PoseNormalizer.from_dict(checkpoint["normalizer"])
    preprocess = PreprocessConfig.from_dict(checkpoint["preprocess"])
    loss_config = PoseLossConfig.from_dict(checkpoint["loss_config"])
    return model, normalizer, preprocess, loss_config


@dataclass(frozen=True)
class LoadedNPECheckpoint:
    path: Path
    sha256: str
    checkpoint: dict[str, Any]
    model: NPEModel
    normalizer: PoseNormalizer
    preprocess: PreprocessConfig
    loss_config: PoseLossConfig


def load_repro_npe_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedNPECheckpoint:
    """Load, schema-check, and instantiate one local reproducible NPE."""

    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    validate_repro_checkpoint(checkpoint)
    model, normalizer, preprocess, loss_config = model_from_checkpoint(
        checkpoint, device=device
    )
    return LoadedNPECheckpoint(
        path=checkpoint_path,
        sha256=sha256_file(checkpoint_path),
        checkpoint=checkpoint,
        model=model,
        normalizer=normalizer,
        preprocess=preprocess,
        loss_config=loss_config,
    )
