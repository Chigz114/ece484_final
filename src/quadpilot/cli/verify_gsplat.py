#!/usr/bin/env python3
"""Fail-closed, CPU-only verification of a completed pinned GSplat run.

This tool is intentionally independent of ``reproduce_gsplat_docker.sh``.  It
exists for the narrow recovery case where Nerfstudio finished and wrote the
final checkpoint, but the outer wrapper subsequently failed.  It never starts
Docker, imports Nerfstudio, touches CUDA, or edits ``status.env``.  A successful
verification keeps the two relevant facts separate:

* the original wrapper status remains failed; and
* the independently checked training artifacts pass.

Only ``recovered-postflight.json`` may be added, and only after every gate has
passed.  Existing recovery reports are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

IMAGE_REF = (
    "dromni/nerfstudio@sha256:"
    "ff0107a7db96bb8ee29c638729328b832b268b890c50f2a2ff25988bb84d4f75"
)
IMAGE_DIGEST = IMAGE_REF.rsplit("@", 1)[1]
LPIPS_SIZE_BYTES = 244_408_911
LPIPS_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_RELATIVE_PATH = "torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
EXPECTED_VERSIONS = {
    "nerfstudio": "1.1.4",
    "gsplat": "1.0.0",
    "torch": "2.1.2+cu118",
    "torchvision": "0.16.2+cu118",
    "viser": "0.2.3",
}
EXPECTED_PIP_DEVIATIONS = [
    "ninja 1.11.1.1 is not supported on this platform",
    "rawpy 0.22.0 has requirement numpy>=2.0, but you have numpy 1.26.4.",
]
EXPECTED_NERFSTUDIO_SOURCES = {
    "/home/user/nerfstudio/nerfstudio/configs/method_configs.py": (
        "b004bcf9e7ba5de52d94138a86aae260c58dbd751eb901ae41f0ab3f75a22718"
    ),
    "/home/user/nerfstudio/nerfstudio/plugins/registry.py": (
        "2c955c48ff6b42e7823c90fc36dca9344fc6010c511238337b1528aa36c6930f"
    ),
    "/home/user/nerfstudio/nerfstudio/scripts/train.py": (
        "2a3b31c832427ca6c56b068a9b18039ea616834da5046852e0231c9df1b6d3c9"
    ),
}
EXPECTED_METHOD_ENTRY_POINTS = (
    (
        "bionerf",
        "bionerf.bionerf_config:bionerf_method",
        "bionerf",
        "1.0",
        "/home/user/.local/lib/python3.10/site-packages/bionerf-1.0.dist-info",
        "8dd1975af4901f2d5c8e0f1ec9401e4bae1f016d36c84b049c80f46de0f204f6",
    ),
    (
        "igs2gs",
        "igs2gs.igs2gs_config:igs2gs_method",
        "igs2gs",
        "0.1.0",
        "/home/user/.local/lib/python3.10/site-packages/igs2gs-0.1.0.dist-info",
        "cbdea16aec02408ea1850f18d8c3d21d6a16aeee0e4d759ae255edff724f7a21",
    ),
    (
        "kplanes",
        "kplanes.kplanes_configs:kplanes_method",
        "kplanes_nerfstudio",
        "0.5.2",
        "/home/user/.local/lib/python3.10/site-packages/kplanes_nerfstudio-0.5.2.dist-info",
        "aa826943bd6514ee0a263148382780968e73b5ef39870b2ef740cc3e7ddf2759",
    ),
    (
        "kplanes_dynamic",
        "kplanes.kplanes_configs:kplanes_dynamic_method",
        "kplanes_nerfstudio",
        "0.5.2",
        "/home/user/.local/lib/python3.10/site-packages/kplanes_nerfstudio-0.5.2.dist-info",
        "aa826943bd6514ee0a263148382780968e73b5ef39870b2ef740cc3e7ddf2759",
    ),
    (
        "lerf",
        "lerf.lerf_config:lerf_method",
        "lerf",
        "0.1.1",
        "/home/user/.local/lib/python3.10/site-packages/lerf-0.1.1.dist-info",
        "45a6092164be0da69a528d3d7ec95d01f5e8b82e748ede4ed4a73b8189720507",
    ),
    (
        "lerf_big",
        "lerf.lerf_config:lerf_method_big",
        "lerf",
        "0.1.1",
        "/home/user/.local/lib/python3.10/site-packages/lerf-0.1.1.dist-info",
        "45a6092164be0da69a528d3d7ec95d01f5e8b82e748ede4ed4a73b8189720507",
    ),
    (
        "lerf_lite",
        "lerf.lerf_config:lerf_method_lite",
        "lerf",
        "0.1.1",
        "/home/user/.local/lib/python3.10/site-packages/lerf-0.1.1.dist-info",
        "45a6092164be0da69a528d3d7ec95d01f5e8b82e748ede4ed4a73b8189720507",
    ),
    (
        "nerfplayer_nerfacto",
        "nerfplayer.nerfplayer_config:nerfplayer_nerfacto",
        "nerfplayer",
        "0.0.1",
        "/home/user/.local/lib/python3.10/site-packages/nerfplayer-0.0.1.dist-info",
        "1c89b0236ca4a9811c6d52c4c72b9ea967e20d92d1d8480150c838d1c4c7144b",
    ),
    (
        "nerfplayer_ngp",
        "nerfplayer.nerfplayer_config:nerfplayer_ngp",
        "nerfplayer",
        "0.0.1",
        "/home/user/.local/lib/python3.10/site-packages/nerfplayer-0.0.1.dist-info",
        "1c89b0236ca4a9811c6d52c4c72b9ea967e20d92d1d8480150c838d1c4c7144b",
    ),
)
METHOD_ENTRY_FIELDS = (
    "name",
    "value",
    "distribution",
    "version",
    "dist_path",
    "entry_points_sha256",
)
GAUSSIAN_TAIL_SHAPES = {
    "features_dc": (3,),
    "features_rest": (15, 3),
    "means": (3,),
    "opacities": (1,),
    "quats": (4,),
    "scales": (3,),
}
FATAL_TRAINING_PATTERNS = (
    r"Traceback \(most recent call last\):",
    r"(?:Runtime|Assertion|Value|ModuleNotFound|Import|Memory)Error:",
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"\boom-kill(?:er)?\b",
    r"(?:^|\n)Killed(?:\r?$|\s)",
    r"Segmentation fault",
    r"core dumped",
)


@dataclass(frozen=True)
class SourceProfile:
    name: str
    receipt_sha256: str
    files: int
    images: int
    image_bytes: int
    total_bytes: int
    sparse_points: int


SOURCE_PROFILES: Mapping[str, SourceProfile] = {
    "lemniscate": SourceProfile(
        name="lemniscate",
        receipt_sha256=(
            "6614c5be765ab7456eac95403af4b2c6fb34e757afc263ba3aa7b9f075cd356a"
        ),
        files=1555,
        images=1553,
        image_bytes=3_362_065_056,
        total_bytes=3_370_611_629,
        sparse_points=183_994,
    ),
    "uturn": SourceProfile(
        name="uturn",
        receipt_sha256=(
            "a42c422dc084375e7f2bf5ef530ac7a5409e9abc0d6c5b3fa90ccd840beb6023"
        ),
        files=1442,
        images=1440,
        image_bytes=3_062_618_402,
        total_bytes=3_070_717_413,
        sparse_points=175_292,
    ),
}


class VerificationError(RuntimeError):
    """A recovered GSplat run violated the frozen verification contract."""


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


def _load_json_value(path: Path) -> Any:
    _require(path.is_file(), f"missing JSON artifact: {path}")
    _require(path.stat().st_size > 0, f"empty JSON artifact: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"cannot parse {path}: {exc}") from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _load_json_value(path)
    _require(isinstance(payload, dict), f"{path.name} root must be an object")
    return payload


def _integer(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return int(value)


def _require_utc_timestamp(value: Any, name: str) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is not None,
        f"{name} must be a whole-second UTC timestamp",
    )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise VerificationError(f"{name} is not a valid UTC timestamp") from exc
    return value


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing file for SHA-256: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    _require(path.stat().st_size > 0, f"empty artifact: {path}")
    label = str(path.relative_to(relative_to)) if relative_to is not None else str(path)
    return {
        "path": label,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parse_shell_env(path: Path) -> dict[str, str]:
    _require(path.is_file(), f"missing shell audit record: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", raw)
        _require(
            match is not None, f"invalid shell record line {line_number} in {path.name}"
        )
        key, encoded = match.groups()
        _require(
            key not in result, f"duplicate shell record key {key!r} in {path.name}"
        )
        try:
            words = shlex.split(encoded, posix=True)
        except ValueError as exc:
            raise VerificationError(
                f"invalid shell escaping for {key!r} in {path.name}: {exc}"
            ) from exc
        _require(len(words) == 1, f"shell record {key!r} must encode exactly one value")
        result[key] = words[0]
    _require(result, f"empty shell audit record: {path}")
    return result


def _path_from_record(record: Mapping[str, str], key: str) -> Path:
    value = record.get(key)
    _require(isinstance(value, str) and value, f"provenance is missing {key}")
    path = Path(value)
    _require(path.is_absolute(), f"provenance {key} must be an absolute path")
    return path


def _same_path(actual: Path, expected: Path, name: str) -> None:
    try:
        actual_resolved = actual.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise VerificationError(f"cannot resolve {name}: {exc}") from exc
    _require(
        actual_resolved == expected_resolved, f"{name} does not resolve to {expected}"
    )


def _require_no_symlink_components(path: Path, stop: Path | None = None) -> None:
    path = path.absolute()
    stop_resolved = stop.absolute() if stop is not None else None
    components: list[Path] = []
    cursor = path
    while True:
        components.append(cursor)
        if stop_resolved is not None and cursor == stop_resolved:
            break
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if stop_resolved is not None:
        _require(components[-1] == stop_resolved, f"{path} is outside {stop}")
    for component in reversed(components):
        _require(
            not component.is_symlink(),
            f"symlink is forbidden in audited path: {component}",
        )


def _safe_source_member(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    _require(relative == pure.as_posix(), f"non-canonical receipt path: {relative!r}")
    _require(
        not pure.is_absolute(), f"absolute receipt path is forbidden: {relative!r}"
    )
    _require(
        bool(pure.parts) and all(part not in ("", ".", "..") for part in pure.parts),
        f"unsafe receipt path: {relative!r}",
    )
    path = root.joinpath(*pure.parts)
    _require_no_symlink_components(path, root)
    _require(path.is_file(), f"receipt file is missing or not regular: {relative}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise VerificationError(
            f"receipt path escapes source root: {relative}"
        ) from exc
    return path


def _extract_prefixed_json(text: str, prefix: str, source: str) -> dict[str, Any]:
    matches = [
        line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)
    ]
    _require(
        len(matches) == 1, f"{source} must contain exactly one {prefix.strip()} record"
    )
    try:
        payload = json.loads(
            matches[0],
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            f"cannot parse {prefix.strip()} in {source}: {exc}"
        ) from exc
    _require(
        isinstance(payload, dict), f"{prefix.strip()} in {source} must be an object"
    )
    return payload


def _validate_method_audit(payload: Mapping[str, Any], source: str) -> None:
    _require(
        payload.get("policy") == "built-in-only", f"{source} plugin policy changed"
    )
    _require(
        payload.get("disabled_group") == "nerfstudio.method_configs",
        f"{source} disabled plugin group changed",
    )
    _require(
        payload.get("nerfstudio_source_sha256") == EXPECTED_NERFSTUDIO_SOURCES,
        f"{source} Nerfstudio source hashes changed",
    )
    rows = payload.get("disabled_entry_points")
    _require(isinstance(rows, list), f"{source} disabled_entry_points must be a list")
    actual: list[tuple[Any, ...]] = []
    for index, row in enumerate(rows):
        _require(
            isinstance(row, dict), f"{source} entry point {index} must be an object"
        )
        _require(
            set(row) == set(METHOD_ENTRY_FIELDS),
            f"{source} entry point {index} fields changed",
        )
        actual.append(tuple(row[field] for field in METHOD_ENTRY_FIELDS))
    _require(
        tuple(sorted(actual)) == EXPECTED_METHOD_ENTRY_POINTS,
        f"{source} external method entry points changed",
    )


def _verify_source(
    data_root: Path,
    profile: SourceProfile,
    receipt_summary_path: Path,
    *,
    progress: bool,
) -> dict[str, Any]:
    _require(data_root.is_dir(), f"source data directory is missing: {data_root}")
    _require(
        data_root.name == profile.name,
        "source data directory basename/profile mismatch",
    )
    _require_no_symlink_components(data_root)
    receipt_path = data_root / ".quadpilot_source_receipt.json"
    receipt_record = _file_record(receipt_path)
    _require(
        receipt_record["sha256"] == profile.receipt_sha256,
        "source receipt SHA-256 differs from the frozen profile",
    )
    receipt = _load_json_object(receipt_path)
    _require(
        receipt.get("schema_version") == 1, "source receipt schema_version must be 1"
    )
    _require(
        receipt.get("track") == profile.name, "source receipt track/profile mismatch"
    )
    files = receipt.get("files")
    _require(isinstance(files, dict), "source receipt files must be an object")
    _require(
        len(files) == profile.files,
        "source receipt file count differs from frozen profile",
    )

    image_paths: set[str] = set()
    image_bytes = 0
    total_bytes = 0
    for index, (relative, metadata) in enumerate(files.items(), 1):
        _require(isinstance(relative, str), "source receipt path must be a string")
        _require(
            isinstance(metadata, dict),
            f"receipt metadata must be an object: {relative}",
        )
        size = _integer(
            metadata.get("size_bytes"), f"receipt size_bytes for {relative}"
        )
        expected_hash = metadata.get("sha256")
        _require(
            isinstance(expected_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_hash),
            f"invalid receipt SHA-256 for {relative}",
        )
        path = _safe_source_member(data_root, relative)
        _require(
            path.stat().st_size == size, f"source size differs from receipt: {relative}"
        )
        _require(
            _sha256(path) == expected_hash,
            f"source SHA-256 differs from receipt: {relative}",
        )
        total_bytes += size
        pure = PurePosixPath(relative)
        if (
            len(pure.parts) == 2
            and pure.parts[0] == "images"
            and pure.suffix.lower() == ".png"
        ):
            image_paths.add(relative)
            image_bytes += size
        if progress and (index % 100 == 0 or index == len(files)):
            print(
                f"SOURCE_HASH_PROGRESS {index}/{len(files)}",
                file=sys.stderr,
                flush=True,
            )

    _require(
        len(image_paths) == profile.images,
        "source image count differs from frozen profile",
    )
    _require(
        image_bytes == profile.image_bytes,
        "source image bytes differ from frozen profile",
    )
    _require(
        total_bytes == profile.total_bytes,
        "source total bytes differ from frozen profile",
    )
    _require(
        set(files) == image_paths | {"transforms.json", "sparse_pc.ply"},
        "source receipt contains an unexpected non-image file",
    )

    transforms = _load_json_object(data_root / "transforms.json")
    frames = transforms.get("frames")
    _require(isinstance(frames, list), "transforms.json frames must be a list")
    frame_paths: list[str] = []
    for index, frame in enumerate(frames):
        _require(isinstance(frame, dict), f"transforms frame {index} must be an object")
        value = frame.get("file_path")
        _require(
            isinstance(value, str), f"transforms frame {index} file_path is invalid"
        )
        frame_paths.append(PurePosixPath(value).as_posix())
    _require(
        len(frame_paths) == profile.images,
        "transforms camera count differs from profile",
    )
    _require(
        len(set(frame_paths)) == len(frame_paths),
        "transforms contains duplicate image paths",
    )
    _require(
        set(frame_paths) == image_paths,
        "transforms images differ from the receipt image set",
    )

    sparse_path = data_root / "sparse_pc.ply"
    try:
        with sparse_path.open("rb") as handle:
            header_lines: list[str] = []
            for _ in range(256):
                raw = handle.readline()
                _require(raw, "sparse_pc.ply header is truncated")
                line = raw.decode("ascii").rstrip("\r\n")
                header_lines.append(line)
                if line == "end_header":
                    break
            else:
                raise VerificationError("sparse_pc.ply header exceeds 256 lines")
    except UnicodeDecodeError as exc:
        raise VerificationError("sparse_pc.ply header is not ASCII") from exc
    vertices = [line for line in header_lines if line.startswith("element vertex ")]
    _require(
        len(vertices) == 1, "sparse_pc.ply must declare exactly one vertex element"
    )
    try:
        sparse_points = int(vertices[0].split()[2])
    except (IndexError, ValueError) as exc:
        raise VerificationError("sparse_pc.ply vertex count is invalid") from exc
    _require(
        sparse_points == profile.sparse_points,
        "sparse point count differs from profile",
    )

    summary = _load_json_object(receipt_summary_path)
    expected_summary = {
        "receipt_sha256": profile.receipt_sha256,
        "verified_bytes": profile.total_bytes,
        "verified_files": profile.files,
        "verified_images": profile.images,
    }
    _require(
        summary == expected_summary,
        "saved receipt-verification.json disagrees with source",
    )
    return {
        "status": "PASS",
        "profile": profile.name,
        **expected_summary,
        "image_bytes": profile.image_bytes,
        "transforms_cameras": len(frame_paths),
        "sparse_points": sparse_points,
    }


def _verify_preflight_and_plugins(
    run_root: Path, profile: SourceProfile
) -> dict[str, Any]:
    preflight_path = run_root / "preflight-container.log"
    _require(preflight_path.is_file(), "missing preflight-container.log")
    preflight_text = preflight_path.read_text(encoding="utf-8")
    _require(
        preflight_text.count("PREFLIGHT_OK") == 1,
        "preflight did not finish exactly once",
    )

    versions = _extract_prefixed_json(preflight_text, "VERSIONS ", preflight_path.name)
    _require(versions == EXPECTED_VERSIONS, "pinned package versions changed")
    pip_check = _extract_prefixed_json(
        preflight_text, "PIP_CHECK ", preflight_path.name
    )
    _require(
        pip_check == {"known_deviations": EXPECTED_PIP_DEVIATIONS, "returncode": 1},
        "pip-check deviations differ from the audited pinned image",
    )
    receipt = _extract_prefixed_json(preflight_text, "RECEIPT_OK ", preflight_path.name)
    _require(
        receipt
        == {
            "receipt_sha256": profile.receipt_sha256,
            "verified_bytes": profile.total_bytes,
            "verified_files": profile.files,
            "verified_images": profile.images,
        },
        "preflight receipt record differs from the frozen profile",
    )
    dataset = _extract_prefixed_json(preflight_text, "DATASET ", preflight_path.name)
    _require(
        dataset
        == {
            "cameras": profile.images,
            "dataparser_downscale_factor": 1,
            "images": profile.images,
            "missing_images": 0,
            "sparse_points": profile.sparse_points,
        },
        "preflight dataparser record differs from the frozen profile",
    )
    builtins = _extract_prefixed_json(
        preflight_text, "BUILTIN_METHOD_CONFIGS_OK ", preflight_path.name
    )
    _require(
        builtins == {"method_count": 43, "splatfacto_present": True},
        "built-in method catalog changed",
    )
    saved_builtins = _load_json_object(run_root / "builtin-method-configs.json")
    _require(
        saved_builtins == builtins, "saved built-in method audit differs from preflight"
    )

    cpu_audit = _extract_prefixed_json(
        preflight_text, "METHOD_PLUGIN_AUDIT ", preflight_path.name
    )
    _validate_method_audit(cpu_audit, "CPU preflight")
    saved_cpu_audit = _load_json_object(run_root / "method-plugin-audit.json")
    _require(
        saved_cpu_audit == cpu_audit, "saved CPU plugin audit differs from preflight"
    )

    docker_log_path = run_root / "docker.log"
    _require(docker_log_path.is_file(), "missing docker.log")
    docker_log = docker_log_path.read_text(encoding="utf-8")
    training_audit = _extract_prefixed_json(
        docker_log, "METHOD_PLUGIN_AUDIT ", docker_log_path.name
    )
    _validate_method_audit(training_audit, "training")
    _require(training_audit == cpu_audit, "CPU and training plugin audits differ")
    materialized_training_audit = run_root / "training-method-plugin-audit.json"
    if materialized_training_audit.exists():
        _require(
            _load_json_object(materialized_training_audit) == training_audit,
            "materialized training plugin audit differs from docker.log",
        )

    _require(
        docker_log.count("Training Finished") == 1,
        "docker.log must contain exactly one Training Finished marker",
    )
    _require(
        re.search(r"(?m)^29999 \(100\.00%\)", docker_log) is not None,
        "docker.log does not show final step 29999 at 100%",
    )
    for pattern in FATAL_TRAINING_PATTERNS:
        _require(
            re.search(pattern, docker_log, flags=re.IGNORECASE) is None,
            f"docker.log contains a fatal training marker matching {pattern!r}",
        )
    return {
        "status": "PASS",
        "versions": versions,
        "pip_check": pip_check,
        "dataset": dataset,
        "builtin_method_count": builtins["method_count"],
        "external_method_plugins_disabled": len(EXPECTED_METHOD_ENTRY_POINTS),
        "cpu_training_plugin_audits_equal": True,
        "training_audit_materialized_by_wrapper": materialized_training_audit.is_file(),
        "training_finished_marker_count": 1,
    }


def _yaml_get(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        _require(
            isinstance(current, dict) and key in current,
            f"config.yml is missing {'.'.join(keys)}",
        )
        current = current[key]
    return current


def _verify_config(config_path: Path, track: str, run_id: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise VerificationError(
            "PyYAML is required to inspect config.yml on CPU"
        ) from exc
    try:
        config = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        )
    except Exception as exc:
        raise VerificationError(
            f"cannot parse config.yml with non-instantiating BaseLoader: {exc}"
        ) from exc
    _require(isinstance(config, dict), "config.yml root must be a mapping")

    expected_scalars = {
        ("experiment_name",): track,
        ("timestamp",): run_id,
        ("method_name",): "splatfacto",
        ("max_num_iterations",): "30000",
        ("vis",): "tensorboard",
        ("steps_per_eval_batch",): "0",
        ("steps_per_eval_image",): "0",
        ("steps_per_eval_all_images",): "0",
        ("save_only_latest_checkpoint",): "true",
        ("machine", "seed"): "42",
        ("machine", "num_devices"): "1",
        ("machine", "device_type"): "cuda",
        ("pipeline", "datamanager", "camera_res_scale_factor"): "0.5",
        ("pipeline", "datamanager", "dataparser", "downscale_factor"): "1",
        ("pipeline", "datamanager", "dataparser", "load_3D_points"): "true",
        ("pipeline", "model", "num_downscales"): "2",
        ("pipeline", "model", "resolution_schedule"): "3000",
        ("pipeline", "model", "random_init"): "false",
    }
    for keys, expected in expected_scalars.items():
        actual = _yaml_get(config, *keys)
        _require(
            actual == expected,
            f"config.yml {'.'.join(keys)}={actual!r}, expected {expected!r}",
        )
    _require(_yaml_get(config, "data") == ["/", "data"], "config.yml data is not /data")
    _require(
        _yaml_get(config, "output_dir") == ["/", "outputs"],
        "config.yml output_dir is not /outputs",
    )
    _require(
        _yaml_get(config, "pipeline", "datamanager", "data") == ["/", "data"],
        "config.yml datamanager data is not /data",
    )
    return {
        "status": "PASS",
        "max_num_iterations": 30_000,
        "final_step": 29_999,
        "seed": 42,
        "camera_res_scale_factor": 0.5,
        "dataparser_downscale_factor": 1,
        "splatfacto_num_downscales": 2,
        "splatfacto_resolution_schedule": 3_000,
        "periodic_evaluation_enabled": False,
        "sparse_point_initialization": True,
    }


def _verify_command(
    command_path: Path,
    *,
    data_dir: Path,
    training_output_dir: Path,
    track: str,
    run_id: str,
) -> dict[str, Any]:
    _require(command_path.is_file(), "missing command.sh")
    command = command_path.read_text(encoding="utf-8")
    normalized = command.replace("\\,", ",")
    required = (
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "exec /usr/bin/docker run",
        "--pull=never",
        "--network none",
        "--gpus device=0",
        f"type=bind,src={data_dir},dst=/data,readonly",
        f"type=bind,src={training_output_dir},dst=/outputs",
        f"dst=/cache/{LPIPS_RELATIVE_PATH},readonly",
        "--env MAX_JOBS=4",
        IMAGE_REF,
        "splatfacto",
        "--data /data",
        "--output-dir /outputs",
        f"--experiment-name {track}",
        f"--timestamp {run_id}",
        "--max-num-iterations 30000",
        "--vis tensorboard",
        "--steps-per-eval-batch 0",
        "--steps-per-eval-image 0",
        "--steps-per-eval-all-images 0",
        "--machine.num-devices 1",
        "--pipeline.datamanager.camera-res-scale-factor 0.5",
        "--pipeline.model.num-downscales 2",
        "--pipeline.model.resolution-schedule 3000",
        "nerfstudio-data --downscale-factor 1",
        "METHOD_PLUGIN_AUDIT",
    )
    for fragment in required:
        _require(
            fragment in normalized, f"command.sh is missing frozen fragment: {fragment}"
        )
    return {
        "status": "PASS",
        **_file_record(command_path, relative_to=command_path.parent),
    }


def _verify_image_inspect(path: Path) -> dict[str, Any]:
    payload = _load_json_value(path)
    _require(
        isinstance(payload, list) and len(payload) == 1,
        "image inspect must contain one image",
    )
    image = payload[0]
    _require(isinstance(image, dict), "image inspect entry must be an object")
    _require(image.get("Id") == IMAGE_DIGEST, "pinned Docker image ID changed")
    repo_digests = image.get("RepoDigests")
    _require(isinstance(repo_digests, list), "image RepoDigests must be a list")
    _require(IMAGE_REF in repo_digests, "pinned Docker RepoDigest is absent")
    _require(image.get("Os") == "linux", "pinned Docker image OS changed")
    _require(image.get("Architecture") == "amd64", "pinned Docker architecture changed")
    return {
        "status": "PASS",
        "image_ref": IMAGE_REF,
        "image_id": image["Id"],
        **_file_record(path, relative_to=path.parent),
    }


def inspect_torch_checkpoint(path: Path, expected_step: int) -> dict[str, Any]:
    """Load a locally produced checkpoint on CPU and validate Gaussian state."""
    try:
        import torch
    except ImportError as exc:
        raise VerificationError(
            "PyTorch is required for checkpoint inspection; run this CPU-only verifier "
            "inside the pinned image without exposing a GPU"
        ) from exc
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Torch 2.1.2 accepts weights_only, but retain a clear CPU fallback for
        # older local audit environments.
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise VerificationError(f"cannot load checkpoint on CPU: {exc}") from exc
    _require(isinstance(checkpoint, dict), "checkpoint root must be a mapping")
    _require(
        set(checkpoint) == {"step", "pipeline", "optimizers", "schedulers", "scalers"},
        "checkpoint top-level fields changed",
    )
    step = checkpoint.get("step")
    _require(
        isinstance(step, int) and not isinstance(step, bool),
        "checkpoint step is not an integer",
    )
    _require(step == expected_step, f"checkpoint step={step}, expected {expected_step}")
    pipeline = checkpoint.get("pipeline")
    _require(isinstance(pipeline, Mapping), "checkpoint pipeline must be a mapping")

    tensor_records: dict[str, Any] = {}
    gaussian_count: int | None = None
    for name, tail_shape in GAUSSIAN_TAIL_SHAPES.items():
        key = f"_model.gauss_params.{name}"
        _require(key in pipeline, f"checkpoint is missing Gaussian tensor {key}")
        tensor = pipeline[key]
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        _require(len(shape) == len(tail_shape) + 1, f"{key} rank changed: {shape}")
        _require(shape[1:] == tail_shape, f"{key} shape changed: {shape}")
        _require(shape[0] > 0, f"{key} contains no Gaussians")
        if gaussian_count is None:
            gaussian_count = shape[0]
        _require(shape[0] == gaussian_count, f"{key} Gaussian count disagrees")
        _require(
            bool(getattr(tensor, "is_floating_point")()), f"{key} is not floating point"
        )
        device = str(getattr(tensor, "device", ""))
        _require(device == "cpu", f"{key} was not loaded on CPU")
        try:
            finite = bool(torch.isfinite(tensor).all().item())
        except Exception as exc:
            raise VerificationError(
                f"cannot check finiteness for {key}: {exc}"
            ) from exc
        _require(finite, f"{key} contains NaN or infinity")
        tensor_records[name] = {
            "shape": list(shape),
            "dtype": str(getattr(tensor, "dtype", "")),
            "device": device,
            "finite": True,
        }
    _require(gaussian_count is not None, "checkpoint has no Gaussian tensors")
    return {
        "status": "PASS",
        "load_device": "cpu",
        "weights_only": False,
        "step": step,
        "gaussian_count": gaussian_count,
        "gaussian_tensors": tensor_records,
    }


def _verify_existing_artifact_manifest(
    path: Path,
    config_hash: str,
    checkpoint_name: str,
    checkpoint_hash: str,
) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    _require(path.is_file(), "training-artifacts.sha256 is not a regular file")
    expected = {
        "config.yml": config_hash,
        f"nerfstudio_models/{checkpoint_name}": checkpoint_hash,
    }
    actual: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, "training-artifacts.sha256 has an invalid line")
        digest, name = match.groups()
        _require(name not in actual, "training-artifacts.sha256 has a duplicate path")
        actual[name] = digest
    _require(
        actual == expected, "training-artifacts.sha256 disagrees with actual artifacts"
    )
    return {"present": True, "sha256": _sha256(path)}


CheckpointInspector = Callable[[Path, int], dict[str, Any]]


def verify_run(
    run_root: Path,
    *,
    track: str,
    profile: SourceProfile | None = None,
    checkpoint_inspector: CheckpointInspector = inspect_torch_checkpoint,
    progress: bool = False,
) -> dict[str, Any]:
    """Verify a failed wrapper run whose final training artifacts may be sound."""
    _require(
        track in SOURCE_PROFILES or profile is not None,
        f"unsupported track profile: {track}",
    )
    selected = profile if profile is not None else SOURCE_PROFILES[track]
    _require(selected.name == track, "injected source profile name/track mismatch")
    _require(run_root.is_dir(), f"run root is missing: {run_root}")
    _require(not run_root.is_symlink(), "run root must not be a symlink")
    run_root = run_root.resolve(strict=True)
    run_id = run_root.name

    recovery_path = run_root / "recovered-postflight.json"
    _require(
        not recovery_path.exists(),
        "recovered-postflight.json already exists; refusing overwrite",
    )

    status_path = run_root / "status.env"
    status_bytes_before = status_path.read_bytes() if status_path.is_file() else b""
    _require(status_bytes_before, "missing or empty original status.env")
    status_hash_before = hashlib.sha256(status_bytes_before).hexdigest()
    status = _parse_shell_env(status_path)
    _require(
        status.get("result") == "failed",
        "recovery verifier requires original wrapper result=failed",
    )
    try:
        wrapper_exit_code = int(status.get("exit_code", ""))
    except ValueError as exc:
        raise VerificationError("status.env exit_code is not an integer") from exc
    _require(
        wrapper_exit_code > 0, "recovery verifier requires a nonzero wrapper exit code"
    )
    _require_utc_timestamp(status.get("finished_utc"), "status.env finished_utc")

    provenance_path = run_root / "provenance.env"
    provenance = _parse_shell_env(provenance_path)
    required_provenance = {
        "schema_version": "1",
        "mode": "train-30k",
        "image_ref": IMAGE_REF,
        "track": track,
        "run_id": run_id,
        "data_mount_mode": "readonly",
        "expected_source_receipt_sha256": selected.receipt_sha256,
        "half_res_linear_scale": "0.5",
        "half_res_pixel_fraction": "0.25",
        "full_resolution_reproduction": "false",
        "lpips_alexnet_expected_size_bytes": str(LPIPS_SIZE_BYTES),
        "lpips_alexnet_expected_sha256": LPIPS_SHA256,
        "lpips_alexnet_actual_size_bytes": str(LPIPS_SIZE_BYTES),
        "lpips_alexnet_actual_sha256": LPIPS_SHA256,
        "lpips_alexnet_cache_verified": "true",
        "method_plugin_policy": "built-in-only",
        "max_num_iterations": "30000",
        "expected_final_step": "29999",
        "expected_checkpoint_name": "step-000029999.ckpt",
        "splatfacto_num_downscales": "2",
        "splatfacto_resolution_schedule": "3000",
        "periodic_evaluation_enabled": "false",
        "training_max_jobs": "4",
        "cpu_preflight_executed": "true",
    }
    for key, expected in required_provenance.items():
        _require(
            provenance.get(key) == expected,
            f"provenance {key!r} differs from {expected!r}",
        )
    if "source_profile" in provenance:
        _require(
            provenance["source_profile"] == track,
            "provenance source_profile/track mismatch",
        )
    optional_profile_counts = {
        "expected_source_receipt_files": selected.files,
        "expected_source_receipt_images": selected.images,
        "expected_source_receipt_image_bytes": selected.image_bytes,
        "expected_source_receipt_total_bytes": selected.total_bytes,
        "expected_source_sparse_points": selected.sparse_points,
    }
    for key, expected in optional_profile_counts.items():
        if key in provenance:
            _require(
                provenance[key] == str(expected),
                f"provenance {key} differs from profile",
            )
    _require_utc_timestamp(provenance.get("started_utc"), "provenance started_utc")
    if "started_utc" in status:
        _require(
            provenance["started_utc"] == status["started_utc"],
            "status/provenance start time mismatch",
        )
    # The wrapper's EXIT trap intentionally records terminal state only in
    # status.env.  Older/current run provenance may therefore have no terminal
    # fields; if a future wrapper adds them, they must agree exactly.
    if "finished_utc" in provenance:
        _require(
            provenance["finished_utc"] == status.get("finished_utc"),
            "status/provenance finish time mismatch",
        )
    if "result" in provenance:
        _require(provenance["result"] == "failed", "status/provenance result mismatch")
    if "exit_code" in provenance:
        _require(
            provenance["exit_code"] == str(wrapper_exit_code),
            "status/provenance exit code mismatch",
        )

    recorded_run = _path_from_record(provenance, "run_dir")
    _same_path(recorded_run, run_root, "provenance run_dir")
    data_dir = _path_from_record(provenance, "data_dir")
    training_output_dir = _path_from_record(provenance, "training_output_dir")
    _same_path(training_output_dir, run_root / "training-output", "training_output_dir")

    lpips_path = _path_from_record(provenance, "lpips_alexnet_cache_path")
    _require_no_symlink_components(lpips_path)
    lpips_record = _file_record(lpips_path)
    _require(
        lpips_record["size_bytes"] == LPIPS_SIZE_BYTES, "LPIPS AlexNet size changed"
    )
    _require(lpips_record["sha256"] == LPIPS_SHA256, "LPIPS AlexNet SHA-256 changed")

    source_report = _verify_source(
        data_dir,
        selected,
        run_root / "receipt-verification.json",
        progress=progress,
    )
    preflight_report = _verify_preflight_and_plugins(run_root, selected)
    image_report = _verify_image_inspect(run_root / "docker-image-inspect.json")
    command_report = _verify_command(
        run_root / "command.sh",
        data_dir=data_dir,
        training_output_dir=training_output_dir,
        track=track,
        run_id=run_id,
    )

    train_run_dir = training_output_dir / track / "splatfacto" / run_id
    _require(
        train_run_dir.is_dir(), f"missing Nerfstudio run directory: {train_run_dir}"
    )
    config_path = train_run_dir / "config.yml"
    checkpoint_dir = train_run_dir / "nerfstudio_models"
    checkpoint_name = "step-000029999.ckpt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_files = (
        sorted(checkpoint_dir.glob("step-*.ckpt")) if checkpoint_dir.is_dir() else []
    )
    _require(
        checkpoint_files == [checkpoint_path],
        "checkpoint directory must contain exactly step-000029999.ckpt",
    )
    _require(checkpoint_path.stat().st_size > 0, "final checkpoint is empty")

    config_report = _verify_config(config_path, track, run_id)
    config_record = _file_record(config_path, relative_to=run_root)
    checkpoint_record = _file_record(checkpoint_path, relative_to=run_root)
    checkpoint_report = checkpoint_inspector(checkpoint_path, 29_999)
    _require(
        isinstance(checkpoint_report, dict)
        and checkpoint_report.get("status") == "PASS",
        "checkpoint inspector did not return PASS",
    )
    manifest_report = _verify_existing_artifact_manifest(
        run_root / "training-artifacts.sha256",
        config_record["sha256"],
        checkpoint_name,
        checkpoint_record["sha256"],
    )

    status_bytes_after = status_path.read_bytes()
    status_hash_after = hashlib.sha256(status_bytes_after).hexdigest()
    _require(
        status_bytes_after == status_bytes_before,
        "status.env changed during verification",
    )

    artifact_records = {
        "provenance_env": _file_record(provenance_path, relative_to=run_root),
        "status_env": _file_record(status_path, relative_to=run_root),
        "command": command_report,
        "docker_log": _file_record(run_root / "docker.log", relative_to=run_root),
        "preflight_log": _file_record(
            run_root / "preflight-container.log", relative_to=run_root
        ),
        "method_plugin_audit": _file_record(
            run_root / "method-plugin-audit.json", relative_to=run_root
        ),
        "builtin_method_configs": _file_record(
            run_root / "builtin-method-configs.json", relative_to=run_root
        ),
        "docker_image_inspect": image_report,
        "source_receipt": {
            "path": str(data_dir / ".quadpilot_source_receipt.json"),
            "size_bytes": (data_dir / ".quadpilot_source_receipt.json").stat().st_size,
            "sha256": selected.receipt_sha256,
        },
        "lpips_alexnet": {"path": str(lpips_path), **lpips_record},
        "config": config_record,
        "checkpoint": checkpoint_record,
    }
    return {
        "schema_version": 1,
        "verifier": "verify_repro_gsplat_run.py",
        "verified_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "run_root": str(run_root),
        "run_id": run_id,
        "track": track,
        "mode": "train-30k",
        "classification": "WRAPPER_FAILED_TRAINING_ARTIFACTS_VERIFIED",
        "overall_success": False,
        "wrapper_status": {
            "status": "failed",
            "exit_code": wrapper_exit_code,
            "status_env_sha256_before": status_hash_before,
            "status_env_sha256_after": status_hash_after,
            "original_status_preserved": True,
        },
        "training_artifacts": {
            "status": "PASS",
            "config": config_report,
            "checkpoint": checkpoint_report,
            "unique_expected_checkpoint": True,
            "existing_wrapper_artifact_manifest": manifest_report,
        },
        "source_data": source_report,
        "preflight_and_plugin_audit": preflight_report,
        "docker_image": image_report,
        "lpips_alexnet": {
            "status": "PASS",
            "size_bytes": LPIPS_SIZE_BYTES,
            "sha256": LPIPS_SHA256,
        },
        "provenance": {
            "status": "PASS",
            "schema_version": 1,
            "legacy_profile_count_fields_absent": sorted(
                key for key in optional_profile_counts if key not in provenance
            ),
        },
        "artifact_sha256": artifact_records,
        "recovery_semantics": (
            "The training artifacts independently pass, but the original wrapper remains "
            "failed. This report does not convert the run into an overall success."
        ),
    }


def write_recovery_report(run_root: Path, report: Mapping[str, Any]) -> Path:
    """Publish a recovery report atomically without ever replacing a prior one."""
    _require(
        report.get("overall_success") is False,
        "recovery report must not claim overall success",
    )
    training = report.get("training_artifacts")
    wrapper = report.get("wrapper_status")
    _require(
        isinstance(training, dict) and training.get("status") == "PASS",
        "artifacts are not PASS",
    )
    _require(
        isinstance(wrapper, dict) and wrapper.get("status") == "failed",
        "wrapper is not failed",
    )
    target = run_root / "recovered-postflight.json"
    _require(
        not target.exists(),
        "recovered-postflight.json already exists; refusing overwrite",
    )
    serialized = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=".recovered-postflight.", suffix=".tmp", dir=run_root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise VerificationError(
                "recovered-postflight.json appeared concurrently; refusing overwrite"
            ) from exc
        temporary.unlink()
        try:
            directory_fd = os.open(run_root, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root", type=Path, help="existing failed-wrapper train-30k run root"
    )
    parser.add_argument(
        "--track",
        required=True,
        choices=sorted(SOURCE_PROFILES),
        help="explicit frozen source-data profile; never inferred",
    )
    parser.add_argument(
        "--write-recovered-postflight",
        action="store_true",
        help="atomically add recovered-postflight.json after every gate passes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = verify_run(args.run_root, track=args.track, progress=True)
        written: Path | None = None
        if args.write_recovered_postflight:
            written = write_recovery_report(args.run_root.resolve(strict=True), report)
        summary = {
            "classification": report["classification"],
            "overall_success": report["overall_success"],
            "wrapper_status": report["wrapper_status"],
            "training_artifacts": {
                "status": report["training_artifacts"]["status"],
                "checkpoint": report["training_artifacts"]["checkpoint"],
            },
            "source_data": report["source_data"],
            "report_written": str(written) if written is not None else None,
        }
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except VerificationError as exc:
        print(f"VERIFY_REPRO_GSPLAT_RUN_FAILED: {exc}", file=sys.stderr)
        return 2
    except (
        Exception
    ) as exc:  # fail closed without an unaudited traceback in normal CLI use
        print(
            f"VERIFY_REPRO_GSPLAT_RUN_FAILED: unexpected {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
