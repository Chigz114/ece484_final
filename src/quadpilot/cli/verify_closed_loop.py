#!/usr/bin/env python3
"""Fail-closed verification for a completed visual closed-loop run.

This verifier is intentionally post-hoc and CPU-only.  It never imports the
renderer or NPE model: it validates the saved schema, recomputes the canonical
strict gate evaluation from the saved truth states, and hashes the explicitly
selected model assets for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.simulation.evaluation import evaluate_ordered_gates  # noqa: E402
from quadpilot.simulation.tracks import TRACKS, TrackConfig, get_track  # noqa: E402

CORE_ARRAY_SHAPES: Mapping[str, tuple[int, ...]] = {
    "states": (7,),
    "observations": (4,),
    "estimated_states": (7,),
    "controls": (4,),
    "camera_to_world": (4, 4),
}
NPZ_ARRAYS = frozenset(
    {
        *CORE_ARRAY_SHAPES,
        "ekf_update_accepted",
        "ekf_mahalanobis",
        "state_step_indices",
        "state_times_s",
        "observation_step_indices",
        "observation_times_s",
        "estimate_step_indices",
        "estimate_times_s",
        "control_step_indices",
        "control_times_s",
        "camera_step_indices",
        "camera_times_s",
    }
)
ESTIMATORS = ("raw", "ekf")
RENDERER_RELATIVE_PATH = "nerfstudio_models/step-000029999.ckpt"
TRANSFORM_RELATIVE_PATH = "dataparser_transforms.json"

# The historical Circle manifest predates the step/Gaussian fields now stored
# for locally reproduced tracks.  Preserve its already-accepted immutable
# contract explicitly; every newer track must carry these facts in its manifest.
LEGACY_RENDERER_CONTRACTS: Mapping[str, tuple[int, int]] = {
    "circle": (29999, 308832),
}


def _expected_files(track: str) -> frozenset[str]:
    return frozenset(
        f"{track}_{estimator}.{suffix}"
        for estimator in ESTIMATORS
        for suffix in ("json", "npz")
    )


class VerificationError(RuntimeError):
    """A closed-loop artifact violated the frozen acceptance contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _finite_float(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be a real number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return int(value)


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing provenance file: {path}")
    _require(path.stat().st_size > 0, f"empty provenance file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise VerificationError(f"JSON contains non-standard non-finite value {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"cannot parse {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} root must be an object")
    return payload


def _json_array(
    payload: Mapping[str, Any], name: str, tail_shape: tuple[int, ...]
) -> np.ndarray:
    _require(name in payload, f"JSON is missing {name}")
    try:
        array = np.asarray(payload[name], dtype=np.float64)
    except Exception as exc:
        raise VerificationError(
            f"JSON {name} is not a rectangular numeric array"
        ) from exc
    expected_ndim = len(tail_shape) + 1
    _require(
        array.ndim == expected_ndim and array.shape[1:] == tail_shape,
        f"JSON {name} has shape {array.shape}, expected (N,{','.join(map(str, tail_shape))})",
    )
    _require(np.isfinite(array).all(), f"JSON {name} contains NaN or infinity")
    return array


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"missing artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual = frozenset(archive.files)
            _require(
                actual == NPZ_ARRAYS,
                f"{path.name} NPZ members differ: missing={sorted(NPZ_ARRAYS - actual)}, "
                f"extra={sorted(actual - NPZ_ARRAYS)}",
            )
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"cannot safely load {path}: {exc}") from exc


def _same_float(
    actual: Any, expected: float, name: str, *, atol: float = 1e-12
) -> None:
    value = _finite_float(actual, name)
    _require(
        math.isclose(value, float(expected), rel_tol=0.0, abs_tol=atol),
        f"{name}={value!r}, expected {expected!r}",
    )


def _validate_axis(
    payload: Mapping[str, Any],
    npz: Mapping[str, np.ndarray],
    *,
    json_name: str,
    npz_prefix: str,
    count: int,
    dt: float,
) -> None:
    axis = payload.get(json_name)
    _require(isinstance(axis, dict), f"sample_alignment.{json_name} must be an object")
    expected_steps = np.arange(count, dtype=np.int64)
    expected_times = expected_steps.astype(np.float64) * dt
    try:
        json_steps = np.asarray(axis["step_indices"], dtype=np.int64)
        json_times = np.asarray(axis["times_s"], dtype=np.float64)
    except Exception as exc:
        raise VerificationError(f"invalid JSON time axis for {json_name}") from exc
    _require(
        np.array_equal(json_steps, expected_steps),
        f"JSON {json_name} step indices are not 0..N-1",
    )
    _require(
        np.allclose(json_times, expected_times, rtol=0.0, atol=1e-12),
        f"JSON {json_name} times do not equal step*dt",
    )
    npz_steps = npz[f"{npz_prefix}_step_indices"]
    npz_times = npz[f"{npz_prefix}_times_s"]
    _require(
        np.issubdtype(npz_steps.dtype, np.integer)
        and np.array_equal(npz_steps, expected_steps),
        f"NPZ {npz_prefix}_step_indices are invalid",
    )
    _require(
        np.issubdtype(npz_times.dtype, np.number)
        and np.isfinite(npz_times).all()
        and np.allclose(npz_times, expected_times, rtol=0.0, atol=1e-12),
        f"NPZ {npz_prefix}_times_s are invalid",
    )


def _compare_strict_evaluation(
    saved: Any,
    states: np.ndarray,
    *,
    track: TrackConfig,
    dt: float,
    laps: int,
    gate_radius: float,
) -> dict[str, Any]:
    _require(isinstance(saved, dict), "metrics.strict_evaluation must be an object")
    computed = evaluate_ordered_gates(
        states,
        track,
        dt=dt,
        laps=laps,
        gate_radius=gate_radius,
    ).to_dict()
    required_crossings = len(track.gate_order) * laps
    _require(
        computed["completed"] is True, "recomputed strict evaluation did not complete"
    )
    _require(
        computed["required_crossings"] == required_crossings,
        f"strict evaluation did not require {required_crossings} crossings",
    )
    _require(
        computed["successful_crossings"] == required_crossings,
        f"strict evaluation was not {required_crossings}/{required_crossings}",
    )

    for key in ("track", "required_crossings", "successful_crossings", "completed"):
        _require(
            saved.get(key) == computed[key],
            f"saved strict_evaluation.{key} disagrees with recomputation",
        )
    for key in (
        "success_rate",
        "mean_gate_error_m",
        "mission_time_s",
        "first_to_last_gate_time_s",
    ):
        _same_float(saved.get(key), float(computed[key]), f"strict_evaluation.{key}")

    saved_crossings = saved.get("crossings")
    computed_crossings = computed["crossings"]
    _require(
        isinstance(saved_crossings, list), "strict_evaluation.crossings must be a list"
    )
    _require(
        len(saved_crossings) == len(computed_crossings),
        "saved crossing count disagrees with recomputation",
    )
    expected_sequence = list(track.gate_order) * laps
    _require(
        [item["gate"] for item in computed_crossings] == expected_sequence,
        f"recomputed crossing sequence is not the canonical {track.name} order",
    )
    for index, (actual, expected) in enumerate(
        zip(saved_crossings, computed_crossings)
    ):
        _require(isinstance(actual, dict), f"crossing {index} must be an object")
        for key in ("gate", "lap"):
            _require(
                actual.get(key) == expected[key],
                f"crossing {index} {key} disagrees with recomputation",
            )
        for key in ("time_seconds", "radial_error_m"):
            _same_float(
                actual.get(key), float(expected[key]), f"crossing {index} {key}"
            )
        try:
            actual_position = np.asarray(actual["position"], dtype=np.float64)
            expected_position = np.asarray(expected["position"], dtype=np.float64)
        except Exception as exc:
            raise VerificationError(f"crossing {index} position is invalid") from exc
        _require(
            actual_position.shape == (3,)
            and np.isfinite(actual_position).all()
            and np.allclose(actual_position, expected_position, rtol=0.0, atol=1e-12),
            f"crossing {index} position disagrees with recomputation",
        )
    return computed


def _validate_run(
    output_dir: Path,
    estimator: str,
    *,
    track: TrackConfig,
    expected_metadata: Mapping[str, Any],
    expected_seed: int,
    expected_device: str,
    expected_max_steps: int,
    expected_dt: float,
    expected_laps: int,
    expected_gate_radius: float,
    expected_hysteresis: float,
    expected_renderer_step: int,
    expected_gaussians: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    json_path = (output_dir / f"{track.name}_{estimator}.json").resolve()
    npz_path = (output_dir / f"{track.name}_{estimator}.npz").resolve()
    payload = _load_json(json_path)
    _require(
        payload.get("schema_version") == 1, f"{json_path.name} schema_version must be 1"
    )
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), f"{json_path.name} metadata must be an object")
    _require(metadata == expected_metadata, "raw and EKF metadata/provenance differ")

    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, dict), f"{json_path.name} artifacts must be an object"
    )
    for key, expected_path in (("json", json_path), ("npz", npz_path)):
        value = artifacts.get(key)
        _require(isinstance(value, str), f"artifacts.{key} must be a path string")
        _require(
            Path(value).expanduser().resolve() == expected_path,
            f"artifacts.{key} does not name {expected_path}",
        )

    arrays = {
        name: _json_array(payload, name, tail_shape)
        for name, tail_shape in CORE_ARRAY_SHAPES.items()
    }
    state_count = len(arrays["states"])
    control_count = len(arrays["controls"])
    _require(
        state_count == control_count + 1,
        "states must contain exactly one more sample than controls",
    )
    for name in ("observations", "estimated_states", "camera_to_world"):
        _require(
            len(arrays[name]) == state_count,
            f"successful {name} must align one-to-one with states",
        )

    initial_state = np.asarray(track.initial_state, dtype=np.float64)
    _require(
        np.allclose(arrays["states"][0], initial_state, rtol=0.0, atol=1e-12),
        f"truth trajectory does not start at the canonical {track.name} initial state",
    )

    accepted_json = payload.get("ekf_update_accepted")
    mahalanobis_json = payload.get("ekf_mahalanobis")
    _require(isinstance(accepted_json, list), "ekf_update_accepted must be a list")
    _require(isinstance(mahalanobis_json, list), "ekf_mahalanobis must be a list")
    _require(
        all(isinstance(value, bool) for value in accepted_json),
        "EKF acceptance entries must be booleans",
    )
    if estimator == "raw":
        _require(
            not accepted_json and not mahalanobis_json,
            "raw result must not contain EKF diagnostics",
        )
    else:
        _require(
            len(accepted_json) == state_count,
            "EKF acceptance count must equal state count",
        )
        _require(
            len(mahalanobis_json) == state_count,
            "EKF Mahalanobis count must equal state count",
        )
        for index, value in enumerate(mahalanobis_json):
            if value is not None:
                _require(
                    _finite_float(value, f"ekf_mahalanobis[{index}]") >= 0.0,
                    "Mahalanobis distance must be nonnegative",
                )

    snapshot_paths = payload.get("snapshot_paths")
    _require(
        snapshot_paths == [],
        "snapshot_paths must be empty for the formal no-frames run",
    )

    npz = _load_npz(npz_path)
    for name, json_array in arrays.items():
        npz_array = npz[name]
        _require(
            np.issubdtype(npz_array.dtype, np.number), f"NPZ {name} must be numeric"
        )
        _require(npz_array.shape == json_array.shape, f"NPZ/JSON {name} shapes differ")
        _require(np.isfinite(npz_array).all(), f"NPZ {name} contains NaN or infinity")
        _require(
            np.array_equal(npz_array, json_array), f"NPZ/JSON {name} values differ"
        )

    accepted_npz = npz["ekf_update_accepted"]
    _require(
        accepted_npz.dtype == np.bool_,
        "NPZ ekf_update_accepted must have boolean dtype",
    )
    _require(
        np.array_equal(accepted_npz, np.asarray(accepted_json, dtype=np.bool_)),
        "NPZ/JSON EKF acceptance differs",
    )
    mahalanobis_expected = np.asarray(
        [np.nan if value is None else value for value in mahalanobis_json],
        dtype=np.float64,
    )
    mahalanobis_npz = npz["ekf_mahalanobis"]
    _require(
        np.issubdtype(mahalanobis_npz.dtype, np.number),
        "NPZ ekf_mahalanobis must be numeric",
    )
    _require(
        np.array_equal(np.isnan(mahalanobis_npz), np.isnan(mahalanobis_expected))
        and np.array_equal(
            mahalanobis_npz[~np.isnan(mahalanobis_npz)],
            mahalanobis_expected[~np.isnan(mahalanobis_expected)],
        ),
        "NPZ/JSON EKF Mahalanobis values differ",
    )

    metrics = payload.get("metrics")
    _require(isinstance(metrics, dict), f"{json_path.name} metrics must be an object")
    _require(
        metrics.get("track") == track.name,
        f"metrics.track must be {track.name}",
    )
    _require(
        metrics.get("estimator") == estimator, f"metrics.estimator must be {estimator}"
    )
    _require(metrics.get("succeeded") is True, f"{estimator} did not succeed")
    _require(
        metrics.get("termination_reason") == "controller_complete",
        f"{estimator} did not terminate by controller completion",
    )
    _require(metrics.get("failure_reason") is None, f"{estimator} has a failure_reason")
    _require(
        metrics.get("controller_completed") is True,
        f"{estimator} controller did not complete",
    )
    steps = _integer(metrics.get("steps"), "metrics.steps")
    _require(
        steps == control_count and 0 < steps < expected_max_steps,
        "successful steps must equal controls and remain below max_steps",
    )
    _require(
        _integer(metrics.get("max_steps"), "metrics.max_steps") == expected_max_steps,
        "metrics.max_steps mismatch",
    )
    _same_float(metrics.get("dt"), expected_dt, "metrics.dt")
    _same_float(metrics.get("duration_s"), steps * expected_dt, "metrics.duration_s")
    _same_float(
        metrics.get("gate_radius_m"), expected_gate_radius, "metrics.gate_radius_m"
    )
    _same_float(
        metrics.get("crossing_hysteresis_m"),
        expected_hysteresis,
        "metrics.crossing_hysteresis_m",
    )
    _require(
        _integer(metrics.get("rendered_observations"), "metrics.rendered_observations")
        == state_count,
        "rendered observation count mismatch",
    )
    _require(
        _integer(metrics.get("snapshots_written"), "metrics.snapshots_written") == 0,
        "formal run wrote snapshots",
    )

    sample_counts = metrics.get("sample_counts")
    _require(isinstance(sample_counts, dict), "metrics.sample_counts must be an object")
    for name, array in arrays.items():
        _require(
            _integer(sample_counts.get(name), f"sample_counts.{name}") == len(array),
            f"sample_counts.{name} mismatch",
        )
    for metric_name in ("raw_npe", "controller_estimate"):
        metric = metrics.get(metric_name)
        _require(isinstance(metric, dict), f"metrics.{metric_name} must be an object")
        _require(
            _integer(metric.get("samples"), f"metrics.{metric_name}.samples")
            == state_count,
            f"metrics.{metric_name}.samples mismatch",
        )

    alignment = payload.get("sample_alignment")
    _require(isinstance(alignment, dict), "sample_alignment must be an object")
    _require(
        alignment.get("contract")
        == "z[k] and xhat[k] observe s[k]; validated u[k] advances s[k] to s[k+1]",
        "sample_alignment contract changed",
    )
    for json_name, npz_prefix, name in (
        ("states", "state", "states"),
        ("observations", "observation", "observations"),
        ("estimated_states", "estimate", "estimated_states"),
        ("controls", "control", "controls"),
        ("camera_to_world", "camera", "camera_to_world"),
    ):
        _validate_axis(
            alignment,
            npz,
            json_name=json_name,
            npz_prefix=npz_prefix,
            count=len(arrays[name]),
            dt=expected_dt,
        )

    strict = _compare_strict_evaluation(
        metrics.get("strict_evaluation"),
        arrays["states"],
        track=track,
        dt=expected_dt,
        laps=expected_laps,
        gate_radius=expected_gate_radius,
    )
    controller_passes = metrics.get("controller_passes")
    _require(
        isinstance(controller_passes, list), "metrics.controller_passes must be a list"
    )
    expected_gates = list(track.gate_order) * expected_laps
    _require(
        len(controller_passes) == len(expected_gates),
        f"controller must record exactly {len(expected_gates)} pass events",
    )
    previous_step = -1
    for index, (item, expected_gate) in enumerate(
        zip(controller_passes, expected_gates)
    ):
        _require(isinstance(item, dict), f"controller pass {index} must be an object")
        _require(
            item.get("gate") == expected_gate,
            f"controller pass sequence is not the canonical {track.name} order",
        )
        pass_step = _integer(item.get("step"), f"controller_passes[{index}].step")
        _require(
            previous_step <= pass_step <= steps,
            f"controller pass {index} has an invalid step",
        )
        previous_step = pass_step
        pass_error = _finite_float(
            item.get("radial_error_m"),
            f"controller_passes[{index}].radial_error_m",
        )
        _require(
            0.0 <= pass_error <= expected_gate_radius,
            f"controller pass {index} radial error lies outside the gate",
        )

    _require(
        _integer(metadata.get("seed"), "metadata.seed") == expected_seed,
        "metadata.seed mismatch",
    )
    _require(metadata.get("device") == expected_device, "metadata.device mismatch")
    _require(
        metadata.get("amp_enabled") is False,
        "formal acceptance requires float32 inference without AMP",
    )
    _require(
        _integer(metadata.get("renderer_checkpoint_step"), "renderer_checkpoint_step")
        == expected_renderer_step,
        "renderer checkpoint step mismatch",
    )
    _require(
        _integer(metadata.get("renderer_gaussian_count"), "renderer_gaussian_count")
        == expected_gaussians,
        "renderer Gaussian count mismatch",
    )
    return payload, arrays, strict


def _asset_provenance(
    manifest_path: Path,
    asset_root: Path,
    *,
    track: str,
    renderer_checkpoint: Path | None,
    dataparser_transform: Path | None,
) -> tuple[Path, Path, str, str, int, int]:
    manifest = _load_json(manifest_path.resolve())
    _require(
        manifest.get("schema_version") == 1, "asset manifest schema_version must be 1"
    )
    try:
        track_manifest = manifest["tracks"][track]
        files = track_manifest["files"]
    except Exception as exc:
        raise VerificationError(
            f"asset manifest lacks the {track} file contract"
        ) from exc

    explicit_paths = (renderer_checkpoint, dataparser_transform)
    _require(
        all(path is None for path in explicit_paths)
        or all(path is not None for path in explicit_paths),
        "--renderer-checkpoint and --dataparser-transform must be supplied together",
    )
    if renderer_checkpoint is None:
        try:
            run = track_manifest["run"]
        except Exception as exc:
            raise VerificationError(
                f"asset manifest lacks the {track} default run"
            ) from exc
        _require(
            isinstance(run, str) and run,
            f"{track} manifest run must be a string",
        )
        run_root = asset_root.expanduser().resolve() / track / "splatfacto" / run
        renderer = (run_root / RENDERER_RELATIVE_PATH).resolve()
        transform = (run_root / TRANSFORM_RELATIVE_PATH).resolve()
    else:
        assert dataparser_transform is not None
        renderer = renderer_checkpoint.expanduser().resolve()
        transform = dataparser_transform.expanduser().resolve()

    def verify(relative: str, path: Path) -> tuple[Path, str, Mapping[str, Any]]:
        try:
            record = files[relative]
            expected_size = _integer(record["size_bytes"], f"manifest {relative} size")
            expected_sha = str(record["sha256"]).lower()
        except Exception as exc:
            raise VerificationError(
                f"asset manifest lacks complete provenance for {relative}"
            ) from exc
        _require(
            len(expected_sha) == 64
            and all(c in "0123456789abcdef" for c in expected_sha),
            f"invalid manifest SHA-256 for {relative}",
        )
        _require(path.is_file(), f"missing {track} asset: {path}")
        _require(
            path.stat().st_size == expected_size, f"{track} asset size mismatch: {path}"
        )
        actual_sha = _sha256(path)
        _require(actual_sha == expected_sha, f"{track} asset SHA-256 mismatch: {path}")
        _require(
            isinstance(record, dict), f"manifest {relative} record must be an object"
        )
        return path, actual_sha, record

    renderer, renderer_sha, renderer_record = verify(RENDERER_RELATIVE_PATH, renderer)
    transform, transform_sha, _ = verify(TRANSFORM_RELATIVE_PATH, transform)

    legacy_contract = LEGACY_RENDERER_CONTRACTS.get(track)
    manifest_step = renderer_record.get("step")
    manifest_gaussians = renderer_record.get("gaussians")
    if manifest_step is None and legacy_contract is not None:
        manifest_step = legacy_contract[0]
    if manifest_gaussians is None and legacy_contract is not None:
        manifest_gaussians = legacy_contract[1]
    _require(
        manifest_step is not None,
        f"manifest lacks renderer checkpoint step for {track}",
    )
    _require(
        manifest_gaussians is not None,
        f"manifest lacks renderer Gaussian count for {track}",
    )
    renderer_step = _integer(manifest_step, f"manifest {track} renderer step")
    gaussian_count = _integer(manifest_gaussians, f"manifest {track} Gaussian count")
    _require(renderer_step >= 0, "manifest renderer step cannot be negative")
    _require(gaussian_count > 0, "manifest Gaussian count must be positive")

    return (
        renderer,
        transform,
        renderer_sha,
        transform_sha,
        renderer_step,
        gaussian_count,
    )


def verify_closed_loop_output(
    output_dir: str | Path,
    *,
    npe_checkpoint: str | Path,
    track: str = "circle",
    manifest_path: str | Path = REPOSITORY_ROOT
    / "configs"
    / "assets"
    / "manifest.json",
    asset_root: str | Path = REPOSITORY_ROOT / "outputs",
    renderer_checkpoint: str | Path | None = None,
    dataparser_transform: str | Path | None = None,
    expected_npe_sha256: str | None = None,
    expected_seed: int = 42,
    expected_device: str = "cuda:0",
    expected_max_steps: int = 1200,
    expected_dt: float = 0.05,
    expected_laps: int = 2,
    expected_gate_radius: float = 0.38,
    expected_hysteresis: float = 0.05,
    expected_renderer_step: int | None = None,
    expected_gaussians: int | None = None,
) -> dict[str, Any]:
    """Verify both formal track artifacts or raise :class:`VerificationError`."""

    config = get_track(track)
    output = Path(output_dir).expanduser().resolve()
    _require(output.is_dir(), f"output directory does not exist: {output}")
    actual_children = frozenset(path.name for path in output.iterdir())
    expected_files = _expected_files(config.name)
    _require(
        actual_children == expected_files,
        f"output members differ: missing={sorted(expected_files - actual_children)}, "
        f"extra={sorted(actual_children - expected_files)}",
    )
    _require(
        isinstance(expected_seed, int) and not isinstance(expected_seed, bool),
        "expected_seed must be an integer",
    )
    _require(
        isinstance(expected_max_steps, int) and expected_max_steps > 0,
        "expected_max_steps must be positive",
    )
    _require(
        isinstance(expected_laps, int) and expected_laps == 2,
        "formal acceptance requires exactly 2 laps",
    )
    for value, name in (
        (expected_dt, "expected_dt"),
        (expected_gate_radius, "expected_gate_radius"),
        (expected_hysteresis, "expected_hysteresis"),
    ):
        _require(
            math.isfinite(float(value)) and float(value) > 0.0,
            f"{name} must be positive and finite",
        )

    npe = Path(npe_checkpoint).expanduser().resolve()
    npe_sha = _sha256(npe)
    if expected_npe_sha256 is not None:
        expected_hash = expected_npe_sha256.lower()
        _require(
            len(expected_hash) == 64
            and all(c in "0123456789abcdef" for c in expected_hash),
            "expected NPE SHA-256 is malformed",
        )
        _require(
            npe_sha == expected_hash,
            "NPE checkpoint does not match --expected-npe-sha256",
        )

    (
        renderer,
        transform,
        renderer_sha,
        transform_sha,
        manifest_renderer_step,
        manifest_gaussians,
    ) = _asset_provenance(
        Path(manifest_path),
        Path(asset_root),
        track=config.name,
        renderer_checkpoint=(
            None if renderer_checkpoint is None else Path(renderer_checkpoint)
        ),
        dataparser_transform=(
            None if dataparser_transform is None else Path(dataparser_transform)
        ),
    )
    if expected_renderer_step is None:
        expected_renderer_step = manifest_renderer_step
    else:
        _require(
            expected_renderer_step == manifest_renderer_step,
            "--expected-renderer-step disagrees with the selected track manifest",
        )
    if expected_gaussians is None:
        expected_gaussians = manifest_gaussians
    else:
        _require(
            expected_gaussians == manifest_gaussians,
            "--expected-gaussians disagrees with the selected track manifest",
        )

    first_payload = _load_json(output / f"{config.name}_raw.json")
    metadata = first_payload.get("metadata")
    _require(
        isinstance(metadata, dict),
        f"{config.name}_raw.json metadata must be an object",
    )
    expected_paths = {
        "npe_checkpoint": npe,
        "renderer_checkpoint": renderer,
        "dataparser_transform": transform,
    }
    for key, expected_path in expected_paths.items():
        value = metadata.get(key)
        _require(isinstance(value, str), f"metadata.{key} must be a path string")
        _require(
            Path(value).expanduser().resolve() == expected_path,
            f"metadata.{key} does not match the selected provenance file",
        )
    _require(
        metadata.get("npe_checkpoint_sha256") == npe_sha,
        "metadata NPE SHA-256 does not match checkpoint bytes",
    )

    results: dict[
        str, tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]
    ] = {}
    for estimator in ESTIMATORS:
        results[estimator] = _validate_run(
            output,
            estimator,
            track=config,
            expected_metadata=metadata,
            expected_seed=expected_seed,
            expected_device=expected_device,
            expected_max_steps=expected_max_steps,
            expected_dt=expected_dt,
            expected_laps=expected_laps,
            expected_gate_radius=expected_gate_radius,
            expected_hysteresis=expected_hysteresis,
            expected_renderer_step=expected_renderer_step,
            expected_gaussians=expected_gaussians,
        )

    raw_arrays = results["raw"][1]
    ekf_arrays = results["ekf"][1]
    _require(
        np.array_equal(raw_arrays["states"][0], ekf_arrays["states"][0]),
        "raw/EKF initial truth states differ",
    )
    _require(
        np.array_equal(
            raw_arrays["camera_to_world"][0], ekf_arrays["camera_to_world"][0]
        ),
        "raw/EKF first camera poses differ",
    )
    _require(
        np.allclose(
            raw_arrays["observations"][0],
            ekf_arrays["observations"][0],
            rtol=0.0,
            atol=1e-5,
        ),
        "raw/EKF first NPE observations differ beyond 1e-5",
    )

    return {
        "status": "PASS",
        "track": config.name,
        "estimators": {
            estimator: {
                "steps": int(results[estimator][0]["metrics"]["steps"]),
                "strict_crossings": int(results[estimator][2]["successful_crossings"]),
                "mean_gate_error_cm": 100.0
                * float(results[estimator][2]["mean_gate_error_m"]),
            }
            for estimator in ESTIMATORS
        },
        "npe_checkpoint": str(npe),
        "npe_checkpoint_sha256": npe_sha,
        "renderer_checkpoint": str(renderer),
        "renderer_checkpoint_sha256": renderer_sha,
        "dataparser_transform": str(transform),
        "dataparser_transform_sha256": transform_sha,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed CPU verification of raw+EKF closed-loop artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--track", choices=sorted(TRACKS), default="circle")
    parser.add_argument("--npe-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-npe-sha256")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "assets" / "manifest.json",
    )
    parser.add_argument("--asset-root", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument(
        "--renderer-checkpoint",
        type=Path,
        help="Explicit checkpoint path; must be paired with --dataparser-transform",
    )
    parser.add_argument(
        "--dataparser-transform",
        type=Path,
        help="Explicit transform path; must be paired with --renderer-checkpoint",
    )
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-device", default="cuda:0")
    parser.add_argument("--expected-max-steps", type=int, default=1200)
    parser.add_argument("--expected-dt", type=float, default=0.05)
    parser.add_argument("--expected-laps", type=int, default=2)
    parser.add_argument("--expected-gate-radius", type=float, default=0.38)
    parser.add_argument("--expected-hysteresis", type=float, default=0.05)
    parser.add_argument(
        "--expected-renderer-step",
        type=int,
        help="Must match the selected track's manifest contract",
    )
    parser.add_argument(
        "--expected-gaussians",
        type=int,
        help="Must match the selected track's manifest contract",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_closed_loop_output(
            args.output_dir,
            npe_checkpoint=args.npe_checkpoint,
            track=args.track,
            manifest_path=args.manifest,
            asset_root=args.asset_root,
            renderer_checkpoint=args.renderer_checkpoint,
            dataparser_transform=args.dataparser_transform,
            expected_npe_sha256=args.expected_npe_sha256,
            expected_seed=args.expected_seed,
            expected_device=args.expected_device,
            expected_max_steps=args.expected_max_steps,
            expected_dt=args.expected_dt,
            expected_laps=args.expected_laps,
            expected_gate_radius=args.expected_gate_radius,
            expected_hysteresis=args.expected_hysteresis,
            expected_renderer_step=args.expected_renderer_step,
            expected_gaussians=args.expected_gaussians,
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
