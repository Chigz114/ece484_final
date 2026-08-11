"""CPU-only tests for the fail-closed visual closed-loop artifact verifier."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quadpilot.cli import verify_closed_loop as verifier
from quadpilot.simulation.evaluation import evaluate_ordered_gates
from quadpilot.simulation.tracks import get_track


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track_states(track_name: str) -> np.ndarray:
    """Tiny path that genuinely crosses one track twice in canonical order."""

    track = get_track(track_name)
    rows = [np.asarray(track.initial_state, dtype=np.float64)]
    for target_index, gate in enumerate(track.ordered_gates() * 2):
        center = np.asarray(gate.center, dtype=np.float64)
        side = float(track.incoming_gate_sides[target_index % len(track.gate_order)])
        before = center + side * 0.1 * gate.normal
        after = center - side * 0.1 * gate.normal
        for position in (before, after):
            state = np.zeros(7, dtype=np.float64)
            state[:3] = position
            state[6] = np.deg2rad(gate.yaw_deg)
            rows.append(state)
    result = np.asarray(rows, dtype=np.float64)
    evaluation = evaluate_ordered_gates(
        result, track, dt=0.05, laps=2, gate_radius=0.38
    )
    if not evaluation.completed:
        raise AssertionError(f"synthetic fixture must strictly complete {track_name}")
    return result


class ClosedLoopArtifactFixture:
    GAUSSIANS = {
        "circle": 308832,
        "lemniscate": 394366,
        "uturn": 437285,
    }

    def __init__(
        self,
        root: Path,
        *,
        track: str = "circle",
        explicit_assets: bool = False,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.track = track
        self.config = get_track(track)
        self.gaussians = self.GAUSSIANS[track]
        self.explicit_assets = explicit_assets
        self.output = root / "output"
        self.output.mkdir()
        self.asset_root = root / "assets"
        manifest_run = "historical-run" if explicit_assets else "test-run"
        self.run_root = (
            root / "explicit-assets" / track
            if explicit_assets
            else self.asset_root / track / "splatfacto" / manifest_run
        )
        (self.run_root / "nerfstudio_models").mkdir(parents=True)
        self.renderer = self.run_root / "nerfstudio_models" / "step-000029999.ckpt"
        self.transform = self.run_root / "dataparser_transforms.json"
        self.renderer.write_bytes(b"synthetic-renderer-checkpoint")
        self.transform.write_text('{"synthetic": true}\n', encoding="utf-8")
        self.npe = root / "best_npe.pth"
        self.npe.write_bytes(b"synthetic-finished-npe-checkpoint")
        self.manifest = root / "manifest.json"
        renderer_record = {
            "size_bytes": self.renderer.stat().st_size,
            "sha256": _sha256(self.renderer),
        }
        if track != "circle":
            renderer_record.update({"step": 29999, "gaussians": self.gaussians})
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tracks": {
                        track: {
                            "run": manifest_run,
                            "files": {
                                "nerfstudio_models/step-000029999.ckpt": renderer_record,
                                "dataparser_transforms.json": {
                                    "size_bytes": self.transform.stat().st_size,
                                    "sha256": _sha256(self.transform),
                                },
                            },
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.metadata = {
            "npe_checkpoint": str(self.npe.resolve()),
            "npe_checkpoint_sha256": _sha256(self.npe),
            "renderer_checkpoint": str(self.renderer.resolve()),
            "renderer_checkpoint_step": 29999,
            "renderer_gaussian_count": self.gaussians,
            "dataparser_transform": str(self.transform.resolve()),
            "device": "cuda:0",
            "cuda_toolchain": {"CUDA_HOME": "/synthetic/cuda-11.8"},
            "amp_enabled": False,
            "seed": 42,
        }
        self.states = _track_states(track)
        self._write_run("raw")
        self._write_run("ekf")

    def _write_run(self, estimator: str) -> None:
        states = self.states.copy()
        if estimator == "ekf":
            # Closed-loop trajectories are allowed to diverge after the shared
            # first sensor sample.  Keep this tiny perturbation inside each gate.
            states[1:, 2] += 0.001
        count = len(states)
        controls = np.zeros((count - 1, 4), dtype=np.float64)
        observations = states[:, [0, 1, 2, 6]].copy()
        estimates = states.copy()
        camera = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
        camera[:, :3, 3] = states[:, :3]
        evaluation = evaluate_ordered_gates(
            states, self.config, dt=0.05, laps=2, gate_radius=0.38
        )
        if not evaluation.completed:
            raise AssertionError("fixture perturbation must preserve strict completion")
        accepted = [] if estimator == "raw" else [True] * count
        mahalanobis_json = [] if estimator == "raw" else [None] + [0.1] * (count - 1)
        mahalanobis_npz = np.asarray(
            [np.nan if value is None else value for value in mahalanobis_json],
            dtype=np.float64,
        )
        steps = count - 1
        names = {
            "states": states,
            "observations": observations,
            "estimated_states": estimates,
            "controls": controls,
            "camera_to_world": camera,
        }

        def axis(length: int) -> dict[str, list[float] | list[int]]:
            indices = np.arange(length, dtype=np.int64)
            return {
                "step_indices": indices.tolist(),
                "times_s": (indices.astype(np.float64) * 0.05).tolist(),
            }

        stem = f"{self.track}_{estimator}"
        json_path = (self.output / f"{stem}.json").resolve()
        npz_path = (self.output / f"{stem}.npz").resolve()
        payload = {
            "schema_version": 1,
            "metadata": dict(self.metadata),
            "metrics": {
                "track": self.track,
                "estimator": estimator,
                "succeeded": True,
                "termination_reason": "controller_complete",
                "failure_reason": None,
                "steps": steps,
                "duration_s": steps * 0.05,
                "dt": 0.05,
                "max_steps": 1200,
                "gate_radius_m": 0.38,
                "crossing_hysteresis_m": 0.05,
                "controller_completed": True,
                "controller_passes": [
                    {
                        "step": index + 1,
                        "gate": crossing.gate,
                        "radial_error_m": crossing.radial_error_m,
                    }
                    for index, crossing in enumerate(evaluation.crossings)
                ],
                "strict_evaluation": evaluation.to_dict(),
                "raw_npe": {"samples": count},
                "controller_estimate": {"samples": count},
                "truth_position_step_jitter_cm": 0.0,
                "ekf_updates_accepted": sum(accepted),
                "ekf_updates_rejected": len(accepted) - sum(accepted),
                "rendered_observations": count,
                "snapshots_written": 0,
                "sample_counts": {name: len(array) for name, array in names.items()},
            },
            "sample_alignment": {
                "contract": (
                    "z[k] and xhat[k] observe s[k]; validated u[k] advances "
                    "s[k] to s[k+1]"
                ),
                **{name: axis(len(array)) for name, array in names.items()},
            },
            **{name: array.tolist() for name, array in names.items()},
            "ekf_update_accepted": accepted,
            "ekf_mahalanobis": mahalanobis_json,
            "snapshot_paths": [],
            "artifacts": {"json": str(json_path), "npz": str(npz_path)},
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        state_steps = np.arange(len(states), dtype=np.int64)
        control_steps = np.arange(len(controls), dtype=np.int64)
        np.savez_compressed(
            npz_path,
            **names,
            ekf_update_accepted=np.asarray(accepted, dtype=np.bool_),
            ekf_mahalanobis=mahalanobis_npz,
            state_step_indices=state_steps,
            state_times_s=state_steps.astype(np.float64) * 0.05,
            observation_step_indices=state_steps,
            observation_times_s=state_steps.astype(np.float64) * 0.05,
            estimate_step_indices=state_steps,
            estimate_times_s=state_steps.astype(np.float64) * 0.05,
            control_step_indices=control_steps,
            control_times_s=control_steps.astype(np.float64) * 0.05,
            camera_step_indices=state_steps,
            camera_times_s=state_steps.astype(np.float64) * 0.05,
        )

    def verify(self) -> dict[str, object]:
        explicit = (
            {
                "renderer_checkpoint": self.renderer,
                "dataparser_transform": self.transform,
            }
            if self.explicit_assets
            else {}
        )
        return verifier.verify_closed_loop_output(
            self.output,
            npe_checkpoint=self.npe,
            track=self.track,
            manifest_path=self.manifest,
            asset_root=self.asset_root,
            expected_npe_sha256=_sha256(self.npe),
            **explicit,
        )

    def read_json(self, estimator: str) -> dict[str, object]:
        return json.loads(
            (self.output / f"{self.track}_{estimator}.json").read_text(encoding="utf-8")
        )

    def write_json(self, estimator: str, payload: dict[str, object]) -> None:
        (self.output / f"{self.track}_{estimator}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def mutate_npz(self, estimator: str, name: str, value: np.ndarray) -> None:
        path = self.output / f"{self.track}_{estimator}.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
        arrays[name] = value
        np.savez_compressed(path, **arrays)


class ClosedLoopVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = ClosedLoopArtifactFixture(Path(self.temporary.name))

    def test_valid_both_run_passes_and_cli_returns_zero(self) -> None:
        report = self.fixture.verify()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["track"], "circle")
        self.assertEqual(report["estimators"]["raw"]["strict_crossings"], 8)
        self.assertEqual(report["estimators"]["ekf"]["strict_crossings"], 8)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = verifier.main(
                [
                    str(self.fixture.output),
                    "--npe-checkpoint",
                    str(self.fixture.npe),
                    "--expected-npe-sha256",
                    _sha256(self.fixture.npe),
                    "--manifest",
                    str(self.fixture.manifest),
                    "--asset-root",
                    str(self.fixture.asset_root),
                ]
            )
        self.assertEqual(status, 0)

    def test_lemniscate_explicit_assets_and_canonical_order_pass(self) -> None:
        lemniscate = ClosedLoopArtifactFixture(
            Path(self.temporary.name) / "lemniscate",
            track="lemniscate",
            explicit_assets=True,
        )
        default_run = (
            lemniscate.asset_root / "lemniscate" / "splatfacto" / "historical-run"
        )
        self.assertFalse(default_run.exists())

        report = lemniscate.verify()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["track"], "lemniscate")
        payload = lemniscate.read_json("raw")
        self.assertEqual(
            [item["gate"] for item in payload["metrics"]["controller_passes"]],
            ["Gate D", "Gate A", "Gate B", "Gate C"] * 2,
        )

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = verifier.main(
                [
                    str(lemniscate.output),
                    "--track",
                    "lemniscate",
                    "--npe-checkpoint",
                    str(lemniscate.npe),
                    "--expected-npe-sha256",
                    _sha256(lemniscate.npe),
                    "--manifest",
                    str(lemniscate.manifest),
                    "--asset-root",
                    str(lemniscate.asset_root),
                    "--renderer-checkpoint",
                    str(lemniscate.renderer),
                    "--dataparser-transform",
                    str(lemniscate.transform),
                    "--expected-renderer-step",
                    "29999",
                    "--expected-gaussians",
                    "394366",
                ]
            )
        self.assertEqual(status, 0)

    def test_wrong_track_hash_gaussian_and_half_explicit_assets_fail(self) -> None:
        lemniscate = ClosedLoopArtifactFixture(
            Path(self.temporary.name) / "lemniscate-rejections",
            track="lemniscate",
            explicit_assets=True,
        )

        with self.assertRaisesRegex(
            verifier.VerificationError, "output members differ"
        ):
            verifier.verify_closed_loop_output(
                lemniscate.output,
                track="circle",
                npe_checkpoint=lemniscate.npe,
                manifest_path=lemniscate.manifest,
                asset_root=lemniscate.asset_root,
                renderer_checkpoint=lemniscate.renderer,
                dataparser_transform=lemniscate.transform,
            )

        with self.assertRaisesRegex(
            verifier.VerificationError, "must be supplied together"
        ):
            verifier.verify_closed_loop_output(
                lemniscate.output,
                track="lemniscate",
                npe_checkpoint=lemniscate.npe,
                manifest_path=lemniscate.manifest,
                asset_root=lemniscate.asset_root,
                renderer_checkpoint=lemniscate.renderer,
            )

        with self.assertRaisesRegex(
            verifier.VerificationError, "expected-gaussians disagrees"
        ):
            verifier.verify_closed_loop_output(
                lemniscate.output,
                track="lemniscate",
                npe_checkpoint=lemniscate.npe,
                manifest_path=lemniscate.manifest,
                asset_root=lemniscate.asset_root,
                renderer_checkpoint=lemniscate.renderer,
                dataparser_transform=lemniscate.transform,
                expected_gaussians=394365,
            )

        original_size = lemniscate.renderer.stat().st_size
        lemniscate.renderer.write_bytes(b"x" * original_size)
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 mismatch"):
            lemniscate.verify()

    def test_missing_artifact_and_frames_directory_fail_closed(self) -> None:
        (self.fixture.output / "circle_ekf.npz").unlink()
        with self.assertRaisesRegex(
            verifier.VerificationError, "missing=.*circle_ekf.npz"
        ):
            self.fixture.verify()

        self.fixture._write_run("ekf")
        (self.fixture.output / "circle_raw_frames").mkdir()
        with self.assertRaisesRegex(
            verifier.VerificationError, "extra=.*circle_raw_frames"
        ):
            self.fixture.verify()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = verifier.main(
                [
                    str(self.fixture.output),
                    "--npe-checkpoint",
                    str(self.fixture.npe),
                    "--manifest",
                    str(self.fixture.manifest),
                    "--asset-root",
                    str(self.fixture.asset_root),
                ]
            )
        self.assertEqual(status, 1)

    def test_json_npz_mismatch_and_nonfinite_npz_fail_closed(self) -> None:
        path = self.fixture.output / "circle_raw.npz"
        with np.load(path, allow_pickle=False) as archive:
            observations = archive["observations"].copy()
        observations[1, 0] += 0.25
        self.fixture.mutate_npz("raw", "observations", observations)
        with self.assertRaisesRegex(
            verifier.VerificationError, "NPZ/JSON observations values differ"
        ):
            self.fixture.verify()

        self.fixture._write_run("raw")
        with np.load(path, allow_pickle=False) as archive:
            states = archive["states"].copy()
        states[1, 0] = np.nan
        self.fixture.mutate_npz("raw", "states", states)
        with self.assertRaisesRegex(
            verifier.VerificationError, "NPZ states contains NaN"
        ):
            self.fixture.verify()

        self.fixture._write_run("raw")
        payload = self.fixture.read_json("raw")
        payload["states"][1][0] = float("nan")
        self.fixture.write_json("raw", payload)
        with self.assertRaisesRegex(
            verifier.VerificationError, "non-standard non-finite"
        ):
            self.fixture.verify()

    def test_metric_sample_count_mismatch_fails_closed(self) -> None:
        payload = self.fixture.read_json("raw")
        payload["metrics"]["sample_counts"]["states"] += 1
        self.fixture.write_json("raw", payload)
        with self.assertRaisesRegex(
            verifier.VerificationError, "sample_counts.states mismatch"
        ):
            self.fixture.verify()

    def test_recomputed_strict_evaluator_rejects_fabricated_success(self) -> None:
        payload = self.fixture.read_json("raw")
        states = np.asarray(payload["states"], dtype=np.float64)
        states[1:, :3] = states[0, :3]
        payload["states"] = states.tolist()
        self.fixture.write_json("raw", payload)
        self.fixture.mutate_npz("raw", "states", states)
        with self.assertRaisesRegex(
            verifier.VerificationError, "recomputed strict evaluation did not complete"
        ):
            self.fixture.verify()

    def test_saved_strict_crossing_order_must_match_recomputation(self) -> None:
        payload = self.fixture.read_json("raw")
        payload["metrics"]["strict_evaluation"]["crossings"][0]["gate"] = "Gate D"
        self.fixture.write_json("raw", payload)
        with self.assertRaisesRegex(
            verifier.VerificationError, "crossing 0 gate disagrees"
        ):
            self.fixture.verify()

    def test_raw_and_ekf_first_sensor_sample_must_be_fair(self) -> None:
        payload = self.fixture.read_json("ekf")
        observations = np.asarray(payload["observations"], dtype=np.float64)
        observations[0, 0] += 1e-3
        payload["observations"] = observations.tolist()
        self.fixture.write_json("ekf", payload)
        self.fixture.mutate_npz("ekf", "observations", observations)
        with self.assertRaisesRegex(
            verifier.VerificationError, "first NPE observations differ"
        ):
            self.fixture.verify()

    def test_npe_and_renderer_provenance_are_hashed(self) -> None:
        self.fixture.npe.write_bytes(b"replaced-npe")
        with self.assertRaisesRegex(verifier.VerificationError, "NPE SHA-256"):
            self.fixture.verify()

        self.fixture.npe.write_bytes(b"synthetic-finished-npe-checkpoint")
        self.fixture.renderer.write_bytes(b"replaced-renderer")
        with self.assertRaisesRegex(verifier.VerificationError, "asset size mismatch"):
            self.fixture.verify()


if __name__ == "__main__":
    unittest.main()
