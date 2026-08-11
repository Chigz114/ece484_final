"""CPU-only tests for independent recovery verification of a GSplat run."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import shlex
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from quadpilot.cli import verify_gsplat as verifier


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shell_record(values: dict[str, object]) -> str:
    return (
        "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())
        + "\n"
    )


def _method_audit() -> dict[str, object]:
    return {
        "disabled_entry_points": [
            dict(zip(verifier.METHOD_ENTRY_FIELDS, row))
            for row in verifier.EXPECTED_METHOD_ENTRY_POINTS
        ],
        "disabled_group": "nerfstudio.method_configs",
        "nerfstudio_source_sha256": verifier.EXPECTED_NERFSTUDIO_SOURCES,
        "policy": "built-in-only",
    }


class SyntheticRecoveredRun:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_id = "synthetic_lem_train30k"
        self.run = root / "outputs" / "lemniscate" / "train-30k" / self.run_id
        self.run.mkdir(parents=True)
        self.data = root / "sources" / "lemniscate"
        (self.data / "images").mkdir(parents=True)
        self.cache = root / "cache"
        self.lpips = self.cache / verifier.LPIPS_RELATIVE_PATH
        self.lpips.parent.mkdir(parents=True)
        self.lpips.write_bytes(b"synthetic-alexnet-cache")
        self.lpips_size = self.lpips.stat().st_size
        self.lpips_sha = _sha256(self.lpips)

        (self.data / "images" / "a.png").write_bytes(b"png-a")
        (self.data / "images" / "b.png").write_bytes(b"png-bb")
        transforms = {
            "frames": [
                {"file_path": "images/a.png", "transform_matrix": [[1, 0], [0, 1]]},
                {"file_path": "images/b.png", "transform_matrix": [[1, 0], [0, 1]]},
            ]
        }
        (self.data / "transforms.json").write_text(
            json.dumps(transforms, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.data / "sparse_pc.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            "0 0 0\n1 0 0\n0 1 0\n",
            encoding="ascii",
        )
        receipt_files: dict[str, dict[str, object]] = {}
        for relative in (
            "images/a.png",
            "images/b.png",
            "sparse_pc.ply",
            "transforms.json",
        ):
            path = self.data / Path(*relative.split("/"))
            receipt_files[relative] = {
                "file_id": f"synthetic-{relative}",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        self.receipt = self.data / ".quadpilot_source_receipt.json"
        self.receipt.write_text(
            json.dumps(
                {
                    "files": receipt_files,
                    "images_folder_id": "synthetic",
                    "schema_version": 1,
                    "track": "lemniscate",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        image_bytes = sum(
            receipt_files[name]["size_bytes"]
            for name in ("images/a.png", "images/b.png")
        )
        total_bytes = sum(item["size_bytes"] for item in receipt_files.values())
        self.profile = verifier.SourceProfile(
            name="lemniscate",
            receipt_sha256=_sha256(self.receipt),
            files=4,
            images=2,
            image_bytes=int(image_bytes),
            total_bytes=int(total_bytes),
            sparse_points=3,
        )

        self.training_output = self.run / "training-output"
        self.train_run = (
            self.training_output / "lemniscate" / "splatfacto" / self.run_id
        )
        self.checkpoint_dir = self.train_run / "nerfstudio_models"
        self.checkpoint_dir.mkdir(parents=True)
        self.checkpoint = self.checkpoint_dir / "step-000029999.ckpt"
        self.checkpoint.write_bytes(b"synthetic-checkpoint")
        self.config = self.train_run / "config.yml"
        self.config.write_text(
            """data: [/, data]
experiment_name: lemniscate
max_num_iterations: 30000
method_name: splatfacto
output_dir: [/, outputs]
timestamp: synthetic_lem_train30k
vis: tensorboard
steps_per_eval_batch: 0
steps_per_eval_image: 0
steps_per_eval_all_images: 0
save_only_latest_checkpoint: true
machine:
  seed: 42
  num_devices: 1
  device_type: cuda
pipeline:
  datamanager:
    data: [/, data]
    camera_res_scale_factor: 0.5
    dataparser:
      downscale_factor: 1
      load_3D_points: true
  model:
    num_downscales: 2
    resolution_schedule: 3000
    random_init: false
""",
            encoding="utf-8",
        )

        self.status_values = {
            "started_utc": "2026-08-09T00:00:00Z",
            "finished_utc": "2026-08-09T01:00:00Z",
            "exit_code": 2,
            "result": "failed",
        }
        self.status = self.run / "status.env"
        self.status.write_text(_shell_record(self.status_values), encoding="utf-8")
        provenance_values: dict[str, object] = {
            "schema_version": 1,
            "started_utc": self.status_values["started_utc"],
            "mode": "train-30k",
            "image_ref": verifier.IMAGE_REF,
            "track": "lemniscate",
            "source_profile": "lemniscate",
            "run_id": self.run_id,
            "data_dir": self.data,
            "data_mount_mode": "readonly",
            "expected_source_receipt_sha256": self.profile.receipt_sha256,
            "expected_source_receipt_files": self.profile.files,
            "expected_source_receipt_images": self.profile.images,
            "expected_source_receipt_image_bytes": self.profile.image_bytes,
            "expected_source_receipt_total_bytes": self.profile.total_bytes,
            "expected_source_sparse_points": self.profile.sparse_points,
            "run_dir": self.run,
            "training_output_dir": self.training_output,
            "half_res_linear_scale": 0.5,
            "half_res_pixel_fraction": 0.25,
            "full_resolution_reproduction": "false",
            "lpips_alexnet_cache_path": self.lpips,
            "lpips_alexnet_expected_size_bytes": self.lpips_size,
            "lpips_alexnet_expected_sha256": self.lpips_sha,
            "lpips_alexnet_actual_size_bytes": self.lpips_size,
            "lpips_alexnet_actual_sha256": self.lpips_sha,
            "lpips_alexnet_cache_verified": "true",
            "method_plugin_policy": "built-in-only",
            "max_num_iterations": 30000,
            "expected_final_step": 29999,
            "expected_checkpoint_name": "step-000029999.ckpt",
            "splatfacto_num_downscales": 2,
            "splatfacto_resolution_schedule": 3000,
            "periodic_evaluation_enabled": "false",
            "training_max_jobs": 4,
            "cpu_preflight_executed": "true",
            "finished_utc": self.status_values["finished_utc"],
            "exit_code": 2,
            "result": "failed",
        }
        self.provenance = self.run / "provenance.env"
        self.provenance.write_text(_shell_record(provenance_values), encoding="utf-8")

        receipt_summary = {
            "receipt_sha256": self.profile.receipt_sha256,
            "verified_bytes": self.profile.total_bytes,
            "verified_files": self.profile.files,
            "verified_images": self.profile.images,
        }
        (self.run / "receipt-verification.json").write_text(
            json.dumps(receipt_summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        audit = _method_audit()
        audit_json = json.dumps(audit, sort_keys=True)
        (self.run / "method-plugin-audit.json").write_text(
            audit_json + "\n", encoding="utf-8"
        )
        builtins = {"method_count": 43, "splatfacto_present": True}
        (self.run / "builtin-method-configs.json").write_text(
            json.dumps(builtins, sort_keys=True) + "\n", encoding="utf-8"
        )
        preflight_lines = [
            "METHOD_PLUGIN_AUDIT " + audit_json,
            "VERSIONS " + json.dumps(verifier.EXPECTED_VERSIONS, sort_keys=True),
            "PIP_CHECK "
            + json.dumps(
                {"known_deviations": verifier.EXPECTED_PIP_DEVIATIONS, "returncode": 1},
                sort_keys=True,
            ),
            "RECEIPT_OK " + json.dumps(receipt_summary, sort_keys=True),
            "DATASET "
            + json.dumps(
                {
                    "cameras": self.profile.images,
                    "dataparser_downscale_factor": 1,
                    "images": self.profile.images,
                    "missing_images": 0,
                    "sparse_points": self.profile.sparse_points,
                },
                sort_keys=True,
            ),
            "BUILTIN_METHOD_CONFIGS_OK " + json.dumps(builtins, sort_keys=True),
            "PREFLIGHT_OK",
        ]
        (self.run / "preflight-container.log").write_text(
            "\n".join(preflight_lines) + "\n", encoding="utf-8"
        )
        (self.run / "docker.log").write_text(
            "METHOD_PLUGIN_AUDIT "
            + audit_json
            + "\n29999 (100.00%)\nTraining Finished\n",
            encoding="utf-8",
        )
        (self.run / "docker-image-inspect.json").write_text(
            json.dumps(
                [
                    {
                        "Id": verifier.IMAGE_DIGEST,
                        "RepoDigests": [verifier.IMAGE_REF],
                        "Os": "linux",
                        "Architecture": "amd64",
                    }
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        command_fragments = [
            "#!/usr/bin/env bash",
            "set -Eeuo pipefail",
            "exec /usr/bin/docker run",
            "--pull=never",
            "--network none",
            "--gpus device=0",
            f"type=bind,src={self.data},dst=/data,readonly",
            f"type=bind,src={self.training_output},dst=/outputs",
            f"dst=/cache/{verifier.LPIPS_RELATIVE_PATH},readonly",
            "--env MAX_JOBS=4",
            verifier.IMAGE_REF,
            "METHOD_PLUGIN_AUDIT",
            "splatfacto",
            "--data /data",
            "--output-dir /outputs",
            "--experiment-name lemniscate",
            f"--timestamp {self.run_id}",
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
        ]
        (self.run / "command.sh").write_text(
            "\n".join(command_fragments) + "\n", encoding="utf-8"
        )

    def checkpoint_report(self) -> dict[str, object]:
        return {
            "status": "PASS",
            "load_device": "cpu",
            "step": 29999,
            "gaussian_count": 7,
            "gaussian_tensors": {},
        }

    def verify(self) -> dict[str, object]:
        with (
            mock.patch.object(verifier, "LPIPS_SIZE_BYTES", self.lpips_size),
            mock.patch.object(verifier, "LPIPS_SHA256", self.lpips_sha),
        ):
            return verifier.verify_run(
                self.run,
                track="lemniscate",
                profile=self.profile,
                checkpoint_inspector=lambda _path, _step: self.checkpoint_report(),
            )


class RecoveredGsplatRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticRecoveredRun(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_artifacts_preserve_failed_wrapper_as_separate_dimension(
        self,
    ) -> None:
        before = self.fixture.status.read_bytes()
        report = self.fixture.verify()
        self.assertEqual(
            report["classification"], "WRAPPER_FAILED_TRAINING_ARTIFACTS_VERIFIED"
        )
        self.assertFalse(report["overall_success"])
        self.assertEqual(report["wrapper_status"]["status"], "failed")
        self.assertEqual(report["wrapper_status"]["exit_code"], 2)
        self.assertEqual(report["training_artifacts"]["status"], "PASS")
        self.assertEqual(self.fixture.status.read_bytes(), before)
        self.assertFalse((self.fixture.run / "recovered-postflight.json").exists())

    def test_status_env_is_authoritative_when_provenance_has_no_terminal_fields(
        self,
    ) -> None:
        lines = self.fixture.provenance.read_text(encoding="utf-8").splitlines()
        terminal = ("finished_utc=", "exit_code=", "result=")
        self.fixture.provenance.write_text(
            "\n".join(line for line in lines if not line.startswith(terminal)) + "\n",
            encoding="utf-8",
        )
        report = self.fixture.verify()
        self.assertEqual(report["wrapper_status"]["status"], "failed")
        self.assertEqual(report["wrapper_status"]["exit_code"], 2)

    def test_matches_real_layout_where_status_has_no_started_timestamp(self) -> None:
        lines = self.fixture.status.read_text(encoding="utf-8").splitlines()
        self.fixture.status.write_text(
            "\n".join(line for line in lines if not line.startswith("started_utc="))
            + "\n",
            encoding="utf-8",
        )
        report = self.fixture.verify()
        self.assertEqual(report["wrapper_status"]["status"], "failed")

    def test_atomic_report_write_never_overwrites_or_changes_status(self) -> None:
        before = self.fixture.status.read_bytes()
        report = self.fixture.verify()
        target = verifier.write_recovery_report(self.fixture.run, report)
        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertFalse(saved["overall_success"])
        self.assertEqual(saved["wrapper_status"]["status"], "failed")
        self.assertEqual(saved["training_artifacts"]["status"], "PASS")
        self.assertEqual(self.fixture.status.read_bytes(), before)
        with self.assertRaisesRegex(verifier.VerificationError, "refusing overwrite"):
            verifier.write_recovery_report(self.fixture.run, report)

    def test_requires_exactly_one_expected_checkpoint(self) -> None:
        (self.fixture.checkpoint_dir / "step-000027999.ckpt").write_bytes(b"extra")
        with self.assertRaisesRegex(
            verifier.VerificationError, "exactly step-000029999"
        ):
            self.fixture.verify()

    def test_fails_closed_on_training_exception_even_with_finished_marker(self) -> None:
        docker_log = self.fixture.run / "docker.log"
        docker_log.write_text(
            docker_log.read_text(encoding="utf-8")
            + "Traceback (most recent call last):\nRuntimeError: boom\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            verifier.VerificationError, "fatal training marker"
        ):
            self.fixture.verify()

    def test_fails_closed_when_cpu_and_training_plugin_audits_differ(self) -> None:
        docker_log = self.fixture.run / "docker.log"
        changed = _method_audit()
        changed["policy"] = "load-all"
        docker_log.write_text(
            "METHOD_PLUGIN_AUDIT "
            + json.dumps(changed, sort_keys=True)
            + "\n29999 (100.00%)\nTraining Finished\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            verifier.VerificationError, "plugin policy changed"
        ):
            self.fixture.verify()

    def test_fails_closed_on_source_or_lpips_digest_change(self) -> None:
        (self.fixture.data / "images" / "a.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(
            verifier.VerificationError, "source size differs|source SHA-256"
        ):
            self.fixture.verify()

        # Rebuild, then independently exercise the LPIPS gate.
        self.tearDown()
        self.setUp()
        self.fixture.lpips.write_bytes(b"tampered-lpips")
        with self.assertRaisesRegex(verifier.VerificationError, "LPIPS AlexNet"):
            self.fixture.verify()

    def test_fails_closed_on_config_semantic_change(self) -> None:
        text = self.fixture.config.read_text(encoding="utf-8")
        self.fixture.config.write_text(
            text.replace("num_downscales: 2", "num_downscales: 0"), encoding="utf-8"
        )
        with self.assertRaisesRegex(verifier.VerificationError, "num_downscales"):
            self.fixture.verify()

    def test_cli_failure_does_not_materialize_a_report(self) -> None:
        self.fixture.checkpoint.unlink()
        with (
            mock.patch.object(verifier, "LPIPS_SIZE_BYTES", self.fixture.lpips_size),
            mock.patch.object(verifier, "LPIPS_SHA256", self.fixture.lpips_sha),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = verifier.main(
                [
                    str(self.fixture.run),
                    "--track",
                    "lemniscate",
                    "--write-recovered-postflight",
                ]
            )
        self.assertEqual(result, 2)
        self.assertFalse((self.fixture.run / "recovered-postflight.json").exists())


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.dtype = "torch.float32"
        self.device = "cpu"

    def is_floating_point(self) -> bool:
        return True


def _fake_checkpoint(
    nonfinite: str | None = None, bad_shape: str | None = None
) -> dict[str, object]:
    pipeline: dict[str, _FakeTensor] = {}
    count = 5
    for name, tail in verifier.GAUSSIAN_TAIL_SHAPES.items():
        shape = (count, *tail)
        if name == bad_shape:
            shape = (count, 2)
        values = np.zeros(shape, dtype=np.float32)
        if name == nonfinite:
            values.flat[0] = math.nan
        pipeline[f"_model.gauss_params.{name}"] = _FakeTensor(values)
    return {
        "step": 29999,
        "pipeline": pipeline,
        "optimizers": {},
        "schedulers": {},
        "scalers": {},
    }


class TorchCheckpointInspectorTests(unittest.TestCase):
    def _fake_torch(self, checkpoint: dict[str, object]) -> types.ModuleType:
        module = types.ModuleType("torch")
        module.load = lambda *_args, **_kwargs: checkpoint  # type: ignore[attr-defined]
        module.isfinite = lambda tensor: np.isfinite(tensor.values)  # type: ignore[attr-defined]
        return module

    def test_validates_step_shape_count_cpu_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-000029999.ckpt"
            path.write_bytes(b"fake")
            with mock.patch.dict(
                "sys.modules", {"torch": self._fake_torch(_fake_checkpoint())}
            ):
                report = verifier.inspect_torch_checkpoint(path, 29999)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["gaussian_count"], 5)
        self.assertEqual(report["step"], 29999)
        self.assertTrue(
            all(item["finite"] for item in report["gaussian_tensors"].values())
        )

    def test_rejects_nonfinite_and_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-000029999.ckpt"
            path.write_bytes(b"fake")
            for payload, message in (
                (_fake_checkpoint(nonfinite="means"), "contains NaN"),
                (_fake_checkpoint(bad_shape="quats"), "shape changed"),
            ):
                with self.subTest(message=message):
                    with mock.patch.dict(
                        "sys.modules", {"torch": self._fake_torch(payload)}
                    ):
                        with self.assertRaisesRegex(
                            verifier.VerificationError, message
                        ):
                            verifier.inspect_torch_checkpoint(path, 29999)


if __name__ == "__main__":
    unittest.main()
