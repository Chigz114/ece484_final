from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quadpilot.hardware.readiness import (
    check_hardware_readiness,
    estimate_similarity_transform,
    similarity_payload,
)


class HardwareReadinessTests(unittest.TestCase):
    def test_similarity_transform_recovers_known_mapping(self) -> None:
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 0.5]],
            dtype=np.float64,
        )
        angle = np.deg2rad(32.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        target = 1.07 * (source @ rotation.T) + np.asarray([2.0, -1.0, 0.3])
        transform, report = estimate_similarity_transform(
            source,
            target,
            source_frame="nerf_world",
            target_frame="vicon_world",
        )
        self.assertAlmostEqual(transform.scale, 1.07, places=12)
        np.testing.assert_allclose(transform.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(transform.translation, [2.0, -1.0, 0.3], atol=1e-12)
        self.assertLess(report["rmse_m"], 1e-12)

    def test_collinear_calibration_fails_closed(self) -> None:
        points = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        with self.assertRaisesRegex(ValueError, "collinear"):
            estimate_similarity_transform(
                points,
                points,
                source_frame="nerf_world",
                target_frame="vicon_world",
            )

    def test_template_is_blocked(self) -> None:
        template = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "hardware"
            / "preflight.template.json"
        )
        report = check_hardware_readiness(template)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["hardware_commands_executed"])
        self.assertIn("manual_safety_checks", report["blockers"])

    def test_complete_prop_off_fixture_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "hardware-evidence"
            evidence.mkdir()

            transform, residuals = estimate_similarity_transform(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 2, 3], [2, 2, 3], [1, 3, 3], [1, 2, 4]],
                source_frame="nerf_world",
                target_frame="vicon_world",
            )
            calibration = similarity_payload(
                transform,
                residuals,
                input_sha256="0" * 64,
                accepted_rmse_m=0.03,
            )
            intrinsics = {
                "fx": 500.0,
                "fy": 501.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": 640,
                "height": 480,
                "calibration_rms_px": 0.4,
            }
            extrinsics = {"matrix_body_from_camera": np.eye(4).tolist()}

            def write(name: str, payload: dict) -> tuple[str, str]:
                path = evidence / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                return str(path.relative_to(root)), sha

            evidence_specs = {}
            for key, name, payload in (
                ("vicon_from_nerf", "vicon_from_nerf.json", calibration),
                ("camera_intrinsics", "camera_intrinsics.json", intrinsics),
                ("body_from_camera", "body_from_camera.json", extrinsics),
            ):
                path, sha = write(name, payload)
                evidence_specs[key] = {"path": path, "sha256": sha}

            config = {
                "schema_version": 1,
                "stage": "bench_prop_off",
                "evidence": evidence_specs,
                "topics": {
                    "pose": "/cf/pose",
                    "odom": "/cf/odom",
                    "image": "/cf/camera/image_rect",
                    "setpoint": "/cf/setpoint",
                    "command": "/cf/cmd_acc",
                    "estop": "/cf/estop",
                },
                "frames": {
                    "npe": "nerf_world",
                    "pose": "vicon_world",
                    "command": "world_accel_yaw_rate",
                },
                "timeouts_s": {
                    "pose": 0.1,
                    "image": 0.2,
                    "setpoint": 0.2,
                    "command": 0.1,
                },
                "geofence_m": {"min": [-2, -2, -1], "max": [2, 2, 1]},
                "control_limits": {
                    "max_acceleration_mps2": 2.0,
                    "max_yaw_rate_rad_s": 1.0,
                },
                "manual_safety_checks": {
                    "propellers_removed": True,
                    "operator_present": True,
                    "physical_estop_tested": True,
                    "radio_kill_tested": True,
                    "vicon_occlusion_tested": True,
                    "command_sign_tested": True,
                    "geofence_tested": True,
                    "no_people_in_test_volume": True,
                    "battery_secured": True,
                },
            }
            config_path = root / "hardware-preflight.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report = check_hardware_readiness(config_path)
            self.assertEqual(report["status"], "READY_FOR_PROP_OFF_BENCH")
            self.assertEqual(report["blockers"], [])


if __name__ == "__main__":
    unittest.main()
