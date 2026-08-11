"""CPU-only regression tests for the real visual closed-loop harness."""

from __future__ import annotations

import json
import sys
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from quadpilot.cli import simulate_closed_loop as closed_loop_cli
from quadpilot.estimation.ekf import PoseEKF
from quadpilot.perception.npe import PoseNormalizer
from quadpilot.simulation.runner import run_oracle_simulation
from quadpilot.simulation.tracks import TRACKS
from quadpilot.simulation.visual_loop import (
    TorchNPEPredictor,
    oracle_pose_observation,
    run_visual_closed_loop,
    save_visual_loop_result,
    true_state_to_camera_matrix,
)


class OracleEquivalenceTests(unittest.TestCase):
    def test_oracle_provider_matches_strict_baseline(self) -> None:
        for track in TRACKS:
            with self.subTest(track=track):
                expected = run_oracle_simulation(
                    track,
                    max_steps=1200,
                    crossing_hysteresis_m=0.05,
                )
                actual = run_visual_closed_loop(
                    track,
                    observation_provider=oracle_pose_observation,
                    estimator="raw",
                    max_steps=1200,
                    crossing_hysteresis_m=0.05,
                )
                self.assertTrue(actual.succeeded)
                self.assertEqual(actual.evaluation, expected.evaluation)
                np.testing.assert_allclose(actual.states, expected.states, atol=0.0)
                np.testing.assert_allclose(actual.controls, expected.controls, atol=0.0)

    def test_truth_state_is_converted_to_level_camera_pose(self) -> None:
        state = np.array([1.2, -3.4, 0.5, 8.0, 9.0, 10.0, np.pi / 2])
        matrix = true_state_to_camera_matrix(state)
        np.testing.assert_allclose(matrix[:3, 3], state[:3], atol=0.0)
        expected_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(matrix[:3, :3], expected_rotation, atol=1e-12)

    def test_shared_renderer_starts_raw_and_ekf_from_identical_sensor_sample(
        self,
    ) -> None:
        class DeterministicRenderer:
            def __init__(self) -> None:
                self.matrices: list[np.ndarray] = []

            def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
                self.matrices.append(camera_to_world.copy())
                return np.full((4, 5, 3), 127, dtype=np.uint8)

        renderer = DeterministicRenderer()

        def predictor(_rgb: np.ndarray) -> np.ndarray:
            matrix = renderer.matrices[-1]
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            return np.r_[matrix[:3, 3], yaw]

        raw = run_visual_closed_loop(
            "circle",
            renderer=renderer,
            predictor=predictor,
            estimator="raw",
            max_steps=2,
        )
        raw_render_count = len(renderer.matrices)
        ekf = run_visual_closed_loop(
            "circle",
            renderer=renderer,
            predictor=predictor,
            estimator="ekf",
            ekf_outlier_threshold=None,
            max_steps=2,
        )
        self.assertGreater(raw_render_count, 0)
        np.testing.assert_array_equal(raw.states[0], ekf.states[0])
        np.testing.assert_array_equal(raw.camera_to_world[0], ekf.camera_to_world[0])
        np.testing.assert_array_equal(raw.observations[0], ekf.observations[0])
        np.testing.assert_array_equal(
            renderer.matrices[0], renderer.matrices[raw_render_count]
        )


class ClosedLoopCliContractTests(unittest.TestCase):
    def test_formal_defaults_are_both_1200_steps_and_two_laps(self) -> None:
        with patch.object(sys, "argv", ["reproduce_npe_closed_loop.py"]):
            args = closed_loop_cli.parse_args()
        self.assertEqual(args.track, "circle")
        self.assertEqual(args.estimator, "both")
        self.assertEqual(args.max_steps, 1200)
        self.assertEqual(args.laps, 2)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.dt, 0.05)
        self.assertEqual(args.snapshot_every, 0)
        self.assertIsNone(args.cuda_home)

    def test_default_circle_renderer_assets_follow_the_recovered_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            fake_npe = Path(temporary) / "best_npe.pth"
            fake_npe.write_bytes(b"existence-only test fixture")
            args = Namespace(
                track="circle",
                renderer_checkpoint=None,
                dataparser_transform=None,
                npe_checkpoint=fake_npe,
            )
            renderer, transform, npe = closed_loop_cli._resolve_assets(args)
        run_dir = Path(
            "/home/chi/UAV/quadpilot-data/gsplat_outputs/circle/"
            "splatfacto/2025-05-09_144210"
        )
        self.assertEqual(
            renderer,
            (run_dir / "nerfstudio_models" / "step-000029999.ckpt").resolve(),
        )
        self.assertEqual(transform, (run_dir / "dataparser_transforms.json").resolve())
        self.assertEqual(npe, fake_npe.resolve())


class InjectionAndSafetyTests(unittest.TestCase):
    def test_fake_renderer_and_predictor_receive_truth_camera_image(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.matrices: list[np.ndarray] = []

            def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
                self.matrices.append(camera_to_world.copy())
                image = np.zeros((4, 5, 3), dtype=np.uint8)
                image[:, :, 1] = 100
                return image

        renderer = Renderer()

        def predictor(_rgb: np.ndarray) -> np.ndarray:
            matrix = renderer.matrices[-1]
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            return np.r_[matrix[:3, 3], yaw]

        result = run_visual_closed_loop(
            "circle",
            renderer=renderer,
            predictor=predictor,
            max_steps=2,
        )
        self.assertEqual(len(renderer.matrices), 2)
        self.assertEqual(result.camera_to_world.shape, (2, 4, 4))
        np.testing.assert_allclose(
            renderer.matrices[0], true_state_to_camera_matrix(result.states[0])
        )
        np.testing.assert_allclose(
            result.observations[0], result.states[0, [0, 1, 2, 6]]
        )

    def test_degenerate_sincos_prediction_stops_before_control(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.zeros((len(images), 5), dtype=images.dtype)

        def transform(_image: object) -> torch.Tensor:
            return torch.zeros((3, 8, 8), dtype=torch.float32)

        class Renderer:
            def render_rgb(self, _camera_to_world: np.ndarray) -> np.ndarray:
                image = np.zeros((8, 8, 3), dtype=np.uint8)
                image[:, :, 0] = np.arange(8, dtype=np.uint8)
                return image

        predictor = TorchNPEPredictor(
            ZeroModel(),
            PoseNormalizer((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            transform,
            device="cpu",
        )
        result = run_visual_closed_loop(
            "circle", renderer=Renderer(), predictor=predictor, max_steps=5
        )
        self.assertEqual(result.termination_reason, "observation_failure")
        self.assertIn("degenerate", result.failure_reason or "")
        self.assertEqual(result.controls.shape, (0, 4))
        self.assertEqual(result.states.shape, (1, 7))
        self.assertIsNone(result.evaluation)

    def test_second_observation_failure_does_not_apply_a_second_control(self) -> None:
        calls = 0

        def fail_after_one(state: np.ndarray, _index: int) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second frame failed")
            return state[[0, 1, 2, 6]].copy()

        result = run_visual_closed_loop(
            "circle", observation_provider=fail_after_one, max_steps=5
        )
        self.assertEqual(result.termination_reason, "observation_failure")
        self.assertEqual(len(result.controls), 1)
        self.assertEqual(len(result.states), 2)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(len(result.estimated_states), 1)

    def test_raw_velocity_never_uses_truth_velocity(self) -> None:
        moving_track = replace(
            TRACKS["circle"],
            name="circle_raw_no_truth_velocity",
            initial_state=(-0.4, -0.5, -0.3, 4.0, -3.0, 2.0, -np.pi / 2),
        )
        fixed_pose = np.asarray(moving_track.initial_state)[[0, 1, 2, 6]]

        def fixed_observation(_state: np.ndarray, _index: int) -> np.ndarray:
            return fixed_pose.copy()

        result = run_visual_closed_loop(
            moving_track,
            observation_provider=fixed_observation,
            estimator="raw",
            max_steps=3,
        )
        np.testing.assert_allclose(result.estimated_states[:, 3:6], 0.0, atol=0.0)
        self.assertFalse(np.allclose(result.states[0, 3:6], 0.0))

    def test_ekf_predicts_with_previous_applied_control(self) -> None:
        class RecordingEKF(PoseEKF):
            def __init__(self) -> None:
                super().__init__()
                self.prediction_controls: list[np.ndarray] = []

            def predict(self, control: np.ndarray, dt: float) -> np.ndarray:
                self.prediction_controls.append(np.asarray(control).copy())
                return super().predict(control, dt)

        ekf = RecordingEKF()
        result = run_visual_closed_loop(
            "circle",
            observation_provider=oracle_pose_observation,
            estimator="ekf",
            ekf=ekf,
            ekf_outlier_threshold=None,
            max_steps=4,
        )
        self.assertEqual(len(ekf.prediction_controls), 3)
        for index, predicted_with in enumerate(ekf.prediction_controls):
            np.testing.assert_allclose(predicted_with, result.controls[index])

    def test_nonfinite_ekf_diagnostic_fails_before_control_and_remains_saveable(
        self,
    ) -> None:
        class InvalidDiagnosticEKF(PoseEKF):
            def update(
                self,
                observation: np.ndarray,
                *,
                outlier_threshold: float | None = 4.0,
            ) -> object:
                del outlier_threshold
                state = np.r_[observation[:3], np.zeros(3), observation[3]]
                return type(
                    "Update",
                    (),
                    {
                        "state": state,
                        "accepted": False,
                        "mahalanobis_distance": float("inf"),
                    },
                )()

        result = run_visual_closed_loop(
            "circle",
            observation_provider=oracle_pose_observation,
            estimator="ekf",
            ekf=InvalidDiagnosticEKF(),
            max_steps=3,
        )
        self.assertEqual(result.termination_reason, "estimation_failure")
        self.assertEqual(len(result.states), 1)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(len(result.estimated_states), 0)
        self.assertEqual(len(result.controls), 0)
        with TemporaryDirectory() as temporary:
            json_path, _ = save_visual_loop_result(result, temporary)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("Mahalanobis", payload["metrics"]["failure_reason"])

    def test_json_and_npz_preserve_core_arrays_and_metrics(self) -> None:
        result = run_visual_closed_loop(
            "circle",
            observation_provider=oracle_pose_observation,
            estimator="raw",
            max_steps=3,
        )
        with TemporaryDirectory() as temporary:
            json_path, npz_path = save_visual_loop_result(
                result, Path(temporary), metadata={"test": True}
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            arrays = np.load(npz_path)
            self.assertTrue(payload["metadata"]["test"])
            self.assertEqual(payload["metrics"]["steps"], result.steps)
            np.testing.assert_allclose(arrays["states"], result.states)
            np.testing.assert_allclose(arrays["observations"], result.observations)
            np.testing.assert_allclose(arrays["controls"], result.controls)
            np.testing.assert_array_equal(arrays["state_step_indices"], [0, 1, 2, 3])
            np.testing.assert_allclose(arrays["state_times_s"], [0.0, 0.05, 0.1, 0.15])
            np.testing.assert_array_equal(arrays["control_step_indices"], [0, 1, 2])
            self.assertEqual(
                payload["sample_alignment"]["contract"],
                "z[k] and xhat[k] observe s[k]; validated u[k] advances s[k] to s[k+1]",
            )


if __name__ == "__main__":
    unittest.main()
