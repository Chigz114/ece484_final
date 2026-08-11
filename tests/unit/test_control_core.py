"""Regression tests for the recovered Quad Pilots control core."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from quadpilot.control.controller import LegacyVisionControlCore
from quadpilot.control.dynamics import body_acceleration_to_world, step_dynamics
from quadpilot.control.trajectory import generate_gate_transition
from quadpilot.datasets.generation import (
    CameraIntrinsics,
    PoseBounds,
    ReproDatasetGenerator,
    normalize_rgb,
    pose_to_camera_matrix,
)
from quadpilot.estimation.ekf import PoseEKF
from quadpilot.perception.renderer import PoseTransform
from quadpilot.simulation.evaluation import evaluate_ordered_gates
from quadpilot.simulation.runner import run_oracle_simulation, run_pose_simulation
from quadpilot.simulation.tracks import TRACKS


class DynamicsTests(unittest.TestCase):
    def test_body_x_at_ninety_degrees_maps_to_world_y(self) -> None:
        actual = body_acceleration_to_world(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        np.testing.assert_allclose(actual, [0.0, 1.0, 0.0], atol=1e-12)

    def test_zero_control_preserves_stationary_state(self) -> None:
        state = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.3])
        actual = step_dynamics(state, np.zeros(4), dt=0.02)
        np.testing.assert_allclose(actual, state, atol=1e-12)


class IncomingGateSideContractTests(unittest.TestCase):
    @staticmethod
    def _initial_path(track_name: str) -> np.ndarray:
        track = TRACKS[track_name]
        first_gate = track.ordered_gates()[0]
        start = np.asarray(track.initial_state[:3], dtype=np.float64)
        direction = np.asarray(first_gate.center) - start
        direction /= np.linalg.norm(direction)
        return generate_gate_transition(
            start,
            direction,
            np.asarray(first_gate.center),
            first_gate.normal,
            straight_dist=0.6,
            track=track.name,
        )

    @staticmethod
    def _transition_path(track_name: str, previous_index: int) -> np.ndarray:
        track = TRACKS[track_name]
        gates = track.ordered_gates()
        next_index = (previous_index + 1) % len(gates)
        return generate_gate_transition(
            np.asarray(gates[previous_index].center),
            gates[previous_index].normal,
            np.asarray(gates[next_index].center),
            gates[next_index].normal,
            straight_dist=0.8,
            is_lap_transition=previous_index == len(gates) - 1,
            track=track.name,
        )

    @staticmethod
    def _incoming_side(path: np.ndarray, center: np.ndarray, normal: np.ndarray) -> int:
        distances = np.dot(path - center, normal)
        nonzero_distances = distances[np.abs(distances) > 1e-9]
        if not len(nonzero_distances):
            raise AssertionError("ideal trajectory never approaches the gate plane")
        return int(np.sign(nonzero_distances[-1]))

    def test_schema_rejects_bad_incoming_side_lengths_and_values(self) -> None:
        track = TRACKS["uturn"]
        invalid_sides = (
            (-1, -1, -1),
            (-1, -1, 0, -1),
            (-1, -1, -1.0, -1),
            (-1, -1, True, -1),
            [-1, -1, -1, -1],
        )
        for sides in invalid_sides:
            with (
                self.subTest(sides=sides),
                self.assertRaisesRegex(ValueError, "incoming_gate_sides"),
            ):
                replace(track, incoming_gate_sides=sides)

    def test_explicit_sides_match_every_recovered_ideal_approach(self) -> None:
        for track_name, track in TRACKS.items():
            with self.subTest(track=track_name, approach="initial"):
                gates = track.ordered_gates()
                actual = self._incoming_side(
                    self._initial_path(track_name),
                    np.asarray(gates[0].center),
                    gates[0].normal,
                )
                self.assertEqual(actual, track.incoming_gate_sides[0])

            for previous_index in range(len(gates)):
                next_index = (previous_index + 1) % len(gates)
                with self.subTest(
                    track=track_name,
                    approach=f"{gates[previous_index].name}->{gates[next_index].name}",
                ):
                    actual = self._incoming_side(
                        self._transition_path(track_name, previous_index),
                        np.asarray(gates[next_index].center),
                        gates[next_index].normal,
                    )
                    self.assertEqual(actual, track.incoming_gate_sides[next_index])

        self.assertEqual(TRACKS["uturn"].incoming_gate_sides, (-1, -1, -1, -1))

    def test_controller_rearms_each_gate_without_false_incoming_misses(self) -> None:
        for track_name, track in TRACKS.items():
            with self.subTest(track=track_name):
                controller = LegacyVisionControlCore(track, total_laps=1)
                observed_events: list[str] = []
                for gate, incoming_side in zip(
                    track.ordered_gates(), track.incoming_gate_sides
                ):
                    center = np.asarray(gate.center)
                    before = center + 0.1 * incoming_side * gate.normal
                    after = center - 0.1 * incoming_side * gate.normal
                    approach = controller.step(
                        np.append(before, 0.0), velocity_estimate=np.zeros(3)
                    )
                    self.assertIsNone(
                        approach.event,
                        f"{track_name} {gate.name} was falsely handled as a miss",
                    )
                    crossing = controller.step(
                        np.append(after, 0.0), velocity_estimate=np.zeros(3)
                    )
                    if crossing.event is not None:
                        observed_events.append(crossing.event)
                    self.assertEqual(crossing.event, "pass")

                self.assertEqual(observed_events, ["pass"] * len(track.gate_order))
                self.assertTrue(controller.completed)
                self.assertEqual(
                    tuple(event[1] for event in controller.pass_events),
                    track.gate_order,
                )

    def test_strict_evaluator_accepts_ideal_lap_for_all_tracks(self) -> None:
        for track_name, track in TRACKS.items():
            with self.subTest(track=track_name):
                paths = [self._initial_path(track_name)]
                paths.extend(
                    self._transition_path(track_name, index)
                    for index in range(len(track.gate_order) - 1)
                )
                states = np.vstack([paths[0], *(path[1:] for path in paths[1:])])
                evaluation = evaluate_ordered_gates(
                    states, track, dt=0.05, laps=1, gate_radius=0.38
                )
                self.assertTrue(evaluation.completed)
                self.assertEqual(evaluation.successful_crossings, 4)
                self.assertEqual(
                    tuple(crossing.gate for crossing in evaluation.crossings),
                    track.gate_order,
                )


class OracleClosedLoopTests(unittest.TestCase):
    def test_all_tracks_complete_two_strict_laps(self) -> None:
        for track in TRACKS:
            with self.subTest(track=track):
                result = run_oracle_simulation(track, max_steps=1200)
                self.assertTrue(result.completed_by_controller)
                self.assertTrue(result.evaluation.completed)
                self.assertEqual(result.evaluation.successful_crossings, 8)
                self.assertEqual(result.evaluation.success_rate, 1.0)
                self.assertLess(result.evaluation.mean_gate_error_m or np.inf, 0.05)
                self.assertLess(result.steps, 600)
                self.assertEqual(
                    tuple(event[1] for event in result.controller_passes),
                    TRACKS[track].gate_order * 2,
                )

    def test_duration_uses_requested_time_step(self) -> None:
        result = run_oracle_simulation("circle", max_steps=2, dt=0.02)
        self.assertAlmostEqual(result.duration_s, result.steps * 0.02)

    def test_repeated_trajectory_cannot_exceed_one_hundred_percent(self) -> None:
        result = run_oracle_simulation("circle", max_steps=1200)
        repeated = np.vstack([result.states, result.states, result.states])
        evaluation = evaluate_ordered_gates(repeated, "circle", laps=2)
        self.assertLessEqual(evaluation.success_rate, 1.0)
        self.assertLessEqual(evaluation.successful_crossings, 8)


class EstimationTests(unittest.TestCase):
    def test_ekf_handles_yaw_wrap_and_keeps_covariance_psd(self) -> None:
        ekf = PoseEKF()
        ekf.initialize(np.array([0.0, 0.0, 0.0, np.pi - 0.01]))
        ekf.predict(np.array([0.0, 0.0, 0.0, 1.0]), dt=0.05)
        update = ekf.update(
            np.array([0.0, 0.0, 0.0, -np.pi + 0.04]),
            outlier_threshold=None,
        )
        self.assertTrue(update.accepted)
        self.assertLess(abs(update.state[6] + np.pi), 0.1)
        np.testing.assert_allclose(ekf.covariance, ekf.covariance.T, atol=1e-12)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(ekf.covariance))), -1e-12
        )

    def test_ekf_hysteresis_completes_seeded_npe_like_stress(self) -> None:
        for track in TRACKS:
            for seed in range(5):
                with self.subTest(track=track, seed=seed):
                    result = run_pose_simulation(
                        track,
                        max_steps=1200,
                        estimator="ekf",
                        position_noise_std=0.05,
                        yaw_noise_std=np.deg2rad(1.0),
                        crossing_hysteresis_m=0.05,
                        seed=seed,
                    )
                    self.assertTrue(result.completed_by_controller)
                    self.assertTrue(result.evaluation.completed)
                    self.assertLess(result.evaluation.mean_gate_error_m or np.inf, 0.08)


class DatasetGenerationTests(unittest.TestCase):
    def test_float_rgb_is_scaled_and_bad_shapes_are_rejected(self) -> None:
        image = np.linspace(0.0, 1.0, 4 * 3 * 3).reshape(3, 4, 3)
        normalized = normalize_rgb(image, width=4, height=3)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(int(normalized.min()), 0)
        self.assertEqual(int(normalized.max()), 255)
        with self.assertRaises(ValueError):
            normalize_rgb(np.zeros((3, 4, 4)), width=4, height=3)

    def test_failures_do_not_create_pose_image_misalignment(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.calls = 0

            def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
                self.calls += 1
                if self.calls in {2, 4}:
                    raise RuntimeError("intentional render failure")
                x_value = camera_to_world[0, 3]
                image = np.zeros((6, 8, 3), dtype=np.uint8)
                image[:, :, 0] = np.arange(8, dtype=np.uint8)
                image[:, :, 1] = int((x_value + 5.0) * 10.0) % 255
                image[:, :, 2] = 200
                return image

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            metadata = ReproDatasetGenerator(
                Renderer(),
                output,
                track="test",
                bounds=PoseBounds(x=(-1, 1), y=(-1, 1), z=(-1, 1)),
                intrinsics=CameraIntrinsics(width=8, height=6, fx=5, fy=5, cx=4, cy=3),
                seed=7,
            ).generate(5, maximum_failures=2)
            self.assertEqual(metadata["n_frames"], 5)
            self.assertEqual(metadata["render_failures"], 2)
            self.assertEqual(metadata["attempts"], 7)
            self.assertEqual(
                sorted(path.name for path in (output / "images").glob("*.png")),
                [f"frame_{index:05d}.png" for index in range(5)],
            )
            records = [
                json.loads(line)
                for line in (output / "samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["sample_id"] for record in records], list(range(5))
            )
            self.assertEqual([record["pose"] for record in records], metadata["poses"])


class RendererCoordinateTests(unittest.TestCase):
    def test_circle_gate_a_pose_matches_recovered_dataparser_transform(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        run_dir = json.loads(
            (repository_root / "configs" / "assets" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )["tracks"]["circle"]["runtime"]["renderer_run_dir"]
        transform_path = Path(run_dir) / "dataparser_transforms.json"
        if not transform_path.is_file():
            self.skipTest("recovered Circle dataparser transform is unavailable")
        pose = np.array([-0.3, -2.8, -0.4, 0.0, 0.0, -np.pi / 2.0])
        actual = PoseTransform(transform_path).to_nerfstudio_c2w(
            pose_to_camera_matrix(pose)
        )
        expected = np.array(
            [
                [-0.584951222, -0.028173886, 0.810578942, 0.391613956],
                [0.810578942, -0.055022836, 0.583038807, -0.237141754],
                [0.028173886, 0.998087525, 0.055022836, -0.048376037],
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
