"""CPU-only end-to-end checks for the Quad Pilots coordinate contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from quadpilot_repro.data_generation import (
    BASE_DATASET_BOUNDS,
    CameraIntrinsics,
    LAUNCH_CORRIDOR_BOUNDS,
    PoseBounds,
    ReproDatasetGenerator,
    pose_to_camera_matrix,
    resolve_dataset_bounds,
)
from quadpilot_repro.gate_sampling import project_world_point_to_image
from quadpilot_repro.npe import PoseNormalizer, decode_predictions
from quadpilot_repro.renderer import PoseTransform
from quadpilot_repro.simulation import run_oracle_simulation
from quadpilot_repro.tracks import get_track
from quadpilot_repro.visual_loop import TorchNPEPredictor, run_visual_closed_loop


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CIRCLE_TRANSFORM = (
    REPOSITORY_ROOT
    / "outputs"
    / "circle"
    / "splatfacto"
    / "2025-05-09_144210"
    / "dataparser_transforms.json"
)


class DatasetRegionContractTests(unittest.TestCase):
    def test_lemniscate_launch_corridor_covers_the_out_of_bounds_start(self) -> None:
        base = resolve_dataset_bounds("lemniscate", "base")
        launch = resolve_dataset_bounds("lemniscate", "launch-corridor")
        self.assertIs(base, BASE_DATASET_BOUNDS["lemniscate"])
        self.assertIs(launch, LAUNCH_CORRIDOR_BOUNDS["lemniscate"])

        initial = get_track("lemniscate").initial_state
        self.assertLess(initial[1], base.y[0])
        self.assertTrue(launch.x[0] <= initial[0] <= launch.x[1])
        self.assertTrue(launch.y[0] <= initial[1] <= launch.y[1])
        self.assertTrue(launch.z[0] <= initial[2] <= launch.z[1])
        self.assertTrue(launch.yaw[0] <= initial[6] <= launch.yaw[1])

    def test_launch_corridor_is_fail_closed_for_other_tracks_or_regions(self) -> None:
        with self.assertRaisesRegex(ValueError, "not defined"):
            resolve_dataset_bounds("circle", "launch-corridor")
        with self.assertRaisesRegex(ValueError, "not defined"):
            resolve_dataset_bounds("lemniscate", "unknown")


def circle_boundary_poses() -> list[np.ndarray]:
    bounds = BASE_DATASET_BOUNDS["circle"]
    return [
        np.array([x, y, z, 0.0, 0.0, yaw], dtype=np.float64)
        for x in bounds.x
        for y in bounds.y
        for z in bounds.z
        for yaw in (-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi)
    ]


def angle_error(first: float, second: float) -> float:
    return float(np.arctan2(np.sin(first - second), np.cos(first - second)))


class RendererTransformContractTests(unittest.TestCase):
    def setUp(self) -> None:
        if not CIRCLE_TRANSFORM.is_file():
            self.skipTest("recovered Circle dataparser transform is unavailable")
        self.transform = PoseTransform(CIRCLE_TRANSFORM)
        self.dataparser_h = np.eye(4, dtype=np.float64)
        self.dataparser_h[:3, :] = self.transform.dataparser_transform
        # Legacy original-world to COLMAP-world row remapping in PoseTransform.
        self.world_axis_h = np.eye(4, dtype=np.float64)
        self.world_axis_h[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        )

    def test_circle_boundary_pose_round_trip_through_dataparser(self) -> None:
        nerfstudio_to_opencv = np.diag([1.0, -1.0, -1.0])
        for pose in circle_boundary_poses():
            with self.subTest(pose=pose.tolist()):
                body_c2w = pose_to_camera_matrix(pose)
                nerfstudio_c2w = self.transform.to_nerfstudio_c2w(body_c2w)

                normalized_h = np.eye(4, dtype=np.float64)
                normalized_h[:3, :] = nerfstudio_c2w
                normalized_h[:3, 3] /= self.transform.scale
                converted = (
                    np.linalg.inv(self.world_axis_h)
                    @ np.linalg.inv(self.dataparser_h)
                    @ normalized_h
                )
                recovered_rotation = (
                    converted[:3, :3]
                    @ nerfstudio_to_opencv
                    @ self.transform.body_from_opencv.T
                )
                recovered_yaw = float(
                    np.arctan2(recovered_rotation[1, 0], recovered_rotation[0, 0])
                )
                np.testing.assert_allclose(
                    converted[:3, 3], pose[:3], rtol=0.0, atol=2e-6
                )
                np.testing.assert_allclose(
                    recovered_rotation, body_c2w[:3, :3], rtol=0.0, atol=2e-6
                )
                self.assertLess(abs(angle_error(recovered_yaw, pose[5])), 2e-6)

    def test_world_point_projection_agrees_before_and_after_dataparser(self) -> None:
        intrinsics = CameraIntrinsics()
        # Nerfstudio scales all scene/camera translations into normalized
        # scene units.  Projection ratios remain unchanged, while labels stay
        # in the original world meters tested below.
        expected_camera = (
            np.array([-0.2, 0.1, -1.2]) * self.transform.scale
        )
        for pose in circle_boundary_poses():
            with self.subTest(pose=pose.tolist()):
                body_c2w = pose_to_camera_matrix(pose)
                point_world = (
                    body_c2w[:3, :3] @ np.array([1.2, 0.2, 0.1]) + pose[:3]
                )
                direct = project_world_point_to_image(
                    pose, point_world, intrinsics
                )

                point_axis = self.world_axis_h @ np.r_[point_world, 1.0]
                point_normalized = self.dataparser_h @ point_axis
                point_normalized[:3] *= self.transform.scale
                nerfstudio_c2w = self.transform.to_nerfstudio_c2w(body_c2w)
                camera_coordinates = nerfstudio_c2w[:, :3].T @ (
                    point_normalized[:3] - nerfstudio_c2w[:, 3]
                )
                np.testing.assert_allclose(
                    camera_coordinates, expected_camera, rtol=0.0, atol=2e-6
                )
                u_px = (
                    intrinsics.fx
                    * camera_coordinates[0]
                    / -camera_coordinates[2]
                    + intrinsics.cx
                )
                v_px = (
                    intrinsics.fy
                    * -camera_coordinates[1]
                    / -camera_coordinates[2]
                    + intrinsics.cy
                )
                # RenderOnlySplatRenderer deliberately emits float32 c2w;
                # boundary-pose cancellation remains far below one pixel.
                self.assertAlmostEqual(direct.u_px, float(u_px), delta=1e-3)
                self.assertAlmostEqual(direct.v_px, float(v_px), delta=1e-3)


class NPELabelContractTests(unittest.TestCase):
    def test_normalizer_decodes_original_world_meters_and_wrapped_yaw(self) -> None:
        bounds = BASE_DATASET_BOUNDS["circle"]
        normalizer = PoseNormalizer(
            mean=tuple((low + high) / 2.0 for low, high in (bounds.x, bounds.y, bounds.z)),
            std=(1.3, 1.7, 0.4),
        )
        for pose in circle_boundary_poses():
            with self.subTest(pose=pose.tolist()):
                encoded = normalizer.encode_pose(pose)
                decoded = decode_predictions(encoded.unsqueeze(0), normalizer)[0]
                expected_normalized = (
                    pose[:3] - np.asarray(normalizer.mean)
                ) / np.asarray(normalizer.std)
                np.testing.assert_allclose(
                    encoded[:3].numpy(), expected_normalized, rtol=0.0, atol=2e-7
                )
                np.testing.assert_allclose(
                    decoded[:3].numpy(), pose[:3], rtol=0.0, atol=5e-7
                )
                self.assertLess(
                    abs(angle_error(float(decoded[3]), float(pose[5]))), 5e-7
                )


class ClosedLoopCoordinateContractTests(unittest.TestCase):
    def test_render_label_decode_controller_chain_stays_in_original_world(self) -> None:
        normalizer = PoseNormalizer(
            mean=(-2.35, -3.9, -0.2),
            std=(1.5, 1.8, 0.4),
        )

        class EncodedPoseModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = torch.zeros(5, dtype=torch.float32)

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.output.to(images.device).unsqueeze(0).expand(len(images), -1)

        model = EncodedPoseModel()

        class LabelOracleRenderer:
            def __init__(self) -> None:
                self.cameras: list[np.ndarray] = []

            def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
                self.cameras.append(camera_to_world.copy())
                yaw = float(
                    np.arctan2(camera_to_world[1, 0], camera_to_world[0, 0])
                )
                label = np.r_[camera_to_world[:3, 3], 0.0, 0.0, yaw]
                model.output = normalizer.encode_pose(label)
                image = np.zeros((8, 8, 3), dtype=np.uint8)
                image[:, :, 0] = np.arange(8, dtype=np.uint8)
                image[:, :, 1] = 100
                return image

        renderer = LabelOracleRenderer()
        predictor = TorchNPEPredictor(
            model,
            normalizer,
            lambda _image: torch.zeros((3, 8, 8), dtype=torch.float32),
            device="cpu",
        )
        actual = run_visual_closed_loop(
            "circle",
            renderer=renderer,
            predictor=predictor,
            estimator="raw",
            max_steps=3,
            crossing_hysteresis_m=0.05,
        )
        expected = run_oracle_simulation(
            "circle", max_steps=3, crossing_hysteresis_m=0.05
        )
        np.testing.assert_allclose(
            actual.observations[:, :3],
            actual.states[: len(actual.observations), :3],
            rtol=0.0,
            atol=5e-7,
        )
        yaw_error = np.arctan2(
            np.sin(
                actual.observations[:, 3]
                - actual.states[: len(actual.observations), 6]
            ),
            np.cos(
                actual.observations[:, 3]
                - actual.states[: len(actual.observations), 6]
            ),
        )
        np.testing.assert_allclose(yaw_error, 0.0, rtol=0.0, atol=5e-7)
        np.testing.assert_allclose(actual.controls, expected.controls, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(actual.states, expected.states, rtol=0.0, atol=1e-6)
        for index, camera in enumerate(renderer.cameras):
            np.testing.assert_allclose(
                camera[:3, 3], actual.states[index, :3], rtol=0.0, atol=1e-12
            )


class CoordinateValidationTests(unittest.TestCase):
    def test_invalid_intrinsics_pose_and_dataparser_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            CameraIntrinsics(width=640.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CameraIntrinsics(fx=float("nan"))
        with self.assertRaises(ValueError):
            CameraIntrinsics(cx=640.0)
        with self.assertRaises(ValueError):
            pose_to_camera_matrix(np.array([0.0, 0.0, np.nan, 0.0, 0.0, 0.0]))

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataparser_transforms.json"
            path.write_text(
                json.dumps(
                    {
                        "transform": np.c_[np.eye(3), np.zeros(3)].tolist(),
                        "scale": float("nan"),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                PoseTransform(path)
            path.write_text(
                json.dumps(
                    {
                        "transform": np.c_[np.diag([2.0, 1.0, 1.0]), np.zeros(3)].tolist(),
                        "scale": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "proper rotation"):
                PoseTransform(path)

    def test_dataset_metadata_declares_label_frame_axes_and_units(self) -> None:
        class Renderer:
            def render_rgb(self, _camera_to_world: np.ndarray) -> np.ndarray:
                image = np.zeros((6, 8, 3), dtype=np.uint8)
                image[:, :, 0] = np.arange(8, dtype=np.uint8)
                image[:, :, 1] = 100
                return image

        with TemporaryDirectory() as temporary:
            metadata = ReproDatasetGenerator(
                Renderer(),
                Path(temporary) / "dataset",
                track="contract-test",
                bounds=PoseBounds(x=(-1, 1), y=(-1, 1), z=(-1, 1)),
                intrinsics=CameraIntrinsics(
                    width=8, height=6, fx=5.0, fy=5.0, cx=4.0, cy=3.0
                ),
                seed=11,
            ).generate(1)
        self.assertEqual(metadata["pose_coordinate_frame"], "original_nerf_world")
        self.assertEqual(metadata["pose_units"], ["m", "m", "m", "rad", "rad", "rad"])
        self.assertIn("+X forward", metadata["body_axis_convention"])
        self.assertIn("optical +Z equals body +X", metadata["camera_axis_convention"])


if __name__ == "__main__":
    unittest.main()
