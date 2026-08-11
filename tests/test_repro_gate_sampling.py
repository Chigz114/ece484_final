"""Tests for transparent gate-focused NPE pose sampling."""

from __future__ import annotations

import unittest

import numpy as np

from quadpilot_repro.data_generation import BASE_DATASET_BOUNDS, CameraIntrinsics
from quadpilot_repro.gate_sampling import (
    GateFocusConfig,
    GateFocusedPoseSampler,
    project_world_point_to_image,
)
from quadpilot_repro.tracks import get_track


class GateFocusedSamplingTests(unittest.TestCase):
    def test_samples_stay_in_bounds_and_on_incoming_side(self) -> None:
        track = get_track("circle")
        bounds = BASE_DATASET_BOUNDS["circle"]
        sampler = GateFocusedPoseSampler(track, bounds)
        rng = np.random.default_rng(4242)
        seen: set[str] = set()
        for _ in range(1000):
            sample = sampler.sample(rng)
            pose = sample.pose
            annotations = dict(sample.annotations or {})
            seen.add(str(annotations["focus_gate"]))
            self.assertTrue(bounds.x[0] <= pose[0] <= bounds.x[1])
            self.assertTrue(bounds.y[0] <= pose[1] <= bounds.y[1])
            self.assertTrue(bounds.z[0] <= pose[2] <= bounds.z[1])
            gate = track.gates[str(annotations["focus_gate"])]
            signed_plane_distance = float(
                np.dot(pose[:3] - np.asarray(gate.center), gate.normal)
            )
            self.assertLess(signed_plane_distance, 0.0)
            expected_yaw = float(np.arctan2(gate.normal[1], gate.normal[0]))
            yaw_error = float(
                (pose[5] - expected_yaw + np.pi) % (2 * np.pi) - np.pi
            )
            self.assertLessEqual(abs(np.rad2deg(yaw_error)), 25.0 + 1e-10)
            projection = project_world_point_to_image(
                pose, np.asarray(gate.center)
            )
            self.assertTrue(projection.is_visible(CameraIntrinsics(), margin_px=32.0))
            self.assertAlmostEqual(
                projection.u_px, float(annotations["gate_center_u_px"])
            )
            self.assertAlmostEqual(
                projection.v_px, float(annotations["gate_center_v_px"])
            )
        self.assertEqual(seen, set(track.gate_order))

    def test_default_sampler_is_margin_visible_and_gate_balanced_on_all_tracks(self) -> None:
        intrinsics = CameraIntrinsics()
        for track_name in ("circle", "uturn", "lemniscate"):
            with self.subTest(track=track_name):
                track = get_track(track_name)
                sampler = GateFocusedPoseSampler(
                    track, BASE_DATASET_BOUNDS[track_name], intrinsics=intrinsics
                )
                rng = np.random.default_rng(20260810)
                counts = {name: 0 for name in track.gate_order}
                for _ in range(4000):
                    sample = sampler.sample(rng)
                    annotations = dict(sample.annotations or {})
                    gate_name = str(annotations["focus_gate"])
                    counts[gate_name] += 1
                    projection = project_world_point_to_image(
                        sample.pose, np.asarray(track.gates[gate_name].center), intrinsics
                    )
                    self.assertTrue(
                        projection.is_visible(intrinsics, margin_px=32.0)
                    )
                # The gate is selected before any pose/FOV rejection, so each
                # gate remains close to its requested 25% share.
                for count in counts.values():
                    self.assertGreater(count, 850)
                    self.assertLess(count, 1150)

    def test_projection_matches_renderer_body_camera_axis_contract(self) -> None:
        intrinsics = CameraIntrinsics()
        pose = np.zeros(6, dtype=np.float64)
        center = project_world_point_to_image(
            pose, np.array([1.0, 0.0, 0.0]), intrinsics
        )
        self.assertAlmostEqual(center.u_px, intrinsics.cx)
        self.assertAlmostEqual(center.v_px, intrinsics.cy)
        self.assertAlmostEqual(center.depth_m, 1.0)

        right_and_above = project_world_point_to_image(
            pose, np.array([1.0, -0.1, 0.1]), intrinsics
        )
        self.assertAlmostEqual(
            right_and_above.u_px, intrinsics.cx + 0.1 * intrinsics.fx
        )
        self.assertAlmostEqual(
            right_and_above.v_px, intrinsics.cy - 0.1 * intrinsics.fy
        )

    def test_original_near_lateral_extreme_is_rejected_fail_closed(self) -> None:
        track = get_track("circle")
        gate = track.gates["Gate D"]
        normal = gate.normal
        tangent = np.array([-normal[1], normal[0], 0.0])
        position = (
            np.asarray(gate.center)
            - 0.35 * normal
            + 0.55 * tangent
            + np.array([0.0, 0.0, 0.32])
        )
        gate_yaw = np.arctan2(normal[1], normal[0])
        pose = np.r_[position, 0.0, 0.0, gate_yaw + np.deg2rad(25.0)]
        projection = project_world_point_to_image(pose, np.asarray(gate.center))
        self.assertFalse(projection.is_visible(CameraIntrinsics(), margin_px=32.0))

        class ExtremeRNG:
            values = iter((0.35, 0.55, 0.32, np.deg2rad(25.0)))

            def integers(self, _low: int, _high: int) -> int:
                return 3  # Gate D keeps this extreme position inside pose bounds.

            def uniform(self, _low: float, _high: float) -> float:
                return float(next(self.values))

        sampler = GateFocusedPoseSampler(
            track,
            BASE_DATASET_BOUNDS["circle"],
            GateFocusConfig(maximum_rejections=1),
        )
        with self.assertRaisesRegex(RuntimeError, "margin-safe camera FOV"):
            sampler.sample(ExtremeRNG())  # type: ignore[arg-type]

    def test_seed_is_reproducible(self) -> None:
        sampler = GateFocusedPoseSampler(
            "lemniscate", BASE_DATASET_BOUNDS["lemniscate"]
        )
        first = sampler.sample(np.random.default_rng(9))
        second = sampler.sample(np.random.default_rng(9))
        np.testing.assert_array_equal(first.pose, second.pose)
        self.assertEqual(dict(first.annotations or {}), dict(second.annotations or {}))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GateFocusConfig(min_approach_distance_m=2.0, max_approach_distance_m=1.0)
        with self.assertRaises(ValueError):
            GateFocusConfig(image_margin_px=-1.0)
        with self.assertRaises(ValueError):
            GateFocusConfig(max_lateral_offset_m=float("nan"))
        with self.assertRaises(ValueError):
            GateFocusConfig(min_approach_distance_m=float("inf"))
        with self.assertRaises(ValueError):
            GateFocusConfig(max_vertical_offset_m=True)
        with self.assertRaises(ValueError):
            GateFocusConfig(max_yaw_jitter_deg="25")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GateFocusConfig(maximum_rejections=True)
        with self.assertRaises(ValueError):
            GateFocusConfig(maximum_rejections=100.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GateFocusedPoseSampler(
                "circle",
                BASE_DATASET_BOUNDS["circle"],
                GateFocusConfig(image_margin_px=240.0),
            )


if __name__ == "__main__":
    unittest.main()
