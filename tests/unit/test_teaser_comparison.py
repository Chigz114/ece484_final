from __future__ import annotations

import unittest

import numpy as np

from quadpilot.verification.teaser import (
    compare_teaser_metrics,
    legacy_dyn_poses,
    position_jitter_cm,
)


class TeaserComparisonTests(unittest.TestCase):
    def _fixture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        controls = np.zeros((5, 4), dtype=np.float64)
        controls[:, 0] = 0.2
        states = [np.asarray([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)]
        from quadpilot.control.dynamics import step_dynamics

        for control in controls:
            states.append(step_dynamics(states[-1], control, dt=0.05))
        state_array = np.asarray(states)
        observations = state_array[:, [0, 1, 2, 6]].copy()
        observations[:, 0] += 0.01
        estimates = state_array.copy()
        estimates[:, 0] += 0.005
        return state_array, observations, estimates, controls

    def test_comparison_uses_legacy_gate_clipped_window(self) -> None:
        states, observations, estimates, controls = self._fixture()
        report = compare_teaser_metrics(
            states=states,
            observations=observations,
            estimated_states=estimates,
            controls=controls,
            controller_pass_steps=[1, 3, 5],
            dt=0.05,
            dyn_seed=42,
        )
        self.assertEqual(report["metric_window"]["samples"], 4)
        self.assertAlmostEqual(report["reproduced"]["NPE"]["mean_cm"], 1.0)
        self.assertAlmostEqual(report["reproduced"]["EKF"]["mean_cm"], 0.5)
        self.assertIn("DYN", report["delta_percent_vs_teaser"])

    def test_legacy_dyn_is_deterministic_and_consumes_yaw_noise(self) -> None:
        states, _, _, controls = self._fixture()
        first = legacy_dyn_poses(states, controls, dt=0.05, seed=42)
        second = legacy_dyn_poses(states, controls, dt=0.05, seed=42)
        different = legacy_dyn_poses(states, controls, dt=0.05, seed=43)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(first.shape, (6, 4))

    def test_invalid_pass_window_fails_closed(self) -> None:
        states, observations, estimates, controls = self._fixture()
        with self.assertRaisesRegex(ValueError, "ordered"):
            compare_teaser_metrics(
                states=states,
                observations=observations,
                estimated_states=estimates,
                controls=controls,
                controller_pass_steps=[5, 1],
                dt=0.05,
                dyn_seed=42,
            )

    def test_jitter_matches_legacy_definition(self) -> None:
        poses = np.asarray([[0, 0, 0], [0.01, 0, 0], [0.04, 0, 0]], dtype=np.float64)
        self.assertAlmostEqual(position_jitter_cm(poses), 1.0)


if __name__ == "__main__":
    unittest.main()
