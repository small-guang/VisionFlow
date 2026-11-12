import math
import unittest
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
import utils.keyframe.person_pose_geometry as person_pose_geometry
from utils.keyframe.person_pose_geometry import (
    PersonTorsoAxisResult,
    estimate_person_torso_axis,
)


def _keypoints():
    return np.full((17, 3), np.nan, dtype=np.float64)


class PersonTorsoAxisTests(unittest.TestCase):
    def test_torso_visibility_uses_selected_config(self):
        points = _keypoints()
        points[5] = (20.0, 10.0, 0.6)
        points[11] = (20.0, 50.0, 0.6)
        self.assertTrue(estimate_person_torso_axis(points, {"w": 100, "h": 100}).valid)

        geometry = algorithm_config.person.pose_geometry.model_copy(
            update={"torso_visible_threshold": 0.7}
        )
        person = algorithm_config.person.model_copy(update={"pose_geometry": geometry})
        selected = algorithm_config.model_copy(update={"person": person})
        with patch.object(person_pose_geometry, "algorithm_config", selected):
            result = estimate_person_torso_axis(points, {"w": 100, "h": 100})

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "pose_axis_missing")

    def test_two_reliable_chains_combine_to_down_axis_and_scale(self):
        points = _keypoints()
        points[5] = (20.0, 10.0, 0.9)
        points[11] = (20.0, 50.0, 0.9)
        points[6] = (40.0, 10.0, 0.8)
        points[12] = (40.0, 50.0, 0.8)

        result = estimate_person_torso_axis(points, {"w": 100, "h": 100})

        self.assertTrue(result.valid)
        self.assertEqual(result.chain_count, 2)
        self.assertAlmostEqual(result.down_axis[0], 0.0, places=7)
        self.assertAlmostEqual(result.down_axis[1], 1.0, places=7)
        self.assertAlmostEqual(result.torso_scale, 40.0, places=7)

    def test_one_reliable_chain_is_valid(self):
        points = _keypoints()
        points[5] = (20.0, 10.0, 0.9)
        points[11] = (35.0, 50.0, 0.9)

        result = estimate_person_torso_axis(points, {"w": 100, "h": 100})

        self.assertTrue(result.valid)
        self.assertEqual(result.chain_count, 1)
        self.assertAlmostEqual(result.torso_scale, 42.7200187266, places=7)

    def test_torso_scale_is_confidence_weighted_across_unequal_lengths(self):
        points = _keypoints()
        points[5] = (20.0, 10.0, 0.9)
        points[11] = (20.0, 50.0, 0.9)
        points[6] = (40.0, 10.0, 0.6)
        points[12] = (40.0, 70.0, 0.6)

        result = estimate_person_torso_axis(points, {"w": 100, "h": 100})

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.torso_scale, 48.0, places=12)

    def test_large_finite_chains_use_stable_weighted_scale(self):
        points = _keypoints()
        points[5] = (0.0, 0.0, 0.9)
        points[11] = (1.5e308, 0.0, 0.9)
        points[6] = (0.0, 0.0, 0.6)
        points[12] = (1.5e308, 0.0, 0.6)

        result = estimate_person_torso_axis(points, {"w": 1.0e308, "h": 1.0e308})

        self.assertTrue(result.valid)
        self.assertEqual(result.chain_count, 2)
        self.assertTrue(math.isclose(result.torso_scale, 1.5e308, rel_tol=1e-15))

    def test_public_result_valid_requires_finite_normalized_axis_and_metadata(self):
        valid = PersonTorsoAxisResult((1.0, 1.0e-8), 2.0, 1, "ok")
        self.assertTrue(valid.valid)
        invalid_results = (
            PersonTorsoAxisResult((float("nan"), 1.0), 2.0, 1, "ok"),
            PersonTorsoAxisResult((0.0, 0.0), 2.0, 1, "ok"),
            PersonTorsoAxisResult((2.0, 0.0), 2.0, 1, "ok"),
            PersonTorsoAxisResult((1.0, 0.0), 2.0, 0, "ok"),
            PersonTorsoAxisResult((1.0, 0.0), float("inf"), 1, "ok"),
            PersonTorsoAxisResult((1.0, 0.0), 2.0, 1, "other"),
        )
        for result in invalid_results:
            with self.subTest(result=result):
                self.assertFalse(result.valid)

    def test_short_visible_chains_return_short_reason(self):
        points = _keypoints()
        points[5] = (20.0, 10.0, 0.9)
        points[11] = (20.0, 15.0, 0.9)

        result = estimate_person_torso_axis(points, {"w": 100, "h": 100})

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "pose_axis_short")

    def test_opposed_chains_return_conflict_reason(self):
        points = _keypoints()
        points[5] = (20.0, 10.0, 0.9)
        points[11] = (20.0, 50.0, 0.9)
        points[6] = (40.0, 50.0, 0.8)
        points[12] = (40.0, 10.0, 0.8)

        result = estimate_person_torso_axis(points, {"w": 100, "h": 100})

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "pose_axis_conflict")

    def test_invalid_inputs_return_missing_reason_without_raising(self):
        cases = (
            None,
            np.zeros((16, 3), dtype=np.float64),
            np.zeros((17, 2), dtype=np.float64),
        )
        for keypoints in cases:
            with self.subTest(keypoints=keypoints):
                result = estimate_person_torso_axis(
                    keypoints, {"w": 100, "h": 100}
                )
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, "pose_axis_missing")

        points = _keypoints()
        points[5] = (20.0, 10.0, 0.9)
        points[11] = (20.0, 50.0, 0.9)
        for rect in ({}, {"w": 0, "h": 100}, {"w": float("nan"), "h": 100}):
            with self.subTest(rect=rect):
                result = estimate_person_torso_axis(points, rect)
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, "pose_axis_missing")


if __name__ == "__main__":
    unittest.main()
