import math
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from utils.quality.face_orientation import (
    FACE_ORIENTATION_VALUES,
    FaceOrientation,
    FaceOrientationResult,
    classify_face_orientation,
    estimate_face_orientation,
    normalize_face_orientation,
)


class FaceOrientationClassifierTests(unittest.TestCase):
    def test_pitch_and_roll_do_not_change_front_side_label(self):
        self.assertIs(
            classify_face_orientation(80, 0, 80), FaceOrientation.FRONT
        )

    def test_yaw_threshold_is_twenty_degrees_exclusive_for_front(self):
        self.assertIs(classify_face_orientation(0, 19.99, 0), FaceOrientation.FRONT)
        self.assertIs(classify_face_orientation(0, 20.0, 0), FaceOrientation.SIDE)
        self.assertIs(classify_face_orientation(0, -20.0, 0), FaceOrientation.SIDE)
        self.assertIs(classify_face_orientation(0, 89, 0), FaceOrientation.SIDE)

    def test_side_yaw_threshold_rejects_invalid_values_and_allows_zero(self):
        for threshold in (-1, math.nan, math.inf, "bad"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    classify_face_orientation(0, 0, 0, threshold)
        self.assertIs(classify_face_orientation(0, 0, 0, 0.0), FaceOrientation.SIDE)

    def test_missing_non_numeric_and_nonfinite_angles_are_unknown(self):
        bad = (None, "bad", math.nan, math.inf, -math.inf)
        for value in bad:
            with self.subTest(value=value):
                self.assertIs(
                    classify_face_orientation(value, 0, 0), FaceOrientation.UNKNOWN
                )
                self.assertIs(
                    classify_face_orientation(0, value, 0), FaceOrientation.UNKNOWN
                )
                self.assertIs(
                    classify_face_orientation(0, 0, value), FaceOrientation.UNKNOWN
                )

    def test_normalize_face_orientation(self):
        self.assertEqual(normalize_face_orientation(" FRONT "), "front")
        self.assertEqual(normalize_face_orientation("ſIDE"), "side")
        self.assertEqual(normalize_face_orientation(FaceOrientation.SIDE), "side")
        self.assertEqual(normalize_face_orientation("no_face"), "no_face")
        self.assertEqual(normalize_face_orientation("profile"), "unknown")
        self.assertEqual(normalize_face_orientation(None), "unknown")
        self.assertEqual(
            FACE_ORIENTATION_VALUES,
            frozenset({"front", "side", "no_face", "unknown"}),
        )

    def test_result_is_frozen_and_label_is_string(self):
        result = FaceOrientationResult(FaceOrientation.FRONT, 1.0, 2.0, 3.0)
        self.assertEqual(result.label, "front")
        with self.assertRaises(FrozenInstanceError):
            result.pitch = 5.0

    @patch("utils.quality.face_orientation.calculate_pose_from_landmarks")
    def test_estimate_handles_bad_pose_and_calculation_errors(self, calculate):
        for pose in (None, (), (1, 2), (1, "bad", 3), (1, math.nan, 3)):
            calculate.return_value = pose
            result = estimate_face_orientation([], (100, 100))
            self.assertEqual(result, FaceOrientationResult(FaceOrientation.UNKNOWN))

        calculate.side_effect = RuntimeError("bad landmarks")
        result = estimate_face_orientation([], (100, 100))
        self.assertEqual(result, FaceOrientationResult(FaceOrientation.UNKNOWN))

    @patch("utils.quality.face_orientation.calculate_pose_from_landmarks", return_value=(1, 20, 3))
    def test_estimate_returns_pose_angles_and_classification(self, calculate):
        result = estimate_face_orientation([], (100, 100))
        self.assertEqual(result.orientation, FaceOrientation.SIDE)
        self.assertEqual((result.pitch, result.yaw, result.roll), (1.0, 20.0, 3.0))


if __name__ == "__main__":
    unittest.main()
