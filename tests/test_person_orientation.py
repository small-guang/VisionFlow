import math
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
from utils.quality import person_orientation as person_orientation_module
from utils.quality.person_orientation import (
    PersonOrientation,
    PersonOrientationResult,
    estimate_person_orientation,
    normalize_person_orientation,
)


def _pose(shoulders=(40.0, 10.0, 70.0, 10.0), hips=(40.0, 110.0, 70.0, 110.0), confidence=0.9):
    points = np.full((17, 3), np.nan, dtype=np.float64)
    points[5] = (shoulders[0], shoulders[1], confidence)
    points[6] = (shoulders[2], shoulders[3], confidence)
    points[11] = (hips[0], hips[1], confidence)
    points[12] = (hips[2], hips[3], confidence)
    return points


def _pose_for_ratio(ratio, confidence=0.9):
    # Vertical torso axis and a shoulder pair whose left-right span has the
    # requested sign and magnitude relative to the 100px torso scale.
    half_span = ratio * 50.0
    return _pose(
        shoulders=(50.0 + half_span, 10.0, 50.0 - half_span, 10.0),
        hips=(50.0 + half_span, 110.0, 50.0 - half_span, 110.0),
        confidence=confidence,
    )


class PersonOrientationTests(unittest.TestCase):
    def test_public_values_and_side_threshold(self):
        self.assertEqual({item.value for item in PersonOrientation}, {"front", "side", "back", "unknown"})
        self.assertEqual(normalize_person_orientation(PersonOrientation.FRONT), "front")
        self.assertEqual(normalize_person_orientation("  BaCk "), "back")
        self.assertEqual(normalize_person_orientation(None), "unknown")
        self.assertEqual(normalize_person_orientation("profile"), "unknown")

    def test_signed_ratios_use_exact_side_band_boundaries(self):
        expected = {
            0.8: PersonOrientation.FRONT,
            0.251: PersonOrientation.FRONT,
            0.25: PersonOrientation.SIDE,
            0.10: PersonOrientation.SIDE,
            0.0: PersonOrientation.SIDE,
            -0.10: PersonOrientation.SIDE,
            -0.25: PersonOrientation.SIDE,
            -0.251: PersonOrientation.BACK,
            -0.8: PersonOrientation.BACK,
        }
        for ratio, orientation in expected.items():
            with self.subTest(ratio=ratio):
                result = estimate_person_orientation(_pose_for_ratio(ratio), {"w": 100, "h": 120}, (120, 100, 3))
                self.assertEqual(result.orientation, orientation)
                self.assertAlmostEqual(result.signed_span_ratio, ratio, places=5)

    def test_classification_is_invariant_under_quarter_turns(self):
        base = _pose_for_ratio(0.8)
        results = []
        for quarter_turns in range(4):
            angle = quarter_turns * math.pi / 2.0
            rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
            rotated = base.copy()
            rotated[:, :2] = rotated[:, :2] @ rotation.T
            image_shape = (300, 300, 3)
            # Translate each rotated pose into the image while preserving all spans.
            rotated[:, 0] += 150.0
            rotated[:, 1] += 150.0
            results.append(estimate_person_orientation(rotated, {"w": 300, "h": 300}, image_shape))
        self.assertEqual([result.orientation for result in results], [PersonOrientation.FRONT] * 4)

    def test_one_complete_horizontal_pair_uses_one_reliable_torso_chain(self):
        points = np.full((17, 3), np.nan, dtype=np.float64)
        points[5] = (90.0, 10.0, 0.9)
        points[6] = (10.0, 10.0, 0.9)
        points[11] = (90.0, 110.0, 0.9)
        result = estimate_person_orientation(points, {"w": 120, "h": 120}, (120, 120, 3))
        self.assertEqual(result.orientation, PersonOrientation.FRONT)
        self.assertEqual(result.evidence_count, 1)
        self.assertIsNone(result.hip_span_ratio)

    def test_unequal_pair_confidence_fusion_can_cross_front_threshold(self):
        points = np.full((17, 3), np.nan, dtype=np.float64)
        # The left/right shoulder span is about +.44 with confidence .9;
        # the centered hip pair contributes zero with confidence .5.  The
        # weighted score is front, while an unweighted mean would be side.
        points[5] = (72.5, 10.0, 0.9)
        points[6] = (27.5, 10.0, 0.9)
        points[11] = (50.0, 110.0, 0.5)
        points[12] = (50.0, 110.0, 0.5)
        result = estimate_person_orientation(points, {"w": 100, "h": 120}, (120, 100, 3))
        self.assertEqual(result.orientation, PersonOrientation.FRONT)
        self.assertEqual(result.evidence_count, 2)
        self.assertAlmostEqual(result.shoulder_span_ratio, 0.439, delta=0.01)
        self.assertAlmostEqual(result.hip_span_ratio, 0.0, places=12)
        self.assertGreater(result.signed_span_ratio, algorithm_config.person.orientation.side_span_ratio)
        self.assertLess((result.shoulder_span_ratio + result.hip_span_ratio) / 2.0, algorithm_config.person.orientation.side_span_ratio)

    def test_classification_reads_configured_side_span(self):
        orientation_config = algorithm_config.person.orientation.model_copy(
            update={"side_span_ratio": 0.5}
        )
        alternate = SimpleNamespace(person=algorithm_config.person.model_copy(update={"orientation": orientation_config}))
        with patch.object(person_orientation_module, "algorithm_config", alternate):
            result = estimate_person_orientation(
                _pose_for_ratio(0.4), {"w": 100, "h": 120}, (120, 100, 3)
            )
        self.assertEqual(result.orientation, PersonOrientation.SIDE)

    def test_unexpected_axis_error_is_contained_as_internal_error(self):
        with patch(
            "utils.quality.person_orientation.estimate_person_torso_axis",
            side_effect=RuntimeError("unexpected"),
        ):
            result = estimate_person_orientation(
                _pose_for_ratio(0.8), {"w": 100, "h": 120}, (120, 100, 3)
            )
        self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
        self.assertEqual(result.reason, "internal_error")
        self.assertIsNone(result.signed_span_ratio)
        self.assertIsNone(result.shoulder_span_ratio)
        self.assertIsNone(result.hip_span_ratio)
        self.assertEqual(result.evidence_count, 0)

    def test_opposed_strong_pair_evidence_returns_pair_conflict(self):
        # Both chains differ by less than the shared 30° axis tolerance while
        # the shoulder and hip horizontal spans are opposite and just beyond
        # the side band.
        points = np.full((17, 3), np.nan, dtype=np.float64)
        points[5] = (75.0, 10.0, 0.9)
        points[6] = (125.0, 10.0, 0.9)
        points[11] = (126.0, 201.0, 0.9)
        points[12] = (74.0, 201.0, 0.9)
        result = estimate_person_orientation(points, {"w": 160, "h": 240}, (240, 160, 3))
        self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
        self.assertEqual(result.reason, "pair_conflict")
        self.assertEqual(result.evidence_count, 2)

    def test_shared_axis_conflict_is_propagated(self):
        points = _pose()
        points[6, 1] = 10.0
        points[12, 1] = 110.0
        points[5, 0] = 10.0
        points[11, 0] = 100.0
        result = estimate_person_orientation(points, {"w": 120, "h": 120}, (120, 120, 3))
        self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
        self.assertEqual(result.reason, "pose_axis_conflict")

    def test_low_confidence_and_out_of_image_evidence_fail_closed(self):
        low = _pose_for_ratio(0.8, confidence=0.49)
        outside = _pose_for_ratio(0.8)
        outside[[5, 6, 11, 12], 0] = -1.0
        for points in (low, outside):
            with self.subTest(points=points):
                result = estimate_person_orientation(points, {"w": 100, "h": 120}, (120, 100, 3))
                self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
                self.assertEqual(result.reason, "pose_axis_missing")
                self.assertIsNone(result.signed_span_ratio)
                self.assertIsNone(result.shoulder_span_ratio)
                self.assertIsNone(result.hip_span_ratio)
                self.assertEqual(result.evidence_count, 0)

    def test_exact_confidence_threshold_is_visible(self):
        result = estimate_person_orientation(
            _pose_for_ratio(0.8, confidence=0.5),
            {"w": 100, "h": 120},
            (120, 100, 3),
        )
        self.assertEqual(result.orientation, PersonOrientation.FRONT)
        self.assertEqual(result.evidence_count, 2)

    def test_no_horizontal_pair_has_exact_empty_reason(self):
        points = np.full((17, 3), np.nan, dtype=np.float64)
        points[5] = (90.0, 10.0, 0.9)
        points[11] = (90.0, 110.0, 0.9)
        result = estimate_person_orientation(points, {"w": 120, "h": 120}, (120, 120, 3))
        self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
        self.assertEqual(result.reason, "horizontal_pair_missing")
        self.assertIsNone(result.signed_span_ratio)
        self.assertIsNone(result.shoulder_span_ratio)
        self.assertIsNone(result.hip_span_ratio)
        self.assertEqual(result.evidence_count, 0)

    def test_invalid_inputs_and_nonfinite_values_fail_closed_without_raising(self):
        cases = (
            None,
            np.zeros((16, 3)),
            np.zeros((17, 2)),
            [[1, 2, 3]],
            np.full((17, 3), "not-a-number", dtype=object),
        )
        for points in cases:
            with self.subTest(points=points):
                result = estimate_person_orientation(points, {"w": 100, "h": 100}, (100, 100, 3))
                self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
                self.assertEqual(result.reason, "pose_axis_missing")
                self.assertIsNone(result.signed_span_ratio)
                self.assertIsNone(result.shoulder_span_ratio)
                self.assertIsNone(result.hip_span_ratio)
                self.assertEqual(result.evidence_count, 0)
        for shape in (None, (), (0, 100, 3), (100, 0, 3), (100, float("nan"), 3)):
            with self.subTest(shape=shape):
                result = estimate_person_orientation(_pose_for_ratio(0.8), {"w": 100, "h": 100}, shape)
                self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
                self.assertEqual(result.reason, "pose_axis_missing")
        for value in (float("nan"), float("inf"), -float("inf")):
            points = _pose_for_ratio(0.8)
            points[[5, 6, 11, 12], 0] = value
            result = estimate_person_orientation(points, {"w": 100, "h": 120}, (120, 100, 3))
            self.assertEqual(result.orientation, PersonOrientation.UNKNOWN)
        zero_scale = _pose_for_ratio(0.8)
        zero_scale[11, :2] = zero_scale[5, :2]
        zero_scale[12, :2] = zero_scale[6, :2]
        self.assertEqual(
            estimate_person_orientation(zero_scale, {"w": 100, "h": 120}, (120, 100, 3)).orientation,
            PersonOrientation.UNKNOWN,
        )

    def test_input_array_is_unchanged(self):
        points = _pose_for_ratio(0.8)
        before = points.copy()
        estimate_person_orientation(points, {"w": 100, "h": 120}, (120, 100, 3))
        np.testing.assert_array_equal(points, before)

    def test_result_is_frozen_and_label_is_canonical(self):
        result = PersonOrientationResult(PersonOrientation.FRONT, 0.8, 0.8, 0.8, 2)
        self.assertEqual(result.label, "front")
        with self.assertRaises(FrozenInstanceError):
            result.orientation = PersonOrientation.BACK


if __name__ == "__main__":
    unittest.main()
