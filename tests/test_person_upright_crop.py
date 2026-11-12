import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.algorithm_config import algorithm_config
import utils.keyframe.person_crop as person_crop
from utils.keyframe.person_crop import (
    PersonRotationDecision,
    build_person_crop,
    estimate_person_upright_rotation,
    project_mask_to_person_crop,
    project_rect_to_person_crop,
    transform_keypoints_to_person_crop,
)


def _horizontal_person_keypoints() -> np.ndarray:
    points = np.zeros((17, 3), dtype=np.float64)
    points[0] = (22.0, 45.0, 0.95)
    points[5] = (25.0, 44.0, 0.9)
    points[11] = (65.0, 44.0, 0.9)
    points[6] = (25.0, 56.0, 0.9)
    points[12] = (65.0, 56.0, 0.9)
    return points


def _force_rotation(angle: float):
    return patch(
        "utils.keyframe.person_crop.estimate_person_upright_rotation",
        return_value=PersonRotationDecision(float(angle), True, "applied"),
    )


def _force_clockwise_rotation():
    return _force_rotation(-90.0)


class PersonUprightAngleTests(unittest.TestCase):
    def test_dead_zone_uses_selected_config(self):
        points = _horizontal_person_keypoints()
        crop = algorithm_config.person.crop.model_copy(
            update={"upright_dead_zone_degrees": 100.0}
        )
        person = algorithm_config.person.model_copy(update={"crop": crop})
        selected = algorithm_config.model_copy(update={"person": person})
        with patch.object(person_crop, "algorithm_config", selected):
            decision = estimate_person_upright_rotation(
                points, {"x": 0, "y": 0, "w": 100, "h": 100}, enabled=True
            )

        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "upright_dead_zone")

    RECT = {"x": 0, "y": 0, "w": 100, "h": 100}

    @staticmethod
    def _coco17_keypoints(
        *,
        left_axis: tuple[float, float] = (0.0, 30.0),
        right_axis: tuple[float, float] = (0.0, 30.0),
        left_score: float = 0.9,
        right_score: float = 0.9,
    ) -> np.ndarray:
        points = np.zeros((17, 3), dtype=np.float64)
        left_shoulder = np.array([30.0, 30.0])
        right_shoulder = np.array([70.0, 30.0])
        points[5] = (*left_shoulder, left_score)
        points[11] = (*(left_shoulder + np.asarray(left_axis)), left_score)
        points[6] = (*right_shoulder, right_score)
        points[12] = (*(right_shoulder + np.asarray(right_axis)), right_score)
        return points

    @staticmethod
    def _axis_for_angle(angle_degrees: float, length: float = 30.0):
        radians = math.radians(angle_degrees)
        return (-math.sin(radians) * length, math.cos(radians) * length)

    def _decision(self, points: np.ndarray, *, enabled: bool = True):
        return estimate_person_upright_rotation(
            points,
            self.RECT,
            enabled=enabled,
        )

    def test_downward_axis_is_in_upright_dead_zone(self):
        decision = self._decision(self._coco17_keypoints())

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "upright_dead_zone")

    def test_rightward_axis_rotates_minus_90_degrees(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(30.0, 0.0),
                right_axis=(30.0, 0.0),
            )
        )

        self.assertAlmostEqual(decision.angle, -90.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_leftward_axis_rotates_plus_90_degrees(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(-30.0, 0.0),
                right_axis=(-30.0, 0.0),
            )
        )

        self.assertAlmostEqual(decision.angle, 90.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_upward_axis_normalizes_to_minus_180_degrees(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(0.0, -30.0),
                right_axis=(0.0, -30.0),
            )
        )

        self.assertEqual(decision.angle, -180.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_disabled_returns_without_validating_pose(self):
        decision = estimate_person_upright_rotation(
            object(),
            {},
            enabled=False,
        )

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "disabled")

    def test_one_valid_chain_is_enough(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(30.0, 0.0),
                right_axis=(30.0, 0.0),
                left_score=0.49,
            )
        )

        self.assertAlmostEqual(decision.angle, -90.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_score_at_visible_threshold_is_accepted(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(30.0, 0.0),
                left_score=0.5,
                right_score=0.49,
            )
        )

        self.assertAlmostEqual(decision.angle, -90.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_visible_axes_shorter_than_ten_percent_of_bbox_diagonal_are_rejected(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(0.0, 14.0),
                right_axis=(0.0, 14.0),
            )
        )

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "pose_axis_short")

    def test_axis_at_ten_percent_of_bbox_diagonal_is_accepted(self):
        minimum_axis_length = (
            math.hypot(self.RECT["w"], self.RECT["h"]) * 0.10
        )
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(minimum_axis_length, 0.0),
                right_score=0.49,
            )
        )

        self.assertAlmostEqual(decision.angle, -90.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_axes_disagreeing_by_more_than_30_degrees_conflict(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(0.0, 30.0),
                right_axis=(30.0, 0.0),
            )
        )

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "pose_axis_conflict")

    def test_exactly_30_degree_disagreement_is_combined_after_unit_normalization(self):
        decision = self._decision(
            self._coco17_keypoints(
                left_axis=(0.0, 20.0),
                right_axis=self._axis_for_angle(-30.0, length=60.0),
            )
        )

        self.assertAlmostEqual(decision.angle, -15.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_unit_axes_at_60_and_90_degrees_are_not_in_conflict(self):
        points = np.zeros((17, 3), dtype=np.float64)
        points[5] = (0.0, 0.0, 0.9)
        points[11] = (*self._axis_for_angle(60.0), 0.9)
        points[6] = (0.0, 0.0, 0.9)
        points[12] = (*self._axis_for_angle(90.0), 0.9)

        decision = self._decision(points)

        self.assertAlmostEqual(decision.angle, 75.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_valid_chains_are_weighted_by_their_minimum_endpoint_scores(self):
        points = self._coco17_keypoints(
            left_axis=self._axis_for_angle(12.0, length=25.0),
            right_axis=self._axis_for_angle(36.0, length=55.0),
            left_score=0.95,
            right_score=0.90,
        )
        points[11, 2] = 0.85
        points[6, 2] = 0.55

        left_axis = points[11, :2] - points[5, :2]
        right_axis = points[12, :2] - points[6, :2]
        left_unit = left_axis / np.linalg.norm(left_axis)
        right_unit = right_axis / np.linalg.norm(right_axis)
        weighted_axis = (
            left_unit * min(points[5, 2], points[11, 2])
            + right_unit * min(points[6, 2], points[12, 2])
        )
        weighted_axis /= np.linalg.norm(weighted_axis)
        expected_angle = -math.degrees(
            math.atan2(float(weighted_axis[0]), float(weighted_axis[1]))
        )
        expected_angle = (expected_angle + 180.0) % 360.0 - 180.0

        equal_weight_axis = left_unit + right_unit
        equal_weight_angle = -math.degrees(
            math.atan2(float(equal_weight_axis[0]), float(equal_weight_axis[1]))
        )
        self.assertGreater(abs(expected_angle - equal_weight_angle), 1.0)

        decision = self._decision(points)

        self.assertAlmostEqual(decision.angle, expected_angle)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_non_finite_coordinates_are_reported_missing(self):
        points = self._coco17_keypoints()
        points[5, 0] = np.nan
        points[6, 1] = np.inf

        decision = self._decision(points)

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "pose_axis_missing")

    def test_extreme_finite_coordinates_are_safe_under_strict_numpy_errors(self):
        points = np.zeros((17, 3), dtype=np.float64)
        points[5] = (-1e308, 0.0, 0.9)
        points[11] = (1e308, 0.0, 0.9)
        points[6] = (-1e308, 1.0, 0.9)
        points[12] = (1e308, 1.0, 0.9)

        previous_errors = np.seterr(over="raise", invalid="raise")
        try:
            decision = self._decision(points)
        finally:
            np.seterr(**previous_errors)

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "pose_axis_missing")

    def test_wrong_keypoint_shape_is_reported_missing(self):
        decision = self._decision(np.zeros((16, 3), dtype=np.float64))

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "pose_axis_missing")

    def test_unusable_confidence_is_reported_missing(self):
        decision = self._decision(
            self._coco17_keypoints(left_score=0.49, right_score=0.49)
        )

        self.assertEqual(decision.angle, 0.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "pose_axis_missing")

    def test_invalid_rect_and_unconvertible_points_are_reported_missing(self):
        cases = [
            (self._coco17_keypoints(), {"x": 0, "y": 0, "w": 0, "h": 100}),
            (self._coco17_keypoints(), {"x": 0, "y": 0, "w": np.inf, "h": 100}),
            (object(), self.RECT),
        ]

        for points, rect in cases:
            with self.subTest(points=type(points).__name__, rect=rect):
                decision = estimate_person_upright_rotation(
                    points,
                    rect,
                    enabled=True,
                )
                self.assertEqual(decision.angle, 0.0)
                self.assertFalse(decision.should_rotate)
                self.assertEqual(decision.reason, "pose_axis_missing")

    def test_seven_degrees_is_inside_dead_zone(self):
        axis = self._axis_for_angle(7.0)

        decision = self._decision(
            self._coco17_keypoints(left_axis=axis, right_axis=axis)
        )

        self.assertAlmostEqual(decision.angle, 7.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "upright_dead_zone")

    def test_eight_degrees_is_applied(self):
        axis = self._axis_for_angle(8.0)

        decision = self._decision(
            self._coco17_keypoints(left_axis=axis, right_axis=axis)
        )

        self.assertAlmostEqual(decision.angle, 8.0)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.reason, "applied")

    def test_weighted_axis_rounding_at_dead_zone_boundary_stays_inside(self):
        points = np.zeros((17, 3), dtype=np.float64)
        points[5] = (0.0, 0.0, 0.9)
        points[11] = (0.0, 40.0, 0.9)
        points[6] = (0.0, 0.0, 0.6)
        points[12] = (*self._axis_for_angle(20.049674556250654, 40.0), 0.6)

        decision = self._decision(points)

        candidates = []
        for shoulder_index, hip_index in ((5, 11), (6, 12)):
            vector = points[hip_index, :2] - points[shoulder_index, :2]
            length = math.hypot(float(vector[0]), float(vector[1]))
            candidates.append(
                (
                    vector / length,
                    min(points[shoulder_index, 2], points[hip_index, 2]),
                )
            )
        legacy_axis = sum(
            (unit_axis * weight for unit_axis, weight in candidates),
            start=np.zeros(2, dtype=np.float64),
        )
        legacy_axis /= math.hypot(
            float(legacy_axis[0]), float(legacy_axis[1])
        )
        legacy_angle = -math.degrees(
            math.atan2(float(legacy_axis[0]), float(legacy_axis[1]))
        )
        legacy_angle = (legacy_angle + 180.0) % 360.0 - 180.0

        self.assertEqual(decision.angle, legacy_angle)
        self.assertLess(decision.angle, 8.0)
        self.assertFalse(decision.should_rotate)
        self.assertEqual(decision.reason, "upright_dead_zone")


class PersonUprightCropPixelTests(unittest.TestCase):
    def test_horizontal_person_rotates_clockwise_and_tight_crops_support(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        image[45:55, 20:80] = (10, 20, 230)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[45:55, 20:80] = 7
        points = _horizontal_person_keypoints()

        result = build_person_crop(
            image,
            {"x": 20, "y": 40, "w": 60, "h": 20},
            mask,
            keypoints=points,
            rotation_enabled=True,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.rotation_applied)
        self.assertAlmostEqual(result.rotation_angle, -90.0)
        self.assertEqual(result.image.shape[:2], (66, 18))
        self.assertNotEqual(result.image.shape[0], 3 * result.image.shape[1])
        self.assertEqual(result.mask.shape, result.image.shape[:2])
        self.assertEqual(result.mask.dtype, mask.dtype)
        self.assertTrue(result.image.flags.c_contiguous)
        self.assertTrue(result.mask.flags.c_contiguous)
        self.assertEqual(set(np.unique(result.mask)), {0, 7})
        self.assertEqual(result.letterbox_padding_ratio, 0.0)
        ys, xs = np.nonzero(result.mask)
        self.assertGreater(int(ys.max() - ys.min()), int(xs.max() - xs.min()))
        self.assertGreaterEqual(int(xs.min()), 2)
        self.assertGreaterEqual(int(ys.min()), 2)
        self.assertGreaterEqual(result.image.shape[1] - int(xs.max()) - 1, 2)
        self.assertGreaterEqual(result.image.shape[0] - int(ys.max()) - 1, 2)

        crop_points = transform_keypoints_to_person_crop(points, result)
        nose_x, nose_y = crop_points[0, :2]
        self.assertGreaterEqual(float(nose_x), 0.0)
        self.assertLess(float(nose_x), result.image.shape[1])
        self.assertGreaterEqual(float(nose_y), 0.0)
        self.assertLess(float(nose_y), result.image.shape[0])

    def test_forced_rotation_with_zero_pose_tight_crops_from_mask(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[45:55, 20:80] = 7

        with _force_clockwise_rotation():
            result = build_person_crop(
                image,
                {"x": 20, "y": 40, "w": 60, "h": 20},
                mask,
                keypoints=np.zeros((17, 3), dtype=np.float64),
                rotation_enabled=True,
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.rotation_applied)
        self.assertEqual(result.rotation_reason, "applied")
        self.assertEqual(result.image.shape[:2], (66, 14))
        self.assertNotEqual(result.image.shape[0], 3 * result.image.shape[1])
        self.assertGreater(int(np.count_nonzero(result.mask)), 0)
        ys, xs = np.nonzero(result.mask)
        self.assertGreaterEqual(int(xs.min()), 2)
        self.assertGreaterEqual(int(ys.min()), 2)

    def test_invalid_mask_tight_crops_from_reliable_visible_pose(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        points = _horizontal_person_keypoints()

        result = build_person_crop(
            image,
            {"x": 20, "y": 40, "w": 60, "h": 20},
            np.zeros((1, 1), dtype=np.uint8),
            keypoints=points,
            rotation_enabled=True,
        )

        self.assertIsNotNone(result)
        self.assertIsNone(result.mask)
        self.assertEqual(result.image.shape[:2], (50, 18))
        crop_points = transform_keypoints_to_person_crop(points, result)
        visible = points[:, 2] >= 0.5
        self.assertTrue(np.all(crop_points[visible, 0] >= 0.0))
        self.assertTrue(np.all(crop_points[visible, 0] < result.image.shape[1]))
        self.assertTrue(np.all(crop_points[visible, 1] >= 0.0))
        self.assertTrue(np.all(crop_points[visible, 1] < result.image.shape[0]))

    def test_missing_mask_and_pose_tight_crops_from_rotated_bbox(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        rect = {"x": 20, "y": 40, "w": 60, "h": 20}

        with _force_clockwise_rotation():
            result = build_person_crop(
                image,
                rect,
                np.zeros((1, 1), dtype=np.uint8),
                margin_ratio=0.5,
                keypoints=np.zeros((17, 3), dtype=np.float64),
                rotation_enabled=True,
            )

        self.assertIsNotNone(result)
        self.assertIsNone(result.mask)
        self.assertTrue(result.rotation_applied)
        self.assertEqual(result.rotation_reason, "applied")
        self.assertEqual(result.image.shape[:2], (66, 25))
        projected_rect = project_rect_to_person_crop(rect, result)
        self.assertGreater(int(np.count_nonzero(projected_rect)), 0)

    def test_disabled_rotation_is_pixel_identical_to_legacy_call(self):
        image = np.arange(120 * 80 * 3, dtype=np.uint8).reshape(120, 80, 3)
        mask = np.zeros((120, 80), dtype=np.uint8)
        mask[20:100, 20:60] = 1
        rect = {"x": 20, "y": 20, "w": 40, "h": 80}
        points = PersonUprightAngleTests._coco17_keypoints(
            left_axis=(30.0, 0.0),
            right_axis=(30.0, 0.0),
        )

        legacy = build_person_crop(image, rect, mask)
        disabled = build_person_crop(
            image,
            rect,
            mask,
            keypoints=points,
            rotation_enabled=False,
        )

        self.assertIsNotNone(legacy)
        self.assertIsNotNone(disabled)
        np.testing.assert_array_equal(disabled.image, legacy.image)
        np.testing.assert_array_equal(disabled.mask, legacy.mask)
        self.assertEqual(disabled.requested_xyxy, legacy.requested_xyxy)
        self.assertEqual(disabled.padding, legacy.padding)
        self.assertEqual(disabled.rotation_reason, "disabled")


class PersonUprightProjectionTests(unittest.TestCase):
    def _result(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[45:55, 20:80] = 1
        points = _horizontal_person_keypoints()
        return (
            build_person_crop(
                image,
                {"x": 20, "y": 40, "w": 60, "h": 20},
                mask,
                keypoints=points,
                rotation_enabled=True,
            ),
            points,
            mask,
        )

    def test_keypoints_and_target_mask_share_the_transform(self):
        result, points, mask = self._result()

        crop_points = transform_keypoints_to_person_crop(points, result)
        projected = project_mask_to_person_crop(mask, result, (100, 120, 3))

        np.testing.assert_array_equal(projected, result.mask)
        self.assertTrue(np.any(result.source_to_crop[:, 2] != 0.0))
        self.assertGreater(crop_points[11, 1], crop_points[5, 1])
        self.assertAlmostEqual(crop_points[11, 0], crop_points[5, 0], delta=1.0)

    def test_non_right_angles_and_image_edges_share_mask_and_validity_projection(
        self,
    ):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        cases = (
            (
                "interior_30",
                30.0,
                {"x": 30, "y": 25, "w": 50, "h": 40},
                (slice(30, 60), slice(35, 75)),
            ),
            (
                "top_left_45",
                45.0,
                {"x": 0, "y": 0, "w": 36, "h": 32},
                (slice(0, 26), slice(0, 30)),
            ),
            (
                "bottom_right_minus_30",
                -30.0,
                {"x": 84, "y": 66, "w": 36, "h": 34},
                (slice(72, 100), slice(90, 120)),
            ),
        )

        for name, angle, rect, mask_slice in cases:
            with self.subTest(case=name):
                mask = np.zeros(image.shape[:2], dtype=np.uint16)
                mask[mask_slice] = 17
                with _force_rotation(angle):
                    result = build_person_crop(
                        image,
                        rect,
                        mask,
                        keypoints=np.zeros((17, 3), dtype=np.float64),
                        rotation_enabled=True,
                    )

                self.assertIsNotNone(result)
                self.assertTrue(result.rotation_applied)
                self.assertAlmostEqual(result.rotation_angle, angle)
                self.assertGreater(result.image.size, 0)
                self.assertIsNotNone(result.mask)
                projected_mask = project_mask_to_person_crop(
                    mask,
                    result,
                    image.shape,
                )
                np.testing.assert_array_equal(projected_mask, result.mask)

                validity = np.zeros(image.shape[:2], dtype=np.uint8)
                source_x1, source_y1, source_x2, source_y2 = (
                    result.source_xyxy
                )
                validity[source_y1:source_y2, source_x1:source_x2] = 1
                projected_validity = cv2.warpAffine(
                    validity,
                    result.source_to_crop,
                    (result.image.shape[1], result.image.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                expected_padding_ratio = 1.0 - (
                    np.count_nonzero(projected_validity)
                    / projected_validity.size
                )
                self.assertAlmostEqual(
                    result.padding_ratio,
                    expected_padding_ratio,
                )

    def test_noncontiguous_mask_dtypes_survive_thirty_degree_projection(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        rect = {"x": 20, "y": 25, "w": 70, "h": 50}
        cases = (
            (np.bool_, True),
            (np.uint16, 257),
            (np.int32, -7),
            (np.float32, 3.25),
        )

        for dtype, foreground_value in cases:
            with self.subTest(dtype=np.dtype(dtype).name):
                storage = np.zeros((100, 240), dtype=dtype)
                mask = storage[:, ::2]
                mask[35:65, 30:85] = foreground_value
                self.assertFalse(mask.flags.c_contiguous)

                with _force_rotation(30.0):
                    result = build_person_crop(
                        image,
                        rect,
                        mask,
                        keypoints=np.zeros((17, 3), dtype=np.float64),
                        rotation_enabled=True,
                    )

                self.assertIsNotNone(result)
                self.assertIsNotNone(result.mask)
                self.assertTrue(result.rotation_applied)
                self.assertAlmostEqual(result.rotation_angle, 30.0)
                self.assertEqual(result.mask.dtype, mask.dtype)
                self.assertTrue(result.mask.flags.c_contiguous)
                projected = project_mask_to_person_crop(
                    mask,
                    result,
                    image.shape,
                )
                np.testing.assert_array_equal(projected, result.mask)
                self.assertIn(foreground_value, np.unique(result.mask))

    def test_padding_diagnostics_match_final_projected_source_validity(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[0:5, 0:20] = 1

        with _force_clockwise_rotation():
            result = build_person_crop(
                image,
                {"x": 0, "y": 0, "w": 60, "h": 20},
                mask,
                keypoints=np.zeros((17, 3), dtype=np.float64),
                rotation_enabled=True,
            )

        validity = np.zeros(image.shape[:2], dtype=np.uint8)
        source_x1, source_y1, source_x2, source_y2 = result.source_xyxy
        validity[source_y1:source_y2, source_x1:source_x2] = 1
        projected_validity = cv2.warpAffine(
            validity,
            result.source_to_crop,
            (result.image.shape[1], result.image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        expected_padding_ratio = 1.0 - (
            np.count_nonzero(projected_validity) / projected_validity.size
        )
        self.assertAlmostEqual(result.padding_ratio, expected_padding_ratio)
        self.assertGreater(result.padding_ratio, 0.0)
        self.assertEqual(result.letterbox_padding_ratio, 0.0)

        ys, xs = np.nonzero(projected_validity)
        expected_content = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        self.assertEqual(result.content_xyxy, expected_content)
        self.assertEqual(
            result.padding,
            (
                expected_content[0],
                expected_content[1],
                result.image.shape[1] - expected_content[2],
                result.image.shape[0] - expected_content[3],
            ),
        )
        requested_x1, requested_y1, requested_x2, requested_y2 = (
            result.requested_xyxy
        )
        requested_area = (requested_x2 - requested_x1) * (
            requested_y2 - requested_y1
        )
        source_area = (source_x2 - source_x1) * (source_y2 - source_y1)
        self.assertAlmostEqual(
            result.source_overflow_ratio,
            (requested_area - source_area) / requested_area,
        )

    def test_mask_outside_natural_window_is_not_sampled(self):
        result, _, _ = self._result()
        foreign = np.zeros((100, 120), dtype=np.uint8)
        foreign[:, 100:120] = 1

        projected = project_mask_to_person_crop(
            foreign,
            result,
            (100, 120, 3),
        )

        self.assertEqual(int(np.count_nonzero(projected)), 0)

    def test_bbox_fallback_becomes_a_rotated_polygon(self):
        result, _, _ = self._result()

        projected = project_rect_to_person_crop(
            {"x": 50, "y": 45, "w": 20, "h": 10},
            result,
        )

        self.assertGreater(int(np.count_nonzero(projected)), 0)
        ys, xs = np.nonzero(projected)
        self.assertGreater(int(ys.max() - ys.min()), int(xs.max() - xs.min()))


class PersonUprightFallbackTests(unittest.TestCase):
    def test_tight_support_failure_keeps_the_full_rotated_aabb(self):
        image = np.full((100, 120, 3), 80, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[45:55, 20:80] = 1
        rect = {"x": 20, "y": 40, "w": 60, "h": 20}

        with patch(
            "utils.keyframe.person_crop._expand_clip_xyxy",
            side_effect=ValueError("bad support"),
        ):
            result = build_person_crop(
                image,
                rect,
                mask,
                keypoints=_horizontal_person_keypoints(),
                rotation_enabled=True,
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.rotation_applied)
        self.assertEqual(result.rotation_reason, "applied")
        self.assertEqual(result.image.shape[:2], (66, 23))
        self.assertNotEqual(result.image.shape[0], 3 * result.image.shape[1])
        self.assertEqual(result.mask.shape, result.image.shape[:2])
        self.assertEqual(result.letterbox_padding_ratio, 0.0)

    def test_opencv_failure_returns_legacy_crop_with_named_reason(self):
        image = np.full((100, 120, 3), 80, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[45:55, 20:80] = 1
        rect = {"x": 20, "y": 40, "w": 60, "h": 20}
        points = PersonUprightAngleTests._coco17_keypoints(
            left_axis=(40.0, 0.0),
            right_axis=(40.0, 0.0),
        )
        legacy = build_person_crop(image, rect, mask)

        with patch(
            "utils.keyframe.person_crop.cv2.warpAffine",
            side_effect=RuntimeError("boom"),
        ):
            result = build_person_crop(
                image,
                rect,
                mask,
                keypoints=points,
                rotation_enabled=True,
            )

        self.assertEqual(result.rotation_reason, "rotation_failed")
        self.assertFalse(result.rotation_applied)
        np.testing.assert_array_equal(result.image, legacy.image)
        np.testing.assert_array_equal(result.mask, legacy.mask)

    def test_invalid_warped_mask_shape_uses_the_same_legacy_fallback(self):
        image = np.full((100, 120, 3), 80, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[45:55, 20:80] = 1
        rect = {"x": 20, "y": 40, "w": 60, "h": 20}
        points = PersonUprightAngleTests._coco17_keypoints(
            left_axis=(40.0, 0.0),
            right_axis=(40.0, 0.0),
        )
        legacy = build_person_crop(image, rect, mask)

        with patch(
            "utils.keyframe.person_crop._warp_mask_nearest",
            return_value=np.zeros((1, 1), dtype=np.uint8),
        ):
            result = build_person_crop(
                image,
                rect,
                mask,
                keypoints=points,
                rotation_enabled=True,
            )

        self.assertEqual(result.rotation_reason, "rotation_failed")
        self.assertFalse(result.rotation_applied)
        np.testing.assert_array_equal(result.image, legacy.image)
        np.testing.assert_array_equal(result.mask, legacy.mask)


if __name__ == "__main__":
    unittest.main()
