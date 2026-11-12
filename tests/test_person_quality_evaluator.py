import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
from tests.person_quality_test_support import (
    PERSON_TEST_IMAGE_SHAPE,
    _evaluate_person_test_quality,
    _person_test_mask,
    _person_test_segment,
)
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
)
from utils.quality.person import evaluator, mask as person_mask
from utils.quality.person.policy import (
    PERSON_ACTION_BUILD_GALLERY,
    PERSON_ACTION_DISCARD,
    PERSON_ACTION_IDENTIFY_ONLY,
)
from utils.quality.quality_reasons import QualityReason


class PersonQualityTests(unittest.TestCase):
    def test_evaluator_uses_configured_bbox_min_width(self):
        edge = algorithm_config.person.quality.edge.model_copy(
            update={"min_bbox_width_px": 25}
        )
        alternate = SimpleNamespace(person=SimpleNamespace(quality=algorithm_config.person.quality.model_copy(update={"edge": edge})))
        with patch.object(evaluator, "algorithm_config", alternate):
            decision = _evaluate_person_test_quality(
                _person_test_segment(rect={"x": 40, "y": 40, "w": 24, "h": 80})
            )
        self.assertIn(QualityReason.BBOX_TOO_SMALL, decision.quality_reasons)

    def test_bbox_size_area_and_aspect_threshold_boundaries(self):
        cases = (
            ({"x": 40, "y": 40, "w": 23, "h": 80}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_TOO_SMALL, True),
            ({"x": 40, "y": 40, "w": 24, "h": 80}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_TOO_SMALL, False),
            ({"x": 40, "y": 40, "w": 40, "h": 47}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_TOO_SMALL, True),
            ({"x": 40, "y": 40, "w": 40, "h": 48}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_TOO_SMALL, False),
            ({"x": 100, "y": 100, "w": 40, "h": 49}, (1000, 1000, 3), QualityReason.BBOX_AREA_TOO_SMALL, True),
            ({"x": 100, "y": 100, "w": 40, "h": 50}, (1000, 1000, 3), QualityReason.BBOX_AREA_TOO_SMALL, False),
            ({"x": 0, "y": 0, "w": 170, "h": 200}, (200, 200, 3), QualityReason.BBOX_AREA_TOO_LARGE, False),
            ({"x": 0, "y": 0, "w": 171, "h": 200}, (200, 200, 3), QualityReason.BBOX_AREA_TOO_LARGE, True),
            ({"x": 40, "y": 40, "w": 60, "h": 48}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_ASPECT_RATIO_ABNORMAL, False),
            ({"x": 40, "y": 40, "w": 61, "h": 48}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_ASPECT_RATIO_ABNORMAL, True),
            ({"x": 40, "y": 40, "w": 24, "h": 120}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_ASPECT_RATIO_ABNORMAL, False),
            ({"x": 40, "y": 40, "w": 24, "h": 121}, PERSON_TEST_IMAGE_SHAPE, QualityReason.BBOX_ASPECT_RATIO_ABNORMAL, True),
        )
        for rect, shape, reason, expected in cases:
            with self.subTest(rect=rect, reason=reason):
                decision = _evaluate_person_test_quality(
                    _person_test_segment(rect=rect), shape=shape
                )
                self.assertEqual(reason in decision.quality_reasons, expected)

    def test_other_person_overlap_threshold_boundaries(self):
        current = _person_test_segment()
        cases = (
            (15, 40, QualityReason.BBOX_OVERLAPS_OTHER_PERSON, False),
            (16, 40, QualityReason.BBOX_OVERLAPS_OTHER_PERSON, True),
            (40, 40, QualityReason.BBOX_HEAVILY_OVERLAPS_OTHER_PERSON, False),
            (40, 41, QualityReason.BBOX_HEAVILY_OVERLAPS_OTHER_PERSON, True),
        )
        for width, height, reason, expected in cases:
            other = {"rect": {"x": 40, "y": 40, "w": width, "h": height}}
            with self.subTest(width=width, height=height):
                decision = _evaluate_person_test_quality(current, [current, other])
                self.assertEqual(reason in decision.quality_reasons, expected)

    def test_mask_bbox_and_component_ratio_boundaries(self):
        for pixels, reason, expected in (
            (479, QualityReason.MASK_BBOX_RATIO_TOO_LOW, True),
            (480, QualityReason.MASK_BBOX_RATIO_TOO_LOW, False),
            (999, QualityReason.MASK_BBOX_RATIO_LOW, True),
            (1000, QualityReason.MASK_BBOX_RATIO_LOW, False),
        ):
            mask = np.zeros(PERSON_TEST_IMAGE_SHAPE[:2], dtype=np.uint8)
            mask[50:50 + pixels // 20, 45:65] = 1
            mask[50 + pixels // 20, 45:45 + pixels % 20] = 1
            with self.subTest(pixels=pixels):
                decision = _evaluate_person_test_quality(_person_test_segment(mask=mask))
                self.assertEqual(reason in decision.quality_reasons, expected)

        for largest, expected in ((550, False), (540, True)):
            mask = np.zeros(PERSON_TEST_IMAGE_SHAPE[:2], dtype=np.uint8)
            mask[50:50 + largest // 25, 45:70] = 1
            mask[50 + largest // 25, 45:45 + largest % 25] = 1
            other = 1000 - largest
            mask[100:100 + other // 25, 45:70] = 1
            mask[100 + other // 25, 45:45 + other % 25] = 1
            with self.subTest(largest=largest):
                decision = _evaluate_person_test_quality(_person_test_segment(mask=mask))
                self.assertEqual(QualityReason.MASK_FRAGMENTED in decision.quality_reasons, expected)

    def test_mask_edge_contact_pixel_boundary(self):
        base = np.zeros(PERSON_TEST_IMAGE_SHAPE[:2], dtype=np.uint8)
        base[70:90, 50:70] = 1
        for touch_pixels, expected in ((3, 0), (4, 1)):
            mask = base.copy()
            mask[40, 50:50 + touch_pixels] = 1
            with self.subTest(touch_pixels=touch_pixels):
                _, touched_edges = person_mask._mask_bbox_edge_stats(
                    mask, {"x": 40, "y": 40, "w": 40, "h": 100}
                )
                self.assertEqual(touched_edges, expected)

        for touch_pixels, expected in ((2, frozenset()), (3, frozenset({"left"}))):
            mask = base.copy()
            mask[70:70 + touch_pixels, 0] = 1
            with self.subTest(image_touch_pixels=touch_pixels):
                self.assertEqual(person_mask._mask_image_edge_directions(mask), expected)

    def test_high_quality_mask_and_crop_builds_gallery(self):
        decision = _evaluate_person_test_quality(
            _person_test_segment(mask=_person_test_mask())
        )
        self.assertEqual(decision.person_quality_level, QUALITY_LEVEL_ENROLL)
        self.assertEqual(decision.person_quality_action, PERSON_ACTION_BUILD_GALLERY)
        self.assertEqual(decision.mask_score, 10.0)
        self.assertEqual(decision.crop_score, 10.0)
        self.assertEqual(decision.quality_reasons, [])

    def test_mask_pixel_boundaries_preserve_existing_grading(self):
        cases = [
            (5999, QUALITY_LEVEL_DROP, PERSON_ACTION_DISCARD),
            (6000, QUALITY_LEVEL_RECOGNITION, PERSON_ACTION_IDENTIFY_ONLY),
            (8399, QUALITY_LEVEL_RECOGNITION, PERSON_ACTION_IDENTIFY_ONLY),
            (8400, QUALITY_LEVEL_ENROLL, PERSON_ACTION_BUILD_GALLERY),
        ]

        for mask_pixel_count, expected_level, expected_action in cases:
            with self.subTest(mask_pixel_count=mask_pixel_count):
                decision = _evaluate_person_test_quality(
                    _person_test_segment(mask_pixel_count=mask_pixel_count)
                )
                self.assertEqual(decision.person_quality_level, expected_level)
                self.assertEqual(decision.person_quality_action, expected_action)

    def test_low_mask_bbox_ratio_drops_or_caps(self):
        drop = _evaluate_person_test_quality(
            _person_test_segment(mask=_person_test_mask(boxes=[(45, 45, 55, 55)]))
        )
        cap = _evaluate_person_test_quality(
            _person_test_segment(mask=_person_test_mask(boxes=[(45, 50, 65, 90)]))
        )

        self.assertEqual(drop.person_quality_level, QUALITY_LEVEL_DROP)
        self.assertIn(QualityReason.MASK_BBOX_RATIO_TOO_LOW, drop.quality_reasons)
        self.assertEqual(cap.person_quality_level, QUALITY_LEVEL_RECOGNITION)
        self.assertIn(QualityReason.MASK_BBOX_RATIO_LOW, cap.quality_reasons)

    def test_multiple_mask_components_cap_and_fragmentation_drops(self):
        cap = _evaluate_person_test_quality(
            _person_test_segment(
                mask=_person_test_mask(
                    boxes=[
                        (45, 50, 65, 100),
                        (70, 60, 90, 100),
                    ]
                )
            )
        )
        drop = _evaluate_person_test_quality(
            _person_test_segment(
                mask=_person_test_mask(
                    boxes=[
                        (45, 50, 60, 90),
                        (65, 50, 80, 90),
                        (85, 50, 100, 90),
                    ]
                )
            )
        )

        self.assertEqual(cap.person_quality_level, QUALITY_LEVEL_RECOGNITION)
        self.assertIn(QualityReason.MASK_MULTIPLE_COMPONENTS, cap.quality_reasons)
        self.assertEqual(drop.person_quality_level, QUALITY_LEVEL_DROP)
        self.assertIn(QualityReason.MASK_FRAGMENTED, drop.quality_reasons)

    def test_mask_touching_multiple_bbox_edges_is_reported(self):
        decision = _evaluate_person_test_quality(
            _person_test_segment(mask=_person_test_mask(boxes=[(40, 40, 80, 140)]))
        )

        self.assertEqual(decision.person_quality_level, QUALITY_LEVEL_ENROLL)
        self.assertIn(
            QualityReason.MASK_TOUCHING_BBOX_EDGE,
            decision.quality_reasons,
        )

    def test_crop_geometry_gates(self):
        cases = [
            (
                {"x": 40, "y": 40, "w": 20, "h": 80},
                QUALITY_LEVEL_DROP,
                QualityReason.BBOX_TOO_SMALL,
                PERSON_TEST_IMAGE_SHAPE,
            ),
            (
                {"x": 10, "y": 10, "w": 190, "h": 190},
                QUALITY_LEVEL_RECOGNITION,
                QualityReason.BBOX_AREA_TOO_LARGE,
                (200, 200, 3),
            ),
            (
                {"x": 40, "y": 40, "w": 100, "h": 50},
                QUALITY_LEVEL_RECOGNITION,
                QualityReason.BBOX_ASPECT_RATIO_ABNORMAL,
                (200, 200, 3),
            ),
            (
                {"x": 2, "y": 40, "w": 40, "h": 100},
                QUALITY_LEVEL_ENROLL,
                QualityReason.BBOX_NEAR_IMAGE_EDGE,
                PERSON_TEST_IMAGE_SHAPE,
            ),
            (
                {"x": 0, "y": 0, "w": 40, "h": 200},
                QUALITY_LEVEL_ENROLL,
                QualityReason.BBOX_NEAR_IMAGE_EDGE,
                PERSON_TEST_IMAGE_SHAPE,
            ),
        ]

        for rect, expected_level, expected_reason, shape in cases:
            with self.subTest(reason=expected_reason):
                decision = _evaluate_person_test_quality(
                    _person_test_segment(rect=rect), shape=shape
                )
                self.assertEqual(decision.person_quality_level, expected_level)
                self.assertIn(expected_reason, decision.quality_reasons)

    def test_other_person_overlap_caps_or_drops(self):
        current = _person_test_segment(mask=_person_test_mask())
        cap_other = _person_test_segment(rect={"x": 60, "y": 80, "w": 30, "h": 40})
        drop_other = _person_test_segment(rect={"x": 45, "y": 50, "w": 40, "h": 70})

        cap = _evaluate_person_test_quality(current, [current, cap_other], index=0)
        drop = _evaluate_person_test_quality(current, [current, drop_other], index=0)

        self.assertEqual(cap.person_quality_level, QUALITY_LEVEL_RECOGNITION)
        self.assertIn(
            QualityReason.BBOX_OVERLAPS_OTHER_PERSON,
            cap.quality_reasons,
        )
        self.assertEqual(drop.person_quality_level, QUALITY_LEVEL_DROP)
        self.assertIn(
            QualityReason.BBOX_HEAVILY_OVERLAPS_OTHER_PERSON,
            drop.quality_reasons,
        )

    def test_invalid_confidence_drops_but_normal_confidence_does_not_cap(self):
        invalid = _evaluate_person_test_quality(_person_test_segment(conf=0))
        normal = _evaluate_person_test_quality(_person_test_segment(conf=0.5))

        self.assertEqual(invalid.person_quality_level, QUALITY_LEVEL_DROP)
        self.assertIn(QualityReason.INVALID_CONFIDENCE, invalid.quality_reasons)
        self.assertEqual(normal.person_quality_level, QUALITY_LEVEL_ENROLL)
        self.assertNotIn(QualityReason.INVALID_CONFIDENCE, normal.quality_reasons)

    def test_missing_or_invalid_mask_falls_back_to_mask_pixel_score(self):
        missing = _evaluate_person_test_quality(_person_test_segment(mask=None))
        invalid_shape = _evaluate_person_test_quality(
            _person_test_segment(mask=np.zeros((10, 10), dtype=np.uint8))
        )

        self.assertEqual(missing.person_quality_level, QUALITY_LEVEL_ENROLL)
        self.assertIn(QualityReason.MASK_MISSING, missing.quality_reasons)
        self.assertEqual(invalid_shape.person_quality_level, QUALITY_LEVEL_ENROLL)
        self.assertIn(QualityReason.MASK_SHAPE_INVALID, invalid_shape.quality_reasons)
        self.assertTrue(
            all(
                isinstance(reason, QualityReason)
                for reason in invalid_shape.quality_reasons
            )
        )
