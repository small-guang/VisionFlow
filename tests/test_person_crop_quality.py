import unittest

import numpy as np

from utils.keyframe.person_crop import PersonCropResult, build_person_crop
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
)
from utils.quality.person.crop import evaluate_person_crop_quality
from utils.quality.quality_reasons import QualityReason


class PersonCropQualityTests(unittest.TestCase):
    @staticmethod
    def _crop_result(
        padding_ratio: float,
        *,
        letterbox_padding_ratio: float = 0.0,
        source_overflow_ratio: float = 0.0,
        size: int = 100,
    ) -> PersonCropResult:
        return PersonCropResult(
            image=np.zeros((size, size, 3), dtype=np.uint8),
            mask=np.ones((size, size), dtype=np.uint8),
            requested_xyxy=(0, 0, size, size),
            source_xyxy=(0, 0, size, size),
            content_xyxy=(0, 0, size, size),
            padding=(0, 0, 0, 0),
            padding_ratio=padding_ratio,
            letterbox_padding_ratio=letterbox_padding_ratio,
            source_overflow_ratio=source_overflow_ratio,
        )

    @staticmethod
    def _segment(size: int = 100) -> dict:
        return {"rect": {"x": 0, "y": 0, "w": size, "h": size}}

    def test_rotated_foreign_mask_uses_the_same_affine_projection(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        target = np.zeros((100, 120), dtype=np.uint8)
        target[45:55, 20:80] = 1
        foreign = np.zeros((100, 120), dtype=np.uint8)
        foreign[45:55, 50:70] = 1
        rect = {"x": 20, "y": 40, "w": 60, "h": 20}
        points = np.zeros((17, 3), dtype=np.float32)
        points[5] = (30.0, 50.0, 0.9)
        points[11] = (70.0, 50.0, 0.9)
        points[6] = (30.0, 55.0, 0.9)
        points[12] = (70.0, 55.0, 0.9)
        result = build_person_crop(
            image,
            rect,
            target,
            keypoints=points,
            rotation_enabled=True,
        )
        segments = [
            {"rect": rect, "mask": target},
            {
                "rect": {"x": 50, "y": 45, "w": 20, "h": 10},
                "mask": foreign,
            },
        ]

        decision = evaluate_person_crop_quality(
            result,
            segments[0],
            segments,
            image.shape,
            segment_index=0,
        )

        self.assertGreater(decision.metrics.foreign_person_pixel_count, 0)
        self.assertEqual(
            decision.metrics.foreign_person_evidence_source,
            "mask",
        )

    @staticmethod
    def _foreign_mask(*, edge_pixels: int = 0, core_pixels: int = 0) -> np.ndarray:
        mask = np.zeros((100, 100), dtype=np.uint8)
        edge_region = np.zeros_like(mask, dtype=bool)
        edge_region[:, :15] = True
        edge_region[:, 85:] = True
        core_region = np.zeros_like(mask, dtype=bool)
        core_region[:, 15:85] = True
        for region, count in (
            (edge_region, edge_pixels),
            (core_region, core_pixels),
        ):
            coordinates = np.argwhere(region)
            selected = coordinates[:count]
            mask[selected[:, 0], selected[:, 1]] = 1
        return mask

    def test_padding_ratios_are_recorded_without_gating(self):
        segment = self._segment()
        for padding_ratio in (0.20, 0.5001, 0.90):
            with self.subTest(padding_ratio=padding_ratio):
                decision = evaluate_person_crop_quality(
                    self._crop_result(
                        padding_ratio,
                        letterbox_padding_ratio=0.75,
                        source_overflow_ratio=0.25,
                    ),
                    segment,
                    [segment],
                    (100, 100, 3),
                    segment_index=0,
                )
                self.assertFalse(decision.reject)
                self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)
                self.assertEqual(decision.quality_reasons, [])
                self.assertEqual(decision.metrics.padding_ratio, padding_ratio)
                self.assertEqual(decision.metrics.letterbox_padding_ratio, 0.75)
                self.assertEqual(decision.metrics.source_overflow_ratio, 0.25)

    def _foreign_decision(
        self,
        *,
        edge_pixels: int = 0,
        core_pixels: int = 0,
        extra_others: list[dict] | None = None,
    ):
        segment = self._segment()
        other = {
            "rect": {"x": 0, "y": 0, "w": 100, "h": 100},
            "mask": self._foreign_mask(
                edge_pixels=edge_pixels,
                core_pixels=core_pixels,
            ),
        }
        return evaluate_person_crop_quality(
            self._crop_result(0.0),
            segment,
            [segment, other, *(extra_others or [])],
            (100, 100, 3),
            segment_index=0,
        )

    def test_foreign_person_below_diagnostic_threshold_is_ignored(self):
        decision = self._foreign_decision(edge_pixels=99)
        self.assertFalse(decision.reject)
        self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)
        self.assertFalse(decision.has_second_person)
        self.assertEqual(decision.metrics.foreign_person_ratio, 0.0099)
        self.assertEqual(decision.metrics.foreign_person_evidence_source, "mask")

    def test_mild_edge_foreign_person_does_not_cap_gallery(self):
        decision = self._foreign_decision(edge_pixels=500)
        self.assertFalse(decision.reject)
        self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)
        self.assertTrue(decision.has_second_person)
        self.assertTrue(decision.metrics.foreign_person_edge_dominant)
        self.assertEqual(decision.metrics.foreign_person_ratio, 0.05)
        self.assertEqual(decision.metrics.foreign_person_core_ratio, 0.0)
        self.assertIn(
            QualityReason.CROP_CONTAINS_OTHER_PERSON_MILD,
            decision.quality_reasons,
        )

    def test_foreign_person_at_total_cap_threshold_caps_recognition(self):
        decision = self._foreign_decision(edge_pixels=1000)
        self.assertFalse(decision.reject)
        self.assertEqual(decision.cap_level, QUALITY_LEVEL_RECOGNITION)
        self.assertIn(QualityReason.CROP_CONTAINS_OTHER_PERSON, decision.quality_reasons)

    def test_foreign_person_at_total_reject_threshold_drops(self):
        decision = self._foreign_decision(edge_pixels=3000)
        self.assertTrue(decision.reject)
        self.assertEqual(decision.cap_level, QUALITY_LEVEL_DROP)
        self.assertTrue(decision.has_second_person)
        self.assertIn(
            QualityReason.CROP_HEAVILY_CONTAINS_OTHER_PERSON,
            decision.quality_reasons,
        )

    def test_core_thresholds_override_lower_total_ratio(self):
        capped = self._foreign_decision(core_pixels=500)
        rejected = self._foreign_decision(core_pixels=2000)

        self.assertEqual(capped.metrics.foreign_person_ratio, 0.05)
        self.assertEqual(capped.metrics.foreign_person_core_ratio, 0.05)
        self.assertEqual(capped.cap_level, QUALITY_LEVEL_RECOGNITION)
        self.assertFalse(rejected.metrics.foreign_person_edge_dominant)
        self.assertEqual(rejected.metrics.foreign_person_core_ratio, 0.20)
        self.assertTrue(rejected.reject)

    def test_edge_dominant_boundary_is_inclusive(self):
        decision = self._foreign_decision(edge_pixels=800, core_pixels=200)

        self.assertTrue(decision.metrics.foreign_person_edge_dominant)

    def test_valid_mask_outside_crop_does_not_fall_back_to_bbox(self):
        segment = self._segment()
        other_mask = np.zeros((200, 200), dtype=np.uint8)
        other_mask[150:160, 150:160] = 1
        other = {
            "rect": {"x": 0, "y": 0, "w": 50, "h": 50},
            "mask": other_mask,
        }

        decision = evaluate_person_crop_quality(
            self._crop_result(0.0),
            segment,
            [segment, other],
            (200, 200, 3),
            segment_index=0,
        )

        self.assertEqual(decision.metrics.foreign_person_pixel_count, 0)
        self.assertEqual(
            decision.metrics.foreign_person_bbox_fallback_pixel_count,
            0,
        )
        self.assertEqual(decision.metrics.foreign_person_evidence_source, "none")
        self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)

    def test_bbox_fallback_is_diagnostic_only(self):
        segment = self._segment()
        other = {"rect": {"x": 0, "y": 0, "w": 10, "h": 100}}

        decision = evaluate_person_crop_quality(
            self._crop_result(0.0),
            segment,
            [segment, other],
            (100, 100, 3),
            segment_index=0,
        )

        self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)
        self.assertTrue(decision.has_second_person)
        self.assertEqual(decision.metrics.foreign_person_ratio, 0.0)
        self.assertEqual(
            decision.metrics.foreign_person_bbox_fallback_ratio,
            0.10,
        )
        self.assertEqual(
            decision.metrics.foreign_person_evidence_source,
            "bbox_fallback",
        )
        self.assertIn(
            QualityReason.CROP_OTHER_PERSON_BBOX_FALLBACK,
            decision.quality_reasons,
        )

    def test_mask_and_bbox_fallback_are_reported_as_mixed(self):
        fallback = {"rect": {"x": 0, "y": 0, "w": 5, "h": 100}}

        decision = self._foreign_decision(
            edge_pixels=500,
            extra_others=[fallback],
        )

        self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)
        self.assertEqual(
            decision.metrics.foreign_person_evidence_source,
            "mixed",
        )
        self.assertIn(
            QualityReason.CROP_CONTAINS_OTHER_PERSON_MILD,
            decision.quality_reasons,
        )
        self.assertIn(
            QualityReason.CROP_OTHER_PERSON_BBOX_FALLBACK,
            decision.quality_reasons,
        )
