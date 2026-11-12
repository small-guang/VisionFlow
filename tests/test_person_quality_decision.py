import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from core.algorithm_config import algorithm_config
from utils.keyframe import keyframe_flow_helpers
from utils.quality.person import policy

from utils.quality.person.decision import (
    drop_person_quality_decision,
    merge_person_crop_quality,
    merge_person_visual_quality,
)
from utils.quality.person.models import (
    PersonCropMetrics,
    PersonCropQualityDecision,
    PersonQualityDecision,
    PersonVisualMetrics,
    PersonVisualQualityDecision,
)
from utils.quality.person.policy import (
    PERSON_ACTION_BUILD_GALLERY,
    PERSON_ACTION_DISCARD,
    PERSON_ACTION_IDENTIFY_ONLY,
)
from utils.quality.quality_reasons import QualityReason
from utils.reid.person_body import BODY_FULL, BODY_HALF_UPPER


class PersonCropDecisionMergeTests(unittest.TestCase):
    @staticmethod
    def _base_quality(
        visible_keypoint_count=13,
        *,
        dropped=False,
        half_body_pose_eligible=False,
    ):
        return PersonQualityDecision(
            person_quality_score=0.0 if dropped else 9.0,
            person_quality_level=0 if dropped else 2,
            person_quality_action=(
                PERSON_ACTION_DISCARD if dropped else PERSON_ACTION_BUILD_GALLERY
            ),
            mask_score=10.0,
            crop_score=10.0,
            quality_reasons=(
                [QualityReason.MASK_AREA_TOO_SMALL] if dropped else []
            ),
            pose_score=9.0,
            visible_keypoint_count=visible_keypoint_count,
            torso_keypoint_count=4 if half_body_pose_eligible else 0,
            shoulder_hip_chain_count=2 if half_body_pose_eligible else 0,
            half_body_pose_eligible=half_body_pose_eligible,
            body_completeness=(
                BODY_HALF_UPPER if half_body_pose_eligible else BODY_FULL
            ),
            is_half_body=half_body_pose_eligible,
        )

    @staticmethod
    def _crop(
        *,
        reject=False,
        cap_level=2,
        crop_score=10.0,
        reasons=(),
        metrics_overrides=None,
    ):
        metrics = {
            "padding_ratio": 0.90,
            "target_mask_pixel_count": 1000,
            "foreign_person_pixel_count": 0,
            "foreign_person_ratio": 0.0,
            "letterbox_padding_ratio": 0.75,
            "source_overflow_ratio": 0.25,
        }
        metrics.update(metrics_overrides or {})
        return PersonCropQualityDecision(
            reject=reject,
            cap_level=cap_level,
            crop_score=crop_score,
            quality_reasons=list(reasons),
            metrics=PersonCropMetrics(**metrics),
        )

    def test_padding_metrics_are_preserved_without_limiting_quality(self):
        decision = merge_person_crop_quality(
            self._base_quality(),
            self._crop(),
        )

        self.assertEqual(decision.person_quality_level, 2)
        self.assertEqual(decision.person_quality_action, PERSON_ACTION_BUILD_GALLERY)
        self.assertEqual(decision.padding_ratio, 0.90)
        self.assertEqual(decision.letterbox_padding_ratio, 0.75)
        self.assertEqual(decision.source_overflow_ratio, 0.25)

    def test_foreign_person_cap_and_reject_remain_effective(self):
        capped = merge_person_crop_quality(
            self._base_quality(half_body_pose_eligible=True),
            self._crop(
                cap_level=1,
                crop_score=6.99,
                reasons=(QualityReason.CROP_CONTAINS_OTHER_PERSON,),
                metrics_overrides={
                    "foreign_person_pixel_count": 100,
                    "foreign_person_ratio": 0.10,
                    "foreign_person_core_pixel_count": 20,
                    "foreign_person_core_ratio": 0.02,
                    "foreign_person_bbox_fallback_pixel_count": 10,
                    "foreign_person_bbox_fallback_ratio": 0.01,
                    "foreign_person_edge_dominant": True,
                    "foreign_person_evidence_source": "mixed",
                },
            ),
        )
        rejected = merge_person_crop_quality(
            self._base_quality(),
            self._crop(
                reject=True,
                cap_level=0,
                crop_score=0.0,
                reasons=(QualityReason.CROP_HEAVILY_CONTAINS_OTHER_PERSON,),
            ),
        )

        self.assertEqual(capped.person_quality_level, 1)
        self.assertEqual(capped.person_quality_action, PERSON_ACTION_IDENTIFY_ONLY)
        self.assertTrue(capped.half_body_pose_eligible)
        self.assertEqual(capped.torso_keypoint_count, 4)
        self.assertEqual(capped.shoulder_hip_chain_count, 2)
        self.assertEqual(capped.foreign_person_ratio, 0.10)
        self.assertEqual(capped.foreign_person_core_ratio, 0.02)
        self.assertEqual(capped.foreign_person_pixel_count, 100)
        self.assertEqual(capped.foreign_person_core_pixel_count, 20)
        self.assertEqual(capped.foreign_person_bbox_fallback_ratio, 0.01)
        self.assertEqual(capped.foreign_person_bbox_fallback_pixel_count, 10)
        self.assertTrue(capped.foreign_person_edge_dominant)
        self.assertEqual(capped.foreign_person_evidence_source, "mixed")
        self.assertEqual(rejected.person_quality_level, 0)
        self.assertEqual(rejected.person_quality_action, PERSON_ACTION_DISCARD)

    def test_visual_hard_reject_still_drops_sample(self):
        merged = merge_person_crop_quality(
            self._base_quality(),
            self._crop(
                metrics_overrides={
                    "foreign_person_pixel_count": 50,
                    "foreign_person_ratio": 0.05,
                    "foreign_person_core_pixel_count": 0,
                    "foreign_person_core_ratio": 0.0,
                    "foreign_person_edge_dominant": True,
                    "foreign_person_evidence_source": "mask",
                }
            ),
        )
        visual = PersonVisualQualityDecision(
            reject=True,
            cap_level=0,
            crop_score=0.0,
            quality_reasons=[QualityReason.BLURRY_LAPLACIAN],
            metrics=PersonVisualMetrics(
                laplacian_variance=0.0,
                black_clip_ratio=0.0,
                dark_ratio=0.0,
                white_clip_ratio=0.0,
                bright_ratio=0.0,
                p5=10.0,
                p10=20.0,
                p90=200.0,
                p95=220.0,
                contrast_span=210.0,
                used_foreground=True,
                sample_pixel_count=1000,
            ),
        )

        decision = merge_person_visual_quality(merged, visual)

        self.assertEqual(decision.person_quality_level, 0)
        self.assertEqual(decision.person_quality_action, PERSON_ACTION_DISCARD)
        self.assertIn(QualityReason.BLURRY_LAPLACIAN, decision.quality_reasons)
        self.assertEqual(decision.foreign_person_ratio, 0.05)
        self.assertTrue(decision.foreign_person_edge_dominant)
        self.assertEqual(decision.foreign_person_evidence_source, "mask")


    def test_forced_drop_preserves_half_body_pose_diagnostics(self):
        base = replace(
            self._base_quality(half_body_pose_eligible=True),
            foreign_person_ratio=0.05,
            foreign_person_pixel_count=50,
            foreign_person_edge_dominant=True,
            foreign_person_evidence_source="mask",
        )
        decision = drop_person_quality_decision(
            QualityReason.EMPTY_CROP,
            base_decision=base,
        )

        self.assertEqual(decision.person_quality_level, 0)
        self.assertEqual(decision.torso_keypoint_count, 4)
        self.assertEqual(decision.shoulder_hip_chain_count, 2)
        self.assertTrue(decision.half_body_pose_eligible)
        self.assertEqual(decision.foreign_person_ratio, 0.05)
        self.assertEqual(decision.foreign_person_pixel_count, 50)
        self.assertTrue(decision.foreign_person_edge_dominant)
        self.assertEqual(decision.foreign_person_evidence_source, "mask")


class PersonQualityScorePolicyTests(unittest.TestCase):
    def test_default_policy_weights_preserve_current_score(self):
        self.assertAlmostEqual(policy._combined_person_quality_score(10, 10, 10), 10.0)
        self.assertAlmostEqual(policy._combined_person_quality_score(8, 6, 4), 6.8)

    def test_combined_score_consumes_configured_weights(self):
        general = algorithm_config.person.quality.general.model_copy(update={
            "pose_weight": 1.0,
            "mask_weight": 0.0,
            "crop_weight": 0.0,
        })
        alternate = SimpleNamespace(person=SimpleNamespace(quality=SimpleNamespace(general=general)))
        with patch.object(policy, "algorithm_config", alternate):
            self.assertAlmostEqual(policy._combined_person_quality_score(8, 6, 4), 8.0)

    def test_mask_score_consumes_configured_score_ceiling(self):
        general = algorithm_config.person.quality.general.model_copy(update={"score_max": 8.0})
        alternate = SimpleNamespace(person=SimpleNamespace(quality=SimpleNamespace(general=general)))
        with patch.object(keyframe_flow_helpers, "algorithm_config", alternate):
            self.assertAlmostEqual(
                keyframe_flow_helpers.person_quality_score_from_mask(12000), 8.0
            )
