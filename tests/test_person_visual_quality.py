import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
from tests.person_quality_test_support import (
    _evaluate_person_test_quality,
    _person_test_mask,
    _person_test_segment,
)
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
)
from utils.quality.person.decision import merge_person_visual_quality
from utils.quality.person.models import PersonVisualMetrics
from utils.quality.person.policy import (
    PERSON_ACTION_DISCARD,
    PERSON_ACTION_IDENTIFY_ONLY,
)
from utils.quality.person.visual import (
    calculate_person_visual_metrics,
    classify_person_visual_metrics,
    evaluate_person_visual_quality,
    format_person_visual_thresholds,
)
from utils.quality.person import visual as person_quality_visual
from utils.quality.quality_reasons import QualityReason


class PersonVisualQualityTests(unittest.TestCase):
    def test_visual_classification_reads_configured_reject_threshold(self):
        visual = algorithm_config.person.quality.visual.model_copy(
            update={"laplacian_reject_threshold": 16.0}
        )
        alternate = SimpleNamespace(
            person=SimpleNamespace(quality=algorithm_config.person.quality.model_copy(update={"visual": visual}))
        )
        with patch.object(person_quality_visual, "algorithm_config", alternate):
            decision = classify_person_visual_metrics(
                self._metrics(laplacian_variance=15.0)
            )
        self.assertTrue(decision.reject)
        self.assertIn(QualityReason.BLURRY_LAPLACIAN, decision.quality_reasons)

    def test_visual_threshold_formatter_preserves_default_text(self):
        reasons = [
            QualityReason.BLURRY_LAPLACIAN,
            QualityReason.UNDEREXPOSED,
            QualityReason.OVEREXPOSED,
            QualityReason.LOW_CONTRAST,
        ]
        text = format_person_visual_thresholds(reasons)
        self.assertEqual(
            text,
            "预检阈值：Laplacian<10丢弃、<20降级；"
            "欠曝丢弃条件为黑场占比>=0.80或P95<=20，"
            "降级条件为暗像素占比>=0.70或P90<=40；"
            "过曝丢弃条件为白场占比>=0.80或P5>=245，"
            "降级条件为亮像素占比>=0.70或P10>=220；"
            "对比度<8丢弃、<20降级",
        )

    def test_visual_threshold_formatter_uses_configured_values(self):
        reasons = [QualityReason.BLURRY_LAPLACIAN, QualityReason.OVEREXPOSED]
        visual = algorithm_config.person.quality.visual.model_copy(update={
            "laplacian_reject_threshold": 11.0,
            "overexposed_reject_p5": 246.0,
            "exposure_reject_ratio": 0.8000001,
        })
        alternate = SimpleNamespace(person=SimpleNamespace(quality=SimpleNamespace(visual=visual)))
        with patch.object(person_quality_visual, "algorithm_config", alternate):
            adjusted = format_person_visual_thresholds(reasons)
        self.assertIn("Laplacian<11", adjusted)
        self.assertIn("P5>=246", adjusted)
        self.assertIn("占比>=0.8000001", adjusted)

    def test_visual_threshold_formatter_keeps_full_non_ratio_precision(self):
        visual = algorithm_config.person.quality.visual.model_copy(update={
            "laplacian_reject_threshold": 10.123456789,
            "underexposed_reject_p95": 20.123456789,
            "overexposed_reject_p5": 245.123456789,
            "contrast_reject_threshold": 8.123456789,
        })
        alternate = SimpleNamespace(person=SimpleNamespace(quality=SimpleNamespace(visual=visual)))
        with patch.object(person_quality_visual, "algorithm_config", alternate):
            text = format_person_visual_thresholds(list(QualityReason))
        for expected in (
            "Laplacian<10.123456789",
            "P95<=20.123456789",
            "P5>=245.123456789",
            "对比度<8.123456789",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

    def _metrics(self, **changes):
        metrics = PersonVisualMetrics(
            laplacian_variance=100.0,
            black_clip_ratio=0.0,
            dark_ratio=0.0,
            white_clip_ratio=0.0,
            bright_ratio=0.0,
            p5=50.0,
            p10=60.0,
            p90=180.0,
            p95=200.0,
            contrast_span=150.0,
            used_foreground=True,
            sample_pixel_count=1000,
        )
        return replace(metrics, **changes)

    def test_laplacian_threshold_boundaries(self):
        cases = [
            (9.99, True, QUALITY_LEVEL_DROP),
            (10.0, False, QUALITY_LEVEL_RECOGNITION),
            (19.99, False, QUALITY_LEVEL_RECOGNITION),
            (20.0, False, QUALITY_LEVEL_ENROLL),
        ]
        for value, expected_reject, expected_cap in cases:
            with self.subTest(value=value):
                decision = classify_person_visual_metrics(
                    self._metrics(laplacian_variance=value)
                )
                self.assertEqual(decision.reject, expected_reject)
                self.assertEqual(decision.cap_level, expected_cap)
                if value < 20.0:
                    self.assertIn(
                        QualityReason.BLURRY_LAPLACIAN,
                        decision.quality_reasons,
                    )

    def test_underexposure_threshold_boundaries(self):
        reject_ratio = classify_person_visual_metrics(
            self._metrics(black_clip_ratio=0.80)
        )
        reject_percentile = classify_person_visual_metrics(
            self._metrics(p95=20.0, contrast_span=100.0)
        )
        cap_ratio = classify_person_visual_metrics(
            self._metrics(dark_ratio=0.70)
        )
        cap_percentile = classify_person_visual_metrics(
            self._metrics(p90=40.0)
        )
        pass_decision = classify_person_visual_metrics(
            self._metrics(
                black_clip_ratio=0.799,
                dark_ratio=0.699,
                p90=40.01,
                p95=20.01,
            )
        )

        self.assertTrue(reject_ratio.reject)
        self.assertTrue(reject_percentile.reject)
        self.assertEqual(cap_ratio.cap_level, QUALITY_LEVEL_RECOGNITION)
        self.assertEqual(cap_percentile.cap_level, QUALITY_LEVEL_RECOGNITION)
        self.assertNotIn(QualityReason.UNDEREXPOSED, pass_decision.quality_reasons)

    def test_overexposure_threshold_boundaries(self):
        reject_ratio = classify_person_visual_metrics(
            self._metrics(white_clip_ratio=0.80)
        )
        reject_percentile = classify_person_visual_metrics(
            self._metrics(p5=245.0, contrast_span=10.0)
        )
        cap_ratio = classify_person_visual_metrics(
            self._metrics(bright_ratio=0.70)
        )
        cap_percentile = classify_person_visual_metrics(
            self._metrics(p10=220.0)
        )
        pass_decision = classify_person_visual_metrics(
            self._metrics(
                white_clip_ratio=0.799,
                bright_ratio=0.699,
                p5=244.99,
                p10=219.99,
            )
        )

        self.assertTrue(reject_ratio.reject)
        self.assertTrue(reject_percentile.reject)
        self.assertEqual(cap_ratio.cap_level, QUALITY_LEVEL_RECOGNITION)
        self.assertEqual(cap_percentile.cap_level, QUALITY_LEVEL_RECOGNITION)
        self.assertNotIn(QualityReason.OVEREXPOSED, pass_decision.quality_reasons)

    def test_contrast_threshold_boundaries(self):
        cases = [
            (7.99, True, QUALITY_LEVEL_DROP),
            (8.0, False, QUALITY_LEVEL_RECOGNITION),
            (19.99, False, QUALITY_LEVEL_RECOGNITION),
            (20.0, False, QUALITY_LEVEL_ENROLL),
        ]
        for value, expected_reject, expected_cap in cases:
            with self.subTest(value=value):
                decision = classify_person_visual_metrics(
                    self._metrics(contrast_span=value)
                )
                self.assertEqual(decision.reject, expected_reject)
                self.assertEqual(decision.cap_level, expected_cap)

    def test_foreground_metrics_ignore_sharp_background_and_mask_boundary(self):
        y, x = np.indices((100, 100))
        checker = np.where((x + y) % 2 == 0, 40, 210).astype(np.uint8)
        image = np.repeat(checker[:, :, None], 3, axis=2)
        image[20:80, 30:70] = 120
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 1

        foreground = calculate_person_visual_metrics(image, mask)
        whole_crop = calculate_person_visual_metrics(image)

        self.assertTrue(foreground.used_foreground)
        self.assertLess(foreground.laplacian_variance, 1.0)
        self.assertGreater(whole_crop.laplacian_variance, 100.0)

    def test_missing_invalid_or_tiny_mask_falls_back_to_whole_crop(self):
        y, x = np.indices((60, 40))
        gray = np.where((x // 2 + y // 2) % 2 == 0, 60, 180).astype(np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        masks = [
            None,
            np.zeros((10, 10, 2), dtype=np.uint8),
            np.pad(np.ones((1, 1), dtype=np.uint8), ((0, 59), (0, 39))),
        ]

        for mask in masks:
            with self.subTest(mask_shape=getattr(mask, "shape", None)):
                metrics = calculate_person_visual_metrics(image, mask)
                self.assertFalse(metrics.used_foreground)
                self.assertEqual(metrics.sample_pixel_count, 224 * 224)

    def test_visual_decision_merges_with_existing_quality(self):
        quality = _evaluate_person_test_quality(
            _person_test_segment(mask=_person_test_mask())
        )
        cap = classify_person_visual_metrics(self._metrics(contrast_span=10.0))
        drop = classify_person_visual_metrics(
            self._metrics(laplacian_variance=0.0)
        )

        capped = merge_person_visual_quality(quality, cap)
        rejected = merge_person_visual_quality(quality, drop)

        self.assertEqual(capped.person_quality_level, QUALITY_LEVEL_RECOGNITION)
        self.assertAlmostEqual(capped.person_quality_score, 9.5485)
        self.assertEqual(capped.crop_score, 6.99)
        self.assertEqual(capped.person_quality_action, PERSON_ACTION_IDENTIFY_ONLY)
        self.assertEqual(rejected.person_quality_level, QUALITY_LEVEL_DROP)
        self.assertEqual(rejected.person_quality_score, 0.0)
        self.assertEqual(rejected.crop_score, 0.0)
        self.assertEqual(rejected.person_quality_action, PERSON_ACTION_DISCARD)

    def test_normal_textured_foreground_passes(self):
        y, x = np.indices((80, 40))
        gray = np.where((x // 2 + y // 2) % 2 == 0, 60, 180).astype(np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        mask = np.ones((80, 40), dtype=np.uint8)

        decision = evaluate_person_visual_quality(image, mask)

        self.assertFalse(decision.reject)
        self.assertEqual(decision.cap_level, QUALITY_LEVEL_ENROLL)
        self.assertEqual(decision.quality_reasons, [])
