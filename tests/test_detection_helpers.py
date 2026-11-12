import unittest
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

import utils.keyframe.detection_helpers as detection_helpers
from core.algorithm_config import algorithm_config
from utils.keyframe import keyframe_flow_helpers

from utils.keyframe.detection_helpers import (
    clip_xyxy_to_rect,
    expand_square_rect,
    parse_keyframe_context,
    parse_person_segments_result,
)
from utils.keyframe.person_crop import PersonCropResult, build_person_crop
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
    REID_SOURCE_BUFFER,
    REID_SOURCE_MILVUS,
    extract_quality_score,
    face_quality_level_from_score,
    format_reid_source,
    merge_search_results,
    normalize_numpy_vector,
    person_quality_level_from_score,
    person_quality_score_from_mask,
    quality_level_from_score,
)

class DetectionHelperTests(unittest.TestCase):
    def setUp(self):
        # 本组 URL 用例验证 OSS 旧格式，不能由开发机选择的 MinIO 后端改变。
        config = ModuleType("core.config")
        config.settings = SimpleNamespace(STORAGE_BACKEND="oss")
        storage = patch.dict(sys.modules, {"core.config": config})
        storage.start()
        self.addCleanup(storage.stop)

    def test_face_quality_level_uses_selected_config_not_imported_snapshot(self):
        quality = algorithm_config.face.quality.model_copy(
            update={"recognition_min_score": 10.0, "enroll_min_score": 20.0}
        )
        face = algorithm_config.face.model_copy(update={"quality": quality})
        selected = algorithm_config.model_copy(update={"face": face})
        with patch.object(keyframe_flow_helpers, "algorithm_config", selected):
            self.assertEqual(
                keyframe_flow_helpers.face_quality_level_from_score(15.0),
                QUALITY_LEVEL_RECOGNITION,
            )

    def test_pose_mask_sample_radius_uses_configured_minimum_and_ratio(self):
        visible_points = np.asarray([[10.0, 10.0, 0.9]], dtype=np.float32)
        cases = (
            (20, 2, 0.01, 2),
            (100, 2, 0.04, 4),
        )
        for size, minimum, ratio, expected_radius in cases:
            with self.subTest(size=size, minimum=minimum, ratio=ratio):
                mask = np.zeros((20, 20), dtype=np.uint8)
                mask[10, 10 + expected_radius] = 1
                assignment = algorithm_config.person.assignment.model_copy(update={
                    "pose_mask_sample_radius_min_px": minimum,
                    "pose_mask_sample_radius_ratio": ratio,
                })
                person = algorithm_config.person.model_copy(update={"assignment": assignment})
                selected = algorithm_config.model_copy(update={"person": person})
                with patch.object(detection_helpers, "algorithm_config", selected):
                    self.assertEqual(
                        detection_helpers.pose_mask_agreement(
                            visible_points,
                            mask,
                            {"w": size, "h": size},
                        ),
                        1.0,
                    )

    def test_build_person_crop_records_source_to_crop_for_normal_and_padded_cases(
        self,
    ):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cases = [
            ({"x": 40, "y": 30, "w": 20, "h": 30}, (-39.0, -28.0)),
            ({"x": 0, "y": 0, "w": 10, "h": 20}, (1.0, 1.0)),
        ]

        for rect, expected_translation in cases:
            with self.subTest(rect=rect):
                result = build_person_crop(image, rect)
                self.assertIsNotNone(result)
                self.assertEqual(result.source_to_crop.shape, (2, 3))
                self.assertEqual(result.source_to_crop.dtype, np.float64)
                self.assertEqual(
                    tuple(result.source_to_crop[:, 2]),
                    expected_translation,
                )
                source_x1, source_y1, _, _ = result.source_xyxy
                content_x1, content_y1, _, _ = result.content_xyxy
                mapped = result.source_to_crop @ np.asarray(
                    [source_x1, source_y1, 1.0], dtype=np.float64
                )
                np.testing.assert_array_equal(
                    mapped,
                    np.asarray([content_x1, content_y1], dtype=np.float64),
                )

    def test_person_crop_result_source_to_crop_defaults_to_none(self):
        result = PersonCropResult(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            mask=None,
            requested_xyxy=(0, 0, 1, 1),
            source_xyxy=(0, 0, 1, 1),
            content_xyxy=(0, 0, 1, 1),
            padding=(0, 0, 0, 0),
            padding_ratio=0.0,
        )

        self.assertIsNone(result.source_to_crop)

    def test_parse_pano_keyframe_context(self):
        self.assertTrue(
            hasattr(detection_helpers, "parse_pano_keyframe_context"),
            "pano keyframe parser must be available",
        )
        parse_pano_keyframe_context = (
            detection_helpers.parse_pano_keyframe_context
        )
        uid = "55555555-5555-4555-8555-555555555555"

        ctx = parse_pano_keyframe_context(
            {
                "sn": "CAM001",
                "timestamp": "1710000000000",
                "keyframe": "CAM001/picture/main.jpg",
                "panoKeyframeUrl": (
                    "https://oss.example.com/CAM001/picture/pano.jpg?x=1"
                ),
                "uId": uid,
            }
        )

        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.sn, "CAM001")
        self.assertEqual(ctx.timestamp, 1710000000000)
        self.assertEqual(ctx.object_name, "CAM001/picture/pano.jpg")
        self.assertEqual(ctx.keyframe_id, uid)
        self.assertIs(ctx.is_pano_keyframe, True)
        self.assertIsNone(parse_pano_keyframe_context({}))
        self.assertIsNone(
            parse_pano_keyframe_context({"panoKeyframeUrl": "  "})
        )

    def test_pano_keyframe_context_inherits_main_context_metadata(self):
        uid = "99999999-9999-4999-8999-999999999999"

        ctx = detection_helpers.parse_pano_keyframe_context(
            {
                "keyframe": "CAM001/picture/main.jpg",
                "panoKeyframeUrl": "CAM999/picture/pano.jpg",
                "uId": uid,
            },
            now_ms=1710000000123,
        )

        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.sn, "CAM001")
        self.assertEqual(ctx.timestamp, 1710000000123)
        self.assertEqual(ctx.object_name, "CAM999/picture/pano.jpg")
        self.assertEqual(ctx.keyframe_id, uid)

    def test_pano_short_name_uses_main_inferred_sn(self):
        ctx = detection_helpers.parse_pano_keyframe_context(
            {
                "keyframe": "CAM001/picture/main.jpg",
                "panoKeyframeUrl": "pano",
                "uId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            },
            now_ms=1710000000123,
        )

        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.sn, "CAM001")
        self.assertEqual(ctx.object_name, "CAM001/picture/pano.jpg")

    def test_parse_keyframe_context_from_short_name(self):
        uid = "11111111-1111-4111-8111-111111111111"
        ctx = parse_keyframe_context(
            {
                "sn": "CAM001",
                "timestamp": "1710000000000",
                "keyframe": "abc123",
                "uId": uid,
            }
        )

        self.assertEqual(ctx.sn, "CAM001")
        self.assertEqual(ctx.timestamp, 1710000000000)
        self.assertEqual(ctx.object_name, "CAM001/picture/abc123.jpg")
        self.assertEqual(ctx.keyframe_id, uid)
        self.assertIs(ctx.is_pano_keyframe, False)

    def test_parse_keyframe_context_preserves_uid(self):
        uid = "22222222-2222-4222-8222-222222222222"
        ctx = parse_keyframe_context(
            {
                "sn": "CAM001",
                "timestamp": "1710000000000",
                "keyframe": "abc123",
                "uId": uid,
            }
        )

        self.assertEqual(ctx.keyframe_id, uid)

    def test_parse_keyframe_context_preserves_camel_case_aliases(self):
        uid = "44444444-4444-4444-8444-444444444444"
        cases = (
            ({"sn": "CAM001", "keyFrame": "frame-a"}, "CAM001/picture/frame-a.jpg"),
            (
                {"keyFrameUrl": "https://oss.example.com/CAM002/picture/frame-b.jpg"},
                "CAM002/picture/frame-b.jpg",
            ),
        )

        for payload, expected_object_name in cases:
            with self.subTest(payload=payload):
                ctx = parse_keyframe_context({**payload, "uId": uid}, now_ms=123)
                self.assertEqual(ctx.object_name, expected_object_name)
                self.assertEqual(ctx.keyframe_id, uid)

    def test_parse_keyframe_context_rejects_legacy_keyframe_id_without_uid(self):
        with self.assertRaises(ValueError):
            parse_keyframe_context(
                {
                    "sn": "CAM001",
                    "timestamp": "1710000000000",
                    "keyframe": "abc123",
                    "keyframeId": "987654321",
                }
            )

    def test_parse_keyframe_context_from_url_and_inferred_sn(self):
        uid = "33333333-3333-4333-8333-333333333333"
        ctx = parse_keyframe_context(
            {
                "keyframeUrl": "https://oss.example.com/CAM002/picture/frame.jpg?x=1",
                "uId": uid,
            },
            now_ms=123,
        )

        self.assertEqual(ctx.sn, "CAM002")
        self.assertEqual(ctx.timestamp, 123)
        self.assertEqual(ctx.object_name, "CAM002/picture/frame.jpg")
        self.assertEqual(ctx.keyframe_id, uid)

    def test_clip_xyxy_to_rect_clamps_to_image(self):
        rect = clip_xyxy_to_rect([-10, 5, 110, 50], (40, 100, 3))

        self.assertEqual(rect, {"x": 0, "y": 5, "w": 100, "h": 35})

    def test_expand_square_rect_stays_inside_image(self):
        expanded = expand_square_rect({"x": 45, "y": 10, "w": 10, "h": 20}, (40, 60, 3))

        self.assertEqual(expanded, (30, 0, 60, 40))

    def test_build_person_crop_adds_margin_without_letterboxing(self):
        image = np.zeros((200, 120, 3), dtype=np.uint8)
        result = build_person_crop(
            image,
            {"x": 40, "y": 40, "w": 40, "h": 100},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.image.shape, (110, 44, 3))
        self.assertEqual(result.requested_xyxy, (38, 35, 82, 145))
        self.assertEqual(result.source_xyxy, (38, 35, 82, 145))
        self.assertEqual(result.content_xyxy, (0, 0, 44, 110))
        self.assertEqual(result.padding, (0, 0, 0, 0))
        self.assertEqual(result.padding_ratio, 0.0)
        self.assertEqual(result.letterbox_padding_ratio, 0.0)
        self.assertEqual(result.source_overflow_ratio, 0.0)

    def test_two_to_one_person_keeps_its_natural_window(self):
        image = np.zeros((200, 160, 3), dtype=np.uint8)
        result = build_person_crop(
            image,
            {"x": 60, "y": 50, "w": 40, "h": 80},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.image.shape, (88, 44, 3))
        self.assertEqual(result.requested_xyxy, (58, 46, 102, 134))
        self.assertEqual(result.source_xyxy, (58, 46, 102, 134))
        self.assertEqual(result.content_xyxy, (0, 0, 44, 88))
        self.assertEqual(result.padding, (0, 0, 0, 0))
        self.assertEqual(result.padding_ratio, 0.0)
        self.assertEqual(result.letterbox_padding_ratio, 0.0)
        self.assertEqual(result.source_overflow_ratio, 0.0)

    def test_build_person_crop_uses_bbox_margin_for_varied_aspect_ratios(self):
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        cases = [
            (
                {"x": 80, "y": 80, "w": 40, "h": 20},
                (22, 44, 3),
                (78, 79, 122, 101),
            ),
            (
                {"x": 80, "y": 60, "w": 10, "h": 31},
                (35, 12, 3),
                (79, 58, 91, 93),
            ),
            (
                {"x": 80, "y": 60, "w": 11, "h": 32},
                (36, 13, 3),
                (79, 58, 92, 94),
            ),
        ]

        for rect, expected_shape, expected_requested in cases:
            with self.subTest(rect=rect):
                result = build_person_crop(image, rect)
                self.assertIsNotNone(result)
                self.assertEqual(result.image.shape, expected_shape)
                self.assertEqual(result.requested_xyxy, expected_requested)
                self.assertEqual(result.source_xyxy, expected_requested)
                self.assertEqual(
                    result.content_xyxy,
                    (0, 0, expected_shape[1], expected_shape[0]),
                )
                self.assertEqual(result.padding, (0, 0, 0, 0))
                self.assertEqual(result.letterbox_padding_ratio, 0.0)

    def test_build_person_crop_pads_image_and_mask_at_top_left(self):
        image = np.full((60, 40, 3), (10, 20, 30), dtype=np.uint8)
        mask = np.zeros((60, 40), dtype=np.uint8)
        mask[0:20, 0:10] = 1
        result = build_person_crop(
            image,
            {"x": 0, "y": 0, "w": 10, "h": 20},
            mask,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.image.shape, (22, 12, 3))
        self.assertEqual(result.requested_xyxy, (-1, -1, 11, 21))
        self.assertEqual(result.source_xyxy, (0, 0, 11, 21))
        self.assertEqual(result.content_xyxy, (1, 1, 12, 22))
        self.assertEqual(result.padding, (1, 1, 0, 0))
        self.assertAlmostEqual(result.padding_ratio, 1.0 / 8.0)
        self.assertEqual(result.letterbox_padding_ratio, 0.0)
        self.assertAlmostEqual(result.source_overflow_ratio, 1.0 / 8.0)
        np.testing.assert_array_equal(result.image[0, 0], algorithm_config.person.crop.border_bgr)
        np.testing.assert_array_equal(result.image[1, 1], (10, 20, 30))
        self.assertIsNotNone(result.mask)
        self.assertEqual(result.mask.dtype, mask.dtype)
        self.assertTrue(result.mask.flags.c_contiguous)
        self.assertEqual(int(np.count_nonzero(result.mask[:1, :])), 0)
        self.assertEqual(int(np.count_nonzero(result.mask[:, :1])), 0)
        self.assertEqual(int(np.count_nonzero(result.mask[1:21, 1:11])), 200)

    def test_build_person_crop_rejects_output_over_allocation_budget(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        rect = {"x": 5, "y": 5, "w": 10, "h": 10}

        for margin_ratio in (1e20, 1e100, 1e308):
            with self.subTest(margin_ratio=margin_ratio):
                self.assertIsNone(
                    build_person_crop(
                        image,
                        rect,
                        margin_ratio=margin_ratio,
                    )
                )

    def test_build_person_crop_pads_each_image_edge(self):
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        cases = [
            ({"x": 0, "y": 20, "w": 10, "h": 20}, 0),
            ({"x": 50, "y": 20, "w": 10, "h": 20}, 2),
            ({"x": 20, "y": 0, "w": 10, "h": 20}, 1),
            ({"x": 20, "y": 40, "w": 10, "h": 20}, 3),
        ]

        for rect, padding_index in cases:
            with self.subTest(rect=rect):
                result = build_person_crop(image, rect)
                self.assertIsNotNone(result)
                self.assertGreater(result.padding[padding_index], 0)
                self.assertEqual(result.image.shape, (22, 12, 3))
                self.assertEqual(result.letterbox_padding_ratio, 0.0)

    def test_build_person_crop_handles_invalid_inputs_and_invalid_mask(self):
        image = np.zeros((60, 40, 3), dtype=np.uint8)
        valid_rect = {"x": 1, "y": 1, "w": 10, "h": 10}
        invalid_images = [
            None,
            image.tolist(),
            np.zeros((60, 40), dtype=np.uint8),
            np.zeros((60, 40, 4), dtype=np.uint8),
            np.zeros((0, 40, 3), dtype=np.uint8),
        ]
        for invalid_image in invalid_images:
            with self.subTest(invalid_image_type=type(invalid_image)):
                self.assertIsNone(build_person_crop(invalid_image, valid_rect))

        invalid_rects = [
            {"x": 1, "y": 1, "w": 0, "h": 10},
            {"x": -1, "y": 1, "w": 10, "h": 10},
            {"x": 35, "y": 1, "w": 10, "h": 10},
            {"x": float("inf"), "y": 1, "w": 10, "h": 10},
            {"x": "invalid", "y": 1, "w": 10, "h": 10},
        ]
        for invalid_rect in invalid_rects:
            with self.subTest(invalid_rect=invalid_rect):
                self.assertIsNone(build_person_crop(image, invalid_rect))

        for invalid_margin in (float("nan"), float("inf"), -0.01, object()):
            with self.subTest(invalid_margin=invalid_margin):
                self.assertIsNone(
                    build_person_crop(
                        image,
                        valid_rect,
                        margin_ratio=invalid_margin,
                    )
                )

        converted = build_person_crop(
            image,
            {"x": "1", "y": "1", "w": "10", "h": "10"},
            margin_ratio="0.1",
        )
        self.assertIsNotNone(converted)
        self.assertEqual(converted.image.shape, (12, 12, 3))

        result = build_person_crop(
            image,
            {"x": 1, "y": 1, "w": 10, "h": 10},
            np.ones((10, 10), dtype=np.uint8),
        )
        self.assertIsNotNone(result)
        self.assertIsNone(result.mask)

    def test_parse_person_segments_filters_person_and_counts_mask_pixels(self):
        detections = np.array(
            [
                [1, 2, 5, 7, 0.9, 0],
                [10, 10, 20, 20, 0.8, 2],
            ],
            dtype=np.float32,
        )
        masks = np.zeros((2, 12, 30), dtype=np.uint8)
        masks[0, 2:5, 3:7] = 1
        masks[1, :, :] = 1

        persons = parse_person_segments_result(
            detections=detections,
            masks=masks,
            num_detections=np.array([2], dtype=np.int32),
            image_shape=(12, 30, 3),
        )

        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0]["rect"], {"x": 1, "y": 2, "w": 4, "h": 5})
        self.assertEqual(persons[0]["maskPixelCount"], 12)


class KeyframeFlowHelperTests(unittest.TestCase):
    def test_normalize_numpy_vector_handles_zero_and_nonzero_vectors(self):
        normalized = normalize_numpy_vector(np.array([[3.0, 4.0]], dtype=np.float32))
        self.assertTrue(np.allclose(normalized, np.array([0.6, 0.8], dtype=np.float32)))

        zero = np.zeros((1, 2), dtype=np.float32)
        self.assertTrue(np.array_equal(normalize_numpy_vector(zero), np.array([0.0, 0.0])))

    def test_extract_quality_score_accepts_scalar_and_array_payloads(self):
        self.assertEqual(extract_quality_score(6.5), 6.5)
        self.assertEqual(
            extract_quality_score({"quality_score": np.array([7.25], dtype=np.float32)}),
            7.25,
        )

        with self.assertRaises(ValueError):
            extract_quality_score({"quality_score": np.array([], dtype=np.float32)})

    def test_quality_helpers_preserve_separate_threshold_behavior(self):
        self.assertEqual(
            face_quality_level_from_score(algorithm_config.face.quality.recognition_min_score - 0.01),
            QUALITY_LEVEL_DROP,
        )
        self.assertEqual(
            face_quality_level_from_score(algorithm_config.face.quality.recognition_min_score),
            QUALITY_LEVEL_RECOGNITION,
        )
        self.assertEqual(
            face_quality_level_from_score(algorithm_config.face.quality.enroll_min_score),
            QUALITY_LEVEL_ENROLL,
        )
        self.assertEqual(
            person_quality_level_from_score(algorithm_config.person.quality.general.recognition_min_score - 0.01),
            QUALITY_LEVEL_DROP,
        )
        self.assertEqual(
            person_quality_level_from_score(algorithm_config.person.quality.general.recognition_min_score),
            QUALITY_LEVEL_RECOGNITION,
        )
        self.assertEqual(
            person_quality_level_from_score(algorithm_config.person.quality.general.enroll_min_score),
            QUALITY_LEVEL_ENROLL,
        )
        self.assertEqual(
            quality_level_from_score(
                6.0,
                recognition_min_score=4.0,
                enroll_min_score=6.0,
            ),
            QUALITY_LEVEL_ENROLL,
        )
        self.assertEqual(
            quality_level_from_score(
                6.0,
                recognition_min_score=6.5,
                enroll_min_score=8.0,
            ),
            QUALITY_LEVEL_DROP,
        )
        self.assertEqual(
            person_quality_score_from_mask(algorithm_config.person.quality.general.max_mask_pixels),
            10.0,
        )

    def test_merge_search_results_prefers_pending_only_on_higher_score(self):
        self.assertEqual(
            merge_search_results("milvus", 0.8, "milvus-hit", "pending", 0.9, "pending-hit"),
            ("pending", 0.9, "pending-hit"),
        )
        self.assertEqual(
            merge_search_results("milvus", 0.8, "milvus-hit", "pending", 0.8, "pending-hit"),
            ("milvus", 0.8, "milvus-hit"),
        )
        self.assertEqual(
            merge_search_results(None, 0.4, "milvus-hit", None, 0.3, "pending-hit"),
            (None, 0.4, "milvus-hit"),
        )

    def test_format_reid_source_labels_known_sources(self):
        self.assertEqual(format_reid_source(REID_SOURCE_MILVUS), "向量库")
        self.assertEqual(format_reid_source(REID_SOURCE_BUFFER), "缓存")
        self.assertEqual(format_reid_source("custom"), "custom")





if __name__ == "__main__":
    unittest.main()
