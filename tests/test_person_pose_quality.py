import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
if "utils.storage.oss_util" not in sys.modules:
    oss_util_module = types.ModuleType("utils.storage.oss_util")
    oss_util_module.AsyncOSSClient = object
    sys.modules["utils.storage.oss_util"] = oss_util_module

from tests.person_pose_test_support import (
    IMAGE_SHAPE,
    PERSON_RECT,
    full_body_keypoints,
    pose_result,
    segment_result,
)
from core.person_keyframe_processor import _pose_keypoints_to_crop
from utils.keyframe.detection_helpers import (
    match_person_segments_to_poses,
    parse_person_poses_result,
)
from utils.quality.person.evaluator import evaluate_person_quality
from utils.quality.person.pose import (
    calculate_person_pose_metrics,
)
from utils.keyframe.person_crop import PersonCropResult
from utils.quality.person import pose as person_quality_pose
from utils.quality.quality_reasons import PoseQualityReason, QualityReason
from utils.records.recognition_record_payloads import (
    build_person_detection_record,
    build_person_recognition_record,
)
from utils.reid.person_body import (
    BODY_FULL,
    BODY_HALF_LOWER,
    BODY_HALF_UPPER,
    BODY_PARTIAL,
    BODY_THREE_QUARTER_LOWER,
    BODY_THREE_QUARTER_UPPER,
)


class PersonPoseUnitTests(unittest.TestCase):
    def test_pose_score_weights_preserve_ten_point_maximum(self):
        metrics = calculate_person_pose_metrics(
            full_body_keypoints(), PERSON_RECT, IMAGE_SHAPE
        )
        self.assertAlmostEqual(metrics.pose_score, 10.0)

        pose_config = algorithm_config.person.quality.pose.model_copy(
            update={"visible_score_weight": 4.0}
        )
        alternate = SimpleNamespace(
            person=SimpleNamespace(quality=algorithm_config.person.quality.model_copy(update={"pose": pose_config}))
        )
        with patch.object(person_quality_pose, "algorithm_config", alternate):
            adjusted = calculate_person_pose_metrics(
                full_body_keypoints(), PERSON_RECT, IMAGE_SHAPE
            )
        self.assertAlmostEqual(adjusted.pose_score, 9.0)

    def test_pose_parser_and_deterministic_iou_matching(self):
        detections = np.array(
            [
                [52, 10, 72, 50, 0.8, 0],
                [2, 10, 22, 50, 0.9, 0],
                [80, 10, 100, 50, 0.7, 2],
            ],
            dtype=np.float32,
        )
        keypoints = np.zeros((3, 17, 3), dtype=np.float32)
        keypoints[0] = full_body_keypoints({"x": 52, "y": 10, "w": 20, "h": 40})
        keypoints[1] = full_body_keypoints({"x": 2, "y": 10, "w": 20, "h": 40})
        poses = parse_person_poses_result(
            detections,
            keypoints,
            np.array([3], dtype=np.int32),
            (100, 120, 3),
        )
        segments = [
            segment_result(rect={"x": 0, "y": 10, "w": 20, "h": 40}),
            segment_result(rect={"x": 50, "y": 10, "w": 20, "h": 40}),
        ]

        matched = match_person_segments_to_poses(segments, poses)

        self.assertEqual(len(poses), 2)
        self.assertEqual(matched[0]["pose"]["rect"]["x"], 2)
        self.assertEqual(matched[1]["pose"]["rect"]["x"], 52)
        self.assertTrue(all(item["poseMatchIou"] >= 0.3 for item in matched))

        threshold_match = match_person_segments_to_poses(
            [segment_result(rect={"x": 0, "y": 0, "w": 20, "h": 40})],
            [
                pose_result(
                    rect={"x": 10, "y": 0, "w": 20, "h": 40},
                    keypoints=full_body_keypoints(
                        {"x": 0, "y": 0, "w": 20, "h": 40}
                    ),
                )
            ],
        )
        below_threshold = match_person_segments_to_poses(
            [segment_result(rect={"x": 0, "y": 0, "w": 20, "h": 40})],
            [
                pose_result(
                    rect={"x": 11, "y": 0, "w": 20, "h": 40},
                    keypoints=full_body_keypoints(
                        {"x": 0, "y": 0, "w": 20, "h": 40}
                    ),
                )
            ],
        )
        self.assertIsNotNone(threshold_match[0]["pose"])
        self.assertIsNone(below_threshold[0]["pose"])

    def test_mask_first_matching_corrects_overlapping_pose_order(self):
        shared_rect = {"x": 10, "y": 10, "w": 100, "h": 140}
        left_mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
        right_mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
        left_mask[10:150, 10:60] = 1
        right_mask[10:150, 60:110] = 1
        segments = [
            {**segment_result(rect=shared_rect), "mask": left_mask},
            {**segment_result(rect=shared_rect), "mask": right_mask},
        ]
        left_points = full_body_keypoints({"x": 15, "y": 10, "w": 40, "h": 140})
        right_points = full_body_keypoints({"x": 65, "y": 10, "w": 40, "h": 140})
        poses = [
            {**pose_result(rect=shared_rect, keypoints=right_points), "tag": "right"},
            {**pose_result(rect=shared_rect, keypoints=left_points), "tag": "left"},
        ]

        matched = match_person_segments_to_poses(segments, poses)

        self.assertEqual(matched[0]["pose"]["tag"], "left")
        self.assertEqual(matched[1]["pose"]["tag"], "right")
        self.assertGreaterEqual(matched[0]["poseMaskAgreement"], 0.60)
        self.assertGreaterEqual(matched[1]["poseMaskAgreement"], 0.60)
        self.assertGreater(matched[0]["poseMatchScore"], 0.0)

    def test_invalid_mask_leaves_pose_unmatched(self):
        segment = {**segment_result(), "mask": None}

        matched = match_person_segments_to_poses([segment], [pose_result()])

        self.assertIsNone(matched[0]["pose"])
        self.assertEqual(matched[0]["poseMatchScore"], 0.0)

    def test_pose_keypoints_translate_into_padded_crop_coordinates(self):
        points = full_body_keypoints()
        crop_result = PersonCropResult(
            image=np.zeros((300, 100, 3), dtype=np.uint8),
            mask=None,
            requested_xyxy=(-10, -20, 90, 280),
            source_xyxy=(0, 0, 90, 200),
            content_xyxy=(10, 20, 100, 220),
            padding=(10, 20, 0, 80),
            padding_ratio=0.0,
            source_to_crop=np.asarray(
                [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]],
                dtype=np.float64,
            ),
        )

        translated = _pose_keypoints_to_crop(points, crop_result)

        np.testing.assert_allclose(translated[:, 0], points[:, 0] + 10)
        np.testing.assert_allclose(translated[:, 1], points[:, 1] + 20)
        np.testing.assert_allclose(translated[:, 2], points[:, 2])

    def test_visibility_boundary_and_body_completeness(self):
        metrics = calculate_person_pose_metrics(
            full_body_keypoints(score=algorithm_config.person.quality.pose.visible_threshold),
            PERSON_RECT,
            IMAGE_SHAPE,
        )

        self.assertEqual(metrics.visible_keypoint_count, 17)
        self.assertEqual(metrics.body_completeness, "full_body")
        self.assertGreaterEqual(metrics.pose_score, 9.9)

    def test_visibility_minimum_partial_drop_and_structured_upper_half_gallery(self):
        five_points = full_body_keypoints(score=0.49)
        visible_indices = [5, 6, 11, 12, 13, 14]
        five_points[visible_indices[:5], 2] = 0.5
        six_points = five_points.copy()
        six_points[visible_indices[5], 2] = 0.5
        five_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=five_points)],
        )[0]
        five_segment["poseInferenceFailed"] = False
        six_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=six_points)],
        )[0]
        six_segment["poseInferenceFailed"] = False

        half_points = full_body_keypoints()
        half_points[13:, 2] = 0.0
        twelve_half_points = half_points.copy()
        twelve_half_points[4, 2] = 0.0
        eleven_half_points = half_points.copy()
        eleven_half_points[[3, 4], 2] = 0.0
        ten_half_points = half_points.copy()
        ten_half_points[[2, 3, 4], 2] = 0.0
        single_torso_half_points = half_points.copy()
        single_torso_half_points[6, 2] = 0.0
        single_torso_half_points[15, 2] = 0.9
        half_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=half_points)],
        )[0]
        half_segment["poseInferenceFailed"] = False
        twelve_half_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=twelve_half_points)],
        )[0]
        twelve_half_segment["poseInferenceFailed"] = False
        eleven_half_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=eleven_half_points)],
        )[0]
        eleven_half_segment["poseInferenceFailed"] = False
        ten_half_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=ten_half_points)],
        )[0]
        ten_half_segment["poseInferenceFailed"] = False
        single_torso_half_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=single_torso_half_points)],
        )[0]
        single_torso_half_segment["poseInferenceFailed"] = False

        five = evaluate_person_quality(five_segment, [five_segment], IMAGE_SHAPE)
        six = evaluate_person_quality(six_segment, [six_segment], IMAGE_SHAPE)
        half = evaluate_person_quality(half_segment, [half_segment], IMAGE_SHAPE)
        twelve_half = evaluate_person_quality(
            twelve_half_segment, [twelve_half_segment], IMAGE_SHAPE
        )
        eleven_half = evaluate_person_quality(
            eleven_half_segment, [eleven_half_segment], IMAGE_SHAPE
        )
        ten_half = evaluate_person_quality(
            ten_half_segment, [ten_half_segment], IMAGE_SHAPE
        )
        single_torso_half = evaluate_person_quality(
            single_torso_half_segment, [single_torso_half_segment], IMAGE_SHAPE
        )

        self.assertEqual(five.person_quality_level, 0)
        self.assertIn(
            PoseQualityReason.POSE_VISIBLE_KEYPOINTS_TOO_FEW,
            five.quality_reasons,
        )
        self.assertEqual(six.person_quality_level, 0)
        self.assertNotIn(
            PoseQualityReason.POSE_VISIBLE_KEYPOINTS_TOO_FEW,
            six.quality_reasons,
        )
        self.assertIn(
            PoseQualityReason.POSE_BODY_TYPE_NOT_SEARCHABLE,
            six.quality_reasons,
        )
        self.assertEqual(half.person_quality_level, 2)
        self.assertTrue(half.is_half_body)
        self.assertTrue(half.half_body_pose_eligible)
        self.assertEqual(half.torso_keypoint_count, 4)
        self.assertEqual(half.shoulder_hip_chain_count, 2)
        self.assertEqual(half.body_completeness, BODY_HALF_UPPER)
        self.assertEqual(twelve_half.visible_keypoint_count, 12)
        self.assertEqual(twelve_half.person_quality_level, 2)
        self.assertTrue(twelve_half.half_body_pose_eligible)
        self.assertEqual(eleven_half.visible_keypoint_count, 11)
        self.assertEqual(eleven_half.person_quality_level, 2)
        self.assertTrue(eleven_half.half_body_pose_eligible)
        self.assertEqual(ten_half.visible_keypoint_count, 10)
        self.assertEqual(ten_half.person_quality_level, 1)
        self.assertFalse(ten_half.half_body_pose_eligible)
        self.assertEqual(single_torso_half.visible_keypoint_count, 13)
        self.assertEqual(single_torso_half.body_completeness, BODY_HALF_UPPER)
        self.assertEqual(single_torso_half.person_quality_level, 1)
        self.assertEqual(single_torso_half.torso_keypoint_count, 3)
        self.assertEqual(single_torso_half.shoulder_hip_chain_count, 1)
        self.assertFalse(single_torso_half.half_body_pose_eligible)

    def test_detailed_body_completeness_uses_same_side_continuous_chains(self):
        full = full_body_keypoints()

        three_quarter_upper = full.copy()
        three_quarter_upper[[15, 16], 2] = 0.0

        three_quarter_lower = full.copy()
        three_quarter_lower[:5, 2] = 0.0

        half_upper = full.copy()
        half_upper[13:, 2] = 0.0

        half_lower = full.copy()
        half_lower[:11, 2] = 0.0

        partial = full.copy()
        partial[11:, 2] = 0.0

        cross_side = full.copy()
        cross_side[[6, 13, 15], 2] = 0.0

        cases = [
            (full, BODY_FULL),
            (three_quarter_upper, BODY_THREE_QUARTER_UPPER),
            (three_quarter_lower, BODY_THREE_QUARTER_LOWER),
            (half_upper, BODY_HALF_UPPER),
            (half_lower, BODY_HALF_LOWER),
            (partial, BODY_PARTIAL),
        ]
        for points, expected in cases:
            with self.subTest(expected=expected):
                metrics = calculate_person_pose_metrics(
                    points,
                    PERSON_RECT,
                    IMAGE_SHAPE,
                )
                self.assertEqual(metrics.body_completeness, expected)

        cross_metrics = calculate_person_pose_metrics(
            cross_side,
            PERSON_RECT,
            IMAGE_SHAPE,
        )
        self.assertNotEqual(cross_metrics.body_completeness, BODY_FULL)

    def test_lower_three_quarter_caps_and_lower_half_drops(self):
        lower_three_quarter = full_body_keypoints()
        lower_three_quarter[:5, 2] = 0.0
        lower_half = full_body_keypoints()
        lower_half[:11, 2] = 0.0

        decisions = []
        for points in (lower_three_quarter, lower_half):
            segment = match_person_segments_to_poses(
                [segment_result()],
                [pose_result(keypoints=points)],
            )[0]
            segment["poseInferenceFailed"] = False
            decisions.append(evaluate_person_quality(segment, [segment], IMAGE_SHAPE))

        self.assertEqual(decisions[0].body_completeness, BODY_THREE_QUARTER_LOWER)
        self.assertEqual(decisions[0].person_quality_level, 1)
        self.assertIn(PoseQualityReason.POSE_HEAD_INCOMPLETE, decisions[0].quality_reasons)
        self.assertEqual(decisions[1].body_completeness, BODY_HALF_LOWER)
        self.assertEqual(decisions[1].person_quality_level, 0)
        self.assertIn(
            PoseQualityReason.POSE_BODY_TYPE_NOT_SEARCHABLE,
            decisions[1].quality_reasons,
        )

    def test_invalid_pose_keypoint_shape_drops(self):
        invalid_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=np.zeros((16, 3), dtype=np.float32))],
        )[0]
        invalid_segment["poseInferenceFailed"] = False

        decision = evaluate_person_quality(
            invalid_segment,
            [invalid_segment],
            IMAGE_SHAPE,
        )

        self.assertEqual(decision.person_quality_level, 0)
        self.assertIn(PoseQualityReason.POSE_NOT_MATCHED, decision.quality_reasons)

    def test_quality_evaluation_without_pose_always_drops(self):
        segment = segment_result()

        decision = evaluate_person_quality(segment, [segment], IMAGE_SHAPE)

        self.assertEqual(decision.person_quality_level, 0)
        self.assertIn(PoseQualityReason.POSE_NOT_MATCHED, decision.quality_reasons)

    def test_combined_score_preserves_mask_recognition_and_enroll_boundaries(self):
        recognition_segment = match_person_segments_to_poses(
            [segment_result(mask_pixel_count=6000)],
            [pose_result()],
        )[0]
        recognition_segment["poseInferenceFailed"] = False
        enroll_segment = match_person_segments_to_poses(
            [segment_result(mask_pixel_count=8400)],
            [pose_result()],
        )[0]
        enroll_segment["poseInferenceFailed"] = False

        recognition = evaluate_person_quality(
            recognition_segment,
            [recognition_segment],
            IMAGE_SHAPE,
        )
        enroll = evaluate_person_quality(
            enroll_segment,
            [enroll_segment],
            IMAGE_SHAPE,
        )

        self.assertAlmostEqual(recognition.person_quality_score, 8.5, places=4)
        self.assertEqual(recognition.person_quality_level, 1)
        self.assertIn(QualityReason.MASK_AREA_BELOW_ENROLL, recognition.quality_reasons)
        self.assertAlmostEqual(enroll.person_quality_score, 9.1, places=4)
        self.assertEqual(enroll.person_quality_level, 2)

    def test_pose_bbox_and_mask_mismatch_prevent_binding(self):
        mismatched_points = full_body_keypoints()
        mismatched_points[:, 0] += 45
        bbox_segment = match_person_segments_to_poses(
            [segment_result()],
            [pose_result(keypoints=mismatched_points)],
        )[0]
        bbox_segment["poseInferenceFailed"] = False

        mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
        mask[30:130, 30:50] = 1
        mask_segment = match_person_segments_to_poses(
            [{**segment_result(), "mask": mask}],
            [pose_result()],
        )[0]
        mask_segment["poseInferenceFailed"] = False

        bbox_decision = evaluate_person_quality(
            bbox_segment,
            [bbox_segment],
            IMAGE_SHAPE,
        )
        mask_decision = evaluate_person_quality(
            mask_segment,
            [mask_segment],
            IMAGE_SHAPE,
        )

        self.assertEqual(bbox_decision.person_quality_level, 0)
        self.assertIn(PoseQualityReason.POSE_NOT_MATCHED, bbox_decision.quality_reasons)
        self.assertEqual(mask_decision.person_quality_level, 0)
        self.assertIn(PoseQualityReason.POSE_NOT_MATCHED, mask_decision.quality_reasons)

    def test_single_image_or_mask_edge_is_diagnostic_without_downgrade(self):
        image_edge_rect = {"x": 0, "y": 20, "w": 50, "h": 120}
        image_edge_segment = match_person_segments_to_poses(
            [segment_result(rect=image_edge_rect)],
            [pose_result(rect=image_edge_rect)],
        )[0]
        image_edge_segment["poseInferenceFailed"] = False

        mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
        mask[30:130, 30:75] = 1
        mask_edge_segment = match_person_segments_to_poses(
            [{**segment_result(), "mask": mask}],
            [pose_result()],
        )[0]
        mask_edge_segment["poseInferenceFailed"] = False

        image_edge = evaluate_person_quality(
            image_edge_segment,
            [image_edge_segment],
            IMAGE_SHAPE,
        )
        mask_edge = evaluate_person_quality(
            mask_edge_segment,
            [mask_edge_segment],
            IMAGE_SHAPE,
        )

        self.assertEqual(image_edge.person_quality_level, 2)
        self.assertEqual(image_edge.crop_score, 10.0)
        self.assertFalse(image_edge.is_truncated)
        self.assertIn(QualityReason.BBOX_NEAR_IMAGE_EDGE, image_edge.quality_reasons)
        self.assertEqual(mask_edge.person_quality_level, 2)
        self.assertFalse(mask_edge.is_truncated)
        self.assertIn(
            QualityReason.MASK_TOUCHING_BBOX_EDGE,
            mask_edge.quality_reasons,
        )

    def test_one_to_four_near_image_edges_are_diagnostic_only(self):
        cases = [
            ({"x": 7, "y": 20, "w": 40, "h": 100}, 1),
            ({"x": 7, "y": 7, "w": 40, "h": 100}, 2),
            ({"x": 7, "y": 7, "w": 106, "h": 100}, 3),
            ({"x": 7, "y": 7, "w": 106, "h": 146}, 4),
        ]

        for rect, expected_count in cases:
            with self.subTest(expected_count=expected_count):
                segment = match_person_segments_to_poses(
                    [segment_result(rect=rect)],
                    [pose_result(rect=rect)],
                )[0]
                segment["poseInferenceFailed"] = False

                decision = evaluate_person_quality(segment, [segment], IMAGE_SHAPE)

                self.assertEqual(decision.person_quality_level, 2)
                self.assertEqual(decision.crop_score, 10.0)
                self.assertFalse(decision.is_truncated)
                self.assertEqual(decision.near_image_edge_count, expected_count)
                self.assertIn(
                    QualityReason.BBOX_NEAR_IMAGE_EDGE,
                    decision.quality_reasons,
                )

    def test_confirmed_bbox_overflow_threshold_boundaries(self):
        rect = {"x": 0, "y": 20, "w": 40, "h": 120}

        def decision_for_ratio(ratio):
            raw_width = rect["w"] / (1.0 - ratio)
            segment = segment_result(rect=rect)
            segment["xyxy"] = [
                rect["w"] - raw_width,
                rect["y"],
                rect["x"] + rect["w"],
                rect["y"] + rect["h"],
            ]
            segment["pose"] = pose_result(rect=rect)
            segment["poseInferenceFailed"] = False
            return evaluate_person_quality(segment, [segment], IMAGE_SHAPE)

        at_cap = decision_for_ratio(0.20)
        above_cap = decision_for_ratio(0.2001)
        at_reject = decision_for_ratio(0.50)
        above_reject = decision_for_ratio(0.5001)

        self.assertAlmostEqual(at_cap.bbox_overflow_ratio, 0.20)
        self.assertEqual(at_cap.person_quality_level, 2)
        self.assertTrue(at_cap.is_truncated)
        self.assertEqual(above_cap.person_quality_level, 1)
        self.assertIn(
            QualityReason.BBOX_CONTENT_OVERFLOW_HIGH,
            above_cap.quality_reasons,
        )
        self.assertEqual(at_reject.person_quality_level, 1)
        self.assertEqual(above_reject.person_quality_level, 0)
        self.assertIn(
            QualityReason.BBOX_CONTENT_OVERFLOW_TOO_HIGH,
            above_reject.quality_reasons,
        )

    def test_three_confirmed_truncated_edges_drop(self):
        rect = {"x": 0, "y": 0, "w": 120, "h": 140}
        segment = segment_result(rect=rect)
        segment["xyxy"] = [-5, -5, 125, 140]
        segment["pose"] = pose_result(rect=rect)
        segment["poseInferenceFailed"] = False

        decision = evaluate_person_quality(segment, [segment], IMAGE_SHAPE)

        self.assertEqual(decision.truncated_edge_count, 3)
        self.assertTrue(decision.is_truncated)
        self.assertEqual(decision.person_quality_level, 0)
        self.assertIn(
            QualityReason.PERSON_TRUNCATED_SEVERE,
            decision.quality_reasons,
        )

    def test_bottom_frame_truncation_uses_pose_without_raw_overflow(self):
        rect = {"x": 40, "y": 40, "w": 40, "h": 120}
        points = full_body_keypoints(rect)
        points[13:, 2] = 0.0
        segment = segment_result(rect=rect)
        segment["pose"] = pose_result(rect=rect, keypoints=points)
        segment["poseInferenceFailed"] = False

        decision = evaluate_person_quality(segment, [segment], IMAGE_SHAPE)

        self.assertEqual(decision.body_completeness, BODY_HALF_UPPER)
        self.assertEqual(decision.visible_keypoint_count, 13)
        self.assertEqual(decision.person_quality_level, 2)
        self.assertTrue(decision.is_truncated)
        self.assertEqual(decision.truncated_edge_count, 1)

    def test_payload_contract_keeps_raw_points_only_on_detection(self):
        points = full_body_keypoints()
        diagnostics = {
            "pose_score": 9.5,
            "visible_keypoint_count": 17,
            "torso_keypoint_count": 4,
            "shoulder_hip_chain_count": 2,
            "half_body_pose_eligible": False,
            "body_completeness": "full_body",
            "is_half_body": False,
            "is_truncated": False,
            "has_second_person": False,
        }
        detection = build_person_detection_record(
            keyframe_id="uid",
            picture_vector_id=1,
            crop_url="crop.jpg",
            dev_sn="CAM001",
            timestamp=1,
            detect_time=2,
            people_rect=PERSON_RECT,
            confidence=0.9,
            mask_pixel_count=12000,
            status=1,
            ai_id="ai",
            info="",
            person_keypoint=points,
            **diagnostics,
        )
        recognition = build_person_recognition_record(
            dev_sn="CAM001",
            keyframe_id="uid",
            picture_vector_id=1,
            image_url="image.jpg",
            face_id="0",
            reid="reid",
            score=0.9,
            detect_time=2,
            timestamp=1,
            people_rect=PERSON_RECT,
            info="",
            reid_status=2,
            ai_id="ai",
            **diagnostics,
        )

        self.assertEqual(detection["personKeypoint"]["format"], "coco17")
        self.assertEqual(len(detection["personKeypoint"]["points"]), 17)
        self.assertNotIn("personKeypoint", recognition)
        self.assertEqual(recognition["bodyCompleteness"], "full_body")
        for record in (detection, recognition):
            self.assertEqual(record["torsoKeypointCount"], 4)
            self.assertEqual(record["shoulderHipChainCount"], 2)
            self.assertIs(record["halfBodyPoseEligible"], False)
        json.dumps(detection, allow_nan=False)
        json.dumps(recognition, allow_nan=False)
