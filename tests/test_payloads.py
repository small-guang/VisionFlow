import json
import unittest

from utils.quality.quality_reasons import QualityReason
from utils.quality.face_orientation import FaceOrientation
from utils.quality.person_orientation import PersonOrientation
from utils.records.recognition_record_payloads import (
    build_face_detection_record,
    build_face_recognition_record,
    build_person_detection_record,
    build_person_recognition_record,
    build_vector_record,
)


class PayloadTests(unittest.TestCase):
    def test_sbs_vector_record_uses_sbs_contract(self):
        record = build_vector_record(
            vector_id=99,
            embedding=[0.1] * 2048,
            keyframe_id="uid",
            picture_vector_id=42,
            timestamp=171,
            keyframe_url="CAM001/picture/k.jpg",
            target_type="person_sbs",
            ai_id="ai-1",
            vector_db_name="alogSBS",
            is_pano_keyframe=True,
        )

        self.assertEqual(record["targetType"], "person_sbs")
        self.assertEqual(record["vectorDbName"], "alogSBS")
        self.assertEqual(record["vectorDim"], 2048)
        self.assertEqual(
            record["modelName"],
            "Person ReID Vector | FastReID SBS R50-IBN (DukeMTMC)",
        )
        self.assertIs(record["isPanoKeyframe"], True)

    def test_build_face_detection_record_includes_quality_fields(self):
        uid = "11111111-1111-4111-8111-111111111111"
        record = build_face_detection_record(
            keyframe_id=uid,
            picture_vector_id="456",
            crop_url="CAM001/face_crop/f.jpg",
            dev_sn="CAM001",
            timestamp="171",
            detect_time="172",
            face_keypoint=[[1, 2]],
            face_rect={"x": "1", "y": "2", "w": "3", "h": "4"},
            confidence="0.88",
            status=True,
            ai_id="ai-1",
            info="正常",
            face_quality_score="6.5",
            face_quality_level="2",
        )

        self.assertEqual(record["keyframeId"], uid)
        self.assertEqual(record["pictureVectorId"], 456)
        self.assertEqual(record["confidence"], 0.88)
        self.assertEqual(record["crop_url"], "CAM001/face_crop/f.jpg")
        self.assertNotIn("keyframeUrl", record)
        self.assertEqual(record["faceQualityScore"], 6.5)
        self.assertEqual(record["faceQualityLevel"], 2)
        self.assertIs(record["isPanoKeyframe"], False)

    def test_build_person_detection_record(self):
        uid = "22222222-2222-4222-8222-222222222222"
        record = build_person_detection_record(
            keyframe_id=uid,
            picture_vector_id="456",
            crop_url="CAM001/person_crop/p.jpg",
            dev_sn="CAM001",
            timestamp="171",
            detect_time="172",
            people_rect={"x": "1", "y": "2", "w": "3", "h": "4"},
            confidence="0.88",
            mask_pixel_count="5001",
            status=True,
            ai_id="ai-1",
            info="正常",
            person_quality_score="10",
            person_quality_level="2",
            person_quality_action="build_gallery",
            mask_score="10",
            crop_score="9.5",
            torso_keypoint_count="4",
            shoulder_hip_chain_count="2",
            half_body_pose_eligible=True,
            foreign_person_ratio="0.08",
            foreign_person_core_ratio="0.01",
            foreign_person_pixel_count="800",
            foreign_person_core_pixel_count="100",
            foreign_person_bbox_fallback_ratio="0.02",
            foreign_person_bbox_fallback_pixel_count="200",
            foreign_person_edge_dominant=True,
            foreign_person_evidence_source="mixed",
            padding_ratio="0.4",
            letterbox_padding_ratio="0.3",
            source_overflow_ratio="0.1",
            bbox_overflow_ratio="0.05",
            near_image_edge_count="2",
            truncated_edge_count="1",
            is_pano_keyframe=True,
            quality_reasons=[QualityReason.MASK_AREA_BELOW_ENROLL],
        )

        self.assertEqual(record["keyframeId"], uid)
        self.assertEqual(record["pictureVectorId"], 456)
        self.assertEqual(record["crop_url"], "CAM001/person_crop/p.jpg")
        self.assertNotIn("keyframeUrl", record)
        self.assertEqual(record["devSn"], "CAM001")
        self.assertEqual(record["timestamp"], 171)
        self.assertEqual(record["detectTime"], 172)
        self.assertEqual(record["peopleRect"], {"x": 1, "y": 2, "w": 3, "h": 4})
        self.assertEqual(record["confidence"], 0.88)
        self.assertEqual(record["maskPixelCount"], 5001)
        self.assertEqual(record["personQualityScore"], 10.0)
        self.assertEqual(record["personQualityLevel"], 2)
        self.assertEqual(record["personQualityAction"], "build_gallery")
        self.assertEqual(record["maskScore"], 10.0)
        self.assertEqual(record["cropScore"], 9.5)
        self.assertEqual(record["torsoKeypointCount"], 4)
        self.assertEqual(record["shoulderHipChainCount"], 2)
        self.assertIs(record["halfBodyPoseEligible"], True)
        self.assertEqual(record["foreignPersonRatio"], 0.08)
        self.assertEqual(record["foreignPersonCoreRatio"], 0.01)
        self.assertEqual(record["foreignPersonPixelCount"], 800)
        self.assertEqual(record["foreignPersonCorePixelCount"], 100)
        self.assertEqual(record["foreignPersonBBoxFallbackRatio"], 0.02)
        self.assertEqual(record["foreignPersonBBoxFallbackPixelCount"], 200)
        self.assertIs(record["foreignPersonEdgeDominant"], True)
        self.assertEqual(record["foreignPersonEvidenceSource"], "mixed")
        self.assertEqual(record["paddingRatio"], 0.4)
        self.assertEqual(record["letterboxPaddingRatio"], 0.3)
        self.assertEqual(record["sourceOverflowRatio"], 0.1)
        self.assertEqual(record["bboxOverflowRatio"], 0.05)
        self.assertEqual(record["nearImageEdgeCount"], 2)
        self.assertEqual(record["truncatedEdgeCount"], 1)
        self.assertEqual(record["qualityReasons"], ["mask_area_below_enroll"])
        self.assertIs(type(record["qualityReasons"][0]), str)
        json.dumps(record)
        self.assertEqual(record["status"], 1)
        self.assertEqual(record["aiId"], "ai-1")
        self.assertIs(record["isPanoKeyframe"], True)
        self.assertNotIn("trackId", record)
        self.assertNotIn("trackStatus", record)

    def test_recognition_records_do_not_include_track_fields(self):
        uid = "33333333-3333-4333-8333-333333333333"
        face_record = build_face_recognition_record(
            dev_sn="CAM001",
            keyframe_id=uid,
            picture_vector_id=456,
            face_url="CAM001/face_crop/f.jpg",
            timestamp=171,
            face_id="face-1",
            score=0.9,
            ai_id="ai-1",
            face_status=0,
            info="正常",
            face_quality_score=6.5,
            face_quality_level=2,
        )
        person_record = build_person_recognition_record(
            dev_sn="CAM001",
            keyframe_id=uid,
            picture_vector_id=456,
            image_url="CAM001/picture/k.jpg",
            face_id="0",
            reid="reid-1",
            score=0.8,
            detect_time=172,
            timestamp=171,
            people_rect={"x": 1, "y": 2, "w": 3, "h": 4},
            info="正常",
            reid_status=0,
            ai_id="ai-2",
            is_pano_keyframe=True,
            person_quality_score=6.0,
            person_quality_level=1,
            person_quality_action="identify_only",
            mask_score=6.0,
            crop_score=10.0,
            torso_keypoint_count=4,
            shoulder_hip_chain_count=2,
            half_body_pose_eligible=True,
            foreign_person_ratio=0.08,
            foreign_person_core_ratio=0.01,
            foreign_person_pixel_count=800,
            foreign_person_core_pixel_count=100,
            foreign_person_bbox_fallback_ratio=0.02,
            foreign_person_bbox_fallback_pixel_count=200,
            foreign_person_edge_dominant=True,
            foreign_person_evidence_source="mask",
            padding_ratio=0.4,
            letterbox_padding_ratio=0.3,
            source_overflow_ratio=0.1,
            bbox_overflow_ratio=0.05,
            near_image_edge_count=2,
            truncated_edge_count=1,
            quality_reasons=[QualityReason.MASK_AREA_BELOW_ENROLL],
        )

        self.assertEqual(face_record["faceStatus"], 0)
        self.assertEqual(face_record["faceQualityScore"], 6.5)
        self.assertEqual(face_record["faceQualityLevel"], 2)
        self.assertIs(face_record["isPanoKeyframe"], False)
        self.assertEqual(person_record["reidStatus"], 0)
        self.assertIs(person_record["isPanoKeyframe"], True)
        self.assertEqual(person_record["reidState"], "")
        self.assertEqual(person_record["reidEnrollmentState"], "")
        self.assertEqual(person_record["reidEnrollmentReason"], "")
        self.assertEqual(person_record["reidDecision"], "SKIPPED")
        self.assertEqual(person_record["reidDecisionReason"], "")
        self.assertEqual(person_record["reidSearchMode"], "NOT_EXECUTED")
        self.assertEqual(person_record["reidSource"], "")
        self.assertIs(person_record["isNewReid"], False)
        self.assertEqual(person_record["canonicalFaceId"], "0")
        self.assertEqual(person_record["reidTop1Score"], 0.0)
        self.assertEqual(person_record["reidMeanScore"], 0.0)
        self.assertEqual(person_record["reidClassScore"], 0.0)
        self.assertEqual(person_record["reidSupportCount"], 0)
        self.assertEqual(person_record["reidRunnerUpScore"], 0.0)
        self.assertEqual(person_record["reidMargin"], 0.0)
        self.assertEqual(person_record["reidRequiredMargin"], 0.0)
        self.assertEqual(person_record["sameFrameExcludedReidCount"], 0)
        self.assertEqual(person_record["personQualityScore"], 6.0)
        self.assertEqual(person_record["personQualityLevel"], 1)
        self.assertEqual(person_record["personQualityAction"], "identify_only")
        self.assertEqual(person_record["maskScore"], 6.0)
        self.assertEqual(person_record["cropScore"], 10.0)
        self.assertEqual(person_record["torsoKeypointCount"], 4)
        self.assertEqual(person_record["shoulderHipChainCount"], 2)
        self.assertIs(person_record["halfBodyPoseEligible"], True)
        self.assertEqual(person_record["foreignPersonRatio"], 0.08)
        self.assertEqual(person_record["foreignPersonCoreRatio"], 0.01)
        self.assertEqual(person_record["foreignPersonPixelCount"], 800)
        self.assertEqual(person_record["foreignPersonCorePixelCount"], 100)
        self.assertEqual(person_record["foreignPersonBBoxFallbackRatio"], 0.02)
        self.assertEqual(
            person_record["foreignPersonBBoxFallbackPixelCount"],
            200,
        )
        self.assertIs(person_record["foreignPersonEdgeDominant"], True)
        self.assertEqual(person_record["foreignPersonEvidenceSource"], "mask")
        self.assertEqual(person_record["paddingRatio"], 0.4)
        self.assertEqual(person_record["letterboxPaddingRatio"], 0.3)
        self.assertEqual(person_record["sourceOverflowRatio"], 0.1)
        self.assertEqual(person_record["bboxOverflowRatio"], 0.05)
        self.assertEqual(person_record["nearImageEdgeCount"], 2)
        self.assertEqual(person_record["truncatedEdgeCount"], 1)
        self.assertEqual(person_record["qualityReasons"], ["mask_area_below_enroll"])
        self.assertIs(type(person_record["qualityReasons"][0]), str)
        json.dumps(person_record)

        for record in [face_record, person_record]:
            self.assertEqual(record["keyframeId"], uid)
            self.assertEqual(record["pictureVectorId"], 456)
            self.assertNotIn("trackId", record)
            self.assertNotIn("trackStatus", record)

    def test_face_orientation_contract_for_all_record_builders(self):
        cases = [
            (
                "face_detection",
                build_face_detection_record,
                {
                    "keyframe_id": "uid",
                    "picture_vector_id": 1,
                    "crop_url": "crop.jpg",
                    "dev_sn": "CAM001",
                    "timestamp": 1,
                    "detect_time": 1,
                    "face_keypoint": [],
                    "face_rect": {},
                    "confidence": 0.9,
                    "status": 1,
                    "ai_id": "ai-1",
                    "info": "",
                },
                "front",
                FaceOrientation.FRONT,
            ),
            (
                "face_recognition",
                build_face_recognition_record,
                {
                    "dev_sn": "CAM001",
                    "keyframe_id": "uid",
                    "picture_vector_id": 1,
                    "face_url": "face.jpg",
                    "timestamp": 1,
                    "face_id": "face-1",
                    "score": 0.9,
                    "ai_id": "ai-1",
                    "face_status": 1,
                    "info": "",
                },
                "side",
                FaceOrientation.SIDE,
            ),
            (
                "person_detection",
                build_person_detection_record,
                {
                    "keyframe_id": "uid",
                    "picture_vector_id": 1,
                    "crop_url": "crop.jpg",
                    "dev_sn": "CAM001",
                    "timestamp": 1,
                    "detect_time": 1,
                    "people_rect": {},
                    "confidence": 0.9,
                    "mask_pixel_count": 1,
                    "status": 1,
                    "ai_id": "ai-1",
                    "info": "",
                },
                "no_face",
                FaceOrientation.NO_FACE,
            ),
            (
                "person_recognition",
                build_person_recognition_record,
                {
                    "dev_sn": "CAM001",
                    "keyframe_id": "uid",
                    "picture_vector_id": 1,
                    "image_url": "frame.jpg",
                    "face_id": "0",
                    "reid": "reid-1",
                    "score": 0.9,
                    "detect_time": 1,
                    "timestamp": 1,
                    "people_rect": {},
                    "info": "",
                    "reid_status": 1,
                    "ai_id": "ai-1",
                },
                "unknown",
                FaceOrientation.UNKNOWN,
            ),
        ]

        for name, builder, kwargs, valid_label, enum_value in cases:
            with self.subTest(builder=name, input="valid"):
                record = builder(**kwargs, face_orientation=f"  {valid_label.upper()}  ")
                self.assertEqual(record["faceOrientation"], valid_label)
                self.assertIs(type(record["faceOrientation"]), str)
                json.dumps(record)
            with self.subTest(builder=name, input="invalid"):
                record = builder(**kwargs, face_orientation="profile")
                self.assertEqual(record["faceOrientation"], "unknown")
                self.assertIs(type(record["faceOrientation"]), str)
                json.dumps(record)
            with self.subTest(builder=name, input="default"):
                record = builder(**kwargs)
                self.assertEqual(record["faceOrientation"], "unknown")
                self.assertIs(type(record["faceOrientation"]), str)
                json.dumps(record)
            with self.subTest(builder=name, input="enum"):
                record = builder(**kwargs, face_orientation=enum_value)
                self.assertEqual(record["faceOrientation"], enum_value.value)
                self.assertIs(type(record["faceOrientation"]), str)
                json.dumps(record)

    def test_person_orientation_contract_for_person_records_only(self):
        person_detection_kwargs = {
            "keyframe_id": "uid",
            "picture_vector_id": 1,
            "crop_url": "crop.jpg",
            "dev_sn": "CAM001",
            "timestamp": 1,
            "detect_time": 1,
            "people_rect": {},
            "confidence": 0.9,
            "mask_pixel_count": 1,
            "status": 1,
            "ai_id": "ai-1",
            "info": "",
        }
        person_recognition_kwargs = {
            "dev_sn": "CAM001",
            "keyframe_id": "uid",
            "picture_vector_id": 1,
            "image_url": "frame.jpg",
            "face_id": "0",
            "reid": "reid-1",
            "score": 0.9,
            "detect_time": 1,
            "timestamp": 1,
            "people_rect": {},
            "info": "",
            "reid_status": 1,
            "ai_id": "ai-1",
        }

        for name, builder, kwargs in (
            (
                "person_detection",
                build_person_detection_record,
                person_detection_kwargs,
            ),
            (
                "person_recognition",
                build_person_recognition_record,
                person_recognition_kwargs,
            ),
        ):
            with self.subTest(builder=name, input="enum"):
                record = builder(**kwargs, person_orientation=PersonOrientation.BACK)
                self.assertEqual(record["personOrientation"], "back")
                self.assertIs(type(record["personOrientation"]), str)
                json.dumps(record, allow_nan=False)
            with self.subTest(builder=name, input="whitespace_case"):
                record = builder(**kwargs, person_orientation=" SIDE ")
                self.assertEqual(record["personOrientation"], "side")
                self.assertIs(type(record["personOrientation"]), str)
                json.dumps(record, allow_nan=False)
            for invalid in ("profile", None, 123):
                with self.subTest(builder=name, input=repr(invalid)):
                    record = builder(**kwargs, person_orientation=invalid)
                    self.assertEqual(record["personOrientation"], "unknown")
                    self.assertIs(type(record["personOrientation"]), str)
                    json.dumps(record, allow_nan=False)
            with self.subTest(builder=name, input="default"):
                record = builder(**kwargs)
                self.assertEqual(record["personOrientation"], "unknown")
                self.assertIs(type(record["personOrientation"]), str)
                json.dumps(record, allow_nan=False)

        face_detection_record = build_face_detection_record(
            keyframe_id="uid",
            picture_vector_id=1,
            crop_url="crop.jpg",
            dev_sn="CAM001",
            timestamp=1,
            detect_time=1,
            face_keypoint=[],
            face_rect={},
            confidence=0.9,
            status=1,
            ai_id="ai-1",
            info="",
        )
        face_recognition_record = build_face_recognition_record(
            dev_sn="CAM001",
            keyframe_id="uid",
            picture_vector_id=1,
            face_url="face.jpg",
            timestamp=1,
            face_id="face-1",
            score=0.9,
            ai_id="ai-1",
            face_status=1,
            info="",
        )
        vector_record = build_vector_record(
            vector_id=1,
            embedding=[0.1, 0.2],
            keyframe_id="uid",
            picture_vector_id=1,
            timestamp=1,
            keyframe_url="frame.jpg",
            target_type="person",
            ai_id="ai-1",
        )
        for record in (
            face_detection_record,
            face_recognition_record,
            vector_record,
        ):
            self.assertNotIn("personOrientation", record)
            json.dumps(record, allow_nan=False)

    def test_person_recognition_enrollment_fields_are_validated(self):
        common = {
            "dev_sn": "CAM001",
            "keyframe_id": "uid",
            "picture_vector_id": 1,
            "image_url": "frame.jpg",
            "face_id": "0",
            "reid": "reid-1",
            "score": 0.9,
            "detect_time": 1,
            "timestamp": 1,
            "people_rect": {"x": 1, "y": 2, "w": 3, "h": 4},
            "info": "等待扩充",
            "reid_status": 2,
            "reid_state": "ACTIVE",
            "ai_id": "ai-1",
        }
        with self.assertRaisesRegex(ValueError, "reidEnrollmentState"):
            build_person_recognition_record(
                **common,
                reid_enrollment_state="PROVISIONAL",
                reid_enrollment_reason="awaiting_body_consensus",
            )
        with self.assertRaisesRegex(ValueError, "reidState"):
            build_person_recognition_record(
                **{**common, "reid_state": "PROVISIONAL"},
                reid_enrollment_state="FORMAL_PENDING",
            )
        with self.assertRaisesRegex(ValueError, "reidEnrollmentState"):
            build_person_recognition_record(
                **common,
                reid_enrollment_state="INSERTED",
            )

    def test_person_recognition_reid_diagnostics_are_normalized(self):
        record = build_person_recognition_record(
            dev_sn="CAM001",
            keyframe_id="uid",
            picture_vector_id=1,
            image_url="frame.jpg",
            face_id="face-1",
            reid="reid-1",
            score=0.9,
            detect_time=1,
            timestamp=1,
            people_rect={"x": 1, "y": 2, "w": 3, "h": 4},
            info="matched",
            reid_status=2,
            reid_state="ACTIVE",
            ai_id="ai-1",
            reid_decision="match_existing",
            reid_decision_reason="accepted",
            reid_search_mode="same_face",
            reid_source="milvus+buffer",
            is_new_reid=False,
            canonical_face_id="face-1",
            reid_top1_score=0.9,
            reid_mean_score=0.85,
            reid_class_score=0.88,
            reid_support_count=3,
            reid_runner_up_score=float("nan"),
            reid_margin=float("inf"),
            reid_required_margin=0.04,
            same_frame_excluded_reid_count=-1,
        )

        self.assertEqual(record["reidDecision"], "MATCH_EXISTING")
        self.assertEqual(record["reidDecisionReason"], "accepted")
        self.assertEqual(record["reidSearchMode"], "SAME_FACE")
        self.assertEqual(record["reidSource"], "MILVUS+BUFFER")
        self.assertEqual(record["canonicalFaceId"], "face-1")
        self.assertEqual(record["reidTop1Score"], 0.9)
        self.assertEqual(record["reidMeanScore"], 0.85)
        self.assertEqual(record["reidClassScore"], 0.88)
        self.assertEqual(record["reidSupportCount"], 3)
        self.assertEqual(record["reidRunnerUpScore"], 0.0)
        self.assertEqual(record["reidMargin"], 0.0)
        self.assertEqual(record["reidRequiredMargin"], 0.04)
        self.assertEqual(record["sameFrameExcludedReidCount"], 0)
        json.dumps(record)

    def test_person_payload_preserves_raw_reason_strings(self):
        record = build_person_detection_record(
            keyframe_id="uid",
            picture_vector_id=1,
            crop_url="crop.jpg",
            dev_sn="CAM001",
            timestamp=171,
            detect_time=172,
            people_rect={"x": 1, "y": 2, "w": 3, "h": 4},
            confidence=0.8,
            mask_pixel_count=100,
            status=1,
            ai_id="ai-1",
            info="",
            quality_reasons=["future_reason"],
        )

        self.assertEqual(record["qualityReasons"], ["future_reason"])
        self.assertEqual(record["torsoKeypointCount"], 0)
        self.assertEqual(record["shoulderHipChainCount"], 0)
        self.assertIs(record["halfBodyPoseEligible"], False)
        self.assertEqual(record["foreignPersonRatio"], 0.0)
        self.assertEqual(record["foreignPersonCoreRatio"], 0.0)
        self.assertEqual(record["foreignPersonPixelCount"], 0)
        self.assertEqual(record["foreignPersonCorePixelCount"], 0)
        self.assertEqual(record["foreignPersonBBoxFallbackRatio"], 0.0)
        self.assertEqual(record["foreignPersonBBoxFallbackPixelCount"], 0)
        self.assertIs(record["foreignPersonEdgeDominant"], False)
        self.assertEqual(record["foreignPersonEvidenceSource"], "none")

    def test_person_geometry_metrics_reject_nonfinite_values(self):
        record = build_person_detection_record(
            keyframe_id="uid",
            picture_vector_id=1,
            crop_url="crop.jpg",
            dev_sn="CAM001",
            timestamp=171,
            detect_time=172,
            people_rect={"x": 1, "y": 2, "w": 3, "h": 4},
            confidence=0.8,
            mask_pixel_count=100,
            status=1,
            ai_id="ai-1",
            info="",
            padding_ratio=float("nan"),
            letterbox_padding_ratio=float("inf"),
            source_overflow_ratio=float("-inf"),
            bbox_overflow_ratio="invalid",
            near_image_edge_count=float("inf"),
            truncated_edge_count=None,
            foreign_person_ratio=float("nan"),
            foreign_person_core_ratio=float("inf"),
            foreign_person_bbox_fallback_ratio=float("-inf"),
            foreign_person_pixel_count=-1,
            foreign_person_core_pixel_count="invalid",
            foreign_person_bbox_fallback_pixel_count=None,
            foreign_person_evidence_source="invalid",
        )

        for field in (
            "paddingRatio",
            "letterboxPaddingRatio",
            "sourceOverflowRatio",
            "bboxOverflowRatio",
        ):
            self.assertEqual(record[field], 0.0)
        self.assertEqual(record["nearImageEdgeCount"], 0)
        self.assertEqual(record["truncatedEdgeCount"], 0)
        self.assertEqual(record["foreignPersonRatio"], 0.0)
        self.assertEqual(record["foreignPersonCoreRatio"], 0.0)
        self.assertEqual(record["foreignPersonBBoxFallbackRatio"], 0.0)
        self.assertEqual(record["foreignPersonPixelCount"], 0)
        self.assertEqual(record["foreignPersonCorePixelCount"], 0)
        self.assertEqual(record["foreignPersonBBoxFallbackPixelCount"], 0)
        self.assertEqual(record["foreignPersonEvidenceSource"], "none")
        json.dumps(record, allow_nan=False)

    def test_person_geometry_metrics_are_clamped_to_public_ranges(self):
        record = build_person_detection_record(
            keyframe_id="uid",
            picture_vector_id=1,
            crop_url="crop.jpg",
            dev_sn="CAM001",
            timestamp=171,
            detect_time=172,
            people_rect={"x": 1, "y": 2, "w": 3, "h": 4},
            confidence=0.8,
            mask_pixel_count=100,
            status=1,
            ai_id="ai-1",
            info="",
            padding_ratio=2.0,
            letterbox_padding_ratio=-1.0,
            source_overflow_ratio=3.0,
            bbox_overflow_ratio=-2.0,
            near_image_edge_count=9,
            truncated_edge_count=-1,
            torso_keypoint_count=9,
            shoulder_hip_chain_count=-1,
            foreign_person_ratio=-1.0,
            foreign_person_core_ratio=-2.0,
            foreign_person_bbox_fallback_ratio=-3.0,
            foreign_person_evidence_source="MIXED",
        )

        self.assertEqual(record["paddingRatio"], 1.0)
        self.assertEqual(record["letterboxPaddingRatio"], 0.0)
        self.assertEqual(record["sourceOverflowRatio"], 1.0)
        self.assertEqual(record["bboxOverflowRatio"], 0.0)
        self.assertEqual(record["nearImageEdgeCount"], 4)
        self.assertEqual(record["truncatedEdgeCount"], 0)
        self.assertEqual(record["torsoKeypointCount"], 4)
        self.assertEqual(record["shoulderHipChainCount"], 0)
        self.assertEqual(record["foreignPersonRatio"], 0.0)
        self.assertEqual(record["foreignPersonCoreRatio"], 0.0)
        self.assertEqual(record["foreignPersonBBoxFallbackRatio"], 0.0)
        self.assertEqual(record["foreignPersonEvidenceSource"], "mixed")

    def test_vector_record_includes_keyframe_id(self):
        uid = "44444444-4444-4444-8444-444444444444"
        record = build_vector_record(
            vector_id="456",
            embedding=[0.1, 0.2, 0.3],
            keyframe_id=uid,
            picture_vector_id="456",
            timestamp=171,
            keyframe_url="CAM001/picture/k.jpg",
            target_type="picture",
            ai_id="",
            dev_sn="CAM001",
            vector_db_name="alogPictureClip",
        )

        self.assertEqual(record["vectorId"], 456)
        self.assertEqual(record["keyframeId"], uid)
        self.assertEqual(record["pictureVectorId"], 456)
        self.assertIs(record["isPanoKeyframe"], False)
        self.assertEqual(record["keyframeUrl"], "CAM001/picture/k.jpg")


if __name__ == "__main__":
    unittest.main()
