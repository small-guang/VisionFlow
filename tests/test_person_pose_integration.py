import asyncio
from datetime import datetime
import importlib
import importlib.util
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from tests.image_execution_test_support import InlineImageExecutor

from core.algorithm_config import algorithm_config
from core.infer.results import FaceQualityResult, FeatureResult
from tests.person_pose_test_support import (
    IMAGE_SHAPE,
    PERSON_RECT,
    full_body_keypoints,
    pose_result,
    segment_result,
    textured_image,
)

if "utils.storage.oss_util" not in sys.modules:
    oss_util_module = types.ModuleType("utils.storage.oss_util")
    oss_util_module.AsyncOSSClient = object
    sys.modules["utils.storage.oss_util"] = oss_util_module

from core.person_keyframe_processor import (
    PersonKeyframeProcessor,
    _pose_keypoints_to_crop,
)
from core.person_pipeline import PersonPipelineRuntime
from utils.quality import face_quality
from utils.quality.face_orientation import FaceOrientation, FaceOrientationResult
from utils.quality.person_orientation import (
    PersonOrientation,
    PersonOrientationResult,
)
from utils.reid import person_face_recognition as person_face
from utils.keyframe.detection_helpers import (
    KeyframeContext,
    person_segments_can_share_reid,
    same_frame_reid_exclusions,
)
from utils.quality.person.policy import (
    PERSON_ACTION_BUILD_GALLERY,
    PERSON_ACTION_IDENTIFY_ONLY,
)
from utils.reid.person_body import (
    BODY_FULL,
    BODY_HALF_LOWER,
    BODY_HALF_UPPER,
    BODY_PARTIAL,
    BODY_THREE_QUARTER_LOWER,
    BODY_THREE_QUARTER_UPPER,
    BODY_UNKNOWN,
    gallery_body_types_for_query,
    person_sbs_threshold,
)
from utils.quality.quality_reasons import QualityReason
from utils.keyframe.keyframe_flow_helpers import (
    encode_image_to_bytes,
)
from utils.quality.person.visual import evaluate_person_visual_quality
from utils.reid.person_identity import PersonIdentityService
from utils.reid.person_search import PersonReidSearchService
from utils.reid.reid_models import (
    PersonReidSearchResult,
    REID_SEARCH_MODE_ANONYMOUS_GLOBAL,
    REID_SEARCH_MODE_SAME_FACE,
    REID_SOURCE_BUFFER,
    REID_SOURCE_MILVUS,
    ReidCandidate,
)
from utils.reid.reid_resolution import resolve_reid_classes
from utils.reid.reid_enrollment_state import (
    REID_ENROLLMENT_STATE_FORMAL_PENDING,
)
from utils.reid.reid_state import (
    REID_STATE_ACTIVE,
    REID_STATE_UNASSIGNED,
)
from utils.buffers.sbs_feature_buffer import SbsFeatureBuffer

class DummyLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class CapturingLogger(DummyLogger):
    def __init__(self):
        self.exceptions = []

    def exception(self, message):
        self.exceptions.append(message)


class FakeOSS:
    async def upload_object(self, image_bytes, object_name):
        return object_name


class FakeBuffer:
    def __init__(self):
        self.detections = []
        self.recognitions = []
        self.features = []
        self.clip_features = []
        self.sbs = types.SimpleNamespace(
            offer_feature=self.offer_person_feature,
            attach_new_reid_recognition=self.attach_new_reid_recognition,
            asearch_candidates_with_hit_info=self.search_person_candidates,
        )
        self.tbps = types.SimpleNamespace(add_feature=self.add_person_clip_feature)

    async def add_person_detection_record(self, record):
        self.detections.append(record)

    async def add_person_recognition_record(self, record):
        self.recognitions.append(record)

    async def add_person_feature(self, **kwargs):
        self.features.append(kwargs)

    async def offer_person_feature(self, **kwargs):
        defer_recognition = kwargs.pop("defer_new_reid_recognition", False)
        self.features.append(kwargs)
        return types.SimpleNamespace(
            reason="test_enrollment",
            reid_state=REID_STATE_ACTIVE,
            enrollment_state=REID_ENROLLMENT_STATE_FORMAL_PENDING,
            recognition_release=(object() if defer_recognition else None),
        )

    async def attach_new_reid_recognition(self, _release, record, on_ready):
        await on_ready(record)

    async def search_person_candidates(self, *_args, **_kwargs):
        return [], 0.0, "", ""

    async def add_person_clip_feature(self, **kwargs):
        self.clip_features.append(kwargs)


class FakeTriton:
    def __init__(self, poses=None, pose_error=None, segments=None):
        self.poses = [pose_result()] if poses is None else poses
        self.segments = [segment_result()] if segments is None else segments
        self.pose_error = pose_error
        self.tbps_calls = 0
        self.sbs_calls = 0

    async def extract_person_segments(self, image):
        return self.segments

    async def extract_person_poses(self, image):
        if self.pose_error is not None:
            raise self.pose_error
        return self.poses

    async def extract_tbps_image_features(self, image_data):
        self.tbps_calls += 1
        return FeatureResult(np.ones(512, dtype=np.float32))

    async def extract_sbs_features(self, image_data):
        self.sbs_calls += 1
        return FeatureResult(np.ones(2048, dtype=np.float32))


def person_result(
    reid=None,
    score=0.0,
    vector_hit_reid="",
    source="",
    *,
    state=None,
    canonical_face_id="",
    reason="accepted",
    search_mode=REID_SEARCH_MODE_ANONYMOUS_GLOBAL,
    top1_score=None,
    mean_score=None,
    class_score=None,
    runner_up_score=0.0,
    margin=None,
    required_margin=0.08,
    support_count=1,
):
    resolved_score = float(score)
    return PersonReidSearchResult(
        reid=reid,
        score=resolved_score,
        vector_hit_reid=vector_hit_reid,
        source=source,
        state=state or (REID_STATE_ACTIVE if reid else REID_STATE_UNASSIGNED),
        canonical_face_id=canonical_face_id,
        reason=reason if reid else "no_candidate",
        support_count=support_count if reid else 0,
        top1_score=(resolved_score if top1_score is None and reid else top1_score or 0.0),
        mean_score=(resolved_score if mean_score is None and reid else mean_score or 0.0),
        class_score=(resolved_score if class_score is None and reid else class_score or 0.0),
        runner_up_score=runner_up_score,
        margin=(resolved_score if margin is None and reid else margin or 0.0),
        required_margin=required_margin,
        search_mode=search_mode,
    )


async def no_person_match(*args, **kwargs):
    return person_result()


async def no_face_match(*args, **kwargs):
    return None, 0.0, "", ""


async def observed_face_match(*args, **kwargs):
    return {
        "face_id": "face-a",
        "info": "test face",
        "score": 0.9,
        "face_orientation": "front",
    }


def person_search_service(person_db, sbs_buffer):
    return PersonReidSearchService(
        person_sbs_embedding_db=person_db,
        sbs_buffer=sbs_buffer,
    )


def person_processor(
    *,
    triton,
    buffer,
    search_persons=no_person_match,
    match_face=None,
    upright_device_sns=frozenset(),
    logger=None,
    person_face_reuse_mode="off",
):
    logger = logger or DummyLogger()
    identity_service = PersonIdentityService(
        identity_lock=asyncio.Lock(),
        search_persons=search_persons,
        sbs_buffer=buffer.sbs,
        logger=logger,
    )
    runtime = None
    if match_face is not None:
        runtime = PersonPipelineRuntime(
            encode_image=encode_image_to_bytes,
            evaluate_visual_quality=evaluate_person_visual_quality,
            match_face=match_face,
            pose_keypoints_to_crop=_pose_keypoints_to_crop,
        )
    return PersonKeyframeProcessor(
        image_executor=InlineImageExecutor(),
        triton_client=triton,
        feature_buffer=buffer,
        identity_service=identity_service,
        search_faces_with_pending_locked=no_face_match,
        logger=logger,
        runtime=runtime,
        upright_device_sns=upright_device_sns,
        person_face_reuse_mode=person_face_reuse_mode,
    )


async def enqueue_formal_sbs(buffer, **kwargs):
    await buffer._enqueue_formal_feature(
        **kwargs,
        crop_url="",
        quality_score=9.0,
    )


async def resolve_buffer_candidates(buffer, embedding, **kwargs):
    candidates, highest_score, highest_reid, _ = (
        await buffer.asearch_candidates_with_hit_info(embedding, **kwargs)
    )
    resolution = resolve_reid_classes(
        candidates,
        query_face_id=kwargs.get("face_id"),
        query_device_id=(kwargs.get("device_ids") or [""])[0],
    )
    if resolution.reid is None:
        return None, highest_score, highest_reid
    return resolution.reid, resolution.score, highest_reid


class SameFramePersonReidPolicyTests(unittest.TestCase):
    @staticmethod
    def _segment(rect, mask):
        return {"rect": dict(rect), "mask": mask}

    def test_duplicate_thresholds_are_inclusive(self):
        first_mask = np.zeros((4, 20), dtype=np.uint8)
        second_mask = np.zeros_like(first_mask)
        first_mask[0, :10] = 1
        second_mask[0, :7] = 1
        first = self._segment({"x": 0, "y": 0, "w": 17, "h": 10}, first_mask)
        second = self._segment({"x": 3, "y": 0, "w": 17, "h": 10}, second_mask)

        self.assertTrue(person_segments_can_share_reid(first, second))

    def test_either_overlap_below_threshold_prevents_sharing(self):
        mask = np.ones((4, 20), dtype=np.uint8)
        low_bbox_overlap = self._segment(
            {"x": 4, "y": 0, "w": 17, "h": 10},
            mask,
        )
        first = self._segment({"x": 0, "y": 0, "w": 17, "h": 10}, mask)
        self.assertFalse(person_segments_can_share_reid(first, low_bbox_overlap))

        low_mask_overlap = np.zeros_like(mask)
        low_mask_overlap[0, :6] = 1
        first_mask = np.zeros_like(mask)
        first_mask[0, :10] = 1
        self.assertFalse(
            person_segments_can_share_reid(
                self._segment(first["rect"], first_mask),
                self._segment(first["rect"], low_mask_overlap),
            )
        )

    def test_invalid_or_empty_masks_fail_closed(self):
        rect = {"x": 0, "y": 0, "w": 10, "h": 10}
        valid = np.ones((10, 10), dtype=np.uint8)
        invalid_masks = [None, np.zeros((10, 10), dtype=np.uint8), np.ones(10), np.ones((9, 10))]
        for invalid_mask in invalid_masks:
            with self.subTest(mask=invalid_mask):
                self.assertFalse(
                    person_segments_can_share_reid(
                        self._segment(rect, valid),
                        self._segment(rect, invalid_mask),
                    )
                )

    def test_reid_is_excluded_when_any_previous_segment_is_not_duplicate(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        current = self._segment({"x": 0, "y": 0, "w": 10, "h": 10}, mask)
        duplicate = self._segment({"x": 0, "y": 0, "w": 10, "h": 10}, mask)
        separate = self._segment({"x": 20, "y": 0, "w": 10, "h": 10}, mask)

        self.assertEqual(
            same_frame_reid_exclusions(
                {"shared": [duplicate, separate], "duplicate-only": [duplicate]},
                current,
            ),
            {"shared"},
        )


class CombinedPersonSearchExclusionTests(unittest.TestCase):
    def test_missing_latest_candidate_interface_fails_fast(self):
        class PendingSource:
            async def asearch_candidates_with_hit_info(self, *_args, **_kwargs):
                return [], 0.0, "", ""

        service = person_search_service(object(), PendingSource())

        with self.assertRaises(AttributeError):
            asyncio.run(
                service.search(
                    np.ones(2048, dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=None,
                    query_device_id="CAM001",
                )
            )

    def test_exclusions_are_forwarded_to_milvus_and_pending(self):
        calls = []

        class MilvusSource:
            async def asearch_similar_person_candidates_with_hit_info(
                self,
                embedding,
                **kwargs,
            ):
                calls.append(("milvus", kwargs))
                return (
                    [
                        ReidCandidate(
                            "milvus-alternative",
                            0.80,
                            source=REID_SOURCE_MILVUS,
                            threshold=0.60,
                            body_completeness=BODY_FULL,
                            device_id="CAM001",
                            prototype_id=1,
                        ),
                        ReidCandidate(
                            "milvus-alternative",
                            0.79,
                            source=REID_SOURCE_MILVUS,
                            threshold=0.60,
                            body_completeness=BODY_FULL,
                            device_id="CAM001",
                            prototype_id=2,
                        )
                    ],
                    0.80,
                    "milvus-alternative",
                    REID_SOURCE_MILVUS,
                )

        class PendingSource:
            async def asearch_candidates_with_hit_info(self, embedding, **kwargs):
                calls.append(("pending", kwargs))
                return [], 0.75, "pending-alternative", REID_SOURCE_BUFFER

        service = person_search_service(MilvusSource(), PendingSource())

        result = asyncio.run(
            service.search(
                np.ones(2048, dtype=np.float32),
                query_body_completeness=BODY_FULL,
                device_ids=None,
                excluded_reids={"occupied"},
                query_device_id="CAM001",
            )
        )

        self.assertEqual(result.reid, "milvus-alternative")
        self.assertAlmostEqual(result.score, 0.798)
        self.assertEqual(result.vector_hit_reid, "milvus-alternative")
        self.assertEqual(result.source, REID_SOURCE_MILVUS)
        self.assertEqual(result.search_mode, REID_SEARCH_MODE_ANONYMOUS_GLOBAL)
        self.assertAlmostEqual(result.top1_score, 0.80)
        self.assertAlmostEqual(result.mean_score, 0.795)
        self.assertAlmostEqual(result.class_score, 0.798)
        self.assertEqual(result.support_count, 2)
        self.assertEqual(result.runner_up_score, 0.0)
        self.assertAlmostEqual(result.margin, 0.798)
        self.assertEqual(result.required_margin, 0.08)
        self.assertEqual([name for name, _ in calls], ["milvus", "pending"])
        for _, kwargs in calls:
            self.assertEqual(kwargs["excluded_reids"], frozenset({"occupied"}))

    def test_known_face_miss_uses_one_same_face_search(self):
        calls = []

        class MilvusSource:
            async def asearch_similar_person_candidates_with_hit_info(
                self, _embedding, **kwargs
            ):
                calls.append(("milvus", kwargs))
                return [], 0.0, "", ""

        class PendingSource:
            async def asearch_candidates_with_hit_info(self, _embedding, **kwargs):
                calls.append(("pending", kwargs))
                return [], 0.0, "", ""

        service = person_search_service(MilvusSource(), PendingSource())
        result = asyncio.run(
            service.search(
                np.ones(2048, dtype=np.float32),
                query_body_completeness=BODY_FULL,
                device_ids=None,
                face_id="face-a",
                query_device_id="CAM001",
            )
        )

        self.assertIsNone(result.reid)
        self.assertEqual(result.search_mode, REID_SEARCH_MODE_SAME_FACE)
        self.assertEqual([name for name, _ in calls], ["milvus", "pending"])
        self.assertTrue(all(kwargs["face_id"] == "face-a" for _, kwargs in calls))

    def test_known_face_exact_match_reports_same_face_search_mode(self):
        class MilvusSource:
            async def asearch_similar_person_candidates_with_hit_info(
                self,
                _embedding,
                **kwargs,
            ):
                self.last_kwargs = kwargs
                return (
                    [
                        ReidCandidate(
                            "face-reid",
                            score,
                            source=REID_SOURCE_MILVUS,
                            threshold=0.56,
                            body_completeness=BODY_FULL,
                            face_id="face-a",
                            device_id="CAM001",
                            prototype_id=index,
                        )
                        for index, score in enumerate((0.90, 0.88), start=1)
                    ],
                    0.90,
                    "face-reid",
                    REID_SOURCE_MILVUS,
                )

        class PendingSource:
            async def asearch_candidates_with_hit_info(self, *_args, **_kwargs):
                return [], 0.0, "", ""

        milvus = MilvusSource()
        service = person_search_service(milvus, PendingSource())

        result = asyncio.run(
            service.search(
                np.ones(2048, dtype=np.float32),
                query_body_completeness=BODY_FULL,
                device_ids=None,
                face_id="face-a",
                query_device_id="CAM001",
            )
        )

        self.assertEqual(result.reid, "face-reid")
        self.assertEqual(result.search_mode, REID_SEARCH_MODE_SAME_FACE)
        self.assertEqual(result.canonical_face_id, "face-a")
        self.assertEqual(result.required_margin, 0.0)
        self.assertAlmostEqual(result.top1_score, 0.90)
        self.assertAlmostEqual(result.mean_score, 0.89)
        self.assertAlmostEqual(result.class_score, 0.896)
        self.assertEqual(milvus.last_kwargs["face_id"], "face-a")


class PersonBodyReidPolicyTests(unittest.TestCase):
    @staticmethod
    def _pending_buffer():
        async def on_vectors_inserted(*_args):
            return None

        class NoopSbsDb:
            collection_name = "alogSBS"

            async def aapply_prototype_updates(self, batch):
                return []

        return SbsFeatureBuffer(
            vector_db=NoopSbsDb(),
            max_batch_size=100,
            on_vectors_inserted=on_vectors_inserted,
        )

    def test_body_pair_threshold_matrix(self):
        expected = {
            (BODY_FULL, BODY_FULL): 0.56,
            (BODY_FULL, BODY_THREE_QUARTER_UPPER): 0.53,
            (BODY_THREE_QUARTER_UPPER, BODY_FULL): 0.53,
            (BODY_THREE_QUARTER_UPPER, BODY_THREE_QUARTER_UPPER): 0.56,
            (BODY_THREE_QUARTER_LOWER, BODY_FULL): 0.55,
            (BODY_THREE_QUARTER_LOWER, BODY_THREE_QUARTER_UPPER): 0.55,
            (BODY_FULL, BODY_HALF_UPPER): 0.55,
            (BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER): 0.55,
            (BODY_HALF_UPPER, BODY_FULL): 0.55,
            (BODY_HALF_UPPER, BODY_THREE_QUARTER_UPPER): 0.55,
            (BODY_HALF_UPPER, BODY_HALF_UPPER): 0.56,
        }

        for pair, threshold in expected.items():
            with self.subTest(pair=pair):
                self.assertEqual(person_sbs_threshold(*pair), threshold)

    def test_searchable_types_use_compatible_gallery_body_types(self):
        expected = (BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER)
        for query_type in (
            BODY_FULL,
            BODY_THREE_QUARTER_UPPER,
            BODY_HALF_UPPER,
        ):
            with self.subTest(query_type=query_type):
                self.assertEqual(gallery_body_types_for_query(query_type), expected)

        self.assertEqual(
            gallery_body_types_for_query(BODY_THREE_QUARTER_LOWER),
            (BODY_FULL, BODY_THREE_QUARTER_UPPER),
        )

        for query_type in (BODY_HALF_LOWER, BODY_PARTIAL, BODY_UNKNOWN):
            with self.subTest(query_type=query_type):
                self.assertEqual(gallery_body_types_for_query(query_type), ())

    def test_incompatible_body_pair_has_no_threshold(self):
        with self.assertRaisesRegex(ValueError, "Incompatible person ReID body pair"):
            person_sbs_threshold(BODY_HALF_LOWER, BODY_FULL)

    def test_body_pair_thresholds_are_strict(self):
        for threshold in (0.56, 0.53, 0.55):
            with self.subTest(threshold=threshold):
                exact_resolution = resolve_reid_classes(
                    [
                        ReidCandidate(
                            "reid",
                            threshold,
                            threshold=threshold,
                            face_id="face-a",
                        )
                    ],
                    query_face_id="face-a",
                )
                above_resolution = resolve_reid_classes(
                    [
                        ReidCandidate(
                            "reid",
                            threshold + 0.0001,
                            threshold=threshold,
                            face_id="face-a",
                        )
                    ],
                    query_face_id="face-a",
                )
                self.assertIsNone(exact_resolution.reid)
                self.assertEqual(above_resolution.reid, "reid")

    def test_full_upper_and_half_queries_are_symmetric_in_pending(self):
        async def run():
            buffer = self._pending_buffer()
            records = (
                ("upper-reid", BODY_THREE_QUARTER_UPPER, [1.0, 0.0, 0.0]),
                ("full-reid", BODY_FULL, [0.0, 1.0, 0.0]),
                ("half-reid", BODY_HALF_UPPER, [0.0, 0.0, 1.0]),
            )
            for repeat in range(2):
                for index, (reid, body_type, embedding) in enumerate(records):
                    await enqueue_formal_sbs(buffer,
                        face_id="0",
                        re_id=reid,
                        body_completeness=body_type,
                        embedding=np.array(embedding, dtype=np.float32),
                        device_id="CAM001",
                        keyframe_id=f"{reid}-{repeat}",
                        picture_vector_id=repeat * 10 + index,
                        keyframe_url="crop.jpg",
                        person_rect=PERSON_RECT,
                        timestamp=9999999999999,
                        ai_id="ai",
                    )
            return (
                await resolve_buffer_candidates(buffer,
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                ),
                await resolve_buffer_candidates(buffer,
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    query_body_completeness=BODY_THREE_QUARTER_UPPER,
                ),
                await resolve_buffer_candidates(buffer,
                    np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_HALF_UPPER,
                ),
                await resolve_buffer_candidates(buffer,
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_HALF_UPPER,
                ),
                await resolve_buffer_candidates(buffer,
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    query_body_completeness=BODY_HALF_UPPER,
                ),
            )

        self.assertEqual(
            asyncio.run(run()),
            (
                ("half-reid", 1.0, "half-reid"),
                ("half-reid", 1.0, "half-reid"),
                ("full-reid", 1.0, "full-reid"),
                ("upper-reid", 1.0, "upper-reid"),
                ("half-reid", 1.0, "half-reid"),
            ),
        )

    def test_pending_body_pair_threshold_boundaries(self):
        def normalized_vector(score):
            return np.array(
                [score, np.sqrt(1.0 - score * score)],
                dtype=np.float32,
            )

        def threshold_floor(threshold):
            value = np.float32(threshold)
            if float(value) > threshold:
                value = np.nextafter(value, np.float32(-np.inf))
            return float(value)

        async def run():
            buffer = self._pending_buffer()
            records = (
                ("same-boundary", BODY_FULL, threshold_floor(0.56), "CAM-ENROLL"),
                ("same-above", BODY_FULL, 0.5601, "CAM-ENROLL"),
                (
                    "cross-boundary",
                    BODY_THREE_QUARTER_UPPER,
                    threshold_floor(0.53),
                    "CAM-ENROLL",
                ),
                ("cross-above", BODY_THREE_QUARTER_UPPER, 0.5301, "CAM-ENROLL"),
                ("identify-boundary", BODY_FULL, threshold_floor(0.55), "CAM-ID"),
                ("identify-above", BODY_FULL, 0.5501, "CAM-ID"),
                (
                    "half-same-boundary",
                    BODY_HALF_UPPER,
                    threshold_floor(0.56),
                    "CAM-ID",
                ),
                ("half-same-above", BODY_HALF_UPPER, 0.5601, "CAM-ID"),
            )
            for index, (reid, body_type, score, device_id) in enumerate(records):
                await enqueue_formal_sbs(buffer,
                    face_id="0",
                    re_id=reid,
                    body_completeness=body_type,
                    embedding=normalized_vector(score),
                    device_id=device_id,
                    keyframe_id=reid,
                    picture_vector_id=index,
                    keyframe_url="crop.jpg",
                    person_rect=PERSON_RECT,
                    timestamp=9999999999999,
                    ai_id="ai",
                )
            enroll_candidates, _, _, _ = await buffer.asearch_candidates_with_hit_info(
                np.array([1.0, 0.0], dtype=np.float32),
                query_body_completeness=BODY_FULL,
                device_ids=["CAM-ENROLL"],
            )
            identify_candidates, _, _, _ = (
                await buffer.asearch_candidates_with_hit_info(
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_HALF_UPPER,
                    device_ids=["CAM-ID"],
                )
            )
            return enroll_candidates, identify_candidates

        enroll_candidates, identify_candidates = asyncio.run(run())
        self.assertEqual(
            [(item.reid, item.threshold) for item in enroll_candidates],
            [
                ("same-boundary", 0.56),
                ("same-above", 0.56),
                ("cross-boundary", 0.53),
                ("cross-above", 0.53),
            ],
        )
        self.assertEqual(
            [(item.reid, item.threshold) for item in identify_candidates],
            [
                ("identify-boundary", 0.55),
                ("identify-above", 0.55),
                ("half-same-boundary", 0.56),
                ("half-same-above", 0.56),
            ],
        )

    def test_pending_search_excludes_reid_before_highest_hit_selection(self):
        async def run():
            buffer = self._pending_buffer()
            for index, (reid, embedding) in enumerate(
                (
                    ("excluded", np.array([1.0, 0.0], dtype=np.float32)),
                    ("alternative", np.array([0.9, np.sqrt(0.19)], dtype=np.float32)),
                )
            ):
                await enqueue_formal_sbs(buffer,
                    face_id="0",
                    re_id=reid,
                    body_completeness=BODY_FULL,
                    embedding=embedding,
                    device_id="CAM001",
                    keyframe_id=reid,
                    picture_vector_id=index,
                    keyframe_url="crop.jpg",
                    person_rect=PERSON_RECT,
                    timestamp=9999999999999,
                    ai_id="ai",
                )
            return await buffer.asearch_candidates_with_hit_info(
                np.array([1.0, 0.0], dtype=np.float32),
                query_body_completeness=BODY_FULL,
                excluded_reids={"excluded"},
            )

        candidates, highest_score, highest_reid, highest_source = asyncio.run(run())
        self.assertEqual([candidate.reid for candidate in candidates], ["alternative"])
        self.assertAlmostEqual(highest_score, 0.9, places=6)
        self.assertEqual(highest_reid, "alternative")
        self.assertEqual(highest_source, "buffer")




class PersonPoseFlowTests(unittest.TestCase):
    def _run_flow(
        self,
        triton,
        search_persons=no_person_match,
        buffer=None,
        match_face=None,
        logger=None,
        upright_device_sns=frozenset(),
        person_face_reuse_mode="off",
        frame_faces=(),
    ):
        buffer = buffer or FakeBuffer()
        processor = person_processor(
            triton=triton,
            buffer=buffer,
            search_persons=search_persons,
            match_face=match_face,
            logger=logger,
            upright_device_sns=upright_device_sns,
            person_face_reuse_mode=person_face_reuse_mode,
        )
        ctx = KeyframeContext(
            sn="CAM001",
            timestamp=1,
            object_name="CAM001/picture/k.jpg",
            keyframe_id="uid",
        )
        asyncio.run(
            processor.process_detection(
                ctx,
                textured_image(),
                "uid",
                1,
                ctx.object_name,
                FakeOSS(),
                frame_faces=frame_faces,
            )
        )
        return buffer

    @staticmethod
    def _two_person_triton(*, second_keypoints=None):
        first_rect = {"x": 5, "y": 20, "w": 35, "h": 120}
        second_rect = {"x": 75, "y": 20, "w": 35, "h": 120}
        return FakeTriton(
            segments=[segment_result(first_rect), segment_result(second_rect)],
            poses=[
                pose_result(first_rect),
                pose_result(second_rect, keypoints=second_keypoints),
            ],
        )

    def test_missing_pose_interface_fails_before_segment_inference(self):
        class NoPoseTriton:
            def __init__(self):
                self.segment_calls = 0

            async def extract_person_segments(self, image):
                self.segment_calls += 1
                return [segment_result()]

        triton = NoPoseTriton()

        with self.assertRaisesRegex(
            RuntimeError,
            "requires a callable Triton extract_person_poses interface",
        ):
            self._run_flow(triton)

        self.assertEqual(triton.segment_calls, 0)

    def test_legacy_tuple_person_search_result_fails_fast(self):
        async def legacy_tuple(*_args, **_kwargs):
            return "old-reid", 0.9, "old-reid", "buffer"

        with self.assertRaisesRegex(
            TypeError,
            "must return PersonReidSearchResult",
        ):
            self._run_flow(FakeTriton(), search_persons=legacy_tuple)

    def test_missing_offer_feature_interface_fails_fast(self):
        buffer = FakeBuffer()
        buffer.sbs = types.SimpleNamespace()

        with self.assertRaises(AttributeError):
            self._run_flow(
                FakeTriton(),
                buffer=buffer,
                match_face=observed_face_match,
            )

    def test_anonymous_active_reid_match_stays_active_without_enrollment(self):
        async def active_match(*_args, **_kwargs):
            return person_result(
                "active-reid",
                0.90,
                "active-reid",
                "缓存",
                state=REID_STATE_ACTIVE,
            )

        offer_count = 0

        async def unexpected_offer(**_kwargs):
            nonlocal offer_count
            offer_count += 1
            raise AssertionError("anonymous observations must not expand SBS features")

        buffer = FakeBuffer()
        buffer.sbs = types.SimpleNamespace(
            add_feature=buffer.add_person_feature,
            offer_feature=unexpected_offer,
        )
        processor = person_processor(
            triton=FakeTriton(),
            buffer=buffer,
            search_persons=active_match,
        )

        for index in range(3):
            ctx = KeyframeContext(
                sn="CAM001",
                timestamp=1_000 + index * 100,
                object_name=f"CAM001/picture/k-{index}.jpg",
                keyframe_id=f"uid-{index}",
            )
            asyncio.run(
                processor.process_detection(
                    ctx,
                    textured_image(),
                    ctx.keyframe_id,
                    index + 1,
                    ctx.object_name,
                    FakeOSS(),
                )
            )

        self.assertEqual(
            [record["reidStatus"] for record in buffer.recognitions],
            [2, 2, 2],
        )
        self.assertEqual(
            [record["reidState"] for record in buffer.recognitions],
            [REID_STATE_ACTIVE, REID_STATE_ACTIVE, REID_STATE_ACTIVE],
        )
        self.assertEqual(
            [record["reidEnrollmentState"] for record in buffer.recognitions],
            ["", "", ""],
        )
        self.assertEqual(
            [record["reidEnrollmentReason"] for record in buffer.recognitions],
            ["", "", ""],
        )
        self.assertEqual(offer_count, 0)

    def test_pose_failure_and_no_match_fail_closed_before_tbps(self):
        failed_triton = FakeTriton(pose_error=RuntimeError("pose unavailable"))
        unmatched_triton = FakeTriton(poses=[])

        failed = self._run_flow(failed_triton)
        unmatched = self._run_flow(unmatched_triton)

        self.assertEqual(failed.detections[0]["personQualityLevel"], 0)
        self.assertIn("pose_inference_failed", failed.detections[0]["qualityReasons"])
        self.assertEqual(unmatched.detections[0]["personQualityLevel"], 0)
        self.assertIn("pose_not_matched", unmatched.detections[0]["qualityReasons"])
        self.assertEqual(failed_triton.tbps_calls, 0)
        self.assertEqual(unmatched_triton.tbps_calls, 0)

    def test_person_orientation_is_calculated_once_and_shared_by_records(self):
        result = PersonOrientationResult(
            orientation=PersonOrientation.FRONT,
            signed_span_ratio=0.75,
            shoulder_span_ratio=0.8,
            hip_span_ratio=0.7,
            evidence_count=2,
        )
        with patch(
            "core.person_pipeline.observation.estimate_person_orientation",
            return_value=result,
        ) as estimator:
            buffer = self._run_flow(FakeTriton(), match_face=observed_face_match)

        self.assertEqual(estimator.call_count, 1)
        self.assertEqual(buffer.detections[0]["personOrientation"], "front")
        self.assertEqual(buffer.recognitions[0]["personOrientation"], "front")

    def test_on_missing_shared_face_runs_the_existing_rotated_local_callback_once(self):
        person_rect = {"x": 20, "y": 28, "w": 80, "h": 64}
        upright_rect = {"x": 28, "y": 20, "w": 64, "h": 80}
        points = full_body_keypoints(upright_rect)
        center = np.array([60.0, 60.0], dtype=np.float32)
        xy = points[:, :2] - center
        points[:, 0] = center[0] + xy[:, 1]
        points[:, 1] = center[1] - xy[:, 0]
        triton = FakeTriton(
            segments=[segment_result(rect=person_rect)],
            poses=[pose_result(rect=person_rect, keypoints=points)],
        )
        local_calls = []

        async def local_face_match(**kwargs):
            local_calls.append(kwargs)
            return {
                "face_id": "local-face",
                "info": "local fallback",
                "face_orientation": "front",
            }

        try:
            buffer = self._run_flow(
                triton,
                match_face=local_face_match,
                upright_device_sns=frozenset({"CAM001"}),
                person_face_reuse_mode="on",
            )
        except TypeError as error:
            self.fail(f"person pipeline did not pass the frame and binding: {error}")

        self.assertEqual(len(local_calls), 1)
        self.assertEqual(local_calls[0]["person_crop"].shape[:2], (88, 72))
        self.assertEqual(
            local_calls[0]["person_mask"].shape,
            local_calls[0]["person_crop"].shape[:2],
        )
        self.assertEqual(local_calls[0]["pose_keypoints"].shape, (17, 3))
        self.assertFalse(
            np.allclose(local_calls[0]["pose_keypoints"][:, :2], points[:, :2])
        )
        self.assertIn("旋转状态 applied", buffer.detections[0]["info"])
        self.assertEqual(buffer.recognitions[0]["aiFaceId"], "local-face")

    def test_dropped_person_emits_estimated_person_orientation_without_recognition(self):
        result = PersonOrientationResult(
            orientation=PersonOrientation.BACK,
            signed_span_ratio=-0.75,
            evidence_count=2,
        )
        with patch(
            "core.person_pipeline.observation.estimate_person_orientation",
            return_value=result,
        ) as estimator:
            buffer = self._run_flow(
                FakeTriton(segments=[segment_result(mask_pixel_count=1)])
            )

        self.assertEqual(estimator.call_count, 1)
        self.assertEqual(buffer.detections[0]["personOrientation"], "back")
        self.assertEqual(buffer.recognitions, [])

    def test_pose_failure_or_unmatched_pose_emits_unknown_person_orientation(self):
        for name, triton in (
            ("inference_failure", FakeTriton(pose_error=RuntimeError("unavailable"))),
            ("unmatched", FakeTriton(poses=[])),
        ):
            with self.subTest(name=name):
                buffer = self._run_flow(triton)

                self.assertEqual(
                    buffer.detections[0]["personOrientation"],
                    "unknown",
                )

    def test_preparation_failure_retains_estimated_person_orientation(self):
        result = PersonOrientationResult(
            orientation=PersonOrientation.SIDE,
            signed_span_ratio=0.0,
            evidence_count=2,
        )
        with (
            patch(
                "core.person_pipeline.observation.estimate_person_orientation",
                return_value=result,
            ) as estimator,
            patch(
                "core.person_pipeline.pipeline.PersonPreparationService.prepare_visual",
                new=AsyncMock(return_value=None),
            ),
        ):
            buffer = self._run_flow(FakeTriton())

        self.assertEqual(estimator.call_count, 1)
        self.assertEqual(buffer.detections[0]["personOrientation"], "side")
        self.assertEqual(buffer.recognitions, [])

    def test_dropped_person_analyzes_orientation_once_without_recognition(self):
        calls = []

        async def orientation_only_match(**kwargs):
            calls.append(kwargs["enable_recognition"])
            return {
                "face_id": "0",
                "info": "orientation only",
                "score": 0.0,
                "face_orientation": "SIDE",
            }

        triton = FakeTriton(segments=[segment_result(mask_pixel_count=1)])
        buffer = self._run_flow(triton, match_face=orientation_only_match)

        self.assertEqual(calls, [False])
        self.assertEqual(len(buffer.detections), 1)
        self.assertEqual(buffer.detections[0]["faceOrientation"], "side")
        self.assertIs(type(buffer.detections[0]["faceOrientation"]), str)
        self.assertEqual(buffer.recognitions, [])

    def test_recognized_person_reuses_one_orientation_for_both_records(self):
        calls = []

        async def recognized_face_match(**kwargs):
            calls.append(kwargs.get("enable_recognition", "omitted"))
            return {
                "face_id": "face-a",
                "info": "face matched",
                "score": 0.9,
                "face_orientation": FaceOrientation.FRONT,
            }

        buffer = self._run_flow(FakeTriton(), match_face=recognized_face_match)

        self.assertEqual(calls, [True])
        self.assertEqual(len(buffer.detections), 1)
        self.assertEqual(len(buffer.recognitions), 1)
        self.assertEqual(buffer.detections[0]["faceOrientation"], "front")
        self.assertEqual(buffer.recognitions[0]["faceOrientation"], "front")
        self.assertIs(type(buffer.detections[0]["faceOrientation"]), str)
        self.assertIs(type(buffer.recognitions[0]["faceOrientation"]), str)

    def test_required_enable_recognition_callback_receives_quality_mode(self):
        calls = []

        async def required_recognition_match(
            person_crop,
            person_mask,
            pose_keypoints,
            device_id,
            timestamp,
            triton_client,
            search_callback,
            normalize_vector,
            similarity_threshold,
            logger,
            *,
            enable_recognition,
            image_executor,
        ):
            self.assertIsInstance(image_executor, InlineImageExecutor)
            calls.append(enable_recognition)
            return {
                "face_id": "required-face" if enable_recognition else "0",
                "info": "recognized" if enable_recognition else "orientation only",
                "score": 0.9 if enable_recognition else 0.0,
                "face_orientation": "front",
            }

        recognition_only_points = full_body_keypoints()
        recognition_only_points[:5, 2] = 0.0
        cases = (
            ("drop", FakeTriton(segments=[segment_result(mask_pixel_count=1)]), 0, False),
            (
                "recognition",
                FakeTriton(poses=[pose_result(keypoints=recognition_only_points)]),
                1,
                True,
            ),
            ("enroll", FakeTriton(), 2, True),
        )

        for name, triton, expected_level, expected_recognition in cases:
            with self.subTest(name=name):
                buffer = self._run_flow(
                    triton,
                    match_face=required_recognition_match,
                )

                self.assertEqual(calls[-1], expected_recognition)
                self.assertEqual(
                    buffer.detections[0]["personQualityLevel"], expected_level
                )
                self.assertEqual(buffer.detections[0]["faceOrientation"], "front")
                if expected_recognition:
                    self.assertEqual(
                        buffer.recognitions[0]["aiFaceId"], "required-face"
                    )
                else:
                    self.assertEqual(buffer.recognitions, [])

    def test_missing_face_orientation_normalizes_unknown(self):
        calls = []

        async def face_match(**kwargs):
            calls.append(kwargs["enable_recognition"])
            return {
                "face_id": "face-a",
                "info": "face matched",
                "score": 0.9,
            }

        buffer = self._run_flow(FakeTriton(), match_face=face_match)

        self.assertEqual(calls, [True])
        self.assertEqual(buffer.detections[0]["faceOrientation"], "unknown")
        self.assertEqual(buffer.recognitions[0]["faceOrientation"], "unknown")
        self.assertEqual(buffer.recognitions[0]["aiFaceId"], "face-a")

    def test_callback_without_recognition_mode_fails_closed(self):
        calls = []
        logger = CapturingLogger()

        async def unsupported_match(
            person_crop,
            person_mask,
            pose_keypoints,
            device_id,
            timestamp,
            triton_client,
            search_callback,
            normalize_vector,
            similarity_threshold,
            logger,
        ):
            calls.append(device_id)
            return {
                "face_id": "unexpected-face",
                "info": "unsupported callback",
                "score": 0.91,
            }

        triton = FakeTriton(segments=[segment_result(mask_pixel_count=1)])
        buffer = self._run_flow(
            triton,
            match_face=unsupported_match,
            logger=logger,
        )

        self.assertEqual(calls, [])
        self.assertEqual(len(logger.exceptions), 1)
        self.assertIn("enable_recognition", logger.exceptions[0])
        self.assertEqual(buffer.detections[0]["faceOrientation"], "unknown")
        self.assertEqual(buffer.recognitions, [])

    def test_thirteen_keypoints_with_variable_aspect_crop_can_build_gallery(self):
        rect = {"x": 10, "y": 30, "w": 100, "h": 100}
        points = full_body_keypoints(rect)
        points[[1, 2, 3, 4], 2] = 0.0

        class HighPaddingTriton(FakeTriton):
            async def extract_person_segments(self, image):
                return [segment_result(rect=rect)]

        triton = HighPaddingTriton(poses=[pose_result(rect=rect, keypoints=points)])

        buffer = self._run_flow(triton)

        detection = buffer.detections[0]
        self.assertEqual(detection["visibleKeypointCount"], 13)
        self.assertEqual(detection["personQualityLevel"], 2)
        self.assertEqual(detection["personQualityAction"], PERSON_ACTION_BUILD_GALLERY)
        self.assertNotIn("crop_padding_ratio_high", detection["qualityReasons"])
        self.assertNotIn("crop_padding_ratio_too_high", detection["qualityReasons"])
        self.assertEqual(detection["letterboxPaddingRatio"], 0.0)
        self.assertEqual(triton.sbs_calls, 1)
        self.assertEqual(triton.tbps_calls, 1)
        self.assertEqual(buffer.features, [])
        self.assertEqual(len(buffer.clip_features), 1)
        self.assertEqual(buffer.clip_features[0]["face_id"], "0")
        self.assertEqual(buffer.clip_features[0]["re_id"], "0")

    def test_full_three_quarter_and_high_quality_half_build_gallery(self):
        full_triton = FakeTriton()
        three_quarter_upper_points = full_body_keypoints()
        three_quarter_upper_points[[15, 16], 2] = 0.0
        three_quarter_upper_triton = FakeTriton(
            poses=[pose_result(keypoints=three_quarter_upper_points)]
        )
        half_points = full_body_keypoints()
        half_points[13:, 2] = 0.0
        half_triton = FakeTriton(poses=[pose_result(keypoints=half_points)])

        full = self._run_flow(full_triton, match_face=observed_face_match)
        three_quarter_upper = self._run_flow(
            three_quarter_upper_triton,
            match_face=observed_face_match,
        )
        half = self._run_flow(half_triton, match_face=observed_face_match)

        self.assertEqual(full.detections[0]["personQualityLevel"], 2)
        self.assertEqual(len(full.features), 1)
        self.assertEqual(len(full.clip_features), 1)
        self.assertEqual(full.features[0]["body_completeness"], BODY_FULL)
        self.assertNotIn("body_completeness", full.clip_features[0])
        self.assertEqual(
            three_quarter_upper.detections[0]["bodyCompleteness"],
            BODY_THREE_QUARTER_UPPER,
        )
        self.assertEqual(three_quarter_upper.detections[0]["personQualityLevel"], 2)
        self.assertEqual(
            three_quarter_upper.features[0]["body_completeness"],
            BODY_THREE_QUARTER_UPPER,
        )
        self.assertNotIn(
            "body_completeness",
            three_quarter_upper.clip_features[0],
        )
        self.assertEqual(full_triton.sbs_calls, 1)
        self.assertEqual(full_triton.tbps_calls, 1)
        self.assertEqual(len(full.detections[0]["personKeypoint"]["points"]), 17)
        self.assertNotIn("personKeypoint", full.recognitions[0])
        self.assertEqual(half.detections[0]["personQualityLevel"], 2)
        self.assertTrue(half.detections[0]["isHalfBody"])
        self.assertEqual(half.features[0]["body_completeness"], BODY_HALF_UPPER)
        self.assertEqual(len(half.features), 1)
        self.assertEqual(len(half.clip_features), 1)
        self.assertEqual(half_triton.sbs_calls, 1)
        self.assertEqual(half_triton.tbps_calls, 1)
        self.assertEqual(half.recognitions[0]["reidState"], REID_STATE_ACTIVE)

    def test_structured_half_upper_visibility_controls_gallery_building(self):
        cases = [
            ("twelve", [4], 12, True),
            ("eleven", [3, 4], 11, True),
            ("ten", [2, 3, 4], 10, False),
        ]

        for name, hidden_head_points, visible_count, eligible in cases:
            with self.subTest(name=name):
                points = full_body_keypoints()
                points[13:, 2] = 0.0
                points[hidden_head_points, 2] = 0.0
                triton = FakeTriton(poses=[pose_result(keypoints=points)])

                buffer = self._run_flow(triton)

                detection = buffer.detections[0]
                recognition = buffer.recognitions[0]
                self.assertEqual(
                    detection["visibleKeypointCount"], visible_count
                )
                self.assertEqual(detection["torsoKeypointCount"], 4)
                self.assertEqual(detection["shoulderHipChainCount"], 2)
                self.assertIs(detection["halfBodyPoseEligible"], eligible)
                self.assertEqual(
                    recognition["halfBodyPoseEligible"], eligible
                )
                self.assertEqual(
                    detection["personQualityLevel"], 2 if eligible else 1
                )
                self.assertEqual(triton.sbs_calls, 1)
                self.assertEqual(triton.tbps_calls, 1 if eligible else 0)

    def test_mild_edge_second_person_can_still_build_gallery(self):
        target_rect = {"x": 40, "y": 20, "w": 40, "h": 120}
        other_rect = {"x": 79, "y": 20, "w": 35, "h": 120}
        triton = FakeTriton(
            segments=[segment_result(target_rect), segment_result(other_rect)],
            poses=[pose_result(target_rect), pose_result(other_rect)],
        )

        buffer = self._run_flow(triton)

        detection = buffer.detections[0]
        recognition = buffer.recognitions[0]
        self.assertEqual(detection["personQualityLevel"], 2)
        self.assertEqual(detection["personQualityAction"], "build_gallery")
        self.assertTrue(detection["hasSecondPerson"])
        self.assertAlmostEqual(detection["foreignPersonRatio"], 0.075)
        self.assertEqual(detection["foreignPersonCoreRatio"], 0.0)
        self.assertTrue(detection["foreignPersonEdgeDominant"])
        self.assertIn("核心区占比 0.000", detection["info"])
        self.assertIn("证据来源 mask", detection["info"])
        self.assertEqual(detection["foreignPersonEvidenceSource"], "mask")
        self.assertIn(
            "crop_contains_other_person_mild",
            detection["qualityReasons"],
        )
        self.assertEqual(recognition["foreignPersonRatio"], 0.075)
        self.assertEqual(triton.sbs_calls, 2)
        self.assertEqual(triton.tbps_calls, 2)

    def test_same_frame_non_overlapping_people_reselect_distinct_existing_reids(self):
        calls = []

        async def search_persons(
            embedding,
            query_body_completeness,
            device_ids,
            face_id=None,
            excluded_reids=None,
            **_kwargs,
        ):
            excluded = frozenset(excluded_reids or ())
            calls.append(excluded)
            if "shared-reid" in excluded:
                return person_result(
                    "alternate-reid", 0.91, "alternate-reid", "鍚戦噺搴?"
                )
            return person_result(
                "shared-reid", 0.95, "shared-reid", "鍚戦噺搴?"
            )

        buffer = self._run_flow(
            self._two_person_triton(),
            search_persons=search_persons,
        )

        self.assertEqual(calls, [frozenset(), frozenset({"shared-reid"})])
        self.assertEqual(
            [record["reid"] for record in buffer.recognitions],
            ["shared-reid", "alternate-reid"],
        )
        self.assertEqual([record["reidStatus"] for record in buffer.recognitions], [2, 2])
        self.assertEqual(
            [record["reidDecision"] for record in buffer.recognitions],
            ["MATCH_EXISTING", "MATCH_EXISTING"],
        )
        self.assertEqual(
            [record["sameFrameExcludedReidCount"] for record in buffer.recognitions],
            [0, 1],
        )
        self.assertEqual(buffer.features, [])
        self.assertEqual(
            [feature["re_id"] for feature in buffer.clip_features],
            ["shared-reid", "alternate-reid"],
        )
        self.assertIn("同帧互斥已排除 1 个 reid", buffer.recognitions[1]["info"])

    def test_same_frame_anonymous_level_two_without_alternative_stays_unassigned(self):
        async def search_persons(
            embedding,
            query_body_completeness,
            device_ids,
            face_id=None,
            excluded_reids=None,
            **_kwargs,
        ):
            if "shared-reid" in (excluded_reids or ()):
                return person_result()
            return person_result(
                "shared-reid", 0.95, "shared-reid", "鍚戦噺搴?"
            )

        buffer = self._run_flow(
            self._two_person_triton(),
            search_persons=search_persons,
        )

        self.assertEqual(
            [record["reid"] for record in buffer.recognitions],
            ["shared-reid", "0"],
        )
        self.assertEqual([record["reidStatus"] for record in buffer.recognitions], [2, 0])
        self.assertEqual(
            [record["reidState"] for record in buffer.recognitions],
            [REID_STATE_ACTIVE, REID_STATE_UNASSIGNED],
        )
        self.assertEqual(
            [record["reidDecision"] for record in buffer.recognitions],
            ["MATCH_EXISTING", "UNASSIGNED"],
        )
        self.assertEqual(buffer.recognitions[1]["reidSource"], "")
        self.assertIs(buffer.recognitions[1]["isNewReid"], False)
        self.assertEqual(buffer.recognitions[1]["sameFrameExcludedReidCount"], 1)
        self.assertEqual(buffer.features, [])
        self.assertEqual(
            [feature["re_id"] for feature in buffer.clip_features],
            ["shared-reid", "0"],
        )

    def test_same_frame_level_one_without_alternative_stays_unmatched(self):
        second_rect = {"x": 75, "y": 20, "w": 35, "h": 120}
        second_keypoints = full_body_keypoints(second_rect)
        second_keypoints[[2, 3, 4, 13, 14, 15, 16], 2] = 0.0

        async def search_persons(
            embedding,
            query_body_completeness,
            device_ids,
            face_id=None,
            excluded_reids=None,
            **_kwargs,
        ):
            if "shared-reid" in (excluded_reids or ()):
                return person_result()
            return person_result(
                "shared-reid", 0.95, "shared-reid", "鍚戦噺搴?"
            )

        buffer = self._run_flow(
            self._two_person_triton(second_keypoints=second_keypoints),
            search_persons=search_persons,
        )

        self.assertEqual(
            [record["reid"] for record in buffer.recognitions],
            ["shared-reid", "0"],
        )
        self.assertEqual([record["reidStatus"] for record in buffer.recognitions], [2, 0])
        self.assertEqual(buffer.recognitions[1]["reidDecision"], "UNASSIGNED")
        self.assertIs(buffer.recognitions[1]["isNewReid"], False)
        self.assertIs(buffer.recognitions[1]["halfBodyPoseEligible"], False)
        self.assertEqual(buffer.recognitions[1]["sameFrameExcludedReidCount"], 1)
        self.assertEqual(buffer.features, [])
        self.assertEqual(
            [feature["re_id"] for feature in buffer.clip_features],
            ["shared-reid"],
        )

    def test_same_frame_exclusion_never_falls_back_outside_known_face(self):
        calls = []

        async def fake_face_match(**kwargs):
            return {"face_id": "face-1", "info": "face bound", "score": 0.9}

        async def search_persons(
            embedding,
            query_body_completeness,
            device_ids,
            face_id=None,
            excluded_reids=None,
            **_kwargs,
        ):
            excluded = frozenset(excluded_reids or ())
            calls.append((face_id, excluded))
            if not excluded:
                return person_result(
                    "shared-reid", 0.95, "shared-reid", "鍚戦噺搴?"
                )
            if face_id == "face-1":
                return person_result()
            return person_result(
                "alternate-reid", 0.91, "alternate-reid", "鍚戦噺搴?"
            )

        buffer = self._run_flow(
            self._two_person_triton(),
            search_persons=search_persons,
            match_face=fake_face_match,
        )

        self.assertEqual(
            calls,
            [
                ("face-1", frozenset()),
                ("face-1", frozenset({"shared-reid"})),
            ],
        )
        assigned_reids = [record["reid"] for record in buffer.recognitions]
        self.assertEqual(assigned_reids[0], "shared-reid")
        self.assertNotIn(assigned_reids[1], {"shared-reid", "alternate-reid", "0"})
        self.assertEqual(
            [record["reidStatus"] for record in buffer.recognitions],
            [2, 1],
        )

    def test_lower_three_quarter_recognizes_existing_reid_without_insert(self):
        points = full_body_keypoints()
        points[:5, 2] = 0.0
        triton = FakeTriton(poses=[pose_result(keypoints=points)])
        query_types = []

        async def match_existing(
            embedding,
            query_body_completeness,
            device_ids,
            face_id=None,
            **_kwargs,
        ):
            query_types.append(query_body_completeness)
            return person_result(
                "existing-reid", 0.9, "existing-reid", "向量库"
            )

        buffer = self._run_flow(triton, search_persons=match_existing)

        self.assertEqual(buffer.detections[0]["bodyCompleteness"], BODY_THREE_QUARTER_LOWER)
        self.assertEqual(buffer.detections[0]["personQualityLevel"], 1)
        self.assertEqual(query_types, [BODY_THREE_QUARTER_LOWER])
        self.assertEqual(buffer.recognitions[0]["reid"], "existing-reid")
        self.assertEqual(buffer.recognitions[0]["reidStatus"], 2)
        self.assertEqual(len(buffer.features), 0)
        self.assertEqual(len(buffer.clip_features), 0)
        self.assertEqual(triton.sbs_calls, 1)
        self.assertEqual(triton.tbps_calls, 0)




class PersonReidFacePriorityTests(unittest.TestCase):
    def _run_flow(self, face_id, search_results):
        search_face_ids = []

        async def fake_face_match(**kwargs):
            self.assertEqual(kwargs["person_mask"].shape, kwargs["person_crop"].shape[:2])
            return {"face_id": face_id, "info": "人脸绑定测试", "score": 0.8}

        async def search_persons(
            embedding,
            query_body_completeness,
            device_ids,
            face_id=None,
            **_kwargs,
        ):
            self.assertEqual(query_body_completeness, BODY_FULL)
            search_face_ids.append(face_id)
            return search_results[face_id]

        buffer = FakeBuffer()
        processor = person_processor(
            triton=FakeTriton(),
            buffer=buffer,
            search_persons=search_persons,
            match_face=fake_face_match,
        )
        ctx = KeyframeContext(
            sn="CAM001",
            timestamp=1,
            object_name="CAM001/picture/k.jpg",
            keyframe_id="uid",
        )
        asyncio.run(
            processor.process_detection(
                ctx,
                textured_image(),
                "uid",
                1,
                ctx.object_name,
                FakeOSS(),
            )
        )
        return buffer, search_face_ids

    def test_face_id_hit_precedes_global_reid_search(self):
        buffer, search_face_ids = self._run_flow(
            "face-1",
            {
                "face-1": person_result(
                    "face-reid",
                    0.98,
                    "face-reid",
                    "向量库",
                    canonical_face_id="face-1",
                )
            },
        )

        self.assertEqual(search_face_ids, ["face-1"])
        self.assertEqual(buffer.recognitions[0]["reid"], "face-reid")
        self.assertEqual(buffer.recognitions[0]["reidState"], "ACTIVE")
        self.assertEqual(buffer.recognitions[0]["reidDecision"], "MATCH_EXISTING")
        self.assertEqual(buffer.recognitions[0]["canonicalFaceId"], "face-1")
        self.assertIn("仅查询 faceId=face-1 的纯净外观簇", buffer.recognitions[0]["info"])

    def test_face_id_miss_creates_new_reid_without_global_fallback(self):
        buffer, search_face_ids = self._run_flow(
            "face-1",
            {
                "face-1": person_result(score=0.40),
                None: person_result(
                    "global-reid", 0.97, "global-reid", "向量库"
                ),
            },
        )

        self.assertEqual(search_face_ids, ["face-1"])
        self.assertNotEqual(buffer.recognitions[0]["reid"], "global-reid")
        self.assertNotEqual(buffer.recognitions[0]["reid"], "0")
        self.assertEqual(buffer.recognitions[0]["reidStatus"], 1)
        self.assertEqual(buffer.recognitions[0]["reidState"], "ACTIVE")
        self.assertEqual(buffer.recognitions[0]["reidDecision"], "CREATE_NEW")
        self.assertEqual(buffer.recognitions[0]["reidSource"], "NEW")
        self.assertIs(buffer.recognitions[0]["isNewReid"], True)
        self.assertEqual(buffer.recognitions[0]["canonicalFaceId"], "face-1")
        self.assertIn("仅查询 faceId=face-1 的纯净外观簇", buffer.recognitions[0]["info"])

    def test_unmatched_face_uses_global_reid_only(self):
        buffer, search_face_ids = self._run_flow(
            "0",
            {
                None: person_result(
                    "global-reid", 0.97, "global-reid", "向量库"
                )
            },
        )

        self.assertEqual(search_face_ids, [None])
        self.assertEqual(buffer.recognitions[0]["reid"], "global-reid")
        self.assertEqual(buffer.recognitions[0]["reidState"], "ACTIVE")

    def test_anonymous_miss_does_not_create_reid_or_inherit_candidate_face(self):
        rejected_candidate_face_id = "face-from-rejected-candidate"
        buffer, search_face_ids = self._run_flow(
            "0",
            {
                None: PersonReidSearchResult(
                    reid=None,
                    score=0.7932420969009399,
                    vector_hit_reid="rejected-vector-reid",
                    source="milvus",
                    canonical_face_id=rejected_candidate_face_id,
                    reason="ambiguous_margin",
                    support_count=2,
                    top1_score=0.7740814089775085,
                    mean_score=0.7647985517978668,
                    class_score=0.7703682661056519,
                    runner_up_score=0.7629443407058716,
                    margin=0.007423925399780318,
                    required_margin=0.08,
                    search_mode=REID_SEARCH_MODE_ANONYMOUS_GLOBAL,
                )
            },
        )

        recognition = buffer.recognitions[0]

        self.assertEqual(search_face_ids, [None])
        self.assertEqual(recognition["reid"], "0")
        self.assertEqual(recognition["reidStatus"], 0)
        self.assertEqual(recognition["reidState"], REID_STATE_UNASSIGNED)
        self.assertEqual(recognition["reidEnrollmentState"], "")
        self.assertEqual(recognition["reidEnrollmentReason"], "")
        self.assertEqual(recognition["reidDecision"], "UNASSIGNED")
        self.assertEqual(recognition["reidDecisionReason"], "ambiguous_margin")
        self.assertEqual(recognition["canonicalFaceId"], "0")
        self.assertEqual(recognition["vectorHitReid"], "rejected-vector-reid")
        self.assertEqual(buffer.features, [])
        self.assertEqual(len(buffer.clip_features), 1)
        self.assertEqual(buffer.clip_features[0]["face_id"], "0")
        self.assertEqual(buffer.clip_features[0]["re_id"], "0")


class PersonReidStorageFilterTests(unittest.TestCase):
    def test_milvus_person_search_adds_exact_face_id_filter(self):
        package_name = "person_gallery_filter_under_test"
        module_name = f"{package_name}.SBS_feature"
        tbps_module_name = f"{package_name}.TBPS_feature"
        person_feature_module_name = f"{package_name}.person_feature"
        picture_module_name = f"{package_name}.picture_feature"
        original_pymilvus = sys.modules.get("pymilvus")

        class FakeMilvusVectorDB:
            @staticmethod
            def _to_milvus_value(value):
                return list(value)

            @staticmethod
            def _hit_entity(hit):
                return hit.get("entity") or hit

            @staticmethod
            def _hit_score(hit):
                return float(hit.get("distance", hit.get("score", 0.0)))

            @staticmethod
            def _records_from_columns(field_names, columns):
                batch_size = len(columns[0]) if isinstance(columns[0], list) else 1
                return [
                    {
                        field_name: (
                            columns[column_index][row_index]
                            if isinstance(columns[column_index], list)
                            else columns[column_index]
                        )
                        for column_index, field_name in enumerate(field_names)
                    }
                    for row_index in range(batch_size)
                ]

            def _embedding_index_params(self):
                return {"field_name": "embedding"}

        pymilvus_module = types.ModuleType("pymilvus")
        pymilvus_module.DataType = types.SimpleNamespace(
            INT64="INT64",
            FLOAT="FLOAT",
            FLOAT_VECTOR="FLOAT_VECTOR",
            VARCHAR="VARCHAR",
            JSON="JSON",
        )
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []
        picture_module = types.ModuleType(picture_module_name)
        picture_module.MilvusVectorDB = FakeMilvusVectorDB
        sys.modules["pymilvus"] = pymilvus_module
        sys.modules[package_name] = package_module
        sys.modules[picture_module_name] = picture_module
        try:
            zilliz_path = Path(__file__).resolve().parents[1] / "db" / "zilliz"
            person_feature_spec = importlib.util.spec_from_file_location(
                person_feature_module_name,
                zilliz_path / "person_feature.py",
            )
            person_feature_module = importlib.util.module_from_spec(
                person_feature_spec
            )
            sys.modules[person_feature_module_name] = person_feature_module
            person_feature_spec.loader.exec_module(person_feature_module)

            module_path = zilliz_path / "SBS_feature.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            module.logger = types.SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                error=lambda *_args, **_kwargs: None,
            )

            tbps_spec = importlib.util.spec_from_file_location(
                tbps_module_name,
                zilliz_path / "TBPS_feature.py",
            )
            tbps_module = importlib.util.module_from_spec(tbps_spec)
            sys.modules[tbps_module_name] = tbps_module
            tbps_spec.loader.exec_module(tbps_module)

            class FakeMilvusClient:
                def __init__(self):
                    self.description = {
                        "auto_id": False,
                        "fields": [
                            {
                                "name": name,
                                **(
                                    {"type": "VARCHAR", "auto_id": False}
                                    if name == "id"
                                    else {}
                                ),
                            }
                            for name in (
                                "id",
                                "faceId",
                                "reId",
                                "bodyCompleteness",
                                "cropUrl",
                                "qualityScore",
                                "embedding",
                                "deviceId",
                                "trackId",
                                "keyframeId",
                                "personRect",
                                "timestamp",
                            )
                        ]
                    }
                    self.search_calls = []
                    self.bind_records = []
                    self.upsert_calls = []
                    self.insert_calls = []
                    self.stored_hits_by_body = {
                        BODY_FULL: [
                            {
                                "entity": {
                                    "reId": "full-reid",
                                    "bodyCompleteness": BODY_FULL,
                                },
                                "distance": 0.81,
                            },
                            {
                                "entity": {
                                    "reId": "incompatible",
                                    "bodyCompleteness": BODY_HALF_UPPER,
                                },
                                "distance": 0.99,
                            },
                        ],
                        BODY_THREE_QUARTER_UPPER: [
                            {
                                "entity": {
                                    "reId": "upper-reid",
                                    "bodyCompleteness": BODY_THREE_QUARTER_UPPER,
                                },
                                "distance": 0.67,
                            },
                        ],
                        BODY_HALF_UPPER: [
                            {
                                "entity": {
                                    "reId": "half-reid",
                                    "bodyCompleteness": BODY_HALF_UPPER,
                                },
                                "distance": 0.68,
                            },
                        ],
                    }

                @staticmethod
                def gallery_body_type(filter_expr):
                    for body_type in (
                        BODY_FULL,
                        BODY_THREE_QUARTER_UPPER,
                        BODY_HALF_UPPER,
                    ):
                        if f'bodyCompleteness == "{body_type}"' in filter_expr:
                            return body_type
                    raise AssertionError(f"missing single-body filter: {filter_expr}")

                def search(self, **kwargs):
                    self.search_calls.append(kwargs)
                    body_type = self.gallery_body_type(kwargs["filter"])
                    hits = list(self.stored_hits_by_body[body_type])
                    marker = "reId not in ["
                    if marker in kwargs["filter"]:
                        excluded_json = (
                            kwargs["filter"].split(marker, 1)[1].split("]", 1)[0]
                        )
                        excluded_reids = set(json.loads(f"[{excluded_json}]"))
                        hits = [
                            hit
                            for hit in hits
                            if hit["entity"].get("reId") not in excluded_reids
                        ]
                    return [hits[: kwargs["limit"]]]

                def describe_collection(self, **kwargs):
                    return self.description

                def query(self, **kwargs):
                    return [dict(record) for record in self.bind_records]

                def upsert(self, **kwargs):
                    data = [dict(record) for record in kwargs["data"]]
                    self.upsert_calls.append(data)
                    by_id = {str(record["id"]): record for record in data}
                    self.bind_records = [
                        by_id.get(str(record["id"]), record)
                        for record in self.bind_records
                    ]
                    return {"upsert_count": len(data)}

                def insert(self, **kwargs):
                    self.insert_calls.append(kwargs["data"])
                    return {"insert_count": len(kwargs["data"])}

            client = FakeMilvusClient()
            gallery = module.SbsReidGallery.__new__(module.SbsReidGallery)
            gallery.collection_name = "alogSBS"
            gallery.client = client
            gallery.search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            fixed_now = datetime(2026, 7, 21, 12, 0, 0)
            module.datetime = types.SimpleNamespace(now=lambda: fixed_now)
            expected_cutoff = int(
                (
                    fixed_now
                    - module.timedelta(days=algorithm_config.reid.classification.query_window_days)
                ).timestamp()
                * 1000
            )
            candidates, highest_score, highest_reid, _ = (
                asyncio.run(
                    gallery.asearch_similar_person_candidates_with_hit_info(
                        [0.1] * 2048,
                        query_body_completeness=BODY_THREE_QUARTER_LOWER,
                        device_ids=["CAM001"],
                        face_id='face-"1',
                    )
                )
            )

            self.assertEqual(len(client.search_calls), 2)
            for call in client.search_calls:
                self.assertIn(f"timestamp >= {expected_cutoff}", call["filter"])
                self.assertIn('deviceId == "CAM001"', call["filter"])
                self.assertIn('faceId == "face-\\"1"', call["filter"])
                self.assertEqual(call["limit"], 50)
                self.assertEqual(
                    call["output_fields"],
                    [
                        "id",
                        "reId",
                        "faceId",
                        "bodyCompleteness",
                        "deviceId",
                    ],
                )
                self.assertEqual(call["group_by_field"], "reId")
                self.assertEqual(call["group_size"], 3)
            self.assertIn(
                'bodyCompleteness == "full_body"',
                client.search_calls[0]["filter"],
            )
            self.assertNotIn(
                "three_quarter_upper",
                client.search_calls[0]["filter"],
            )
            self.assertIn(
                'bodyCompleteness == "three_quarter_upper"',
                client.search_calls[1]["filter"],
            )
            self.assertNotIn(
                'bodyCompleteness == "full_body"',
                client.search_calls[1]["filter"],
            )
            self.assertEqual(
                [candidate.reid for candidate in candidates],
                ["full-reid", "upper-reid"],
            )
            self.assertEqual(highest_score, 0.81)
            self.assertEqual(highest_reid, "full-reid")

            replay_timestamp = 1_700_000_000_000
            replay_filter = gallery._build_person_reid_base_filter(
                query_timestamp=replay_timestamp,
            )
            expected_replay_cutoff = replay_timestamp - 7 * 24 * 60 * 60 * 1000
            self.assertIn(f"timestamp >= {expected_replay_cutoff}", replay_filter)

            client.search_calls.clear()
            excluded_candidates, excluded_score, excluded_reid, _ = (
                asyncio.run(
                    gallery.asearch_similar_person_candidates_with_hit_info(
                        [0.1] * 2048,
                        query_body_completeness=BODY_FULL,
                        face_id='face-"1',
                        excluded_reids={"full-reid", 'quoted-"id'},
                    )
                )
            )
            self.assertEqual(
                [candidate.reid for candidate in excluded_candidates],
                ["upper-reid", "half-reid"],
            )
            self.assertEqual((excluded_score, excluded_reid), (0.68, "half-reid"))
            for call in client.search_calls:
                self.assertIn('faceId == "face-\\"1"', call["filter"])
                self.assertIn(
                    'reId not in ["full-reid", "quoted-\\"id"]',
                    call["filter"],
                )

            gallery._validate_existing_schema()
            gallery._prototype_update_lock = threading.RLock()
            gallery._insert_prototype(
                faceId="face-a",
                reId="appearance-reid",
                bodyCompleteness=BODY_FULL,
                cropUrl="CAM001/person_crop/person.jpg",
                embedding=[1.0] + [0.0] * 2047,
                deviceId="CAM001",
                trackId=0,
                keyframeId=2,
                personRect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=2_000,
                qualityScore=9.0,
            )
            inserted_record = client.insert_calls[-1][0]
            inserted_id = inserted_record["id"]
            self.assertEqual(len(inserted_id), 36)
            self.assertEqual(inserted_id.count("-"), 4)
            self.assertEqual(
                inserted_record["cropUrl"],
                "CAM001/person_crop/person.jpg",
            )
            with self.assertRaisesRegex(ValueError, "faceId"):
                gallery._insert_prototype(
                    faceId="0",
                    reId="missing-face-reid",
                    bodyCompleteness=BODY_FULL,
                    cropUrl="",
                    embedding=[1.0] + [0.0] * 2047,
                    deviceId="CAM001",
                    trackId=0,
                    keyframeId=3,
                    personRect={"x": 1, "y": 2, "w": 3, "h": 4},
                    timestamp=3_000,
                    qualityScore=9.0,
                )

            client.description = {
                "auto_id": True,
                "fields": [
                    {
                        "name": name,
                        **(
                            {"type": "INT64", "auto_id": True}
                            if name == "id"
                            else {}
                        ),
                    }
                    for name in module.SBS_REQUIRED_FIELDS
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "caller-generated"):
                gallery._validate_existing_schema()

            client.description = {"fields": [{"name": "reId"}]}
            with self.assertRaisesRegex(RuntimeError, "fresh alogSBS schema"):
                gallery._validate_existing_schema()

            client.description = {
                "auto_id": False,
                "fields": [
                    {
                        "name": name,
                        **(
                            {"type": "VARCHAR", "auto_id": False}
                            if name == "id"
                            else {}
                        ),
                    }
                    for name in (*module.SBS_REQUIRED_FIELDS, "prototypeType")
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "unexpected=.*prototypeType"):
                gallery._validate_existing_schema()

            client.search_calls.clear()
            full_candidates, _, _, _ = asyncio.run(
                gallery.asearch_similar_person_candidates_with_hit_info(
                    [0.1] * 2048,
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                )
            )
            self.assertEqual(len(client.search_calls), 3)
            self.assertEqual(
                [candidate.body_completeness for candidate in full_candidates],
                [BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER],
            )
            self.assertEqual(
                [candidate.threshold for candidate in full_candidates],
                [0.56, 0.53, 0.55],
            )
            self.assertTrue(
                all(call["limit"] == 50 for call in client.search_calls)
            )

            client.search_calls.clear()
            half_candidates, _, _, _ = asyncio.run(
                gallery.asearch_similar_person_candidates_with_hit_info(
                    [0.1] * 2048,
                    query_body_completeness=BODY_HALF_UPPER,
                    device_ids=["CAM001"],
                )
            )
            self.assertEqual(len(client.search_calls), 3)
            self.assertEqual(
                [candidate.body_completeness for candidate in half_candidates],
                [BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER],
            )
            self.assertEqual(
                [candidate.threshold for candidate in half_candidates],
                [0.55, 0.55, 0.56],
            )

            client.search_calls.clear()
            client.stored_hits_by_body = {
                BODY_FULL: [
                    {
                        "entity": {
                            "reId": f"crowded-{index}",
                            "bodyCompleteness": BODY_FULL,
                        },
                        "distance": 0.90,
                    }
                    for index in range(125)
                ],
                BODY_THREE_QUARTER_UPPER: [
                    {
                        "entity": {
                            "reId": "upper-survives",
                            "bodyCompleteness": BODY_THREE_QUARTER_UPPER,
                        },
                        "distance": 0.81,
                    }
                ],
                BODY_HALF_UPPER: [],
            }
            crowded_candidates, _, _, _ = (
                asyncio.run(
                    gallery.asearch_similar_person_candidates_with_hit_info(
                        [0.1] * 2048,
                        query_body_completeness=BODY_FULL,
                    )
                )
            )
            self.assertEqual(len(crowded_candidates), 51)
            self.assertIn(
                "upper-survives",
                [candidate.reid for candidate in crowded_candidates],
            )

            client.stored_hits_by_body = {
                BODY_FULL: [
                    {
                        "entity": {
                            "reId": "raw-highest-full",
                            "bodyCompleteness": BODY_FULL,
                        },
                            "distance": 0.65,
                    }
                ],
                BODY_THREE_QUARTER_UPPER: [
                    {
                        "entity": {
                            "reId": "accepted-upper",
                            "bodyCompleteness": BODY_THREE_QUARTER_UPPER,
                        },
                        "distance": 0.64,
                    }
                ],
                BODY_HALF_UPPER: [
                    {
                        "entity": {
                            "reId": "exact-half-boundary",
                            "bodyCompleteness": BODY_HALF_UPPER,
                        },
                        "distance": 0.55,
                    }
                ],
            }
            threshold_candidates, raw_score, raw_reid, raw_source = (
                asyncio.run(
                    gallery.asearch_similar_person_candidates_with_hit_info(
                        [0.1] * 2048,
                        query_body_completeness=BODY_FULL,
                    )
                )
            )
            self.assertEqual(
                [candidate.reid for candidate in threshold_candidates],
                [
                    "raw-highest-full",
                    "accepted-upper",
                    "exact-half-boundary",
                ],
            )
            self.assertEqual((raw_score, raw_reid), (0.65, "raw-highest-full"))
            self.assertEqual(raw_source, "milvus")

            client.search_calls.clear()
            no_search_result = asyncio.run(
                gallery.asearch_similar_person_candidates_with_hit_info(
                    [0.1] * 2048,
                    query_body_completeness=BODY_HALF_LOWER,
                )
            )
            self.assertEqual(no_search_result, ([], 0.0, "", ""))
            self.assertEqual(client.search_calls, [])

            class BlockingMilvusClient(FakeMilvusClient):
                def __init__(self, *, fail_body_type=None):
                    super().__init__()
                    self.barrier = threading.Barrier(3)
                    self.lock = threading.Lock()
                    self.fail_body_type = fail_body_type
                    self.stored_hits_by_body = {
                        BODY_FULL: [
                            {
                                "entity": {
                                    "reId": "tie-full",
                                    "bodyCompleteness": BODY_FULL,
                                },
                                "distance": 0.81,
                            }
                        ],
                        BODY_THREE_QUARTER_UPPER: [
                            {
                                "entity": {
                                    "reId": "tie-upper",
                                    "bodyCompleteness": BODY_THREE_QUARTER_UPPER,
                                },
                                "distance": 0.81,
                            }
                        ],
                        BODY_HALF_UPPER: [
                            {
                                "entity": {
                                    "reId": "tie-half",
                                    "bodyCompleteness": BODY_HALF_UPPER,
                                },
                                "distance": 0.81,
                            }
                        ],
                    }

                def search(self, **kwargs):
                    body_type = self.gallery_body_type(kwargs["filter"])
                    with self.lock:
                        self.search_calls.append(kwargs)
                    self.barrier.wait(timeout=2)
                    if body_type == self.fail_body_type:
                        raise RuntimeError(f"{body_type} query failed")
                    return [
                        self.stored_hits_by_body[body_type][: kwargs["limit"]]
                    ]

            parallel_client = BlockingMilvusClient()
            gallery.client = parallel_client
            parallel_result = asyncio.run(
                gallery.asearch_similar_person_candidates_with_hit_info(
                    [0.1] * 2048,
                    query_body_completeness=BODY_FULL,
                )
            )
            self.assertEqual(len(parallel_client.search_calls), 3)
            self.assertEqual(
                [candidate.body_completeness for candidate in parallel_result[0]],
                [BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER],
            )
            self.assertEqual(parallel_result[2], "tie-full")

            failing_client = BlockingMilvusClient(
                fail_body_type=BODY_THREE_QUARTER_UPPER
            )
            gallery.client = failing_client
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                asyncio.run(
                    gallery.asearch_similar_person_candidates_with_hit_info(
                        [0.1] * 2048,
                        query_body_completeness=BODY_FULL,
                    )
                )
            self.assertEqual(len(failing_client.search_calls), 3)
            gallery.client = client

            class FakeSchema:
                def __init__(self):
                    self.field_names = []
                    self.fields = {}

                def add_field(self, name, *_args, **_kwargs):
                    self.field_names.append(name)
                    self.fields[name] = _kwargs

            class ExistingCollectionClient:
                def __init__(self):
                    self.schema = None
                    self.describe_calls = 0
                    self.insert_calls = []

                def create_schema(self, **_kwargs):
                    self.schema = FakeSchema()
                    return self.schema

                def has_collection(self, _collection_name):
                    return True

                def describe_collection(self, **_kwargs):
                    self.describe_calls += 1
                    return {
                        "auto_id": False,
                        "fields": [
                            {
                                "name": name,
                                **(
                                    {"type": "VARCHAR", "auto_id": False}
                                    if name == "id"
                                    else {}
                                ),
                            }
                            for name in (
                                "id",
                                "faceId",
                                "reId",
                                "bodyCompleteness",
                                "cropUrl",
                                "qualityScore",
                                "embedding",
                                "deviceId",
                                "trackId",
                                "keyframeId",
                                "personRect",
                                "timestamp",
                            )
                        ]
                    }

                def insert(self, **kwargs):
                    self.insert_calls.append(kwargs)
                    return {"primary_keys": [1]}

            tbps_client = ExistingCollectionClient()
            tbps_gallery = tbps_module.TbpsGallery.__new__(
                tbps_module.TbpsGallery
            )
            self.assertEqual(tbps_module.TbpsGallery.COLLECTION_NAME, "alogTBPS")
            self.assertEqual(tbps_module.TbpsGallery.EMBEDDING_DIM, 512)
            self.assertFalse(hasattr(module, "PersonReidGallery"))
            self.assertFalse(hasattr(tbps_module, "PersonReidGallery"))
            tbps_gallery.collection_name = "alogTBPS"
            tbps_gallery.dim = 512
            tbps_gallery.client = tbps_client
            tbps_gallery._create_schema()
            self.assertEqual(
                tbps_client.schema.field_names,
                [
                    "id",
                    "faceId",
                    "reId",
                    "embedding",
                    "deviceId",
                    "trackId",
                    "keyframeId",
                    "personRect",
                    "timestamp",
                ],
            )
            self.assertEqual(tbps_client.describe_calls, 0)
            tbps_gallery.insert_embeddings(
                faceId=["face-1"],
                reId=["reid-1"],
                embedding=[[0.1] * 512],
                deviceId=["CAM001"],
                trackId=[0],
                keyframeId=[42],
                personRect=[{"x": 1, "y": 2, "w": 3, "h": 4}],
                timestamp=[171],
            )
            self.assertNotIn(
                "bodyCompleteness",
                tbps_client.insert_calls[0]["data"][0],
            )

            sbs_client = ExistingCollectionClient()
            sbs_gallery = module.SbsReidGallery.__new__(module.SbsReidGallery)
            self.assertEqual(module.SbsReidGallery.COLLECTION_NAME, "alogSBS")
            self.assertEqual(module.SbsReidGallery.EMBEDDING_DIM, 2048)
            sbs_gallery.collection_name = "alogSBS"
            sbs_gallery.dim = 2048
            sbs_gallery.client = sbs_client
            sbs_gallery._create_schema()
            self.assertEqual(
                sbs_client.schema.field_names,
                [
                    "id",
                    "faceId",
                    "reId",
                    "bodyCompleteness",
                    "cropUrl",
                    "qualityScore",
                    "embedding",
                    "deviceId",
                    "trackId",
                    "keyframeId",
                    "personRect",
                    "timestamp",
                ],
            )
            self.assertEqual(sbs_client.schema.fields["cropUrl"]["max_length"], 512)
            self.assertEqual(sbs_client.describe_calls, 1)
            sbs_gallery._insert_prototype(
                faceId=["face-1"],
                reId=["reid-1"],
                bodyCompleteness=[BODY_FULL],
                cropUrl=["CAM001/person_crop/person.jpg"],
                embedding=[[0.1] * 2048],
                deviceId=["CAM001"],
                trackId=[987654321],
                keyframeId=[42],
                personRect=[{"x": 1, "y": 2, "w": 3, "h": 4}],
                timestamp=[171],
                qualityScore=[9.0],
            )
            inserted_sbs_record = sbs_client.insert_calls[-1]["data"][0]
            self.assertEqual(inserted_sbs_record["trackId"], 0)
            self.assertEqual(
                inserted_sbs_record["cropUrl"],
                "CAM001/person_crop/person.jpg",
            )
            sbs_gallery._insert_prototype(
                faceId=["face-2"],
                reId=["reid-2"],
                bodyCompleteness=[BODY_FULL],
                cropUrl=[""],
                embedding=[[0.1] * 2048],
                deviceId=["CAM001"],
                trackId=[0],
                keyframeId=[43],
                personRect=[{"x": 1, "y": 2, "w": 3, "h": 4}],
                timestamp=[172],
                qualityScore=[9.0],
            )
            self.assertEqual(sbs_client.insert_calls[-1]["data"][0]["cropUrl"], "")

            class ReplacementClient(ExistingCollectionClient):
                def __init__(self, *, fail_delete=False):
                    super().__init__()
                    self.fail_delete = fail_delete
                    self.events = []
                    angles = [0.0, 0.05, 0.10, 0.15, 0.20]
                    self.records = [
                        {
                            "id": index + 1,
                            "faceId": "face-1",
                            "reId": "reid-1",
                            "bodyCompleteness": BODY_FULL,
                            "cropUrl": f"crop-{index}.jpg",
                            "embedding": [float(np.cos(angle)), float(np.sin(angle))],
                            "deviceId": "CAM001",
                            "trackId": 0,
                            "keyframeId": index + 1,
                            "personRect": {"x": 1, "y": 2, "w": 3, "h": 4},
                            "timestamp": index + 1,
                            "qualityScore": 9.0,
                        }
                        for index, angle in enumerate(angles)
                    ]

                def query(self, **kwargs):
                    self.query_kwargs = kwargs
                    return list(self.records)

                def delete(self, **kwargs):
                    self.events.append(("delete", kwargs))
                    if self.fail_delete:
                        raise RuntimeError("delete failed")

                def insert(self, **kwargs):
                    self.events.append(("insert", kwargs))
                    return {"primary_keys": [99]}

            replacement_item = {
                "faceId": "face-1",
                "reId": "reid-1",
                "bodyCompleteness": BODY_FULL,
                "cropUrl": "CAM001/person_crop/replacement.jpg",
                "embedding": [float(np.cos(0.8)), float(np.sin(0.8))],
                "deviceId": "CAM001",
                "pictureVectorId": 99,
                "personRect": {"x": 1, "y": 2, "w": 3, "h": 4},
                "timestamp": 999,
                "qualityScore": 9.5,
            }
            replacement_client = ReplacementClient()
            sbs_gallery.client = replacement_client
            replacement_outcome = sbs_gallery.apply_prototype_updates(
                [replacement_item]
            )[0]
            self.assertEqual(replacement_outcome["action"], "replace")
            self.assertEqual(
                [event_name for event_name, _ in replacement_client.events],
                ["delete", "insert"],
            )
            self.assertEqual(
                replacement_client.query_kwargs["consistency_level"],
                "Strong",
            )
            self.assertEqual(
                replacement_client.events[-1][1]["data"][0]["trackId"],
                0,
            )

            failing_replacement_client = ReplacementClient(fail_delete=True)
            sbs_gallery.client = failing_replacement_client
            frozen_outcome = sbs_gallery.apply_prototype_updates(
                [replacement_item]
            )[0]
            self.assertEqual(frozen_outcome["action"], "frozen")
            self.assertEqual(frozen_outcome["reason"], "replacement_delete_failed")
            self.assertEqual(
                [event_name for event_name, _ in failing_replacement_client.events],
                ["delete"],
            )

            class CommitThenTimeoutClient:
                def __init__(self):
                    self.records = []
                    self.insert_calls = 0

                def query(self, **_kwargs):
                    return [dict(record) for record in self.records]

                def insert(self, **kwargs):
                    self.insert_calls += 1
                    self.records.extend(
                        dict(record) for record in kwargs["data"]
                    )
                    raise RuntimeError("timeout after commit")

            uncertain_item = {
                "prototypeId": "00000000-0000-0000-0000-000000000123",
                "faceId": "face-uncertain",
                "reId": "reid-uncertain",
                "bodyCompleteness": BODY_FULL,
                "cropUrl": "CAM001/person_crop/uncertain.jpg",
                "embedding": [1.0, 0.0],
                "deviceId": "CAM001",
                "pictureVectorId": 123,
                "personRect": {"x": 1, "y": 2, "w": 3, "h": 4},
                "timestamp": 123,
                "qualityScore": 9.5,
            }
            uncertain_client = CommitThenTimeoutClient()
            sbs_gallery.client = uncertain_client

            first_outcome = sbs_gallery.apply_prototype_updates(
                [uncertain_item]
            )[0]
            retry_outcome = sbs_gallery.apply_prototype_updates(
                [uncertain_item]
            )[0]

            self.assertTrue(first_outcome["retry"])
            self.assertEqual(retry_outcome["action"], "already_persisted")
            self.assertTrue(retry_outcome["persisted"])
            self.assertEqual(
                retry_outcome["insert_result"]["primary_keys"],
                [uncertain_item["prototypeId"]],
            )
            self.assertEqual(uncertain_client.insert_calls, 1)

            class ZeroWriteClient:
                def query(self, **_kwargs):
                    return []

                def insert(self, **_kwargs):
                    return {"insert_count": 0, "primary_keys": []}

            zero_write_item = {
                **uncertain_item,
                "prototypeId": "00000000-0000-0000-0000-000000000124",
                "reId": "reid-zero-write",
            }
            sbs_gallery.client = ZeroWriteClient()
            zero_write_outcome = sbs_gallery.apply_prototype_updates(
                [zero_write_item]
            )[0]
            self.assertTrue(zero_write_outcome["retry"])
            self.assertFalse(zero_write_outcome["persisted"])

            class ExistingSiblingPrototypeClient:
                def query(self, **_kwargs):
                    return [
                        {
                            **uncertain_item,
                            "id": "00000000-0000-0000-0000-000000000125",
                        }
                    ]

                def insert(self, **_kwargs):
                    raise AssertionError("redundant prototype must not insert")

            redundant_item = {
                **uncertain_item,
                "prototypeId": "00000000-0000-0000-0000-000000000126",
            }
            sbs_gallery.client = ExistingSiblingPrototypeClient()
            redundant_outcome = sbs_gallery.apply_prototype_updates(
                [redundant_item]
            )[0]
            self.assertFalse(redundant_outcome.get("retry", False))
            self.assertTrue(redundant_outcome["persisted"])
            self.assertIsNone(redundant_outcome["insert_result"])

            class NewCollectionClient(ExistingCollectionClient):
                def __init__(self):
                    super().__init__()
                    self.created = None

                def has_collection(self, _collection_name):
                    return False

                def create_collection(self, **kwargs):
                    self.created = kwargs

            rebuilt_client = NewCollectionClient()
            rebuilt_gallery = module.SbsReidGallery.__new__(
                module.SbsReidGallery
            )
            rebuilt_gallery.collection_name = "alogSBS"
            rebuilt_gallery.dim = 2048
            rebuilt_gallery.client = rebuilt_client
            rebuilt_gallery._create_schema()
            self.assertEqual(rebuilt_client.created["collection_name"], "alogSBS")
            self.assertEqual(
                rebuilt_client.created["properties"],
                {"collection.ttl.seconds": 604800},
            )
        finally:
            sys.modules.pop(module_name, None)
            sys.modules.pop(tbps_module_name, None)
            sys.modules.pop(person_feature_module_name, None)
            sys.modules.pop(picture_module_name, None)
            sys.modules.pop(package_name, None)
            if original_pymilvus is None:
                sys.modules.pop("pymilvus", None)
            else:
                sys.modules["pymilvus"] = original_pymilvus

    def test_pending_person_search_filters_face_id(self):
        module_names = ["utils.buffers.buffer", "aiokafka", "core.config", "db.zilliz"]
        originals = {name: sys.modules.get(name) for name in module_names}
        for name in module_names:
            sys.modules.pop(name, None)

        aiokafka_module = types.ModuleType("aiokafka")
        aiokafka_module.AIOKafkaProducer = object
        config_module = types.ModuleType("core.config")
        config_module.settings = types.SimpleNamespace(
            KAFKA_FACE_DETECTION_RECORD_TOPIC="face-detection",
            KAFKA_FACE_RECOGNITION_RECORD_TOPIC="face-recognition",
            KAFKA_PERSON_DETECTION_RECORD_TOPIC="person-detection",
            KAFKA_PERSON_RECOGNITION_RECORD_TOPIC="person-recognition",
            KAFKA_VECTOR_RECORD_TOPIC="vector",
            kafka_client_config={},
        )
        zilliz_module = types.ModuleType("db.zilliz")

        class NoopSbsDb:
            collection_name = "alogSBS"

            async def aapply_prototype_updates(self, batch):
                return []

        zilliz_module.face_embedding_db = object()
        zilliz_module.person_clip_embedding_db = object()
        zilliz_module.person_sbs_embedding_db = NoopSbsDb()
        zilliz_module.picture_clip_embedding_db = object()
        sys.modules["aiokafka"] = aiokafka_module
        sys.modules["core.config"] = config_module
        sys.modules["db.zilliz"] = zilliz_module
        try:
            buffer_module = importlib.import_module("utils.buffers.buffer")
            feature_buffer = buffer_module.FeatureBuffer(max_batch_size=10)

            async def run():
                for face_id, reid in (("face-1", "reid-1"), ("face-2", "reid-2")):
                    await enqueue_formal_sbs(feature_buffer.sbs,
                        face_id=face_id,
                        re_id=reid,
                        body_completeness=BODY_FULL,
                        embedding=np.array([1.0, 0.0], dtype=np.float32),
                        device_id="CAM001",
                        keyframe_id=reid,
                        picture_vector_id=1,
                        keyframe_url="crop.jpg",
                        person_rect=PERSON_RECT,
                        timestamp=9999999999999,
                        ai_id="ai",
                    )
                return await resolve_buffer_candidates(feature_buffer.sbs,
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                    face_id="face-1",
                )

            self.assertEqual(asyncio.run(run()), ("reid-1", 1.0, "reid-1"))
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                if originals[name] is not None:
                    sys.modules[name] = originals[name]


class FakePersonFaceTriton:
    def __init__(self, quality_score):
        self.quality_score = quality_score
        self.detection_calls = 0
        self.quality_calls = 0
        self.quality_inputs = []
        self.feature_calls = 0
        self.keypoint_calls = 0
        self.detections = [{"xyxy": [10, 10, 55, 55], "conf": 0.9, "cls": 0}]
        self.landmarks = np.array(
            [[15, 20], [50, 20], [32, 32], [20, 45], [45, 45]],
            dtype=np.float32,
        )
        self.keypoint_error = None

    async def extract_face_detections(self, image):
        self.detection_calls += 1
        return self.detections

    async def extract_face_keypoints(self, expanded_crop, bbox):
        self.keypoint_calls += 1
        if self.keypoint_error is not None:
            raise self.keypoint_error
        return self.landmarks.copy()

    async def extract_face_quality(self, face_crop, landmarks):
        self.quality_calls += 1
        self.quality_inputs.append(
            (np.asarray(face_crop).copy(), np.asarray(landmarks).copy())
        )
        return FaceQualityResult(self.quality_score)

    async def extract_face_features(self, expanded_crop, landmarks):
        self.feature_calls += 1
        return FeatureResult(np.ones(512, dtype=np.float32))


class PersonFaceRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.original_person_precheck = person_face.meets_pre_extraction_quality_criteria
        self.original_person_feature_check = person_face.is_high_quality_feature
        self.original_laplacian = face_quality.calculate_laplacian_variance
        self.original_pose = face_quality.calculate_pose_from_landmarks
        self.original_orientation_estimator = getattr(
            person_face,
            "estimate_face_orientation",
            None,
        )
        person_face.estimate_face_orientation = lambda landmarks, image_shape: (
            FaceOrientationResult(FaceOrientation.FRONT, 0.0, 0.0, 0.0)
        )

    def tearDown(self):
        person_face.meets_pre_extraction_quality_criteria = self.original_person_precheck
        person_face.is_high_quality_feature = self.original_person_feature_check
        face_quality.calculate_laplacian_variance = self.original_laplacian
        face_quality.calculate_pose_from_landmarks = self.original_pose
        if self.original_orientation_estimator is None:
            delattr(person_face, "estimate_face_orientation")
        else:
            person_face.estimate_face_orientation = self.original_orientation_estimator

    def test_local_centered_face_preserves_expanded_crop_pose_coordinates(self):
        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.detections = [
            {"xyxy": [30, 30, 70, 70], "conf": 0.9, "cls": 0}
        ]
        triton.landmarks = np.array(
            [[27, 32], [53, 32], [40, 43], [30, 58], [51, 58]],
            dtype=np.float32,
        )

        async def unexpected_search(*args, **kwargs):
            raise AssertionError("orientation-only local path must not search")

        with patch.object(
            person_face,
            "estimate_face_orientation",
            wraps=self.original_orientation_estimator,
        ) as estimator:
            result = asyncio.run(
                person_face.match_face_id_from_person_crop_with_info(
                    person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                    person_mask=np.ones((100, 100), dtype=np.uint8),
                    device_id="CAM001",
                    timestamp=1710000000000,
                    triton_client=triton,
                    search_callback=unexpected_search,
                    logger=DummyLogger(),
                    enable_recognition=False,
                )
            )

        estimator.assert_called_once()
        pose_landmarks, pose_shape = estimator.call_args.args
        # 关键点已相对 expanded crop；不能再次减去检测框在裁图中的原点 (20, 20)。
        np.testing.assert_array_equal(pose_landmarks, triton.landmarks)
        self.assertEqual(pose_shape, (80, 80, 3))
        self.assertEqual(result["face_orientation"], "front")
        for angle in ("pitch", "yaw", "roll"):
            self.assertTrue(np.isfinite(result[angle]), angle)

    def test_local_edge_face_preserves_expanded_crop_pose_coordinates(self):
        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.detections = [
            {"xyxy": [70, 30, 98, 58], "conf": 0.9, "cls": 0}
        ]
        triton.landmarks = np.array(
            [[16, 20], [40, 20], [28, 31], [18, 41], [37, 41]],
            dtype=np.float32,
        )

        async def unexpected_search(*args, **kwargs):
            raise AssertionError("orientation-only local path must not search")

        with patch.object(
            person_face,
            "estimate_face_orientation",
            wraps=self.original_orientation_estimator,
        ) as estimator:
            result = asyncio.run(
                person_face.match_face_id_from_person_crop_with_info(
                    person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                    person_mask=np.ones((100, 100), dtype=np.uint8),
                    device_id="CAM001",
                    timestamp=1710000000000,
                    triton_client=triton,
                    search_callback=unexpected_search,
                    logger=DummyLogger(),
                    enable_recognition=False,
                )
            )

        estimator.assert_called_once()
        pose_landmarks, pose_shape = estimator.call_args.args
        # 边缘裁图已截断为非正方形，关键点不能再减去检测框原点 (14, 14)。
        np.testing.assert_array_equal(pose_landmarks, triton.landmarks)
        self.assertEqual(pose_shape, (56, 44, 3))
        self.assertEqual(result["face_orientation"], "front")
        for angle in ("pitch", "yaw", "roll"):
            self.assertTrue(np.isfinite(result[angle]), angle)

    def test_candidate_reuses_complete_shared_artifacts_and_searches_again(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        triton = FakePersonFaceTriton(quality_score=0.0)
        normalized = np.zeros(512, dtype=np.float32)
        normalized[0] = 1.0
        landmarks = np.array(
            [[15, 20], [50, 20], [32, 32], [20, 45], [45, 45]],
            dtype=np.float32,
        )
        search_calls = 0

        async def fake_search(embedding, threshold, device_ids):
            nonlocal search_calls
            search_calls += 1
            np.testing.assert_array_equal(embedding, normalized)
            self.assertEqual(threshold, algorithm_config.face.matching.general_similarity_threshold)
            self.assertIsNone(device_ids)
            return "face-shared", 0.91, "face-shared"

        result = asyncio.run(
            person_face.recognize_person_face_candidate_with_info(
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                bbox=(10, 10, 55, 55),
                landmarks_in_image=landmarks,
                quality_score=5.5,
                normalized_embedding=normalized,
                raw_embedding_norm=0.25,
                ownership_info="shared owner",
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=fake_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_id"], "face-shared")
        self.assertIn("shared owner", result["info"])
        self.assertEqual(triton.keypoint_calls, 0)
        self.assertEqual(triton.quality_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(search_calls, 1)

    def test_candidate_applies_face_id_scalar_contract_to_search_result(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        person_face.is_high_quality_feature = lambda *args, **kwargs: (True, None)
        triton = FakePersonFaceTriton(quality_score=4.5)
        normalized = np.zeros(512, dtype=np.float32)
        normalized[0] = 1.0
        search_calls = 0

        cases = (
            ("container", ["bad-id"], person_face.UNMATCHED_FACE_ID),
            (
                "UUID string",
                "123e4567-e89b-12d3-a456-426614174000",
                "123e4567-e89b-12d3-a456-426614174000",
            ),
            ("integer", 42, "42"),
        )
        for name, raw_face_id, expected in cases:
            with self.subTest(name=name):
                async def fake_search(*args, **kwargs):
                    nonlocal search_calls
                    search_calls += 1
                    return raw_face_id, 0.91, raw_face_id

                result = asyncio.run(
                    person_face.recognize_person_face_candidate_with_info(
                        image=np.zeros((100, 100, 3), dtype=np.uint8),
                        bbox=(10, 10, 55, 55),
                        landmarks_in_image=np.ones((5, 2), dtype=np.float32) * 30,
                        quality_score=5.5,
                        normalized_embedding=normalized,
                        raw_embedding_norm=0.25,
                        device_id="CAM001",
                        timestamp=1710000000000,
                        triton_client=triton,
                        search_callback=fake_search,
                        logger=DummyLogger(),
                    )
                )

                self.assertEqual(result["face_id"], expected)
        self.assertEqual(search_calls, len(cases))

    def test_candidate_rejects_shared_feature_below_person_norm_threshold(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        triton = FakePersonFaceTriton(quality_score=0.0)
        normalized = np.zeros(512, dtype=np.float32)
        normalized[0] = 1.0
        search_calls = 0

        async def unexpected_search(*args, **kwargs):
            nonlocal search_calls
            search_calls += 1
            raise AssertionError("search should not run for a low-norm shared feature")

        result = asyncio.run(
            person_face.recognize_person_face_candidate_with_info(
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                bbox=(10, 10, 55, 55),
                landmarks_in_image=np.ones((5, 2), dtype=np.float32) * 30,
                quality_score=5.5,
                normalized_embedding=normalized,
                raw_embedding_norm=0.19,
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertIn("人脸特征质量未通过", result["info"])
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(search_calls, 0)

    def test_candidate_ignores_incomplete_shared_feature_pair(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        person_face.is_high_quality_feature = lambda *args, **kwargs: (True, None)
        normalized = np.zeros(512, dtype=np.float32)
        normalized[0] = 1.0

        incomplete_pairs = (
            {"normalized_embedding": normalized},
            {"raw_embedding_norm": 0.25},
        )
        for supplied in incomplete_pairs:
            with self.subTest(supplied=tuple(supplied)):
                triton = FakePersonFaceTriton(quality_score=0.0)
                search_calls = 0

                async def fake_search(*args, **kwargs):
                    nonlocal search_calls
                    search_calls += 1
                    return "face-fresh", 0.8, "face-fresh"

                result = asyncio.run(
                    person_face.recognize_person_face_candidate_with_info(
                        image=np.zeros((100, 100, 3), dtype=np.uint8),
                        bbox=(10, 10, 55, 55),
                        landmarks_in_image=np.ones((5, 2), dtype=np.float32) * 30,
                        quality_score=5.5,
                        device_id="CAM001",
                        timestamp=1710000000000,
                        triton_client=triton,
                        search_callback=fake_search,
                        logger=DummyLogger(),
                        **supplied,
                    )
                )

                self.assertEqual(result["face_id"], "face-fresh")
                self.assertEqual(triton.feature_calls, 1)
                self.assertEqual(search_calls, 1)

    def test_candidate_infers_landmarks_when_supplied_shape_is_not_exact(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        person_face.is_high_quality_feature = lambda *args, **kwargs: (True, None)
        normalized = np.zeros(512, dtype=np.float32)
        normalized[0] = 1.0
        malformed_landmarks = (
            np.ones((4, 3), dtype=np.float32),
            np.ones((6, 2), dtype=np.float32),
            np.ones(10, dtype=np.float32),
        )

        for supplied in malformed_landmarks:
            with self.subTest(shape=supplied.shape):
                triton = FakePersonFaceTriton(quality_score=0.0)

                async def fake_search(*args, **kwargs):
                    return "face-inferred", 0.8, "face-inferred"

                result = asyncio.run(
                    person_face.recognize_person_face_candidate_with_info(
                        image=np.zeros((100, 100, 3), dtype=np.uint8),
                        bbox=(10, 10, 55, 55),
                        landmarks_in_image=supplied,
                        quality_score=5.5,
                        normalized_embedding=normalized,
                        raw_embedding_norm=0.25,
                        device_id="CAM001",
                        timestamp=1710000000000,
                        triton_client=triton,
                        search_callback=fake_search,
                        logger=DummyLogger(),
                    )
                )

                self.assertEqual(result["face_id"], "face-inferred")
                self.assertEqual(triton.keypoint_calls, 1)

    def test_candidate_extracts_feature_for_non_unit_normalized_embedding(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        person_face.is_high_quality_feature = lambda *args, **kwargs: (True, None)
        zero_norm = np.zeros(512, dtype=np.float32)
        norm_two = np.zeros(512, dtype=np.float32)
        norm_two[0] = 2.0

        for supplied in (zero_norm, norm_two):
            with self.subTest(norm=float(np.linalg.norm(supplied))):
                triton = FakePersonFaceTriton(quality_score=0.0)

                async def fake_search(embedding, *args, **kwargs):
                    self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
                    return "face-fresh", 0.8, "face-fresh"

                result = asyncio.run(
                    person_face.recognize_person_face_candidate_with_info(
                        image=np.zeros((100, 100, 3), dtype=np.uint8),
                        bbox=(10, 10, 55, 55),
                        landmarks_in_image=np.ones((5, 2), dtype=np.float32) * 30,
                        quality_score=5.5,
                        normalized_embedding=supplied,
                        raw_embedding_norm=0.25,
                        device_id="CAM001",
                        timestamp=1710000000000,
                        triton_client=triton,
                        search_callback=fake_search,
                        logger=DummyLogger(),
                    )
                )

                self.assertEqual(result["face_id"], "face-fresh")
                self.assertEqual(triton.feature_calls, 1)

    def test_candidate_reuses_low_quality_score_and_stops_before_feature(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        triton = FakePersonFaceTriton(quality_score=5.5)
        search_calls = 0

        async def unexpected_search(*args, **kwargs):
            nonlocal search_calls
            search_calls += 1
            raise AssertionError("search should not run below the quality threshold")

        result = asyncio.run(
            person_face.recognize_person_face_candidate_with_info(
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                bbox=(10, 10, 55, 55),
                landmarks_in_image=np.ones((5, 2), dtype=np.float32) * 30,
                quality_score=0.99,
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertIn("0.99", result["info"])
        self.assertEqual(triton.quality_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(search_calls, 0)

    def test_candidate_orientation_only_uses_shared_landmarks_and_stops(self):
        calls = {"estimate": 0, "precheck": 0, "search": 0}

        def estimate_orientation(landmarks, image_shape):
            calls["estimate"] += 1
            return FaceOrientationResult(FaceOrientation.SIDE, 2.0, 35.0, -1.0)

        def unexpected_precheck(*args, **kwargs):
            calls["precheck"] += 1
            raise AssertionError("precheck should not run in orientation-only mode")

        async def unexpected_search(*args, **kwargs):
            calls["search"] += 1
            raise AssertionError("search should not run in orientation-only mode")

        person_face.estimate_face_orientation = estimate_orientation
        person_face.meets_pre_extraction_quality_criteria = unexpected_precheck
        triton = FakePersonFaceTriton(quality_score=5.5)
        result = asyncio.run(
            person_face.recognize_person_face_candidate_with_info(
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                bbox=(10, 10, 55, 55),
                landmarks_in_image=np.ones((5, 2), dtype=np.float32) * 30,
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
                enable_recognition=False,
            )
        )

        self.assertEqual(result["face_orientation"], "side")
        self.assertEqual(result["pitch"], 2.0)
        self.assertEqual(result["yaw"], 35.0)
        self.assertEqual(result["roll"], -1.0)
        self.assertEqual(triton.keypoint_calls, 0)
        self.assertEqual(triton.quality_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(calls, {"estimate": 1, "precheck": 0, "search": 0})

    def test_owned_face_selection_preserves_original_detection_index(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        detections = [
            None,
            {"xyxy": [5, 10, 45, 55], "conf": 0.99},
            {"xyxy": [55, 10, 95, 55], "conf": 0.70},
        ]
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[:5] = np.array(
            [[75, 25, 0.9], [72, 23, 0.9], [78, 23, 0.9], [68, 26, 0.9], [82, 26, 0.9]],
            dtype=np.float32,
        )

        selection = person_face.select_owned_face_detection(
            detections, (100, 100, 3), mask, pose
        )

        self.assertEqual(selection.detection_index, 2)
        self.assertEqual(selection.bbox, (55, 10, 95, 55))
        self.assertIn("Pose头部匹配=True", selection.info)

    def test_orientation_only_stops_before_recognition_models_and_search(self):
        calls = {"estimate": 0, "precheck": 0, "normalize": 0, "search": 0}

        def unexpected_precheck(*args, **kwargs):
            calls["precheck"] += 1
            raise AssertionError("precheck should not run in orientation-only mode")

        def unexpected_normalize(*args, **kwargs):
            calls["normalize"] += 1
            raise AssertionError("normalization should not run in orientation-only mode")

        async def unexpected_search(*args, **kwargs):
            calls["search"] += 1
            raise AssertionError("search should not run in orientation-only mode")

        person_face.meets_pre_extraction_quality_criteria = unexpected_precheck

        def estimate_orientation(landmarks, image_shape):
            calls["estimate"] += 1
            return FaceOrientationResult(FaceOrientation.SIDE, 2.0, 35.0, -1.0)

        person_face.estimate_face_orientation = estimate_orientation
        triton = FakePersonFaceTriton(quality_score=4.5)

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                normalize_vector=unexpected_normalize,
                logger=DummyLogger(),
                enable_recognition=False,
            )
        )

        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertEqual(result["face_orientation"], "side")
        self.assertEqual(result["pitch"], 2.0)
        self.assertEqual(result["yaw"], 35.0)
        self.assertEqual(result["roll"], -1.0)
        self.assertEqual(triton.detection_calls, 1)
        self.assertEqual(triton.keypoint_calls, 1)
        self.assertEqual(triton.quality_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(
            calls,
            {"estimate": 1, "precheck": 0, "normalize": 0, "search": 0},
        )

    def test_orientation_estimator_exception_reports_unknown_once(self):
        estimate_calls = 0

        def failing_estimator(landmarks, image_shape):
            nonlocal estimate_calls
            estimate_calls += 1
            raise RuntimeError("pose unavailable")

        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run in orientation-only mode")

        person_face.estimate_face_orientation = failing_estimator
        triton = FakePersonFaceTriton(quality_score=4.5)
        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
                enable_recognition=False,
            )
        )

        self.assertEqual(estimate_calls, 1)
        self.assertEqual(result["face_orientation"], "unknown")
        self.assertIsNone(result["pitch"])
        self.assertIsNone(result["yaw"])
        self.assertIsNone(result["roll"])

    def test_non_list_face_detection_responses_report_unknown(self):
        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run for malformed detections")

        malformed_responses = (
            None,
            ({"xyxy": [10, 10, 55, 55], "conf": 0.9},),
        )
        for detections in malformed_responses:
            with self.subTest(detections=detections):
                triton = FakePersonFaceTriton(quality_score=4.5)
                triton.detections = detections
                try:
                    result = asyncio.run(
                        person_face.match_face_id_from_person_crop_with_info(
                            person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                            person_mask=np.ones((100, 100), dtype=np.uint8),
                            device_id="CAM001",
                            timestamp=1710000000000,
                            triton_client=triton,
                            search_callback=unexpected_search,
                            logger=DummyLogger(),
                        )
                    )
                except Exception as exc:
                    self.fail(f"malformed detections must fail closed: {exc}")
                self.assertEqual(result["face_orientation"], "unknown")
                self.assertEqual(triton.keypoint_calls, 0)

    def test_all_invalid_face_detection_boxes_report_unknown(self):
        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run for invalid detections")

        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.detections = [
            {"xyxy": []},
            {"xyxy": [1, 2, 3]},
            {"xyxy": [10, 10, 10, 20]},
        ]
        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_orientation"], "unknown")
        self.assertEqual(triton.keypoint_calls, 0)

    def test_non_dict_detection_is_skipped_when_valid_detection_exists(self):
        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run in orientation-only mode")

        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.detections = [
            None,
            "malformed",
            {"xyxy": [10, 10, 55, 55], "conf": 0.9},
        ]
        try:
            result = asyncio.run(
                person_face.match_face_id_from_person_crop_with_info(
                    person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                    person_mask=np.ones((100, 100), dtype=np.uint8),
                    device_id="CAM001",
                    timestamp=1710000000000,
                    triton_client=triton,
                    search_callback=unexpected_search,
                    logger=DummyLogger(),
                    enable_recognition=False,
                )
            )
        except Exception as exc:
            self.fail(f"non-dict detections must be skipped safely: {exc}")

        self.assertEqual(result["face_orientation"], "front")
        self.assertEqual(triton.keypoint_calls, 1)

    def test_empty_face_detections_report_no_face_without_keypoints_or_search(self):
        search_calls = 0

        async def unexpected_search(*args, **kwargs):
            nonlocal search_calls
            search_calls += 1
            raise AssertionError("search should not run without a face detection")

        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.detections = []
        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_orientation"], "no_face")
        self.assertEqual(triton.keypoint_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(search_calls, 0)

    def test_invalid_or_missing_person_mask_reports_unknown(self):
        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run with an invalid person mask")

        invalid_masks = (
            None,
            np.zeros((100, 100), dtype=np.uint8),
            np.ones((99, 100), dtype=np.uint8),
        )
        for invalid_mask in invalid_masks:
            with self.subTest(mask=invalid_mask):
                triton = FakePersonFaceTriton(quality_score=4.5)
                if invalid_mask is None:
                    triton.detections = []
                result = asyncio.run(
                    person_face.match_face_id_from_person_crop_with_info(
                        person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                        person_mask=invalid_mask,
                        device_id="CAM001",
                        timestamp=1710000000000,
                        triton_client=triton,
                        search_callback=unexpected_search,
                        logger=DummyLogger(),
                    )
                )
                self.assertEqual(result["face_orientation"], "unknown")
                self.assertEqual(triton.keypoint_calls, 0)

    def test_keypoint_failure_reports_unknown(self):
        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run after keypoint failure")

        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.keypoint_error = RuntimeError("keypoints unavailable")
        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_orientation"], "unknown")
        self.assertEqual(triton.keypoint_calls, 1)
        self.assertEqual(triton.feature_calls, 0)

    def test_default_and_person_bind_face_width_boundaries(self):
        face_quality.calculate_laplacian_variance = lambda image: 100.0
        face_quality.calculate_pose_from_landmarks = (
            lambda landmarks, image_shape: (0.0, 0.0, 0.0)
        )
        image = np.ones((80, 80, 3), dtype=np.uint8) * 127
        normalized_landmarks = np.array(
            [[0.2, 0.25], [0.8, 0.25], [0.5, 0.5], [0.3, 0.8], [0.7, 0.8]],
            dtype=np.float32,
        )
        gates = {
            "standalone": {},
            "person_bind": {
                "laplacian_threshold": algorithm_config.person_face.binding.laplacian_threshold,
                "min_face_width": algorithm_config.person_face.binding.min_face_width_px,
                "max_pose_abs_degrees": algorithm_config.person_face.binding.max_pose_abs_degrees,
                "geometry_std_ratio_threshold": (
                    algorithm_config.person_face.binding.geometry_std_ratio_threshold
                ),
            },
        }
        for gate, thresholds in gates.items():
            for width, expected_good in ((19, False), (20, False), (21, True), (45, True)):
                with self.subTest(gate=gate, width=width):
                    is_good, reason = face_quality.meets_pre_extraction_quality_criteria(
                        image,
                        10 + normalized_landmarks * width,
                        [10, 10, 10 + width, 10 + width],
                        DummyLogger(),
                        pose_image_shape=image.shape,
                        **thresholds,
                    )
                    self.assertEqual(is_good, expected_good)
                    self.assertIs(
                        reason,
                        None if expected_good else QualityReason.FACE_WIDTH_TOO_NARROW,
                    )

    def test_feature_quality_check_returns_quality_reason(self):
        is_good, reason = face_quality.is_high_quality_feature(
            np.zeros((1, 512), dtype=np.float32)
        )
        self.assertFalse(is_good)
        self.assertIs(reason, QualityReason.LOW_FEATURE_QUALITY_NORM)

    def test_person_face_quality_at_bind_threshold_matches_existing_face_id(self):
        seen = {}

        def fake_precheck(*args, **kwargs):
            seen.update(kwargs)
            return True, None

        def fake_feature_check(embedding, **kwargs):
            seen["norm_threshold"] = kwargs.get("norm_threshold")
            return True, None

        async def fake_search(embedding, threshold, device_ids):
            seen["similarity_threshold"] = threshold
            seen["device_ids"] = device_ids
            return "face-1", 0.82, "face-1"

        person_face.meets_pre_extraction_quality_criteria = fake_precheck
        person_face.is_high_quality_feature = fake_feature_check
        triton = FakePersonFaceTriton(
            quality_score=algorithm_config.person_face.binding.quality_min_score
        )
        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=fake_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_id"], "face-1")
        self.assertEqual(result["face_orientation"], "front")
        self.assertEqual(result["pitch"], 0.0)
        self.assertEqual(result["yaw"], 0.0)
        self.assertEqual(result["roll"], 0.0)
        self.assertEqual(len(triton.quality_inputs), 1)
        quality_image, quality_landmarks = triton.quality_inputs[0]
        self.assertGreater(quality_image.shape[0], 45)
        self.assertGreater(quality_image.shape[1], 45)
        np.testing.assert_array_equal(quality_landmarks, triton.landmarks)
        self.assertEqual(triton.feature_calls, 1)
        self.assertEqual(seen["laplacian_threshold"], algorithm_config.person_face.binding.laplacian_threshold)
        self.assertEqual(seen["min_face_width"], algorithm_config.person_face.binding.min_face_width_px)
        self.assertEqual(seen["max_pose_abs_degrees"], algorithm_config.person_face.binding.max_pose_abs_degrees)
        self.assertEqual(
            seen["geometry_std_ratio_threshold"],
            algorithm_config.person_face.binding.geometry_std_ratio_threshold,
        )
        self.assertEqual(seen["norm_threshold"], algorithm_config.person_face.binding.feature_norm_threshold)
        self.assertEqual(seen["similarity_threshold"], algorithm_config.face.matching.general_similarity_threshold)
        self.assertIsNone(seen["device_ids"])

    def test_person_face_quality_below_bind_threshold_returns_zero(self):
        person_face.meets_pre_extraction_quality_criteria = lambda *args, **kwargs: (True, None)
        triton = FakePersonFaceTriton(
            quality_score=algorithm_config.person_face.binding.quality_min_score - 0.01
        )

        async def unexpected_search(*args, **kwargs):
            raise AssertionError("search should not run below the bind threshold")

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )
        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertEqual(triton.feature_calls, 0)

    def test_person_bind_accepts_quality_below_standalone_recognition_floor(self):
        person_face.meets_pre_extraction_quality_criteria = (
            lambda *args, **kwargs: (True, None)
        )
        quality_score = algorithm_config.face.quality.recognition_min_score - 0.01
        self.assertGreaterEqual(quality_score, algorithm_config.person_face.binding.quality_min_score)
        triton = FakePersonFaceTriton(quality_score=quality_score)
        search_calls = 0

        async def fake_search(*args, **kwargs):
            nonlocal search_calls
            search_calls += 1
            return "face-1", 0.9, "face-1"

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=fake_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_id"], "face-1")
        self.assertEqual(triton.feature_calls, 1)
        self.assertEqual(search_calls, 1)

    def test_person_face_search_failure_returns_zero(self):
        person_face.meets_pre_extraction_quality_criteria = lambda *args, **kwargs: (True, None)
        person_face.is_high_quality_feature = lambda *args, **kwargs: (True, None)
        triton = FakePersonFaceTriton(quality_score=5.5)

        async def failing_search(*args, **kwargs):
            raise RuntimeError("search unavailable")

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=failing_search,
                logger=DummyLogger(),
            )
        )

        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertEqual(triton.feature_calls, 1)

    def test_mask_selects_target_face_over_higher_confidence_other_person(self):
        person_face.meets_pre_extraction_quality_criteria = lambda *args, **kwargs: (True, None)
        person_face.is_high_quality_feature = lambda *args, **kwargs: (True, None)
        triton = FakePersonFaceTriton(quality_score=5.5)
        triton.detections = [
            {"xyxy": [60, 10, 98, 55], "conf": 0.99, "cls": 0},
            {"xyxy": [10, 10, 55, 55], "conf": 0.70, "cls": 0},
        ]
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, :58] = 1

        async def fake_search(*args, **kwargs):
            return "face-target", 0.83, "face-target"

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=mask,
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=fake_search,
                logger=DummyLogger(),
            )
        )
        self.assertEqual(result["face_id"], "face-target")
        self.assertIn("人脸归属候选 1/2", result["info"])
        self.assertEqual(triton.keypoint_calls, 1)

    def test_pose_head_disambiguates_faces_inside_same_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        detections = [
            {"xyxy": [5, 10, 45, 55], "conf": 0.99},
            {"xyxy": [55, 10, 95, 55], "conf": 0.70},
        ]
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[:5] = np.array(
            [[75, 25, 0.9], [72, 23, 0.9], [78, 23, 0.9], [68, 26, 0.9], [82, 26, 0.9]],
            dtype=np.float32,
        )
        bbox, info = person_face._select_best_face_bbox(
            detections, (100, 100, 3), mask, pose
        )
        self.assertEqual(bbox, (55, 10, 95, 55))
        self.assertIn("Pose头部匹配=True", info)

    def test_pose_head_mismatch_reports_no_face_without_keypoints(self):
        search_calls = 0

        async def unexpected_search(*args, **kwargs):
            nonlocal search_calls
            search_calls += 1
            raise AssertionError("gallery search should not run for a Pose-head mismatch")

        pose = np.zeros((17, 3), dtype=np.float32)
        pose[:5] = np.array(
            [[85, 85, 0.9], [83, 84, 0.9], [87, 84, 0.9], [82, 86, 0.9], [88, 86, 0.9]],
            dtype=np.float32,
        )
        triton = FakePersonFaceTriton(quality_score=4.5)
        triton.detections = [{"xyxy": [10, 10, 55, 55], "conf": 0.9, "cls": 0}]

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=np.ones((100, 100), dtype=np.uint8),
                pose_keypoints=pose,
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
                enable_recognition=False,
            )
        )

        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertEqual(result["face_orientation"], "no_face")
        self.assertEqual(triton.detection_calls, 1)
        self.assertEqual(triton.keypoint_calls, 0)
        self.assertEqual(triton.quality_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertEqual(search_calls, 0)
        self.assertIn("均不匹配Pose头部", result["info"])

    def test_no_mask_owned_face_skips_feature_and_gallery_search(self):
        triton = FakePersonFaceTriton(quality_score=4.5)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[70:100, :] = 1

        async def unexpected_search(*args, **kwargs):
            raise AssertionError("gallery search should not run without an owned face")

        result = asyncio.run(
            person_face.match_face_id_from_person_crop_with_info(
                person_crop=np.zeros((100, 100, 3), dtype=np.uint8),
                person_mask=mask,
                device_id="CAM001",
                timestamp=1710000000000,
                triton_client=triton,
                search_callback=unexpected_search,
                logger=DummyLogger(),
            )
        )
        self.assertEqual(result["face_id"], person_face.UNMATCHED_FACE_ID)
        self.assertEqual(triton.keypoint_calls, 0)
        self.assertEqual(triton.feature_calls, 0)
        self.assertIn("均不满足人物mask", result["info"])


if __name__ == "__main__":
    unittest.main()
