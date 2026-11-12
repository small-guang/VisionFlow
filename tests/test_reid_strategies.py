import asyncio
from types import SimpleNamespace
import unittest

import numpy as np

from tests.image_execution_test_support import InlineImageExecutor

from core.keyframe_processor import KeyframeProcessor
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
    UNMATCHED_REID,
)
from utils.reid.person_identity import (
    REID_DECISION_CREATE_NEW,
    REID_DECISION_MATCH_EXISTING,
    REID_DECISION_UNASSIGNED,
    IdentityOutcome,
    IdentityRequest,
    PersonIdentityService,
    ReidAssignmentPolicy,
)
from utils.reid.person_search import ReidQueryPlanner
from utils.reid.reid_enrollment_state import REID_ENROLLMENT_STATE_FORMAL_PENDING
from utils.reid.reid_models import (
    REID_SEARCH_MODE_ANONYMOUS_GLOBAL,
    REID_SEARCH_MODE_SAME_FACE,
    REID_SOURCE_MILVUS,
    PersonReidSearchResult,
    ReidCandidate,
)
from utils.reid.reid_resolution import ReidClassResolver
from utils.reid.reid_state import REID_STATE_ACTIVE


def candidate(
    reid,
    score,
    *,
    face_id="",
    device_id="device-1",
    prototype_id="p",
):
    return ReidCandidate(
        reid=reid,
        score=score,
        threshold=0.5,
        source=REID_SOURCE_MILVUS,
        face_id=face_id,
        device_id=device_id,
        prototype_id=prototype_id,
    )


class ReidStrategyTests(unittest.TestCase):
    def test_query_planner_selects_one_attempt_per_query(self):
        planner = ReidQueryPlanner()
        anonymous = planner.plan("0")
        self.assertEqual([attempt.mode for attempt in anonymous], [REID_SEARCH_MODE_ANONYMOUS_GLOBAL])

        known_face = planner.plan("face-1")
        self.assertEqual(
            [attempt.mode for attempt in known_face],
            [REID_SEARCH_MODE_SAME_FACE],
        )
        self.assertEqual(known_face[0].candidate_face_id, "face-1")

    def test_profiles_preserve_resolution_contract(self):
        resolver = ReidClassResolver()
        cases = (
            (
                "same-face",
                [candidate("same", 0.60, face_id="face-1")],
                {"query_face_id": "face-1", "query_device_id": "other"},
                "same",
                "accepted",
            ),
            (
                "anonymous-body",
                [
                    candidate("body", 0.80, prototype_id="p1"),
                    candidate("body", 0.78, prototype_id="p2"),
                ],
                {"query_device_id": "device-1"},
                "body",
                "accepted",
            ),
            (
                "unsupported-singleton",
                [candidate("weak", 0.80)],
                {"query_device_id": "device-1"},
                None,
                "no_compatible_class",
            ),
        )
        for name, candidates, kwargs, expected_reid, expected_reason in cases:
            with self.subTest(name=name):
                result = resolver.resolve(candidates, **kwargs)
                self.assertEqual(result.reid, expected_reid)
                self.assertEqual(result.reason, expected_reason)

    def test_assignment_policy_match_create_and_unassigned(self):
        policy = ReidAssignmentPolicy()
        matched = policy.decide(
            PersonReidSearchResult(
                reid="known",
                source=REID_SOURCE_MILVUS,
                state=REID_STATE_ACTIVE,
            ),
            quality_level=QUALITY_LEVEL_RECOGNITION,
            face_id="0",
        )
        self.assertEqual(matched.action, REID_DECISION_MATCH_EXISTING)
        self.assertEqual(matched.reid, "known")

        created = policy.decide(
            PersonReidSearchResult(reid=None),
            quality_level=QUALITY_LEVEL_ENROLL,
            face_id="face-a",
        )
        self.assertEqual(created.action, REID_DECISION_CREATE_NEW)
        self.assertEqual(created.state, REID_STATE_ACTIVE)

        anonymous_unassigned = policy.decide(
            PersonReidSearchResult(reid=None),
            quality_level=QUALITY_LEVEL_ENROLL,
            face_id="0",
        )
        self.assertEqual(anonymous_unassigned.action, REID_DECISION_UNASSIGNED)
        self.assertEqual(anonymous_unassigned.reid, UNMATCHED_REID)

        unassigned = policy.decide(
            PersonReidSearchResult(reid=None),
            quality_level=QUALITY_LEVEL_RECOGNITION,
            face_id="0",
        )
        self.assertEqual(unassigned.action, REID_DECISION_UNASSIGNED)
        self.assertEqual(unassigned.reid, UNMATCHED_REID)


class PersonServiceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_search_and_identity_services_are_reused(self):
        class Triton:
            async def extract_person_segments(self, _image):
                return []

            async def extract_person_poses(self, _image):
                return []

        processor = KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=Triton(),
            feature_buffer=SimpleNamespace(sbs=object()),
            face_embedding_db=object(),
            person_sbs_embedding_db=object(),
        )
        pipeline = processor.person_processor._pipeline
        search_service = processor._person_reid_search
        identity_service = processor._person_identity_service
        self.assertIs(pipeline.identity_service, identity_service)
        self.assertIs(identity_service.search_persons.__self__, search_service)

        ctx = KeyframeContext(
            sn="device-1",
            timestamp=1,
            object_name="device-1/picture/frame.jpg",
            keyframe_id="frame",
        )
        for _ in range(2):
            await processor.person_processor.process_detection(
                ctx,
                np.zeros((8, 8, 3), dtype=np.uint8),
                "frame",
                1,
                ctx.object_name,
                object(),
            )

        self.assertIs(processor.person_processor._pipeline, pipeline)
        self.assertIs(processor._person_reid_search, search_service)
        self.assertIs(processor._person_identity_service, identity_service)


class PersonIdentityServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_identity_service_forwards_pano_flag_to_sbs_enrollment(self):
        offer_calls = []

        async def search(_embedding, **_kwargs):
            return PersonReidSearchResult(
                reid="reid-1",
                score=0.9,
                state=REID_STATE_ACTIVE,
                reason="accepted",
                canonical_face_id="face-a",
            )

        class Buffer:
            async def offer_feature(self, **kwargs):
                offer_calls.append(kwargs)
                return SimpleNamespace(
                    enrollment_state=REID_ENROLLMENT_STATE_FORMAL_PENDING,
                    reid_state=REID_STATE_ACTIVE,
                    reason="queued",
                    recognition_release=None,
                )

        async def run():
            service = PersonIdentityService(
                identity_lock=asyncio.Lock(),
                search_persons=search,
                sbs_buffer=Buffer(),
                logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
            )
            request = IdentityRequest(
                embedding=np.asarray([1.0], dtype=np.float32),
                observed_face_id="face-a",
                person_quality_level=QUALITY_LEVEL_ENROLL,
                person_quality_score=8.0,
                body_completeness="full_body",
                device_id="device-1",
                timestamp=1,
                keyframe_id="frame-1",
                picture_vector_id=42,
                keyframe_url="image.jpg",
                person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
                ai_id="ai-1",
                segment_index=0,
                is_pano_keyframe=True,
            )
            await service.assign(request, register_assignment=lambda _reid: None)

        asyncio.run(run())
        self.assertEqual(len(offer_calls), 1)
        self.assertIs(offer_calls[0]["is_pano_keyframe"], True)
        self.assertNotIn("defer_new_reid_recognition", offer_calls[0])

    async def test_anonymous_level_two_match_skips_enrollment(self):
        async def search(_embedding, **_kwargs):
            return PersonReidSearchResult(
                reid="known-reid",
                score=0.9,
                state=REID_STATE_ACTIVE,
                reason="accepted",
            )

        class Buffer:
            def __init__(self):
                self.offer_calls = []

            async def offer_feature(self, **kwargs):
                self.offer_calls.append(kwargs)
                raise AssertionError("anonymous match must not be offered for enrollment")

        buffer = Buffer()
        service = PersonIdentityService(
            identity_lock=asyncio.Lock(),
            search_persons=search,
            sbs_buffer=buffer,
            logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        )
        request = IdentityRequest(
            embedding=np.asarray([1], dtype=np.float32),
            observed_face_id="0",
            person_quality_level=QUALITY_LEVEL_ENROLL,
            person_quality_score=8.0,
            body_completeness="full_body",
            device_id="device-1",
            timestamp=1,
            keyframe_id="frame-1",
            picture_vector_id=1,
            keyframe_url="image.jpg",
            person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
            ai_id="person-1",
            segment_index=0,
        )

        outcome = await service.assign(
            request,
            register_assignment=lambda _reid: None,
        )

        self.assertEqual(outcome.reid, "known-reid")
        self.assertEqual(outcome.state, REID_STATE_ACTIVE)
        self.assertEqual(outcome.enrollment_state, "")
        self.assertEqual(buffer.offer_calls, [])


class PersonPipelineCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_reid_emitter_does_not_receive_release_keyword(self):
        from core.person_pipeline.models import FaceMatchOutcome
        from core.person_pipeline.pipeline import PersonDetectionPipeline
        from utils.reid.person_identity import IdentityOutcome

        emitted_kwargs = []

        async def legacy_emitter(*_args, **kwargs):
            if "new_reid_recognition_release" in kwargs:
                raise TypeError("legacy emitter does not accept release keyword")
            emitted_kwargs.append(kwargs)

        pipeline = object.__new__(PersonDetectionPipeline)
        pipeline.recognition_emitter = legacy_emitter
        prepared = SimpleNamespace(
            observation=SimpleNamespace(
                detect_time=7,
                people_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                ai_id="ai-1",
                person_orientation="front",
            ),
            quality_decision=SimpleNamespace(
                person_quality_score=8.0,
                person_quality_level=2,
                person_quality_action="build_gallery",
                mask_score=8.0,
                crop_score=8.0,
                quality_reasons=[],
            ),
            crop_path="device-1/person_crop/person.jpg",
        )
        await pipeline._emit_recognition(
            KeyframeContext(
                sn="device-1",
                timestamp=1,
                object_name="device-1/picture/frame.jpg",
                keyframe_id="frame-1",
            ),
            "frame-1",
            42,
            "device-1/picture/frame.jpg",
            prepared,
            SimpleNamespace(clip_failure=""),
            FaceMatchOutcome("face-a", "face matched", "front"),
            IdentityOutcome(
                reid="known-reid",
                status=2,
                state=REID_STATE_ACTIVE,
                decision=REID_DECISION_MATCH_EXISTING,
                is_new=False,
            ),
        )

        self.assertEqual(len(emitted_kwargs), 1)


class PersonIdentityLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_enrollment_result_missing_release_field_fails_fast(self):
        class IncompleteBuffer:
            async def offer_feature(self, **kwargs):
                return SimpleNamespace(
                    enrollment_state=REID_ENROLLMENT_STATE_FORMAL_PENDING,
                    reid_state=REID_STATE_ACTIVE,
                    reason="queued",
                )

        service = PersonIdentityService(
            identity_lock=asyncio.Lock(),
            search_persons=lambda *_args, **_kwargs: None,
            sbs_buffer=IncompleteBuffer(),
            logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        )
        request = IdentityRequest(
            embedding=np.asarray([1.0], dtype=np.float32),
            observed_face_id="face-1",
            person_quality_level=QUALITY_LEVEL_ENROLL,
            person_quality_score=8.0,
            body_completeness="full_body",
            device_id="camera-1",
            timestamp=1,
            keyframe_id="frame-1",
            picture_vector_id=1,
            keyframe_url="frame.jpg",
            person_rect={"x": 0, "y": 0, "w": 20, "h": 40},
            ai_id="ai-1",
            segment_index=0,
        )
        outcome = IdentityOutcome(reid="person-1", state=REID_STATE_ACTIVE)
        with self.assertRaisesRegex(AttributeError, "recognition_release"):
            await service._enroll(request, outcome)

    async def test_lock_covers_search_through_enrollment(self):
        search_calls = []
        enrollment_started = asyncio.Event()
        release_first = asyncio.Event()

        async def search(embedding, **_kwargs):
            marker = int(embedding[0])
            search_calls.append(marker)
            return PersonReidSearchResult(
                reid=f"reid-{marker}",
                score=0.9,
                state=REID_STATE_ACTIVE,
                reason="accepted",
            )

        class Buffer:
            async def offer_feature(self, **kwargs):
                if kwargs["ai_id"] == "first":
                    enrollment_started.set()
                    await release_first.wait()
                return SimpleNamespace(
                    enrollment_state=REID_ENROLLMENT_STATE_FORMAL_PENDING,
                    reid_state=REID_STATE_ACTIVE,
                    reason="queued",
                    recognition_release=None,
                )

        logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
        service = PersonIdentityService(
            identity_lock=asyncio.Lock(),
            search_persons=search,
            sbs_buffer=Buffer(),
            logger=logger,
        )

        def request(marker, ai_id):
            return IdentityRequest(
                embedding=np.asarray([marker], dtype=np.float32),
                observed_face_id="face-a",
                person_quality_level=QUALITY_LEVEL_ENROLL,
                person_quality_score=8.0,
                body_completeness="full_body",
                device_id="device-1",
                timestamp=1,
                keyframe_id=f"frame-{marker}",
                picture_vector_id=marker,
                keyframe_url="image.jpg",
                person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
                ai_id=ai_id,
                segment_index=0,
            )

        first = asyncio.create_task(
            service.assign(request(1, "first"), register_assignment=lambda _reid: None)
        )
        await enrollment_started.wait()
        second = asyncio.create_task(
            service.assign(request(2, "second"), register_assignment=lambda _reid: None)
        )
        await asyncio.sleep(0)
        self.assertEqual(search_calls, [1])

        release_first.set()
        await asyncio.gather(first, second)
        self.assertEqual(search_calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
