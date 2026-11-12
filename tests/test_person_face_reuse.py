import asyncio
import inspect
import types
import unittest
from unittest.mock import patch

import numpy as np

from core.face_observation import FaceProcessingStage, FrameFaceObservation
from core.infer.results import FaceQualityResult, FeatureResult
from core.person_pipeline import face_reuse
from core.person_pipeline.features import PersonFeatureService
from core.person_pipeline.face_reuse import FrameFaceBinding, bind_frame_faces_to_people
from core.person_pipeline.models import (
    FaceMatchOutcome,
    PersonObservation,
    PersonPipelineRuntime,
    PreparedPerson,
)
from tests.person_pose_test_support import full_body_keypoints
from utils.keyframe.detection_helpers import KeyframeContext
from utils.reid import person_face_recognition as person_face
from utils.reid.face_matching import FaceSampleMatch, resolve_face_matches


IMAGE_SHAPE = (100, 100, 3)
KEY_FRAME = np.ones((120, 120, 3), dtype=np.uint8)
LANDMARKS = np.array(
    [[35, 40], [65, 40], [50, 55], [40, 70], [60, 70]],
    dtype=np.float32,
)
NORMALIZED_EMBEDDING = np.concatenate(
    (np.ones(1, dtype=np.float32), np.zeros(511, dtype=np.float32))
)


def person(*, index, rect, mask, keypoints=None):
    return PersonObservation(
        segment_index=index,
        segment={"rect": dict(rect), "mask": mask},
        people_rect=dict(rect),
        mask_pixel_count=int(np.count_nonzero(mask)),
        confidence=0.9,
        person_keypoints=keypoints,
        person_orientation_result=types.SimpleNamespace(label="front"),
        quality_decision=None,
        ai_id=f"person-{index}",
        detect_time=0,
    )


def mask_with_rect(rect):
    mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
    x1, y1 = rect["x"], rect["y"]
    x2, y2 = x1 + rect["w"], y1 + rect["h"]
    mask[y1:y2, x1:x2] = 1
    return mask


def frame_face(
    *,
    assigned_face_id="0",
    orientation="front",
    completed_stage=FaceProcessingStage.IDENTITY,
    failure_reason="",
    source_landmarks=LANDMARKS,
    quality_score=5.5,
    normalized_embedding=NORMALIZED_EMBEDDING,
    raw_embedding_norm=4.0,
):
    return FrameFaceObservation(
        source_rect=(20, 20, 80, 80),
        confidence=0.9,
        expanded_source_rect=(14, 14, 86, 86),
        source_landmarks=source_landmarks,
        orientation=orientation,
        quality_score=quality_score,
        normalized_embedding=normalized_embedding,
        raw_embedding_norm=raw_embedding_norm,
        assigned_face_id=assigned_face_id,
        completed_stage=completed_stage,
        failure_reason=failure_reason,
    )


def invalid_landmark_chain_observations():
    return (
        (
            "missing_landmarks",
            frame_face(
                orientation="front",
                completed_stage=FaceProcessingStage.LANDMARKS,
                source_landmarks=None,
                quality_score=None,
                normalized_embedding=None,
                raw_embedding_norm=None,
            ),
        ),
        (
            "malformed_landmarks",
            frame_face(
                orientation="side",
                completed_stage=FaceProcessingStage.LANDMARKS,
                source_landmarks=np.ones((4, 2), dtype=np.float32),
                quality_score=None,
                normalized_embedding=None,
                raw_embedding_norm=None,
            ),
        ),
    )


def invalid_identity_chain_observations():
    missing_bbox = frame_face(assigned_face_id="face-direct")
    object.__setattr__(missing_bbox, "source_rect", None)
    return (
        ("missing_bbox", missing_bbox),
        (
            "missing_landmarks",
            frame_face(
                assigned_face_id="face-direct",
                source_landmarks=None,
            ),
        ),
        (
            "missing_quality",
            frame_face(
                assigned_face_id="face-direct",
                quality_score=None,
            ),
        ),
        (
            "malformed_embedding",
            frame_face(
                assigned_face_id="face-direct",
                normalized_embedding=np.ones(512, dtype=np.float32),
            ),
        ),
        (
            "missing_embedding",
            frame_face(
                assigned_face_id="face-direct",
                normalized_embedding=None,
            ),
        ),
        (
            "missing_raw_norm",
            frame_face(
                assigned_face_id="face-direct",
                raw_embedding_norm=None,
            ),
        ),
    )


class RecordingFaceTriton:
    def __init__(self):
        self.keypoint_calls = []
        self.quality_calls = []
        self.feature_calls = []

    @property
    def total_face_calls(self):
        return len(self.keypoint_calls) + len(self.quality_calls) + len(self.feature_calls)

    async def extract_face_keypoints(self, expanded_crop, bbox):
        self.keypoint_calls.append((expanded_crop, bbox))
        return LANDMARKS.copy()

    async def extract_face_quality(self, face_crop, landmarks):
        self.quality_calls.append((face_crop, landmarks))
        return FaceQualityResult(5.5)

    async def extract_face_features(self, expanded_crop, landmarks):
        self.feature_calls.append((expanded_crop, landmarks))
        return FeatureResult(np.ones(512, dtype=np.float32))


class RecordingSearch:
    def __init__(self, face_id="face-1", *, error=None):
        self.face_id = face_id
        self.error = error
        self.calls = []

    async def __call__(self, embedding, threshold, device_ids):
        self.calls.append((embedding, threshold, device_ids))
        if self.error is not None:
            raise self.error
        matched = self.face_id
        return matched, 0.91, matched or ""


class RecordingLogger:
    def __init__(self):
        self.exceptions = []

    def __getattr__(self, name):
        if name == "exception":
            return self.exceptions.append
        return lambda *args, **kwargs: None


class RecordingReuseService:
    def __init__(self, attempt):
        self.result = attempt
        self.calls = []

    async def attempt(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FaceReuseFrameStatsTests(unittest.TestCase):
    def _new_stats(self, **kwargs):
        stats_type = getattr(face_reuse, "FaceReuseFrameStats", None)
        self.assertIsNotNone(
            stats_type,
            "FaceReuseFrameStats must summarize reuse decisions per image",
        )
        return stats_type(**kwargs)

    def test_records_each_resolution_source_and_rotation(self):
        stats = self._new_stats(
            person_count=4,
            frame_face_count=3,
            bound_count=3,
        )

        stats.record(
            FaceMatchOutcome("face-1", "", "front", source="direct_id_reuse"),
            rotation_applied=True,
        )
        stats.record(
            FaceMatchOutcome("face-2", "", "front", source="partial_reuse"),
            rotation_applied=False,
        )
        stats.record(
            FaceMatchOutcome("0", "", "side", source="shared_orientation"),
            rotation_applied=True,
        )
        stats.record(
            FaceMatchOutcome("0", "", source="local_fallback"),
            rotation_applied=False,
        )

        self.assertEqual(stats.person_count, 4)
        self.assertEqual(stats.frame_face_count, 3)
        self.assertEqual(stats.bound_count, 3)
        self.assertEqual(stats.direct_id_reuse_count, 1)
        self.assertEqual(stats.partial_reuse_count, 1)
        self.assertEqual(stats.shared_orientation_count, 1)
        self.assertEqual(stats.local_fallback_count, 1)
        self.assertEqual(stats.rotated_person_count, 2)

    def test_id_disagreement_counts_only_nonzero_ids(self):
        stats = self._new_stats(person_count=3, frame_face_count=1)

        stats.record(
            FaceMatchOutcome(
                "local-face",
                "",
                source="local_fallback",
                shared_face_id="shared-face",
            ),
            rotation_applied=False,
        )
        stats.record(
            FaceMatchOutcome(
                "same-face",
                "",
                source="local_fallback",
                shared_face_id="same-face",
            ),
            rotation_applied=False,
        )
        stats.record(
            FaceMatchOutcome(
                "local-face",
                "",
                source="local_fallback",
                shared_face_id="0",
            ),
            rotation_applied=False,
        )

        self.assertEqual(stats.local_fallback_count, 3)
        self.assertEqual(stats.id_disagreement_count, 1)

    def test_any_failed_shared_attempt_with_local_nonzero_counts_recovery(self):
        stats = self._new_stats(person_count=3, frame_face_count=1)

        stats.record(
            FaceMatchOutcome(
                "local-face",
                "",
                source="local_fallback",
                fallback_reason="shared_face_id_zero",
            ),
            rotation_applied=False,
        )
        stats.record(
            FaceMatchOutcome(
                "0",
                "",
                source="local_fallback",
                fallback_reason="shared_face_id_zero",
            ),
            rotation_applied=False,
        )
        stats.record(
            FaceMatchOutcome(
                "another-local-face",
                "",
                source="local_fallback",
                fallback_reason="no_owned_face",
            ),
            rotation_applied=False,
        )

        self.assertEqual(stats.local_fallback_count, 3)
        self.assertEqual(stats.fallback_recovered_count, 2)

    def test_face_outcome_shared_diagnostics_have_backward_compatible_defaults(self):
        outcome = FaceMatchOutcome("0", "legacy constructor")

        self.assertEqual(getattr(outcome, "shared_source", None), "")
        self.assertEqual(getattr(outcome, "shared_fallback_reason", None), "")

    @staticmethod
    def _with_shared_diagnostics(outcome, **diagnostics):
        for name, value in diagnostics.items():
            object.__setattr__(outcome, name, value)
        return outcome

    def test_shared_success_counts_source_and_rotation_independently(self):
        stats = self._new_stats(person_count=3, frame_face_count=3, bound_count=3)
        cases = (
            ("direct_id_reuse", "face-direct", "front", True),
            ("partial_reuse", "face-partial", "side", False),
            ("shared_orientation", "0", "front", True),
        )

        for source, shared_face_id, shared_orientation, rotated in cases:
            outcome = self._with_shared_diagnostics(
                FaceMatchOutcome(
                    shared_face_id,
                    "local result",
                    shared_orientation,
                    source="local_fallback",
                    shared_face_id=shared_face_id,
                    shared_orientation=shared_orientation,
                ),
                shared_source=source,
                shared_fallback_reason="",
            )
            stats.record(outcome, rotation_applied=rotated)

        self.assertEqual(stats.direct_id_reuse_count, 1)
        self.assertEqual(stats.partial_reuse_count, 1)
        self.assertEqual(stats.shared_orientation_count, 1)
        self.assertEqual(stats.local_fallback_count, 3)
        self.assertEqual(getattr(stats, "shared_success_rotated_count", None), 2)
        self.assertEqual(getattr(stats, "shared_success_unrotated_count", None), 1)
        self.assertEqual(getattr(stats, "local_fallback_rotated_count", None), 2)
        self.assertEqual(getattr(stats, "local_fallback_unrotated_count", None), 1)

    def test_id_disagreement_categories_are_mutually_exclusive_and_aggregate(self):
        stats = self._new_stats(person_count=4, frame_face_count=4)
        comparisons = (
            ("shared-face", "0", "", "direct_id_reuse"),
            ("0", "local-face", "no_owned_face", ""),
            ("shared-a", "local-b", "", "direct_id_reuse"),
            ("same-face", "same-face", "", "direct_id_reuse"),
        )

        for shared_id, local_id, reason, source in comparisons:
            outcome = self._with_shared_diagnostics(
                FaceMatchOutcome(
                    local_id,
                    "local result",
                    "front",
                    source="local_fallback",
                    fallback_reason=reason,
                    shared_face_id=shared_id,
                    shared_orientation="front",
                ),
                shared_source=source,
                shared_fallback_reason=reason,
            )
            stats.record(outcome, rotation_applied=False)

        self.assertEqual(
            getattr(stats, "shared_nonzero_local_zero_count", None),
            1,
        )
        self.assertEqual(
            getattr(stats, "local_nonzero_shared_zero_count", None),
            1,
        )
        self.assertEqual(getattr(stats, "different_nonzero_id_count", None), 1)
        self.assertEqual(stats.id_disagreement_count, 3)

    def test_orientation_disagreement_excludes_unknown_no_face_and_equal(self):
        stats = self._new_stats(person_count=4, frame_face_count=4)
        for shared_orientation, local_orientation in (
            ("front", "side"),
            ("front", "front"),
            ("unknown", "side"),
            ("no_face", "front"),
        ):
            outcome = self._with_shared_diagnostics(
                FaceMatchOutcome(
                    "same-face",
                    "local result",
                    local_orientation,
                    source="local_fallback",
                    shared_face_id="same-face",
                    shared_orientation=shared_orientation,
                ),
                shared_source="direct_id_reuse",
                shared_fallback_reason="",
            )
            stats.record(outcome, rotation_applied=False)

        self.assertEqual(getattr(stats, "orientation_disagreement_count", None), 1)

    def test_fallback_recovery_reasons_and_rotation_are_stratified(self):
        stats = self._new_stats(person_count=4, frame_face_count=2)
        cases = (
            ("no_owned_face", "local-a", True),
            ("shared_face_id_zero", "local-b", False),
            ("shared_resolution_failed", "0", True),
            ("", "local-off", False),
        )

        for reason, local_id, rotated in cases:
            outcome = self._with_shared_diagnostics(
                FaceMatchOutcome(
                    local_id,
                    "local result",
                    source="local_fallback",
                    fallback_reason=reason,
                ),
                shared_source="partial_reuse" if reason == "shared_face_id_zero" else "",
                shared_fallback_reason=reason,
            )
            stats.record(outcome, rotation_applied=rotated)

        self.assertEqual(stats.fallback_recovered_count, 2)
        self.assertEqual(
            getattr(stats, "local_nonzero_recovery_rotated_count", None),
            1,
        )
        self.assertEqual(
            getattr(stats, "local_nonzero_recovery_unrotated_count", None),
            1,
        )
        self.assertEqual(
            getattr(stats, "fallback_reason_counts", None),
            {
                "no_owned_face": 1,
                "shared_face_id_zero": 1,
                "shared_resolution_failed": 1,
            },
        )

    def test_unusable_shared_source_without_reason_can_still_recover_locally(self):
        stats = self._new_stats(person_count=1, frame_face_count=1)
        outcome = self._with_shared_diagnostics(
            FaceMatchOutcome(
                "local-face",
                "local result",
                source="local_fallback",
            ),
            shared_source="partial_reuse",
        )

        stats.record(outcome, rotation_applied=False)

        self.assertEqual(stats.fallback_recovered_count, 1)
        self.assertEqual(stats.local_nonzero_recovery_unrotated_count, 1)


class PersonFeatureReuseModeTests(unittest.TestCase):
    def setUp(self):
        self.ctx = KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/main.jpg",
            keyframe_id="keyframe-id",
        )
        self.key_frame = np.full((120, 120, 3), 7, dtype=np.uint8)
        self.rotated_crop = np.full((40, 20, 3), 13, dtype=np.uint8)
        self.person_mask = np.ones((40, 20), dtype=np.uint8)
        self.transformed_pose = object()
        observation = person(
            index=0,
            rect={"x": 10, "y": 10, "w": 20, "h": 40},
            mask=np.ones(IMAGE_SHAPE[:2], dtype=np.uint8),
            keypoints=LANDMARKS,
        )
        crop_result = types.SimpleNamespace(
            image=self.rotated_crop,
            mask=self.person_mask,
        )
        self.prepared = PreparedPerson(
            observation=observation,
            crop_result=crop_result,
            quality_decision=None,
            crop_info="",
        )
        self.binding = FrameFaceBinding(frame_face(), "bound")

    def _service(self, *, mode, local_callback, shared_attempt):
        runtime = PersonPipelineRuntime(
            encode_image=lambda image: b"",
            evaluate_visual_quality=lambda image, mask: None,
            match_face=local_callback,
            pose_keypoints_to_crop=lambda keypoints, crop: self.transformed_pose,
        )
        try:
            service = PersonFeatureService(
                triton_client=RecordingFaceTriton(),
                feature_buffer=object(),
                search_faces_with_pending_locked=RecordingSearch(),
                logger=RecordingLogger(),
                runtime=runtime,
                reuse_mode=mode,
            )
        except TypeError as error:
            self.fail(f"PersonFeatureService lacks mode-aware construction: {error}")
        service.reuse_service = RecordingReuseService(shared_attempt)
        return service

    def _match(self, service, *, enable_recognition=True, binding=None):
        try:
            return asyncio.run(
                service.match_face(
                    self.ctx,
                    self.key_frame,
                    self.prepared,
                    binding or self.binding,
                    enable_recognition=enable_recognition,
                )
            )
        except TypeError as error:
            self.fail(f"PersonFeatureService lacks the mode-aware match API: {error}")

    @staticmethod
    def _install_real_reuse(service, *, search):
        service.reuse_service = face_reuse.PersonFaceReuseService(
            triton_client=service.triton_client,
            search_faces_with_pending_locked=search,
            logger=service.logger,
        )

    def test_observe_mode_is_rejected(self):
        async def local_callback(**kwargs):
            return {
                "face_id": "local-face",
                "info": "local result",
                "face_orientation": "front",
            }

        with self.assertRaisesRegex(ValueError, "invalid person face reuse mode"):
            self._service(
                mode="observe",
                local_callback=local_callback,
                shared_attempt=face_reuse.SharedFaceAttempt(),
            )

    def test_off_skips_shared_attempt_and_passes_recognition_mode(self):
        local_calls = []

        async def local_callback(**kwargs):
            local_calls.append(kwargs)
            return {
                "face_id": "local-face",
                "info": "local result",
                "face_orientation": "side",
            }

        service = self._service(
            mode="off",
            local_callback=local_callback,
            shared_attempt=face_reuse.SharedFaceAttempt(
                outcome=FaceMatchOutcome("shared-face", "shared", "front")
            ),
        )

        outcome = self._match(service)

        self.assertEqual(outcome.face_id, "local-face")
        self.assertEqual(outcome.source, "local_fallback")
        self.assertEqual(service.reuse_service.calls, [])
        self.assertEqual(len(local_calls), 1)
        self.assertIs(local_calls[0]["person_crop"], self.rotated_crop)
        self.assertIs(local_calls[0]["person_mask"], self.person_mask)
        self.assertIs(local_calls[0]["pose_keypoints"], self.transformed_pose)
        self.assertEqual(local_calls[0]["device_id"], "CAM001")
        self.assertEqual(local_calls[0]["timestamp"], 1710000000000)
        self.assertIs(local_calls[0]["enable_recognition"], True)




    def test_on_returns_nonzero_shared_identity_without_calling_local(self):
        async def unexpected_local(**kwargs):
            raise AssertionError("local fallback must not run for a reliable shared identity")

        shared = FaceMatchOutcome(
            "shared-face",
            "shared result",
            "front",
            source="direct_id_reuse",
        )
        service = self._service(
            mode="on",
            local_callback=unexpected_local,
            shared_attempt=face_reuse.SharedFaceAttempt(
                outcome=shared,
                source="direct_id_reuse",
            ),
        )

        outcome = self._match(service)

        self.assertIs(outcome, shared)
        self.assertEqual(len(service.reuse_service.calls), 1)

    def test_on_rotated_person_forces_the_existing_local_face_flow(self):
        local_calls = []

        async def local_callback(**kwargs):
            local_calls.append(kwargs)
            return {
                "face_id": "local-face",
                "info": "local rotated result",
                "face_orientation": "front",
            }

        self.prepared.crop_result.rotation_applied = True
        service = self._service(
            mode="on",
            local_callback=local_callback,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )
        self._install_real_reuse(
            service,
            search=RecordingSearch("shared-face"),
        )
        binding = FrameFaceBinding(
            frame_face(assigned_face_id="shared-face"),
            "bound",
        )

        outcome = self._match(service, binding=binding)

        self.assertEqual(outcome.face_id, "local-face")
        self.assertEqual(outcome.source, "local_fallback")
        self.assertEqual(outcome.fallback_reason, "person_rotated")
        self.assertEqual(len(local_calls), 1)

    def test_on_rejects_container_face_id_and_runs_local_fallback(self):
        local_calls = []

        async def local_callback(**kwargs):
            local_calls.append(kwargs)
            return {
                "face_id": "local-face",
                "info": "local result",
                "face_orientation": "side",
            }

        service = self._service(
            mode="on",
            local_callback=local_callback,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )
        service.reuse_service = face_reuse.PersonFaceReuseService(
            triton_client=service.triton_client,
            search_faces_with_pending_locked=service.search_faces_with_pending_locked,
            logger=service.logger,
        )
        binding = FrameFaceBinding(
            frame_face(assigned_face_id=["bad-id"]),
            "bound",
        )

        outcome = self._match(service, binding=binding)

        self.assertEqual(len(local_calls), 1)
        self.assertEqual(outcome.face_id, "local-face")
        self.assertEqual(outcome.orientation, "side")
        self.assertTrue(outcome.fallback_reason)

    def test_on_invalid_raw_face_ids_never_enter_partial_search(self):
        invalid_ids = (
            ("list", ["bad-id"]),
            ("dict", {"id": "bad-id"}),
            ("float", 42.0),
            ("bool", True),
        )
        for id_kind, raw_face_id in invalid_ids:
            with self.subTest(id_kind=id_kind):
                local_calls = []

                async def local_callback(**kwargs):
                    local_calls.append(kwargs)
                    return {
                        "face_id": "local-face",
                        "info": "local result",
                        "face_orientation": "side",
                    }

                search = RecordingSearch("shared-search-hit")
                service = self._service(
                    mode="on",
                    local_callback=local_callback,
                    shared_attempt=face_reuse.SharedFaceAttempt(),
                )
                self._install_real_reuse(
                    service,
                    search=search,
                )
                binding = FrameFaceBinding(
                    frame_face(assigned_face_id=raw_face_id),
                    "bound",
                )

                with (
                    patch.object(
                        person_face,
                        "meets_pre_extraction_quality_criteria",
                        return_value=(True, None),
                    ),
                    patch.object(
                        person_face,
                        "is_high_quality_feature",
                        return_value=(True, None),
                    ),
                ):
                    outcome = self._match(service, binding=binding)

                self.assertEqual(search.calls, [])
                self.assertEqual(len(local_calls), 1)
                self.assertEqual(outcome.face_id, "local-face")
                self.assertEqual(outcome.orientation, "side")
                self.assertEqual(
                    outcome.fallback_reason,
                    "shared_face_id_malformed",
                )

    def test_on_valid_zero_face_ids_remain_eligible_for_partial_search(self):
        for raw_face_id in ("0", 0):
            with self.subTest(raw_face_id=raw_face_id):
                async def unexpected_local(**kwargs):
                    raise AssertionError(
                        "valid unmatched ID should permit reliable partial reuse"
                    )

                search = RecordingSearch("shared-search-hit")
                service = self._service(
                    mode="on",
                    local_callback=unexpected_local,
                    shared_attempt=face_reuse.SharedFaceAttempt(),
                )
                self._install_real_reuse(
                    service,
                    search=search,
                )
                binding = FrameFaceBinding(
                    frame_face(assigned_face_id=raw_face_id),
                    "bound",
                )

                with (
                    patch.object(
                        person_face,
                        "meets_pre_extraction_quality_criteria",
                        return_value=(True, None),
                    ),
                    patch.object(
                        person_face,
                        "is_high_quality_feature",
                        return_value=(True, None),
                    ),
                ):
                    outcome = self._match(service, binding=binding)

                self.assertEqual(outcome.face_id, "shared-search-hit")
                self.assertEqual(outcome.source, "partial_reuse")
                self.assertEqual(len(search.calls), 1)

    def test_on_valid_nonzero_scalar_ids_remain_direct_identities(self):
        for raw_face_id, expected in (("face-123", "face-123"), (42, "42")):
            with self.subTest(raw_face_id=raw_face_id):
                async def unexpected_local(**kwargs):
                    raise AssertionError("reliable direct identity must skip local")

                search = RecordingSearch("unexpected-search-hit")
                service = self._service(
                    mode="on",
                    local_callback=unexpected_local,
                    shared_attempt=face_reuse.SharedFaceAttempt(),
                )
                self._install_real_reuse(
                    service,
                    search=search,
                )
                binding = FrameFaceBinding(
                    frame_face(assigned_face_id=raw_face_id),
                    "bound",
                )

                outcome = self._match(service, binding=binding)

                self.assertEqual(outcome.face_id, expected)
                self.assertEqual(outcome.source, "direct_id_reuse")
                self.assertEqual(search.calls, [])

    def test_on_direct_identity_with_invalid_promised_chain_runs_local_once(self):
        for name, observation in invalid_identity_chain_observations():
            with self.subTest(name=name):
                local_calls = []

                async def local_callback(**kwargs):
                    local_calls.append(kwargs)
                    return {
                        "face_id": "local-face",
                        "info": "local result",
                        "face_orientation": "side",
                    }

                search = RecordingSearch("unexpected-search-hit")
                service = self._service(
                    mode="on",
                    local_callback=local_callback,
                    shared_attempt=face_reuse.SharedFaceAttempt(),
                )
                self._install_real_reuse(
                    service,
                    search=search,
                )

                outcome = self._match(
                    service,
                    binding=FrameFaceBinding(observation, "bound"),
                )

                self.assertEqual(len(local_calls), 1)
                self.assertEqual(outcome.face_id, "local-face")
                self.assertEqual(outcome.orientation, "side")
                self.assertEqual(service.triton_client.total_face_calls, 0)
                self.assertEqual(search.calls, [])

        async def unexpected_local(**kwargs):
            raise AssertionError("complete identity must retain the direct fast path")

        search = RecordingSearch("unexpected-search-hit")
        service = self._service(
            mode="on",
            local_callback=unexpected_local,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )
        self._install_real_reuse(
            service,
            search=search,
        )

        outcome = self._match(
            service,
            binding=FrameFaceBinding(
                frame_face(assigned_face_id="face-direct"),
                "bound",
            ),
        )

        self.assertEqual(outcome.face_id, "face-direct")
        self.assertEqual(outcome.source, "direct_id_reuse")
        self.assertEqual(service.triton_client.total_face_calls, 0)
        self.assertEqual(search.calls, [])

    def test_on_runs_local_for_shared_zero_or_failure_and_preserves_fallback_reason(self):
        cases = (
            (
                "zero",
                face_reuse.SharedFaceAttempt(
                    outcome=FaceMatchOutcome(
                        "0",
                        "no hit",
                        "front",
                        source="partial_reuse",
                    ),
                    source="partial_reuse",
                    fallback_reason="shared_face_id_zero",
                ),
                "shared_face_id_zero",
            ),
            (
                "failure",
                face_reuse.SharedFaceAttempt(
                    fallback_reason="shared_resolution_failed"
                ),
                "shared_resolution_failed",
            ),
        )
        for name, shared_attempt, fallback_reason in cases:
            with self.subTest(name=name):
                local_calls = []

                async def local_callback(**kwargs):
                    local_calls.append(kwargs)
                    return {
                        "face_id": "local-face",
                        "info": "local result",
                        "face_orientation": "side",
                    }

                service = self._service(
                    mode="on",
                    local_callback=local_callback,
                    shared_attempt=shared_attempt,
                )

                outcome = self._match(service)

                self.assertEqual(len(local_calls), 1)
                self.assertEqual(outcome.face_id, "local-face")
                self.assertEqual(outcome.orientation, "side")
                self.assertEqual(outcome.fallback_reason, fallback_reason)
                self.assertEqual(
                    getattr(outcome, "shared_source", None),
                    shared_attempt.source,
                )
                self.assertEqual(
                    getattr(outcome, "shared_fallback_reason", None),
                    fallback_reason,
                )

    def test_on_orientation_only_returns_reliable_shared_orientation_without_local(self):
        async def unexpected_local(**kwargs):
            raise AssertionError("local orientation must not run for reliable shared orientation")

        shared = FaceMatchOutcome(
            "0",
            "shared orientation",
            "side",
            source="shared_orientation",
        )
        service = self._service(
            mode="on",
            local_callback=unexpected_local,
            shared_attempt=face_reuse.SharedFaceAttempt(
                outcome=shared,
                source="shared_orientation",
            ),
        )

        outcome = self._match(service, enable_recognition=False)

        self.assertIs(outcome, shared)
        self.assertEqual(len(service.reuse_service.calls), 1)

    def test_on_orientation_only_detection_stage_runs_local_fallback(self):
        local_calls = []

        async def local_callback(**kwargs):
            local_calls.append(kwargs)
            return {
                "face_id": "0",
                "info": "local orientation",
                "face_orientation": "side",
            }

        service = self._service(
            mode="on",
            local_callback=local_callback,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )
        service.reuse_service = face_reuse.PersonFaceReuseService(
            triton_client=service.triton_client,
            search_faces_with_pending_locked=service.search_faces_with_pending_locked,
            logger=service.logger,
        )
        binding = FrameFaceBinding(
            frame_face(
                orientation="front",
                completed_stage=FaceProcessingStage.DETECTION,
            ),
            "bound",
        )

        outcome = self._match(
            service,
            enable_recognition=False,
            binding=binding,
        )

        self.assertEqual(len(local_calls), 1)
        self.assertEqual(outcome.face_id, "0")
        self.assertEqual(outcome.orientation, "side")
        self.assertEqual(outcome.fallback_reason, "shared_orientation_incomplete")

    def test_on_orientation_with_invalid_promised_landmarks_runs_local_once(self):
        for name, observation in invalid_landmark_chain_observations():
            with self.subTest(name=name):
                local_calls = []

                async def local_callback(**kwargs):
                    local_calls.append(kwargs)
                    return {
                        "face_id": "0",
                        "info": "local orientation",
                        "face_orientation": "front",
                    }

                search = RecordingSearch("unexpected-search-hit")
                service = self._service(
                    mode="on",
                    local_callback=local_callback,
                    shared_attempt=face_reuse.SharedFaceAttempt(),
                )
                self._install_real_reuse(
                    service,
                    search=search,
                )

                outcome = self._match(
                    service,
                    enable_recognition=False,
                    binding=FrameFaceBinding(observation, "bound"),
                )

                self.assertEqual(len(local_calls), 1)
                self.assertEqual(outcome.face_id, "0")
                self.assertEqual(outcome.orientation, "front")
                self.assertEqual(service.triton_client.total_face_calls, 0)
                self.assertEqual(search.calls, [])

        async def unexpected_local(**kwargs):
            raise AssertionError("valid landmarks must retain shared orientation")

        search = RecordingSearch("unexpected-search-hit")
        service = self._service(
            mode="on",
            local_callback=unexpected_local,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )
        self._install_real_reuse(
            service,
            search=search,
        )

        outcome = self._match(
            service,
            enable_recognition=False,
            binding=FrameFaceBinding(
                frame_face(
                    orientation="side",
                    completed_stage=FaceProcessingStage.LANDMARKS,
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
                "bound",
            ),
        )

        self.assertEqual(outcome.face_id, "0")
        self.assertEqual(outcome.orientation, "front")
        self.assertEqual(outcome.source, "shared_orientation")
        self.assertEqual(service.triton_client.total_face_calls, 0)
        self.assertEqual(search.calls, [])

    def test_on_ambiguous_nonduplicate_people_each_run_local_fallback(self):
        local_calls = []

        async def local_callback(**kwargs):
            local_calls.append(kwargs)
            return {
                "face_id": "local-face",
                "info": "local result",
                "face_orientation": "front",
            }

        service = self._service(
            mode="on",
            local_callback=local_callback,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )
        self._install_real_reuse(
            service,
            search=RecordingSearch(),
        )
        shared_mask = np.ones(IMAGE_SHAPE[:2], dtype=np.uint8)
        shared_observation = FrameFaceObservation(
            source_rect=(40, 20, 60, 40),
            confidence=0.9,
        )
        people = [
            person(
                index=0,
                rect={"x": 0, "y": 0, "w": 30, "h": 100},
                mask=shared_mask,
            ),
            person(
                index=1,
                rect={"x": 70, "y": 0, "w": 30, "h": 100},
                mask=shared_mask,
            ),
        ]
        bindings = bind_frame_faces_to_people(
            [shared_observation],
            people,
            IMAGE_SHAPE,
        )

        outcomes = [self._match(service, binding=binding) for binding in bindings]

        self.assertEqual(len(local_calls), 2)
        self.assertEqual([outcome.face_id for outcome in outcomes], ["local-face"] * 2)
        self.assertEqual(
            [outcome.fallback_reason for outcome in outcomes],
            ["ambiguous_non_duplicate_people"] * 2,
        )

    def test_local_callback_exception_retains_face_id_zero_behavior(self):
        async def failing_local(**kwargs):
            raise RuntimeError("local callback failed")

        service = self._service(
            mode="off",
            local_callback=failing_local,
            shared_attempt=face_reuse.SharedFaceAttempt(),
        )

        outcome = self._match(service)

        self.assertEqual(outcome.face_id, "0")
        self.assertEqual(outcome.orientation, "unknown")
        self.assertIn("local callback failed", outcome.info)


class PersonFaceReuseServiceTests(unittest.TestCase):
    def setUp(self):
        self.ctx = KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/main.jpg",
            keyframe_id="keyframe-id",
        )

    def _service(self, *, triton=None, search=None, logger=None):
        service_type = getattr(face_reuse, "PersonFaceReuseService", None)
        self.assertIsNotNone(
            service_type,
            "PersonFaceReuseService must resolve reliable shared face stages",
        )
        return service_type(
            triton_client=triton or RecordingFaceTriton(),
            search_faces_with_pending_locked=search or RecordingSearch(),
            logger=logger or RecordingLogger(),
        )

    def _attempt(self, service, observation, *, enable_recognition=True, reason="bound"):
        return asyncio.run(
            service.attempt(
                ctx=self.ctx,
                key_frame=KEY_FRAME,
                binding=FrameFaceBinding(observation, reason),
                enable_recognition=enable_recognition,
            )
        )

    def test_stage_is_fixed_to_feature_and_not_configurable(self):
        self.assertNotIn(
            "max_stage",
            inspect.signature(face_reuse.PersonFaceReuseService).parameters,
        )
        service = face_reuse.PersonFaceReuseService(
            triton_client=RecordingFaceTriton(),
            search_faces_with_pending_locked=RecordingSearch(),
            logger=RecordingLogger(),
        )

        self.assertFalse(hasattr(service, "max_stage"))

    def test_reuses_uniquely_bound_reliable_nonzero_identity_without_face_calls(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch()
        service = self._service(triton=triton, search=search)

        attempt = self._attempt(service, frame_face(assigned_face_id="face-1"))

        self.assertEqual(attempt.outcome.face_id, "face-1")
        self.assertEqual(attempt.source, "direct_id_reuse")
        self.assertEqual(triton.total_face_calls, 0)
        self.assertEqual(search.calls, [])

    def test_direct_identity_recomputes_person_rule_orientation(self):
        service = self._service()

        attempt = self._attempt(
            service,
            frame_face(assigned_face_id="face-1", orientation="side"),
        )

        self.assertEqual(attempt.outcome.face_id, "face-1")
        self.assertEqual(attempt.outcome.orientation, "front")
        self.assertEqual(attempt.source, "direct_id_reuse")

    def test_direct_identity_requires_a_valid_complete_structural_chain(self):
        for name, observation in invalid_identity_chain_observations():
            with self.subTest(name=name):
                triton = RecordingFaceTriton()
                search = RecordingSearch("unexpected-search-hit")
                service = self._service(
                    triton=triton,
                    search=search,
                )

                attempt = self._attempt(service, observation)

                self.assertIsNone(attempt.outcome)
                self.assertTrue(attempt.fallback_reason)
                self.assertEqual(triton.total_face_calls, 0)
                self.assertEqual(search.calls, [])

        triton = RecordingFaceTriton()
        search = RecordingSearch("unexpected-search-hit")
        service = self._service(
            triton=triton,
            search=search,
        )

        attempt = self._attempt(
            service,
            frame_face(assigned_face_id="face-direct"),
        )

        self.assertEqual(attempt.outcome.face_id, "face-direct")
        self.assertEqual(attempt.source, "direct_id_reuse")
        self.assertEqual(triton.total_face_calls, 0)
        self.assertEqual(search.calls, [])


    def test_direct_resolution_rejects_container_face_id_before_coercion(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch()
        service = self._service(triton=triton, search=search)
        observation = frame_face(assigned_face_id=["bad-id"])

        attempt = self._attempt(service, observation)

        self.assertIsNone(attempt.outcome)
        self.assertTrue(attempt.fallback_reason)
        self.assertEqual(triton.total_face_calls, 0)
        self.assertEqual(search.calls, [])

    def test_quality_stage_reuses_landmarks_and_quality_but_extracts_a_feature(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch()
        service = self._service(triton=triton, search=search)

        with (
            patch.object(
                person_face,
                "meets_pre_extraction_quality_criteria",
                return_value=(True, None),
            ),
            patch.object(person_face, "is_high_quality_feature", return_value=(True, None)),
        ):
            attempt = self._attempt(
                service,
                frame_face(
                    completed_stage=FaceProcessingStage.QUALITY,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
            )

        self.assertEqual(attempt.outcome.face_id, "face-1")
        self.assertEqual(attempt.source, "partial_reuse")
        self.assertEqual(triton.keypoint_calls, [])
        self.assertEqual(triton.quality_calls, [])
        self.assertEqual(len(triton.feature_calls), 1)
        self.assertEqual(len(search.calls), 1)

    def test_partial_reuse_applies_general_threshold_and_dual_sample_confirmation(self):
        for count, expected in ((1, "0"), (2, "face-1")):
            with self.subTest(count=count):
                samples = [FaceSampleMatch("face-1", 0.49, "one"),
                           FaceSampleMatch("face-1", 0.46, "two")][:count]

                async def search(embedding, threshold, device_ids):
                    return resolve_face_matches(samples, threshold)

                service = self._service(search=search)
                with (
                    patch.object(person_face, "meets_pre_extraction_quality_criteria",
                                 return_value=(True, None)),
                    patch.object(person_face, "is_high_quality_feature",
                                 return_value=(True, None)),
                ):
                    attempt = self._attempt(service, frame_face())
                self.assertEqual(attempt.outcome.face_id, expected)

    def test_quality_stage_reuses_completed_quality_despite_feature_failure(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch()
        service = self._service(triton=triton, search=search)
        observation = frame_face(
            completed_stage=FaceProcessingStage.QUALITY,
            failure_reason="Face feature extraction failed: producer timeout",
            normalized_embedding=None,
            raw_embedding_norm=None,
        )

        with (
            patch.object(
                person_face,
                "meets_pre_extraction_quality_criteria",
                return_value=(True, None),
            ),
            patch.object(person_face, "is_high_quality_feature", return_value=(True, None)),
        ):
            attempt = self._attempt(service, observation)

        self.assertEqual(attempt.outcome.face_id, "face-1")
        self.assertEqual(attempt.source, "partial_reuse")
        self.assertEqual(triton.keypoint_calls, [])
        self.assertEqual(triton.quality_calls, [])
        self.assertEqual(len(triton.feature_calls), 1)
        self.assertEqual(len(search.calls), 1)

    def test_feature_reuses_complete_artifacts_but_runs_person_gates_and_locked_search(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch()
        service = self._service(triton=triton, search=search)
        gate_calls = {"precheck": 0, "feature": 0}

        def accept_precheck(*args, **kwargs):
            gate_calls["precheck"] += 1
            return True, None

        def accept_feature(*args, **kwargs):
            gate_calls["feature"] += 1
            return True, None

        with (
            patch.object(
                person_face,
                "meets_pre_extraction_quality_criteria",
                side_effect=accept_precheck,
            ),
            patch.object(
                person_face,
                "is_high_quality_feature",
                side_effect=accept_feature,
            ),
        ):
            attempt = self._attempt(service, frame_face())

        self.assertEqual(attempt.outcome.face_id, "face-1")
        self.assertEqual(attempt.source, "partial_reuse")
        self.assertEqual(triton.total_face_calls, 0)
        self.assertEqual(gate_calls, {"precheck": 1, "feature": 1})
        self.assertEqual(len(search.calls), 1)
        np.testing.assert_array_equal(search.calls[0][0], NORMALIZED_EMBEDDING)

    def test_feature_reuses_valid_embedding_despite_independent_feature_gate_failure(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch()
        service = self._service(triton=triton, search=search)
        gate_calls = {"precheck": 0, "feature": 0}

        def accept_precheck(*args, **kwargs):
            gate_calls["precheck"] += 1
            return True, None

        def accept_feature(*args, **kwargs):
            gate_calls["feature"] += 1
            return True, None

        observation = frame_face(
            completed_stage=FaceProcessingStage.FEATURE,
            failure_reason="Face feature quality gate failed in frame pipeline",
        )
        with (
            patch.object(
                person_face,
                "meets_pre_extraction_quality_criteria",
                side_effect=accept_precheck,
            ),
            patch.object(
                person_face,
                "is_high_quality_feature",
                side_effect=accept_feature,
            ),
        ):
            attempt = self._attempt(service, observation)

        self.assertEqual(attempt.outcome.face_id, "face-1")
        self.assertEqual(attempt.source, "partial_reuse")
        self.assertEqual(triton.total_face_calls, 0)
        self.assertEqual(gate_calls, {"precheck": 1, "feature": 1})
        self.assertEqual(len(search.calls), 1)
        np.testing.assert_array_equal(search.calls[0][0], NORMALIZED_EMBEDDING)

    def test_partial_reuse_completes_only_stages_missing_from_observation(self):
        cases = (
            (
                "analysis_detection",
                frame_face(
                    completed_stage=FaceProcessingStage.DETECTION,
                    failure_reason="frame keypoint extraction failed",
                    source_landmarks=None,
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
                (1, 1, 1),
            ),
            (
                "analysis_landmarks",
                frame_face(
                    completed_stage=FaceProcessingStage.LANDMARKS,
                    failure_reason="frame quality extraction failed",
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
                (0, 1, 1),
            ),
            (
                "feature_detection",
                frame_face(
                    completed_stage=FaceProcessingStage.DETECTION,
                    failure_reason="frame keypoint extraction failed",
                    source_landmarks=None,
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
                (1, 1, 1),
            ),
            (
                "feature_landmarks",
                frame_face(
                    completed_stage=FaceProcessingStage.LANDMARKS,
                    failure_reason="frame quality extraction failed",
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
                (0, 1, 1),
            ),
            (
                "feature_quality",
                frame_face(
                    completed_stage=FaceProcessingStage.QUALITY,
                    failure_reason="frame feature extraction failed",
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
                (0, 0, 1),
            ),
            (
                "feature_feature",
                frame_face(completed_stage=FaceProcessingStage.FEATURE),
                (0, 0, 0),
            ),
        )
        for name, observation, expected_calls in cases:
            with self.subTest(name=name):
                triton = RecordingFaceTriton()
                search = RecordingSearch("face-completed")
                service = self._service(
                    triton=triton,
                    search=search,
                )

                with (
                    patch.object(
                        person_face,
                        "meets_pre_extraction_quality_criteria",
                        return_value=(True, None),
                    ),
                    patch.object(
                        person_face,
                        "is_high_quality_feature",
                        return_value=(True, None),
                    ),
                ):
                    attempt = self._attempt(service, observation)

                self.assertEqual(attempt.outcome.face_id, "face-completed")
                self.assertEqual(attempt.source, "partial_reuse")
                self.assertEqual(
                    (
                        len(triton.keypoint_calls),
                        len(triton.quality_calls),
                        len(triton.feature_calls),
                    ),
                    expected_calls,
                )
                self.assertEqual(len(search.calls), 1)

    def test_shared_search_zero_returns_zero_outcome_and_requests_local_fallback(self):
        triton = RecordingFaceTriton()
        search = RecordingSearch(0)
        service = self._service(triton=triton, search=search)

        with (
            patch.object(
                person_face,
                "meets_pre_extraction_quality_criteria",
                return_value=(True, None),
            ),
            patch.object(person_face, "is_high_quality_feature", return_value=(True, None)),
        ):
            attempt = self._attempt(service, frame_face())

        self.assertEqual(attempt.outcome.face_id, "0")
        self.assertEqual(attempt.source, "partial_reuse")
        self.assertEqual(attempt.fallback_reason, "shared_face_id_zero")
        self.assertEqual(len(search.calls), 1)

    def test_malformed_or_incomplete_shared_facts_request_fallback_without_face_calls(self):
        malformed_source_rect = frame_face(
            completed_stage=FaceProcessingStage.DETECTION,
            source_landmarks=None,
            quality_score=None,
            normalized_embedding=None,
            raw_embedding_norm=None,
        )
        object.__setattr__(
            malformed_source_rect,
            "source_rect",
            ("not-a-coordinate", 20, 80, 80),
        )
        malformed_cases = (
            (
                "identity_not_completed",
                frame_face(
                    assigned_face_id="face-1",
                    completed_stage=FaceProcessingStage.FEATURE,
                ),
            ),
            (
                "identity_failure",
                frame_face(assigned_face_id="face-1", failure_reason="search failed"),
            ),
            (
                "identity_malformed_failure",
                frame_face(assigned_face_id="face-1", failure_reason=[]),
            ),
            (
                "analysis_without_landmarks",
                frame_face(
                    completed_stage=FaceProcessingStage.LANDMARKS,
                    source_landmarks=None,
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
            ),
            (
                "analysis_quality_stage_without_quality",
                frame_face(
                    completed_stage=FaceProcessingStage.QUALITY,
                    quality_score=None,
                    normalized_embedding=None,
                    raw_embedding_norm=None,
                ),
            ),
            (
                "analysis_detection_with_malformed_bbox",
                malformed_source_rect,
            ),
            (
                "feature_not_unit_normalized",
                frame_face(normalized_embedding=np.ones(512, dtype=np.float32)),
            ),
            (
                "identity_missing_stage",
                types.SimpleNamespace(assigned_face_id="face-1"),
            ),
            (
                "analysis_missing_fields",
                types.SimpleNamespace(assigned_face_id="0"),
            ),
        )
        for name, observation in malformed_cases:
            with self.subTest(name=name):
                triton = RecordingFaceTriton()
                search = RecordingSearch()
                service = self._service(
                    triton=triton,
                    search=search,
                )

                try:
                    attempt = self._attempt(service, observation)
                except Exception as error:
                    self.fail(f"malformed shared facts escaped for {name}: {error}")

                self.assertIsNone(attempt.outcome)
                self.assertTrue(attempt.fallback_reason)
                self.assertEqual(triton.total_face_calls, 0)
                self.assertEqual(search.calls, [])

    def test_unhashable_completed_stage_never_escapes_shared_attempt(self):
        for malformed_stage in ([], {}):
            with self.subTest(malformed_stage=malformed_stage):
                triton = RecordingFaceTriton()
                search = RecordingSearch()
                service = self._service(
                    triton=triton,
                    search=search,
                )

                try:
                    attempt = self._attempt(
                        service,
                        frame_face(completed_stage=malformed_stage),
                    )
                except Exception as error:
                    self.fail(f"malformed completed_stage escaped attempt: {error}")

                self.assertIsNone(attempt.outcome)
                self.assertEqual(
                    attempt.fallback_reason,
                    "shared_feature_incomplete",
                )
                self.assertEqual(triton.total_face_calls, 0)
                self.assertEqual(search.calls, [])

    def test_shared_resolution_exception_requests_fallback_instead_of_escaping(self):
        logger = RecordingLogger()
        service = self._service(logger=logger)

        with patch.object(
            face_reuse,
            "recognize_person_face_candidate_with_info",
            side_effect=RuntimeError("shared resolver failed"),
            create=True,
        ):
            attempt = self._attempt(service, frame_face())

        self.assertIsNone(attempt.outcome)
        self.assertEqual(attempt.fallback_reason, "shared_resolution_failed")
        self.assertEqual(len(logger.exceptions), 1)

    def test_orientation_only_recomputes_person_rule_orientation_from_shared_landmarks(self):
        service = self._service()
        cases = (
            ("front", "bound", "front", ""),
            ("side", "bound", "front", ""),
            ("unknown", "bound", "front", ""),
            ("front", "conflict", None, "conflict"),
        )

        for orientation, reason, expected_orientation, expected_fallback in cases:
            with self.subTest(orientation=orientation, reason=reason):
                attempt = self._attempt(
                    service,
                    frame_face(orientation=orientation),
                    enable_recognition=False,
                    reason=reason,
                )
                if expected_orientation is None:
                    self.assertIsNone(attempt.outcome)
                    self.assertEqual(attempt.fallback_reason, expected_fallback)
                else:
                    self.assertEqual(attempt.outcome.face_id, "0")
                    self.assertEqual(attempt.outcome.orientation, expected_orientation)
                    self.assertEqual(attempt.source, "shared_orientation")

    def test_orientation_only_requires_completed_landmarks(self):
        service = self._service()

        detection_attempt = self._attempt(
            service,
            frame_face(
                orientation="front",
                completed_stage=FaceProcessingStage.DETECTION,
            ),
            enable_recognition=False,
        )
        landmarks_attempt = self._attempt(
            service,
            frame_face(
                orientation="FRONT",
                completed_stage=FaceProcessingStage.LANDMARKS,
            ),
            enable_recognition=False,
        )

        self.assertIsNone(detection_attempt.outcome)
        self.assertEqual(
            detection_attempt.fallback_reason,
            "shared_orientation_incomplete",
        )
        self.assertEqual(landmarks_attempt.outcome.orientation, "front")
        self.assertEqual(landmarks_attempt.source, "shared_orientation")

    def test_orientation_only_requires_a_valid_landmark_structural_chain(self):
        for name, observation in invalid_landmark_chain_observations():
            with self.subTest(name=name):
                triton = RecordingFaceTriton()
                search = RecordingSearch("unexpected-search-hit")
                service = self._service(
                    triton=triton,
                    search=search,
                )

                attempt = self._attempt(
                    service,
                    observation,
                    enable_recognition=False,
                )

                self.assertIsNone(attempt.outcome)
                self.assertTrue(attempt.fallback_reason)
                self.assertEqual(triton.total_face_calls, 0)
                self.assertEqual(search.calls, [])

        triton = RecordingFaceTriton()
        search = RecordingSearch("unexpected-search-hit")
        service = self._service(
            triton=triton,
            search=search,
        )

        attempt = self._attempt(
            service,
            frame_face(
                orientation="front",
                completed_stage=FaceProcessingStage.LANDMARKS,
                quality_score=None,
                normalized_embedding=None,
                raw_embedding_norm=None,
            ),
            enable_recognition=False,
        )

        self.assertEqual(attempt.outcome.face_id, "0")
        self.assertEqual(attempt.outcome.orientation, "front")
        self.assertEqual(attempt.source, "shared_orientation")
        self.assertEqual(triton.total_face_calls, 0)
        self.assertEqual(search.calls, [])


class FrameFaceBindingTests(unittest.TestCase):
    def test_binds_face_owned_by_person_mask(self):
        rect = {"x": 10, "y": 10, "w": 60, "h": 80}
        frame_face = FrameFaceObservation(source_rect=(25, 20, 45, 40), confidence=0.9)

        bindings = bind_frame_faces_to_people(
            frame_faces=[frame_face],
            people=[person(index=0, rect=rect, mask=mask_with_rect(rect))],
            image_shape=IMAGE_SHAPE,
        )

        self.assertIs(bindings[0].frame_face, frame_face)
        self.assertEqual(bindings[0].reason, "bound")

    def test_rejects_face_with_center_outside_person_mask(self):
        rect = {"x": 10, "y": 10, "w": 20, "h": 20}
        frame_face = FrameFaceObservation(source_rect=(50, 20, 70, 40), confidence=0.9)

        bindings = bind_frame_faces_to_people(
            frame_faces=[frame_face],
            people=[person(index=0, rect=rect, mask=mask_with_rect(rect))],
            image_shape=IMAGE_SHAPE,
        )

        self.assertIsNone(bindings[0].frame_face)
        self.assertEqual(bindings[0].reason, "no_owned_face")

    def test_pose_head_selects_lower_confidence_face_that_contains_head(self):
        rect = {"x": 0, "y": 0, "w": 100, "h": 100}
        keypoints = full_body_keypoints({"x": 60, "y": 10, "w": 30, "h": 80})
        high_confidence_wrong_face = FrameFaceObservation(
            source_rect=(10, 10, 30, 30), confidence=0.99
        )
        lower_confidence_head_face = FrameFaceObservation(
            source_rect=(65, 10, 85, 30), confidence=0.50
        )

        bindings = bind_frame_faces_to_people(
            frame_faces=[high_confidence_wrong_face, lower_confidence_head_face],
            people=[
                person(
                    index=0,
                    rect=rect,
                    mask=mask_with_rect(rect),
                    keypoints=keypoints,
                )
            ],
            image_shape=IMAGE_SHAPE,
        )

        self.assertIs(bindings[0].frame_face, lower_confidence_head_face)
        self.assertEqual(bindings[0].reason, "bound")

    def test_clears_shared_face_claimed_by_non_duplicate_people(self):
        shared_mask = np.ones(IMAGE_SHAPE[:2], dtype=np.uint8)
        frame_face = FrameFaceObservation(source_rect=(40, 20, 60, 40), confidence=0.9)
        people = [
            person(index=0, rect={"x": 0, "y": 0, "w": 30, "h": 100}, mask=shared_mask),
            person(index=1, rect={"x": 70, "y": 0, "w": 30, "h": 100}, mask=shared_mask),
        ]

        bindings = bind_frame_faces_to_people(
            frame_faces=[frame_face], people=people, image_shape=IMAGE_SHAPE
        )

        self.assertEqual(
            [(binding.frame_face, binding.reason) for binding in bindings],
            [
                (None, "ambiguous_non_duplicate_people"),
                (None, "ambiguous_non_duplicate_people"),
            ],
        )

    def test_keeps_shared_face_for_duplicate_person_observations(self):
        rect = {"x": 10, "y": 10, "w": 80, "h": 80}
        shared_mask = mask_with_rect(rect)
        frame_face = FrameFaceObservation(source_rect=(40, 20, 60, 40), confidence=0.9)
        people = [
            person(index=0, rect=rect, mask=shared_mask),
            person(index=1, rect=rect, mask=shared_mask.copy()),
        ]

        bindings = bind_frame_faces_to_people(
            frame_faces=[frame_face], people=people, image_shape=IMAGE_SHAPE
        )

        self.assertEqual([binding.reason for binding in bindings], ["bound", "bound"])
        self.assertIs(bindings[0].frame_face, frame_face)
        self.assertIs(bindings[1].frame_face, frame_face)

    def test_ignores_malformed_frame_face_bbox(self):
        rect = {"x": 10, "y": 10, "w": 60, "h": 80}
        malformed_frame_face = types.SimpleNamespace(source_rect=(10, 20, 30), confidence=0.9)

        bindings = bind_frame_faces_to_people(
            frame_faces=[malformed_frame_face],
            people=[person(index=0, rect=rect, mask=mask_with_rect(rect))],
            image_shape=IMAGE_SHAPE,
        )

        self.assertEqual(len(bindings), 1)
        self.assertIsNone(bindings[0].frame_face)
        self.assertEqual(bindings[0].reason, "no_owned_face")

    def test_preserves_people_input_order_and_length(self):
        left_rect = {"x": 0, "y": 0, "w": 45, "h": 100}
        right_rect = {"x": 55, "y": 0, "w": 45, "h": 100}
        left_face = FrameFaceObservation(source_rect=(10, 20, 30, 40), confidence=0.9)
        right_face = FrameFaceObservation(source_rect=(65, 20, 85, 40), confidence=0.8)
        people = [
            person(index=7, rect=right_rect, mask=mask_with_rect(right_rect)),
            person(index=3, rect=left_rect, mask=mask_with_rect(left_rect)),
        ]

        bindings = bind_frame_faces_to_people(
            frame_faces=[left_face, right_face], people=people, image_shape=IMAGE_SHAPE
        )

        self.assertEqual(len(bindings), 2)
        self.assertIs(bindings[0].frame_face, right_face)
        self.assertIs(bindings[1].frame_face, left_face)


if __name__ == "__main__":
    unittest.main()
