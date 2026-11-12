import asyncio
import math
import unittest
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
from utils.buffers.sbs_feature_buffer import SbsFeatureBuffer
from utils.reid.reid_enrollment_state import (
    REID_ENROLLMENT_STATE_FORMAL_PENDING,
    REID_ENROLLMENT_STATE_REJECTED,
)
from utils.reid.person_body import BODY_FULL, person_sbs_threshold
from utils.reid.prototype_policy import (
    PROTOTYPE_ACTION_FROZEN,
    PROTOTYPE_ACTION_INSERT,
    PROTOTYPE_ACTION_REJECT_FACE_CONFLICT,
    PROTOTYPE_ACTION_REJECT_FACE_REQUIRED,
    PROTOTYPE_ACTION_REJECT_OUTLIER,
    PROTOTYPE_ACTION_REPLACE,
    PROTOTYPE_ACTION_SKIP_DUPLICATE,
    plan_prototype_update,
)
from utils.reid.reid_models import (
    REID_SOURCE_MILVUS,
    ReidCandidate,
)
from utils.reid.reid_resolution import resolve_reid_classes
from utils.reid.reid_rules import AnonymousBodyPolicy, BodySupportRule, ReidClassFacts
from utils.reid.reid_state import REID_STATE_ACTIVE


def vector_with_cosine(score: float) -> np.ndarray:
    return np.asarray([score, np.sqrt(max(0.0, 1.0 - score * score))], dtype=np.float32)


def vector_at_angle(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray([np.cos(radians), np.sin(radians)], dtype=np.float32)


async def enqueue_formal_sbs(buffer, **kwargs):
    await buffer._enqueue_formal_feature(
        **kwargs,
        crop_url="",
        quality_score=9.0,
    )


def candidate(
    reid,
    score,
    *,
    face_id="",
    device_id="CAM-A",
    prototype_id=1,
):
    return ReidCandidate(
        reid=reid,
        score=score,
        source=REID_SOURCE_MILVUS,
        threshold=0.56,
        body_completeness=BODY_FULL,
        face_id=face_id,
        device_id=device_id,
        prototype_id=prototype_id,
    )


class ReidClassPurityTests(unittest.TestCase):
    def test_resolver_reads_configured_local_top1_at_decision_time(self):
        classification = algorithm_config.reid.classification.model_copy(
            update={"body_local_top1": 0.85}
        )
        reid = algorithm_config.reid.model_copy(
            update={"classification": classification}
        )
        custom_config = algorithm_config.model_copy(update={"reid": reid})
        hits = [
            candidate("supported", 0.80, prototype_id=index)
            for index in (1, 2)
        ]

        with patch("utils.reid.reid_resolution.algorithm_config", custom_config):
            resolution = resolve_reid_classes(hits, query_device_id="CAM-A")

        self.assertIsNone(resolution.reid)
        self.assertEqual(resolution.reason, "no_compatible_class")

    def test_reid_candidate_contains_no_legacy_lifecycle_fields(self):
        hit = candidate("r", 0.9)
        self.assertFalse(hasattr(hit, "identity_state"))
        self.assertFalse(hasattr(hit, "prototype_type"))
        self.assertFalse(hasattr(hit, "evidence_count"))

    def test_different_face_cannot_reuse_similar_body_reid(self):
        resolution = resolve_reid_classes(
            [candidate("face-a-reid", 0.99, face_id="face-a")],
            query_face_id="face-b",
            query_device_id="CAM-A",
        )
        self.assertIsNone(resolution.reid)
        self.assertEqual(resolution.reason, "no_compatible_class")
        self.assertEqual(resolution.best_score, 0.0)
        self.assertEqual(resolution.mean_score, 0.0)
        self.assertEqual(resolution.score, 0.0)
        self.assertEqual(resolution.required_margin, 0.0)

    def test_same_face_may_choose_between_multiple_reids(self):
        resolution = resolve_reid_classes(
            [
                candidate("appearance-1", 0.91, face_id="face-a", prototype_id=1),
                candidate("appearance-2", 0.82, face_id="face-a", prototype_id=2),
            ],
            query_face_id="face-a",
            query_device_id="CAM-A",
        )
        self.assertEqual(resolution.reid, "appearance-1")
        self.assertEqual(resolution.state, REID_STATE_ACTIVE)
        self.assertAlmostEqual(resolution.best_score, 0.91)
        self.assertAlmostEqual(resolution.mean_score, 0.91)
        self.assertAlmostEqual(resolution.score, 0.91)
        self.assertAlmostEqual(resolution.runner_up_score, 0.82)
        self.assertAlmostEqual(resolution.margin, 0.09)
        self.assertEqual(resolution.required_margin, 0.0)

    def test_anonymous_top1_thresholds_are_current_and_strict(self):
        for label, threshold, device_id in (
            ("local", 0.70, "CAM-A"),
            ("cross-device", 0.74, "CAM-B"),
        ):
            with self.subTest(label=label):
                at_boundary = resolve_reid_classes(
                    [
                        candidate("supported", threshold, device_id=device_id, prototype_id=1),
                        candidate("supported", threshold, device_id=device_id, prototype_id=2),
                    ],
                    query_device_id="CAM-A",
                )
                immediately_above = resolve_reid_classes(
                    [
                        candidate(
                            "supported",
                            math.nextafter(threshold, math.inf),
                            device_id=device_id,
                            prototype_id=1,
                        ),
                        candidate(
                            "supported",
                            math.nextafter(threshold, math.inf),
                            device_id=device_id,
                            prototype_id=2,
                        ),
                    ],
                    query_device_id="CAM-A",
                )

                self.assertIsNone(at_boundary.reid)
                self.assertEqual(immediately_above.reid, "supported")
                self.assertEqual(immediately_above.support_count, 2)

    def test_anonymous_mean_thresholds_are_current_and_strict(self):
        policy = AnonymousBodyPolicy()

        def facts(mean_score, device_ids):
            selected = tuple(
                candidate(
                    "supported",
                    mean_score,
                    device_id=device_id,
                    prototype_id=index,
                )
                for index, device_id in enumerate(device_ids, start=1)
            )
            return ReidClassFacts(
                reid="supported",
                query_face="",
                query_device="CAM-A",
                canonical_face="",
                selected=selected,
                best_score=mean_score,
                mean_score=mean_score,
                raw_runner_score=0.0,
                raw_margin=mean_score,
            )

        # Isolate the configured mean gate; resolver Top1 filtering can preclude it.
        self.assertFalse(
            policy.evaluate(
                facts(0.66, ("CAM-A", "CAM-B"))
            ).accepted
        )
        self.assertTrue(
            policy.evaluate(
                facts(
                    math.nextafter(0.66, math.inf),
                    ("CAM-A", "CAM-B"),
                )
            ).accepted
        )
        self.assertFalse(
            policy.evaluate(
                facts(0.70, ("CAM-B", "CAM-C"))
            ).accepted
        )
        self.assertTrue(
            policy.evaluate(
                facts(
                    math.nextafter(0.70, math.inf),
                    ("CAM-B", "CAM-C"),
                )
            ).accepted
        )

    def test_unique_face_anchor_single_support_uses_raw_runner_margin(self):
        accepted = resolve_reid_classes(
            [
                candidate("anchored", 0.78, face_id="face-a"),
                candidate("raw-runner", 0.45, face_id="face-b"),
            ],
            query_device_id="CAM-A",
        )
        ambiguous = resolve_reid_classes(
            [
                candidate("anchored", 0.78, face_id="face-a"),
                candidate("raw-runner", 0.74, face_id="face-b"),
            ],
            query_device_id="CAM-A",
        )
        self.assertEqual(accepted.reid, "anchored")
        self.assertEqual(accepted.reason, "unique_face_anchor")
        self.assertEqual(accepted.support_count, 1)
        self.assertAlmostEqual(accepted.raw_margin, 0.33)
        self.assertIsNone(ambiguous.reid)

    def test_unique_anchor_raw_margin_ignores_same_face_reids(self):
        resolution = resolve_reid_classes(
            [
                candidate("appearance-1", 0.78, face_id="face-a"),
                candidate("appearance-2", 0.77, face_id="face-a"),
                candidate("other-person", 0.45, face_id="face-b"),
            ],
            query_device_id="CAM-A",
        )
        self.assertEqual(resolution.reid, "appearance-1")
        self.assertEqual(resolution.reason, "unique_face_anchor")
        self.assertAlmostEqual(resolution.raw_runner_up_score, 0.45)
        self.assertAlmostEqual(resolution.raw_margin, 0.33)

    def test_unique_anchor_raw_margin_keeps_different_face_competitor(self):
        resolution = resolve_reid_classes(
            [
                candidate("appearance-1", 0.78, face_id="face-a"),
                candidate("appearance-2", 0.77, face_id="face-a"),
                candidate("other-person", 0.74, face_id="face-b"),
            ],
            query_device_id="CAM-A",
        )
        self.assertIsNone(resolution.reid)

    def test_unique_face_anchor_raw_margin_is_inclusive_at_point_fifteen(self):
        rule = BodySupportRule()
        selected = (candidate("anchored", 0.71, face_id="face-a"),)

        def facts(raw_margin):
            return ReidClassFacts(
                reid="anchored",
                query_face="",
                query_device="CAM-A",
                canonical_face="face-a",
                selected=selected,
                best_score=0.71,
                mean_score=0.71,
                raw_runner_score=0.71 - raw_margin,
                raw_margin=raw_margin,
            )

        at_boundary = rule.evaluate(facts(0.15))
        immediately_below = rule.evaluate(
            facts(math.nextafter(0.15, -math.inf))
        )

        self.assertTrue(at_boundary.accepted)
        self.assertEqual(at_boundary.reason, "unique_face_anchor")
        self.assertFalse(immediately_below.accepted)

    def test_no_compatible_raw_diagnostics_use_identity_runner(self):
        resolution = resolve_reid_classes(
            [
                candidate("face-a-1", 0.90, face_id="face-a", prototype_id=1),
                candidate("face-a-2", 0.89, face_id="face-a", prototype_id=2),
                candidate("conflict", 0.76, face_id="face-b", prototype_id=3),
                candidate("conflict", 0.75, face_id="face-c", prototype_id=4),
            ],
            query_device_id="CAM-A",
        )
        self.assertIsNone(resolution.reid)
        self.assertEqual(resolution.reason, "no_compatible_class")
        self.assertAlmostEqual(resolution.raw_runner_up_score, 0.76)
        self.assertAlmostEqual(resolution.raw_margin, 0.14)

    def test_unique_face_anchor_cross_device_requires_above_point_seventy_six(self):
        at_boundary = resolve_reid_classes(
            [
                candidate(
                    "anchored",
                    0.76,
                    face_id="face-a",
                    device_id="CAM-B",
                ),
                candidate("runner", 0.40, face_id="face-b"),
            ],
            query_device_id="CAM-A",
        )
        above = resolve_reid_classes(
            [
                candidate(
                    "anchored",
                    0.77,
                    face_id="face-a",
                    device_id="CAM-B",
                ),
                candidate("runner", 0.40, face_id="face-b"),
            ],
            query_device_id="CAM-A",
        )
        self.assertIsNone(at_boundary.reid)
        self.assertEqual(above.reid, "anchored")
        self.assertEqual(above.reason, "unique_face_anchor")

    def test_same_face_similarity_at_body_threshold_does_not_force_reid_reuse(self):
        resolution = resolve_reid_classes(
            [candidate("white-outfit", 0.56, face_id="face-a")],
            query_face_id="face-a",
            query_device_id="CAM-A",
        )
        self.assertIsNone(resolution.reid)
        self.assertEqual(resolution.reason, "no_compatible_class")


class ViewpointBridgeRegressionTests(unittest.TestCase):
    VIEW_ANGLES = (0.0, 35.0, 17.5)
    EMBEDDINGS = tuple(vector_at_angle(angle) for angle in VIEW_ANGLES)

    def test_all_three_anonymous_views_resolve_to_face_anchored_appearance(self):
        anchor = self.EMBEDDINGS[0]
        for index, query in enumerate(self.EMBEDDINGS):
            candidates = [
                ReidCandidate(
                    reid="appearance-reid",
                    score=float(np.dot(query, anchor)),
                    source=REID_SOURCE_MILVUS,
                    threshold=person_sbs_threshold(BODY_FULL, BODY_FULL),
                    body_completeness=BODY_FULL,
                    face_id="face-a",
                    device_id="CAM-A",
                    prototype_id=f"anchor-{index}",
                ),
                ReidCandidate(
                    reid="raw-runner",
                    score=0.40,
                    source=REID_SOURCE_MILVUS,
                    threshold=0.56,
                    face_id="other-face",
                    device_id="CAM-A",
                ),
            ]
            resolution = resolve_reid_classes(
                candidates,
                query_device_id="CAM-A",
            )
            self.assertEqual(resolution.reid, "appearance-reid")
            self.assertEqual(resolution.reason, "unique_face_anchor")
            self.assertEqual(resolution.support_count, 1)

    def test_canonical_face_cannot_bypass_observed_face_requirement(self):
        async def run():
            buffer = SbsEnrollmentGateTests.buffer()
            outcome = await buffer.offer_feature(
                observed_face_id="0",
                canonical_face_id="face-a",
                re_id="appearance-reid",
                identity_state=REID_STATE_ACTIVE,
                body_completeness=BODY_FULL,
                embedding=self.EMBEDDINGS[0],
                device_id="CAM-A",
                keyframe_id="frame-0",
                picture_vector_id=0,
                keyframe_url="frame.jpg",
                person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
                timestamp=1_000,
                ai_id="ai-0",
            )
            return outcome, buffer

        outcome, buffer = asyncio.run(run())
        self.assertEqual(outcome.reason, "face_required")
        self.assertEqual(
            outcome.enrollment_state,
            REID_ENROLLMENT_STATE_REJECTED,
        )
        self.assertEqual(buffer.pending, {})
        self.assertEqual(buffer.queue.qsize(), 0)
        self.assertEqual(buffer._canonical_face_by_reid, {})

    def test_anonymous_face_requirement_precedes_canonical_face_conflict(self):
        async def run():
            buffer = SbsEnrollmentGateTests.buffer()
            buffer._canonical_face_by_reid["appearance-reid"] = "other-face"
            outcome = await buffer.offer_feature(
                observed_face_id="0",
                canonical_face_id="face-a",
                re_id="appearance-reid",
                identity_state=REID_STATE_ACTIVE,
                body_completeness=BODY_FULL,
                embedding=self.EMBEDDINGS[0],
                device_id="CAM-A",
                keyframe_id="frame-0",
                picture_vector_id=0,
                keyframe_url="frame.jpg",
                person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
                timestamp=1_000,
                ai_id="ai-0",
            )
            return outcome, buffer

        outcome, buffer = asyncio.run(run())
        self.assertEqual(outcome.reason, "face_required")
        self.assertEqual(
            outcome.enrollment_state,
            REID_ENROLLMENT_STATE_REJECTED,
        )
        self.assertEqual(buffer.pending, {})
        self.assertEqual(buffer.queue.qsize(), 0)
        self.assertEqual(
            buffer._canonical_face_by_reid,
            {"appearance-reid": "other-face"},
        )

    def test_same_face_close_appearances_reuse_best_reid_without_margin(self):
        resolution = resolve_reid_classes(
            [
                candidate("appearance-1", 0.85, face_id="face-a", prototype_id=1),
                candidate("appearance-2", 0.83, face_id="face-a", prototype_id=2),
            ],
            query_face_id="face-a",
            query_device_id="CAM-A",
        )
        self.assertEqual(resolution.reid, "appearance-1")
        self.assertEqual(resolution.reason, "accepted")
        self.assertAlmostEqual(resolution.best_score, 0.85)
        self.assertAlmostEqual(resolution.mean_score, 0.85)
        self.assertAlmostEqual(resolution.score, 0.85)
        self.assertAlmostEqual(resolution.runner_up_score, 0.83)
        self.assertAlmostEqual(resolution.margin, 0.02)
        self.assertEqual(resolution.required_margin, 0.0)

    def test_anonymous_same_face_reids_compete_as_one_identity(self):
        resolution = resolve_reid_classes(
            [
                candidate("appearance-1", 0.90, face_id="face-a", prototype_id=1),
                candidate("appearance-1", 0.86, face_id="face-a", prototype_id=2),
                candidate("appearance-2", 0.88, face_id="face-a", prototype_id=1),
                candidate("appearance-2", 0.85, face_id="face-a", prototype_id=2),
            ],
            query_device_id="CAM-A",
        )
        self.assertEqual(resolution.reid, "appearance-1")
        self.assertEqual(resolution.canonical_face_id, "face-a")
        self.assertEqual(resolution.reason, "accepted")
        self.assertEqual(resolution.runner_up_score, 0.0)
        self.assertEqual(resolution.required_margin, 0.08)

    def test_anonymous_different_faces_still_require_face_margin(self):
        resolution = resolve_reid_classes(
            [
                candidate("appearance-1", 0.90, face_id="face-a", prototype_id=1),
                candidate("appearance-1", 0.86, face_id="face-a", prototype_id=2),
                candidate("appearance-2", 0.88, face_id="face-b", prototype_id=1),
                candidate("appearance-2", 0.85, face_id="face-b", prototype_id=2),
            ],
            query_device_id="CAM-A",
        )
        self.assertIsNone(resolution.reid)
        self.assertEqual(resolution.reason, "ambiguous_margin")
        self.assertLess(resolution.margin, 0.08)

    def test_body_only_active_class_requires_multiple_supporting_prototypes(self):
        single = resolve_reid_classes(
            [candidate("body", 0.90, face_id="")],
            query_device_id="CAM-A",
        )
        supported = resolve_reid_classes(
            [
                candidate("body", 0.90, face_id="", prototype_id=1),
                candidate("body", 0.86, face_id="", prototype_id=2),
            ],
            query_device_id="CAM-A",
        )
        self.assertIsNone(single.reid)
        self.assertEqual(supported.reid, "body")
        self.assertEqual(supported.state, REID_STATE_ACTIVE)
        self.assertAlmostEqual(supported.best_score, 0.90)
        self.assertAlmostEqual(supported.mean_score, 0.88)
        self.assertAlmostEqual(supported.score, 0.892)
        self.assertEqual(supported.support_count, 2)
        self.assertEqual(supported.required_margin, 0.08)

class PrototypeUpdatePolicyTests(unittest.TestCase):
    @staticmethod
    def record(
        prototype_id,
        vector,
        *,
        face_id="face-a",
        device_id="CAM-A",
        quality=9.0,
        timestamp=1,
    ):
        return {
            "id": prototype_id,
            "faceId": face_id,
            "reId": "reid-a",
            "embedding": np.asarray(vector, dtype=np.float32),
            "deviceId": device_id,
            "bodyCompleteness": BODY_FULL,
            "qualityScore": quality,
            "timestamp": timestamp,
        }

    def test_first_prototype_inserts(self):
        plan = plan_prototype_update([], self.record(0, [1.0, 0.0]))
        self.assertEqual(plan.action, PROTOTYPE_ACTION_INSERT)

    def test_face_conflicts_and_missing_faces_are_rejected(self):
        face_class = [self.record(1, [1.0, 0.0], face_id="face-a")]
        self.assertEqual(
            plan_prototype_update(
                face_class,
                self.record(2, [0.9, 0.43589], face_id="face-b"),
            ).action,
            PROTOTYPE_ACTION_REJECT_FACE_CONFLICT,
        )
        self.assertEqual(
            plan_prototype_update(
                face_class,
                self.record(2, [0.9, 0.43589], face_id="0"),
            ).action,
            PROTOTYPE_ACTION_REJECT_FACE_REQUIRED,
        )

    def test_duplicate_skips_and_new_mode_rejects(self):
        existing = [self.record(1, [1.0, 0.0])]
        duplicate = plan_prototype_update(
            existing,
            self.record(2, vector_with_cosine(0.97)),
        )
        outlier = plan_prototype_update(
            existing,
            self.record(3, vector_with_cosine(0.40)),
        )
        self.assertEqual(duplicate.action, PROTOTYPE_ACTION_SKIP_DUPLICATE)
        self.assertEqual(outlier.action, PROTOTYPE_ACTION_REJECT_OUTLIER)

    def test_full_domain_replaces_most_redundant_prototype(self):
        existing = [
            self.record(1, [1.0, 0.0]),
            self.record(2, vector_with_cosine(0.94), quality=7.0),
            self.record(3, vector_with_cosine(0.92), quality=9.0),
        ]
        plan = plan_prototype_update(
            existing,
            self.record(4, vector_with_cosine(0.75)),
            max_total=3,
            max_per_domain=3,
        )
        self.assertEqual(plan.action, PROTOTYPE_ACTION_REPLACE)
        self.assertIn(plan.replace_id, {2, 3})

    def test_full_diverse_gallery_freezes_instead_of_forced_eviction(self):
        existing = [
            self.record(1, [1.0, 0.0]),
            self.record(2, [0.0, 1.0]),
        ]
        plan = plan_prototype_update(
            existing,
            self.record(3, [0.70710678, 0.70710678]),
            max_total=2,
            max_per_domain=2,
        )
        self.assertEqual(plan.action, PROTOTYPE_ACTION_FROZEN)


class SbsEnrollmentGateTests(unittest.TestCase):
    @staticmethod
    def buffer():
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

    def test_buffer_does_not_expose_provisional_state(self):
        buffer = self.buffer()
        self.assertFalse(hasattr(buffer, "provisional"))
        self.assertFalse(hasattr(buffer, "anchor_provisional"))
        self.assertFalse(hasattr(buffer, "_provisional_lock"))
        self.assertFalse(hasattr(buffer, "_anchor_provisional_lock"))

    def test_only_current_observed_face_enrolls_immediately(self):
        async def run():
            buffer = self.buffer()
            common = {
                "identity_state": REID_STATE_ACTIVE,
                "body_completeness": BODY_FULL,
                "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                "device_id": "CAM-A",
                "picture_vector_id": 1,
                "keyframe_url": "frame.jpg",
                "crop_url": "CAM-A/person_crop/a.jpg",
                "person_rect": {"x": 0, "y": 0, "w": 10, "h": 20},
                "ai_id": "ai",
                "quality_score": 9.0,
            }
            anonymous_outcome = await buffer.offer_feature(
                **common,
                observed_face_id="0",
                canonical_face_id="face-from-existing-reid",
                re_id="existing-reid",
                keyframe_id="anonymous-frame",
                timestamp=1_000,
            )
            face_outcome = await buffer.offer_feature(
                **common,
                observed_face_id="face-a",
                canonical_face_id="face-a",
                re_id="face-reid",
                keyframe_id="face-frame",
                timestamp=2_000,
            )
            return anonymous_outcome, face_outcome, buffer

        anonymous_outcome, face_outcome, buffer = asyncio.run(run())
        self.assertFalse(anonymous_outcome.enrolled)
        self.assertEqual(anonymous_outcome.reason, "face_required")
        self.assertEqual(
            anonymous_outcome.enrollment_state,
            REID_ENROLLMENT_STATE_REJECTED,
        )
        self.assertTrue(face_outcome.enrolled)
        self.assertEqual(face_outcome.reid_state, REID_STATE_ACTIVE)
        self.assertEqual(
            face_outcome.enrollment_state,
            REID_ENROLLMENT_STATE_FORMAL_PENDING,
        )
        self.assertEqual(buffer.queue.qsize(), 1)
        self.assertEqual(len(buffer.pending), 1)
        face_record = next(iter(buffer.pending.values()))
        self.assertEqual(face_record["reId"], "face-reid")
        self.assertEqual(face_record["faceId"], "face-a")
        self.assertEqual(face_record["cropUrl"], "CAM-A/person_crop/a.jpg")

    def test_anonymous_active_identity_is_rejected_without_buffer_search_state(self):
        async def run():
            buffer = self.buffer()
            embedding = np.asarray([1.0, 0.0], dtype=np.float32)
            outcome = await buffer.offer_feature(
                observed_face_id="0",
                re_id="active-reid",
                identity_state=REID_STATE_ACTIVE,
                body_completeness=BODY_FULL,
                embedding=embedding,
                device_id="CAM-A",
                keyframe_id="frame-1",
                picture_vector_id=1,
                keyframe_url="frame.jpg",
                person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
                timestamp=1_000,
                ai_id="ai",
            )
            candidates, *_ = await buffer.asearch_candidates_with_hit_info(
                embedding,
                BODY_FULL,
                device_ids=["CAM-A"],
                query_timestamp=1_100,
            )
            return outcome, candidates, buffer

        outcome, candidates, buffer = asyncio.run(run())
        self.assertEqual(outcome.reason, "face_required")
        self.assertEqual(
            outcome.enrollment_state,
            REID_ENROLLMENT_STATE_REJECTED,
        )
        self.assertEqual(candidates, [])
        self.assertEqual(buffer.pending, {})
        self.assertEqual(buffer.queue.qsize(), 0)

    def test_enrollment_outcome_states_distinguish_queue_and_rejection(self):
        async def run():
            buffer = self.buffer()
            common = {
                "identity_state": REID_STATE_ACTIVE,
                "body_completeness": BODY_FULL,
                "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                "device_id": "CAM-A",
                "picture_vector_id": 1,
                "keyframe_url": "frame.jpg",
                "person_rect": {"x": 0, "y": 0, "w": 10, "h": 20},
                "ai_id": "ai",
            }
            face = await buffer.offer_feature(
                **common,
                re_id="face-reid",
                observed_face_id="face-a",
                canonical_face_id="face-a",
                keyframe_id="face-frame",
                timestamp=1_000,
            )
            anonymous = await buffer.offer_feature(
                **common,
                re_id="anonymous-reid",
                observed_face_id="0",
                keyframe_id="anonymous-frame",
                timestamp=1_100,
            )
            empty_reid = await buffer.offer_feature(
                **common,
                re_id="",
                observed_face_id="0",
                keyframe_id="empty-reid-frame",
                timestamp=1_200,
            )
            return face, anonymous, empty_reid, buffer.queue.qsize()

        face, anonymous, empty_reid, queue_size = asyncio.run(run())
        self.assertEqual(
            face.enrollment_state,
            REID_ENROLLMENT_STATE_FORMAL_PENDING,
        )
        self.assertEqual(face.reason, "face_confirmed")
        self.assertEqual(
            anonymous.enrollment_state,
            REID_ENROLLMENT_STATE_REJECTED,
        )
        self.assertEqual(anonymous.reason, "face_required")
        self.assertEqual(
            empty_reid.enrollment_state,
            REID_ENROLLMENT_STATE_REJECTED,
        )
        self.assertEqual(empty_reid.reason, "empty_reid")
        self.assertEqual(queue_size, 1)
        self.assertFalse(hasattr(face, "pending"))
        self.assertFalse(hasattr(face, "promoted_count"))

    def test_repeated_anonymous_offers_remain_rejected(self):
        async def run():
            buffer = self.buffer()
            embedding = np.asarray([1.0, 0.0], dtype=np.float32)
            common = {
                "observed_face_id": "0",
                "re_id": "short-lived-reid",
                "identity_state": REID_STATE_ACTIVE,
                "body_completeness": BODY_FULL,
                "embedding": embedding,
                "device_id": "CAM-A",
                "picture_vector_id": 1,
                "keyframe_url": "frame.jpg",
                "person_rect": {"x": 0, "y": 0, "w": 10, "h": 20},
                "ai_id": "ai",
            }
            snapshots = []
            for index, timestamp in enumerate((1_000, 1_100), start=1):
                outcome = await buffer.offer_feature(
                    **common,
                    keyframe_id=f"frame-{index}",
                    timestamp=timestamp,
                )
                snapshots.append(
                    {
                        "enrolled": outcome.enrolled,
                        "enrollment_state": outcome.enrollment_state,
                        "reason": outcome.reason,
                        "pending_empty": buffer.pending == {},
                        "queue_size": buffer.queue.qsize(),
                        "canonical_faces": dict(buffer._canonical_face_by_reid),
                    }
                )
            return snapshots

        snapshots = asyncio.run(run())
        expected = {
            "enrolled": False,
            "enrollment_state": REID_ENROLLMENT_STATE_REJECTED,
            "reason": "face_required",
            "pending_empty": True,
            "queue_size": 0,
            "canonical_faces": {},
        }
        self.assertEqual(snapshots, [expected, expected])

class SbsFlushTests(unittest.TestCase):
    def test_successful_items_complete_while_storage_errors_are_requeued(self):
        async def run():
            callbacks = []

            async def on_vectors_inserted(*args):
                callbacks.append(args)

            class FakeSbsDb:
                collection_name = "alogSBS"

                async def aapply_prototype_updates(self, batch):
                    return [
                        {
                            "item": batch[0],
                            "action": "insert",
                            "persisted": True,
                            "insert_result": {"primary_keys": [1]},
                        },
                        {
                            "item": batch[1],
                            "action": "retry_error",
                            "reason": "storage_error",
                            "retry": True,
                            "persisted": False,
                            "insert_result": None,
                        },
                    ]

            buffer = SbsFeatureBuffer(
                vector_db=FakeSbsDb(),
                max_batch_size=100,
                on_vectors_inserted=on_vectors_inserted,
            )
            for index in range(2):
                await enqueue_formal_sbs(buffer,
                    face_id="face-a",
                    re_id=f"reid-{index}",
                    body_completeness=BODY_FULL,
                    embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                    device_id="CAM-A",
                    keyframe_id=f"frame-{index}",
                    picture_vector_id=index,
                    keyframe_url="frame.jpg",
                    person_rect={"x": 0, "y": 0, "w": 10, "h": 20},
                    timestamp=1_000 + index,
                    ai_id="ai",
                )
            await buffer.flush()
            return buffer, callbacks

        buffer, callbacks = asyncio.run(run())
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(buffer.queue.qsize(), 1)
        self.assertEqual(len(buffer.pending), 1)
        self.assertEqual(next(iter(buffer.pending.values()))["reId"], "reid-1")


if __name__ == "__main__":
    unittest.main()
