"""Bind whole-frame face observations to their owning person observations."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.face_observation import (
    FaceProcessingStage,
    FrameFaceObservation,
    landmarks_in_expanded_crop,
    normalize_face_id_scalar,
)
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.detection_helpers import person_segments_can_share_reid
from utils.quality.face_orientation import (
    estimate_face_orientation,
    normalize_face_orientation,
)
from utils.reid.person_face_recognition import select_owned_face_detection

from .models import FaceMatchOutcome, PersonObservation


_NORMALIZED_EMBEDDING_NORM_TOLERANCE = 1e-4
_STAGE_LEVEL = {
    FaceProcessingStage.DETECTION: 0,
    FaceProcessingStage.LANDMARKS: 1,
    FaceProcessingStage.QUALITY: 2,
    FaceProcessingStage.FEATURE: 3,
    FaceProcessingStage.IDENTITY: 4,
}


async def recognize_person_face_candidate_with_info(**kwargs: Any) -> dict[str, Any]:
    module = importlib.import_module("utils.reid.person_face_recognition")
    callback = getattr(module, "recognize_person_face_candidate_with_info")
    return await callback(**kwargs)


@dataclass(frozen=True)
class FrameFaceBinding:
    frame_face: FrameFaceObservation | None = None
    reason: str = "no_owned_face"


@dataclass(frozen=True)
class SharedFaceAttempt:
    outcome: FaceMatchOutcome | None = None
    source: str = ""
    fallback_reason: str = ""


@dataclass
class FaceReuseFrameStats:
    person_count: int
    frame_face_count: int
    bound_count: int = 0
    direct_id_reuse_count: int = 0
    partial_reuse_count: int = 0
    shared_orientation_count: int = 0
    local_fallback_count: int = 0
    fallback_recovered_count: int = 0
    id_disagreement_count: int = 0
    shared_nonzero_local_zero_count: int = 0
    local_nonzero_shared_zero_count: int = 0
    different_nonzero_id_count: int = 0
    orientation_disagreement_count: int = 0
    rotated_person_count: int = 0
    shared_success_rotated_count: int = 0
    shared_success_unrotated_count: int = 0
    local_fallback_rotated_count: int = 0
    local_fallback_unrotated_count: int = 0
    local_nonzero_recovery_rotated_count: int = 0
    local_nonzero_recovery_unrotated_count: int = 0
    fallback_reason_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        outcome: FaceMatchOutcome,
        rotation_applied: bool,
    ) -> None:
        shared_source = outcome.shared_source or outcome.source
        shared_face_id = outcome.shared_face_id
        shared_orientation = outcome.shared_orientation
        if not outcome.shared_source and outcome.source in {
            "direct_id_reuse",
            "partial_reuse",
            "shared_orientation",
        }:
            shared_face_id = outcome.face_id
            shared_orientation = outcome.orientation
        shared_orientation = normalize_face_orientation(shared_orientation)
        local_orientation = normalize_face_orientation(outcome.orientation)
        shared_id_nonzero = shared_face_id not in {"", "0"}
        local_id_nonzero = outcome.face_id not in {"", "0"}
        shared_success = bool(
            (
                shared_source in {"direct_id_reuse", "partial_reuse"}
                and shared_id_nonzero
            )
            or (
                shared_source == "shared_orientation"
                and shared_orientation in {"front", "side"}
            )
        )

        if shared_source == "direct_id_reuse" and shared_success:
            self.direct_id_reuse_count += 1
        elif shared_source == "partial_reuse" and shared_success:
            self.partial_reuse_count += 1
        elif shared_source == "shared_orientation" and shared_success:
            self.shared_orientation_count += 1
        if outcome.source == "local_fallback":
            self.local_fallback_count += 1
            if rotation_applied:
                self.local_fallback_rotated_count += 1
            else:
                self.local_fallback_unrotated_count += 1

        if rotation_applied:
            self.rotated_person_count += 1
        if shared_success:
            if rotation_applied:
                self.shared_success_rotated_count += 1
            else:
                self.shared_success_unrotated_count += 1

        shared_reason = outcome.shared_fallback_reason or outcome.fallback_reason
        if shared_reason:
            self.fallback_reason_counts[shared_reason] = (
                self.fallback_reason_counts.get(shared_reason, 0) + 1
            )
        shared_attempted = bool(
            outcome.shared_source or shared_reason or shared_id_nonzero
        )
        if outcome.source == "local_fallback" and shared_attempted:
            if shared_id_nonzero and not local_id_nonzero:
                self.shared_nonzero_local_zero_count += 1
            elif local_id_nonzero and not shared_id_nonzero:
                self.local_nonzero_shared_zero_count += 1
            elif (
                shared_id_nonzero
                and local_id_nonzero
                and shared_face_id != outcome.face_id
            ):
                self.different_nonzero_id_count += 1
            if (
                shared_orientation in {"front", "side"}
                and local_orientation in {"front", "side"}
                and shared_orientation != local_orientation
            ):
                self.orientation_disagreement_count += 1

        self.id_disagreement_count = (
            self.shared_nonzero_local_zero_count
            + self.local_nonzero_shared_zero_count
            + self.different_nonzero_id_count
        )
        recovered = bool(
            outcome.source == "local_fallback"
            and shared_attempted
            and not shared_success
            and local_id_nonzero
        )
        if recovered:
            self.fallback_recovered_count += 1
            if rotation_applied:
                self.local_nonzero_recovery_rotated_count += 1
            else:
                self.local_nonzero_recovery_unrotated_count += 1

    def format_fallback_reasons(self) -> str:
        if not self.fallback_reason_counts:
            return "none"
        return "|".join(
            f"{reason}:{count}"
            for reason, count in sorted(self.fallback_reason_counts.items())
        )


def _parse_completed_stage(value: Any) -> FaceProcessingStage | None:
    if isinstance(value, FaceProcessingStage):
        return value
    if not isinstance(value, str):
        return None
    try:
        return FaceProcessingStage(value)
    except ValueError:
        return None


def _has_no_failure_reason(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def _has_valid_source_rect(frame_face: FrameFaceObservation) -> bool:
    try:
        x1, y1, x2, y2 = frame_face.source_rect
        return bool(x2 > x1 and y2 > y1)
    except Exception:
        return False


def _has_valid_landmarks(frame_face: FrameFaceObservation) -> bool:
    try:
        landmarks = np.asarray(frame_face.source_landmarks, dtype=np.float32)
    except Exception:
        return False
    return bool(
        landmarks.shape == (5, 2) and np.all(np.isfinite(landmarks))
    )


def _person_rule_orientation(frame_face: FrameFaceObservation) -> str:
    landmarks = landmarks_in_expanded_crop(frame_face)
    if landmarks is None:
        return "unknown"
    try:
        x1, y1, x2, y2 = frame_face.expanded_source_rect
        image_shape = (y2 - y1, x2 - x1, 3)
        orientation = estimate_face_orientation(landmarks, image_shape).orientation
    except Exception:
        return "unknown"
    return normalize_face_orientation(orientation)


def _has_valid_quality(frame_face: FrameFaceObservation) -> bool:
    try:
        quality_score = float(frame_face.quality_score)
    except Exception:
        return False
    return math.isfinite(quality_score)


def _has_valid_feature(frame_face: FrameFaceObservation) -> bool:
    try:
        embedding = np.asarray(frame_face.normalized_embedding, dtype=np.float32)
        raw_norm = float(frame_face.raw_embedding_norm)
        normalized_norm = float(np.linalg.norm(embedding))
    except Exception:
        return False
    return bool(
        embedding.ndim == 1
        and embedding.size > 0
        and np.all(np.isfinite(embedding))
        and abs(normalized_norm - 1.0)
        <= _NORMALIZED_EMBEDDING_NORM_TOLERANCE
        and math.isfinite(raw_norm)
        and raw_norm > 0.0
    )


def _validated_observation_stage(
    frame_face: FrameFaceObservation,
) -> FaceProcessingStage | None:
    completed_stage = _parse_completed_stage(frame_face.completed_stage)
    if completed_stage is None or not _has_valid_source_rect(frame_face):
        return None

    stage_level = _STAGE_LEVEL[completed_stage]
    if (
        stage_level >= _STAGE_LEVEL[FaceProcessingStage.LANDMARKS]
        and not _has_valid_landmarks(frame_face)
    ):
        return None
    if (
        stage_level >= _STAGE_LEVEL[FaceProcessingStage.QUALITY]
        and not _has_valid_quality(frame_face)
    ):
        return None
    if (
        stage_level >= _STAGE_LEVEL[FaceProcessingStage.FEATURE]
        and not _has_valid_feature(frame_face)
    ):
        return None
    return completed_stage


class PersonFaceReuseService:
    def __init__(
        self,
        *,
        triton_client: Any,
        search_faces_with_pending_locked: Callable[..., Any],
        logger: Any,
        image_executor: Any = None,
    ) -> None:
        self.triton_client = triton_client
        self.search_faces_with_pending_locked = search_faces_with_pending_locked
        self.logger = logger
        self.image_executor = image_executor

    async def attempt(
        self,
        *,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        binding: FrameFaceBinding,
        enable_recognition: bool,
    ) -> SharedFaceAttempt:
        frame_face = binding.frame_face
        if frame_face is None or binding.reason != "bound":
            return SharedFaceAttempt(fallback_reason=binding.reason)
        if not isinstance(frame_face, FrameFaceObservation):
            return SharedFaceAttempt(fallback_reason="shared_face_malformed")
        if not frame_face.assigned_face_id_is_valid:
            return SharedFaceAttempt(fallback_reason="shared_face_id_malformed")

        completed_stage = _validated_observation_stage(frame_face)

        if not enable_recognition:
            if completed_stage not in {
                FaceProcessingStage.LANDMARKS,
                FaceProcessingStage.QUALITY,
                FaceProcessingStage.FEATURE,
                FaceProcessingStage.IDENTITY,
            }:
                return SharedFaceAttempt(
                    fallback_reason="shared_orientation_incomplete"
                )
            orientation = _person_rule_orientation(frame_face)
            if orientation in {"front", "side"}:
                outcome = FaceMatchOutcome(
                    face_id="0",
                    info="复用整图人脸朝向",
                    orientation=orientation,
                    source="shared_orientation",
                )
                return SharedFaceAttempt(
                    outcome=outcome,
                    source="shared_orientation",
                )
            return SharedFaceAttempt(fallback_reason="shared_orientation_unknown")

        assigned_face_id = normalize_face_id_scalar(frame_face.assigned_face_id)
        if assigned_face_id != "0":
            identity_is_reliable = (
                completed_stage is FaceProcessingStage.IDENTITY
                and _has_no_failure_reason(frame_face.failure_reason)
            )
            if identity_is_reliable:
                outcome = FaceMatchOutcome(
                    face_id=assigned_face_id,
                    info="复用整图人脸身份",
                    orientation=_person_rule_orientation(frame_face),
                    source="direct_id_reuse",
                )
                return SharedFaceAttempt(outcome=outcome, source="direct_id_reuse")
            return SharedFaceAttempt(fallback_reason="shared_face_id_unreliable")

        if completed_stage is None:
            return SharedFaceAttempt(fallback_reason="shared_feature_incomplete")

        stage_level = _STAGE_LEVEL[completed_stage]
        shared_landmarks = (
            frame_face.source_landmarks
            if stage_level >= _STAGE_LEVEL[FaceProcessingStage.LANDMARKS]
            else None
        )
        shared_quality = (
            frame_face.quality_score
            if stage_level >= _STAGE_LEVEL[FaceProcessingStage.QUALITY]
            else None
        )

        shared_embedding = (
            frame_face.normalized_embedding
            if stage_level >= _STAGE_LEVEL[FaceProcessingStage.FEATURE]
            else None
        )
        shared_norm = (
            frame_face.raw_embedding_norm
            if stage_level >= _STAGE_LEVEL[FaceProcessingStage.FEATURE]
            else None
        )
        try:
            result = await recognize_person_face_candidate_with_info(
                image=key_frame,
                bbox=frame_face.source_rect,
                landmarks_in_image=shared_landmarks,
                quality_score=shared_quality,
                normalized_embedding=shared_embedding,
                raw_embedding_norm=shared_norm,
                ownership_info=binding.reason,
                device_id=ctx.sn,
                timestamp=ctx.timestamp,
                triton_client=self.triton_client,
                search_callback=self.search_faces_with_pending_locked,
                logger=self.logger,
                enable_recognition=True,
                image_executor=self.image_executor,
            )
            face_id = normalize_face_id_scalar(result.get("face_id"))
            outcome = FaceMatchOutcome(
                face_id=face_id,
                info=str(result.get("info") or ""),
                orientation=normalize_face_orientation(
                    result.get("face_orientation")
                ),
                source="partial_reuse",
            )
        except Exception as error:
            self.logger.exception(
                f"Shared person face resolution failed; falling back: {error}"
            )
            return SharedFaceAttempt(fallback_reason="shared_resolution_failed")

        if outcome.face_id == "0":
            return SharedFaceAttempt(
                outcome=outcome,
                source="partial_reuse",
                fallback_reason="shared_face_id_zero",
            )
        return SharedFaceAttempt(outcome=outcome, source="partial_reuse")


def bind_frame_faces_to_people(
    frame_faces: Sequence[FrameFaceObservation],
    people: Sequence[PersonObservation],
    image_shape: tuple[int, ...],
) -> list[FrameFaceBinding]:
    """Return one deterministic face binding for each input person observation."""
    detections = []
    for frame_face_index, frame_face in enumerate(frame_faces):
        try:
            x1, y1, x2, y2 = frame_face.source_rect
            if x2 <= x1 or y2 <= y1:
                continue
        except (AttributeError, TypeError, ValueError):
            continue
        detections.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "conf": frame_face.confidence,
                "frame_face_index": frame_face_index,
            }
        )

    selected_by_person: list[int | None] = [None] * len(people)
    for person_index, person in enumerate(people):
        selection = select_owned_face_detection(
            detections,
            image_shape,
            person.segment.get("mask"),
            person.person_keypoints,
        )
        if selection is not None:
            selected_by_person[person_index] = int(
                detections[selection.detection_index]["frame_face_index"]
            )

    claimants: dict[int, list[int]] = {}
    for person_index, frame_face_index in enumerate(selected_by_person):
        if frame_face_index is not None:
            claimants.setdefault(frame_face_index, []).append(person_index)

    bindings = [FrameFaceBinding() for _ in people]
    for frame_face_index, person_indices in claimants.items():
        duplicates = all(
            person_segments_can_share_reid(
                people[left].segment,
                people[right].segment,
            )
            for offset, left in enumerate(person_indices)
            for right in person_indices[offset + 1 :]
        )
        if len(person_indices) > 1 and not duplicates:
            for person_index in person_indices:
                bindings[person_index] = FrameFaceBinding(
                    reason="ambiguous_non_duplicate_people"
                )
            continue
        for person_index in person_indices:
            bindings[person_index] = FrameFaceBinding(
                frame_face=frame_faces[frame_face_index],
                reason="bound",
            )
    return bindings
