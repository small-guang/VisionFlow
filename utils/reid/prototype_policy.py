"""Bounded SBS prototype update planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.algorithm_config import algorithm_config

from .reid_models import finite_float, normalize_face_id


PROTOTYPE_ACTION_INSERT = "insert"
PROTOTYPE_ACTION_REPLACE = "replace"
PROTOTYPE_ACTION_SKIP_DUPLICATE = "skip_duplicate"
PROTOTYPE_ACTION_REJECT_FACE_CONFLICT = "reject_face_conflict"
PROTOTYPE_ACTION_REJECT_FACE_REQUIRED = "reject_face_required"
PROTOTYPE_ACTION_REJECT_OUTLIER = "reject_outlier"
PROTOTYPE_ACTION_FROZEN = "frozen"


@dataclass(frozen=True)
class PrototypeUpdatePlan:
    action: str
    reason: str
    replace_id: str | int | None = None

    @property
    def writes_vector(self) -> bool:
        return self.action in {PROTOTYPE_ACTION_INSERT, PROTOTYPE_ACTION_REPLACE}


def _as_normalized_vector(value: Any) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float32)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return np.asarray([], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.asarray([], dtype=np.float32)
    return vector / norm


def _record_value(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _domain_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(_record_value(record, "deviceId", "device_id", default="") or ""),
        str(
            _record_value(
                record,
                "bodyCompleteness",
                "body_completeness",
                default="",
            )
            or ""
        ),
    )


def plan_prototype_update(
    existing_records: Sequence[Mapping[str, Any]],
    new_record: Mapping[str, Any],
    *,
    max_total: int | None = None,
    max_per_domain: int | None = None,
    duplicate_threshold: float | None = None,
    class_floor: float | None = None,
) -> PrototypeUpdatePlan:
    """Plan an insert/replace without widening a class to absorb a new mode."""

    prototype_config = algorithm_config.reid.prototype
    if max_total is None:
        max_total = prototype_config.max_per_reid
    if max_per_domain is None:
        max_per_domain = prototype_config.max_per_domain
    if duplicate_threshold is None:
        duplicate_threshold = prototype_config.duplicate_threshold
    if class_floor is None:
        class_floor = prototype_config.class_floor
    if max_total <= 0 or max_per_domain <= 0:
        raise ValueError("prototype limits must be positive")

    new_face = normalize_face_id(_record_value(new_record, "faceId", "face_id"))
    if not new_face:
        return PrototypeUpdatePlan(
            PROTOTYPE_ACTION_REJECT_FACE_REQUIRED,
            "face_required",
        )
    existing_faces = {
        face_id
        for face_id in (
            normalize_face_id(_record_value(record, "faceId", "face_id"))
            for record in existing_records
        )
        if face_id
    }
    if len(existing_faces) > 1:
        return PrototypeUpdatePlan(
            PROTOTYPE_ACTION_REJECT_FACE_CONFLICT,
            "existing_reid_has_multiple_faces",
        )
    canonical_face = next(iter(existing_faces), "")
    if new_face and canonical_face and new_face != canonical_face:
        return PrototypeUpdatePlan(
            PROTOTYPE_ACTION_REJECT_FACE_CONFLICT,
            "query_face_conflicts_with_reid_anchor",
        )
    new_vector = _as_normalized_vector(_record_value(new_record, "embedding"))
    if new_vector.size == 0:
        return PrototypeUpdatePlan(PROTOTYPE_ACTION_REJECT_OUTLIER, "invalid_vector")

    normalized_existing: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for record in existing_records:
        vector = _as_normalized_vector(_record_value(record, "embedding"))
        if vector.shape == new_vector.shape:
            normalized_existing.append((record, vector))
    if not normalized_existing:
        return PrototypeUpdatePlan(PROTOTYPE_ACTION_INSERT, "first_prototype")

    similarities = [
        float(np.dot(new_vector, vector)) for _, vector in normalized_existing
    ]
    max_similarity = max(similarities)
    if max_similarity >= float(duplicate_threshold):
        return PrototypeUpdatePlan(
            PROTOTYPE_ACTION_SKIP_DUPLICATE,
            "prototype_is_redundant",
        )
    if max_similarity < float(class_floor):
        return PrototypeUpdatePlan(
            PROTOTYPE_ACTION_REJECT_OUTLIER,
            "prototype_would_create_new_mode",
        )

    new_domain = _domain_key(new_record)
    domain_records = [
        pair for pair in normalized_existing if _domain_key(pair[0]) == new_domain
    ]
    if len(existing_records) < max_total and len(domain_records) < max_per_domain:
        return PrototypeUpdatePlan(PROTOTYPE_ACTION_INSERT, "adds_safe_coverage")

    domain_counts: dict[tuple[str, str], int] = {}
    for record, _ in normalized_existing:
        key = _domain_key(record)
        domain_counts[key] = domain_counts.get(key, 0) + 1

    if domain_records:
        replacement_pool = domain_records
    else:
        overrepresented = {key for key, count in domain_counts.items() if count > 1}
        replacement_pool = [
            pair
            for pair in normalized_existing
            if _domain_key(pair[0]) in overrepresented
        ]

    safe_pool: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for record, vector in replacement_pool:
        key = _domain_key(record)
        if domain_counts.get(key, 0) <= 1:
            continue
        safe_pool.append((record, vector))
    if not safe_pool:
        return PrototypeUpdatePlan(PROTOTYPE_ACTION_FROZEN, "no_safe_replacement")

    def redundancy(
        pair: tuple[Mapping[str, Any], np.ndarray],
    ) -> tuple[float, float, int]:
        record, vector = pair
        peers = [
            peer_vector
            for peer_record, peer_vector in normalized_existing
            if peer_record is not record
            and _domain_key(peer_record) == _domain_key(record)
        ]
        max_peer = max(
            (float(np.dot(vector, peer)) for peer in peers),
            default=-1.0,
        )
        quality = finite_float(
            _record_value(record, "qualityScore", "quality_score", default=0.0)
        )
        timestamp = int(_record_value(record, "timestamp", default=0) or 0)
        return (max_peer, -quality, -timestamp)

    replace_pair = max(safe_pool, key=redundancy)
    replace_record, _ = replace_pair
    if redundancy(replace_pair)[0] < prototype_config.replace_redundancy_floor:
        return PrototypeUpdatePlan(PROTOTYPE_ACTION_FROZEN, "no_redundant_prototype")

    replace_id = _record_value(replace_record, "id")
    if replace_id is None:
        return PrototypeUpdatePlan(
            PROTOTYPE_ACTION_FROZEN,
            "replacement_has_no_primary_key",
        )
    return PrototypeUpdatePlan(
        PROTOTYPE_ACTION_REPLACE,
        "replace_most_redundant_prototype",
        replace_id=replace_id,
    )
