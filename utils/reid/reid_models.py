"""Shared immutable types and constants for person ReID decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .reid_state import REID_STATE_UNASSIGNED


REID_SOURCE_MILVUS = "milvus"
REID_SOURCE_BUFFER = "buffer"

REID_SEARCH_MODE_NOT_EXECUTED = "NOT_EXECUTED"
REID_SEARCH_MODE_SAME_FACE = "SAME_FACE"
REID_SEARCH_MODE_ANONYMOUS_GLOBAL = "ANONYMOUS_GLOBAL"


@dataclass(frozen=True)
class ReidCandidate:
    reid: str
    score: float
    source: str = ""
    threshold: float = 0.0
    body_completeness: str = ""
    face_id: str = ""
    device_id: str = ""
    prototype_id: str | int | None = None


@dataclass(frozen=True)
class ReidClassResolution:
    reid: str | None
    score: float = 0.0
    best_score: float = 0.0
    mean_score: float = 0.0
    runner_up_score: float = 0.0
    margin: float = 0.0
    required_margin: float = 0.0
    raw_runner_up_score: float = 0.0
    raw_margin: float = 0.0
    source: str = ""
    support_count: int = 0
    canonical_face_id: str = ""
    state: str = REID_STATE_UNASSIGNED
    reason: str = "no_candidate"


@dataclass(frozen=True)
class PersonReidSearchResult:
    reid: str | None
    score: float = 0.0
    vector_hit_reid: str = ""
    source: str = ""
    state: str = REID_STATE_UNASSIGNED
    canonical_face_id: str = ""
    reason: str = "no_candidate"
    support_count: int = 0
    top1_score: float = 0.0
    mean_score: float = 0.0
    class_score: float = 0.0
    runner_up_score: float = 0.0
    margin: float = 0.0
    required_margin: float = 0.0
    raw_runner_up_score: float = 0.0
    raw_margin: float = 0.0
    search_mode: str = REID_SEARCH_MODE_NOT_EXECUTED


def normalize_face_id(value: Any) -> str:
    normalized = str(value or "").strip()
    return "" if normalized in {"", "0"} else normalized


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return converted if math.isfinite(converted) else default
