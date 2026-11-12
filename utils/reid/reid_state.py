"""Shared ReID lifecycle states exposed to recognition records."""

from __future__ import annotations

from typing import Any


REID_STATE_UNASSIGNED = ""  # 未分配 ReID 时识别记录使用空状态
REID_STATE_ACTIVE = "ACTIVE"  # 已分配的正式 ReID，可参与召回
REID_STATES = frozenset(
    {
        REID_STATE_UNASSIGNED,
        REID_STATE_ACTIVE,
    }
)


def normalize_reid_state(value: Any) -> str:
    """Normalize and validate a recognition-layer ReID lifecycle state."""

    normalized = str(value or "").strip().upper()
    if normalized not in REID_STATES:
        raise ValueError(f"Unsupported reidState: {value!r}")
    return normalized


def require_assigned_reid_state(value: Any) -> str:
    """Require a lifecycle state for a ReID that has already been assigned."""

    normalized = normalize_reid_state(value)
    if normalized != REID_STATE_ACTIVE:
        raise ValueError("assigned ReID state must be ACTIVE")
    return normalized
