"""Current SBS enrollment stage exposed on person recognition records."""

from __future__ import annotations

from typing import Any


REID_ENROLLMENT_STATE_NONE = ""  # 本帧未尝试提交 SBS 扩充样本
REID_ENROLLMENT_STATE_FORMAL_PENDING = "FORMAL_PENDING"  # 样本已进入正式异步写队列
REID_ENROLLMENT_STATE_REJECTED = "REJECTED"  # 样本在进入正式写队列前被拒绝
REID_ENROLLMENT_STATES = frozenset(
    {
        REID_ENROLLMENT_STATE_NONE,
        REID_ENROLLMENT_STATE_FORMAL_PENDING,
        REID_ENROLLMENT_STATE_REJECTED,
    }
)


def normalize_reid_enrollment_state(value: Any) -> str:
    """Normalize and validate the current observation's SBS enrollment stage."""

    normalized = str(value or "").strip().upper()
    if normalized not in REID_ENROLLMENT_STATES:
        raise ValueError(f"Unsupported reidEnrollmentState: {value!r}")
    return normalized
