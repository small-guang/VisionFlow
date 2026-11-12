"""Shared geometry helpers for reliable COCO person torso poses."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.algorithm_config import algorithm_config


PERSON_TORSO_CHAIN_DISAGREEMENT_TOLERANCE_DEGREES = 1e-9  # 躯干链方向夹角比较的浮点容差


@dataclass(frozen=True)
class PersonTorsoAxisResult:
    down_axis: tuple[float, float] | None = None
    torso_scale: float | None = None
    chain_count: int = 0
    reason: str = "pose_axis_missing"

    @property
    def valid(self) -> bool:
        if self.down_axis is None or self.torso_scale is None:
            return False
        try:
            axis_x, axis_y = (float(value) for value in self.down_axis)
            torso_scale = float(self.torso_scale)
            chain_count = float(self.chain_count)
        except (TypeError, ValueError, OverflowError):
            return False
        axis_norm = math.hypot(axis_x, axis_y)
        return (
            all(math.isfinite(value) for value in (axis_x, axis_y))
            and math.isfinite(axis_norm)
            and axis_norm > 0.0
            and abs(axis_norm - 1.0) <= 1e-6
            and math.isfinite(torso_scale)
            and torso_scale > 0.0
            and math.isfinite(chain_count)
            and chain_count > 0.0
            and self.reason == "ok"
        )


def estimate_person_torso_axis(
    keypoints: Any,
    rect: Any,
) -> PersonTorsoAxisResult:
    """Estimate a confidence-weighted, downward torso axis from COCO-17 points."""
    geometry = algorithm_config.person.pose_geometry
    try:
        points = np.asarray(keypoints, dtype=np.float64)
    except Exception:
        return PersonTorsoAxisResult(reason="pose_axis_missing")
    if points.shape != (17, 3):
        return PersonTorsoAxisResult(reason="pose_axis_missing")

    try:
        width = float(rect.get("w"))
        height = float(rect.get("h"))
    except Exception:
        return PersonTorsoAxisResult(reason="pose_axis_missing")
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0.0
        or height <= 0.0
    ):
        return PersonTorsoAxisResult(reason="pose_axis_missing")
    diagonal = math.hypot(width, height)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        return PersonTorsoAxisResult(reason="pose_axis_missing")

    minimum_axis_length = diagonal * geometry.torso_min_axis_diagonal_ratio
    candidates: list[tuple[np.ndarray, float, float]] = []
    has_short_visible_axis = False
    for shoulder_index, hip_index in ((5, 11), (6, 12)):
        shoulder = points[shoulder_index]
        hip = points[hip_index]
        if not np.isfinite(shoulder).all() or not np.isfinite(hip).all():
            continue
        if (
            shoulder[2] < geometry.torso_visible_threshold
            or hip[2] < geometry.torso_visible_threshold
        ):
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            vector = hip[:2] - shoulder[:2]
        if not np.isfinite(vector).all():
            continue
        length = math.hypot(float(vector[0]), float(vector[1]))
        if not math.isfinite(length):
            continue
        if length < minimum_axis_length:
            has_short_visible_axis = True
            continue
        weight = float(min(shoulder[2], hip[2]))
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            unit_axis = vector / length
        if not np.isfinite(unit_axis).all():
            continue
        candidates.append((unit_axis, length, weight))

    if not candidates:
        reason = "pose_axis_short" if has_short_visible_axis else "pose_axis_missing"
        return PersonTorsoAxisResult(reason=reason)

    if len(candidates) == 2:
        dot_product = float(np.dot(candidates[0][0], candidates[1][0]))
        if not math.isfinite(dot_product):
            return PersonTorsoAxisResult(reason="pose_axis_conflict")
        dot_product = max(-1.0, min(1.0, dot_product))
        disagreement = math.degrees(math.acos(dot_product))
        if disagreement > (
            geometry.torso_max_chain_disagreement_degrees
            + PERSON_TORSO_CHAIN_DISAGREEMENT_TOLERANCE_DEGREES
        ):
            return PersonTorsoAxisResult(reason="pose_axis_conflict")

    with np.errstate(over="ignore", invalid="ignore"):
        combined_axis = sum(
            (unit_axis * weight for unit_axis, _, weight in candidates),
            start=np.zeros(2, dtype=np.float64),
        )
    if not np.isfinite(combined_axis).all():
        return PersonTorsoAxisResult(reason="pose_axis_conflict")
    combined_length = math.hypot(
        float(combined_axis[0]), float(combined_axis[1])
    )
    if not math.isfinite(combined_length) or combined_length <= 0.0:
        return PersonTorsoAxisResult(reason="pose_axis_conflict")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        combined_axis /= combined_length
    if not np.isfinite(combined_axis).all():
        return PersonTorsoAxisResult(reason="pose_axis_conflict")

    maximum_length = max(length for _, length, _ in candidates)
    maximum_weight = max(weight for _, _, weight in candidates)
    if (
        not math.isfinite(maximum_length)
        or maximum_length <= 0.0
        or not math.isfinite(maximum_weight)
        or maximum_weight <= 0.0
    ):
        return PersonTorsoAxisResult(reason="pose_axis_conflict")
    scaled_weights = [
        weight / maximum_weight for _, _, weight in candidates
    ]
    scaled_lengths = [
        length / maximum_length for _, length, _ in candidates
    ]
    total_scaled_weight = sum(scaled_weights)
    if (
        not all(math.isfinite(value) and value > 0.0 for value in scaled_weights)
        or not all(math.isfinite(value) and value > 0.0 for value in scaled_lengths)
        or not math.isfinite(total_scaled_weight)
        or total_scaled_weight <= 0.0
    ):
        return PersonTorsoAxisResult(reason="pose_axis_conflict")
    scaled_torso_scale = (
        sum(length * weight for length, weight in zip(scaled_lengths, scaled_weights))
        / total_scaled_weight
    )
    if not math.isfinite(scaled_torso_scale) or scaled_torso_scale <= 0.0:
        return PersonTorsoAxisResult(reason="pose_axis_conflict")
    torso_scale = maximum_length * scaled_torso_scale
    if not math.isfinite(torso_scale) or torso_scale <= 0.0:
        return PersonTorsoAxisResult(reason="pose_axis_conflict")
    return PersonTorsoAxisResult(
        down_axis=(float(combined_axis[0]), float(combined_axis[1])),
        torso_scale=float(torso_scale),
        chain_count=len(candidates),
        reason="ok",
    )
