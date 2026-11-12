"""Resolve face identities from shared gallery and pending sample evidence."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

from core.algorithm_config import algorithm_config


@dataclass(frozen=True)
class FaceSampleMatch:
    face_id: str
    score: float
    sample_id: str = ""
    crop_path: str = ""


def resolve_face_matches(
    samples: Iterable[FaceSampleMatch], threshold: float
) -> tuple[str | None, float, str]:
    """Confirm the highest-scoring identity, requiring two samples in the gray band.

    Keep the best raw hit on rejection so callers can distinguish insufficient
    identity evidence from a genuine miss. Equal scores retain source order.
    """
    valid = [
        sample for sample in samples
        if sample.face_id and sample.face_id != "0" and math.isfinite(sample.score)
    ]
    if not valid:
        return None, 0.0, ""

    best = max(valid, key=lambda sample: sample.score)
    rejected = (None, best.score, best.face_id)
    if best.score < threshold:
        return rejected
    if best.score >= algorithm_config.face.matching.single_sample_similarity_threshold:
        return best.face_id, best.score, best.face_id

    seen_samples: set[str] = set()
    seen_crops: set[str] = set()
    support_count = 0
    for sample in sorted(valid, key=lambda sample: sample.score, reverse=True):
        if sample.face_id != best.face_id or sample.score < threshold:
            continue
        # A flush can expose the same observation in both sources. Repeated
        # records or the same crop under another sampleId are only one vote.
        duplicate = (
            bool(sample.sample_id) and sample.sample_id in seen_samples
        ) or (bool(sample.crop_path) and sample.crop_path in seen_crops)
        if sample.sample_id:
            seen_samples.add(sample.sample_id)
        if sample.crop_path:
            seen_crops.add(sample.crop_path)
        if duplicate or not (sample.sample_id or sample.crop_path):
            continue
        support_count += 1
        if support_count >= 2:
            return best.face_id, best.score, best.face_id
    return rejected
