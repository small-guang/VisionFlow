"""Shared fixtures for person-quality unit tests."""

import numpy as np

from utils.quality.person.evaluator import evaluate_person_quality


PERSON_TEST_IMAGE_SHAPE = (200, 120, 3)

PERSON_TEST_GOOD_RECT = {"x": 40, "y": 40, "w": 40, "h": 100}

def _person_test_pose(rect):
    normalized = [
        (0.50, 0.10),
        (0.45, 0.08),
        (0.55, 0.08),
        (0.38, 0.12),
        (0.62, 0.12),
        (0.30, 0.25),
        (0.70, 0.25),
        (0.22, 0.40),
        (0.78, 0.40),
        (0.18, 0.55),
        (0.82, 0.55),
        (0.38, 0.55),
        (0.62, 0.55),
        (0.38, 0.75),
        (0.62, 0.75),
        (0.38, 0.92),
        (0.62, 0.92),
    ]
    return {
        "keypoints": np.asarray(
            [
                [
                    rect["x"] + rect["w"] * x_ratio,
                    rect["y"] + rect["h"] * y_ratio,
                    0.9,
                ]
                for x_ratio, y_ratio in normalized
            ],
            dtype=np.float32,
        )
    }

def _person_test_mask(rect=PERSON_TEST_GOOD_RECT, boxes=None, shape=PERSON_TEST_IMAGE_SHAPE):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if boxes is None:
        boxes = [(rect["x"] + 5, rect["y"] + 10, rect["x"] + 35, rect["y"] + 90)]
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    return mask

def _person_test_segment(
    mask_pixel_count=12000,
    rect=None,
    conf=0.8,
    mask=None,
    pose=None,
):
    if rect is None:
        rect = dict(PERSON_TEST_GOOD_RECT)
    return {
        "rect": rect,
        "conf": conf,
        "maskPixelCount": mask_pixel_count,
        "mask": mask,
        "pose": _person_test_pose(rect) if pose is None else pose,
    }

def _evaluate_person_test_quality(
    segment, all_segments=None, shape=PERSON_TEST_IMAGE_SHAPE, index=0
):
    return evaluate_person_quality(
        segment,
        all_segments if all_segments is not None else [segment],
        shape,
        segment_index=index,
    )

