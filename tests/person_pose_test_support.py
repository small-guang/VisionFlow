"""Shared Pose fixtures used by unit and integration tests."""

import numpy as np


IMAGE_SHAPE = (160, 120, 3)

PERSON_RECT = {"x": 40, "y": 20, "w": 40, "h": 120}

def full_body_keypoints(rect=PERSON_RECT, score=0.9):
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
    return np.array(
        [
            [
                rect["x"] + rect["w"] * x_ratio,
                rect["y"] + rect["h"] * y_ratio,
                score,
            ]
            for x_ratio, y_ratio in normalized
        ],
        dtype=np.float32,
    )

def pose_result(rect=PERSON_RECT, keypoints=None):
    return {
        "rect": dict(rect),
        "xyxy": [
            rect["x"],
            rect["y"],
            rect["x"] + rect["w"],
            rect["y"] + rect["h"],
        ],
        "conf": 0.9,
        "cls": 0,
        "keypoints": full_body_keypoints(rect) if keypoints is None else keypoints,
    }

def segment_result(rect=PERSON_RECT, mask_pixel_count=12000):
    mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
    x1 = max(0, rect["x"])
    y1 = max(0, rect["y"])
    x2 = min(IMAGE_SHAPE[1], rect["x"] + rect["w"])
    y2 = min(IMAGE_SHAPE[0], rect["y"] + rect["h"])
    mask[y1:y2, x1:x2] = 1
    return {
        "rect": dict(rect),
        "xyxy": [
            rect["x"],
            rect["y"],
            rect["x"] + rect["w"],
            rect["y"] + rect["h"],
        ],
        "conf": 0.9,
        "cls": 0,
        "maskPixelCount": mask_pixel_count,
        "mask": mask,
    }

def textured_image():
    y, x = np.indices(IMAGE_SHAPE[:2])
    gray = np.where((x // 2 + y // 2) % 2 == 0, 60, 180).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)

