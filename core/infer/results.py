from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from utils.person_mask import PersonMask


class Rect(TypedDict):
    x: int
    y: int
    w: int
    h: int


class FaceDetection(TypedDict):
    xyxy: list[float]
    conf: float
    cls: int


class PersonSegment(TypedDict):
    xyxy: list[float]
    rect: Rect
    conf: float
    cls: int
    maskPixelCount: int
    mask: PersonMask | np.ndarray


class PersonPose(TypedDict):
    xyxy: list[float]
    rect: Rect
    conf: float
    cls: int
    keypoints: np.ndarray | None


@dataclass(frozen=True)
class FeatureResult:
    vector: np.ndarray


@dataclass(frozen=True)
class FaceQualityResult:
    score: float
