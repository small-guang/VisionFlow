from core.infer.contracts import MODEL_CONTRACTS

from .base import ModelAdapter
from .embeddings import (
    ClipImageAdapter,
    FaceRecognitionAdapter,
    SbsImageAdapter,
    TbpsImageAdapter,
)
from .face import (
    FaceDetectionAdapter,
    FaceKeypointAdapter,
    FaceQualityAdapter,
)
from .person import PersonPoseAdapter, PersonSegmentationAdapter


def build_adapters() -> dict[str, ModelAdapter]:
    adapters: dict[str, ModelAdapter] = {
        "clip_image": ClipImageAdapter(),
        "face_rec": FaceRecognitionAdapter(),
        "tbps_image": TbpsImageAdapter(),
        "sbs_image": SbsImageAdapter(),
        "face_qa": FaceQualityAdapter(),
        "face_kps": FaceKeypointAdapter(),
        "face_det": FaceDetectionAdapter(),
        "person_seg": PersonSegmentationAdapter(),
        "person_pose": PersonPoseAdapter(),
    }
    if set(adapters) != set(MODEL_CONTRACTS):
        raise RuntimeError("adapter registry must match MODEL_CONTRACTS")
    return adapters


__all__ = [
    "ClipImageAdapter",
    "FaceDetectionAdapter",
    "FaceKeypointAdapter",
    "FaceQualityAdapter",
    "FaceRecognitionAdapter",
    "ModelAdapter",
    "PersonPoseAdapter",
    "PersonSegmentationAdapter",
    "SbsImageAdapter",
    "TbpsImageAdapter",
    "build_adapters",
]
