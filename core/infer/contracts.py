from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class TensorContract:
    name: str
    datatype: str
    dims: tuple[int, ...]


@dataclass(frozen=True)
class ModelContract:
    key: str
    model_name: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    api_contract_version: int = 1

    def input(self, name: str) -> TensorContract:
        return next(tensor for tensor in self.inputs if tensor.name == name)

    def output(self, name: str) -> TensorContract:
        return next(tensor for tensor in self.outputs if tensor.name == name)


def tensor(name: str, datatype: str, *dims: int) -> TensorContract:
    return TensorContract(name=name, datatype=datatype, dims=dims)


MODEL_CONTRACTS: Mapping[str, ModelContract] = MappingProxyType(
    {
        "clip_image": ModelContract(
            key="clip_image",
            model_name="clip_ensemble_trt",
            inputs=(tensor("IMAGE_BYTES", "BYTES", 1),),
            outputs=(tensor("unnorm_image_features", "FP32", 512),),
        ),
        "face_rec": ModelContract(
            key="face_rec",
            model_name="face_recognition_pipeline",
            inputs=(
                tensor("RAW_IMAGE", "UINT8", -1, -1, 3),
                tensor("LANDMARK", "FP32", 5, 2),
            ),
            outputs=(tensor("EMBEDDING", "FP32", 512),),
        ),
        "tbps_image": ModelContract(
            key="tbps_image",
            model_name="TBPS_clip_image_ensemble",
            api_contract_version=2,
            inputs=(tensor("IMAGE_BYTES", "BYTES", 1),),
            outputs=(tensor("image_feat", "FP32", 512),),
        ),
        "sbs_image": ModelContract(
            key="sbs_image",
            model_name="sbs_r50_ibn_image_ensemble",
            api_contract_version=2,
            inputs=(tensor("IMAGE_BYTES", "BYTES", 1),),
            outputs=(tensor("embedding", "FP32", 2048),),
        ),
        "face_qa": ModelContract(
            key="face_qa",
            model_name="face_qa_ensemble",
            inputs=(
                tensor("RAW_FACE_IMAGE", "UINT8", -1, -1, 3),
                tensor("LANDMARK", "FP32", 5, 2),
            ),
            outputs=(tensor("quality_score", "FP32", 1),),
            api_contract_version=2,
        ),
        "face_kps": ModelContract(
            key="face_kps",
            model_name="face_keypoint_2d106_ensemble",
            inputs=(
                tensor("RAW_IMAGE", "UINT8", -1, -1, 3),
                tensor("BBOX", "FP32", 4),
            ),
            outputs=(tensor("LANDMARKS_106", "FP32", 106, 2),),
        ),
        "face_det": ModelContract(
            key="face_det",
            model_name="yolov8n_face_ensemble",
            api_contract_version=2,
            inputs=(tensor("RAW_IMAGE", "UINT8", -1, -1, 3),),
            outputs=(
                tensor("DETECTIONS", "FP32", 100, 6),
                tensor("NUM_DETECTIONS", "INT32", 1),
            ),
        ),
        "person_seg": ModelContract(
            key="person_seg",
            model_name="yolo26s_seg_ensemble",
            api_contract_version=3,
            inputs=(tensor("RAW_IMAGE", "UINT8", -1, -1, 3),),
            outputs=(
                tensor("DETECTIONS", "FP32", -1, 6),
                tensor("MASK_CROPS_PACKED", "UINT8", -1, 160, 20),
                tensor("NUM_DETECTIONS", "INT32", 1),
            ),
        ),
        "person_pose": ModelContract(
            key="person_pose",
            model_name="yolo26m_pose_ensemble",
            api_contract_version=2,
            inputs=(tensor("RAW_IMAGE", "UINT8", -1, -1, 3),),
            outputs=(
                tensor("DETECTIONS", "FP32", 100, 6),
                tensor("KEYPOINTS", "FP32", 100, 17, 3),
                tensor("NUM_DETECTIONS", "INT32", 1),
            ),
        ),
    }
)
