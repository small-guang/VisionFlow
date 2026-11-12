from collections.abc import Sequence

import numpy as np

from core.infer.batching import NormalizedRequest
from core.infer.errors import InferenceResponseError
from core.infer.results import FaceDetection, FaceQualityResult

from .base import (
    ModelAdapter,
    normalized_request,
    require_count,
    require_finite,
    require_float32_input,
    require_output_tensor,
    require_uint8_image,
    stack_inputs,
)


class _RawImageAdapter(ModelAdapter):
    input_name = "RAW_IMAGE"

    def normalize(self, image: np.ndarray) -> NormalizedRequest:
        value = require_uint8_image(image)
        return normalized_request(
            {self.input_name: value},
            (self.input_name,),
            context={"image_shape": tuple(value.shape)},
        )

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return stack_inputs(requests, (self.input_name,))


class FaceQualityAdapter(_RawImageAdapter):
    input_name = "RAW_FACE_IMAGE"

    def __init__(self) -> None:
        super().__init__("face_qa")

    def normalize(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> NormalizedRequest:
        inputs = {
            self.input_name: require_uint8_image(image),
            "LANDMARK": require_float32_input(
                landmarks, name="landmarks", shape=(5, 2)
            ),
        }
        return normalized_request(
            inputs, (self.input_name, "LANDMARK")
        )

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return stack_inputs(requests, (self.input_name, "LANDMARK"))

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        require_output_tensor(
            outputs,
            name="quality_score",
            batch_size=batch_size,
            dtype=np.float32,
            dims=(1,),
        )

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> FaceQualityResult:
        del request
        try:
            score = outputs["quality_score"][index]
        except (KeyError, IndexError) as exc:
            raise InferenceResponseError(
                f"quality_score is missing sample {index}"
            ) from exc
        require_finite(score, "quality_score")
        return FaceQualityResult(score=float(score[0]))


class FaceKeypointAdapter(ModelAdapter):
    LMK106_RIGHT_EYE_IDX = tuple(range(33, 43))
    LMK106_LEFT_EYE_IDX = tuple(range(87, 97))
    LMK106_NOSE_IDX = 86
    LMK106_MOUTH_RIGHT_IDX = 52
    LMK106_MOUTH_LEFT_IDX = 61

    def __init__(self) -> None:
        super().__init__("face_kps")

    def normalize(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> NormalizedRequest:
        inputs = {
            "RAW_IMAGE": require_uint8_image(image),
            "BBOX": require_float32_input(
                bbox, name="bbox", shape=(4,)
            ),
        }
        return normalized_request(inputs, ("RAW_IMAGE", "BBOX"))

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return stack_inputs(requests, ("RAW_IMAGE", "BBOX"))

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        require_output_tensor(
            outputs,
            name="LANDMARKS_106",
            batch_size=batch_size,
            dtype=np.float32,
            dims=(106, 2),
        )

    @classmethod
    def _lmk106_to_lmk5(cls, landmarks: np.ndarray) -> np.ndarray:
        output = np.empty((5, 2), dtype=np.float32)
        output[0] = np.mean(
            landmarks[list(cls.LMK106_RIGHT_EYE_IDX)],
            axis=0,
            dtype=np.float64,
        )
        output[1] = np.mean(
            landmarks[list(cls.LMK106_LEFT_EYE_IDX)],
            axis=0,
            dtype=np.float64,
        )
        output[2] = landmarks[cls.LMK106_NOSE_IDX]
        output[3] = landmarks[cls.LMK106_MOUTH_RIGHT_IDX]
        output[4] = landmarks[cls.LMK106_MOUTH_LEFT_IDX]
        return output

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> np.ndarray:
        del request
        try:
            landmarks = outputs["LANDMARKS_106"][index]
        except (KeyError, IndexError) as exc:
            raise InferenceResponseError(
                f"LANDMARKS_106 is missing sample {index}"
            ) from exc
        require_finite(landmarks, "LANDMARKS_106")
        result = np.array(
            self._lmk106_to_lmk5(landmarks),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        require_finite(result, "LANDMARKS_5")
        return result


class FaceDetectionAdapter(_RawImageAdapter):
    def __init__(self) -> None:
        super().__init__("face_det")

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        require_output_tensor(
            outputs,
            name="DETECTIONS",
            batch_size=batch_size,
            dtype=np.float32,
            dims=(100, 6),
        )
        require_output_tensor(
            outputs,
            name="NUM_DETECTIONS",
            batch_size=batch_size,
            dtype=np.int32,
            dims=(1,),
        )

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> list[FaceDetection]:
        del request
        try:
            detections = outputs["DETECTIONS"][index]
            count_value = outputs["NUM_DETECTIONS"][index]
        except (KeyError, IndexError) as exc:
            raise InferenceResponseError(
                f"face detection outputs are missing sample {index}"
            ) from exc
        require_finite(detections, "DETECTIONS")
        count = require_count(count_value)
        return [
            FaceDetection(
                xyxy=[float(value) for value in row[:4]],
                conf=float(row[4]),
                cls=int(row[5]),
            )
            for row in detections[:count]
        ]
