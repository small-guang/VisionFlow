from collections.abc import Sequence

import numpy as np

from core.infer.batching import NormalizedRequest
from core.infer.errors import InferenceResponseError
from core.infer.results import FeatureResult

from .base import (
    ModelAdapter,
    normalized_request,
    pack_encoded_bytes,
    require_clip_bytes,
    require_encoded_bytes,
    require_finite,
    require_float32_input,
    require_output_tensor,
    require_uint8_image,
    stack_inputs,
)


class _FeatureAdapter(ModelAdapter):
    output_name: str
    width: int

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        require_output_tensor(
            outputs,
            name=self.output_name,
            batch_size=batch_size,
            dtype=np.float32,
            dims=(self.width,),
        )

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        require_finite(vector, self.output_name)
        return np.array(vector, dtype=np.float32, order="C", copy=True)

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> FeatureResult:
        del request
        try:
            vector = outputs[self.output_name][index]
        except (KeyError, IndexError) as exc:
            raise InferenceResponseError(
                f"{self.output_name} is missing sample {index}"
            ) from exc
        return FeatureResult(vector=self._prepare_vector(vector))


class ClipImageAdapter(_FeatureAdapter):
    output_name = "unnorm_image_features"
    width = 512

    def __init__(self) -> None:
        super().__init__("clip_image")

    def normalize(self, image_bytes: bytes) -> NormalizedRequest:
        value = require_clip_bytes(image_bytes)
        inputs = {"IMAGE_BYTES": np.asarray([value], dtype=object)}
        return normalized_request(inputs, ("IMAGE_BYTES",))

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return pack_encoded_bytes(requests)


class FaceRecognitionAdapter(_FeatureAdapter):
    output_name = "EMBEDDING"
    width = 512

    def __init__(self) -> None:
        super().__init__("face_rec")

    def normalize(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> NormalizedRequest:
        inputs = {
            "RAW_IMAGE": require_uint8_image(image),
            "LANDMARK": require_float32_input(
                landmarks, name="landmarks", shape=(5, 2)
            ),
        }
        return normalized_request(inputs, ("RAW_IMAGE", "LANDMARK"))

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return stack_inputs(requests, ("RAW_IMAGE", "LANDMARK"))


class _EncodedFeatureAdapter(_FeatureAdapter):
    def normalize(self, image_bytes: bytes | np.ndarray) -> NormalizedRequest:
        value = require_encoded_bytes(image_bytes)
        inputs = {"IMAGE_BYTES": np.asarray([value], dtype=object)}
        return normalized_request(inputs, ("IMAGE_BYTES",))

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return pack_encoded_bytes(requests)


class TbpsImageAdapter(_EncodedFeatureAdapter):
    output_name = "image_feat"
    width = 512

    def __init__(self) -> None:
        super().__init__("tbps_image")


class SbsImageAdapter(_EncodedFeatureAdapter):
    output_name = "embedding"
    width = 2048

    def __init__(self) -> None:
        super().__init__("sbs_image")

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        require_finite(vector, self.output_name)
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(norm) or norm <= 0.0:
            raise InferenceResponseError(
                "embedding must be finite and non-zero"
            )
        normalized = vector.astype(np.float64) / norm
        return np.array(
            normalized, dtype=np.float32, order="C", copy=True
        )
