from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from core.infer.batching import NormalizedRequest
from core.infer.errors import InferenceResponseError
from core.infer.results import PersonPose, PersonSegment
from core.performance import runtime_performance
from utils.keyframe.detection_helpers import (
    PERSON_CLASS_ID,
    clip_xyxy_to_rect,
    parse_person_poses_result,
)
from utils.person_mask import PersonMask

from .base import (
    ModelAdapter,
    normalized_request,
    require_count,
    require_finite,
    require_output_tensor,
    require_uint8_image,
    stack_inputs,
)


def _image_shape(request: NormalizedRequest) -> tuple[int, ...]:
    context = request.context
    if not isinstance(context, Mapping):
        raise InferenceResponseError("request context is missing image_shape")
    shape = context.get("image_shape")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(type(item) is not int or item < 1 for item in shape)
    ):
        raise InferenceResponseError("request context has invalid image_shape")
    return shape


def _person_seg_response_error(
    error: InferenceResponseError,
) -> InferenceResponseError:
    message = str(error)
    if not message.startswith("person_seg: "):
        message = f"person_seg: {message}"
    return InferenceResponseError(message)


class _PersonAdapter(ModelAdapter):
    def normalize(self, image: np.ndarray) -> NormalizedRequest:
        value = require_uint8_image(image)
        return normalized_request(
            {"RAW_IMAGE": value},
            ("RAW_IMAGE",),
            context={"image_shape": tuple(value.shape)},
        )

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        return stack_inputs(requests, ("RAW_IMAGE",))

    @staticmethod
    def _validate_detection_outputs(
        outputs: dict[str, np.ndarray],
        batch_size: int,
        *,
        dims: tuple[int, ...] = (100, 6),
    ) -> tuple[np.ndarray, np.ndarray]:
        detections = require_output_tensor(
            outputs,
            name="DETECTIONS",
            batch_size=batch_size,
            dtype=np.float32,
            dims=dims,
        )
        counts = require_output_tensor(
            outputs,
            name="NUM_DETECTIONS",
            batch_size=batch_size,
            dtype=np.int32,
            dims=(1,),
        )
        return detections, counts


class PersonSegmentationAdapter(_PersonAdapter):
    def __init__(self) -> None:
        super().__init__("person_seg")

    def _validated_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        detections, counts = self._validate_detection_outputs(
            outputs, batch_size, dims=(-1, 6)
        )
        mask_crops = require_output_tensor(
            outputs,
            name="MASK_CROPS_PACKED",
            batch_size=batch_size,
            dtype=np.uint8,
            dims=(-1, 160, 20),
        )
        if mask_crops.shape[:2] != detections.shape[:2]:
            raise InferenceResponseError(
                "MASK_CROPS_PACKED and DETECTIONS must share batch "
                "and detection dimensions"
            )
        capacity = detections.shape[1]
        if capacity > 100:
            raise InferenceResponseError(
                "DETECTIONS/MASK_CROPS_PACKED capacity must be in [1, 100], "
                f"got {capacity}"
            )
        validated_counts = []
        for batch_index in range(batch_size):
            count = require_count(counts[batch_index])
            if count > capacity:
                raise InferenceResponseError(
                    f"NUM_DETECTIONS exceeds dynamic capacity {capacity} "
                    f"at batch index {batch_index}"
                )
            validated_counts.append(count)
        expected_capacity = max(1, max(validated_counts, default=0))
        if capacity != expected_capacity:
            raise InferenceResponseError(
                "NUM_DETECTIONS requires canonical DETECTIONS/MASK_CROPS_PACKED "
                f"capacity {expected_capacity}, got {capacity}"
            )
        for batch_index, count in enumerate(validated_counts):
            require_finite(
                detections[batch_index, :count],
                f"DETECTIONS batch index {batch_index}",
            )
        for batch_index, count in enumerate(validated_counts):
            if np.any(detections[batch_index, count:]) or np.any(
                mask_crops[batch_index, count:]
            ):
                raise InferenceResponseError(
                    "DETECTIONS/MASK_CROPS_PACKED padding must be zero "
                    "at batch index "
                    f"{batch_index}"
                )
        return detections, mask_crops, counts

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        try:
            self._validated_outputs(outputs, batch_size)
        except InferenceResponseError as exc:
            raise _person_seg_response_error(exc) from exc

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> list[PersonSegment]:
        # 逐图同步解包包含 mask 展开、缩放、PersonMask 构造和像素计数，不含 RPC。
        with runtime_performance.measure("person_seg_unpack"):
            try:
                return self._unpack_one(outputs, index, request)
            except InferenceResponseError as exc:
                raise _person_seg_response_error(exc) from exc

    def _unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> list[PersonSegment]:
        try:
            detections = outputs["DETECTIONS"][index]
            mask_crops = outputs["MASK_CROPS_PACKED"][index]
            count_value = outputs["NUM_DETECTIONS"][index]
        except (KeyError, IndexError) as exc:
            raise InferenceResponseError(
                "DETECTIONS/MASK_CROPS_PACKED/NUM_DETECTIONS outputs are missing "
                f"sample {index}"
            ) from exc
        count = require_count(count_value)
        if count > len(detections):
            raise InferenceResponseError(
                "NUM_DETECTIONS exceeds dynamic DETECTIONS capacity "
                f"{len(detections)}"
            )
        require_finite(detections[:count], "DETECTIONS")
        image_shape = _image_shape(request)
        image_h, image_w = image_shape[:2]
        validated_detections = []
        for detection_index, row in enumerate(detections[:count]):
            rect = clip_xyxy_to_rect(row[:4], image_shape)
            if rect is None:
                raise InferenceResponseError(
                    f"DETECTIONS[{detection_index}] has no valid clipped "
                    "bbox area"
                )
            validated_detections.append((row, rect, detection_index))

        results = []
        for row, rect, detection_index in validated_detections:
            if int(row[5]) != PERSON_CLASS_ID:
                continue
            crop = np.unpackbits(
                mask_crops[detection_index], axis=-1, count=160, bitorder="big"
            )
            resized = cv2.resize(
                crop,
                (rect["w"], rect["h"]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = PersonMask(resized, (rect["x"], rect["y"]), (image_h, image_w))
            results.append(
                {
                    "xyxy": [float(value) for value in row[:4]],
                    "rect": rect,
                    "conf": float(row[4]),
                    "cls": int(row[5]),
                    "mask": mask,
                    "maskPixelCount": mask.count(),
                }
            )
        return results


class PersonPoseAdapter(_PersonAdapter):
    def __init__(self) -> None:
        super().__init__("person_pose")

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        detections, _ = self._validate_detection_outputs(outputs, batch_size)
        keypoints = require_output_tensor(
            outputs,
            name="KEYPOINTS",
            batch_size=batch_size,
            dtype=np.float32,
            dims=(100, 17, 3),
        )
        if keypoints.shape[:2] != detections.shape[:2]:
            raise InferenceResponseError(
                "KEYPOINTS and DETECTIONS must share batch and detection dimensions"
            )

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> list[PersonPose]:
        try:
            detections = outputs["DETECTIONS"][index]
            keypoints = outputs["KEYPOINTS"][index]
            count_value = outputs["NUM_DETECTIONS"][index]
        except (KeyError, IndexError) as exc:
            raise InferenceResponseError(
                f"person pose outputs are missing sample {index}"
            ) from exc
        require_finite(detections, "DETECTIONS")
        require_finite(keypoints, "KEYPOINTS")
        require_count(count_value)
        results = parse_person_poses_result(
            detections=detections,
            keypoints=keypoints,
            num_detections=count_value,
            image_shape=_image_shape(request),
        )
        for result in results:
            if result["keypoints"] is not None:
                result["keypoints"] = np.array(
                    result["keypoints"],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
        return results
