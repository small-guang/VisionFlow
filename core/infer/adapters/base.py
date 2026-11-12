from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from core.infer.batching import NormalizedRequest
from core.infer.contracts import MODEL_CONTRACTS, ModelContract
from core.infer.errors import InferenceInputError, InferenceResponseError


class ModelAdapter:
    def __init__(self, key: str) -> None:
        self.contract: ModelContract = MODEL_CONTRACTS[key]

    def normalize(self, *args: object) -> NormalizedRequest:
        raise NotImplementedError

    def pack(
        self, requests: Sequence[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def validate_outputs(
        self, outputs: dict[str, np.ndarray], batch_size: int
    ) -> None:
        raise NotImplementedError

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> object:
        raise NotImplementedError


def require_uint8_image(
    image: np.ndarray, name: str = "image"
) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise InferenceInputError(f"{name} must be a UINT8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise InferenceInputError(
            f"{name} must have non-empty shape [H,W,3]"
        )
    return image


def require_float32_input(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise InferenceInputError(f"{name} must be an FP32 NumPy array")
    if value.shape != shape or value.size == 0:
        raise InferenceInputError(f"{name} must have shape {shape}")
    if not bool(np.all(np.isfinite(value))):
        raise InferenceInputError(f"{name} must contain only finite values")
    return value


def require_encoded_bytes(
    value: bytes | np.ndarray, name: str = "encoded image"
) -> bytes:
    if isinstance(value, bytes):
        return require_clip_bytes(value)
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise InferenceInputError(f"{name} must be bytes or a UINT8 NumPy array")
    if value.ndim != 1 or value.size == 0:
        raise InferenceInputError(f"{name} must have non-empty shape [L]")
    return value.tobytes()


def require_clip_bytes(value: bytes) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise InferenceInputError("image bytes must be non-empty bytes")
    return value


def compatibility_key(
    inputs: Mapping[str, np.ndarray], input_names: Iterable[str]
) -> tuple:
    return tuple(
        (name, tuple(inputs[name].shape), inputs[name].dtype.name)
        for name in input_names
    )


def normalized_request(
    inputs: dict[str, np.ndarray],
    input_names: Iterable[str],
    *,
    context: object | None = None,
) -> NormalizedRequest:
    return NormalizedRequest(
        inputs=inputs,
        compatibility_key=compatibility_key(inputs, input_names),
        context=context,
    )


def _request_tensor_matching_key(
    request: NormalizedRequest, name: str
) -> np.ndarray:
    value = request.inputs[name]
    if not isinstance(value, np.ndarray) or value.size == 0:
        raise InferenceInputError(
            f"{name} must remain a non-empty NumPy array until packing"
        )
    matches = [
        item
        for item in request.compatibility_key
        if isinstance(item, tuple) and item and item[0] == name
    ]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise InferenceInputError(
            f"{name} has an invalid compatibility key"
        )
    expected_shape = tuple(matches[0][1])
    expected_dtype = matches[0][2]
    if tuple(value.shape) != expected_shape or value.dtype.name != expected_dtype:
        raise InferenceInputError(
            f"{name} no longer matches its compatibility key"
        )
    if np.issubdtype(value.dtype, np.floating) and not bool(
        np.all(np.isfinite(value))
    ):
        raise InferenceInputError(f"{name} must contain only finite values")
    return value


def stack_inputs(
    requests: Sequence[NormalizedRequest], input_names: Iterable[str]
) -> dict[str, np.ndarray]:
    if not requests:
        raise InferenceInputError("cannot pack an empty request batch")
    names = tuple(input_names)
    try:
        packed = {}
        for name in names:
            values = [
                _request_tensor_matching_key(item, name) for item in requests
            ]
            first = values[0]
            if len(values) == 1:
                packed[name] = np.ascontiguousarray(first)[np.newaxis, ...]
                continue
            if any(
                value.shape != first.shape or value.dtype != first.dtype
                for value in values
            ):
                raise InferenceInputError(
                    "batch requests contain incompatible input tensors"
                )
            # Materialize slices directly into one C-order batch allocation.
            output = np.empty((len(values), *first.shape), dtype=first.dtype)
            np.stack(values, axis=0, out=output)
            packed[name] = output
        return packed
    except InferenceInputError:
        raise
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise InferenceInputError(
            "batch requests contain incompatible input tensors"
        ) from exc


def pack_encoded_bytes(
    requests: Sequence[NormalizedRequest], name: str = "IMAGE_BYTES"
) -> dict[str, np.ndarray]:
    if not requests:
        raise InferenceInputError("cannot pack an empty request batch")
    output = np.empty((len(requests), 1), dtype=object)
    try:
        for index, request in enumerate(requests):
            value = _request_tensor_matching_key(request, name)
            if value.dtype != object or value.shape != (1,):
                raise InferenceInputError(
                    f"{name} must have object shape [1] before packing"
                )
            output[index, 0] = require_clip_bytes(value[0])
    except InferenceInputError:
        raise
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise InferenceInputError(
            f"batch requests contain invalid {name} tensors"
        ) from exc
    return {name: output}


def require_batched_output(
    outputs: Mapping[str, np.ndarray], name: str, batch_size: int
) -> np.ndarray:
    value = outputs.get(name)
    if not isinstance(value, np.ndarray) or value.ndim == 0:
        actual = None if value is None else getattr(value, "shape", None)
        raise InferenceResponseError(
            f"{name} must be a batched NumPy array, got {actual}"
        )
    if value.shape[0] != batch_size:
        raise InferenceResponseError(
            f"{name} must have batch size {batch_size}, got {value.shape}"
        )
    return value


def require_output_tensor(
    outputs: Mapping[str, np.ndarray],
    *,
    name: str,
    batch_size: int,
    dtype: np.dtype | type,
    dims: tuple[int, ...],
) -> np.ndarray:
    value = require_batched_output(outputs, name, batch_size)
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise InferenceResponseError(
            f"{name} must have dtype {expected_dtype.name}, "
            f"got {value.dtype.name}"
        )
    if value.ndim != len(dims) + 1:
        raise InferenceResponseError(
            f"{name} must have shape [B,{','.join(map(str, dims))}], "
            f"got {value.shape}"
        )
    for axis, expected in enumerate(dims, start=1):
        actual = value.shape[axis]
        if (expected == -1 and actual < 1) or (
            expected != -1 and actual != expected
        ):
            raise InferenceResponseError(
                f"{name} has invalid dimension at axis {axis}: "
                f"expected {expected}, got {actual}"
            )
    return value


def require_finite(value: np.ndarray, name: str) -> None:
    if not bool(np.all(np.isfinite(value))):
        raise InferenceResponseError(f"{name} contains non-finite values")


def require_count(value: object, *, name: str = "NUM_DETECTIONS") -> int:
    scalar = np.asarray(value)
    if scalar.size != 1 or not np.issubdtype(scalar.dtype, np.integer):
        raise InferenceResponseError(f"{name} must contain one integer")
    count = int(scalar.reshape(-1)[0])
    if count < 0 or count > 100:
        raise InferenceResponseError(
            f"{name} must be an integer in [0, 100], got {count}"
        )
    return count
