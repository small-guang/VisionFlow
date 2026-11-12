"""Request-local measurements; payload sizes exclude network/protobuf framing."""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np


def tensor_payload_bytes(tensors: Mapping[str, np.ndarray]) -> int | None:
    """Count tensor bytes without materializing numeric arrays or BYTES blobs.

    Triton BYTES elements have a four-byte length prefix. Object-array nbytes
    measures pointers, so it must never be used as their payload size. Return
    None for malformed tensors rather than letting telemetry affect inference.
    """
    try:
        total = 0
        for tensor in tensors.values():
            if not isinstance(tensor, np.ndarray):
                return None
            if tensor.dtype.kind in {"O", "S", "U"}:
                total += sum(
                    4 + len(value if isinstance(value, bytes) else str(value).encode("utf-8"))
                    for value in tensor.flat
                )
            else:
                total += tensor.nbytes
        return total
    except Exception:
        return None


@dataclass
class InferenceMetrics:
    request_bytes: int | None = None
    response_bytes: int | None = None
    pack_ms: float | None = None
    pack_queue_wait_ms: float | None = 0.0
    executor_queue_wait_ms: float | None = None
    client_wait_ms: float = 0.0
    serialize_ms: float | None = None
    rpc_ms: float | None = None
    rpc_attempts: int = 0
    caller_cancelled: bool = False

    def log_fields(self, batch_wait_ms: float) -> dict[str, int | float | None]:
        fields = {
            name: round(value, 3) if isinstance(value, float) else value
            for name, value in vars(self).items()
        }
        fields["batch_wait_ms"] = round(batch_wait_ms, 3)
        fields["queue_total_ms"] = (
            round(
                batch_wait_ms + self.executor_queue_wait_ms
                + self.pack_queue_wait_ms + self.client_wait_ms, 3,
            )
            if self.executor_queue_wait_ms is not None and self.pack_queue_wait_ms is not None
            else None
        )
        return fields


# Set inside the executor operation, not on the submitter task: executor lanes
# are long-lived tasks. Reset in finally so retries/cancellation cannot leak a
# previous batch's measurements into the next operation on the lane.
active_inference_metrics: ContextVar[InferenceMetrics | None] = ContextVar(
    "active_inference_metrics", default=None
)
