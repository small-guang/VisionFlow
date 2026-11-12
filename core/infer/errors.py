class TritonInferenceError(RuntimeError):
    """Base error for ALog's inference boundary."""


class TritonContractError(TritonInferenceError):
    pass


class InferenceInputError(TritonInferenceError, ValueError):
    pass


class InferenceQueueTimeout(TritonInferenceError, TimeoutError):
    pass


class InferenceServerTimeout(TritonInferenceError, TimeoutError):
    pass


class InferenceUnavailable(TritonInferenceError):
    pass


class InferenceResponseError(TritonInferenceError):
    pass
