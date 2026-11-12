"""身份协调边界：共享状态由服务拥有，帧流水线只提交观察。"""

from .face import FaceIdentityOutcome, FaceIdentityRequest, FaceIdentityService

__all__ = ["FaceIdentityOutcome", "FaceIdentityRequest", "FaceIdentityService"]
