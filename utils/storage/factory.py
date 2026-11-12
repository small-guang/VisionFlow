from typing import TYPE_CHECKING

from .base import ObjectStorageClient

if TYPE_CHECKING:
    from core.config import Settings


def create_storage_client(
    config: "Settings | None" = None, *, execution=None,
) -> ObjectStorageClient:
    """按配置仅初始化所选后端，避免加载未启用后端的 SDK。"""
    if config is None:
        from core.config import settings
        config = settings

    resource_args = {"execution": execution} if execution is not None else {}

    if config.STORAGE_BACKEND == "oss":
        from .oss_util import AsyncOSSClient
        return AsyncOSSClient(
            access_key_id=config.OSS_ACCESS_KEY_ID,
            access_key_secret=config.OSS_ACCESS_KEY_SECRET,
            endpoint=config.OSS_ENDPOINT,
            bucket_name=config.OSS_BUCKET_NAME,
            **resource_args,
        )
    if config.STORAGE_BACKEND == "minio":
        from .minio_util import AsyncMinIOClient
        return AsyncMinIOClient(
            endpoint=config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            bucket_name=config.MINIO_BUCKET_NAME,
            secure=config.MINIO_SECURE,
            **resource_args,
        )
    raise ValueError("Unsupported STORAGE_BACKEND; expected oss or minio")
