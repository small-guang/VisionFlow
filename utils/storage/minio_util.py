"""Async adapter for MinIO with the same object-key contract as OSS."""

from io import BytesIO

import loguru
from minio import Minio
from minio.error import S3Error

from core.config import settings
from utils.storage.execution import run_storage_operation


class AsyncMinIOClient:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket_name: str | None = None,
        secure: bool | None = None,
        *,
        execution=None,
    ):
        self.execution = execution
        self.endpoint = settings.MINIO_ENDPOINT if endpoint is None else endpoint
        self.access_key = settings.MINIO_ACCESS_KEY if access_key is None else access_key
        self.secret_key = settings.MINIO_SECRET_KEY if secret_key is None else secret_key
        self.bucket_name = settings.MINIO_BUCKET_NAME if bucket_name is None else bucket_name
        self.secure = settings.MINIO_SECURE if secure is None else secure

        if not all((self.endpoint, self.access_key, self.secret_key, self.bucket_name)):
            raise ValueError(
                "MinIO配置不完整，请设置MINIO_ENDPOINT、MINIO_ACCESS_KEY、"
                "MINIO_SECRET_KEY和MINIO_BUCKET_NAME"
            )

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    async def upload_object(
        self, image_data: bytes, object_name: str, headers: dict | None = None
    ) -> str | None:
        """Upload image bytes and return their object key, or None on failure."""
        content_type = "image/jpeg"
        if headers:
            for name, value in headers.items():
                if name.lower() == "content-type":
                    content_type = value
                    break

        def upload() -> None:
            with BytesIO(image_data) as data:
                self.client.put_object(
                    self.bucket_name,
                    object_name,
                    data,
                    len(image_data),
                    content_type=content_type,
                )

        try:
            await run_storage_operation(self.execution, "upload", upload)
            return object_name
        except Exception as exc:
            loguru.logger.error("MinIO上传失败，错误类型: {}", type(exc).__name__)
            return None

    async def get_object(
        self, object_name: str, thumbnail_params: str | None = None
    ) -> bytes | None:
        """Read original object bytes; OSS thumbnail operations are unsupported."""
        if thumbnail_params:
            loguru.logger.warning("MinIO不支持OSS缩略图参数")
            return None

        def download() -> bytes:
            response = self.client.get_object(self.bucket_name, object_name)
            try:
                return response.read()
            finally:
                try:
                    response.close()
                finally:
                    response.release_conn()

        try:
            return await run_storage_operation(self.execution, "download", download)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                loguru.logger.warning("MinIO对象不存在，错误类型: {}", type(exc).__name__)
            else:
                loguru.logger.error("MinIO获取对象失败，错误类型: {}", type(exc).__name__)
            return None
        except Exception as exc:
            loguru.logger.error("MinIO获取对象失败，错误类型: {}", type(exc).__name__)
            return None
