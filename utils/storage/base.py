from typing import Protocol


class ObjectStorageClient(Protocol):
    """图片存储契约；上传成功返回对象 key，读写失败返回 None。"""

    async def upload_object(
        self, image_data: bytes, object_name: str,
        headers: dict[str, str] | None = None,
    ) -> str | None: ...

    async def get_object(
        self, object_name: str, thumbnail_params: str | None = None,
    ) -> bytes | None: ...
