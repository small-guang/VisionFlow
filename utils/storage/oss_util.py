import oss2
from utils.storage.execution import run_storage_operation
import loguru
from core.config import settings

class AsyncOSSClient:
    def __init__(self, access_key_id=None, access_key_secret=None, endpoint=None, bucket_name=None, *, execution=None):
        self.execution = execution
        self.access_key_id = access_key_id or settings.OSS_ACCESS_KEY_ID
        self.access_key_secret = access_key_secret or settings.OSS_ACCESS_KEY_SECRET
        self.endpoint = endpoint or settings.OSS_ENDPOINT
        self.bucket_name = bucket_name or settings.OSS_BUCKET_NAME

        if not all([self.access_key_id, self.access_key_secret, self.endpoint, self.bucket_name]):
            loguru.logger.error("OSS配置不完整，请设置OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET、OSS_ENDPOINT和OSS_BUCKET_NAME环境变量或初始化参数")
            raise ValueError("OSS配置不完整")

        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

    async def upload_object(self, image_data, object_name, headers=None):
        """
        异步上传对象到OSS

        参数:
            image_data: bytes, 图片的二进制数据
            object_name: str, OSS中的对象名称
            headers: dict, 可选的HTTP头
        返回:
            str: 对象名称 (成功时)
            None: (失败时)
        """
        try:
            if headers:
                result = await run_storage_operation(self.execution, "upload", self.bucket.put_object, object_name, image_data, headers)
            else:
                result = await run_storage_operation(self.execution, "upload", self.bucket.put_object, object_name, image_data)
            if result.status == 200:
                loguru.logger.info(f"OSS上传成功: {object_name}")
                return object_name
            else:
                loguru.logger.error(f"OSS上传失败: {object_name}, 状态码: {result.status}")
                return None
        except Exception as e:
            loguru.logger.error(f"OSS上传异常: {object_name}, 错误类型: {type(e).__name__}")
            return None

    async def get_object(self, object_name, thumbnail_params=None):
        """
        异步从OSS获取对象

        参数:
            object_name: str, OSS中的对象名称
            thumbnail_params: str, 缩略图参数 (可选)
        返回:
            bytes: 对象数据 (成功时)
            None: (失败时)
        """
        try:
            if thumbnail_params:
                processed_object_name = f"{object_name}?x-oss-process={thumbnail_params}"
            else:
                processed_object_name = object_name
            
            def download():
                obj = self.bucket.get_object(processed_object_name)
                return obj.read()

            data = await run_storage_operation(self.execution, "download", download)
            return data
        except oss2.exceptions.NoSuchKey:
            loguru.logger.warning(f"OSS对象不存在: {object_name}")
            return None
        except Exception as e:
            loguru.logger.error(f"OSS获取对象失败: {object_name}, 错误: {str(e)}")
            return None
