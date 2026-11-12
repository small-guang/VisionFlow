from core.performance import runtime_performance

from .base import ObjectStorageClient
from .image_decode import _decode_image


async def get_image_by_oss_name(img_name, oss_client: ObjectStorageClient, *, image_executor):
    """从所选对象存储获取图片并转换为OpenCV格式
    Args:
        img_name: 图片对象名称
        image_executor: 调用方持有的图片 CPU 执行器
    Returns:
        numpy.ndarray: OpenCV格式的图像数据
    Raises:
        ValueError: 当图片解码失败时抛出
    """
    if image_executor is None:
        raise ValueError("image_executor is required")
    img_bytes = await oss_client.get_object(img_name)
    if not img_bytes:
        raise ValueError("从对象存储获取图片数据为空")

    # 下载完成后才进入解码名额等待；CPU 解码不再占用事件循环线程。
    timed = getattr(image_executor, "run_timed", None)
    if callable(timed):
        img = await timed("image_decode", _decode_image, img_bytes, metrics=runtime_performance)
    else:
        img = await image_executor.run(_decode_image, img_bytes)

    return img, img_bytes
