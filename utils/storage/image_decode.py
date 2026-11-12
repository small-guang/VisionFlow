"""原图解码操作；执行资源由调用方注入并持有。"""

from __future__ import annotations

import cv2
import numpy as np


def _decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败")
    return image
