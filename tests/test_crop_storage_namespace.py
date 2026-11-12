"""裁图元数据必须采用存储实际返回的 object key，不丢失输出命名空间。"""

import unittest

from core.application import OutputNamespacedStorage
from utils.keyframe.keyframe_flow_helpers import upload_crop_bytes


class MemoryStorage:
    def __init__(self, *, fail=False):
        self.objects = {}
        self.fail = fail

    async def upload_object(self, image_data, object_name, headers=None):
        if self.fail:
            return None
        self.objects[object_name] = image_data
        return object_name

    async def get_object(self, object_name, thumbnail_params=None):
        return self.objects.get(object_name)


class CropStorageNamespaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_face_and_person_metadata_keys_can_read_namespaced_crops(self):
        storage = MemoryStorage()
        storage.objects["camera/picture/source.jpg"] = b"source"
        isolated = OutputNamespacedStorage(storage, "/replay/run-1/")
        for crop_type in ("face", "person"):
            with self.subTest(crop_type=crop_type):
                key = await upload_crop_bytes(isolated, b"crop", "camera", crop_type)
                self.assertTrue(key.startswith(f"replay/run-1/camera/{crop_type}/"))
                self.assertEqual(await isolated.get_object(key), b"crop")
        self.assertEqual(await isolated.get_object("camera/picture/source.jpg"), b"source")

    async def test_empty_namespace_retains_existing_crop_key_layout(self):
        storage = MemoryStorage()
        for device_sn, expected in (("camera", "camera/face/"), ("", "face/")):
            with self.subTest(device_sn=device_sn):
                key = await upload_crop_bytes(
                    OutputNamespacedStorage(storage, ""), b"crop", device_sn, "face",
                )
                self.assertTrue(key.startswith(expected))
                self.assertEqual(storage.objects[key], b"crop")

    async def test_upload_failure_does_not_emit_nonexistent_crop_key(self):
        storage = MemoryStorage(fail=True)
        key = await upload_crop_bytes(
            OutputNamespacedStorage(storage, "replay"), b"crop", "camera", "person",
        )
        self.assertEqual(key, "")
        self.assertEqual(storage.objects, {})


if __name__ == "__main__":
    unittest.main()
