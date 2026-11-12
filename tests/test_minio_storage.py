import asyncio
import importlib.util
import io
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from minio.error import S3Error


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils/storage/minio_util.py"


class FakeResponse:
    def __init__(self, main_thread, payload=b"downloaded", read_error=None, close_error=None):
        self.main_thread = main_thread
        self.payload = payload
        self.read_error = read_error
        self.close_error = close_error
        self.events = []

    def _record(self, event):
        if threading.get_ident() == self.main_thread:
            raise AssertionError(f"{event} blocked the event loop thread")
        self.events.append(event)

    def read(self):
        self._record("read")
        if self.read_error:
            raise self.read_error
        return self.payload

    def close(self):
        self._record("close")
        if self.close_error:
            raise self.close_error

    def release_conn(self):
        self._record("release_conn")


class FakeMinio:
    def __init__(self, main_thread, response):
        self.main_thread = main_thread
        self.response = response
        self.uploads = []
        self.downloads = []
        self.upload_error = None
        self.download_error = None

    def put_object(self, bucket_name, object_name, data, length, content_type="application/octet-stream"):
        if threading.get_ident() == self.main_thread:
            raise AssertionError("put_object blocked the event loop thread")
        if self.upload_error:
            raise self.upload_error
        if not isinstance(data, io.BytesIO):
            raise AssertionError("upload data must be a BytesIO stream")
        self.uploads.append((bucket_name, object_name, data.read(), length, content_type))
        return SimpleNamespace(object_name=object_name, etag="test-etag", version_id=None)

    def get_object(self, bucket_name, object_name):
        if threading.get_ident() == self.main_thread:
            raise AssertionError("get_object blocked the event loop thread")
        self.downloads.append((bucket_name, object_name))
        if self.download_error:
            raise self.download_error
        return self.response


class MinioStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertTrue(MODULE_PATH.exists(), "AsyncMinIOClient implementation is missing")
        config = ModuleType("core.config")
        config.settings = SimpleNamespace(
            MINIO_ENDPOINT="minio.test:9000",
            MINIO_ACCESS_KEY="test-access",
            MINIO_SECRET_KEY="test-secret",
            MINIO_BUCKET_NAME="pictures",
            MINIO_SECURE=True,
        )
        spec = importlib.util.spec_from_file_location("_minio_storage_under_test", MODULE_PATH)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"core.config": config}):
            spec.loader.exec_module(self.module)
        self.main_thread = threading.get_ident()
        self.response = FakeResponse(self.main_thread)
        self.sdk = FakeMinio(self.main_thread, self.response)
        self.constructor_patch = patch.object(self.module, "Minio", return_value=self.sdk)
        self.constructor = self.constructor_patch.start()
        self.addCleanup(self.constructor_patch.stop)
        self.client = self.module.AsyncMinIOClient()

    async def test_upload_returns_input_key_and_streams_jpeg_off_event_loop(self):
        key = "CAM001/face_crop/image.jpg"
        self.assertEqual(await self.client.upload_object(b"\xff\xd8jpeg", key), key)
        self.assertEqual(self.sdk.uploads, [("pictures", key, b"\xff\xd8jpeg", 6, "image/jpeg")])

    async def test_upload_uses_case_insensitive_content_type_override(self):
        self.assertEqual(
            await self.client.upload_object(b"png", "image.jpg", {"content-type": "image/png"}),
            "image.jpg",
        )
        self.assertEqual(self.sdk.uploads[0][-1], "image/png")

    async def test_constructor_uses_settings_and_preserves_explicit_false(self):
        self.constructor.assert_called_once_with(
            "minio.test:9000", access_key="test-access", secret_key="test-secret", secure=True,
        )
        client = self.module.AsyncMinIOClient(
            endpoint="other.test:9000", access_key="other-access", secret_key="other-secret",
            bucket_name="other-pictures", secure=False,
        )
        self.assertEqual(await client.upload_object(b"jpg", "override.jpg"), "override.jpg")
        self.constructor.assert_called_with(
            "other.test:9000", access_key="other-access", secret_key="other-secret", secure=False,
        )
        self.assertEqual(self.sdk.uploads[0][0], "other-pictures")

    async def test_constructor_rejects_incomplete_configuration(self):
        for field in ("endpoint", "access_key", "secret_key", "bucket_name"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module.AsyncMinIOClient(**{field: ""})

    async def test_download_returns_bytes_and_releases_response_off_event_loop(self):
        self.assertEqual(await self.client.get_object("image.jpg"), b"downloaded")
        self.assertEqual(self.sdk.downloads, [("pictures", "image.jpg")])
        self.assertEqual(self.response.events, ["read", "close", "release_conn"])

    async def test_failed_read_always_closes_and_releases_response(self):
        self.response.read_error = OSError("sensitive-url?credential=test-secret")
        self.assertIsNone(await self.client.get_object("image.jpg"))
        self.assertEqual(self.response.events, ["read", "close", "release_conn"])

    async def test_failed_close_still_releases_response(self):
        self.response.close_error = OSError("close failure")
        self.assertIsNone(await self.client.get_object("image.jpg"))
        self.assertEqual(self.response.events, ["read", "close", "release_conn"])

    async def test_missing_object_returns_none(self):
        self.sdk.download_error = S3Error(None, "NoSuchKey", "not found", None, None, None)
        self.assertIsNone(await self.client.get_object("missing.jpg"))
        self.assertEqual(self.response.events, [])

    async def test_thumbnail_parameters_are_rejected_without_fetching_original(self):
        self.assertIsNone(await self.client.get_object("image.jpg", "image/resize,w_100"))
        self.assertEqual(self.sdk.downloads, [])

    async def test_empty_thumbnail_parameters_fetch_original(self):
        self.assertEqual(await self.client.get_object("image.jpg", ""), b"downloaded")

    async def test_upload_errors_return_none_without_logging_secrets_or_urls(self):
        self.sdk.upload_error = RuntimeError("https://test-access:test-secret@minio.test:9000/image")
        messages = []
        sink = self.module.loguru.logger.add(lambda message: messages.append(str(message)))
        try:
            self.assertIsNone(await self.client.upload_object(b"jpg", "image.jpg"))
        finally:
            self.module.loguru.logger.remove(sink)
        logged = "".join(messages)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("test-secret", logged)
        self.assertNotIn("test-access", logged)
        self.assertNotIn("https://", logged)

    async def test_download_errors_return_none_without_logging_secrets_or_urls(self):
        self.sdk.download_error = RuntimeError("https://test-access:test-secret@minio.test:9000/image")
        messages = []
        sink = self.module.loguru.logger.add(lambda message: messages.append(str(message)))
        try:
            self.assertIsNone(await self.client.get_object("image.jpg"))
        finally:
            self.module.loguru.logger.remove(sink)
        logged = "".join(messages)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("test-secret", logged)
        self.assertNotIn("test-access", logged)
        self.assertNotIn("https://", logged)


if __name__ == "__main__":
    unittest.main()
