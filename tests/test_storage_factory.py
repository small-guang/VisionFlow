import importlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

from tests.test_triton_contract_cli import _MainImport


class StorageFactoryTests(unittest.IsolatedAsyncioTestCase):
    def factory(self):
        self.assertIsNotNone(importlib.util.find_spec("utils.storage.factory"),
                             "storage backend selection is not implemented")
        return importlib.import_module("utils.storage.factory").create_storage_client

    def config(self, backend):
        return types.SimpleNamespace(
            STORAGE_BACKEND=backend,
            OSS_ACCESS_KEY_ID="oss-test-key", OSS_ACCESS_KEY_SECRET="oss-test-secret",
            OSS_ENDPOINT="https://oss.example.com", OSS_BUCKET_NAME="oss-images",
            MINIO_ENDPOINT="minio.example.com:9000", MINIO_ACCESS_KEY="minio-test-key",
            MINIO_SECRET_KEY="minio-test-secret", MINIO_BUCKET_NAME="minio-images",
            MINIO_SECURE=False,
        )

    async def test_oss_selects_existing_client_with_oss_connection_settings(self):
        create = self.factory()
        created = []
        oss = types.ModuleType("utils.storage.oss_util")
        class Client:
            def __init__(self, **connection):
                created.append(connection)
            async def get_object(self, key):
                return b"oss:" + key.encode()
        oss.AsyncOSSClient = Client
        with patch.dict(sys.modules, {"utils.storage.oss_util": oss,
                                       "utils.storage.minio_util": None}):
            client = create(self.config("oss"))
            self.assertEqual(await client.get_object("cam/picture/a.jpg"), b"oss:cam/picture/a.jpg")
        self.assertEqual(created, [{"access_key_id": "oss-test-key",
            "access_key_secret": "oss-test-secret", "endpoint": "https://oss.example.com",
            "bucket_name": "oss-images"}])

    async def test_minio_selects_minio_settings_without_importing_oss(self):
        create = self.factory()
        created = []
        minio = types.ModuleType("utils.storage.minio_util")
        class Client:
            def __init__(self, **connection):
                created.append(connection)
            async def get_object(self, key):
                return b"minio:" + key.encode()
        minio.AsyncMinIOClient = Client
        with patch.dict(sys.modules, {"utils.storage.minio_util": minio,
                                       "utils.storage.oss_util": None}):
            client = create(self.config("minio"))
            self.assertEqual(await client.get_object("cam/picture/a.jpg"), b"minio:cam/picture/a.jpg")
        self.assertEqual(created, [{"endpoint": "minio.example.com:9000",
            "access_key": "minio-test-key", "secret_key": "minio-test-secret",
            "bucket_name": "minio-images", "secure": False}])

    async def test_unknown_backend_is_not_silently_routed_to_oss(self):
        create = self.factory()
        with self.assertRaisesRegex(ValueError, "STORAGE_BACKEND"):
            create(self.config("s3"))

    async def test_worker_initialization_uses_selected_storage_factory(self):
        class FactoryProbe(Exception):
            pass
        with _MainImport() as main:
            with patch.object(main, "create_storage_client", create=True,
                              side_effect=FactoryProbe("selected backend")):
                try:
                    await main.run_worker_service()
                except Exception as error:
                    self.assertIsInstance(error, FactoryProbe)
                else:
                    self.fail("worker did not initialize the configured storage backend")


if __name__ == "__main__":
    unittest.main()
