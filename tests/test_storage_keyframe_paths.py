import unittest
from types import SimpleNamespace
from unittest.mock import patch

import core.config as config
from utils.keyframe.detection_helpers import (
    parse_keyframe_context,
    parse_pano_keyframe_context,
)


class StorageKeyframePathTests(unittest.TestCase):
    def setUp(self):
        self.storage_settings = SimpleNamespace(
            STORAGE_BACKEND="minio",
            MINIO_ENDPOINT="minio.example:9000",
            MINIO_BUCKET_NAME="frames",
            MINIO_SECURE=False,
        )
        self.settings_patch = patch.object(config, "settings", self.storage_settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def parse(self, value, **fields):
        return parse_keyframe_context(
            {"keyframe": value, "uId": "frame-id", **fields}, now_ms=1234
        )

    def test_minio_url_removes_bucket_before_inferring_device_sn(self):
        context = self.parse("http://minio.example:9000/frames/CAM001/picture/main.jpg")
        self.assertEqual(context.object_name, "CAM001/picture/main.jpg")
        self.assertEqual(context.sn, "CAM001")
        self.assertEqual(context.keyframe_id, "frame-id")

    def test_minio_signed_url_decodes_key_without_retaining_query(self):
        context = self.parse(
            "http://MINIO.EXAMPLE:9000/frames/CAM%20001/picture/a%3Fb%20c.jpg"
            "?X-Amz-Signature=signature#ignored"
        )
        self.assertEqual(context.object_name, "CAM 001/picture/a?b c.jpg")
        self.assertEqual(context.sn, "CAM 001")

    def test_minio_url_decodes_bucket_segment(self):
        context = self.parse("http://minio.example:9000/fr%61mes/CAM001/picture/main.jpg")
        self.assertEqual(context.object_name, "CAM001/picture/main.jpg")

    def test_minio_full_url_preserves_slashes_within_object_key(self):
        context = self.parse("http://minio.example:9000/frames/%2FCAM001/picture/main.jpg")
        self.assertEqual(context.object_name, "/CAM001/picture/main.jpg")

    def test_minio_rejects_explicit_zero_port_instead_of_using_default(self):
        self.storage_settings.MINIO_ENDPOINT = "minio.example"
        with self.assertRaises(ValueError):
            self.parse("http://minio.example:0/frames/CAM001/picture/main.jpg")

    def test_raw_keys_keep_first_segment_even_when_it_is_bucket_name(self):
        for value in ("frames/picture/main.jpg", "/frames/picture/main.jpg?signature=x"):
            with self.subTest(value=value):
                context = self.parse(value)
                self.assertEqual(context.object_name, "frames/picture/main.jpg")
                self.assertEqual(context.sn, "frames")

    def test_short_name_keeps_existing_sn_and_extension_rules(self):
        context = self.parse("main?signature=x", sn="CAM001")
        self.assertEqual(context.object_name, "CAM001/picture/main.jpg")

    def test_main_and_pano_urls_use_same_bucket_resolution(self):
        payload = {
            "keyframeUrl": "http://minio.example:9000/frames/CAM001/picture/main.jpg",
            "panoKeyframeUrl": "http://minio.example:9000/frames/CAM001/picture/pano%20one.jpg?x=1",
            "timestamp": "1000",
            "uId": "frame-id",
        }
        context = parse_pano_keyframe_context(payload)
        self.assertEqual(context.object_name, "CAM001/picture/pano one.jpg")
        self.assertEqual(context.sn, "CAM001")
        self.assertEqual(context.timestamp, 1000)
        self.assertEqual(context.keyframe_id, "frame-id")
        self.assertTrue(context.is_pano_keyframe)

    def test_short_pano_name_inherits_sn_from_minio_main_url(self):
        context = parse_pano_keyframe_context(
            {
                "keyframe": "http://minio.example:9000/frames/CAM001/picture/main.jpg",
                "panoKeyframeUrl": "pano",
                "uId": "frame-id",
            }
        )
        self.assertEqual(context.object_name, "CAM001/picture/pano.jpg")

    def test_minio_rejects_url_with_different_bucket(self):
        with self.assertRaises(ValueError):
            self.parse("http://minio.example:9000/other/CAM001/picture/main.jpg")

    def test_minio_rejects_untrusted_endpoint_or_credentials(self):
        for value in (
            "http://untrusted.example:9000/frames/CAM001/picture/main.jpg",
            "http://minio.example:9001/frames/CAM001/picture/main.jpg",
            "http://user:pass@minio.example:9000/frames/CAM001/picture/main.jpg",
            "ftp://minio.example:9000/frames/CAM001/picture/main.jpg",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_minio_rejects_pano_url_with_different_bucket(self):
        with self.assertRaises(ValueError):
            parse_pano_keyframe_context(
                {
                    "keyframe": "CAM001/picture/main.jpg",
                    "panoKeyframeUrl": "http://minio.example:9000/other/CAM001/picture/pano.jpg",
                    "uId": "frame-id",
                }
            )

    def test_minio_rejects_bucket_only_url_and_empty_key(self):
        for path in ("frames", "frames/", "/frames/CAM001/picture/main.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.parse("http://minio.example:9000/" + path)

    def test_minio_supports_endpoint_without_explicit_port(self):
        self.storage_settings.MINIO_ENDPOINT = "minio.example"
        for value in (
            "http://minio.example/frames/CAM001/picture/main.jpg",
            "http://minio.example:80/frames/CAM001/picture/main.jpg",
        ):
            with self.subTest(value=value):
                self.assertEqual(self.parse(value).object_name, "CAM001/picture/main.jpg")

    def test_minio_secure_endpoint_uses_https_default_port(self):
        self.storage_settings.MINIO_ENDPOINT = "minio.example"
        self.storage_settings.MINIO_SECURE = True
        self.assertEqual(
            self.parse("https://minio.example/frames/CAM001/picture/main.jpg").object_name,
            "CAM001/picture/main.jpg",
        )

    def test_oss_retains_existing_key_and_url_rules(self):
        self.storage_settings.STORAGE_BACKEND = "oss"
        cases = (
            ("/CAM001/picture/main.jpg?x=1", "CAM001/picture/main.jpg", "CAM001"),
            ("https://oss.example/CAM001/picture/a%20b.jpg?x=1", "CAM001/picture/a b.jpg", "CAM001"),
            ("http://minio.example:9000/frames/CAM001/picture/main.jpg", "frames/CAM001/picture/main.jpg", "frames"),
        )
        for value, expected, sn in cases:
            with self.subTest(value=value):
                context = self.parse(value)
                self.assertEqual(context.object_name, expected)
                self.assertEqual(context.sn, sn)


if __name__ == "__main__":
    unittest.main()
