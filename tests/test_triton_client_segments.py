from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.infer.capabilities import ModelCapability
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.results import FaceQualityResult, FeatureResult
from core.infer.triton_client import UnifiedTritonClient


def facade_settings():
    return SimpleNamespace(
        TRITON_SERVER_URL="triton.example:8001",
        TRITON_BATCH_TARGET_DEFAULT=1,
        TRITON_BATCH_WAIT_MS_DEFAULT=0.0,
        TRITON_BATCH_TARGETS={},
        TRITON_BATCH_WAIT_MS={},
        TRITON_QUEUE_TIMEOUT_SECONDS=5.0,
        TRITON_INFER_TIMEOUT_SECONDS=30.0,
        TRITON_CONFIG_TIMEOUT_SECONDS=5.0,
        TRITON_CLIENT_POOL_SIZE=2,
        TRITON_EXECUTOR_CONCURRENCY=8,
        TRITON_EXECUTOR_QUEUE_CAPACITY=64,
        TRITON_TRANSIENT_RETRIES=0,
    )


def target_one_capabilities():
    return {
        key: ModelCapability(
            key=key,
            model_name=contract.model_name,
            api_contract_version=contract.api_contract_version,
            server_max_batch=4,
            target_batch=1,
        )
        for key, contract in MODEL_CONTRACTS.items()
    }


class FriendlyTransport:
    async def start(self):
        pass

    async def is_server_ready(self):
        return True

    async def close(self):
        pass

    async def infer(self, contract, inputs, *, request_id):
        del request_id
        key = contract.key
        if key == "clip_image":
            return {
                "unnorm_image_features": np.ones((1, 512), dtype=np.float32)
            }
        if key == "face_rec":
            return {"EMBEDDING": np.ones((1, 512), dtype=np.float32)}
        if key == "tbps_image":
            return {"image_feat": np.ones((1, 512), dtype=np.float32)}
        if key == "face_qa":
            return {"quality_score": np.array([[8.0]], dtype=np.float32)}
        if key == "face_kps":
            return {
                "LANDMARKS_106": np.zeros((1, 106, 2), dtype=np.float32)
            }
        if key == "face_det":
            detections = np.zeros((1, 100, 6), dtype=np.float32)
            detections[0, 0] = [1, 1, 4, 4, 0.9, 0]
            return {
                "DETECTIONS": detections,
                "NUM_DETECTIONS": np.array([[1]], dtype=np.int32),
            }
        if key == "person_seg":
            batch = inputs["RAW_IMAGE"].shape[0]
            detections = np.zeros((batch, 1, 6), dtype=np.float32)
            detections[:, 0] = [1, 1, 4, 4, 0.9, 0]
            return {
                "DETECTIONS": detections,
                "MASK_CROPS_PACKED": np.full(
                    (batch, 1, 160, 20), 255, dtype=np.uint8
                ),
                "NUM_DETECTIONS": np.ones((batch, 1), dtype=np.int32),
            }
        raise AssertionError(f"unexpected inference for {key}")


class TritonClientFriendlyReturnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = UnifiedTritonClient(
            settings=facade_settings(), transport=FriendlyTransport()
        )
        discovery = AsyncMock(return_value=target_one_capabilities())
        with patch(
            "core.infer.triton_client.discover_all_capabilities", discovery
        ):
            await self.client.initialize()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_remaining_friendly_methods_return_business_types(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        encoded = np.arange(10, dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        bbox = np.array([1, 1, 4, 4], dtype=np.float32)

        clip = await self.client.extract_image_features(b"jpeg")
        face = await self.client.extract_face_features(image, landmarks)
        tbps = await self.client.extract_tbps_image_features(encoded)
        quality = await self.client.extract_face_quality(image, landmarks)
        detections = await self.client.extract_face_detections(image)
        keypoints = await self.client.extract_face_keypoints(image, bbox)
        segments = await self.client.extract_person_segments(image)

        self.assertIsInstance(clip, FeatureResult)
        self.assertEqual(clip.vector.shape, (512,))
        self.assertIsInstance(face, FeatureResult)
        self.assertEqual(face.vector.shape, (512,))
        self.assertIsInstance(tbps, FeatureResult)
        self.assertEqual(tbps.vector.shape, (512,))
        self.assertIsInstance(quality, FaceQualityResult)
        self.assertEqual(quality.score, 8.0)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 1)
        self.assertEqual(keypoints.shape, (5, 2))
        self.assertIsInstance(segments, list)
        self.assertEqual(segments[0]["maskPixelCount"], 9)


if __name__ == "__main__":
    unittest.main()
