from __future__ import annotations

import asyncio
import gc
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.infer.capabilities import ModelCapability
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.results import FaceQualityResult, FeatureResult
from core.infer.triton_client import UnifiedTritonClient


def target_one_settings():
    return SimpleNamespace(
        TRITON_SERVER_URL="unused:8001",
        TRITON_BATCH_TARGET_DEFAULT=1,
        TRITON_BATCH_WAIT_MS_DEFAULT=0.0,
        TRITON_BATCH_TARGETS={},
        TRITON_BATCH_WAIT_MS={},
        TRITON_QUEUE_TIMEOUT_SECONDS=1.0,
        TRITON_INFER_TIMEOUT_SECONDS=2.0,
        TRITON_CONFIG_TIMEOUT_SECONDS=3.0,
        TRITON_CLIENT_POOL_SIZE=1,
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
            server_max_batch=8,
            target_batch=1,
        )
        for key, contract in MODEL_CONTRACTS.items()
    }


class CapturingTransport:
    def __init__(self):
        self.requests: dict[str, dict[str, np.ndarray]] = {}
        self.closed = False

    async def start(self):
        pass

    async def is_server_ready(self):
        return True

    async def close(self):
        self.closed = True

    async def infer(self, contract, inputs, *, request_id):
        del request_id
        self.requests[contract.key] = {
            name: np.array(value, copy=True) for name, value in inputs.items()
        }
        batch = next(iter(inputs.values())).shape[0]
        if contract.key == "clip_image":
            return {
                "unnorm_image_features": np.ones(
                    (batch, 512), dtype=np.float32
                )
            }
        if contract.key == "face_rec":
            return {"EMBEDDING": np.ones((batch, 512), dtype=np.float32)}
        if contract.key == "tbps_image":
            return {"image_feat": np.ones((batch, 512), dtype=np.float32)}
        if contract.key == "sbs_image":
            return {"embedding": np.ones((batch, 2048), dtype=np.float32)}
        if contract.key == "face_qa":
            return {
                "quality_score": np.full((batch, 1), 8.0, dtype=np.float32)
            }
        if contract.key == "face_kps":
            return {
                "LANDMARKS_106": np.zeros(
                    (batch, 106, 2), dtype=np.float32
                )
            }
        detections = np.zeros((batch, 100, 6), dtype=np.float32)
        detections[:, 0] = [1, 1, 4, 4, 0.9, 0]
        counts = np.ones((batch, 1), dtype=np.int32)
        if contract.key == "face_det":
            return {
                "DETECTIONS": detections,
                "NUM_DETECTIONS": counts,
            }
        if contract.key == "person_seg":
            segment_detections = np.zeros(
                (batch, 1, 6), dtype=np.float32
            )
            segment_detections[:, 0] = [1, 1, 4, 4, 0.9, 0]
            return {
                "DETECTIONS": segment_detections,
                "MASK_CROPS_PACKED": np.full(
                    (batch, 1, 160, 20), 255, dtype=np.uint8
                ),
                "NUM_DETECTIONS": counts,
            }
        if contract.key == "person_pose":
            keypoints = np.zeros(
                (batch, 100, 17, 3), dtype=np.float32
            )
            keypoints[:, 0, :, 2] = 0.9
            return {
                "DETECTIONS": detections,
                "KEYPOINTS": keypoints,
                "NUM_DETECTIONS": counts,
            }
        raise AssertionError(f"unexpected inference for {contract.key}")


class TritonBatchContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_sample_keeps_shared_numeric_storage_alive_during_rpc(self):
        entered, release = asyncio.Event(), asyncio.Event()
        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3).copy()
        owner_ref = weakref.ref(image)
        observed = {}

        class GatedTransport(CapturingTransport):
            async def infer(self, contract, inputs, *, request_id):
                observed["shared"] = np.shares_memory(
                    inputs["RAW_IMAGE"], owner_ref()
                )
                entered.set()
                await release.wait()
                observed["payload"] = inputs["RAW_IMAGE"].tobytes()
                return await super().infer(
                    contract, inputs, request_id=request_id
                )

        client = UnifiedTritonClient(
            settings=target_one_settings(), transport=GatedTransport()
        )
        with patch(
            "core.infer.triton_client.discover_all_capabilities",
            new=AsyncMock(return_value=target_one_capabilities()),
        ):
            await client.initialize()
        inference = asyncio.create_task(client.extract_face_detections(image))
        del image
        try:
            await asyncio.wait_for(entered.wait(), 1)
            gc.collect()
            self.assertIsNotNone(owner_ref())
            self.assertTrue(observed["shared"])
        finally:
            release.set()
            await inference
            await client.close()
        self.assertEqual(observed["payload"], bytes(range(36)))

    async def test_encoded_facade_batches_native_and_legacy_bytes_without_padding(self):
        settings = target_one_settings()
        settings.TRITON_BATCH_TARGET_DEFAULT = 2
        settings.TRITON_BATCH_WAIT_MS_DEFAULT = 20.0
        capabilities = {
            key: replace(value, target_batch=2)
            for key, value in target_one_capabilities().items()
        }
        transport = CapturingTransport()
        client = UnifiedTritonClient(settings=settings, transport=transport)
        with patch(
            "core.infer.triton_client.discover_all_capabilities",
            new=AsyncMock(return_value=capabilities),
        ):
            await client.initialize()
        native = b"abc\x00\x00"
        legacy = np.asarray([8, 0], dtype=np.uint8)
        try:
            for key, extract in (
                ("sbs_image", client.extract_sbs_features),
                ("tbps_image", client.extract_tbps_image_features),
            ):
                with self.subTest(key=key):
                    results = await asyncio.gather(
                        extract(native), extract(legacy), return_exceptions=True
                    )
                    self.assertTrue(
                        all(isinstance(result, FeatureResult) for result in results),
                        results,
                    )
                    packed = transport.requests[key]["IMAGE_BYTES"]
                    self.assertEqual(packed.shape, (2, 1))
                    self.assertEqual(packed.dtype, object)
                    self.assertIs(packed[0, 0], native)
                    self.assertEqual(packed[1, 0], b"\x08\x00")
        finally:
            await client.close()

    async def test_target_one_facade_keeps_wire_batch_and_hides_result_batch(self):
        transport = CapturingTransport()
        client = UnifiedTritonClient(
            settings=target_one_settings(), transport=transport
        )
        with patch(
            "core.infer.triton_client.discover_all_capabilities",
            new=AsyncMock(return_value=target_one_capabilities()),
        ):
            await client.initialize()

        image = np.zeros((10, 12, 3), dtype=np.uint8)
        encoded = np.arange(17, dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        bbox = np.array([1, 1, 4, 4], dtype=np.float32)

        results = {
            "clip_image": await client.extract_image_features(b"jpeg"),
            "face_rec": await client.extract_face_features(image, landmarks),
            "tbps_image": await client.extract_tbps_image_features(encoded),
            "sbs_image": await client.extract_sbs_features(encoded),
            "face_qa": await client.extract_face_quality(image, landmarks),
            "face_kps": await client.extract_face_keypoints(image, bbox),
            "face_det": await client.extract_face_detections(image),
            "person_seg": await client.extract_person_segments(image),
            "person_pose": await client.extract_person_poses(image),
        }

        self.assertEqual(set(transport.requests), set(MODEL_CONTRACTS))
        for key, inputs in transport.requests.items():
            with self.subTest(model=key):
                self.assertTrue(inputs)
                self.assertTrue(
                    all(value.shape[0] == 1 for value in inputs.values())
                )
        self.assertEqual(
            transport.requests["clip_image"]["IMAGE_BYTES"].shape,
            (1, 1),
        )
        for key in (
            "face_rec",
            "face_kps",
            "face_det",
            "person_seg",
            "person_pose",
        ):
            self.assertEqual(
                transport.requests[key]["RAW_IMAGE"].shape,
                (1, 10, 12, 3),
            )
        self.assertEqual(
            transport.requests["face_qa"]["RAW_FACE_IMAGE"].shape,
            (1, 10, 12, 3),
        )
        self.assertEqual(
            transport.requests["face_qa"]["LANDMARK"].shape,
            (1, 5, 2),
        )

        self.assertIsInstance(results["clip_image"], FeatureResult)
        self.assertEqual(results["clip_image"].vector.shape, (512,))
        self.assertEqual(results["face_rec"].vector.shape, (512,))
        self.assertEqual(results["tbps_image"].vector.shape, (512,))
        self.assertEqual(results["sbs_image"].vector.shape, (2048,))
        self.assertIsInstance(results["face_qa"], FaceQualityResult)
        self.assertIsInstance(results["face_qa"].score, float)
        self.assertEqual(results["face_kps"].shape, (5, 2))
        self.assertEqual(len(results["face_det"]), 1)
        self.assertEqual(results["person_seg"][0]["mask"].image_shape, (10, 12))
        self.assertEqual(results["person_seg"][0]["mask"].data.shape, (3, 3))
        self.assertEqual(results["person_seg"][0]["mask"].origin, (1, 1))
        self.assertEqual(
            results["person_pose"][0]["keypoints"].shape, (17, 3)
        )

        await client.close()
        self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
