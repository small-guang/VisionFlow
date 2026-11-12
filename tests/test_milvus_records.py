import asyncio
import importlib
import importlib.util
import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from utils.reid.person_body import BODY_FULL, BODY_HALF_LOWER, BODY_HALF_UPPER
from utils.quality.person.models import PersonQualityDecision
from utils.reid.reid_models import (
    REID_SOURCE_BUFFER,
    ReidCandidate,
)
from utils.reid.reid_resolution import resolve_reid_classes


_MISSING = object()


class RecordingVectorDb:
    def __init__(self, collection_name, dim):
        self.collection_name = collection_name
        self.dim = dim


class RecordingSbsVectorDb(RecordingVectorDb):
    def __init__(self):
        super().__init__("alogSBS", 2048)


class RecordingTbpsVectorDb(RecordingVectorDb):
    def __init__(self):
        super().__init__("alogTBPS", 512)


class DummyLogger:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class FakeVectorDb:
    def __init__(self, collection_name, primary_keys=None, result_key="primary_keys"):
        self.collection_name = collection_name
        self.primary_keys = primary_keys or []
        self.result_key = result_key
        self.calls = []

    async def ainsert_embeddings(self, **kwargs):
        self.calls.append(kwargs)
        result = {self.result_key: self.primary_keys}
        if self.collection_name == "alogFace":
            result["outcomes"] = [
                {"action": "inserted", "persisted": True, "primary_key": key}
                for key in self.primary_keys
            ]
        return result

    async def aapply_prototype_updates(self, batch):
        self.calls.append(
            {
                "faceId": [item["faceId"] for item in batch],
                "reId": [item["reId"] for item in batch],
                "bodyCompleteness": [item["bodyCompleteness"] for item in batch],
                "cropUrl": [item["cropUrl"] for item in batch],
                "embedding": [item["embedding"] for item in batch],
                "deviceId": [item["deviceId"] for item in batch],
                "trackId": [0 for _ in batch],
                "keyframeId": [item["pictureVectorId"] for item in batch],
                "personRect": [item["personRect"] for item in batch],
                "timestamp": [item["timestamp"] for item in batch],
                "qualityScore": [item["qualityScore"] for item in batch],
            }
        )
        primary_keys = list(self.primary_keys)
        return [
            {
                "item": item,
                "action": "insert",
                "reason": "test_insert",
                "retry": False,
                "persisted": True,
                "insert_result": {
                    self.result_key: primary_keys[index : index + 1]
                },
            }
            for index, item in enumerate(batch)
        ]

class IterablePrimaryKeys:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)


class BlockingVectorDb(FakeVectorDb):
    def __init__(self, collection_name, started, proceed, primary_keys=None):
        super().__init__(collection_name, primary_keys=primary_keys)
        self.started = started
        self.proceed = proceed

    async def ainsert_embeddings(self, **kwargs):
        self.started.set()
        await self.proceed.wait()
        return await super().ainsert_embeddings(**kwargs)

    async def aapply_prototype_updates(self, batch):
        self.started.set()
        await self.proceed.wait()
        return await super().aapply_prototype_updates(batch)


class FailingVectorDb(FakeVectorDb):
    async def ainsert_embeddings(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("write failed")

    async def aapply_prototype_updates(self, batch):
        self.calls.append(batch)
        raise RuntimeError("write failed")


async def enqueue_formal_sbs(buffer, **kwargs):
    await buffer._enqueue_formal_feature(
        **kwargs,
        crop_url="",
        quality_score=8.0,
    )


async def resolve_buffer_candidates(buffer, embedding, **kwargs):
    candidates, highest_score, highest_reid, _ = (
        await buffer.asearch_candidates_with_hit_info(embedding, **kwargs)
    )
    resolution = resolve_reid_classes(
        candidates,
        query_face_id=kwargs.get("face_id"),
        query_device_id=(kwargs.get("device_ids") or [""])[0],
    )
    if resolution.reid is None:
        return None, highest_score, highest_reid
    return resolution.reid, resolution.score, highest_reid


class MilvusRecordColumnTests(unittest.TestCase):
    def setUp(self):
        self.module_names = ["core.config", "pymilvus"]
        self.original_modules = {
            name: sys.modules.get(name, _MISSING) for name in self.module_names
        }

        config_module = types.ModuleType("core.config")
        config_module.settings = types.SimpleNamespace(
            milvus_uri="http://localhost:19530",
            milvus_token="token",
        )
        sys.modules["core.config"] = config_module

        pymilvus_module = types.ModuleType("pymilvus")
        pymilvus_module.DataType = types.SimpleNamespace(
            INT64="INT64",
            FLOAT_VECTOR="FLOAT_VECTOR",
            VARCHAR="VARCHAR",
        )
        pymilvus_module.MilvusClient = object
        sys.modules["pymilvus"] = pymilvus_module

        module_path = Path(__file__).resolve().parents[1] / "db" / "zilliz" / "picture_feature.py"
        spec = importlib.util.spec_from_file_location(
            "picture_feature_under_test", module_path
        )
        self.picture_feature = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.picture_feature)

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_single_item_batch_unwraps_column_values(self):
        records = self.picture_feature.MilvusVectorDB._records_from_columns(
            ["faceId", "embedding", "faceKeyPoints"],
            [
                ["face-1"],
                [np.array([0.1, 0.2], dtype=np.float32)],
                [[{"x": 1, "y": 2}]],
            ],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["faceId"], "face-1")
        self.assertAlmostEqual(records[0]["embedding"][0], 0.1)
        self.assertAlmostEqual(records[0]["embedding"][1], 0.2)
        self.assertEqual(records[0]["faceKeyPoints"], [{"x": 1, "y": 2}])

    def test_single_item_person_batch_unwraps_face_and_reid_values(self):
        records = self.picture_feature.MilvusVectorDB._records_from_columns(
            ["faceId", "reId", "embedding", "personRect"],
            [
                ["face-1"],
                ["reid-1"],
                [np.array([0.1, 0.2], dtype=np.float32)],
                [{"x": 1, "y": 2, "w": 3, "h": 4}],
            ],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["faceId"], "face-1")
        self.assertEqual(records[0]["reId"], "reid-1")
        self.assertAlmostEqual(records[0]["embedding"][0], 0.1)
        self.assertAlmostEqual(records[0]["embedding"][1], 0.2)
        self.assertEqual(records[0]["personRect"], {"x": 1, "y": 2, "w": 3, "h": 4})

    def test_scalar_single_record_preserves_list_field_values(self):
        records = self.picture_feature.MilvusVectorDB._records_from_columns(
            ["faceId", "embedding", "faceKeyPoints"],
            [
                "face-1",
                np.array([0.1, 0.2], dtype=np.float32),
                [{"x": 1, "y": 2}],
            ],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["faceId"], "face-1")
        self.assertAlmostEqual(records[0]["embedding"][0], 0.1)
        self.assertAlmostEqual(records[0]["embedding"][1], 0.2)
        self.assertEqual(records[0]["faceKeyPoints"], [{"x": 1, "y": 2}])

    def test_multi_item_batch_builds_one_record_per_row(self):
        records = self.picture_feature.MilvusVectorDB._records_from_columns(
            ["faceId", "embedding"],
            [
                ["face-1", "face-2"],
                [
                    np.array([0.1, 0.2], dtype=np.float32),
                    np.array([0.3, 0.4], dtype=np.float32),
                ],
            ],
        )

        self.assertEqual([record["faceId"] for record in records], ["face-1", "face-2"])
        self.assertAlmostEqual(records[0]["embedding"][0], 0.1)
        self.assertAlmostEqual(records[0]["embedding"][1], 0.2)
        self.assertAlmostEqual(records[1]["embedding"][0], 0.3)
        self.assertAlmostEqual(records[1]["embedding"][1], 0.4)

    def test_face_gallery_storage_error_is_not_reported_as_a_search_miss(self):
        package_name = "face_gallery_failure_under_test"
        face_module_name = f"{package_name}.face_feature"
        picture_module_name = f"{package_name}.picture_feature"
        package_path = Path(__file__).resolve().parents[1] / "db" / "zilliz"
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package
        sys.modules[picture_module_name] = self.picture_feature

        try:
            spec = importlib.util.spec_from_file_location(
                face_module_name,
                package_path / "face_feature.py",
            )
            face_feature = importlib.util.module_from_spec(spec)
            sys.modules[face_module_name] = face_feature
            spec.loader.exec_module(face_feature)

            class FailingSearchClient:
                def search(self, **_kwargs):
                    raise RuntimeError("face gallery unavailable")

            gallery = object.__new__(face_feature.FaceGallery)
            gallery.client = FailingSearchClient()
            gallery.collection_name = "alogFace"

            with self.assertRaisesRegex(RuntimeError, "face gallery unavailable"):
                gallery.search_similar_faces_with_hit_info(
                    np.ones(512, dtype=np.float32),
                    threshold=0.48,
                )
        finally:
            sys.modules.pop(face_module_name, None)
            sys.modules.pop(picture_module_name, None)
            sys.modules.pop(package_name, None)


class ZillizCollectionConfigurationTests(unittest.TestCase):
    def test_sbs_collection_is_2048_dimensional_and_other_dimensions_stay_512(self):
        package_name = "zilliz_config_under_test"
        module_names = [
            package_name,
            f"{package_name}.face_feature",
            f"{package_name}.picture_feature",
            f"{package_name}.SBS_feature",
            f"{package_name}.TBPS_feature",
        ]
        original_modules = {
            name: sys.modules.get(name, _MISSING) for name in module_names
        }

        try:
            symbols = {
                "face_feature": "FaceGallery",
                "picture_feature": "MilvusVectorDB",
                "SBS_feature": "SbsReidGallery",
                "TBPS_feature": "TbpsGallery",
            }
            for child_name, symbol_name in symbols.items():
                child_module = types.ModuleType(f"{package_name}.{child_name}")
                fake_class = {
                    "SBS_feature": RecordingSbsVectorDb,
                    "TBPS_feature": RecordingTbpsVectorDb,
                }.get(child_name, RecordingVectorDb)
                setattr(child_module, symbol_name, fake_class)
                sys.modules[child_module.__name__] = child_module

            package_path = Path(__file__).resolve().parents[1] / "db" / "zilliz"
            spec = importlib.util.spec_from_file_location(
                package_name,
                package_path / "__init__.py",
                submodule_search_locations=[str(package_path)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)

            self.assertEqual(module.person_sbs_embedding_db.collection_name, "alogSBS")
            self.assertEqual(module.person_sbs_embedding_db.dim, 2048)
            self.assertEqual(module.face_embedding_db.dim, 512)
            self.assertEqual(module.person_clip_embedding_db.dim, 512)
            self.assertEqual(module.picture_clip_embedding_db.dim, 512)
        finally:
            for name, original_module in original_modules.items():
                if original_module is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module


class FeatureBufferRoutingTests(unittest.TestCase):
    def test_detection_emitter_requires_quality_decision(self):
        from core.person_keyframe_processor import PersonKeyframeProcessor
        from utils.keyframe.detection_helpers import KeyframeContext

        processor = object.__new__(PersonKeyframeProcessor)
        processor.feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        ctx = KeyframeContext(sn="CAM001", timestamp=1, object_name="frame.jpg")

        async def emit():
            await processor.queue_person_detection_record(
                ctx, "frame-1", 1, "crop.jpg", {"x": 1, "y": 2, "w": 30, "h": 60},
                "ai-1", 1, 1, 0.9, 1200, "detected",
            )

        with self.assertRaisesRegex(TypeError, "quality_decision"):
            asyncio.run(emit())

    def test_recognition_emitter_requires_quality_decision(self):
        from core.person_keyframe_processor import PersonKeyframeProcessor
        from utils.keyframe.detection_helpers import KeyframeContext

        processor = object.__new__(PersonKeyframeProcessor)
        processor.feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        ctx = KeyframeContext(sn="CAM001", timestamp=1, object_name="frame.jpg")

        async def emit():
            await processor.queue_person_recognition_record(
                ctx, "frame-1", 1, "frame.jpg", "crop.jpg", "face-1", "reid-1",
                0.9, 1, {"x": 1, "y": 2, "w": 30, "h": 60}, "matched", 2, "ai-1", "",
            )

        with self.assertRaisesRegex(TypeError, "quality_decision"):
            asyncio.run(emit())
    def setUp(self):
        self.module_names = [
            "utils.buffers.buffer",
            "utils.buffers.person_feature_buffer",
            "utils.buffers.sbs_feature_buffer",
            "utils.buffers.tbps_feature_buffer",
            "aiokafka",
            "loguru",
            "core.config",
            "db.zilliz",
        ]
        self.original_modules = {
            name: sys.modules.get(name, _MISSING) for name in self.module_names
        }
        for name in self.module_names:
            sys.modules.pop(name, None)

        aiokafka_module = types.ModuleType("aiokafka")
        aiokafka_module.AIOKafkaProducer = object
        sys.modules["aiokafka"] = aiokafka_module

        loguru_module = types.ModuleType("loguru")
        loguru_module.logger = DummyLogger()
        sys.modules["loguru"] = loguru_module

        settings = types.SimpleNamespace(
            KAFKA_FACE_DETECTION_RECORD_TOPIC="face-detection",
            KAFKA_FACE_RECOGNITION_RECORD_TOPIC="face-recognition",
            KAFKA_PERSON_DETECTION_RECORD_TOPIC="person-detection",
            KAFKA_PERSON_RECOGNITION_RECORD_TOPIC="person-recognition",
            KAFKA_VECTOR_RECORD_TOPIC="vector",
            kafka_client_config={},
            FACE_FRAME_CONCURRENCY=4,
            FACE_GLOBAL_CONCURRENCY=24,
            PERSON_FRAME_CONCURRENCY=6,
            PERSON_GLOBAL_CONCURRENCY=32,
            person_upright_device_sns=frozenset(),
            person_face_reuse_mode="on",
        )
        core_config_module = types.ModuleType("core.config")
        core_config_module.settings = settings
        sys.modules["core.config"] = core_config_module

        self.face_db = FakeVectorDb("alogFace", primary_keys=[77])
        self.person_db = FakeVectorDb("alogTBPS", primary_keys=[88])
        self.sbs_db = FakeVectorDb("alogSBS", primary_keys=[99])
        self.picture_db = FakeVectorDb("alogPictureClip")
        db_zilliz_module = types.ModuleType("db.zilliz")
        db_zilliz_module.face_embedding_db = self.face_db
        db_zilliz_module.person_clip_embedding_db = self.person_db
        db_zilliz_module.person_sbs_embedding_db = self.sbs_db
        db_zilliz_module.picture_clip_embedding_db = self.picture_db
        sys.modules["db.zilliz"] = db_zilliz_module

        self.buffer_module = importlib.import_module("utils.buffers.buffer")

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_picture_flush_uses_picture_vector_id_as_milvus_primary_key(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        uid = "11111111-1111-4111-8111-111111111111"

        async def run():
            await feature_buffer.add_picture_feature(
                keyframe_id=uid,
                picture_vector_id=42,
                embedding=[0.1, 0.2],
                device_id="CAM001",
                image_path="CAM001/picture/k.jpg",
                timestamp=171,
                is_pano_keyframe=True,
            )
            await feature_buffer._flush_pictures()

        asyncio.run(run())

        self.assertEqual(self.picture_db.calls[0]["keyframeId"], [42])
        record = feature_buffer.vector_record_buffer.get_nowait()
        self.assertEqual(record["vectorId"], 42)
        self.assertEqual(record["keyframeId"], uid)
        self.assertEqual(record["pictureVectorId"], 42)
        self.assertIs(record["isPanoKeyframe"], True)
        self.assertNotIn("isPanoKeyframe", self.picture_db.calls[0])

    def test_face_flush_keeps_uid_in_vector_record_and_milvus_uses_picture_id(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        uid = "22222222-2222-4222-8222-222222222222"

        async def run():
            await feature_buffer.add_face_feature(
                face_id="face-1",
                embedding=[0.1, 0.2],
                device_id="CAM001",
                keyframe_id=uid,
                picture_vector_id=42,
                keyframe_url="CAM001/picture/k.jpg",
                face_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                face_key_points=[],
                face_crop_path="CAM001/face_crop/f.jpg",
                timestamp=171,
                ai_id="ai-1",
                quality_score=8.25,
                is_pano_keyframe=True,
            )
            await feature_buffer._flush_faces()

        asyncio.run(run())

        self.assertEqual(self.face_db.calls[0]["keyframeId"], [42])
        self.assertEqual(self.face_db.calls[0]["qualityScore"], [8.25])
        record = feature_buffer.vector_record_buffer.get_nowait()
        self.assertEqual(record["vectorId"], 77)
        self.assertIs(record["isPanoKeyframe"], True)
        self.assertNotIn("isPanoKeyframe", self.face_db.calls[0])
        self.assertEqual(record["keyframeId"], uid)
        self.assertEqual(record["pictureVectorId"], 42)

    def test_face_quality_skip_clears_pending_and_only_written_sample_emits_vector_record(self):
        class SelectiveFaceDb(FakeVectorDb):
            async def ainsert_embeddings(self, **kwargs):
                return {"ids": [88], "outcomes": [
                    {"action": "skipped", "persisted": True, "primary_key": None},
                    {"action": "replaced", "persisted": True, "primary_key": 88},
                ]}

        feature_buffer = self.buffer_module.FeatureBuffer(
            max_batch_size=10, face_db=SelectiveFaceDb("alogFace"),
        )

        async def run():
            for index, quality in enumerate((7.1, 9.0)):
                await feature_buffer.add_face_feature(
                    face_id="face-1", embedding=[1.0, 0.0], device_id="CAM001",
                    keyframe_id=f"frame-{index}", picture_vector_id=42 + index,
                    keyframe_url="picture.jpg", face_rect={}, face_key_points=[],
                    face_crop_path=f"face-{index}.jpg", timestamp=100 + index,
                    ai_id="ai-1", quality_score=quality,
                    new_face_recognition_record={"aiFaceId": "face-1", "faceStatus": 1}
                    if index == 0 else None,
                )
            await feature_buffer._flush_faces()

        asyncio.run(run())
        self.assertTrue(feature_buffer.face_buffer.empty())
        self.assertEqual(feature_buffer.pending_faces, {})
        self.assertEqual(feature_buffer.vector_record_buffer.qsize(), 1)
        record = feature_buffer.vector_record_buffer.get_nowait()
        self.assertEqual(record["vectorId"], 88)
        self.assertEqual(record["keyframeId"], "frame-1")
        # The skipped first observation can release its birth record because a
        # different observation of the same identity is already durable.
        self.assertEqual(feature_buffer.face_recognition_record_buffer.qsize(), 1)

    def test_face_failed_flush_retries_with_same_sample_id(self):
        failing = FailingVectorDb("alogFace")
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10, face_db=failing)

        async def run():
            await feature_buffer.add_face_feature(
                face_id="face-1", embedding=[1.0, 0.0], device_id="CAM001",
                keyframe_id="frame", picture_vector_id=42, keyframe_url="picture.jpg",
                face_rect={}, face_key_points=[], face_crop_path="face.jpg",
                timestamp=100, ai_id="ai-1", quality_score=8.0,
            )
            await feature_buffer._flush_faces()
            await feature_buffer._flush_faces()

        asyncio.run(run())
        self.assertEqual(feature_buffer.face_buffer.qsize(), 1)
        self.assertEqual(len(feature_buffer.pending_faces), 1)
        self.assertEqual(len(failing.calls[0]["sampleId"][0]), 36)
        self.assertEqual(failing.calls[0]["sampleId"], failing.calls[1]["sampleId"])

    def test_cancelled_face_flush_does_not_republish_acknowledged_notifications(self):
        async def run():
            feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=2)
            feature_buffer.face_features.vector_db = FakeVectorDb("alogFace", primary_keys=[77, 78])
            blocked = asyncio.Event()
            sent = []

            class Producer:
                should_block = True

                async def send_and_wait(self, topic, payload):
                    record = json.loads(payload)
                    if topic == "face-recognition" and record["aiFaceId"] == "face-1" and self.should_block:
                        blocked.set()
                        await asyncio.Event().wait()
                    sent.append((topic, record))

            producer = Producer()
            feature_buffer._kafka_producer = producer

            async def add(index):
                await feature_buffer.add_face_feature(
                    face_id=f"face-{index}", embedding=[1.0, 0.0], device_id="CAM001",
                    keyframe_id=f"frame-{index}", picture_vector_id=42 + index,
                    keyframe_url="picture.jpg", face_rect={}, face_key_points=[],
                    face_crop_path=f"face-{index}.jpg", timestamp=100 + index,
                    ai_id=f"ai-{index}", quality_score=8.0,
                    new_face_recognition_record={"aiFaceId": f"face-{index}", "faceStatus": 1},
                )

            await add(0)
            task = asyncio.create_task(add(1))
            await asyncio.wait_for(blocked.wait(), timeout=3)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            producer.should_block = False
            await feature_buffer._flush_faces()
            await feature_buffer._flush_face_recognition_records()
            await feature_buffer._flush_vector_records()
            self.assertEqual([r["aiFaceId"] for topic, r in sent if topic == "face-recognition"], ["face-0", "face-1"])
            self.assertEqual([r["vectorId"] for topic, r in sent if topic == "vector"], [77, 78])
            self.assertEqual(feature_buffer.pending_faces, {})

        asyncio.run(run())

    def test_person_flush_keeps_uid_in_vector_record_and_milvus_uses_picture_id(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        uid = "33333333-3333-4333-8333-333333333333"

        async def run():
            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-1",
                re_id="reid-1",
                body_completeness=BODY_FULL,
                is_pano_keyframe=True,
                embedding=[0.1] * 2048,
                device_id="CAM001",
                keyframe_id=uid,
                picture_vector_id=42,
                keyframe_url="CAM001/picture/k.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=171,
                ai_id="ai-1",
            )
            await feature_buffer.sbs.flush()

        asyncio.run(run())

        self.assertEqual(self.sbs_db.calls[0]["keyframeId"], [42])
        self.assertEqual(self.sbs_db.calls[0]["bodyCompleteness"], [BODY_FULL])
        self.assertEqual(len(self.sbs_db.calls[0]["embedding"][0]), 2048)
        record = feature_buffer.vector_record_buffer.get_nowait()
        self.assertEqual(record["vectorId"], 99)
        self.assertIs(record["isPanoKeyframe"], True)
        self.assertNotIn("isPanoKeyframe", self.sbs_db.calls[0])
        self.assertEqual(record["keyframeId"], uid)
        self.assertEqual(record["pictureVectorId"], 42)
        self.assertEqual(record["targetType"], "person_sbs")
        self.assertEqual(record["vectorDbName"], "alogSBS")
        self.assertEqual(record["modelName"], "Person ReID Vector | FastReID SBS R50-IBN (DukeMTMC)")

    def test_person_gallery_buffers_accept_half_upper_and_reject_half_lower(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)

        async def add_sbs():
            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-1",
                re_id="reid-1",
                body_completeness=BODY_HALF_UPPER,
                embedding=[0.1] * 2048,
                device_id="CAM001",
                keyframe_id="uid",
                picture_vector_id=42,
                keyframe_url="crop.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=171,
                ai_id="ai-1",
            )

        asyncio.run(add_sbs())

        async def add_non_enrollable_sbs():
            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-1",
                re_id="reid-1",
                body_completeness=BODY_HALF_LOWER,
                embedding=[0.1] * 2048,
                device_id="CAM001",
                keyframe_id="uid-2",
                picture_vector_id=43,
                keyframe_url="crop-2.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=172,
                ai_id="ai-1",
            )

        with self.assertRaisesRegex(ValueError, "gallery accepts only"):
            asyncio.run(add_non_enrollable_sbs())

    def test_person_clip_flush_is_independent_and_does_not_enter_pending(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)

        async def run():
            await feature_buffer.tbps.add_feature(
                face_id="face-1",
                re_id="reid-1",
                embedding=[0.1] * 512,
                is_pano_keyframe=True,
                device_id="CAM001",
                keyframe_id="uid",
                picture_vector_id=42,
                keyframe_url="CAM001/picture/k.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=171,
                ai_id="ai-1",
            )
            await feature_buffer.tbps.flush()

        asyncio.run(run())

        self.assertEqual(len(self.person_db.calls), 1)
        self.assertNotIn("bodyCompleteness", self.person_db.calls[0])
        self.assertEqual(len(self.sbs_db.calls), 0)
        self.assertEqual(feature_buffer.sbs.pending, {})
        record = feature_buffer.vector_record_buffer.get_nowait()
        self.assertEqual(record["targetType"], "person")
        self.assertEqual(record["vectorDbName"], "alogTBPS")
        self.assertEqual(record["vectorDim"], 512)
        self.assertIs(record["isPanoKeyframe"], True)
        self.assertNotIn("isPanoKeyframe", self.person_db.calls[0])
        self.assertFalse(hasattr(feature_buffer, "add_person_feature"))
        self.assertFalse(hasattr(feature_buffer, "add_person_clip_feature"))
        self.assertFalse(hasattr(feature_buffer, "pending_persons"))

    def test_split_person_buffers_retry_independently_after_write_failure(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.sbs.vector_db = FailingVectorDb("alogSBS")
        feature_buffer.tbps.vector_db = FailingVectorDb("alogTBPS")

        async def run():
            common = {
                "face_id": "face-1",
                "re_id": "reid-1",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "device_id": "CAM001",
                "keyframe_id": "uid",
                "picture_vector_id": 42,
                "keyframe_url": "crop.jpg",
                "person_rect": {"x": 1, "y": 2, "w": 3, "h": 4},
                "timestamp": int(time.time() * 1000),
                "ai_id": "ai-1",
            }
            await enqueue_formal_sbs(feature_buffer.sbs,
                body_completeness=BODY_FULL,
                **common,
            )
            await feature_buffer.tbps.add_feature(**common)
            await feature_buffer.sbs.flush()
            await feature_buffer.tbps.flush()

        asyncio.run(run())

        self.assertEqual(feature_buffer.sbs.queue.qsize(), 1)
        self.assertEqual(feature_buffer.tbps.queue.qsize(), 1)
        self.assertEqual(len(feature_buffer.sbs.pending), 1)
        self.assertFalse(hasattr(feature_buffer.tbps, "pending"))

    def test_new_person_recognition_waits_for_sbs_insert_retry(self):
        from core.person_keyframe_processor import PersonKeyframeProcessor
        from core.person_pipeline.models import FaceMatchOutcome
        from core.person_pipeline.pipeline import PersonDetectionPipeline
        from utils.keyframe.detection_helpers import KeyframeContext
        from utils.reid.person_identity import IdentityRequest, PersonIdentityService
        from utils.reid.reid_models import PersonReidSearchResult

        class RecordingKafkaProducer:
            def __init__(self):
                self.records = []

            async def send_and_wait(self, topic, value):
                self.records.append((topic, json.loads(value.decode("utf-8"))))

        async def search_miss(*_args, **_kwargs):
            return PersonReidSearchResult(reid=None, score=0.25)

        class RetryingSbsVectorDb(FakeVectorDb):
            async def aapply_prototype_updates(self, batch):
                self.calls.append(batch)
                return [
                    {
                        "item": item,
                        "action": "retry_error",
                        "reason": "storage_error",
                        "retry": True,
                        "persisted": False,
                        "insert_result": None,
                    }
                    for item in batch
                ]

        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=1)
        feature_buffer.sbs.vector_db = RetryingSbsVectorDb("alogSBS")
        kafka_producer = RecordingKafkaProducer()
        feature_buffer._kafka_producer = kafka_producer

        person_processor = object.__new__(PersonKeyframeProcessor)
        person_processor.feature_buffer = feature_buffer
        person_processor.records = feature_buffer.records
        person_processor.sbs_features = feature_buffer.sbs
        person_pipeline = object.__new__(PersonDetectionPipeline)
        person_pipeline.recognition_emitter = (
            person_processor.queue_person_recognition_record
        )

        ctx = KeyframeContext(
            sn="CAM001",
            timestamp=171,
            object_name="CAM001/picture/k.jpg",
            keyframe_id="uid",
        )
        person_rect = {"x": 1, "y": 2, "w": 30, "h": 60}
        prepared = types.SimpleNamespace(
            observation=types.SimpleNamespace(
                detect_time=7,
                people_rect=person_rect,
                ai_id="ai-1",
                person_orientation="front",
            ),
            quality_decision=PersonQualityDecision(
                person_quality_score=8.0,
                person_quality_level=2,
                person_quality_action="build_gallery",
                mask_score=8.0,
                crop_score=8.0,
                quality_reasons=[],
            ),
            crop_path="CAM001/person_crop/p.jpg",
        )
        embeddings = types.SimpleNamespace(clip_failure="")
        face = FaceMatchOutcome("face-1", "face matched", "front")

        async def run():
            identity_service = PersonIdentityService(
                identity_lock=asyncio.Lock(),
                search_persons=search_miss,
                sbs_buffer=feature_buffer.sbs,
                logger=DummyLogger(),
                id_factory=lambda: "new-reid",
            )
            identity = await identity_service.assign(
                IdentityRequest(
                    embedding=np.ones(2048, dtype=np.float32),
                    observed_face_id="face-1",
                    person_quality_level=2,
                    person_quality_score=8.0,
                    body_completeness=BODY_FULL,
                    device_id="CAM001",
                    timestamp=171,
                    keyframe_id="uid",
                    picture_vector_id=42,
                    keyframe_url="CAM001/picture/k.jpg",
                    crop_url="CAM001/person_crop/p.jpg",
                    person_rect=person_rect,
                    ai_id="ai-1",
                    segment_index=0,
                ),
                register_assignment=lambda _reid: None,
            )
            await person_pipeline._emit_recognition(
                ctx,
                "uid",
                42,
                "CAM001/picture/k.jpg",
                prepared,
                embeddings,
                face,
                identity,
            )

            self.assertTrue(feature_buffer.person_recognition_record_buffer.empty())
            await feature_buffer.flush_all()
            self.assertEqual(
                [
                    record
                    for topic, record in kafka_producer.records
                    if topic == "person-recognition"
                ],
                [],
            )
            self.assertEqual(feature_buffer.sbs.queue.qsize(), 1)
            self.assertEqual(len(feature_buffer.sbs.pending), 1)

            feature_buffer.sbs.vector_db = FakeVectorDb(
                "alogSBS",
                primary_keys=[99],
            )
            await feature_buffer.flush_all()
            await feature_buffer.flush_all()
            return identity

        identity = asyncio.run(run())

        recognition_records = [
            record
            for topic, record in kafka_producer.records
            if topic == "person-recognition"
        ]
        self.assertEqual(len(recognition_records), 1)
        self.assertEqual(recognition_records[0]["reid"], identity.reid)
        self.assertEqual(recognition_records[0]["reidStatus"], 1)
        self.assertEqual(recognition_records[0]["reidDecision"], "CREATE_NEW")
        self.assertIs(recognition_records[0]["isNewReid"], True)
        self.assertEqual(feature_buffer.sbs.pending, {})
        self.assertTrue(feature_buffer.sbs.queue.empty())

    def test_new_person_recognition_releases_if_sbs_insert_finishes_first(self):
        class RecordingKafkaProducer:
            def __init__(self):
                self.records = []

            async def send_and_wait(self, topic, value):
                self.records.append((topic, json.loads(value.decode("utf-8"))))

        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=1)
        kafka_producer = RecordingKafkaProducer()
        feature_buffer._kafka_producer = kafka_producer
        recognition = {
            "reid": "new-reid-early",
            "reidStatus": 1,
            "reidDecision": "CREATE_NEW",
            "isNewReid": True,
        }

        async def run():
            enrollment = await feature_buffer.sbs.offer_feature(
                face_id="face-1",
                observed_face_id="face-1",
                canonical_face_id="face-1",
                re_id="new-reid-early",
                identity_state="ACTIVE",
                body_completeness=BODY_FULL,
                embedding=np.ones(2048, dtype=np.float32),
                device_id="CAM001",
                keyframe_id="uid-early",
                picture_vector_id=43,
                keyframe_url="CAM001/picture/k2.jpg",
                crop_url="CAM001/person_crop/p2.jpg",
                person_rect={"x": 1, "y": 2, "w": 30, "h": 60},
                timestamp=172,
                ai_id="ai-2",
                quality_score=8.0,
                defer_new_reid_recognition=True,
            )
            self.assertTrue(feature_buffer.sbs.queue.empty())
            self.assertEqual(feature_buffer.sbs.pending, {})
            self.assertTrue(feature_buffer.person_recognition_record_buffer.empty())

            await feature_buffer.sbs.attach_new_reid_recognition(
                enrollment.recognition_release,
                recognition,
                feature_buffer.add_person_recognition_record,
            )
            await feature_buffer.sbs.attach_new_reid_recognition(
                enrollment.recognition_release,
                recognition,
                feature_buffer.add_person_recognition_record,
            )
            await feature_buffer.flush_all()

        asyncio.run(run())

        recognition_records = [
            record
            for topic, record in kafka_producer.records
            if topic == "person-recognition"
        ]
        self.assertEqual(recognition_records, [recognition])

    def test_new_person_release_is_once_under_attach_persist_race(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        recognition = {
            "reid": "new-reid-race",
            "reidDecision": "CREATE_NEW",
        }
        ready_records = []

        async def capture_ready_record(record):
            await asyncio.sleep(0)
            ready_records.append(record)

        async def run():
            enrollment = await feature_buffer.sbs.offer_feature(
                face_id="face-race",
                observed_face_id="face-race",
                canonical_face_id="face-race",
                re_id="new-reid-race",
                identity_state="ACTIVE",
                body_completeness=BODY_FULL,
                embedding=np.ones(2048, dtype=np.float32),
                device_id="CAM001",
                keyframe_id="uid-race",
                picture_vector_id=46,
                keyframe_url="CAM001/picture/race.jpg",
                crop_url="CAM001/person_crop/race.jpg",
                person_rect={"x": 1, "y": 2, "w": 30, "h": 60},
                timestamp=173,
                ai_id="ai-race",
                quality_score=8.0,
                defer_new_reid_recognition=True,
            )
            pending_item = (await feature_buffer.sbs._pending_snapshot())[0]
            await asyncio.gather(
                *(
                    feature_buffer.sbs.attach_new_reid_recognition(
                        enrollment.recognition_release,
                        recognition,
                        capture_ready_record,
                    )
                    for _ in range(4)
                ),
                *(
                    feature_buffer.sbs._mark_new_reid_persisted(pending_item)
                    for _ in range(4)
                ),
            )

        asyncio.run(run())
        self.assertEqual(ready_records, [recognition])

    def test_new_person_recognition_without_sbs_release_fails_closed(self):
        from core.person_keyframe_processor import PersonKeyframeProcessor
        from utils.keyframe.detection_helpers import KeyframeContext

        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        processor = object.__new__(PersonKeyframeProcessor)
        processor.feature_buffer = feature_buffer
        ctx = KeyframeContext(
            sn="CAM001",
            timestamp=171,
            object_name="CAM001/picture/k.jpg",
            keyframe_id="uid",
        )

        async def run():
            with self.assertRaisesRegex(RuntimeError, "SBS persistence release"):
                await processor.queue_person_recognition_record(
                    ctx,
                    "uid",
                    42,
                    "CAM001/picture/k.jpg",
                    "CAM001/person_crop/p.jpg",
                    "face-1",
                    "new-reid",
                    0.0,
                    7,
                    {"x": 1, "y": 2, "w": 30, "h": 60},
                    "created",
                    1,
                    "ai-1",
                    "",
                    quality_decision=PersonQualityDecision(
                        person_quality_score=8.0,
                        person_quality_level=2,
                        person_quality_action="build_gallery",
                        mask_score=8.0,
                        crop_score=8.0,
                        quality_reasons=[],
                    ),
                    is_new_reid=True,
                    reid_decision="CREATE_NEW",
                )

        asyncio.run(run())
        self.assertTrue(feature_buffer.person_recognition_record_buffer.empty())

    def test_vector_notification_failure_does_not_retract_persisted_new_reid(self):
        class IdempotentSbsVectorDb:
            collection_name = "alogSBS"

            def __init__(self):
                self.persisted_ids = set()

            async def aapply_prototype_updates(self, batch):
                outcomes = []
                for item in batch:
                    prototype_id = item["prototypeId"]
                    already_persisted = prototype_id in self.persisted_ids
                    self.persisted_ids.add(prototype_id)
                    outcomes.append(
                        {
                            "item": item,
                            "action": (
                                "already_persisted"
                                if already_persisted
                                else "insert"
                            ),
                            "reason": "test_idempotency",
                            "retry": False,
                            "persisted": True,
                            "insert_result": {
                                "primary_keys": [prototype_id]
                            },
                        }
                    )
                return outcomes

        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.sbs.vector_db = IdempotentSbsVectorDb()
        notification_attempts = []
        successful_notifications = []
        ready_records = []

        async def notify_vectors(batch, result, target_type, collection_name):
            notification_attempts.append(batch[0]["prototypeId"])
            if len(notification_attempts) == 1:
                raise RuntimeError("vector notification failed")
            successful_notifications.append(
                (batch, result, target_type, collection_name)
            )

        async def capture_ready_record(record):
            ready_records.append(record)

        feature_buffer.sbs.on_vectors_inserted = notify_vectors

        async def run():
            enrollment = await feature_buffer.sbs.offer_feature(
                face_id="face-notify",
                observed_face_id="face-notify",
                canonical_face_id="face-notify",
                re_id="reid-notify",
                identity_state="ACTIVE",
                body_completeness=BODY_FULL,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="uid-notify",
                picture_vector_id=45,
                keyframe_url="CAM001/picture/notify.jpg",
                crop_url="CAM001/person_crop/notify.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=int(time.time() * 1000),
                ai_id="ai-notify",
                quality_score=8.0,
                defer_new_reid_recognition=True,
            )
            await feature_buffer.sbs.attach_new_reid_recognition(
                enrollment.recognition_release,
                {"reid": "reid-notify", "reidDecision": "CREATE_NEW"},
                capture_ready_record,
            )

            await feature_buffer.sbs.flush()
            self.assertEqual(
                ready_records,
                [{"reid": "reid-notify", "reidDecision": "CREATE_NEW"}],
            )
            self.assertEqual(feature_buffer.sbs.queue.qsize(), 1)
            self.assertEqual(len(feature_buffer.sbs.pending), 1)

            await feature_buffer.sbs.flush()

        asyncio.run(run())
        self.assertEqual(len(notification_attempts), 2)
        self.assertEqual(len(successful_notifications), 1)
        self.assertEqual(
            ready_records,
            [{"reid": "reid-notify", "reidDecision": "CREATE_NEW"}],
        )
        self.assertTrue(feature_buffer.sbs.queue.empty())
        self.assertEqual(feature_buffer.sbs.pending, {})

    def test_sibling_prototype_can_release_retried_new_reid_once(self):
        class ReorderedSbsVectorDb:
            collection_name = "alogSBS"

            def __init__(self):
                self.call_count = 0

            async def aapply_prototype_updates(self, batch):
                self.call_count += 1
                if self.call_count == 1:
                    return [
                        {
                            "item": batch[0],
                            "action": "retry_error",
                            "reason": "storage_error",
                            "retry": True,
                            "persisted": False,
                            "insert_result": None,
                        },
                        {
                            "item": batch[1],
                            "action": "insert",
                            "reason": "first_prototype",
                            "retry": False,
                            "persisted": True,
                            "insert_result": {
                                "primary_keys": [batch[1]["prototypeId"]]
                            },
                        },
                    ]
                return [
                    {
                        "item": batch[0],
                        "action": "skip_duplicate",
                        "reason": "prototype_is_redundant",
                        "retry": False,
                        "persisted": True,
                        "insert_result": None,
                    }
                ]

        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.sbs.vector_db = ReorderedSbsVectorDb()
        ready_records = []

        async def capture_ready_record(record):
            ready_records.append(record)

        async def offer(*, keyframe_id, picture_vector_id, defer):
            return await feature_buffer.sbs.offer_feature(
                face_id="face-sibling",
                observed_face_id="face-sibling",
                canonical_face_id="face-sibling",
                re_id="reid-sibling",
                identity_state="ACTIVE",
                body_completeness=BODY_FULL,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id=keyframe_id,
                picture_vector_id=picture_vector_id,
                keyframe_url=f"CAM001/picture/{keyframe_id}.jpg",
                crop_url=f"CAM001/person_crop/{keyframe_id}.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=int(time.time() * 1000),
                ai_id=keyframe_id,
                quality_score=8.0,
                defer_new_reid_recognition=defer,
            )

        async def run():
            enrollment = await offer(
                keyframe_id="initial",
                picture_vector_id=47,
                defer=True,
            )
            await offer(
                keyframe_id="sibling",
                picture_vector_id=48,
                defer=False,
            )
            await feature_buffer.sbs.attach_new_reid_recognition(
                enrollment.recognition_release,
                {"reid": "reid-sibling", "reidDecision": "CREATE_NEW"},
                capture_ready_record,
            )

            await feature_buffer.sbs.flush()
            self.assertEqual(ready_records, [])
            self.assertEqual(feature_buffer.sbs.queue.qsize(), 1)

            await feature_buffer.sbs.flush()
            await feature_buffer.sbs.flush()

        asyncio.run(run())
        self.assertEqual(
            ready_records,
            [{"reid": "reid-sibling", "reidDecision": "CREATE_NEW"}],
        )
        self.assertTrue(feature_buffer.sbs.queue.empty())
        self.assertEqual(feature_buffer.sbs.pending, {})

    def test_new_face_recognition_is_released_once_after_insert_retry_succeeds(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.face_features.vector_db = FailingVectorDb("alogFace")

        async def run():
            try:
                await feature_buffer.add_face_feature(
                    face_id="new-face",
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                    device_id="CAM001",
                    keyframe_id="uid",
                    picture_vector_id=42,
                    keyframe_url="CAM001/picture/k.jpg",
                    face_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                    face_key_points=[],
                    face_crop_path="CAM001/face_crop/f.jpg",
                    timestamp=171,
                    ai_id="ai-1",
                    quality_score=8.0,
                    new_face_recognition_record={
                        "aiFaceId": "new-face",
                        "faceStatus": 1,
                    },
                )
            except TypeError as error:
                self.fail(f"face enrollment does not accept a success record: {error}")

            self.assertTrue(feature_buffer.face_recognition_record_buffer.empty())
            await feature_buffer._flush_faces()
            self.assertTrue(feature_buffer.face_recognition_record_buffer.empty())
            self.assertEqual(feature_buffer.face_buffer.qsize(), 1)
            self.assertEqual(len(feature_buffer.pending_faces), 1)

            feature_buffer.face_features.vector_db = FakeVectorDb(
                "alogFace",
                primary_keys=[77],
            )
            await feature_buffer._flush_faces()

        asyncio.run(run())

        self.assertTrue(feature_buffer.face_buffer.empty())
        self.assertEqual(feature_buffer.pending_faces, {})
        self.assertEqual(feature_buffer.face_recognition_record_buffer.qsize(), 1)
        recognition = feature_buffer.face_recognition_record_buffer.get_nowait()
        self.assertEqual(recognition["aiFaceId"], "new-face")
        self.assertEqual(recognition["faceStatus"], 1)

    def test_face_processor_does_not_publish_new_face_before_real_buffer_insert(self):
        from core.face_keyframe_processor import FaceKeyframeProcessor
        from core.infer.results import FaceQualityResult, FeatureResult
        from utils.keyframe.detection_helpers import KeyframeContext

        class FaceTriton:
            async def extract_face_detections(self, _image):
                return [{"xyxy": [50, 40, 90, 80], "conf": 0.9}]

            async def extract_face_keypoints(self, _crop, _bbox):
                return np.array(
                    [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                    dtype=np.float32,
                )

            async def extract_face_quality(self, _crop, _landmarks):
                return FaceQualityResult(8.0)

            async def extract_face_features(self, _crop, _landmarks):
                return FeatureResult(np.ones(512, dtype=np.float32))

        async def search_miss(*_args, **_kwargs):
            return None, 0.25, ""

        class RecordingKafkaProducer:
            def __init__(self):
                self.records = []

            async def send_and_wait(self, topic, value):
                self.records.append((topic, json.loads(value.decode("utf-8"))))

        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.face_features.vector_db = FailingVectorDb("alogFace")
        kafka_producer = RecordingKafkaProducer()
        feature_buffer._kafka_producer = kafka_producer
        processor = FaceKeyframeProcessor(
            triton_client=FaceTriton(),
            feature_buffer=feature_buffer,
            face_identity_lock=asyncio.Lock(),
            search_faces_with_pending=search_miss,
            logger=DummyLogger(),
        )

        async def run():
            with (
                patch(
                    "core.face_keyframe_processor.upload_crop_bytes",
                    new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
                ),
                patch(
                    "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                    return_value=(True, ""),
                ),
            ):
                observations = await processor.process_detection(
                    KeyframeContext(
                        sn="CAM001",
                        timestamp=171,
                        object_name="CAM001/picture/k.jpg",
                        keyframe_id="uid",
                    ),
                    np.zeros((120, 120, 3), dtype=np.uint8),
                    "uid",
                    42,
                    "CAM001/picture/k.jpg",
                    object(),
                )

            self.assertTrue(feature_buffer.face_recognition_record_buffer.empty())
            await feature_buffer.flush_all()
            self.assertTrue(feature_buffer.face_recognition_record_buffer.empty())
            self.assertEqual(
                [
                    record
                    for topic, record in kafka_producer.records
                    if topic == "face-recognition"
                ],
                [],
            )

            feature_buffer.face_features.vector_db = FakeVectorDb(
                "alogFace",
                primary_keys=[77],
            )
            await feature_buffer.flush_all()
            return observations[0].assigned_face_id

        new_face_id = asyncio.run(run())

        self.assertEqual(feature_buffer.pending_faces, {})
        self.assertTrue(feature_buffer.face_recognition_record_buffer.empty())
        recognition_records = [
            record
            for topic, record in kafka_producer.records
            if topic == "face-recognition"
        ]
        self.assertEqual(len(recognition_records), 1)
        recognition = recognition_records[0]
        self.assertEqual(recognition["aiFaceId"], new_face_id)
        self.assertNotEqual(recognition["aiFaceId"], "0")
        self.assertEqual(recognition["faceStatus"], 1)

    def test_face_flush_expands_iterable_milvus_ids(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.face_features.vector_db = FakeVectorDb(
            "alogFace",
            primary_keys=IterablePrimaryKeys([77, 78]),
            result_key="ids",
        )

        async def run():
            for index in range(2):
                await feature_buffer.add_face_feature(
                    face_id=f"face-{index}",
                    embedding=[0.1, 0.2],
                    device_id="CAM001",
                    keyframe_id=f"uid-{index}",
                    picture_vector_id=42 + index,
                    keyframe_url=f"CAM001/picture/k{index}.jpg",
                    face_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                    face_key_points=[],
                    face_crop_path=f"CAM001/face_crop/f{index}.jpg",
                    timestamp=171,
                    ai_id="ai-1",
                    quality_score=8.0,
                )
            await feature_buffer._flush_faces()

        asyncio.run(run())

        self.assertEqual(
            [
                feature_buffer.vector_record_buffer.get_nowait()["vectorId"]
                for _ in range(2)
            ],
            [77, 78],
        )

    def test_person_flush_expands_iterable_milvus_ids(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.sbs.vector_db = FakeVectorDb(
            "alogSBS",
            primary_keys=IterablePrimaryKeys([88, 89]),
            result_key="ids",
        )

        async def run():
            for index in range(2):
                await enqueue_formal_sbs(feature_buffer.sbs,
                    face_id=f"face-{index}",
                    re_id=f"reid-{index}",
                    body_completeness=BODY_FULL,
                    embedding=[0.1, 0.2],
                    device_id="CAM001",
                    keyframe_id=f"uid-{index}",
                    picture_vector_id=42 + index,
                    keyframe_url=f"CAM001/picture/k{index}.jpg",
                    person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                    timestamp=171,
                    ai_id="ai-1",
                )
            await feature_buffer.sbs.flush()

        asyncio.run(run())

        self.assertEqual(
            [
                feature_buffer.vector_record_buffer.get_nowait()["vectorId"]
                for _ in range(2)
            ],
            [88, 89],
        )

    def test_pending_gray_face_needs_two_distinct_samples(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        feature_buffer.pending_faces = {
            1: {"faceId": "face-1", "sampleId": "one", "embedding": [0.48, 0.0],
                "deviceId": "CAM001", "faceCropPath": "one.jpg"},
        }

        async def search():
            return await feature_buffer.asearch_pending_faces_with_hit_info(
                np.array([1.0, 0.0], dtype=np.float32), threshold=0.45,
                device_ids=["CAM001"],
            )

        self.assertIsNone(asyncio.run(search())[0])
        feature_buffer.pending_faces[2] = {
            "faceId": "face-1", "sampleId": "two", "embedding": [0.46, 0.0],
            "deviceId": "CAM002", "faceCropPath": "two.jpg",
        }
        self.assertIsNone(asyncio.run(search())[0])
        feature_buffer.pending_faces[2]["deviceId"] = "CAM001"
        self.assertEqual(asyncio.run(search())[0], "face-1")
        feature_buffer.pending_faces[2]["sampleId"] = "one"
        self.assertIsNone(asyncio.run(search())[0])

    def test_face_pending_search_survives_flush_until_insert_success(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        started = asyncio.Event()
        proceed = asyncio.Event()
        blocking_db = BlockingVectorDb(
            "alogFace",
            started=started,
            proceed=proceed,
            primary_keys=[77],
        )
        feature_buffer.face_features.vector_db = blocking_db

        async def run():
            await feature_buffer.add_face_feature(
                face_id="face-1",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="uid",
                picture_vector_id=42,
                keyframe_url="CAM001/picture/k.jpg",
                face_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                face_key_points=[],
                face_crop_path="CAM001/face_crop/f.jpg",
                timestamp=171,
                ai_id="ai-1",
                quality_score=8.0,
            )

            self.assertEqual(
                await feature_buffer.asearch_pending_faces_with_hit_info(
                    np.array([1.0, 0.0], dtype=np.float32),
                    threshold=0.5,
                    device_ids=["CAM001"],
                ),
                ("face-1", 1.0, "face-1"),
            )

            flush_task = asyncio.create_task(feature_buffer._flush_faces())
            await started.wait()
            self.assertTrue(feature_buffer.face_buffer.empty())
            self.assertEqual(
                await feature_buffer.asearch_pending_faces_with_hit_info(
                    np.array([1.0, 0.0], dtype=np.float32),
                    threshold=0.5,
                    device_ids=["CAM001"],
                ),
                ("face-1", 1.0, "face-1"),
            )

            proceed.set()
            await flush_task
            self.assertEqual(
                await feature_buffer.asearch_pending_faces_with_hit_info(
                    np.array([1.0, 0.0], dtype=np.float32),
                    threshold=0.5,
                    device_ids=["CAM001"],
                ),
                (None, 0.0, ""),
            )

        asyncio.run(run())

    def test_person_pending_search_survives_flush_until_insert_success(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        started = asyncio.Event()
        proceed = asyncio.Event()
        blocking_db = BlockingVectorDb(
            "alogSBS",
            started=started,
            proceed=proceed,
            primary_keys=[88],
        )
        feature_buffer.sbs.vector_db = blocking_db
        now_ms = int(time.time() * 1000)

        async def run():
            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-1",
                re_id="reid-1",
                body_completeness=BODY_FULL,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="uid",
                picture_vector_id=42,
                keyframe_url="CAM001/picture/k.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=now_ms,
                ai_id="ai-1",
            )

            self.assertEqual(
                await resolve_buffer_candidates(feature_buffer.sbs,
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                    face_id="face-1",
                ),
                ("reid-1", 1.0, "reid-1"),
            )

            flush_task = asyncio.create_task(feature_buffer.sbs.flush())
            await started.wait()
            self.assertTrue(feature_buffer.sbs.queue.empty())
            self.assertEqual(
                await resolve_buffer_candidates(feature_buffer.sbs,
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                    face_id="face-1",
                ),
                ("reid-1", 1.0, "reid-1"),
            )

            proceed.set()
            await flush_task
            self.assertEqual(
                await resolve_buffer_candidates(feature_buffer.sbs,
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                    face_id="face-1",
                ),
                (None, 0.0, ""),
            )

        asyncio.run(run())

    def test_cancelled_person_flush_requeues_new_reid_until_retry_succeeds(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        started = asyncio.Event()
        proceed = asyncio.Event()
        feature_buffer.sbs.vector_db = BlockingVectorDb(
            "alogSBS",
            started=started,
            proceed=proceed,
            primary_keys=[88],
        )
        ready_records = []

        async def capture_ready_record(record):
            ready_records.append(record)

        async def run():
            enrollment = await feature_buffer.sbs.offer_feature(
                face_id="face-cancelled",
                observed_face_id="face-cancelled",
                canonical_face_id="face-cancelled",
                re_id="reid-cancelled",
                identity_state="ACTIVE",
                body_completeness=BODY_FULL,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="uid-cancelled",
                picture_vector_id=44,
                keyframe_url="CAM001/picture/cancelled.jpg",
                crop_url="CAM001/person_crop/cancelled.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=int(time.time() * 1000),
                ai_id="ai-cancelled",
                quality_score=8.0,
                defer_new_reid_recognition=True,
            )
            await feature_buffer.sbs.attach_new_reid_recognition(
                enrollment.recognition_release,
                {"reid": "reid-cancelled", "reidDecision": "CREATE_NEW"},
                capture_ready_record,
            )

            flush_task = asyncio.create_task(feature_buffer.sbs.flush())
            await started.wait()
            flush_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await flush_task

            self.assertEqual(feature_buffer.sbs.queue.qsize(), 1)
            self.assertEqual(len(feature_buffer.sbs.pending), 1)
            self.assertEqual(ready_records, [])

            feature_buffer.sbs.vector_db = FakeVectorDb(
                "alogSBS",
                primary_keys=[88],
            )
            await feature_buffer.sbs.flush()

            self.assertTrue(feature_buffer.sbs.queue.empty())
            self.assertEqual(feature_buffer.sbs.pending, {})
            self.assertEqual(
                ready_records,
                [{"reid": "reid-cancelled", "reidDecision": "CREATE_NEW"}],
            )

        asyncio.run(run())

    def test_pending_person_search_known_face_never_uses_other_face(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        now_ms = int(time.time() * 1000)

        async def run():
            for index in range(3):
                await enqueue_formal_sbs(feature_buffer.sbs,
                    face_id="face-low",
                    re_id="many-low",
                    body_completeness=BODY_FULL,
                    embedding=np.array([0.75, 0.0], dtype=np.float32),
                    device_id="CAM001",
                    keyframe_id=f"low-{index}",
                    picture_vector_id=100 + index,
                    keyframe_url=f"CAM001/picture/low-{index}.jpg",
                    person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                    timestamp=now_ms,
                    ai_id="ai-low",
                )

            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-high",
                re_id="few-high",
                body_completeness=BODY_FULL,
                embedding=np.array([0.95, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="high",
                picture_vector_id=200,
                keyframe_url="CAM001/picture/high.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=now_ms,
                ai_id="ai-high",
            )

            result = await resolve_buffer_candidates(feature_buffer.sbs,
                np.array([1.0, 0.0], dtype=np.float32),
                query_body_completeness=BODY_FULL,
                device_ids=["CAM001"],
                face_id="face-low",
            )

            self.assertEqual(result[0], "many-low")
            self.assertAlmostEqual(result[1], 0.75, places=6)
            self.assertEqual(result[2], "many-low")

        asyncio.run(run())

    def test_pending_person_candidates_include_buffer_source(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        now_ms = int(time.time() * 1000)

        async def run():
            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-1",
                re_id="buffer-reid",
                body_completeness=BODY_FULL,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="buffer",
                picture_vector_id=300,
                keyframe_url="CAM001/picture/buffer.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=now_ms,
                ai_id="ai-buffer",
            )
            await enqueue_formal_sbs(feature_buffer.sbs,
                face_id="face-above",
                re_id="above-threshold",
                body_completeness=BODY_FULL,
                embedding=np.array([0.59, 0.0], dtype=np.float32),
                device_id="CAM001",
                keyframe_id="above",
                picture_vector_id=301,
                keyframe_url="CAM001/picture/above.jpg",
                person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                timestamp=now_ms,
                ai_id="ai-above",
            )

            candidates, highest_score, highest_reid, highest_source = (
                await feature_buffer.sbs.asearch_candidates_with_hit_info(
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                )
            )

            self.assertEqual(
                candidates,
                [
                    ReidCandidate(
                        reid="buffer-reid",
                        score=1.0,
                        source=REID_SOURCE_BUFFER,
                        threshold=0.56,
                        body_completeness=BODY_FULL,
                        face_id="face-1",
                        device_id="CAM001",
                        prototype_id=feature_buffer.sbs.pending[1]["prototypeId"],
                    ),
                    ReidCandidate(
                        reid="above-threshold",
                        score=float(np.float32(0.59)),
                        source=REID_SOURCE_BUFFER,
                        threshold=0.56,
                        body_completeness=BODY_FULL,
                        face_id="face-above",
                        device_id="CAM001",
                        prototype_id=feature_buffer.sbs.pending[2]["prototypeId"],
                    ),
                ],
            )
            self.assertEqual(highest_score, 1.0)
            self.assertEqual(highest_reid, "buffer-reid")
            self.assertEqual(highest_source, REID_SOURCE_BUFFER)

        asyncio.run(run())

    def test_pending_person_candidates_use_seven_day_query_window(self):
        feature_buffer = self.buffer_module.FeatureBuffer(max_batch_size=10)
        now_ms = int(time.time() * 1000)
        day_ms = 24 * 60 * 60 * 1000

        async def run():
            for reid, age_days in (("six-day-reid", 6), ("eight-day-reid", 8)):
                await enqueue_formal_sbs(feature_buffer.sbs,
                    face_id="face-1",
                    re_id=reid,
                    body_completeness=BODY_FULL,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                    device_id="CAM001",
                    keyframe_id=reid,
                    picture_vector_id=400 + age_days,
                    keyframe_url=f"CAM001/picture/{reid}.jpg",
                    person_rect={"x": 1, "y": 2, "w": 3, "h": 4},
                    timestamp=now_ms - age_days * day_ms,
                    ai_id=f"ai-{reid}",
                )

            candidates, highest_score, highest_reid, _ = (
                await feature_buffer.sbs.asearch_candidates_with_hit_info(
                    np.array([1.0, 0.0], dtype=np.float32),
                    query_body_completeness=BODY_FULL,
                    device_ids=["CAM001"],
                )
            )

            self.assertEqual(
                [candidate.reid for candidate in candidates],
                ["six-day-reid"],
            )
            self.assertEqual(highest_score, 1.0)
            self.assertEqual(highest_reid, "six-day-reid")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
