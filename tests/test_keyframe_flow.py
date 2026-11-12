import asyncio
import importlib
import inspect
import json
import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import numpy as np

from tests.image_execution_test_support import InlineImageExecutor

from core.algorithm_config import algorithm_config
from core.face_observation import FaceProcessingStage, FrameFaceObservation
from core.infer.results import FaceQualityResult, FeatureResult
from core.performance import RuntimePerformance
from utils.reid.person_body import BODY_FULL, BODY_THREE_QUARTER_UPPER
from utils.reid.reid_models import (
    REID_SOURCE_BUFFER,
    REID_SOURCE_MILVUS,
    ReidCandidate,
)
from utils.reid.reid_enrollment_state import (
    REID_ENROLLMENT_STATE_FORMAL_PENDING,
)
from utils.reid.reid_state import REID_STATE_ACTIVE
from utils.reid.person_face_recognition import select_owned_face_detection
from utils.reid.face_matching import FaceSampleMatch

_MISSING = object()


def normal_person_test_image(height=120, width=120):
    y, x = np.indices((height, width))
    gray = np.where((x // 2 + y // 2) % 2 == 0, 60, 180).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def moderately_underexposed_person_test_image(height=120, width=120):
    y, x = np.indices((height, width))
    gray = np.where((x // 3 + y // 3) % 5 == 0, 80, 0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def person_pose_result(rect):
    normalized = [
        (0.50, 0.10),
        (0.45, 0.08),
        (0.55, 0.08),
        (0.38, 0.12),
        (0.62, 0.12),
        (0.30, 0.25),
        (0.70, 0.25),
        (0.22, 0.40),
        (0.78, 0.40),
        (0.18, 0.55),
        (0.82, 0.55),
        (0.38, 0.55),
        (0.62, 0.55),
        (0.38, 0.75),
        (0.62, 0.75),
        (0.38, 0.92),
        (0.62, 0.92),
    ]
    return {
        "rect": dict(rect),
        "xyxy": [
            rect["x"],
            rect["y"],
            rect["x"] + rect["w"],
            rect["y"] + rect["h"],
        ],
        "conf": 0.9,
        "cls": 0,
        "keypoints": np.asarray(
            [
                [
                    rect["x"] + rect["w"] * x_ratio,
                    rect["y"] + rect["h"] * y_ratio,
                    0.9,
                ]
                for x_ratio, y_ratio in normalized
            ],
            dtype=np.float32,
        ),
    }


class DummyLogger:
    def __init__(self):
        self.info_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def debug(self, message):
        self.info(message)

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class OverflowingFloat:
    def __float__(self):
        raise OverflowError("cannot convert test value")


class FakeBuffer:
    def __init__(self):
        self.picture_features = []
        self.face_features = []
        self.face_detection_records = []
        self.face_recognition_records = []
        self.person_detection_records = []
        self.person_features = []
        self.person_clip_features = []
        self.person_recognition_records = []
        self.pending_face_result = (None, 0.0, "")
        self.pending_face_samples = None
        self.pending_person_result = (None, 0.0, "")
        self.pending_face_search_calls = 0
        self.pending_person_search_calls = 0
        self.sbs = types.SimpleNamespace(
            offer_feature=self.offer_person_feature,
            attach_new_reid_recognition=self.attach_new_reid_recognition,
            asearch_candidates_with_hit_info=(
                self.asearch_pending_person_candidates_with_hit_info
            ),
        )
        self.tbps = types.SimpleNamespace(
            add_feature=self.add_person_clip_feature,
        )

    async def add_picture_feature(self, **kwargs):
        self.picture_features.append(kwargs)

    async def add_face_detection_record(self, record):
        self.face_detection_records.append(record)

    async def add_face_recognition_record(self, record):
        self.face_recognition_records.append(record)

    async def add_face_feature(self, **kwargs):
        new_face_recognition_record = kwargs.pop(
            "new_face_recognition_record",
            None,
        )
        self.face_features.append(kwargs)
        if new_face_recognition_record is not None:
            self.face_recognition_records.append(new_face_recognition_record)

    async def add_person_detection_record(self, record):
        self.person_detection_records.append(record)

    async def add_person_feature(self, **kwargs):
        self.person_features.append(kwargs)

    async def offer_person_feature(self, **kwargs):
        defer_recognition = kwargs.pop("defer_new_reid_recognition", False)
        self.person_features.append(kwargs)
        return types.SimpleNamespace(
            reason="test_enrollment",
            reid_state=REID_STATE_ACTIVE,
            enrollment_state=REID_ENROLLMENT_STATE_FORMAL_PENDING,
            recognition_release=(object() if defer_recognition else None),
        )

    async def attach_new_reid_recognition(self, _release, record, on_ready):
        await on_ready(record)

    async def add_person_clip_feature(self, **kwargs):
        self.person_clip_features.append(kwargs)

    async def add_person_recognition_record(self, record):
        self.person_recognition_records.append(record)

    async def asearch_pending_faces_with_hit_info(
        self, embedding, threshold=0.5, device_ids=None
    ):
        self.pending_face_search_calls += 1
        return self.pending_face_result

    async def asearch_pending_face_samples(self, embedding, device_ids=None):
        _, score, hit_id = await self.asearch_pending_faces_with_hit_info(
            embedding, device_ids=device_ids
        )
        if self.pending_face_samples is not None:
            return list(self.pending_face_samples)
        return [FaceSampleMatch(hit_id, score, "pending-one")] if hit_id else []

    async def asearch_pending_persons_with_hit_info(
        self, embedding, threshold=0.95, device_ids=None
    ):
        self.pending_person_search_calls += 1
        return self.pending_person_result

    async def asearch_pending_person_candidates_with_hit_info(
        self,
        embedding,
        query_body_completeness,
        device_ids=None,
        face_id=None,
        excluded_reids=None,
        query_timestamp=None,
    ):
        self.pending_person_search_calls += 1
        reid, score, hit_reid = self.pending_person_result
        candidates = []
        if reid is not None and score > 0.75:
            candidates.extend(
                ReidCandidate(
                    reid=reid,
                    score=score,
                    source=REID_SOURCE_BUFFER,
                    threshold=0.75,
                    body_completeness=BODY_FULL,
                    face_id=str(face_id or "face-buffer"),
                    device_id="CAM001",
                    prototype_id=index,
                )
                for index in (1, 2)
            )
        return candidates, score, hit_reid, REID_SOURCE_BUFFER if hit_reid else ""


class FakeTriton:
    def __init__(self):
        self.image_feature_calls = 0
        self.face_detection_calls = 0
        self.person_segment_calls = 0
        self.tbps_calls = 0
        self.sbs_calls = 0

    async def extract_image_features(self, image_bytes):
        self.image_feature_calls += 1
        return FeatureResult(np.ones(512, dtype=np.float32))

    async def extract_face_detections(self, image):
        self.face_detection_calls += 1
        return []

    async def extract_person_segments(self, image):
        self.person_segment_calls += 1
        tiny_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        tiny_mask[1:3, 1:3] = 1
        person_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        person_mask[10:90, 10:40] = 1
        return [
            {
                "rect": {"x": 1, "y": 1, "w": 2, "h": 2},
                "conf": 0.9,
                "maskPixelCount": 1000,
                "mask": tiny_mask,
            },
            {
                "rect": {"x": 10, "y": 10, "w": 30, "h": 80},
                "conf": 0.8,
                "maskPixelCount": 12000,
                "mask": person_mask,
            },
        ]

    async def extract_person_poses(self, image):
        return [
            person_pose_result({"x": 1, "y": 1, "w": 2, "h": 2}),
            person_pose_result({"x": 10, "y": 10, "w": 30, "h": 80}),
        ]

    async def extract_tbps_image_features(self, image_data):
        self.tbps_calls += 1
        return FeatureResult(np.ones(512, dtype=np.float32))

    async def extract_sbs_features(self, image_data):
        self.sbs_calls += 1
        return FeatureResult(np.ones(2048, dtype=np.float32))


class FakePersonDb:
    async def asearch_similar_person_candidates_with_hit_info(
        self,
        embedding,
        query_body_completeness,
        device_ids=None,
        face_id=None,
        excluded_reids=None,
        query_timestamp=None,
    ):
        return [], 0.0, "", ""


class FakeFaceDb:
    def __init__(self, face_id=None, score=0.25, hit_face_id="", samples=None):
        self.face_id = face_id
        self.score = score
        self.hit_face_id = hit_face_id
        self.search_calls = 0
        self.samples = samples

    async def asearch_similar_faces_with_hit_info(
        self, embedding, threshold=0.5, device_ids=None
    ):
        self.search_calls += 1
        return self.face_id, self.score, self.hit_face_id

    async def asearch_face_samples(self, embedding, device_ids=None):
        _, score, hit_id = await self.asearch_similar_faces_with_hit_info(
            embedding, device_ids=device_ids
        )
        if self.samples is not None:
            return list(self.samples)
        return [FaceSampleMatch(hit_id, score, "milvus-one")] if hit_id else []


class FailingFaceDb(FakeFaceDb):
    async def asearch_similar_faces_with_hit_info(
        self, embedding, threshold=0.5, device_ids=None
    ):
        self.search_calls += 1
        raise RuntimeError("face gallery unavailable")


class FakePersonReidDb:
    def __init__(self, reid=None, score=0.25, hit_reid=""):
        self.reid = reid
        self.score = score
        self.hit_reid = hit_reid
        self.search_calls = 0
        self.last_threshold = None
        self.last_embedding = None

    async def asearch_similar_person_candidates_with_hit_info(
        self,
        embedding,
        query_body_completeness,
        device_ids=None,
        face_id=None,
        excluded_reids=None,
        query_timestamp=None,
    ):
        self.search_calls += 1
        self.last_threshold = 0.75
        self.last_embedding = np.asarray(embedding)
        candidates = []
        if self.reid is not None and self.score > 0.75:
            candidates.extend(
                [
                    ReidCandidate(
                        reid=self.reid,
                        score=self.score,
                        source=REID_SOURCE_MILVUS,
                        threshold=0.75,
                        body_completeness=BODY_FULL,
                        device_id="CAM001",
                        prototype_id=index,
                    )
                    for index in (1, 2)
                ]
            )
        return (
            candidates,
            self.score,
            self.hit_reid,
            REID_SOURCE_MILVUS if self.hit_reid else "",
        )


class FakeOSS:
    async def upload_object(self, image_bytes, object_name):
        return object_name


class FailingOSS:
    async def upload_object(self, image_bytes, object_name):
        return None


class CapturingOSS:
    def __init__(self):
        self.uploads = []

    async def upload_object(self, image_bytes, object_name):
        self.uploads.append((bytes(image_bytes), object_name))
        return object_name


class FaceOnlyTriton:
    def __init__(self, quality_score):
        self.quality_score = quality_score
        self.feature_calls = 0

    async def extract_face_detections(self, image):
        return [{"xyxy": [30, 30, 90, 90], "conf": 0.75}]

    async def extract_face_keypoints(self, expanded_crop, bbox):
        return np.array(
            [[40, 40], [80, 40], [60, 60], [45, 80], [75, 80]],
            dtype=np.float32,
        )

    async def extract_face_quality(self, face_crop, landmarks):
        return FaceQualityResult(self.quality_score)

    async def extract_face_features(self, expanded_crop, landmarks):
        self.feature_calls += 1
        return FeatureResult(np.ones(512, dtype=np.float32))


class PersonOnlyTriton:
    def __init__(
        self,
        mask_pixel_count,
        *,
        sbs_error=None,
        sbs_result=None,
        clip_error=None,
        clip_result=None,
        person_rect=None,
        pose_keypoints=None,
    ):
        self.mask_pixel_count = mask_pixel_count
        self.sbs_error = sbs_error
        self.sbs_result = sbs_result
        self.clip_error = clip_error
        self.clip_result = clip_result
        self.person_rect = dict(
            person_rect or {"x": 10, "y": 10, "w": 30, "h": 80}
        )
        self.pose_keypoints = pose_keypoints
        self.tbps_calls = 0
        self.sbs_calls = 0
        self.tbps_inputs = []
        self.sbs_inputs = []

    async def extract_person_segments(self, image):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1 = max(0, int(self.person_rect["x"]))
        y1 = max(0, int(self.person_rect["y"]))
        x2 = min(image.shape[1], x1 + int(self.person_rect["w"]))
        y2 = min(image.shape[0], y1 + int(self.person_rect["h"]))
        mask[y1:y2, x1:x2] = 1
        return [
            {
                "rect": dict(self.person_rect),
                "conf": 0.8,
                "maskPixelCount": self.mask_pixel_count,
                "mask": mask,
            }
        ]

    async def extract_person_poses(self, image):
        result = person_pose_result(self.person_rect)
        if self.pose_keypoints is not None:
            result["keypoints"] = np.asarray(self.pose_keypoints).copy()
        return [result]

    async def extract_tbps_image_features(self, image_data):
        self.tbps_calls += 1
        self.tbps_inputs.append(image_data)
        if self.clip_error is not None:
            raise self.clip_error
        if self.clip_result is not None:
            return self.clip_result
        return FeatureResult(np.ones(512, dtype=np.float32))

    async def extract_sbs_features(self, image_data):
        self.sbs_calls += 1
        self.sbs_inputs.append(image_data)
        if self.sbs_error is not None:
            raise self.sbs_error
        if self.sbs_result is not None:
            return self.sbs_result
        return FeatureResult(np.ones(2048, dtype=np.float32))


class FacePersonReuseTriton(FakeTriton):
    def __init__(self, *, frame_has_face=True):
        super().__init__()
        self.frame_has_face = frame_has_face
        self.face_keypoint_calls = 0
        self.face_quality_calls = 0
        self.face_feature_calls = 0

    async def extract_face_detections(self, image):
        self.face_detection_calls += 1
        if self.face_detection_calls == 1 and not self.frame_has_face:
            return []
        return [{"xyxy": [45, 15, 65, 35], "conf": 0.95}]

    async def extract_face_keypoints(self, expanded_crop, bbox):
        self.face_keypoint_calls += 1
        return np.asarray(
            [[8, 8], [22, 8], [15, 15], [10, 22], [20, 22]],
            dtype=np.float32,
        )

    async def extract_face_quality(self, face_crop, landmarks):
        self.face_quality_calls += 1
        return FaceQualityResult(8.0)

    async def extract_face_features(self, expanded_crop, landmarks):
        self.face_feature_calls += 1
        return FeatureResult(np.ones(512, dtype=np.float32))

    async def extract_person_segments(self, image):
        self.person_segment_calls += 1
        person_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        person_mask[10:110, 10:100] = 1
        return [
            {
                "rect": {"x": 10, "y": 10, "w": 90, "h": 100},
                "conf": 0.9,
                "maskPixelCount": 12000,
                "mask": person_mask,
            }
        ]

    async def extract_person_poses(self, image):
        return [person_pose_result({"x": 10, "y": 10, "w": 90, "h": 100})]


class KeyframeFlowTests(unittest.TestCase):
    def setUp(self):
        self.module_names = [
            "main",
            "uvloop",
            "aiokafka",
            "aiokafka.structs",
            "cv2",
            "loguru",
            "sqlalchemy",
            "sqlalchemy.future",
            "core.config",
            "core.infer",
            "core.face_keyframe_processor",
            "core.person_keyframe_processor",
            "core.keyframe_processor",
            "db.zilliz",
            "db.pgsql.cam_device",
            "utils.buffers.buffer",
            "utils.storage.image_util",
            "utils.storage.oss_util",
            "utils.identifiers.id_generator",
            "utils.quality.face_quality",
            "utils.quality.face_orientation",
            "utils.keyframe.keyframe_flow_helpers",
            "utils.reid.person_face_recognition",
        ]
        self.original_modules = {
            name: sys.modules.get(name, _MISSING) for name in self.module_names
        }
        for name in self.module_names:
            sys.modules.pop(name, None)

        uvloop_module = types.ModuleType("uvloop")
        uvloop_module.install = lambda: None
        sys.modules["uvloop"] = uvloop_module

        aiokafka_module = types.ModuleType("aiokafka")
        aiokafka_module.AIOKafkaConsumer = object
        structs_module = types.ModuleType("aiokafka.structs")
        structs_module.ConsumerRecord = object
        sys.modules["aiokafka"] = aiokafka_module
        sys.modules["aiokafka.structs"] = structs_module

        loguru_module = types.ModuleType("loguru")
        self.logger = DummyLogger()
        loguru_module.logger = self.logger
        sys.modules["loguru"] = loguru_module

        sqlalchemy_module = types.ModuleType("sqlalchemy")
        sqlalchemy_future_module = types.ModuleType("sqlalchemy.future")
        sqlalchemy_future_module.select = lambda *args, **kwargs: object()
        sys.modules["sqlalchemy"] = sqlalchemy_module
        sys.modules["sqlalchemy.future"] = sqlalchemy_future_module

        settings = types.SimpleNamespace(
            LOG_LEVEL="INFO",
            KAFKA_GROUP_ID="group",
            KAFKA_AUTO_OFFSET_RESET="earliest",
            KAFKA_SECURITY_PROTOCOL="PLAINTEXT",
            KAFKA_SASL_MECHANISM="PLAIN",
            KAFKA_USERNAME="user",
            KAFKA_PASSWORD="pass",
            kafka_bootstrap_servers="localhost:9092",
            person_upright_device_sns=frozenset(),
            person_face_reuse_mode="on",
            FACE_FRAME_CONCURRENCY=4,
            FACE_GLOBAL_CONCURRENCY=24,
            FACE_PREPARE_LOOKAHEAD=2,
            FACE_SEARCH_PREFETCH_ENABLED=True,
            PERSON_FRAME_CONCURRENCY=6,
            PERSON_GLOBAL_CONCURRENCY=32,
            PERSON_PREPARE_LOOKAHEAD=2,
            IMAGE_CPU_WORKERS=4,
        )
        core_config_module = types.ModuleType("core.config")
        core_config_module.settings = settings
        sys.modules["core.config"] = core_config_module

        core_infer_module = types.ModuleType("core.infer")
        core_infer_module.UnifiedTritonClient = FakeTriton
        sys.modules["core.infer"] = core_infer_module

        db_zilliz_module = types.ModuleType("db.zilliz")
        db_zilliz_module.face_embedding_db = object()
        db_zilliz_module.person_clip_embedding_db = FakePersonDb()
        db_zilliz_module.person_sbs_embedding_db = FakePersonDb()
        sys.modules["db.zilliz"] = db_zilliz_module

        cam_device_module = types.ModuleType("db.pgsql.cam_device")
        cam_device_module.SessionLocal = object
        sys.modules["db.pgsql.cam_device"] = cam_device_module

        buffer_module = types.ModuleType("utils.buffers.buffer")
        buffer_module.feature_buffer = types.SimpleNamespace(sbs=object())
        sys.modules["utils.buffers.buffer"] = buffer_module

        async def fake_get_image_by_oss_name(object_name, oss_client, *, image_executor):
            return normal_person_test_image(), b"raw-image"

        image_util_module = types.ModuleType("utils.storage.image_util")
        image_util_module.get_image_by_oss_name = fake_get_image_by_oss_name
        sys.modules["utils.storage.image_util"] = image_util_module

        oss_util_module = types.ModuleType("utils.storage.oss_util")
        oss_util_module.AsyncOSSClient = object
        sys.modules["utils.storage.oss_util"] = oss_util_module

        id_generator_module = types.ModuleType("utils.identifiers.id_generator")
        id_generator_module.id_generator = types.SimpleNamespace(generate=lambda: 42)
        sys.modules["utils.identifiers.id_generator"] = id_generator_module

        face_quality_module = types.ModuleType("utils.quality.face_quality")
        face_quality_module.calculate_laplacian_variance = lambda image: 100.0
        face_quality_module.calculate_pose_from_landmarks = (
            lambda landmarks, image_shape: (0.0, 0.0, 0.0)
        )
        face_quality_module.is_high_quality_feature = lambda embedding: (True, None)
        face_quality_module.meets_pre_extraction_quality_criteria = (
            lambda image, landmarks, bbox, logger, **kwargs: (True, None)
        )
        sys.modules["utils.quality.face_quality"] = face_quality_module

        self.person_face_crops = []
        self.person_face_id = "0"

        async def fake_face_match(**kwargs):
            self.person_face_crops.append(np.asarray(kwargs["person_crop"]).copy())
            return {
                "face_id": self.person_face_id,
                "info": "未检测到人脸",
                "score": 0.0,
            }

        person_face_module = types.ModuleType("utils.reid.person_face_recognition")
        person_face_module.match_face_id_from_person_crop_with_info = fake_face_match
        person_face_module.select_owned_face_detection = select_owned_face_detection
        sys.modules["utils.reid.person_face_recognition"] = person_face_module

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_dispatcher_transfers_ownership_but_join_waits_for_real_handler(self):
        main = importlib.import_module("main")
        from core.queue_metrics import PipelineMetrics

        async def run():
            queue = asyncio.Queue()
            for message in ("first", "second"):
                await queue.put(message)
            stop_event = asyncio.Event()
            stop_event.set()
            release = asyncio.Event()
            both_started = asyncio.Event()
            started, completed, errors = [], [], []
            metrics = PipelineMetrics()
            storage = object()

            async def handler(message, client):
                self.assertIs(client, storage)
                started.append(message)
                if len(started) == 2:
                    both_started.set()
                await release.wait()
                completed.append(message)

            scheduler = main.MessageScheduler(
                lambda message: main.worker_process_message(message, storage),
                capacity=2, on_error=errors.append, metrics=metrics,
            )
            with patch.object(main, "worker_process_message", handler), patch.object(
                main, "feature_buffer", types.SimpleNamespace(wait_for_capacity=AsyncMock()),
            ), patch.object(main, "pipeline_metrics", metrics):
                dispatcher = asyncio.create_task(main.processing_worker(
                    "test-dispatcher", queue, storage, stop_event, scheduler,
                ))
                joined = asyncio.create_task(queue.join())
                try:
                    await asyncio.wait_for(dispatcher, timeout=2)
                    await asyncio.wait_for(both_started.wait(), timeout=2)
                    await asyncio.sleep(0)
                    self.assertTrue(queue.empty())
                    self.assertFalse(joined.done())
                    self.assertEqual(completed, [])
                    self.assertEqual(scheduler.queue_snapshot()["inflight_messages"], 2)
                    self.assertEqual(metrics.snapshot()["stages"]["message_processing"]["active"], 2)
                    release.set()
                    await asyncio.wait_for(joined, timeout=2)
                    await scheduler.close()
                finally:
                    release.set()
                    dispatcher.cancel()
                    joined.cancel()
                    await asyncio.gather(dispatcher, joined, return_exceptions=True)
                    await scheduler.close()
            self.assertEqual(started, ["first", "second"])
            self.assertEqual(completed, ["first", "second"])
            self.assertEqual(errors, [])
            self.assertEqual(scheduler.queue_snapshot()["inflight_messages"], 0)

        asyncio.run(run())

    def test_cancelling_dispatcher_preserves_admitted_business(self):
        main = importlib.import_module("main")
        from core.queue_metrics import PipelineMetrics

        async def run():
            queue = asyncio.Queue()
            for message in ("admitted", "waiting"):
                await queue.put(message)
            release = asyncio.Event()
            started, completed, cancelled, errors = [], [], [], []
            metrics = PipelineMetrics()
            storage = object()

            async def handler(message, client):
                self.assertIs(client, storage)
                started.append(message)
                try:
                    await release.wait()
                    completed.append(message)
                except asyncio.CancelledError:
                    cancelled.append(message)
                    raise

            scheduler = main.MessageScheduler(
                lambda message: main.worker_process_message(message, storage),
                capacity=1, on_error=errors.append, metrics=metrics,
            )

            async def admission_is_waiting():
                while scheduler.queue_snapshot()["admission_waiters"] != 1 or not started:
                    await asyncio.sleep(0)

            with patch.object(main, "worker_process_message", handler), patch.object(
                main, "feature_buffer", types.SimpleNamespace(wait_for_capacity=AsyncMock()),
            ), patch.object(main, "pipeline_metrics", metrics):
                dispatcher = asyncio.create_task(main.processing_worker(
                    "test-dispatcher", queue, storage, asyncio.Event(), scheduler,
                ))
                joined = asyncio.create_task(queue.join())
                try:
                    await asyncio.wait_for(admission_is_waiting(), timeout=2)
                    self.assertEqual(started, ["admitted"])
                    dispatcher.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await dispatcher
                    self.assertEqual(cancelled, [])
                    self.assertEqual(completed, [])
                    self.assertFalse(joined.done())
                    snapshot = scheduler.queue_snapshot()
                    self.assertEqual((snapshot["inflight_messages"], snapshot["admission_waiters"]), (1, 0))
                    self.assertEqual(metrics.snapshot()["stages"]["message_admission"]["waiting"], 0)
                    release.set()
                    await asyncio.wait_for(joined, timeout=2)
                    await scheduler.close()
                finally:
                    release.set()
                    dispatcher.cancel()
                    joined.cancel()
                    await asyncio.gather(dispatcher, joined, return_exceptions=True)
                    await scheduler.close()
            self.assertEqual(started, ["admitted"])
            self.assertEqual(completed, ["admitted"])
            self.assertEqual(cancelled, [])
            self.assertEqual(errors, [])
            self.assertEqual(scheduler.queue_snapshot()["inflight_messages"], 0)

        asyncio.run(run())

    def test_shutdown_stop_event_drains_queued_and_admitted_messages(self):
        main = importlib.import_module("main")
        from core.queue_metrics import PipelineMetrics

        async def run():
            queue = asyncio.Queue()
            for message in range(3):
                await queue.put(message)
            stop_event = asyncio.Event()
            releases = [asyncio.Event() for _ in range(3)]
            starts = [asyncio.Event() for _ in range(3)]
            started, completed, errors = [], [], []
            metrics = PipelineMetrics()
            storage = object()

            async def handler(message, client):
                self.assertIs(client, storage)
                started.append(message)
                starts[message].set()
                await releases[message].wait()
                completed.append(message)

            scheduler = main.MessageScheduler(
                lambda message: main.worker_process_message(message, storage),
                capacity=1, on_error=errors.append, metrics=metrics,
            )

            async def admission_is_waiting():
                while scheduler.queue_snapshot()["admission_waiters"] != 1:
                    await asyncio.sleep(0)

            with patch.object(main, "worker_process_message", handler), patch.object(
                main, "feature_buffer", types.SimpleNamespace(wait_for_capacity=AsyncMock()),
            ), patch.object(main, "pipeline_metrics", metrics):
                dispatcher = asyncio.create_task(main.processing_worker(
                    "test-dispatcher", queue, storage, stop_event, scheduler,
                ))
                joined = asyncio.create_task(queue.join())
                try:
                    await asyncio.wait_for(starts[0].wait(), timeout=2)
                    await asyncio.wait_for(admission_is_waiting(), timeout=2)
                    self.assertEqual(queue.qsize(), 1)
                    stop_event.set()
                    self.assertFalse(joined.done())
                    self.assertFalse(dispatcher.done())
                    releases[0].set()
                    await asyncio.wait_for(starts[1].wait(), timeout=2)
                    self.assertFalse(joined.done())
                    releases[1].set()
                    await asyncio.wait_for(starts[2].wait(), timeout=2)
                    await asyncio.wait_for(dispatcher, timeout=2)
                    self.assertTrue(queue.empty())
                    self.assertFalse(joined.done())
                    self.assertEqual(completed, [0, 1])
                    releases[2].set()
                    await asyncio.wait_for(joined, timeout=2)
                    await scheduler.close()
                finally:
                    for release in releases:
                        release.set()
                    dispatcher.cancel()
                    joined.cancel()
                    await asyncio.gather(dispatcher, joined, return_exceptions=True)
                    await scheduler.close()
            self.assertEqual(started, [0, 1, 2])
            self.assertEqual(completed, [0, 1, 2])
            self.assertEqual(errors, [])
            snapshot = scheduler.queue_snapshot()
            self.assertEqual((snapshot["inflight_messages"], snapshot["admission_waiters"]), (0, 0))
            self.assertTrue(snapshot["closed"])

        asyncio.run(run())

    def test_person_body_groups_and_pending_use_one_class_resolution(self):
        main = importlib.import_module("main")

        class GroupedPersonDb:
            async def asearch_similar_person_candidates_with_hit_info(
                self,
                embedding,
                query_body_completeness,
                device_ids=None,
                face_id=None,
                **_kwargs,
            ):
                return (
                    [
                        ReidCandidate(
                            "combined-reid",
                            0.86,
                            REID_SOURCE_MILVUS,
                            0.56,
                            BODY_FULL,
                            device_id="CAM001",
                            prototype_id=1,
                        ),
                        ReidCandidate(
                            "combined-reid",
                            0.84,
                            REID_SOURCE_MILVUS,
                            0.53,
                            BODY_THREE_QUARTER_UPPER,
                            device_id="CAM001",
                            prototype_id=2,
                        ),
                        ReidCandidate(
                            "single-group-reid",
                            0.88,
                            REID_SOURCE_MILVUS,
                            0.56,
                            BODY_FULL,
                            device_id="CAM001",
                            prototype_id=3,
                        ),
                    ],
                    0.79,
                    "single-group-reid",
                    REID_SOURCE_MILVUS,
                )

        fake_buffer = FakeBuffer()
        fake_buffer.pending_person_result = (
            "combined-reid",
            0.83,
            "combined-reid",
        )
        processor = main.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FakeTriton(),
            feature_buffer=fake_buffer,
            face_embedding_db=object(),
            person_sbs_embedding_db=GroupedPersonDb(),
        )

        result = asyncio.run(
            processor._person_reid_search.search(
                np.ones(2048, dtype=np.float32),
                BODY_FULL,
                ["CAM001"],
            )
        )

        self.assertEqual(result.reid, "combined-reid")
        self.assertEqual(result.state, "ACTIVE")
        self.assertEqual(fake_buffer.pending_person_search_calls, 1)

    def test_keyframe_contexts_transport_their_own_frame_faces(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FakeTriton(),
            feature_buffer=FakeBuffer(),
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonDb(),
        )
        main_faces = [object()]
        pano_faces = [object()]

        class FakeFaceProcessor:
            async def process_detection(self, ctx, *args, **kwargs):
                return pano_faces if ctx.is_pano_keyframe else main_faces

        class FakePersonProcessor:
            def __init__(self):
                self.seen = {}

            async def process_detection(self, ctx, *args, frame_faces=()):
                self.seen[ctx.object_name] = frame_faces

        async def skip_picture_feature(*args, **kwargs):
            return None

        processor.face_processor = FakeFaceProcessor()
        person = FakePersonProcessor()
        processor.person_processor = person
        processor.store_picture_feature = skip_picture_feature
        main_ctx = processor_module.KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/main.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
        )
        pano_ctx = processor_module.KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/pano.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
            is_pano_keyframe=True,
        )

        asyncio.run(
            processor._process_keyframe_context(main_ctx, oss_client=FakeOSS())
        )
        asyncio.run(
            processor._process_keyframe_context(pano_ctx, oss_client=FakeOSS())
        )

        self.assertIs(person.seen[main_ctx.object_name], main_faces)
        self.assertIs(person.seen[pano_ctx.object_name], pano_faces)

    def test_legacy_person_detection_call_defaults_to_empty_frame_faces(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FakeTriton(),
            feature_buffer=FakeBuffer(),
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonDb(),
        )

        class FakePersonProcessor:
            def __init__(self):
                self.seen = None

            async def process_detection(self, *args, frame_faces=()):
                self.seen = frame_faces

        person = FakePersonProcessor()
        processor.person_processor = person
        ctx = processor_module.KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/main.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
        )

        asyncio.run(
            processor.process_person_detection(
                ctx,
                normal_person_test_image(),
                ctx.keyframe_id,
                42,
                ctx.object_name,
                FakeOSS(),
            )
        )

        self.assertIs(person.seen, ())

    def test_reuse_mode_override_reaches_person_pipeline_without_stage_setting(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FakeTriton(),
            feature_buffer=FakeBuffer(),
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonDb(),
            person_face_reuse_mode="off",
        )

        self.assertEqual(processor.person_processor._pipeline.reuse_mode, "off")
        self.assertFalse(
            hasattr(processor.person_processor._pipeline, "reuse_max_stage")
        )

    def test_reuse_stage_is_not_part_of_runtime_constructors(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        person_module = importlib.import_module("core.person_keyframe_processor")
        pipeline_module = importlib.import_module("core.person_pipeline.pipeline")
        features_module = importlib.import_module("core.person_pipeline.features")

        constructors = (
            (processor_module.KeyframeProcessor, "person_face_reuse_max_stage"),
            (person_module.PersonKeyframeProcessor, "person_face_reuse_max_stage"),
            (pipeline_module.PersonDetectionPipeline, "person_face_reuse_max_stage"),
            (features_module.PersonFeatureService, "reuse_max_stage"),
        )
        for constructor, removed_parameter in constructors:
            with self.subTest(constructor=constructor.__name__):
                self.assertNotIn(
                    removed_parameter,
                    inspect.signature(constructor).parameters,
                )

    def test_reuse_off_skips_binding_and_supplies_aligned_disabled_bindings(self):
        pipeline_module = importlib.import_module("core.person_pipeline.pipeline")
        pipeline = pipeline_module.PersonDetectionPipeline.__new__(
            pipeline_module.PersonDetectionPipeline
        )
        observations = [
            types.SimpleNamespace(segment={"index": 0}),
            types.SimpleNamespace(segment={"index": 1}),
        ]

        class FakeObservations:
            async def detect(self, ctx, key_frame):
                return observations

        seen_bindings = []

        async def capture_observation(self, *args):
            seen_bindings.append(args[-1])

        pipeline.observations = FakeObservations()
        pipeline.logger = DummyLogger()
        pipeline.reuse_mode = "off"
        pipeline.frame_concurrency = 4
        pipeline.prepare_lookahead = 2
        pipeline._preparation_slots = asyncio.Semaphore(16)
        pipeline._commit_observation = AsyncMock()
        pipeline._prepare_observation = types.MethodType(
            capture_observation,
            pipeline,
        )

        with patch.object(
            pipeline_module,
            "bind_frame_faces_to_people",
            side_effect=AssertionError("reuse-off must not bind frame faces"),
            create=True,
        ):
            asyncio.run(
                pipeline.process(
                    object(),
                    normal_person_test_image(),
                    "keyframe-id",
                    42,
                    "CAM001/picture/main.jpg",
                    FakeOSS(),
                    frame_faces=[object()],
                )
            )

        self.assertEqual(
            [binding.reason for binding in seen_bindings],
            ["reuse_off", "reuse_off"],
        )

    def test_reuse_enabled_passes_aligned_binding_to_observation_processing(self):
        pipeline_module = importlib.import_module("core.person_pipeline.pipeline")
        face_reuse_module = importlib.import_module("core.person_pipeline.face_reuse")
        pipeline = pipeline_module.PersonDetectionPipeline.__new__(
            pipeline_module.PersonDetectionPipeline
        )
        observation = types.SimpleNamespace(segment={"index": 0})
        frame_faces = [object()]
        binding = face_reuse_module.FrameFaceBinding(reason="bound")

        class FakeObservations:
            async def detect(self, ctx, key_frame):
                return [observation]

        seen_bindings = []

        async def capture_observation(self, *args):
            seen_bindings.append(args[-1])

        def bind_faces(actual_faces, actual_people, image_shape):
            self.assertIs(actual_faces, frame_faces)
            self.assertEqual(actual_people, [observation])
            self.assertEqual(image_shape, (120, 120, 3))
            return [binding]

        pipeline.observations = FakeObservations()
        pipeline.logger = DummyLogger()
        pipeline.reuse_mode = "on"
        pipeline.frame_concurrency = 4
        pipeline.prepare_lookahead = 2
        pipeline._preparation_slots = asyncio.Semaphore(16)
        pipeline._commit_observation = AsyncMock()
        pipeline._prepare_observation = types.MethodType(
            capture_observation,
            pipeline,
        )

        with patch.object(
            pipeline_module,
            "bind_frame_faces_to_people",
            side_effect=bind_faces,
            create=True,
        ):
            asyncio.run(
                pipeline.process(
                    object(),
                    normal_person_test_image(),
                    "keyframe-id",
                    42,
                    "CAM001/picture/main.jpg",
                    FakeOSS(),
                    frame_faces=frame_faces,
                )
            )

        self.assertEqual(seen_bindings, [binding])

    def _reuse_processor(
        self,
        *,
        mode,
        frame_has_face=True,
        shared_face_id="shared-face",
        local_face_id="local-face",
        local_orientation="front",
    ):
        processor_module = importlib.import_module("core.keyframe_processor")
        fake_buffer = FakeBuffer()
        fake_triton = FacePersonReuseTriton(frame_has_face=frame_has_face)
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=FakeFaceDb(
                face_id=shared_face_id,
                score=0.99,
                hit_face_id=shared_face_id if shared_face_id != "0" else "",
            ),
            person_sbs_embedding_db=FakePersonDb(),
            person_face_reuse_mode=mode,
        )
        runtime = processor.person_processor._pipeline.features.runtime

        async def local_face_match(**kwargs):
            await kwargs["triton_client"].extract_face_detections(
                kwargs["person_crop"]
            )
            return {
                "face_id": local_face_id,
                "info": "local face result",
                "face_orientation": local_orientation,
            }

        processor.person_processor._pipeline.features.runtime = replace(
            runtime,
            match_face=local_face_match,
        )
        ctx = processor_module.KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/reuse.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
        )
        asyncio.run(
            processor._process_keyframe_context(ctx, oss_client=FakeOSS())
        )
        return fake_buffer, fake_triton

    def _last_reuse_summary(self):
        summaries = [
            message
            for message in self.logger.info_messages
            if message.startswith("Person face reuse frame summary:")
        ]
        self.assertTrue(summaries)
        return summaries[-1]

    def test_on_identity_reuses_frame_face_and_saves_person_detection_call(self):
        fake_buffer, fake_triton = self._reuse_processor(mode="on")

        self.assertEqual(fake_triton.face_detection_calls, 1)
        self.assertEqual(
            fake_buffer.person_detection_records[0]["faceOrientation"],
            "front",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["aiFaceId"],
            "shared-face",
        )
        self.assertIn(
            "Person face reuse frame summary: mode=on, people=1, "
            "frame_faces=1, bound=1, direct=1, partial=0, orientation=0, "
            "fallback=0, fallback_recovered=0, id_disagreement=0, "
            "id_shared_only=0, id_local_only=0, id_different=0, "
            "orientation_disagreement=0, fallback_reasons=none, rotated=0, "
            "shared_success_rotated=0, shared_success_unrotated=1, "
            "fallback_rotated=0, fallback_unrotated=0, recovery_rotated=0, "
            "recovery_unrotated=0",
            self.logger.info_messages,
        )

    def test_on_without_frame_face_keeps_one_local_detection_call(self):
        fake_buffer, fake_triton = self._reuse_processor(
            mode="on",
            frame_has_face=False,
        )

        self.assertEqual(fake_triton.face_detection_calls, 2)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["aiFaceId"],
            "local-face",
        )




    def test_off_keeps_local_call_and_does_not_publish_reuse_diagnostics(self):
        fake_buffer, fake_triton = self._reuse_processor(mode="off")

        self.assertEqual(fake_triton.face_detection_calls, 2)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["aiFaceId"],
            "local-face",
        )
        for record in (
            fake_buffer.person_detection_records
            + fake_buffer.person_recognition_records
        ):
            self.assertFalse(
                {
                    "direct",
                    "partial",
                    "fallback",
                    "fallbackRecovered",
                    "idDisagreement",
                }
                & record.keys()
            )

    def test_worker_runs_picture_face_and_person_paths(self):
        main = importlib.import_module("main")
        fake_buffer = FakeBuffer()
        fake_triton = FakeTriton()
        self.person_face_id = "face-1"
        device_sn = "CAM001"
        uid = "11111111-1111-4111-8111-111111111111"
        main.KEYFRAME_PROCESSOR = main.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=object(),
            person_sbs_embedding_db=FakePersonDb(),
        )

        msg = types.SimpleNamespace(
            topic="PanoramicPersonDetSync",
            offset=7,
            value=json.dumps(
                {
                    "sn": device_sn,
                    "timestamp": 1710000000000,
                    "keyframe": "frame001",
                    "uId": uid,
                }
            ).encode("utf-8"),
        )

        asyncio.run(main.worker_process_message(msg, oss_client=FakeOSS()))

        self.assertEqual(fake_triton.image_feature_calls, 1)
        self.assertEqual(fake_triton.face_detection_calls, 1)
        self.assertEqual(fake_triton.person_segment_calls, 1)
        self.assertEqual(fake_triton.tbps_calls, 1)
        self.assertEqual(len(fake_buffer.picture_features), 1)
        self.assertEqual(fake_buffer.picture_features[0]["keyframe_id"], uid)
        self.assertEqual(fake_buffer.picture_features[0]["picture_vector_id"], 42)
        self.assertIs(fake_buffer.picture_features[0]["is_pano_keyframe"], False)
        self.assertEqual(len(fake_buffer.person_detection_records), 2)
        self.assertEqual(
            [record["status"] for record in fake_buffer.person_detection_records],
            [0, 1],
        )
        self.assertEqual(
            [
                record["personQualityLevel"]
                for record in fake_buffer.person_detection_records
            ],
            [0, 2],
        )
        self.assertEqual(
            [record["keyframeId"] for record in fake_buffer.person_detection_records],
            [uid, uid],
        )
        self.assertEqual(
            [record["isPanoKeyframe"] for record in fake_buffer.person_detection_records],
            [False, False],
        )
        self.assertEqual(
            [
                record["pictureVectorId"]
                for record in fake_buffer.person_detection_records
            ],
            [42, 42],
        )
        self.assertTrue(
            fake_buffer.person_detection_records[0]["crop_url"].startswith(
                f"{device_sn}/person_crop/"
            )
        )
        self.assertTrue(
            fake_buffer.person_detection_records[1]["crop_url"].startswith(
                f"{device_sn}/person_crop/"
            )
        )
        self.assertEqual(len(fake_buffer.person_features), 1)
        self.assertEqual(fake_buffer.person_features[0]["keyframe_id"], uid)
        self.assertEqual(fake_buffer.person_features[0]["picture_vector_id"], 42)
        self.assertIs(fake_buffer.person_features[0]["is_pano_keyframe"], False)
        self.assertIs(fake_buffer.person_clip_features[0]["is_pano_keyframe"], False)
        self.assertEqual(len(fake_buffer.person_recognition_records), 1)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["keyframeId"],
            uid,
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["pictureVectorId"],
            42,
        )
        self.assertIs(
            fake_buffer.person_recognition_records[0]["isPanoKeyframe"],
            False,
        )

    def test_payload_processes_main_and_pano_keyframes(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        performance = RuntimePerformance()
        self.enterContext(patch.object(processor_module, "runtime_performance", performance))
        fake_buffer = FakeBuffer()
        fake_triton = FakeTriton()
        loaded_names = []
        generated_ids = iter((101, 102))
        device_sn = "CAM001"
        uid = "66666666-6666-4666-8666-666666666666"
        main_name = f"{device_sn}/picture/main.jpg"
        pano_name = f"{device_sn}/picture/pano.jpg"

        async def load_image(object_name, oss_client):
            loaded_names.append(object_name)
            return normal_person_test_image(), object_name.encode("utf-8")

        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonDb(),
            keyframe_id_generator=types.SimpleNamespace(
                generate=lambda: next(generated_ids)
            ),
            image_loader=load_image,
        )

        asyncio.run(
            processor.process_payload(
                {
                    "sn": device_sn,
                    "timestamp": 1710000000000,
                    "keyframe": main_name,
                    "panoKeyframeUrl": pano_name,
                    "uId": uid,
                },
                oss_client=FakeOSS(),
                topic="PanoramicPersonDetSync",
                offset=8,
            )
        )

        self.assertEqual(loaded_names, [main_name, pano_name])
        self.assertEqual(fake_triton.image_feature_calls, 2)
        self.assertEqual(fake_triton.face_detection_calls, 2)
        self.assertEqual(fake_triton.person_segment_calls, 2)
        self.assertEqual(
            [item["picture_vector_id"] for item in fake_buffer.picture_features],
            [101, 102],
        )
        self.assertEqual(
            [item["image_path"] for item in fake_buffer.picture_features],
            [main_name, pano_name],
        )
        self.assertEqual(
            [item["keyframe_id"] for item in fake_buffer.picture_features],
            [uid, uid],
        )
        self.assertEqual(
            [item["is_pano_keyframe"] for item in fake_buffer.picture_features],
            [False, True],
        )
        self.assertEqual(
            [
                record["pictureVectorId"]
                for record in fake_buffer.person_detection_records
            ],
            [101, 101, 102, 102],
        )
        self.assertEqual(
            [record["isPanoKeyframe"] for record in fake_buffer.person_detection_records],
            [False, False, True, True],
        )
        summaries = [
            message
            for message in self.logger.info_messages
            if message.startswith("Person face reuse frame summary:")
        ]
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all("mode=on" in message for message in summaries))
        self.assertEqual(
            performance.snapshot()["images_completed_by_topic"],
            {"PanoramicPersonDetSync": 2},
        )

    def test_payload_skips_duplicate_pano_keyframe(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        performance = RuntimePerformance()
        self.enterContext(patch.object(processor_module, "runtime_performance", performance))
        fake_buffer = FakeBuffer()
        loaded_names = []
        generated_ids = iter((201, 202))
        device_sn = "CAM001"
        uid = "77777777-7777-4777-8777-777777777777"
        object_name = f"{device_sn}/picture/same.jpg"

        async def load_image(current_object_name, oss_client):
            loaded_names.append(current_object_name)
            return normal_person_test_image(), current_object_name.encode("utf-8")

        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FakeTriton(),
            feature_buffer=fake_buffer,
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonDb(),
            keyframe_id_generator=types.SimpleNamespace(
                generate=lambda: next(generated_ids)
            ),
            image_loader=load_image,
        )

        asyncio.run(
            processor.process_payload(
                {
                    "sn": device_sn,
                    "timestamp": 1710000000000,
                    "keyframe": object_name,
                    "panoKeyframeUrl": (
                        f"https://oss.example.com/{object_name}?token=test"
                    ),
                    "uId": uid,
                },
                oss_client=FakeOSS(),
            )
        )

        self.assertEqual(loaded_names, [object_name])
        self.assertEqual(len(fake_buffer.picture_features), 1)
        self.assertIs(fake_buffer.picture_features[0]["is_pano_keyframe"], False)
        self.assertEqual(performance.snapshot()["images_completed_by_topic"], {"": 1})

    def test_image_completion_does_not_count_missing_or_failed_processing(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        ctx = processor_module.KeyframeContext(
            sn="CAM001", timestamp=1710000000000,
            object_name="CAM001/picture/main.jpg",
            keyframe_id="88888888-8888-4888-8888-888888888888",
        )
        for failure in ("missing", "download", "person"):
            with self.subTest(failure=failure):
                performance = RuntimePerformance()
                fake_buffer = FakeBuffer()
                loader = AsyncMock(return_value=(normal_person_test_image(), b"image"))
                if failure == "missing":
                    loader.return_value = (None, b"")
                elif failure == "download":
                    loader.side_effect = RuntimeError("download failed")
                processor = processor_module.KeyframeProcessor(
                    image_executor=InlineImageExecutor(),
                    triton_client=FakeTriton(),
                    feature_buffer=fake_buffer,
                    face_embedding_db=FakeFaceDb(),
                    person_sbs_embedding_db=FakePersonDb(),
                    image_loader=loader,
                )
                if failure == "person":
                    processor.process_person_detection = AsyncMock(
                        side_effect=RuntimeError("person processing failed")
                    )
                with patch.object(processor_module, "runtime_performance", performance):
                    if failure == "missing":
                        self.assertFalse(asyncio.run(processor._process_keyframe_context(
                            ctx, oss_client=FakeOSS(), topic="PtzPersonDetSync",
                        )))
                    else:
                        with self.assertRaises(RuntimeError):
                            asyncio.run(processor._process_keyframe_context(
                                ctx, oss_client=FakeOSS(), topic="PtzPersonDetSync",
                            ))
                if failure == "person":
                    self.assertEqual(len(fake_buffer.picture_features), 1)
                self.assertEqual(performance.snapshot()["images_completed_by_topic"], {})

    def test_traced_completion_follows_all_frame_stages_and_separates_pano(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        performance = RuntimePerformance()
        run_id = "123456789abc4def8123456789abcdef"
        ctx = processor_module.KeyframeContext(
            sn="CAM001", timestamp=1710000000000, object_name="CAM001/picture/main.jpg",
            keyframe_id=f"alogtest-v1:{run_id}:0",
        )
        stages = []
        expected_before = 0

        async def stage(name):
            stages.append(name)
            current = performance.snapshot()["replay_runs"].get(run_id)
            completed = current["main"]["completed_count"] + current["pano"]["completed_count"] if current else 0
            self.assertEqual(completed, expected_before)

        processor = object.__new__(processor_module.KeyframeProcessor)
        processor.image_loader = AsyncMock(return_value=(normal_person_test_image(), b"image"))
        processor.keyframe_id_generator = types.SimpleNamespace(generate=lambda: 42)

        async def picture(*args, **kwargs):
            await stage("picture")

        async def faces(*args, **kwargs):
            await stage("face")
            return []

        async def persons(*args, **kwargs):
            await stage("person")

        processor.store_picture_feature = picture
        processor.process_face_detection = faces
        processor.process_person_detection = persons
        with patch.object(processor_module, "runtime_performance", performance):
            self.assertTrue(asyncio.run(processor._process_keyframe_context(
                ctx, oss_client=FakeOSS(), topic="PtzPersonDetSync",
            )))
        self.assertEqual(stages, ["picture", "face", "person"])
        run = performance.snapshot()["replay_runs"][run_id]
        self.assertEqual(run["main"]["completed_count"], 1)
        self.assertEqual(run["pano"]["completed_count"], 0)
        expected_before = 1
        with patch.object(processor_module, "runtime_performance", performance):
            self.assertTrue(asyncio.run(processor._process_keyframe_context(
                replace(ctx, is_pano_keyframe=True, object_name="CAM001/picture/pano.jpg"),
                oss_client=FakeOSS(), topic="PtzPersonDetSync",
            )))
        run = performance.snapshot()["replay_runs"][run_id]
        self.assertEqual(run["main"]["completed_count"], 1)
        self.assertEqual(run["pano"]["completed_count"], 1)

    def test_pano_failure_does_not_fail_main_keyframe(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        fake_buffer = FakeBuffer()
        fake_triton = FakeTriton()
        loaded_names = []
        device_sn = "CAM001"
        uid = "88888888-8888-4888-8888-888888888888"
        main_name = f"{device_sn}/picture/main.jpg"
        pano_name = f"{device_sn}/picture/pano.jpg"

        async def load_image(object_name, oss_client):
            loaded_names.append(object_name)
            if object_name == pano_name:
                raise RuntimeError("pano download failed")
            return normal_person_test_image(), object_name.encode("utf-8")

        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonDb(),
            keyframe_id_generator=types.SimpleNamespace(generate=lambda: 301),
            image_loader=load_image,
        )

        try:
            asyncio.run(
                processor.process_payload(
                    {
                        "sn": device_sn,
                        "timestamp": 1710000000000,
                        "keyframe": main_name,
                        "panoKeyframeUrl": pano_name,
                        "uId": uid,
                    },
                    oss_client=FakeOSS(),
                    topic="PanoramicPersonDetSync",
                    offset=9,
                )
            )
        except RuntimeError as exc:
            self.fail(f"pano failure escaped process_payload: {exc}")

        self.assertEqual(loaded_names, [main_name, pano_name])
        self.assertEqual(fake_triton.image_feature_calls, 1)
        self.assertEqual(len(fake_buffer.picture_features), 1)
        self.assertEqual(fake_buffer.picture_features[0]["image_path"], main_name)
        self.assertIs(fake_buffer.picture_features[0]["is_pano_keyframe"], False)

    def test_process_payload_requires_uid(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        fake_buffer = FakeBuffer()
        fake_triton = FakeTriton()
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=object(),
            person_sbs_embedding_db=FakePersonDb(),
        )

        with self.assertRaises(ValueError):
            asyncio.run(
                processor.process_payload(
                    {
                        "sn": "CAM001",
                        "timestamp": 1710000000000,
                        "keyframe": "frame001",
                    },
                    oss_client=FakeOSS(),
                )
            )
        self.assertEqual(fake_buffer.picture_features, [])
        self.assertEqual(fake_buffer.person_detection_records, [])

    def test_quality_grading_helpers(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")

        face_enroll_threshold = algorithm_config.face.quality.enroll_min_score
        person_enroll_threshold = algorithm_config.person.quality.general.enroll_min_score
        recognition_mask = int(
            algorithm_config.person.quality.general.max_mask_pixels
            * algorithm_config.person.quality.general.recognition_min_score
            / 10.0
        )
        enroll_mask = int(
            algorithm_config.person.quality.general.max_mask_pixels
            * algorithm_config.person.quality.general.enroll_min_score
            / 10.0
        )

        self.assertEqual(
            helper_module.face_quality_level_from_score(
                algorithm_config.face.quality.recognition_min_score - 0.01
            ),
            0,
        )
        self.assertEqual(
            helper_module.face_quality_level_from_score(
                algorithm_config.face.quality.recognition_min_score
            ),
            1,
        )
        self.assertEqual(
            helper_module.face_quality_level_from_score(
                algorithm_config.face.quality.recognition_min_score + 0.01
            ),
            1,
        )
        self.assertEqual(
            helper_module.face_quality_level_from_score(face_enroll_threshold - 0.01),
            1,
        )
        self.assertEqual(
            helper_module.face_quality_level_from_score(face_enroll_threshold),
            2,
        )
        self.assertAlmostEqual(
            helper_module.person_quality_score_from_mask(
                algorithm_config.person.quality.general.max_mask_pixels
            ),
            10.0,
        )
        self.assertEqual(
            helper_module.person_quality_level_from_score(
                helper_module.person_quality_score_from_mask(recognition_mask - 1)
            ),
            0,
        )
        self.assertEqual(
            helper_module.person_quality_level_from_score(
                helper_module.person_quality_score_from_mask(recognition_mask)
            ),
            1,
        )
        self.assertEqual(
            helper_module.person_quality_level_from_score(
                helper_module.person_quality_score_from_mask(enroll_mask - 1)
            ),
            1,
        )
        self.assertEqual(
            helper_module.person_quality_level_from_score(
                helper_module.person_quality_score_from_mask(enroll_mask)
            ),
            2,
        )
        self.assertEqual(
            helper_module.person_quality_level_from_score(person_enroll_threshold),
            2,
        )

    def _run_face_quality_flow(
        self,
        quality_score,
        face_db=None,
        oss_client=None,
        pending_face_result=None,
        is_pano_keyframe=False,
        pending_face_samples=None,
    ):
        processor_module = importlib.import_module("core.keyframe_processor")
        fake_buffer = FakeBuffer()
        fake_buffer.pending_face_samples = pending_face_samples
        if pending_face_result is not None:
            fake_buffer.pending_face_result = pending_face_result
        fake_triton = FaceOnlyTriton(quality_score)
        fake_face_db = face_db or FakeFaceDb()
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=fake_face_db,
            person_sbs_embedding_db=FakePersonReidDb(),
        )
        ctx = processor_module.KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/k.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
            is_pano_keyframe=is_pano_keyframe,
        )
        asyncio.run(
            processor.process_face_detection(
                ctx,
                np.zeros((120, 120, 3), dtype=np.uint8),
                ctx.keyframe_id,
                42,
                "CAM001/picture/k.jpg",
                oss_client or FakeOSS(),
            )
        )
        return fake_buffer, fake_triton, fake_face_db

    def test_face_level_zero_does_not_recognize_or_insert(self):
        fake_buffer, fake_triton, fake_face_db = self._run_face_quality_flow(
            algorithm_config.face.quality.recognition_min_score - 0.01
        )

        self.assertEqual(fake_buffer.face_detection_records[0]["status"], 0)
        self.assertTrue(
            fake_buffer.face_detection_records[0]["crop_url"].startswith(
                "CAM001/face_crop/"
            )
        )
        self.assertNotEqual(
            fake_buffer.face_detection_records[0]["crop_url"],
            "CAM001/picture/k.jpg",
        )
        self.assertEqual(fake_buffer.face_detection_records[0]["confidence"], 0.75)
        self.assertEqual(fake_buffer.face_detection_records[0]["faceQualityLevel"], 0)
        self.assertEqual(fake_triton.feature_calls, 0)
        self.assertEqual(fake_face_db.search_calls, 0)
        self.assertEqual(len(fake_buffer.face_features), 0)
        self.assertEqual(len(fake_buffer.face_recognition_records), 0)

    def test_face_level_zero_returns_quality_observation(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        observation_module = importlib.import_module("core.face_observation")
        quality_score = algorithm_config.face.quality.recognition_min_score - 0.01
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FaceOnlyTriton(quality_score),
            feature_buffer=FakeBuffer(),
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=FakePersonReidDb(),
        )
        ctx = processor_module.KeyframeContext(
            sn="CAM001",
            timestamp=1710000000000,
            object_name="CAM001/picture/k.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
        )

        observations = asyncio.run(
            processor.face_processor.process_detection(
                ctx,
                np.zeros((120, 120, 3), dtype=np.uint8),
                ctx.keyframe_id,
                42,
                ctx.object_name,
                FakeOSS(),
            )
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source_rect, (30, 30, 90, 90))
        self.assertEqual(
            observations[0].completed_stage,
            observation_module.FaceProcessingStage.QUALITY,
        )
        self.assertEqual(observations[0].quality_score, quality_score)

    def test_face_level_one_unmatched_recognizes_without_insert(self):
        for quality_score in (algorithm_config.face.quality.recognition_min_score,
                              algorithm_config.face.quality.enroll_min_score - 0.01):
            with self.subTest(quality_score=quality_score):
                fake_buffer, fake_triton, fake_face_db = self._run_face_quality_flow(
                    quality_score
                )

                detection = fake_buffer.face_detection_records[0]
                self.assertEqual(detection["status"], 1)
                self.assertEqual(detection["confidence"], 0.75)
                self.assertEqual(detection["faceQualityScore"], quality_score)
                self.assertEqual(detection["faceQualityLevel"], 1)
                self.assertIs(detection["isPanoKeyframe"], False)
                self.assertEqual(fake_triton.feature_calls, 1)
                self.assertEqual(fake_face_db.search_calls, 1)
                self.assertEqual(len(fake_buffer.face_features), 0)
                recognition = fake_buffer.face_recognition_records[0]
                self.assertEqual(recognition["faceUrl"], detection["crop_url"])
                self.assertEqual(recognition["aiFaceId"], "0")
                self.assertEqual(recognition["faceStatus"], 0)
                self.assertIs(recognition["isPanoKeyframe"], False)

    def test_face_level_one_matched_recognizes_without_insert(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            algorithm_config.face.quality.enroll_min_score - 0.01,
            face_db=FakeFaceDb(face_id="face-1", score=0.8, hit_face_id="face-1"),
        )

        self.assertEqual(len(fake_buffer.face_features), 0)
        self.assertEqual(fake_buffer.face_recognition_records[0]["aiFaceId"], "face-1")
        self.assertEqual(fake_buffer.face_recognition_records[0]["faceStatus"], 2)

    def test_face_level_two_pending_match_reuses_face_id_and_inserts(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            pending_face_result=("pending-face", 0.75, "pending-face"),
        )

        self.assertEqual(fake_buffer.pending_face_search_calls, 1)
        self.assertEqual(fake_buffer.face_recognition_records[0]["aiFaceId"], "pending-face")
        self.assertEqual(fake_buffer.face_recognition_records[0]["faceStatus"], 2)
        self.assertEqual(fake_buffer.face_recognition_records[0]["score"], 0.75)
        self.assertEqual(
            fake_buffer.face_recognition_records[0]["vectorHitFaceId"],
            "pending-face",
        )
        self.assertEqual(len(fake_buffer.face_features), 1)
        self.assertEqual(fake_buffer.face_features[0]["face_id"], "pending-face")
        self.assertEqual(fake_buffer.face_features[0]["quality_score"], 8.0)

    def test_face_match_uses_highest_score_and_ties_prefer_milvus(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            face_db=FakeFaceDb(face_id="milvus-face", score=0.8, hit_face_id="milvus-face"),
            pending_face_result=("pending-face", 0.9, "pending-face"),
        )

        self.assertEqual(fake_buffer.face_recognition_records[0]["aiFaceId"], "pending-face")
        self.assertEqual(
            fake_buffer.face_recognition_records[0]["vectorHitFaceId"],
            "pending-face",
        )
        self.assertEqual(fake_buffer.face_recognition_records[0]["score"], 0.9)

        tie_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            face_db=FakeFaceDb(face_id="milvus-face", score=0.8, hit_face_id="milvus-face"),
            pending_face_result=("pending-face", 0.8, "pending-face"),
        )

        self.assertEqual(tie_buffer.face_recognition_records[0]["aiFaceId"], "milvus-face")
        self.assertEqual(
            tie_buffer.face_recognition_records[0]["vectorHitFaceId"],
            "milvus-face",
        )
        self.assertEqual(tie_buffer.face_recognition_records[0]["score"], 0.8)

    def test_face_level_two_unmatched_generates_id_and_inserts(self):
        fake_buffer, fake_triton, fake_face_db = self._run_face_quality_flow(8.0)

        self.assertEqual(fake_buffer.face_detection_records[0]["faceQualityLevel"], 2)
        self.assertEqual(fake_triton.feature_calls, 1)
        self.assertEqual(fake_face_db.search_calls, 1)
        self.assertEqual(len(fake_buffer.face_features), 1)
        self.assertIs(fake_buffer.face_features[0]["is_pano_keyframe"], False)
        self.assertTrue(
            fake_buffer.face_features[0]["face_crop_path"].startswith(
                "CAM001/face_crop/"
            )
        )
        self.assertNotEqual(fake_buffer.face_recognition_records[0]["aiFaceId"], "0")
        self.assertEqual(fake_buffer.face_recognition_records[0]["faceStatus"], 1)

    def test_gray_confirmed_face_is_recognized_without_enrolling_its_feature(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            face_db=FakeFaceDb(
                face_id="face-1", score=0.48, hit_face_id="face-1",
                samples=[FaceSampleMatch("face-1", 0.48, "one"),
                         FaceSampleMatch("face-1", 0.46, "two")],
            ),
        )
        self.assertEqual(fake_buffer.face_recognition_records[0]["aiFaceId"], "face-1")
        self.assertEqual(fake_buffer.face_recognition_records[0]["faceStatus"], 2)
        self.assertEqual(fake_buffer.face_features, [])

    def test_gray_samples_split_between_gallery_and_pending_confirm_identity(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            face_db=FakeFaceDb(
                score=0.48, hit_face_id="face-1",
                samples=[FaceSampleMatch("face-1", 0.48, "one")],
            ),
            pending_face_samples=[FaceSampleMatch("face-1", 0.46, "two")],
        )
        recognition = fake_buffer.face_recognition_records[0]
        self.assertEqual(recognition["aiFaceId"], "face-1")
        self.assertEqual(recognition["faceStatus"], 2)
        self.assertEqual(recognition["score"], 0.48)
        self.assertEqual(fake_buffer.face_features, [])

    def test_same_sample_in_both_sources_does_not_confirm_gray_identity(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            face_db=FakeFaceDb(
                score=0.48, hit_face_id="face-1",
                samples=[FaceSampleMatch("face-1", 0.48, "same")],
            ),
            pending_face_samples=[FaceSampleMatch("face-1", 0.48, "same")],
        )
        recognition = fake_buffer.face_recognition_records[0]
        self.assertEqual(recognition["aiFaceId"], "0")
        self.assertEqual(recognition["faceStatus"], 0)
        self.assertEqual(fake_buffer.face_features, [])

    def test_pending_snapshot_covers_a_flush_during_gallery_search(self):
        processor_module = importlib.import_module("core.keyframe_processor")
        buffer = FakeBuffer()
        buffer.pending_face_samples = [FaceSampleMatch("face-1", 0.46, "two")]

        class FlushingFaceDb(FakeFaceDb):
            async def asearch_face_samples(self, embedding, device_ids=None):
                # The pending sample is persisted after this search's read point,
                # then removed from pending before the database reply arrives.
                buffer.pending_face_samples.clear()
                return [FaceSampleMatch("face-1", 0.48, "one")]

        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=FakeTriton(), feature_buffer=buffer,
            face_embedding_db=FlushingFaceDb(),
            person_sbs_embedding_db=FakePersonReidDb(),
        )
        result = asyncio.run(processor._search_faces_with_pending_locked(
            np.array([1.0, 0.0], dtype=np.float32), 0.45, None
        ))
        self.assertEqual(result, ("face-1", 0.48, "face-1"))

    def test_gray_unconfirmed_face_does_not_create_a_new_identity(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            face_db=FakeFaceDb(score=0.48, hit_face_id="face-1"),
        )
        recognition = fake_buffer.face_recognition_records[0]
        self.assertEqual(recognition["aiFaceId"], "0")
        self.assertEqual(recognition["faceStatus"], 0)
        self.assertEqual(recognition["score"], 0.48)
        self.assertEqual(recognition["vectorHitFaceId"], "face-1")
        self.assertEqual(fake_buffer.face_features, [])

    def test_face_gallery_failure_does_not_create_or_enroll_a_new_face(self):
        failing_db = FailingFaceDb()

        try:
            fake_buffer, _, _ = self._run_face_quality_flow(
                8.0,
                face_db=failing_db,
            )
        except RuntimeError as error:
            self.fail(f"face gallery failure escaped the face boundary: {error}")

        self.assertEqual(failing_db.search_calls, 1)
        self.assertEqual(fake_buffer.face_features, [])
        self.assertEqual(len(fake_buffer.face_recognition_records), 1)
        recognition = fake_buffer.face_recognition_records[0]
        self.assertEqual(recognition["aiFaceId"], "0")
        self.assertEqual(recognition["faceStatus"], 0)
        self.assertIn("人脸向量库检索失败", recognition["info"])
        self.assertIn("未生成新 face_id", recognition["info"])

    def test_pano_face_context_propagates_to_records_and_feature_handoff(self):
        fake_buffer, _, _ = self._run_face_quality_flow(
            8.0,
            is_pano_keyframe=True,
        )

        self.assertIs(fake_buffer.face_detection_records[0]["isPanoKeyframe"], True)
        self.assertIs(fake_buffer.face_recognition_records[0]["isPanoKeyframe"], True)
        self.assertIs(fake_buffer.face_features[0]["is_pano_keyframe"], True)

    def _run_person_quality_flow(
        self,
        mask_pixel_count,
        person_db=None,
        oss_client=None,
        pending_person_result=None,
        key_frame=None,
        triton=None,
        observed_face_id="0",
        is_pano_keyframe=False,
        upright_device_sns=frozenset(),
        person_face_reuse_mode=None,
        frame_faces=(),
        sn="CAM001",
    ):
        processor_module = importlib.import_module("core.keyframe_processor")
        fake_buffer = FakeBuffer()
        if pending_person_result is not None:
            fake_buffer.pending_person_result = pending_person_result
        fake_triton = triton or PersonOnlyTriton(mask_pixel_count)
        fake_person_db = person_db or FakePersonReidDb()
        processor = processor_module.KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=fake_triton,
            feature_buffer=fake_buffer,
            face_embedding_db=FakeFaceDb(),
            person_sbs_embedding_db=fake_person_db,
            person_upright_device_sns=upright_device_sns,
            person_face_reuse_mode=person_face_reuse_mode,
        )
        ctx = processor_module.KeyframeContext(
            sn=sn,
            timestamp=1710000000000,
            object_name=f"{sn}/picture/k.jpg",
            keyframe_id="11111111-1111-4111-8111-111111111111",
            is_pano_keyframe=is_pano_keyframe,
        )
        self.person_face_id = observed_face_id
        asyncio.run(
            processor.process_person_detection(
                ctx,
                key_frame if key_frame is not None else normal_person_test_image(),
                ctx.keyframe_id,
                42,
                ctx.object_name,
                oss_client or FakeOSS(),
                frame_faces=frame_faces,
            )
        )
        return fake_buffer, fake_triton, fake_person_db

    def _rotated_flow_fixture(self):
        # Keep the synthetic horizontal box within the existing structural
        # quality thresholds so downstream visual/feature stages are exercised.
        person_rect = {"x": 20, "y": 28, "w": 80, "h": 64}
        upright_rect = {"x": 28, "y": 20, "w": 64, "h": 80}
        points = person_pose_result(upright_rect)["keypoints"]
        center = np.array([60.0, 60.0], dtype=np.float32)
        xy = points[:, :2] - center
        points[:, 0] = center[0] + xy[:, 1]
        points[:, 1] = center[1] - xy[:, 0]
        return person_rect, points

    def _person_mask_for_quality_score(self, score):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        return int(algorithm_config.person.quality.general.max_mask_pixels * score / 10.0)

    def test_eps_normal_rotates_without_matching_manual_device_list(self):
        helper = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_rect, points = self._rotated_flow_fixture()

        for sn in (
            "541288EPS11C00165165AF62",
            "541288eps11c00165165af62",
            "541288EpS11C00165165AF62",
        ):
            for upright_device_sns in (frozenset(), frozenset({"MANUAL-CAM"})):
                with self.subTest(sn=sn, upright_device_sns=upright_device_sns):
                    triton = PersonOnlyTriton(
                        algorithm_config.person.quality.general.max_mask_pixels,
                        person_rect=person_rect,
                        pose_keypoints=points,
                    )
                    buffer, _, _ = self._run_person_quality_flow(
                        triton.mask_pixel_count,
                        triton=triton,
                        upright_device_sns=upright_device_sns,
                        sn=sn,
                    )

                    self.assertIn(
                        "旋转状态 applied",
                        buffer.person_detection_records[0]["info"],
                    )
                    self.assertEqual(self.person_face_crops[-1].shape[:2], (88, 72))

    def test_configured_non_eps_normal_rotates_but_unconfigured_does_not(self):
        helper = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_rect, points = self._rotated_flow_fixture()
        triton = PersonOnlyTriton(
            algorithm_config.person.quality.general.max_mask_pixels,
            person_rect=person_rect,
            pose_keypoints=points,
        )

        configured, _, _ = self._run_person_quality_flow(
            triton.mask_pixel_count,
            triton=triton,
            upright_device_sns=frozenset({"CAM001"}),
        )
        unconfigured, _, _ = self._run_person_quality_flow(
            triton.mask_pixel_count,
            triton=triton,
            upright_device_sns=frozenset(),
        )

        self.assertIn(
            "旋转状态 applied",
            configured.person_detection_records[0]["info"],
        )
        self.assertIn(
            "旋转状态 disabled",
            unconfigured.person_detection_records[0]["info"],
        )

    def test_pano_rotates_without_device_list(self):
        helper = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_rect, points = self._rotated_flow_fixture()
        triton = PersonOnlyTriton(
            algorithm_config.person.quality.general.max_mask_pixels,
            person_rect=person_rect,
            pose_keypoints=points,
        )

        buffer, _, _ = self._run_person_quality_flow(
            triton.mask_pixel_count,
            triton=triton,
            is_pano_keyframe=True,
            upright_device_sns=frozenset(),
        )

        self.assertIn(
            "旋转状态 applied",
            buffer.person_detection_records[0]["info"],
        )
        self.assertIs(
            buffer.person_detection_records[0]["isPanoKeyframe"],
            True,
        )

    def test_on_rotated_person_forces_local_face_flow_with_the_rotated_crop(self):
        helper = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_rect, points = self._rotated_flow_fixture()
        triton = PersonOnlyTriton(
            algorithm_config.person.quality.general.max_mask_pixels,
            person_rect=person_rect,
            pose_keypoints=points,
        )
        unmatched_frame_face = FrameFaceObservation(
            source_rect=(20, 45, 38, 75),
            confidence=0.9,
            source_landmarks=np.asarray(
                [[24, 52], [34, 52], [29, 58], [25, 66], [33, 66]],
                dtype=np.float32,
            ),
            orientation="front",
            assigned_face_id="0",
            completed_stage=FaceProcessingStage.IDENTITY,
        )
        self.person_face_id = "local-after-zero"

        buffer, _, _ = self._run_person_quality_flow(
            triton.mask_pixel_count,
            triton=triton,
            upright_device_sns=frozenset({"CAM001"}),
            person_face_reuse_mode="on",
            frame_faces=[unmatched_frame_face],
            observed_face_id="local-after-zero",
        )

        self.assertEqual(len(self.person_face_crops), 1)
        self.assertIn("旋转状态 applied", buffer.person_detection_records[0]["info"])
        self.assertEqual(
            buffer.person_recognition_records[0]["aiFaceId"],
            "local-after-zero",
        )
        self.assertEqual(self.person_face_crops[0].shape[:2], (88, 72))
        self.assertEqual(
            self._last_reuse_summary(),
            "Person face reuse frame summary: mode=on, people=1, "
            "frame_faces=1, bound=1, direct=0, partial=0, orientation=0, "
            "fallback=1, fallback_recovered=1, id_disagreement=1, "
            "id_shared_only=0, id_local_only=1, id_different=0, "
            "orientation_disagreement=0, "
            "fallback_reasons=person_rotated:1, rotated=1, "
            "shared_success_rotated=0, shared_success_unrotated=0, "
            "fallback_rotated=1, fallback_unrotated=0, recovery_rotated=1, "
            "recovery_unrotated=0",
        )


    def test_person_rotation_failure_falls_back_and_continues(self):
        helper = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_rect, points = self._rotated_flow_fixture()
        triton = PersonOnlyTriton(
            algorithm_config.person.quality.general.max_mask_pixels,
            person_rect=person_rect,
            pose_keypoints=points,
        )
        oss = CapturingOSS()

        with patch(
            "utils.keyframe.person_crop.cv2.warpAffine",
            side_effect=RuntimeError("boom"),
        ):
            buffer, _, _ = self._run_person_quality_flow(
                triton.mask_pixel_count,
                triton=triton,
                oss_client=oss,
                upright_device_sns=frozenset({"CAM001"}),
            )

        self.assertIn(
            "旋转状态 rotation_failed",
            buffer.person_detection_records[0]["info"],
        )
        self.assertEqual(len(oss.uploads), 1)
        self.assertEqual(len(triton.sbs_inputs), 1)

    def test_person_level_zero_does_not_recognize_or_insert(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_module = importlib.import_module("core.person_keyframe_processor")
        visual_calls = []
        original_visual_precheck = person_module.evaluate_person_visual_quality

        def track_visual_precheck(*args):
            visual_calls.append(args)
            return original_visual_precheck(*args)

        person_module.evaluate_person_visual_quality = track_visual_precheck
        fake_buffer, fake_triton, fake_person_db = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.recognition_min_score
            )
            - 1
        )

        self.assertEqual(fake_buffer.person_detection_records[0]["status"], 0)
        self.assertTrue(
            fake_buffer.person_detection_records[0]["crop_url"].startswith(
                "CAM001/person_crop/"
            )
        )
        self.assertNotEqual(
            fake_buffer.person_detection_records[0]["crop_url"],
            "CAM001/picture/k.jpg",
        )
        self.assertEqual(fake_buffer.person_detection_records[0]["confidence"], 0.8)
        self.assertEqual(fake_buffer.person_detection_records[0]["personQualityLevel"], 0)
        self.assertEqual(fake_triton.tbps_calls, 0)
        self.assertEqual(fake_person_db.search_calls, 0)
        self.assertEqual(len(fake_buffer.person_features), 0)
        self.assertEqual(len(fake_buffer.person_recognition_records), 0)
        self.assertEqual(visual_calls, [])

    def test_person_blurry_laplacian_precheck_drops_before_reid(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        gradient = np.linspace(60, 180, 120, dtype=np.uint8)
        blurred_image = np.repeat(gradient[None, :], 120, axis=0)
        blurred_image = np.repeat(blurred_image[:, :, None], 3, axis=2)

        fake_buffer, fake_triton, fake_person_db = self._run_person_quality_flow(
            algorithm_config.person.quality.general.max_mask_pixels,
            key_frame=blurred_image,
        )

        record = fake_buffer.person_detection_records[0]
        self.assertEqual(record["status"], 0)
        self.assertEqual(record["personQualityScore"], 0.0)
        self.assertEqual(record["personQualityLevel"], 0)
        self.assertTrue(record["crop_url"].startswith("CAM001/person_crop/"))
        self.assertIn("图像预检指标", record["info"])
        self.assertIn("Laplacian方差", record["info"])
        self.assertIn("Laplacian<10丢弃", record["info"])
        self.assertIn("图像模糊", record["info"])
        self.assertEqual(fake_triton.tbps_calls, 0)
        self.assertEqual(fake_person_db.search_calls, 0)
        self.assertEqual(len(fake_buffer.person_features), 0)
        self.assertEqual(len(fake_buffer.person_recognition_records), 0)

    def test_person_moderate_underexposure_identifies_without_insert(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        fake_buffer, fake_triton, fake_person_db = self._run_person_quality_flow(
            algorithm_config.person.quality.general.max_mask_pixels,
            key_frame=moderately_underexposed_person_test_image(),
        )

        record = fake_buffer.person_detection_records[0]
        self.assertEqual(record["status"], 1)
        self.assertEqual(record["personQualityLevel"], 1)
        self.assertEqual(record["personQualityAction"], "identify_only")
        self.assertEqual(record["cropScore"], 6.99)
        self.assertIn("underexposed", record["qualityReasons"])
        self.assertIn("暗像素占比", record["info"])
        self.assertIn("降级条件", record["info"])
        self.assertEqual(fake_triton.tbps_calls, 0)
        self.assertEqual(fake_triton.sbs_calls, 1)
        self.assertEqual(fake_person_db.search_calls, 1)
        self.assertEqual(len(fake_buffer.person_features), 0)
        self.assertEqual(len(fake_buffer.person_recognition_records), 1)

    def test_person_level_one_unmatched_recognizes_without_insert(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        fake_buffer, fake_triton, fake_person_db = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.recognition_min_score
            )
        )

        self.assertEqual(fake_buffer.person_detection_records[0]["status"], 1)
        self.assertEqual(fake_buffer.person_detection_records[0]["confidence"], 0.8)
        self.assertEqual(fake_buffer.person_detection_records[0]["personQualityLevel"], 1)
        self.assertEqual(
            fake_buffer.person_detection_records[0]["personQualityAction"],
            "identify_only",
        )
        self.assertIn("门控原因", fake_buffer.person_detection_records[0]["info"])
        self.assertIn(
            "基础mask分未达到建库门槛",
            fake_buffer.person_detection_records[0]["info"],
        )
        self.assertIn("maskScore", fake_buffer.person_detection_records[0])
        self.assertIn("cropScore", fake_buffer.person_detection_records[0])
        self.assertIsInstance(
            fake_buffer.person_detection_records[0]["qualityReasons"],
            list,
        )
        self.assertEqual(fake_triton.tbps_calls, 0)
        self.assertEqual(fake_triton.sbs_calls, 1)
        self.assertEqual(fake_person_db.search_calls, 1)
        self.assertEqual(len(fake_buffer.person_features), 0)
        self.assertTrue(
            fake_buffer.person_detection_records[0]["crop_url"].startswith(
                "CAM001/person_crop/"
            )
        )
        self.assertEqual(fake_buffer.person_recognition_records[0]["reid"], "0")
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidStatus"], 0)
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidState"], "")
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidDecision"],
            "UNASSIGNED",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidDecisionReason"],
            "no_compatible_class",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidSearchMode"],
            "ANONYMOUS_GLOBAL",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidRequiredMargin"],
            0.08,
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidEnrollmentState"],
            "",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidEnrollmentReason"],
            "",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["personQualityAction"],
            "identify_only",
        )
        self.assertIn("maskScore", fake_buffer.person_recognition_records[0])
        self.assertIn("cropScore", fake_buffer.person_recognition_records[0])
        self.assertIsInstance(
            fake_buffer.person_recognition_records[0]["qualityReasons"],
            list,
        )

    def test_person_level_one_matched_recognizes_without_insert(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        fake_buffer, _, fake_person_db = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.recognition_min_score
            ),
            person_db=FakePersonReidDb(reid="reid-1", score=0.99, hit_reid="reid-1"),
        )

        self.assertEqual(len(fake_buffer.person_features), 0)
        self.assertEqual(fake_buffer.person_recognition_records[0]["reid"], "reid-1")
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidStatus"], 2)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidDecision"],
            "MATCH_EXISTING",
        )
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidSource"], "MILVUS")
        self.assertIs(fake_buffer.person_recognition_records[0]["isNewReid"], False)
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidSupportCount"], 2)
        self.assertAlmostEqual(
            fake_buffer.person_recognition_records[0]["reidClassScore"],
            0.99,
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidState"],
            "ACTIVE",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidEnrollmentState"],
            "",
        )
        self.assertEqual(fake_person_db.last_threshold, 0.75)

    def test_person_level_two_pending_match_reuses_reid_and_inserts(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        fake_buffer, _, _ = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.enroll_min_score
            ),
            pending_person_result=("pending-reid", 0.99, "pending-reid"),
            observed_face_id="face-1",
        )

        self.assertEqual(fake_buffer.pending_person_search_calls, 1)
        self.assertEqual(fake_buffer.person_recognition_records[0]["reid"], "pending-reid")
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidStatus"], 2)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidState"],
            "ACTIVE",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidEnrollmentState"],
            "FORMAL_PENDING",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidEnrollmentReason"],
            "test_enrollment",
        )
        self.assertEqual(fake_buffer.person_recognition_records[0]["score"], 0.99)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["vectorHitReid"],
            "pending-reid",
        )
        self.assertEqual(len(fake_buffer.person_features), 1)
        self.assertEqual(len(fake_buffer.person_clip_features), 1)
        self.assertEqual(fake_buffer.person_features[0]["re_id"], "pending-reid")

    def test_person_level_two_unmatched_generates_reid_and_inserts(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        fake_buffer, fake_triton, fake_person_db = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.enroll_min_score
            ),
            observed_face_id="face-1",
        )

        self.assertEqual(fake_buffer.person_detection_records[0]["personQualityLevel"], 2)
        self.assertEqual(
            fake_buffer.person_detection_records[0]["personQualityAction"],
            "build_gallery",
        )
        self.assertEqual(fake_triton.tbps_calls, 1)
        self.assertEqual(fake_triton.sbs_calls, 1)
        self.assertEqual(fake_person_db.search_calls, 1)
        self.assertEqual(fake_person_db.last_embedding.shape, (2048,))
        self.assertTrue(np.all(np.isfinite(fake_person_db.last_embedding)))
        self.assertAlmostEqual(
            float(np.linalg.norm(fake_person_db.last_embedding)),
            1.0,
            places=6,
        )
        self.assertEqual(len(fake_buffer.person_features), 1)
        self.assertNotEqual(fake_buffer.person_recognition_records[0]["reid"], "")
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidStatus"], 1)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidDecision"],
            "CREATE_NEW",
        )
        self.assertEqual(fake_buffer.person_recognition_records[0]["reidSource"], "NEW")
        self.assertIs(fake_buffer.person_recognition_records[0]["isNewReid"], True)
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["canonicalFaceId"],
            "face-1",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidState"],
            "ACTIVE",
        )
        self.assertEqual(
            fake_buffer.person_recognition_records[0]["reidEnrollmentState"],
            "FORMAL_PENDING",
        )
        self.assertEqual(len(fake_buffer.person_clip_features), 1)

    def test_pano_person_context_propagates_to_records_and_feature_handoffs(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        fake_buffer, _, _ = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.enroll_min_score
            ),
            observed_face_id="face-1",
            is_pano_keyframe=True,
        )

        self.assertIs(fake_buffer.person_detection_records[0]["isPanoKeyframe"], True)
        self.assertIs(fake_buffer.person_recognition_records[0]["isPanoKeyframe"], True)
        self.assertIs(fake_buffer.person_clip_features[0]["is_pano_keyframe"], True)
        self.assertIs(fake_buffer.person_features[0]["is_pano_keyframe"], True)

    def test_person_crop_is_shared_across_quality_oss_sbs_clip_and_face(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        person_module = importlib.import_module("core.person_keyframe_processor")
        person_rect, points = self._rotated_flow_fixture()
        triton = PersonOnlyTriton(
            algorithm_config.person.quality.general.max_mask_pixels,
            person_rect=person_rect,
            pose_keypoints=points,
        )
        oss = CapturingOSS()
        encoded_images = []
        visual_inputs = []
        original_encode = person_module.encode_image_to_bytes
        original_visual = person_module.evaluate_person_visual_quality
        encoded_bytes = b"shared-person-crop\x00"

        def capture_encode(image):
            encoded_images.append(np.asarray(image).copy())
            return encoded_bytes

        def capture_visual(image, mask):
            visual_inputs.append(
                (
                    np.asarray(image).copy(),
                    None if mask is None else np.asarray(mask).copy(),
                )
            )
            return original_visual(image, mask)

        person_module.encode_image_to_bytes = capture_encode
        person_module.evaluate_person_visual_quality = capture_visual
        try:
            fake_buffer, _, _ = self._run_person_quality_flow(
                triton.mask_pixel_count,
                triton=triton,
                oss_client=oss,
                observed_face_id="face-1",
                upright_device_sns=frozenset({"CAM001"}),
            )
        finally:
            person_module.encode_image_to_bytes = original_encode
            person_module.evaluate_person_visual_quality = original_visual

        self.assertEqual(len(encoded_images), 1)
        person_crop = encoded_images[0]
        self.assertNotEqual(person_crop.shape[0], 3 * person_crop.shape[1])
        self.assertEqual(
            fake_buffer.person_detection_records[0]["letterboxPaddingRatio"],
            0.0,
        )
        self.assertIn(
            "旋转状态 applied",
            fake_buffer.person_detection_records[0]["info"],
        )
        self.assertEqual(len(visual_inputs), 1)
        np.testing.assert_array_equal(visual_inputs[0][0], person_crop)
        self.assertEqual(visual_inputs[0][1].shape, person_crop.shape[:2])
        self.assertEqual(len(self.person_face_crops), 1)
        np.testing.assert_array_equal(self.person_face_crops[0], person_crop)

        self.assertIs(triton.sbs_inputs[0], encoded_bytes)
        self.assertIs(triton.tbps_inputs[0], encoded_bytes)
        self.assertIs(oss.uploads[0][0], encoded_bytes)
        self.assertEqual(
            fake_buffer.person_features[0]["crop_url"],
            oss.uploads[0][1],
        )
        expected_rect = person_rect
        self.assertEqual(fake_buffer.person_features[0]["person_rect"], expected_rect)
        self.assertEqual(
            fake_buffer.person_detection_records[0]["peopleRect"],
            expected_rect,
        )

    def test_person_sbs_failure_keeps_clip_but_does_not_allocate_id(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        triton = PersonOnlyTriton(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.enroll_min_score
            ),
            sbs_error=RuntimeError("sbs unavailable"),
        )
        fake_buffer, _, fake_person_db = self._run_person_quality_flow(
            triton.mask_pixel_count,
            triton=triton,
            observed_face_id="face-1",
        )

        self.assertEqual(fake_person_db.search_calls, 0)
        self.assertEqual(len(fake_buffer.person_features), 0)
        self.assertEqual(len(fake_buffer.person_clip_features), 1)
        self.assertEqual(fake_buffer.person_clip_features[0]["re_id"], "0")
        record = fake_buffer.person_recognition_records[0]
        self.assertEqual(record["reid"], "0")
        self.assertEqual(record["reidStatus"], 0)
        self.assertEqual(record["reidState"], "")
        self.assertEqual(record["reidDecision"], "SKIPPED")
        self.assertEqual(record["reidDecisionReason"], "sbs_inference_failed")
        self.assertEqual(record["reidSearchMode"], "NOT_EXECUTED")
        self.assertEqual(record["reidSource"], "")
        self.assertIs(record["isNewReid"], False)
        self.assertIn("SBS 推理失败", record["info"])

    def test_person_sbs_invalid_results_expose_reason_codes(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        mask_pixel_count = self._person_mask_for_quality_score(
            algorithm_config.person.quality.general.enroll_min_score
        )
        cases = [
            (object(), "sbs_embedding_invalid"),
            (FeatureResult(np.ones(3, dtype=np.float32)), "sbs_embedding_invalid"),
            (FeatureResult("bad"), "sbs_embedding_invalid"),
            (
                FeatureResult(np.asarray([object()], dtype=object)),
                "sbs_embedding_invalid",
            ),
            (
                FeatureResult(
                    np.asarray([OverflowingFloat()], dtype=object)
                ),
                "sbs_embedding_invalid",
            ),
            (FeatureResult(np.zeros(2048, dtype=np.float32)), "sbs_embedding_zero"),
        ]

        for sbs_result, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                fake_buffer, _, fake_person_db = self._run_person_quality_flow(
                    mask_pixel_count,
                    triton=PersonOnlyTriton(
                        mask_pixel_count,
                        sbs_result=sbs_result,
                    ),
                )
                record = fake_buffer.person_recognition_records[0]
                self.assertEqual(fake_person_db.search_calls, 0)
                self.assertEqual(record["reid"], "0")
                self.assertEqual(record["reidDecision"], "SKIPPED")
                self.assertEqual(record["reidDecisionReason"], expected_reason)
                self.assertEqual(record["reidSearchMode"], "NOT_EXECUTED")
                self.assertEqual(record["reidTop1Score"], 0.0)
                self.assertEqual(record["sameFrameExcludedReidCount"], 0)

    def test_person_clip_invalid_vectors_do_not_escape_feature_flow(self):
        helper_module = importlib.import_module(
            "utils.keyframe.keyframe_flow_helpers"
        )
        mask_pixel_count = self._person_mask_for_quality_score(
            algorithm_config.person.quality.general.enroll_min_score
        )
        invalid_vectors = [
            "bad",
            np.asarray([object()], dtype=object),
            np.asarray([OverflowingFloat()], dtype=object),
        ]

        for vector in invalid_vectors:
            with self.subTest(vector=repr(vector)):
                fake_buffer, _, fake_person_db = self._run_person_quality_flow(
                    mask_pixel_count,
                    triton=PersonOnlyTriton(
                        mask_pixel_count,
                        clip_result=FeatureResult(vector),
                    ),
                    observed_face_id="face-1",
                )
                record = fake_buffer.person_recognition_records[0]
                self.assertEqual(fake_person_db.search_calls, 1)
                self.assertEqual(len(fake_buffer.person_features), 1)
                self.assertEqual(len(fake_buffer.person_clip_features), 0)
                self.assertIn("CLIP embedding 无效", record["info"])

    def test_person_clip_failure_does_not_block_sbs_assignment(self):
        helper_module = importlib.import_module("utils.keyframe.keyframe_flow_helpers")
        triton = PersonOnlyTriton(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.enroll_min_score
            ),
            clip_error=RuntimeError("clip unavailable"),
        )
        fake_buffer, _, fake_person_db = self._run_person_quality_flow(
            triton.mask_pixel_count,
            triton=triton,
            observed_face_id="face-1",
        )

        self.assertEqual(fake_person_db.search_calls, 1)
        self.assertEqual(len(fake_buffer.person_features), 1)
        self.assertEqual(len(fake_buffer.person_clip_features), 0)
        record = fake_buffer.person_recognition_records[0]
        self.assertNotEqual(record["reid"], "0")
        self.assertEqual(record["reidStatus"], 1)
        self.assertEqual(record["reidState"], "ACTIVE")
        self.assertIn("CLIP 推理失败", record["info"])

    def test_detection_records_do_not_fallback_to_keyframe_when_crop_upload_fails(self):
        face_buffer, face_triton, face_db = self._run_face_quality_flow(
            6.5,
            oss_client=FailingOSS(),
        )

        self.assertEqual(face_buffer.face_detection_records[0]["crop_url"], "")
        self.assertEqual(face_triton.feature_calls, 0)
        self.assertEqual(face_db.search_calls, 0)
        self.assertEqual(len(face_buffer.face_recognition_records), 0)

        person_buffer, person_triton, person_db = self._run_person_quality_flow(
            self._person_mask_for_quality_score(
                algorithm_config.person.quality.general.enroll_min_score
            ),
            oss_client=FailingOSS(),
            observed_face_id="face-1",
        )

        self.assertEqual(person_buffer.person_detection_records[0]["crop_url"], "")
        self.assertEqual(person_triton.tbps_calls, 1)
        self.assertEqual(person_triton.sbs_calls, 1)
        self.assertEqual(person_db.search_calls, 1)
        self.assertEqual(person_buffer.person_features[0]["crop_url"], "")


if __name__ == "__main__":
    unittest.main()
