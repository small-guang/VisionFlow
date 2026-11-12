"""Behavioral coverage for the algorithm configuration at consumer boundaries."""

import importlib
import inspect
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from core.algorithm_config import algorithm_config
from utils.reid import person_face_recognition as person_face
from utils.reid import reid_models, reid_resolution
from utils.reid.reid_rules import ReidClassFacts


class _NoopBuffer:
    def __init__(self, **kwargs):
        from utils.buffers.background_writer import WriteQueue

        self.queue = WriteQueue(unit="vector_item")
        self.pending = {}

    async def flush(self):
        pass


def _load_feature_buffer():
    db_stub = types.ModuleType("db.zilliz")
    for name in (
        "face_embedding_db",
        "person_clip_embedding_db",
        "person_sbs_embedding_db",
        "picture_clip_embedding_db",
    ):
        setattr(db_stub, name, None)
    settings_stub = types.ModuleType("core.config")
    settings_stub.settings = types.SimpleNamespace(**{
        name: "test"
        for name in (
            "KAFKA_FACE_DETECTION_RECORD_TOPIC",
            "KAFKA_FACE_RECOGNITION_RECORD_TOPIC",
            "KAFKA_PERSON_DETECTION_RECORD_TOPIC",
            "KAFKA_PERSON_RECOGNITION_RECORD_TOPIC",
            "KAFKA_VECTOR_RECORD_TOPIC",
        )
    })
    sbs_stub = types.ModuleType("utils.buffers.sbs_feature_buffer")
    sbs_stub.SbsFeatureBuffer = _NoopBuffer
    tbps_stub = types.ModuleType("utils.buffers.tbps_feature_buffer")
    tbps_stub.TbpsFeatureBuffer = _NoopBuffer
    module_name = "utils.buffers.buffer"
    previous = sys.modules.get(module_name)
    with patch.dict(sys.modules, {
        "db.zilliz": db_stub,
        "core.config": settings_stub,
        "utils.buffers.sbs_feature_buffer": sbs_stub,
        "utils.buffers.tbps_feature_buffer": tbps_stub,
    }):
        module = importlib.import_module(module_name)
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous
    return module.FeatureBuffer


class AlgorithmConfigRuntimeTests(unittest.TestCase):
    def test_code_owned_contract_values_remain_fixed(self):
        self.assertEqual(person_face.UNMATCHED_FACE_ID, "0")
        self.assertEqual(person_face.PERSON_FACE_NORMALIZED_EMBEDDING_NORM_TOLERANCE, 1e-4)
        self.assertEqual(person_face.PERSON_FACE_HEAD_KEYPOINT_INDICES, (0, 1, 2, 3, 4))

    def test_pending_face_search_requires_explicit_threshold(self):
        signature = inspect.signature(
            _load_feature_buffer().asearch_pending_faces_with_hit_info
        )
        with self.assertRaises(TypeError):
            signature.bind(object(), np.array([1.0], dtype=np.float32))

    def test_reid_class_score_uses_configured_weights(self):
        facts = ReidClassFacts(
            reid="person-1",
            query_face="face-1",
            query_device="camera-1",
            canonical_face="face-1",
            selected=(reid_models.ReidCandidate(reid="person-1", score=0.8),),
            best_score=0.8,
            mean_score=0.7,
            raw_runner_score=0.0,
            raw_margin=0.8,
        )
        score = reid_resolution.ReidClassResolver._score(facts, "accepted")
        self.assertAlmostEqual(score.score, 0.60 * 0.80 + 0.40 * 0.70)

        classification = algorithm_config.reid.classification.model_copy(update={
            "best_score_weight": 0.25,
            "mean_score_weight": 0.75,
        })
        reid = algorithm_config.reid.model_copy(update={"classification": classification})
        selected = algorithm_config.model_copy(update={"reid": reid})
        with patch.object(reid_resolution, "algorithm_config", selected):
            alternate = reid_resolution.ReidClassResolver._score(facts, "accepted")
        self.assertAlmostEqual(alternate.score, 0.25 * 0.80 + 0.75 * 0.70)

    def test_pose_head_at_configured_expanded_face_boundary_is_owned(self):
        face = {"xyxy": [40, 40, 60, 60], "conf": 0.8}
        mask = np.ones((100, 100), dtype=np.uint8)
        pose = np.zeros((17, 3), dtype=np.float32)
        margin = algorithm_config.person_face.ownership.head_bbox_margin_ratio
        boundary = 40 - 20 * margin
        for axis, at_edge, beyond_edge in (
            ("x", (boundary, 50), (boundary - 0.1, 50)),
            ("y", (50, boundary), (50, boundary - 0.1)),
        ):
            with self.subTest(axis=axis):
                pose[0] = (*at_edge, 1)
                owned = person_face.select_owned_face_detection(
                    [face], (100, 100, 3), mask, pose
                )
                self.assertIsNotNone(owned)
                self.assertEqual(owned.bbox, (40, 40, 60, 60))

                pose[0] = (*beyond_edge, 1)
                outside = person_face.select_owned_face_detection(
                    [face], (100, 100, 3), mask, pose
                )
                self.assertIsNone(outside)


if __name__ == "__main__":
    unittest.main()
