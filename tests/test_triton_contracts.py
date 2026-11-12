import unittest
from dataclasses import FrozenInstanceError
from typing import get_type_hints

import numpy as np

from utils.person_mask import PersonMask

from core.infer.contracts import MODEL_CONTRACTS, ModelContract, TensorContract
from core.infer.results import (
    FaceDetection,
    FaceQualityResult,
    FeatureResult,
    PersonPose,
    PersonSegment,
    Rect,
)


EXPECTED_MODEL_KEYS = {
    "clip_image",
    "face_rec",
    "tbps_image",
    "sbs_image",
    "face_qa",
    "face_kps",
    "face_det",
    "person_seg",
    "person_pose",
}

EXPECTED_CONTRACTS = {
    "clip_image": (
        "clip_ensemble_trt",
        (("IMAGE_BYTES", "BYTES", (1,)),),
        (("unnorm_image_features", "FP32", (512,)),),
    ),
    "face_rec": (
        "face_recognition_pipeline",
        (
            ("RAW_IMAGE", "UINT8", (-1, -1, 3)),
            ("LANDMARK", "FP32", (5, 2)),
        ),
        (("EMBEDDING", "FP32", (512,)),),
    ),
    "tbps_image": (
        "TBPS_clip_image_ensemble",
        (("IMAGE_BYTES", "BYTES", (1,)),),
        (("image_feat", "FP32", (512,)),),
    ),
    "sbs_image": (
        "sbs_r50_ibn_image_ensemble",
        (("IMAGE_BYTES", "BYTES", (1,)),),
        (("embedding", "FP32", (2048,)),),
    ),
    "face_qa": (
        "face_qa_ensemble",
        (
            ("RAW_FACE_IMAGE", "UINT8", (-1, -1, 3)),
            ("LANDMARK", "FP32", (5, 2)),
        ),
        (("quality_score", "FP32", (1,)),),
    ),
    "face_kps": (
        "face_keypoint_2d106_ensemble",
        (
            ("RAW_IMAGE", "UINT8", (-1, -1, 3)),
            ("BBOX", "FP32", (4,)),
        ),
        (("LANDMARKS_106", "FP32", (106, 2)),),
    ),
    "face_det": (
        "yolov8n_face_ensemble",
        (("RAW_IMAGE", "UINT8", (-1, -1, 3)),),
        (
            ("DETECTIONS", "FP32", (100, 6)),
            ("NUM_DETECTIONS", "INT32", (1,)),
        ),
    ),
    "person_seg": (
        "yolo26s_seg_ensemble",
        (("RAW_IMAGE", "UINT8", (-1, -1, 3)),),
        (
            ("DETECTIONS", "FP32", (-1, 6)),
            ("MASK_CROPS_PACKED", "UINT8", (-1, 160, 20)),
            ("NUM_DETECTIONS", "INT32", (1,)),
        ),
    ),
    "person_pose": (
        "yolo26m_pose_ensemble",
        (("RAW_IMAGE", "UINT8", (-1, -1, 3)),),
        (
            ("DETECTIONS", "FP32", (100, 6)),
            ("KEYPOINTS", "FP32", (100, 17, 3)),
            ("NUM_DETECTIONS", "INT32", (1,)),
        ),
    ),
}


def tensor_signature(contract):
    return tuple((tensor.name, tensor.datatype, tensor.dims) for tensor in contract)


class TritonContractRegistryTests(unittest.TestCase):
    def test_registry_contains_exactly_the_supported_models(self):
        self.assertEqual(set(MODEL_CONTRACTS), EXPECTED_MODEL_KEYS)

    def test_each_model_declares_versioned_non_empty_io(self):
        for key, contract in MODEL_CONTRACTS.items():
            with self.subTest(key=key):
                expected_version = {
                    "person_seg": 3,
                    "face_qa": 2,
                    "face_det": 2,
                    "person_pose": 2,
                    "sbs_image": 2,
                    "tbps_image": 2,
                }.get(key, 1)
                self.assertEqual(
                    contract.api_contract_version, expected_version
                )
                self.assertTrue(contract.inputs)
                self.assertTrue(contract.outputs)

    def test_person_segment_declares_batch_local_detection_and_mask_capacity(self):
        contract = MODEL_CONTRACTS["person_seg"]

        self.assertEqual(contract.output("DETECTIONS").dims, (-1, 6))
        self.assertEqual(
            contract.output("MASK_CROPS_PACKED").dims, (-1, 160, 20)
        )
        self.assertEqual(contract.output("NUM_DETECTIONS").dims, (1,))
        self.assertEqual(contract.api_contract_version, 3)

    def test_registry_matches_the_exact_wire_contracts(self):
        for key, (model_name, inputs, outputs) in EXPECTED_CONTRACTS.items():
            with self.subTest(key=key):
                contract = MODEL_CONTRACTS[key]
                self.assertEqual(contract.key, key)
                self.assertEqual(contract.model_name, model_name)
                self.assertEqual(tensor_signature(contract.inputs), inputs)
                self.assertEqual(tensor_signature(contract.outputs), outputs)

    def test_model_names_are_unique(self):
        names = [contract.model_name for contract in MODEL_CONTRACTS.values()]

        self.assertEqual(len(names), len(set(names)))

    def test_contract_lookup_returns_named_tensor(self):
        contract = MODEL_CONTRACTS["face_rec"]

        self.assertEqual(contract.input("LANDMARK").dims, (5, 2))
        self.assertEqual(contract.output("EMBEDDING").datatype, "FP32")

    def test_contract_values_are_frozen(self):
        tensor_contract = MODEL_CONTRACTS["clip_image"].inputs[0]
        model_contract = MODEL_CONTRACTS["clip_image"]

        with self.assertRaises(FrozenInstanceError):
            tensor_contract.datatype = "UINT8"
        with self.assertRaises(FrozenInstanceError):
            model_contract.model_name = "another_model"

        self.assertIsInstance(tensor_contract, TensorContract)
        self.assertIsInstance(model_contract, ModelContract)


class TritonResultDeclarationTests(unittest.TestCase):
    def test_typed_result_mappings_declare_exact_required_fields(self):
        expected_hints = {
            Rect: {"x": int, "y": int, "w": int, "h": int},
            FaceDetection: {
                "xyxy": list[float],
                "conf": float,
                "cls": int,
            },
            PersonSegment: {
                "xyxy": list[float],
                "rect": Rect,
                "conf": float,
                "cls": int,
                "maskPixelCount": int,
                "mask": PersonMask | np.ndarray,
            },
            PersonPose: {
                "xyxy": list[float],
                "rect": Rect,
                "conf": float,
                "cls": int,
                "keypoints": np.ndarray | None,
            },
        }

        for result_type, expected in expected_hints.items():
            with self.subTest(result_type=result_type.__name__):
                self.assertEqual(get_type_hints(result_type), expected)
                self.assertEqual(result_type.__required_keys__, frozenset(expected))

    def test_typed_result_mappings_accept_the_public_fields(self):
        rect = Rect(x=1, y=2, w=3, h=4)
        face = FaceDetection(xyxy=[1.0, 2.0, 3.0, 4.0], conf=0.9, cls=0)
        mask = np.ones((2, 2), dtype=np.uint8)
        segment = PersonSegment(
            xyxy=[1.0, 2.0, 3.0, 4.0],
            rect=rect,
            conf=0.8,
            cls=1,
            maskPixelCount=4,
            mask=mask,
        )
        pose = PersonPose(
            xyxy=[1.0, 2.0, 3.0, 4.0],
            rect=rect,
            conf=0.7,
            cls=2,
            keypoints=None,
        )

        self.assertEqual(face["xyxy"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(segment["maskPixelCount"], 4)
        self.assertIs(segment["mask"], mask)
        self.assertIsNone(pose["keypoints"])

    def test_value_results_hold_numpy_data_and_are_frozen(self):
        vector = np.array([1.0, 2.0], dtype=np.float32)
        feature = FeatureResult(vector=vector)
        quality = FaceQualityResult(score=0.75)

        self.assertIs(feature.vector, vector)
        self.assertEqual(quality.score, 0.75)
        with self.assertRaises(FrozenInstanceError):
            feature.vector = np.array([], dtype=np.float32)
        with self.assertRaises(FrozenInstanceError):
            quality.score = 0.5


if __name__ == "__main__":
    unittest.main()
