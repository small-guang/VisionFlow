import unittest
from typing import get_type_hints
from unittest.mock import patch

import numpy as np

import core.infer.adapters.person as person_adapter_module
from core.infer.adapters import build_adapters
from core.infer.adapters.base import ModelAdapter
from core.infer.batching import NormalizedRequest
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import InferenceInputError, InferenceResponseError
from core.infer.results import FaceQualityResult, FeatureResult


def raw_image(height=10, width=12):
    return np.arange(height * width * 3, dtype=np.uint8).reshape(
        height, width, 3
    )


def batched_detections():
    values = np.zeros((2, 100, 6), dtype=np.float32)
    values[0, 0] = [1, 1, 4, 4, 0.9, 0]
    values[0, 1] = [2, 2, 8, 8, 0.8, 1]
    values[1, 0] = [3, 3, 9, 9, 0.7, 0]
    values[1, 1] = [4, 4, 7, 7, 0.6, 0]
    return values


def batched_counts(first=1, second=2):
    return np.asarray([[first], [second]], dtype=np.int32)


def batched_segment_outputs():
    detections = np.zeros((2, 2, 6), dtype=np.float32)
    detections[0, 0] = [1, 1, 4, 4, 0.9, 0]
    detections[1, 0] = [3, 3, 9, 9, 0.8, 0]
    detections[1, 1] = [4, 4, 7, 7, 0.7, 0]
    crops = np.zeros((2, 2, 160, 20), dtype=np.uint8)
    crops[0, 0] = 255
    crops[1, 0] = 255
    crops[1, 1] = 255
    return {
        "DETECTIONS": detections,
        "MASK_CROPS_PACKED": crops,
        "NUM_DETECTIONS": batched_counts(1, 2),
    }


class AdapterRegistryTests(unittest.TestCase):
    def test_registry_contains_exactly_the_nine_contract_keys(self):
        adapters = build_adapters()

        self.assertEqual(set(adapters), set(MODEL_CONTRACTS))
        for key, adapter in adapters.items():
            self.assertIs(adapter.contract, MODEL_CONTRACTS[key])
            for method in (
                "normalize",
                "pack",
                "validate_outputs",
                "unpack_one",
            ):
                self.assertTrue(callable(getattr(adapter, method, None)))

    def test_base_adapter_declares_the_stable_interface_and_registry_type(self):
        expected_methods = {
            "normalize",
            "pack",
            "validate_outputs",
            "unpack_one",
        }

        self.assertEqual(
            get_type_hints(build_adapters)["return"],
            dict[str, ModelAdapter],
        )
        self.assertTrue(expected_methods <= set(ModelAdapter.__dict__))
        self.assertTrue(
            all(
                isinstance(adapter, ModelAdapter)
                for adapter in build_adapters().values()
            )
        )


class AdapterInputTests(unittest.TestCase):
    def setUp(self):
        self.adapters = build_adapters()

    def test_raw_and_multi_input_pack_shapes_are_explicitly_batched(self):
        image_a = raw_image()
        image_b = raw_image().copy()
        face_rec = self.adapters["face_rec"]
        face_qa = self.adapters["face_qa"]
        face_kps = self.adapters["face_kps"]
        raw_adapter = self.adapters["face_det"]

        raw_requests = [
            raw_adapter.normalize(image_a),
            raw_adapter.normalize(image_b),
        ]
        landmarks = np.arange(10, dtype=np.float32).reshape(5, 2)
        face_requests = [
            face_rec.normalize(image_a, landmarks),
            face_rec.normalize(image_b, landmarks.copy()),
        ]
        try:
            quality_requests = [
                face_qa.normalize(image_a, landmarks),
                face_qa.normalize(image_b, landmarks.copy()),
            ]
        except TypeError as exc:
            self.fail(f"face quality adapter must accept five landmarks: {exc}")
        bbox = np.asarray([1, 2, 8, 9], dtype=np.float32)
        keypoint_requests = [
            face_kps.normalize(image_a, bbox),
            face_kps.normalize(image_b, bbox.copy()),
        ]

        self.assertEqual(
            raw_adapter.pack(raw_requests)["RAW_IMAGE"].shape,
            (2, 10, 12, 3),
        )
        packed_face = face_rec.pack(face_requests)
        self.assertEqual(packed_face["RAW_IMAGE"].shape, (2, 10, 12, 3))
        self.assertEqual(packed_face["LANDMARK"].shape, (2, 5, 2))
        packed_quality = face_qa.pack(quality_requests)
        self.assertEqual(
            packed_quality["RAW_FACE_IMAGE"].shape, (2, 10, 12, 3)
        )
        self.assertEqual(packed_quality["LANDMARK"].shape, (2, 5, 2))
        packed_keypoints = face_kps.pack(keypoint_requests)
        self.assertEqual(packed_keypoints["RAW_IMAGE"].shape, (2, 10, 12, 3))
        self.assertEqual(packed_keypoints["BBOX"].shape, (2, 4))

    def test_clip_bytes_pack_as_object_tensor_with_shape_b_1(self):
        adapter = self.adapters["clip_image"]
        requests = [adapter.normalize(b"abc"), adapter.normalize(b"defgh")]

        packed = adapter.pack(requests)["IMAGE_BYTES"]

        self.assertEqual(packed.shape, (2, 1))
        self.assertEqual(packed.dtype, np.dtype(object))
        self.assertEqual(packed.tolist(), [[b"abc"], [b"defgh"]])

    def test_uint8_encoded_inputs_share_a_key_and_pack_exact_bytes(self):
        for key in ("tbps_image", "sbs_image"):
            with self.subTest(key=key):
                adapter = self.adapters[key]
                first = adapter.normalize(
                    np.asarray([1, 2, 3], dtype=np.uint8)
                )
                second = adapter.normalize(
                    np.asarray([1, 2, 3, 4, 5], dtype=np.uint8)
                )

                self.assertEqual(first.compatibility_key, second.compatibility_key)
                packed = adapter.pack([first, second])["IMAGE_BYTES"]
                self.assertEqual(packed.shape, (2, 1))
                self.assertEqual(packed.dtype, object)
                self.assertEqual(packed[0, 0], b"\x01\x02\x03")
                self.assertEqual(packed[1, 0], b"\x01\x02\x03\x04\x05")

    def test_raw_compatibility_key_includes_every_input_shape_and_dtype(self):
        face_rec = self.adapters["face_rec"]
        landmarks = np.zeros((5, 2), dtype=np.float32)
        first = face_rec.normalize(raw_image(10, 12), landmarks)
        different_image = face_rec.normalize(raw_image(11, 12), landmarks)

        self.assertNotEqual(first.compatibility_key, different_image.compatibility_key)
        self.assertIn(("RAW_IMAGE", (10, 12, 3), "uint8"), first.compatibility_key)
        self.assertIn(("LANDMARK", (5, 2), "float32"), first.compatibility_key)

    def test_pack_copies_noncontiguous_inputs_to_contiguous_arrays(self):
        face_rec = self.adapters["face_rec"]
        image = raw_image()[:, ::-1, :]
        landmarks = np.arange(20, dtype=np.float32).reshape(10, 2)[::2]

        request = face_rec.normalize(image, landmarks)

        packed = face_rec.pack([request])
        self.assertTrue(packed["RAW_IMAGE"].flags.c_contiguous)
        self.assertTrue(packed["LANDMARK"].flags.c_contiguous)
        np.testing.assert_array_equal(packed["RAW_IMAGE"][0], image)
        np.testing.assert_array_equal(packed["LANDMARK"][0], landmarks)

    def test_raw_image_context_keeps_only_shape_metadata(self):
        image = raw_image()
        for key in ("face_det", "person_seg", "person_pose"):
            with self.subTest(key=key):
                request = self.adapters[key].normalize(image)
                self.assertEqual(request.context, {"image_shape": image.shape})
                self.assertEqual(set(request.context), {"image_shape"})
                self.assertIsNot(request.context.get("image"), image)

    def test_normalize_rejects_empty_wrong_dtype_and_wrong_shape_inputs(self):
        cases = [
            ("face_det empty", lambda: self.adapters["face_det"].normalize(
                np.empty((0, 12, 3), dtype=np.uint8)
            )),
            ("face_det dtype", lambda: self.adapters["face_det"].normalize(
                np.zeros((10, 12, 3), dtype=np.float32)
            )),
            ("face_det shape", lambda: self.adapters["face_det"].normalize(
                np.zeros((10, 12), dtype=np.uint8)
            )),
            ("clip empty", lambda: self.adapters["clip_image"].normalize(b"")),
            ("clip type", lambda: self.adapters["clip_image"].normalize("abc")),
            ("encoded empty", lambda: self.adapters["sbs_image"].normalize(
                np.asarray([], dtype=np.uint8)
            )),
            ("encoded dtype", lambda: self.adapters["sbs_image"].normalize(
                np.asarray([1, 2], dtype=np.int32)
            )),
            ("encoded shape", lambda: self.adapters["sbs_image"].normalize(
                np.zeros((2, 2), dtype=np.uint8)
            )),
            ("landmark dtype", lambda: self.adapters["face_rec"].normalize(
                raw_image(), np.zeros((5, 2), dtype=np.float64)
            )),
            ("landmark shape", lambda: self.adapters["face_rec"].normalize(
                raw_image(), np.zeros((10,), dtype=np.float32)
            )),
            ("bbox dtype", lambda: self.adapters["face_kps"].normalize(
                raw_image(), np.zeros(4, dtype=np.float64)
            )),
            ("bbox shape", lambda: self.adapters["face_kps"].normalize(
                raw_image(), np.zeros((2, 2), dtype=np.float32)
            )),
        ]

        for name, operation in cases:
            with self.subTest(name=name), self.assertRaises(InferenceInputError):
                operation()

    def test_pack_revalidates_mutated_normalized_request_inputs(self):
        clip = self.adapters["clip_image"]
        clip_request = clip.normalize(b"abc")
        clip_request.inputs["IMAGE_BYTES"] = [b"abc"]

        encoded = self.adapters["sbs_image"]
        encoded_request = encoded.normalize(
            np.asarray([1, 2, 3], dtype=np.uint8)
        )
        encoded_request.inputs["IMAGE_BYTES"] = [1, 2, 3]

        raw = self.adapters["face_det"]
        raw_object = raw.normalize(raw_image())
        raw_object.inputs["RAW_IMAGE"] = object()
        raw_object_array = raw.normalize(raw_image())
        raw_object_array.inputs["RAW_IMAGE"] = np.empty(
            raw_image().shape, dtype=object
        )

        face_rec = self.adapters["face_rec"]
        landmarks = np.zeros((5, 2), dtype=np.float32)
        nonfinite_float = face_rec.normalize(raw_image(), landmarks)
        nonfinite_float.inputs["LANDMARK"][0, 0] = np.nan

        cases = [
            ("clip list", clip, clip_request),
            ("encoded list", encoded, encoded_request),
            ("raw object", raw, raw_object),
            ("raw object array", raw, raw_object_array),
            ("float NaN", face_rec, nonfinite_float),
        ]
        for name, adapter, request in cases:
            with self.subTest(name=name), self.assertRaises(
                InferenceInputError
            ):
                adapter.pack([request])


class AdapterOutputTests(unittest.TestCase):
    def setUp(self):
        self.adapters = build_adapters()
        self.raw_requests = {
            key: [
                self.adapters[key].normalize(raw_image()),
                self.adapters[key].normalize(raw_image()),
            ]
            for key in ("face_det", "person_seg", "person_pose")
        }
        landmarks = np.zeros((5, 2), dtype=np.float32)
        self.raw_requests["face_qa"] = [
            self.adapters["face_qa"].normalize(raw_image(), landmarks),
            self.adapters["face_qa"].normalize(
                raw_image(), landmarks.copy()
            ),
        ]

    def test_all_embedding_adapters_return_b_two_feature_results(self):
        cases = {
            "clip_image": ("unnorm_image_features", 512),
            "face_rec": ("EMBEDDING", 512),
            "tbps_image": ("image_feat", 512),
            "sbs_image": ("embedding", 2048),
        }
        for key, (output_name, width) in cases.items():
            with self.subTest(key=key):
                adapter = self.adapters[key]
                outputs = {
                    output_name: np.arange(
                        1, 2 * width + 1, dtype=np.float32
                    ).reshape(2, width)
                }
                adapter.validate_outputs(outputs, 2)
                requests = [
                    NormalizedRequest({}, (), None),
                    NormalizedRequest({}, (), None),
                ]
                results = [
                    adapter.unpack_one(outputs, index, requests[index])
                    for index in range(2)
                ]

                self.assertTrue(
                    all(isinstance(item, FeatureResult) for item in results)
                )
                self.assertTrue(all(item.vector.shape == (width,) for item in results))
                self.assertTrue(
                    all(item.vector.dtype == np.float32 for item in results)
                )
                self.assertTrue(all(item.vector.flags.c_contiguous for item in results))
                self.assertTrue(
                    all(np.all(np.isfinite(item.vector)) for item in results)
                )
                if key == "sbs_image":
                    for item in results:
                        self.assertAlmostEqual(
                            float(np.linalg.norm(item.vector.astype(np.float64))),
                            1.0,
                            places=6,
                        )

    def test_feature_results_do_not_share_memory_with_batch_outputs(self):
        cases = {
            "clip_image": ("unnorm_image_features", 512),
            "face_rec": ("EMBEDDING", 512),
            "tbps_image": ("image_feat", 512),
            "sbs_image": ("embedding", 2048),
        }
        request = NormalizedRequest({}, (), None)
        for key, (name, width) in cases.items():
            with self.subTest(key=key):
                outputs = {
                    name: np.ones((2, width), dtype=np.float32)
                }
                adapter = self.adapters[key]
                adapter.validate_outputs(outputs, 2)

                result = adapter.unpack_one(outputs, 0, request)

                self.assertFalse(
                    np.shares_memory(result.vector, outputs[name])
                )

    def test_sbs_normalization_divides_in_fp64_before_casting_to_fp32(self):
        adapter = self.adapters["sbs_image"]
        vector = np.random.default_rng(7).normal(size=2048).astype(np.float32)
        outputs = {"embedding": np.stack((vector, vector), axis=0)}
        expected = (
            vector.astype(np.float64)
            / np.linalg.norm(vector.astype(np.float64))
        ).astype(np.float32)
        adapter.validate_outputs(outputs, 2)

        result = adapter.unpack_one(
            outputs, 0, NormalizedRequest({}, (), None)
        )

        np.testing.assert_array_equal(result.vector, expected)

    def test_face_quality_returns_per_sample_business_results(self):
        adapter = self.adapters["face_qa"]
        outputs = {
            "quality_score": np.asarray([[0.75], [0.25]], dtype=np.float32)
        }
        adapter.validate_outputs(outputs, 2)

        results = [
            adapter.unpack_one(outputs, index, self.raw_requests["face_qa"][index])
            for index in range(2)
        ]

        self.assertEqual(results, [FaceQualityResult(0.75), FaceQualityResult(0.25)])

    def test_face_keypoints_convert_each_106_point_sample_to_five_points(self):
        adapter = self.adapters["face_kps"]
        landmarks = np.arange(2 * 106 * 2, dtype=np.float32).reshape(2, 106, 2)
        outputs = {"LANDMARKS_106": landmarks}
        adapter.validate_outputs(outputs, 2)
        requests = [NormalizedRequest({}, (), None)] * 2

        first = adapter.unpack_one(outputs, 0, requests[0])
        second = adapter.unpack_one(outputs, 1, requests[1])

        expected_first = np.vstack(
            (
                landmarks[0, 33:43].mean(axis=0),
                landmarks[0, 87:97].mean(axis=0),
                landmarks[0, 86],
                landmarks[0, 52],
                landmarks[0, 61],
            )
        ).astype(np.float32)
        np.testing.assert_allclose(first, expected_first)
        self.assertEqual(first.shape, (5, 2))
        self.assertEqual(second.shape, (5, 2))
        self.assertFalse(np.array_equal(first, second))

    def test_face_keypoint_eye_means_remain_finite_at_fp32_maximum(self):
        adapter = self.adapters["face_kps"]
        landmarks = np.full(
            (1, 106, 2), np.finfo(np.float32).max, dtype=np.float32
        )
        outputs = {"LANDMARKS_106": landmarks}
        adapter.validate_outputs(outputs, 1)

        result = adapter.unpack_one(
            outputs, 0, NormalizedRequest({}, (), None)
        )

        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(
            result, np.full((5, 2), np.finfo(np.float32).max)
        )

    def test_face_detection_slices_each_sample_using_its_own_count(self):
        adapter = self.adapters["face_det"]
        outputs = {
            "DETECTIONS": batched_detections(),
            "NUM_DETECTIONS": batched_counts(1, 2),
        }
        adapter.validate_outputs(outputs, 2)

        first = adapter.unpack_one(outputs, 0, self.raw_requests["face_det"][0])
        second = adapter.unpack_one(outputs, 1, self.raw_requests["face_det"][1])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        self.assertEqual(first[0]["xyxy"], [1.0, 1.0, 4.0, 4.0])
        self.assertEqual(second[1]["xyxy"], [4.0, 4.0, 7.0, 7.0])
        self.assertAlmostEqual(first[0]["conf"], 0.9, places=6)
        self.assertIsInstance(first[0]["cls"], int)

    def test_person_segmentation_keeps_detection_mask_alignment_per_sample(self):
        adapter = self.adapters["person_seg"]
        outputs = batched_segment_outputs()
        adapter.validate_outputs(outputs, 2)

        first = adapter.unpack_one(outputs, 0, self.raw_requests["person_seg"][0])
        second = adapter.unpack_one(outputs, 1, self.raw_requests["person_seg"][1])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        self.assertEqual(first[0]["maskPixelCount"], 9)
        self.assertEqual(second[0]["maskPixelCount"], 36)
        self.assertEqual(second[1]["maskPixelCount"], 9)

    def test_person_segmentation_resizes_crop_only_inside_bbox(self):
        adapter = self.adapters["person_seg"]
        outputs = {
            "DETECTIONS": np.asarray(
                [[[2, 2, 6, 6, 0.9, 0]]], dtype=np.float32
            ),
            "MASK_CROPS_PACKED": np.zeros(
                (1, 1, 160, 20), dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        outputs["MASK_CROPS_PACKED"][0, 0, :, :10] = 255
        request = adapter.normalize(raw_image())
        adapter.validate_outputs(outputs, 1)

        mask = adapter.unpack_one(outputs, 0, request)[0]["mask"]

        self.assertEqual(mask.image_shape, (10, 12))
        self.assertEqual(mask.origin, (2, 2))
        self.assertEqual(mask.data.shape, (4, 4))
        self.assertEqual(mask.data.dtype, np.uint8)
        np.testing.assert_array_equal(mask.data, [[1, 1, 0, 0]] * 4)
        self.assertEqual(mask.count((0, 0, 12, 2)), 0)
        self.assertEqual(mask.count((0, 6, 12, 10)), 0)

    def test_person_segmentation_accepts_zero_count_with_one_wire_slot(self):
        adapter = self.adapters["person_seg"]
        outputs = {
            "DETECTIONS": np.zeros((1, 1, 6), dtype=np.float32),
            "MASK_CROPS_PACKED": np.zeros(
                (1, 1, 160, 20), dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.zeros((1, 1), dtype=np.int32),
        }
        request = adapter.normalize(raw_image())

        adapter.validate_outputs(outputs, 1)

        self.assertEqual(adapter.unpack_one(outputs, 0, request), [])

    def test_person_segmentation_accepts_canonical_capacity_from_counts(self):
        adapter = self.adapters["person_seg"]
        dynamic_two = batched_segment_outputs()
        all_empty = {
            "DETECTIONS": np.zeros((2, 1, 6), dtype=np.float32),
            "MASK_CROPS_PACKED": np.zeros(
                (2, 1, 160, 20), dtype=np.uint8
            ),
            "NUM_DETECTIONS": batched_counts(0, 0),
        }

        adapter.validate_outputs(dynamic_two, 2)
        adapter.validate_outputs(all_empty, 2)

    def test_person_segmentation_rejects_noncanonical_capacity_from_counts(self):
        adapter = self.adapters["person_seg"]
        overpadded = batched_segment_outputs()
        overpadded["DETECTIONS"] = np.pad(
            overpadded["DETECTIONS"], ((0, 0), (0, 1), (0, 0))
        )
        overpadded["MASK_CROPS_PACKED"] = np.pad(
            overpadded["MASK_CROPS_PACKED"], ((0, 0), (0, 1), (0, 0), (0, 0))
        )
        all_empty_overpadded = {
            "DETECTIONS": np.zeros((2, 2, 6), dtype=np.float32),
            "MASK_CROPS_PACKED": np.zeros(
                (2, 2, 160, 20), dtype=np.uint8
            ),
            "NUM_DETECTIONS": batched_counts(0, 0),
        }

        for name, outputs, expected, actual in (
            ("counts one two", overpadded, 2, 3),
            ("all empty", all_empty_overpadded, 1, 2),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                InferenceResponseError,
                f"NUM_DETECTIONS.*capacity {expected}, got {actual}",
            ):
                adapter.validate_outputs(outputs, 2)

    def test_person_segmentation_rejects_invalid_dynamic_capacity_and_alignment(self):
        adapter = self.adapters["person_seg"]
        cases = []

        empty = {
            "DETECTIONS": np.zeros((2, 0, 6), dtype=np.float32),
            "MASK_CROPS_PACKED": np.zeros(
                (2, 0, 160, 20), dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.zeros((2, 1), dtype=np.int32),
        }
        cases.append(("zero capacity", empty, "axis 1"))

        too_many = {
            "DETECTIONS": np.zeros((2, 101, 6), dtype=np.float32),
            "MASK_CROPS_PACKED": np.zeros(
                (2, 101, 160, 20), dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.zeros((2, 1), dtype=np.int32),
        }
        cases.append(("capacity 101", too_many, r"\[1, 100\]"))

        mismatched_batch = batched_segment_outputs()
        mismatched_batch["MASK_CROPS_PACKED"] = np.zeros(
            (1, 2, 160, 20), dtype=np.uint8
        )
        cases.append(("mismatched batch", mismatched_batch, "batch size 2"))

        mismatched_capacity = batched_segment_outputs()
        mismatched_capacity["MASK_CROPS_PACKED"] = np.zeros(
            (2, 1, 160, 20), dtype=np.uint8
        )
        cases.append(
            ("mismatched capacity", mismatched_capacity, "share batch")
        )

        for name, outputs, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                InferenceResponseError, message
            ):
                adapter.validate_outputs(outputs, 2)

    def test_person_segmentation_rejects_wrong_shapes_names_and_dtypes(self):
        adapter = self.adapters["person_seg"]
        cases = []

        old_name = batched_segment_outputs()
        old_name["MASK_CROPS"] = old_name.pop("MASK_CROPS_PACKED")
        cases.append(("old mask name", old_name, "MASK_CROPS_PACKED"))

        wrong_detection_tail = batched_segment_outputs()
        wrong_detection_tail["DETECTIONS"] = np.zeros(
            (2, 2, 5), dtype=np.float32
        )
        cases.append(
            ("detection tail", wrong_detection_tail, "DETECTIONS")
        )

        wrong_crop_tail = batched_segment_outputs()
        wrong_crop_tail["MASK_CROPS_PACKED"] = np.zeros(
            (2, 2, 160, 19), dtype=np.uint8
        )
        cases.append(("crop tail", wrong_crop_tail, "MASK_CROPS_PACKED"))

        wrong_count_shape = batched_segment_outputs()
        wrong_count_shape["NUM_DETECTIONS"] = np.ones(
            (2,), dtype=np.int32
        )
        cases.append(
            ("count shape", wrong_count_shape, "NUM_DETECTIONS")
        )

        for tensor_name, dtype in (
            ("DETECTIONS", np.float64),
            ("MASK_CROPS_PACKED", np.int32),
            ("NUM_DETECTIONS", np.int64),
        ):
            wrong_dtype = batched_segment_outputs()
            wrong_dtype[tensor_name] = wrong_dtype[tensor_name].astype(dtype)
            cases.append(
                (
                    f"{tensor_name} dtype",
                    wrong_dtype,
                    f"{tensor_name}.*dtype",
                )
            )

        for name, outputs, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                InferenceResponseError, message
            ):
                adapter.validate_outputs(outputs, 2)

    def test_person_segmentation_rejects_counts_outside_dynamic_capacity(self):
        adapter = self.adapters["person_seg"]
        for invalid, message in ((-1, r"\[0, 100\]"), (3, "capacity 2")):
            with self.subTest(invalid=invalid):
                outputs = batched_segment_outputs()
                outputs["NUM_DETECTIONS"][0, 0] = invalid
                with self.assertRaisesRegex(InferenceResponseError, message):
                    adapter.validate_outputs(outputs, 2)

    def test_person_segmentation_reports_count_before_nonzero_padding(self):
        adapter = self.adapters["person_seg"]
        outputs = batched_segment_outputs()
        outputs["NUM_DETECTIONS"][0, 0] = 3
        outputs["MASK_CROPS_PACKED"][0, 1, 0, 0] = 2

        with self.assertRaises(InferenceResponseError) as caught:
            adapter.validate_outputs(outputs, 2)

        self.assertIn("NUM_DETECTIONS", str(caught.exception))
        self.assertNotIn("padding", str(caught.exception))

    def test_person_segmentation_accepts_all_packed_byte_values_and_rejects_padding(self):
        adapter = self.adapters["person_seg"]

        all_bytes = batched_segment_outputs()
        all_bytes["MASK_CROPS_PACKED"][0, 0].flat[:256] = np.arange(
            256, dtype=np.uint8
        )
        adapter.validate_outputs(all_bytes, 2)

        detection_padding = batched_segment_outputs()
        detection_padding["DETECTIONS"][0, 1, 0] = 1
        with self.assertRaisesRegex(InferenceResponseError, "padding"):
            adapter.validate_outputs(detection_padding, 2)

        crop_padding = batched_segment_outputs()
        crop_padding["MASK_CROPS_PACKED"][0, 1, 0, 0] = 1
        with self.assertRaisesRegex(InferenceResponseError, "padding"):
            adapter.validate_outputs(crop_padding, 2)

    def test_person_segmentation_rejects_nonfinite_valid_detection_prefix(self):
        adapter = self.adapters["person_seg"]
        outputs = batched_segment_outputs()
        outputs["DETECTIONS"][1, 1, 4] = np.inf

        with self.assertRaisesRegex(InferenceResponseError, "non-finite"):
            adapter.validate_outputs(outputs, 2)

    def test_person_segmentation_valid_nonperson_is_not_padding_and_is_filtered(self):
        adapter = self.adapters["person_seg"]
        outputs = {
            "DETECTIONS": np.asarray(
                [[[1, 1, 4, 4, 0.9, 7]]], dtype=np.float32
            ),
            "MASK_CROPS_PACKED": np.full(
                (1, 1, 160, 20), 255, dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        request = adapter.normalize(raw_image())

        adapter.validate_outputs(outputs, 1)

        self.assertEqual(adapter.unpack_one(outputs, 0, request), [])

    def test_person_segmentation_clips_rounded_bbox_and_rejects_empty_bbox(self):
        adapter = self.adapters["person_seg"]
        request = adapter.normalize(raw_image())
        crops = np.full((1, 1, 160, 20), 255, dtype=np.uint8)
        edge = {
            "DETECTIONS": np.asarray(
                [[[-2.4, -1.4, 3.4, 4.4, 0.9, 0]]], dtype=np.float32
            ),
            "MASK_CROPS_PACKED": crops,
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        adapter.validate_outputs(edge, 1)

        result = adapter.unpack_one(edge, 0, request)[0]

        self.assertEqual(result["rect"], {"x": 0, "y": 0, "w": 3, "h": 4})
        self.assertEqual(result["maskPixelCount"], 12)

        invalid = {
            "DETECTIONS": np.asarray(
                [[[3.1, 3.1, 3.4, 5.4, 0.9, 0]]], dtype=np.float32
            ),
            "MASK_CROPS_PACKED": crops,
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        adapter.validate_outputs(invalid, 1)
        with self.assertRaisesRegex(InferenceResponseError, "no valid clipped"):
            adapter.unpack_one(invalid, 0, request)

    def test_person_segmentation_validates_nonperson_bbox_before_filtering(self):
        adapter = self.adapters["person_seg"]
        outputs = {
            "DETECTIONS": np.asarray(
                [[[8, 8, 4, 4, 0.9, 7]]], dtype=np.float32
            ),
            "MASK_CROPS_PACKED": np.full(
                (1, 1, 160, 20), 255, dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        request = adapter.normalize(raw_image())
        adapter.validate_outputs(outputs, 1)

        with self.assertRaisesRegex(InferenceResponseError, "no valid clipped"):
            adapter.unpack_one(outputs, 0, request)

    def test_person_segmentation_validation_error_has_model_and_tensor_context(self):
        adapter = self.adapters["person_seg"]
        outputs = batched_segment_outputs()
        outputs["MASK_CROPS_PACKED"] = outputs["MASK_CROPS_PACKED"].astype(
            np.int32
        )

        with self.assertRaises(InferenceResponseError) as caught:
            adapter.validate_outputs(outputs, 2)

        message = str(caught.exception)
        self.assertTrue(message.startswith("person_seg: "))
        self.assertEqual(message.count("person_seg:"), 1)
        self.assertIn("MASK_CROPS_PACKED", message)

    def test_person_segmentation_bbox_error_has_model_and_detection_context(self):
        adapter = self.adapters["person_seg"]
        outputs = {
            "DETECTIONS": np.asarray(
                [[[8, 8, 4, 4, 0.9, 0]]], dtype=np.float32
            ),
            "MASK_CROPS_PACKED": np.full(
                (1, 1, 160, 20), 255, dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        request = adapter.normalize(raw_image())
        adapter.validate_outputs(outputs, 1)

        with self.assertRaises(InferenceResponseError) as caught:
            adapter.unpack_one(outputs, 0, request)

        message = str(caught.exception)
        self.assertTrue(message.startswith("person_seg: "))
        self.assertEqual(message.count("person_seg:"), 1)
        self.assertIn("DETECTIONS[0]", message)
        self.assertIn("bbox", message)

    def test_person_pose_keeps_detection_keypoint_alignment_per_sample(self):
        adapter = self.adapters["person_pose"]
        keypoints = np.zeros((2, 100, 17, 3), dtype=np.float32)
        keypoints[0, 0, :, 0] = 11
        keypoints[1, 0, :, 0] = 22
        keypoints[1, 1, :, 0] = 33
        outputs = {
            "DETECTIONS": batched_detections(),
            "KEYPOINTS": keypoints,
            "NUM_DETECTIONS": batched_counts(1, 2),
        }
        adapter.validate_outputs(outputs, 2)

        first = adapter.unpack_one(outputs, 0, self.raw_requests["person_pose"][0])
        second = adapter.unpack_one(outputs, 1, self.raw_requests["person_pose"][1])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        np.testing.assert_array_equal(first[0]["keypoints"][:, 0], 11)
        np.testing.assert_array_equal(second[0]["keypoints"][:, 0], 22)
        np.testing.assert_array_equal(second[1]["keypoints"][:, 0], 33)

    def test_person_results_own_only_the_returned_mask_and_keypoints(self):
        counts = np.asarray([[1]], dtype=np.int32)

        segment = self.adapters["person_seg"]
        segment_detections = np.asarray(
            [[[1, 1, 4, 4, 0.9, 0]]], dtype=np.float32
        )
        crops = np.full((1, 1, 160, 20), 255, dtype=np.uint8)
        segment_outputs = {
            "DETECTIONS": segment_detections,
            "MASK_CROPS_PACKED": crops,
            "NUM_DETECTIONS": counts,
        }
        segment_request = segment.normalize(raw_image())
        segment.validate_outputs(segment_outputs, 1)

        segment_result = segment.unpack_one(
            segment_outputs, 0, segment_request
        )[0]

        pose = self.adapters["person_pose"]
        pose_detections = np.zeros((1, 100, 6), dtype=np.float32)
        pose_detections[0, 0] = [1, 1, 4, 4, 0.9, 0]
        keypoints = np.zeros((1, 100, 17, 3), dtype=np.float32)
        pose_outputs = {
            "DETECTIONS": pose_detections,
            "KEYPOINTS": keypoints,
            "NUM_DETECTIONS": counts,
        }
        pose_request = pose.normalize(raw_image())
        pose.validate_outputs(pose_outputs, 1)

        pose_result = pose.unpack_one(pose_outputs, 0, pose_request)[0]

        self.assertFalse(
            np.shares_memory(segment_result["mask"].data, crops)
        )
        self.assertFalse(
            np.shares_memory(pose_result["keypoints"], keypoints)
        )
        self.assertEqual(segment_result["mask"].image_shape, (10, 12))
        self.assertEqual(segment_result["mask"].origin, (1, 1))
        self.assertEqual(segment_result["mask"].data.shape, (3, 3))
        self.assertEqual(segment_result["mask"].data.dtype, np.uint8)
        self.assertTrue(segment_result["mask"].data.flags.c_contiguous)
        self.assertTrue(segment_result["mask"].data.flags.owndata)
        self.assertEqual(pose_result["keypoints"].shape, (17, 3))

    def test_person_segmentation_does_not_allocate_full_image_masks(self):
        adapter = self.adapters["person_seg"]
        height, width = 720, 1280
        outputs = {
            "DETECTIONS": np.asarray(
                [
                    [
                        [10, 10, 300, 400, 0.9, 0],
                        [500, 200, 1000, 700, 0.8, 0],
                    ]
                ],
                dtype=np.float32,
            ),
            "MASK_CROPS_PACKED": np.full(
                (1, 2, 160, 20), 255, dtype=np.uint8
            ),
            "NUM_DETECTIONS": np.asarray([[2]], dtype=np.int32),
        }
        request = adapter.normalize(
            np.zeros((height, width, 3), dtype=np.uint8)
        )
        adapter.validate_outputs(outputs, 1)
        with patch.object(person_adapter_module.np, "zeros", wraps=np.zeros) as zeros_spy:
            results = adapter.unpack_one(outputs, 0, request)

        self.assertNotIn((height, width), [call.args[0] for call in zeros_spy.call_args_list])
        self.assertEqual(len(results), 2)
        for result in results:
            rect, mask = result["rect"], result["mask"]
            self.assertEqual(mask.image_shape, (height, width))
            self.assertEqual(mask.origin, (rect["x"], rect["y"]))
            self.assertEqual(mask.data.nbytes, rect["w"] * rect["h"])
            self.assertEqual(mask.data.dtype, np.uint8)
            self.assertTrue(mask.data.flags.c_contiguous)
            self.assertTrue(mask.data.flags.owndata)

    def test_missing_output_wrong_batch_wrong_width_and_wrong_dtype_are_rejected(self):
        cases = [
            (
                "missing",
                self.adapters["clip_image"],
                {},
            ),
            (
                "wrong batch",
                self.adapters["clip_image"],
                {"unnorm_image_features": np.ones((1, 512), dtype=np.float32)},
            ),
            (
                "wrong width",
                self.adapters["clip_image"],
                {"unnorm_image_features": np.ones((2, 511), dtype=np.float32)},
            ),
            (
                "wrong dtype",
                self.adapters["clip_image"],
                {"unnorm_image_features": np.ones((2, 512), dtype=np.float64)},
            ),
        ]
        for name, adapter, outputs in cases:
            with self.subTest(name=name), self.assertRaises(InferenceResponseError):
                adapter.validate_outputs(outputs, 2)

    def test_structurally_misaligned_person_outputs_are_rejected_for_whole_batch(self):
        detections = batched_detections()
        cases = [
            (
                self.adapters["person_seg"],
                {
                    "DETECTIONS": np.zeros((2, 2, 6), dtype=np.float32),
                    "MASK_CROPS_PACKED": np.zeros(
                        (2, 1, 160, 20), dtype=np.uint8
                    ),
                    "NUM_DETECTIONS": batched_counts(),
                },
            ),
            (
                self.adapters["person_pose"],
                {
                    "DETECTIONS": detections,
                    "KEYPOINTS": np.zeros((2, 99, 17, 3), dtype=np.float32),
                    "NUM_DETECTIONS": batched_counts(),
                },
            ),
        ]
        for adapter, outputs in cases:
            with self.subTest(key=adapter.contract.key), self.assertRaises(
                InferenceResponseError
            ):
                adapter.validate_outputs(outputs, 2)

    def test_nonfinite_second_feature_isolated_to_second_unpack(self):
        adapter = self.adapters["clip_image"]
        values = np.ones((2, 512), dtype=np.float32)
        values[1, 10] = np.nan
        outputs = {"unnorm_image_features": values}
        adapter.validate_outputs(outputs, 2)
        request = NormalizedRequest({}, (), None)

        first = adapter.unpack_one(outputs, 0, request)

        self.assertIsInstance(first, FeatureResult)
        with self.assertRaises(InferenceResponseError):
            adapter.unpack_one(outputs, 1, request)

    def test_nonfinite_values_are_rejected_per_sample_by_all_float_adapters(self):
        cases = []
        for key, output_name, shape in (
            ("face_rec", "EMBEDDING", (2, 512)),
            ("tbps_image", "image_feat", (2, 512)),
            ("sbs_image", "embedding", (2, 2048)),
            ("face_qa", "quality_score", (2, 1)),
            ("face_kps", "LANDMARKS_106", (2, 106, 2)),
        ):
            outputs = {output_name: np.ones(shape, dtype=np.float32)}
            outputs[output_name][1].flat[0] = np.nan
            cases.append((key, outputs))
        detections = batched_detections()
        detections[1, 0, 0] = np.nan
        cases.extend(
            [
                (
                    "face_det",
                    {
                        "DETECTIONS": detections,
                        "NUM_DETECTIONS": batched_counts(),
                    },
                ),
                (
                    "person_pose",
                    {
                        "DETECTIONS": detections.copy(),
                        "KEYPOINTS": np.zeros((2, 100, 17, 3), dtype=np.float32),
                        "NUM_DETECTIONS": batched_counts(),
                    },
                ),
            ]
        )
        for key, outputs in cases:
            with self.subTest(key=key):
                adapter = self.adapters[key]
                adapter.validate_outputs(outputs, 2)
                request = self.raw_requests.get(
                    key, [NormalizedRequest({}, (), None)] * 2
                )
                adapter.unpack_one(outputs, 0, request[0])
                with self.assertRaises(InferenceResponseError):
                    adapter.unpack_one(outputs, 1, request[1])

    def test_invalid_counts_and_zero_sbs_vector_are_rejected_per_sample(self):
        for key in ("face_det", "person_pose"):
            for invalid in (-1, 101):
                with self.subTest(key=key, invalid=invalid):
                    outputs = {
                        "DETECTIONS": batched_detections(),
                        "NUM_DETECTIONS": batched_counts(1, invalid),
                    }
                    if key == "person_pose":
                        outputs["KEYPOINTS"] = np.zeros(
                            (2, 100, 17, 3), dtype=np.float32
                        )
                    adapter = self.adapters[key]
                    adapter.validate_outputs(outputs, 2)
                    adapter.unpack_one(
                        outputs, 0, self.raw_requests[key][0]
                    )
                    with self.assertRaises(InferenceResponseError):
                        adapter.unpack_one(
                            outputs, 1, self.raw_requests[key][1]
                        )

        sbs = self.adapters["sbs_image"]
        outputs = {"embedding": np.ones((2, 2048), dtype=np.float32)}
        outputs["embedding"][1] = 0
        sbs.validate_outputs(outputs, 2)
        sbs.unpack_one(outputs, 0, NormalizedRequest({}, (), None))
        with self.assertRaises(InferenceResponseError):
            sbs.unpack_one(outputs, 1, NormalizedRequest({}, (), None))


if __name__ == "__main__":
    unittest.main()
