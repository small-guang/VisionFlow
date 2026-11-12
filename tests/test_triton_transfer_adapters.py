import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

import core.infer.adapters.person as person_adapter_module
from core.infer.adapters import build_adapters
from core.infer.errors import InferenceInputError, InferenceResponseError


class EncodedImageTransferTests(unittest.TestCase):
    def test_native_bytes_preserve_identity_lengths_and_nuls(self):
        values = [b"\x00a\x00", b"longer\x00payload\x00\x00"]
        for key in ("sbs_image", "tbps_image"):
            with self.subTest(key=key):
                adapter = build_adapters()[key]
                try:
                    requests = [adapter.normalize(value) for value in values]
                except InferenceInputError as error:
                    self.fail(f"native encoded bytes must be accepted: {error}")
                self.assertEqual(
                    requests[0].compatibility_key, requests[1].compatibility_key
                )

                packed = adapter.pack(requests)["IMAGE_BYTES"]

                self.assertEqual(packed.shape, (2, 1))
                self.assertEqual(packed.dtype, object)
                self.assertEqual(packed.tolist(), [[values[0]], [values[1]]])
                for index, value in enumerate(values):
                    self.assertIs(
                        requests[index].inputs["IMAGE_BYTES"][0], value
                    )
                    self.assertIs(packed[index, 0], value)

    def test_legacy_uint8_slices_convert_once_to_exact_immutable_bytes(self):
        for key in ("sbs_image", "tbps_image"):
            with self.subTest(key=key):
                adapter = build_adapters()[key]
                original = np.asarray([7, 99, 0, 99, 8, 99, 0], dtype=np.uint8)
                request = adapter.normalize(original[::2])

                converted = request.inputs["IMAGE_BYTES"][0]
                self.assertIsInstance(converted, bytes)
                self.assertEqual(converted, b"\x07\x00\x08\x00")
                original[:] = 4
                packed = adapter.pack([request])["IMAGE_BYTES"]
                self.assertIs(packed[0, 0], converted)
                self.assertEqual(packed[0, 0], b"\x07\x00\x08\x00")

    def test_encoded_input_validation_rejects_empty_or_mutable_payloads(self):
        for key in ("sbs_image", "tbps_image"):
            adapter = build_adapters()[key]
            for value in (
                b"", "jpeg", bytearray(b"jpeg"), memoryview(b"jpeg")
            ):
                with self.subTest(key=key, value=value), self.assertRaises(
                    InferenceInputError
                ):
                    adapter.normalize(value)


class NumericInputOwnershipTests(unittest.TestCase):
    def test_single_sample_shares_input_storage_and_retains_owner(self):
        adapter = build_adapters()["face_rec"]
        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3).copy()
        landmarks = np.arange(10, dtype=np.float32).reshape(5, 2).copy()
        image_ref, landmarks_ref = weakref.ref(image), weakref.ref(landmarks)
        request = adapter.normalize(image, landmarks)

        packed = adapter.pack([request])

        self.assertTrue(np.shares_memory(packed["RAW_IMAGE"], image))
        self.assertTrue(np.shares_memory(packed["LANDMARK"], landmarks))
        self.assertEqual(packed["RAW_IMAGE"].shape, (1, 3, 4, 3))
        self.assertEqual(packed["LANDMARK"].shape, (1, 5, 2))
        del image, landmarks, request
        gc.collect()
        self.assertIsNotNone(image_ref())
        self.assertIsNotNone(landmarks_ref())
        self.assertEqual(packed["RAW_IMAGE"].tobytes(), bytes(range(36)))
        np.testing.assert_array_equal(packed["LANDMARK"][0, -1], [8, 9])
        del packed
        gc.collect()
        self.assertIsNone(image_ref())
        self.assertIsNone(landmarks_ref())

    def test_noncontiguous_inputs_materialize_only_when_packed(self):
        adapter = build_adapters()["face_rec"]
        image = np.arange(72, dtype=np.uint8).reshape(3, 8, 3)[:, ::2, :]
        landmarks = np.arange(20, dtype=np.float32).reshape(10, 2)[::2]
        request = adapter.normalize(image, landmarks)
        self.assertIs(request.inputs["RAW_IMAGE"], image)
        self.assertIs(request.inputs["LANDMARK"], landmarks)

        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                packed = adapter.pack([request] * batch_size)
                for name, value in request.inputs.items():
                    self.assertTrue(packed[name].flags.c_contiguous)
                    self.assertFalse(np.shares_memory(packed[name], value))
                    for row in packed[name]:
                        np.testing.assert_array_equal(row, value)

    def test_multiple_samples_own_one_contiguous_batch(self):
        adapter = build_adapters()["face_det"]
        first = np.asfortranarray(np.full((3, 4, 3), 7, dtype=np.uint8))
        second = np.asfortranarray(np.full((3, 4, 3), 9, dtype=np.uint8))
        requests = [adapter.normalize(first), adapter.normalize(second)]
        self.assertIs(requests[0].inputs["RAW_IMAGE"], first)
        self.assertIs(requests[1].inputs["RAW_IMAGE"], second)

        packed = adapter.pack(requests)["RAW_IMAGE"]

        self.assertTrue(packed.flags.c_contiguous)
        self.assertTrue(packed.flags.owndata)
        self.assertFalse(np.shares_memory(packed, first))
        self.assertFalse(np.shares_memory(packed, second))
        np.testing.assert_array_equal(packed[0], np.full((3, 4, 3), 7))
        np.testing.assert_array_equal(packed[1], np.full((3, 4, 3), 9))


class PackedMaskTransferTests(unittest.TestCase):
    def test_explicit_big_order_bits_reconstruct_exact_bbox_mask(self):
        adapter = build_adapters()["person_seg"]
        request = adapter.normalize(np.zeros((180, 180, 3), dtype=np.uint8))
        packed = np.zeros((1, 1, 160, 20), dtype=np.uint8)
        packed[0, 0, 0, :4] = [0x80, 0x01, 0xAA, 0x55]
        packed[0, 0, 159, 19] = 0x01
        outputs = {
            "DETECTIONS": np.asarray([[[10, 12, 170, 172, 0.9, 0]]], np.float32),
            "MASK_CROPS_PACKED": packed,
            "NUM_DETECTIONS": np.asarray([[1]], np.int32),
        }
        try:
            adapter.validate_outputs(outputs, 1)
        except InferenceResponseError as error:
            self.fail(f"valid packed masks must be accepted: {error}")

        result = adapter.unpack_one(outputs, 0, request)[0]

        expected = np.zeros((160, 160), dtype=np.uint8)
        expected[0, :32] = [
            1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1,
        ]
        expected[159, 159] = 1
        np.testing.assert_array_equal(result["mask"].data, expected)
        self.assertEqual(result["mask"].origin, (10, 12))
        self.assertEqual(result["maskPixelCount"], 11)
        self.assertFalse(np.shares_memory(result["mask"].data, packed))

    def test_zero_and_all_one_masks_unpack_only_used_person_slots(self):
        adapter = build_adapters()["person_seg"]
        request = adapter.normalize(np.zeros((10, 12, 3), dtype=np.uint8))
        packed = np.zeros((2, 3, 160, 20), dtype=np.uint8)
        packed[1, 0] = 255
        packed[1, 1] = 0xAA
        detections = np.zeros((2, 3, 6), dtype=np.float32)
        detections[0, 0] = [1, 1, 4, 4, 0.9, 0]
        detections[1] = [
            [1, 1, 4, 4, 0.9, 0],
            [2, 2, 5, 5, 0.8, 7],
            [3, 3, 6, 6, 0.7, 0],
        ]
        outputs = {
            "DETECTIONS": detections,
            "MASK_CROPS_PACKED": packed,
            "NUM_DETECTIONS": np.asarray([[1], [3]], np.int32),
        }
        try:
            adapter.validate_outputs(outputs, 2)
        except InferenceResponseError as error:
            self.fail(f"valid packed masks must be accepted: {error}")

        with patch.object(
            person_adapter_module.np, "unpackbits", wraps=np.unpackbits
        ) as unpack:
            first = adapter.unpack_one(outputs, 0, request)
            second = adapter.unpack_one(outputs, 1, request)

        self.assertEqual([item["maskPixelCount"] for item in first], [0])
        self.assertEqual([item["maskPixelCount"] for item in second], [9, 0])
        self.assertEqual(unpack.call_count, 3)
        for call in unpack.call_args_list:
            self.assertEqual(call.args[0].shape, (160, 20))
            self.assertEqual(
                call.kwargs, {"axis": -1, "count": 160, "bitorder": "big"}
            )


if __name__ == "__main__":
    unittest.main()
