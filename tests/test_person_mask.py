import unittest
import weakref

import numpy as np

from core.infer.adapters import build_adapters


class PersonMaskStorageTests(unittest.TestCase):
    def test_segmentation_retains_bbox_pixels_without_full_frame_allocation(self):
        adapter = build_adapters()["person_seg"]
        request = adapter.normalize(np.zeros((1080, 1920, 3), dtype=np.uint8))
        outputs = {
            "DETECTIONS": np.asarray([[[100, 100, 300, 600, 0.9, 0]]], dtype=np.float32),
            "MASK_CROPS_PACKED": np.full((1, 1, 160, 20), 255, dtype=np.uint8),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        adapter.validate_outputs(outputs, 1)
        result = adapter.unpack_one(outputs, 0, request)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["maskPixelCount"], 100000)
        mask = result[0]["mask"]
        data = mask if isinstance(mask, np.ndarray) else mask.data
        self.assertLessEqual(data.nbytes, 100000)

    def test_twenty_masks_keep_the_same_storage_at_1080p_and_4k(self):
        adapter = build_adapters()["person_seg"]
        for height, width in ((1080, 1920), (2160, 3840)):
            with self.subTest(resolution=(height, width)):
                request = adapter.normalize(np.zeros((height, width, 3), dtype=np.uint8))
                outputs = {
                    "DETECTIONS": np.tile(np.asarray(
                        [[[100, 100, 300, 600, 0.9, 0]]], dtype=np.float32
                    ), (1, 20, 1)),
                    "MASK_CROPS_PACKED": np.full((1, 20, 160, 20), 255, dtype=np.uint8),
                    "NUM_DETECTIONS": np.asarray([[20]], dtype=np.int32),
                }
                adapter.validate_outputs(outputs, 1)
                wire_ref = weakref.ref(outputs["MASK_CROPS_PACKED"])
                image_ref = weakref.ref(request.inputs["RAW_IMAGE"])
                result = adapter.unpack_one(outputs, 0, request)
                self.assertEqual(sum(item["mask"].data.nbytes for item in result), 2000000)
                self.assertTrue(all(item["mask"].data.flags.owndata for item in result))
                del outputs, request
                self.assertIsNone(wire_ref())
                self.assertIsNone(image_ref())


class PersonMaskGeometryTests(unittest.TestCase):
    def setUp(self):
        from utils.person_mask import PersonMask

        self.Mask = PersonMask
        self.data = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        self.mask = PersonMask(self.data, (2, 1), (5, 7))

    def test_global_window_has_same_zeros_as_an_image_canvas(self):
        np.testing.assert_array_equal(
            self.mask.window((1, 0, 6, 4)),
            [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0],
             [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]],
        )

    def test_windows_clip_at_image_edges_and_preserve_empty_windows(self):
        np.testing.assert_array_equal(
            self.mask.window((-20, -20, 3, 2)),
            [[0, 0, 0], [0, 0, 1]],
        )
        self.assertEqual(self.mask.window((8, 1, 10, 4)).shape, (3, 0))
        self.assertEqual(self.mask.window((4, 3, 2, 1)).size, 0)

    def test_contained_window_shares_only_local_storage(self):
        window = self.mask.window((3, 1, 5, 3))
        self.assertTrue(np.shares_memory(window, self.data))
        np.testing.assert_array_equal(window, [[0, 1], [1, 1]])

    def test_counts_use_global_coordinates_without_padding_allocations(self):
        self.assertEqual(self.mask.count(), 4)
        self.assertEqual(self.mask.count((0, 0, 3, 2)), 1)
        self.assertEqual(self.mask.count((4, 1, 5, 3)), 2)
        self.assertEqual(self.mask.count((5, 3, 7, 5)), 0)

    def test_intersection_aligns_different_origins(self):
        other = self.Mask(np.ones((2, 2), dtype=np.uint8), (3, 2), (5, 7))
        self.assertEqual(self.mask.intersection_count(other), 2)
        self.assertEqual(other.intersection_count(self.mask), 2)
        far = self.Mask(np.ones((1, 1), dtype=np.uint8), (6, 4), (5, 7))
        self.assertEqual(self.mask.intersection_count(far), 0)

    def test_invalid_geometry_is_rejected(self):
        for origin, shape in [
            ((-1, 0), (5, 7)), ((6, 4), (5, 7)),
            ((0.5, 0), (5, 7)), ((True, 0), (5, 7)),
            ((0, 0), (0, 7)), ((0, 0), (5, 7, 3)),
        ]:
            with self.subTest(origin=origin, shape=shape), self.assertRaises(ValueError):
                self.Mask(self.data, origin, shape)

    def test_legacy_arrays_remain_valid_without_copying_their_data(self):
        from utils.person_mask import as_person_mask

        result = as_person_mask(self.data, (2, 3, 3))
        self.assertIs(result.data, self.data)
        self.assertEqual(result.origin, (0, 0))
        self.assertIsNone(as_person_mask(self.data, (5, 7)))
        self.assertIsNone(as_person_mask(self.mask, (2, 3)))
        self.assertIsNone(as_person_mask(None))
        self.assertIsNone(as_person_mask(np.zeros((0, 3), dtype=np.uint8)))

    def test_implicit_array_conversion_never_materializes_an_image_mask(self):
        self.assertEqual(np.asarray(self.mask).ndim, 0)
