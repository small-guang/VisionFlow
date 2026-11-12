"""Compare local masks with independently embedded dense reference canvases."""

import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from core.infer.adapters import build_adapters
from core.person_pipeline.face_reuse import bind_frame_faces_to_people
from core.person_pipeline.observation import PersonObservationService
from tests.person_quality_test_support import _person_test_pose
from utils.keyframe.detection_helpers import (
    match_person_segments_to_poses,
    person_segments_can_share_reid,
    pose_mask_agreement,
)
from utils.keyframe.person_crop import (
    PersonRotationDecision,
    build_person_crop,
    project_mask_to_person_crop,
)
from utils.person_mask import PersonMask
from utils.quality.person.crop import evaluate_person_crop_quality
from utils.quality.person.evaluator import evaluate_person_quality
from utils.quality.person.mask import (
    _connected_component_areas,
    _mask_bbox_edge_stats,
    _mask_image_edge_directions,
    _valid_mask,
)
from utils.reid.person_face_recognition import select_owned_face_detection


def mask_pair(data, origin, shape):
    x, y = origin
    dense = np.zeros(shape[:2], dtype=data.dtype)
    dense[y:y + data.shape[0], x:x + data.shape[1]] = data
    return dense, PersonMask(data.copy(), origin, shape[:2])


def segment(mask, rect):
    data = mask.data if isinstance(mask, PersonMask) else mask
    return {
        "rect": rect, "conf": 0.9, "mask": mask,
        "maskPixelCount": int(np.count_nonzero(data)),
        "pose": {**_person_test_pose(rect), "rect": rect},
    }


class PersonMaskConsumerParityTests(unittest.TestCase):
    SHAPE = (200, 120, 3)
    RECT = {"x": 40, "y": 40, "w": 40, "h": 100}

    def setUp(self):
        self.data = np.ones((100, 40), dtype=np.uint8)
        self.data[30:38, 10:20] = 0
        self.dense, self.local = mask_pair(self.data, (40, 40), self.SHAPE)

    def test_pose_sampling_and_matching_preserve_global_coordinates(self):
        points = np.asarray([[45, 45, 1], [39, 45, 1], [20, 20, 1]], dtype=float)
        self.assertEqual(pose_mask_agreement(points, self.local, self.RECT), 2 / 3)
        dense_segment, local_segment = [segment(m, self.RECT) for m in (self.dense, self.local)]
        poses = [dense_segment["pose"]]
        dense_match = match_person_segments_to_poses([dense_segment], poses)[0]
        local_match = match_person_segments_to_poses([local_segment], poses)[0]
        self.assertIs(local_match["pose"], poses[0])
        for key in ("poseMatchIou", "poseMaskAgreement", "poseMatchScore"):
            self.assertEqual(local_match[key], dense_match[key])

    def test_duplicate_iou_handles_offsets_empty_disjoint_and_legacy_masks(self):
        for origin, data in [
            ((41, 40), self.data), ((70, 40), self.data),
            ((0, 0), np.ones((10, 10), dtype=np.uint8)),
            ((40, 40), np.zeros_like(self.data)),
        ]:
            other_dense, other_local = mask_pair(data, origin, self.SHAPE)
            expected = person_segments_can_share_reid(
                segment(self.dense, self.RECT), segment(other_dense, self.RECT)
            )
            with self.subTest(origin=origin, empty=not data.any()):
                for first, second in [(self.local, other_local), (self.local, other_dense)]:
                    self.assertEqual(person_segments_can_share_reid(
                        segment(first, self.RECT), segment(second, self.RECT)
                    ), expected)
        self.assertTrue(person_segments_can_share_reid(
            segment(self.local, self.RECT), segment(self.dense, self.RECT)
        ))

    def test_quality_components_and_bbox_edges_match_dense_masks(self):
        for data in (self.data, np.zeros_like(self.data), self.data * 255):
            for origin in ((40, 40), (0, 0), (80, 100)):
                dense, local = mask_pair(data, origin, self.SHAPE)
                rect = {"x": origin[0], "y": origin[1], "w": 40, "h": 100}
                ds, ls = segment(dense, rect), segment(local, rect)
                with self.subTest(origin=origin, value=int(data.max())):
                    self.assertEqual(evaluate_person_quality(ds, [ds], self.SHAPE),
                                     evaluate_person_quality(ls, [ls], self.SHAPE))
                    dense_bool = _valid_mask(dense, *self.SHAPE[:2])
                    local_bool = _valid_mask(local, *self.SHAPE[:2])
                    self.assertIsNotNone(local_bool)
                    self.assertEqual(sorted(_connected_component_areas(dense_bool)),
                                     sorted(_connected_component_areas(local_bool)))
                    self.assertEqual(_mask_bbox_edge_stats(dense_bool, rect),
                                     _mask_bbox_edge_stats(local_bool, rect))

    def test_image_edge_band_uses_original_frame_dimensions(self):
        # At 1080p the two-pixel edge band includes x=1; a one-pixel ROI band would miss it.
        shape = (1080, 1920)
        for origin, expected in [
            ((1, 200), {"left"}), ((300, 1), {"top"}),
            ((1899, 200), {"right"}), ((300, 979), {"bottom"}),
            ((300, 200), set()),
        ]:
            dense, local = mask_pair(np.ones((100, 20), dtype=np.uint8), origin, shape)
            with self.subTest(origin=origin):
                self.assertEqual(_mask_image_edge_directions(local), frozenset(expected))
                self.assertEqual(_mask_image_edge_directions(local), _mask_image_edge_directions(dense))

    def test_face_coverage_includes_zero_pixels_outside_local_support(self):
        faces = [{"xyxy": [20, 40, 50, 60], "conf": 0.99},
                 {"xyxy": [40, 40, 60, 60], "conf": 0.9}]
        for pose in (None, _person_test_pose(self.RECT)["keypoints"]):
            with self.subTest(has_pose=pose is not None):
                expected = select_owned_face_detection(faces, self.SHAPE, self.dense, pose)
                actual = select_owned_face_detection(faces, self.SHAPE, self.local, pose)
                self.assertIsNotNone(expected)
                self.assertEqual(expected.detection_index, 1)
                self.assertEqual(actual, expected)

    def test_shared_frame_face_binding_still_allows_duplicate_people(self):
        face = SimpleNamespace(source_rect=(40, 40, 60, 60), confidence=0.9)
        for masks in ((self.dense, self.dense), (self.local, self.local), (self.local, self.dense)):
            people = [SimpleNamespace(segment=segment(m, self.RECT), person_keypoints=None) for m in masks]
            bindings = bind_frame_faces_to_people([face], people, self.SHAPE)
            self.assertEqual([b.reason for b in bindings], ["bound", "bound"])
            self.assertTrue(all(b.frame_face is face for b in bindings))

    def test_crops_and_foreign_projection_are_pixel_exact_including_rotation(self):
        image = np.arange(np.prod(self.SHAPE), dtype=np.uint8).reshape(self.SHAPE)
        for origin in ((40, 40), (0, 0), (80, 100)):
            for angle in (0.0, -90.0, 27.5):
                for dtype in (np.uint8, np.bool_, np.uint64):
                    dense, local = mask_pair(self.data.astype(dtype), origin, self.SHAPE)
                    foreign_dense, foreign_local = mask_pair(
                        np.ones((80, 15), dtype=dtype), (origin[0] + 15, origin[1] + 10), self.SHAPE
                    )
                    rect = {"x": origin[0], "y": origin[1], "w": 40, "h": 100}
                    decision = PersonRotationDecision(angle, bool(angle), "applied" if angle else "disabled")
                    with self.subTest(origin=origin, angle=angle, dtype=dtype), patch(
                        "utils.keyframe.person_crop.estimate_person_upright_rotation", return_value=decision
                    ):
                        dc = build_person_crop(image, rect, dense, rotation_enabled=True)
                        lc = build_person_crop(image, rect, local, rotation_enabled=True)
                        self.assertIsNotNone(lc)
                        self.assertIsNotNone(lc.mask)
                        self.assertEqual(lc.rotation_applied, bool(angle))
                        for field in fields(dc):
                            expected, actual = getattr(dc, field.name), getattr(lc, field.name)
                            if isinstance(expected, np.ndarray):
                                np.testing.assert_array_equal(actual, expected, err_msg=field.name)
                                self.assertEqual(actual.dtype, expected.dtype)
                            else:
                                self.assertEqual(actual, expected, field.name)
                        np.testing.assert_array_equal(
                            project_mask_to_person_crop(foreign_local, lc, self.SHAPE),
                            project_mask_to_person_crop(foreign_dense, dc, self.SHAPE),
                        )
                        ds, ls = segment(dense, rect), segment(local, rect)
                        dense_quality = evaluate_person_crop_quality(
                            dc, ds, [ds, segment(foreign_dense, rect)], self.SHAPE, segment_index=0
                        )
                        local_quality = evaluate_person_crop_quality(
                            lc, ls, [ls, segment(foreign_local, rect)], self.SHAPE, segment_index=0
                        )
                        self.assertGreater(dense_quality.metrics.foreign_person_pixel_count, 0)
                        self.assertEqual(local_quality, dense_quality)


class PersonMaskObservationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_output_reaches_observation_quality_and_pose_binding(self):
        image = np.zeros((200, 120, 3), dtype=np.uint8)
        rect = {"x": 40, "y": 40, "w": 40, "h": 100}
        adapter = build_adapters()["person_seg"]
        outputs = {
            "DETECTIONS": np.asarray([[[40, 40, 80, 140, 0.9, 0]]], dtype=np.float32),
            "MASK_CROPS_PACKED": np.full((1, 1, 160, 20), 255, dtype=np.uint8),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
        }
        adapter.validate_outputs(outputs, 1)
        result = adapter.unpack_one(outputs, 0, adapter.normalize(image))
        poses = [{**_person_test_pose(rect), "rect": rect}]
        client = SimpleNamespace(extract_person_segments=AsyncMock(return_value=result),
                                 extract_person_poses=AsyncMock(return_value=poses))
        observations = await PersonObservationService(triton_client=client, logger=Mock()).detect(
            SimpleNamespace(sn="mask-test"), image
        )
        self.assertEqual(len(observations), 1)
        self.assertIs(observations[0].segment["pose"], poses[0])
        self.assertEqual(observations[0].segment["poseMaskAgreement"], 1.0)
        self.assertEqual(observations[0].mask_pixel_count, 4000)
        dense, _ = mask_pair(np.ones((100, 40), dtype=np.uint8), (40, 40), image.shape)
        reference = {**observations[0].segment, "mask": dense}
        self.assertEqual(observations[0].quality_decision,
                         evaluate_person_quality(reference, [reference], image.shape, segment_index=0))
