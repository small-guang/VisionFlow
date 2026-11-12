import math
import unittest

import numpy as np

from core.face_observation import (
    FaceProcessingStage,
    FrameFaceObservation,
    landmarks_in_expanded_crop,
)


class FrameFaceObservationTests(unittest.TestCase):
    def test_face_id_accepts_only_supported_raw_scalars(self):
        cases = (
            ("plain string", "face-123", "face-123"),
            ("trimmed UUID", "  123e4567-e89b-12d3-a456-426614174000  ", "123e4567-e89b-12d3-a456-426614174000"),
            ("integer", 42, "42"),
            ("integer zero", 0, "0"),
            ("string zero", "0", "0"),
            ("blank string", "   ", "0"),
            ("list", ["bad-id"], "0"),
            ("dict", {"id": "bad-id"}, "0"),
            ("float", 42.0, "0"),
            ("true", True, "0"),
            ("false", False, "0"),
            ("tuple", ("bad-id",), "0"),
        )

        for name, raw_face_id, expected in cases:
            with self.subTest(name=name):
                observation = FrameFaceObservation(
                    source_rect=(1, 2, 3, 4),
                    confidence=0.9,
                    assigned_face_id=raw_face_id,
                )
                self.assertEqual(observation.assigned_face_id, expected)

    def test_face_id_retains_raw_scalar_validity_after_normalization(self):
        cases = (
            ("string zero", "0", "0", True),
            ("integer zero", 0, "0", True),
            ("nonzero string", "face-123", "face-123", True),
            ("nonzero integer", 42, "42", True),
            ("list", ["bad-id"], "0", False),
            ("dict", {"id": "bad-id"}, "0", False),
            ("float", 42.0, "0", False),
            ("bool", True, "0", False),
        )

        for name, raw_face_id, normalized, is_valid in cases:
            with self.subTest(name=name):
                observation = FrameFaceObservation(
                    source_rect=(1, 2, 3, 4),
                    confidence=0.9,
                    assigned_face_id=raw_face_id,
                )
                self.assertEqual(observation.assigned_face_id, normalized)
                self.assertEqual(
                    observation.assigned_face_id_is_valid,
                    is_valid,
                )

    def test_arrays_are_copied_and_read_only(self):
        landmarks = np.array([[12, 22], [18, 22], [15, 26], [13, 30], [17, 30]], dtype=np.float32)
        embedding = np.arange(512, dtype=np.float32)
        observation = FrameFaceObservation(
            source_rect=(10, 20, 20, 32),
            confidence=0.9,
            expanded_source_rect=(5, 15, 25, 35),
            source_landmarks=landmarks,
            normalized_embedding=embedding,
            raw_embedding_norm=4.0,
            completed_stage=FaceProcessingStage.FEATURE,
        )
        landmarks[:] = 0
        embedding[:] = 0
        self.assertEqual(observation.source_landmarks[0].tolist(), [12.0, 22.0])
        self.assertEqual(float(observation.normalized_embedding[1]), 1.0)
        with self.assertRaises(ValueError):
            observation.source_landmarks[0, 0] = 99

    def test_landmarks_are_translated_into_expanded_crop(self):
        observation = FrameFaceObservation(
            source_rect=(10, 20, 20, 32),
            confidence=0.9,
            expanded_source_rect=(5, 15, 25, 35),
            source_landmarks=np.array([[12, 22], [18, 22], [15, 26], [13, 30], [17, 30]], dtype=np.float32),
            completed_stage=FaceProcessingStage.LANDMARKS,
        )
        actual = landmarks_in_expanded_crop(observation)
        np.testing.assert_allclose(actual[0], [7.0, 7.0])

    def test_missing_or_malformed_geometry_returns_none(self):
        observation = FrameFaceObservation(
            source_rect=(10, 20, 20, 32),
            confidence=0.9,
            completed_stage=FaceProcessingStage.DETECTION,
        )
        self.assertIsNone(landmarks_in_expanded_crop(observation))

    def test_rejects_non_four_value_rect(self):
        with self.assertRaises(ValueError):
            FrameFaceObservation(source_rect=(1, 2, 3), confidence=0.9)

    def test_rejects_nonfinite_embedding_norm(self):
        with self.assertRaises(ValueError):
            FrameFaceObservation(source_rect=(1, 2, 3, 4), confidence=0.9, raw_embedding_norm=math.inf)

    def test_rejects_nonfinite_or_non_vector_embedding(self):
        for embedding in (np.array([1.0, math.nan]), np.ones((2, 2))):
            with self.subTest(embedding=embedding):
                with self.assertRaises(ValueError):
                    FrameFaceObservation(source_rect=(1, 2, 3, 4), confidence=0.9, normalized_embedding=embedding)


if __name__ == "__main__":
    unittest.main()
