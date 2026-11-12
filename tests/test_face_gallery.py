import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from pymilvus import MilvusClient
from core.algorithm_config import algorithm_config


def load_gallery_modules():
    """Load the real adapter without starting the package's four live clients."""
    directory = Path(__file__).resolve().parents[1] / "db" / "zilliz"
    package_name = "face_quality_gallery_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    config = types.ModuleType("core.config")
    config.settings = types.SimpleNamespace(
        milvus_uri="http://localhost:19530",
        milvus_token="test-token",
    )
    with patch.dict(sys.modules, {package_name: package, "core.config": config}):
        modules = []
        for name in ("picture_feature", "face_feature"):
            module_name = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(module_name, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        return modules


class MemoryMilvusClient:
    """Only the external Milvus operations are replaced; policy stays real."""

    create_schema = staticmethod(MilvusClient.create_schema)

    def __init__(self):
        self.rows = {}
        self.schema = None
        self.fail_before_write = False
        self.lose_write_reply = False
        self.empty_write_reply = False
        self.max_count_seen = 0

    def has_collection(self, collection_name):
        return self.schema is not None

    def create_collection(self, *, collection_name, schema, **kwargs):
        self.schema = schema.to_dict()

    def describe_collection(self, collection_name):
        return copy.deepcopy(self.schema)

    def list_indexes(self, **kwargs):
        return ["embedding"]

    def load_collection(self, collection_name):
        pass

    def query(self, *, collection_name, filter, output_fields, **kwargs):
        field, raw_value = filter.split(" == ", 1)
        value = json.loads(raw_value)
        # Let another worker race if the adapter's update lock is removed.
        time.sleep(0.001)
        rows = [
            {key: copy.deepcopy(row[key]) for key in output_fields}
            for row in self.rows.values()
            if row[field] == value
        ]
        return rows[:kwargs.get("limit", len(rows))]

    def query_iterator(self, *, batch_size, **kwargs):
        rows = self.query(**kwargs)

        class Iterator:
            def next(self):
                batch = rows[:batch_size]
                del rows[:batch_size]
                return batch

            def close(self):
                pass

        return Iterator()

    def upsert(self, *, collection_name, data):
        if self.fail_before_write:
            self.fail_before_write = False
            raise RuntimeError("write unavailable")
        if self.empty_write_reply:
            return {"upsert_count": 0, "ids": []}
        for row in data:
            self.rows[row["id"]] = copy.deepcopy(row)
        self.max_count_seen = max(self.max_count_seen, len(self.rows))
        if self.lose_write_reply:
            self.lose_write_reply = False
            raise RuntimeError("write reply lost")
        return {"upsert_count": len(data), "ids": [row["id"] for row in data]}

    def delete(self, *, collection_name, ids):
        for key in ids:
            self.rows.pop(key, None)
        return {"delete_count": len(ids)}


class FaceGalleryQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.picture_module, cls.face_module = load_gallery_modules()

    def setUp(self):
        self.client = MemoryMilvusClient()
        self.gallery = self.make_gallery()

    def make_gallery(self, limit=20):
        with patch.object(self.picture_module, "MilvusClient", return_value=self.client) as factory:
            factory.prepare_index_params.side_effect = MilvusClient.prepare_index_params
            return self.face_module.FaceGallery(
                collection_name="alogFace", dim=2, max_vectors_per_face=limit
            )

    def test_default_capacity_uses_algorithm_profile(self):
        profile = types.SimpleNamespace(
            face=types.SimpleNamespace(
                gallery=types.SimpleNamespace(max_vectors_per_id=7)
            )
        )
        with patch.object(self.face_module, "algorithm_config", profile):
            gallery = self.make_gallery(limit=None)
        self.assertEqual(gallery.max_vectors_per_face, 7)

    def write(self, quality, sample, *, face="face-1", timestamp=100, device="CAM001"):
        return self.gallery.insert_embeddings(
            faceId=face,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            deviceId=device,
            trackId=0,
            keyframeId=42,
            faceRect={},
            faceKeyPoints=[],
            faceCropPath=f"{device}/face_crop/{sample}.jpg",
            timestamp=timestamp,
            qualityScore=quality,
            sampleId=sample,
        )

    def search_hits(self, hits, threshold=0.45):
        # Only the external search is stubbed; exercise the adapter's decision.
        with patch.object(self.client, "search", return_value=[hits], create=True):
            return self.gallery.search_similar_faces_with_hit_info(
                np.array([1.0, 0.0], dtype=np.float32), threshold=threshold
            )

    @staticmethod
    def hit(score, sample, face="face-1", crop=None):
        return {
            "id": sample,
            "distance": score,
            "entity": {
                "faceId": face,
                "deviceId": "CAM001",
                "sampleId": sample,
                "faceCropPath": crop or f"CAM001/face_crop/{sample}.jpg",
            },
        }

    def test_gray_match_requires_two_distinct_samples_of_the_same_identity(self):
        cases = [
            ([self.hit(0.48, "one")], None),
            ([self.hit(0.49, "one"), self.hit(0.42, "two")], None),
            ([self.hit(0.48, "one"), self.hit(0.46, "two")], "face-1"),
            ([self.hit(0.48, "one"), self.hit(0.46, "two", "face-2")], None),
            ([self.hit(0.48, "one"), self.hit(0.48, "one")], None),
            ([self.hit(0.48, "one", crop="same.jpg"),
              self.hit(0.48, "two", crop="same.jpg")], None),
        ]
        for hits, expected in cases:
            with self.subTest(hits=hits):
                face_id, score, hit_id = self.search_hits(hits)
                self.assertEqual(face_id, expected)
                self.assertEqual(score, hits[0]["distance"])
                self.assertEqual(hit_id, "face-1")

    def test_match_threshold_boundaries_and_custom_lower_threshold(self):
        cases = [
            (0.45, [self.hit(0.55, "one")], "face-1"),
            (0.45, [self.hit(0.549, "one")], None),
            (0.45, [self.hit(0.45, "one"), self.hit(0.45, "two")], "face-1"),
            (0.45, [self.hit(0.449, "one"), self.hit(0.449, "two")], None),
            (0.48, [self.hit(0.49, "one"), self.hit(0.47, "two")], None),
            (0.6, [self.hit(0.55, "one")], None),
        ]
        for threshold, hits, expected in cases:
            with self.subTest(threshold=threshold, hits=hits):
                self.assertEqual(self.search_hits(hits, threshold)[0], expected)

    def test_best_single_sample_cannot_be_diluted_by_extra_history(self):
        result = self.search_hits([
            self.hit(0.8, "one", "face-a"),
            self.hit(0.46, "two", "face-a"),
            self.hit(0.7, "three", "face-b"),
        ])
        self.assertEqual(result, ("face-a", 0.8, "face-a"))

    def test_uncertain_best_identity_does_not_fall_back_to_a_weaker_identity(self):
        result = self.search_hits([
            self.hit(0.49, "one", "face-a"),
            self.hit(0.48, "two", "face-b"),
            self.hit(0.47, "three", "face-b"),
        ])
        self.assertEqual(result, (None, 0.49, "face-a"))

    def test_more_than_twenty_observations_keep_twenty_best_scores(self):
        for i in range(40):
            self.write(7.0 + i / 20, f"sample-{i}")
        self.assertEqual(len(self.client.rows), 20)
        self.assertEqual(self.client.max_count_seen, 20)
        self.assertEqual(
            sorted(row["qualityScore"] for row in self.client.rows.values()),
            [8.0 + i / 20 for i in range(20)],
        )

    def test_lower_score_is_skipped_without_changing_existing_vectors(self):
        self.gallery = self.make_gallery(limit=1)
        self.write(9.0, "best")
        before = copy.deepcopy(self.client.rows)
        result = self.write(7.1, "worse", timestamp=999)
        self.assertEqual(self.client.rows, before)
        self.assertEqual(result["outcomes"][0]["action"], "skipped")
        self.assertIsNone(result["outcomes"][0]["primary_key"])
        self.assertTrue(result["outcomes"][0]["persisted"])

    def test_higher_score_replaces_same_integer_primary_key_across_devices(self):
        self.gallery = self.make_gallery(limit=1)
        self.write(7.1, "old")
        old_id = next(iter(self.client.rows))
        self.write(9.0, "best", device="CAM002")
        self.assertEqual(list(self.client.rows), [old_id])
        self.assertIsInstance(old_id, int)
        self.assertEqual(self.client.rows[old_id]["deviceId"], "CAM002")
        self.assertEqual(self.client.rows[old_id]["qualityScore"], 9.0)

    def test_equal_quality_prefers_newer_capture_and_rejects_out_of_order_sample(self):
        self.gallery = self.make_gallery(limit=1)
        self.write(8.1, "old", timestamp=100)
        self.write(8.1, "new", timestamp=200)
        self.write(8.1, "late-arrival", timestamp=150)
        self.assertEqual([row["sampleId"] for row in self.client.rows.values()], ["new"])

    def test_exact_same_rank_does_not_churn_gallery(self):
        self.gallery = self.make_gallery(limit=1)
        self.write(8.1, "first")
        self.write(8.1, "second")
        self.assertEqual([row["sampleId"] for row in self.client.rows.values()], ["first"])

    def test_capacity_is_per_face_across_all_devices(self):
        self.gallery = self.make_gallery(limit=2)
        for i in range(5):
            for face in ("face-a", "face-b"):
                self.write(7.0 + i / 10, f"{face}-{i}", face=face, device=f"CAM{i}")
        self.assertEqual(len(self.client.rows), 4)
        for face in ("face-a", "face-b"):
            self.assertEqual(sorted(
                row["qualityScore"] for row in self.client.rows.values() if row["faceId"] == face
            ), [7.3, 7.4])

    def test_retry_after_lost_reply_does_not_add_duplicate(self):
        self.client.lose_write_reply = True
        with self.assertRaisesRegex(RuntimeError, "reply lost"):
            self.write(8.0, "retry")
        result = self.write(8.0, "retry")
        self.assertEqual(len(self.client.rows), 1)
        self.assertEqual(result["outcomes"][0]["action"], "already_persisted")

    def test_failed_replacement_preserves_old_vector_and_retry_improves_it(self):
        self.gallery = self.make_gallery(limit=1)
        self.write(7.5, "old")
        self.client.fail_before_write = True
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.write(9.0, "new")
        self.assertEqual([row["sampleId"] for row in self.client.rows.values()], ["old"])
        self.write(9.0, "new")
        self.assertEqual([row["sampleId"] for row in self.client.rows.values()], ["new"])

    def test_empty_write_confirmation_is_a_failure(self):
        self.client.empty_write_reply = True
        with self.assertRaisesRegex(RuntimeError, "confirm"):
            self.write(8.0, "empty")
        self.assertEqual(self.client.rows, {})

    def test_invalid_scores_never_enter_gallery(self):
        for score in (None, float("nan"), float("inf"), -1,
                      algorithm_config.face.quality.enroll_min_score - 0.01, 10.1):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    self.write(score, "invalid")
        self.assertEqual(self.client.rows, {})

    def test_parallel_writes_preserve_capacity_and_best_samples(self):
        self.gallery = self.make_gallery(limit=3)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.write, 7.0 + i / 10, f"parallel-{i}") for i in range(20)]
            for future in futures:
                future.result()
        self.assertEqual(sorted(row["qualityScore"] for row in self.client.rows.values()), [8.7, 8.8, 8.9])
        self.assertEqual(self.client.max_count_seen, 3)

    def test_lowered_limit_prunes_worst_existing_samples_on_next_write(self):
        for i in range(4):
            self.write(7.0 + i / 10, f"sample-{i}")
        self.gallery = self.make_gallery(limit=2)
        self.write(7.0, "skipped")
        self.assertEqual(sorted(row["qualityScore"] for row in self.client.rows.values()), [7.2, 7.3])

    def test_existing_legacy_schema_requires_migration(self):
        self.client.schema["auto_id"] = True
        with self.assertRaisesRegex(RuntimeError, "Incompatible face gallery schema"):
            self.make_gallery()

    def test_migrated_identity_over_one_thousand_rows_is_fully_pruned(self):
        self.client.rows = {
            index: {"id": index, "faceId": "face-1", "sampleId": f"legacy-{index}",
                    "qualityScore": 7.0, "timestamp": index}
            for index in range(1, 1006)
        }
        self.write(8.0, "new")
        self.assertEqual(len(self.client.rows), 20)
        self.assertEqual(sorted(
            row["timestamp"] for row in self.client.rows.values() if row["qualityScore"] == 7.0
        ), list(range(987, 1006)))


if __name__ == "__main__":
    unittest.main()
