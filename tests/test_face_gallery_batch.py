"""Face 批内读取复用保持逐条质量决策、数据库确认和整批重试边界。"""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from tests.test_face_gallery import MemoryMilvusClient, load_gallery_modules


class BatchMilvusClient(MemoryMilvusClient):
    def __init__(self):
        super().__init__()
        self.queries = []
        self.writes = []
        self.deletes = []
        self.fail_write_number = None
        self.failure_mode = "before"
        self.fail_delete_number = None
        self.observe = lambda: None

    def query_iterator(self, **kwargs):
        self.observe()
        self.queries.append(copy.deepcopy(kwargs))
        return super().query_iterator(**kwargs)

    def upsert(self, **kwargs):
        self.observe()
        self.writes.append(copy.deepcopy(kwargs["data"]))
        fail = len(self.writes) == self.fail_write_number
        if fail and self.failure_mode == "before":
            raise RuntimeError("write unavailable")
        result = super().upsert(**kwargs)
        if fail and self.failure_mode == "lost_reply":
            raise RuntimeError("write reply lost")
        if fail and self.failure_mode == "short_confirmation":
            return {"upsert_count": 0}
        return result

    def delete(self, **kwargs):
        self.observe()
        self.deletes.append(list(kwargs["ids"]))
        result = super().delete(**kwargs)
        if len(self.deletes) == self.fail_delete_number:
            return {"delete_count": 0}
        return result


class FaceGalleryBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.face_module = load_gallery_modules()

    def make_gallery(self, limit=3):
        with patch.object(self.face_module.MilvusVectorDB, "__init__", return_value=None):
            gallery = self.face_module.FaceGallery(max_vectors_per_face=limit)
        gallery.collection_name = "alogFace"
        gallery.dim = 2
        gallery.client = BatchMilvusClient()
        return gallery

    @staticmethod
    def record(sample, quality=8.0, *, face="face-a", timestamp=100):
        return dict(
            faceId=face, embedding=[1.0, 0.0], deviceId="camera", trackId=0,
            keyframeId=1, faceRect={}, faceKeyPoints=[], faceCropPath=f"{sample}.jpg",
            timestamp=timestamp, qualityScore=quality, sampleId=sample,
        )

    @staticmethod
    def write_batch(gallery, records):
        return gallery.insert_embeddings(**{
            key: [record[key] for record in records] for key in records[0]
        })

    def assert_matches_sequential(self, records, *, limit):
        batch, sequential = self.make_gallery(limit), self.make_gallery(limit)
        actual = self.write_batch(batch, records)
        expected = [sequential.insert_embeddings(**record)["outcomes"][0] for record in records]
        self.assertEqual(actual["outcomes"], expected)
        self.assertEqual(batch.client.rows, sequential.client.rows)
        self.assertEqual(batch.client.writes, sequential.client.writes)
        self.assertEqual(len(batch.client.queries), len({record["faceId"] for record in records}))
        self.assertTrue(all(query["consistency_level"] == "Strong" for query in batch.client.queries))
        return batch, actual

    def test_interleaved_faces_and_repeated_samples_keep_original_write_order(self):
        records = [
            self.record("a-old", 7),
            self.record("b-old", 8, face="face-b"),
            self.record("a-new", 9),
            self.record("a-old", 7),  # 已被替换的 sampleId 不应误当作仍持久化。
            self.record("b-old", 8, face="face-b"),
            self.record("a-new", 9),
            self.record("b-new", 8, face="face-b", timestamp=200),
        ]
        _, result = self.assert_matches_sequential(records, limit=1)
        self.assertEqual([row["action"] for row in result["outcomes"]], [
            "inserted", "inserted", "replaced", "skipped",
            "already_persisted", "already_persisted", "replaced",
        ])

    def test_batch_capacity_and_equal_quality_timestamp_match_sequential_updates(self):
        records = [self.record(f"sample-{i}", 7 + (i % 5) / 2,
                               face=f"face-{i % 2}", timestamp=(i * 7) % 13)
                   for i in range(30)]
        records.extend([dict(records[-1]), self.record("same-rank", 9, face="face-0", timestamp=12)])
        batch, _ = self.assert_matches_sequential(records, limit=3)
        for face in ("face-0", "face-1"):
            self.assertLessEqual(sum(row["faceId"] == face for row in batch.client.rows.values()), 3)

    def test_retry_discards_batch_view_after_partial_or_unconfirmed_write(self):
        for mode, persisted_before_retry in (("before", 1), ("lost_reply", 2),
                                              ("short_confirmation", 2)):
            with self.subTest(mode=mode):
                gallery = self.make_gallery()
                records = [self.record(f"sample-{i}") for i in range(3)]
                gallery.client.fail_write_number = 2
                gallery.client.failure_mode = mode
                with self.assertRaises(RuntimeError):
                    self.write_batch(gallery, records)
                self.assertEqual(len(gallery.client.rows), persisted_before_retry)
                self.assertEqual(len(gallery.client.writes), 2)
                self.assertEqual(gallery.candidate_revision(), (2, 0))
                gallery.client.fail_write_number = None
                result = self.write_batch(gallery, records)
                self.assertEqual([row["action"] for row in result["outcomes"]],
                                 ["already_persisted"] * persisted_before_retry
                                 + ["inserted"] * (3 - persisted_before_retry))
                self.assertEqual(len(gallery.client.rows), 3)
                self.assertEqual(gallery.client.max_count_seen, 3)
                self.assertEqual(len(gallery.client.queries), 2)
                self.assertEqual(gallery.candidate_revision(), (4, 0))

    def test_failed_replacement_does_not_advance_view_or_process_later_record(self):
        gallery = self.make_gallery(limit=1)
        gallery.insert_embeddings(**self.record("old", 7))
        records = [self.record("better", 9), self.record("later", 8)]
        gallery.client.fail_write_number = 2
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.write_batch(gallery, records)
        self.assertEqual([row["sampleId"] for row in gallery.client.rows.values()], ["old"])
        gallery.client.fail_write_number = None
        result = self.write_batch(gallery, records)
        self.assertEqual([row["action"] for row in result["outcomes"]], ["replaced", "skipped"])
        self.assertEqual([row["sampleId"] for row in gallery.client.rows.values()], ["better"])

    def test_capacity_cleanup_failure_requeries_actual_remaining_rows_on_retry(self):
        gallery = self.make_gallery(limit=2)
        gallery.client.rows = {
            index: dict(self.record(f"legacy-{index}", 7, timestamp=index), id=index)
            for index in range(1005)
        }
        gallery.client.fail_delete_number = 2
        records = [self.record("new", 9), self.record("second", 8)]
        with self.assertRaisesRegex(RuntimeError, "capacity cleanup"):
            self.write_batch(gallery, records)
        self.assertEqual(len(gallery.client.writes), 0)
        self.assertEqual(len(gallery.client.rows), 2)
        gallery.client.fail_delete_number = None
        result = self.write_batch(gallery, records)
        self.assertEqual([row["action"] for row in result["outcomes"]], ["replaced", "replaced"])
        self.assertEqual({row["sampleId"] for row in gallery.client.rows.values()}, {"new", "second"})
        self.assertEqual(len(gallery.client.queries), 2)

    def test_successful_batch_view_does_not_survive_the_next_batch(self):
        gallery = self.make_gallery(limit=1)
        self.write_batch(gallery, [self.record("old", 7)])
        row = next(iter(gallery.client.rows.values()))
        row.update(sampleId="persisted-elsewhere", qualityScore=9)
        result = self.write_batch(gallery, [self.record("lower", 8)])
        self.assertEqual(result["outcomes"][0]["action"], "skipped")
        self.assertEqual(len(gallery.client.queries), 2)
        self.assertEqual(next(iter(gallery.client.rows.values()))["sampleId"], "persisted-elsewhere")

    def test_whole_batch_validation_precedes_reads_or_writes(self):
        gallery = self.make_gallery()
        records = [self.record("valid"), self.record("invalid", float("nan"))]
        with self.assertRaises(ValueError):
            self.write_batch(gallery, records)
        self.assertEqual(gallery.client.queries, [])
        self.assertEqual(gallery.client.writes, [])
        self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_queries_cleanup_and_all_writes_keep_the_whole_batch_revision_active(self):
        gallery = self.make_gallery(limit=2)
        gallery.client.rows = {
            index: dict(self.record(f"old-{index}", 7), id=index) for index in range(3)
        }
        states = []
        gallery.client.observe = lambda: states.append(gallery.candidate_revision())
        self.write_batch(gallery, [self.record("one", 9), self.record("two", 8),
                                   self.record("three", 8, face="face-b")])
        self.assertEqual(states, [(1, 1)] * len(states))
        self.assertEqual(len(states), 6)  # 两身份查询，一次清理，三次逐条upsert。
        self.assertEqual(gallery.candidate_revision(), (2, 0))


if __name__ == "__main__":
    unittest.main()
