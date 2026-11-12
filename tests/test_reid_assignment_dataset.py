import json
from pathlib import Path
import unittest

import numpy as np

from utils.reid.person_body import gallery_body_types_for_query, person_sbs_threshold
from utils.reid.prototype_policy import plan_prototype_update
from utils.reid.reid_models import (
    REID_SOURCE_MILVUS,
    ReidCandidate,
)
from utils.reid.reid_resolution import resolve_reid_classes


DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "alog_reid_e2e_v2"
    / "reid"
    / "assignment_purity"
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@unittest.skipUnless(
    DATASET_ROOT.is_dir(),
    "assignment_purity dataset is not available in this workspace",
)
class AssignmentPurityDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocols = json.loads((DATASET_ROOT / "protocols.json").read_text("utf-8"))
        cls.samples = read_jsonl(DATASET_ROOT / "samples.jsonl")
        cls.sample_by_id = {row["sample_id"]: row for row in cls.samples}
        cls.embeddings = np.load(DATASET_ROOT / "embeddings.npy", mmap_mode="r")

    def test_dataset_protocol_matches_product_identity_contract(self):
        self.assertEqual(self.protocols["identity_axis"], "person_id")
        self.assertIn("no more than one person", self.protocols["assignment_invariant"])
        self.assertIn("materially different appearances", self.protocols["allowed"])
        self.assertEqual(
            self.protocols["appearance_assignment_target"],
            "one person and one appearance should converge to one reid",
        )

    def test_all_confirmed_pollution_pairs_are_rejected_by_face_invariant(self):
        checked = 0
        for pair in read_jsonl(DATASET_ROOT / "pairs.jsonl"):
            if pair["pair_type"] != "same_observed_reid_different_person_pollution":
                continue
            left = self.sample_by_id[pair["left_sample_id"]]
            right = self.sample_by_id[pair["right_sample_id"]]
            resolution = resolve_reid_classes(
                [
                    ReidCandidate(
                        reid="observed-polluted-reid",
                        score=float(pair["cosine_similarity"]),
                        source=REID_SOURCE_MILVUS,
                        threshold=-1.0,
                        body_completeness=left["body_completeness"],
                        face_id=left["person_id"],
                        device_id=left["device_id"],
                    )
                ],
                query_face_id=right["person_id"],
                query_device_id=right["device_id"],
            )
            self.assertIsNone(resolution.reid, pair["pair_id"])
            self.assertEqual(resolution.reason, "no_compatible_class")
            checked += 1
        self.assertEqual(checked, 919)

    def test_alias_stress_keeps_two_pure_reids_per_person(self):
        alias_root = DATASET_ROOT / "expected_db" / "alias_stress"
        records = read_jsonl(alias_root / "records.jsonl")
        by_reid = {}
        by_person = {}
        for record in records:
            by_reid.setdefault(record["alias_stress_reid"], set()).add(
                record["person_id"]
            )
            by_person.setdefault(record["person_id"], set()).add(
                record["alias_stress_reid"]
            )
        self.assertEqual(len(by_reid), 60)
        self.assertTrue(all(len(persons) == 1 for persons in by_reid.values()))
        self.assertEqual(len(by_person), 30)
        self.assertTrue(all(len(reids) == 2 for reids in by_person.values()))

    def test_replay_writeback_remains_pure_while_allowing_fragmentation(self):
        replay_root = DATASET_ROOT / "replay"
        replay_samples = read_jsonl(replay_root / "samples.jsonl")
        replay_vectors = np.load(replay_root / "embeddings.npy", mmap_mode="r")
        gallery = {}
        assignments = {}
        new_counter = 0
        writes = 0
        cross_device_matches = 0

        for sample in sorted(replay_samples, key=lambda row: int(row["timestamp"])):
            query = np.asarray(
                replay_vectors[int(sample["embedding_index"])],
                dtype=np.float32,
            )
            allowed_bodies = gallery_body_types_for_query(sample["body_completeness"])
            candidates = []
            for reid, records in gallery.items():
                for prototype_index, record in enumerate(records):
                    if record["bodyCompleteness"] not in allowed_bodies:
                        continue
                    score = float(np.dot(query, record["embedding"]))
                    candidates.append(
                        ReidCandidate(
                            reid=reid,
                            score=score,
                            source=REID_SOURCE_MILVUS,
                            threshold=person_sbs_threshold(
                                sample["body_completeness"],
                                record["bodyCompleteness"],
                            ),
                            body_completeness=record["bodyCompleteness"],
                            face_id=record["faceId"],
                            device_id=record["deviceId"],
                            prototype_id=prototype_index,
                        )
                    )
            resolution = resolve_reid_classes(
                candidates,
                query_face_id=sample["person_id"],
                query_device_id=sample["device_id"],
            )
            assigned_reid = resolution.reid
            if assigned_reid is not None and all(
                record["deviceId"] != sample["device_id"]
                for record in gallery[assigned_reid]
            ):
                cross_device_matches += 1
            if assigned_reid is None and int(sample["quality_level"]) >= 2:
                new_counter += 1
                assigned_reid = f"dataset-reid-{new_counter:04d}"
            if assigned_reid is None:
                continue

            assignments.setdefault(assigned_reid, set()).add(sample["person_id"])
            if int(sample["quality_level"]) < 2:
                continue
            new_record = {
                "id": writes + 1,
                "faceId": sample["person_id"],
                "reId": assigned_reid,
                "bodyCompleteness": sample["body_completeness"],
                "embedding": query,
                "deviceId": sample["device_id"],
                "qualityScore": float(sample["quality_score"]),
                "timestamp": int(sample["timestamp"]),
            }
            plan = plan_prototype_update(gallery.get(assigned_reid, []), new_record)
            if plan.writes_vector:
                gallery.setdefault(assigned_reid, []).append(new_record)
                writes += 1

        impure = {reid: persons for reid, persons in assignments.items() if len(persons) > 1}
        self.assertGreater(writes, 0)
        self.assertGreater(cross_device_matches, 0)
        self.assertEqual(impure, {})
        self.assertTrue(all(len(records) <= 24 for records in gallery.values()))

    def test_body_only_replay_matches_current_lower_threshold_baseline(self):
        expected_root = DATASET_ROOT / "expected_db"
        gallery_records = read_jsonl(expected_root / "records.jsonl")
        gallery_vectors = np.load(expected_root / "embeddings.npy", mmap_mode="r")
        replay_root = DATASET_ROOT / "replay"
        replay_samples = read_jsonl(replay_root / "samples.jsonl")
        replay_vectors = np.load(replay_root / "embeddings.npy", mmap_mode="r")
        person_by_reid = {
            record["expected_reid_id"]: record["person_id"]
            for record in gallery_records
        }
        accepted = 0
        correct = 0
        false_matches = 0

        for sample in replay_samples:
            query = np.asarray(
                replay_vectors[int(sample["embedding_index"])],
                dtype=np.float32,
            )
            allowed_bodies = gallery_body_types_for_query(
                sample["body_completeness"]
            )
            candidates = []
            for index, record in enumerate(gallery_records):
                if record["bodyCompleteness"] not in allowed_bodies:
                    continue
                score = float(
                    np.dot(
                        query,
                        np.asarray(gallery_vectors[index], dtype=np.float32),
                    )
                )
                candidates.append(
                    ReidCandidate(
                        reid=record["expected_reid_id"],
                        score=score,
                        source=REID_SOURCE_MILVUS,
                        threshold=person_sbs_threshold(
                            sample["body_completeness"],
                            record["bodyCompleteness"],
                        ),
                        body_completeness=record["bodyCompleteness"],
                        face_id=record["person_id"],
                        device_id=record["deviceId"],
                        prototype_id=index,
                    )
                )
            resolution = resolve_reid_classes(
                candidates,
                query_device_id=sample["device_id"],
            )
            if resolution.reid is None:
                continue
            accepted += 1
            if person_by_reid[resolution.reid] == sample["person_id"]:
                correct += 1
            else:
                false_matches += 1

        # Intentional characterization of current thresholds, not a purity release gate.
        self.assertEqual((accepted, correct, false_matches), (8, 4, 4))


if __name__ == "__main__":
    unittest.main()
