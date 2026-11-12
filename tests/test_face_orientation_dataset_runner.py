import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tests.image_execution_test_support import InlineImageExecutor

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
import run_face_orientation_dataset as runner


class DummyLogger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


@dataclass
class FakeObservation:
    people_rect: dict
    segment: dict
    person_keypoints: object = None


class FakeFaceClient:
    def __init__(self):
        self.keypoint_calls = 0
        self.keypoint_images = []
        self.keypoint_bboxes = []
        self.detections = [{"xyxy": [2, 3, 12, 13], "conf": 0.8}]
        self.landmarks = np.array(
            [[3, 3], [8, 3], [6, 6], [4, 9], [8, 9]], dtype=np.float32
        )
        self.client = SimpleNamespace(batch_tasks={}, pool=SimpleNamespace(_pool=[]))

    async def extract_face_detections(self, _image):
        return self.detections

    async def extract_face_keypoints(self, _image, _bbox):
        self.keypoint_calls += 1
        self.keypoint_images.append(_image.copy())
        self.keypoint_bboxes.append(_bbox.copy())
        return self.landmarks


class FakeObservationService:
    async def detect(self, _ctx, image):
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        return [
            FakeObservation(
                people_rect={"x": 1, "y": 2, "w": 14, "h": 15},
                segment={"mask": mask},
                person_keypoints=np.zeros((17, 3), dtype=np.float32),
            )
        ]


class FakePreparationService:
    async def prepare(self, _ctx, image, _oss, observation, _segments):
        crop = SimpleNamespace(image=image[1:17, 1:17], mask=np.ones((16, 16), dtype=np.uint8))
        return SimpleNamespace(
            observation=observation,
            crop_result=crop,
            cropped=crop.image,
            mask_crop=crop.mask,
        )


class CloseableGrpcClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FailingGrpcClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True
        raise RuntimeError("close failed")


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def _services(self, matcher=None):
        async def default_matcher(**_kwargs):
            return {
                "face_orientation": "front",
                "pitch": 1.0,
                "yaw": 2.0,
                "roll": 3.0,
                "info": "fake face match",
            }

        return runner.RunnerServices(
            clip_rect=runner._fallback_clip_rect,
            expand_rect=runner._fallback_expand_rect,
            estimate_orientation=lambda _landmarks, _shape: SimpleNamespace(
                label=" SIDE ", pitch=np.float32(1.25), yaw=np.float64(31.5), roll=2
            ),
            normalize_orientation=lambda value: str(value).strip().lower(),
            observation_factory=FakeObservationService,
            preparation_factory=FakePreparationService,
            pose_keypoints_to_crop=lambda points, _crop: points,
            matcher=matcher or default_matcher,
            logger=DummyLogger(),
        )

    def test_production_count_validator_accepts_any_nonempty_jpg_collection(self):
        for count in (1, 19):
            with self.subTest(count=count):
                accepted = [Path(f"image-{index}.jpg") for index in range(count)]
                self.assertEqual(runner.validate_image_count(accepted), accepted)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected at least 1 JPG image for production evaluation, found 0",
        ):
            runner.validate_image_count([])

    async def test_collect_input_images_accepts_only_top_level_lowercase_jpg_files(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            nested_dir = input_dir / "nested"
            nested_dir.mkdir(parents=True)
            for name in ("zebra.jpg", "alpha.jpg", "upper.JPG", "mixed.Jpg", "photo.jpeg"):
                (input_dir / name).write_bytes(b"image")
            (nested_dir / "nested.jpg").write_bytes(b"image")

            self.assertEqual(
                [path.name for path in runner.collect_input_images(input_dir)],
                ["alpha.jpg", "zebra.jpg"],
            )

            uppercase_only = Path(directory) / "uppercase-only"
            uppercase_only.mkdir()
            (uppercase_only / "source.JPG").write_bytes(b"image")
            constructed = []
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                await runner.run_production(
                    uppercase_only,
                    Path(directory) / "output",
                    client_factory=lambda: constructed.append(True),
                )

        self.assertEqual(constructed, [])

    async def test_production_evaluation_processes_all_sorted_input_jpgs(self):
        client = FakeFaceClient()
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            for index in range(18, -1, -1):
                path = input_dir / f"source-{index:02d}.jpg"
                path.write_bytes(
                    cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
                )
            (input_dir / "ignored.jpeg").write_bytes(b"not a jpg")

            result = await runner.run_evaluation(
                runner.collect_input_images(input_dir),
                Path(directory) / "output",
                client,
                self._services(),
            )

        expected_names = [f"source-{index:02d}.jpg" for index in range(19)]
        self.assertEqual(result["inputCount"], 19)
        self.assertEqual([image["image"] for image in result["images"]], expected_names)

    async def test_face_evaluation_calls_keypoints_once_and_normalizes_angles(self):
        client = FakeFaceClient()
        client.detections = [{"xyxy": [10, 10, 14, 14], "conf": 0.8}]
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        estimated = []

        def estimate(landmarks, shape):
            estimated.append((landmarks, shape))
            return SimpleNamespace(label=" SIDE ", pitch=1.25, yaw=31.5, roll=2)

        faces = await runner.evaluate_faces(
            image,
            client,
            replace(self._services(), estimate_orientation=estimate),
        )

        self.assertEqual(client.keypoint_calls, 1)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["faceOrientation"], "side")
        self.assertEqual(faces[0]["pitch"], 1.25)
        self.assertEqual(faces[0]["yaw"], 31.5)
        self.assertEqual(faces[0]["roll"], 2.0)
        self.assertEqual(client.keypoint_images[0].shape, (8, 8, 3))
        np.testing.assert_array_equal(
            client.keypoint_bboxes[0], np.array([2, 2, 6, 6], dtype=np.float32)
        )
        self.assertEqual(estimated[0][1], image.shape)
        np.testing.assert_array_equal(
            estimated[0][0],
            np.array(
                [[11, 11], [16, 11], [14, 14], [12, 17], [16, 17]],
                dtype=np.float32,
            ),
        )

    async def test_face_evaluation_maps_malformed_landmarks_to_unknown_safely(self):
        from utils.quality.face_orientation import (
            estimate_face_orientation,
            normalize_face_orientation,
        )

        client = FakeFaceClient()
        client.landmarks = "not-landmarks"
        services = replace(
            self._services(),
            estimate_orientation=estimate_face_orientation,
            normalize_orientation=normalize_face_orientation,
        )

        faces = await runner.evaluate_faces(
            np.zeros((24, 24, 3), dtype=np.uint8), client, services
        )

        self.assertEqual(client.keypoint_calls, 1)
        self.assertEqual(faces[0]["faceOrientation"], "unknown")

    async def test_person_evaluation_disables_recognition_and_never_searches(self):
        calls = []

        async def matcher(**kwargs):
            calls.append(kwargs)
            return {
                "face_orientation": " SIDE ",
                "pitch": 5,
                "yaw": 25,
                "roll": 1,
                "info": "person result",
            }

        def forbidden_search(*_args, **_kwargs):
            raise AssertionError("person evaluation must not search identities")

        image = np.zeros((24, 24, 3), dtype=np.uint8)
        people = await runner.evaluate_persons(
            image,
            FakeFaceClient(),
            self._services(matcher),
            runner.KeyframeContext("dataset", 0, "sample.jpg"),
            forbidden_search,
        )

        self.assertEqual(len(people), 1)
        self.assertFalse(calls[0]["enable_recognition"])
        self.assertEqual(people[0]["faceOrientation"], "side")
        self.assertEqual(people[0]["peopleRect"]["w"], 14)

    async def test_evaluation_writes_json_annotations_and_closes_runtime_resources(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        async def batch_worker():
            await asyncio.Event().wait()

        batch_task = asyncio.create_task(batch_worker())
        client.client.batch_tasks = {"face": batch_task}
        client.client.pool = SimpleNamespace(_pool=[grpc_client])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            source = input_dir / "source.jpg"
            self.assertTrue(cv2.imwrite(str(source), np.zeros((24, 24, 3), dtype=np.uint8)))
            output = root / "output"

            result = await runner.run_evaluation(
                [source], output, client, self._services(), production=False
            )

            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["inputCount"], 1)
            self.assertEqual(payload["summary"]["faces"]["side"], 1)
            self.assertEqual(payload["summary"]["persons"]["front"], 1)
            self.assertTrue((output / "source-orientation.jpg").is_file())
            self.assertTrue(grpc_client.closed)
            self.assertTrue(batch_task.done())
            self.assertTrue(batch_task.cancelled())

    async def test_second_image_failure_leaves_previous_artifacts_unchanged(self):
        class SecondImageFailingClient(FakeFaceClient):
            def __init__(self):
                super().__init__()
                self.detection_calls = 0

            async def extract_face_detections(self, image):
                self.detection_calls += 1
                if self.detection_calls == 2:
                    raise RuntimeError("second image failed")
                return await super().extract_face_detections(image)

        client = SecondImageFailingClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            output.mkdir()
            encoded = cv2.imencode(
                ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
            )[1].tobytes()
            sources = [input_dir / "first.jpg", input_dir / "second.jpg"]
            for source in sources:
                source.write_bytes(encoded)

            previous_result = {
                "inputCount": 1,
                "images": [
                    {
                        "image": "old.jpg",
                        "annotated": "old-orientation.jpg",
                        "faces": [],
                        "persons": [],
                    }
                ],
                "summary": {"faces": {}, "persons": {}},
            }
            (output / "results.json").write_text(
                json.dumps(previous_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "old-orientation.jpg").write_bytes(b"previous annotation")
            (output / "user-notes.txt").write_bytes(b"keep me")
            before_files = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            before_siblings = {path.name for path in root.iterdir()}

            with self.assertRaisesRegex(RuntimeError, "second image failed"):
                await runner.run_evaluation(
                    sources, output, client, self._services(), production=False
                )

            after_files = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            self.assertEqual(after_files, before_files)
            self.assertEqual({path.name for path in root.iterdir()}, before_siblings)
            self.assertTrue(grpc_client.closed)

    async def test_repeated_run_removes_only_previous_manifest_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            encoded = cv2.imencode(
                ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
            )[1].tobytes()
            first = input_dir / "first.jpg"
            second = input_dir / "second.jpg"
            first.write_bytes(encoded)
            second.write_bytes(encoded)

            first_client = FakeFaceClient()
            first_grpc = CloseableGrpcClient()
            first_client.client.pool = SimpleNamespace(_pool=[first_grpc])
            await runner.run_evaluation(
                [first, second], output, first_client, self._services(), production=False
            )
            (output / "user-notes.txt").write_bytes(b"keep me")

            second_client = FakeFaceClient()
            second_grpc = CloseableGrpcClient()
            second_client.client.pool = SimpleNamespace(_pool=[second_grpc])
            result = await runner.run_evaluation(
                [first], output, second_client, self._services(), production=False
            )

            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result, payload)
            self.assertEqual(
                [image["annotated"] for image in payload["images"]],
                ["first-orientation.jpg"],
            )
            self.assertTrue((output / "first-orientation.jpg").is_file())
            self.assertIsNotNone(
                cv2.imdecode(
                    np.frombuffer(
                        (output / "first-orientation.jpg").read_bytes(), dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )
            )
            self.assertFalse((output / "second-orientation.jpg").exists())
            self.assertEqual((output / "user-notes.txt").read_bytes(), b"keep me")
            self.assertTrue(first_grpc.closed)
            self.assertTrue(second_grpc.closed)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"results.json", "first-orientation.jpg", "user-notes.txt"},
            )

    async def test_publish_failure_rolls_back_previous_artifacts(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            output.mkdir()
            encoded = cv2.imencode(
                ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
            )[1].tobytes()
            sources = [input_dir / "first.jpg", input_dir / "second.jpg"]
            for source in sources:
                source.write_bytes(encoded)

            previous_result = {
                "inputCount": 1,
                "images": [
                    {
                        "image": "old.jpg",
                        "annotated": "old-orientation.jpg",
                        "faces": [],
                        "persons": [],
                    }
                ],
                "summary": {"faces": {}, "persons": {}},
            }
            (output / "results.json").write_text(
                json.dumps(previous_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "old-orientation.jpg").write_bytes(b"previous annotation")
            (output / "user-notes.txt").write_bytes(b"keep me")
            before_files = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            before_siblings = {path.name for path in root.iterdir()}
            real_replace = os.replace
            failed_once = False

            def fail_results_publish(source, destination):
                nonlocal failed_once
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not failed_once
                    and source_path.name == "results.json"
                    and destination_path == output / "results.json"
                    and source_path.parent != output
                ):
                    failed_once = True
                    raise OSError("injected results publish failure")
                return real_replace(source, destination)

            with patch.object(runner.os, "replace", side_effect=fail_results_publish):
                with self.assertRaisesRegex(OSError, "injected results publish failure"):
                    await runner.run_evaluation(
                        sources, output, client, self._services(), production=False
                    )

            after_files = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            self.assertTrue(failed_once)
            self.assertEqual(after_files, before_files)
            self.assertEqual({path.name for path in root.iterdir()}, before_siblings)
            self.assertTrue(grpc_client.closed)

    async def test_post_commit_cleanup_failure_keeps_new_final_artifacts(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            output.mkdir()
            source = input_dir / "first.jpg"
            source.write_bytes(
                cv2.imencode(
                    ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                )[1].tobytes()
            )
            previous_result = {
                "inputCount": 1,
                "images": [
                    {
                        "image": "old.jpg",
                        "annotated": "old-orientation.jpg",
                        "faces": [],
                        "persons": [],
                    }
                ],
                "summary": {"faces": {}, "persons": {}},
            }
            (output / "results.json").write_text(
                json.dumps(previous_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "old-orientation.jpg").write_bytes(b"old annotation")
            real_rmtree = runner.shutil.rmtree
            failed_once = False

            def partially_remove_backup_then_fail(path, *args, **kwargs):
                nonlocal failed_once
                candidate = Path(path)
                if not failed_once and ".output.backup-" in candidate.name:
                    failed_once = True
                    (candidate / "old-orientation.jpg").unlink()
                    raise OSError("injected post-commit cleanup failure")
                return real_rmtree(path, *args, **kwargs)

            with patch.object(
                runner.shutil,
                "rmtree",
                side_effect=partially_remove_backup_then_fail,
            ):
                try:
                    await runner.run_evaluation(
                        [source], output, client, self._services(), production=False
                    )
                except BaseException as error:
                    raised = error
                else:
                    self.fail("post-commit cleanup failure was not raised")

            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(type(raised).__name__, "PostCommitCleanupError")
            self.assertTrue(failed_once)
            self.assertEqual(payload["images"][0]["image"], "first.jpg")
            self.assertTrue((output / "first-orientation.jpg").is_file())
            self.assertFalse((output / "old-orientation.jpg").exists())
            self.assertTrue(grpc_client.closed)
            self.assertTrue(raised.residual_paths)
            for path in raised.residual_paths:
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.exists())

    async def test_rollback_restore_failure_preserves_recovery_artifacts(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            output.mkdir()
            source = input_dir / "first.jpg"
            source.write_bytes(
                cv2.imencode(
                    ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                )[1].tobytes()
            )
            previous_result = {
                "inputCount": 1,
                "images": [
                    {
                        "image": "old.jpg",
                        "annotated": "old-orientation.jpg",
                        "faces": [],
                        "persons": [],
                    }
                ],
                "summary": {"faces": {}, "persons": {}},
            }
            (output / "results.json").write_text(
                json.dumps(previous_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "old-orientation.jpg").write_bytes(b"old annotation")
            real_replace = os.replace
            publish_failed = False
            restore_failed = False

            def fail_publish_and_one_restore(source_path, destination_path):
                nonlocal publish_failed, restore_failed
                source_path = Path(source_path)
                destination_path = Path(destination_path)
                if (
                    not publish_failed
                    and source_path.name == "results.json"
                    and ".output.stage-" in source_path.parent.name
                    and destination_path == output / "results.json"
                ):
                    publish_failed = True
                    raise OSError("injected publish failure")
                if (
                    publish_failed
                    and not restore_failed
                    and ".output.backup-" in source_path.parent.name
                    and destination_path == output / "old-orientation.jpg"
                ):
                    restore_failed = True
                    raise OSError("injected restore failure")
                return real_replace(source_path, destination_path)

            with patch.object(
                runner.os, "replace", side_effect=fail_publish_and_one_restore
            ):
                try:
                    await runner.run_evaluation(
                        [source], output, client, self._services(), production=False
                    )
                except BaseException as error:
                    raised = error
                else:
                    self.fail("publish recovery failure was not raised")

            self.assertEqual(type(raised).__name__, "PublishRecoveryError")
            self.assertTrue(publish_failed)
            self.assertTrue(restore_failed)
            self.assertIsInstance(raised.__cause__, OSError)
            self.assertIn("injected publish failure", str(raised.__cause__))
            self.assertTrue(grpc_client.closed)
            self.assertTrue(raised.recovery_paths)
            for path in raised.recovery_paths:
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.exists())
            recovery_files = {
                path.name
                for directory_path in raised.recovery_paths
                for path in directory_path.rglob("*")
                if path.is_file()
            }
            self.assertIn("old-orientation.jpg", recovery_files)
            self.assertIn("first-orientation.jpg", recovery_files)

    async def test_evaluation_and_staging_cleanup_failures_are_both_preserved(self):
        class BrokenClient(FakeFaceClient):
            async def extract_face_detections(self, _image):
                raise RuntimeError("inference failed")

        client = BrokenClient()
        failing_grpc = FailingGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[failing_grpc])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(
                cv2.imencode(
                    ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                )[1].tobytes()
            )

            with patch.object(
                runner,
                "_remove_temporary_directory",
                side_effect=OSError("staging cleanup failed"),
            ):
                with self.assertRaises(runner.EvaluationCleanupError) as raised:
                    await runner.run_evaluation(
                        [source], root / "output", client, self._services(), production=False
                    )

            artifact_error = raised.exception.primary_error
            self.assertEqual(type(artifact_error).__name__, "ArtifactCleanupError")
            self.assertEqual(str(artifact_error.primary_error), "inference failed")
            self.assertEqual(
                [str(error) for error in artifact_error.cleanup_errors],
                ["staging cleanup failed"],
            )
            self.assertEqual(len(raised.exception.cleanup_errors), 1)
            self.assertTrue(failing_grpc.closed)

    async def test_invalid_previous_manifest_does_not_claim_user_annotation(self):
        cases = {
            "mismatched": ("owner.jpg", "user-orientation.jpg"),
            "traversal": ("../user.jpg", "user-orientation.jpg"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case_name, (image_name, annotated_name) in cases.items():
                with self.subTest(case_name=case_name):
                    case_root = root / case_name
                    input_dir = case_root / "input"
                    output = case_root / "output"
                    input_dir.mkdir(parents=True)
                    output.mkdir()
                    source = input_dir / "new.jpg"
                    source.write_bytes(
                        cv2.imencode(
                            ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                        )[1].tobytes()
                    )
                    manifest = {
                        "inputCount": 1,
                        "images": [
                            {
                                "image": image_name,
                                "annotated": annotated_name,
                                "faces": [],
                                "persons": [],
                            }
                        ],
                        "summary": {"faces": {}, "persons": {}},
                    }
                    (output / "results.json").write_text(
                        json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (output / annotated_name).write_bytes(b"user owned")

                    await runner.run_evaluation(
                        [source],
                        output,
                        FakeFaceClient(),
                        self._services(),
                        production=False,
                    )

                    user_annotation = output / annotated_name
                    self.assertTrue(user_annotation.is_file())
                    self.assertEqual(user_annotation.read_bytes(), b"user owned")

    async def test_casefold_duplicate_annotation_basenames_fail_preflight(self):
        filesystem_name_key = getattr(runner, "_filesystem_name_key", os.path.normcase)
        if filesystem_name_key("Source.jpg") != filesystem_name_key("source.jpg"):
            self.skipTest("filesystem names are case-sensitive on this platform")
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "Source.jpg"
            second = root / "second" / "source.jpg"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"must not be read")
            second.write_bytes(b"must not be read")
            with patch.object(
                runner,
                "_read_jpg",
                side_effect=AssertionError("case collision must fail before reading"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "unique annotated output basename"
                ):
                    await runner.run_evaluation(
                        [first, second],
                        root / "output",
                        client,
                        self._services(),
                        production=False,
                    )

            self.assertTrue(grpc_client.closed)
            self.assertFalse((root / "output").exists())

    async def test_case_only_repeated_run_keeps_new_manifest_annotation(self):
        filesystem_name_key = getattr(runner, "_filesystem_name_key", os.path.normcase)
        if filesystem_name_key("Foo.jpg") != filesystem_name_key("foo.jpg"):
            self.skipTest("filesystem names are case-sensitive on this platform")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "first-input"
            second_input = root / "second-input"
            output = root / "output"
            first_input.mkdir()
            second_input.mkdir()
            encoded = cv2.imencode(
                ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
            )[1].tobytes()
            first = first_input / "Foo.jpg"
            second = second_input / "foo.jpg"
            first.write_bytes(encoded)
            second.write_bytes(encoded)

            await runner.run_evaluation(
                [first], output, FakeFaceClient(), self._services(), production=False
            )
            await runner.run_evaluation(
                [second], output, FakeFaceClient(), self._services(), production=False
            )

            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            annotated_name = payload["images"][0]["annotated"]
            self.assertEqual(annotated_name, "foo-orientation.jpg")
            self.assertTrue((output / annotated_name).is_file())
            self.assertEqual(
                len(
                    {
                        filesystem_name_key(path.name)
                        for path in output.iterdir()
                        if path.name.endswith("-orientation.jpg")
                    }
                ),
                1,
            )

    async def test_case_only_precommit_failure_restores_original_names_and_bytes(self):
        if runner._filesystem_name_key("Foo.jpg") != runner._filesystem_name_key(
            "foo.jpg"
        ):
            self.skipTest("filesystem names are case-sensitive on this platform")
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "first-input"
            second_input = root / "second-input"
            output = root / "output"
            first_input.mkdir()
            second_input.mkdir()
            encoded = cv2.imencode(
                ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
            )[1].tobytes()
            first = first_input / "Foo.jpg"
            second = second_input / "foo.jpg"
            first.write_bytes(encoded)
            second.write_bytes(encoded)
            await runner.run_evaluation(
                [first], output, FakeFaceClient(), self._services(), production=False
            )
            (output / "user-notes.txt").write_bytes(b"keep exact bytes")
            before = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            real_replace = os.replace
            failed_once = False

            def fail_results_publish(source_path, destination_path):
                nonlocal failed_once
                source_path = Path(source_path)
                destination_path = Path(destination_path)
                if (
                    not failed_once
                    and source_path.name == "results.json"
                    and ".output.stage-" in source_path.parent.name
                    and destination_path == output / "results.json"
                ):
                    failed_once = True
                    raise OSError("injected case-only publish failure")
                return real_replace(source_path, destination_path)

            with patch.object(
                runner.os, "replace", side_effect=fail_results_publish
            ):
                with self.assertRaisesRegex(
                    OSError, "injected case-only publish failure"
                ):
                    await runner.run_evaluation(
                        [second], output, client, self._services(), production=False
                    )

            after = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            self.assertTrue(failed_once)
            self.assertEqual(after, before)
            self.assertTrue(grpc_client.closed)

    async def test_unmanaged_annotation_target_is_not_overwritten(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            output.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(
                cv2.imencode(
                    ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                )[1].tobytes()
            )
            (output / "source-orientation.jpg").write_bytes(b"user annotation")
            (output / "user-notes.txt").write_bytes(b"keep exact bytes")
            before = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            before_siblings = {path.name for path in root.iterdir()}

            with self.assertRaisesRegex(
                RuntimeError, "not managed by the previous results.json manifest"
            ):
                await runner.run_evaluation(
                    [source], output, client, self._services(), production=False
                )

            after = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertEqual({path.name for path in root.iterdir()}, before_siblings)
            self.assertTrue(grpc_client.closed)

    async def test_staging_and_backup_use_resolved_destination_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            lexical_bridge = root / "bridge"
            input_dir.mkdir()
            lexical_bridge.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(
                cv2.imencode(
                    ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                )[1].tobytes()
            )
            lexical_output = lexical_bridge / os.pardir / "output"
            resolved_output = lexical_output.resolve(strict=False)
            real_mkdtemp = tempfile.mkdtemp
            temporary_parents = []

            def record_mkdtemp(*args, **kwargs):
                temporary_parents.append(Path(kwargs["dir"]))
                return real_mkdtemp(*args, **kwargs)

            with patch.object(
                runner.tempfile, "mkdtemp", side_effect=record_mkdtemp
            ):
                await runner.run_evaluation(
                    [source],
                    lexical_output,
                    FakeFaceClient(),
                    self._services(),
                    production=False,
                )

            self.assertEqual(len(temporary_parents), 2)
            self.assertEqual(
                temporary_parents,
                [resolved_output.parent, resolved_output.parent],
            )
            self.assertTrue((resolved_output / "results.json").is_file())

    async def test_backup_creation_failure_leaves_no_new_output_or_staging(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(
                cv2.imencode(
                    ".jpg", np.zeros((24, 24, 3), dtype=np.uint8)
                )[1].tobytes()
            )
            real_mkdtemp = tempfile.mkdtemp
            calls = 0

            def fail_second_mkdtemp(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("backup creation failed")
                return real_mkdtemp(*args, **kwargs)

            with patch.object(
                runner.tempfile, "mkdtemp", side_effect=fail_second_mkdtemp
            ):
                with self.assertRaisesRegex(OSError, "backup creation failed"):
                    await runner.run_evaluation(
                        [source],
                        root / "output",
                        client,
                        self._services(),
                        production=False,
                    )

            self.assertEqual(calls, 2)
            self.assertTrue(grpc_client.closed)
            self.assertEqual({path.name for path in root.iterdir()}, {"input"})

    async def test_duplicate_annotation_basenames_fail_preflight_and_close_client(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "source.jpg"
            second = root / "second" / "source.jpg"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"must not be read")
            second.write_bytes(b"must not be read")

            with patch.object(
                runner,
                "_read_jpg",
                side_effect=AssertionError("duplicate targets must fail before image reading"),
            ):
                with self.assertRaisesRegex(RuntimeError, "unique annotated output basename"):
                    await runner.run_evaluation(
                        [first, second],
                        root / "output",
                        client,
                        self._services(),
                        production=False,
                    )

            self.assertTrue(grpc_client.closed)
            self.assertFalse((root / "output").exists())

    async def test_non_production_evaluation_rejects_an_empty_image_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected at least 1 JPG image for production evaluation, found 0",
            ):
                await runner.run_evaluation(
                    [],
                    Path(directory) / "output",
                    FakeFaceClient(),
                    self._services(),
                    production=False,
                )

    async def test_unicode_paths_use_byte_based_jpeg_io_and_draw_expected_colors(self):
        client = FakeFaceClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "输入目录"
            root.mkdir()
            source = root / "样本.jpg"
            source.write_bytes(cv2.imencode(".jpg", np.zeros((24, 24, 3), dtype=np.uint8))[1].tobytes())
            output = Path(directory) / "输出目录"

            with (
                patch.object(runner.cv2, "imread", side_effect=AssertionError("must not use imread")),
                patch.object(runner.cv2, "imwrite", side_effect=AssertionError("must not use imwrite")),
            ):
                await runner.run_evaluation(
                    [source], output, client, self._services(), production=False
                )

            decoded = cv2.imdecode(
                np.frombuffer((output / "样本-orientation.jpg").read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(decoded)
            face_pixel = decoded[14, 12]
            person_pixel = decoded[16, 1]
            self.assertGreater(int(face_pixel[0]), int(face_pixel[1]))  # face blue box
            self.assertGreater(int(face_pixel[0]), int(face_pixel[2]))
            self.assertGreater(int(person_pixel[1]), int(person_pixel[0]))  # person green box
            self.assertGreater(int(person_pixel[1]), int(person_pixel[2]))

    async def test_cleanup_attempts_every_closable_when_one_close_fails(self):
        client = FakeFaceClient()
        first, second = FailingGrpcClient(), CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[first, second])

        with self.assertRaises(runner.CleanupError) as raised:
            await runner.close_client_resources(client)

        self.assertEqual(len(raised.exception.errors), 1)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)

    async def test_mid_run_failure_preserves_inference_and_cleanup_failures(self):
        class BrokenClient(FakeFaceClient):
            async def extract_face_detections(self, _image):
                raise RuntimeError("inference failed")

        client = BrokenClient()
        failed, closed = FailingGrpcClient(), CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[failed, closed])
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes())
            with self.assertRaises(runner.EvaluationCleanupError) as raised:
                await runner.run_evaluation(
                    [source], Path(directory) / "output", client, self._services(), production=False
                )

        self.assertEqual(str(raised.exception.primary_error), "inference failed")
        self.assertEqual(len(raised.exception.cleanup_errors), 1)
        self.assertTrue(closed.closed)

    async def test_preflight_failure_still_closes_supplied_client(self):
        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                await runner.run_evaluation(
                    [],
                    Path(directory) / "output",
                    client,
                    self._services(),
                )

        self.assertTrue(grpc_client.closed)

    async def test_list_and_output_path_preflight_failures_close_supplied_client(self):
        class ExplodingPaths:
            def __iter__(self):
                raise RuntimeError("path list failed")

        class ExplodingOutputPath:
            def __fspath__(self):
                raise RuntimeError("output path failed")

        client = FakeFaceClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with self.assertRaisesRegex(RuntimeError, "path list failed"):
            await runner.run_evaluation(
                ExplodingPaths(), Path("output"), client, self._services(), production=False
            )
        self.assertTrue(grpc_client.closed)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            source.write_bytes(
                cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
            )
            client = FakeFaceClient()
            grpc_client = CloseableGrpcClient()
            client.client.pool = SimpleNamespace(_pool=[grpc_client])
            with self.assertRaisesRegex(RuntimeError, "output path failed"):
                await runner.run_evaluation(
                    [source], ExplodingOutputPath(), client, self._services(), production=False
                )
            self.assertTrue(grpc_client.closed)

    async def test_run_production_empty_input_does_not_construct_client(self):
        constructed = []

        def client_factory():
            constructed.append(True)
            return FakeFaceClient()

        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected at least 1 JPG image for production evaluation, found 0",
            ):
                await runner.run_production(
                    input_dir, Path(directory) / "output", client_factory=client_factory
                )

        self.assertEqual(constructed, [])

    async def test_initialization_failure_closes_partially_created_client(self):
        class InitializationFailingClient(FakeFaceClient):
            async def initialize(self):
                raise RuntimeError("initialization failed")

        client = InitializationFailingClient()
        grpc_client = CloseableGrpcClient()
        client.client.pool = SimpleNamespace(_pool=[grpc_client])
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            for index in range(12):
                Path(input_dir, f"source-{index}.jpg").write_bytes(
                    cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
                )
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                await runner.run_production(
                    input_dir,
                    Path(directory) / "output",
                    client_factory=lambda: client,
                )

        self.assertTrue(grpc_client.closed)

    async def test_run_production_rejects_same_input_and_output_before_construction(self):
        constructed = []
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            for index in range(12):
                Path(input_dir, f"source-{index}.jpg").write_bytes(
                    cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
                )
            with self.assertRaisesRegex(RuntimeError, "Input and output directories must differ"):
                await runner.run_production(
                    input_dir,
                    input_dir,
                    client_factory=lambda: constructed.append(True),
                )

        self.assertEqual(constructed, [])

    async def test_directory_overlap_is_rejected_before_constructing_production_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "source.jpg").write_bytes(b"image")
            cases = {
                "same": (
                    os.path.join(os.fspath(input_dir), os.pardir, input_dir.name),
                    input_dir,
                ),
                "output-is-input-child": (input_dir, input_dir / "output"),
                "input-is-output-child": (input_dir / "nested", input_dir),
            }
            for name, (source_dir, destination) in cases.items():
                with self.subTest(name=name):
                    source = Path(source_dir) / "source.jpg"
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(b"image")
                    constructed = []
                    with self.assertRaisesRegex(
                        RuntimeError, "directories must .*not overlap"
                    ):
                        await runner.run_production(
                            source_dir,
                            destination,
                            client_factory=lambda: constructed.append(True),
                        )
                    self.assertEqual(constructed, [])

    def test_directory_validation_allows_distinct_unicode_and_ordinary_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary_input = root / "input"
            ordinary_output = root / "output"
            nfc_directory = root / "caf\u00e9"
            nfd_directory = root / "cafe\u0301"
            for path in (ordinary_input, ordinary_output):
                path.mkdir()

            self.assertNotEqual(nfc_directory, nfd_directory)
            runner._validate_distinct_directories(ordinary_input, ordinary_output)
            runner._validate_distinct_directories(nfc_directory, nfd_directory)

    async def test_evaluation_rejects_leaf_symlink_resolving_under_output_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "source.jpg"
            target = output_dir / "target.jpg"
            target.write_bytes(b"image")
            try:
                source.symlink_to(target)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"leaf symlinks are unavailable: {error}")

            client = FakeFaceClient()
            grpc_client = CloseableGrpcClient()
            client.client.pool = SimpleNamespace(_pool=[grpc_client])
            with patch.object(
                runner,
                "_read_jpg",
                side_effect=AssertionError("image reading must not start before output guard"),
            ), patch.object(
                runner,
                "evaluate_faces",
                side_effect=AssertionError("evaluation must not start before output guard"),
            ), patch.object(
                runner,
                "_write_jpg",
                side_effect=AssertionError("image writing must not start before output guard"),
            ):
                with self.assertRaisesRegex(RuntimeError, "directories must .*not overlap"):
                    await runner.run_evaluation(
                        [source], output_dir, client, self._services(), production=False
                    )
            self.assertTrue(grpc_client.closed)

            constructed = []
            with self.assertRaisesRegex(RuntimeError, "directories must .*not overlap"):
                await runner.run_production(
                    input_dir,
                    output_dir,
                    client_factory=lambda: constructed.append(True),
                )
            self.assertEqual(constructed, [])

    async def test_evaluation_rejects_hardlinked_output_targets_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target_name in ("source-orientation.jpg", "results.json"):
                with self.subTest(target_name=target_name):
                    input_dir = root / target_name / "input"
                    output_dir = root / target_name / "output"
                    input_dir.mkdir(parents=True)
                    output_dir.mkdir(parents=True)
                    source = input_dir / "source.jpg"
                    source.write_bytes(b"image")
                    os.link(source, output_dir / target_name)

                    client = FakeFaceClient()
                    grpc_client = CloseableGrpcClient()
                    client.client.pool = SimpleNamespace(_pool=[grpc_client])
                    with patch.object(
                        runner,
                        "_read_jpg",
                        side_effect=AssertionError("image reading must not start before output guard"),
                    ), patch.object(
                        runner,
                        "evaluate_faces",
                        side_effect=AssertionError("evaluation must not start before output guard"),
                    ), patch.object(
                        runner,
                        "_write_jpg",
                        side_effect=AssertionError("image writing must not start before output guard"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "same physical file"):
                            await runner.run_evaluation(
                                [source], output_dir, client, self._services(), production=False
                            )
                    self.assertTrue(grpc_client.closed)

    async def test_production_rejects_hardlinked_output_targets_before_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target_name in ("source-orientation.jpg", "results.json"):
                with self.subTest(target_name=target_name):
                    input_dir = root / target_name / "input"
                    output_dir = root / target_name / "output"
                    input_dir.mkdir(parents=True)
                    output_dir.mkdir(parents=True)
                    source = input_dir / "source.jpg"
                    source.write_bytes(b"image")
                    os.link(source, output_dir / target_name)
                    constructed = []

                    with self.assertRaisesRegex(RuntimeError, "same physical file"):
                        await runner.run_production(
                            input_dir,
                            output_dir,
                            client_factory=lambda: constructed.append(True),
                        )
                    self.assertEqual(constructed, [])

    async def test_evaluation_rejects_every_overlapping_source_parent_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "same": lambda output_dir, case_root: [output_dir / "same.jpg"],
                "source-parent-contains-output": lambda output_dir, case_root: [
                    case_root / "parent.jpg",
                    case_root / "safe" / "safe.jpg",
                ],
                "source-parent-is-inside-output": lambda output_dir, case_root: [
                    output_dir / "nested" / "child.jpg"
                ],
            }
            for name, make_paths in cases.items():
                with self.subTest(name=name):
                    case_root = root / name
                    output_dir = case_root / "output"
                    paths = make_paths(output_dir, case_root)
                    for path in paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"image")
                    client = FakeFaceClient()
                    grpc_client = CloseableGrpcClient()
                    client.client.pool = SimpleNamespace(_pool=[grpc_client])
                    with patch.object(
                        runner,
                        "_read_jpg",
                        side_effect=AssertionError("image reading must not start before output guard"),
                    ), patch.object(
                        runner,
                        "evaluate_faces",
                        side_effect=AssertionError("evaluation must not start before output guard"),
                    ), patch.object(
                        runner,
                        "_write_jpg",
                        side_effect=AssertionError("image writing must not start before output guard"),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "directories must .*not overlap"
                        ):
                            await runner.run_evaluation(
                                paths,
                                output_dir,
                                client,
                                self._services(),
                                production=False,
                            )
                    self.assertTrue(grpc_client.closed)

    def test_default_services_wires_real_production_person_components(self):
        from core.person_pipeline.observation import PersonObservationService
        from core.person_pipeline.preparation import PersonPreparationService
        from utils.quality.face_orientation import estimate_face_orientation
        from utils.reid.person_face_recognition import match_face_id_from_person_crop_with_info

        client = FakeFaceClient()
        executor = InlineImageExecutor()
        services = runner.default_services(client, image_executor=executor)

        self.assertIsInstance(services.observation_factory(), PersonObservationService)
        self.assertIsInstance(services.preparation_factory(), PersonPreparationService)
        self.assertIs(services.observation_factory().image_executor, executor)
        self.assertIs(services.preparation_factory().image_executor, executor)
        self.assertIs(services.estimate_orientation, estimate_face_orientation)
        self.assertIs(services.matcher.func, match_face_id_from_person_crop_with_info)
        self.assertIs(services.matcher.keywords["image_executor"], executor)

    async def test_production_owns_image_executor_on_success_failure_and_cancellation(self):
        from core.config import settings
        from core.execution import AsyncWorkExecutor

        class Client(FakeFaceClient):
            async def initialize(self):
                pass

        for outcome in (None, RuntimeError("evaluation failed"), asyncio.CancelledError()):
            with self.subTest(outcome=type(outcome).__name__), tempfile.TemporaryDirectory() as directory:
                input_dir = Path(directory) / "input"
                input_dir.mkdir()
                Path(input_dir, "source.jpg").write_bytes(
                    cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
                )
                executors = []

                def make_executor(workers, **kwargs):
                    executor = AsyncWorkExecutor(workers, **kwargs)
                    executors.append(executor)
                    return executor

                async def evaluate(paths, destination, client, services, **kwargs):
                    executor = services.preparation_factory().image_executor
                    self.assertIs(executor, services.observation_factory().image_executor)
                    self.assertEqual(await executor.run(lambda: 42), 42)
                    await runner.close_client_resources(client)
                    if outcome is not None:
                        raise outcome
                    return {"inputCount": len(paths)}

                with patch("core.execution.AsyncWorkExecutor", side_effect=make_executor), patch.object(
                    runner, "run_evaluation", side_effect=evaluate,
                ):
                    if outcome is None:
                        result = await runner.run_production(
                            input_dir, Path(directory) / "output", client_factory=Client,
                        )
                        self.assertEqual(result, {"inputCount": 1})
                    else:
                        with self.assertRaises(type(outcome)) as raised:
                            await runner.run_production(
                                input_dir, Path(directory) / "output", client_factory=Client,
                            )
                        self.assertIs(raised.exception, outcome)
                self.assertEqual(len(executors), 1)
                self.assertEqual(executors[0].workers, settings.IMAGE_CPU_WORKERS)
                pool = executors[0]._pool
                self.assertTrue(pool._closed)
                self.assertTrue(pool._executor._shutdown)
                self.assertTrue(all(not thread.is_alive() for thread in pool._executor._threads))

    def test_cli_help_runs_when_executed_as_a_script(self):
        script = Path(__file__).parent / "run_face_orientation_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input INPUT", result.stdout)

    async def test_output_directory_cannot_be_input_directory(self):
        client = FakeFaceClient()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            source.write_bytes(cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes())
            with self.assertRaisesRegex(RuntimeError, "Input and output directories must differ"):
                await runner.run_evaluation(
                    [source], Path(directory), client, self._services(), production=False
                )

    async def test_output_directory_cannot_contain_any_mixed_source_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            external_source = input_dir / "external.jpg"
            output_source = output_dir / "already-in-output.jpg"
            encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
            external_source.write_bytes(encoded)
            output_source.write_bytes(encoded)

            with patch.object(
                runner,
                "evaluate_faces",
                side_effect=AssertionError("evaluation must not start before output guard"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Input and output directories must differ"):
                    await runner.run_evaluation(
                        [external_source, output_source],
                        output_dir,
                        FakeFaceClient(),
                        self._services(),
                        production=False,
                    )


if __name__ == "__main__":
    unittest.main()
