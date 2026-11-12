import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
import run_face_orientation_dataset as base_runner
import run_person_orientation_dataset as runner


class DummyLogger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeClient:
    def __init__(self):
        self.client = SimpleNamespace(batch_tasks={}, pool=SimpleNamespace(_pool=[]))

    async def extract_face_detections(self, _image):
        return []


class PersonRunnerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _observation(result=...):
        values = {
            "people_rect": {"x": 1, "y": 1, "w": 6, "h": 7},
            "segment": {"mask": np.ones((8, 8), dtype=np.uint8)},
            "person_keypoints": np.zeros((17, 3), dtype=np.float32),
        }
        if result is not ...:
            values["person_orientation_result"] = result
        return SimpleNamespace(**values)

    @classmethod
    def _services(cls, observation, *, preparation=None, matcher=None):
        class ObservationService:
            async def detect(self, _context, _image):
                return [observation]

        class PreparationService:
            async def prepare(self, _context, image, _store, current, _segments):
                if preparation is not None:
                    return await preparation(current, image)
                return None

        async def default_matcher(**_kwargs):
            return {
                "face_orientation": "side",
                "pitch": 1,
                "yaw": 2,
                "roll": 3,
                "info": "matched face",
            }

        return base_runner.RunnerServices(
            clip_rect=base_runner._fallback_clip_rect,
            expand_rect=base_runner._fallback_expand_rect,
            estimate_orientation=lambda _landmarks, _shape: None,
            normalize_orientation=lambda value: str(value).strip().lower()
            if value is not None
            else "unknown",
            observation_factory=ObservationService,
            preparation_factory=PreparationService,
            pose_keypoints_to_crop=lambda points, _crop: points,
            matcher=matcher or default_matcher,
            logger=DummyLogger(),
        )

    def test_parse_args_uses_person_dataset_defaults(self):
        args = runner.parse_args([])

        self.assertEqual(args.input, "datasets/test")
        self.assertEqual(args.output, "exports/person_orientation_eval")

    def test_profiles_are_frozen_and_generate_distinct_annotation_names(self):
        source = Path("nested/sample.jpg")

        self.assertEqual(
            base_runner.FACE_PROFILE.annotation_name(source),
            "sample-orientation.jpg",
        )
        self.assertEqual(
            base_runner.PERSON_PROFILE.annotation_name(source),
            "sample-person-orientation.jpg",
        )
        with self.assertRaises(FrozenInstanceError):
            base_runner.PERSON_PROFILE.annotation_tag = "changed"

    def test_profile_annotation_tag_rejects_unsafe_tokens(self):
        for annotation_tag in (
            "",
            ".",
            "..",
            "person/orientation",
            "person\\orientation",
            "person:orientation",
            "person_orientation",
            "Person-Orientation",
            "person orientation",
            "person*orientation",
            "person?orientation",
            "person|orientation",
            "person<orientation",
            "person>orientation",
            'person"orientation',
        ):
            with self.subTest(annotation_tag=annotation_tag):
                with self.assertRaisesRegex(ValueError, "annotation_tag"):
                    base_runner.EvaluationProfile(annotation_tag=annotation_tag)

    def test_package_import_exercises_parse_args_from_repo_root(self):
        command = (
            "from tests import run_person_orientation_dataset as runner; "
            "args = runner.parse_args([]); "
            "print(args.input); print(args.output)"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["datasets/test", "exports/person_orientation_eval"],
        )

    async def test_wrapper_forwards_person_profile(self):
        expected = {"inputCount": 19}
        with patch.object(
            runner,
            "base_run_production",
            new=AsyncMock(return_value=expected),
        ) as delegated:
            actual = await runner.run_production("input", "output", client_factory="factory")

        self.assertIs(actual, expected)
        delegated.assert_awaited_once_with(
            "input",
            "output",
            client_factory="factory",
            profile=base_runner.PERSON_PROFILE,
        )

    def test_cli_help_runs_without_initializing_triton(self):
        script = Path(__file__).parent / "run_person_orientation_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input INPUT", result.stdout)

    async def test_person_profile_uses_stored_results_for_nineteen_inputs(self):
        orientation = SimpleNamespace(
            label="front",
            signed_span_ratio=np.float32(float("nan")),
            shoulder_span_ratio=np.float64(float("inf")),
            hip_span_ratio=float("-inf"),
            evidence_count=2,
            reason="classified",
        )
        observation = self._observation(orientation)
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            encoded = cv2.imencode(
                ".jpg", np.zeros((8, 8, 3), dtype=np.uint8)
            )[1].tobytes()
            paths = []
            for index in range(19):
                path = input_dir / f"sample-{index:02d}.jpg"
                path.write_bytes(encoded)
                paths.append(path)

            real_json_dumps = json.dumps
            with patch(
                "utils.quality.person_orientation.estimate_person_orientation",
                side_effect=AssertionError("evaluator must use the stored result"),
            ) as estimator, patch.object(
                base_runner.json, "dumps", wraps=real_json_dumps
            ) as serializer:
                result = await base_runner.run_evaluation(
                    paths,
                    output_dir,
                    client,
                    self._services(observation),
                    production=False,
                    profile=base_runner.PERSON_PROFILE,
                )
            estimator.assert_not_called()
            serializer.assert_called_once()
            self.assertIs(serializer.call_args.kwargs.get("allow_nan"), False)
            manifest_text = (output_dir / "results.json").read_text(encoding="utf-8")
            payload = json.loads(
                manifest_text
            )
            self.assertNotIn("NaN", manifest_text)
            self.assertNotIn("Infinity", manifest_text)

            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.jpg")),
                [f"sample-{index:02d}-person-orientation.jpg" for index in range(19)],
            )

        self.assertEqual(result, payload)
        self.assertEqual(result["inputCount"], 19)
        self.assertEqual(result["evaluationProfile"], "person-orientation")
        self.assertEqual(result["processedCount"], 19)
        self.assertEqual(result["personCount"], 19)
        self.assertEqual(
            result["personOrientationSummary"],
            {"front": 19, "side": 0, "back": 0, "unknown": 0},
        )
        first = result["images"][0]
        self.assertEqual(first["annotated"], "sample-00-person-orientation.jpg")
        self.assertEqual(len(first["persons"]), 1)
        self.assertEqual(
            first["persons"][0],
            {
                "peopleRect": {"x": 1, "y": 1, "w": 6, "h": 7},
                "faceOrientation": "unknown",
                "pitch": None,
                "yaw": None,
                "roll": None,
                "info": "person preparation returned no result",
                "personOrientation": "front",
                "signedSpanRatio": None,
                "shoulderSpanRatio": None,
                "hipSpanRatio": None,
                "orientationEvidenceCount": 2,
                "personOrientationReason": "classified",
            },
        )

    async def test_person_profile_retains_face_match_failure_with_body_diagnostics(self):
        orientation = SimpleNamespace(
            label="back",
            signed_span_ratio=-0.6,
            shoulder_span_ratio=-0.5,
            hip_span_ratio=-0.7,
            evidence_count=2,
            reason="classified",
        )
        observation = self._observation(orientation)

        async def preparation(current, image):
            crop = SimpleNamespace(
                image=image.copy(), mask=np.ones(image.shape[:2], dtype=np.uint8)
            )
            return SimpleNamespace(
                observation=current,
                crop_result=crop,
                cropped=crop.image,
                mask_crop=crop.mask,
            )

        async def matcher(**_kwargs):
            raise RuntimeError("face unavailable")

        people = await base_runner.evaluate_persons(
            np.zeros((8, 8, 3), dtype=np.uint8),
            FakeClient(),
            self._services(observation, preparation=preparation, matcher=matcher),
            base_runner.KeyframeContext("dataset", 0, "sample.jpg"),
            base_runner._search_must_not_run,
            profile=base_runner.PERSON_PROFILE,
        )

        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["personOrientation"], "back")
        self.assertEqual(people[0]["faceOrientation"], "unknown")
        self.assertIsNone(people[0]["pitch"])
        self.assertIn("face unavailable", people[0]["info"])

    async def test_person_profile_missing_and_nonfinite_diagnostics_are_json_safe(self):
        malformed = SimpleNamespace(
            label="side",
            signed_span_ratio=float("nan"),
            shoulder_span_ratio=float("inf"),
            hip_span_ratio="not-a-number",
            evidence_count=-4,
            reason=None,
        )
        cases = (
            (
                "missing",
                self._observation(),
                {
                    "personOrientation": "unknown",
                    "signedSpanRatio": None,
                    "shoulderSpanRatio": None,
                    "hipSpanRatio": None,
                    "orientationEvidenceCount": 0,
                    "personOrientationReason": "",
                },
            ),
            (
                "nonfinite",
                self._observation(malformed),
                {
                    "personOrientation": "side",
                    "signedSpanRatio": None,
                    "shoulderSpanRatio": None,
                    "hipSpanRatio": None,
                    "orientationEvidenceCount": 0,
                    "personOrientationReason": "",
                },
            ),
        )
        for name, observation, expected in cases:
            with self.subTest(name=name):
                people = await base_runner.evaluate_persons(
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    FakeClient(),
                    self._services(observation),
                    base_runner.KeyframeContext("dataset", 0, "sample.jpg"),
                    base_runner._search_must_not_run,
                    profile=base_runner.PERSON_PROFILE,
                )
                for field, value in expected.items():
                    self.assertEqual(people[0][field], value)
                json.dumps(people, allow_nan=False)

    async def test_person_annotation_contains_body_score_and_face_label(self):
        orientation = SimpleNamespace(
            label="front",
            signed_span_ratio=0.625,
            shoulder_span_ratio=0.6,
            hip_span_ratio=0.65,
            evidence_count=2,
            reason="classified",
        )
        observation = self._observation(orientation)

        async def preparation(current, image):
            crop = SimpleNamespace(
                image=image.copy(), mask=np.ones(image.shape[:2], dtype=np.uint8)
            )
            return SimpleNamespace(
                observation=current,
                crop_result=crop,
                cropped=crop.image,
                mask_crop=crop.mask,
            )

        labels = []
        with patch.object(
            base_runner,
            "_draw_rect",
            side_effect=lambda _image, _rect, label, _color: labels.append(label),
        ):
            await base_runner.evaluate_persons(
                np.zeros((8, 8, 3), dtype=np.uint8),
                FakeClient(),
                self._services(observation, preparation=preparation),
                base_runner.KeyframeContext("dataset", 0, "sample.jpg"),
                base_runner._search_must_not_run,
                canvas=np.zeros((8, 8, 3), dtype=np.uint8),
                profile=base_runner.PERSON_PROFILE,
            )

        self.assertEqual(len(labels), 1)
        self.assertIn("body front", labels[0])
        self.assertIn("score=0.625", labels[0])
        self.assertIn("face side", labels[0])

    def test_summary_can_count_body_orientation(self):
        self.assertEqual(
            base_runner._summary(
                [
                    {"personOrientation": "front"},
                    {"personOrientation": "front"},
                    {"personOrientation": "unknown"},
                ],
                field="personOrientation",
            ),
            {"front": 2, "unknown": 1},
        )

    def test_previous_manifest_ownership_is_profile_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            manifest = {
                "evaluationProfile": "person-orientation",
                "inputCount": 1,
                "images": [
                    {
                        "image": "source.jpg",
                        "annotated": "source-person-orientation.jpg",
                    }
                ],
            }
            (destination / "results.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            self.assertEqual(
                base_runner._load_previous_managed_annotations(
                    destination, base_runner.PERSON_PROFILE
                ),
                {"source-person-orientation.jpg": "source-person-orientation.jpg"},
            )
            self.assertEqual(
                base_runner._load_previous_managed_annotations(
                    destination, base_runner.FACE_PROFILE
                ),
                {},
            )

    async def _assert_profile_switch_is_rejected(self, first_profile, second_profile):
        observation = self._observation(
            SimpleNamespace(
                label="front",
                signed_span_ratio=0.75,
                shoulder_span_ratio=0.8,
                hip_span_ratio=0.7,
                evidence_count=2,
                reason="classified",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(
                cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
            )
            await base_runner.run_evaluation(
                [source],
                output_dir,
                FakeClient(),
                self._services(observation),
                production=False,
                profile=first_profile,
            )
            original_names = sorted(path.name for path in output_dir.iterdir())
            original_manifest = (output_dir / "results.json").read_bytes()
            self.assertEqual(
                json.loads(original_manifest)["evaluationProfile"],
                first_profile.annotation_tag,
            )
            original_annotation = (
                output_dir / first_profile.annotation_name(source)
            ).read_bytes()

            with patch.object(
                base_runner,
                "_read_jpg",
                side_effect=AssertionError("profile mismatch must fail before inference"),
            ):
                with self.assertRaisesRegex(RuntimeError, "different evaluation profile"):
                    await base_runner.run_evaluation(
                        [source],
                        output_dir,
                        FakeClient(),
                        self._services(observation),
                        production=False,
                        profile=second_profile,
                    )

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()), original_names
            )
            self.assertEqual(
                (output_dir / "results.json").read_bytes(), original_manifest
            )
            self.assertEqual(
                (output_dir / first_profile.annotation_name(source)).read_bytes(),
                original_annotation,
            )

    async def test_face_manifest_rejects_person_profile_without_mutation(self):
        await self._assert_profile_switch_is_rejected(
            base_runner.FACE_PROFILE, base_runner.PERSON_PROFILE
        )

    async def test_person_manifest_rejects_face_profile_without_mutation(self):
        await self._assert_profile_switch_is_rejected(
            base_runner.PERSON_PROFILE, base_runner.FACE_PROFILE
        )

    def test_legacy_manifest_without_profile_is_owned_by_face_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            manifest = {
                "inputCount": 1,
                "images": [
                    {
                        "image": "source.jpg",
                        "annotated": "source-orientation.jpg",
                    }
                ],
            }
            (destination / "results.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            self.assertEqual(
                base_runner._load_previous_managed_annotations(
                    destination, base_runner.FACE_PROFILE
                ),
                {"source-orientation.jpg": "source-orientation.jpg"},
            )
            self.assertEqual(
                base_runner._load_previous_managed_annotations(
                    destination, base_runner.PERSON_PROFILE
                ),
                {},
            )

    async def test_person_profile_rejects_hardlinked_annotation_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "source.jpg"
            source.write_bytes(
                cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
            )
            try:
                Path(output_dir, "source-person-orientation.jpg").hardlink_to(source)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"hard links are unavailable: {error}")

            with self.assertRaisesRegex(RuntimeError, "same physical file"):
                await base_runner.run_evaluation(
                    [source],
                    output_dir,
                    FakeClient(),
                    self._services(self._observation()),
                    production=False,
                    profile=base_runner.PERSON_PROFILE,
                )


if __name__ == "__main__":
    unittest.main()
