"""Public contract tests for the algorithm TOML configuration."""

import os
from copy import deepcopy
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from core import algorithm_config as algorithm_config_module
from core.algorithm_config import AlgorithmConfig, AlgorithmConfigError, load_algorithm_config


class AlgorithmConfigTests(unittest.TestCase):
    """Verify loading and validation through the public configuration API."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._temporary_path = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _copy_default_and_replace(
        self, old: str, new: str, occurrence: int | None = None
    ) -> Path:
        """Copy the checked-in profile after replacing one intended occurrence."""
        default_path = Path(__file__).resolve().parents[1] / "config" / "algorithm.toml"
        contents = default_path.read_text(encoding="utf-8")
        if occurrence is None:
            self.assertEqual(contents.count(old), 1)
            replacement_start = contents.index(old)
        else:
            self.assertGreaterEqual(contents.count(old), occurrence)
            start = 0
            for _ in range(occurrence):
                start = contents.find(old, start)
                self.assertNotEqual(start, -1)
                start += len(old)
            replacement_start = start - len(old)
        updated = (
            contents[:replacement_start]
            + new
            + contents[replacement_start + len(old) :]
        )
        self.assertEqual(updated[:replacement_start], contents[:replacement_start])
        self.assertEqual(
            updated[replacement_start : replacement_start + len(new)], new
        )
        self.assertEqual(
            updated[replacement_start + len(new) :],
            contents[replacement_start + len(old) :],
        )
        path = self._temporary_path / "algorithm.toml"
        path.write_text(updated, encoding="utf-8")
        return path

    def test_checked_in_default_is_complete_and_immutable(self):
        config = load_algorithm_config()
        self.assertEqual(config.face.matching.general_similarity_threshold, 0.38)
        self.assertEqual(config.person.assignment.pose_match_mask_weight, 0.70)
        self.assertEqual(config.person_face.binding.quality_min_score, 1.5)
        self.assertEqual(config.reid.prototype.max_per_reid, 24)
        with self.assertRaises(ValidationError):
            config.face.matching.general_similarity_threshold = 0.9

    def test_face_gallery_capacity_is_required_and_bounded(self):
        default_path = Path(__file__).resolve().parents[1] / "config" / "algorithm.toml"
        raw = tomllib.loads(default_path.read_text(encoding="utf-8"))
        self.assertEqual(load_algorithm_config().face.gallery.max_vectors_per_id, 20)

        for value in (1, 1000):
            with self.subTest(valid=value):
                changed = deepcopy(raw)
                changed["face"]["gallery"]["max_vectors_per_id"] = value
                self.assertEqual(
                    AlgorithmConfig.model_validate(changed).face.gallery.max_vectors_per_id,
                    value,
                )

        for value in (0, 1001, True, 20.0):
            with self.subTest(invalid=value):
                changed = deepcopy(raw)
                changed["face"]["gallery"]["max_vectors_per_id"] = value
                with self.assertRaises(ValidationError) as failure:
                    AlgorithmConfig.model_validate(changed)
                self.assertIn("face.gallery.max_vectors_per_id", str(failure.exception))

        missing = deepcopy(raw)
        del missing["face"]["gallery"]["max_vectors_per_id"]
        with self.assertRaises(ValidationError) as failure:
            AlgorithmConfig.model_validate(missing)
        self.assertIn("face.gallery.max_vectors_per_id", str(failure.exception))

    def test_legacy_face_capacity_process_variable_is_rejected(self):
        with patch.dict(os.environ, {"FACE_MAX_VECTORS_PER_ID": "7"}):
            with self.assertRaisesRegex(
                AlgorithmConfigError, "face.gallery.max_vectors_per_id"
            ):
                load_algorithm_config()

    def test_legacy_face_capacity_dotenv_variable_is_rejected(self):
        root, default, _ = self._project_profiles()
        (root / ".env").write_text("FACE_MAX_VECTORS_PER_ID=7\n", encoding="utf-8")
        with (
            patch.object(algorithm_config_module, "PROJECT_ROOT", root),
            patch.object(algorithm_config_module, "DEFAULT_ALGORITHM_CONFIG_PATH", default),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(
                AlgorithmConfigError, "face.gallery.max_vectors_per_id"
            ):
                load_algorithm_config()

    def test_relative_override_is_resolved_from_project_root(self):
        with patch.dict(os.environ, {"ALGORITHM_CONFIG_PATH": "config/algorithm.toml"}):
            config = load_algorithm_config()
        self.assertEqual(config.person.crop.border_bgr, (104, 116, 124))

    def _project_profiles(self) -> tuple[Path, Path, Path]:
        root = self._temporary_path
        default = root / "config" / "algorithm.toml"
        default.parent.mkdir()
        source = Path(__file__).resolve().parents[1] / "config" / "algorithm.toml"
        shutil.copyfile(source, default)
        custom = root / "custom.toml"
        custom.write_text(
            source.read_text(encoding="utf-8").replace(
                "general_similarity_threshold = 0.38",
                "general_similarity_threshold = 0.44",
                1,
            ),
            encoding="utf-8",
        )
        return root, default, custom

    def test_project_dotenv_selects_profile_for_direct_startup(self):
        root, default, _ = self._project_profiles()
        (root / ".env").write_text("ALGORITHM_CONFIG_PATH=custom.toml\n", encoding="utf-8")
        with (
            patch.object(algorithm_config_module, "PROJECT_ROOT", root),
            patch.object(algorithm_config_module, "DEFAULT_ALGORITHM_CONFIG_PATH", default),
            patch.dict(os.environ, {}, clear=True),
        ):
            config = load_algorithm_config()
        self.assertEqual(config.face.matching.general_similarity_threshold, 0.44)

    def test_process_environment_overrides_project_dotenv(self):
        root, default, _ = self._project_profiles()
        (root / ".env").write_text("ALGORITHM_CONFIG_PATH=custom.toml\n", encoding="utf-8")
        with (
            patch.object(algorithm_config_module, "PROJECT_ROOT", root),
            patch.object(algorithm_config_module, "DEFAULT_ALGORITHM_CONFIG_PATH", default),
            patch.dict(os.environ, {"ALGORITHM_CONFIG_PATH": "config/algorithm.toml"}, clear=True),
        ):
            config = load_algorithm_config()
        self.assertEqual(config.face.matching.general_similarity_threshold, 0.38)

    def test_explicit_path_overrides_process_environment(self):
        root, default, custom = self._project_profiles()
        with (
            patch.object(algorithm_config_module, "PROJECT_ROOT", root),
            patch.object(algorithm_config_module, "DEFAULT_ALGORITHM_CONFIG_PATH", default),
            patch.dict(os.environ, {"ALGORITHM_CONFIG_PATH": str(default)}, clear=True),
        ):
            config = load_algorithm_config(custom)
        self.assertEqual(config.face.matching.general_similarity_threshold, 0.44)

    def test_absolute_environment_override_uses_selected_file(self):
        root, default, custom = self._project_profiles()
        with (
            patch.object(algorithm_config_module, "PROJECT_ROOT", root),
            patch.object(algorithm_config_module, "DEFAULT_ALGORITHM_CONFIG_PATH", default),
            patch.dict(os.environ, {"ALGORITHM_CONFIG_PATH": str(custom)}, clear=True),
        ):
            config = load_algorithm_config()
        self.assertEqual(config.face.matching.general_similarity_threshold, 0.44)

    def test_missing_dotenv_selected_file_does_not_fall_back_to_default(self):
        root, default, _ = self._project_profiles()
        (root / ".env").write_text("ALGORITHM_CONFIG_PATH=missing.toml\n", encoding="utf-8")
        with (
            patch.object(algorithm_config_module, "PROJECT_ROOT", root),
            patch.object(algorithm_config_module, "DEFAULT_ALGORITHM_CONFIG_PATH", default),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(AlgorithmConfigError, "missing.toml"):
                load_algorithm_config()

    def test_missing_file_reports_selected_path(self):
        missing = Path("does-not-exist.toml")
        with self.assertRaisesRegex(AlgorithmConfigError, "does-not-exist.toml"):
            load_algorithm_config(missing)

    def test_unknown_key_is_rejected_with_full_location(self):
        path = self._copy_default_and_replace(
            "general_similarity_threshold = 0.38",
            "general_similarity_threshold = 0.38\nunknown_knob = 1",
        )
        with self.assertRaisesRegex(AlgorithmConfigError, "face.matching.unknown_knob"):
            load_algorithm_config(path)

    def test_malformed_toml_reports_selected_file(self):
        path = self._copy_default_and_replace(
            "general_similarity_threshold = 0.38",
            "general_similarity_threshold = [",
        )
        with self.assertRaises(AlgorithmConfigError) as failure:
            load_algorithm_config(path)
        self.assertIn(str(path), str(failure.exception))

    def test_unreadable_selected_file_reports_permission_error(self):
        selected = self._temporary_path / "unreadable.toml"
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            with self.assertRaises(AlgorithmConfigError) as failure:
                load_algorithm_config(selected)
        self.assertIn(str(selected), str(failure.exception))
        self.assertIn("denied", str(failure.exception))

    def test_missing_required_key_reports_full_location(self):
        path = self._copy_default_and_replace(
            "general_similarity_threshold = 0.38", ""
        )
        with self.assertRaisesRegex(
            AlgorithmConfigError, "face.matching.general_similarity_threshold"
        ):
            load_algorithm_config(path)

    def test_wrong_numeric_type_reports_full_location(self):
        path = self._copy_default_and_replace(
            "class_top_k = 3", 'class_top_k = "3"'
        )
        with self.assertRaisesRegex(
            AlgorithmConfigError, "reid.classification.class_top_k"
        ):
            load_algorithm_config(path)

    def test_nonfinite_floats_are_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                path = self._copy_default_and_replace(
                    "laplacian_threshold = 12.0",
                    f"laplacian_threshold = {value}",
                    occurrence=1,
                )
                with self.assertRaisesRegex(
                    AlgorithmConfigError, "face.quality.laplacian_threshold"
                ):
                    load_algorithm_config(path)

    def test_boolean_is_not_accepted_as_numeric(self):
        path = self._copy_default_and_replace(
            "class_top_k = 3", "class_top_k = true"
        )
        with self.assertRaisesRegex(AlgorithmConfigError, "reid.classification.class_top_k"):
            load_algorithm_config(path)

    def test_assignment_weights_must_sum_to_one(self):
        path = self._copy_default_and_replace(
            "pose_match_iou_weight = 0.30",
            "pose_match_iou_weight = 0.40",
        )
        with self.assertRaisesRegex(AlgorithmConfigError, "sum to 1.0"):
            load_algorithm_config(path)

    def test_quality_threshold_order_is_enforced(self):
        path = self._copy_default_and_replace(
            "recognition_min_score = 2.0",
            "recognition_min_score = 5.6",
            occurrence=1,
        )
        with self.assertRaisesRegex(AlgorithmConfigError, "recognition.*enroll"):
            load_algorithm_config(path)

    def test_every_cross_field_relationship_accepts_boundary_and_rejects_violation(self):
        default_path = Path(__file__).resolve().parents[1] / "config" / "algorithm.toml"
        raw = tomllib.loads(default_path.read_text(encoding="utf-8"))
        cases = (
            ("face.quality.recognition_min_score", 5.5, 5.51, "face.quality"),
            ("person.assignment.pose_match_iou_weight", 0.30, 0.31, "person.assignment"),
            ("person.quality.general.recognition_min_score", 7.0, 7.01, "person.quality.general"),
            ("person.quality.general.recognition_score_gap", 6.99, 7.0, "person.quality.general"),
            ("person.quality.general.crop_weight", 0.15, 0.16, "person.quality.general"),
            ("person.quality.crop.foreign_cap_ratio", 0.30, 0.31, "person.quality.crop"),
            ("person.quality.crop.foreign_core_cap_ratio", 0.20, 0.21, "person.quality.crop"),
            ("person.quality.edge.mask_edge_cap_count", 3, 4, "person.quality.edge"),
            ("person.quality.edge.mask_edge_cap_ratio", 0.40, 0.41, "person.quality.edge"),
            ("person.quality.edge.bbox_overflow_cap_ratio", 0.50, 0.51, "person.quality.edge"),
            ("person.quality.edge.other_person_overlap_cap_ratio", 0.40, 0.41, "person.quality.edge"),
            ("person.quality.edge.mask_bbox_reject_ratio", 0.25, 0.26, "person.quality.edge"),
            ("person.quality.edge.min_bbox_image_ratio", 0.849, 0.85, "person.quality.edge"),
            ("person.quality.edge.min_bbox_aspect_ratio", 4.99, 5.0, "person.quality.edge"),
            ("person.quality.pose.min_vertical_span_ratio", 0.65, 0.66, "person.quality.pose"),
            ("person.quality.visual.mask_erosion_kernel", 7, 6, "person.quality.visual"),
            ("person.quality.visual.laplacian_reject_threshold", 20.0, 20.01, "person.quality.visual"),
            ("person.quality.visual.black_clip_threshold", 20, 21, "person.quality.visual"),
            ("person.quality.visual.bright_threshold", 250, 251, "person.quality.visual"),
            ("person.quality.visual.exposure_cap_ratio", 0.80, 0.81, "person.quality.visual"),
            ("person.quality.visual.underexposed_reject_p95", 40.0, 40.01, "person.quality.visual"),
            ("person.quality.visual.overexposed_cap_p10", 245.0, 245.01, "person.quality.visual"),
            ("person.quality.visual.contrast_reject_threshold", 20.0, 20.01, "person.quality.visual"),
            ("person.quality.pose.visible_score_weight", 5.0, 5.01, "person.quality"),
            ("reid.classification.mean_score_weight", 0.40, 0.41, "reid.classification"),
            ("reid.prototype.max_per_domain", 24, 25, "reid.prototype"),
            ("reid.prototype.class_floor", 0.90, 0.91, "reid.prototype"),
            ("reid.prototype.replace_redundancy_floor", 0.95, 0.96, "reid.prototype"),
        )
        for field, boundary, violation, location in cases:
            with self.subTest(field=field):
                keys = field.split(".")
                for value, valid in ((boundary, True), (violation, False)):
                    changed = deepcopy(raw)
                    section = changed
                    for key in keys[:-1]:
                        section = section[key]
                    section[keys[-1]] = value
                    if valid:
                        AlgorithmConfig.model_validate(changed)
                    else:
                        with self.assertRaises(ValidationError) as failure:
                            AlgorithmConfig.model_validate(changed)
                        self.assertIn(location, str(failure.exception))
