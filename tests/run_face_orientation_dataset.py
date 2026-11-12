"""Run face-orientation checks against a local JPG dataset.

This module intentionally lives in ``tests``: it is a manually invoked evaluator,
not a test case.  Its runtime dependencies are injected so its behavior can be
unit-tested without a live Triton server.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.keyframe.detection_helpers import KeyframeContext


MIN_PRODUCTION_IMAGE_COUNT = 1
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
PERSON_ORIENTATION_LABELS = ("front", "side", "back", "unknown")


@dataclass(frozen=True)
class EvaluationProfile:
    annotation_tag: str = "orientation"
    include_person_orientation: bool = False
    retain_all_persons: bool = False

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.annotation_tag) is None:
            raise ValueError(
                "annotation_tag must contain lowercase ASCII letters, digits, and hyphens"
            )

    def annotation_name(self, path: str | Path) -> str:
        return f"{Path(path).stem}-{self.annotation_tag}.jpg"


FACE_PROFILE = EvaluationProfile()
PERSON_PROFILE = EvaluationProfile(
    annotation_tag="person-orientation",
    include_person_orientation=True,
    retain_all_persons=True,
)


@dataclass(frozen=True)
class RunnerServices:
    clip_rect: Callable[[Iterable[Any], tuple[int, ...]], dict[str, int] | None]
    expand_rect: Callable[[dict[str, int], tuple[int, ...]], tuple[int, int, int, int] | None]
    estimate_orientation: Callable[[Any, tuple[int, ...]], Any]
    normalize_orientation: Callable[[Any], str]
    observation_factory: Callable[[], Any]
    preparation_factory: Callable[[], Any]
    pose_keypoints_to_crop: Callable[[Any, Any], Any]
    matcher: Callable[..., Any]
    logger: Any


class CleanupError(RuntimeError):
    """All failures encountered while releasing evaluator resources."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        super().__init__(
            "Failed to close dataset evaluator resources: "
            + "; ".join(str(error) for error in errors)
        )


class EvaluationCleanupError(RuntimeError):
    """Preserves an evaluation failure together with cleanup failures."""

    def __init__(
        self,
        primary_error: BaseException,
        cleanup_errors: list[BaseException],
    ) -> None:
        self.primary_error = primary_error
        self.cleanup_errors = cleanup_errors
        super().__init__(
            f"Evaluation failed: {primary_error}; cleanup also failed: "
            + "; ".join(str(error) for error in cleanup_errors)
        )


class ArtifactCleanupError(RuntimeError):
    """Preserves a primary failure when its temporary artifacts cannot be removed."""

    def __init__(
        self,
        primary_error: BaseException,
        cleanup_errors: list[BaseException],
        residual_paths: Iterable[Path],
    ) -> None:
        self.primary_error = primary_error
        self.cleanup_errors = cleanup_errors
        self.residual_paths = tuple(residual_paths)
        super().__init__(
            f"Operation failed: {primary_error}; artifact cleanup also failed: "
            + "; ".join(str(error) for error in cleanup_errors)
            + _format_recovery_paths(self.residual_paths)
        )


class PublishRecoveryError(RuntimeError):
    """A pre-commit publish failure whose recovery artifacts must be retained."""

    def __init__(
        self,
        publish_error: BaseException,
        rollback_errors: list[BaseException],
        recovery_paths: Iterable[Path],
    ) -> None:
        self.publish_error = publish_error
        self.rollback_errors = rollback_errors
        self.recovery_paths = tuple(recovery_paths)
        super().__init__(
            f"Publish failed: {publish_error}; rollback also failed: "
            + "; ".join(str(error) for error in rollback_errors)
            + _format_recovery_paths(self.recovery_paths)
        )


class PostCommitCleanupError(RuntimeError):
    """The new manifest is committed, but one or more temporary paths remain."""

    def __init__(
        self,
        cleanup_errors: list[BaseException],
        residual_paths: Iterable[Path],
    ) -> None:
        self.cleanup_errors = cleanup_errors
        self.residual_paths = tuple(residual_paths)
        super().__init__(
            "Evaluation artifacts were committed, but cleanup failed: "
            + "; ".join(str(error) for error in cleanup_errors)
            + _format_recovery_paths(self.residual_paths)
        )


def _format_recovery_paths(paths: Iterable[Path]) -> str:
    values = [os.fspath(path) for path in paths]
    return f"; retained paths: {', '.join(values)}" if values else ""


class _LocalObjectStore:
    """Minimal OSS substitute: preparation only needs a successful object name."""

    async def upload_object(self, _image_bytes: bytes, object_name: str) -> str:
        return object_name


def _fallback_clip_rect(
    xyxy: Iterable[Any], image_shape: tuple[int, ...]
) -> dict[str, int] | None:
    values = list(xyxy)
    if len(values) < 4 or len(image_shape) < 2:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    x1 = int(max(0, min(width, round(float(values[0])))))
    y1 = int(max(0, min(height, round(float(values[1])))))
    x2 = int(max(0, min(width, round(float(values[2])))))
    y2 = int(max(0, min(height, round(float(values[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _fallback_expand_rect(
    rect: dict[str, int], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int] | None:
    x, y = int(rect["x"]), int(rect["y"])
    width, height = int(rect["w"]), int(rect["h"])
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if width <= 0 or height <= 0:
        return None
    side = max(width, height) * 2.0
    center_x, center_y = x + width / 2.0, y + height / 2.0
    x1 = int(max(0, center_x - side / 2.0))
    y1 = int(max(0, center_y - side / 2.0))
    x2 = int(min(image_width, center_x + side / 2.0))
    y2 = int(min(image_height, center_y + side / 2.0))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _orientation_fields(result: Any, normalize: Callable[[Any], str]) -> dict[str, Any]:
    raw_label = getattr(result, "label", None)
    if raw_label is None:
        raw_label = getattr(result, "orientation", None)
    return {
        "faceOrientation": normalize(raw_label),
        "pitch": _finite_float(getattr(result, "pitch", None)),
        "yaw": _finite_float(getattr(result, "yaw", None)),
        "roll": _finite_float(getattr(result, "roll", None)),
    }


def _person_orientation_fields(observation: Any) -> dict[str, Any]:
    result = getattr(observation, "person_orientation_result", None)
    if result is None:
        return {
            "personOrientation": "unknown",
            "signedSpanRatio": None,
            "shoulderSpanRatio": None,
            "hipSpanRatio": None,
            "orientationEvidenceCount": 0,
            "personOrientationReason": "",
        }

    label = getattr(result, "label", None)
    normalized_label = (
        label.strip().casefold()
        if isinstance(label, str)
        and label.strip().casefold() in PERSON_ORIENTATION_LABELS
        else "unknown"
    )
    try:
        evidence_count = int(getattr(result, "evidence_count", 0))
    except (TypeError, ValueError, OverflowError):
        evidence_count = 0
    evidence_count = max(0, evidence_count)
    reason = getattr(result, "reason", "")
    return {
        "personOrientation": normalized_label,
        "signedSpanRatio": _finite_float(getattr(result, "signed_span_ratio", None)),
        "shoulderSpanRatio": _finite_float(
            getattr(result, "shoulder_span_ratio", None)
        ),
        "hipSpanRatio": _finite_float(getattr(result, "hip_span_ratio", None)),
        "orientationEvidenceCount": evidence_count,
        "personOrientationReason": "" if reason is None else str(reason),
    }


def _landmarks_to_full_image(
    landmarks: Any,
    x_offset: int,
    y_offset: int,
) -> np.ndarray | None:
    """Translate crop-local landmarks without letting malformed model data escape."""
    try:
        points = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim == 1 and points.size % 2 == 0:
        points = points.reshape(-1, 2)
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    translated = points.copy()
    translated[:, 0] += x_offset
    translated[:, 1] += y_offset
    return translated


def _face_label(prefix: str, orientation: dict[str, Any]) -> str:
    angles = " ".join(
        f"{name[0]}={'?' if orientation[name] is None else orientation[name]:}"
        for name in ("pitch", "yaw", "roll")
    )
    return f"{prefix} {orientation['faceOrientation']} {angles}"


def _person_profile_label(record: dict[str, Any]) -> str:
    score = record.get("signedSpanRatio")
    rendered_score = "?" if score is None else str(score)
    return (
        f"body {record['personOrientation']} score={rendered_score} "
        f"face {record['faceOrientation']}"
    )


def _draw_rect(
    image: np.ndarray,
    rect: dict[str, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = (int(rect[key]) for key in ("x", "y", "w", "h"))
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        image,
        label,
        (x, max(12, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def _read_jpg(path: Path) -> np.ndarray:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read JPG image: {path}")
    return image


def _write_jpg(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError(f"Could not encode annotated JPG: {path}")
    path.write_bytes(encoded.tobytes())


async def evaluate_faces(
    image: np.ndarray,
    client: Any,
    services: RunnerServices,
    canvas: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Evaluate whole-image faces, with landmarks expressed in expanded-crop space."""
    detections = await client.extract_face_detections(image)
    if not isinstance(detections, list):
        raise RuntimeError("Face detector returned a non-list response")

    faces: list[dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        rect = services.clip_rect(detection.get("xyxy", []), image.shape)
        if rect is None:
            continue
        expanded = services.expand_rect(rect, image.shape)
        if expanded is None:
            continue
        x1, y1, x2, y2 = expanded
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        bbox_in_crop = np.asarray(
            [rect["x"] - x1, rect["y"] - y1, rect["x"] + rect["w"] - x1, rect["y"] + rect["h"] - y1],
            dtype=np.float32,
        )
        landmarks = await client.extract_face_keypoints(crop, bbox_in_crop)
        landmarks_full = _landmarks_to_full_image(landmarks, x1, y1)
        orientation = _orientation_fields(
            services.estimate_orientation(landmarks_full, image.shape),
            services.normalize_orientation,
        )
        record = {
            "bbox": rect,
            "confidence": _finite_float(detection.get("conf")) or 0.0,
            **orientation,
        }
        faces.append(record)
        if canvas is not None:
            _draw_rect(canvas, rect, _face_label("face", orientation), BLUE)
    return faces


async def evaluate_persons(
    image: np.ndarray,
    client: Any,
    services: RunnerServices,
    context: KeyframeContext,
    search_callback: Callable[..., Any],
    canvas: np.ndarray | None = None,
    *,
    profile: EvaluationProfile = FACE_PROFILE,
) -> list[dict[str, Any]]:
    """Run the production person observation/preparation path without ReID search."""
    observations = await services.observation_factory().detect(context, image)
    preparation = services.preparation_factory()
    all_segments = [observation.segment for observation in observations]
    object_store = _LocalObjectStore()
    people: list[dict[str, Any]] = []
    for observation in observations:
        prepared = await preparation.prepare(
            context, image, object_store, observation, all_segments
        )
        if prepared is None:
            if not profile.retain_all_persons:
                continue
            orientation = {
                "faceOrientation": "unknown",
                "pitch": None,
                "yaw": None,
                "roll": None,
            }
            info = "person preparation returned no result"
        else:
            try:
                pose_keypoints = services.pose_keypoints_to_crop(
                    observation.person_keypoints, prepared.crop_result
                )
                match = await services.matcher(
                    person_crop=prepared.cropped,
                    device_id=context.sn,
                    timestamp=context.timestamp,
                    triton_client=client,
                    search_callback=search_callback,
                    person_mask=prepared.mask_crop,
                    pose_keypoints=pose_keypoints,
                    enable_recognition=False,
                    logger=services.logger,
                )
                orientation = {
                    "faceOrientation": services.normalize_orientation(
                        match.get("face_orientation")
                    ),
                    "pitch": _finite_float(match.get("pitch")),
                    "yaw": _finite_float(match.get("yaw")),
                    "roll": _finite_float(match.get("roll")),
                }
                info = str(match.get("info", ""))
            except Exception as error:
                if not profile.retain_all_persons:
                    raise
                orientation = {
                    "faceOrientation": "unknown",
                    "pitch": None,
                    "yaw": None,
                    "roll": None,
                }
                info = f"face matching failed: {error}"
        rect = dict(observation.people_rect)
        record = {"peopleRect": rect, **orientation, "info": info}
        if profile.include_person_orientation:
            record.update(_person_orientation_fields(observation))
        people.append(record)
        if canvas is not None:
            label = (
                _person_profile_label(record)
                if profile.include_person_orientation
                else _face_label("person", orientation)
            )
            _draw_rect(canvas, rect, label, GREEN)
    return people


def _summary(
    records: Iterable[dict[str, Any]], field: str = "faceOrientation"
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get(field, "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def validate_image_count(
    image_paths: Iterable[Path], minimum_count: int = MIN_PRODUCTION_IMAGE_COUNT
) -> list[Path]:
    paths = list(image_paths)
    if len(paths) < minimum_count:
        raise RuntimeError(
            f"Expected at least {minimum_count} JPG image for production evaluation, found {len(paths)}"
        )
    return paths


def collect_input_images(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix == ".jpg"),
        key=lambda path: path.name,
    )


def _normalized_resolved_directory(directory: str | Path) -> Path:
    resolved = Path(directory).resolve(strict=False)
    return Path(os.path.normcase(str(resolved)))


def _normalized_lexical_directory(directory: str | Path) -> Path:
    return Path(os.path.normcase(os.path.abspath(os.fspath(directory))))


def _validate_distinct_directories(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    resolve_paths: bool = True,
    normalized_output: Path | None = None,
) -> None:
    normalize = _normalized_resolved_directory if resolve_paths else _normalized_lexical_directory
    source = normalize(input_dir)
    destination = normalized_output or normalize(output_dir)
    try:
        source.relative_to(destination)
        overlaps = True
    except ValueError:
        try:
            destination.relative_to(source)
            overlaps = True
        except ValueError:
            overlaps = False
    if overlaps:
        raise RuntimeError("Input and output directories must differ and must not overlap")


def _existing_file_identities(paths: Iterable[Path]) -> set[tuple[int, int] | tuple[str, str]]:
    identities: set[tuple[int, int] | tuple[str, str]] = set()
    for path in dict.fromkeys(paths):
        try:
            status = path.stat()
        except FileNotFoundError:
            continue
        inode = getattr(status, "st_ino", 0)
        if inode:
            identities.add((status.st_dev, inode))
        else:
            identities.add(("resolved-path", os.path.normcase(str(path.resolve(strict=False)))))
    return identities


def _validate_output_targets_do_not_alias_inputs(
    image_paths: Iterable[Path],
    destination: Path,
    profile: EvaluationProfile = FACE_PROFILE,
) -> None:
    paths = list(image_paths)
    output_targets = {destination / "results.json"}
    output_targets.update(destination / profile.annotation_name(path) for path in paths)
    if _existing_file_identities(paths) & _existing_file_identities(output_targets):
        raise RuntimeError("Input image and output target refer to the same physical file")


def _validate_evaluation_paths(
    image_paths: Iterable[Path],
    output_dir: str | Path,
    profile: EvaluationProfile = FACE_PROFILE,
) -> None:
    paths = list(image_paths)
    destination = Path(output_dir)
    lexical_destination = _normalized_lexical_directory(destination)
    resolved_destination = _normalized_resolved_directory(destination)
    for source_dir in {image_path.parent for image_path in paths}:
        _validate_distinct_directories(
            source_dir,
            destination,
            resolve_paths=False,
            normalized_output=lexical_destination,
        )
    for source_dir in {image_path.resolve(strict=False).parent for image_path in paths}:
        _validate_distinct_directories(
            source_dir,
            destination,
            normalized_output=resolved_destination,
        )
    _validate_output_targets_do_not_alias_inputs(paths, destination, profile)


def _is_safe_basename(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not value or value in {".", ".."} or "\x00" in value or ":" in value:
        return False
    candidate = Path(value)
    return (
        not candidate.is_absolute()
        and not candidate.drive
        and candidate.name == value
        and "/" not in value
        and "\\" not in value
    )


def _filesystem_name_key(name: str) -> str:
    return os.path.normcase(name)


def _is_safe_jpg_basename(value: Any) -> bool:
    return _is_safe_basename(value) and value.endswith(".jpg")


def _is_safe_managed_annotation_basename(
    value: Any, profile: EvaluationProfile = FACE_PROFILE
) -> bool:
    return _is_safe_basename(value) and value.endswith(f"-{profile.annotation_tag}.jpg")


def _load_previous_manifest_annotations(
    destination: Path,
) -> tuple[str, dict[str, str]] | None:
    manifest_path = destination / "results.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, IsADirectoryError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    annotation_tag = manifest.get("evaluationProfile", FACE_PROFILE.annotation_tag)
    if not isinstance(annotation_tag, str):
        return None
    try:
        owner_profile = EvaluationProfile(annotation_tag=annotation_tag)
    except ValueError:
        return None
    images = manifest.get("images")
    input_count = manifest.get("inputCount")
    if (
        not isinstance(images, list)
        or type(input_count) is not int
        or input_count != len(images)
    ):
        return None
    image_keys: set[str] = set()
    managed: dict[str, str] = {}
    for item in images:
        if not isinstance(item, dict):
            return None
        image_name = item.get("image")
        annotated_name = item.get("annotated")
        if not _is_safe_jpg_basename(image_name):
            return None
        if not _is_safe_managed_annotation_basename(annotated_name, owner_profile):
            return None
        if annotated_name != owner_profile.annotation_name(image_name):
            return None
        image_key = _filesystem_name_key(image_name)
        annotated_key = _filesystem_name_key(annotated_name)
        if image_key in image_keys or annotated_key in managed:
            return None
        image_keys.add(image_key)
        managed[annotated_key] = annotated_name
    return annotation_tag, managed


def _load_previous_managed_annotations(
    destination: Path, profile: EvaluationProfile = FACE_PROFILE
) -> dict[str, str]:
    previous = _load_previous_manifest_annotations(destination)
    if previous is None or previous[0] != profile.annotation_tag:
        return {}
    return previous[1]


def _validate_existing_manifest_profile(
    destination: Path, profile: EvaluationProfile
) -> dict[str, str]:
    previous = _load_previous_manifest_annotations(destination)
    if previous is None:
        return {}
    existing_tag, managed = previous
    if existing_tag != profile.annotation_tag:
        raise RuntimeError(
            "Output directory contains results from a different evaluation profile: "
            f"existing={existing_tag}, requested={profile.annotation_tag}"
        )
    return managed


def _path_exists_without_following_symlinks(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _remove_temporary_directory(path: Path) -> None:
    if _path_exists_without_following_symlinks(path):
        shutil.rmtree(path)


def _existing_absolute_paths(paths: Iterable[Path | None]) -> tuple[Path, ...]:
    return tuple(
        path.resolve(strict=False)
        for path in paths
        if path is not None and _path_exists_without_following_symlinks(path)
    )


def _cleanup_temporary_directories(paths: Iterable[Path | None]) -> list[BaseException]:
    errors: list[BaseException] = []
    for path in paths:
        if path is None:
            continue
        try:
            _remove_temporary_directory(path)
        except BaseException as error:
            errors.append(error)
    return errors


def _publish_staged_artifacts(
    staging: Path,
    destination: Path,
    annotated_names: Iterable[str],
    previous_managed: dict[str, str],
) -> None:
    new_managed = {
        _filesystem_name_key(name): name for name in annotated_names
    }
    orphans = {
        key: name for key, name in previous_managed.items() if key not in new_managed
    }
    affected_names = set(new_managed.values()) | set(orphans.values()) | {"results.json"}
    destination_existed = _path_exists_without_following_symlinks(destination)
    backup: Path | None = None
    backed_up: list[str] = []
    published: list[str] = []

    def backup_existing(name: str) -> None:
        if backup is None:
            raise AssertionError("backup directory has not been created")
        target = destination / name
        if _path_exists_without_following_symlinks(target):
            backed_up.append(name)
            try:
                os.replace(target, backup / name)
            except BaseException:
                backed_up.pop()
                raise

    try:
        destination.mkdir(parents=True, exist_ok=True)
        for key, name in new_managed.items():
            if (
                key not in previous_managed
                and _path_exists_without_following_symlinks(destination / name)
            ):
                raise RuntimeError(
                    f"Annotation target is not managed by the previous results.json manifest: "
                    f"{destination / name}"
                )
        for name in affected_names:
            target = destination / name
            if target.is_dir() and not target.is_symlink():
                raise RuntimeError(f"Output artifact target is a directory: {target}")
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.backup-",
                dir=os.fspath(destination.parent),
            )
        )
        for key, name in sorted(new_managed.items()):
            previous_name = previous_managed.get(key)
            if previous_name is not None:
                backup_existing(previous_name)
            os.replace(staging / name, destination / name)
            published.append(name)
        for name in sorted(orphans.values(), key=_filesystem_name_key):
            backup_existing(name)

        backup_existing("results.json")
        os.replace(staging / "results.json", destination / "results.json")
        published.append("results.json")
    except BaseException as publish_error:
        rollback_errors: list[BaseException] = []
        try:
            staging.mkdir(parents=True, exist_ok=True)
        except BaseException as error:
            rollback_errors.append(error)
        for name in reversed(published):
            target = destination / name
            try:
                if _path_exists_without_following_symlinks(target):
                    os.replace(target, staging / name)
            except BaseException as error:
                rollback_errors.append(error)
        for name in reversed(backed_up):
            saved = backup / name
            try:
                if _path_exists_without_following_symlinks(saved):
                    os.replace(saved, destination / name)
            except BaseException as error:
                rollback_errors.append(error)
        if not destination_existed and not rollback_errors:
            try:
                destination.rmdir()
            except BaseException as error:
                rollback_errors.append(error)
        if rollback_errors:
            recovery_paths = _existing_absolute_paths((staging, backup))
            raise PublishRecoveryError(
                publish_error, rollback_errors, recovery_paths
            ) from publish_error
        cleanup_errors = _cleanup_temporary_directories((staging, backup))
        if cleanup_errors:
            residual_paths = _existing_absolute_paths((staging, backup))
            raise ArtifactCleanupError(
                publish_error, cleanup_errors, residual_paths
            ) from publish_error
        raise

    cleanup_errors = _cleanup_temporary_directories((staging, backup))
    if cleanup_errors:
        residual_paths = _existing_absolute_paths((staging, backup))
        raise PostCommitCleanupError(
            cleanup_errors, residual_paths
        ) from cleanup_errors[0]


async def _search_must_not_run(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Identity search must not run during orientation evaluation")


async def run_evaluation(
    image_paths: Iterable[Path],
    output_dir: str | Path,
    client: Any,
    services: RunnerServices,
    *,
    production: bool = True,
    profile: EvaluationProfile = FACE_PROFILE,
) -> dict[str, Any]:
    """Evaluate a supplied collection and always release the supplied runtime."""
    evaluation_error: BaseException | None = None
    try:
        paths = list(image_paths)
        paths = validate_image_count(paths)
        lexical_destination = Path(output_dir)
        _validate_evaluation_paths(paths, lexical_destination, profile)
        destination = lexical_destination.resolve(strict=False)
        annotated_names = [profile.annotation_name(image_path) for image_path in paths]
        annotated_keys = [_filesystem_name_key(name) for name in annotated_names]
        if len(annotated_keys) != len(set(annotated_keys)):
            raise RuntimeError("Each input image must have a unique annotated output basename")
        previous_managed = _validate_existing_manifest_profile(destination, profile)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.stage-",
                dir=os.fspath(destination.parent),
            )
        )
        try:
            images: list[dict[str, Any]] = []
            all_faces: list[dict[str, Any]] = []
            all_people: list[dict[str, Any]] = []
            for image_path, annotated_name in zip(paths, annotated_names):
                source = _read_jpg(image_path)
                annotated = source.copy()
                context = KeyframeContext(
                    sn="dataset",
                    timestamp=0,
                    object_name=str(image_path),
                    keyframe_id=image_path.stem,
                )
                faces = await evaluate_faces(source, client, services, annotated)
                persons = await evaluate_persons(
                    source,
                    client,
                    services,
                    context,
                    _search_must_not_run,
                    annotated,
                    profile=profile,
                )
                _write_jpg(staging / annotated_name, annotated)
                images.append(
                    {
                        "image": image_path.name,
                        "annotated": annotated_name,
                        "faces": faces,
                        "persons": persons,
                    }
                )
                all_faces.extend(faces)
                all_people.extend(persons)
            result = {
                "evaluationProfile": profile.annotation_tag,
                "inputCount": len(paths),
                "images": images,
                "summary": {"faces": _summary(all_faces), "persons": _summary(all_people)},
            }
            if profile.include_person_orientation:
                person_orientation_summary = {
                    label: 0 for label in PERSON_ORIENTATION_LABELS
                }
                person_orientation_summary.update(
                    _summary(all_people, field="personOrientation")
                )
                result.update(
                    {
                        "processedCount": len(images),
                        "personCount": len(all_people),
                        "personOrientationSummary": person_orientation_summary,
                    }
                )
            (staging / "results.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        except BaseException as primary_error:
            cleanup_errors = _cleanup_temporary_directories((staging,))
            if cleanup_errors:
                residual_paths = _existing_absolute_paths((staging,))
                raise ArtifactCleanupError(
                    primary_error, cleanup_errors, residual_paths
                ) from primary_error
            raise
        _publish_staged_artifacts(
            staging, destination, annotated_names, previous_managed
        )
        return result
    except BaseException as error:
        evaluation_error = error
        raise
    finally:
        try:
            await close_client_resources(client)
        except CleanupError as cleanup_error:
            if evaluation_error is None:
                raise
            raise EvaluationCleanupError(
                evaluation_error, cleanup_error.errors
            ) from evaluation_error


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value


async def close_client_resources(client: Any) -> None:
    """Cancel batching workers and close every pooled gRPC client once."""
    errors: list[BaseException] = []
    nested = getattr(client, "client", None)
    batch_tasks = getattr(nested, "batch_tasks", getattr(client, "batch_tasks", {}))
    tasks = (
        list(batch_tasks.values())
        if isinstance(batch_tasks, dict)
        else list(batch_tasks or [])
    )
    for task in tasks:
        try:
            if hasattr(task, "done") and not task.done():
                task.cancel()
        except BaseException as error:
            errors.append(error)
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors.extend(
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        )

    closables: list[Any] = []
    pool = getattr(nested, "pool", getattr(client, "pool", None))
    closables.extend(getattr(pool, "_pool", []) or [])
    closables.extend(item for item in (nested, client) if item is not None)
    closed: set[int] = set()
    for resource in closables:
        if id(resource) in closed:
            continue
        closed.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                await _maybe_await(close())
            except BaseException as error:
                errors.append(error)
    if errors:
        raise CleanupError(errors)


def default_services(client: Any, *, image_executor: Any) -> RunnerServices:
    """Build production services lazily so ``--help`` has no Triton import."""
    from core.config import settings
    from core.person_keyframe_processor import _pose_keypoints_to_crop
    from core.person_pipeline.models import PersonPipelineRuntime
    from core.person_pipeline.observation import PersonObservationService
    from core.person_pipeline.preparation import PersonPreparationService
    from utils.keyframe.detection_helpers import clip_xyxy_to_rect, expand_square_rect
    from utils.keyframe.keyframe_flow_helpers import encode_image_to_bytes
    from utils.quality.face_orientation import (
        estimate_face_orientation,
        normalize_face_orientation,
    )
    from utils.quality.person.visual import evaluate_person_visual_quality
    from utils.reid.person_face_recognition import match_face_id_from_person_crop_with_info

    logger = logging.getLogger(__name__)
    runtime = PersonPipelineRuntime(
        encode_image=encode_image_to_bytes,
        evaluate_visual_quality=evaluate_person_visual_quality,
        match_face=match_face_id_from_person_crop_with_info,
        pose_keypoints_to_crop=_pose_keypoints_to_crop,
    )
    return RunnerServices(
        clip_rect=clip_xyxy_to_rect,
        expand_rect=expand_square_rect,
        estimate_orientation=estimate_face_orientation,
        normalize_orientation=normalize_face_orientation,
        observation_factory=lambda: PersonObservationService(
            triton_client=client, logger=logger, image_executor=image_executor,
        ),
        preparation_factory=lambda: PersonPreparationService(
            logger=logger,
            runtime=runtime,
            upright_device_sns=frozenset(settings.person_upright_device_sns),
            image_executor=image_executor,
        ),
        pose_keypoints_to_crop=_pose_keypoints_to_crop,
        matcher=partial(match_face_id_from_person_crop_with_info, image_executor=image_executor),
        logger=logger,
    )


async def run_production(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    client_factory: Callable[[], Any] | None = None,
    profile: EvaluationProfile = FACE_PROFILE,
) -> dict[str, Any]:
    """Create and initialize Triton only for a real evaluator invocation."""
    image_paths = validate_image_count(collect_input_images(input_dir))
    _validate_evaluation_paths(image_paths, output_dir, profile)
    client: Any = None
    image_executor: Any = None
    evaluation_owns_client = False
    primary_error: BaseException | None = None
    try:
        if client_factory is None:
            from core.infer import UnifiedTritonClient

            client_factory = UnifiedTritonClient
        client = client_factory()
        await client.initialize()
        from core.config import settings
        from core.execution import AsyncWorkExecutor
        from core.performance import runtime_performance

        image_executor = AsyncWorkExecutor(
            settings.IMAGE_CPU_WORKERS, name="image_cpu", metrics=runtime_performance,
        )
        services = default_services(client, image_executor=image_executor)
        evaluation_owns_client = True
        return await run_evaluation(
            image_paths,
            output_dir,
            client,
            services,
            production=False,
            profile=profile,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if client is not None and not evaluation_owns_client:
            try:
                await close_client_resources(client)
            except CleanupError as cleanup_error:
                cleanup_errors.extend(cleanup_error.errors)
        if image_executor is not None:
            try:
                await image_executor.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            if primary_error is None:
                raise CleanupError(cleanup_errors)
            raise EvaluationCleanupError(
                primary_error, cleanup_errors
            ) from primary_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/test", help="directory containing JPGs")
    parser.add_argument(
        "--output", default="exports/face_orientation_eval", help="output directory"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run_production(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
