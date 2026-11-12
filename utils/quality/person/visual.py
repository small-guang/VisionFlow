"""Visual precheck metrics and classification for person crops."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from core.algorithm_config import algorithm_config
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
)

from ..quality_reasons import QualityReason
from .mask import _cv2_module
from .models import PersonVisualMetrics, PersonVisualQualityDecision
from .policy import _deduplicate_reasons


PERSON_VISUAL_SIZE = (224, 224)  # 图像预检统一缩放尺寸，与 TBPS 输入尺寸一致

def calculate_person_visual_metrics(
    person_crop: np.ndarray,
    mask_crop: np.ndarray | None = None,
) -> PersonVisualMetrics:
    visual_config = algorithm_config.person.quality.visual
    if person_crop is None or person_crop.size == 0:
        raise ValueError("person crop is empty")

    cv2_module = _cv2_module()
    if cv2_module is None:
        raise RuntimeError("OpenCV is required for person visual quality checks")

    resized = cv2_module.resize(
        person_crop,
        PERSON_VISUAL_SIZE,
        interpolation=cv2_module.INTER_CUBIC,
    )
    if resized.ndim == 3:
        gray = cv2_module.cvtColor(resized, cv2_module.COLOR_BGR2GRAY)
    else:
        gray = resized
    gray = np.asarray(gray, dtype=np.uint8)

    foreground = _resize_foreground_mask(mask_crop, cv2_module)
    use_foreground = (
        foreground is not None
        and int(np.count_nonzero(foreground)) >= visual_config.min_foreground_pixels
    )
    exposure_selector = foreground if use_foreground else np.ones(gray.shape, dtype=bool)

    sharpness_selector = exposure_selector
    if use_foreground:
        eroded = cv2_module.erode(
            foreground.astype(np.uint8),
            np.ones(
                (
                    visual_config.mask_erosion_kernel,
                    visual_config.mask_erosion_kernel,
                ),
                dtype=np.uint8,
            ),
            iterations=visual_config.mask_erosion_iterations,
        ).astype(bool)
        if int(np.count_nonzero(eroded)) >= visual_config.min_foreground_pixels:
            sharpness_selector = eroded

    selected_gray = gray[exposure_selector]
    laplacian = cv2_module.Laplacian(gray, cv2_module.CV_64F)
    selected_laplacian = laplacian[sharpness_selector]
    p5, p10, p90, p95 = [
        float(value) for value in np.percentile(selected_gray, [5, 10, 90, 95])
    ]

    return PersonVisualMetrics(
        laplacian_variance=float(np.var(selected_laplacian)),
        black_clip_ratio=float(
            np.mean(selected_gray <= visual_config.black_clip_threshold)
        ),
        dark_ratio=float(np.mean(selected_gray <= visual_config.dark_threshold)),
        white_clip_ratio=float(
            np.mean(selected_gray >= visual_config.white_clip_threshold)
        ),
        bright_ratio=float(np.mean(selected_gray >= visual_config.bright_threshold)),
        p5=p5,
        p10=p10,
        p90=p90,
        p95=p95,
        contrast_span=float(p95 - p5),
        used_foreground=bool(use_foreground),
        sample_pixel_count=int(selected_gray.size),
    )

def classify_person_visual_metrics(
    metrics: PersonVisualMetrics,
) -> PersonVisualQualityDecision:
    visual_config = algorithm_config.person.quality.visual
    general_config = algorithm_config.person.quality.general
    reject_reasons: list[QualityReason] = []
    cap_reasons: list[QualityReason] = []

    def add_reason(target: list[QualityReason], reason: QualityReason) -> None:
        if reason not in target:
            target.append(reason)

    if metrics.laplacian_variance < visual_config.laplacian_reject_threshold:
        add_reason(reject_reasons, QualityReason.BLURRY_LAPLACIAN)
    elif metrics.laplacian_variance < visual_config.laplacian_cap_threshold:
        add_reason(cap_reasons, QualityReason.BLURRY_LAPLACIAN)

    if (
        metrics.black_clip_ratio >= visual_config.exposure_reject_ratio
        or metrics.p95 <= visual_config.underexposed_reject_p95
    ):
        add_reason(reject_reasons, QualityReason.UNDEREXPOSED)
    elif (
        metrics.dark_ratio >= visual_config.exposure_cap_ratio
        or metrics.p90 <= visual_config.underexposed_cap_p90
    ):
        add_reason(cap_reasons, QualityReason.UNDEREXPOSED)

    if (
        metrics.white_clip_ratio >= visual_config.exposure_reject_ratio
        or metrics.p5 >= visual_config.overexposed_reject_p5
    ):
        add_reason(reject_reasons, QualityReason.OVEREXPOSED)
    elif (
        metrics.bright_ratio >= visual_config.exposure_cap_ratio
        or metrics.p10 >= visual_config.overexposed_cap_p10
    ):
        add_reason(cap_reasons, QualityReason.OVEREXPOSED)

    if metrics.contrast_span < visual_config.contrast_reject_threshold:
        add_reason(reject_reasons, QualityReason.LOW_CONTRAST)
    elif metrics.contrast_span < visual_config.contrast_cap_threshold:
        add_reason(cap_reasons, QualityReason.LOW_CONTRAST)

    reasons = _deduplicate_reasons([*reject_reasons, *cap_reasons])
    if reject_reasons:
        return PersonVisualQualityDecision(
            reject=True,
            cap_level=QUALITY_LEVEL_DROP,
            crop_score=0.0,
            quality_reasons=reasons,
            metrics=metrics,
        )
    if cap_reasons:
        return PersonVisualQualityDecision(
            reject=False,
            cap_level=QUALITY_LEVEL_RECOGNITION,
            crop_score=(
                general_config.enroll_min_score - general_config.recognition_score_gap
            ),
            quality_reasons=reasons,
            metrics=metrics,
        )
    return PersonVisualQualityDecision(
        reject=False,
        cap_level=QUALITY_LEVEL_ENROLL,
        crop_score=general_config.score_max,
        quality_reasons=[],
        metrics=metrics,
    )

def evaluate_person_visual_quality(
    person_crop: np.ndarray,
    mask_crop: np.ndarray | None = None,
) -> PersonVisualQualityDecision:
    return classify_person_visual_metrics(
        calculate_person_visual_metrics(person_crop, mask_crop)
    )

def format_person_visual_metrics(metrics: PersonVisualMetrics) -> str:
    source = "mask前景" if metrics.used_foreground else "整张crop"
    return (
        f"采样区域={source}，Laplacian方差={metrics.laplacian_variance:.2f}，"
        f"黑场占比={metrics.black_clip_ratio:.3f}，暗像素占比={metrics.dark_ratio:.3f}，"
        f"白场占比={metrics.white_clip_ratio:.3f}，亮像素占比={metrics.bright_ratio:.3f}，"
        f"P5/P10/P90/P95={metrics.p5:.1f}/{metrics.p10:.1f}/"
        f"{metrics.p90:.1f}/{metrics.p95:.1f}，对比度={metrics.contrast_span:.1f}"
    )

def format_person_visual_thresholds(
    reasons: Iterable[QualityReason],
) -> str:
    visual_config = algorithm_config.person.quality.visual
    def ratio_text(value: float) -> str:
        return f"{value:.2f}" if value == round(value, 2) else str(value)

    def numeric_text(value: float) -> str:
        return str(value).removesuffix(".0")

    reason_set = set(reasons)
    thresholds: list[str] = []
    if QualityReason.BLURRY_LAPLACIAN in reason_set:
        thresholds.append(
            f"Laplacian<{numeric_text(visual_config.laplacian_reject_threshold)}丢弃、"
            f"<{numeric_text(visual_config.laplacian_cap_threshold)}降级"
        )
    if QualityReason.UNDEREXPOSED in reason_set:
        thresholds.append(
            f"欠曝丢弃条件为黑场占比>={ratio_text(visual_config.exposure_reject_ratio)}"
            f"或P95<={numeric_text(visual_config.underexposed_reject_p95)}，"
            f"降级条件为暗像素占比>={ratio_text(visual_config.exposure_cap_ratio)}"
            f"或P90<={numeric_text(visual_config.underexposed_cap_p90)}"
        )
    if QualityReason.OVEREXPOSED in reason_set:
        thresholds.append(
            f"过曝丢弃条件为白场占比>={ratio_text(visual_config.exposure_reject_ratio)}"
            f"或P5>={numeric_text(visual_config.overexposed_reject_p5)}，"
            f"降级条件为亮像素占比>={ratio_text(visual_config.exposure_cap_ratio)}"
            f"或P10>={numeric_text(visual_config.overexposed_cap_p10)}"
        )
    if QualityReason.LOW_CONTRAST in reason_set:
        thresholds.append(
            f"对比度<{numeric_text(visual_config.contrast_reject_threshold)}丢弃、"
            f"<{numeric_text(visual_config.contrast_cap_threshold)}降级"
        )
    return "预检阈值：" + "；".join(thresholds)

def _resize_foreground_mask(mask_crop: Any, cv2_module: Any) -> np.ndarray | None:
    if mask_crop is None:
        return None
    try:
        mask_array = np.asarray(mask_crop)
    except (TypeError, ValueError):
        return None
    if mask_array.ndim != 2 or mask_array.size == 0:
        return None
    resized = cv2_module.resize(
        mask_array.astype(np.uint8),
        PERSON_VISUAL_SIZE,
        interpolation=cv2_module.INTER_NEAREST,
    )
    return np.asarray(resized).astype(bool)
