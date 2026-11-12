from typing import Any

from .quality_reasons import (
    PoseQualityReason,
    QualityReason,
    quality_reason_value,
)


QUALITY_REASON_MESSAGES: dict[QualityReason, str] = {
    QualityReason.FACE_WIDTH_TOO_NARROW: "人脸宽度过窄",
    QualityReason.KEYPOINTS_OUTSIDE_BBOX: "关键点超出检测框",
    QualityReason.NOSE_OUTSIDE_EYES_HORIZONTAL: "鼻尖未位于双眼水平范围内",
    QualityReason.NOSE_OUTSIDE_EYES_MOUTH_VERTICAL: "鼻尖未位于眼睛和嘴角垂直范围内",
    QualityReason.BAD_POSE: "姿态角过大",
    QualityReason.GEOMETRIC_INSTABILITY: "关键点几何关系不稳定",
    QualityReason.BLURRY_LAPLACIAN: "图像模糊",
    QualityReason.UNDEREXPOSED: "图像欠曝",
    QualityReason.OVEREXPOSED: "图像过曝",
    QualityReason.LOW_CONTRAST: "图像对比度过低",
    QualityReason.LOW_FEATURE_QUALITY_NORM: "特征向量质量过低",
    QualityReason.MASK_AREA_TOO_SMALL: "mask面积过小，未达到识别门槛",
    QualityReason.MASK_AREA_BELOW_ENROLL: "基础mask分未达到建库门槛",
    QualityReason.INVALID_CONFIDENCE: "检测置信度异常",
    QualityReason.INVALID_IMAGE_SHAPE: "原图尺寸异常",
    QualityReason.INVALID_BBOX: "人型检测框无效",
    QualityReason.EMPTY_CROP: "人型裁剪图为空",
    QualityReason.BBOX_TOO_SMALL: "人型检测框过小",
    QualityReason.BBOX_AREA_TOO_SMALL: "人型检测框占整图比例过小",
    QualityReason.BBOX_AREA_TOO_LARGE: "人型检测框占整图比例过大",
    QualityReason.BBOX_ASPECT_RATIO_ABNORMAL: "人型检测框高宽比异常",
    QualityReason.BBOX_THREE_IMAGE_EDGES: "人型检测框严重贴近图像边缘",
    QualityReason.BBOX_NEAR_IMAGE_EDGE: "人型检测框靠近图像边缘",
    QualityReason.BBOX_HEAVILY_OVERLAPS_OTHER_PERSON: "人型检测框与其他人严重重叠",
    QualityReason.BBOX_OVERLAPS_OTHER_PERSON: "人型检测框与其他人重叠",
    QualityReason.MASK_MISSING: "mask缺失，退回mask像素数评分",
    QualityReason.MASK_SHAPE_INVALID: "mask尺寸异常，退回mask像素数评分",
    QualityReason.MASK_BBOX_RATIO_TOO_LOW: "mask占检测框比例过低",
    QualityReason.MASK_BBOX_RATIO_LOW: "mask占检测框比例偏低",
    QualityReason.MASK_FRAGMENTED: "mask连通域过于破碎",
    QualityReason.MASK_MULTIPLE_COMPONENTS: "mask存在多个大连通区域",
    QualityReason.MASK_TOUCHING_BBOX_EDGE_SEVERE: "mask严重贴近检测框边缘",
    QualityReason.MASK_TOUCHING_BBOX_EDGE: "mask贴近检测框边缘",
    QualityReason.CROP_PADDING_RATIO_HIGH: "人型裁剪补边占比偏高",
    QualityReason.CROP_PADDING_RATIO_TOO_HIGH: "人型裁剪补边占比过高",
    QualityReason.BBOX_CONTENT_OVERFLOW_HIGH: "原始人型框越界比例偏高",
    QualityReason.BBOX_CONTENT_OVERFLOW_TOO_HIGH: "原始人型框越界比例过高",
    QualityReason.PERSON_TRUNCATED: "人体被原图边界截断",
    QualityReason.PERSON_TRUNCATED_SEVERE: "人体被原图多边严重截断",
    QualityReason.CROP_CONTAINS_OTHER_PERSON_MILD: "人型裁剪边缘轻度混入其他人物",
    QualityReason.CROP_OTHER_PERSON_BBOX_FALLBACK: "其他人物mask无效，使用bbox诊断",
    QualityReason.CROP_CONTAINS_OTHER_PERSON: "人型裁剪混入其他人物",
    QualityReason.CROP_HEAVILY_CONTAINS_OTHER_PERSON: "人型裁剪严重混入其他人物",
}

POSE_QUALITY_REASON_MESSAGES: dict[PoseQualityReason, str] = {
    PoseQualityReason.POSE_INFERENCE_FAILED: "人体姿态模型调用失败",
    PoseQualityReason.POSE_NOT_MATCHED: "未匹配到人体姿态",
    PoseQualityReason.POSE_KEYPOINTS_INVALID: "人体关键点格式无效",
    PoseQualityReason.POSE_VISIBLE_KEYPOINTS_TOO_FEW: "可见人体关键点过少",
    PoseQualityReason.POSE_TORSO_INCOMPLETE: "肩髋关键点严重缺失",
    PoseQualityReason.POSE_SHOULDER_HIP_CHAIN_INCOMPLETE: "肩髋链路不完整",
    PoseQualityReason.POSE_LOWER_BODY_INCOMPLETE: "下半身关键点不完整",
    PoseQualityReason.POSE_HALF_BODY: "检测为半身或局部人体",
    PoseQualityReason.POSE_HEAD_INCOMPLETE: "头部关键点缺失，禁止建库",
    PoseQualityReason.POSE_BODY_TYPE_NOT_SEARCHABLE: "身型结构不支持ReID查询",
    PoseQualityReason.POSE_KEYPOINTS_OUTSIDE_BBOX: "人体关键点与分割框不一致",
    PoseQualityReason.POSE_MASK_MISMATCH: "人体关键点与mask不一致",
    PoseQualityReason.POSE_GEOMETRY_UNSTABLE: "人体关键点纵向结构不稳定",
}


def describe_quality_reason(reason: Any) -> str:
    if reason is None or reason == "":
        return "未知原因"
    reason_value = quality_reason_value(reason)
    try:
        reason_code = QualityReason(reason_value)
    except ValueError:
        try:
            pose_reason_code = PoseQualityReason(reason_value)
        except ValueError:
            return reason_value
        return POSE_QUALITY_REASON_MESSAGES.get(pose_reason_code, reason_value)
    return QUALITY_REASON_MESSAGES.get(reason_code, reason_value)


def format_quality_reasons(reasons: Any) -> str:
    if reasons is None:
        return ""
    if isinstance(reasons, str):
        reason_items = [reasons] if reasons else []
    elif isinstance(reasons, (list, tuple, set)):
        reason_items = [item for item in reasons if quality_reason_value(item)]
    else:
        reason_items = [reasons] if quality_reason_value(reasons) else []

    labels: list[str] = []
    seen: set[str] = set()
    for reason in reason_items:
        label = describe_quality_reason(quality_reason_value(reason))
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return "、".join(labels)


def append_quality_reasons_info(info: Any, reasons: Any) -> str:
    base_info = "" if info is None else str(info)
    reason_text = format_quality_reasons(reasons)
    if not reason_text:
        return base_info
    separator = "，" if base_info else ""
    return f"{base_info}{separator}门控原因：{reason_text}"
