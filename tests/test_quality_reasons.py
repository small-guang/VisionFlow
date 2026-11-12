import unittest

from utils.quality.info_messages import QUALITY_REASON_MESSAGES, describe_quality_reason
from utils.quality.quality_reasons import QualityReason, quality_reason_value


EXPECTED_QUALITY_REASON_VALUES = (
    "face_width_too_narrow",
    "keypoints_outside_bbox",
    "nose_outside_eyes_horizontal",
    "nose_outside_eyes_mouth_vertical",
    "bad_pose",
    "geometric_instability",
    "blurry_laplacian",
    "underexposed",
    "overexposed",
    "low_contrast",
    "low_feature_quality_norm",
    "mask_area_too_small",
    "mask_area_below_enroll",
    "invalid_confidence",
    "invalid_image_shape",
    "invalid_bbox",
    "empty_crop",
    "bbox_too_small",
    "bbox_area_too_small",
    "bbox_area_too_large",
    "bbox_aspect_ratio_abnormal",
    "bbox_three_image_edges",
    "bbox_near_image_edge",
    "bbox_content_overflow_high",
    "bbox_content_overflow_too_high",
    "person_truncated",
    "person_truncated_severe",
    "bbox_heavily_overlaps_other_person",
    "bbox_overlaps_other_person",
    "mask_missing",
    "mask_shape_invalid",
    "mask_bbox_ratio_too_low",
    "mask_bbox_ratio_low",
    "mask_fragmented",
    "mask_multiple_components",
    "mask_touching_bbox_edge_severe",
    "mask_touching_bbox_edge",
    "crop_padding_ratio_high",
    "crop_padding_ratio_too_high",
    "crop_contains_other_person_mild",
    "crop_other_person_bbox_fallback",
    "crop_contains_other_person",
    "crop_heavily_contains_other_person",
)


class QualityReasonTests(unittest.TestCase):
    def test_enum_values_match_public_reason_codes(self):
        self.assertEqual(
            tuple(reason.value for reason in QualityReason),
            EXPECTED_QUALITY_REASON_VALUES,
        )

    def test_message_table_covers_every_enum_member(self):
        self.assertEqual(set(QUALITY_REASON_MESSAGES), set(QualityReason))
        self.assertTrue(all(QUALITY_REASON_MESSAGES.values()))

    def test_description_accepts_enum_string_and_unknown_reason(self):
        enum_description = describe_quality_reason(
            QualityReason.FACE_WIDTH_TOO_NARROW
        )

        self.assertEqual(
            describe_quality_reason("face_width_too_narrow"),
            enum_description,
        )
        self.assertEqual(describe_quality_reason("future_reason"), "future_reason")

    def test_reason_value_converts_enum_and_preserves_string(self):
        self.assertEqual(
            quality_reason_value(QualityReason.MASK_AREA_TOO_SMALL),
            "mask_area_too_small",
        )
        self.assertEqual(quality_reason_value("future_reason"), "future_reason")


if __name__ == "__main__":
    unittest.main()
