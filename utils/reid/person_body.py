"""Person body-completeness categories and ReID compatibility rules."""

from __future__ import annotations

from typing import Any

from core.algorithm_config import algorithm_config


BODY_FULL = "full_body"
BODY_THREE_QUARTER_UPPER = "three_quarter_upper"
BODY_THREE_QUARTER_LOWER = "three_quarter_lower"
BODY_HALF_UPPER = "half_body_upper"
BODY_HALF_LOWER = "half_body_lower"
BODY_PARTIAL = "partial"
BODY_UNKNOWN = "unknown"

PERSON_BODY_COMPLETENESS_VALUES = frozenset(
    {
        BODY_FULL,
        BODY_THREE_QUARTER_UPPER,
        BODY_THREE_QUARTER_LOWER,
        BODY_HALF_UPPER,
        BODY_HALF_LOWER,
        BODY_PARTIAL,
        BODY_UNKNOWN,
    }
)

PERSON_GALLERY_BODY_TYPES = frozenset(
    {BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER}
)

PERSON_REID_GALLERY_TYPES_BY_QUERY: dict[str, tuple[str, ...]] = {
    BODY_FULL: (BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER),
    BODY_THREE_QUARTER_UPPER: (
        BODY_FULL,
        BODY_THREE_QUARTER_UPPER,
        BODY_HALF_UPPER,
    ),
    BODY_THREE_QUARTER_LOWER: (BODY_FULL, BODY_THREE_QUARTER_UPPER),
    BODY_HALF_UPPER: (BODY_FULL, BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER),
}

_ENROLLABLE_CROSS_BODY_PAIRS = frozenset(
    {
        (BODY_FULL, BODY_THREE_QUARTER_UPPER),
        (BODY_THREE_QUARTER_UPPER, BODY_FULL),
    }
)
_OTHER_CROSS_BODY_PAIRS = frozenset(
    {
        (BODY_FULL, BODY_HALF_UPPER),
        (BODY_THREE_QUARTER_UPPER, BODY_HALF_UPPER),
        (BODY_THREE_QUARTER_LOWER, BODY_FULL),
        (BODY_THREE_QUARTER_LOWER, BODY_THREE_QUARTER_UPPER),
        (BODY_HALF_UPPER, BODY_FULL),
        (BODY_HALF_UPPER, BODY_THREE_QUARTER_UPPER),
    }
)


def person_body_value(value: Any) -> str:
    """Return a valid body-completeness value or raise a clear error."""
    normalized = str(value or "").strip()
    if normalized not in PERSON_BODY_COMPLETENESS_VALUES:
        raise ValueError(f"Unsupported person bodyCompleteness: {value!r}")
    return normalized


def gallery_body_value(value: Any) -> str:
    """Return a valid gallery body type, rejecting non-enrollable values."""
    normalized = person_body_value(value)
    if normalized not in PERSON_GALLERY_BODY_TYPES:
        raise ValueError(
            "Person gallery accepts only full_body, three_quarter_upper, "
            "or half_body_upper, "
            f"got {normalized!r}"
        )
    return normalized


def gallery_body_types_for_query(query_body_completeness: Any) -> tuple[str, ...]:
    query_type = person_body_value(query_body_completeness)
    return PERSON_REID_GALLERY_TYPES_BY_QUERY.get(query_type, ())


def person_sbs_threshold(
    query_body_completeness: Any,
    gallery_body_completeness: Any,
) -> float:
    query_type = person_body_value(query_body_completeness)
    gallery_type = gallery_body_value(gallery_body_completeness)
    if query_type == gallery_type:
        return algorithm_config.reid.body.same_body_threshold
    pair = (query_type, gallery_type)
    if pair in _ENROLLABLE_CROSS_BODY_PAIRS:
        return algorithm_config.reid.body.enrollable_cross_body_threshold
    if pair in _OTHER_CROSS_BODY_PAIRS:
        return algorithm_config.reid.body.other_cross_body_threshold
    raise ValueError(
        "Incompatible person ReID body pair: "
        f"query={query_type!r}, gallery={gallery_type!r}"
    )
