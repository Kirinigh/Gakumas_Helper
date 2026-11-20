"""Validate boxed OCR evidence supplied to offline Live readers."""

import math
from typing import Any


def _valid_ocr_result(result: Any) -> bool:
    raw_box = getattr(result, "box", ())
    raw_score = getattr(result, "score", None)
    return bool(
        isinstance(raw_box, (list, tuple))
        and len(raw_box) == 4
        and all(type(value) is int for value in raw_box)
        and raw_box[2] > 0
        and raw_box[3] > 0
        and not isinstance(raw_score, bool)
        and isinstance(raw_score, (int, float))
        and math.isfinite(float(raw_score))
        and 0.0 <= float(raw_score) <= 1.0
    )
