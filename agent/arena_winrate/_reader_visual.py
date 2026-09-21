"""Pure screen geometry and OCR layout rules used by the Maa reader.

No controller, capture, clock or mutable reader state is owned here.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any
from collections.abc import Sequence

from .reader import ArenaReaderError

CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR = 0.75


_PItemPanelRow = tuple[
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, tuple[float, float, float, float]], ...],
]


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _box(item: Any) -> tuple[int, int, int, int]:
    value = _value(item, "box")
    if value is None or len(value) != 4:
        raise ArenaReaderError("recognition_box_missing", "recognition result has no four-value box")
    return tuple(int(round(number)) for number in value)


def _text(item: Any) -> str:
    return str(_value(item, "text", ""))


def _has_support_bonus_heading(text: str) -> bool:
    # The member-page footnote also contains this phrase; it is not a popup.
    return sum(row.strip() == "サポートボーナス" for row in text.splitlines()) == 1


def _support_bonus_from_overlay(
    text: str,
    *,
    close_count: int,
) -> float | None:
    """Resolve only a complete same-frame support-bonus overlay."""

    bonuses = re.findall(
        r"(?<![0-9])\+([0-9]+(?:\.[0-9]+)?)%",
        text,
    )
    if not _has_support_bonus_heading(text) or len(bonuses) != 1 or close_count != 1:
        return None
    value = float(bonuses[0]) / 100
    if not 0 <= value <= 1:
        raise ArenaReaderError(
            "support_bonus_invalid",
            f"support bonus is outside [0,1]: {value}",
        )
    return value


def _p_item_interaction_box(
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Keep Maa's click target inside the central, unobscured P-item artwork."""

    x, y, width, height = box
    if width < 16 or height < 16:
        raise ArenaReaderError(
            "p_item_interaction_box_invalid",
            f"P-item box is too small for a safe click: {box!r}",
        )
    inset_x = max(2, width // 4)
    inset_y = max(2, height // 4)
    return (
        x + inset_x,
        y + inset_y,
        width - inset_x * 2,
        height - inset_y * 2,
    )


def _card_rows_shifted(
    actual: Sequence[tuple[int, int, int, int]],
    expected: Sequence[tuple[int, int, int, int]],
    *,
    tolerance: int = 3,
) -> bool:
    return any(
        abs(actual_value - expected_value) > tolerance
        for actual_box, expected_box in zip(actual, expected, strict=True)
        for actual_value, expected_value in zip(actual_box, expected_box, strict=True)
    )


def _card_content_generation_deltas(
    actual: Sequence[tuple[str, Any]],
    expected: Sequence[tuple[str, Any]],
    *,
    maximum_mean_absolute_error: float = (
        CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
    ),
) -> tuple[dict[str, Any], ...]:
    """Measure why a rendered-card generation differs after geometry settles."""

    import numpy as np

    if len(actual) != len(expected) or not actual:
        return (
            {
                "slot": 0,
                "visual_group_changed": False,
                "shape_changed": True,
                "mean_absolute_error": None,
                "shifted": True,
            },
        )
    deltas: list[dict[str, Any]] = []
    for slot, ((actual_group, actual_query), (expected_group, expected_query)) in enumerate(
        zip(actual, expected, strict=True),
        start=1,
    ):
        actual_array = np.asarray(actual_query)
        expected_array = np.asarray(expected_query)
        visual_group_changed = actual_group != expected_group
        shape_changed = actual_array.shape != expected_array.shape
        motion = None
        zero_mean_motion = None
        edge_motion = None
        structural_correlation = None
        mean_color_shift = None
        if not shape_changed:
            actual_float = actual_array.astype(np.float32)
            expected_float = expected_array.astype(np.float32)
            difference = actual_float - expected_float
            motion = float(
                np.mean(np.abs(difference))
            )
            channel_shift = np.mean(difference, axis=(0, 1), keepdims=True)
            mean_color_shift = float(np.mean(np.abs(channel_shift)))
            zero_mean_motion = float(
                np.mean(np.abs(difference - channel_shift))
            )
            actual_gray = np.mean(actual_float, axis=2)
            expected_gray = np.mean(expected_float, axis=2)
            actual_edges = np.concatenate(
                (
                    np.diff(actual_gray, axis=0).ravel(),
                    np.diff(actual_gray, axis=1).ravel(),
                )
            )
            expected_edges = np.concatenate(
                (
                    np.diff(expected_gray, axis=0).ravel(),
                    np.diff(expected_gray, axis=1).ravel(),
                )
            )
            edge_motion = float(np.mean(np.abs(actual_edges - expected_edges)))
            actual_centered = actual_float.ravel() - float(np.mean(actual_float))
            expected_centered = expected_float.ravel() - float(
                np.mean(expected_float)
            )
            denominator = float(
                np.linalg.norm(actual_centered) * np.linalg.norm(expected_centered)
            )
            structural_correlation = (
                1.0
                if denominator == 0.0 and motion == 0.0
                else (
                    0.0
                    if denominator == 0.0
                    else float(
                        np.dot(actual_centered, expected_centered) / denominator
                    )
                )
            )
        deltas.append(
            {
                "slot": slot,
                "visual_group_changed": visual_group_changed,
                "shape_changed": shape_changed,
                "mean_absolute_error": motion,
                "mean_color_shift": mean_color_shift,
                "zero_mean_absolute_error": zero_mean_motion,
                "edge_mean_absolute_error": edge_motion,
                "structural_correlation": structural_correlation,
                "shifted": (
                    visual_group_changed
                    or shape_changed
                    or motion is None
                    or motion > maximum_mean_absolute_error
                ),
            }
        )
    return tuple(deltas)


def _aligned_card_content_query(
    image: Any,
    box: tuple[int, int, int, int],
    shift_x: int,
    shift_y: int,
    *,
    current_side: bool,
) -> Any:
    """Return the existing 16x16 content query on one common overlap.

    ``shift_x``/``shift_y`` describe where the current render moved relative
    to the frozen source.  Both sides discard the non-overlapping outer
    pixel before running the exact 96x96 -> 16x16 production resize.  This
    is deliberately narrower than changing the global MAE threshold.
    """

    import cv2
    import numpy as np

    if shift_x not in {-1, 0, 1} or shift_y not in {-1, 0, 1}:
        raise ArenaReaderError(
            "skill_card_source_alignment_invalid",
            f"source-card alignment is outside the fixed +/-1 domain: "
            f"({shift_x}, {shift_y})",
        )
    array = np.asarray(image)
    x, y, width, height = box
    overlap_width = width - abs(shift_x)
    overlap_height = height - abs(shift_y)
    if (
        array.ndim != 3
        or array.shape[2] not in {3, 4}
        or overlap_width < 8
        or overlap_height < 8
    ):
        raise ArenaReaderError(
            "skill_card_source_alignment_invalid",
            f"source-card alignment ROI is invalid: box={box!r}",
        )
    offset_x = max(0, shift_x if current_side else -shift_x)
    offset_y = max(0, shift_y if current_side else -shift_y)
    crop_x = x + offset_x
    crop_y = y + offset_y
    crop = array[
        crop_y : crop_y + overlap_height,
        crop_x : crop_x + overlap_width,
        :3,
    ]
    if crop.shape[:2] != (overlap_height, overlap_width):
        raise ArenaReaderError(
            "skill_card_source_alignment_invalid",
            f"source-card alignment ROI is outside the frame: box={box!r}; "
            f"shift=({shift_x}, {shift_y}); current_side={current_side}",
        )
    live = cv2.resize(
        np.ascontiguousarray(crop),
        (96, 96),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(
        live,
        (16, 16),
        interpolation=cv2.INTER_AREA,
    ).astype(np.uint8)


def _reference_business_ids(identity: dict[str, Any]) -> tuple[int, ...]:
    """Return every high-confidence clean-reference ID in one observation."""

    raw_candidates: Sequence[dict[str, Any]]
    if identity.get("status") == "MEASURED":
        raw_candidates = (identity,)
    elif identity.get("status") == "MEASURED_CANDIDATES":
        raw_candidates = tuple(
            candidate
            for candidate in identity.get("candidates", ())
            if isinstance(candidate, dict)
        )
    else:
        return ()
    candidate_ids = {
        int(candidate["reference_business_id"])
        for candidate in raw_candidates
        if candidate.get("identity_low_confidence") is not True
        and isinstance(candidate.get("reference_business_id"), int)
        and not isinstance(candidate.get("reference_business_id"), bool)
        and int(candidate["reference_business_id"]) > 0
    }
    return tuple(sorted(candidate_ids))


def _reference_visual_groups(identity: dict[str, Any]) -> tuple[str, ...]:
    """Return every high-confidence visual family in one observation."""

    raw_candidates: Sequence[dict[str, Any]]
    if identity.get("status") == "MEASURED":
        raw_candidates = (identity,)
    elif identity.get("status") == "MEASURED_CANDIDATES":
        raw_candidates = tuple(
            candidate
            for candidate in identity.get("candidates", ())
            if isinstance(candidate, dict)
        )
    else:
        return ()
    groups = {
        str(candidate["reference_visual_group"])
        for candidate in raw_candidates
        if candidate.get("identity_low_confidence") is not True
        and isinstance(candidate.get("reference_visual_group"), str)
        and str(candidate["reference_visual_group"]).strip()
    }
    return tuple(sorted(groups))


def _reference_identity_candidates(
    identity: dict[str, Any],
) -> tuple[tuple[int, str], ...]:
    """Return every high-confidence business-ID/visual-family pair."""

    if not isinstance(identity, dict):
        return ()
    raw_candidates: Sequence[dict[str, Any]]
    if identity.get("status") == "MEASURED":
        raw_candidates = (identity,)
    elif identity.get("status") == "MEASURED_CANDIDATES":
        raw_value = identity.get("candidates")
        if (
            not isinstance(raw_value, Sequence)
            or isinstance(raw_value, (str, bytes))
            or not raw_value
            or any(not isinstance(candidate, dict) for candidate in raw_value)
        ):
            return ()
        raw_candidates = tuple(raw_value)
    else:
        return ()
    candidates: set[tuple[int, str]] = set()
    for candidate in raw_candidates:
        business_id = candidate.get("reference_business_id")
        visual_group = candidate.get("reference_visual_group")
        if (
            candidate.get("status") != "MEASURED"
            or candidate.get("identity_low_confidence") is not False
            or not isinstance(business_id, int)
            or isinstance(business_id, bool)
            or business_id < 1
            or not isinstance(visual_group, str)
            or not visual_group.strip()
        ):
            return ()
        candidates.add((business_id, visual_group))
    if len(candidates) != len(raw_candidates):
        return ()
    return tuple(sorted(candidates))


def _unique_reference_business_id(identity: dict[str, Any]) -> int | None:
    """Return one high-confidence clean-reference ID, never a shortlist."""

    candidate_ids = _reference_business_ids(identity)
    if len(candidate_ids) != 1:
        return None
    return candidate_ids[0]


def _skill_card_title_row_has_neutral_ink(
    image: Any,
    box: tuple[int, int, int, int],
) -> bool:
    """Reject coloured effect links while retaining the neutral title ink."""

    import numpy as np

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ArenaReaderError(
            "skill_card_detail_title_image_invalid",
            "skill-card title evidence is not a colour image",
        )
    image_height, image_width = values.shape[:2]
    left, top, width, height = box
    right = left + width
    bottom = top + height
    if (
        width < 1
        or height < 1
        or left < 0
        or top < 0
        or right > image_width
        or bottom > image_height
    ):
        raise ArenaReaderError(
            "skill_card_detail_title_box_invalid",
            f"title OCR box is outside the capture: {box!r}",
        )
    crop = values[top:bottom, left:right, :3].astype(np.int16)
    channel_min = np.min(crop, axis=2)
    channel_max = np.max(crop, axis=2)
    ink = channel_min < 235
    ink_count = int(np.count_nonzero(ink))
    minimum_ink = max(8, int(round(width * height * 0.004)))
    if ink_count < minimum_ink:
        return False
    neutral = ink & ((channel_max - channel_min) <= 45)
    dark_neutral = neutral & (channel_max <= 205)
    return bool(
        int(np.count_nonzero(neutral)) / ink_count >= 0.72
        and int(np.count_nonzero(dark_neutral)) >= minimum_ink
    )


def _skill_card_detail_overlay_guard_boxes(
    image: Any,
) -> tuple[tuple[int, int, int, int], ...]:
    """Cover every observed vertical placement of the adaptive detail panel."""

    height, width = image.shape[:2]
    if width < 320 or height < 568:
        raise ArenaReaderError(
            "skill_card_detail_overlay_guard_invalid",
            f"capture {width}x{height} is too small for overlay evidence",
        )
    horizontal_bands = ((0.00, 0.62), (0.38, 1.00))
    bands = ((0.00, 0.18), (0.14, 0.32), (0.28, 0.48))
    return tuple(
        (
            int(round(width * left_ratio)),
            int(round(height * top_ratio)),
            int(round(width * (right_ratio - left_ratio))),
            int(round(height * (bottom_ratio - top_ratio))),
        )
        for left_ratio, right_ratio in horizontal_bands
        for top_ratio, bottom_ratio in bands
    )


def _skill_card_interaction_box(
    card_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the deterministic, unobscured card-body click anchor.

    Maa treats the supplied rectangle as an action target rather than a
    promise to use its centre.  A broad rectangle can therefore land on
    the cost, type, or ``+`` overlays.  Live evidence showed that the old
    upper-art point could leave the source grid unchanged while this
    central body point opened the same cards on the bounded retry.  Make
    that proven point the first contact without widening the hit region.
    """

    x, y, width, height = card_box
    if width < 8 or height < 8:
        raise ArenaReaderError(
            "skill_card_interaction_box_invalid",
            f"card box is too small for a stable interaction region: {card_box!r}",
        )
    return (
        x + int(round(width * 0.48)),
        y + int(round(height * 0.48)),
        1,
        1,
    )


def _skill_card_retry_interaction_box(
    card_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the same safe body point for one longer-contact retry."""

    x, y, width, height = card_box
    if width < 8 or height < 8:
        raise ArenaReaderError(
            "skill_card_interaction_box_invalid",
            f"card box is too small for a stable interaction region: {card_box!r}",
        )
    return (
        x + int(round(width * 0.48)),
        y + int(round(height * 0.48)),
        1,
        1,
    )


def _title_anchored_effect_roi(
    frame_width: int,
    frame_height: int,
    title_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Cover one effect panel from its title instead of a fixed screen column."""

    title_x, title_y, title_width, title_height = title_box
    # OCR boxes are intentionally tight and may cover only the first title
    # glyphs.  The effect rows can extend far to either side of that box.
    # Keep the retry bounded to the title's panel, but guarantee at least
    # two thirds of the normalized screen width so a shifted panel cannot
    # truncate its invariant action rows.
    minimum_width = int(round(frame_width * 0.66))
    left = max(0, title_x - int(round(frame_width * 0.12)))
    top = max(0, title_y - int(round(frame_height * 0.015)))
    right = min(
        frame_width,
        title_x + title_width + int(round(frame_width * 0.55)),
    )
    if right - left < minimum_width:
        left = max(0, min(left, frame_width - minimum_width))
        right = min(frame_width, max(right, left + minimum_width))
        if right - left < minimum_width:
            left = max(0, right - minimum_width)
    bottom = min(
        frame_height,
        max(
            title_y + title_height + int(round(frame_height * 0.25)),
            top + int(round(frame_height * 0.295)),
        ),
    )
    if right - left < 32 or bottom - top < 32:
        raise ArenaReaderError(
            "skill_card_detail_effect_roi_invalid",
            "title-anchored detail effect ROI is too small: "
            f"frame={frame_width}x{frame_height}, title={title_box!r}",
        )
    return left, top, right - left, bottom - top


def _background_parameter_column_left(
    frame_width: int,
    frame_height: int,
    title_box: tuple[int, int, int, int],
    items: Sequence[Any],
) -> int | None:
    """Locate a proven member-stat column exposed beside a left popover.

    The member detail's parameter labels live in one fixed right-hand
    column.  A left-clamped skill-card popover can leave the lower labels
    visible beside its effect text, so a deliberately wide title-bound ROI
    may otherwise join a parameter value to the last effect token.  Treat
    the column as background only when two distinct, exact parameter
    labels each have their own aligned numeric value below them.  One
    label or an unpaired number is intentionally insufficient because the
    same words and digits may legitimately occur inside card effects.
    """

    title_x, title_y, title_width, _ = title_box
    title_right = title_x + title_width
    _, roi_top, _, roi_height = _title_anchored_effect_roi(
        frame_width,
        frame_height,
        title_box,
    )
    roi_bottom = roi_top + roi_height
    column_min_x = int(round(frame_width * 0.55))
    column_max_x = int(round(frame_width * 0.72))
    labels: list[tuple[str, tuple[int, int, int, int]]] = []
    numbers: list[tuple[int, int, int, int]] = []
    for item in items:
        text = re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", _text(item)),
        )
        box = _box(item)
        x, _, width, height = box
        if width < 1 or height < 1:
            continue
        if (
            text in {"ボーカル", "ダンス", "ビジュアル", "体力"}
            and column_min_x <= x <= column_max_x
        ):
            labels.append((text, box))
        elif (
            re.fullmatch(r"[0-9]{1,6}", text) is not None
            and column_min_x <= x <= column_max_x
        ):
            numbers.append(box)

    pair_candidates: list[
        tuple[
            float,
            float,
            str,
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ]
    ] = []
    for label, label_box in labels:
        label_x, label_y, label_width, label_height = label_box
        label_centre_x = label_x + label_width / 2.0
        label_centre_y = label_y + label_height / 2.0
        for number_box in numbers:
            number_x, number_y, number_width, number_height = number_box
            number_centre_x = number_x + number_width / 2.0
            number_centre_y = number_y + number_height / 2.0
            if (
                number_centre_y
                >= label_centre_y + max(4.0, label_height * 0.35)
                and number_y
                <= label_y + label_height + frame_height * 0.045
                and abs(number_centre_x - label_centre_x)
                <= frame_width * 0.04
                and abs(number_x - label_x) <= frame_width * 0.04
            ):
                pair_candidates.append(
                    (
                        abs(number_centre_x - label_centre_x),
                        number_centre_y - label_centre_y,
                        label,
                        label_box,
                        number_box,
                    )
                )

    # Two different parameter rows in one narrow x lane prove the fixed
    # background column.  Values are matched one-to-one; sharing one OCR
    # number between two labels cannot manufacture the required proof.
    unique_pairs: list[
        tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]
    ] = []
    seen_labels: set[str] = set()
    seen_numbers: set[tuple[int, int, int, int]] = set()
    for _, _, label, label_box, number_box in sorted(pair_candidates):
        if label in seen_labels or number_box in seen_numbers:
            continue
        seen_labels.add(label)
        seen_numbers.add(number_box)
        unique_pairs.append((label, label_box, number_box))
    if len(unique_pairs) < 2:
        return None
    pair_top = min(
        min(pair[1][1], pair[2][1]) for pair in unique_pairs
    )
    pair_bottom = max(
        max(
            pair[1][1] + pair[1][3],
            pair[2][1] + pair[2][3],
        )
        for pair in unique_pairs
    )
    if pair_top >= roi_bottom or pair_bottom <= roi_top:
        return None
    # A title-bound ROI deliberately starts slightly above the title.  A
    # background stat row can therefore touch only that top margin while
    # the actual popover begins below it.  Such a row cannot justify
    # clipping the entire right side: doing so would discard legitimate
    # effect tokens rendered to the right of the condition text (for
    # example ``好調状態の場合、集中``).  Require the proven background
    # column to reach into the title/effect panel itself.
    if pair_bottom <= title_y:
        return None
    label_order = {
        "ボーカル": 0,
        "ダンス": 1,
        "ビジュアル": 2,
        "体力": 3,
    }
    ordered_pairs = sorted(unique_pairs, key=lambda value: label_order[value[0]])
    for earlier, later in zip(ordered_pairs, ordered_pairs[1:], strict=False):
        earlier_index = label_order[earlier[0]]
        later_index = label_order[later[0]]
        earlier_y = earlier[1][1] + earlier[1][3] / 2.0
        later_y = later[1][1] + later[1][3] / 2.0
        index_gap = later_index - earlier_index
        row_step = (later_y - earlier_y) / index_gap
        if not frame_height * 0.02 <= row_step <= frame_height * 0.08:
            return None
    label_lefts = [pair[1][0] for pair in unique_pairs]
    number_lefts = [pair[2][0] for pair in unique_pairs]
    lane_tolerance = max(4, int(round(frame_width * 0.035)))
    if (
        max(label_lefts) - min(label_lefts) > lane_tolerance
        or max(number_lefts) - min(number_lefts) > lane_tolerance
    ):
        return None
    column_left = min((*label_lefts, *number_lefts))
    boundary = column_left - max(2, int(round(frame_width * 0.008)))
    if boundary <= title_right + max(4, int(round(frame_width * 0.02))):
        return None
    return boundary


def _matching_ocr_items(items: Sequence[Any], expected: str) -> tuple[Any, ...]:
    """Apply Maa's expected search to already filtered, ordered boxes."""
    pattern = re.compile(expected)
    return tuple(item for item in items if pattern.search(_text(item)))


def _spatial_ocr_rows(
    items: Sequence[Any],
) -> tuple[
    tuple[
        str,
        tuple[int, int, int, int],
        tuple[tuple[str, tuple[int, int, int, int]], ...],
    ],
    ...,
]:
    """Rebuild visual reading rows with their union boxes.

    Maa's vertical ordering compares top coordinates. Two tokens on one
    visual line can therefore be reversed when the right token has a taller
    box that begins a few pixels earlier. Cluster only strongly overlapping
    vertical spans, then order members left-to-right and rows top-to-bottom.
    """

    entries: list[tuple[int, int, int, int, str]] = []
    for item in items:
        value = _text(item).strip()
        if not value:
            continue
        x, y, width, height = _box(item)
        if width < 1 or height < 1:
            raise ArenaReaderError(
                "recognition_box_invalid",
                f"OCR result has an invalid box: {(x, y, width, height)!r}",
            )
        entries.append((x, y, width, height, value))

    def median(values: Sequence[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    rows: list[list[tuple[int, int, int, int, str]]] = []
    for entry in sorted(
        entries,
        key=lambda value: (value[1] + value[3] / 2.0, value[0]),
    ):
        _, y, _, height, _ = entry
        centre = y + height / 2.0
        candidates: list[tuple[float, int]] = []
        for row_index, row in enumerate(rows):
            row_centres = [item[1] + item[3] / 2.0 for item in row]
            row_heights = [float(item[3]) for item in row]
            row_centre = median(row_centres)
            row_height = median(row_heights)
            height_ratio = max(height, row_height) / min(height, row_height)
            overlap = min(
                y + height,
                row_centre + row_height / 2.0,
            ) - max(
                y,
                row_centre - row_height / 2.0,
            )
            overlap_ratio = max(0.0, overlap) / min(height, row_height)
            centre_gap = abs(centre - row_centre)
            if (
                height_ratio <= 1.50
                and overlap_ratio >= 0.50
                and centre_gap <= 0.65 * max(height, row_height)
            ):
                candidates.append((centre_gap, row_index))
        if candidates:
            _, row_index = min(candidates)
            rows[row_index].append(entry)
        else:
            rows.append([entry])

    rows.sort(
        key=lambda row: (
            median([item[1] + item[3] / 2.0 for item in row]),
            min(item[0] for item in row),
        )
    )
    spatial_rows: list[
        tuple[
            str,
            tuple[int, int, int, int],
            tuple[tuple[str, tuple[int, int, int, int]], ...],
        ]
    ] = []
    for row in rows:
        ordered_row = sorted(row, key=lambda value: (value[0], value[1]))
        left = min(item[0] for item in ordered_row)
        top = min(item[1] for item in ordered_row)
        right = max(item[0] + item[2] for item in ordered_row)
        bottom = max(item[1] + item[3] for item in ordered_row)
        spatial_rows.append(
            (
                "".join(item[4] for item in ordered_row),
                (left, top, right - left, bottom - top),
                tuple(
                    (item[4], (item[0], item[1], item[2], item[3]))
                    for item in ordered_row
                ),
            )
        )
    return tuple(spatial_rows)


def _spatial_ocr_text(items: Sequence[Any]) -> str:
    """Flatten spatial OCR rows into newline-separated text."""

    return "\n".join(
        text
        for text, _, _ in _spatial_ocr_rows(items)
    )


def _p_item_detail_panel_rows(
    image: Any,
    items: Sequence[Any],
    *,
    background_only: bool = False,
) -> tuple[
    tuple[
        str,
        tuple[float, float, float, float],
        tuple[tuple[str, tuple[float, float, float, float]], ...],
    ],
    ...,
]:
    """Read spatial OCR rows and normalized boxes from the left panel.

    P-item titles move vertically with source-page content, so geometry
    identifies the panel rather than guessing a title band. The transaction
    layer removes frozen source rows and treats only the first new row as
    title evidence; later effect prose remains visible for diagnostics but
    cannot participate in catalog matching.

    The background-only view contains just the same-panel atoms excluded
    by the title height gate. It never supplies title or page signatures.
    """

    height, width = image.shape[:2]
    if width < 320 or height < 568:
        raise ArenaReaderError(
            "p_item_detail_title_roi_invalid",
            f"capture {width}x{height} is too small for P-item title geometry",
        )
    scale_x = width / 720.0
    scale_y = height / 1280.0
    panel_items: list[Any] = []
    for item in items:
        x, y, item_width, item_height = _box(item)
        normalized_x = x / scale_x
        normalized_y = y / scale_y
        normalized_width = item_width / scale_x
        normalized_height = item_height / scale_y
        normalized_text = re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", _text(item)),
        )
        if (
            normalized_width < 16
            or normalized_height <= 0
            or (18 <= normalized_height <= 45) == background_only
            or not normalized_text
            or not any(character.isalpha() for character in normalized_text)
        ):
            continue
        in_left_panel = (
            18 <= normalized_x
            and normalized_x <= 360
            and normalized_x + normalized_width <= 650
            # The live top-panel title box can begin at y=15.  Reserve a
            # +/-3 px detector-box tolerance without admitting rows above
            # y=12; all horizontal, size, frozen-source and first-new-row
            # identity gates remain intact.
            and 12 <= normalized_y <= 270
        )
        if in_left_panel:
            panel_items.append(item)
    if not panel_items:
        return ()

    spatial_rows = _spatial_ocr_rows(panel_items)

    def isolated_ascii_artwork_noise(
        text: str,
        row_box: tuple[int, int, int, int],
        components: tuple[tuple[str, tuple[int, int, int, int]], ...],
    ) -> bool:
        normalized = re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", text),
        )
        if (
            len(components) != 1
            or re.fullmatch(r"[A-Za-z]", normalized) is None
        ):
            return False
        row_x, row_y, row_width, row_height = row_box
        normalized_row_box = (
            row_x / scale_x,
            row_y / scale_y,
            row_width / scale_x,
            row_height / scale_y,
        )
        if not (
            normalized_row_box[0] <= 30
            and normalized_row_box[1] <= 45
            and normalized_row_box[2] <= 24
            and normalized_row_box[3] <= 22
        ):
            # A low-confidence letter at the normal title position remains
            # the first new row.  Removing it could promote effect prose to
            # identity evidence, so geometry outside the observed artwork
            # corner must continue to fail closed.
            return False
        component_text, component_box = components[0]
        scores = tuple(
            float(_value(item, "score", 1.0))
            for item in panel_items
            if _text(item).strip() == component_text
            and _box(item) == component_box
        )
        # Full-screen OCR can expose one low-confidence Latin glyph from
        # the P-item artwork above the real title.  Filter only a complete
        # isolated row after spatial joining so split titles such as
        # ``P`` + ``っち+`` retain their Latin component.
        return bool(scores) and max(scores) < 0.50

    return tuple(
        (
            text,
            (
                x / scale_x,
                y / scale_y,
                row_width / scale_x,
                row_height / scale_y,
            ),
            tuple(
                (
                    component_text,
                    (
                        component_x / scale_x,
                        component_y / scale_y,
                        component_width / scale_x,
                        component_height / scale_y,
                    ),
                )
                for component_text, (
                    component_x,
                    component_y,
                    component_width,
                    component_height,
                ) in components
            ),
        )
        for text, (x, y, row_width, row_height), components in spatial_rows
        if not isolated_ascii_artwork_noise(
            text,
            (x, y, row_width, row_height),
            components,
        )
    )


def _isolated_p_item_title_row(
    image: Any, source_images: Sequence[Any], atoms: Sequence[dict[str, Any]],
) -> _PItemPanelRow | None:
    """Recover only the top row of a proven new panel, never search prose for a name."""
    from arena_winrate._detail_title_region import _white_detail_panels, isolated_p_item_body_region

    panels = _white_detail_panels(image)
    if len(panels) != 1:
        return None
    px, py, pw, ph = panels[0]
    height, width = image.shape[:2]
    sx, sy = width / 720, height / 1280
    inside = []
    for atom in atoms:
        x, y, w, h = (v * scale for v, scale in zip(atom["box"], (sx, sy, sx, sy), strict=True))
        if px <= x < x + w <= px + pw and py <= y < y + h <= py + ph:
            inside.append({**atom, "box": (x, y, w, h)})
        elif px <= x + w / 2 <= px + pw and py <= y + h / 2 <= py + ph:
            return None
    rows = _p_item_detail_panel_rows(image, inside)
    if not rows:
        return None
    title_box = tuple(v * scale for v, scale in zip(rows[0][1], (sx, sy, sx, sy), strict=True))
    if title_box[1] - py > height * .018:
        return None  # A missing title must not promote the first effect line.
    if isolated_p_item_body_region(image, source_images, title_box) is None:
        return None
    return rows[0]


def _p_item_detail_title_text(
    image: Any,
    items: Sequence[Any],
) -> str:
    """Flatten normalized left-panel rows for diagnostics and tests."""

    return "\n".join(
        text
        for text, _, _ in _p_item_detail_panel_rows(
            image,
            items,
        )
    )


def _box_center_point(
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Collapse one uniquely recognized control to its deterministic centre."""

    x, y, width, height = box
    if width < 1 or height < 1:
        raise ArenaReaderError(
            "maa_click_box_invalid",
            f"recognized control has an invalid box: {box!r}",
        )
    return (x + width // 2, y + height // 2, 1, 1)


def _p_item_source_proof_boxes(
    source_image: Any,
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    source_height, source_width = source_image.shape[:2]
    if source_width < 320 or source_height < 568:
        raise ArenaReaderError(
            "p_item_source_restore_evidence_invalid",
            "P-item source capture is too small for detail-panel restore evidence",
        )
    scale_x = source_width / 720.0
    scale_y = source_height / 1280.0

    def scaled_box(
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = box
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(8, int(round(width * scale_x))),
            max(8, int(round(height * scale_y))),
        )

    panel_boxes = (
        scaled_box((18, 16, 331, 45)),
        scaled_box((100, 132, 320, 58)),
    )
    return (*boxes, *panel_boxes)


def _p_item_border_only_change(
    source: Any, current: Any, boxes: Sequence[tuple[int, int, int, int]], threshold: float,
) -> bool:
    """Ignore sparse border shine only; preserve artwork and the '+' corner."""
    import numpy as np

    deltas = []
    for x, y, width, height in boxes:
        before = source[y:y + height, x:x + width]
        after = current[y:y + height, x:x + width]
        if before.shape != (height, width, 3) or after.shape != before.shape:
            return False
        deltas.append(after.astype(np.int16) - before.astype(np.int16))
    for delta in deltas:
        height, width = delta.shape[:2]
        magnitude = np.abs(delta)
        core = np.zeros((height, width), dtype=bool)
        mx, my = max(1, round(width / 8)), max(1, round(height / 8))
        core[my:height - my, mx:width - mx] = True
        corner = np.zeros_like(core)
        corner[:round(height * 0.40), round(width * 0.60):] = True
        if float(magnitude[core].mean()) > threshold:
            return False
        # The upgrade corner stays protected. Only a pixel-identical shine
        # observed in at least THREE distinct slots can explain changes to
        # its outer edge; a removed '+' in one slot cannot borrow this proof.
        votes = np.zeros_like(core, dtype=np.uint8)
        if len(deltas) == 4:
            for other in deltas:
                if other.shape == delta.shape:
                    votes += np.all(np.abs(other - delta) <= 1, axis=2)
        corner_error = magnitude.copy()
        corner_error[(votes >= 3) & ~core] = 0
        if float(corner_error[corner].mean()) > threshold:
            return False
        changed = magnitude.max(axis=2) > 1
        if float(changed.mean()) > 0.08:
            return False
        border_delta = delta[~core]
        # A shine brightens (or dims) pixels; mixed colour replacement is
        # not accepted as decoration, even when the centre is unchanged.
        if not (np.all(border_delta >= -1) or np.all(border_delta <= 1)):
            return False
    return True


def _skill_card_empty_observation(
    image: Any,
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Detect only a nearly uniform neutral/dark blank, never card art."""

    import cv2
    import numpy as np

    x, y, width, height = box
    crop = image[y : y + height, x : x + width, :3]
    if crop.size == 0 or width < 16 or height < 16:
        raise ArenaReaderError(
            "skill_card_empty_slot_input_invalid",
            f"skill-card slot crop is empty or too small: {box!r}",
        )
    inset = max(2, int(round(min(width, height) * 0.06)))
    inner = crop[inset:-inset, inset:-inset]
    if inner.size == 0:
        inner = crop
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 32, 96)
    gray_mean = float(np.mean(gray))
    gray_stddev = float(np.std(gray))
    gray_p95 = float(np.percentile(gray, 95))
    dark_pixel_ratio = float(np.mean(gray <= 24))
    edge_density = float(np.mean(edges > 0))
    channel_range = np.max(inner, axis=2) - np.min(inner, axis=2)
    mean_channel_range = float(np.mean(channel_range))
    uniform_neutral_placeholder = bool(
        mean_channel_range <= 8.0
        and gray_stddev <= 8.0
        and edge_density <= 0.01
    )
    uniform_dark_placeholder = bool(
        gray_mean <= 12.0
        and gray_stddev <= 6.0
        and gray_p95 <= 24.0
        and dark_pixel_ratio >= 0.97
        and edge_density <= 0.015
    )
    empty = bool(
        uniform_neutral_placeholder or uniform_dark_placeholder
    )
    return {
        "empty": empty,
        "uniform_neutral_placeholder": uniform_neutral_placeholder,
        "uniform_dark_placeholder": uniform_dark_placeholder,
        "gray_mean": round(gray_mean, 6),
        "gray_stddev": round(gray_stddev, 6),
        "gray_p95": round(gray_p95, 6),
        "dark_pixel_ratio": round(dark_pixel_ratio, 6),
        "edge_density": round(edge_density, 6),
        "mean_channel_range": round(mean_channel_range, 6),
    }
