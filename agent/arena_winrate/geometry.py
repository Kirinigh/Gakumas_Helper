"""Pure geometry helpers for read-only arena screen parsing."""

from __future__ import annotations

from typing import TypeVar
from statistics import median
from collections.abc import Mapping, Sequence

PARAMETER_LABEL_ORDER = ("ボーカル", "ダンス", "ビジュアル", "体力")
P_ITEM_SCREEN_SLOT_TO_ENGINE_SLOT = (1, 4, 3, 2)

_T = TypeVar("_T")


def p_item_screen_order_to_engine_order(
    values: Sequence[_T],
) -> tuple[_T, _T, _T, _T]:
    """Translate the four visible P-item cells into simulator slot order.

    Contest member details render the three non-primary P-items in reverse
    insertion order. The simulator URL/engine contract keeps insertion order,
    so visible left-to-right slots ``1,2,3,4`` map to engine slots
    ``1,4,3,2``. The permutation is fixed by two independently confirmed
    stage-172 loadouts and is also its own inverse.
    """

    if len(values) != 4:
        raise ValueError("exactly four visible P-item slots are required")
    first, second, third, fourth = values
    return first, fourth, third, second


def p_item_screen_slot_to_engine_slot(screen_slot: int) -> int:
    """Return the one-based simulator slot for a one-based visible slot."""

    if screen_slot not in (1, 2, 3, 4):
        raise ValueError("P-item screen slot must be in 1..4")
    return P_ITEM_SCREEN_SLOT_TO_ENGINE_SLOT[screen_slot - 1]


def infer_parameter_label_boxes(
    observed: Mapping[str, tuple[int, int, int, int]],
    *,
    frame_height: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Resolve four fixed parameter rows from three or four OCR labels.

    The member detail layout always renders the four rows in a fixed order.
    One missing Japanese label may therefore be inferred from the other three,
    but two missing rows or inconsistent geometry remain a hard failure.
    """

    if frame_height <= 0:
        raise ValueError("parameter-label frame height must be positive")
    known = {
        PARAMETER_LABEL_ORDER.index(label): tuple(map(int, box))
        for label, box in observed.items()
        if label in PARAMETER_LABEL_ORDER and len(box) == 4
    }
    if len(known) not in (3, 4):
        raise ValueError("three or four unique parameter labels are required")
    ordered_known = sorted(known.items())
    if any(
        later[1][1] <= earlier[1][1]
        for earlier, later in zip(ordered_known, ordered_known[1:], strict=False)
    ):
        raise ValueError("parameter labels are not vertically ordered")
    if len(known) == 4:
        return tuple(known[index] for index in range(4))

    indices = [float(index) for index, _ in ordered_known]
    y_values = [float(box[1]) for _, box in ordered_known]
    mean_index = sum(indices) / len(indices)
    mean_y = sum(y_values) / len(y_values)
    denominator = sum((index - mean_index) ** 2 for index in indices)
    if denominator <= 0:
        raise ValueError("parameter-label row spacing cannot be fitted")
    step = sum(
        (index - mean_index) * (y - mean_y)
        for index, y in zip(indices, y_values, strict=True)
    ) / denominator
    intercept = mean_y - step * mean_index
    if not frame_height * 0.02 <= step <= frame_height * 0.08:
        raise ValueError(f"parameter-label row step is implausible: {step:.3f}")
    residual = max(
        abs(y - (intercept + step * index))
        for index, y in zip(indices, y_values, strict=True)
    )
    if residual > max(4.0, frame_height * 0.006):
        raise ValueError(f"parameter-label rows are inconsistent: residual={residual:.3f}")

    missing_index = next(index for index in range(4) if index not in known)
    widths = [box[2] for box in known.values()]
    heights = [box[3] for box in known.values()]
    x_values = [box[0] for box in known.values()]
    inferred = (
        int(round(median(x_values))),
        int(round(intercept + step * missing_index)),
        int(round(median(widths))),
        int(round(median(heights))),
    )
    if inferred[1] < 0 or inferred[1] + inferred[3] > int(frame_height * 0.25):
        raise ValueError("inferred parameter-label row is outside the detail header")
    known[missing_index] = inferred
    return tuple(known[index] for index in range(4))


def excluded_duplicate_card_flags(
    card_boxes: Sequence[tuple[int, int, int, int]],
    marker_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[bool, ...]:
    """Map visible ``重複``/``制限`` OCR markers to six fixed card slots.

    Either marker is rendered over the upper half of a grey excluded card. OCR
    geometry, rather than the overlaid card artwork, is authoritative here:
    the artwork is deliberately not a usable card-identity input.
    """

    flags = [False] * len(card_boxes)
    for marker_x, marker_y, marker_width, marker_height in marker_boxes:
        marker_centre_x = marker_x + marker_width / 2
        marker_centre_y = marker_y + marker_height / 2
        matches = [
            index
            for index, (card_x, card_y, card_width, card_height) in enumerate(card_boxes)
            if card_x - card_width * 0.10 <= marker_centre_x <= card_x + card_width * 1.10
            and card_y - card_height * 0.10 <= marker_centre_y <= card_y + card_height * 0.62
        ]
        if len(matches) == 1:
            flags[matches[0]] = True
    return tuple(flags)


def stable_excluded_duplicate_card_flags(
    observations: Sequence[Sequence[bool]],
) -> tuple[bool | None, ...]:
    """Require a two-of-three marker consensus and surface one-off hits."""

    if len(observations) != 3 or not observations:
        raise ValueError("exactly three duplicate-marker observations are required")
    width = len(observations[0])
    if any(len(observation) != width for observation in observations):
        raise ValueError("duplicate-marker observations must have equal widths")
    result: list[bool | None] = []
    for index in range(width):
        positive = sum(bool(observation[index]) for observation in observations)
        result.append(True if positive >= 2 else False if positive == 0 else None)
    return tuple(result)


def resolve_excluded_duplicate_card_flags(
    primary: Sequence[Sequence[bool]],
    *,
    targeted: Sequence[Sequence[bool]] | None = None,
    corrective: Sequence[Sequence[bool]] | None = None,
) -> tuple[bool | None, ...]:
    """Resolve one-off OCR hits without weakening persistent duplicate markers.

    The broad OCR pass remains authoritative when it already has a two-of-three
    consensus.  For one-off slots, a card-local enlarged OCR pass may add
    positive evidence.  If that is still inconclusive, exactly one fresh
    three-frame corrective read may replace only the unresolved slots.
    """

    stable = stable_excluded_duplicate_card_flags(primary)
    unresolved = tuple(index for index, value in enumerate(stable) if value is None)
    if not unresolved and targeted is None:
        return stable
    width = len(stable)
    if targeted is not None:
        if len(targeted) != 3 or any(len(frame) != width for frame in targeted):
            raise ValueError("targeted duplicate-marker observations must be 3 x width")
        combined = tuple(
            tuple(bool(primary[frame][slot]) or bool(targeted[frame][slot]) for slot in range(width))
            for frame in range(3)
        )
        targeted_stable = stable_excluded_duplicate_card_flags(combined)
        stable = targeted_stable
        unresolved = tuple(index for index, value in enumerate(stable) if value is None)
        if not unresolved:
            return stable
    if corrective is None:
        return stable
    if len(corrective) != 3 or any(len(frame) != width for frame in corrective):
        raise ValueError("corrective duplicate-marker observations must be 3 x width")
    corrective_stable = stable_excluded_duplicate_card_flags(corrective)
    return tuple(
        corrective_stable[index] if index in unresolved else stable[index]
        for index in range(width)
    )


def canonical_skill_card_rows(
    frame_shape: Sequence[int],
    detector_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Recover fixed six-column detail rows from noisy play-card detections.

    The contest member detail uses a fixed six-column card grid. The upstream
    play-card detector can miss columns or return a second, taller box for the
    same card when customization badges are present. Treat detector output as
    row/column anchors only and synthesize click ROIs after at least four
    distinct columns agree on one row. This deliberately fails closed when a
    screen does not provide enough geometric evidence.
    """

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")

    left = int(round(width * 0.039))
    pitch = int(round(width * 0.1264))
    card_width = int(round(width * 0.114))
    card_height = int(round(height * 0.066))
    expected_lefts = tuple(left + pitch * index for index in range(6))
    expected_centers = tuple(value + card_width / 2 for value in expected_lefts)

    plausible: list[tuple[int, int, int, int, int]] = []
    for raw_box in detector_boxes:
        x, y, box_width, box_height = (int(value) for value in raw_box)
        if not int(width * 0.10) <= box_width <= int(width * 0.14):
            continue
        if not 0.9 <= box_height / box_width <= 1.9:
            continue
        if not int(height * 0.15) <= y <= int(height * 0.70):
            continue
        center_x = x + box_width / 2
        column = min(range(6), key=lambda index: abs(expected_centers[index] - center_x))
        if abs(expected_centers[column] - center_x) > pitch * 0.42:
            continue
        plausible.append((x, y, box_width, box_height, column))
    plausible = list(dict.fromkeys(plausible))

    def column_anchor_y(
        items: Sequence[tuple[int, int, int, int, int]],
        column: int,
    ) -> float:
        column_items = tuple(item for item in items if item[4] == column)
        dimension_error = min(
            (
                abs(item[2] - card_width) + abs(item[3] - card_height)
                for item in column_items
            ),
            default=None,
        )
        if dimension_error is None:
            raise ValueError("column anchor requires at least one detector box")
        # A normal card face and its taller customization overlay can occupy
        # the same fixed column.  The synthesized row is defined by the card
        # face, so prefer the detection whose dimensions best match that face;
        # only equally good, distinct y observations receive an internal vote.
        best_y_values = {
            item[1]
            for item in column_items
            if abs(item[2] - card_width) + abs(item[3] - card_height)
            == dimension_error
        }
        return float(median(best_y_values))

    def column_weighted_y(
        items: Sequence[tuple[int, int, int, int, int]],
    ) -> float:
        columns = {item[4] for item in items}
        column_anchors = tuple(
            column_anchor_y(items, column)
            for column in sorted(columns)
        )
        return float(median(column_anchors))

    clusters: list[list[tuple[int, int, int, int, int]]] = []
    y_tolerance = max(10, int(round(height * 0.018)))
    for item in sorted(plausible, key=lambda value: (value[1], value[0], value[3])):
        compatible_clusters = tuple(
            candidate
            for candidate in clusters
            if abs(column_weighted_y(candidate) - item[1]) <= y_tolerance
        )
        cluster = min(
            compatible_clusters,
            key=lambda candidate: abs(column_weighted_y(candidate) - item[1]),
            default=None,
        )
        if cluster is None:
            clusters.append([item])
        else:
            cluster.append(item)

    cluster_columns = [({item[4] for item in cluster}, cluster) for cluster in clusters]
    maximum_support = max(
        (len(columns) for columns, _ in cluster_columns),
        default=0,
    )
    # A one-card/tall-overlay detection can form an earlier false row beside a
    # complete real row.  Sparse recovery remains available when it is the
    # only evidence, but it must never outrank a row backed by at least four
    # distinct fixed columns.
    minimum_support = 4 if maximum_support >= 4 else maximum_support
    rows: list[tuple[tuple[int, int, int, int], ...]] = []
    for columns, cluster in cluster_columns:
        if not columns or len(columns) < minimum_support:
            continue
        # Each fixed column is one geometric vote.  The detector can emit
        # several boxes for the same card (for example a normal face plus a
        # taller customized-card overlay), so every column first selects its
        # card-face anchor and then contributes exactly one vote.  Duplicate
        # multiplicity therefore cannot move the row or its cluster support.
        row_y = int(round(column_weighted_y(cluster)))
        rows.append(tuple((column_x, row_y, card_width, card_height) for column_x in expected_lefts))
    return tuple(sorted(rows, key=lambda row: row[0][1]))


def expected_scrolled_secondary_skill_card_row(
    frame_shape: Sequence[int],
    primary_row: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the fixed shifted second-row geometry for a scrolled detail page."""

    if len(frame_shape) < 2 or len(primary_row) != 6:
        return ()
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return ()
    primary_y = int(primary_row[0][1])
    if primary_y > int(height * 0.23):
        return ()
    left = int(round(width * 0.214))
    pitch = int(round(width * 0.1264))
    card_width = int(round(width * 0.114))
    card_height = int(round(height * 0.066))
    row_y = primary_y + int(round(height * 0.075))
    return tuple(
        (left + pitch * index, row_y, card_width, card_height)
        for index in range(6)
    )


def canonical_scrolled_secondary_skill_card_row(
    frame_shape: Sequence[int],
    detector_boxes: Sequence[tuple[int, int, int, int]],
    primary_row: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Recover the right-shifted second row after the upper detail is scrolled.

    The two rows overlap after the short scroll. Detector NMS therefore keeps
    the complete first row but may retain only one card from the right-shifted
    second row. Require that card to agree with any fixed shifted column and
    the expected row offset, then derive the remaining fixed slots. Exact card
    identity checks remain a separate downstream gate.
    """

    expected_row = expected_scrolled_secondary_skill_card_row(
        frame_shape,
        primary_row,
    )
    if not expected_row:
        return ()
    height, width = int(frame_shape[0]), int(frame_shape[1])
    row_y = expected_row[0][1]
    expected_lefts = tuple(box[0] for box in expected_row)
    for raw_box in detector_boxes:
        x, y, box_width, box_height = (int(value) for value in raw_box)
        if min(abs(x - expected_left) for expected_left in expected_lefts) > int(
            width * 0.02
        ):
            continue
        if abs(y - row_y) > int(height * 0.02):
            continue
        if not int(width * 0.10) <= box_width <= int(width * 0.14):
            continue
        if not 0.9 <= box_height / box_width <= 1.25:
            continue
        return expected_row
    return ()
