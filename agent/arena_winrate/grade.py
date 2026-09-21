"""Contest Grade slot limits and conservative visible-slot classification."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

import numpy as np

GRADE_STAGE_MEMBER_CAPS: dict[int, tuple[int, int, int]] = {
    1: (1, 1, 1),
    2: (2, 1, 1),
    3: (2, 2, 2),
    4: (2, 2, 2),
    5: (2, 2, 2),
    6: (2, 2, 2),
    7: (3, 3, 3),
}

MEMBER_SLOT_MIN_GRAY_STDDEV = 28.0
MEMBER_SLOT_MIN_EDGE_DENSITY = 0.05
EMPTY_SLOT_MAX_INNER_GRAY_MEAN = 55.0
EMPTY_SLOT_MIN_DARK_PIXEL_RATIO = 0.75
EMPTY_PLACEHOLDER_MAX_INNER_GRAY_MEAN = 100.0
EMPTY_PLACEHOLDER_MAX_GRAY_STDDEV = 36.0
EMPTY_PLACEHOLDER_MAX_EDGE_DENSITY = 0.12


class MemberSlotState(str, Enum):
    OCCUPIED = "occupied"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class MemberSlotMetrics:
    gray_mean: float
    gray_stddev: float
    inner_gray_mean: float
    inner_dark_pixel_ratio: float
    edge_density: float


def stage_member_cap(grade: int, stage_number: int) -> int:
    """Return the maximum enabled member slots for one Grade and stage."""

    if grade not in GRADE_STAGE_MEMBER_CAPS:
        raise ValueError(f"contest Grade must be from 1 to 7: {grade!r}")
    if stage_number not in (1, 2, 3):
        raise ValueError(f"contest stage must be from 1 to 3: {stage_number!r}")
    return GRADE_STAGE_MEMBER_CAPS[grade][stage_number - 1]


def fixed_member_slot_boxes(
    frame_width: int,
    avatar_top: int,
    avatar_height: int,
    slot_cap: int,
    *,
    group_center_ratio: float = 0.6430555556,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the enabled prefix of the fixed three-column rehearsal layout."""

    if frame_width < 1 or avatar_height < 1:
        raise ValueError("frame width and avatar height must be positive")
    if slot_cap not in (1, 2, 3):
        raise ValueError(f"slot cap must be from 1 to 3: {slot_cap!r}")
    if not 0.0 < group_center_ratio < 1.0:
        raise ValueError("member-slot group center ratio must be inside (0,1)")
    avatar_width = int(frame_width * 0.125)
    gap = int(round(frame_width / 120))
    group_width = 3 * avatar_width + 2 * gap
    group_center = int(round(frame_width * group_center_ratio))
    group_left = group_center - group_width // 2
    return tuple(
        (group_left + index * (avatar_width + gap), avatar_top, avatar_width, avatar_height)
        for index in range(slot_cap)
    )


def team_stage_total_anchors(
    total_anchors: tuple[tuple[int, int, int, int], ...],
    *,
    own_team: bool,
) -> tuple[tuple[int, int, int, int], ...]:
    """Select the three team-side totals from rehearsal or opponent comparison."""

    ordered = tuple(sorted(total_anchors, key=lambda box: (box[1], box[0])))
    if own_team:
        if len(ordered) != 3:
            raise ValueError(f"own team requires three stage totals, found {len(ordered)}")
        return ordered
    if len(ordered) != 6:
        raise ValueError(f"opponent comparison requires six stage totals, found {len(ordered)}")
    selected: list[tuple[int, int, int, int]] = []
    for index in range(0, 6, 2):
        pair = tuple(sorted(ordered[index : index + 2], key=lambda box: box[0]))
        if abs(pair[0][1] - pair[1][1]) > 12 or pair[0][0] >= pair[1][0]:
            raise ValueError(f"opponent stage total pair is ambiguous: {pair!r}")
        selected.append(pair[1])
    return tuple(selected)


def measure_member_slot(crop: np.ndarray) -> MemberSlotMetrics:
    """Measure one already-located slot without retaining its screenshot."""

    if crop.ndim != 3 or crop.shape[2] < 3 or crop.shape[0] < 10 or crop.shape[1] < 10:
        raise ValueError(f"member-slot crop has an unsupported shape: {crop.shape!r}")
    gray = (
        crop[:, :, 0].astype(np.float32) * 0.114
        + crop[:, :, 1].astype(np.float32) * 0.587
        + crop[:, :, 2].astype(np.float32) * 0.299
    ).astype(np.uint8)
    inset_y = max(1, int(gray.shape[0] * 0.1))
    inset_x = max(1, int(gray.shape[1] * 0.1))
    inner = gray[inset_y:-inset_y, inset_x:-inset_x]
    gray_int = gray.astype(np.int16)
    gradient = np.zeros_like(gray, dtype=np.uint8)
    gradient[:, 1:] = np.maximum(
        gradient[:, 1:],
        np.abs(gray_int[:, 1:] - gray_int[:, :-1]).clip(0, 255).astype(np.uint8),
    )
    gradient[1:, :] = np.maximum(
        gradient[1:, :],
        np.abs(gray_int[1:, :] - gray_int[:-1, :]).clip(0, 255).astype(np.uint8),
    )
    return MemberSlotMetrics(
        gray_mean=float(gray.mean()),
        gray_stddev=float(gray.std()),
        inner_gray_mean=float(inner.mean()),
        inner_dark_pixel_ratio=float((inner <= 45).mean()),
        edge_density=float((gradient > 25).mean()),
    )


def _has_centered_empty_slot_dash(crop: np.ndarray) -> bool:
    """Recognize the explicit dash on a flat placeholder of either team color."""
    height, width = crop.shape[:2]
    dy, dx = max(1, int(height * 0.1)), max(1, int(width * 0.1))
    inner = crop[dy:-dy, dx:-dx, :3].astype(np.float32)
    background = np.median(inner.reshape(-1, 3), axis=0)
    foreground = np.max(np.abs(inner - background), axis=2) > 16
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        return False
    ih, iw = foreground.shape
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bar_width, bar_height = right - left, bottom - top
    # Every non-background pixel must fit one short, filled central bar. A
    # portrait, extra mark, loading frame or texture cannot become an empty slot.
    return bool(
        0.18 * iw <= bar_width <= 0.45 * iw
        and 0.035 * ih <= bar_height <= 0.13 * ih
        and bar_width >= 2.5 * bar_height
        and abs((left + right) / 2 - iw / 2) <= 0.12 * iw
        and abs((top + bottom) / 2 - ih / 2) <= 0.12 * ih
        and foreground.sum() >= 0.85 * bar_width * bar_height
        and np.all(np.median(inner[foreground], axis=0) - background >= 20)
    )


def classify_member_slot(crop: np.ndarray) -> tuple[MemberSlotState, MemberSlotMetrics]:
    """Require a known blank or portrait; weak non-black slots remain ambiguous."""

    metrics = measure_member_slot(crop)
    if (
        metrics.inner_gray_mean <= EMPTY_SLOT_MAX_INNER_GRAY_MEAN
        and metrics.inner_dark_pixel_ratio >= EMPTY_SLOT_MIN_DARK_PIXEL_RATIO
    ):
        return MemberSlotState.EMPTY, metrics
    if (
        metrics.inner_gray_mean <= EMPTY_PLACEHOLDER_MAX_INNER_GRAY_MEAN
        and metrics.gray_stddev <= EMPTY_PLACEHOLDER_MAX_GRAY_STDDEV
        and metrics.edge_density <= EMPTY_PLACEHOLDER_MAX_EDGE_DENSITY
    ):
        return MemberSlotState.EMPTY, metrics
    if (
        metrics.gray_stddev >= MEMBER_SLOT_MIN_GRAY_STDDEV
        and metrics.edge_density >= MEMBER_SLOT_MIN_EDGE_DENSITY
    ):
        return MemberSlotState.OCCUPIED, metrics
    if _has_centered_empty_slot_dash(crop):
        return MemberSlotState.EMPTY, metrics
    return MemberSlotState.AMBIGUOUS, metrics
