"""Predict an old background through a separately measured straight-edge shadow.

No candidate pixel participates in fitting. A caller still owns source votes,
raw stroke contradictions, the exposed-area gate and the unchanged 4/32 tests.
"""

from __future__ import annotations

import numpy as np

Box = tuple[int, int, int, int]
_MIN_DONORS = 12
_BRIGHT = 250
_RESIDUAL = 4
_STRONG_DELTA = 32


def _box_valid(box: Box, width: int, height: int) -> bool:
    return (isinstance(box, tuple) and len(box) == 4
            and all(type(value) is int for value in box)
            and box[2] > 0 and box[3] > 0 and box[0] >= 0 and box[1] >= 0
            and box[0] + box[2] <= width and box[1] + box[3] <= height)


def _rotate_box(box: Box, width: int, height: int, turns: int) -> Box:
    x, y, w, h = box
    if turns == 1:
        return y, width - x - w, h, w
    if turns == 2:
        return width - x - w, height - y - h, w, h
    if turns == 3:
        return height - y - h, x, h, w
    return box


def _two_blocks(available: np.ndarray, before: int, after: int, span: int) -> tuple[slice, slice] | None:
    """Select by geometry and source availability, never by observed ratios."""
    count = len(available)

    def sufficient(block: slice) -> bool:
        return bool(np.all(available[block].sum(axis=0) >= _MIN_DONORS))

    first, second = slice(0, before), slice(after, count)
    if sufficient(first) and sufficient(second):
        return first, second
    # Both independently supported strips may precede/follow the candidate:
    # the proved straight edge has one translation-invariant normal profile.
    intervals = sorted(((0, before), (after, count)), key=lambda item: item[1] - item[0], reverse=True)
    for start, stop in intervals:
        middle = (start + stop) // 2
        if (stop - start) / 2 < span:
            continue
        first, second = slice(start, middle), slice(middle, stop)
        if sufficient(first) and sufficient(second):
            return first, second
    return None


def _right_projection(image, source, box: Box, panel: Box, core, exposed, margin: int):
    x, y, w, h = box
    px, py, pw, ph = panel
    edge = px + pw
    if not (py <= y < y + h <= py + ph and x < edge + margin and x + w > edge):
        return None
    if not exposed.any() or np.any(exposed.any(axis=0) != exposed.all(axis=0)):
        return None
    columns = np.flatnonzero(exposed.all(axis=0)) + x
    if not len(columns) or np.any(columns < edge):
        return None
    shadow_columns = columns[columns < edge + margin]
    if not len(shadow_columns):
        return None

    # A curved/cropped edge cannot supply this one-dimensional projection.
    rightmost = np.max(np.where(core, np.arange(pw)[None, :], -1), axis=1)
    flat = rightmost == pw - 1
    top, bottom = y - py, y + h - py
    if not flat[top:bottom].all():
        return None
    start, stop = top, bottom
    while start and flat[start - 1]:
        start -= 1
    while stop < ph and flat[stop]:
        stop += 1
    start += margin
    stop -= margin
    if not (start <= top < bottom <= stop):
        return None

    # Sample the entire admitted shadow band plus an independent near-unity
    # outer anchor. Points under the white paint are excluded by the caller.
    first_distance = int(shadow_columns.min() - edge)
    normal = np.arange(edge + first_distance, edge + margin + 2)
    if normal[-1] >= image.shape[1]:
        return None
    old = source[py + start:py + stop, normal].astype(np.float32)
    current = image[py + start:py + stop, normal].astype(np.float32)
    source_white = (old.min(axis=2) >= _BRIGHT) & (np.ptp(old, axis=2) <= _RESIDUAL)
    outside = np.ones(stop - start, bool)
    outside[top - start:bottom - start] = False
    available = source_white & outside[:, None]
    blocks = _two_blocks(available, top - start, bottom - start, h)
    if blocks is None:
        return None
    # Every source-white donor is checked, including points outside the two
    # fitting strips. A median must never silently choose one lighting party.
    if (np.any(np.ptp(current, axis=2)[available] > _RESIDUAL)
            or np.any((current - old)[available] > _RESIDUAL)
            or np.any((old - current)[available] > _STRONG_DELTA)):
        return None
    profiles = []
    for block in blocks:
        mask = available[block]
        ratios = current[block] / np.maximum(old[block], 1)
        profile = np.array([np.median(ratios[:, column][mask[:, column]])
                            for column in range(len(normal))], dtype=np.float32)
        profiles.append(np.minimum(profile, 1))
    for profile in profiles:
        residual = np.abs(current - old * profile[None, :, None])
        if np.any(residual[available] > _RESIDUAL):
            return None
        # One level accommodates 8-bit quantization, not a second shadow.
        if (np.any(np.diff(profile) < -1 / _BRIGHT)
                or profile[-1] < 1 - _RESIDUAL / _BRIGHT
                or profile[0] >= 1 - _RESIDUAL / _BRIGHT):
            return None
    profile = (profiles[0] + profiles[1]) / 2
    predicted = source[y:y + h, x:x + w].astype(np.float32).copy()
    for column in shadow_columns:
        predicted[:, column - x] *= profile[column - normal[0]]
    return predicted


def shadow_source_projection(
    image: np.ndarray, source: np.ndarray, box: Box, panel: Box, core: np.ndarray, *, exposed: np.ndarray,
) -> np.ndarray | None:
    """Return float32 RGB old-source pixels, with unexposed pixels unchanged.

    The profile is measured from two disjoint source-white strips outside the
    candidate, on the same continuous straight edge and away from its corners.
    Both strips need at least 12 points per sampled column. Their centers are
    separated by at least the candidate's tangential size, even on one side.
    Every available donor must agree within four original RGB levels. No fit
    reaches beyond the existing seven-pixel-at-720px shadow band.
    """
    if (not isinstance(image, np.ndarray) or not isinstance(source, np.ndarray)
            or image.dtype != np.uint8 or source.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or source.shape != image.shape):
        return None
    height, width = image.shape[:2]
    if (not _box_valid(box, width, height) or not _box_valid(panel, width, height)
            or not isinstance(core, np.ndarray) or core.dtype != np.bool_
            or core.shape != (panel[3], panel[2])
            or not isinstance(exposed, np.ndarray) or exposed.dtype != np.bool_
            or exposed.shape != (box[3], box[2])):
        return None
    x, y, w, h = box
    px, py, pw, ph = panel
    sides = [x + w > px + pw and py <= y and y + h <= py + ph,
             y + h > py + ph and px <= x and x + w <= px + pw,
             x < px and py <= y and y + h <= py + ph,
             y < py and px <= x and x + w <= px + pw]
    if sum(sides) != 1:
        return None
    turns = sides.index(True)
    margin = max(3, round(min(width, height) * .01))
    prediction = _right_projection(
        np.rot90(image, turns), np.rot90(source, turns),
        _rotate_box(box, width, height, turns), _rotate_box(panel, width, height, turns),
        np.rot90(core, turns), np.rot90(exposed, turns), margin,
    )
    return None if prediction is None else np.rot90(prediction, -turns).copy()
