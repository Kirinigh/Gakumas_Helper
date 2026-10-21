"""Same-frame pixel proof for repeated four-point effect decorations.

This does not classify OCR text or decide whether a fragment may be removed.
Callers must first prove an isolated decoration slot and complete surrounding
semantics. A repeated dark character is not sufficient: every component must
also have the filled, symmetric, four-point shape of the game's decoration.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from ._detail_text_layout import Box


def has_repeated_four_point_decoration(
    image: Any, glyph_box: Box, reference_rows: Sequence[Box],
) -> bool:
    """Prove one star from at least two independent text rows on this image.

    Reference boxes are complete effect text rows, not boxes of recognized
    decoration characters. All three small grayscale components must be in
    the same left lane, fully visible, independently star-shaped, and similar.
    No capture, OCR, external template, catalog answer, or cached proof is used.
    """
    import cv2
    import numpy as np

    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or len(reference_rows) < 2):
        return False
    if any(len(box) != 4 or any(not isinstance(value, int) for value in box)
           or box[2] <= 0 or box[3] <= 0 for box in (glyph_box, *reference_rows)):
        return False
    font_height = float(np.median([box[3] for box in reference_rows]))
    x, y, width, height = glyph_box
    if (not 0.35 * font_height <= width <= 0.85 * font_height
            or not 0.35 * font_height <= height <= 0.85 * font_height):
        return False
    centers = [box[1] + box[3] / 2 for box in reference_rows]
    if any(abs(first - second) < 0.7 * font_height
           for index, first in enumerate(centers) for second in centers[index + 1:]):
        return False
    center_x = x + width / 2
    if any(center_x + 0.75 * font_height >= box[0] for box in reference_rows):
        return False

    def component(center_y: float):
        # Stay within this text row: a taller square can touch title strokes
        # above the first reference or the next row's decoration.
        radius = int(round(0.6 * font_height))
        left, top = int(round(center_x)) - radius, int(round(center_y)) - radius
        right, bottom = left + 2 * radius, top + 2 * radius
        if left < 0 or top < 0 or right > image.shape[1] or bottom > image.shape[0]:
            return None
        patch = image[top:bottom, left:right].astype(np.int16)
        lightness = patch.mean(axis=2)
        mask = ((patch.max(axis=2) - patch.min(axis=2) <= 24)
                & (lightness >= 65) & (lightness <= 215)).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = [index for index in range(1, len(stats))
                      if stats[index, cv2.CC_STAT_AREA] >= max(4, 0.01 * font_height**2)]
        if len(candidates) != 1:
            return None
        index = candidates[0]
        a, b, w, h, area = (int(value) for value in stats[index])
        if (a == 0 or b == 0 or a + w == mask.shape[1] or b + h == mask.shape[0]
                or not 0.35 * font_height <= w <= 0.85 * font_height
                or not 0.35 * font_height <= h <= 0.85 * font_height
                or not 0.8 <= w / h <= 1.25 or not 0.25 <= area / (w * h) <= 0.6
                or abs(left + a + w / 2 - center_x) > 0.25 * font_height
                or abs(top + b + h / 2 - center_y) > 0.3 * font_height):
            return None
        shape = labels[b:b + h, a:a + w] == index
        # A four-point star is filled, thin at all four tips and widest in
        # the middle. Holes, crossed strokes and numeral-shaped ends fail.
        for line in (*shape, *shape.T):
            positions = np.flatnonzero(line)
            if not len(positions) or positions[-1] - positions[0] + 1 != len(positions):
                return None
        row_widths, column_heights = shape.sum(axis=1), shape.sum(axis=0)
        if (max(row_widths[:max(1, h // 5)]) > 0.5 * w
                or max(row_widths[-max(1, h // 5):]) > 0.5 * w
                or max(column_heights[:max(1, w // 5)]) > 0.5 * h
                or max(column_heights[-max(1, w // 5):]) > 0.5 * h
                or max(row_widths[h // 3:2 * h // 3 + 1]) < 0.85 * w
                or max(column_heights[w // 3:2 * w // 3 + 1]) < 0.85 * h):
            return None
        # Repeated '+' signs also have four narrow ends, but their arms form
        # an abrupt step. Require several discrete taper widths on both
        # axes, with no single step accounting for most of the expansion.
        # Use integer profile differences: antialiasing can change a native
        # 14-pixel star's width by six pixels at its waist.
        for profile in (row_widths.astype(int), column_heights.astype(int)):
            if (len(np.unique(profile)) < 4
                    or 2 * np.abs(np.diff(profile)).max() > profile.max() - profile.min()):
                return None
        for flipped in (shape[::-1], shape[:, ::-1]):
            if np.logical_and(shape, flipped).sum() / np.logical_or(shape, flipped).sum() < 0.7:
                return None
        return ((left + a, top + b, w, h),
                cv2.resize(shape.astype(np.uint8), (24, 24), interpolation=cv2.INTER_NEAREST))

    candidate = component(y + height / 2)
    if candidate is None:
        return False
    a, b, w, h = candidate[0]
    overlap = max(0, min(x + width, a + w) - max(x, a)) * max(0, min(y + height, b + h) - max(y, b))
    # Native OCR can box just the digit-like part of a star (both preserved
    # failures cover 57% of its component). The independent shape and two
    # same-frame references still have to prove the whole component.
    if overlap < 0.55 * max(width * height, w * h):
        return False
    for center in centers:
        reference = component(center)
        if reference is None:
            return False
        if (abs(reference[0][0] - a) > 0.15 * font_height
                or abs(reference[0][2] - w) > 0.15 * font_height
                or abs(reference[0][3] - h) > 0.15 * font_height
                or np.logical_and(candidate[1], reference[1]).sum()
                / np.logical_or(candidate[1], reference[1]).sum() < 0.8):
            return False
    return True
