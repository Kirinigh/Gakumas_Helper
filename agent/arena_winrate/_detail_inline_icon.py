"""Bounded same-frame evidence for a colored diamond between two text atoms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ._detail_text_layout import Box


def has_inline_icon_gap(image: Any, left: Box, right: Box) -> bool:
    """Prove a visible non-text separator; never infer a field or its value."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        return False
    x, y, w, h = left
    a, b, _, d = right
    scale = min(h, d)
    gap = a - x - w
    if scale <= 0 or not 0 < gap <= 2 * scale:
        return False
    pad = max(1, round(scale / 4))
    l, t = x + w - pad, min(y, b) - pad
    r, bottom = a + pad, max(y + h, b + d) + pad
    if l < 0 or t < 0 or r > image.shape[1] or bottom > image.shape[0]:
        return False
    patch = image[t:bottom, l:r].astype(np.int16)
    chroma = patch.max(axis=2) - patch.min(axis=2)
    # Keep the existing neutral/color split (24 levels), including the
    # icon's connected antialiased rim rather than only its saturated core.
    mask = ((chroma >= 24) & (patch.max(axis=2) >= 150)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= max(4, scale * scale / 8)]
    if len(components) != 1:
        return False
    index = components[0]
    cx, cy, width, height, area = map(int, stats[index])
    if (cx == 0 or cy == 0 or cx + width == mask.shape[1] or cy + height == mask.shape[0]
            or not .6 * scale <= width <= 1.5 * scale
            or not .6 * scale <= height <= 1.5 * scale
            or not .75 <= width / height <= 1.33
            or not .25 <= area / (width * height) <= .8
            or abs(t + cy + height / 2 - (y + h / 2 + b + d / 2) / 2) > .35 * scale):
        return False
    component = labels == index
    core_points = cv2.findNonZero((component & (chroma >= 48)).astype(np.uint8))
    if core_points is None:
        return False
    # The saturated core proves shape; its connected antialiased rim proves
    # pixel coverage. A colored shadow need not itself be a four-point shape.
    core_hull = cv2.convexHull(core_points)
    corners = cv2.approxPolyDP(core_hull, .04 * cv2.arcLength(core_hull, True), True)[:, 0, :]
    if len(corners) != 4:
        return False
    center_x, center_y = cx + (width - 1) / 2, cy + (height - 1) / 2
    if (abs(corners[corners[:, 1].argmin(), 0] - center_x) > width / 4
            or abs(corners[corners[:, 1].argmax(), 0] - center_x) > width / 4
            or abs(corners[corners[:, 0].argmin(), 1] - center_y) > height / 4
            or abs(corners[corners[:, 0].argmax(), 1] - center_y) > height / 4):
        return False
    covered = np.zeros(mask.shape, np.uint8)
    hull = cv2.convexHull(cv2.findNonZero(component.astype(np.uint8)))
    cv2.fillConvexPoly(covered, hull, 1)
    gap_pixels = np.zeros(mask.shape, bool)
    gap_pixels[:, x + w - l:a - l] = True
    # A number/operator in the gap cannot be hidden by a neighboring icon.
    return not np.any(gap_pixels & ~covered.astype(bool) & (patch.min(axis=2) < 198))
