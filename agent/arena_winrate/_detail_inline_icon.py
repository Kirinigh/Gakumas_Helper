"""Bounded same-frame evidence for a colored diamond between two text atoms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ._detail_text_layout import Box


def has_inline_icon_gap(image: Any, left: Box, right: Box) -> bool:
    """Prove a visible non-text separator; never infer a field or its value."""
    if not _is_bgr_image(image):
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
    center_y = (y + h / 2 + b + d / 2) / 2 - t
    icon = _filled_diamond(patch, scale, center_y)
    if icon is None:
        return False
    _, covered = icon
    gap_pixels = np.zeros(covered.shape, bool)
    gap_pixels[:, x + w - l:a - l] = True
    # A number/operator in the gap cannot be hidden by a neighboring icon.
    return not np.any(gap_pixels & ~covered & (patch.min(axis=2) < 198))


def has_repeated_inline_icon_atom(image: Any, atom: Box, field: Box, reference: Box) -> bool:
    """Prove an OCR box belongs to the same visible icon as a second field."""
    if not _is_bgr_image(image):
        return False
    x, y, w, h = atom
    fx, fy, _, fh = field
    rx, ry, _, rh = reference
    if not (0 < fh and .8 * fh <= rh <= 1.25 * fh and rh < fy - ry <= 4 * fh
            and fx - 2 * fh <= x < fx and x + w <= fx + fh / 8
            and .8 * fh <= w <= 1.5 * fh and .8 * fh <= h <= 1.5 * fh):
        return False
    pad = max(1, round(fh / 8))
    earlier = (x + rx - fx - pad, y + ry - fy, w + 2 * pad, h)
    first = _inline_atom_shape(image, atom, fy + fh / 2, fh)
    if first is None:
        return False
    second = _inline_atom_shape(image, earlier, ry + rh / 2, rh)
    return second is not None and _same_icon_core_pixels(first, second)


def _is_bgr_image(image: Any) -> bool:
    return (isinstance(image, np.ndarray) and image.dtype == np.uint8
            and image.ndim == 3 and image.shape[2] == 3)


def _inline_atom_shape(image, box, center_y, scale):
    x, y, width, height = box
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        return None
    patch = image[y:y + height, x:x + width].astype(np.int16)
    icon = _filled_diamond(patch, scale, center_y - y)
    if icon is None:
        return None
    (cx, cy, w, h), covered = icon
    if abs(cx + w / 2 - width / 2) > .25 * scale:
        return None
    darkest = patch.min(axis=2)
    chroma = patch.max(axis=2) - darkest
    dark = darkest < 198
    # An attached number must leave its ink either beside the icon or in its
    # interior. Only faint neutral rims above/below the entire icon are allowed.
    outside = dark & ~covered
    if (np.any(outside[cy:cy + h])
            or np.any(outside & ((darkest < 180) | (chroma >= 24)))
            or np.any(dark & (chroma < 24) & covered)):
        return None
    return (cx, cy, w, h), patch, covered


def _same_icon_core_pixels(first, second) -> bool:
    (x, y, w, h), patch, covered = first
    (a, b, rw, rh), reference, _ = second
    if abs(w - rw) > 1 or abs(h - rh) > 1:
        return False
    # Ignore the one-pixel antialiased rim, but compare the pictogram as well
    # as the colored interior: the diamond alone cannot excuse overlaid ink.
    inside = cv2.erode(covered.astype(np.uint8), np.ones((3, 3), np.uint8))[y:y + h, x:x + w] != 0
    if not np.any(inside):
        return False
    pixels = patch[y:y + h, x:x + w][inside]
    for top in range(b - 1, b + 2):
        for left in range(a - 1, a + 2):
            if not (0 <= top <= reference.shape[0] - h and 0 <= left <= reference.shape[1] - w):
                continue
            match = reference[top:top + h, left:left + w][inside]
            if np.all(np.abs(pixels - match) <= 32):
                return True
    return False


def _filled_diamond(patch, scale, center_y):
    chroma = patch.max(axis=2) - patch.min(axis=2)
    # Keep the existing neutral/color split (24 levels), including the
    # icon's connected antialiased rim rather than only its saturated core.
    mask = ((chroma >= 24) & (patch.max(axis=2) >= 150)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= max(4, scale * scale / 8)]
    if len(components) != 1:
        return None
    index = components[0]
    cx, cy, width, height, area = map(int, stats[index])
    if (cx == 0 or cy == 0 or cx + width == mask.shape[1] or cy + height == mask.shape[0]
            or not .6 * scale <= width <= 1.5 * scale
            or not .6 * scale <= height <= 1.5 * scale
            or not .75 <= width / height <= 1.33
            or not .25 <= area / (width * height) <= .8
            or abs(cy + height / 2 - center_y) > .35 * scale):
        return None
    component = labels == index
    core_points = cv2.findNonZero((component & (chroma >= 48)).astype(np.uint8))
    if core_points is None:
        return None
    # The saturated core proves shape; its connected antialiased rim proves
    # pixel coverage. A colored shadow need not itself be a four-point shape.
    core_hull = cv2.convexHull(core_points)
    corners = cv2.approxPolyDP(core_hull, .04 * cv2.arcLength(core_hull, True), True)[:, 0, :]
    if len(corners) != 4:
        return None
    center_x, center_y = cx + (width - 1) / 2, cy + (height - 1) / 2
    if (abs(corners[corners[:, 1].argmin(), 0] - center_x) > width / 4
            or abs(corners[corners[:, 1].argmax(), 0] - center_x) > width / 4
            or abs(corners[corners[:, 0].argmin(), 1] - center_y) > height / 4
            or abs(corners[corners[:, 0].argmax(), 1] - center_y) > height / 4):
        return None
    covered = np.zeros(mask.shape, np.uint8)
    hull = cv2.convexHull(cv2.findNonZero(component.astype(np.uint8)))
    cv2.fillConvexPoly(covered, hull, 1)
    return (cx, cy, width, height), covered.astype(bool)
