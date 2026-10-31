"""Independently witnessed horizontal shadow symmetry at the top panel rim."""

from __future__ import annotations

import cv2
import numpy as np


def mirror_confirms_top_weak_ink(image, source, box, panel, core, weak, field_values) -> bool:
    """Use held-out mirror neighborhoods only after the physical field passes.

    Candidate observations select neither witnesses nor predictions. Two
    separate top/bottom bands must validate reflection, and two disjoint
    neighbor groups must each predict a held-out reference and the target.
    """
    x, y, _, _ = box
    px, py, pw, ph = panel
    rows, columns = np.nonzero(weak)
    ys, xs = rows + y, columns + x
    if (not len(xs) or np.any(ys > py) or not isinstance(field_values, np.ndarray)
            or field_values.shape != (len(xs), 3) or not np.isfinite(field_values).all()):
        return False
    margin = max(3, round(image.shape[1] * .01))
    pad = margin + 3
    left, right = max(0, px - pad), min(image.shape[1], px + pw + pad)
    top, bottom = max(0, py - pad), min(image.shape[0], py + ph + pad)
    canvas = np.zeros((bottom - top, right - left), np.uint8)
    canvas[py - top:py - top + ph, px - left:px - left + pw] = core
    distance = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    old = source[top:bottom, left:right]
    current = image[top:bottom, left:right]
    low = cv2.erode(old.min(axis=2), np.ones((3, 3), np.uint8))
    high = cv2.dilate(old.max(axis=2), np.ones((3, 3), np.uint8))
    flat = (low >= 240) & (high.astype(np.int16) - low.astype(np.int16) <= 4)
    yy, xx = np.indices(canvas.shape)
    xx += left
    yy += top
    mirror = 2 * px + pw - 1 - xx
    valid = (mirror > left) & (mirror < right - 1)
    mi = np.clip(mirror - left, 0, right - left - 1)
    ry = yy - top
    ex, ey, ew, eh = box
    excluded = (xx >= ex - 1) & (xx <= ex + ew) & (yy >= ey - 1) & (yy <= ey + eh)
    pairs = (valid & (xx < px + pw / 2) & flat & flat[ry, mi]
             & (distance >= 3) & (distance < pad) & (distance[ry, mi] >= 3) & (distance[ry, mi] < pad)
             & ~excluded & ~excluded[ry, mi])
    for band in (yy < py, yy >= py + ph):
        selected = pairs & band
        if np.count_nonzero(selected) < 12:
            return False
        a, b = old[selected].astype(float), old[ry, mi][selected].astype(float)
        av, bv = current[selected].astype(float), current[ry, mi][selected].astype(float)
        if (np.any(np.ptp(av, axis=1) > 4) or np.any(np.ptp(bv, axis=1) > 4)
                or np.any(av - a > 4) or np.any(bv - b > 4)
                or np.any(av < 198) or np.any(bv < 198)
                or np.any(np.abs(av - a * bv / b) > 4)
                or np.any(np.abs(bv - b * av / a) > 4)):
            return False

    groups = tuple((radius, offsets) for radius in (1, 2) for offsets in (
        ((-radius, 0), (radius, 0), (0, -radius), (0, radius)),
        ((-radius, -radius), (radius, -radius), (-radius, radius), (radius, radius))))
    for sample_index, (tx, ty) in enumerate(zip(xs, ys, strict=True)):
        mx = 2 * px + pw - 1 - int(tx)
        if not left + 3 <= mx < right - 3 or not top + 3 <= ty < bottom - 3:
            return False
        if (excluded[int(ty) - top, mx - left] or not flat[int(ty) - top, mx - left]
                or distance[int(ty) - top, mx - left] <= 3):
            return False
        reference_old = source[ty, mx].astype(float)
        reference = image[ty, mx].astype(float)
        if np.ptp(reference) > 4 or np.any(reference < 198) or np.any(reference - reference_old > 4):
            return False
        predictions = []
        admitted_radii = set()
        for radius, group in groups:
            coords = tuple((mx + dx, int(ty) + dy) for dx, dy in group)
            # Admission uses only geometry and the old source. An observed
            # disagreement can never select a more convenient neighbor group.
            if any(excluded[by - top, ax - left] or not flat[by - top, ax - left]
                   or distance[by - top, ax - left] <= 2 for ax, by in coords):
                continue
            admitted_radii.add(radius)
            a = np.array([source[by, ax] for ax, by in coords], dtype=float)
            av = np.array([image[by, ax] for ax, by in coords], dtype=float)
            if np.any(np.ptp(av, axis=1) > 4) or np.any(av - a > 4) or np.any(av < 198):
                return False
            ratio = np.mean(av / a, axis=0)
            if np.any(np.abs(reference - reference_old * ratio) > 4):
                return False
            predicted = source[ty, tx].astype(float) * ratio
            if np.any(np.abs(image[ty, tx].astype(float) - predicted) > 4):
                return False
            predictions.append(predicted)
        if admitted_radii != {1, 2}:
            return False
        values = np.asarray(predictions)
        if (np.any(np.ptp(values, axis=0) > 4)
                or np.any(np.abs(values.mean(axis=0) - field_values[sample_index]) > 4)):
            return False
    return True
