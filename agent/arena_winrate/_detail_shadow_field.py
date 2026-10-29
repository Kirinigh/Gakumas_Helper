"""Prove a local old source through an independently measured 2-D shadow.

The candidate never trains or selects the field. Two separated source-flat
groups must explain one another before their predictions can be averaged.
The caller retains raw ink contradictions and the ordinary source votes.
"""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]
_MIN_DONORS = 12
_RESIDUAL = 4
_STRONG_DELTA = 32
_SIGMAS = tuple(range(1, 7))
_OFFSETS = tuple((sigma, dx, dy) for sigma in _SIGMAS for dx in range(-1, 2) for dy in range(4))


def _box_valid(box: Box, width: int, height: int) -> bool:
    return (isinstance(box, tuple) and len(box) == 4
            and all(type(value) is int for value in box)
            and box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0
            and box[0] + box[2] <= width and box[1] + box[3] <= height)


def _one_side(box: Box, panel: Box) -> bool:
    x, y, w, h = box
    px, py, pw, ph = panel
    sides = (x + w > px + pw and py <= y and y + h <= py + ph,
             y + h > py + ph and px <= x and x + w <= px + pw,
             x < px and py <= y and y + h <= py + ph,
             y < py and px <= x and x + w <= px + pw)
    return sum(sides) == 1


def _complete_core(core: np.ndarray) -> bool:
    rows, columns = np.nonzero(core)
    if (not len(rows) or core.mean() < .75 or rows.min() != 0 or columns.min() != 0
            or rows.max() != core.shape[0] - 1 or columns.max() != core.shape[1] - 1):
        return False
    count, _ = cv2.connectedComponents(core.astype(np.uint8))
    if count != 2:
        return False
    hull = np.zeros(core.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(core.astype(np.uint8))), 1)
    # Rasterizing an eroded convex contour can leave a one-pixel staircase
    # outside its next convex hull. Validate its interior without repainting
    # that staircase or accepting a disconnected/concave interior.
    interior = cv2.erode(hull, np.ones((3, 3), np.uint8)).astype(bool)
    return not bool(np.any(interior & ~core))


def _passes(observed: np.ndarray, predicted: np.ndarray) -> bool:
    residual = np.abs(observed - predicted)
    # Every source-flat donor is a physical-field witness. A strong outlier
    # cannot be discarded or hidden inside the ordinary one-percent budget.
    return bool(residual.size and residual.mean() <= _RESIDUAL
                and np.max(residual) <= _STRONG_DELTA
                and (residual.max(axis=1) > _STRONG_DELTA).mean() <= .01)


def _fit_fields(ambient: np.ndarray, drop: np.ndarray, old: np.ndarray, current: np.ndarray):
    """Fit all fixed physical kernels using two-variable sufficient statistics.

    Amplitudes are nonnegative and sum to at most one. Include both axes and
    the sum-one boundary, so the constrained fit does not clip a wrong model.
    Model ordering breaks ties independently of all candidate observations.
    """
    a = np.repeat(ambient, 3, axis=1) * old.ravel()[None, :]
    d = np.repeat(drop, 3, axis=1) * old.ravel()[None, :]
    target = (old - current).ravel()
    aa = np.einsum("ij,ij->i", a, a)[:, None]
    dd = np.einsum("ij,ij->i", d, d)[None, :]
    ad = a @ d.T
    at = (a @ target)[:, None]
    dt = (d @ target)[None, :]
    determinant = aa * dd - ad * ad
    shape = ad.shape
    first = np.divide(at * dd - dt * ad, determinant,
                      out=np.zeros(shape), where=determinant > 1e-9)
    second = np.divide(dt * aa - at * ad, determinant,
                       out=np.zeros(shape), where=determinant > 1e-9)
    interior = (determinant > 1e-9) & (first >= 0) & (second >= 0) & (first + second <= 1)
    edge_a = np.broadcast_to(np.clip(at / np.maximum(aa, 1e-12), 0, 1), shape)
    edge_d = np.broadcast_to(np.clip(dt / np.maximum(dd, 1e-12), 0, 1), shape)
    difference = aa + dd - 2 * ad
    sum_one = np.clip(np.divide(at - dt - ad + dd, difference,
                                out=np.zeros(shape), where=difference > 1e-9), 0, 1)
    candidates = ((first, second, interior), (edge_a, np.zeros(shape), True),
                  (np.zeros(shape), edge_d, True), (sum_one, 1 - sum_one, True))
    best = None
    for ca, cd, valid in candidates:
        # The omitted target-square term is constant over all models.
        loss = ca * ca * aa + cd * cd * dd + 2 * ca * cd * ad - 2 * ca * at - 2 * cd * dt
        loss = np.where(valid, loss, np.inf)
        index = np.unravel_index(int(np.argmin(loss)), shape)
        value = float(loss[index])
        if best is None or value < best[0]:
            best = value, index[0], index[1], float(ca[index]), float(cd[index])
    return best[1:]


def _samples_supported(group, bins, xx, yy, panel: Box, margin: int, is_left: bool) -> bool:
    px, py, pw, ph = panel
    side = xx < px if is_left else xx >= px + pw
    return (all(np.count_nonzero(group & (bins == level)) >= _MIN_DONORS
                for level in range(3, margin + 3))
            and all(np.count_nonzero(group & region) >= _MIN_DONORS
                    for region in (side, yy < py, yy >= py + ph)))


def shadow_field_projection(
    image: np.ndarray, source: np.ndarray, box: Box, panel: Box, core: np.ndarray, *, exposed: np.ndarray,
) -> np.ndarray | None:
    """Return old RGB through a proved physical shadow, otherwise ``None``.

    Screen-space kernels use sigma 1..6, horizontal offset -1..1 and downward
    offset 0..3. No image rotation can turn a downward shadow into a sideways
    one. Only exposed pixels within the existing shadow margin are projected.
    All other pixels retain the supplied source exactly. The caller must have
    already established a unique panel and rejected raw stroke conflicts.
    """
    if (not isinstance(image, np.ndarray) or not isinstance(source, np.ndarray)
            or image.dtype != np.uint8 or source.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or source.shape != image.shape):
        return None
    height, width = image.shape[:2]
    if (not _box_valid(box, width, height) or not _box_valid(panel, width, height)
            or not _one_side(box, panel)
            or not isinstance(core, np.ndarray) or core.dtype != np.bool_
            or core.shape != (panel[3], panel[2]) or not _complete_core(core)
            or not isinstance(exposed, np.ndarray) or exposed.dtype != np.bool_
            or exposed.shape != (box[3], box[2]) or not exposed.any()):
        return None
    x, y, w, h = box
    px, py, pw, ph = panel
    margin = max(3, round(width * .01))
    support = margin + 3
    if px < support or py < support or px + pw + support > width or py + ph + support > height:
        return None  # A cropped edge cannot independently support a complete field.
    padding = max(30, support)
    left, top = max(0, px - padding), max(0, py - padding)
    right, bottom = min(width, px + pw + padding), min(height, py + ph + padding)
    if x < left or y < top or x + w > right or y + h > bottom:
        return None
    shape = bottom - top, right - left
    canvas = np.zeros(shape, np.uint8)
    canvas[py - top:py - top + ph, px - left:px - left + pw] = core
    distance = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    covered = cv2.dilate(canvas, np.ones((5, 5), np.uint8)).astype(bool)
    region = np.s_[y - top:y - top + h, x - left:x - left + w]
    if np.any(exposed & covered[region]):
        return None
    target = exposed & (distance[region] < margin)
    if not target.any():
        return None

    yy, xx = np.indices(shape)
    xx += left
    yy += top
    previous = source[top:bottom, left:right]
    current = image[top:bottom, left:right].astype(np.float64)
    low = cv2.erode(previous.min(axis=2), np.ones((3, 3), np.uint8))
    high = cv2.dilate(previous.max(axis=2), np.ones((3, 3), np.uint8))
    # The one-pixel exclusion also prevents the source qualification window
    # from touching candidate pixels. All selection precedes observed tests.
    outside = ~((xx >= x - 1) & (xx < x + w + 1) & (yy >= y - 1) & (yy < y + h + 1))
    qualified = ((distance >= 3) & (distance < margin + 3) & ~covered & outside
                 & (low >= 240) & (high.astype(np.int16) - low.astype(np.int16) <= _RESIDUAL))
    bins = np.floor(distance).astype(np.int16)
    gap = support
    split = px + pw // 2
    groups = (qualified & (xx < split - gap), qualified & (xx >= split + gap))
    if not all(_samples_supported(group, bins, xx, yy, panel, margin, side == 0)
               for side, group in enumerate(groups)):
        return None
    # Do not silently discard a black/red/bright contaminant after choosing
    # source-flat witnesses. Strong new darkness is incompatible with this
    # weak, near-edge field even if its area fits a global percentage budget.
    witnesses = qualified
    old = previous[witnesses].astype(np.float64)
    observed = current[witnesses]
    if (np.any(np.ptp(observed, axis=1) > _RESIDUAL)
            or np.any(observed - old > _RESIDUAL)
            or np.any(observed.min(axis=1) < 230 - _STRONG_DELTA)):
        return None

    surface = cv2.dilate(canvas, np.ones((3, 3), np.uint8)).astype(np.float32)
    fields = [cv2.GaussianBlur(surface, (0, 0), float(sigma), borderType=cv2.BORDER_CONSTANT)
              for sigma in _SIGMAS]
    rows, columns = np.nonzero(witnesses)
    ambient = np.stack([field[rows, columns] for field in fields]).astype(np.float64)
    drop = np.stack([fields[sigma - 1][rows - dy, columns - dx]
                     for sigma, dx, dy in _OFFSETS]).astype(np.float64)
    model_predictions = []
    candidate_old = source[y:y + h, x:x + w].astype(np.float32)
    for group in groups:
        selection = group[witnesses]
        a_index, d_index, ca, cd = _fit_fields(ambient[:, selection], drop[:, selection],
                                            old[selection], observed[selection])
        field_ratio = 1 - ca * ambient[a_index] - cd * drop[d_index]
        # Both training and held-out witnesses (including the center gap) must
        # be explained. No candidate outcome selects or repairs this model.
        predicted = old * field_ratio[:, None]
        if (not all(_passes(observed[group[witnesses]], predicted[group[witnesses]]) for group in groups)
                or not _passes(observed, predicted)):
            return None
        sigma, dx, dy = _OFFSETS[d_index]
        rows_c, columns_c = np.indices((h, w))
        rows_c += y - top
        columns_c += x - left
        ratio = (1 - ca * fields[a_index][rows_c, columns_c]
                 - cd * fields[sigma - 1][rows_c - dy, columns_c - dx])
        prediction = candidate_old.copy()
        prediction[target] *= ratio[target, None]
        model_predictions.append(prediction)
    agreement = np.abs(model_predictions[0] - model_predictions[1])[target]
    if agreement.mean() > _RESIDUAL or (agreement.max(axis=1) > _STRONG_DELTA).mean() > .01:
        return None
    return ((model_predictions[0] + model_predictions[1]) / 2).astype(np.float32)
