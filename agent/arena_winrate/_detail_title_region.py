"""Failure-only geometry for a newly opened skill-card title, without OCR guessing."""

from collections.abc import Sequence

import cv2
import numpy as np


def isolated_title_region(
    image: np.ndarray,
    sources: Sequence[np.ndarray],
    card_box: tuple[int, int, int, int],
    items: Sequence[tuple[str, tuple[int, int, int, int]]],
) -> tuple[int, int, int, int] | None:
    """Require one white panel, new against frozen sources, on the same page.

    Pixel thresholds select geometry only; they never authorize a card identity.
    Unknown themes, incomplete borders, page motion and multiple panels abstain.
    """
    if (
        image.ndim != 3 or image.shape[2] != 3
        or len(sources) != 3 or any(source.shape != image.shape for source in sources)
    ):
        return None
    height, width = image.shape[:2]
    if width < 320 or height < 568:
        return None
    sx, sy, sw, sh = card_box
    if not (sw > 0 and sh > 0 and 0 <= sx < sx + sw <= width and 0 <= sy < sy + sh <= height):
        return None
    # White fill is connected around the title and effect glyphs. Other page
    # surfaces are grey; artwork islands must also pass rectangular edge tests.
    white = (image.min(axis=2) >= 250).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    panels = []
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        x, y, w, h = map(int, (x, y, w, h))
        if not (
            width * .40 <= w <= width * .65
            and height * .08 <= h <= height * .45
            and y + h <= height * .60 and area >= w * h * .62
        ):
            continue
        inset = max(3, round(width * .004))
        corner = max(8, round(width * .03))
        component = labels[y:y + h, x:x + w] == label
        if min(
            component[inset, corner:-corner].mean(),
            component[-inset - 1, corner:-corner].mean(),
            component[corner:-corner, inset].mean(),
            component[corner:-corner, -inset - 1].mean(),
        ) < .88:
            continue
        panels.append((x, y, w, h))
    # Do not select whichever of several panels happens to name a known card.
    if len(panels) != 1:
        return None
    x, y, w, h = panels[0]
    if not (x <= sx + sw / 2 <= x + w and y - sh <= sy + sh / 2 <= y + h + sh * 2):
        return None
    margin = max(5, round(width * .018))
    outside = np.zeros((height, width), dtype=bool)
    outside[:round(height * .48)] = True
    outside[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin] = False
    if outside.sum() < width * height * .10:
        return None
    votes = 0
    current = image.astype(np.int16)
    for source in sources:
        changed = np.max(np.abs(current - source.astype(np.int16)), axis=2) > 12
        if changed[y:y + h, x:x + w].mean() >= .12 and changed[outside].mean() <= .04:
            votes += 1
    if votes < 2:
        return None

    left, right = x + max(3, round(width * .006)), x + w - max(3, round(width * .012))
    top, bottom = y + max(3, round(height * .004)), y + round(height * .036)
    # Card effect lines below the header distinguish this panel from a blank
    # overlay. They only bound the crop; their text never supplies the title.
    body_rows = []
    for text, (bx, by, bw, bh) in items:
        if (
            left <= bx and bx + bw <= right and y + height * .026 <= by
            and by + bh <= y + h and bh > 0
            and any(token in text for token in ("消費", "スコア", "元気", "好印象", "集中", "強気", "全力", "スキルカード"))
        ):
            body_rows.append(by)
    if len(set(body_rows)) < 2:
        return None
    bottom = min(bottom, min(body_rows))
    if bottom - top < height * .018:
        return None
    # An existing OCR atom must intersect the header. It may span background
    # digits, but a title crop must never be inferred from an effect row alone.
    if not any(
        bx < right and bx + bw > left and by < bottom and by + bh > top
        for _, (bx, by, bw, bh) in items
    ):
        return None
    return left, top, right - left, bottom - top
