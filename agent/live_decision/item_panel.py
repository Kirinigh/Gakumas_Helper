"""Read existing HIF item overview and badge evidence using caller-supplied OCR."""

import re
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .ocr_evidence import _valid_ocr_result
from .status_rules import _normalize_status_text

if __package__ == "agent.live_decision":
    from agent.p_item_recognition.reference import PItemReferenceRuntime, measure_p_item_content_generation
else:
    from p_item_recognition.reference import PItemReferenceRuntime, measure_p_item_content_generation


def _inside(row, roi):
    x, y, w, h = roi
    rx, ry, rw, rh = row["box"]
    return rw > 0 and rh > 0 and x <= rx and y <= ry and rx + rw <= x + w and ry + rh <= y + h


def _unique_inventory_text(rows, text, roi):
    matches = [row for row in rows if row.score >= 0.85 and _inside({"box": row.box}, roi) and _normalize_status_text(row.text) == text]
    if len(matches) != 1:
        raise ValueError(f"item overview marker is missing or ambiguous: {text}")


@dataclass(frozen=True)
class ItemPanelRow:
    box: tuple[int, int, int, int]
    icon_box: tuple[int, int, int, int]
    name: str
    used_count: int | None
    use_limit: int | None
    limit_scope: str | None
    confidence: float

    @property
    def content_key(self):
        return self.name, self.used_count, self.use_limit, self.limit_scope


@dataclass(frozen=True)
class ItemPanelPage:
    rows: tuple[ItemPanelRow, ...]
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool
    partial_row_count: int

    @property
    def content_key(self):
        return tuple(row.content_key for row in self.rows), self.at_top, self.at_bottom


@dataclass(frozen=True)
class ItemRemainingCounter:
    value: int
    box: tuple[int, int, int, int]
    confidence: float


def _explicit_counter_candidates(crop, mask, x, y, w, h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx == 0 or by == 0 or bx + bw >= w or by + bh >= h:
            continue
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        if not (
            10 <= min(bw, bh) <= max(bw, bh) <= 64
            and 0.85 <= bw / bh <= 1.18
            and 0.60 <= area / (bw * bh) <= 0.86
            and hull_area > 0
            and perimeter > 0
            and area / hull_area >= 0.85
            and 4 * math.pi * hull_area / perimeter**2 >= 0.75
        ):
            continue
        # Item artwork can touch the circle's lower edge. Fit its outer hull
        # only for geometry; retain the original contour for visible glyphs,
        # so an obscured digit is never reconstructed from the fitted circle.
        interior = np.zeros((h, w), np.uint8)
        cv2.drawContours(interior, [contour], -1, 255, cv2.FILLED)
        glyph = (crop.min(axis=2) >= 220) & (interior > 0)
        yy, xx = np.where(glyph)
        if not len(xx) or len(xx) / (bw * bh) < 0.05:
            continue
        if xx.min() <= bx or xx.max() >= bx + bw - 1 or yy.min() <= by or yy.max() >= by + bh - 1:
            continue
        digit = 255 - glyph[by : by + bh, bx : bx + bw].astype(np.uint8) * 255
        digit = cv2.copyMakeBorder(digit, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
        candidates.append(((x + bx, y + by, bw, bh), cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)))
    return candidates


def read_explicit_item_counter(read_ocr, image, search_box, *, use_limit, client_size=None):
    """Read a visible grey-circle counter on the battle-item description page.

    D-068: this is the remaining count, including an explicit zero. Missing
    badges are unknown here; the HUD's exhausted appearance is a separate gate.
    """
    if type(use_limit) is not int or not 1 <= use_limit <= 99:
        raise ValueError("item counter requires an observed finite use limit")
    if getattr(image, "ndim", None) != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("item counter frame is invalid")
    if len(search_box) != 4 or any(type(v) is not int for v in search_box):
        raise ValueError("item counter search box is invalid")
    x, y, w, h = search_box
    if min(x, y) < 0 or min(w, h) < 1 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("item counter search box is clipped")
    crop = image[y : y + h, x : x + w]
    gray = ((np.ptp(crop, axis=2) <= 12) & (crop.min(axis=2) >= 90) & (crop.max(axis=2) <= 200)).astype(np.uint8) * 255
    # Prefer the actual separated circle. Closing a healthy badge can join
    # adjacent item artwork (1929). Only repair a gap when no original circle
    # exists; multiple original circles remain ambiguous.
    candidates = _explicit_counter_candidates(crop, gray, x, y, w, h)
    if not candidates:
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        candidates = _explicit_counter_candidates(crop, closed, x, y, w, h)
    if not candidates and client_size == (359, 638):
        # The compact hat art can touch the grey badge through tinted edge
        # pixels. Keep its neutral pixels and read the original white glyph.
        neutral = gray & (np.ptp(crop, axis=2) == 0).astype(np.uint8) * 255
        candidates = _explicit_counter_candidates(crop, neutral, x, y, w, h)
    if len(candidates) != 1:
        raise ValueError("explicit remaining-use badge is missing or ambiguous")
    box, digit = candidates[0]
    rows = read_ocr(digit, "LiveItemRemainingCounter", only_rec=True)
    if len(rows) != 1 or not _valid_ocr_result(rows[0]) or rows[0].score < 0.95 or not isinstance(getattr(rows[0], "text", None), str):
        raise ValueError("item remaining-use digit is unresolved")
    text = _normalize_status_text(rows[0].text)
    if not re.fullmatch(r"0|[1-9][0-9]?", text):
        raise ValueError("item remaining-use digit is invalid")
    value = int(text)
    if value > use_limit:
        raise ValueError("item remaining-use count exceeds its limit")
    return ItemRemainingCounter(value, box, rows[0].score)


def item_counter_consensus(observations):
    observations = tuple(observations)
    if any(
        not isinstance(o, ItemRemainingCounter)
        or type(o.value) is not int
        or not 0 <= o.value <= 99
        or type(o.confidence) not in (int, float)
        or not math.isfinite(o.confidence)
        or not 0.95 <= o.confidence <= 1
        or len(o.box) != 4
        or any(type(v) is not int for v in o.box)
        or min(o.box[:2]) < 0
        or min(o.box[2:]) < 1
        for o in observations
    ):
        raise ValueError("item remaining-use observation is invalid")
    if len(observations) != 3 or any(o.value != observations[0].value for o in observations[1:]):
        raise ValueError("item remaining-use count has no three-frame consensus")
    if any(max(abs(a - b) for a, b in zip(o.box, observations[0].box)) > 2 for o in observations[1:]):
        raise ValueError("item remaining-use badge has not settled")
    return observations[len(observations) // 2]


def item_panel_consensus(pages):
    pages = tuple(pages)
    if len(pages) != 3 or any(p.content_key != pages[0].content_key for p in pages[1:]):
        raise ValueError("item overview semantics have not settled")
    for page in pages[1:]:
        if max(abs(a - b) for a, b in zip(page.thumb, pages[0].thumb)) > 2 or any(
            max(abs(x - y) for x, y in zip(a.box, b.box)) > 2 for a, b in zip(page.rows, pages[0].rows)
        ):
            raise ValueError("item overview geometry has not settled")
    return pages[len(pages) // 2]


def item_panel_stable_geometry(images, *, expected_thumb_height=None):
    """Require both row geometry and the existing reference content gate.

    Rounded row coordinates can remain identical during subpixel scrolling.
    Such frames are unsuitable for item identity even when OCR text agrees.
    """
    if len(images) != 3:
        raise ValueError("item overview requires three frames")
    geometries = [item_panel_geometry(image, expected_thumb_height=expected_thumb_height) for image in images]
    first = geometries[0]
    if any(
        len(g[0]) != len(first[0])
        or g[2:] != first[2:]
        or max(abs(a - b) for a, b in zip(g[1], first[1])) > 1
        or any(max(abs(a - b) for a, b in zip(x, y)) > 1 for x, y in zip(g[0], first[0]))
        for g in geometries[1:]
    ):
        raise ValueError("item overview geometry has not settled")
    boxes = tuple((x + 14, y + 14, 142, 142) for x, y, _, _ in first[0])
    errors = measure_p_item_content_generation(images, boxes)
    if max(errors) > PItemReferenceRuntime().content_generation_max_mean_abs_error:
        raise ValueError("item overview rendered content has not settled")
    return first


def item_panel_geometry(image, *, expected_thumb_height=None):
    if getattr(image, "shape", ()) != (1280, 720, 3):
        raise ValueError("item overview resolution is uncalibrated")
    # Same white-row segmentation as the status list, with this overview's
    # larger icon area and viewport. Clipped rows never become observations.
    mask = (image[100:1007, 20:700].min(axis=2) >= 235).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = sorted(
        (
            (int(x + 20), int(y + 100), int(w), int(h))
            for x, y, w, h, area in stats[1:]
            if 665 <= w <= 680 and h >= 40 and area / (w * h) >= 0.65
        ),
        key=lambda b: b[1],
    )
    # The upper viewport has a 20-pixel fade: a clipped row can begin as a
    # separate bright component inside it. Only rows below the fade have a
    # visible top edge (the unscrolled first row begins at y=123).
    complete = tuple(b for b in boxes if b[1] > 120 and b[1] + b[3] < 995 and b[3] >= 165)
    if not complete:
        raise ValueError("item overview has no complete row")
    if any(not 7 <= b[1] - a[1] - a[3] <= 20 for a, b in zip(boxes, boxes[1:])):
        raise ValueError("item overview row spacing is inconsistent")
    strip = image[115:1001, 703].astype(np.int16)
    eligible = strip[(np.ptp(strip, axis=1) <= 32) & (strip.min(axis=1) >= 90) & (strip.max(axis=1) <= 180)]
    colors, counts = np.unique(eligible, axis=0, return_counts=True)
    spans = []
    for color, count in zip(colors, counts):
        if count < 30:
            continue
        solid = np.max(np.abs(image[115:1001, 703:705].astype(np.int16) - color), axis=(1, 2)) <= 2
        edges = np.diff(np.r_[False, solid, False].astype(np.int8))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            span = (int(a + 115), int(b + 115))
            if b - a >= 30 and not any(max(abs(x - y) for x, y in zip(span, old)) <= 3 for old in spans):
                spans.append(span)
    if expected_thumb_height is not None:
        if type(expected_thumb_height) is not int or not 30 <= expected_thumb_height <= 886:
            raise ValueError("item overview observed scrollbar height is invalid")
        # A scroll-only transaction keeps the scrollbar thumb's size. The
        # already observed first page distinguishes it from a short uniform
        # background strip without choosing the longest or darkest candidate.
        spans = [span for span in spans if abs(span[1] - span[0] - expected_thumb_height) <= 2]
    if len(spans) != 1:
        raise ValueError("item overview scrollbar is not unique")
    top, bottom = spans[0]
    # The solid thumb interior ends at 992 on the actual bottom page; the
    # remaining viewport pixels are footer padding and the rounded end cap.
    return complete, (top, bottom), 115 <= top <= 120, 990 <= bottom <= 994, len(boxes) - len(complete)


def item_limit_text_image(source):
    """Exclude pale bullet stars and normalize the actual dark-text extent."""
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8 or not source.size:
        raise ValueError("item limit crop is invalid")
    ink = source.min(axis=2) < 120
    yy, xx = np.where(ink)
    if not len(xx):
        raise ValueError("item limit has no visible text")
    glyphs = 255 - ink[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].astype(np.uint8) * 255
    glyphs = cv2.copyMakeBorder(glyphs, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(glyphs, cv2.COLOR_GRAY2BGR)


def _item_limit_semantics(text, *, repeated_delimiters=False):
    opening, closing = (r"\(+", r"\)+") if repeated_delimiters else (r"\(", r"\)")
    match = re.fullmatch(opening + r"(プロデュース中|試験・ステージ内)(\d{1,2})回" + closing, text)
    return (match[1], int(match[2])) if match else None


def _confirmed_item_text(read_ocr, image, row, *, usage_limit=False, client_size=None):
    if type(row["score"]) not in {int, float} or not math.isfinite(row["score"]):
        raise ValueError("item text confidence is invalid")
    if row["score"] >= 0.95:
        return row["text"], row["score"]
    if row["score"] < 0.85:
        raise ValueError("item text confidence is insufficient")
    x, y, w, h = row["box"]
    source = image[y : y + h, x : x + w]
    if usage_limit:
        if client_size == (359, 638) and _item_limit_semantics(row["text"]) is not None:
            # Binarization erases thin strokes at this scale. The original
            # line can independently confirm the exact stage limit and count.
            found = read_ocr(source, "LiveInventoryItemText", only_rec=True)
            if (
                len(found) == 1
                and _valid_ocr_result(found[0])
                and found[0].score >= 0.95
                and isinstance(getattr(found[0], "text", None), str)
                and _normalize_status_text(found[0].text) == row["text"]
            ):
                return row["text"], found[0].score
        crop = item_limit_text_image(source)
    else:
        crop = cv2.copyMakeBorder(source, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    found = read_ocr(crop, "LiveInventoryItemText", only_rec=True)
    if len(found) != 1 or not _valid_ocr_result(found[0]) or found[0].score < 0.95 or not isinstance(getattr(found[0], "text", None), str):
        raise ValueError("item text reread is unresolved")
    text = _normalize_status_text(found[0].text)
    if text != row["text"]:
        # A duplicated outer parenthesis still has to agree on scope and count.
        canonical = _item_limit_semantics(text) if usage_limit else None
        original = _item_limit_semantics(row["text"], repeated_delimiters=True) if usage_limit else None
        if canonical is None or canonical != original:
            raise ValueError("item text reread changed its meaning")
    return text, found[0].score


def read_item_panel(read_ocr, image, *, expected_thumb_height=None, client_size=None):
    """Read an existing overview using the supplied OCR function."""
    boxes, thumb, at_top, at_bottom, partial = item_panel_geometry(image, expected_thumb_height=expected_thumb_height)
    rows = tuple(read_ocr(image, "LiveInventoryItemsOverview"))
    if any(not _valid_ocr_result(row) or not isinstance(getattr(row, "text", None), str) for row in rows):
        raise ValueError("item overview OCR evidence is invalid")
    _unique_inventory_text(rows, "所持Pアイテム", (15, 25, 285, 65))
    _unique_inventory_text(rows, "閉じる", (210, 1120, 290, 85))
    rows = [dict(text=_normalize_status_text(r.text), score=r.score, box=tuple(r.box)) for r in rows]

    observed = []
    for box in boxes:
        x, y, w, h = box
        headers = [r for r in rows if _inside(r, (180, y + 3, 485, 42))]
        if len(headers) != 1:
            raise ValueError("item name is missing or ambiguous")
        name, confidence = _confirmed_item_text(read_ocr, image, headers[0])
        uses = [r for r in rows if _inside(r, (30, y + 155, 155, max(h - 155, 1))) and "回発動済み" in r["text"]]
        if len(uses) > 1:
            raise ValueError("item used counter is ambiguous")
        used = None
        if uses:
            value, score = _confirmed_item_text(read_ocr, image, uses[0])
            confidence = min(confidence, score)
            match = re.fullmatch(r"(\d{1,2})回発動済み", value)
            if not match:
                raise ValueError("item used counter is incomplete")
            used = int(match[1])
        limits = [
            r for r in rows if _inside(r, (180, y + 45, 490, h - 48)) and ("プロデュース中" in r["text"] or "試験・ステージ内" in r["text"])
        ]
        if len(limits) > 1:
            raise ValueError("item usage scopes are ambiguous")
        limit, scope = None, None
        if limits:
            value, score = _confirmed_item_text(read_ocr, image, limits[0], usage_limit=True, client_size=client_size)
            confidence = min(confidence, score)
            match = re.fullmatch(r"\((プロデュース中|試験・ステージ内)(\d{1,2})回\)", value)
            if not match:
                raise ValueError("item usage limit is incomplete")
            scope, limit = match[1], int(match[2])
        if used is not None and limit is not None and used > limit:
            raise ValueError("item used count exceeds its limit")
        observed.append(ItemPanelRow(box, (x + 14, y + 14, 142, 142), name, used, limit, scope, confidence))
    return ItemPanelPage(tuple(observed), thumb, at_top, at_bottom, partial)
