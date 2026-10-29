"""Attribute one skill tooltip before any spatial or numeric text assembly."""

import re
import unicodedata
from typing import Any
from itertools import combinations
from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np

from ._reader_visual import _box, _text, _spatial_ocr_rows
from ._detail_shadow_field import shadow_field_projection, shadow_field_confirms_weak_ink
from ._detail_title_region import _white_detail_panels, detail_title_in_header
from ._detail_shadow_source import shadow_source_projection
from ._decoration_pixel_evidence import has_four_point_decoration_consensus

Box = tuple[int, int, int, int]
_EFFECT = re.compile(r"スコア|元気|集中|好印象|好調|やる気|全力|体力消費|ターン追加|スキルカード|消費|増加|減少|不可|ない|無効|以外|不|無|非|未|禁|除")
_OPEN_OPERAND = re.compile(r"(?:[+の]|消費|追加|値増加|コスト|スコア|コア|元気|体力|集中|やる気|好印象|好調|全力値)$")


@dataclass(frozen=True, slots=True)
class SkillDetailDomain:
    panel: Box | None
    title: Box | None
    owned: tuple[Any, ...] = ()
    background: tuple[Any, ...] = ()
    unknown: tuple[Any, ...] = ()
    reason: str = ""
    blocked: bool = False

    @property
    def complete(self) -> bool:
        return self.panel is not None and not self.unknown and not self.reason


class SkillDetailText(str):
    """A semantic input carries its attribution state, even when text is empty."""

    def __new__(cls, text: str, domain: SkillDetailDomain):
        value = super().__new__(cls, text)
        value.domain = domain
        return value


def _contains(outer: Box, inner: Box) -> bool:
    x, y, w, h = outer
    a, b, c, d = inner
    return c > 0 and d > 0 and x <= a and y <= b and a + c <= x + w and b + d <= y + h


def _new_panel(image: np.ndarray, sources: Sequence[np.ndarray], panel: Box) -> bool:
    """The existing 12/.12/.04 two-source gate, without int16 frame copies."""
    ih, iw = image.shape[:2]
    x, y, w, h = panel
    margin = max(5, round(iw * .018))
    outside = np.zeros((ih, iw), np.uint8)
    outside[:round(ih * .48)] = 255
    outside[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin] = 0
    count = cv2.countNonZero(outside)
    if count < iw * ih * .1:
        return False
    votes = 0
    for source in sources:
        changed = cv2.bitwise_not(cv2.inRange(cv2.absdiff(image, source), (0, 0, 0), (12, 12, 12)))
        if (cv2.countNonZero(changed[y:y + h, x:x + w]) >= w * h * .12
                and cv2.countNonZero(cv2.bitwise_and(changed, outside)) <= count * .04):
            votes += 1
    return votes >= 2


def _old_pixels(image: np.ndarray, sources: Sequence[np.ndarray], box: Box, panel: Box) -> bool:
    x, y, w, h = box
    px, py, pw, ph = panel
    # Border/shadow pixels are newly covered, not evidence that the old ink
    # outside a rounded panel changed. Require substantial exposed source ink.
    margin = max(3, round(image.shape[1] * .01))
    mask = np.ones((h, w), bool)
    l, t = max(0, px - margin - x), max(0, py - margin - y)
    r, b = min(w, px + pw + margin - x), min(h, py + ph + margin - y)
    if l < r and t < b:
        mask[t:b, l:r] = False
    if mask.sum() < w * h * .25:
        return False
    current = image[y:y + h, x:x + w].astype(np.int16)
    votes = 0
    for source in sources:
        delta = np.abs(current - source[y:y + h, x:x + w].astype(np.int16))[mask]
        if delta.mean() <= 4 and (delta.max(axis=1) > 32).mean() <= .01:
            votes += 1
    return votes >= 2


def _panel_core(image: np.ndarray, panel: Box) -> np.ndarray:
    """Reconstruct the convex rounded surface, without text holes or shadow."""
    x, y, w, h = panel
    white = cv2.inRange(image[y:y + h, x:x + w], (250, 250, 250), (255, 255, 255))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    component = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    core = np.zeros_like(component)
    cv2.fillConvexPoly(core, cv2.convexHull(cv2.findNonZero(component)), 1)
    # One pixel of edge antialiasing is not foreground text. Genuine crossing
    # operands are still retained through the incomplete-operand gate below.
    return cv2.erode(core, np.ones((3, 3), np.uint8)).astype(bool)


def _surface_coverage(core: np.ndarray, panel: Box, box: Box, radius: int) -> np.ndarray:
    """Map the existing surface dilation without clipping at its crop edge."""
    px, py, pw, ph = panel
    x, y, w, h = box
    covered = np.zeros((h, w), bool)
    surface = cv2.dilate(
        np.pad(core.astype(np.uint8), radius), np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8),
    ).astype(bool)
    l, t = max(x, px - radius), max(y, py - radius)
    r, b = min(x + w, px + pw + radius), min(y + h, py + ph + radius)
    if l < r and t < b:
        covered[t - y:b - y, l - x:r - x] = surface[
            t - py + radius:b - py + radius, l - px + radius:r - px + radius,
        ]
    return covered


def _old_edge_pixels(
    image: np.ndarray, sources: Sequence[np.ndarray], box: Box, panel: Box, core: np.ndarray,
) -> bool:
    """Prove nearby source ink when the ordinary shadow mask has no room.

    This does not relax an observed pixel contradiction. Only a box near
    the panel with insufficient ordinary exposed area can use this proof.
    New ink in a rounded corner and changes to exposed strokes still veto it.
    """
    x, y, w, h = box
    px, py, pw, ph = panel
    left, top, right, bottom = max(x, px), max(y, py), min(x + w, px + pw), min(y + h, py + ph)
    margin = max(3, round(image.shape[1] * .01))
    shadow_width = max(0, min(x + w, px + pw + margin) - max(x, px - margin))
    shadow_height = max(0, min(y + h, py + ph + margin) - max(y, py - margin))
    if w * h - shadow_width * shadow_height >= w * h * .25:
        return False
    covered = _surface_coverage(core, panel, box, 2)
    exposed = ~covered
    if exposed.sum() < w * h * .25:
        return False
    local_core = np.zeros((h, w), bool)
    if left < right and top < bottom:
        local_core[top - y:bottom - y, left - x:right - x] = core[top - py:bottom - py, left - px:right - px]
    # The existing radius also covers paint outside the contour's crop bbox.
    current = image[y:y + h, x:x + w].astype(np.int16)
    current_dark = current.min(axis=2) < 230
    if np.any(current_dark & local_core):
        return False
    votes = 0
    for source in sources:
        previous = source[y:y + h, x:x + w].astype(np.int16)
        delta = np.abs(current - previous)
        visible_delta = delta[exposed]
        previous_dark = previous.min(axis=2) < 230
        current_strong = current.min(axis=2) < 230 - 32
        previous_strong = previous.min(axis=2) < 230 - 32
        # A blank tail cannot explain a character. Strong exposed ink must
        # remain at the same pixels; the aggregate 1% allowance is not a
        # licence to add/remove one stroke in a small glyph.
        if (not np.any(exposed & current_strong & previous_dark)
                or np.any(exposed & (current_strong | previous_strong) & (delta.max(axis=2) > 32))):
            continue
        # White panel coverage can brighten old ink inside its rectangle.
        # A new dark stroke outside the actual surface has no such explanation,
        # even if an unchanged old right-hand tail passes the exposed-area gate.
        if np.any(~local_core & current_dark & ((previous - current).max(axis=2) > 32)):
            continue
        if np.any(~covered & previous_dark & ((current - previous).max(axis=2) > 32)):
            continue
        if visible_delta.mean() > 4 or (visible_delta.max(axis=1) > 32).mean() > .01:
            projected = shadow_source_projection(image, source, box, panel, core, exposed=exposed)
            if projected is None:
                projected = shadow_field_projection(image, source, box, panel, core, exposed=exposed)
            if projected is None:
                continue
            residual = np.abs(current - projected)[exposed]
            if residual.mean() > 4 or (residual.max(axis=1) > 32).mean() > .01:
                continue
        votes += 1
    return votes >= 2


def _white_cover_pixels(current: np.ndarray, previous: np.ndarray) -> bool:
    """Prove a neutral white layer over old colored edge pixels.

    Preserve the existing all-channel brightening proof first. For colored
    sources, fit only the source's lowest/highest channels in linear sRGB;
    its remaining channel independently checks the neutral mixing model.
    """
    if current.shape != previous.shape or current.size == 0 or current.shape[-1] != 3:
        return False
    current = current.reshape(-1, 3).astype(np.float64)
    previous = previous.reshape(-1, 3).astype(np.float64)
    gain = current - previous
    fallback = np.any(gain <= 32, axis=1)
    if not np.any(fallback):
        return True
    observed, source, gain = current[fallback], previous[fallback], gain[fallback]
    if (np.any(gain < -4) or np.any(gain.max(axis=1) <= 32)
            or np.any(np.ptp(source, axis=1) <= 32)):
        return False

    def linear_srgb(rgb: np.ndarray) -> np.ndarray:
        scaled = rgb / 255
        return np.where(scaled <= .04045, scaled / 12.92, ((scaled + .055) / 1.055) ** 2.4)

    low, high = source.argmin(axis=1), source.argmax(axis=1)
    rows = np.arange(len(source))
    source_linear, observed_linear = linear_srgb(source), linear_srgb(observed)
    slope = ((observed_linear[rows, high] - observed_linear[rows, low])
             / (source_linear[rows, high] - source_linear[rows, low]))
    intercept = observed_linear[rows, low] - slope * source_linear[rows, low]
    # Quantized endpoint values may just cross a physical boundary. Clip the
    # endpoint-derived model, then require the original four-level tolerance
    # at every channel; the middle channel never chooses/fits the model.
    intercept = np.clip(intercept, 0, 1)
    slope = np.clip(slope, 0, 1 - intercept)
    prediction = intercept[:, None] + slope[:, None] * source_linear
    predicted_rgb = 255 * np.where(
        prediction <= .0031308, prediction * 12.92,
        1.055 * prediction ** (1 / 2.4) - .055,
    )
    return bool(np.all(np.abs(predicted_rgb - observed) <= 4))


def _old_border_pixels(
    image: np.ndarray, sources: Sequence[np.ndarray], box: Box, panel: Box, core: np.ndarray,
) -> bool:
    """Explain white-surface antialiasing over an otherwise intact old label.

    The convex core can reach the crop's outermost pixel. That pixel is not
    necessarily text: a white tooltip blends with old page ink at its edge.
    Only a weak, source-backed rim with three white pixels immediately inward
    can use this proof. Interior ink and new/darkened strokes remain unknown.
    The caller excludes operands and effect clauses independently of pixels.
    """
    x, y, w, h = box
    px, py, pw, ph = panel
    l, t, r, b = max(x, px), max(y, py), min(x + w, px + pw), min(y + h, py + ph)
    if l >= r or t >= b or min(pw, ph) < 5:
        return False
    current = image[y:y + h, x:x + w].astype(np.int16)
    local_core = np.zeros((h, w), bool)
    local_core[t - y:b - y, l - x:r - x] = core[t - py:b - py, l - px:r - px]
    ink = (current.min(axis=2) < 230) & local_core
    rows, cols = np.nonzero(ink)
    if not len(rows) or np.any(current.min(axis=2)[ink] < 230 - 32):
        return False
    global_x, global_y = cols + x, rows + y
    left, right = global_x == px, global_x == px + pw - 1
    top, bottom = global_y == py, global_y == py + ph - 1
    if not np.all(left | right | top | bottom):
        return False
    # Requiring the whole adjacent strip also excludes a real glyph that
    # starts at the rim and continues into the tooltip's readable surface.
    dx, dy = left.astype(int) - right.astype(int), top.astype(int) - bottom.astype(int)
    for step in (1, 2, 3):
        inward_x, inward_y = global_x + dx * step, global_y + dy * step
        if (not np.all(core[inward_y - py, inward_x - px])
                or np.any(image[inward_y, inward_x].min(axis=1) < 250)):
            return False

    # Keep the normal exposed-area contract, with an actual old glyph at the
    # same positions. A blank external tail cannot authenticate a new label.
    margin = max(3, round(image.shape[1] * .01))
    exposed = np.ones((h, w), bool)
    a, c = max(0, px - margin - x), min(w, px + pw + margin - x)
    d, e = max(0, py - margin - y), min(h, py + ph + margin - y)
    if a < c and d < e:
        exposed[d:e, a:c] = False
    if exposed.sum() < w * h * .25:
        return False
    current_dark = current.min(axis=2) < 230
    current_strong = current.min(axis=2) < 230 - 32
    votes = 0
    for source in sources:
        previous = source[y:y + h, x:x + w].astype(np.int16)
        delta = np.abs(current - previous)
        visible_delta = delta[exposed]
        previous_strong = previous.min(axis=2) < 230 - 32
        if (visible_delta.mean() > 4 or (visible_delta.max(axis=1) > 32).mean() > .01
                or not np.any(exposed & current_strong & previous_strong)
                or np.any(exposed & (current_strong | previous_strong) & (delta.max(axis=2) > 32))
                or not np.all(previous_strong[ink])
                or not _white_cover_pixels(current[ink], previous[ink])
                or np.any(~local_core & current_dark & ((previous - current).max(axis=2) > 32))):
            continue
        votes += 1
    return votes >= 2


def _body_marker_is_decoration(item: Any, body: Sequence[Any], image: np.ndarray | None) -> bool:
    """A separately proved bullet is not an unfinished arithmetic operator."""
    if image is None or _text(item).strip() not in {"+", "＋"}:
        return False
    x, y, w, h = _box(item)
    center_y = y + h / 2
    anchors = [_box(other) for other in body
               if sum(char.isalpha() for char in _text(other)) >= 2
               and _box(other)[0] > x + w + max(h, _box(other)[3]) * .5
               and abs(_box(other)[1] + _box(other)[3] / 2 - center_y) >= .7 * max(h, _box(other)[3])]
    nearest = sorted(anchors, key=lambda box: abs(box[1] + box[3] / 2 - center_y))[:4]
    return any(has_four_point_decoration_consensus(image, (x, y, w, h), pair)
               for pair in combinations(nearest, 2))


def _unfinished_body_operand(
    items: Sequence[Any], panel: Box, title: Box, *, image: np.ndarray | None = None,
) -> bool:
    """Keep open operands without treating an icon-split linker as a row end."""
    body = [item for item in items if _contains(panel, _box(item))
            and _box(item)[1] >= title[1] + title[3]]
    # Use a structural view only: a proved bullet must not stand between a
    # wrapped label and its value. Keep every original atom in the domain.
    structural = [item for item in body if not _body_marker_is_decoration(item, body, image)]
    opened = [item for item in structural if _OPEN_OPERAND.search(_text(item).strip())]
    if not opened:
        return False
    rows = _spatial_ocr_rows(structural)
    for item in opened:
        text, box = _text(item).strip(), _box(item)
        index = next((i for i, (_, _, parts) in enumerate(rows)
                      if (text, box) in parts), None)
        if text.endswith("の") and index is not None:
            parts = rows[index][2]
            part_index = parts.index((text, box))
            if part_index + 1 < len(parts):
                following_text, following_box = parts[part_index + 1]
                gap = following_box[0] - box[0] - box[2]
                if (0 <= gap <= max(box[3], following_box[3]) * .75
                        and re.match(r"[\u3040-\u30ff\u3400-\u9fff]", following_text)):
                    continue
        # This closes a known UI wrap, not arbitrary fragmented arithmetic.
        # A written operator, negative value or another field stays open.
        if not re.search(r"(?:スキルカード使用数|ターン)追加$", text):
            return True
        if index is not None and index + 1 < len(rows):
            _, box, _ = rows[index]
            following, next_box, _ = rows[index + 1]
            gap = next_box[1] - box[1] - box[3]
            if (re.fullmatch(r"[+＋][1-9][0-9]*", following.strip())
                    and 0 <= gap <= max(box[3], next_box[3]) * .75):
                continue
        return True
    return False


def _old_title_overlap_pixels(
    image: np.ndarray, sources: Sequence[np.ndarray], box: Box, panel: Box, title: Box, core: np.ndarray,
) -> bool:
    """Separate an old header number's rectangle from a complete title glyph.

    OCR rectangles are not foreground masks. A partially covered background
    number can include the left edge of a newly drawn, independently confirmed
    title. Every intersecting glyph must belong wholly to that title and have
    substantial support outside the background rectangle. Body rows and newly
    isolated strokes cannot borrow the title's rectangle as a blanket mask.
    """
    x, y, w, h = box
    px, py, pw, ph = panel
    if not _contains(panel, title) or y + h > title[1] + title[3] or _contains(panel, box):
        return False
    l, t, r, b = max(x, px), max(y, py), min(x + w, px + pw), min(y + h, py + ph)
    if l >= r or t >= b:
        return False
    foreground = ((image[py:py + ph, px:px + pw].min(axis=2) < 230) & core).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    overlapping = labels[t - py:b - py, l - px:r - px]
    glyphs = np.unique(overlapping[overlapping != 0])
    if not len(glyphs):
        return False
    for glyph in glyphs:
        gx, gy, gw, gh, area = map(int, stats[glyph])
        if (not _contains(title, (px + gx, py + gy, gw, gh))
                or np.count_nonzero(overlapping == glyph) > area * .75):
            return False

    margin = max(3, round(image.shape[1] * .01))
    exposed = np.ones((h, w), bool)
    a, c = max(0, px - margin - x), min(w, px + pw + margin - x)
    d, e = max(0, py - margin - y), min(h, py + ph + margin - y)
    if a < c and d < e:
        exposed[d:e, a:c] = False
    if exposed.sum() < w * h * .25:
        return False
    current = image[y:y + h, x:x + w].astype(np.int16)
    local_core = np.zeros((h, w), bool)
    local_core[t - y:b - y, l - x:r - x] = core[t - py:b - py, l - px:r - px]
    # Keep the eroded surface plus its three-pixel paint/antialias transition
    # even at the crop edge. Strong new strokes still veto the source proof.
    # Clipping this dilation to the panel bbox would misclassify its genuine
    # outer shadow as new ink, repeating the original rectangle assumption.
    covered = _surface_coverage(core, panel, box, 3)
    current_dark = current.min(axis=2) < 230
    current_strong = current.min(axis=2) < 230 - 32
    votes = 0
    for source in sources:
        previous = source[y:y + h, x:x + w].astype(np.int16)
        delta = np.abs(current - previous)
        visible_delta = delta[exposed]
        previous_strong = previous.min(axis=2) < 230 - 32
        previous_dark = previous.min(axis=2) < 230
        if (visible_delta.mean() > 4 or (visible_delta.max(axis=1) > 32).mean() > .01
                or not np.any(exposed & current_strong & previous_strong)
                or np.any(exposed & (current_strong | previous_strong) & (delta.max(axis=2) > 32))
                or np.any(~local_core & current_dark & ((previous - current).max(axis=2) > 32))
                or np.any(~covered & previous_dark & ((current - previous).max(axis=2) > 32))):
            continue
        weak = ~covered & current_dark & ~previous_dark & ((previous - current).max(axis=2) > 4)
        if weak.any() and not shadow_field_confirms_weak_ink(image, source, box, panel, core, weak=weak):
            continue
        votes += 1
    return votes >= 2


def _numeric_background_decoration(text: str) -> bool:
    """OCR decoration is not an operand; pixel ownership still needs proof."""
    normalized = unicodedata.normalize("NFKC", text)
    # Operators, game percentages/points and Japanese effect/unit words must
    # never borrow the title proof. ASCII OCR confusions and non-math symbols
    # can, but only alongside a digit and the unchanged old-pixel proof.
    operator_names = ("PLUS", "MINUS", "MULTIPLICATION", "DIVISION", "EQUAL", "GREATER", "LESS", "PERCENT", "FRACTION", "RATIO")
    return (any("0" <= char <= "9" for char in normalized)
            and "pt" not in normalized.casefold()
            and all(char.isascii() and char.isalnum()
                    or (unicodedata.category(char) == "So"
                        and not any(word in unicodedata.name(char, "") for word in operator_names))
                    for char in normalized))


def _isolated_background_number(text: str, box: Box, panel: Box, title: Box, *, title_verified: bool) -> bool:
    """A distant backdrop digit cannot supply any tooltip body operand.

    Keep operators and units out of this route. Boundary atoms still need
    the original pixel proofs; only a full two-font horizontal gap qualifies.
    The caller has already proved the unique surface and its source change.
    """
    x, _, w, h = box
    px, _, pw, _ = panel
    gap = max(px - x - w, x - px - pw)
    return (title_verified is True and gap >= 2 * max(h, title[3])
            and text.isascii() and text.isalnum() and _numeric_background_decoration(text))


def _old_isolated_glyph(image: np.ndarray, sources: Sequence[np.ndarray], box: Box) -> bool:
    """Keep old light glyphs without confusing their colored ground with ink.

    Only the far-domain route may use this evidence. The union includes newly
    neutral pixels as well, so new dark/white strokes cannot select themselves
    out. Flat sources and erased observations supply no old-symbol evidence.
    """
    x, y, w, h = box
    current = image[y:y + h, x:x + w].astype(np.int16)
    if np.ptp(current, axis=(0, 1)).max() < 32:
        return False
    current_neutral = np.ptp(current, axis=2) <= 24
    votes = 0
    for source in sources:
        previous = source[y:y + h, x:x + w].astype(np.int16)
        previous_neutral = np.ptp(previous, axis=2) <= 24
        witnesses = previous_neutral | current_neutral
        if (np.ptp(previous, axis=(0, 1)).max() < 32
                or not np.any(previous_neutral & current_neutral
                              & (previous.min(axis=2) >= 198) & (current.min(axis=2) >= 198))):
            continue
        delta = np.abs(current - previous)[witnesses]
        if delta.mean() <= 4 and delta.max() <= 32:
            votes += 1
    return votes >= 2


def attribute_skill_detail(
    image: np.ndarray, sources: Sequence[np.ndarray], items: Sequence[Any], *, title_box: Box | None = None,
    title_verified: bool = False,
) -> SkillDetailDomain:
    """OWNED / proved BACKGROUND / UNATTRIBUTED, without consulting card values."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or len(sources) != 3
            or any(not isinstance(source, np.ndarray) or source.shape != image.shape or source.dtype != image.dtype for source in sources)):
        return SkillDetailDomain(None, None, reason="source_frames_unavailable")
    ih, iw = image.shape[:2]
    panels = _white_detail_panels(image, full_canvas=True)
    if len(panels) != 1:
        return SkillDetailDomain(None, None, reason="panel_not_unique")
    panel = panels[0]
    px, py, pw, ph = panel
    for item in items:
        if not _contains((0, 0, iw, ih), _box(item)):
            return SkillDetailDomain(None, None, reason="invalid_ocr_box")
    if title_box is None:
        header = [item for item in items if detail_title_in_header(panel, _box(item), ih)]
        rows = _spatial_ocr_rows(header)
        if not rows:
            return SkillDetailDomain(None, None, reason="title_region_unavailable")
        title_box = rows[0][1]
    if not detail_title_in_header(panel, title_box, ih):
        return SkillDetailDomain(None, None, reason="title_region_unconfirmed")
    if not _new_panel(image, sources, panel):
        return SkillDetailDomain(None, None, reason="panel_source_unconfirmed")
    owned, background, unknown = [], [], []
    blocked = False
    core = _panel_core(image, panel)
    incomplete = _unfinished_body_operand(items, panel, title_box, image=image)
    for item in items:
        box, text = _box(item), _text(item).strip()
        if not text:
            continue
        if _contains(panel, box):
            owned.append(item)
            continue
        if (_isolated_background_number(text, box, panel, title_box, title_verified=title_verified)
                and _old_isolated_glyph(image, sources, box)):
            background.append(item)
            continue
        x, y, w, h = box
        l, t, r, bottom = max(x, px), max(y, py), min(x + w, px + pw), min(y + h, py + ph)
        crossing_rows = np.zeros(0, bool)
        if l < r and t < bottom:
            crossing_rows = np.any((image[t:bottom, l:r].min(axis=2) < 230)
                                   & core[t - py:bottom - py, l - px:r - px], axis=1)
        crossing_ink = bool(crossing_rows.any())
        overlaps = x < px + pw and x + w > px and y < py + ph and y + h > py
        relevant = (title_box[1] <= y + h and y < py + ph) or bool(_EFFECT.search(text))
        decoration = re.fullmatch(r"[A-Za-z0-9()+−\-]+", text) is not None
        possible_operand = any(character.isnumeric() for character in text) or text[:1] in {"+", "-", "−"}
        empty_surface_overlap = (overlaps and not crossing_ink and decoration and not _EFFECT.search(text)
                                 and not (incomplete and possible_operand))
        # A wide OCR box is not a foreground mask. Only its ink intersecting
        # the panel determines whether it reaches the body. The old rim still
        # needs the complete source/three-white-pixel proof below.
        verified_number = title_verified is True and _numeric_background_decoration(text)
        header_number = (verified_number and crossing_ink
                         and t + int(np.flatnonzero(crossing_rows)[-1]) < title_box[1] + title_box[3])
        old_border = (crossing_ink and (not decoration or header_number or (verified_number and not incomplete))
                      and not _EFFECT.search(text)
                      and (not (incomplete and possible_operand) or header_number)
                      and _old_border_pixels(image, sources, box, panel, core))
        old_title_overlap = (crossing_ink and title_verified is True and not incomplete
                             and _numeric_background_decoration(text)
                             and _old_title_overlap_pixels(image, sources, box, panel, title_box, core))
        old_header_label = (title_verified is True and text.isalpha() and not _EFFECT.search(text)
                            and title_box[1] <= y and y + h <= title_box[1] + title_box[3])
        if old_border or old_title_overlap or (not crossing_ink and (empty_surface_overlap or not relevant or _old_pixels(image, sources, box, panel)
                                               or ((not incomplete or old_header_label)
                                                   and _old_edge_pixels(image, sources, box, panel, core)))):
            background.append(item)
        else:
            unknown.append(item)
            # A clipped glyph can be re-observed in the existing bounded crop.
            # New complete external semantics must never disappear via a crop.
            if not overlaps or (not crossing_ink and (not decoration or _EFFECT.search(text))):
                blocked = True
    return SkillDetailDomain(panel, title_box, tuple(owned), tuple(background), tuple(unknown),
                             "unattributed_atoms" if unknown else "", blocked)


def project_domain_items(items: Sequence[Any], crop: Box, scale: int = 2) -> tuple[Any, ...] | None:
    """Map a crop observation to the unchanged parent-image coordinate space."""
    x, y, w, h = crop
    result = []
    for item in items:
        a, b, c, d = _box(item)
        if not _contains((0, 0, w * scale, h * scale), (a, b, c, d)):
            return None
        result.append({"text": _text(item), "box": (x + a // scale, y + b // scale,
                       (a + c + scale - 1) // scale - a // scale,
                       (b + d + scale - 1) // scale - b // scale)})
    return tuple(result)


def reobserved_unknowns(domain: SkillDetailDomain, projected: Sequence[Any]) -> bool:
    """A bounded view may not silently omit previously visible crossing ink."""
    left, top, width, height = domain.panel
    used = set()
    for item in domain.unknown:
        x, y, w, h = _box(item)
        matches = []
        for index, other in enumerate(projected):
            a, b, c, d = _box(other)
            overlap = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
            if (_text(item).strip() == _text(other).strip()
                    and left < a and a + c < left + width and top < b and b + d < top + height
                    and max(w / c, c / w, h / d, d / h) <= 1.5
                    and overlap >= min(w * h, c * d) * .9):
                matches.append(index)
        if len(matches) != 1 or matches[0] in used:
            return False
        used.add(matches[0])
    return True
