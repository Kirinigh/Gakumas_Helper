"""Attribute one skill tooltip before any spatial or numeric text assembly."""

import re
from typing import Any
from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np

from ._reader_visual import _box, _text, _spatial_ocr_rows
from ._detail_title_region import _white_detail_panels

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


def attribute_skill_detail(
    image: np.ndarray, sources: Sequence[np.ndarray], items: Sequence[Any], *, title_box: Box | None = None,
) -> SkillDetailDomain:
    """OWNED / proved BACKGROUND / UNATTRIBUTED, without consulting card values."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or len(sources) != 3
            or any(not isinstance(source, np.ndarray) or source.shape != image.shape or source.dtype != image.dtype for source in sources)):
        return SkillDetailDomain(None, None, reason="source_frames_unavailable")
    ih, iw = image.shape[:2]
    panels = _white_detail_panels(image)
    if len(panels) != 1:
        return SkillDetailDomain(None, None, reason="panel_not_unique")
    panel = panels[0]
    px, py, pw, ph = panel
    for item in items:
        if not _contains((0, 0, iw, ih), _box(item)):
            return SkillDetailDomain(None, None, reason="invalid_ocr_box")
    if title_box is None:
        header = [item for item in items if _contains(panel, _box(item))
                  and _box(item)[1] + _box(item)[3] <= py + min(ph * .35, ih * .05)]
        rows = _spatial_ocr_rows(header)
        if not rows:
            return SkillDetailDomain(None, None, reason="title_region_unavailable")
        title_box = rows[0][1]
    if (not _contains(panel, title_box)
            or title_box[1] + title_box[3] > py + min(ph * .35, ih * .05)
            or not _new_panel(image, sources, panel)):
        return SkillDetailDomain(None, None, reason="panel_source_unconfirmed")
    owned, background, unknown = [], [], []
    blocked = False
    core = _panel_core(image, panel)
    incomplete = any(_contains(panel, _box(item)) and _box(item)[1] >= title_box[1] + title_box[3]
                     and _OPEN_OPERAND.search(_text(item).strip()) for item in items)
    for item in items:
        box, text = _box(item), _text(item).strip()
        if not text:
            continue
        if _contains(panel, box):
            owned.append(item)
            continue
        x, y, w, h = box
        l, t, r, bottom = max(x, px), max(y, py), min(x + w, px + pw), min(y + h, py + ph)
        crossing_ink = (l < r and t < bottom
                        and np.any((image[t:bottom, l:r].min(axis=2) < 230) & core[t - py:bottom - py, l - px:r - px]))
        overlaps = x < px + pw and x + w > px and y < py + ph and y + h > py
        relevant = (title_box[1] <= y + h and y < py + ph) or bool(_EFFECT.search(text))
        decoration = re.fullmatch(r"[A-Za-z0-9()+−\-]+", text) is not None
        possible_operand = any(character.isnumeric() for character in text) or text[:1] in {"+", "-", "−"}
        empty_surface_overlap = (overlaps and not crossing_ink and decoration and not _EFFECT.search(text)
                                 and not (incomplete and possible_operand))
        if not crossing_ink and (empty_surface_overlap or not relevant or _old_pixels(image, sources, box, panel)):
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
