"""Numeric fields shared by DMM's H.I.F. and Idol Road battle HUDs."""
from __future__ import annotations

import re

import cv2
import numpy as np

# Maa coordinates after full-frame normalization. Exclude the heart and bar.
EXAM_STAMINA_ROI = (573, 212, 49, 32)
EXAM_GENKI_ROI = (660, 175, 55, 34)


def exam_failed_header(image, ocr):
    """Hiro recognizes a failed result even when the player's row is hidden."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return False
    return any('不合格' in ''.join(row.text.split()) for row in ocr(image, roi=(20, 120, 240, 80)))


def read_lesson_goal(image, ocr):
    """Read Hiro's practice target and displayed attribute value from one frame."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    labels = ''.join(row.text.replace(' ', '') for row in ocr(image, roi=(230, 72, 260, 52)))
    if re.search(r'PERFECTまで', labels, re.IGNORECASE):
        goal = 'perfect'
    elif re.search(r'CLEARまで', labels, re.IGNORECASE):
        goal = 'clear'
    else:
        return None

    def value(roi):
        text = ''.join(row.text for row in ocr(image, roi=roi))
        if not any(word in text for word in ('込み', 'ナス', 'ボーナス')):
            if match := re.search(r'(\d{1,4})\s*$', text.replace(',', '')):
                return int(match[1])
        return None

    return dict(goal=goal, left=value((300, 125, 120, 47)), parameter=value((560, 45, 140, 37)))


def lesson_turns_shown(image, read_turns):
    """Hiro's dark countdown survives when SELECT replaces the lesson target."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return False
    if cv2.cvtColor(image[61:119, 30:94], cv2.COLOR_BGR2GRAY).mean() >= 175:
        return False  # An early PERFECT can replace it with a light reward label.
    turns = read_turns(image)
    return turns is not None and turns >= 1


def read_exam_rank_label(text):
    """Read Hiro's result-row ordinal even if OCR misreads its first glyph."""
    match = re.fullmatch(r'([1-6IlS])(st|nd|rd|th)[lI1|]?', ''.join(text.split()))
    if match is None:
        return None
    ordinal = {'st': 1, 'nd': 2, 'rd': 3}.get(match[2])
    return ordinal or int(match[1].translate(str.maketrans('IlS', '115')))


def read_exam_ranking(image, ocr):
    """Use Hiro's white player border to bind the visible result row."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    buttons = ''.join(row.text.replace(' ', '') for row in ocr(image, roi=(0, 1060, 720, 180)))
    if not re.search(r'次へ|再挑戦', buttons):
        return None
    scores = []
    for row in ocr(image, roi=(195, 250, 525, 840)):
        label = re.sub(r'\s+', '', row.text)
        if score := re.fullmatch(r'(\d[\d,.]*)([Pp][TtlI1]?)?', label):
            if score[2] or len(score[1]) >= 2:
                digits = re.sub(r'[^0-9]', '', score[1])
                scores.append((row.box[1] + row.box[3] // 2 - 22, int(digits)))
    scores.sort()
    hsv = cv2.cvtColor(image[:, 20:32], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 225), (179, 40, 255)) != 0
    player = [(y, score) for y, score in scores if white[y - 50:y + 50].mean() >= .08]
    if len(player) != 1:
        return None
    y, score = player[0]
    # Hiro associates the rank label with its own row before falling back to
    # row order. A missing opponent score must not move the player's rank.
    labels = set()
    for row in ocr(image, roi=(0, 250, 190, 840)):
        if abs(row.box[1] + row.box[3] // 2 - y) <= 40:
            if (rank := read_exam_rank_label(row.text)) is not None:
                labels.add(rank)
    if len(labels) == 1:
        return dict(rank=labels.pop(), score=score)
    if len(labels) > 1 or not 2 <= len(scores) <= 6 or abs(scores[0][0] - 332) > 20:
        return None
    return dict(rank=next(i for i, (row_y, _) in enumerate(scores, 1) if row_y == y), score=score)


def battle_dialog_header(image):
    """Locate Hiro's orange dialog band before running text recognition."""
    if not isinstance(image, np.ndarray):
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv[:, 30:690], (8, 110, 190), (30, 255, 255))
    coverage = (orange != 0).mean(axis=1)
    white = cv2.inRange(hsv, (0, 0, 226), (179, 39, 255))
    for keep in (.55, .45, .3):
        edges = np.flatnonzero(np.diff(np.r_[False, coverage >= keep, False]))
        for start, end in reversed(tuple(zip(edges[::2], edges[1::2], strict=True))):
            beginning = np.flatnonzero(coverage[start:end] >= .55)
            if not beginning.size:
                continue
            top, bottom = int(start + beginning[0]), int(end)
            if not 25 <= bottom - top <= 70:
                continue
            title = cv2.inRange(hsv[top:bottom, 40:320], (0, 0, 236), (179, 39, 255))
            if (title != 0).mean() < .03:
                continue
            body = white[bottom + 10:bottom + 60, 60:660]
            if keep == .55 or (body.shape[0] >= 10 and (body != 0).mean() >= .6):
                return top, bottom
    return None


def battle_choice_buttons(image):
    """Locate the pale choice panel and its paired footer buttons in one frame."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None

    def fill(box, low, high):
        x, y, width, height = box
        hsv = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
        return cv2.countNonZero(cv2.inRange(hsv, low, high)) / (width * height)

    cancel = (110, 1147, 218, 30)
    confirm = (391, 1148, 222, 29)
    white = (0, 0, 198), (179, 71, 255)
    if (fill(cancel, *white) >= .62 and fill(confirm, (0, 52, 145), (28, 255, 255)) >= .62
            and fill((39, 120, 650, 195), *white) >= .80):
        return cancel, confirm
    return None


def valid_stamina(value: object, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def outlined_number_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Find a same-frame OCR crop using the outlined white HUD glyphs.

    Adapt Hiro's no-font fallback to Maa: trim only before the first
    digit-shaped component, retaining the right side and any trailing glyph.
    """
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    x, y, w, h = roi
    scale = 3
    crop = cv2.resize(image[y:y + h, x:x + w], (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (179, 45, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    dark = hsv[:, :, 2] < 170
    outline = np.ones((9, 9), np.uint8)
    left_edges = []
    outlined = False
    for component, (left, top, width, height, area) in enumerate(stats[1:], start=1):
        if not (h * scale * .4 <= height <= h * scale * .85 and width <= height
                and area >= 18 * scale and 0 < top and top + height < h * scale):
            continue
        # Keep a possibly occluded first glyph even if its outline is broken.
        left_edges.append(int(left))
        part = (labels == component).astype(np.uint8)
        ring = (cv2.dilate(part, outline) != 0) & (part == 0)
        outlined |= bool(ring.any() and dark[ring].mean() >= .55)
    trim = max(0, min(left_edges, default=0) // scale - 3)
    return (x + trim, y, w - trim, h) if outlined and trim else None
