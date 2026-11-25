"""Read live Tower status values from the same frame as the playable hand."""

from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np

from agent.utils.runtime_paths import resolve_runtime_path

_FIELDS = ('good_impression_turns', 'review_score_percent', 'motivation')
_HIRO_FIELDS = {'concentration': 'examlessonbuff', 'good_condition_turns': 'examparameterbuff'}
_SIZE = 42
_GLYPH_RADIUS = 23
_COLUMN = (15, 226, 59, 640)


@lru_cache(maxsize=1)
def _references():
    relative = Path('resource/base/image/produce/idol_road_status_glyphs.png')
    path = resolve_runtime_path(Path('assets') / relative, relative)
    sheet = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if sheet is None or sheet.shape != (_SIZE, _SIZE * len(_FIELDS)):
        raise ValueError('idol_road_status_references_invalid')
    references = {field: (glyph,) for field, glyph in zip(_FIELDS, np.hsplit(sheet, len(_FIELDS)), strict=True)}
    for field, name in _HIRO_FIELDS.items():
        relative = relative.with_name(f'idol_road_{name}.png')
        icon_path = resolve_runtime_path(Path('assets') / relative, relative)
        if icon_path.is_file():
            icon = cv2.imdecode(np.frombuffer(icon_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            if icon is not None and icon.ndim == 3 and icon.shape[2] == 4:
                references[field] = tuple(cv2.resize(icon[..., 3], (size, size), interpolation=cv2.INTER_AREA)
                                          for size in (30, 34, 38, 42))
    return references


def white_glyph(image, x, y):
    half = _SIZE // 2
    patch = image[y - half:y + half, x - half:x + half]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 215), (180, 70, 255))
    row, col = np.ogrid[-half:half, -half:half]
    white[np.abs(row) + np.abs(col) > _GLYPH_RADIUS] = 0
    return white


def read_tower_status_values(image, read_value):
    """Match blue diamond glyphs and read their adjacent live numbers.

    Hiro distinguishes a status by its icon, then reads the value beside it.
    The DMM references are glyphs isolated from this client's screenshots.
    An absent or unread slot contributes no update to the tracked value.
    """
    x, y, width, height = _COLUMN
    hsv = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (88, 95, 140), (118, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, boxes, _ = cv2.connectedComponentsWithStats(blue)
    seen = {}
    for left, top, w, h, area in boxes[1:]:
        center_x, center_y = int(x + left + w // 2), int(y + top + h // 2)
        if not (44 <= w <= 57 and 44 <= h <= 57 and area >= 450 and 40 <= center_x <= 47):
            continue
        patches = tuple(white_glyph(image, center_x + dx, center_y + dy)
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        matches = sorted(((max(cv2.minMaxLoc(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED))[1]
                               for patch in patches for template in templates), field)
                          for field, templates in _references().items()), reverse=True)
        (score, field), (second, _) = matches[:2]
        if score < .72 or score - second < .12:
            continue
        suffix = {'percent': True} if field == 'review_score_percent' else {'percent': False}
        if field == 'good_condition_turns':
            suffix['turns'] = True
        value = read_value(image, (center_x + 11, center_y - 4, 68, 37), **suffix)
        if value is not None:
            seen.setdefault(field, set()).add(value)
    return {field: next(iter(values)) for field, values in seen.items() if len(values) == 1}
