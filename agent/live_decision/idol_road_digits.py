"""Single-frame glyph recognition for small outlined Tower ranking scores."""

import cv2
import numpy as np

# White glyph masks from two archived Maa ranking strips (11-23 01:24 and
# 02:18), normalized to 16 x 20. Each hex word is one row, left bit first.
_GLYPH_ROWS = (
    '07e01ff83ffc7f7e781ff80ff00ff00ff00ff00f'
    'f00ff00ff00ff00ff80f781f7e3e3ffe1ff807e0',  # 0
    '038003f003fc03fc03fc03c403c003c003c003c0'
    '03c003c003c003c003c003c007e07fffffffffff',  # 1
    '07e03ff87ffc7c3ef81ff00ff00ff80078007c00'
    '3e001f000f8007e003e001f001fc3ffc7fff7fff',  # 2
    '7ffe3ffe1ffe0fc007c007e007e00ff03ff87ff8'
    '7c00fc00f800f800f800fc007f003ffc1fff03ff',  # 3
    '300038001c001c001e001f001f801fc01fc01fe0'
    '1ef01c783c783e7cfffeffff7fff3e003c001c00',  # 4
    '3ff03ff03ff800780038003c01fc0ffc1ffc3ffc'
    '7c007800f800f800f8007c007e003ffe1fff03ff',  # 5
    '0700078007c003e001e001f001f81ffc3ffc7ffe'
    'fc1ff00ff00ff00ff00ff81f7ffe3ffc1ff807e0',  # 6
    'ffffffff7ffe7e003e003e001e000f000f800780'
    '07c003c003e001e001f000f000f80078007c003c',  # 7
    '07e01ff83ffc7c3e781e781e781e3c3c3ffc1ff8'
    '3ffc7c3ef81ff80ff81ff81f7c3e7ffe1ff80ff0',  # 8
    '07e01ff83ffc7e7ef81ff00ff00ff00ff81ff81f'
    '7ffe3ffc3ff81fc00f800f8007c003e001f001f0',  # 9
)
_GLYPHS = np.array([
    [int(rows[offset:offset + 4], 16) for offset in range(0, len(rows), 4)]
    for rows in _GLYPH_ROWS
], dtype=np.uint16)
_GLYPHS = (_GLYPHS[:, :, None] & (1 << np.arange(16))) != 0
_OUTLINE = np.ones((9, 9), np.uint8)
_OPEN = np.ones((3, 3), np.uint8)


def read_ranking_digits(image, roi):
    """Return a score only when the outlined glyphs give a clear reading."""
    x, y, width, height = roi
    crop = image[y:y + height, x:x + width]
    hsv = cv2.cvtColor(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 192), (179, 65, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, _OPEN)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    dark = hsv[:, :, 2] < 170
    parts = []
    for label, (left, top, w, h, area) in enumerate(stats[1:], 1):
        right, bottom = left + w, top + h
        if not (30 <= h <= 99 and 10 <= w <= h and area >= 135
                and 0 < top < bottom < white.shape[0] and 0 < left < right < white.shape[1]):
            continue
        neighborhood = np.s_[max(0, top - 4):bottom + 4, max(0, left - 4):right + 4]
        glyph = (labels[neighborhood] == label).astype(np.uint8)
        ring = (cv2.dilate(glyph, _OUTLINE) != 0) & (glyph == 0)
        if dark[neighborhood][ring].mean() >= .50:
            parts.append((left, bottom, h, labels[top:bottom, left:right] == label))
    if not parts:
        return None

    tallest = max(part[2] for part in parts)
    baseline = max(part[1] for part in parts)
    # Holes in large digits can expose small white artwork behind the score.
    line = sorted((part for part in parts if part[2] >= .78 * tallest and part[1] >= baseline - .13 * tallest),
                  key=lambda part: part[0])
    if not line or line[0][0] > .8 * tallest:
        return None
    if any(not .55 * tallest <= following[0] - current[0] <= 1.2 * tallest
           for current, following in zip(line, line[1:], strict=False)):
        return None

    digits = []
    for _, _, _, mask in line:
        glyph = cv2.resize(mask.astype(np.float32), (16, 20), interpolation=cv2.INTER_AREA) >= .5
        overlap = np.count_nonzero(_GLYPHS & glyph, axis=(1, 2)) / np.count_nonzero(_GLYPHS | glyph, axis=(1, 2))
        other, best = np.argsort(overlap)[-2:]
        if overlap[best] < .72 or overlap[best] - overlap[other] < .10:
            return None
        digits.append(str(best))
    return int(''.join(digits))
