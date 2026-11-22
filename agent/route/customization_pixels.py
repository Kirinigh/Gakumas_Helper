"""Read customization facts without inventing card, option or purchase targets.

The interval entry is calibrated against a retained local main-page capture.
List/count ROIs follow miyu1019 fb8fd00c; they still need local menu captures.
These count facts alone cannot construct a CustomizationMenu.
"""
from __future__ import annotations

import re


def read_interval_customization_entry(image, rows):
    from .production import _row
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('unsupported customization interval image')
    _row(rows, r'スキルカード', (320, 880, 280, 90))
    caption = _row(rows, r'カスタマイズ', (320, 900, 280, 80))
    capacity = _row(rows, r'カスタマイズ可能枚数あと\d+枚', (310, 950, 285, 60))
    remaining = int(re.fullmatch(r'カスタマイズ可能枚数あと(\d+)枚', capacity['text'])[1])
    x, y, w, h = caption['box']
    crop = image[y:y+h, x:x+w].astype('int16')
    green = (crop[:, :, 1] > crop[:, :, 0] + 15) & (crop[:, :, 1] > crop[:, :, 2] + 15)
    if float(green.mean()) < .6:
        raise ValueError('current customization entry is not visibly enabled')
    return dict(remaining=remaining, target=tuple(caption['box']))


def read_customization_list(rows):
    from .production import _row, _interval_resources
    current, maximum, currency = _interval_resources(rows)
    _row(rows, r'カスタマイズするスキルカードを1つ選んでください', (40, 275, 640, 85))
    capacity = _row(rows, r'カスタマイズ可能スキルカードあと\d+枚', (180, 1078, 360, 18))
    remaining = int(re.fullmatch(r'カスタマイズ可能スキルカードあと(\d+)枚', capacity['text'])[1])
    return dict(scene='card_customize_list', stamina=current, max_stamina=maximum,
        currency=currency, customization_remaining=remaining, target=None)


def read_customization_counts(rows, option_maxima):
    """Source-defined two/three columns; absent/contradictory text stays unknown."""
    from .production import _row
    if (not isinstance(option_maxima, tuple) or len(option_maxima) not in (2, 3)
            or any(not isinstance(pair, tuple) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0]
                or type(pair[1]) is not int or pair[1] <= 0 for pair in option_maxima)
            or len(dict(option_maxima)) != len(option_maxima)):
        raise ValueError('customization option columns require exact catalog maxima')
    remaining = []
    for index, (key, maximum) in enumerate(option_maxima):
        fields = []
        for pattern, roi in ((r'あと\d+', (40 + index*220, 875, 200, 30)),
                (r'カスタマイズ上限', (40 + index*220, 980, 200, 35))):
            # Distinguish no field from duplicate/weak competing evidence.
            x, y, w, h = roi
            candidates = [row for row in rows if re.fullmatch(pattern, row['text'])
                and x <= row['box'][0] and y <= row['box'][1]
                and row['box'][0]+row['box'][2] <= x+w and row['box'][1]+row['box'][3] <= y+h]
            if candidates:
                fields.append(_row(candidates, pattern, roi))
        if len(fields) != 1:
            raise ValueError('customization remaining count is missing or contradictory')
        text = fields[0]['text']
        value = 0 if text == 'カスタマイズ上限' else int(text[2:])
        if not 0 <= value <= maximum:
            raise ValueError('customization remaining count exceeds catalog maximum')
        remaining.append((key, value))
    total = _row(rows, r'合計あと\d+回', (425, 1090, 155, 40))
    return dict(remaining=tuple(remaining), total_remaining=int(re.fullmatch(r'合計あと(\d+)回', total['text'])[1]))
