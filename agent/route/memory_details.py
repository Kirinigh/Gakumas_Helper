"""Read the two detail pages of the formation memory picker."""
from __future__ import annotations

import re

from .activity_production import canonical


def read_memory_detail(rows, catalog):
    """An unbound detail is an expected grant, without proof that it took effect."""
    page_numbers = set()
    lines = []
    for row in rows:
        if row['score'] < .95:
            continue
        x, y, width, height = row['box']
        text = canonical(row['text'])
        if not text:
            continue
        cx, cy = x + width / 2, y + height / 2
        if 310 <= cx <= 410 and 480 <= cy <= 580:
            marker = re.fullmatch(r'([12])[/1lI|]2', text)
            if marker:
                page_numbers.add(int(marker[1]))
        lines.append((cy, cx, y, y + height, text))
    footer = {text for cy, _, _, _, text in lines if cy >= 1000}
    if len(page_numbers) != 1 or not {'キャンセル', '決定'} <= footer or not any('成詳' in t for t in footer):
        return None
    lines.sort()
    page = page_numbers.pop()
    if page == 2:
        return dict(page=2, abilities=_abilities(lines))

    timing_rows = {text for cy, _, _, _, text in lines if 180 <= cy <= 215
        and '獲得' in text and '獲得可能' not in text}
    timings = set()
    for timing in timing_rows:
        if '開始時' in timing:
            timings.add('start')
        if '中間試験' in timing or 'オーディション後' in timing:
            timings.add('mid')
    names = {text for cy, cx, _, _, text in lines if 222 <= cy <= 260 and cx >= 192}
    matches = [card for card in catalog.entries.values() if canonical(card.name) in names]
    card = matches[0] if len(matches) == 1 else None
    effects = [text for cy, cx, _, _, text in lines if 270 <= cy <= 510 and cx >= 192
        and not re.fullmatch(r'[12][/1lI|]2', text)]
    return dict(page=1, expected_timing=next(iter(timings)) if len(timings) == 1 else None,
        card=dict(card_id=card.card_id, name=card.name, upgraded=card.upgraded) if card is not None else None,
        effects=effects or None)


def _abilities(lines):
    abilities = []
    for left in (True, False):
        parts = []
        bottom = 0
        for cy, cx, top, end, text in lines:
            if (cx < 360) != left or not 120 <= cy <= 515 or text == '獲得可能スキルカード' or (
                    310 <= cx <= 410 and 480 <= cy and re.fullmatch(r'[12][/1lI|]2', text)):
                continue
            if parts and top - bottom > 22:
                abilities.append(parts)
                parts = []
            parts.append(text)
            bottom = max(end, bottom)
        if parts:
            abilities.append(parts)
    return [dict(text=''.join(text for text in parts if text != '発動不可'), disabled='発動不可' in parts)
        for parts in abilities if any(text != '発動不可' for text in parts)] or None


def inventory(entries, catalog):
    """Group retained pages by the caller's stable observed slot, never by card name or stats."""
    slots = {}
    for entry in entries:
        slot = entry['slot']
        if not isinstance(slot, str) or not slot.strip():
            raise ValueError('a saved memory entry needs its observed slot ID')
        pages = slots.setdefault(slot, {1: [], 2: []})
        for frame in entry['frames']:
            detail = read_memory_detail(frame['rows'], catalog)
            if detail is not None and detail not in pages[detail['page']]:
                pages[detail['page']].append(detail)
    memories = []
    for ordinal, pages in enumerate(slots.values(), 1):
        first, second = (values[0] if len(values) == 1 else {} for values in pages.values())
        memories.append(dict(memory=f'memory-{ordinal}', card=first.get('card'),
            expected_timing=first.get('expected_timing'), effects=first.get('effects'),
            abilities=second.get('abilities'), observed_pages=[page for page, values in pages.items() if values],
            ambiguous_pages=[page for page, values in pages.items() if len(values) > 1]))
    return memories
