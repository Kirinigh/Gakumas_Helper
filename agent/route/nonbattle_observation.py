"""HIF non-battle OCR facts calibrated against retained 004--141 reports.

These are read-only page candidates. They deliberately provide no click targets,
run identity, complete deck, slot occupancy or permission from text alone.
"""
from __future__ import annotations

import re
import math
import unicodedata
from dataclasses import dataclass


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


@dataclass(frozen=True)
class NonbattleFacts:
    scene: str
    currency: int | None
    stamina: int | None
    max_stamina: int | None
    exchange: tuple[str, str] | None = None
    selected_card_id: str | None = None
    reason: str = 'candidate_requires_complete_inventory_and_stability'


def _rows(rows):
    accepted = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('malformed HIF OCR row')
        box, text, score = (row.get(key) for key in ('box', 'text', 'score'))
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(type(value) is not int for value in box) or min(box[:2]) < 0
                or min(box[2:]) <= 0 or box[0] + box[2] > 720 or box[1] + box[3] > 1280
                or not isinstance(text, str) or type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('malformed HIF OCR geometry/confidence')
        if score >= .95:
            accepted.append(dict(text=normalized(text), box=tuple(box)))
    return tuple(accepted)


def _within(box, region):
    x, y, width, height = box
    left, top, wide, high = region
    return left <= x and top <= y and x + width <= left + wide and y + height <= top + high


def _unique(rows, pattern, region):
    matches = [row['text'] for row in rows if _within(row['box'], region)
               and re.fullmatch(pattern, row['text'])]
    if len(matches) > 1:
        raise ValueError('ambiguous HIF non-battle field')
    return matches[0] if matches else None


def _card_id(catalog, name):
    matches = [card.card_id for card in catalog.entries.values()
               if normalized(card.name) == name and card.plan in ('sense', 'free')]
    if len(matches) != 1:
        raise ValueError('unresolved HIF receipt card identity')
    return matches[0]


def _exchange_receipt_text(rows):
    lines = sorted((row for row in rows if _within(row['box'], (80, 950, 560, 110))),
        key=lambda row: (row['box'][1], row['box'][0]))
    return ''.join(row['text'] for row in lines)


def exchange_receipt_text(rows):
    """Join physical result lines even when the completion verb wraps."""
    return _exchange_receipt_text(_rows(rows))


def parse_nonbattle_facts(rows, *, card_catalog):
    rows = _rows(rows)
    currency_text = _unique(rows, r'\d{1,6}', (350, 85, 100, 65))
    stamina_text = _unique(rows, r'\d{1,3}/\d{1,3}', (285, 35, 165, 75))
    stamina = maximum = None
    if stamina_text:
        stamina, maximum = map(int, stamina_text.split('/'))
        if not 0 <= stamina <= maximum <= 100 or maximum == 0:
            raise ValueError('contradictory HIF non-battle stamina')
    currency = int(currency_text) if currency_text is not None else None
    scenes = set()
    for scene, text, region in (
        ('card_choice', 'チェンジで獲得するスキルカードを選んでください', (50, 260, 620, 80)),
        ('card_exchange', 'チェンジするスキルカードを選択してください', (60, 540, 610, 100)),
        ('drink_offer', '受け取るPドリンクを選んでください。', (100, 580, 530, 90)),
        ('consult', 'Pポイントと交換するものを選んでください', (75, 260, 570, 90)),
    ):
        if _unique(rows, re.escape(normalized(text)), region) is not None:
            scenes.add(scene)
    # A split result message is read in physical line order, not OCR result order.
    receipt_text = _exchange_receipt_text(rows)
    exchange = None
    match = re.fullmatch(r'(.+)を(.+)にチェンジしました[。]?', receipt_text)
    if match:
        exchange = tuple(_card_id(card_catalog, name) for name in match.groups())
        scenes.add('card_exchange_result')
    if len(scenes) != 1:
        return NonbattleFacts('unknown', currency, stamina, maximum,
            reason='nonbattle_scene_not_uniquely_observed')
    scene = next(iter(scenes))
    selected = None
    if scene == 'card_exchange':
        title = _unique(rows, r'.+', (180, 245, 430, 60))
        if title is not None:
            selected = _card_id(card_catalog, title)
    return NonbattleFacts(scene, currency, stamina, maximum, exchange, selected)


def observe_nonbattle_facts(context, image, *, card_catalog):
    if getattr(image, 'shape', ())[:2] != (1280, 720):
        raise ValueError('unsupported HIF non-battle capture resolution')
    result = context.run_recognition('ProduceRecognitionScore', image,
        pipeline_override={'ProduceRecognitionScore': {
            'recognition': 'OCR', 'roi': [0, 0, 720, 1280], 'expected': '.+'}})
    if result is None or not result.hit:
        raise ValueError('HIF non-battle OCR unavailable')
    rows = tuple(dict(text=value.text, box=list(value.box), score=value.score)
                 for value in result.filtered_results)
    return parse_nonbattle_facts(rows, card_catalog=card_catalog)


def parse_customization_success(rows, *, card_name):
    """Only the exact current-card success message in its result text area."""
    rows = _rows(rows)
    lines = sorted((row for row in rows if _within(row['box'], (90, 960, 540, 130))),
                   key=lambda row: (row['box'][1], row['box'][0]))
    if not isinstance(card_name, str) or not card_name.strip():
        return False
    expected = normalized(card_name + 'をカスタマイズしました')
    return ''.join(row['text'] for row in lines).rstrip('。') == expected
