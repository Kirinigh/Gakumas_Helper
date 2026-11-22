"""Read the Idol Road formation that the game shows before a challenge.

Screen layout reference: hiro-gakumas-hero 8380622a, vision/tower.py.
The caller supplies Maa OCR and the existing card-art recognizer.
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .high_score.contracts import PlanType

_PLANS = {
    'ProducePlanType_Plan1': PlanType.SENSE,
    'ProducePlanType_Plan2': PlanType.LOGIC,
    'ProducePlanType_Plan3': PlanType.ANOMALY,
}
_DECK_FOR_EXAM_EFFECT = {
    'ExamLessonBuff': 'lesson_buff',
    'ExamParameterBuff': 'parameter_buff',
    'ExamReview': 'review',
    'ExamCardPlayAggressive': 'aggressive',
    'ExamConcentration': 'concentration',
    'ExamFullPower': 'full_power',
}


@dataclass(frozen=True)
class FormationOverview:
    layer: int | None
    idol_name: str | None
    card_title: str | None
    turns: int | None
    multiplier_permils: tuple[int, ...] | None
    max_stamina: int | None
    page: int
    next_page: tuple[int, int] | None
    previous_page: tuple[int, int] | None


@dataclass(frozen=True)
class FormationCard:
    row: int
    column: int
    card_id: str | None
    upgraded: bool


@dataclass(frozen=True)
class FormationIdol:
    idol_card_id: str
    plan: PlanType
    initial_cards: tuple[tuple[str, int], ...] | None


def _only(values):
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def read_formation_overview(rows):
    """Use one canonical 720×1280 OCR result from the active formation page."""
    boxes = tuple((row['text'].strip(), x + w // 2, y + h // 2)
                  for row in rows for x, y, w, h in (row['box'],))
    page = _only((int(text[0]), x, y) for text, x, y in boxes
                 if re.fullmatch(r'[12]/2', text) and 600 <= y <= 720)
    # The challenge button and pager distinguish formation from floor selection.
    if page is None or not any(text == '挑戦する' and y >= 1000 for text, _, y in boxes):
        return None
    number, cx, cy = page
    layer = _only(int(text[5:]) for text, _, _ in boxes if re.fullmatch(r'STAGE\d+', text))
    if number == 2:
        return FormationOverview(layer, None, None, None, None, None, number, None, (cx - 97, cy))

    def number_beside(label, label_y, number_y):
        return _only(int(value) for text, x, y in boxes if text == label and label_y[0] <= y <= label_y[1]
                     for value, nx, ny in boxes if value.isdecimal() and abs(nx - x) < 30 and number_y(y, ny))

    turns = number_beside('ターン', (60, 130), lambda _, y: 30 <= y <= 80)
    stamina = number_beside('体力', (500, 600), lambda y, ny: 15 <= ny - y <= 50)
    rows_by_height = sorted((y, int(match[1]) * 10) for text, x, y in boxes if x > 560
                            and (match := re.fullmatch(r'\D?(\d{2,5})[%％]', text)))
    multipliers = tuple(value for _, value in rows_by_height) if len(rows_by_height) == 3 else None
    idol = _only(text for text, x, y in boxes if 180 <= x <= 400 and 390 <= y <= 425)
    title = _only(text for text, x, y in boxes if 180 <= x <= 330 and 360 <= y < 390)
    return FormationOverview(layer, idol, title, turns, multipliers, stamina, number, (cx + 97, cy), None)


def read_formation_deck(image, identify):
    """Read the active card cells on formation page 2 in formation order.

    ``identify(image, box, upgraded)`` is the existing recognizer adapted to
    the small card-art crop. It returns a local exact card ID, or None when
    unread. Duplicate instances and unread bright cells retain their slots.
    """
    if image.shape != (1280, 720, 3):
        raise ValueError('idol road formation image must be canonical Maa BGR')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    layout = ((415, (70, 160, 250, 340, 430, 520)),
              (515, (195, 285, 375, 465, 555, 645)),
              (607, (195, 285, 375, 465, 555, 645)))
    cards = []
    for row, (y, columns) in enumerate(layout):
        for column, x in enumerate(columns):
            cell = hsv[y - 42:y + 42, x - 42:x + 42, 2]
            if np.mean(cell) < 160:
                continue
            hue, saturation, value = hsv[y - 22:y + 2, x + 15:x + 45].transpose(2, 0, 1)
            red = ((hue <= 8) | (hue >= 172)) & (saturation >= 150) & (value >= 170)
            upgraded = bool(np.count_nonzero(red) >= 15)
            card_id = identify(image, (x - 42, y - 42, 84, 84), upgraded)
            cards.append(FormationCard(row, column, card_id, upgraded))
    return tuple(cards)


def find_idol_card(idol_name, card_title, *, characters, idol_cards):
    """Join the displayed character and card title to one master row."""
    if not idol_name or not card_title:
        return None
    name = ''.join(idol_name.split())
    character_id = _only(row['id'] for row in characters
                         if ''.join((row['lastName'] + row['firstName']).split()) == name)
    if character_id is None:
        return None
    matches = [row for row in idol_cards
               if row['characterId'] == character_id and row['name'] == card_title]
    return matches[0] if len(matches) == 1 else None


def resolve_formation_idol(overview, *, characters, idol_cards, initial_decks):
    """Join displayed name and title to the master, including off-page basic cards."""
    card = find_idol_card(overview.idol_name, overview.card_title, characters=characters, idol_cards=idol_cards)
    if card is None:
        return None
    plan = _PLANS.get(card['planType'])
    if plan is None:
        return None
    initial_cards = None
    theme = _DECK_FOR_EXAM_EFFECT.get(card['examEffectType'].removeprefix('ProduceExamEffectType_'))
    decks = [row for row in initial_decks if row['id'] == f'initial_deck-{theme}'] if theme else []
    if len(decks) == 1 and decks[0].get('produceCardIds'):
        ids = decks[0]['produceCardIds']
        upgrades = decks[0].get('produceCardUpgradeCounts') or (0,) * len(ids)
        if len(ids) == len(upgrades):
            initial_cards = tuple(zip(ids, upgrades, strict=True))
    return FormationIdol(card['id'], plan, initial_cards)
