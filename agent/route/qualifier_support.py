"""Identify HIF qualifier support art and value a rental together with its formation."""
from __future__ import annotations

import re
import json
from math import isfinite
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .formation_review import formation_changes

ATTRS = {'Vocal': 'Vo', 'Dance': 'Da', 'Visual': 'Vi', 'Assist': 'HP'}
STEP2_TILES = ((29, 715), (254, 713), (481, 715), (29, 847), (256, 847))
RENTAL_TILE = (483, 847)


def _feature(art):
    pixels = cv2.resize(cv2.cvtColor(art, cv2.COLOR_BGR2GRAY), (64, 20),
        interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    pixels -= pixels.mean()
    return pixels / max(float(np.linalg.norm(pixels)), 1e-6)


class SupportGallery:
    """Read the local Hiro support/index.json layout using the client's own implementation."""

    def __init__(self, directory):
        folder = Path(directory)
        if not (folder / 'index.json').is_file():
            folder = folder / 'support'
        index = json.loads((folder / 'index.json').read_bytes())
        self.cards, features = [], []
        for stem, record in index.items():
            if re.fullmatch(r'[\w-]+', stem) is None:
                continue
            attr = ATTRS.get(record.get('type', '').removeprefix('SupportCardType_'))
            if attr is None or not re.fullmatch(r's_card-[\w-]+', record.get('id', '')):
                continue
            file = folder / (stem + '.png')
            image = cv2.imdecode(np.frombuffer(file.read_bytes(), np.uint8), cv2.IMREAD_COLOR) if file.is_file() else None
            if image is not None and image.shape[0] >= 4:
                self.cards.append(dict(id=record['id'], name=record.get('name'), attribute=attr))
                features.append(_feature(image[:int(.55 * image.shape[0])]))
        if not features:
            raise ValueError('no usable support art in local gallery')
        self.matrix = np.stack(features)

    def identify(self, image, x, y, *, width=211, spread=8):
        features = []
        for w in (width-6, width-3, width, width+3, width+6):
            h = int(w * .275)
            for dy in range(-spread, spread+1, 2):
                for dx in range(-spread, spread+1, 2):
                    left, top = x+dx, y+dy
                    if left >= 0 and top >= 0 and left+w <= image.shape[1] and top+h <= image.shape[0]:
                        features.append(_feature(image[top:top+h, left:left+w]))
        if not features:
            return None
        scores = self.matrix @ np.stack(features, axis=1)
        i, j = np.unravel_index(scores.argmax(), scores.shape)
        score = float(scores[i, j])
        if score < .7:
            return None
        return dict(self.cards[i], score=round(score, 3))


def _tile_stats(ui, frame, x, y):
    from .qualifier import support_level

    return support_level(frame, x, y, context=ui.context)


def read_own_support(ui, frame, gallery, *, stats=True):
    slots = []
    for x, y in STEP2_TILES:
        card = gallery.identify(frame.image, x, y)
        if card is not None and stats:
            level, breaks = _tile_stats(ui, frame, x, y)
            card.update(level=level, breaks=breaks)
        slots.append(dict(card or {}, observed=card is not None))
    return slots


def _support_picker(frame):
    from .qualifier import label

    if not label(frame, '決定', top=1080) or not label(frame, 'キャンセル', top=1080):
        return None
    if any(re.match(r'P[LI]V', row['text'], re.I) for row in frame.rows):
        return None
    for row in frame.rows:
        if 500 <= row['box'][1] <= 560 and (match := re.fullmatch(r'(\d+)\s*/\s*(\d+)', row['text'])):
            page, total = map(int, match.groups())
            if 1 <= page <= total:
                return page, total
    return None


def _visible_tiles(frame, gallery, target):
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 150)
    orange = bright & (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 22) & (hsv[:, :, 2] >= 200)
    tiles = []
    for x in (33, 255, 476):
        spans = bright[530:1060, x+182:x+202].sum(axis=1) >= 8
        edges = np.flatnonzero(np.diff(np.pad(spans.astype(np.int8), 1)))
        for start, end in zip(edges[::2], edges[1::2]):
            y = (int(start)+int(end))//2 + 530 - 89
            if not 551 <= y <= 942 or not 12 <= end-start <= 30:
                continue
            card = gallery.identify(frame.image, x, y)
            if card:
                corners = (orange[y-12:y+20, x-12:x+20].sum()
                    + orange[y+95:y+130, x-12:x+20].sum())
                tiles.append((card, int(corners), (x+105, y+59, 1, 1)))
    selected = max(range(len(tiles)), key=lambda i: tiles[i][1], default=-1)
    return [(box, i == selected and corners >= 150)
        for i, (card, corners, box) in enumerate(tiles) if card['id'] == target]


def _pick_owned_support(ui, frame, gallery, target):
    from .qualifier import label

    page, total = _support_picker(frame)
    while page > 1:
        ui.click(frame, (233, 533, 1, 1), settle=1.5)
        frame = ui.need(lambda f: _support_picker(f) == (page-1, total), 'previous support picker page')
        page -= 1
    for page in range(1, total+1):
        for _ in range(6):
            previous = frame
            frame = ui.swipe(frame, (360, 685), (360, 960))
            if not _support_picker(frame):
                raise ValueError('support picker closed while scrolling')
            if np.array_equal(frame.image[555:1000], previous.image[555:1000]):
                break
        for _ in range(12):
            matches = _visible_tiles(frame, gallery, target)
            if matches:
                if not matches[0][1]:
                    ui.click(frame, matches[0][0], settle=.8)
                    frame = ui.capture()
                if _support_picker(frame) and any(selected for _, selected in _visible_tiles(frame, gallery, target)):
                    ui.click(frame, label(frame, '決定', top=1080), settle=1.5)
                    return
                break
            previous = frame
            frame = ui.swipe(frame, (360, 960), (360, 685))
            if not _support_picker(frame):
                raise ValueError('support picker closed while scrolling')
            if np.array_equal(frame.image[555:1000], previous.image[555:1000]):
                break
        if page < total:
            ui.click(frame, (485, 533, 1, 1), settle=1.8)
            frame = ui.need(lambda f: _support_picker(f) == (page+1, total), 'next support picker page')
    if _support_picker(frame):
        ui.click(frame, label(frame, 'キャンセル', top=1080))
    raise ValueError(f'configured support card could not be selected: {target}')


def write_own_support(ui, frame, gallery, desired, receipts):
    from .qualifier import _support_step

    current = read_own_support(ui, frame, gallery, stats=False)
    plan = formation_changes(current, desired)
    # Hiro fills unidentified slots only after consuming known off-plan slots.
    moves = plan['moves'] + [dict(slot=slot, current=None, replacement=target) for slot, target in
        zip(plan['unknown_slots'], plan['unplaced'])]
    for move in moves:
        x, y = STEP2_TILES[move['slot']-1]
        target = move['replacement']
        ui.click(frame, (x+105, y+59, 1, 1), settle=1.5)
        picker = ui.need(_support_picker, 'support card picker')
        _pick_owned_support(ui, picker, gallery, target)
        frame = ui.need(lambda f: _support_step(f) and
            (gallery.identify(f.image, x, y) or {}).get('id') == target, 'written support slot')
        receipts.append(move)
    return frame


@dataclass(frozen=True)
class FormationValue:
    """Estimate the six-card qualifier, using locally supplied compiled effects."""

    profiles: dict
    attribute: str
    event_cards: dict
    opportunities: dict

    @classmethod
    def load(cls, path, attribute):
        data = json.loads(Path(path).read_bytes())
        return cls(data['cards'], attribute, data.get('event_cards', {}), data.get('opportunities', {}))

    def _profile(self, card):
        key = card['id']
        return next((p for p in self.profiles.get(key, ()) if (p['level'], p['breaks']) ==
            (card.get('level'), card.get('breaks')) and p.get('skills') is not None and p.get('events') is not None), None)

    def _triggers(self, trigger, lessons, sp):
        for kind, attr in ATTRS.items():
            if attr != 'HP' and f'lesson_{kind.lower()}' in trigger:
                if trigger.endswith('_sp'):
                    return sp[attr]
                if trigger.endswith('_normal'):
                    return lessons[attr] - sp[attr]
                return lessons[attr]
        if 'end_lesson-lesson_sp' in trigger:
            return sum(sp.values())
        if 'end_lesson' in trigger:
            return sum(lessons.values())
        return next((n for prefix, n in self.opportunities.items() if prefix in trigger),
            3 if 'end_audition' in trigger else int('produce_start' in trigger))

    def score(self, own, rental):
        slots = [*own, rental]
        if len(slots) != 6 or not all(card.get('id') for card in slots):
            return None
        if len({card['id'] for card in slots}) != 6 or Counter(card['attribute'] for card in slots)[self.attribute] != 4:
            return None
        profiles = [self._profile(card) for card in slots]
        if any(p is None for p in profiles):
            return None
        lessons = {attr: 6 if attr == self.attribute else 0 for attr in ('Vo', 'Da', 'Vi')}
        skills = [skill for profile in profiles for skill in profile['skills']]
        sp = {}
        for kind, attr in ATTRS.items():
            if attr == 'HP':
                continue
            bonus = sum(s['value'] for s in skills
                if s['effect_type'] == f'ProduceEffectType_Lesson{kind}SpChangeRatePermilAddition') / 1000
            sp[attr] = lessons[attr] * min(1., max(0., .25+bonus))
        value = sum(sp.values()) * 45
        for profile in profiles:
            event_stats = sum(e['value'] for e in profile['events'] if e['type'] == 'parameter')
            value += event_stats
            for event in profile['events']:
                if event['type'] == 'points':
                    value += .4 * event['value']
                elif event['type'] == 'card':
                    value += self.event_cards.get(event['id'], 0)
            for skill in profile['skills']:
                effect = skill['effect_type'].removeprefix('ProduceEffectType_')
                amount = skill['value']
                count = self._triggers(skill.get('trigger_id') or '', lessons, sp)
                if skill.get('limit'):
                    count = min(count, skill['limit'])
                if re.fullmatch(r'(Vocal|Dance|Visual)Addition', effect):
                    value += amount * count
                elif (match := re.fullmatch(r'(Vocal|Dance|Visual)GrowthRateAddition', effect)):
                    base = self.opportunities.get('growth_' + ATTRS[match[1]])
                    if base is not None:
                        value += amount * base / 1000
                elif effect == 'SupportCardEventParameterAdditionValueUp':
                    value += amount * event_stats / 1000
                elif effect == 'SupportCardProduceCardUpgradeProbabilityUp':
                    value += amount * .008
                elif effect == 'LessonPresentProducePointUp':
                    value += amount * count * .016
                elif effect == 'ProducePointAdditionDisableTrigger':
                    value += amount * count * .4
                elif effect == 'StaminaRecoverFix':
                    value += amount * count * 1.5
                elif effect == 'MaxStaminaAddition':
                    value += amount * 3
                elif effect == 'ShopProduceDrinkPriceDiscountMultiple':
                    value += 30
        return value if isfinite(value) else None
