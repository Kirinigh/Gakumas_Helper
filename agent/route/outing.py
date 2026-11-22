"""Hiro outing contracts, with observed quotes and explicit reward instances.

Only the registered, identified offer may be received. Ordinal preference is
separate from physical observation; it never supplies a missing target or ID.
"""
from __future__ import annotations

import cv2

from .ledger import LedgerEffect


def read_outing_schedule(rows, *, selected, calendar=None):
    from .calendar import read_calendar, read_route_resources
    from .production import _row, _text
    if (calendar or read_calendar(rows)) != (3, 'outing'):
        raise ValueError('outing must start from its actual third-day calendar')
    outing = _row(rows, r'おでかけ', (190, 1020, 165, 105))
    # RestAuto checks the common rest control on the day-three activity screen.
    _text(rows, r'休む', (590, 845, 100, 80))
    selects = [row for row in rows if row['text'] == 'SELECT'
        and 1000 <= row['box'][1] <= 1130]
    if selected:
        target = _row(rows, 'SELECT', (200, 1060, 145, 70))
        if len(selects) != 1 or abs(target['box'][0] + target['box'][2]/2 - outing['box'][0] - outing['box'][2]/2) > 15:
            raise ValueError('outing SELECT is not attached to its original tile')
        target = outing
    else:
        if selects:
            raise ValueError('unbound outing preview cannot be selected again')
        target = outing
    return dict(read_route_resources(rows), scene='outing_selected' if selected else 'outing',
        day=3, target=tuple(target['box']), target_key='confirm_outing' if selected else 'select_outing')


def read_outing_options(rows, *, selected):
    from .calendar import read_route_resources
    from .production import _row, _text
    _text(rows, r'おでかけ', (45, 25, 150, 85))
    # Event dialogue varies. The approved first-option policy is identified
    # by its actual row and -50P quote, not one retained story's wording.
    # This only opens a preview; execution still needs every displayed effect.
    # Right-edge costs and trouble badges describe other options. They do
    # not identify the configured first option or its preview target.
    captions = [_row([row for row in rows if row['box'][0] < 300],
        r'\S(?:.*\S)?', (60, y, 615, 80)) for y in (650, 750, 850)]
    resources = read_route_resources(rows)
    if selected:
        # The displayed, capped stamina increase is the exact quote. The
        # percentage text cannot alone supply rounding or cap semantics.
        _text(rows, r'最大体力の60%分回復', (130, 430, 400, 80))
        _text(rows, r'Pドリンクを選択して獲得', (130, 485, 400, 60))
        _text(rows, r'強化済みのスキルカードを選択して獲得', (130, 530, 530, 55))
        _text(rows, r'スキルカードを選択して獲得', (130, 575, 430, 55))
        cost = int(_text(rows, r'-50', (160, 95, 120, 65)))
        recovery = int(_text(rows, r'[+]\d{1,2}', (165, 40, 120, 60)))
        if not 0 <= recovery <= resources['max_stamina'] - resources['stamina'] or (
                recovery == 0 and resources['stamina'] != resources['max_stamina']):
            raise ValueError('outing recovery quote is outside its observed capacity')
        quote = dict(currency_delta=cost, stamina_delta=recovery, drink_rewards=1, card_rewards=2)
    else:
        _text(rows, r'-50', (540, 620, 135, 85))
        quote = None
    if resources['currency'] < 50:
        raise ValueError('outing first option is not affordable')
    return dict(resources, day=3, scene='outing_choice_preview' if selected else 'outing_choice',
        target=tuple(captions[0]['box']), target_key='execute_outing' if selected else 'preview_outing_first', quote=quote,
        option_captions=[dict(row) for row in captions], option_labels=[row['text'] for row in captions])


def outing_offer_boxes(image, *, drink=False):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('outing offer capture is uncalibrated')
    mask = (image[790:1000, 85:635].min(2) < 180).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    # Bottle artwork can form a second large component inside its own tile.
    # Count the outer controls with the registered dimensions and fill.
    tiles = [row for row in stats[1:] if 126 <= row[2] <= 130 and 126 <= row[3] <= 130
        and row[4] >= row[2]*row[3]*(.15 if drink else .2)]
    if len(tiles) != 3:
        raise ValueError('outing requires three complete current offer tiles')
    boxes = sorted((tuple(map(int, row[:4])) for row in tiles))
    boxes = tuple((x + 85, y + 790, width, height) for x, y, width, height in boxes)
    if any(abs(box[0] - x) > 2 or abs(box[1] - 821) > 2
            for box, x in zip(boxes, (158, 296, 434), strict=True)):
        raise ValueError('outing offers are outside the complete registered row')
    return boxes


def outing_effect(state, key, request):
    if state.day != 3 or request.get('run_id') != state.run_id or request.get('day') != 3:
        raise ValueError('outing action lacks its original same-run activity')
    phases = {('schedule', 'select_outing'): 'outing_selected',
        ('outing_selected', 'confirm_outing'): 'outing_choice',
        ('outing_choice', 'preview_outing_first'): 'outing_choice_preview'}
    kwargs = {}
    if (state.phase, key) in phases:
        phase = phases[state.phase, key]
    elif state.phase == 'outing_choice_preview' and key == 'execute_outing':
        quote = request.get('quote') or {}
        if (quote.get('currency_delta') != -50 or type(quote.get('stamina_delta')) is not int
                or quote['stamina_delta'] < 0 or state.stamina + quote['stamina_delta'] > state.max_stamina
                or quote['stamina_delta'] == 0 and state.stamina != state.max_stamina
                or quote.get('drink_rewards') != 1 or quote.get('card_rewards') != 2):
            raise ValueError('outing has no complete affordable original reward quote')
        kwargs.update(currency_delta=-50, stamina_delta=quote['stamina_delta'])
        phase = 'outing_drink_offer'
    else:
        raise ValueError('outing action has no bound source stage')
    return LedgerEffect(f'{state.run_id}:{state.revision}:outing:{key}', key, phase=phase, **kwargs)
