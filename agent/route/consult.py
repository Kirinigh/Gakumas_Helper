"""Default Hiro consultation: observe free entry rewards, then leave the shop."""
from __future__ import annotations

from .ledger import LedgerEffect


def read_consult_schedule(rows, *, selected, calendar=None):
    from .calendar import read_calendar, read_route_resources
    from .production import _row
    if (calendar or read_calendar(rows)) != (6, 'consult_entry'):
        raise ValueError('consultation must start from its actual sixth-day calendar')
    # A free navigation caption uses the route's existing .85 caption
    # threshold, not the .95 business-value gate. Rest is not the target.
    caption = _row(rows, r'相談', (290, 1020, 135, 105), minimum=.85)
    selects = [row for row in rows if row['text'] == 'SELECT' and row['score'] >= .95]
    if selected:
        _row(rows, r'SELECT', (285, 1060, 150, 75))
        if len(selects) != 1:
            raise ValueError('consultation selection is ambiguous')
    elif selects:
        raise ValueError('unbound consultation preview cannot be selected again')
    return dict(read_route_resources(rows), day=6, kind='consult_selected' if selected else 'consult_entry',
        target=tuple(caption['box']), target_key='confirm_consult' if selected else 'select_consult', quote={})


def read_consult_reward_quote(rows):
    import re

    from .production import _text
    gifts = int(re.fullmatch(r'ランダムな特定のPドリンクを(\d)つ獲得', _text(rows,
        r'ランダムな特定のPドリンクを\dつ獲得', (155, 575, 380, 50)))[1])
    points = int(_text(rows, r'Pポイント[+]\d{1,4}', (150, 605, 250, 45)).split('+')[1])
    if not 1 <= gifts <= 4: raise ValueError('consult free reward count is outside four physical slots')
    return dict(drink_rewards=gifts, currency_delta=points)


def read_consult_shop(rows, image):
    from .home import _unique_home_template
    from .calendar import read_route_resources
    from .production import _text
    from .activity_production import obtained_receipt_visible
    _text(rows, r'相談', (45, 15, 150, 85))
    if getattr(image, 'shape', ()) != (1280, 720, 3): raise ValueError('consult shop capture is uncalibrated')
    if obtained_receipt_visible(image):
        raise ValueError('consult shop is covered by an obtained reward')
    if any('ランダムな特定のPドリンク' in row['text'] or 'Pポイント+' in row['text']
            or row['box'][1] >= 625 and row['box'][1] < 750 and 240 < row['box'][0] < 470
            and row['text'].startswith('+') for row in rows):
        raise ValueError('consult shop still contains its free P reward overlay')
    # fb8fd00c HIF.json ProduceHIF__ProduceHIFButton: use only the shop
    # exit from its button alternatives, in its original ROI at default .7.
    end, _ = _unique_home_template(image, 'produce/shop_exit.png', (0, 780, 720, 500), .7)
    return dict(read_route_resources(rows), day=6, kind='consult_shop', target=tuple(end['box']), target_key='finish_consult')


def consult_effect(state, key, *, single_entry=False):
    if state.day != 6: raise ValueError('consultation has no bound current sixth day')
    if state.phase == 'schedule' and key == 'select_consult': phase = 'consult_rewards' if single_entry else 'consult_selected'
    elif state.phase == 'consult_selected' and key == 'confirm_consult': phase = 'consult_rewards'
    elif state.phase == 'consult_shop' and key == 'finish_consult':
        return LedgerEffect(f'{state.run_id}:{state.revision}:finish-consult', key, day=7, phase='interval')
    else: raise ValueError('consultation action is outside its original default source stage')
    return LedgerEffect(f'{state.run_id}:{state.revision}:{key}', key, phase=phase)
