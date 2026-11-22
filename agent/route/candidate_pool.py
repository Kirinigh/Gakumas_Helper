"""Free Hiro-style candidate replacement under the current reward reservation."""
from __future__ import annotations

import re
from copy import deepcopy

import cv2
import numpy as np

from .ownership import locked_hif_record
from .settings_navigation import NonbattleOverlay, pending_overlay

KEY = 'candidate_pool_navigation'
BUTTONS = {'reroll': '再抽選', 'exclude': 'カード除去'}
TITLES = {'reroll': 'スキルカード再抽選確認', 'exclude': 'スキルカード除去確認'}
CONFIRM = {'reroll': '再抽選する', 'exclude': '除去する'}


class CandidatePoolObserved(ValueError):
    def __init__(self, request):
        super().__init__('current candidate pool has a configured free action')
        self.request = request


def candidate_policy(context):
    reader = getattr(context, 'get_node_data', None)
    attach = ((reader('ProduceHifCandidatePoolSettings') or {}).get('attach') or {}) if callable(reader) else {}
    raw = attach.get('preferred', '')
    if raw == '':
        return (), ()
    if not isinstance(raw, str) or re.fullmatch(r'[1-9]\d*(?:,[1-9]\d*)*', raw) is None:
        raise ValueError('preferred reward cards must be comma-separated card IDs')
    order = attach.get('order', 'reroll,exclude')
    if order not in ('reroll,exclude', 'exclude,reroll'):
        raise ValueError('invalid candidate pool preference')
    return tuple(raw.split(',')), tuple(order.split(','))


def preferred_index(offers, preferred, default):
    for card_id in preferred:
        for index, row in enumerate(offers):
            if row.get('card_id') == card_id:
                return index
    return default


def _control(rows, text, *, side=None):
    boxes = [row['box'] for row in rows if row['score'] >= .85 and row['box'][1] >= 990
        and row['text'].lstrip('→✓く') == text and (side is None or (row['box'][0] >= 350) == side)]
    return boxes[0] if len(boxes) == 1 else None


def _capacity(rows, kind):
    counts = []
    for row in rows:
        x, y, _, _ = row['box']
        if row['score'] < .85 or not 990 <= y < 1096 or (x >= 350) != (kind == 'reroll'):
            continue
        match = re.fullmatch(r'(?:あと|残り)(\d+)回', row['text'])
        if match is not None:
            counts.append(int(match[1]))
    return counts[0] if len(counts) == 1 else None


def _art(image, boxes):
    result = []
    for x, y, width, height in boxes:
        patch = image[y+18:y+height-18, x+18:x+width-18]
        result.append(cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (16, 12)).tolist())
    return result


def pending_candidate_pool(record):
    state = pending_overlay(record, KEY) or {}
    return state if state.get('active') else None


def maybe_edit_pool(host, context, image, rows, offers, *, family, default):
    """Interrupt only an explicitly configured and currently proven free offer."""
    if (getattr(context, 'gkh_hif_cultivation_runtime', None) is None
            or host.hooks.transition_gate.pending is not None or not _offer_visible(rows)):
        return
    preferred, order = candidate_policy(context)
    if not preferred or not all(offer['card_id'] is not None for offer in offers) or any(
            offer['card_id'] in preferred for offer in offers):
        return
    ledger = host.ledger.state
    ticket = [family, ledger.day, ledger.revision]
    saved = pending_overlay({'handoff': host.handoff}, KEY) or {}
    earlier = saved.get('history', ())
    if sum(item['ticket'] == ticket for item in earlier) >= 5:
        return
    boxes = [offer['box'] for offer in offers]
    cards = [offer['card_id'] for offer in offers]
    for kind in order:
        if any(item['ticket'] == ticket and item['kind'] == kind and item['result'] != 'changed' for item in earlier):
            continue
        capacity = _capacity(rows, kind)
        if capacity and _control(rows, BUTTONS[kind], side=kind == 'reroll') is not None:
            raise CandidatePoolObserved(dict(ticket=ticket, kind=kind, capacity=capacity,
                boxes=boxes, before=cards, art=_art(image, boxes), target=default))


def _offer_visible(rows):
    return any(row['score'] >= .85 and 445 <= row['box'][1] <= 750
        and ('受け取るスキルカード' in row['text'] or '入れ替えるスキルカード' in row['text']
             or 'スキルカードを選択' in row['text'] or 'スキルカードを選んでください' in row['text']) for row in rows)


def _dialog(rows, kind):
    return any(row['score'] >= .85 and TITLES[kind] in row['text'] for row in rows)


def _has_cost(rows):
    return any(re.search(r'ジュエル|有償|アイテムを消費|APを消費|有料', row['text']) for row in rows)


class CandidatePoolNavigation(NonbattleOverlay):
    key = KEY

    def _finish(self, state, request, result):
        state['history'].append(dict(ticket=request['ticket'], kind=request['kind'],
            before=request['before'], initial_capacity=request['capacity'], result=result,
            inputs=state['steps']))
        state.update(active=None, steps=[], input=None)
        self._save(state)
        return state['history'][-1]

    def run(self, request=None):
        from .exchange import require_selection_corners

        with locked_hif_record(self.runtime.path):
            self.record = self.runtime._read()
            if self.record['ownership']['owner'] != 'task600' or self.record.get('ledger') is None:
                raise ValueError('candidate pool requires its original nonbattle owner and ledger')
            saved = deepcopy(pending_overlay(self.record, self.key)) or dict(history=[], active=None, input=None, steps=[])
            request = saved['active'] or request
            state = self.record['ledger']['state']
            if request is None or request['ticket'][1:] != [state['day'], state['revision']]:
                raise ValueError('candidate pool does not belong to the active reward')
            kind = request['kind']
            saved['active'] = request
            parent = self.context.gkh_hif_route_host
            self.context.gkh_hif_route_host = self
            try:
                deadline = self.runtime.clock.now() + 23_000
                while self.runtime.clock.now() < deadline:
                    at, image, rows = self._screen()
                    previous = saved.get('input')
                    steps = saved['steps']
                    last = previous or (steps[-1] if steps else None)
                    dialog = _dialog(rows, kind)
                    visible = _offer_visible(rows)
                    before = np.asarray(request['art'], dtype='float32')
                    if visible and not dialog and last is not None and last['control'] == 'cancel':
                        if previous is not None:
                            steps.append(previous)
                        return self._finish(saved, request, 'paid_cancelled')
                    if visible and not dialog and last is not None and last['control'] in ('open', 'confirm'):
                        art = np.asarray(_art(image, request['boxes']), dtype='float32')
                        remaining = _capacity(rows, kind)
                        if (remaining is not None and remaining < request['capacity']
                                or np.abs(art - before).mean(axis=(1, 2)).max() > 12):
                            if previous is not None:
                                steps.append(previous)
                            return self._finish(saved, request, 'changed')
                    if previous is not None:
                        # A still-visible confirmation after a submission is
                        # evidence to wait, never an instruction to repeat it.
                        control = previous['control']
                        next_surface = control == 'open' and dialog
                        if visible and control == 'select':
                            try:
                                require_selection_corners(image, request['boxes'][request['target']])
                            except ValueError:
                                pass
                            else:
                                next_surface = True
                        if not next_surface:
                            self.sleep(.2)
                            continue
                        steps.append(previous)
                        saved['input'] = None
                        self._save(saved)
                    if dialog:
                        paid = _has_cost(rows)
                        box = _control(rows, 'キャンセル' if paid else CONFIRM[kind], side=not paid)
                        action = 'cancel' if paid else 'confirm'
                    elif visible:
                        if _capacity(rows, kind) != request['capacity']:
                            if not steps:
                                return self._finish(saved, request, 'unavailable')
                            self.sleep(.2)
                            continue
                        if not steps:
                            art = np.asarray(_art(image, request['boxes']), dtype='float32')
                            if np.abs(art - before).mean(axis=(1, 2)).max() > 12:
                                return self._finish(saved, request, 'unavailable')
                        if kind == 'exclude' and not steps:
                            box, action = request['boxes'][request['target']], 'select'
                        else:
                            box = _control(rows, BUTTONS[kind], side=kind == 'reroll')
                            if kind == 'exclude' and box is not None:
                                button = cv2.cvtColor(image[1080:1114, 40:190], cv2.COLOR_BGR2HSV)
                                if float(np.median(button[:, :, 2])) < 210:
                                    box = None
                            action = 'open'
                    else:
                        box = None
                    if box is None:
                        self.sleep(.2)
                        continue
                    self._click(saved, action, box, at)
                    self.sleep(.8 if action in ('open', 'confirm') else .4)
                raise ValueError('candidate pool operation has no verified fresh result')
            finally:
                self.context.gkh_hif_route_host = parent
