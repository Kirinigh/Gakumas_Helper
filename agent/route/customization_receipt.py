"""Settle one reserved customization from its toast, full census and menu.

No action or retry is issued here. The existing ledger remains pending until
every independently observed card, drink, resource and capacity matches.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from collections import defaultdict
from dataclasses import replace, dataclass

from .engine import has_single_frame_id
from .ledger import HifLedger
from .customization import CustomizationMenu, CustomizationResult, plan_customization


@dataclass(frozen=True)
class CustomizationAfterMenu:
    run_id: str
    day: int
    game_pid: int
    observed_at_ms: int
    frame_ids: tuple[str, ...]
    menu: CustomizationMenu
    currency: int
    stamina: int
    max_stamina: int
    drinks: tuple


def _frames(keys, *, game_pid, after, before=None):
    if not has_single_frame_id(keys):
        raise ValueError('customization receipt needs one current frame')
    captures = [re.fullmatch(r'route-frame-' + str(game_pid) + r'-([0-9]+)', key)
        if isinstance(key, str) else None for key in keys]
    if not all(captures):
        raise ValueError('customization receipt frames belong to a different game')
    times = [int(match[1]) for match in captures]
    if times != sorted(times) or times[0] <= after or before is not None and times[-1] > before:
        raise ValueError('customization receipt frames precede their original submission')
    return times


def _pending(request, ledger):
    pending = ledger.pending
    if (not isinstance(request, dict) or request.get('schema_version') != 1
            or pending is None or pending.effect.target_key != request.get('target_key')
            or pending.effect.transaction_id != request.get('transaction_id')
            or pending.before.run_id != request.get('run_id') or pending.before.day != request.get('day')
            or pending.before.revision != request.get('source_revision')
            or json.loads(json.dumps(ledger.record()['pending']['effect'])) != request.get('effect')
            or list(pending.source_frame_ids) != request.get('source_frame_ids')
            or len(pending.effect.remove_cards) != 1 or len(pending.effect.add_cards) != 1
            or pending.effect.remove_cards[0].instance_id != request.get('card_instance_id')
            or pending.effect.remove_cards[0].card_id != request.get('card_id')):
        raise ValueError('customization receipt differs from its original pending transaction')
    return pending


def begin_customization_receipt(ledger, observation, menu, *, catalog, game_pid, issued_at_ms):
    pending = ledger.pending
    if pending is None or type(game_pid) is not int or game_pid <= 0 or type(issued_at_ms) is not int:
        raise ValueError('customization receipt requires the original reserved execution')
    baseline = HifLedger(pending.before, card_catalog=ledger.card_catalog, drink_ids=ledger.drink_ids,
        settled=ledger.settled)
    planned = plan_customization(observation, baseline, menu, catalog)
    if (planned.effect != pending.effect or planned.decision.proposal is None
            or planned.decision.proposal.target.kind != 'execute_customization'
            or issued_at_ms < observation.revision
            or issued_at_ms - observation.revision + observation.capture_age_ms > 1000
            or sum(card.card_id == menu.card_id for card in pending.before.cards) != 1):
        raise ValueError('customization receipt lacks its unique exact original execution')
    _frames(pending.source_frame_ids, game_pid=game_pid, after=-1, before=issued_at_ms)
    return dict(schema_version=1, stage='awaiting_result', run_id=pending.before.run_id,
        day=pending.before.day, game_pid=game_pid, transaction_id=pending.effect.transaction_id,
        target_key=pending.effect.target_key, source_revision=pending.before.revision,
        card_instance_id=menu.card_instance_id, card_id=menu.card_id, issued_at_ms=issued_at_ms,
        source_frame_ids=list(pending.source_frame_ids), effect=json.loads(json.dumps(ledger.record()['pending']['effect'])))


def confirm_customization_success(request, ledger, result):
    _pending(request, ledger)
    if (request.get('stage') != 'awaiting_result' or not isinstance(result, CustomizationResult)
            or (result.card_instance_id, result.card_id) != (request['card_instance_id'], request['card_id'])):
        raise ValueError('customization success does not name its original physical card')
    _frames(result.frame_ids, game_pid=request['game_pid'], after=request['issued_at_ms'])
    value = deepcopy(request)
    value.update(stage='census', success_frame_ids=list(result.frame_ids))
    return value


def resolve_customization_receipt(request, ledger, inventory, after_menu, *, catalog, now_ms):
    pending = _pending(request, ledger)
    if (request.get('stage') != 'census' or not isinstance(after_menu, CustomizationAfterMenu)
            or after_menu.run_id != request['run_id'] or after_menu.day != request['day']
            or after_menu.game_pid != request['game_pid'] or type(now_ms) is not int
            or type(after_menu.observed_at_ms) is not int
            or not 0 <= now_ms - after_menu.observed_at_ms <= 1000):
        raise ValueError('customization result needs its fresh original current menu')
    success = request.get('success_frame_ids')
    _frames(success, game_pid=request['game_pid'], after=request['issued_at_ms'])
    ticks = _frames(after_menu.frame_ids, game_pid=request['game_pid'], after=request['issued_at_ms'],
        before=after_menu.observed_at_ms)
    if now_ms - ticks[0] > 1000 or set(after_menu.frame_ids) & set(success):
        raise ValueError('customization post-census menu is not independent and current')
    menu, expected = after_menu.menu, pending.expected
    changed = pending.effect.add_cards[0]
    if (not isinstance(menu, CustomizationMenu)
            or (menu.card_instance_id, menu.card_id) != (changed.instance_id, changed.card_id)
            or any(type(value) is not int for value in (after_menu.currency, after_menu.stamina, after_menu.max_stamina))
            or (after_menu.currency, after_menu.stamina, after_menu.max_stamina)
                != (expected.currency, expected.stamina, expected.max_stamina)
            or after_menu.drinks != expected.drinks):
        raise ValueError('customization post-census resources or physical card changed')
    remaining = dict(menu.remaining)
    rules = catalog.rules(changed.card_id)
    if (len(remaining) != len(menu.remaining) or set(remaining) != set(rules)
            or any(type(value) is not int or value != rules[key][0] - dict(changed.customization).get(key, 0)
                for key, value in remaining.items())
            or type(menu.total_remaining) is not int
            or menu.total_remaining != dict(expected.counters).get('custom_remaining:' + changed.instance_id)):
        raise ValueError('customization post-census counts differ from the reserved single change')
    if (not isinstance(inventory, dict) or inventory.get('census_id') != request['transaction_id']
            or inventory.get('complete') is not True or type(inventory.get('total')) is not int
            or inventory['total'] != expected.card_count or not isinstance(inventory.get('slots'), list)
            or len(inventory['slots']) != expected.card_count
            or [row.get('index') for row in inventory['slots']] != list(range(expected.card_count))):
        raise ValueError('customization requires its complete physical card census')
    groups = defaultdict(list)
    for card in expected.cards:
        groups[(card.card_id, card.upgraded, card.customization)].append(card)
    observed = []
    census_times = []
    source_frames = set(request['source_frame_ids']) | set(success)
    for row in inventory['slots']:
        census_times.extend(_frames(row.get('frame_ids'), game_pid=request['game_pid'], after=request['issued_at_ms']))
        if source_frames & set(row['frame_ids']):
            raise ValueError('customization census reused an old source or toast')
        entry = ledger.card_catalog.entries.get(row.get('card_id'))
        custom = row.get('customizations')
        if (entry is None or not isinstance(custom, dict)
                or any(not isinstance(key, str) or type(value) is not int or value <= 0 for key, value in custom.items())
                or row.get('presence') not in ('CONFIDENT_ZERO', 'DETAIL_CANDIDATE')
                or row['presence'] == 'CONFIDENT_ZERO' and custom
                or row['presence'] == 'DETAIL_CANDIDATE' and row.get('detail', {}).get('nonbattle_schema') != 1):
            raise ValueError('customization census has an unresolved permanent variant')
        if row['presence'] == 'DETAIL_CANDIDATE':
            census_times.extend(_frames(row['detail'].get('frame_ids'), game_pid=request['game_pid'], after=request['issued_at_ms']))
            if source_frames & set(row['detail']['frame_ids']):
                raise ValueError('customization census reused an old detail')
        key = (row['card_id'], entry.upgraded, tuple(sorted(custom.items())))
        if not groups[key]:
            raise ValueError('customization census changed another card or omitted its single upgrade')
        observed.append(groups[key].pop(0))
    if any(groups.values()):
        raise ValueError('customization census lost an original owned instance')
    if not census_times or ticks[0] <= max(census_times):
        raise ValueError('customization return menu does not follow its independent full census')
    # Populate business fields from current observations, keeping only the
    # reserved transaction's revision and unchanged phase/counter identities.
    counters = dict(expected.counters)
    counters['custom_remaining:' + changed.instance_id] = menu.total_remaining
    receipt = replace(expected, cards=tuple(sorted(observed, key=lambda card: card.instance_id)),
        card_count=inventory['total'], currency=after_menu.currency, stamina=after_menu.stamina,
        max_stamina=after_menu.max_stamina, drinks=after_menu.drinks, drink_count=len(after_menu.drinks),
        counters=tuple(sorted(counters.items())))
    settled = ledger.confirm(receipt, transaction_id=request['transaction_id'], frame_ids=after_menu.frame_ids)
    value = deepcopy(request)
    value.update(stage='complete', census_id=inventory['census_id'], receipt_frame_ids=list(after_menu.frame_ids),
        receipt_revision=settled.state.revision)
    return settled, value
