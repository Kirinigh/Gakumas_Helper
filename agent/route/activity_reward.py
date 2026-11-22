"""One original activity reservation through free rewards and a fresh census.

Child inputs may only finish the already quoted activity or inspect inventory.
They cannot start another paid operation, buy, consume, delete or customize.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from types import SimpleNamespace
from collections import Counter, defaultdict
from dataclasses import replace

from .ledger import HifLedger, CardInstance, LedgerEffect, DrinkInstance, PartialDeckLedgerState

KINDS = frozenset({'activity_select_drink', 'activity_receive_drink', 'activity_ack_drink',
    'activity_ack_support_card', 'activity_ack_support_ok',
    'activity_select_card', 'activity_receive_card', 'activity_ack_card', 'activity_ack_free_drink',
    'activity_inspect_card', 'activity_inspect_drink',
    'activity_open_drink_detail', 'activity_close_drink_detail',
    'recovery_inventory', 'recovery_inventory_detail', 'recovery_inventory_scroll', 'recovery_close'})


def support_source(pending):
    if pending is None:
        return None
    target = pending.target
    if (target.kind == 'lesson' and target.key == 'lesson:Vi'
            and (pending.source.day, pending.source.scene) == (1, 'lesson')):
        return target.key
    if target.kind in ('confirm_lesson', 'dismiss_exchange_result') and target.key == target.kind:
        return target.key
    return None


def observed_lesson_grants(value):
    """Plain card awards identify each addition on the way into the first classroom."""
    return (value['kind'] == 'support' and value.get('source_target') == 'lesson:Vi'
        and all(grant.get('kind') != 'upgrade' for grant in [*value['cards'], value['current_grant']]))


def request(host, *, ledger=None):
    value = (host.handoff or {}).get('nonbattle_context', {}).get('activity_reward')
    if value is None or isinstance(value, dict) and value.get('stage') in ('complete', 'cancelled'):
        return None
    ledger = host.ledger if ledger is None else ledger
    pending = ledger.pending if ledger is not None else None
    owner = host.ownership or {}
    if (not isinstance(value, dict) or value.get('schema_version') != 1 or pending is None
            or value.get('kind') not in ('outing', 'consult', 'support')
            or pending.effect.target_key != (value.get('source_target', {'outing': 'execute_outing', 'consult': 'confirm_consult'}[value['kind']])
                if value['kind'] != 'support' else value.get('source_target', 'confirm_lesson'))
            or value['kind'] == 'consult' and value.get('source_target','confirm_consult') not in ('select_consult','confirm_consult')
            or value['kind'] == 'consult' and pending.before.phase != (
                'schedule' if value.get('source_target') == 'select_consult' else 'consult_selected')
            or value['kind'] == 'support' and pending.effect.target_key not in ('lesson:Vi', 'confirm_lesson', 'dismiss_exchange_result')
            or pending.effect.target_key == 'lesson:Vi' and (pending.before.day, pending.before.phase) != (1, 'schedule')
            or value.get('run_id') != host.run_id or value.get('game_pid') != host.game_pid
            or value.get('ownership_revision') != owner.get('revision')
            or (owner.get('owner'), owner.get('phase')) != ('task600', 'nonbattle')
            or owner.get('game_pid') != host.game_pid
            or value.get('transaction_id') != pending.effect.transaction_id
            or value.get('source_revision') != pending.before.revision
            or value.get('day') != pending.before.day or value.get('source_frame_ids') != list(pending.source_frame_ids)
            or pending.before.day not in ((1, 4) if value['kind'] == 'support' else (3,) if value['kind'] == 'outing' else (6,))):
        raise ValueError('activity reward does not match its original reservation and owner')
    return value


def completed_consult(host):
    """A settled free-entry activity is the sole cold shop continuation source."""
    value = (host.handoff or {}).get('nonbattle_context', {}).get('activity_reward')
    ledger, owner = host.ledger, host.ownership or {}
    if (not isinstance(value, dict) or value.get('stage') != 'complete' or value.get('kind') != 'consult'
            or value.get('schema_version') != 1 or value.get('run_id') != host.run_id
            or value.get('game_pid') != host.game_pid or owner.get('game_pid') != host.game_pid
            or value.get('ownership_revision') != owner.get('revision')
            or (owner.get('owner'), owner.get('phase')) != ('task600', 'nonbattle')
            or ledger is None or ledger.state.day != 6 or ledger.state.phase != 'consult_shop'
            or ledger.state.revision != value.get('source_revision', -2) + 1
            or not ledger.settled or ledger.settled[-1] != value.get('transaction_id')
            or value.get('resolution', {}).get('kind') != 'consult_shop'):
        raise ValueError('consult shop lacks its original completed free-entry activity')
    frames(value['resolution'], value['source_frame_ids'])
    return value


def permits_navigation(host, ledger, gate):
    value = request(host, ledger=ledger)
    if value is None: return False
    if gate.pending is None: return value['stage'] != 'paid_pending'
    return (gate.pending.target.kind in KINDS and gate.pending.source.day in (value['day'], value['census_day'])
        and (re.fullmatch(r'inventory-detail:\d+', gate.pending.target.key) is not None
            if gate.pending.target.kind == 'recovery_inventory_detail' else
            re.fullmatch(r'inventory-scroll:\d+', gate.pending.target.key) is not None
            if gate.pending.target.kind == 'recovery_inventory_scroll' else gate.pending.target.key == gate.pending.target.kind))


def with_request(handoff, value):
    result = dict(handoff or {})
    context = dict(result.get('nonbattle_context') or {})
    context['activity_reward'] = deepcopy(value)
    result['nonbattle_context'] = context
    return result


def start_request(host, ledger, gate):
    pending, proof = ledger.pending, host._page.activity
    if pending is None or gate.pending is None or not isinstance(proof, dict):
        raise ValueError('activity needs its original quoted source page')
    outing = pending.effect.target_key == 'execute_outing'
    single_entry = pending.effect.target_key == 'select_consult'
    kind = 'outing' if outing else 'consult'
    quote = proof.get('quote') or {}
    if (pending.before.day != (3 if outing else 6)
            or gate.pending.target.kind != pending.effect.target_key
            or proof.get('kind') != ('outing_choice_preview' if outing else 'consult_entry' if single_entry else 'consult_selected')
            or pending.effect.currency_delta != (-50 if outing else 0)
            or not outing and (pending.effect.target_key not in ('select_consult','confirm_consult')
                or pending.effect.stamina_delta or pending.effect.max_stamina_delta)
            or pending.effect.remove_cards or pending.effect.add_cards or pending.effect.remove_drinks
            or pending.effect.add_drinks or pending.effect.counter_deltas
            or outing and (quote.get('currency_delta') != -50 or quote.get('stamina_delta') != pending.effect.stamina_delta
                or quote.get('drink_rewards') != 1 or quote.get('card_rewards') != 2)):
        raise ValueError('activity original cost or reward contract is incomplete')
    return dict(schema_version=1, kind=kind, stage='paid_pending', run_id=host.run_id, game_pid=host.game_pid,
        ownership_revision=host.ownership['revision'], transaction_id=pending.effect.transaction_id,
        source_revision=pending.before.revision, day=pending.before.day, census_day=4 if outing else 6,
        source_frame_ids=list(pending.source_frame_ids), reservation=deepcopy(ledger.record()['pending']),
        quote=deepcopy(quote), cards=[], drinks=[], inventory=None, chosen=None, offers=None,
        drink_identities={}, inspecting_slot=None, inspecting_offer=None,
        **({'source_target':'select_consult'} if single_entry else {}))


def original_offer_slot(value, proof, index):
    original, current = value.get('offers'), proof.get('offers')
    if (type(index) is not int or index not in range(3)
            or not isinstance(original, list) or not isinstance(current, list)
            or len(original) != 3 or len(current) != 3):
        raise ValueError('activity reward lacks its original physical slot')
    old, new = original[index], current[index]
    if (not isinstance(old, dict) or not isinstance(new, dict)
            or old.get('index') != index or new.get('index') != index or not old.get('box') or old['box'] != new.get('box')):
        raise ValueError('activity reward target moved from its original physical slot')
    return old, new


def _validate_support_grant(host, grant):
    entry = host._catalog.entries.get(grant.get('card_id'))
    if entry is None or grant.get('upgraded') is not entry.upgraded:
        raise ValueError('support grant version is unresolved')
    if grant.get('kind') == 'upgrade':
        before = host.ledger.pending.before
        if isinstance(before, PartialDeckLedgerState) and grant.get('source_instance_id') is None:
            source = host._catalog.entries.get(grant.get('source_card_id'))
            if source is None or source.upgraded or not entry.upgraded or entry.name != source.name + '+':
                raise ValueError('support upgrade has no observed base and upgraded variant')
            return
        originals = [card for card in host.ledger.pending.before.cards if card.card_id == grant.get('source_card_id')
            and not card.upgraded and not card.customization]
        if (len(originals) != 1 or originals[0].instance_id != grant.get('source_instance_id')
                or not entry.upgraded or entry.name != host._catalog.entries[originals[0].card_id].name + '+'):
            raise ValueError('support upgrade lacks its exact original permanent instance')


def start_support_request(host, page):
    from .checkpoint import transition_checkpoint
    from .input_receipt import original_input_report

    pending, gate, proof = host.ledger.pending, host.hooks.transition_gate, page.activity
    target = support_source(gate.pending)
    ending = target == 'dismiss_exchange_result'
    phases = {'dismiss_exchange_result': 'schedule', 'confirm_lesson': 'lesson_choice', 'lesson:Vi': None}
    if (pending is None or target is None
            or page.baseline != pending.before
            or pending.effect.target_key != target or pending.before.day not in (1, 4)
            or target == 'lesson:Vi' and (pending.before.day, pending.before.phase) != (1, 'schedule')
            or pending.effect.phase != phases[target]
            or pending.effect.day not in ((pending.before.day + 1,) if ending else (None, pending.before.day))
            or proof.get('kind') != 'support_reward' or proof.get('day') != pending.before.day
            or page.observation.scene != 'support_reward'
            or pending.effect.remove_cards or pending.effect.add_cards or pending.effect.remove_drinks
            or pending.effect.add_drinks or pending.effect.counter_deltas
            or pending.effect.currency_delta or pending.effect.stamina_delta or pending.effect.max_stamina_delta
            or tuple(proof.get(key) for key in ('stamina', 'max_stamina', 'currency')) != (
                pending.before.stamina, pending.before.max_stamina, pending.before.currency)):
        raise ValueError('support grant lacks its original zero-cost submitted lesson')
    if tuple(frames(proof, pending.source_frame_ids)) != page.frame_ids:
        raise ValueError('support grant frames differ from its current page')
    grant = proof.get('grant') or {}
    _validate_support_grant(host, grant)
    original = original_input_report(host, gate)
    return dict(schema_version=1, kind='support', stage='support_reward', run_id=host.run_id,
        game_pid=host.game_pid, ownership_revision=host.ownership['revision'],
        transaction_id=pending.effect.transaction_id, source_revision=pending.before.revision,
        day=pending.before.day, census_day=pending.before.day + 1 if ending else pending.before.day,
        source_target=target, destination_scene='activity_home' if ending else 'support_lesson',
        source_frame_ids=list(pending.source_frame_ids),
        reservation=deepcopy(host.ledger.record()['pending']), original_transition=transition_checkpoint(gate, host.clock.epoch),
        original_result=original['id'], grant_frame_ids=list(proof['frame_ids']), current_grant=deepcopy(grant),
        quote={}, cards=[], drinks=[], inventory=None, chosen=None, offers=None,
        drink_identities={}, inspecting_slot=None, inspecting_offer=None)


def frames(proof, excluded):
    keys = proof.get('frame_ids')
    if (not isinstance(keys, list) or len(keys) != 1
            or any(not isinstance(key, str) or not key for key in keys) or set(keys).intersection(excluded)):
        raise ValueError('activity needs one new current frame')
    return keys


def validate_resources(host, proof):
    pending = host.ledger.pending
    minimum = pending.expected
    if (any(type(proof.get(key)) is not int for key in ('stamina', 'max_stamina', 'currency'))
            or not minimum.stamina <= proof['stamina'] <= proof['max_stamina']
            or proof['max_stamina'] < minimum.max_stamina or proof['max_stamina'] > 100
            or proof['currency'] < minimum.currency):
        raise ValueError('activity receipt contains an unquoted loss or incomplete resources')


def outing_rewards_received(value):
    return (value['kind'] == 'outing' and len(value['drinks']) == 1
            and sum(card.get('kind') != 'upgrade' for card in value['cards']) == 2)


def _support_grant(host, value, proof):
    grant = proof.get('grant')
    if not isinstance(grant, dict):
        raise ValueError('support reward has no observed grant')
    if value['kind'] == 'outing':
        _validate_support_grant(host, grant)
        if (not outing_rewards_received(value) or proof.get('day') != value['day'] or grant.get('kind') != 'upgrade'
                or any(grant['source_instance_id'] is not None and card.get('source_instance_id') == grant['source_instance_id']
                       or card.get('card_id') == grant['source_card_id'] for card in value['cards'])):
            raise ValueError('outing upgrade does not identify one unmodified permanent card after both rewards')
    return deepcopy(grant)


def _post_restore_consult_source(host, value, proof):
    """Census the actual free-entry gains; never invent an unseen quote."""
    from .checkpoint import restore_transition, transition_checkpoint
    from .input_receipt import original_input_report
    from .nonbattle_date import KEY, navigation_closed

    child = (host.handoff or {}).get('nonbattle_context', {}).get(KEY) or {}
    scope = child.get('business_scope') or {}
    canonical = lambda data: json.loads(json.dumps(data, allow_nan=False))
    pending = host.ledger.pending
    if (value['kind'] != 'consult' or proof['kind'] != 'consult_shop' or proof.get('quote')
            or pending.effect.currency_delta or pending.effect.stamina_delta or pending.effect.max_stamina_delta
            or pending.effect.remove_cards or pending.effect.add_cards or pending.effect.remove_drinks or pending.effect.add_drinks):
        raise ValueError('consult current census lacks its unchanged free parent')
    gate = host.hooks.transition_gate
    original = original_input_report(host, gate)
    if child:
        if (not navigation_closed(dict(handoff=host.handoff, transition=transition_checkpoint(gate, host.clock.epoch)))
                or scope.get('run_id') != host.run_id or scope.get('ownership') != host.ownership):
            raise ValueError('consult current census lacks a completed same-run navigation')
        # A successfully closed navigation can remain in the same save for
        # later days. Only a child of this outstanding transaction supplies
        # its resume receipt; an older settled action cannot own this census.
        old_parent = (scope.get('ledger') or {}).get('pending') or {}
        old_transaction = (old_parent.get('effect') or {}).get('transaction_id')
    if child and old_transaction not in host.ledger.settled:
        if (scope.get('ledger') != canonical(host.ledger.record())
                or scope.get('transition') != canonical(transition_checkpoint(gate, host.clock.epoch))
                or child.get('business_receipt', {}).get('observation', {}).get('scene') != 'consult_shop'):
            raise ValueError('consult current census lacks its unchanged free parent and completed same-save return')
        carrier = child.get('route_input_receipt') or {}
        gate = restore_transition(carrier.get('original_transition'), clock_epoch=host.clock.epoch)
        restored = original_input_report(SimpleNamespace(handoff=dict(nonbattle_context=dict(route_input_receipt=carrier)),
            run_id=host.run_id, game_pid=host.game_pid, clock=host.clock), gate)
        if gate.pending.target.kind != 'date_resume_saved' or gate.pending.source.day != value['day']:
            raise ValueError('consult free differences do not follow the original successful save resume')
        source = dict(source='post_restore_current_census', resume_result=restored['id'],
            resume_transition=deepcopy(carrier['original_transition']))
    else:
        if (gate.pending.target.kind != 'activity_ack_free_drink' or gate.pending.source.scene != 'consult_receipt'
                or gate.pending.source.day != value['day'] or not value['drinks']
                or value['drinks'][-1] != value.get('chosen')):
            raise ValueError('consult current census lacks its original successful free-drink acknowledgement')
        source = dict(source='free_entry_current_census', free_result=original['id'],
            free_transition=transition_checkpoint(gate, host.clock.epoch))
    slots, before = proof.get('drink_slots') or (), pending.before
    occupied = [row['slot'] for row in slots if row.get('state') == 'OCCUPIED']
    if (len(slots) != 4 or [row.get('slot') for row in slots] != list(range(4))
            or any(row.get('state') not in ('EMPTY', 'OCCUPIED') for row in slots)
            or not {drink.slot for drink in before.drinks} <= set(occupied)
            or (proof['stamina'], proof['max_stamina']) != (before.stamina, before.max_stamina)
            or proof['currency'] < before.currency):
        raise ValueError('consult restored census has unexplained resource or physical-slot losses')
    return dict(source, frame_ids=list(proof['frame_ids']),
        currency=proof['currency'], stamina=proof['stamina'], max_stamina=proof['max_stamina'],
        occupied_slots=occupied, previous_drink_receipts=deepcopy(value['drinks']))


def navigation_checkpoint(host, gate):
    value, proof = request(host), host._page.activity
    if value is None or not isinstance(proof, dict) or gate.pending.source != host._page.observation:
        raise ValueError('activity child input has no original current source')
    frames(proof, value['source_frame_ids'])
    kind, stage = gate.pending.target.kind, value['stage']
    allowed = {'drink_offer': ('activity_select_drink', 'activity_inspect_drink'),
        'drink_inspection': ('activity_select_drink', 'activity_inspect_drink', 'activity_receive_drink'), 'drink_selected': ('activity_receive_drink',),
        'support_reward': ('activity_ack_support_card',), 'support_ok': ('activity_ack_support_ok',),
        'drink_receipt': ('activity_ack_drink',), 'card_offer': ('activity_select_card', 'activity_inspect_card'),
        'card_inspection': ('activity_select_card', 'activity_inspect_card', 'activity_receive_card'),
        'card_selected': ('activity_receive_card',), 'card_receipt': ('activity_ack_card',),
        'consult_receipt': ('activity_ack_free_drink',)}
    if stage == 'census':
        if kind not in ('recovery_inventory', 'recovery_inventory_detail', 'recovery_inventory_scroll', 'recovery_close',
                'activity_open_drink_detail', 'activity_close_drink_detail'):
            raise ValueError('activity census cannot dispatch a resource operation')
    elif kind not in allowed.get(stage, ()):
        raise ValueError('activity child input is outside its original reward stage')
    if (value['kind'] in ('support', 'outing') and kind == 'activity_ack_support_card'
            and _support_grant(host, value, proof) != value.get('current_grant')):
        raise ValueError('support grant changed before acknowledgement')
    result = deepcopy(value)
    result['navigation_source_frame_ids'] = list(proof['frame_ids'])
    if proof.get('inventory') is not None:
        from .nonbattle_inventory import retain_inventory_registration
        result['inventory'] = retain_inventory_registration(host, host._page, proof['inventory'])
    if kind in ('activity_select_drink', 'activity_select_card', 'activity_inspect_card', 'activity_inspect_drink'):
        result['offers'] = deepcopy(proof['offers'])
    if kind in ('activity_inspect_card', 'activity_inspect_drink'):
        index = proof.get('inspect_offer')
        field = 'card_id' if kind == 'activity_inspect_card' else 'drink_id'
        if type(index) is not int or index not in range(3) or proof['offers'][index].get(field) is not None:
            raise ValueError('activity inspection must target an unresolved physical offer')
        result['inspecting_offer'] = index
    elif kind in ('activity_select_card', 'activity_select_drink'):
        index = proof.get('selected_offer_index')
        field = 'card_id' if kind == 'activity_select_card' else 'drink_id'
        if type(index) is not int or index not in range(3) or proof['offers'][index].get(field) is None:
            raise ValueError('activity reward selection lacks its physical identity')
        result['selected_offer_index'] = index
    if kind in ('activity_receive_drink', 'activity_receive_card', 'activity_ack_free_drink'):
        if not isinstance(proof.get('chosen'), dict):
            raise ValueError('activity free reward has no complete current chosen identity')
        if stage in ('card_inspection', 'drink_inspection'):
            index, field = value.get('inspecting_offer'), 'card_id' if kind == 'activity_receive_card' else 'drink_id'
            old, new = original_offer_slot(value, proof, index)
            if (proof.get('selected_offer_index') != index or not old.get(field)
                    or proof['chosen'].get(field) != old[field] or new.get(field) != old[field]):
                raise ValueError('activity cannot receive an unconfirmed inspected reward')
            result['selected_offer_index'] = index
        result['chosen'] = deepcopy(proof['chosen'])
        if kind == 'activity_receive_card' and not value['cards'] and proof['chosen'].get('upgraded') is not True:
            raise ValueError('activity cannot receive a normal card from its quoted upgraded first reward')
    if kind == 'activity_open_drink_detail':
        if type(proof.get('inspect_slot')) is not int or proof['inspect_slot'] not in range(4):
            raise ValueError('activity drink detail has no unique current physical slot')
        result['inspecting_slot'] = proof['inspect_slot']
    result['source_occupancy'] = deepcopy(proof.get('drink_slots'))
    return result


def confirm_page(host, page):
    value, proof, pending = request(host), page.activity, host.hooks.transition_gate.pending
    if value is None or pending is None or not isinstance(proof, dict):
        raise ValueError('activity destination lacks its original pending action')
    keys = frames(proof, (*value['source_frame_ids'], *value.get('navigation_source_frame_ids', ())))
    if tuple(keys) != page.frame_ids: raise ValueError('activity frame binding changed')
    kind, actual, result = pending.target.kind, proof['kind'], deepcopy(value)
    if kind in ('execute_outing', 'confirm_consult','select_consult'):
        expected = ('drink_offer',) if value['kind'] == 'outing' else ('consult_receipt', 'consult_shop', 'support_reward', 'support_ok')
        if value['stage'] != 'paid_pending' or actual not in expected:
            raise ValueError('activity did not reach its original reward destination')
        validate_resources(host, proof)
        result['stage'] = actual if actual != 'consult_shop' else 'census'
        if actual == 'support_reward':
            result['current_grant'] = deepcopy(proof['grant'])
    elif kind in ('activity_select_drink', 'activity_select_card'):
        card = kind == 'activity_select_card'
        expected, field = ('card_selected', 'card_id') if card else ('drink_selected', 'drink_id')
        index, chosen = value.get('selected_offer_index'), proof.get('chosen') or {}
        old, new = original_offer_slot(value, proof, index)
        if (actual != expected or proof.get('selected_offer_index') != index or not old.get(field)
                or new.get(field) != old[field] or chosen.get(field) != old[field]
                or card and (chosen.get('upgraded') is not old.get('upgraded')
                    or new.get('upgraded') is not old.get('upgraded'))):
            raise ValueError('activity selected reward differs from its original chosen slot')
        result['stage'] = expected
    elif kind == 'activity_receive_drink':
        if actual != 'drink_receipt' or proof.get('chosen') != value['chosen'] or value['drinks']:
            raise ValueError('activity drink receipt differs from the original free choice')
        result.update(stage='drink_receipt', drinks=[deepcopy(value['chosen'])])
        if proof.get('receipt_identity_source') == 'pending_selection_requires_census':
            result['drink_identity_census_required'] = True
    elif kind == 'activity_ack_drink':
        if actual != 'card_offer': raise ValueError('activity drink receipt did not lead to its first card offer')
        result['stage'] = 'card_offer'
    elif kind in ('activity_ack_support_card', 'activity_ack_support_ok'):
        card_ack = kind == 'activity_ack_support_card'
        if value['stage'] != ('support_reward' if card_ack else 'support_ok'):
            raise ValueError('support acknowledgement lacks its original grant')
        if card_ack:
            grant = value['current_grant']
            if actual == 'support_reward' and proof.get('grant') == grant:
                raise ValueError('support grant animation has not advanced')
            result['cards'].append(deepcopy(grant))
        if actual == 'support_reward':
            result.update(stage='support_reward', current_grant=_support_grant(host, result, proof))
        elif (actual == value.get('destination_scene', 'support_lesson')
                and (value['kind'] != 'outing' or proof.get('day') == value['census_day'])):
            result['stage'] = 'census'
        elif card_ack and actual == 'support_ok':
            result['stage'] = 'support_ok'
        elif value['kind'] == 'consult' and actual in ('consult_receipt', 'consult_shop'):
            result['stage'] = actual if actual != 'consult_shop' else 'census'
        else:
            raise ValueError('support acknowledgement did not reach its actual successor')
    elif kind in ('activity_inspect_card', 'activity_inspect_drink'):
        card = kind == 'activity_inspect_card'
        index, inspected, offers = value.get('inspecting_offer'), proof.get('inspected') or {}, proof.get('offers')
        expected = 'card_inspection' if card else 'drink_inspection'
        field = 'card_id' if card else 'drink_id'
        old, new = original_offer_slot(value, proof, index)
        identity = inspected.get(field)
        if actual != expected or inspected.get('index') != index or new.get(field) != identity:
            raise ValueError('activity read-only detail did not resolve its original offer')
        if identity is None:
            chosen_index = proof.get('selected_offer_index')
            original_chosen, chosen = original_offer_slot(value, proof, chosen_index)
            if chosen_index == index or not chosen.get(field) or chosen[field] != original_chosen.get(field):
                raise ValueError('activity read-only detail left the chosen reward unknown')
        elif (identity not in (host._catalog.entries if card else host._drink_ids)
                or old.get('candidate_ids') and identity not in old['candidate_ids']
                or card and inspected.get('upgraded') is not host._catalog.entries[identity].upgraded):
            raise ValueError('activity read-only detail changed its original candidate')
        result.update(stage=expected, offers=deepcopy(offers))
    elif kind == 'activity_receive_card':
        if actual != 'card_receipt' or proof.get('chosen') != value['chosen'] or len(value['cards']) >= 2:
            raise ValueError('activity card receipt differs from the original free choice')
        result.update(stage='card_receipt', cards=[*value['cards'], deepcopy(value['chosen'])])
    elif kind == 'activity_ack_card':
        if len(value['cards']) == 1 and actual == 'card_offer': result['stage'] = 'card_offer'
        elif len(value['cards']) == 2 and actual == 'activity_home' and proof.get('day') == 4: result['stage'] = 'census'
        elif actual == 'support_reward' and value['stage'] == 'card_receipt' and outing_rewards_received(value):
            result.update(stage='support_reward', destination_scene='activity_home', current_grant=_support_grant(host, value, proof))
        else: raise ValueError('activity card receipt did not reach its quoted successor')
    elif kind == 'activity_ack_free_drink':
        if actual not in ('consult_receipt', 'consult_shop') or not isinstance(value.get('chosen'), dict):
            raise ValueError('consult reward receipt did not reach its known successor')
        source_slots = value.get('source_occupancy')
        slots = proof.get('drink_slots')
        if (not isinstance(source_slots, list) or not isinstance(slots, list)
                or len(source_slots) != 4 or len(slots) != 4
                or any(row.get('slot') != i or row.get('state') not in ('EMPTY', 'OCCUPIED')
                    for i, row in enumerate([*source_slots]))
                or [row.get('state') for row in source_slots].count('EMPTY') - [row.get('state') for row in slots].count('EMPTY') != 1
                or any(old['state'] == 'OCCUPIED' and new['state'] != 'OCCUPIED'
                    for old, new in zip(source_slots, slots, strict=True))):
            raise ValueError('consult receipt acknowledgement did not prove exactly one new physical drink')
        result['drinks'] = [*value['drinks'], deepcopy(value['chosen'])]
        result['stage'] = 'consult_receipt' if actual == 'consult_receipt' else 'census'
    elif kind in ('recovery_inventory', 'recovery_inventory_detail', 'recovery_inventory_scroll'):
        inventory = proof.get('inventory') or {}
        if actual != 'activity_inventory' or inventory.get('census_id') != value['transaction_id']:
            raise ValueError('activity census did not reach its original card inventory')
        if kind == 'recovery_inventory_detail' and inventory.get('selected_index') != int(pending.target.key.split(':')[1]):
            raise ValueError('activity census detail did not prove its original physical instance')
        if kind == 'recovery_inventory_scroll':
            from .nonbattle_inventory import flat_scroll_measurement_matches
            registration = inventory.get('registration') or {}
            if (registration.get('source_displacement') != (value.get('inventory') or {}).get('displacement')
                    or registration.get('source_frame_ids') != value.get('navigation_source_frame_ids')
                    or registration.get('receipt_frame_ids') != keys
                    or not flat_scroll_measurement_matches(value.get('inventory') or {}, inventory)):
                raise ValueError('activity census scroll lacks its original measured overlap')
        result['inventory'] = deepcopy(inventory)
    elif kind == 'recovery_close':
        if actual not in ('activity_home', 'consult_shop', 'support_lesson') or proof.get('day') != value['census_day']:
            raise ValueError('activity census did not return to its original actual schedule/shop')
        validate_resources(host, proof)
    elif kind == 'activity_open_drink_detail':
        if (actual != 'activity_drink_detail' or proof.get('inspect_slot') != value.get('inspecting_slot')
                or proof.get('drink_id') not in host._drink_ids):
            raise ValueError('activity detail does not prove its original physical-slot identity')
        result['drink_identities'][str(proof['inspect_slot'])] = proof['drink_id']
    elif kind == 'activity_close_drink_detail':
        if actual not in ('activity_home', 'consult_shop'):
            raise ValueError('activity drink detail did not return to its actual census destination')
        result['inspecting_slot'] = None
    else: raise ValueError('activity cannot settle another action')
    if (value['kind'] == 'consult' and actual == 'consult_shop' and not proof.get('quote')
            and result.get('post_restore_census') is None):
        result['post_restore_census'] = _post_restore_consult_source(host, result, proof)
    # A single causally new occupied slot and one known free receipt give a
    # unique assignment. Multiple distinct new IDs require physical details.
    slots = proof.get('drink_slots')
    if isinstance(slots, list) and len(slots) == 4:
        held = {row.slot for row in host.ledger.pending.before.drinks}
        unbound = [row['slot'] for row in slots if row['state'] == 'OCCUPIED'
            and row['slot'] not in held and str(row['slot']) not in result['drink_identities']]
        remaining = Counter(row['drink_id'] for row in result['drinks']) - Counter(result['drink_identities'].values())
        restored = result.get('post_restore_census')
        if restored is not None:
            occupied = [row['slot'] for row in slots if row['state'] == 'OCCUPIED']
            if (occupied != restored['occupied_slots'] or tuple(proof[key] for key in ('currency', 'stamina', 'max_stamina'))
                    != tuple(restored[key] for key in ('currency', 'stamina', 'max_stamina'))):
                raise ValueError('consult restored current resources changed during its census')
            new_slots = [slot for slot in occupied if slot not in held]
            if not unbound:
                drinks = [dict(drink_id=result['drink_identities'][str(slot)], slot=slot,
                    identity_source='post_restore_detail') for slot in new_slots]
                if Counter(row['drink_id'] for row in restored['previous_drink_receipts']) - Counter(row['drink_id'] for row in drinks):
                    raise ValueError('consult restored details contradict its original verified drink receipts')
                result['drinks'] = drinks
        elif result.get('drink_identity_census_required') is True:
            # An obtained animation may be acknowledged before its decorative
            # title is readable. Its selected ID is an expectation, not a
            # current slot identity: require the existing physical detail path.
            pass
        elif len(unbound) == 1 and sum(remaining.values()) == 1:
            result['drink_identities'][str(unbound[0])] = next(iter(remaining))
        elif unbound and len(remaining) == 1 and sum(remaining.values()) == len(unbound):
            for slot in unbound: result['drink_identities'][str(slot)] = next(iter(remaining))
    if proof.get('inventory') is not None:
        from .nonbattle_inventory import retain_inventory_registration
        result['inventory'] = retain_inventory_registration(host, page, proof['inventory'])
    result['last_destination'] = deepcopy(proof)
    return result


def resolve(host, proof):
    value = request(host)
    if (value is None or value['stage'] != 'census' or proof['kind'] not in ('activity_home', 'consult_shop', 'support_lesson')
            or proof.get('day') != value['census_day']):
        raise ValueError('activity needs its actual complete post-census destination')
    frames(proof, value['source_frame_ids']); validate_resources(host, proof)
    inventory, before = value.get('inventory') or {}, host.ledger.pending.before
    expected_resources = host.ledger.pending.expected
    if value['kind'] in ('outing', 'support'):
        if (proof['stamina'], proof['max_stamina'], proof['currency']) != (
                expected_resources.stamina, expected_resources.max_stamina, expected_resources.currency):
            raise ValueError('activity final resources differ from the original exact quote')
        if observed_lesson_grants(value) and proof['kind'] == 'support_lesson':
            pending = host.ledger.pending
            additions = []
            for index, grant in enumerate(value['cards']):
                _validate_support_grant(host, grant)
                additions.append(CardInstance(f'{pending.effect.transaction_id}:support:{index}',
                    grant['card_id'], grant['upgraded']))
            if not additions:
                raise ValueError('lesson has no acknowledged support grant')
            effect = replace(pending.effect, add_cards=tuple(additions), phase='lesson_choice')
            candidate = HifLedger(before, card_catalog=host._catalog, drink_ids=host._drink_ids,
                settled=host.ledger.settled).prepare(effect, frame_ids=pending.source_frame_ids)
            ledger = candidate.confirm(candidate.pending.expected, transaction_id=effect.transaction_id,
                frame_ids=tuple(proof['frame_ids']))
            return ledger, dict(value, stage='complete', resolution=deepcopy(proof))
    elif value.get('post_restore_census') is not None:
        restored = value['post_restore_census']
        if (restored.get('source') not in ('post_restore_current_census', 'free_entry_current_census')
                or tuple(proof[key] for key in ('currency', 'stamina', 'max_stamina')) != tuple(
                    restored[key] for key in ('currency', 'stamina', 'max_stamina'))
                or proof['currency'] < before.currency
                or (proof['stamina'], proof['max_stamina']) != (before.stamina, before.max_stamina)):
            raise ValueError('consult free post-restore census resources changed')
    else:
        quote = proof.get('quote') or {}
        if (type(quote.get('drink_rewards')) is not int or quote['drink_rewards'] != len(value['drinks'])
                or type(quote.get('currency_delta')) is not int or quote['currency_delta'] < 0
                or proof['currency'] != before.currency + quote['currency_delta']
                or (proof['stamina'], proof['max_stamina']) != (before.stamina, before.max_stamina)):
            raise ValueError('consult free rewards differ from its actual support quote')
    if (inventory.get('complete') is not True or inventory.get('census_id') != value['transaction_id']
            or inventory.get('total') != len(inventory.get('slots', ()))
            or [row.get('index') for row in inventory['slots']] != list(range(inventory.get('total', 0))) or before.counters):
        raise ValueError('activity requires a complete current permanent-card census')
    groups = defaultdict(list)
    removed, upgraded = [], []
    partial = isinstance(before, PartialDeckLedgerState)
    unbound_upgrades = Counter()
    for grant in value['cards']:
        if grant.get('kind') != 'upgrade':
            continue
        if partial and grant.get('source_instance_id') is None:
            _validate_support_grant(host, grant)
            # The award identified the variant only. Preserve known copies still
            # present; attribute an upgrade only if its old variant is missing.
            known = [c for c in before.cards if c.card_id == grant['source_card_id'] and c not in removed]
            remaining = Counter(tuple(sorted(r['customizations'].items())) for r in inventory['slots']
                if r.get('card_id') == grant['source_card_id'] and isinstance(r.get('customizations'), dict))
            for card in known:
                if remaining[card.customization] <= 0:
                    removed.append(card)
                    upgraded.append(replace(card, card_id=grant['card_id'], upgraded=True))
                    break
                remaining[card.customization] -= 1
            else:
                unbound_upgrades[grant['card_id']] += 1
            continue
        matches = [card for card in before.cards if card.instance_id == grant.get('source_instance_id')
            and card.card_id == grant.get('source_card_id') and not card.upgraded and not card.customization]
        entry = host._catalog.entries.get(grant.get('card_id'))
        if (len(matches) != 1 or matches[0] in removed or entry is None or not entry.upgraded
                or entry.name != host._catalog.entries[matches[0].card_id].name + '+'):
            raise ValueError('support upgrade no longer matches its original permanent instance')
        removed.append(matches[0])
        upgraded.append(replace(matches[0], card_id=entry.card_id, upgraded=True))
    for card in before.cards:
        if card not in removed:
            groups[card.card_id, card.upgraded, card.customization].append(card)
    for card in upgraded:
        groups[card.card_id, card.upgraded, card.customization].append(card)
    # The award confirms the card and its variant. Customization is read in the inventory.
    expected_cards = Counter((row['card_id'], row['upgraded']) for row in value['cards'] if row.get('kind') != 'upgrade')
    remaining_grants = expected_cards.copy()
    census_only = Counter()
    observed, added = [], []
    for row in inventory['slots']:
        entry = host._catalog.entries.get(row.get('card_id'))
        if (entry is None or not isinstance(row.get('customizations'), dict)
                or row.get('presence') not in ('CONFIDENT_ZERO', 'DETAIL_CANDIDATE')
                or row['presence'] != 'CONFIDENT_ZERO' and row.get('detail', {}).get('nonbattle_schema') != 1):
            raise ValueError('activity census contains an unresolved permanent variant')
        key = row['card_id'], entry.upgraded, tuple(sorted(row['customizations'].items()))
        if groups[key]: card = groups[key].pop(0)
        else:
            acquired = not partial or remaining_grants[key[:2]] > 0
            source = 'activity' if acquired else 'census'
            card = CardInstance(f'{host.run_id}:{source}:{before.revision}:{row["index"]}', *key)
            if acquired:
                added.append(card)
                remaining_grants[key[:2]] -= 1
            else:
                census_only[key[0]] += 1
        observed.append(card)
    if any(groups.values()): raise ValueError('activity cannot silently remove or customize an old card')
    if unbound_upgrades - census_only:
        raise ValueError('activity census did not contain the observed upgraded card')
    if Counter((card.card_id, card.upgraded) for card in added) != expected_cards:
        raise ValueError('activity current card additions differ from its verified free receipts')
    if value['kind'] == 'outing' and not outing_rewards_received(value):
        raise ValueError('outing quoted reward count is incomplete')
    slots = proof.get('drink_slots') or ()
    if len(slots) != 4 or [row.get('slot') for row in slots] != list(range(4)):
        raise ValueError('activity needs all four current physical drink slots')
    old_drinks = {row.slot: row for row in before.drinks}
    new_slots = [row['slot'] for row in slots if row['state'] == 'OCCUPIED' and row['slot'] not in old_drinks]
    if any(slots[slot]['state'] != 'OCCUPIED' for slot in old_drinks) or len(new_slots) != len(value['drinks']):
        raise ValueError('activity drink receipts do not match the new physical occupancy')
    # Ordered free receipts alone cannot assign multiple new physical slots.
    # Each new slot needs a current read-only detail bound to its exact ID.
    details = proof.get('drink_identities') or {}
    if set(map(int, details)) != set(new_slots):
        raise ValueError('activity new drinks need current physical-slot identities')
    if Counter(details.values()) != Counter(row['drink_id'] for row in value['drinks']):
        raise ValueError('activity current drink identities differ from its free receipts')
    additions = tuple(DrinkInstance(slot, f'{host.run_id}:activity:{before.revision}:drink:{slot}', details[str(slot)]) for slot in new_slots)
    effect = LedgerEffect(host.ledger.pending.effect.transaction_id, host.ledger.pending.effect.target_key,
        remove_cards=tuple(removed), add_cards=(*upgraded, *added), add_drinks=additions,
        currency_delta=proof['currency']-before.currency,
        stamina_delta=proof['stamina']-before.stamina, max_stamina_delta=proof['max_stamina']-before.max_stamina,
        day=value['census_day'], phase=({'outing': 'schedule', 'consult': 'consult_shop'}[value['kind']]
            if value['kind'] != 'support' else 'schedule' if value.get('destination_scene') == 'activity_home' else 'lesson_choice'))
    candidate = HifLedger(before, card_catalog=host._catalog, drink_ids=host._drink_ids, settled=host.ledger.settled).prepare(
        effect, frame_ids=host.ledger.pending.source_frame_ids)
    ledger = candidate.confirm_card_census(observed, frame_ids=tuple(proof['frame_ids']))
    result = deepcopy(value); result.update(stage='complete', resolution=deepcopy(proof))
    return ledger, result
