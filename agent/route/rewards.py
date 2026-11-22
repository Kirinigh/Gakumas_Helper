"""Resolve an already submitted public course from a complete fresh census.

The stamina quote is reserved before input. Variable free rewards never become
an invented quote, and the reservation remains pending during read-only census.
"""
from __future__ import annotations

import re
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import replace

from .ledger import HifLedger, CardInstance, LedgerEffect, PartialDeckLedgerState

REWARD_NAVIGATION = frozenset({'training_reward_ack', 'training_reward_upgrade_ack', 'recovery_inventory',
    'recovery_inventory_detail', 'recovery_inventory_scroll', 'recovery_close',
    'reward_delete_select', 'reward_delete_scroll', 'reward_delete_open', 'reward_delete_confirm', 'reward_delete_result_ack'})
ACTIVE_STAGES = frozenset({'awaiting_ack', 'upgrade', 'deletion', 'census'})
COURSE_RESULT_DESTINATIONS = ('reward_home', 'reward_delete_choice', 'reward_upgrade')


def request(host, *, ledger=None, handoff=None):
    ledger = host.ledger if ledger is None else ledger
    handoff = host.handoff if handoff is None else handoff
    value = (handoff or {}).get('nonbattle_context', {}).get('training_reward')
    if value is None or isinstance(value, dict) and value.get('stage') in ('complete', 'cancelled'):
        return None
    pending = ledger.pending if ledger is not None else None
    owner = host.ownership or {}
    consult_exit = isinstance(value, dict) and value.get('source_target') == 'finish_consult'
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or value.get('stage') not in ('course_pending', *ACTIVE_STAGES)
            or owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
            or owner.get('game_pid') != host.game_pid
            or value.get('ownership_revision') != owner.get('revision')
            or pending is None or pending.effect.target_key != ('finish_consult' if consult_exit else 'confirm_training')
            or pending.before.day not in ((6,) if consult_exit else (2, 5))
            or pending.before.phase != ('consult_shop' if consult_exit else 'training_selected')
            or value.get('run_id') != host.run_id or value.get('game_pid') != host.game_pid
            or value.get('transaction_id') != pending.effect.transaction_id
            or value.get('source_revision') != pending.before.revision
            or value.get('day') != pending.before.day or value.get('next_day') != pending.before.day + 1
            or value.get('quoted_cost') != -pending.effect.stamina_delta
            or value.get('source_frame_ids') != list(pending.source_frame_ids)
            or not consult_exit and (value.get('attribute') not in ('Vo', 'Da', 'Vi') or type(value.get('special')) is not bool)):
        raise ValueError('public-course reward reservation does not match its original submitted course')
    if consult_exit:
        from .consult import consult_effect
        if pending.effect != consult_effect(pending.before, 'finish_consult'):
            raise ValueError('free consultation exit event cannot change its original zero-cost reservation')
    return value


def same_course_animation(host, pending, page):
    if (pending is None or pending.target.kind != 'confirm_training' or page is None
            or page.observation.scene != 'training_animation' or not isinstance(page.reward, dict)):
        return False
    value = request(host)
    return (value is not None and value['stage'] == 'course_pending'
        and page.reward.get('kind') == 'animation' and page.reward.get('day') == value['day']
        and page.reward.get('attribute') == value['attribute'] and page.reward.get('special') is value['special'])


def consult_exit_request(host, gate):
    """Reuse the free-reward reservation for an event after actual shop exit."""
    from .consult import consult_effect
    from .checkpoint import transition_checkpoint
    from .input_receipt import original_input_report

    pending = host.ledger.pending
    if (pending is None or gate.pending is None or gate.pending.target.kind != 'finish_consult'
            or pending.effect != consult_effect(pending.before, 'finish_consult')
            or pending.before != host.ledger.state or gate.pending.source.scene != 'consult_shop'
            or gate.pending.expected_scenes != ('battle_intro', 'reward_delete_choice')
            or gate.pending.source.day != 6 or request(host) is not None):
        raise ValueError('free deletion event lacks its original zero-cost consultation exit')
    report = original_input_report(host, gate)
    context = (host.handoff or {}).get('nonbattle_context') or {}
    consult = context.get('activity_reward') or {}
    if (consult.get('stage') != 'complete' or consult.get('run_id') != host.run_id
            or consult.get('resolution', {}).get('kind') != 'consult_shop'):
        raise ValueError('free deletion event lacks its completed original shop census')
    return dict(schema_version=1, stage='deletion', source_target='finish_consult',
        run_id=host.run_id, game_pid=host.game_pid, ownership_revision=host.ownership['revision'],
        transaction_id=pending.effect.transaction_id, source_revision=pending.before.revision,
        day=6, next_day=7, quoted_cost=0, source_frame_ids=list(pending.source_frame_ids),
        original_transition=transition_checkpoint(gate, host.clock.epoch), original_result=report['id'],
        reservation=deepcopy(host.ledger.record()['pending']), inventory=None,
        previous_course=deepcopy(context.get('training_reward')))


def active(host):
    value = request(host)
    return value is not None and value['stage'] in ACTIVE_STAGES


def permits_navigation(host, ledger, gate):
    value = request(host, ledger=ledger)
    if value is None or value['stage'] not in ACTIVE_STAGES:
        return False
    pending = gate.pending
    if pending is None:
        return True
    return (pending.target.kind in REWARD_NAVIGATION
        and pending.source.day == (value['day'] if pending.target.kind in ('training_reward_ack', 'training_reward_upgrade_ack')
            or pending.target.kind.startswith('reward_delete_') else value['next_day'])
        and (re.fullmatch(r'inventory-detail:\d+', pending.target.key) is not None
            if pending.target.kind == 'recovery_inventory_detail' else
            re.fullmatch(r'inventory-scroll:\d+', pending.target.key) is not None
            if pending.target.kind == 'recovery_inventory_scroll' else pending.target.key == pending.target.kind))


def start_request(host, ledger, gate):
    from .checkpoint import transition_checkpoint
    choice = (host.handoff or {}).get('nonbattle_context', {}).get('training_choice', {})
    pending = ledger.pending
    if (pending is None or gate.pending is None or gate.pending.target.kind != 'confirm_training'
            or pending.effect.target_key != 'confirm_training' or pending.before.day not in (2, 5)
            or choice.get('status') != 'active' or choice.get('day') != pending.before.day
            or choice.get('selected_attribute') not in ('Vo', 'Da', 'Vi')
            or type(choice.get('selected_special')) is not bool
            or choice.get('selected_cost') != -pending.effect.stamina_delta
            or pending.effect.currency_delta or pending.effect.remove_cards or pending.effect.add_cards
            or pending.effect.remove_drinks or pending.effect.add_drinks or pending.effect.counter_deltas):
        raise ValueError('public-course reward intent lacks its original exact cost and selection')
    old = request(host)
    if old is not None:
        raise ValueError('public-course reward intent cannot replace an unresolved submission')
    return dict(schema_version=1, stage='course_pending', run_id=host.run_id, game_pid=host.game_pid,
        ownership_revision=host.ownership['revision'], transaction_id=pending.effect.transaction_id,
        source_revision=pending.before.revision, day=pending.before.day, next_day=pending.before.day + 1,
        quoted_cost=-pending.effect.stamina_delta, attribute=choice['selected_attribute'],
        special=choice['selected_special'], source_frame_ids=list(pending.source_frame_ids),
        original_transition=transition_checkpoint(gate, host.clock.epoch),
        reservation=deepcopy(ledger.record()['pending']), inventory=None)


def reward_frames(proof, source):
    frames = proof.get('frame_ids')
    if (not isinstance(frames, list) or len(frames) != 1
            or any(not isinstance(value, str) or not value for value in frames)
            or set(frames).intersection(source)):
        raise ValueError('public-course reward proof requires fresh distinct observations')
    return frames


def validate_resources(host, proof):
    pending = host.ledger.pending
    before = pending.before
    values = [proof.get(key) for key in ('stamina', 'max_stamina', 'currency')]
    if (any(type(value) is not int or value < 0 for value in values)
            or not before.stamina + pending.effect.stamina_delta <= values[0] <= values[1]
            or values[1] < before.max_stamina or values[1] > 100 or values[2] < before.currency):
        raise ValueError('public-course reward contains an unquoted loss or incomplete resources')
    value = request(host)
    deletion = value.get('deletion') if value is not None else None
    if deletion is not None and deletion.get('status') == 'committed_pending_census':
        result = deletion['result']
        if any(proof[key] < result[key] for key in ('stamina', 'max_stamina', 'currency')):
            raise ValueError('public-course reward lost resources after its observed deletion result')


def confirm_page(host, page):
    """Confirm navigation only; the resource reservation remains unresolved."""
    from .engine import source_frames_match
    value = request(host)
    proof = page.reward
    route = host.hooks.transition_gate.pending
    if value is None or route is None or not isinstance(proof, dict):
        raise ValueError('public-course reward has no current original pending action')
    frames = reward_frames(proof, (*value['source_frame_ids'], *value.get('navigation_source_frame_ids', ())))
    if (tuple(frames) != page.frame_ids or not source_frames_match(page.observation, page.frame_ids)
            or page.observation != host._page.observation):
        raise ValueError('public-course reward page is not bound to the current proof')
    result = deepcopy(value)
    if route.target.kind == 'confirm_training' and value['stage'] == 'course_pending':
        if (proof.get('kind') != 'result' or proof.get('day') != value['day']
                or proof.get('attribute') != value['attribute'] or proof.get('special') is not value['special']):
            raise ValueError('public-course result differs from its original selected course')
        validate_resources(host, proof)
        result.update(stage='awaiting_ack', result=deepcopy(proof))
    elif value['stage'] in ACTIVE_STAGES and route.target.kind in REWARD_NAVIGATION:
        ack = route.target.kind
        if ack == 'training_reward_upgrade_ack':
            if value['stage'] != 'upgrade':
                raise ValueError('course upgrade acknowledgement lacks its original result')
            grant = result.pop('upgrade')['grant']
            if proof.get('grant') == grant:
                raise ValueError('course upgrade result is still displayed')
            result.setdefault('upgrades', []).append(grant)
        if ack in ('training_reward_ack', 'training_reward_upgrade_ack') and proof.get('kind') == 'upgrade':
            from .activity_reward import _validate_support_grant
            grant = proof.get('grant') or {}
            if (value.get('source_target') == 'finish_consult' or proof.get('day') != value['day']
                    or grant.get('kind') != 'upgrade' or grant in result.get('upgrades', ())):
                raise ValueError('course upgrade does not belong to its original reward')
            _validate_support_grant(host, grant)
            validate_resources(host, proof)
            result.update(stage='upgrade', upgrade=deepcopy(proof))
        elif ack in ('training_reward_ack', 'training_reward_upgrade_ack') and proof.get('kind') == 'delete_choice':
            if value['stage'] not in ('awaiting_ack', 'upgrade') or value['special'] is not True or proof.get('day') != value['day']:
                raise ValueError('course deletion is not its original SP-result interruption')
            result.update(stage='deletion', deletion=deepcopy(proof['deletion']))
        elif route.target.kind in ('reward_delete_open', 'reward_delete_confirm') and proof.get('kind') == 'delete_result':
            from .checkpoint import transition_checkpoint
            from .input_receipt import original_input_report
            if (value['stage'] != 'deletion' or proof.get('day') != value['day']
                    or value['deletion'].get('status') != 'active'
                    or proof['deletion']['request'] != value['deletion']['request']):
                raise ValueError('course deletion result changed its original owned instance')
            original_input_report(host, host.hooks.transition_gate)
            validate_resources(host, proof)
            result['deletion'] = dict(deepcopy(value['deletion']), status='committed_pending_census',
                original_transition=transition_checkpoint(host.hooks.transition_gate, host.clock.epoch),
                result=deepcopy(proof))
        elif route.target.kind == 'reward_delete_scroll':
            if (value['stage'] != 'deletion' or proof.get('kind') not in ('delete_choice', 'delete_search')
                    or proof.get('day') != value['day'] or proof['deletion']['request'] != value['deletion']['request']):
                raise ValueError('deletion search changed its original request')
            result['deletion']['last_menu'] = dict(kind=proof['kind'], frame_ids=list(frames))
        elif route.target.kind in ('reward_delete_select', 'reward_delete_open'):
            expected = {'reward_delete_select': 'delete_selected', 'reward_delete_open': 'delete_confirm'}[route.target.kind]
            if (route.target.kind == 'reward_delete_select' and route.source.scene == 'reward_delete_choice'
                    and proof.get('kind') == 'delete_reselect' and value['deletion'].get('last_menu', {}).get('kind') != 'delete_reselect'):
                expected = 'delete_reselect'
            if (value['stage'] != 'deletion' or proof.get('kind') != expected or proof.get('day') != value['day']
                    or proof.get('deletion', {}).get('request') != value['deletion']['request']):
                raise ValueError('course deletion navigation changed its original instance or stage')
            result['deletion'] = dict(deepcopy(proof['deletion']), last_menu=dict(kind=expected, frame_ids=list(frames)))
            if expected == 'delete_reselect':
                from .input_receipt import original_input_report
                result['deletion']['last_menu']['source_result'] = original_input_report(
                    host, host.hooks.transition_gate)['id']
        elif route.target.kind in ('training_reward_ack', 'training_reward_upgrade_ack', 'recovery_close', 'reward_delete_result_ack'):
            if route.target.kind == 'reward_delete_result_ack' and value['deletion'].get('status') != 'committed_pending_census':
                raise ValueError('deletion result cannot advance without its successful original removal')
            if proof.get('kind') != 'home' or proof.get('day') != value['next_day']:
                raise ValueError('public-course reward navigation lacks the actual next calendar day')
            validate_resources(host, proof)
            result.update(stage='census', home=deepcopy(proof))
        elif proof.get('kind') != 'inventory':
            raise ValueError('public-course reward navigation did not reach the owned inventory')
        elif (route.target.kind in ('recovery_inventory_detail', 'recovery_inventory_scroll')
                and optional_course_deck_review(value) and proof['inventory'].get('read_finished') is True
                and proof['inventory'].get('complete') is False and proof['inventory'].get('review_frame_ids') == frames
                and proof['inventory'].get('census_id') == value['transaction_id']):
            # The submitted list navigation yielded no usable registration.
            # Retain its real receipt and partial evidence; no card mutation is claimed.
            pass
        elif route.target.kind == 'recovery_inventory_detail':
            index = int(route.target.key.split(':')[1])
            if proof['inventory'].get('selected_index') != index:
                raise ValueError('public-course census did not confirm its original detail instance')
        elif route.target.kind == 'recovery_inventory_scroll':
            from .nonbattle_inventory import flat_scroll_measurement_matches
            registration = proof['inventory'].get('registration') or {}
            if (registration.get('source_displacement') != (value.get('inventory') or {}).get('displacement')
                    or registration.get('receipt_frame_ids') != frames
                    or not set(registration.get('source_frame_ids', ())).intersection(value.get('navigation_source_frame_ids', ()))
                    or not flat_scroll_measurement_matches(value.get('inventory') or {}, proof['inventory'])):
                raise ValueError('public-course census scroll lacks its original measured registration')
    else:
        raise ValueError('public-course reward cannot settle an unrelated action')
    if proof.get('inventory') is not None:
        from .nonbattle_inventory import retain_inventory_registration
        result['inventory'] = retain_inventory_registration(host, page, proof['inventory'])
    return result


def with_request(handoff, value):
    candidate = deepcopy(handoff or {})
    candidate.setdefault('nonbattle_context', {})['training_reward'] = deepcopy(value)
    return candidate


def optional_course_deck_review(value):
    """An attribute gain alone never calls for a compulsory deck census."""
    result = value.get('result') or {}
    return (value.get('source_target') != 'finish_consult' and type(value.get('special')) is bool
        and result.get('kind') == 'result' and result.get('special') is value['special']
        and not any(value.get(key) for key in ('deletion', 'upgrades', 'upgrade')))


def retained_course_deck(value, before):
    if not optional_course_deck_review(value):
        return False
    inventory = value.get('inventory')
    if inventory is None:
        # A special lesson can lead to a separate card reward. The SP label
        # alone does not invalidate a known deck after an attribute result.
        return bool(before.cards) and not isinstance(before, PartialDeckLedgerState)
    return (inventory.get('census_id') == value['transaction_id']
        and inventory.get('read_finished') is True and inventory.get('complete') is False)


def _reviewed_partial_deck(host, settled, inventory):
    """A failed review preserves old instances and any individually resolved copies."""
    from .recovery_inventory import _card_variant
    state = settled.state
    known = Counter((card.card_id, card.upgraded, card.customization) for card in state.cards)
    observed = Counter()
    additions = []
    for row in inventory['slots']:
        variant = _card_variant(host, row)
        if variant is not None and not row.get('identity_needs_detail'):
            observed[variant] += 1
            if observed[variant] > known[variant]:
                additions.append(CardInstance(f'{host.run_id}:census:{host.ledger.pending.before.revision}:{row["index"]}', *variant))
    total = inventory.get('total')
    if (state.cards and not isinstance(state, PartialDeckLedgerState)
            and not additions and total in (None, state.card_count)):
        return settled
    cards = (*state.cards, *additions)
    if total is None:
        total = state.card_count if state.cards or isinstance(state, PartialDeckLedgerState) else None
    if total is not None and total < len(cards):
        total = None
    partial = PartialDeckLedgerState(**dict(vars(state), cards=cards, card_count=total))
    return HifLedger(partial, card_catalog=host._catalog, drink_ids=host._drink_ids, settled=settled.settled)


def resolve(host, proof, *, observation=None):
    """Reconcile original instances, acknowledged changes and the current census."""
    value = request(host)
    if value is None or value['stage'] != 'census' or proof.get('kind') != 'home' or proof.get('day') != value['next_day']:
        raise ValueError('public-course reward needs its actual post-census calendar')
    reward_frames(proof, (*value['source_frame_ids'], *value.get('result', {}).get('frame_ids', ())))
    validate_resources(host, proof)
    inventory = value.get('inventory')
    pending = host.ledger.pending
    if pending.before.counters:
        raise ValueError('public-course reward counter producer is not bound')
    if retained_course_deck(value, pending.before) or (optional_course_deck_review(value) and proof.get('inventory_opened') is False):
        return _settle(host, value, proof, (), (), None, observation)
    if (not isinstance(inventory, dict) or inventory.get('census_id') != value['transaction_id']
            or inventory.get('complete') is not True
            or inventory.get('total') != len(inventory.get('slots', ()))
            or [row.get('index') for row in inventory['slots']] != list(range(inventory['total']))):
        raise ValueError('public-course reward requires a complete fresh physical card census')
    deleted = _removed_cards(host, value)
    removed, added, observed = list(deleted), [], []
    partial = isinstance(pending.before, PartialDeckLedgerState)
    groups = defaultdict(list)
    for card in pending.before.cards:
        if card not in deleted:
            groups[card.card_id, card.upgraded, card.customization].append(card)
    unknown_upgrades = Counter()
    for grant in value.get('upgrades', ()):
        from .activity_reward import _validate_support_grant
        if not isinstance(grant, dict) or grant.get('kind') != 'upgrade':
            raise ValueError('public-course upgrade is not an acknowledged card variant change')
        _validate_support_grant(host, grant)
        originals = groups[grant['source_card_id'], False, ()]
        instance = grant.get('source_instance_id')
        if instance is not None:
            original = next((card for card in originals if card.instance_id == instance), None)
            if original is None:
                raise ValueError('public-course upgrade cannot change a removed or already upgraded instance')
        else:
            # A partial ledger can lack this base card entirely. Keep known copies
            # still present in the census, and require the new variant in every case.
            remaining = sum(row.get('card_id') == grant['source_card_id'] and row.get('customizations') == {}
                for row in inventory['slots'])
            if len(originals) <= remaining:
                unknown_upgrades[grant['card_id']] += 1
                continue
            original = originals[-1]
        originals.remove(original)
        removed.append(original)
        upgraded = replace(original, card_id=grant['card_id'], upgraded=True)
        added.append(upgraded)
        groups[upgraded.card_id, True, upgraded.customization].append(upgraded)
    for row in inventory['slots']:
        card_id = row.get('card_id')
        entry = host._catalog.entries.get(card_id)
        if (entry is None or not isinstance(row.get('customizations'), dict)
                or row.get('presence') not in ('CONFIDENT_ZERO', 'DETAIL_CANDIDATE')
                or row['presence'] != 'CONFIDENT_ZERO' and row.get('detail', {}).get('nonbattle_schema') != 1):
            raise ValueError('public-course reward census contains an unresolved permanent variant')
        custom = tuple(sorted((str(key), count) for key, count in row['customizations'].items()))
        key = (card_id, entry.upgraded, custom)
        if groups[key]:
            card = groups[key].pop(0)
        else:
            source = 'census' if partial or optional_course_deck_review(value) else 'reward'
            card = CardInstance(f'{host.run_id}:{source}:{pending.before.revision}:{row["index"]}', card_id, entry.upgraded, custom)
            if not partial:
                added.append(card)
            elif not custom and unknown_upgrades[card_id] > 0:
                unknown_upgrades[card_id] -= 1
        observed.append(card)
    if any(groups.values()):
        raise ValueError('public-course free reward cannot remove or customize an existing permanent card')
    if any(unknown_upgrades.values()):
        raise ValueError('public-course census lacks an acknowledged upgraded variant')
    if deleted and any(row['card_id'] == deleted[0].card_id for row in inventory['slots']):
        raise ValueError('course deletion census still contains its removed unique card')
    return _settle(host, value, proof, removed, added, observed, observation)


def _settle(host, value, proof, removed, added, observed, observation):
    pending = host.ledger.pending
    if proof.get('drinks') != [dict(slot=drink.slot, instance_id=drink.instance_id, drink_id=drink.drink_id)
            for drink in pending.before.drinks]:
        raise ValueError('public-course reward drink census has an unresolved identity or loss')
    effect = LedgerEffect(pending.effect.transaction_id, pending.effect.target_key, remove_cards=tuple(removed), add_cards=tuple(added),
        currency_delta=proof['currency'] - pending.before.currency,
        stamina_delta=proof['stamina'] - pending.before.stamina,
        max_stamina_delta=proof['max_stamina'] - pending.before.max_stamina,
        day=value['next_day'], phase='interval' if value.get('source_target') == 'finish_consult' else 'schedule')
    candidate = HifLedger(pending.before, card_catalog=host._catalog, drink_ids=host._drink_ids,
        settled=host.ledger.settled).prepare(effect, frame_ids=pending.source_frame_ids)
    frames = tuple(proof['frame_ids'])
    if observed is None:
        settled = candidate.confirm(candidate.pending.expected, transaction_id=effect.transaction_id, frame_ids=frames, observation=observation)
        if value.get('inventory') or proof.get('inventory_opened') is False:
            settled = _reviewed_partial_deck(host, settled, value.get('inventory') or dict(slots=[]))
        deck_source = 'partial_inventory' if value.get('inventory') else 'retained_before_course'
    else:
        settled = candidate.confirm_card_census(observed, frame_ids=frames, variable_reward=True, observation=observation)
        deck_source = 'observed_inventory'
    result = deepcopy(value)
    result.update(stage='complete', resolution=deepcopy(proof), resolved_transaction_id=effect.transaction_id,
        deck_source=deck_source,
        quoted_stamina_delta=pending.effect.stamina_delta,
        observed_stamina_delta=effect.stamina_delta, observed_currency_delta=effect.currency_delta)
    return settled, result


def _removed_cards(host, value):
    pending = host.ledger.pending
    removed = ()
    deletion = value.get('deletion')
    if deletion is not None:
        from .deletion import DeletionRequest, request_record
        if (deletion.get('status') != 'committed_pending_census' or deletion.get('request', {}).get('fee') != 0
                or deletion.get('original_transition', {}).get('pending', {}).get('target', {}).get('kind')
                    not in ('reward_delete_open', 'reward_delete_confirm')
                or deletion.get('result', {}).get('kind') != 'delete_result'):
            raise ValueError('course deletion has no exact successful removal result')
        removed = tuple(card for card in pending.before.cards
            if request_record(DeletionRequest(card, 0)) == deletion['request'])
        if len(removed) != 1:
            raise ValueError('course deletion does not name one original permanent instance')
        import unicodedata
        name = ''.join(unicodedata.normalize('NFKC', host._catalog.entries[removed[0].card_id].name).split())
        if deletion['result'].get('result_text') != name + 'を削除しました' or deletion['result'].get('day') != value['day']:
            raise ValueError('course deletion result does not name its original removed card')
        reward_frames(deletion['result'], value['source_frame_ids'])
    return removed


def resolve_consult_exit_opening(host, report):
    """Finish a known free deletion from the unchanged Live card census.

    Only permanent cards are consumed here. Battle-entry recovery and fresh
    input authority remain the responsibility of the normal battle handoff.
    """
    from types import SimpleNamespace

    from live_decision.opening_observer import opening_component
    from live_decision.inventory_customization import verified_zero_customization_detail

    from .hif import active_hif_role
    from .input_receipt import original_input_report

    value, gate = request(host), host.hooks.transition_gate
    if (value is None or value.get('source_target') != 'finish_consult' or value['stage'] != 'deletion'
            or gate.pending is None or gate.pending.target.kind != 'reward_delete_result_ack'):
        raise ValueError('opening census lacks its original consultation deletion acknowledgement')
    acknowledgement = original_input_report(host, gate)
    recovery, run = report.get('context_recovery', {}), report.get('observed_live_context', {})
    events = report.get('live_events', ())
    if (report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid
            or recovery.get('game_pid') != host.game_pid or recovery.get('accepted') is not True
            or recovery.get('returned_to_live') is not True
            or (run.get('idol'), run.get('plan'), run.get('max_stamina'))
                != (*active_hif_role(host), host.ledger.state.max_stamina)
            or recovery.get('round_context', {}).get('round_number') != 1
            or (recovery['round_context'].get('run_id'), recovery['round_context'].get('revision'))
                != (run.get('run_id'), run.get('revision'))
            or len(events) != 1 or events[0].get('event') != 'live_inventory_observe'
            or events[0].get('executed') is not False or events[0].get('execution_submit_count') != 0):
        raise ValueError('opening census is not the same untouched first battle')
    component = opening_component(events[0], 'cards', SimpleNamespace(**run))
    before = reward_frames(dict(frame_ids=component.get('before_frame_ids')), value['source_frame_ids'])
    frames = reward_frames(dict(frame_ids=component.get('after_frame_ids')), (*value['source_frame_ids'], *before))
    counts = dict.fromkeys(('手札', '山札', '捨札', '除外'), 0)
    observed = Counter()
    for row in component.get('cards', ()):
        custom, zone = row.get('customizations'), row.get('zone')
        entry = host._catalog.entries.get(row.get('card_id'))
        if (entry is None or zone not in counts or row.get('index') != counts[zone]
                or not isinstance(custom, dict)
                or any(not isinstance(key, str) or not key.isdecimal() or type(count) is not int or count <= 0
                    for key, count in custom.items())
                or not (row.get('source') == 'observed_badge_absence' and not custom
                    or row.get('source') == 'observed_detail' and (custom
                        or len(row.get('frame_ids') or ()) == 1 and verified_zero_customization_detail(row)))):
            raise ValueError('opening census contains an unresolved permanent card')
        counts[zone] += 1
        observed[(row['card_id'], entry.upgraded, tuple(sorted(custom.items())))] += 1
    removed = _removed_cards(host, value)
    pending = host.ledger.pending
    expected = Counter((card.card_id, card.upgraded, card.customization)
        for card in pending.before.cards if card not in removed)
    if (len(removed) != 1 or observed != expected or component.get('zone_counts') != counts
            or counts['手札'] not in (3, 4, 5) or not counts['山札'] or counts['捨札'] or counts['除外']):
        raise ValueError('opening census does not equal the original deck minus its verified deletion')
    # Resources come from the actual deletion result, not the automatically
    # healed Live stamina. The unchanged drinks keep their original instances.
    proof = value['deletion']['result']
    if any(proof.get(key) != getattr(pending.before, key) for key in ('stamina', 'max_stamina', 'currency')):
        raise ValueError('free consultation deletion changed its original resources')
    effect = replace(pending.effect, remove_cards=removed)
    candidate = HifLedger(pending.before, card_catalog=host._catalog, drink_ids=host._drink_ids,
        settled=host.ledger.settled).prepare(effect, frame_ids=pending.source_frame_ids)
    settled = candidate.confirm(candidate.pending.expected, transaction_id=effect.transaction_id,
        frame_ids=tuple(frames), battle_census_frame_id=frames[0])
    result = deepcopy(value)
    result.update(stage='complete', resolved_transaction_id=effect.transaction_id,
        quoted_stamina_delta=0, observed_stamina_delta=0, observed_currency_delta=0,
        resolution=dict(kind='battle_opening', day=7, source_result=report['id'],
            acknowledgement_result=acknowledgement['id'], frame_ids=frames, card_count=sum(counts.values()),
            permanent_component=component))
    return settled, result
