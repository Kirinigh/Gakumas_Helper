"""Adapt an already verified native settlement to the restart observer.

The ordinary source loader verifies the complete action/state chain once.
The second pass only extracts its submitted action order; it neither
replays transitions nor treats selection-only records as business actions.
"""

import json
from dataclasses import asdict

from . import diagnostic_depth_reached
from .native_settlement_resume import load_result, load_verified_source

_PARENTS = ('native_resume_source', 'native_selected_source',
            'native_inprocess_source', 'native_pending_source')
_OTHER_ROOTS = ('native_support_source', 'native_manual_choice_source',
                'native_choice_source', 'native_restart_source',
                'native_settled_restart_source')


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _submissions(row):
    events = row.get('live_events', ())
    if any(event.get('choice_submit_count', 0) for event in events) or row.get('choice_journal'):
        raise ValueError('settled restart chronology does not cover secondary choices')
    return [event for event in events if event.get('execution_submit_count', 0)
            or event.get('end_turn_submit_count', 0)]


def _source_rows(runtime, name, verified):
    """Follow only the primary chain already consumed by load_verified_source."""
    from .native_opening_root import has_native_opening_root

    rows, seen = [], set()
    old_pid = verified['endpoints']['game']['pid']
    identity = verified['observed_live_context']
    while True:
        if name in seen or diagnostic_depth_reached(seen):
            raise ValueError('settled restart chronology is cyclic or unbounded')
        seen.add(name)
        row = load_result(runtime, name, allow_closed_transaction=True)
        if (row['endpoints']['game']['pid'] != old_pid
                or any(row['observed_live_context'][key] != identity[key]
                       for key in ('idol', 'plan', 'max_stamina'))
                or any(row.get(key) for key in _OTHER_ROOTS)):
            raise ValueError('settled restart chronology crosses a different game or unsupported root')
        parents = [row[key] for key in _PARENTS if row.get(key)]
        rows.append(row)
        if has_native_opening_root(row):
            if parents:
                raise ValueError('settled restart opening also names a parent')
            return list(reversed(rows))
        if len(parents) != 1:
            raise ValueError('settled restart chronology lacks one verified native parent')
        name = parents[0]


def _card_entry(event, state, kernel, row):
    from live_decision.history import normalize_history_text
    from live_decision.upgrade import normalize_upgrade_title

    from .native_card_evidence import dispatched_card_refreshes

    action = event['proposal']['candidate_traces']['high_score']['action']
    slot = action.get('source_slot')
    matches = [card for card in state['deck']['hand']
               if card['instance_id'] == action.get('card_instance_id')]
    visible = [card for card in event.get('hand', ()) if card.get('slot') == slot]
    if (action.get('kind') != 'PLAY_CARD' or type(slot) is not int or slot < 0
            or len(matches) != 1 or len(visible) != 1
            or visible[0].get('card_id') != (kernel.cards[matches[0]['card_id']].canonical_card_id
                                           or matches[0]['card_id'])
            or event.get('selected_skill_support') is not None):
        raise ValueError('settled restart card lacks its original bound hand identity')
    # The full source loader already validated A/B and actual dispatch. This
    # pass reads the final B; it does not manufacture a new input certificate.
    _, final = dispatched_card_refreshes(event, row=row)
    turn = final.get('generation', {}).get('turn')
    if final.get('accepted') is not True or type(turn) is not int or turn != state['turn']:
        raise ValueError('settled restart card turn differs from its complete prestate')
    title = kernel.cards[matches[0]['card_id']].name
    receipt = event.get('single_plus_estimate')
    if receipt is not None:
        if (receipt.get('accepted') is not True or receipt.get('submit_count') != 1
                or receipt.get('card_instance_id') != action['card_instance_id']
                or receipt.get('selected_slot') != slot
                or receipt.get('bound_state', {}).get('turn') != turn):
            raise ValueError('settled restart title receipt is not bound to the submitted card')
        definition = receipt.get('projected_card_id', matches[0]['card_id'])
        title = receipt.get('raw_title')
        if (not isinstance(title, str) or not title
                or normalize_upgrade_title(normalize_history_text(title))
                != normalize_upgrade_title(normalize_history_text(kernel.cards[definition].name))):
            raise ValueError('settled restart title differs from the verified card definition')
    return dict(turn=turn, kind='card', title=title)


def _chronology(rows, kernel, expected):
    from live_decision.high_score.settled_restart import verified_replay_action

    openings = [event for event in rows[0]['live_events']
                if event.get('event') == 'live_opening_state' and event.get('accepted') is True]
    if len(openings) != 1 or openings[0].get('strategy_state_bound') is not True:
        raise ValueError('settled restart chronology lacks its complete opening')
    state = openings[0]['state']
    if state['turn'] != 1:
        raise ValueError('settled restart chronology does not start at the native opening')
    chronology, pending = [], False
    for row in rows:
        submitted = _submissions(row)
        if len(submitted) > 1 or (submitted and pending):
            raise ValueError('settled restart action order lacks complete intervening settlement')
        if submitted:
            event = submitted[0]
            if event.get('execution_submit_count') != 1:
                raise ValueError('settled restart action submission count is not one')
            if event.get('event') == 'live_decision_observe':
                chronology.append(_card_entry(event, state, kernel, row))
            elif event.get('event') == 'live_joint_action':
                action = event['proposal']['action']
                transactions = [item for item in event.get('transactions', ())
                                if item.get('execution_submit_count')]
                if (action.get('kind') != 'USE_DRINK' or len(transactions) != 1
                        or transactions[0].get('execution_submit_count') != 1
                        or action.get('drink_id') not in state['drinks']):
                    raise ValueError('settled restart drink has no unique submitted inventory action')
                observation = transactions[0].get('observation', {})
                title = observation.get('name')
                from live_decision.history import normalize_history_text
                if (not isinstance(title, str) or observation.get('available') is not True
                        or normalize_history_text(title)
                        != normalize_history_text(kernel.drinks[action['drink_id']].name)):
                    raise ValueError('settled restart drink title differs from the submitted definition')
                chronology.append(dict(turn=state['turn'], kind='drink', title=title))
            elif not (event.get('event') == 'live_end_turn'
                      and event.get('end_turn_submit_count') == 1
                      and event.get('proposal', {}).get('action', {}).get('kind') == 'END_TURN'):
                raise ValueError('settled restart chronology contains an unsupported business action')
            pending = True
        settlements = [event for event in row['live_events']
                       if event.get('event') == 'live_committed_state' and event.get('accepted') is True]
        if settlements:
            if len(settlements) != 1 or not pending:
                raise ValueError('settled restart chronology has an unpaired completed state')
            after = settlements[0]['expected_state']
            if after['turn'] not in (state['turn'], state['turn'] + 1):
                raise ValueError('settled restart chronology skips a turn')
            if after['turn'] == state['turn'] + 1:
                for before_ref in state.get('scheduled_effects', ()):
                    if (before_ref.get('source_kind') != 'CARD' or before_ref.get('persistent') is not True
                            or before_ref.get('trigger_phase') != 'EVERY_TURN'):
                        continue
                    matches = [ref for ref in after.get('scheduled_effects', ()) if all(
                        ref.get(key) == before_ref.get(key)
                        for key in ('source_id', 'card_id', 'definition_effect_index', 'source_kind', 'persistent'))]
                    # Discarding an exhausted reference does not fire it again.
                    if len(matches) == 1 and matches[0].get('last_fired_turn') == after['turn']:
                        definition = kernel.cards[before_ref['card_id']]
                        effect = definition.persistent_effects[before_ref['definition_effect_index']]
                        if not any(child.kind.value == 'FREE_PLAY_RANDOM' for child in effect.scheduled_effects):
                            continue
                        chronology.append(verified_replay_action(
                            kernel, before_ref, matches[0], after['deck'], after['turn']))
            state, pending = after, False
    if pending or not chronology or not _same(state, asdict(expected)):
        raise ValueError('settled restart chronology does not end at the verified complete state')
    return chronology


def collect_settled_restart_source(runtime, name, module, trail=()):
    """Verify once, then expose existing business evidence to the pure core."""
    bundle, expected, settlement, row = load_verified_source(runtime, name, module, trail)
    reference = row.get('post_history')
    if (settlement.get('accepted') is not True or settlement.get('strategy_state_bound') is not True
            or not isinstance(reference, dict) or reference.get('accepted') is not True
            or reference.get('returned_unchanged') is not True):
        raise ValueError('settled restart source lacks completed settlement and accepted post history')
    rows = _source_rows(runtime, name, row)
    chronology = _chronology(rows, bundle.kernel, expected)
    opening, = [event for event in rows[0]['live_events']
                if event.get('event') == 'live_opening_state' and event.get('accepted') is True]
    return dict(bundle=bundle, expected=expected, settlement=settlement,
                reference_history=reference, chronology=chronology, opening=opening,
                row=row, old_run=module.HifLiveRunContext(**row['observed_live_context']),
                old_pid=row['endpoints']['game']['pid'])


def settled_restart_history_checkpoint(context, recognizer, saved, module):
    """Use the old core only to compare fresh, current-epoch observations."""
    def checkpoint(frames, single_frame=False, *, return_observation=None):
        return module._inventory_live_checkpoint(
            context, frames, recognizer, identity_threshold=.70,
            identity_min_margin=.05, single_frame=single_frame,
            return_observation=return_observation)

    scope = module._inventory_inspected_return_scope(context, saved['settlement']['core'])
    module._bind_inspected_complete_observer(
        context, checkpoint, scope, identity_threshold=.70, identity_min_margin=.05)
    return checkpoint


def _verify_history_source_navigation(row):
    """A partial restart may retain history, but no unaccounted game input."""
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints
    from .native_pending_components import _navigation_dispatch_matches

    history, recovery = row.get('settled_restart_history', {}), row.get('context_recovery', {})
    game = row.get('endpoints', {}).get('game', {})
    size = game.get('client_size')
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
            or recovery.get('accepted') is not True or recovery.get('returned_to_live') is not True
            or recovery.get('context') != row.get('observed_live_context')
            or recovery.get('game_pid') != game.get('pid')
            or row.get('settlement') is not None or _submissions(row)
            or any(event.get('executed') is True or event.get('drinks_consumed', 0)
                   or event.get('execution_selection_click_count', 0) for event in row.get('live_events', ()))):
        raise ValueError('settled restart archived history is not a navigation-only source')
    context_inputs, history_inputs = recovery.get('inputs', ()), history.get('inputs', ())
    dispatches = row.get('native_input_dispatches', ())
    if (type(row.get('input_count')) is not int or row['input_count'] != len(dispatches)
            or any(type(value) is not int for value in (recovery.get('input_click_count'),
                history.get('input_click_count'), history.get('scroll_count')))
            or recovery.get('input_click_count') != len(context_inputs)
            or history.get('input_click_count') != len(history_inputs)
            or len(history_inputs) < 2 or history_inputs[0].get('phase') != 'open'
            or history_inputs[-1].get('phase') != 'close'
            or any(item.get('phase') != 'scroll' for item in history_inputs[1:-1])
            or history.get('scroll_count') != len(history_inputs)-2
            or len(dispatches) != len(context_inputs)+len(history_inputs)):
        raise ValueError('settled restart archived history navigation accounting changed')
    proofs = [dict(item, phase='open', box=item.get('target')) for item in context_inputs]
    proofs += list(history_inputs)
    for actual, proof in zip(dispatches, proofs, strict=True):
        endpoints = proof.get('endpoints')
        _verified_endpoints(endpoints, game['pid'])
        _same_input_host(row['endpoints'], endpoints)
        if endpoints['game'].get('observed_client_size') != list(size):
            raise ValueError('settled restart archived navigation client size changed')
        if proof.get('phase') == 'scroll':
            swipe = proof.get('swipe', ())
            if (len(swipe) != 5 or any(type(value) is not int for value in swipe)
                    or actual.get('duration') != swipe[4] or swipe[4] <= 0):
                raise ValueError('settled restart archived history swipe is incomplete')
            proof = dict(proof, begin=swipe[:2], end=swipe[2:4])
        if not _navigation_dispatch_matches(actual, proof):
            raise ValueError('settled restart archived history input was not its recorded navigation')


def load_settled_restart_history_source(runtime, name, source, module, trail=()):
    """Reverify archived complete history without claiming fresh components."""
    from live_decision.high_score.settled_restart import prepare_settled_restart

    from .native_opening_root import verify_native_round_context

    if name in trail or source == name or diagnostic_depth_reached(trail):
        raise ValueError('settled restart history provenance is cyclic or unbounded')
    row = load_result(runtime, name)
    if (row.get('native_settled_restart_source') != source
            or row.get('native_settled_restart_history_source')
            or any(row.get(key) for key in (*_PARENTS, *_OTHER_ROOTS[:-1]))):
        raise ValueError('settled restart archived history has another source')
    _verify_history_source_navigation(row)
    saved = collect_settled_restart_source(runtime, source, module, (*trail, name))
    verify_native_round_context(row, module, saved['row'], required=True)
    run = module.HifLiveRunContext(**row['observed_live_context'])
    prepared = prepare_settled_restart(saved['bundle'], saved['expected'], settlement=saved['settlement'],
        reference_history=saved['reference_history'], chronology=saved['chronology'], opening=saved['opening'],
        old_run=saved['old_run'], new_run=run, old_pid=saved['old_pid'],
        new_pid=row['endpoints']['game']['pid'], history=row['settled_restart_history'])
    return saved, prepared, row


def load_verified_settled_restart_source(runtime, name, module, trail=()):
    """Recompute the new binding from saved complete observations, without input."""
    from live_decision.high_score.settled_restart import (
        prepare_settled_restart,
        finalize_settled_restart,
        continue_settled_restart_history,
    )

    from .native_opening_root import verify_native_round_context

    if name in trail or diagnostic_depth_reached(trail):
        raise ValueError('settled restart source is cyclic or unbounded')
    row = load_result(runtime, name)
    source = row.get('native_settled_restart_source')
    if (not source or row.get('live_return') is not True or _submissions(row)
            or any(row.get(key) for key in (*_PARENTS, *_OTHER_ROOTS[:-1]))):
        raise ValueError('settled restart completion is not one zero-business-input observation')
    history_source = row.get('native_settled_restart_history_source')
    if history_source:
        saved, prepared, archived = load_settled_restart_history_source(
            runtime, history_source, source, module, (*trail, name))
        if (row['endpoints']['game']['pid'] != archived['endpoints']['game']['pid']
                or row['endpoints']['game']['client_size'] != archived['endpoints']['game']['client_size']
                or row['observed_live_context'] != archived['observed_live_context']
                or not _same(row.get('settled_restart_history'), archived['settled_restart_history'])):
            raise ValueError('settled restart retained history process, epoch or evidence changed')
        verify_native_round_context(row, module, archived, required=True)
        prepared = continue_settled_restart_history(prepared, current_history=row.get('settled_restart_resume_history', {}),
            run=module.HifLiveRunContext(**row['observed_live_context']), game_pid=row['endpoints']['game']['pid'])
    else:
        saved = collect_settled_restart_source(runtime, source, module, (*trail, name))
        verify_native_round_context(row, module, saved['row'])
        prepared = prepare_settled_restart(saved['bundle'], saved['expected'],
            settlement=saved['settlement'], reference_history=saved['reference_history'],
            chronology=saved['chronology'], opening=saved['opening'], old_run=saved['old_run'],
            new_run=module.HifLiveRunContext(**row['observed_live_context']), old_pid=saved['old_pid'],
            new_pid=row['endpoints']['game']['pid'], history=row['settled_restart_history'])
    components_source = row.get('native_settled_restart_components_source')
    if components_source:
        if not history_source:
            raise ValueError('settled components require verified history continuation')
        from .native_settled_restart_components import (
            load_settled_restart_components_source,
            verify_settled_restart_component_completion,
        )

        reusable = load_settled_restart_components_source(runtime, components_source, module,
            prepared=prepared, history_source=archived)
        verify_settled_restart_component_completion(row, row.get('settlement', {}), reusable)
    bundle, state, event = finalize_settled_restart(prepared, settlement=row.get('settlement', {}))
    if not _same(event, row.get('settled_restart_settlement')):
        raise ValueError('settled restart saved binding differs from its complete observations')
    return bundle, state, event, row
