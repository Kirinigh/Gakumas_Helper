"""A closed business checkpoint plus its one pending action, without old UI history.

This restores business/calculation only. The current owner, process-recovery
proof, fresh input observations and cumulative execution budget remain separate.
"""
from types import SimpleNamespace
from dataclasses import replace

from ..hif_recovery_reports import read_recovery_business_report


def _read(runtime, name):
    from .native_search_window import _name
    return read_recovery_business_report(runtime / f'result-{_name(name)}.json')


def load_pending_frontier(runtime, name, module, *, observed_core=None, business_successor=None):
    """Recheck the last closed card, then replay its already-submitted successor.

    The closed receipt is a durable business checkpoint. Its original complete
    prestate and fresh preinput receipt remain mandatory; its action and full
    settlement are recalculated. We do not replay all earlier UI transactions.
    """
    from ..high_score.opening import (
        verified_expired_start_drinks,
        verified_expired_start_scores,
        expected_permanent_inventory_keys,
    )
    from .native_opening_root import PARENT_FIELDS, load_native_opening_root, verify_native_round_context
    from .native_search_cache import _cache_source, verify_selected_cache, _require_current_payload
    from ..high_score.contracts import ActionKind
    from .native_inprocess_resume import _same_input_host, promote_verified_settlement, verify_closed_transaction_record
    from .native_settlement_resume import exact_state, decoded_dispatched_action, replay_pending_submission
    from ..high_score.hif_mcts_cache import definitions
    from .native_rule_progress_resume import verify_rule_progress

    source = _read(runtime, name)
    returned = source.get('observer_returned', {})
    if (source.get('information_boundary_execution') is not True
            or source.get('actual_policy_profile') != 'hif-strong-reference-trial-v1'
            or source.get('transaction_closed') or source.get('live_return') is not False
            or source.get('error') or source.get('choice_journal')
            or source.get('status') != 'STOP' or not isinstance(returned, dict)
            or returned.get('record') != source['id']
            or type(returned.get('sequence')) is not int or returned['sequence'] < 1
            or [key for key in PARENT_FIELDS if source.get(key)] != ['native_inprocess_source']):
        raise ValueError('pending frontier requires one original in-process strong submission')
    parent = _read(runtime, source['native_inprocess_source'])
    verify_closed_transaction_record(parent)
    for field in ('host_attempt', 'endpoints', 'observed_live_context', 'handoff_id',
                  'handoff_opening', 'round_context', 'formal_params', 'actual_policy_profile'):
        if parent.get(field) != source.get(field):
            raise ValueError('pending frontier differs from its completed predecessor: '+field)
    for dispatch in source.get('native_input_dispatches', ()):
        _same_input_host(source['endpoints'], dispatch.get('endpoints', {}))
    owner, payload = _cache_source(runtime, source, business_successor=business_successor)
    profile = 'hif-strong-reference-trial-v1'
    if (owner['id'] != source['id'] or payload is None or payload.get('profile') != profile
            or payload.get('phase') != 'pending_intent'):
        raise ValueError('pending frontier requires its own complete strong calculation')
    _require_current_payload(payload, profile)
    opening = _read(runtime, source['mcts_search_cache']['computation_origin'])
    root, _ = load_native_opening_root(opening, module)
    if (opening.get('handoff_opening') != source.get('handoff_opening')
            or definitions(root.kernel) != payload['candidate']['computation']['definitions']):
        raise ValueError('pending frontier rules differ from the fixed complete opening')
    verify_native_round_context(source, module, parent, required=True)
    event, = [event for event in parent['live_events'] if event.get('event') == 'live_committed_state'
              and event.get('accepted') is True]
    receipt = event['rule_progress']
    state = exact_state(receipt['before'])
    # This first frontier consumes a closed ordinary-card boundary with no
    # remaining drinks. It never guesses a physical slot for a surviving drink.
    if state.drinks or receipt.get('action', {}).get('kind') != 'PLAY_CARD':
        raise ValueError('pending frontier needs an explicit supported closed-card resource map')
    cards = {ref.instance_id: ref for ref in state.deck.all_cards}
    originals = {ref.instance_id: ref for ref in root.state.deck.all_cards}
    if any(cards.get(key) != ref for key, ref in originals.items()):
        raise ValueError('pending frontier changed the confirmed permanent deck')
    generated = {key: ref for key, ref in cards.items() if key not in originals}
    if (set(generated) != {f'task275-created-{index}' for index in range(state.created_card_count)}
            or any(ref.card_id not in root.kernel.cards for ref in generated.values())):
        raise ValueError('pending frontier generated-card ledger is incomplete')
    bundle = replace(root, state=state, drink_slots=(),
        hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(root, state),
        expired_start_scores=verified_expired_start_scores(root, state),
        expired_start_drinks=verified_expired_start_drinks(root, state))
    preinput, = [event for event in parent['live_events'] if event.get('event') == 'live_preinput_state']
    before_baseline = preinput['rule_preinput']['source_baseline']
    bundle, candidates, baseline, _ = replay_pending_submission(runtime, parent, module, bundle, before_baseline)
    action, _ = decoded_dispatched_action(parent, bundle)
    settled = verify_rule_progress(bundle, candidates, baseline, action, event)
    bundle = promote_verified_settlement(bundle, settled, event)
    action, _ = decoded_dispatched_action(source, bundle)
    if action.kind is not ActionKind.PLAY_CARD:
        raise ValueError('pending frontier must retain the original submitted card')
    computation = verify_selected_cache(runtime, source, bundle, SimpleNamespace(profile=profile, action=action),
        business_successor=business_successor)
    restored = replay_pending_submission(runtime, source, module, bundle, event, observed_core=observed_core)
    proof = dict(version=1, scope='closed_business_checkpoint_and_pending_submission',
        opening_source=opening['id'], completed_source=parent['id'], pending_source=source['id'],
        computation=computation, input_authorized=False, previous_ui_history_replayed=False)
    return (*restored, proof)


def _unsubmitted_upstream_plan(row):
    """Retain a computed card before any UI input or a rejected first selection."""
    from ..hif_upstream_ui import UPSTREAM_REVISION

    event, = row['live_events']
    ui = row.get('upstream_ui_journal') or {}
    preparation = row.get('native_upstream_input_preparation') or {}
    proposal = event.get('proposal') or {}
    plan = proposal.get('trace', {}).get('planning') or {}
    inputs = ui.get('inputs', ())
    if (event.get('event') != 'live_upstream_card' or event.get('accepted') is not False
            or event.get('executed') is not False or event.get('execution_submit_count') != 0
            or event.get('ui') != ui or ui.get('kind') != 'upstream_hif_card_ui'
            or ui.get('revision') != UPSTREAM_REVISION or ui.get('response_confirmed') is not False
            or ui.get('playable') is not False
            or proposal.get('outcome') != 'DECIDE' or proposal.get('has_action_intent') is not True
            or proposal.get('execution_authorized') is not False or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('action', {}).get('kind') != 'PLAY_CARD'
            or any(action != plan['action'] for action in
                (proposal.get('action'), event.get('action'), ui.get('action')))):
        raise ValueError('completed frontier selection was not an unsubmitted complete plan')
    untouched = inputs == [] and row.get('native_upstream_input_preparation') is None
    rejected = (len(inputs) == 1 and inputs[0].get('phase') == 'select'
        and inputs[0].get('dispatched') is False and inputs[0].get('dispatch_index') == 0
        and inputs[0].get('job_id') is None and preparation.get('dispatch_state') == 'rejected_before_dispatch'
        and preparation.get('input_may_have_been_dispatched') is False
        and all(preparation.get(key) == inputs[0].get(key) for key in ('x', 'y', 'input_deadline'))
        and preparation.get('authorization_floor') == inputs[0].get('captured_at'))
    if not (untouched or rejected):
        raise ValueError('completed frontier has an unresolved upstream input')
    # Its original completed search-window file and exact cached intent/state
    # are checked by restore_search_cache, never inferred from this event.
    return plan


def verify_completion_continuation(row, completed, proof):
    """A returned, zero-input precheck stopped after a verified completion.

    Only its own complete calculation may continue. This business
    check neither replaces the complete control chain nor grants another window.
    """
    cache = row.get('mcts_search_cache') or {}
    receipt = row.get('handoff_resume') or {}
    returned = row.get('observer_returned') or {}
    events = row.get('live_events', ())
    logical = proof['process_recovery']['logical_context']
    if (row.get('status') != 'STOP' or row.get('live_return') is not False
            or row.get('information_boundary_execution') is not True
            or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
            or row.get('transaction_closed') or row.get('error')
            or row.get('native_rule_resume_source') != completed['id']
            or receipt.get('source') != completed['id'] or receipt.get('boundary_kind') != 'settled'
            or row.get('handoff_id') != completed.get('handoff_id')
            or row.get('actual_policy_profile') != 'hif-strong-reference-trial-v1'
            or row.get('observed_live_context') != logical
            or row.get('execution_live_context') != logical
            or row.get('endpoints', {}).get('game', {}).get('pid') != proof['process_recovery']['new_pid']
            or returned.get('record') != row.get('id')
            or type(returned.get('sequence')) is not int or returned['sequence'] < 1
            or cache.get('record_id') != row.get('id')
            or cache.get('phase') not in ('observed_successor', 'pending_intent')
            or (cache.get('run_id'), cache.get('revision')) != (logical['run_id'], logical['revision'])
            or any(row.get(key) for key in ('native_input_preparation',
                'native_navigation_preparation', 'native_pending_source', 'native_selected_source',
                'choice_journal', 'gray_selection_journal'))
            or len(events) != 1):
        raise ValueError('completed frontier continuation is not a closed zero-input boundary')
    if cache['phase'] == 'pending_intent':
        return _unsubmitted_upstream_plan(row)
    if any(row.get(key) for key in ('mcts_search_window', 'upstream_ui_journal', 'native_upstream_input_preparation')):
        raise ValueError('completed frontier continuation contains an unverified decision')
    event = events[0]
    if (event.get('event') != 'live_decision_exception_stop'
            or event.get('runtime_phase') != 'runtime_prechecks'
            or event.get('executed') is not False or event.get('would_submit') is not False
            or event.get('submit_dispatched') is not False
            or event.get('input_may_have_been_dispatched') is not False
            or any(event.get(key) for key in ('execution_submit_count', 'execution_selection_click_count',
                'identity_probe_click_count', 'execution_selection_events', 'foreground_preparation_events',
                'identity_probe_events', 'shadow_transaction_events', 'authoritative_refresh_events'))):
        raise ValueError('completed frontier continuation may have performed work or input')


def is_completed_frontier_stop(row):
    """Classify an already restored post-commit boundary, not its authority.

    The fixed-entry loader must still verify the referenced real completion and
    full calculation. This only lets its original bounded continuation contract
    carry a zero-input precheck stop whose identity is already known.
    """
    from ..hif_live_recovery import closed_observer_record

    try:
        if (not closed_observer_record(row)
                or row.get('mcts_search_cache_restore', {}).get('restored') is not True
                or row['mcts_search_cache_restore'].get('restored_complete_policy') is not True):
            return False
        receipt = row['handoff_resume']
        logical = {key: value for key, value in receipt['live_context'].items() if key != 'round_number'}
        verify_completion_continuation(row,
            dict(id=receipt['source'], handoff_id=receipt['handoff_id']),
            dict(process_recovery=dict(logical_context=logical,
                new_pid=row['endpoints']['game']['pid'])))
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def load_completed_frontier(runtime, name, module):
    """Finish a returned observer's complete receipt, without repeating its UI.

    A failure after observation remains in the original record. Replaying its
    actual independent identity/history and rule receipt can establish business
    state, but grants neither current process ownership nor new input.
    """
    import json

    from .native_search_cache import restore_search_cache
    from .native_settlement_resume import decoded_dispatched_action
    from .native_rule_progress_resume import promote_rule_bundle, verify_rule_progress
    from .native_rule_process_recovery import process_recovery

    row = _read(runtime, name)
    continuation = None
    if row.get('native_rule_resume_source') and row.get('live_return') is False:
        continuation = row
        name = row['native_rule_resume_source']
        row = _read(runtime, name)
    exit_record = json.loads((runtime / f'{name}.exit.json').read_bytes())
    if (exit_record.get('exit_code') != 0 or row.get('live_return') is not True
            or row.get('transaction_closed') or not row.get('native_pending_source')
            or not isinstance(row.get('pending_business_frontier'), dict)
            or row.get('mcts_search_cache') or row.get('choice_journal')
            or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                   for event in row.get('live_events', ()))):
        raise ValueError('completed frontier lacks a returned zero-submit observer')
    settlement = row['settlement']
    if [event for event in row['live_events'] if event.get('event') == 'live_committed_state'] != [settlement]:
        raise ValueError('completed frontier has no unique original settlement')
    bundle, expected, baseline, pending, proof = load_pending_frontier(
        runtime, row['native_pending_source'], module, observed_core=settlement['core'],
        business_successor=continuation if (continuation or {}).get('mcts_search_cache', {}).get('phase') == 'pending_intent' else None)
    action, _ = decoded_dispatched_action(pending, bundle)
    after = verify_rule_progress(bundle, expected, baseline, action, settlement)
    recovery = process_recovery(row, pending, bundle, action, settlement)
    rebound = promote_rule_bundle(bundle, after, settlement)
    # Retain original business observations; a later zero-input wrapper owns
    # its own spent decision count and calculation, not a new game state.
    row = dict(row, **{key: pending[key] for key in ('handoff_id', 'handoff_opening',
        'actual_policy_profile')}, frontier_binding_source=pending['id'])
    continuation_proof = dict(process_recovery=recovery)
    unsubmitted_plan, continuation_sources = None, []
    if continuation is not None:
        unsubmitted_plan = verify_completion_continuation(continuation, row, continuation_proof)
        previous = continuation
        while previous['id'] != pending['id']:
            if previous['id'] in continuation_sources or len(continuation_sources) >= 64:
                raise ValueError('completed frontier calculation lineage is cyclic or unbounded')
            verify_completion_continuation(previous, row, continuation_proof)
            continuation_sources.append(previous['id'])
            restored = previous.get('mcts_search_cache_restore') or {}
            if restored.get('restored') is not True or restored.get('restored_complete_policy') is not True:
                raise ValueError('completed frontier lost its original complete calculation lineage')
            parent = restored.get('source_record')
            if parent == pending['id']:
                break
            previous = _read(runtime, parent)
    policy, calculation = restore_search_cache(runtime, continuation or pending, rebound,
        **(dict(before=bundle.state, action=action, submitted=True) if continuation is None else {}),
        **(dict(unsubmitted_plan=unsubmitted_plan) if unsubmitted_plan is not None else {}),
        accepted=True,
        information_boundary_execution=True, profile='hif-strong-reference-trial-v1',
        source_drink_slots=bundle.drink_slots)
    if policy is None or calculation.get('restored') is not True:
        raise ValueError('completed frontier could not retain the complete strong computation')
    # These immutable business bindings came from the original dispatched
    # card. Current observations/endpoints keep their own original values.
    return rebound, policy, row, dict(business_frontier=proof, process_recovery=recovery,
        computation=calculation, input_authorized=False, repeated_navigation=0,
        original_failure=row.get('error'),
        **({'continuation_source': continuation['id'], 'continuation_sources': continuation_sources}
           if continuation is not None else {}))


def bind_current_game_instance(restored, current_task):
    """Relate a verified observer PID to the current native process instance.

    Old observer reports did not save process creation times. The live native
    instance must therefore predate the observer's own saved capture. Its
    identity is kept separately; historical endpoint fields are never filled in.
    """
    import re
    from datetime import datetime, timezone

    from route.cultivation_factory import validate_creation_task

    validate_creation_task(current_task)
    bundle, policy, row, proof = restored
    game = current_task['game']
    observed = row['endpoints']['game']
    frames = row.get('frames', ())
    if (observed.get('pid') != game['pid'] or observed.get('integrity_rid') != 0x3000
            or not frames or proof.get('process_recovery', {}).get('new_pid') != game['pid']):
        raise ValueError('completed frontier has no matching current native game')
    first = frames[0]
    path = first.get('path', '')
    match = re.search(r'([0-9]{8}T[0-9]{12}Z)-[a-f0-9]{32}\.png$', path)
    if (match is None or type(first.get('capture_monotonic')) not in (int, float)
            or first['capture_monotonic'] <= 0):
        raise ValueError('completed frontier lacks its original timed capture')
    captured = datetime.strptime(match[1], '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
    delta = captured - datetime(1601, 1, 1, tzinfo=timezone.utc)
    ticks = (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10
    if not game['created_at_100ns'] <= ticks:
        raise ValueError('current game instance was created after the completed observation')
    instance = dict(game=dict(game), source=row['id'], capture_path=path,
        captured_at_100ns=ticks, capture_monotonic=first['capture_monotonic'])
    return bundle, policy, row, dict(proof, current_game_instance=instance)


def completed_handoff_checkpoint(runtime, name, session, module, *, restored=None):
    """Adapt an already verified completion to the existing continuous host."""
    from .native_opening_root import load_native_round_context
    from .native_handoff_resume import _binding

    if session.authorise_input() is not True:
        raise ValueError('completed frontier requires the current owned handoff')
    record = session.record
    handoff = dict(record['handoff'], route_run_id=record['run_id'])
    live = handoff['live_context']
    bundle, policy, row, proof = (restored if restored is not None
        else load_completed_frontier(runtime, name, module))
    _binding(row, live, handoff)
    if (row['id'] != name or row['endpoints']['game']['pid'] != session.game_pid
            or bundle.state.terminal or bundle.state.pending_choice or bundle.state.resolution_stack
            or (bundle.run_id,bundle.revision) != (live['run_id'],live['revision'])):
        raise ValueError('completed frontier differs from the current game or Live epoch')
    original = _read(runtime, row['frontier_binding_source'])
    round_context = load_native_round_context(runtime, original, module)
    if round_context is None or round_context.round_number != live['round_number']:
        raise ValueError('completed frontier lacks its original round')
    return dict(bundle=bundle,state=bundle.state,event=row['settlement'],row=row,policy=policy,
        observer_module=module,cache_report=proof['computation'],round_context=round_context,
        live=live,source_runtime=runtime,untouched_opening=False,selected=False,pending=False,
        expected=None,action=None,completed_observation=True,reconciled_unconsumed=False,
        completed_frontier=proof,receipt=dict(version=1,source=name,handoff_id=handoff['handoff_id'],
            route_run_id=record['run_id'],ownership_revision=record['ownership']['revision'],
            ledger_revision=handoff['ledger_revision'],live_context=live,boundary_kind='settled'))
