"""Replay rule-maintained progress without inventing full panel observations."""

import json
from dataclasses import asdict, replace

from route.battle_policy import FORMAL_PROFILES

SCOPE = 'rule_maintained_information_boundary_v1'
PREINPUT_SCOPE = 'rule_maintained_preinput_v1'


def information_execution_enabled(command):
    enabled = command.get('information_boundary_execution', False)
    if type(enabled) is not bool or (enabled
            and command.get('hif_policy_profile') not in FORMAL_PROFILES):
        raise ValueError('information boundary execution requires the explicit strong reference trial')
    return enabled


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def rule_receipt(event):
    if event.get('verification_scope') != SCOPE:
        raise ValueError('native progress has no supported rule scope')
    return event.get('rule_progress', event)


def source_receipt(event):
    if event.get('verification_scope') == PREINPUT_SCOPE:
        return event.get('rule_preinput', event)
    return rule_receipt(event) if is_rule_progress(event) else event


def is_rule_progress(event):
    return isinstance(event, dict) and event.get('verification_scope') == SCOPE


def verify_rule_progress(bundle, candidates, baseline, action, event):
    """Rebuild the receipt from original candidates and its recorded fresh core."""
    from live_decision.high_score.hif_rule_progress import resolve_rule_progress

    saved = rule_receipt(event)
    source = source_receipt(baseline)
    state, rebuilt = resolve_rule_progress(bundle, candidates, baseline_event=source,
        action=action, core=None if saved.get('observation_mode') == 'deterministic_hand_removal' else saved.get('core'),
        submitted=True, action_confirmation=saved.get('action_confirmation'),
        inventory_hand=saved.get('inventory_hand'),
        hand_slot_identity=saved.get('hand_slot_identity'))
    if state is None or rebuilt.get('accepted') is not True or not _same(saved, rebuilt):
        raise ValueError('native rule progress differs from original action and fresh observation')
    if event.get('observation_mode') == 'pending_core_comparison':
        from live_decision.high_score.hif_rule_progress import resolve_rule_pending

        observed, proof = resolve_rule_pending(bundle, candidates, baseline_event=source,
            action=action, initial_core=event['initial_core'], core=event['core'],
            single_plus_estimate=event.get('single_plus_estimate'),
            hand_slot_identity=saved.get('hand_slot_identity'))
        if (observed != state or proof.get('pending_outcome') != 'after'
                or event.get('pending_outcome') != 'after'
                or any(not _same(event.get(key), value) for key, value in proof.items())):
            raise ValueError('pending rule progress cannot distinguish the original states')
    if (event.get('event') != 'live_committed_state'
            or any(not _same(event.get(key), rebuilt[key]) for key in (
                'verification_scope', 'accepted', 'strategy_state_bound', 'run_id', 'revision',
                'checks', 'core', 'expected_state'))
            or event.get('observed_hand_instance_ids') != [ref.instance_id for ref in state.deck.hand]
            or event.get('components')):
        raise ValueError('native rule progress envelope conflicts with its receipt')
    return state


def verify_rule_preinput(row, bundle, baseline):
    """Recheck the new small preinput observation against the verified source."""
    from live_decision.high_score.hif_rule_progress import resolve_rule_preinput

    events = [event for event in row.get('live_events', ())
        if event.get('event') == 'live_preinput_state']
    if row.get('information_boundary_execution') is not True:
        raise ValueError('native rule preinput requires its unique explicit receipt')
    # A frozen continuation does not run the slow-search revalidation. Its
    # source was already verified by the caller; original A/B and dispatched
    # action evidence remain mandatory later in load_pending_source.
    if not events:
        projections = [event for event in row.get('live_events', ()) if event.get('event') == 'live_hand_projection']
        if len(projections) > 1:
            raise ValueError('native rule hand projection is ambiguous')
        if projections:
            from live_decision.high_score.hif_rule_progress import with_hand_projection

            source = with_hand_projection(bundle, source_receipt(baseline), projections[0])
            key = ('rule_preinput' if baseline.get('verification_scope') == PREINPUT_SCOPE else
                   'rule_progress' if is_rule_progress(baseline) else None)
            return dict(baseline, **{key: source}) if key is not None else source
        return baseline
    if len(events) != 1:
        raise ValueError('native rule preinput requires its unique explicit receipt')
    event, = events
    if (event.get('verification_scope') != PREINPUT_SCOPE
            or not isinstance(event.get('rule_preinput'), dict) or event.get('components')):
        raise ValueError('native rule preinput has no supported receipt')
    saved = event['rule_preinput']
    source = source_receipt(baseline)
    projections = [e for e in row.get('live_events', ()) if e.get('event') == 'live_hand_projection']
    if len(projections) > 1:
        raise ValueError('native rule hand projection is ambiguous')
    if projections:
        from live_decision.high_score.hif_rule_progress import with_hand_projection

        source = with_hand_projection(bundle, source, projections[0])
    state, rebuilt = resolve_rule_preinput(bundle, baseline_event=source,
        core=saved.get('core'))
    if (state != bundle.state or rebuilt.get('accepted') is not True or not _same(saved, rebuilt)
            or any(not _same(event.get(key), rebuilt[key]) for key in (
                'verification_scope', 'accepted', 'strategy_state_bound', 'run_id', 'revision',
                'checks', 'core', 'expected_state'))
            or event.get('observed_hand_instance_ids') != [ref.instance_id for ref in state.deck.hand]
            or any(event.get(key, 0) != 0 for key in (
                'execution_submit_count', 'end_turn_submit_count', 'drinks_consumed'))):
        raise ValueError('native rule preinput differs from its original source and fresh core')
    inspected = baseline.get('retained_hand_identity_inspection', baseline.get('hand_identity_inspection'))
    if inspected is not None:
        # This verified observation made no business transition. Carry its
        # already verified title source even for old producers which omitted it.
        return dict(event, retained_hand_identity_inspection=inspected)
    return event


def restore_unsubmitted_intent(runtime, name, bundle, baseline, parent):
    """Reuse a completed zero-input child window against its verified parent."""
    from .native_search_cache import restore_search_cache
    from .native_settlement_resume import load_result

    row = load_result(runtime, name)
    if (row.get('native_inprocess_source') != parent['id']
            or row.get('host_attempt') != parent.get('host_attempt', parent['id'])
            or row.get('endpoints') != parent.get('endpoints')
            or row.get('observed_live_context') != parent.get('observed_live_context')
            or row.get('information_boundary_execution') is not True
            or row.get('input_count') != 0 or row.get('native_input_dispatches')
            or row.get('gray_selection_journal') or row.get('choice_journal')):
        raise ValueError('unsubmitted rule intent has input or different source')
    events = row.get('live_events', ())
    if ([event.get('event') for event in events] != ['live_preinput_state', 'live_decision_observe']
            or any(event.get(key, 0) for event in events for key in (
                'execution_submit_count', 'end_turn_submit_count', 'execution_selection_click_count',
                'identity_probe_click_count', 'total_input_click_count'))):
        raise ValueError('unsubmitted rule intent has an unsupported transaction')
    verify_rule_preinput(row, bundle, baseline)
    decision = events[-1].get('proposal', {}).get('candidate_traces', {}).get('high_score', {})
    trace = decision.get('trace', {})
    profile = 'hif-strong-reference-trial-v1'
    if (decision.get('has_action_intent') is not True or trace.get('policy_profile') != profile
            or (trace.get('run_id'), trace.get('revision')) != (bundle.run_id, bundle.revision)
            or decision.get('action') != trace.get('planning', {}).get('action')):
        raise ValueError('unsubmitted rule decision is unbound')
    return restore_search_cache(runtime, row, bundle, accepted=True, submitted=False,
        profile=profile, information_boundary_execution=True, unsubmitted_plan=trace['planning'])


def promote_rule_bundle(bundle, state, event):
    """Maintain business slots only; this creates no screen or input evidence."""
    from live_decision.high_score.opening import (
        verified_expired_start_drinks,
        verified_expired_start_scores,
        expected_permanent_inventory_keys,
    )

    receipt = rule_receipt(event)
    if (event.get('accepted') is not True or not _same(event.get('expected_state'), asdict(state))
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)):
        raise ValueError('native rule progress promotion has no bound state')
    slots = tuple(bundle.drink_slots or ())
    action = receipt.get('action', {})
    if action.get('kind') == 'USE_DRINK':
        from live_decision.high_score.contracts import ActionKind, GameAction
        from live_decision.high_score.hif_rule_progress import rule_drink_slots

        slots = rule_drink_slots(bundle, GameAction(**dict(action,kind=ActionKind.USE_DRINK)),
            receipt.get('action_confirmation'))
    if sorted(drink for _, drink in slots) != sorted(state.drinks):
        raise ValueError('native rule progress drink slots differ from remaining resources')
    return replace(bundle, state=state, drink_slots=slots,
        hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(bundle, state),
        expired_start_scores=verified_expired_start_scores(bundle, state),
        expired_start_drinks=verified_expired_start_drinks(bundle, state))


def restore_rule_source(runtime, row, module, trail=()):
    """Bind a verified parent to the same process and epoch, without old history."""
    from .native_opening_root import verify_native_round_context
    from .native_settlement_resume import load_verified_source

    if row.get('handoff_resume') is not None:
        from .native_handoff_resume import restore_handoff_source
        return restore_handoff_source(runtime, row, module, trail)

    name = row.get('native_rule_resume_source')
    if not name or row.get('information_boundary_execution') is not True:
        raise ValueError('native rule resume was not explicitly enabled')
    bundle, state, baseline, parent = load_verified_source(runtime, name, module, (*trail, row['id']))
    verify_native_round_context(row, module, parent)
    if (not is_rule_progress(baseline)
            or row.get('observed_live_context') != parent.get('observed_live_context')
            or row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid']
            or state.terminal):
        raise ValueError('native rule resume source process or epoch changed')
    return promote_rule_bundle(bundle, state, baseline), baseline


def reconcile_legacy_hand_order(bundle, state, event):
    """Correct only an old deterministic receipt's ordering, retaining the original."""
    from live_decision.high_score.hif_mcts_cache import action_from_json
    from live_decision.high_score.hif_rule_progress import resolve_rule_progress

    saved = rule_receipt(event)
    if saved.get('observation_mode') != 'deterministic_hand_removal' or 'hand_order_revision' in saved:
        return state, event, None
    action = action_from_json(saved['action'])
    after, receipt = resolve_rule_progress(bundle,(state,),baseline_event=saved['source_baseline'],
        action=action,core=None,submitted=True,action_confirmation=saved['action_confirmation'])
    if after is None or receipt.get('accepted') is not True:
        raise ValueError('native legacy hand order could not be reconstructed')
    if bundle.kernel._canonical_probability_state(after) != bundle.kernel._canonical_probability_state(state):
        raise ValueError('native legacy hand order reconciliation changed business state')
    corrected = dict(event,rule_progress=receipt,
        **{key:receipt[key] for key in ('checks','core','expected_state')},
        observed_hand_instance_ids=[ref.instance_id for ref in after.deck.hand])
    return after, corrected, dict(version=2,original_receipt=saved,corrected_receipt=receipt,
        submission_authorized=False,reason='preserve_survivor_order_from_verified_before')


def load_verified_rule_source(runtime, name, module, trail, row):
    """Recompute original submitted outcomes; never trust a saved expected state."""
    from .native_opening_root import verify_native_round_context
    from .native_settlement_resume import load_pending_source, decoded_dispatched_action

    if row.get('information_boundary_execution') is not True:
        raise ValueError('native rule progress was not explicitly enabled')
    events = [event for event in row.get('live_events', ())
        if event.get('event') == 'live_committed_state' and event.get('accepted') is True]
    if len(events) != 1 or not is_rule_progress(events[0]):
        raise ValueError('native rule progress requires one accepted receipt')
    if row.get('choice_journal'):
        from .native_choice_resume import load_confirmed_choice_source
        projected, candidates, baseline, verified = load_confirmed_choice_source(runtime, name, module, trail,
            allow_closed_transaction=True)
        settlement = row.get('settlement', row['choice_journal'].get('settlement'))
        if not _same(settlement, events[0]) or not _same(verified, row):
            raise ValueError('native rule choice settlement differs from its actual confirmation journal')
        state = verify_rule_progress(projected.source_bundle, candidates, baseline, projected.action, events[0])
        return projected.source_bundle, state, events[0], row
    pending = row.get('native_pending_source')
    support_receipt = row.get('native_pending_support_receipt')
    if support_receipt is not None and not pending:
        raise ValueError('native support correction requires its original submitted source')
    bundle, candidates, baseline, submitted_row = load_pending_source(runtime, pending or name, module,
        allow_closed_transaction=True, trail=(*trail, name) if pending else trail,
        observed_support_receipt=support_receipt)
    if pending:
        verify_native_round_context(row, module, submitted_row)
        if (row.get('observed_live_context') != submitted_row.get('observed_live_context')
                or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                       for event in row.get('live_events', ()))):
            raise ValueError('native pending rule progress is not zero-submit same-run completion')
    action, _ = decoded_dispatched_action(submitted_row, bundle)
    state = verify_rule_progress(bundle, candidates, baseline, action, events[0])
    if pending and row['endpoints']['game']['pid'] != submitted_row['endpoints']['game']['pid']:
        from .native_rule_process_recovery import verify_process_recovery
        verify_process_recovery(row, submitted_row, bundle, action, events[0])
    return bundle, state, events[0], row if pending else submitted_row
