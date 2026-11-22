"""Resume this acceptance run from complete native records, without inferred fields."""
import re
import copy
import json
import math
from dataclasses import asdict, replace

from route.battle_policy import FORMAL_PROFILES

from . import diagnostic_depth_reached


def _submitted_observation_failure(row):
    """An audited card submit followed only by a failed, zero-input observer."""
    from live_decision.hif_live_recovery import closed_observer_record
    from live_decision.high_score.contracts import ActionKind

    from .native_card_evidence import _fresh

    if (row.get('status') != 'STOP' or row.get('live_return') is not False
            or not closed_observer_record(row) or row.get('transaction_closed')
            or row.get('error') or row.get('failure_event') or row.get('native_pending_source')
            or row.get('native_navigation_preparation')
            or row.get('information_boundary_execution') is not True
            or row.get('native_input_preparation', {}).get('dispatch_state') != 'dispatch_attempted'):
        return None
    events = row.get('live_events', ())
    failures = [e for e in events if e.get('event') == 'live_committed_state']
    cache = row.get('mcts_search_cache', {})
    if (len(failures) != 1 or cache.get('record_id') != row.get('id')
            or cache.get('phase') != 'pending_intent'):
        return None
    failure = failures[0]
    if (failure.get('accepted') is not False or failure.get('executed') is not False
            or failure.get('execution_submit_count') != 0 or failure.get('total_input_click_count') != 0
            or failure not in [{k: v for k, v in event.items() if k != 'mcts_commit'}
                for event in row.get('published_components', ())]
            or any(e is not failure and not (e.get('event') == 'live_preinput_state' and e.get('accepted') is True)
                and e.get('event') != 'live_decision_observe' for e in events)):
        return None
    action, report = decoded_dispatched_action(row)
    for refresh in report.get('authoritative_refresh_events', ()):
        _fresh(refresh, action.source_slot)
    selection = report.get('execution_selection_click_count')
    if (action.kind is not ActionKind.PLAY_CARD or report.get('execution_submit_count') != 1
            or type(selection) is not int or selection not in (0, 1)
            or row.get('input_count') != selection + 1
            or len(row.get('native_input_dispatches', ())) != selection + 1
            or report.get('identity_probe_click_count') != 0):
        return None
    return failure


def is_submitted_completion_stop(row):
    """Recognize the old unresolved dispatch, never assert game consumption."""
    from live_decision.hif_live_recovery import closed_observer_record
    from live_decision.high_score.contracts import ActionKind

    from .native_submitted_navigation import is_pending_observation_stop

    try:
        if (row.get('upstream_ui_journal') is not None
                and row.get('status') == 'STOP' and row.get('live_return') is False
                and not row.get('transaction_closed') and not row.get('native_pending_source')
                and closed_observer_record(row)):
            action, _ = decoded_dispatched_action(row)
            return action.kind is ActionKind.PLAY_CARD
        if is_pending_observation_stop(row):
            return True
        return _submitted_observation_failure(row) is not None
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


def require_pending_completion_checkpoint(checkpoint):
    """Reuse a verified complete receipt, or reconstruct the dispatched intent."""
    from live_decision.high_score.contracts import PlanType
    from live_decision.high_score.hif_information_boundary import classify_information_boundary

    if checkpoint.get('completed_observation'):
        from .native_handoff_policy import phase
        from .native_submitted_navigation import pending_observation_report, completed_pending_observation

        row, policy = checkpoint['row'], checkpoint['policy']
        if (not completed_pending_observation(row) or pending_observation_report(row) is None
                or checkpoint.get('pending') or checkpoint.get('selected') or checkpoint.get('untouched_opening')
                or checkpoint['state'] != checkpoint['bundle'].state
                or phase(policy) != 'observed_successor' or policy._pending is not None
                or checkpoint['cache_report'].get('restored') is not True):
            raise ValueError('HIF completed observation lacks its verified settled computation')
        return
    deferred = (checkpoint.get('policy') is None
        and checkpoint.get('cache_report', {}).get('deferred_until_observed_settlement') is True)
    if (not is_submitted_completion_stop(checkpoint['row']) or checkpoint.get('pending') is not True
            or checkpoint.get('selected') or checkpoint['state'] != checkpoint['bundle'].state
            or not deferred and (checkpoint['policy']._phase != 'pending_intent'
                or checkpoint['policy']._pending != (checkpoint['state'], checkpoint['action']))):
        raise ValueError('HIF completion recovery lacks the original submitted pending action')
    bundle, action = checkpoint['bundle'], checkpoint['action']
    dispatch_source = checkpoint.get('dispatch_source', checkpoint['row'])
    from live_decision.high_score.contracts import ActionKind

    if action.kind is ActionKind.END_TURN:
        require_pending_gallery_checkpoint(dict(checkpoint, row=dispatch_source))
        return
    if (not is_submitted_completion_stop(dispatch_source) or dispatch_source.get('native_pending_source')):
        raise ValueError('HIF completion recovery lacks its original dispatch evidence')
    decoded, _ = decoded_dispatched_action(dispatch_source, bundle)
    if (_submitted_observation_failure(dispatch_source) is not None
            or dispatch_source.get('upstream_ui_journal') is not None):
        # The normal loader already rebuilt all original outcomes and restored
        # the matching pending computation. No OCR reason or hand-size guess
        # should turn this confirmed dispatch into a new submission.
        if decoded != action or not checkpoint['expected']:
            raise ValueError('HIF pending observation changed the original dispatched action')
        return
    boundary = classify_information_boundary(bundle.kernel, bundle.state, action, plan_type=PlanType.SENSE)
    expected = checkpoint['expected']
    states = expected if isinstance(expected, tuple) else (expected,)
    if (decoded != action or boundary.kind != 'deterministic_same_turn'
            or not boundary.successor.deck.hand
            or len(boundary.successor.deck.hand) != len(bundle.state.deck.hand) - 1
            or not states or any(bundle.kernel._canonical_probability_state(state)
                != bundle.kernel._canonical_probability_state(boundary.successor) for state in states)):
        raise ValueError('HIF completion recovery changed the original deterministic outcome')


def is_submitted_hand_identity_stop(row):
    """A successful PLAY reached an ambiguous new hand, never an unsubmitted retry."""
    from live_decision.hif_live_recovery import closed_observer_record
    from live_decision.high_score.contracts import ActionKind
    try:
        from .native_submitted_navigation import is_pending_inventory_read_stop
        if is_pending_inventory_read_stop(row):
            return True
        if (row.get('status') != 'STOP' or row.get('live_return') is not False or row.get('error') or row.get('failure_event')
                or row.get('transaction_closed') or not closed_observer_record(row)
                or not row.get('upstream_ui_journal') or row.get('native_pending_source')):
            return False
        action,submitted = decoded_dispatched_action(row)
        settlements = [e for e in row['live_events'] if e.get('event') == 'live_committed_state']
        if action.kind is not ActionKind.PLAY_CARD or len(settlements) != 1:
            return False
        settlement = settlements[0]
        proof = settlement.get('rule_progress',{})
        # The overall executed flag is false when its settlement fails. The
        # successful submit is proved by the existing dispatcher/audit reader.
        ambiguity = (settlement.get('reason') == 'original_outcome_not_unique'
            and type(proof.get('unique_outcome_count')) is int and proof['unique_outcome_count'] > 1
            and len(proof.get('matching_candidate_indices',())) >= 2)
        families = proof.get('core',{}).get('families',())
        slot_ambiguity = (settlement.get('reason') == 'inventory hand has no unique Live slot assignment'
            and not any(k in proof for k in ('candidate_count','matching_candidate_indices','unique_outcome_count'))
            and any(len(f)>1 and sum(set(other)==set(f) for other in families)>1 for f in families))
        return (submitted.get('execution_submit_count') == 1
            and settlement.get('accepted') is False and (ambiguity or slot_ambiguity)
            and proof.get('reason') == settlement['reason'] and proof.get('submitted') is True
            and proof.get('checks') == dict(submitted=True,baseline_binding=True,action_binding=True,fresh_core=True)
            and proof.get('information_boundary',{}).get('kind') == 'synchronize'
            and settlement.get('core') == proof.get('core'))
    except (ValueError,KeyError,TypeError,AttributeError,IndexError):
        return False


def require_pending_hand_identity_checkpoint(checkpoint):
    from live_decision.high_score.hif_rule_progress import needs_hand_inventory, resolve_rule_progress

    from .native_rule_progress_resume import source_receipt
    if (not is_submitted_hand_identity_stop(checkpoint['row']) or checkpoint.get('pending') is not True
            or checkpoint.get('selected') or checkpoint['state'] != checkpoint['bundle'].state
            or checkpoint['policy']._phase != 'pending_intent'
            or checkpoint['policy']._pending != (checkpoint['state'],checkpoint['action'])):
        raise ValueError('HIF hand identity recovery lacks its original submitted checkpoint')
    saved, = [e['rule_progress'] for e in checkpoint['row']['live_events'] if e.get('event') == 'live_committed_state']
    after,proof = resolve_rule_progress(checkpoint['bundle'],checkpoint['expected'],
        baseline_event=source_receipt(checkpoint['event']),action=checkpoint['action'],core=saved['core'],submitted=True)
    if (after is not None or proof != saved
            or not (needs_hand_inventory(checkpoint['bundle'],checkpoint['expected'],proof)
                or proof.get('reason') == 'inventory hand has no unique Live slot assignment')):
        raise ValueError('HIF hand identity recovery changed original ambiguous outcomes')


def is_submitted_gallery_stop(row):
    """One audited SKIP with an unresolved arrival; reason text grants nothing."""
    from live_decision.hif_live_recovery import closed_observer_record
    from live_decision.high_score.contracts import ActionKind

    try:
        events = row['live_events']
        preparation = row.get('native_input_preparation') or {}
        if (row.get('status') != 'STOP' or row.get('error') or row.get('failure_event')
                or row.get('transaction_closed') or not closed_observer_record(row)
                or not (row.get('native_inprocess_source') or row.get('native_rule_resume_source'))
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches', ())) != 1
                or preparation.get('dispatch_state') != 'dispatch_attempted'
                or (preparation.get('x'), preparation.get('y')) != (663, 802)
                or row.get('native_navigation_preparation')):
            return False
        submitted = [event for event in events if event.get('execution_submit_count') or event.get('end_turn_submit_count')]
        if len(submitted) != 1:
            return False
        event, = submitted
        if (event.get('event') != 'live_end_turn' or event.get('accepted') is not False
                or event.get('end_turn_submit_count') != 1
                or event.get('execution_submit_count') != 1
                or not event.get('arrival_observations')):
            return False
        # Reuse the independent three SKIP readings, exact actual dispatch,
        # post-success arrival and all High-endpoint checks. A proposal or
        # dispatch_attempted marker alone never proves a successful action.
        action, submitted = decoded_dispatched_action(row)
        return action.kind is ActionKind.END_TURN and submitted is event
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def require_pending_gallery_checkpoint(checkpoint):
    """The complete old chain must replay before observing the submitted END."""
    from live_decision.high_score.contracts import ActionKind

    if (not is_submitted_gallery_stop(checkpoint['row']) or checkpoint.get('pending') is not True
            or checkpoint.get('selected') or checkpoint['action'].kind is not ActionKind.END_TURN
            or checkpoint['state'] != checkpoint['bundle'].state):
        raise ValueError('HIF gallery recovery lacks its verified submitted END checkpoint')
    policy = checkpoint['policy']
    deferred = policy is None and checkpoint['cache_report'].get('deferred_until_observed_settlement') is True
    if not deferred and (getattr(policy, '_phase', None) != 'pending_intent'
            or policy._pending != (checkpoint['state'], checkpoint['action'])):
        raise ValueError('HIF turn recovery lacks its original complete computation')
    expected = checkpoint['expected']
    states = expected if isinstance(expected, tuple) else (expected,)
    if (not states or any(state.terminal or state.pending_choice or state.resolution_stack
            or state.turn != checkpoint['state'].turn+1
            or state.turns_remaining != checkpoint['state'].turns_remaining-1 for state in states)):
        raise ValueError('HIF gallery recovery changed the submitted END outcomes')


def _verified_endpoints(endpoints, game_pid):
    if (not isinstance(endpoints, dict) or not {'agent', 'maa', 'game'} <= endpoints.keys()
            or endpoints['game'].get('pid') != game_pid
            or any(type(value.get('pid')) is not int or value['pid'] <= 0
                   or value.get('integrity_rid') != 12288 for value in endpoints.values())):
        raise ValueError('native input permissions are unverified')


def same_endpoint_observation(actual, observed, game_pid):
    """Compare original endpoint evidence while retaining final SDK additions."""
    _verified_endpoints(observed, game_pid)
    _verified_endpoints(actual, game_pid)
    if set(actual) != set(observed):
        return False
    return all(set(value) - set(observed[name]) <= {'created_at_100ns', 'executable'}
        and json.dumps({k:value.get(k) for k in observed[name]}, sort_keys=True)
            == json.dumps(observed[name], sort_keys=True)
        for name,value in actual.items())


def reports_pending_submission(row):
    """Route the reported submission to its dispatch verifier."""
    return row.get('transaction_closed') is not True and any(
        event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
        for event in row.get('live_events', ()))


def decoded_dispatched_action(row, bundle=None):
    """Read one actual card/drink/SKIP dispatch, never a proposal alone."""
    from live_decision.high_score.contracts import ActionKind, GameAction

    game_pid = row['endpoints']['game']['pid']
    dispatches = row.get('native_input_dispatches', ())
    if not dispatches:
        raise ValueError('native input dispatches are absent')
    for entry in dispatches:
        _verified_endpoints(entry.get('endpoints'), game_pid)
    if row.get('upstream_ui_journal') is not None:
        from live_decision.hif_upstream_policy import decode_upstream_action

        action = decode_upstream_action(row, bundle=bundle)
        event_name = 'live_end_turn' if action.kind is ActionKind.END_TURN else 'live_upstream_card'
        events = [event for event in row.get('live_events', ())
            if event.get('event') == event_name]
        if len(events) > 1:
            raise ValueError('upstream UI has more than one action report')
        # A crash can precede final publication. The actual dispatch above is
        # authoritative; this envelope never asserts game consumption.
        event = events[0] if events else dict(event=event_name,
            action=row['upstream_ui_journal']['action'], execution_submit_count=1,
            accepted=False, executed=False)
        if action.kind is ActionKind.END_TURN and not events:
            event['end_turn_submit_count'] = 1
        if event.get('action') != row['upstream_ui_journal']['action']:
            raise ValueError('upstream action publication differs from dispatch intent')
        if events and event.get('ui') != row['upstream_ui_journal']:
            raise ValueError('upstream UI publication differs from its actual journal')
        return action, event
    events = [event for event in row['live_events']
              if event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)]
    if len(events) != 1:
        raise ValueError('native source does not contain exactly one submission')
    event = events[0]
    if event.get('event') != 'live_joint_action':
        raise ValueError('native submission lacks a current dispatch receipt')
    if any(type(event.get(key, 0)) is not int or event.get(key, 0) not in (0, 1)
           for key in ('execution_submit_count', 'end_turn_submit_count')):
        raise ValueError('native submission counts are invalid')
    raw = dict(event['proposal']['action'])
    raw['kind'] = ActionKind(raw['kind'])
    action = GameAction(**raw)
    action.validate()

    def dispatched_click(x, y, endpoints):
        _verified_endpoints(endpoints, game_pid)
        matches = [index for index, entry in enumerate(dispatches) if entry.get('kind') != 'swipe'
                   and (entry.get('x'), entry.get('y')) == (x, y)
                   # The controller audit retains size tuples in memory;
                   # logger events already crossed JSON and contain lists.
                   # Compare the exact persisted shape, retaining every value.
                   and same_endpoint_observation(entry['endpoints'], endpoints, game_pid)]
        if len(matches) != 1:
            raise ValueError('joint confirmation has no unique audited input')
        return matches

    if action.kind is ActionKind.USE_DRINK:
        from ..opening_observer import valid_opening_frame_ids

        transactions = event.get('transactions', ())
        used = [item for item in transactions if item.get('use_submitted') is True]
        if (event.get('event') != 'live_joint_action' or event.get('execution_submit_count') != 1
                or len(used) != 1 or sum(t.get('execution_submit_count', 0) for t in transactions) != 1
                or bundle is None or (action.source_slot, action.drink_id) not in (bundle.drink_slots or ())):
            raise ValueError('native drink lacks one bound submitted bottle')
        used = used[0]
        name_frames = used.get('page_frame_ids', ())
        before_frames = used.get('before_frame_ids', ())
        if (used.get('slot') != action.source_slot or used.get('input_may_have_been_dispatched') is not True
                or used.get('strategy_state_bound') is not True or used.get('calibration_only') is not False
                or used.get('use_control_geometry') != 'miyu_action_box'
                or not valid_opening_frame_ids(before_frames) or len(before_frames) != 1
                or not valid_opening_frame_ids(name_frames) or len(name_frames) != 1):
            raise ValueError('native drink submission evidence is incomplete')
        from live_decision.status_panel import _normalize_status_text
        observation = used.get('observation', {})
        if (observation.get('name') != _normalize_status_text(bundle.kernel.drinks[action.drink_id].name)
                or observation.get('available') is not True or type(observation.get('confidence')) not in (int, float)
                or not 0 <= observation['confidence'] <= 1):
            raise ValueError('native drink identity or availability is unverified')
        confirmations = [item for item in used.get('inputs', ()) if item.get('phase') == 'use']
        if len(confirmations) != 1 or any(step.get('phase') == 'use_retry' for step in used['inputs']):
            raise ValueError('native drink confirmation is not unique')
        confirmation = confirmations[0]
        box = confirmation.get('box', ())
        def valid_box(value):
            return (isinstance(value, (tuple, list)) and len(value) == 4
                and all(type(v) is int for v in value) and min(value[2:]) > 0)
        if not valid_box(box):
            raise ValueError('native drink confirmation geometry is unverified')
        fresh = used.get('use_preinput_observation') or {}
        if ((row.get('formal_params') or {}).get('hif_policy_profile') not in
                FORMAL_PROFILES
                or not valid_box(fresh.get('use_box')) or tuple(fresh['use_box']) != tuple(box)
                or not isinstance(used.get('use_preinput_frame_id'), str) or not used['use_preinput_frame_id']):
            raise ValueError('native drink confirmation differs from its observed current button')
        x, y, width, height = box
        dispatched_click(x + width // 2, y + height // 2, confirmation.get('endpoints'))
    else:
        raise ValueError('native source action is unsupported')
    return action, event


def load_result(runtime, name, *, allow_closed_transaction=False):
    from ..hif_recovery_reports import read_recovery_report

    if not isinstance(name, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', name) is None:
        raise ValueError('invalid native evidence name')
    row = read_recovery_report(runtime / ('result-' + name + '.json'))
    host_attempt = row.get('host_attempt', name)
    if (not isinstance(host_attempt, str)
            or re.fullmatch(r'[0-9]+-[a-z0-9-]+', host_attempt) is None):
        raise ValueError('invalid native host evidence name')
    exit_path = runtime / (host_attempt + '.exit.json')
    finished = json.loads(exit_path.read_text(encoding='utf8')) if exit_path.exists() else {}
    if row.get('id') != name:
        raise ValueError('native evidence is incomplete')
    if finished.get('exit_code') != 0:
        from live_decision.hif_live_recovery import closed_observer_record

        from .native_submitted_navigation import completed_observation_host_exited

        if not closed_observer_record(row) and not completed_observation_host_exited(runtime, row):
            if not allow_closed_transaction:
                raise ValueError('native evidence is incomplete')
            from .native_inprocess_resume import verify_closed_transaction_record
            verify_closed_transaction_record(row)
    return row


def exact_state(raw):
    from live_decision.high_score.golden import _state
    normalized = copy.deepcopy(raw)
    for zone in ('hand', 'draw_pile', 'discard', 'removed', 'held'):
        normalized['deck'][zone] = [card['instance_id'] for card in raw['deck'][zone]]
    state = _state(normalized)
    # The native kernel can settle passion to integer zero at a turn boundary;
    # golden._state casts it to float for fixtures. Preserve the explicit
    # recorded numeric value/type before the strict full-state round trip.
    if 'passion' in raw:
        if type(raw['passion']) not in (int, float) or not math.isfinite(raw['passion']):
            raise ValueError('native passion is not an explicit finite number')
        state = replace(state, passion=raw['passion'])
    if json.dumps(asdict(state), sort_keys=True) != json.dumps(raw, sort_keys=True):
        raise ValueError('native state cannot be restored without defaults or omissions')
    state.validate()
    return state


def load_verified_source(runtime, name, module, trail=()):
    from live_decision.high_score.opening import verify_hif_drink_consumption, verify_hif_committed_observation
    from live_decision.high_score.contracts import ActionKind

    from .native_opening_root import has_native_opening_root, load_native_opening_root, verify_native_round_context
    from .native_support_resume import restore_native_history
    if name in trail or diagnostic_depth_reached(trail):
        raise ValueError('native evidence chain is cyclic or unbounded')
    row = load_result(runtime, name, allow_closed_transaction=True)
    if row.get('pending_business_frontier') is not None:
        from .native_pending_frontier import load_completed_frontier
        restored, _, original, _ = load_completed_frontier(runtime, name, module)
        return restored, restored.state, original['settlement'], original
    from .native_pending_outcome import verify_unconsumed, is_unconsumed_observation
    if is_unconsumed_observation(row):
        bundle, expected, baseline, dispatch = load_pending_source(runtime, name, module, trail=trail)
        action, _ = decoded_dispatched_action(dispatch, bundle)
        state = verify_unconsumed(bundle, expected, baseline, action, row['settlement'],
            dispatch.get('verified_single_plus_estimate'))
        return bundle, state, row['settlement'], row
    from .native_rule_progress_resume import is_rule_progress, load_verified_rule_source
    if any(is_rule_progress(event) and event.get('accepted') is True
           for event in row.get('live_events', ()) if event.get('event') == 'live_committed_state'):
        return load_verified_rule_source(runtime, name, module, trail, row)
    from .native_submitted_navigation import pending_observation_report, completed_pending_observation
    completed_observation = completed_pending_observation(row)
    if completed_observation:
        pending_observation_report(row)
    if row.get('live_return') is not True and not completed_observation:
        raise ValueError('native submission did not complete')
    if row.get('native_manual_choice_source'):
        from .native_manual_choice_restart import load_verified_manual_choice_source
        return load_verified_manual_choice_source(runtime, name, module)
    if row.get('choice_journal'):
        from .native_choice_resume import load_verified_choice_source
        return load_verified_choice_source(runtime, name, module, trail)
    if row.get('native_settled_restart_source'):
        from .native_settled_restart import load_verified_settled_restart_source
        return load_verified_settled_restart_source(runtime, name, module, trail)
    if row.get('native_pending_restart_source'):
        from .native_pending_restart import load_verified_pending_restart_source
        return load_verified_pending_restart_source(runtime, name, module, trail)
    if row.get('native_restart_source'):
        from live_decision.high_score.restart import prepare_hif_restart, finalize_hif_restart

        from .native_restart_source import collect_restart_source

        if any(e.get('execution_submit_count', 0) or e.get('end_turn_submit_count', 0)
               for e in row['live_events']):
            raise ValueError('restart completion contains an additional business submission')
        saved = collect_restart_source(runtime, row['native_restart_source'], module)
        new_run = module.HifLiveRunContext(**row['observed_live_context'])
        prepared = prepare_hif_restart(saved['bundle'], saved['candidates'],
            old_run=saved['old_run'], new_run=new_run, old_pid=saved['old_pid'],
            new_pid=row['endpoints']['game']['pid'], history=row['restart_history'],
            expected_history=saved['expected_history'], action=saved['action'])
        bundle, state, event = finalize_hif_restart(prepared,
            settlement=row['settlement'], baseline_components=saved['baseline']['components'])
        if json.dumps(event, sort_keys=True) != json.dumps(row.get('restart_settlement'), sort_keys=True):
            raise ValueError('restart saved binding differs from its verified complete observations')
        return bundle, state, event, row
    pending = row.get('native_pending_source')
    if row.get('native_pending_components_source') and not pending:
        raise ValueError('pending component source requires a pending native action')
    if row.get('native_pending_support_receipt') is not None and not pending:
        raise ValueError('pending support receipt requires a pending native action')
    expected_support = None
    resumed_components = None
    dispatch_row = row
    historical_witness = None
    inventory_source = None
    if pending:
        saved_settlements = [event for event in row['live_events']
                             if event.get('event') == 'live_committed_state' and event.get('accepted')]
        observed = (exact_state(saved_settlements[0]['expected_state'])
                    if len(saved_settlements) == 1 else None)
        bundle, expected_support, baseline, dispatch_row = load_pending_source(
            runtime, pending, module, observed_state=observed,
            observed_support_receipt=row.get('native_pending_support_receipt'))
        historical_witness = dispatch_row.get('historical_draw_witness')
        verify_native_round_context(row, module, dispatch_row)
        if (row.get('execution_live_context') != dispatch_row['observed_live_context']
                or row['endpoints']['game']['pid'] != dispatch_row['endpoints']['game']['pid']
                or any(row['observed_live_context'][key] != dispatch_row['observed_live_context'][key]
                       for key in ('idol','plan','max_stamina'))
                or any(e.get('execution_submit_count',0) or e.get('end_turn_submit_count',0)
                       for e in row['live_events'])):
            raise ValueError('pending completion is not a zero-submit continuation of the same run')
        row = dict(row, observed_live_context=row['execution_live_context'])
        if row.get('native_pending_components_source'):
            resumed_components = load_pending_components_source(runtime,
                row['native_pending_components_source'], module, pending,
                bundle, expected_support, baseline)
    run = module.HifLiveRunContext(**row['observed_live_context'])
    if pending:
        pass
    elif has_native_opening_root(row):
        bundle, baseline = load_native_opening_root(row, module)
    elif row.get('native_selected_source'):
        from .native_selected_resume import load_selected_source, verify_selected_continuation
        bundle, _, baseline, selected_source = load_selected_source(
            runtime, row['native_selected_source'], module, (*trail, name))
        from .native_opening_root import restore_selected_execution_epoch
        row = restore_selected_execution_epoch(row, selected_source)
        verify_native_round_context(row, module, selected_source)
        verify_selected_continuation(row, selected_source)
    elif row.get('native_inprocess_source'):
        from .native_inprocess_resume import restore_inprocess_source
        bundle, baseline = restore_inprocess_source(runtime, row, module, trail)
    elif row.get('native_rule_resume_source') and row.get('handoff_resume') is not None:
        from .native_handoff_resume import restore_handoff_source
        bundle, baseline = restore_handoff_source(runtime, row, module, trail)
    else:
        bundle, state, previous, parent = load_verified_source(
            runtime, row['native_resume_source'], module, (*trail, name))
        inventory_source = (bundle, state, previous, parent['id'])
        verify_native_round_context(row, module, parent)
        bundle, _, baseline = restore_native_history(runtime, module, bundle, state, settlement=previous,
            reference=row['resume_reference'], current=row['resume_history'],
            old_run=module.HifLiveRunContext(**parent['observed_live_context']), new_run=run,
            old_pid=parent['endpoints']['game']['pid'], new_pid=row['endpoints']['game']['pid'])
    if not pending:
        from .native_hand_support import bind_planning_hand_support
        from .native_preinput_history import verify_preinput_history
        bundle = bind_planning_hand_support(bundle, dispatch_row, runtime=runtime, module=module, trail=trail,
                                            inventory_source=inventory_source)
        verify_preinput_history(dispatch_row, bundle, baseline)
    action, submitted = decoded_dispatched_action(dispatch_row, bundle)
    from .native_single_plus import recorded_single_plus_estimate
    single_plus_estimate = recorded_single_plus_estimate(bundle, action, submitted, source_row=dispatch_row)
    if pending and dispatch_row.get('verified_pending_support_receipt') is not None:
        single_plus_estimate = None
    if not pending and submitted.get('executed' if action.kind is ActionKind.PLAY_CARD else 'accepted') is not True:
        raise ValueError('native submitted action did not complete')
    if action.kind is ActionKind.PLAY_CARD:
        from types import SimpleNamespace

        from live_decision.high_score.live_policy import BoundHifOpeningPolicy

        from .native_selected_support import bind_recorded_selected_support
        support_policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5])
        support_policy.response = SimpleNamespace(action=action)
        if bind_recorded_selected_support(support_policy, submitted, require_submitted=True,
                                          source_row=dispatch_row):
            if single_plus_estimate is not None:
                raise ValueError('native card cannot bind exact support and single-plus estimation together')
            expected_support = support_policy.committed_transition()
        elif single_plus_estimate is not None and not pending:
            from .native_card_evidence import dispatched_card_refreshes
            _, final = dispatched_card_refreshes(submitted, row=dispatch_row, bundle=bundle)
            support_policy.bind_selected_single_plus_estimate(single_plus_estimate, require_submitted=True)
            support_policy.bind_generated_card_resource_preview(final['resource_transaction'],
                                                               final['visual_fields']['score'])
            support_policy.bind_normal_card_score_preview(final['resource_transaction'],
                                                         final['visual_fields']['score'])
            support_policy.bind_single_plus_typed_score_preview(final['resource_transaction'],
                                                               final['visual_fields']['score'])
            expected_support = support_policy.committed_transition()
        if not pending:
            from .native_hand_support import inventory_supported_outcomes
            inventory_expected = inventory_supported_outcomes(
                bundle, action, submitted, dispatch_row, inventory_source)
            if inventory_expected is not None:
                expected_support = inventory_expected
    settlement, = [e for e in row['live_events'] if e.get('event') == 'live_committed_state' and e.get('accepted')]
    if json.dumps(settlement.get('single_plus_estimate'), sort_keys=True) != json.dumps(single_plus_estimate, sort_keys=True):
        raise ValueError('native settlement single-plus receipt differs from original submitted evidence')
    if (settlement.get('run_id') != run.run_id or settlement.get('revision') != run.revision
            or not settlement.get('checks') or not all(v is True for v in settlement['checks'].values())):
        raise ValueError('native settlement binding or checks are incomplete')
    if resumed_components is not None:
        from .native_pending_components import verify_pending_component_completion
        verify_pending_component_completion(row, settlement, resumed_components)
    state = exact_state(settlement['expected_state'])
    baseline_components = baseline.get('components')
    if baseline_components is None and pending and baseline.get('verification_scope') in (
            'rule_maintained_information_boundary_v1', 'rule_maintained_preinput_v1'):
        # Match the full observer: the independently verified rule baseline
        # has no component snapshots; this settlement supplies all five.
        baseline_components = {}
    if baseline_components is None:
        raise ValueError('native settlement baseline has no verified components')
    if bundle.state.hand_support or state.hand_support:
        from live_decision.high_score.hif_support_continuity import hand_support_candidate_continuity
        if not hand_support_candidate_continuity(bundle.state, state):
            raise ValueError('native settlement hand support has no current prestate source')
        if expected_support is None:
            from .native_hand_support import require_supported_outcome
            expected_support = require_supported_outcome(bundle, action, submitted, dispatch_row, state)
    comparison = None
    if completed_observation:
        from live_decision.high_score.opening import resolve_hif_pending_observation

        if expected_support is None:
            raise ValueError('complete pending publication lacks its original rule outcomes')
        # Replay the same pending comparator as the producer, including its
        # rule-only predecessor support. Do not re-interpret its full receipt
        # as an ordinary settlement requiring an older status-panel snapshot.
        resolved, comparison = resolve_hif_pending_observation(bundle, expected_support,
            core=settlement['core'], components=settlement['components'],
            baseline_components=baseline_components, committed_action=action,
            single_plus_estimate=single_plus_estimate)
        if (comparison.get('accepted') is not True or comparison.get('pending_outcome') != 'after'
                or resolved != state):
            raise ValueError('complete pending publication differs from its verified actual outcome')
    elif expected_support is not None:
        possible = expected_support if isinstance(expected_support, tuple) else (expected_support,)
        if single_plus_estimate is not None or state not in possible:
            # The native resolver also preserves already-exhausted item refs
            # for observed zero counters (948: item 442). Re-run that exact
            # producer against the original components; never normalize the
            # saved state or infer a missing effect from its recorded flags.
            from live_decision.high_score.opening import resolve_hif_committed_observation
            resolved, proof = resolve_hif_committed_observation(bundle, possible,
                core=settlement['core'], components=settlement['components'],
                baseline_components=baseline_components, committed_action=action,
                single_plus_estimate=single_plus_estimate)
            if proof.get('accepted') is not True or resolved != state:
                raise ValueError('pending native state is not an outcome of the dispatched action')
    if comparison is None:
        comparison = verify_hif_committed_observation(bundle, state, core=settlement['core'],
            components=settlement['components'], baseline_components=baseline_components,
            committed_action=action)
    if (not comparison['accepted'] or tuple(comparison.get('observed_hand_instance_ids', ()))
            != tuple(settlement.get('observed_hand_instance_ids', ()))):
        raise ValueError('recorded complete components do not verify native state')
    if action.kind is ActionKind.USE_DRINK:
        verify_hif_drink_consumption(baseline['components']['drinks'], settlement['components']['drinks'], action)
        # Keep state and slots at the same pre-action point for the next
        # history verification. Only its accepted rebound adopts post slots.
    if historical_witness is not None:
        # Diagnostic output only. It is never trusted on the next load; the
        # witness and all original component/epoch checks are reconstructed.
        row = dict(row, historical_draw_witness=historical_witness)
    from .native_preinput_history import verify_settlement_history
    settlement = verify_settlement_history(row, bundle, settlement, run=run)
    return bundle, state, settlement, row


def observe_verified_settlement_history(runtime, name, module, context, capture_frames, result):
    """Read only fresh history against one verified settlement's original epoch."""
    from .native_opening_root import load_native_round_context, verify_native_round_context
    from .native_preinput_history import bind_settlement_history

    bundle, expected, saved, source = load_verified_source(runtime, name, module)
    source_run = module.HifLiveRunContext(**source['observed_live_context'])
    current_run = module.HifLiveRunContext(**result['observed_live_context'])
    if (any(source['endpoints']['game'][key] != result['endpoints']['game'][key]
            for key in ('pid', 'client_size'))
            or any(getattr(source_run, key) != getattr(current_run, key)
                   for key in ('idol', 'plan', 'max_stamina'))
            or (saved.get('run_id'), saved.get('revision')) != (source_run.run_id, source_run.revision)
            or (bundle.run_id, bundle.revision) != (source_run.run_id, source_run.revision)):
        raise ValueError('native history observation source process, identity or epoch changed')
    verify_native_round_context(result, module, parent=source)
    round_context = load_native_round_context(runtime, source, module, run=source_run)
    # These are only comparison inputs. Do not publish committed state, restore
    # a search policy/cache, or transfer any archived input credential.
    context.gkh_hif_live_context = source_run
    context.gkh_hif_opening_state = bundle
    context.gkh_hif_round_context = round_context
    context.restore_inspected_family()
    result['native_resume_source'] = source['id']
    result['live_return'] = False
    observed = observe_history(context, module, capture_frames, expected,
        verified_history_settlement=saved)
    result['history_observation'] = observed
    if observed.get('accepted') is True:
        # Validate the complete history and original action/state relation, but
        # deliberately discard the attached settlement: this operation reads.
        baseline = {key: value for key, value in saved.items() if key != 'history_observation'}
        bind_settlement_history(bundle, baseline, observed, run=source_run,
            game_pid=source['endpoints']['game']['pid'])
        result['live_return'] = True
    return observed


def observe_history(context, module, capture_frames, expected=None, *, verified_history_settlement=None):
    from live_decision.history import observe_latest_history
    recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
    context.gkh_hif_card_art = module.FrozenMaskedArt.load(recognizer, module.MODEL_ROOT / 'hif')
    def checkpoint(frames, **kwargs):
        if len(frames) == 1:
            kwargs.setdefault('single_frame', True)
        return module._inventory_live_checkpoint(context, frames, recognizer,
            identity_threshold=.70, identity_min_margin=.05, **kwargs)

    settlement = getattr(context, 'gkh_hif_committed_event', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    historical = verified_history_settlement is not None
    if historical:
        # The caller has loaded/verified this source and checked game identity.
        # Its old epoch remains unchanged; this is only a comparison baseline
        # for new observations, not a published current committed state.
        bundle = getattr(context, 'gkh_hif_opening_state', None)
        settlement = verified_history_settlement
        if (bundle is None or not isinstance(settlement, dict)
                or settlement.get('accepted') is not True or settlement.get('strategy_state_bound') is not True
                or settlement.get('event') != 'live_committed_state' or not isinstance(settlement.get('core'), dict)
                or not isinstance(settlement.get('checks'), dict) or not settlement['checks']
                or not all(value is True for value in settlement['checks'].values())
                or (settlement.get('run_id'), settlement.get('revision')) != (bundle.run_id, bundle.revision)):
            raise ValueError('history observation baseline lacks its verified original settlement')
    if (run is not None and isinstance(settlement, dict) and settlement.get('accepted') is True
            and settlement.get('event') == 'live_committed_state'
            and settlement.get('strategy_state_bound') is True
            and (historical or (getattr(context, 'gkh_hif_committed_state', None) is not None
                and (settlement.get('run_id'), settlement.get('revision')) == (run.run_id, run.revision)))
            and isinstance(settlement.get('core'), dict)):
        scope = module._inventory_inspected_return_scope(context, settlement['core'])
        module._bind_inspected_complete_observer(context, checkpoint, scope,
            identity_threshold=.70, identity_min_margin=.05)
    if expected is not None and not expected.deck.hand:
        context.gkh_hif_expected_empty_hand = (run.run_id, run.revision)
    try:
        return observe_latest_history(context, capture_frames=lambda: capture_frames(context),
            checkpoint=checkpoint, frame_identifier=module.frame_identifier,
            round_context=getattr(context, 'gkh_hif_round_context', None))
    finally:
        context.gkh_hif_expected_empty_hand = None


def load_pending_source(runtime, name, module, *, observed_state=None, observed_support_receipt=None,
                        allow_closed_transaction=False, trail=()):
    """Reconstruct only the already dispatched joint action's settlements."""
    from .native_opening_root import has_native_opening_root, load_native_opening_root, verify_native_round_context
    from .native_support_resume import restore_native_history
    if name in trail or diagnostic_depth_reached(trail):
        raise ValueError('native pending evidence chain is cyclic or unbounded')
    row = load_result(runtime, name, allow_closed_transaction=allow_closed_transaction)
    if row.get('native_pending_source'):
        from .native_submitted_navigation import verify_pending_inventory_parent

        parent = load_result(runtime, row['native_pending_source'], allow_closed_transaction=allow_closed_transaction)
        verify_pending_inventory_parent(row, parent, module)
        # The continuation owns its navigation/cache receipt, but the original
        # row alone owns the business dispatch. Never manufacture a submission.
        return load_pending_source(runtime, parent['id'], module, observed_state=observed_state,
            observed_support_receipt=observed_support_receipt,
            allow_closed_transaction=allow_closed_transaction, trail=(*trail, name))
    # A persisted diagnostic is not a newly computed transition witness.
    row.pop('historical_draw_witness', None)
    row.pop('verified_single_plus_estimate', None)
    row.pop('verified_pending_support_receipt', None)
    inventory_source = None
    if has_native_opening_root(row):
        bundle, baseline = load_native_opening_root(row, module)
    elif row.get('native_selected_source'):
        from .native_selected_resume import load_selected_source, verify_selected_continuation
        bundle, _, baseline, selected_source = load_selected_source(
            runtime, row['native_selected_source'], module, (*trail, name))
        from .native_opening_root import restore_selected_execution_epoch
        row = restore_selected_execution_epoch(row, selected_source)
        verify_native_round_context(row, module, selected_source)
        verify_selected_continuation(row, selected_source)
    elif row.get('native_inprocess_source'):
        from .native_inprocess_resume import restore_inprocess_source
        bundle, baseline = restore_inprocess_source(runtime, row, module, trail)
    elif row.get('native_rule_resume_source'):
        from .native_rule_progress_resume import restore_rule_source
        bundle, baseline = restore_rule_source(runtime, row, module, trail)
    else:
        bundle, state, saved, parent = load_verified_source(runtime, row['native_resume_source'], module,
                                                          (*trail, name))
        inventory_source = (bundle, state, saved, parent['id'])
        verify_native_round_context(row, module, parent)
        run = module.HifLiveRunContext(**row['observed_live_context'])
        bundle, _, baseline = restore_native_history(runtime, module, bundle, state, settlement=saved,
            reference=row['resume_reference'], current=row['resume_history'],
            old_run=module.HifLiveRunContext(**parent['observed_live_context']), new_run=run,
            old_pid=parent['endpoints']['game']['pid'], new_pid=row['endpoints']['game']['pid'])
    return replay_pending_submission(runtime, row, module, bundle, baseline,
        inventory_source=inventory_source, observed_state=observed_state,
        observed_support_receipt=observed_support_receipt)


def replay_pending_submission(runtime, row, module, bundle, baseline, *, inventory_source=None,
                              observed_state=None, observed_support_receipt=None, observed_core=None):
    """Replay the original input from a separately established business boundary.

    This neither validates an execution budget nor grants new UI input. Both
    the ordinary chain reader and a current checkpoint must establish their
    source before calling this same submission/effect verifier.
    """
    from types import SimpleNamespace

    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy

    from .native_hand_support import bind_planning_hand_support
    from .native_preinput_history import verify_preinput_history
    bundle = bind_planning_hand_support(bundle, row, runtime=runtime, module=module,
                                        inventory_source=inventory_source)
    if row.get('information_boundary_execution') is True:
        from .native_rule_progress_resume import verify_rule_preinput
        baseline = verify_rule_preinput(row, bundle, baseline)
    else:
        verify_preinput_history(row, bundle, baseline)
    action, submitted = decoded_dispatched_action(row, bundle)
    from .native_single_plus import recorded_single_plus_estimate
    single_plus_estimate = recorded_single_plus_estimate(bundle, action, submitted, source_row=row)
    hold_receipt = row.get('hold_journal')
    policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5],
                                   hold_ready=bool(hold_receipt))
    policy.response = SimpleNamespace(action=action)
    if row.get('upstream_ui_journal') is not None:
        from ..hif_upstream_support import bind_recorded_support

        bind_recorded_support(policy, row['upstream_ui_journal'])
    if action.kind is ActionKind.PLAY_CARD and row.get('upstream_ui_journal') is None:
        from .native_card_evidence import dispatched_card_refreshes
        from .native_selected_support import bind_recorded_selected_support
        refreshes = dispatched_card_refreshes(submitted, row=row, bundle=bundle)
        if any(e.get('accepted') is not True for e in refreshes):
            raise ValueError('pending card source lacks both fresh input checks')
        # Replay the exact selected tier before deriving its card-only cost
        # contract, in the same order as the live input transaction.
        if (bind_recorded_selected_support(policy, submitted, require_submitted=True, source_row=row)
                and single_plus_estimate is not None):
            raise ValueError('native card cannot bind exact support and single-plus estimation together')
        if single_plus_estimate is not None:
            policy.bind_selected_single_plus_estimate(single_plus_estimate, require_submitted=True)
        policy.bind_generated_card_resource_preview(refreshes[-1]['resource_transaction'],
                                                   refreshes[-1]['visual_fields']['score'])
        # Live _prepare_state consumes ordinary NORMAL-card B forecasts too.
        # Restore that same original observation for an unenhanced card;
        # requiring a multi-plus receipt silently dropped 2006's bound score.
        policy.bind_normal_card_score_preview(refreshes[-1]['resource_transaction'],
                                              refreshes[-1]['visual_fields']['score'])
        if single_plus_estimate is not None:
            policy.bind_single_plus_typed_score_preview(refreshes[-1]['resource_transaction'],
                                                       refreshes[-1]['visual_fields']['score'])
        policy._bind_typed_single_precision_display(refreshes[-1]['resource_transaction'],
                                                    refreshes[-1]['visual_fields']['score'])
    # No score correction is invented from the current screen. Native complete
    # observations must resolve one of these actual rule outcomes or stop.
    from .native_hand_support import inventory_supported_outcomes
    inventory_expected = inventory_supported_outcomes(bundle, action, submitted, row, inventory_source)
    if observed_support_receipt is not None:
        from .native_selected_support import bind_observed_pending_support

        if action.kind is not ActionKind.PLAY_CARD or inventory_expected is not None:
            raise ValueError('pending support conflicts with another action or inventory projection')
        verified = bind_observed_pending_support(policy, row, submitted, observed_support_receipt,
                                                single_plus_estimate=single_plus_estimate,
                                                baseline_event=baseline)
        row = dict(row, verified_pending_support_receipt=verified)
        single_plus_estimate = None
    if hold_receipt:
        from ..hif_hold_transaction import replay_confirmed_hif_hold

        if inventory_expected is not None or observed_support_receipt is not None:
            raise ValueError('HIF hold confirmation conflicts with another action projection')
        card_receipt = row.get('upstream_ui_journal')
        if card_receipt:
            origin_index = max(item['dispatch_index'] for item in card_receipt['inputs']
                if item['phase'] == 'confirm' and item['dispatch_index'] < len(row['native_input_dispatches']))
        else:
            use = next(step for transaction in submitted['transactions'] if transaction.get('use_submitted') is True
                       for step in transaction['inputs'] if step['phase'] == 'use')
            x, y, w, h = use['box']
            origin_index = next(i for i, item in enumerate(row['native_input_dispatches'])
                if item.get('kind') != 'swipe' and (item.get('x'), item.get('y')) == (x + w//2, y + h//2))
        return bundle, replay_confirmed_hif_hold(policy, row, origin_dispatch_index=origin_index), baseline, row
    witness = None
    if observed_state is not None and inventory_expected is None:
        from live_decision.high_score.hif_observed_transition import observed_committed_transition
        witness = observed_committed_transition(policy, observed_state)
    elif observed_core is not None and inventory_expected is None:
        from live_decision.high_score.hif_observed_transition import observed_core_transition
        witness = observed_core_transition(policy, observed_core)
    if inventory_expected is not None:
        expected = inventory_expected
    elif witness is None:
        expected = policy.committed_transition()
    else:
        expected = witness.states[0] if len(witness.states) == 1 else witness.states
        row = dict(row, historical_draw_witness=witness.audit())
    if single_plus_estimate is not None:
        row = dict(row, verified_single_plus_estimate=single_plus_estimate)
    if action.kind is ActionKind.END_TURN:
        from .native_upstream_effects import reconcile_pending_end

        expected, correction = reconcile_pending_end(runtime, row, module, bundle, expected)
        if correction is not None:
            row = dict(row, verified_pending_upstream_effects=correction)
    return bundle, expected, baseline, row


def load_pending_components_source(runtime, name, module, pending_name, bundle, expected, baseline):
    from .native_pending_components import load_pending_components_source as load
    return load(runtime, name, module, pending_name, bundle, expected, baseline)


def pending_inspection_geometry_proposal(frame_candidates, expected, module):
    """Propose positions after one empty initial frame; never authorize input.

    The caller must run the ordinary complete fresh preflight before selection.
    No selected-state or identity evidence is synthesized from these two rows.
    """
    from live_decision.selection import live_selection_geometry_evidence

    states = expected if isinstance(expected, tuple) else (expected,)
    if not states or {len(state.deck.hand) for state in states} != {5}:
        return None
    if len(frame_candidates) != 3 or sorted(map(len, frame_candidates)) != [0, 5, 5]:
        return None
    rows = [row for row in frame_candidates if row]
    if not module._same_live_candidate_geometry(rows[0], rows[1]):
        return None
    for row in rows:
        evidence = live_selection_geometry_evidence(row)
        if (not evidence['available'] or evidence['ambiguous'] or evidence['raised_slots']):
            return None
    return rows[-1]
