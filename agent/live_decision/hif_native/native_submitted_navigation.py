"""Restore current pending settlement and its actual navigation observations."""
import math

from route.engine import has_single_frame_id

from .native_card_evidence import _same


def is_submitted_inventory_navigation_stop(row):
    """A completed inventory return may continue its original pending submission."""
    return is_pending_inventory_read_stop(row)


def is_pending_inventory_read_stop(row):
    """A failed hand read returned to Live without another business input.

    This is only a shape/evidence check. The pending loader must still resolve
    its parent, verify the original submission and replay its complete state.
    """
    try:
        return pending_inventory_read_report(row) is not None
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


def pending_inventory_read_report(row):
    from live_decision.hif_live_recovery import closed_observer_record

    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints
    from .native_pending_components import _navigation_dispatch_matches

    source = row.get('native_pending_source')
    events = row.get('live_events', ())
    if not source or len(events) != 1:
        return None
    event = events[0]
    failures = event.get('failure_events') or ()
    if not any(f.get('phase') == 'inventory_hand_observation' and f.get('inputs') for f in failures):
        return None
    proof = event.get('rule_progress', {})
    receipt = row.get('handoff_resume', {})
    navigation = row.get('native_navigation_preparation', {})
    dispatches = row.get('native_input_dispatches', ())
    if (row.get('status') != 'STOP' or row.get('live_return') is not False
            or not closed_observer_record(row) or row.get('transaction_closed')
            or row.get('error') or row.get('failure_event') or row.get('native_input_preparation')
            or row.get('information_boundary_execution') is not True
            or row.get('settlement') != event or row.get('published_components') != [event]
            or event.get('event') != 'live_committed_state' or event.get('accepted') is not False
            or event.get('executed') is not False or event.get('execution_submit_count') != 0
            or any(event.get(k, 0) for k in ('end_turn_submit_count', 'execution_selection_click_count'))
            or event.get('hand_identity_disambiguation') != proof
            or proof.get('reason') != 'original_outcome_not_unique'
            or proof.get('submitted') is not True
            or type(proof.get('unique_outcome_count')) is not int or proof['unique_outcome_count'] < 2
            or len(proof.get('matching_candidate_indices', ())) < 2
            or proof.get('checks') != dict(submitted=True, baseline_binding=True, action_binding=True, fresh_core=True)
            or proof.get('information_boundary', {}).get('kind') != 'synchronize'
            or event.get('core') != proof.get('core') or event.get('initial_core') != proof.get('core')
            or not has_single_frame_id(proof.get('core', {}).get('frame_ids'))
            or receipt.get('version') != 1 or receipt.get('source') != source
            or receipt.get('boundary_kind') != 'submitted_pending'
            or row.get('input_count') != 2 or len(dispatches) != 2):
        raise ValueError('pending inventory read lacks its zero-submit continuation proof')
    if len(failures) != 1:
        raise ValueError('pending inventory read has conflicting failures')
    failure = failures[0]
    reason = failure.get('read_failure')
    inputs = failure.get('inputs', ())
    returns = failure.get('return_observations', ())
    if (failure.get('phase') != 'inventory_hand_observation' or failure.get('accepted') is not False
            or not reason or failure.get('reason') != reason or event.get('reason') != reason
            or failure.get('live_state_unchanged') is not True or failure.get('input_click_count') != 2
            or [e.get('phase') for e in inputs] != ['open', 'close']
            or not failure.get('page_arrival_observations')
            or failure['page_arrival_observations'][-1].get('accepted') is not True
            or len(returns) != 1 or returns[0].get('accepted') is not True
            or not has_single_frame_id(failure.get('before_frame_ids'))
            or not has_single_frame_id(failure.get('after_frame_ids'))
            or returns[0].get('frame_ids') != failure['after_frame_ids']
            or set(failure['before_frame_ids']) & set(failure['after_frame_ids'])
            or any(not _navigation_dispatch_matches(actual, saved)
                   for actual, saved in zip(dispatches, inputs, strict=True))):
        raise ValueError('pending inventory read did not verifiably return to unchanged Live')
    game_pid = row['endpoints']['game']['pid']
    for actual in dispatches:
        _verified_endpoints(actual.get('endpoints'), game_pid)
        _same_input_host(row['endpoints'], actual['endpoints'])
        if not _same(actual['endpoints'], dispatches[0]['endpoints']) or actual.get('kind', 'click') != 'click':
            raise ValueError('pending inventory read changed its input endpoint')
    floor, checked = (navigation.get(k) for k in ('authorization_floor', 'authorization_checked_at'))
    if (navigation.get('dispatch_state') != 'dispatch_attempted'
            or any(not _same(navigation.get(k), dispatches[-1].get(k)) for k in ('x', 'y', 'endpoints'))
            or navigation.get('input_deadline', False) is not None
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (floor, checked))
            or checked < floor):
        raise ValueError('pending inventory cleanup was not a fresh confirmed navigation')
    return event


def completed_pending_observation(row):
    """A complete business publication survives later host bookkeeping failure.

    This is not a process-exit or input receipt. The normal source loader must
    still verify those and replay the publication against the original action.
    """
    event = row.get('settlement', {})
    components = event.get('components', {})
    return bool(row.get('native_pending_source') and row.get('status') == 'STOP'
        and row.get('live_return') is not True and row.get('transaction_closed') is not True
        and event.get('event') == 'live_committed_state'
        and event.get('accepted') is True and event.get('strategy_state_bound') is True
        and event.get('pending_outcome') == 'after' and event.get('expected_state')
        and event.get('checks') and all(v is True for v in event['checks'].values())
        and set(components) == {'cards', 'status', 'items', 'drinks', 'turns'}
        and all(value.get('accepted') is True and value.get('returned_unchanged') is True
            and (value.get('run_id'), value.get('revision')) == (event.get('run_id'), event.get('revision'))
            for value in components.values()))


def completed_observation_host_exited(runtime, row):
    """Resolve older nested host names through their real cultivation wrapper."""
    import json

    from .native_inprocess_resume import _same_input_host

    if not completed_pending_observation(row):
        return False
    # Older formal children recorded their inner attempt as host_attempt.
    # Find the actual exited wrapper, retaining both original records as-is.
    for path in sorted(runtime.glob('result-task600-*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        exit_path = runtime / f'{path.stem.removeprefix("result-")}.exit.json'
        if not exit_path.exists() or json.loads(exit_path.read_text(encoding='utf-8')).get('exit_code') != 0:
            continue
        outer = json.loads(path.read_text(encoding='utf-8'))
        if outer.get('id') != path.stem.removeprefix('result-'):
            continue
        matches = [battle for battle in outer.get('cultivation', {}).get('battle_outcomes', ())
            if battle.get('last_record') == row['id'] and row['id'] in battle.get('records', ())
            and battle.get('handoff_id') == row.get('handoff_id') and battle.get('status') == row.get('status')]
        if len(matches) == 1:
            _same_input_host(outer['endpoints'], row['endpoints'])
            pending_observation_report(row)
            return True
    return False


def pending_observation_report(row):
    """Read-only continuation of a dispatched intent, independent of OCR reason.

    This receipt does not certify consumption. The recursive loader still
    validates the original dispatch, computation and complete business state.
    """
    from live_decision.hif_live_recovery import closed_observer_record

    from .native_pending_outcome import is_unconsumed_observation
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    source, event = row.get('native_pending_source'), row.get('settlement', {})
    if not source or event.get('event') != 'live_committed_state':
        return None
    if pending_inventory_read_report(row) is not None:
        return None  # The inventory reader accounts for its own open and close.
    geometry = [f for f in event.get('failure_events', ()) if f.get('phase') == 'inventory_live_geometry']
    for proof in geometry:
        # The current crop and empty-hand readers store a singleton list or one scalar ID.
        frame_ids = proof.get('frame_ids', [proof.get('frame_id')])
        if (proof.get('geometry_stable') is not False or proof.get('selection') is not None
                or proof.get('select_evidence') or not has_single_frame_id(frame_ids)):
            raise ValueError('pending observation lacks current unselected hand geometry')
    receipt, cache = row.get('handoff_resume', {}), row.get('mcts_search_cache', {})
    events, dispatches = row.get('live_events', ()), row.get('native_input_dispatches', ())
    complete = completed_pending_observation(row)
    if (row.get('status') != 'STOP' or not complete and (
                row.get('live_return') is not False or not closed_observer_record(row))
            or row.get('transaction_closed')
            or row.get('error') and not (complete or is_unconsumed_observation(row)) or row.get('failure_event')
            or row.get('information_boundary_execution') is not True
            or event not in events or event not in row.get('published_components', ())
            or event.get('executed') is not False or event.get('execution_submit_count') != 0
            or any(e.get('executed') is True or any(e.get(k, 0) for k in
                ('execution_submit_count', 'end_turn_submit_count', 'execution_selection_click_count',
                 'drinks_consumed')) for e in events)
            or receipt.get('version') != 1 or receipt.get('source') != source
            or receipt.get('boundary_kind') != 'submitted_pending'
            or cache and (cache.get('record_id') != row.get('id') or cache.get('phase') != 'pending_intent')
            or not cache and row.get('mcts_search_cache_restore', {}).get('deferred_until_observed_settlement') is not True):
        raise ValueError('pending observation lacks its unchanged dispatched-intent boundary')
    reset = row.get('native_pending_selection_reset')
    reset_count = 0
    if reset is not None:
        if reset.get('source') != source or reset.get('business_submit_count') != 0:
            raise ValueError('pending preview reset changed its business source')
        for step in reset.get('events', ()):
            from ..hif_hand_inspection import verify_upstream_clear

            for proof in verify_upstream_clear(step):
                actual = dispatches[reset_count]
                if (proof.get('dispatch_index') != reset_count
                        or actual.get('kind', 'click') != 'click'
                        or (actual.get('x'), actual.get('y')) != (5, 5)):
                    raise ValueError('upstream cancellation differs from the actual neutral input')
                reset_count += 1
    observed_count = event.get('total_input_click_count')
    if (type(observed_count) is not int or observed_count < 0
            or row.get('input_count') != len(dispatches)
            or len(dispatches) != reset_count + observed_count):
        raise ValueError('pending observation navigation accounting is incomplete')
    if not dispatches and any(row.get(key, {}).get('dispatch_state') == 'dispatch_attempted'
            for key in ('native_input_preparation', 'native_navigation_preparation')):
        raise ValueError('pending observation has an unaccounted dispatch attempt')
    for actual in dispatches:
        _verified_endpoints(actual.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], actual['endpoints'])
    # Navigation receipts, rather than the diagnostic reason or expected hand
    # size, establish that no new business action belongs to this continuation.
    if reset_count and observed_count == 0:
        prepared = row.get('native_upstream_input_preparation', {})
        floor, checked, deadline = (prepared.get(k) for k in
            ('authorization_floor', 'authorization_checked_at', 'input_deadline'))
        if (prepared.get('dispatch_state') != 'dispatch_attempted'
                or any(prepared.get(k) != dispatches[-1].get(k) for k in ('x', 'y', 'endpoints'))
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (floor, checked, deadline))
                or not floor <= checked < deadline or not 0 < deadline-floor <= 1):
            raise ValueError('pending neutral navigation lacks its final fresh dispatch proof')
    return event


def is_pending_observation_stop(row):
    try:
        return pending_observation_report(row) is not None
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


def verify_pending_inventory_parent(row, parent, module):
    """Bind navigation metadata to the actual original submitted action."""
    from .native_opening_root import verify_native_round_context

    if pending_observation_report(row) is not None:
        if (row['native_pending_source'] != parent.get('id')
                or any(row.get(k) != parent.get(k) for k in
                    ('handoff_id', 'handoff_opening', 'observed_live_context'))
                or row.get('execution_live_context') != parent.get('observed_live_context')
                or row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid']
                or row['formal_params']['hif_policy_profile'] != parent['formal_params']['hif_policy_profile']):
            raise ValueError('pending observation changed its original handoff or execution context')
        verify_native_round_context(row, module, parent, required=True)
        return
    event = pending_inventory_read_report(row)
    if event is None or row['native_pending_source'] != parent.get('id'):
        raise ValueError('pending inventory read has no verified original source')
    originals = [e.get('rule_progress') for e in parent.get('live_events', ())
                 if e.get('event') == 'live_committed_state']
    proof = event['rule_progress']
    if (len(originals) != 1 or not isinstance(originals[0], dict)
            or any(row.get(k) != parent.get(k) for k in ('handoff_id', 'handoff_opening', 'observed_live_context'))
            or row.get('execution_live_context') != parent.get('observed_live_context')
            or row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid']
            or row['formal_params']['hif_policy_profile'] != parent['formal_params']['hif_policy_profile']
            or {k: v for k, v in proof.items() if k != 'core'} !=
               {k: v for k, v in originals[0].items() if k != 'core'}
            or any(proof['core'].get(k) != originals[0]['core'].get(k) for k in ('fields', 'families'))):
        raise ValueError('pending inventory read changed its original action or outcome evidence')
    verify_native_round_context(row, module, parent, required=True)
