"""Restore the evaluated first intent after its observed opening and input audit."""
import time
from copy import deepcopy


def is_saved_opening_intent_stop(row):
    from ..hif_live_recovery import closed_observer_record

    if row.get('handoff_resume') is None:
        return is_original_saved_opening_intent(row)
    events = row.get('live_events', ())
    try:
        return bool(row.get('status') == 'STOP' and closed_observer_record(row)
            and row.get('formal_params', {}).get('hif_policy_profile') == 'hif-strong-reference-trial-v1'
            and row.get('input_count') == row.get('total_input_count') == 0
            and row.get('native_input_dispatches') == []
            and row.get('mcts_search_cache', {}).get('phase') == 'pending_intent'
            and row.get('handoff_resume', {}).get('boundary_kind') == 'unsubmitted_opening'
            and len(events) >= 2 and events[0].get('event') == 'live_opening_state'
            and events[0].get('accepted') is True
            and events[-1].get('event') in ('live_decision_exception_stop', 'live_policy_stop', 'live_selection_stop')
            and all(e.get('executed') is False and e.get('execution_submit_count') == 0
                and e.get('execution_selection_click_count', 0) == 0
                and e.get('identity_probe_click_count', 0) == 0
                and not e.get('end_turn_submit_count', 0) and not e.get('choice_submit_count', 0)
                for e in events)
            and not any(row.get(k) for k in ('choice_journal', 'gray_selection_journal',
                'upstream_ui_journal', 'native_pending_source', 'native_selected_source')))
    except (ValueError, KeyError, TypeError):
        return False


def is_original_saved_opening_intent(row):
    from ..hif_live_recovery import closed_observer_record
    from .native_opening_root import PARENT_FIELDS
    from .native_opening_mapping_resume import verify_opening_navigation

    try:
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or (row.get('mcts_search_cache') or {}).get('phase') != 'pending_intent'
                or row.get('formal_params', {}).get('hif_policy_profile') != 'hif-strong-reference-trial-v1'
                or row.get('handoff_resume') is not None or not row.get('handoff_opening')
                or not closed_observer_record(row) or any(row.get(key) is not None for key in
                    (*PARENT_FIELDS, 'native_input_preparation', 'upstream_ui_journal', 'choice_journal',
                     'gray_selection_journal', 'settlement', 'transaction_closed'))):
            return False
        events = row['live_events']
        opening, = [e for e in events if e.get('event') == 'live_opening_state']
        index = events.index(opening)
        published = [e for e in row.get('published_components', ()) if e.get('event') == 'live_opening_state']
        if (opening.get('accepted') is not True or published and published != [opening]
                or any(e.get('executed') or e.get('execution_submit_count') or e.get('end_turn_submit_count')
                    or e.get('choice_submit_count') or e.get('drinks_consumed') for e in events)
                or any(_reported_action_input(event) for event in events[index+1:])
                or type(row.get('total_input_count')) is not int or row['total_input_count'] < row['input_count']):
            return False
        # The observer's top-level possible-input flag also covers passive
        # reading. The SDK records every attempted input before calling Maa.
        inspections = (opening.get('extended_hand_identity') or {}).get('inspections', ())
        if (row.get('native_upstream_input_preparation') is not None and not inspections
                and not opening.get('hand_identity_inspection')):
            return False
        retained = (row.get('opening_navigation_resume') or {}).get('retained_components', {})
        if any(opening['components'].get(key) != value for key, value in retained.items()):
            return False
        verify_opening_navigation(row, events[:index], opening)
        return True
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def _reported_action_input(event, *, detail=False):
    if (any(event.get(key) for key in ('submit_dispatched', 'click_dispatched', 'dispatched',
            'execution_selection_click_count', 'identity_probe_click_count', 'input_click_count',
            'total_input_click_count', 'execution_submit_count', 'end_turn_submit_count',
            'choice_submit_count', 'drinks_consumed')) or detail and event.get('input_may_have_been_dispatched')):
        return True
    return any(_reported_action_input(step, detail=True) for key in ('execution_selection_events',
        'identity_probe_events', 'authoritative_refresh_events', 'foreground_preparation_events',
        'shadow_transaction_events', 'failure_events', 'transactions', 'inputs', 'events')
        for step in event.get(key, ()) if isinstance(step, dict))


def saved_opening_checkpoint(runtime, session, module, source):
    from .native_opening_root import PARENT_FIELDS, load_native_opening_root, load_native_round_context
    from .native_search_cache import _cache_source, restore_search_cache
    from .native_handoff_resume import _binding
    from .native_settlement_resume import load_result
    from .native_opening_mapping_resume import verify_opening_navigation, verify_opening_metadata_source

    if not is_saved_opening_intent_stop(source):
        raise ValueError('saved opening has an incomplete or unaccounted dispatch ledger')
    handoff = dict(session.record['handoff'], route_run_id=session.record['run_id'])
    live = handoff['live_context']
    _binding(source, live, handoff)
    parent = source.get('native_rule_resume_source')
    if source['endpoints']['game']['pid'] != session.game_pid:
        raise ValueError('saved opening source or process changed')
    original = source
    if parent is not None:
        if (parent != source['handoff_resume'].get('source')
                or [k for k in PARENT_FIELDS if source.get(k)] != ['native_rule_resume_source']):
            raise ValueError('saved opening source or process changed')
        original = load_result(runtime, parent)
        _binding(original, live, handoff)
    owner, payload = _cache_source(runtime, source)
    plan = (payload or {}).get('pending_plan')
    if plan is None:
        raise ValueError('saved opening has no recorded pending plan')
    previous_opening = verify_opening_metadata_source(runtime, original)
    if parent is None and previous_opening is not None:
        retained = original['opening_navigation_resume'].get('retained_components', {})
        if any(previous_opening.get('components', {}).get(k) != value for k, value in retained.items()):
            raise ValueError('saved opening retained components differ from their original unbound source')
    bundle, opening = load_native_opening_root(original, module)
    # The original complete observation stays authoritative. The repeated core
    # establishes continuity, and the next host must obtain fresh frames again.
    events = original['live_events']
    verify_opening_navigation(original, events[:events.index(opening)], opening)
    if parent is None and source['total_input_count'] != source['input_count']:
        previous = load_result(runtime, source['automatic_battle']['previous_record'], allow_closed_transaction=True)
        if (not source.get('host_attempt') or source['host_attempt'] != previous.get('host_attempt')
                or source['handoff_id'] != previous.get('handoff_id')
                or source['total_input_count'] != previous['total_input_count'] + source['input_count']):
            raise ValueError('saved opening cumulative input count differs from its preceding host record')
    if parent is not None:
        repeated = deepcopy(source['live_events'][0])
        fresh = repeated.pop('unsubmitted_opening_resume', None)
        if (repeated != opening or not isinstance(fresh, dict)
                or fresh.get('source') != parent or fresh.get('input_click_count') != 0
                or fresh.get('original_observation_reused') is not True
                or any(fresh['fresh_core'][k] != opening['core'][k] for k in ('fields', 'families'))):
            raise ValueError('saved opening changed its independent business state')
    if bundle.state.turn != 1 or bundle.state.score != 0:
        raise ValueError('saved opening changed its independent business state')
    policy, report = restore_search_cache(runtime, owner, bundle, accepted=True,
        profile=handoff['policy_profile'], retrieval_ready=True,
        information_boundary_execution=True, unsubmitted_plan=plan)
    report.update(original_failed_window_modified=False,
        new_search_performed=False, input_authorized=False)
    return dict(bundle=bundle, state=bundle.state, event=opening, row=source,
        policy=policy, cache_report=report, live=live, source_runtime=runtime,
        round_context=load_native_round_context(runtime, source, module),
        untouched_opening=True, selected=False, pending=False, expected=None, action=None,
        receipt=dict(version=1, source=source['id'], handoff_id=handoff['handoff_id'],
            route_run_id=session.record['run_id'], ownership_revision=session.record['ownership']['revision'],
            ledger_revision=handoff['ledger_revision'], live_context=live, boundary_kind='unsubmitted_opening'))


def prepare_saved_opening(runtime, session, module, source, *, attempt,
                          maximum_actions, timeout_seconds, parent_deadline, authorise):
    from ..hif_live_recovery import require_budget, automatic_host_active, constrain_parent_deadline
    from ..hif_battle_engineering_window import append_repair, window_limits

    checkpoint = saved_opening_checkpoint(runtime, session, module, source)
    control = deepcopy(source['automatic_battle'])
    if (control['maximum_actions'], control['timeout_seconds']) != (maximum_actions, timeout_seconds):
        raise ValueError('saved opening cannot replace its cumulative budget')
    deadline, _ = window_limits(control)
    if not automatic_host_active() and time.time() >= min(deadline, control.get('active_parent_deadline', control.get('parent_deadline', deadline))):
        if not callable(authorise):
            raise ValueError('saved opening recovery needs the current engineering grant')
        grant = authorise(source=source, record=session.record)
        control = append_repair(control, source, grant, now=time.time(), parent_deadline=parent_deadline)
    constrain_parent_deadline(control, parent_deadline)
    require_budget(control)
    control['recoveries_used'] += 1
    return control, source, checkpoint
