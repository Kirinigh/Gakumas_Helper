"""Explicit research restart of a lost first window; never automatic fallback."""
import time
from copy import deepcopy


def prepare_opening_recompute(runtime, session, module, *, specification, attempt,
                              maximum_actions, timeout_seconds, parent_deadline, authorise=None):
    from ..hif_live_recovery import (
        identity,
        require_budget,
        load_automatic_chain,
        automatic_host_active,
        closed_observer_record,
        constrain_parent_deadline,
    )
    from .native_opening_root import PARENT_FIELDS, load_native_opening_root, load_native_round_context
    from .native_search_window import PROFILE, _name, _path, _read
    from .native_handoff_resume import _binding
    from ..high_score.hif_mcts_native import HifMctsNativePolicy, profile_configuration
    from .native_opening_mapping_resume import verify_opening_navigation, verify_opening_metadata_source
    from ..hif_battle_engineering_window import window_limits

    if (not isinstance(specification, dict) or set(specification) != {'source', 'reason'}
            or not isinstance(specification['reason'], str) or not specification['reason'].strip()
            or session.authorise_input() is not True):
        raise ValueError('opening recompute requires an explicit current research request')
    _name(attempt)
    source = load_automatic_chain(runtime, identity(session))[-1]
    if specification['source'] != source['id'] or source['id'] == attempt:
        raise ValueError('opening recovery must use the latest closed attempt')
    if source.get('mcts_search_cache'):
        from .native_opening_intent_resume import prepare_saved_opening

        return prepare_saved_opening(runtime, session, module, source, attempt=attempt,
            maximum_actions=maximum_actions, timeout_seconds=timeout_seconds,
            parent_deadline=parent_deadline, authorise=authorise)
    record = session.record
    handoff = dict(record['handoff'], route_run_id=record['run_id'])
    live = handoff['live_context']
    if (specification['source'] != source['id'] or source['id'] == attempt
            or not closed_observer_record(source) or source.get('status') != 'STOP'
            or source.get('live_return') is not False
            or source['formal_params']['hif_policy_profile'] != PROFILE
            or any(source.get(k) for k in (*PARENT_FIELDS, 'transaction_closed', 'mcts_search_cache',
                'native_input_preparation', 'upstream_ui_journal', 'choice_journal', 'gray_selection_journal'))):
        raise ValueError('opening recompute is not the latest untouched first-window failure')
    _binding(source, live, handoff)
    if source['endpoints']['game']['pid'] != session.game_pid:
        raise ValueError('opening recompute belongs to another game process')
    events = source['live_events']
    opening, stopped = events[-2:]
    if (opening.get('event') != 'live_opening_state' or opening.get('accepted') is not True
            or stopped.get('event') not in ('live_decision_exception_stop', 'live_policy_stop')
            or stopped.get('executed') is not False or stopped.get('input_may_have_been_dispatched') is not False
            or any(stopped.get(k, 0) != 0 for k in ('execution_selection_click_count', 'identity_probe_click_count'))
            or any(e.get('executed') is not False or e.get('execution_submit_count') != 0
                or e.get('end_turn_submit_count', 0) or e.get('choice_submit_count', 0) for e in events)
            or any(opening['initial_core'][key] != opening['core'][key] for key in ('fields', 'families'))):
        raise ValueError('opening recompute has possible selection or business input')
    verify_opening_metadata_source(runtime, source)
    verify_opening_navigation(source, events[:-2], opening)
    bundle, opening = load_native_opening_root(source, module)
    if (bundle.state.turn != 1 or bundle.state.score != 0 or bundle.state.deck.discard
            or bundle.state.deck.removed or bundle.state.pending_choice):
        raise ValueError('opening recompute cannot discard a played strategy')
    window = _read(runtime, source['id'], 0)
    if (window != source.get('mcts_search_window') or window['status'] != 'started'
            or window['attempt'] != source['id'] or _path(runtime, source['id'], 1).exists()
            or any(runtime.glob(f"search-cache-{source['id']}-*.json"))
            or any(runtime.glob(f'search-window-{attempt}-*.json'))):
        raise ValueError('opening recompute must retain the original lost first window')
    control = deepcopy(source['automatic_battle'])
    if (control['maximum_actions'], control['timeout_seconds']) != (maximum_actions, timeout_seconds):
        raise ValueError('opening recompute cannot replace the battle budget')
    constrain_parent_deadline(control, parent_deadline)
    require_budget(control)
    control['recoveries_used'] += 1
    _, config = profile_configuration(PROFILE)
    deadline = None if automatic_host_active() else time.perf_counter() + min(parent_deadline, window_limits(control)[0]) - time.time()
    policy = HifMctsNativePolicy(bundle.kernel, run_id=bundle.run_id, revision=bundle.revision,
        planning_seed=config['planning_seed_base'], deadline=deadline,
        retrieval_ready=True, information_boundary_execution=True, profile=PROFILE)
    # No prior tree exists. New-window accounting remains independent and the
    # original durable started record stays unchanged and linked in the report.
    proof = dict(specification, attempt=attempt, new_development_attempt=True, restored=False,
        lost_computation_restored=False, original_failed_window_modified=False, failed_window=window,
        reported_completed_simulations=stopped.get('proposal', {}).get('trace', {}).get('planning', {}).get('completed_simulations'),
        input_authorized=False)
    checkpoint = dict(bundle=bundle, state=bundle.state, event=opening, row=source,
        policy=policy, cache_report=dict(profile=PROFILE, opening_recompute=proof, restored=False),
        round_context=load_native_round_context(runtime, source, module), live=live, source_runtime=runtime,
        untouched_opening=True, selected=False, pending=False, expected=None, action=None,
        receipt=dict(version=1, source=source['id'], handoff_id=handoff['handoff_id'],
            route_run_id=record['run_id'], ownership_revision=record['ownership']['revision'],
            ledger_revision=handoff['ledger_revision'], live_context=live, boundary_kind='unsubmitted_opening'))
    return control, source, checkpoint
