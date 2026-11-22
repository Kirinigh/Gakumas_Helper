"""Replay separate native transactions from one still-bound Maa process."""
import json
from dataclasses import asdict, replace


def stopped_policy_decision(event):
    """Read the original decision from either existing observer envelope."""
    proposal = event['proposal']
    return proposal if event.get('event') == 'live_policy_stop' else proposal['candidate_traces']['high_score']


def is_settled_policy_continuation(row, parent):
    """A zero-input STOP can still have a completely settled parent.

    This identifies the existing committed checkpoint, not a retryable reason.
    The normal loader must replay its settlement and restore the complete cache.
    """
    from ..hif_live_recovery import closed_observer_record

    try:
        if (not isinstance(parent, dict) or row.get('status') != 'STOP'
                or row.get('live_return') is not False or not closed_observer_record(row)
                or row.get('transaction_closed') or row.get('error')
                or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
                or row.get('native_inprocess_source') != parent.get('id')
                or row['automatic_battle']['previous_record'] != parent['id']
                or any(row.get(k) for k in ('native_input_preparation', 'native_navigation_preparation',
                    'native_pending_source', 'native_selected_source', 'native_resume_source',
                    'choice_journal', 'gray_selection_journal', 'upstream_ui_journal',
                    'mcts_search_window', 'mcts_search_cache'))
                or len(row['live_events']) != 1):
            return False
        event = row['live_events'][0]
        direct = event.get('event') == 'live_policy_stop'
        precheck = (event.get('event') == 'live_decision_exception_stop'
            and event.get('runtime_phase') == 'runtime_prechecks'
            and event.get('error_type') == 'ValueError'
            and event.get('submit_dispatched') is False
            and not any(event.get(k) for k in ('proposal', 'selected_skill_support',
                'authoritative_refresh_events', 'single_plus_estimate')))
        if (not (direct or precheck or event.get('event') == 'live_decision_observe') or event.get('executed') is not False
                or not direct and event.get('would_submit') is not False
                or event.get('current_selected_slot') is not None
                or any(event.get(k, 0) != 0 for k in ('execution_click_count', 'execution_submit_count',
                    'execution_selection_click_count', 'identity_probe_click_count', 'end_turn_submit_count',
                    'drinks_consumed', 'total_input_click_count'))
                or any(event.get(k) for k in ('input_may_have_been_dispatched', 'execution_selection_events',
                    'foreground_preparation_events', 'identity_probe_events', 'shadow_transaction_events'))):
            return False
        cache = parent['mcts_search_cache']
        if (cache.get('phase') != 'observed_successor' or cache.get('record_id') != parent['id']
                or row['formal_params']['hif_policy_profile'] != parent['formal_params']['hif_policy_profile']
                or row['total_input_count'] != parent['total_input_count']
                or any(row.get(k) != parent.get(k) for k in ('host_attempt', 'endpoints', 'observed_live_context',
                    'handoff_id', 'handoff_opening', 'round_context'))):
            return False
        verify_closed_transaction_record(parent)
        if precheck:
            return parent['formal_params']['hif_policy_profile'] == 'hif-strong-reference-trial-v1'
        proposal = event['proposal']
        decision = stopped_policy_decision(event)
        trace = decision['trace']
        if (not direct and proposal.get('status') != 'STOP' or proposal.get('action') is not None
                or decision.get('protocol_version') != 'task275-hif-mcts-native-intent-v1'
                or decision.get('outcome') != 'STOP_TASK' or decision.get('has_action_intent') is not False
                or decision.get('action') is not None or decision.get('execution_authorized') is not False
                or trace.get('expanded_nodes') != 0 or trace.get('completed_depth') != 0
                or type(trace.get('planning_window_index')) is not int or trace['planning_window_index'] < 1
                or (trace.get('run_id'), trace.get('revision')) != (cache.get('run_id'), cache.get('revision'))
                or trace.get('policy_profile') != parent['formal_params']['hif_policy_profile']):
            return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def verify_closed_transaction_record(row):
    """Admit a finished action for full verification without inventing an exit.

    This only validates the receipt structure. The caller must still decode
    the dispatched action and verify every recorded settlement component.
    """
    from .native_opening_root import has_native_opening_root, load_native_opening_root
    from .native_choice_resume import has_confirmed_choice
    from .native_settlement_resume import _verified_endpoints
    from .native_rule_progress_resume import is_rule_progress

    root = has_native_opening_root(row)
    journal = row.get('choice_journal') or {}
    settled_choice = row.get('native_choice_source') and journal.get('stage') == 'settled' and has_confirmed_choice(journal)
    receipt = row.get('handoff_resume') or {}
    observed_choice = settled_choice and receipt.get('boundary_kind') == 'confirmed_retrieval' and receipt.get('source') == row['native_choice_source']
    if (row.get('transaction_closed') is not True or row.get('live_return') is not True
            or not row.get('host_attempt')
            or not (row.get('native_inprocess_source') or row.get('native_resume_source')
                    or row.get('native_selected_source') or root or settled_choice
                    or (row.get('information_boundary_execution') is True
                        and (row.get('native_rule_resume_source') or row.get('native_pending_source'))))):
        raise ValueError('native transaction has not closed successfully')
    if root:
        load_native_opening_root(row)
    run = row.get('observed_live_context', {})
    game_pid = row.get('endpoints', {}).get('game', {}).get('pid')
    dispatches = row.get('native_input_dispatches', ())
    pending_rule = row.get('information_boundary_execution') is True and row.get('native_pending_source')
    if not dispatches and not (pending_rule or observed_choice):
        raise ValueError('closed native transaction has no audited input')
    for dispatch in dispatches:
        _verified_endpoints(dispatch.get('endpoints'), game_pid)
        _same_input_host(row['endpoints'], dispatch['endpoints'])
    submissions = [event for event in row.get('live_events', ())
                   if event.get('execution_submit_count', 0)
                   or event.get('end_turn_submit_count', 0)]
    if settled_choice:
        confirmations = sum(event.get('choice_submit_count', 0) for event in row.get('live_events', ()))
        if submissions or confirmations != (0 if observed_choice else 1):
            raise ValueError('closed retrieval changed the number of actual submissions')
    elif pending_rule:
        if submissions:
            raise ValueError('pending native rule completion contains a new submission')
    elif len(submissions) != 1:
        raise ValueError('closed native transaction must contain one submission')
    submitted = submissions[0] if submissions else {}
    if submissions and any(type(submitted.get(key, 0)) is not int or submitted.get(key, 0) not in (0, 1)
           for key in ('execution_submit_count', 'end_turn_submit_count')):
        raise ValueError('closed native transaction has invalid submission counts')
    if submissions and submitted.get('executed' if submitted.get('event') == 'live_decision_observe'
                     else 'accepted') is not True:
        raise ValueError('closed native transaction did not complete its submission')
    settlements = [event for event in row.get('live_events', ())
                   if event.get('event') == 'live_committed_state' and event.get('accepted') is True]
    if len(settlements) != 1:
        raise ValueError('closed native transaction must contain one complete settlement')
    settlement = settlements[0]
    if (not settlement.get('checks') or not all(v is True for v in settlement['checks'].values())
            or settlement.get('run_id') != run.get('run_id')
            or settlement.get('revision') != run.get('revision')
            or not isinstance(settlement.get('expected_state'), dict)):
        raise ValueError('closed native transaction settlement is incomplete')
    if is_rule_progress(settlement):
        if (row.get('information_boundary_execution') is not True
                or settlement.get('strategy_state_bound') is not True
                or not isinstance(settlement.get('rule_progress'), dict)
                or settlement.get('components')):
            raise ValueError('closed native rule progress receipt is incomplete')
    elif not isinstance(settlement.get('components'), dict) or not settlement['components']:
        raise ValueError('closed native transaction complete components are missing')
    elif pending_rule and settlement.get('pending_outcome') != 'after':
        raise ValueError('closed pending transaction lacks a uniquely observed successor')


def _same_input_host(host, actual):
    for name in ('game', 'agent'):
        expected = host.get(name, {})
        if (actual.get(name, {}).get('pid') != expected.get('pid')
                or expected.get('integrity_rid') != 12288):
            raise ValueError('native transaction input host changed')
    # endpoint_evidence historically calls os.getppid() "maapicli". In the
    # pinned launcher that PID is the Python input child, not MaaPiCli itself.
    # Match it to the independently enumerated input chain, never rename it.
    parent = host.get('maapicli', host.get('maa', {}))
    if (parent.get('integrity_rid') != 12288
            or not any(entry.get('pid') == parent.get('pid')
                       for name, entry in actual.items()
                       if name == 'maa' or name.startswith('input_child_'))):
        raise ValueError('native transaction input ancestry changed')


def restore_inprocess_source(runtime, row, module, trail=()):
    """Promote the parent's verified settlement exactly as the live observer does."""
    from .native_opening_root import verify_native_round_context
    from .native_settlement_resume import load_verified_source
    from .native_rule_progress_resume import is_rule_progress

    source = row.get('native_inprocess_source')
    if not source or row.get('native_resume_source'):
        raise ValueError('native in-process source is ambiguous')
    bundle, state, baseline, parent = load_verified_source(
        runtime, source, module, (*trail, row['id']))
    verify_native_round_context(row, module, parent)
    if (not row.get('host_attempt')
            or row['host_attempt'] != parent.get('host_attempt', parent['id'])
            or row.get('observed_live_context') != parent.get('observed_live_context')
            or row.get('endpoints') != parent.get('endpoints')
            or (bundle.run_id, bundle.revision) != (
                row['observed_live_context']['run_id'], row['observed_live_context']['revision'])
            or state.terminal):
        raise ValueError('native in-process source belongs to another live host or epoch')
    parent_dispatches = parent.get('native_input_dispatches', ())
    observed_choice = (parent.get('handoff_resume') or {}).get('boundary_kind') == 'confirmed_retrieval' and parent.get('native_choice_source')
    if not parent_dispatches and not (observed_choice or parent.get('native_pending_source') and is_rule_progress(baseline)):
        raise ValueError('native in-process parent has no actual input host evidence')
    for dispatch in row.get('native_input_dispatches', ()):
        _same_input_host(row['endpoints'], dispatch.get('endpoints', {}))
        if parent_dispatches and dispatch.get('endpoints') != parent_dispatches[-1].get('endpoints'):
            raise ValueError('native in-process actual input processes changed')
    if json.dumps(baseline.get('expected_state'), sort_keys=True) != json.dumps(asdict(state), sort_keys=True):
        raise ValueError('native in-process parent state differs from its complete receipt')
    if is_rule_progress(baseline):
        if row.get('information_boundary_execution') is not True:
            raise ValueError('native in-process rule continuation was not enabled')
    return promote_verified_settlement(bundle, state, baseline), baseline


def promote_verified_settlement(bundle, state, baseline):
    """Reuse the same state/physical-slot promotion after full source replay."""
    from live_decision.high_score.opening import observed_hif_drink_slots, expected_permanent_inventory_keys

    from .native_rule_progress_resume import is_rule_progress, promote_rule_bundle

    if (baseline.get('verification_scope') == 'rule_maintained_preinput_v1'
            and baseline.get('pending_outcome') == 'before'
            and baseline.get('observation_mode') == 'pending_core_comparison'):
        from .native_pending_outcome import _same_value

        if state != bundle.state or not _same_value(baseline.get('expected_state'), asdict(state)):
            raise ValueError('unconsumed core comparison changed its original state')
        return bundle  # The independently replayed proof retained this exact state and definitions.
    if json.dumps(baseline.get('expected_state'), sort_keys=True) != json.dumps(asdict(state), sort_keys=True):
        # The full observer has already ordered its running hand by the actual
        # slots; its immutable receipt retains the rule outcome's hand order.
        # Admit exactly that recorded permutation, preserving every other fact.
        observed = baseline.get('observed_hand_instance_ids', ())
        by_id = {card.instance_id: card for card in state.deck.hand}
        original = baseline.get('expected_state', {}).get('deck', {}).get('hand', ())
        ids = [card.get('instance_id') for card in original]
        if (list(by_id) != list(observed) or len(ids) != len(by_id) or len(set(ids)) != len(ids)
                or set(ids) != set(by_id)):
            raise ValueError('native promotion state differs from its complete receipt')
        unpromoted = replace(state, deck=state.deck.replace_zones(hand=tuple(by_id[key] for key in ids)))
        if json.dumps(baseline.get('expected_state'), sort_keys=True) != json.dumps(asdict(unpromoted), sort_keys=True):
            raise ValueError('native promotion state differs from its complete receipt')
    if is_rule_progress(baseline):
        return promote_rule_bundle(bundle, state, baseline)
    # load_verified_source has already rechecked the parent's real components.
    # Its returned bundle may still be pre-action, exactly like the production
    # context before the next ProduceLiveDecisionObserve call.
    by_id = {card.instance_id: card for card in state.deck.hand}
    observed_hand = baseline.get('observed_hand_instance_ids', ())
    if len(observed_hand) != len(by_id) or set(observed_hand) != set(by_id):
        raise ValueError('native in-process parent hand order is incomplete')
    state = replace(state, deck=state.deck.replace_zones(hand=tuple(
        by_id[key] for key in observed_hand)))
    from live_decision.high_score.opening import verified_expired_start_drinks, verified_expired_start_scores
    bundle = replace(bundle, state=state,
        expired_start_scores=verified_expired_start_scores(bundle, state),
        expired_start_drinks=verified_expired_start_drinks(bundle, state),
        inventory_keys=expected_permanent_inventory_keys(bundle, state),
        hand_instance_ids=tuple(card.instance_id for card in state.deck.hand),
        drink_slots=(observed_hif_drink_slots(baseline['components']['drinks'])
                     if 'slots' in baseline['components']['drinks'] else bundle.drink_slots))
    return bundle
