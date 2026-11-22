"""Restore a verified Live boundary within the existing durable route handoff.

An audited submitted action is only observed to completion, never dispatched
again. Original replay/cache readers validate it before any navigation.
"""
import json
from dataclasses import asdict

from route.battle_policy import NETWORK_PROFILES


def _binding(row, live, handoff):
    opening = row.get('handoff_opening', {})
    run = {key: value for key, value in live.items() if key != 'round_number'}
    if handoff.get('source') == 'current_battle_page':
        inventory_verified = opening.get('opening_inventory_source') == 'current_complete_opening'
    else:
        inventory_verified = opening.get('permanent_inventory_verified') is True
    if (row.get('handoff_id') != handoff['handoff_id']
            or row.get('observed_live_context') != run
            or row.get('formal_params', {}).get('hif_policy_profile') != handoff['policy_profile']
            or opening.get('handoff_id') != handoff['handoff_id']
            or opening.get('route_run_id') != handoff['route_run_id']
            or opening.get('ledger_revision') != handoff['ledger_revision']
            or (opening.get('live_run_id'), opening.get('live_revision'), opening.get('round_number'))
            != (live['run_id'], live['revision'], live['round_number'])
            or opening.get('policy_profile') != handoff['policy_profile']
            or not inventory_verified
            or opening.get('drink_slots_verified') is not True):
        raise ValueError('HIF saved boundary does not belong to the durable battle handoff')


def load_handoff_checkpoint(runtime, name, session, module):
    from .native_opening_root import has_native_opening_root, load_native_opening_root, load_native_round_context
    from .native_handoff_policy import phase, restore_policy
    from .native_inprocess_resume import promote_verified_settlement, verify_closed_transaction_record
    from .native_settlement_resume import load_result, load_verified_source, reports_pending_submission
    from .native_submitted_navigation import (
        is_pending_observation_stop,
        completed_pending_observation,
        is_pending_inventory_read_stop,
    )
    from .native_observed_identity_resume import (
        post_commit_source,
        verify_post_commit_source,
        is_post_commit_identity_stop,
        unsubmitted_drink_inspection,
        restore_unsubmitted_drink_policy,
        restored_post_commit_observation,
        unsubmitted_post_commit_proposal,
        upstream_resource_intent_proposal,
    )

    # Do not infer a closed boundary merely from a successful process exit.
    if session.authorise_input() is not True:
        raise ValueError('HIF saved boundary requires the active shared owner')
    record = session.record
    handoff = dict(record['handoff'], route_run_id=record['run_id'])
    live = handoff.get('live_context')
    if not isinstance(live, dict):
        raise ValueError('HIF saved boundary has no durable Live epoch')
    row = load_result(runtime, name, allow_closed_transaction=True)
    if row.get('pending_business_frontier') is not None:
        from .native_pending_frontier import completed_handoff_checkpoint
        return completed_handoff_checkpoint(runtime, name, session, module)
    _binding(row, live, handoff)
    if row.get('hold_journal') and row.get('transaction_closed') is not True:
        raise ValueError('HIF hold has a submitted origin and no closed result; do not replay')
    if row['endpoints']['game']['pid'] != session.game_pid:
        raise ValueError('HIF saved boundary belongs to another game process')
    from .native_opening_intent_resume import saved_opening_checkpoint, is_original_saved_opening_intent

    if is_original_saved_opening_intent(row):
        from ..hif_live_recovery import identity, load_automatic_chain

        if load_automatic_chain(runtime, identity(session))[-1]['id'] != name:
            raise ValueError('HIF saved opening intent is not the latest battle record')
        return saved_opening_checkpoint(runtime, session, module, row)
    from .native_upstream_selected import (
        original_selection,
        load_selected_checkpoint,
        selected_read_without_input,
        is_unsubmitted_upstream_selection,
    )

    if selected_read_without_input(row):
        row = original_selection(runtime, row)
    if is_unsubmitted_upstream_selection(row):
        return load_selected_checkpoint(runtime, row, session, module, live, handoff)
    from .native_choice_resume import load_choice_source, is_pending_retrieval_stop, is_confirmed_retrieval_stop, load_confirmed_choice_source

    confirmed = is_confirmed_retrieval_stop(row)
    if confirmed or is_pending_retrieval_stop(row, runtime=runtime):
        if confirmed:
            projected, expected, baseline, source = load_confirmed_choice_source(runtime, name, module)
        else:
            projected, baseline, source = load_choice_source(runtime, name, module)
            expected = None
        if handoff['policy_profile'] in NETWORK_PROFILES:
            from .native_fast_cache import restore_fast_retrieval

            policy, report = restore_fast_retrieval(runtime, source, projected)
        else:
            from .native_search_cache import restore_retrieval_cache

            policy, report = restore_retrieval_cache(runtime, source, projected,
                profile=handoff['policy_profile'], information_boundary_execution=True)
        return dict(bundle=projected.source_bundle, state=projected.source_bundle.state,
            event=baseline, row=source, policy=policy, cache_report=report, retrieval=projected,
            round_context=load_native_round_context(runtime, source, module), live=live, source_runtime=runtime,
            selected=False, pending=False, expected=expected, action=projected.action,
            receipt=dict(version=1, source=source['id'], handoff_id=handoff['handoff_id'],
                route_run_id=record['run_id'], ownership_revision=record['ownership']['revision'],
                ledger_revision=handoff['ledger_revision'], live_context=live,
                boundary_kind='confirmed_retrieval' if confirmed else 'pending_retrieval'))
    continued_pending = is_pending_inventory_read_stop(row) or is_pending_observation_stop(row)
    pending = continued_pending or reports_pending_submission(row)
    selected = (not pending and row.get('transaction_closed') is not True
        and (bool(row.get('gray_selection_journal')) or bool(row.get('native_selected_source'))
             or is_cleared_selection_timing_stop(row)
             or any(event.get('execution_selection_click_count') == 1
                    for event in row.get('live_events', ()))))
    untouched = not (pending or selected) and has_native_opening_root(row) and row.get('transaction_closed') is not True
    post_commit_identity = is_post_commit_identity_stop(row)
    drink_proposal = unsubmitted_drink_inspection(row) if post_commit_identity else None
    preinput_proposal = unsubmitted_post_commit_proposal(row) if post_commit_identity else None
    if post_commit_identity and preinput_proposal is None:
        preinput_proposal = upstream_resource_intent_proposal(runtime, row)
    expected, action, dispatch_source = None, None, None
    reconciled_unconsumed = False
    completed_observation = continued_pending and completed_pending_observation(row)
    if completed_observation:
        from .native_settlement_resume import decoded_dispatched_action
        from .native_submitted_navigation import verify_pending_inventory_parent

        # The previous host already published every complete component. Its
        # later bookkeeping failure does not require another information tour.
        bundle, state, baseline, source = load_verified_source(runtime, name, module)
        dispatch_source = load_result(runtime, row['native_pending_source'], allow_closed_transaction=True)
        verify_pending_inventory_parent(row, dispatch_source, module)
        action, _ = decoded_dispatched_action(dispatch_source, bundle)
        rebound = promote_verified_settlement(bundle, state, baseline)
        state, expected, pending = rebound.state, state, False
    elif pending:
        from .native_settlement_resume import load_pending_source, decoded_dispatched_action

        bundle, expected, baseline, source = load_pending_source(runtime, name, module)
        dispatch_source = source
        action, _ = decoded_dispatched_action(source, bundle)
        if continued_pending:
            # Keep the latest navigation boundary and its own complete cache.
            # Only submission decoding follows the already dispatched parent.
            source = dict(row, **{k: source[k] for k in ('verified_single_plus_estimate',
                'verified_pending_support_receipt') if k in source})
        state, rebound = bundle.state, bundle
        from .native_pending_outcome import retry_index, verify_unconsumed, is_unconsumed_observation

        if is_unconsumed_observation(row):
            if retry_index(dispatch_source):
                raise ValueError('HIF verified unconsumed retry limit reached')
            state = verify_unconsumed(bundle, expected, baseline, action, row['settlement'],
                source.get('verified_single_plus_estimate'))
            rebound = promote_verified_settlement(bundle, state, row['settlement'])
            state, baseline = rebound.state, row['settlement']
            pending, reconciled_unconsumed = False, True
    elif selected:
        from .native_selected_resume import load_selected_source

        bundle, state, baseline, source = load_selected_source(runtime, name, module)
        rebound = bundle
    elif untouched:
        from .native_opening_resume import verify_unsubmitted_opening_source
        from .native_opening_mapping_resume import verify_opening_metadata_source

        verify_opening_metadata_source(runtime, row)
        bundle, baseline = load_native_opening_root(row, module)
        verify_unsubmitted_opening_source(row, bundle)
        # Continuous-host records have this existing filename form. Earlier
        # incomplete openings have no bound epoch; a later bound attempt must
        # be recovered from its own receipt, never hidden by this first one.
        for path in runtime.glob('result-*-live*.json'):
            if path.name == f"result-{row['id']}.json":
                continue
            other = json.loads(path.read_text(encoding='utf-8'))
            if (other.get('handoff_id') == handoff['handoff_id']
                    and other.get('handoff_opening') is not None):
                raise ValueError('HIF untouched opening has a later bound attempt')
        state, source, rebound = bundle.state, row, bundle
    elif post_commit_identity:
        bundle, state, baseline, source = load_verified_source(runtime, post_commit_source(row), module)
        verify_post_commit_source(row, source, module)
        rebound = promote_verified_settlement(bundle, state, baseline)
    else:
        verify_closed_transaction_record(row)
        bundle, state, baseline, source = load_verified_source(runtime, name, module)
        rebound = promote_verified_settlement(bundle, state, baseline)
    if ((bundle.run_id, bundle.revision) != (live['run_id'], live['revision'])
            or state.max_stamina != live['max_stamina']
            or state.terminal or state.pending_choice or state.resolution_stack):
        raise ValueError('HIF recovery requires a complete nonterminal settlement')
    round_context = load_native_round_context(runtime, source, module)
    if round_context is None or round_context.round_number != live['round_number']:
        raise ValueError('HIF saved boundary has no matching round proof')
    if drink_proposal is not None:
        policy, report = restore_unsubmitted_drink_policy(runtime, row, rebound, drink_proposal)
    else:
        policy, report = restore_policy(runtime, row if preinput_proposal is not None or restored_post_commit_observation(row) else source, rebound, before=bundle.state, action=action,
            submitted=not (untouched or selected or pending or reconciled_unconsumed), round_context=round_context,
            selected=selected, pending=pending, baseline=baseline, source_drink_slots=bundle.drink_slots,
            unconsumed_source=dispatch_source if reconciled_unconsumed else None,
            **(dict(verified_successors=expected) if completed_observation else {}),
            **(dict(unsubmitted_plan=preinput_proposal['candidate_traces']['high_score']['trace']['planning'])
                if preinput_proposal is not None else {}))
    deferred = pending and report.get('deferred_until_observed_settlement') is True and policy is None
    has_intent = selected or pending or reconciled_unconsumed or preinput_proposal is not None or drink_proposal is not None
    if not deferred and (phase(policy) != ('pending_intent' if has_intent else 'observed_successor')
            or (policy._pending is None) != (not has_intent)):
        raise ValueError('HIF saved computation is not a completely committed successor')
    if preinput_proposal is not None:
        from .native_handoff_policy import pending_action

        restored_action = pending_action(policy, rebound.state)
        original_action = preinput_proposal['candidate_traces']['high_score']['action']
        if any(getattr(restored_action, key) != original_action.get(key)
                for key in ('card_instance_id','drink_id','target_id')):
            raise ValueError('HIF preinput recovery changed the computed action')
    from ..hif_opening_art import restore_reference
    opening_art = restore_reference(row, bundle, baseline) if untouched else None
    return dict(bundle=rebound, state=state, event=baseline, row=row if post_commit_identity else source, policy=policy, observer_module=module,
        **(dict(settled_source=source) if post_commit_identity else {}),
        opening_art_reference=opening_art,
        cache_report=report, round_context=round_context, live=live, source_runtime=runtime,
        untouched_opening=untouched,
        selected=selected, pending=pending, expected=expected, action=action,
        completed_observation=completed_observation,
        reconciled_unconsumed=reconciled_unconsumed,
        **(dict(dispatch_source=dispatch_source) if pending or reconciled_unconsumed else {}),
        **(dict(unconsumed_retry=dict(version=1, retry_index=1, observation_source=name,
            dispatch_source=dispatch_source['id'])) if reconciled_unconsumed else {}),
        receipt=dict(version=1, source=source['id'] if post_commit_identity else name, handoff_id=handoff['handoff_id'],
            route_run_id=record['run_id'], ownership_revision=record['ownership']['revision'],
            ledger_revision=handoff['ledger_revision'], live_context=live,
            boundary_kind='not_consumed' if reconciled_unconsumed else 'submitted_pending' if pending else (
                'selected' if selected else ('unsubmitted_opening' if untouched else 'settled'))))


def bind_handoff_checkpoint(context, checkpoint, result, module):
    from .native_opening_root import verify_native_round_context
    from ..hif_hand_inspection import restore_hand_inspection, restore_pending_drink_opening_inspection

    source, live = checkpoint.get('settled_source', checkpoint['row']), checkpoint['live']
    fresh = context.gkh_hif_live_context
    if (any(getattr(fresh, key) != live[key] for key in ('idol', 'plan', 'max_stamina'))
            or context.gkh_hif_context_game_pid != source['endpoints']['game']['pid']):
        raise ValueError('HIF current identity/process differs from the saved boundary')
    result['observed_live_context'] = asdict(fresh)
    # Original notebook observations can predate the logical execution epoch.
    # Establish the existing verified parent link before checking that proof;
    # neither observation epoch nor frame IDs are relabelled.
    key = ('native_pending_source' if checkpoint.get('pending') else
        'native_selected_source' if checkpoint.get('selected') else 'native_rule_resume_source')
    result.update({key: source['id']}, execution_live_context=source['observed_live_context'])
    verify_native_round_context(result, module, source, required=True)
    run = {key: value for key, value in live.items() if key != 'round_number'}
    context.gkh_hif_live_context = module.HifLiveRunContext(**run)
    context.gkh_hif_round_context = checkpoint['round_context']
    context.gkh_hif_opening_state = checkpoint['bundle']
    if checkpoint.get('untouched_opening'):
        context.gkh_hif_opening_art_reference = checkpoint.get('opening_art_reference')
        context.gkh_hif_unsubmitted_opening_event = checkpoint['event']
        context.gkh_hif_unsubmitted_opening_source = source['id']
        context.gkh_hif_extended_opening_identity = checkpoint['event'].get('extended_hand_identity')
        restore_hand_inspection(context, checkpoint['event'], module, source=source)
    elif checkpoint.get('retrieval') is None:
        context.gkh_hif_committed_state = checkpoint['state']
        context.gkh_hif_committed_event = checkpoint['event']
        inspection_event = source.get('settlement', {}) if checkpoint.get('pending') else checkpoint['event']
        restore_hand_inspection(context, inspection_event, module, source=source, pending=checkpoint.get('pending', False))
        restore_pending_drink_opening_inspection(context, checkpoint, module)
    context.gkh_hif_mcts_policy = checkpoint['policy']
    context.gkh_hif_handoff_opening = dict(source['handoff_opening'])
    result.update(native_rule_resume_source=source['id'], handoff_resume=checkpoint['receipt'],
        observed_live_context=run, execution_live_context=run,
        mcts_search_cache_restore=checkpoint['cache_report'])
    if checkpoint.get('retrieval') is not None:
        from copy import deepcopy

        result.pop('native_rule_resume_source')
        result.update(native_choice_source=source['id'], choice_journal=deepcopy(source['choice_journal']))
    if checkpoint.get('reconciled_unconsumed'):
        result['native_unconsumed_retry'] = checkpoint['unconsumed_retry']
        context.gkh_hif_gray_selected_source = None
    if checkpoint.get('selected'):
        context.gkh_hif_gray_selected_source = source['gray_selection_journal']
        from .native_selected_resume import restore_strong_selection_art

        if checkpoint.get('source_runtime') is not None:
            restore_strong_selection_art(context, source, checkpoint['source_runtime'], module)
        context.gkh_hif_mcts_selected_cache_source = source
        result.pop('native_rule_resume_source')
        result.update(native_selected_source=source['id'], selected_recovery_baseline=checkpoint['event'])
    elif checkpoint.get('pending'):
        result.pop('native_rule_resume_source')
        result['native_pending_source'] = source['id']
        if source.get('verified_pending_upstream_effects') is not None:
            result['native_pending_upstream_effects'] = source['verified_pending_upstream_effects']


def complete_handoff_pending(context, checkpoint, result, module, runtime, params):
    """Reuse the existing pending-settlement reader and fast history replay."""
    from .native_handoff_policy import phase, restore_policy
    from .native_inprocess_resume import promote_verified_settlement

    if checkpoint.get('pending') is not True:
        raise ValueError('HIF completion requires a verified submitted checkpoint')
    bundle, source = checkpoint['bundle'], checkpoint['row']
    from ..high_score.contracts import ActionKind
    from .native_settlement_resume import is_submitted_completion_stop, require_pending_completion_checkpoint

    completion_pending = (is_submitted_completion_stop(source)
        and checkpoint['action'].kind is not ActionKind.END_TURN)
    previous_fresh_core = getattr(context, 'gkh_hif_require_fresh_rule_core', False)
    if completion_pending:
        require_pending_completion_checkpoint(checkpoint)
        # A cold recovery must see public state, rather than reuse the failed
        # removal-only shortcut or turn its prediction into an observation.
        context.gkh_hif_require_fresh_rule_core = True
    try:
        inspection = getattr(context, 'gkh_hif_pending_hand_inspection', None)
        if inspection is not None:
            from ..hif_hand_inspection import finish_upstream_inspection

            reset = dict(source=source['id'], business_submit_count=0, events=[])
            result['native_pending_selection_reset'] = reset
            try:
                finish_upstream_inspection(context, inspection, module)
            finally:
                reset['events'] = inspection.get('pending_inspection_reset', [])
            context.gkh_hif_current_hand_inspection = inspection
            context.gkh_hif_pending_hand_inspection = None
        event = module._observe_hif_opening_state(context, params,
            committed_expected=checkpoint['expected'], baseline_event=checkpoint['event'],
            committed_action=checkpoint['action'],
            single_plus_estimate=source.get('verified_single_plus_estimate'),
            **(dict(pending_observation=True, force_fresh_components=True) if completion_pending else {}))
    finally:
        if completion_pending:
            context.gkh_hif_require_fresh_rule_core = previous_fresh_core
    # The native observer publishes through the host logger before returning.
    # Offline callers may not install that sink, so only append if absent.
    published = json.loads(json.dumps(event, default=module._live_event_json_default))
    if published not in result['live_events']:
        result['live_events'].append(published)
    result['settlement'] = event
    if event.get('accepted') is not True:
        return False
    if event.get('pending_outcome') == 'before':
        # A complete unchanged state is not a completed card. Keep the original
        # attempt and cached intent; the caller must not advance its history.
        result['pending_dispatch_outcome'] = 'not_consumed'
        return False
    terminal = getattr(context, 'gkh_hif_terminal_event', None)
    state = context.gkh_hif_committed_state if terminal is None else checkpoint['expected']
    if terminal is not None:
        return True
    rebound = promote_verified_settlement(bundle, state, event)
    policy, report = restore_policy(runtime, source, rebound, before=bundle.state,
        action=checkpoint['action'], submitted=True, round_context=checkpoint['round_context'],
        source_drink_slots=bundle.drink_slots, verified_successors=checkpoint['expected'])
    if phase(policy) != 'observed_successor' or policy._pending is not None:
        raise ValueError('HIF pending completion did not advance the saved policy')
    context.gkh_hif_opening_state = rebound
    context.gkh_hif_mcts_policy = policy
    result['mcts_search_cache_restore'] = report
    context.gkh_hif_mcts_cache_sink(policy)
    return True


def is_cleared_selection_timing_stop(row):
    """A reconciled intent stopped before its new SELECT, with no dispatch."""
    try:
        from ..hif_live_recovery import closed_observer_record

        receipt = row['native_selection_rebind']
        report, = row['live_events']
        failures = report.get('failure_events', ())
        proposals = [p['proposal'] for p in report.get('identity_probe_events', ())
                     if p.get('phase') == 'hif_candidate_precompute' and p.get('click_dispatched') is False]
        common = (closed_observer_record(row) and row.get('status') == 'STOP'
            and row.get('live_return') is False and not row.get('transaction_closed')
            and not row.get('error') and row.get('input_count') == 0
            and row.get('native_input_dispatches') == []
            and not any(row.get(k) for k in ('native_navigation_preparation',
                'native_selected_source', 'native_pending_source', 'choice_journal', 'gray_selection_journal'))
            and receipt.get('accepted') is True and receipt.get('input_click_count') == 0
            and receipt.get('source') == row.get('native_rule_resume_source')
            and row.get('mcts_search_cache', {}).get('phase') == 'pending_intent')
        if not common:
            return False
        preparation = row.get('native_input_preparation')
        if preparation:
            from .native_card_evidence import _fresh

            refresh, = report['authoritative_refresh_events']
            stage, = report['execution_selection_events']
            action = receipt['action']
            _fresh(refresh, action['source_slot'])
            return (report.get('event') == 'live_decision_observe'
                and report.get('executed') is False and report.get('current_selected_slot') is None
                and report.get('input_may_have_been_dispatched', False) is False and not failures
                and all(report.get(k) == 0 for k in ('execution_submit_count', 'execution_selection_click_count',
                    'identity_probe_click_count', 'selection_click_count', 'execution_click_count', 'total_input_click_count'))
                and refresh.get('phase') == 'authoritative_refresh_a' and refresh.get('expected_selected_slot') is None
                and refresh['proposal']['candidate_traces']['high_score']['action'] == action
                and report['proposal']['candidate_traces']['high_score']['action'] == action
                and stage == dict(phase='execution_selection', accepted=False, attempted_slot=action['source_slot'],
                    authorization_floor=stage.get('authorization_floor'), click_dispatched=False,
                    input_may_have_been_dispatched=False, reason='snapshot_expired_after_input_preparation')
                and preparation.get('dispatch_state') == 'rejected_before_dispatch'
                and preparation.get('input_may_have_been_dispatched') is False
                and preparation.get('reason') == stage['reason']
                and preparation.get('authorization_floor') == stage['authorization_floor']
                and refresh['authorization_floor'] <= stage['authorization_floor'] < refresh['input_deadline']
                and preparation.get('input_deadline') == refresh['input_deadline'])
        return (report.get('event') == 'live_selection_stop' and report.get('runtime_phase') == 'authoritative_refresh_a'
            and report.get('executed') is False and report.get('input_may_have_been_dispatched') is False
            and all(report.get(k) == 0 for k in ('execution_submit_count','execution_selection_click_count','identity_probe_click_count'))
            and not report.get('execution_selection_events') and not report.get('authoritative_refresh_events')
            and 1 <= len(failures) <= 2 and all(
                f.get('phase') == 'authoritative_refresh_a' and f.get('expected_selected_slot') is None
                and f.get('reason') == 'authoritative_live_evidence_window_expired'
                and f.get('accepted') is False and f.get('click_dispatched') is False
                and f.get('input_may_have_been_dispatched') is False for f in failures)
            and len(proposals) == 1 and proposals[0]['candidate_traces']['high_score']['action'] == receipt['action'])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _selected_outcomes(bundle, baseline, journal, action):
    """Replay the original card's rules, without choosing or advancing a policy."""
    from types import SimpleNamespace

    from ..high_score.live_policy import BoundHifOpeningPolicy

    policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    if journal.get('selected_skill_support') is not None:
        policy.bind_selected_skill_support(journal['selected_skill_support'])
    if journal.get('single_plus_estimate') is not None:
        policy.bind_selected_single_plus_estimate(journal['single_plus_estimate'], require_submitted=False)
    return policy.committed_transition()


def verify_cleared_selection(bundle, baseline, source, receipt):
    """Only a unique fresh before-state permits a new normal selection."""
    from ..high_score.hif_mcts_cache import same_json, json_value, action_from_json
    from .native_rule_progress_resume import source_receipt
    from ..high_score.hif_rule_progress import resolve_rule_pending

    journal = source['gray_selection_journal']
    raw = journal['refresh_a']['proposal']['candidate_traces']['high_score']['action']
    action = action_from_json(raw)
    unselected = receipt.get('unselected', {})
    if (receipt.get('source') != source['id'] or receipt.get('action') != json_value(asdict(action))
            or receipt.get('input_click_count') != 0 or receipt.get('accepted') is not True
            or unselected.get('accepted') is not True
            or unselected.get('unselected_geometry_authorized') is not True
            or unselected.get('geometry_stable') is not True
            or unselected.get('select_consensus', {}).get('accepted') is not True
            or unselected['select_consensus'].get('selected_slot') is not None
            or not unselected.get('frame_ids')
            or unselected['select_consensus'].get('observations') != [[] for _ in unselected['frame_ids']]
            or unselected.get('frame_ids') != receipt.get('initial_core', {}).get('frame_ids')
            or (unselected.get('stamina'), unselected.get('shield'))
                != (bundle.state.stamina, bundle.state.genki)):
        raise ValueError('cleared selection lacks its original intent and fresh unselected evidence')
    state, proof = resolve_rule_pending(bundle, _selected_outcomes(bundle, baseline, journal, action),
        baseline_event=source_receipt(baseline), action=action,
        initial_core=receipt['initial_core'], core=receipt['core'],
        single_plus_estimate=journal.get('single_plus_estimate'))
    if (state != bundle.state or proof.get('pending_outcome') != 'before'
            or proof.get('accepted') is not True or not same_json(proof, receipt.get('comparison'))):
        raise ValueError('cleared selection does not uniquely match the unconsumed before-state')
    return action


def reconcile_cleared_selection(context, checkpoint, result, module, params):
    """A cold selected checkpoint may have been deselected while the user played daily tasks.

    Preserve its successful SELECT and pending computation. Two independent
    core/hand observations distinguish before from every original successor;
    only then let the ordinary input path bind a new physical selection.
    """
    from .native_handoff_policy import STRONG, pending_action
    from ..high_score.hif_mcts_cache import json_value
    from .native_rule_progress_resume import source_receipt
    from ..high_score.hif_rule_progress import resolve_rule_pending

    if not checkpoint.get('selected') or getattr(checkpoint['policy'], 'profile', None) != STRONG:
        return
    bundle, baseline, source = checkpoint['bundle'], checkpoint['event'], checkpoint['row']
    def capture():
        if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
            return (context.tasker.controller.post_screencap().wait().get(),)
        return module._capture_hif_panel_frames(context)

    frames = capture()
    unselected = module._observe_live_resource_baseline(context, frames,
        expected_hand_count=len(bundle.state.deck.hand))
    if unselected.get('accepted') is not True:
        return  # The existing selected reader owns selected/ambiguous scenes.
    action = pending_action(checkpoint['policy'], bundle.state)
    journal = source['gray_selection_journal']
    expected = _selected_outcomes(bundle, baseline, journal, action)
    recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
    context.gkh_hif_card_art = module.FrozenMaskedArt.load(recognizer, module.MODEL_ROOT / 'hif')

    def read(images):
        core = module._inventory_live_checkpoint(context, images, recognizer,
            identity_threshold=params['debug_identity_threshold'],
            identity_min_margin=params['debug_identity_min_margin'], single_frame=len(images) == 1)
        return json_value({key: core[key] for key in ('fields', 'families', 'frame_ids')})

    receipt = dict(source=source['id'], action=json_value(asdict(action)),
        input_click_count=0, accepted=False, unselected=json_value(unselected))
    result['native_selection_rebind'] = receipt
    receipt['initial_core'] = read(frames)
    receipt['core'] = read(capture())
    state, comparison = resolve_rule_pending(bundle, expected, baseline_event=source_receipt(baseline),
        action=action, initial_core=receipt['initial_core'], core=receipt['core'],
        single_plus_estimate=journal.get('single_plus_estimate'))
    receipt['comparison'] = comparison
    receipt['accepted'] = state == bundle.state and comparison.get('pending_outcome') == 'before'
    verify_cleared_selection(bundle, baseline, source, receipt)
    # Source history and the complete pending tree remain unchanged. This new
    # record is a normal action with an explicit verified selected predecessor.
    result['native_rule_resume_source'] = result.pop('native_selected_source')
    context.gkh_hif_gray_selected_source = None


def restore_handoff_source(runtime, row, module, trail=()):
    """Replay a later action's parent after a recorded process boundary."""
    from .native_opening_root import verify_native_round_context
    from .native_inprocess_resume import promote_verified_settlement
    from .native_settlement_resume import load_verified_source

    receipt = row.get('handoff_resume', {})
    source = row.get('native_rule_resume_source')
    selected_origin = None
    if receipt.get('boundary_kind') == 'unsubmitted_selected':
        from .native_settlement_resume import load_result
        from .native_upstream_selected import original_selection

        # The first producer named the SELECT record in the receipt while the
        # business pointer correctly named its settled parent. Verify both;
        # do not rewrite the historical receipt or treat SELECT as a card play.
        previous = load_result(runtime, row['automatic_battle']['previous_record'])
        selected_origin = original_selection(runtime, previous)
        if (selected_origin['native_inprocess_source'] != source
                or receipt.get('source') not in (source, selected_origin['id'])
                or receipt.get('selection_source', receipt.get('source')) != selected_origin['id']
                or row.get('upstream_ui_journal', {}).get('action') != selected_origin['upstream_ui_journal']['action']):
            raise ValueError('HIF selected handoff changed its actual selection or business parent')
    if (receipt.get('version') != 1 or selected_origin is None and receipt.get('source') != source
            or row.get('information_boundary_execution') is not True
            or type(receipt.get('ownership_revision')) is not int):
        raise ValueError('HIF cross-process handoff receipt is incomplete')
    untouched = receipt.get('boundary_kind') == 'unsubmitted_opening'
    if untouched:
        from .native_opening_root import load_native_opening_root
        from .native_opening_resume import verify_unsubmitted_opening_source
        from .native_settlement_resume import load_result
        from .native_opening_mapping_resume import verify_opening_metadata_source
        from .native_opening_identity_resume import opening_source_chain

        parent = load_result(runtime, source)
        original, _ = opening_source_chain(runtime, parent)
        verify_opening_metadata_source(runtime, original)
        bundle, baseline = load_native_opening_root(original, module)
        verify_unsubmitted_opening_source(original, bundle, allow_saved_intent=True)
        copied = [event for event in row.get('live_events', ()) if event.get('event') == 'live_opening_state']
        if copied:
            from types import SimpleNamespace

            from live_decision.extended_opening import verify_unsubmitted_opening

            original = dict(copied[0])
            continuity = original.pop('unsubmitted_opening_resume', {})
            if (len(copied) != 1 or original != baseline or continuity.get('source') != source
                    or continuity.get('original_observation_reused') is not True
                    or type(continuity.get('input_click_count')) is not int
                    or continuity['input_click_count'] != 0):
                raise ValueError('HIF handoff republished opening differs from its original source')
            verify_unsubmitted_opening(SimpleNamespace(gkh_hif_opening_state=bundle,
                gkh_hif_live_context=module.HifLiveRunContext(**parent['observed_live_context'])),
                baseline, continuity.get('fresh_core', {}))
            baseline = copied[0]
        state = bundle.state
    elif row.get('native_selection_rebind') is not None:
        from .native_selected_resume import load_selected_source

        bundle, state, baseline, parent = load_selected_source(runtime, source, module, (*trail, row['id']))
        verify_cleared_selection(bundle, baseline, parent, row['native_selection_rebind'])
    else:
        bundle, state, baseline, parent = load_verified_source(runtime, source, module, (*trail, row['id']))
    live = receipt.get('live_context')
    if not isinstance(live, dict):
        raise ValueError('HIF cross-process receipt has no bound Live epoch')
    handoff = dict(handoff_id=receipt['handoff_id'], route_run_id=receipt['route_run_id'],
        ledger_revision=receipt['ledger_revision'], policy_profile=row['formal_params']['hif_policy_profile'])
    if (handoff['ledger_revision'] is None
            and parent.get('handoff_opening', {}).get('opening_inventory_source') == 'current_complete_opening'):
        handoff['source'] = 'current_battle_page'
    _binding(row, live, handoff)
    _binding(parent, live, handoff)
    if selected_origin is not None:
        _binding(selected_origin, live, handoff)
    verify_native_round_context(row, module, parent, required=True)
    if row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid'] or state.terminal:
        raise ValueError('HIF handoff continuation changed its game process or terminal state')
    unchanged = untouched or row.get('native_selection_rebind') is not None
    return (bundle if unchanged else promote_verified_settlement(bundle, state, baseline)), baseline
