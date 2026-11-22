"""Resume read-only hand observation after a completely settled native action."""
import math
from pathlib import Path


def unsubmitted_drink_inspection(row):
    """Return the original proposal only if the bottle inspection sent no input."""
    from ..hif_live_recovery import closed_observer_record

    try:
        event, = row['live_events']
        inspection, = event['transactions']
        proposal = event['proposal']
        cache = row['mcts_search_cache']
        if (row['status'] == 'STOP' and row.get('live_return') is False and not row.get('transaction_closed')
                and not row.get('error') and closed_observer_record(row)
                and row['formal_params']['hif_policy_profile'] == 'hif-strong-reference-trial-v1'
                and row.get('native_inprocess_source') and row['automatic_battle']['previous_record'] == row['native_inprocess_source']
                and row['input_count'] == 0 and row['native_input_dispatches'] == []
                and not any(row.get(key) for key in ('native_input_preparation', 'native_navigation_preparation',
                    'native_upstream_input_preparation', 'upstream_ui_journal', 'choice_journal', 'gray_selection_journal'))
                and cache['phase'] == 'pending_intent' and cache['record_id'] == row['id']
                and event['event'] == 'live_joint_action' and event['accepted'] is False
                and event['execution_submit_count'] == 0 and inspection['phase'] == 'drink_description_inspection'
                and inspection['accepted'] is False and inspection['input_click_count'] == inspection['drinks_consumed'] == 0
                and inspection['inputs'] == [] and not inspection.get('use_submitted')
                and proposal['has_action_intent'] is True and proposal['execution_authorized'] is False
                and proposal['action']['kind'] == 'USE_DRINK' and proposal['action'] == proposal['trace']['planning']['action']):
            return proposal
    except (KeyError, TypeError, ValueError):
        pass
    return None


def restore_unsubmitted_drink_inspection(runtime, row, module):
    """Restore the pre-inspection computation and physical bottle binding without performing UI."""
    from .native_inprocess_resume import restore_inprocess_source, verify_closed_transaction_record
    from .native_settlement_resume import load_result

    proposal = unsubmitted_drink_inspection(row)
    if proposal is None:
        raise ValueError('drink inspection lacks its original zero-input intent')
    parent = load_result(runtime, row['native_inprocess_source'], allow_closed_transaction=True)
    verify_closed_transaction_record(parent)
    if row['handoff_id'] != parent['handoff_id'] or row['total_input_count'] != parent['total_input_count']:
        raise ValueError('unsubmitted drink inspection changed the settled battle or its actual inputs')
    bundle, baseline = restore_inprocess_source(runtime, row, module)
    policy, restored = restore_unsubmitted_drink_policy(runtime, row, bundle, proposal)
    return bundle, baseline, policy, restored


def restore_unsubmitted_drink_policy(runtime, row, bundle, proposal):
    from dataclasses import asdict

    from .native_search_cache import _cache_source
    from .native_handoff_policy import pending_action, restore_policy
    from ..high_score.hif_mcts_cache import same_pending_plan

    owner, payload = _cache_source(runtime, row)
    plan = (payload or {}).get('pending_plan') or {}
    if owner['id'] != row['id'] or not same_pending_plan(plan, dict(proposal['trace']['planning'], action=plan.get('action'))):
        raise ValueError('unsubmitted drink inspection changed its complete evaluated plan')
    policy, restored = restore_policy(runtime, row, bundle, unsubmitted_plan=plan)
    observed = policy._drink_binding.observed_action(pending_action(policy, bundle.state))
    if asdict(observed) != proposal['action']:
        raise ValueError('unsubmitted drink inspection changed its observed bottle')
    return policy, restored


def restored_upstream_resource_intent(row):
    source = row.get('native_rule_resume_source')
    resume = row.get('handoff_resume') or {}
    restored = row.get('mcts_search_cache_restore') or {}
    return (bool(source) and not row.get('native_inprocess_source')
        and resume.get('source') == source and resume.get('boundary_kind') == 'settled'
        and restored.get('restored') is True and restored.get('restored_complete_policy') is True
        and restored.get('new_search_performed') is False
        and restored.get('reason') == 'restored_complete_unsubmitted_intent'
        and restored.get('source_record') == (row.get('automatic_battle') or {}).get('previous_record')
        and row.get('total_input_count') == 0)


def upstream_resource_intent_stop(row):
    """A complete upstream plan stopped before input at its old receipt adapter."""
    try:
        if upstream_unsubmitted_title_stop(row) or upstream_unsubmitted_end_stop(row):
            return True
        event, = row['live_events']
        failure, = event['failure_events']
        frames = failure['frame_ids']
        cache = row['mcts_search_cache']
        restored = restored_upstream_resource_intent(row)
        geometry = (failure.get('geometry_stable') is True
            and failure.get('select_consensus', {}).get('accepted') is True
            and failure['select_consensus'].get('observations') == [[], [], []]
            and failure['select_consensus'].get('selected_slot') is None
            and len(failure['select_evidence']) == 3
            and all(e.get('source') == 'pinned_upstream_select_panel_and_raised_card'
                and e.get('resolved_slots') == [] and e.get('frame_id') == frame
                for e, frame in zip(failure['select_evidence'], frames, strict=True)))
        unobserved = (restored and failure.get('geometry_stable') is False
            and failure.get('select_consensus') is None and failure.get('select_evidence') == [])
        return (row['formal_params']['hif_policy_profile'] == 'hif-strong-reference-trial-v1'
            and cache['phase'] == 'pending_intent' and cache['record_id'] == row['id']
            and (restored or row.get('native_inprocess_source') == row['automatic_battle']['previous_record'])
            and event['runtime_phase'] == 'resource_baseline'
            and event['reason'] == 'authoritative_live_resource_baseline_unresolved'
            and failure.get('accepted') is False and (geometry or unobserved)
            and len(frames) == len(set(frames)) == 3
            and len(failure['resource_observations']) == 3
            and all(reading[name].get('accepted') is True
                and type(reading[name].get('value')) is int
                and .95 <= reading[name].get('confidence', 0) <= 1
                for reading in failure['resource_observations'] for name in ('stamina', 'shield')))
    except (KeyError, TypeError, ValueError):
        return False


def upstream_unsubmitted_title_stop(row):
    """Legacy title normalization failed before the saved END reached its owner."""
    events = row.get('live_events', ())
    cache = row.get('mcts_search_cache') or {}
    return (restored_upstream_resource_intent(row) and len(events) == 1
        and row.get('formal_params', {}).get('hif_policy_profile') == 'hif-strong-reference-trial-v1'
        and cache.get('phase') == 'pending_intent' and cache.get('record_id') == row.get('id')
        and events[0].get('runtime_phase') == 'observe_cards'
        and events[0].get('reason') == 'HIF title layout requires normalized physical card footprints'
        and events[0].get('failure_events') == [] and events[0].get('identity_probe_events') == [])


def upstream_unsubmitted_end_stop(row):
    events = row.get('live_events', ())
    cache = row.get('mcts_search_cache') or {}
    return (restored_upstream_resource_intent(row) and len(events) == 1
        and row.get('formal_params', {}).get('hif_policy_profile') == 'hif-strong-reference-trial-v1'
        and cache.get('phase') == 'pending_intent' and cache.get('record_id') == row.get('id')
        and events[0].get('event') == 'live_end_turn' and events[0].get('accepted') is False
        and events[0].get('end_turn_submit_count') == events[0].get('execution_submit_count') == 0
        and not events[0].get('input_may_have_been_dispatched'))


def upstream_resource_intent_proposal(runtime, row):
    if not upstream_resource_intent_stop(row):
        return None
    from .native_search_cache import _cache_source

    _, payload = _cache_source(runtime, row)
    plan = (payload or {}).get('pending_plan')
    if not isinstance(plan, dict) or plan.get('accepted') is not True:
        raise ValueError('upstream resource recovery lacks its complete evaluated plan')
    if (upstream_unsubmitted_title_stop(row) or upstream_unsubmitted_end_stop(row)) and plan['action']['kind'] != 'END_TURN':
        raise ValueError('legacy title continuation is not an evaluated END intent')
    return dict(candidate_traces=dict(high_score=dict(action=plan['action'], trace=dict(planning=plan))))


def unsubmitted_post_commit_proposal(row):
    """A computed intention stopped in observation, before SELECT or dispatch."""
    try:
        events = row['live_events']
        if len(events) not in (1, 2) or not row.get('native_rule_resume_source') or row.get('native_inprocess_source'):
            return None
        stop = events[-1]
        if len(events) == 2:
            observed = events[0]
            if (observed.get('event') != 'live_preinput_state' or observed.get('accepted') is not False
                    or observed.get('executed') is not False or observed.get('total_input_click_count') != 0
                    or observed.get('execution_submit_count') != 0 or observed.get('components')
                    or stop.get('event') != 'live_decision_exception_stop' or stop.get('runtime_phase') != 'observe_cards'
                    or stop.get('submit_dispatched') is not False
                    or len(observed.get('failure_events', ())) != 1):
                return None
            failure = observed['failure_events'][0]
            if (failure.get('phase') != 'hif_settlement_hand_pending'
                    or len(failure.get('original_failure_events', ())) != 1):
                return None
        else:
            if (stop.get('event') != 'live_selection_stop' or stop.get('runtime_phase') != 'authoritative_refresh_a'
                    or stop.get('input_may_have_been_dispatched') is not False
                    or len(stop.get('failure_events', ())) != 1):
                return None
            failure = stop['failure_events'][0]
            if (failure.get('phase') != 'authoritative_refresh_a' or failure.get('accepted') is not False
                    or failure.get('click_dispatched') is not False
                    or failure.get('input_may_have_been_dispatched') is not False
                    or failure.get('expected_selected_slot') is not None
                    or failure.get('resource_transaction', {}).get('accepted') is not True
                    or failure.get('unselected_geometry_authorized') is not True
                    or not failure.get('retained_hand_families')
                    or not failure.get('identity_revalidation')
                    or any(p.get('accepted') is not True for p in failure['identity_revalidation'])):
                return None
        probes = stop.get('identity_probe_events', ())
        resume = row['handoff_resume']
        restored = row['mcts_search_cache_restore']
        if (stop.get('executed') is not False
                or stop.get('would_submit') is not False
                or any(stop.get(k) != 0 for k in ('execution_submit_count','execution_selection_click_count','identity_probe_click_count'))
                or any(stop.get(k) for k in ('execution_selection_events','authoritative_refresh_events',
                    'foreground_preparation_events','shadow_transaction_events'))
                or [p.get('phase') for p in probes] != ['initial_observation','retained_complete_hand_identity','hif_candidate_precompute']
                or probes[0].get('click_dispatched') is not False
                or probes[0].get('geometry_stable') is not True
                or probes[0].get('identity_source_accepted') is not True
                or probes[0].get('select_consensus', {}).get('selected_slot') is not None
                or probes[0].get('select_consensus', {}).get('accepted') is not True
                or probes[1].get('accepted') is not True or probes[2].get('click_dispatched') is not False
                or resume.get('boundary_kind') != 'settled' or resume.get('source') != row['native_rule_resume_source']
                or (restored.get('source_record') != resume['source'] and not (
                    restored.get('source_record') == row['automatic_battle']['previous_record']
                    and restored.get('restored') is True and restored.get('new_search_performed') is False
                    and restored.get('reason') == 'restored_complete_unsubmitted_intent'))
                or row['mcts_search_cache'].get('phase') != 'pending_intent'
                or row['formal_params'].get('hif_policy_profile') != 'hif-strong-reference-trial-v1'):
            return None
        proposal = probes[2]['proposal']
        if (proposal.get('status') != 'PROPOSE'
                or proposal['candidate_traces']['high_score']['trace']['planning'].get('accepted') is not True):
            return None
        return proposal
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def restored_post_commit_observation(row):
    """The same settled computation was restored, then observed with no input."""
    source = row.get('native_rule_resume_source')
    resume, cache, restored = (row.get(k) or {} for k in
        ('handoff_resume', 'mcts_search_cache', 'mcts_search_cache_restore'))
    return (bool(source) and not row.get('native_inprocess_source')
        and resume.get('source') == source and resume.get('boundary_kind') == 'settled'
        and restored.get('source_record') == source and restored.get('restored') is True
        and restored.get('restored_complete_policy') is True
        and cache.get('phase') == 'observed_successor' and cache.get('record_id') == row.get('id')
        and row.get('total_input_count') == 0)


def post_commit_source(row):
    return (row.get('native_rule_resume_source') if unsubmitted_post_commit_proposal(row) is not None
        or restored_post_commit_observation(row) or restored_upstream_resource_intent(row)
        else row.get('native_inprocess_source'))


def post_commit_failure(row):
    if upstream_unsubmitted_title_stop(row) or upstream_unsubmitted_end_stop(row):
        return {}
    failure = row['live_events'][0]['failure_events'][0]
    if failure.get('phase') == 'authoritative_refresh_a':
        proofs = {p['slot']:p for p in failure.get('retained_hand_families', ())}
        return dict(phase='inventory_live_identity', frame_ids=failure['frame_ids'], frame_id=failure['frame_ids'][-1],
            predictions=[proofs[p['slot']]['original_prediction'] if p['slot'] in proofs else p['fresh_prediction']
                for p in failure['identity_revalidation']])
    return failure['original_failure_events'][0] if unsubmitted_post_commit_proposal(row) is not None else failure


def is_post_commit_identity_stop(row):
    from .native_pending_frontier import is_completed_frontier_stop

    if is_completed_frontier_stop(row) or unsubmitted_drink_inspection(row) is not None:
        return True
    from ..opening_observer import resource_baseline_reobserve_allowed
    from ..hif_live_recovery import closed_observer_record
    from ..hif_upstream_policy import unselected_resource_reobserve_allowed

    try:
        if upstream_unsubmitted_end_stop(row):
            return (row.get('status') == 'STOP' and closed_observer_record(row)
                and row.get('live_return') is False and not row.get('transaction_closed')
                and not row.get('error') and row.get('input_count') == 0
                and row.get('native_input_dispatches') == []
                and not any(row.get(k) for k in ('native_input_preparation', 'native_navigation_preparation',
                    'native_upstream_input_preparation', 'native_pending_source', 'native_selected_source',
                    'native_resume_source', 'choice_journal', 'gray_selection_journal')))
        preinput = unsubmitted_post_commit_proposal(row) is not None
        restored = restored_post_commit_observation(row) or restored_upstream_resource_intent(row)
        if (row.get('status') != 'STOP' or not closed_observer_record(row)
                or row.get('live_return') is not False or row.get('transaction_closed')
                or row.get('error') or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
                or not post_commit_source(row)
                or not preinput and not restored and row['automatic_battle']['previous_record'] != row['native_inprocess_source']
                or any(row.get(k) for k in ('native_input_preparation','native_navigation_preparation',
                    'native_pending_source','native_selected_source','native_resume_source','choice_journal','gray_selection_journal'))
                or not preinput and len(row['live_events']) != 1):
            return False
        event = row['live_events'][-1] if preinput else row['live_events'][0]
        if (not preinput and event.get('event') != 'live_selection_stop'
                or event.get('executed') is not False
                # observe_cards conservatively marks debug_execute as possible
                # input even before any probe. Only this closed native record's
                # independently empty dispatch/preparation ledger can disprove it.
                or row['formal_params'].get('live_decision_mode') != 'opening_execute'
                or event.get('would_submit') is not False
                or any(event.get(k) != 0 for k in ('execution_submit_count','execution_selection_click_count','identity_probe_click_count'))
                or any(event.get(k) for k in ('authoritative_refresh_events','execution_selection_events',
                    'shadow_transaction_events')) or not preinput and event.get('identity_probe_events')
                or any(prep.get('phase') != 'resource_baseline_reobservation'
                    or prep.get('input_click_count') != 0
                    or not (resource_baseline_reobserve_allowed(prep.get('prior', {}),
                        row['formal_params'].get('debug_resource_absolute_min_confidence'))
                        or upstream_resource_intent_stop(row)
                        and prep.get('retry_kind') == 'upstream_transient_geometry'
                        and unselected_resource_reobserve_allowed(prep.get('prior', {})))
                    for prep in event.get('foreground_preparation_events', ()))
                or not preinput and not upstream_unsubmitted_title_stop(row)
                and len(event.get('failure_events', ())) != 1):
            return False
        if upstream_unsubmitted_title_stop(row):
            return True  # Still subject to the closed zero-input ledger checks above.
        failure = post_commit_failure(row)
        if (not preinput and event.get('runtime_phase') == 'observe_cards'
                and event.get('reason') == 'initial Live identity source layout is invalid'):
            frames = failure.get('frame_ids', ())
            rows = failure.get('candidate_boxes', ())
            selected = failure.get('select_consensus', {})
            return (not row.get('mcts_search_cache') and not row.get('mcts_search_cache_restore')
                and failure.get('phase') == 'initial_observation' and failure.get('click_dispatched') is False
                and failure.get('geometry_stable') is True and failure.get('identity_source_accepted') is False
                and failure.get('identity_source_reason') == 'live_identity_source_layout_invalid'
                and len(frames) == len(set(frames)) == 3
                and set(frames) <= {f['frame_id'] for f in row['frames']}
                and failure.get('identity_source_frame_id') == frames[0]
                and len(rows) == 3 and rows[0] == failure.get('identity_source_candidate_boxes')
                and 1 <= len(rows[-1]) <= 5 and rows[-2] == rows[-1]
                and selected.get('accepted') is True and selected.get('selected_slot') is None
                and selected.get('observations') == [[], [], []])
        if event.get('runtime_phase') == 'resource_baseline':
            frames = failure.get('frame_ids', ())
            upstream_intent = upstream_resource_intent_stop(row)
            return (event.get('input_may_have_been_dispatched') is False
                and (restored or upstream_intent or not row.get('mcts_search_cache') and not row.get('mcts_search_cache_restore'))
                and (upstream_intent or resource_baseline_reobserve_allowed(failure,
                    row['formal_params'].get('debug_resource_absolute_min_confidence')))
                and len(frames) == len(set(frames)) == 3
                and set(frames) <= {f['frame_id'] for f in row['frames']})
        refresh_a = preinput and event.get('runtime_phase') == 'authoritative_refresh_a'
        if (event.get('runtime_phase') != 'observe_cards' and not refresh_a
                or not preinput and event.get('reason') != 'inventory navigation Live visual family is unresolved'
                or event.get('input_may_have_been_dispatched') is not (not refresh_a)):
            return False
        predictions = failure['predictions']
        frames = failure['frame_ids']
        if (failure.get('phase') != 'inventory_live_identity' or not 1 <= len(predictions) <= 5
                or [p['slot'] for p in predictions] != list(range(len(predictions)))
                or len(frames) != 3 or len(set(frames)) != 3 or failure['frame_id'] not in frames
                or not set(frames) <= {f['frame_id'] for f in row['frames']}):
            return False
        weak = [p for p in predictions if p.get('accepted') is False]
        if (weak and all(type(p.get('accepted')) is bool for p in predictions)
                and (preinput or not row.get('mcts_search_cache') and not row.get('mcts_search_cache_restore'))
                and all(p.get('reason') == 'below_margin_threshold'
                    and p.get('recognizer') == 'card_art_embedding_top5_hif_masked_v1'
                    and math.isfinite(p['confidence'])
                    and row['formal_params']['debug_identity_threshold'] <= p['confidence'] <= 1
                    and math.isfinite(p['margin'])
                    and 0 <= p['margin'] < row['formal_params']['debug_identity_min_margin'] for p in weak)):
            return True
        return (bool(weak) and all(p.get('accepted') in (True,False) for p in predictions)
            and len(predictions) <= 3
            and all(p.get('reason') == 'hif_full_masked_family_conflict'
                and p.get('recognizer') == 'card_art_embedding_full_masked_conflict_v1'
                and math.isfinite(p['confidence']) and 0 <= p['confidence'] < .85
                and math.isfinite(p['margin']) and p['margin'] >= .05 for p in weak))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def verify_post_commit_source(row, parent, module):
    from .native_opening_root import verify_native_round_context
    from .native_inprocess_resume import verify_closed_transaction_record

    if not is_post_commit_identity_stop(row) or post_commit_source(row) != parent['id']:
        raise ValueError('HIF identity continuation lacks its closed parent')
    verify_closed_transaction_record(parent)
    preinput = unsubmitted_post_commit_proposal(row) is not None
    restored = restored_post_commit_observation(row) or restored_upstream_resource_intent(row)
    if (any(row.get(k) != parent.get(k) for k in ('observed_live_context','handoff_id','handoff_opening'))
            or not preinput and not restored and any(row.get(k) != parent.get(k) for k in ('host_attempt','endpoints'))
            or row['formal_params']['hif_policy_profile'] != parent['formal_params']['hif_policy_profile']
            or not restored and row['total_input_count'] != parent['total_input_count']):
        raise ValueError('HIF identity continuation changed its settled host or business binding')
    if preinput or restored:
        from .native_settlement_resume import _verified_endpoints

        _verified_endpoints(row['endpoints'], parent['endpoints']['game']['pid'])
        if row['endpoints']['game'] != parent['endpoints']['game']:
            raise ValueError('HIF preinput continuation changed its game instance')
    verify_native_round_context(row, module, parent)
    if row['live_events'][-1].get('runtime_phase') == 'authoritative_refresh_a':
        settled = next(e for e in parent['live_events'] if e['event'] == 'live_committed_state')
        generation = row['live_events'][-1]['failure_events'][0]['generation']
        fields = [generation[key] for key in
            ('turns_remaining','stamina','shield','score','score_multiplier','action_order')]
        if fields != settled['core']['fields'] or generation['hand_count'] != len(settled['core']['families']):
            raise ValueError('HIF preinput continuation changed its settled public state')
    if preinput or row['live_events'][0].get('runtime_phase') == 'observe_cards':
        failure = post_commit_failure(row)
        if any(p.get('reason') == 'below_margin_threshold' for p in failure.get('predictions', ())):
            settled = next(e for e in parent['live_events'] if e['event'] == 'live_committed_state')
            families = settled['core']['families']
            if (len(families) != len(failure['predictions'])
                    or any(not p.get('top_k_card_ids') or not set(family) & set(p['top_k_card_ids'][0])
                        for family, p in zip(families, failure['predictions'], strict=True))):
                raise ValueError('HIF observation continuation changed its settled hand families')
    if row['live_events'][0].get('runtime_phase') == 'resource_baseline':
        settled = next(e for e in parent['live_events'] if e['event'] == 'live_committed_state')
        fields = settled['core']['fields']
        failure = row['live_events'][0]['failure_events'][0]
        if (any(reading[name]['value'] != value
                for reading in failure['resource_observations']
                for name, value in zip(('stamina', 'shield'), fields[1:3], strict=True)
                if reading[name].get('accepted') is True or reading[name].get('value') is not None)
                or any(boxes and len(boxes) != len(settled['core']['families'])
                    for boxes in failure['candidate_boxes'])):
            raise ValueError('HIF observation continuation differs from its settled resources or hand')


def require_repaired_post_commit_identity(checkpoint, *, readback=None, runtime=None, context=None):
    import cv2
    from utils.runtime_paths import resolve_runtime_path
    from card_selection.model import frame_identifier, isolate_card_candidates
    from card_selection.embedding import EmbeddingCardRecognizer

    from ..hif_visual import HifFrameTitleCache, hif_unselected_row_candidates
    from ..masked_art import FrozenMaskedArt
    from ..partial_card import classify_live_card_views, normalise_live_candidates
    from .native_handoff_policy import phase

    row = checkpoint['row']
    drink_proposal = unsubmitted_drink_inspection(row)
    has_intent = drink_proposal is not None or unsubmitted_post_commit_proposal(row) is not None or upstream_resource_intent_stop(row)
    if (not is_post_commit_identity_stop(row) or checkpoint.get('untouched_opening')
            or checkpoint.get('selected') or checkpoint.get('pending')
            or phase(checkpoint['policy']) != ('pending_intent' if has_intent else 'observed_successor')
            or (checkpoint['policy']._pending is not None) != has_intent):
        raise ValueError('HIF identity continuation is not a completely settled successor')
    if readback is not None:
        return _require_new_post_commit_observation(checkpoint, readback, runtime, context)
    if drink_proposal is not None:
        # The checkpoint loader verified the actual bottle binding and full
        # saved plan. The normal policy and drink UI own the new transaction.
        return
    if upstream_unsubmitted_end_stop(row):
        return  # Computation only; fresh business values and upstream SKIP bind input.
    if row['live_events'][0].get('runtime_phase') == 'resource_baseline' or upstream_unsubmitted_title_stop(row):
        from types import SimpleNamespace

        from ..selection import same_live_candidate_geometry

        module = checkpoint['observer_module']
        actual = row['frames'][-3:]
        failure = (dict(frame_ids=[f['frame_id'] for f in actual]) if upstream_unsubmitted_title_stop(row)
            else row['live_events'][0]['failure_events'][0])
        raw = [r['raw']['all'] for r in row['live_raw_fields'] if r['node'] == 'ProduceRecognitionCards'][-3:]
        if [f['frame_id'] for f in actual] != failure['frame_ids'] or len(raw) != 3:
            raise ValueError('HIF baseline continuation lacks its original detector frames')
        images = tuple(cv2.imread(frame['path']) for frame in actual)
        if any(image is None or frame_identifier(image) != frame['frame_id']
                for image, frame in zip(images, actual, strict=True)):
            raise ValueError('HIF baseline continuation pixels are missing or changed')
        context = SimpleNamespace(gkh_hif_live_context=module.HifLiveRunContext(**row['observed_live_context']),
            gkh_hif_committed_event=checkpoint['event'], gkh_hif_opening_state=checkpoint['bundle'],
            gkh_hif_upstream_card_ui=upstream_resource_intent_stop(row))
        rows = tuple(module._normalise_observed_live_candidates(context, image,
            [SimpleNamespace(**r) for r in detection] if context.gkh_hif_upstream_card_ui else detection)
            for image, detection in zip(images, raw, strict=True))
        rows, _ = module._reconcile_hif_unselected_detector_gap(context, images, rows)
        stable = bool(rows[0]) and all(same_live_candidate_geometry(rows[0], other) for other in rows[1:])
        if not stable and context.gkh_hif_upstream_card_ui:
            from ..hif_upstream_policy import unselected_row_anchors

            anchors = [unselected_row_anchors(cards, image.shape)
                for image, cards in zip(images, rows, strict=True)]
            stable = bool(anchors[0]) and all(a == anchors[0] for a in anchors[1:])
            if not stable and restored_upstream_resource_intent(row):
                present = [a for a in anchors if a is not None]
                if (present and len(present[0]) == len(checkpoint['state'].deck.hand)
                        and all(a == present[0] for a in present)
                        and all(not cards or a is not None for cards, a in zip(rows, anchors, strict=True))):
                    # A closed zero-input attempt cannot consume the saved
                    # intent. Restore computation only; these incomplete old
                    # frames are not evidence for any input. The runtime still
                    # requires a fresh, complete three-frame resource baseline.
                    return
        if (not rows[0] or len(rows[0]) != len(checkpoint['state'].deck.hand)
                or not stable):
            raise ValueError('HIF baseline continuation current physical row remains unresolved')
        return  # Only restore the settled successor; a fresh observer must bind every input.
    model_root = resolve_runtime_path(Path('assets/resource/base/model/classify/card_embedding'),
        Path('resource/base/model/classify/card_embedding'))
    recognizer = EmbeddingCardRecognizer.load(model_root)
    bank = FrozenMaskedArt.load(recognizer, model_root/'hif')
    failure = post_commit_failure(row)
    paths = {f['frame_id']:f['path'] for f in row['frames']}
    raw = [r['raw']['all'] for r in row['live_raw_fields'] if r['node'] == 'ProduceRecognitionCards'][-3:]
    if [f['frame_id'] for f in row['frames'][-3:]] != failure['frame_ids'] or len(raw) != 3:
        raise ValueError('HIF identity continuation lacks its actual final three detector frames')
    expected = checkpoint['state'].deck.hand
    from types import SimpleNamespace

    module = checkpoint['observer_module']
    context = SimpleNamespace(gkh_hif_live_context=module.HifLiveRunContext(**row['observed_live_context']),
        gkh_hif_committed_event=checkpoint['event'], gkh_hif_opening_state=checkpoint['bundle'],
        gkh_hif_card_art=bank)
    frame_rows = tuple(zip(failure['frame_ids'], raw, strict=True))
    if failure.get('identity_source_reason') == 'live_identity_source_layout_invalid':
        # The first transient crop is retained as a failure, never certified.
        # Both later complete rows must independently match the settled hand.
        frame_rows = frame_rows[-2:]
    for fid, detected in frame_rows:
        image = cv2.imread(paths[fid])
        if image is None or frame_identifier(image) != fid:
            raise ValueError('HIF identity continuation failure pixels are missing or changed')
        candidates = isolate_card_candidates(detected,include_unplayable=True)
        cards = normalise_live_candidates(hif_unselected_row_candidates(image.shape,candidates,
            image=image,title_cache=HifFrameTitleCache(image)) or (),frame_shape=image.shape)
        predictions = classify_live_card_views(recognizer,image,cards,frame_id=fid,
            plan=row['observed_live_context']['plan'],
            acceptance_threshold=row['formal_params']['debug_identity_threshold'],
            min_margin=row['formal_params']['debug_identity_min_margin'],
            **module._live_card_art_profile(context,cards,None))
        predictions = module._corroborate_retained_hif_families(context,image,cards,predictions,
            selected_slot=None,threshold=row['formal_params']['debug_identity_threshold'],evidence=[])
        if (len(predictions) != len(expected) or not predictions or any(not p.accepted
                or expected[p.slot].card_id not in p.top_k_card_ids[0] for p in predictions)):
            raise ValueError('HIF post-commit identity remains unresolved on the saved current frames')


def _require_new_post_commit_observation(checkpoint, readback, runtime, context):
    """Re-read a separate, closed Maa capture; never rewrite the failed pixels.

    This restores computation only. The ordinary observer still captures and
    binds the current game before any selection or submission can occur.
    """
    import re
    import json
    from types import SimpleNamespace

    import cv2
    from card_selection.model import frame_identifier

    row, module = checkpoint['row'], checkpoint['observer_module']
    if (not isinstance(readback, dict) or set(readback) != {'source_record', 'capture_record'}
            or readback['source_record'] != row['id']
            or not isinstance(readback['capture_record'], str)
            or re.fullmatch(r'task275-[a-z0-9-]{1,130}', readback['capture_record']) is None
            or post_commit_failure(row).get('identity_source_reason') != 'live_identity_source_layout_invalid'
            or context is None or runtime is None):
        raise ValueError('HIF readback must explicitly bind the closed layout failure')
    runtime = Path(runtime).resolve()
    source = readback['capture_record']
    report = json.loads((runtime / f'result-{source}.json').read_bytes())
    closed = json.loads((runtime / f'{source}.exit.json').read_bytes())
    endpoints = report.get('endpoints', {})
    frames = report.get('frames', ())
    if (report.get('id') != source or report.get('status') != 'CAPTURED_NO_INPUT'
            or report.get('input_count') != 0 or closed.get('exit_code') != 0
            or set(endpoints) != {'maapicli', 'agent', 'game'}
            or any(type(e.get('pid')) is not int or e['pid'] <= 0
                or e.get('integrity_rid') != 12288 for e in endpoints.values())
            or endpoints['game']['pid'] != row['endpoints']['game']['pid']
            or endpoints['game']['client_size'] != row['endpoints']['game']['client_size']
            or len(frames) != 3 or [f.get('label') for f in frames] != ['before', 'read_only_1', 'read_only_2']
            or report.get('capture_size') != [720, 1280]):
        raise ValueError('HIF readback lacks its same-game closed zero-input Maa capture')
    paths = [Path(f['path']).resolve() for f in frames]
    times = [f.get('capture_monotonic') for f in frames]
    if (len(set(paths)) != 3
            or any(not p.is_relative_to(runtime / '.local/runtime-data/images') for p in paths)
            or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times)
            or not times[0] < times[1] < times[2]):
        raise ValueError('HIF readback lacks three separate capture frames')
    images = tuple(cv2.imread(str(p)) for p in paths)
    if any(image is None or image.shape != (1280, 720, 3) for image in images):
        raise ValueError('HIF readback capture pixels are missing or invalid')
    frame_ids = tuple(frame_identifier(image) for image in images)
    if len(set(frame_ids)) != 3 or set(frame_ids) & {f['frame_id'] for f in row['frames']}:
        raise ValueError('HIF readback cannot re-label the failed saved frames')
    recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
    observed_context = SimpleNamespace(run_recognition=context.run_recognition,
        get_node_data=context.get_node_data,
        gkh_hif_live_context=module.HifLiveRunContext(**row['observed_live_context']),
        gkh_hif_client_size=tuple(endpoints['game']['client_size']),
        gkh_hif_committed_event=checkpoint['event'], gkh_hif_opening_state=checkpoint['bundle'],
        gkh_hif_card_art=module.FrozenMaskedArt.load(recognizer, module.MODEL_ROOT / 'hif'))
    core = module._inventory_live_checkpoint(observed_context, images, recognizer,
        identity_threshold=row['formal_params']['debug_identity_threshold'],
        identity_min_margin=row['formal_params']['debug_identity_min_margin'])
    expected = checkpoint['event']['core']
    # Preserve customised definitions in the restored kernel. Artwork identifies
    # their canonical printed card, not the internal effect-suffixed key.
    kernel = checkpoint['bundle'].kernel
    canonical = [kernel.cards[card.card_id].canonical_card_id or card.card_id
        for card in checkpoint['state'].deck.hand]
    if (json.dumps(core['fields']) != json.dumps(expected['fields'])
            or len(core['families']) != len(expected['families'])
            or any(not set(prior).issubset(current)
                for prior, current in zip(expected['families'], core['families'], strict=True))
            or len(core['families']) != len(checkpoint['state'].deck.hand)
            or any(card_id not in family
                for card_id, family in zip(canonical, core['families'], strict=True))):
        raise ValueError('HIF readback differs from the fully settled successor')
    return dict(source_record=row['id'], capture_record=source, input_count=0,
        frame_ids=frame_ids, capture_monotonic=times, fields=core['fields'], families=core['families'],
        candidate_boxes=core['candidate_boxes'], title_family_corroboration=core['title_family_corroboration'],
        purpose='restore_computation_only_fresh_input_observation_still_required')
