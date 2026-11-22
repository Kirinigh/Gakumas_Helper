"""Resume one held selection after repairing ambiguous non-target artwork."""
import math
from types import SimpleNamespace


def is_unsubmitted_neighbor_identity_stop(row):
    """Classify a closed observation; this grants neither identity nor PLAY."""
    from ..hif_bound_title import in_candidate_band, retryable_low_confidence_family
    from ..hif_live_recovery import closed_observer_record
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    try:
        sources = [row.get(key) for key in ('native_inprocess_source', 'native_rule_resume_source')
                   if row.get(key)]
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or row.get('transaction_closed') or row.get('error') or row.get('failure_event')
                or not closed_observer_record(row)
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches',())) != 1
                or len(sources) != 1 or row.get('native_navigation_preparation')
                or (row.get('native_rule_resume_source') and row.get('information_boundary_execution') is not True)):
            return False
        events = row['live_events']
        if len(events) == 2:
            # The normal selected-source loader independently rebuilds this
            # existing rule-preinput receipt; keep it in the original record.
            from .native_rule_progress_resume import PREINPUT_SCOPE

            preinput = events[0]
            if (preinput.get('event') != 'live_preinput_state'
                    or preinput.get('verification_scope') != PREINPUT_SCOPE
                    or preinput.get('accepted') is not True or preinput.get('executed') is not False
                    or preinput.get('strategy_state_bound') is not True
                    or not preinput.get('checks') or not all(v is True for v in preinput['checks'].values())
                    or any(preinput.get(k, 0) != 0 for k in ('total_input_click_count',
                        'execution_submit_count', 'execution_selection_click_count', 'end_turn_submit_count',
                        'identity_probe_click_count', 'drinks_consumed'))):
                return False
        elif len(events) != 1:
            return False
        report = events[-1]
        if (report.get('event') != 'live_selection_stop' or report.get('executed') is not False
                or report.get('runtime_phase') not in ('execute', 'authoritative_refresh_b')
                or report.get('reason') != 'authoritative_live_identity_family_changed'
                or report.get('execution_selection_click_count') != 1
                or any(report.get(k,0) != 0 for k in ('execution_submit_count','end_turn_submit_count',
                    'identity_probe_click_count','drinks_consumed')) or report.get('foreground_preparation_events')):
            return False
        first = report['authoritative_refresh_events'][0]
        action = first['proposal']['candidate_traces']['high_score']['action']
        slot,count = action['source_slot'],first['generation']['hand_count']
        failure = report['failure_events'][-1]
        ids = failure['frame_ids']
        identities = failure['identity_revalidation']
        if (type(slot) is not int or type(count) is not int or count not in (4,5) or not 0 <= slot < count
                or action.get('kind') != 'PLAY_CARD' or not action.get('card_instance_id')
                or first.get('phase') != 'authoritative_refresh_a' or first.get('accepted') is not True
                or first.get('unselected_geometry_authorized') is not True
                or first.get('safe_decision',{}).get('action') != dict(kind='PLAY_CARD',card_slot=slot)
                or failure.get('phase') != 'authoritative_refresh_b' or failure.get('accepted') is not False
                or failure.get('reason') != report['reason'] or failure.get('click_dispatched') is not False
                or failure.get('expected_selected_slot') != slot or failure.get('geometry_stable') is not True
                or any(failure.get(k) != v for k,v in (('health_votes',3),('playable_votes',3),('secondary_votes',0),('popup_votes',0)))
                or len(ids) != 3 or len(set(ids)) != 3 or len(identities) != count
                or failure['select_consensus'].get('accepted') is not True
                or failure['select_consensus'].get('observations') != [[slot],[slot],[slot]]
                or [r.get('slot') for r in identities] != list(range(count))):
            return False
        age = failure.get('evidence_window_age_ms')
        from .native_card_evidence import evidence_window_limit

        if type(age) not in (int,float) or not math.isfinite(age) or not 0 <= age <= evidence_window_limit(failure):
            return False
        weak = [r for r in identities if r.get('accepted') is not True]
        if not weak or any(r['slot'] == slot or not (
                in_candidate_band(SimpleNamespace(**r['fresh_prediction']),r['expected_family'],.05)
                or retryable_low_confidence_family(SimpleNamespace(**r['fresh_prediction']),r['expected_family'],.05))
                for r in weak):
            return False
        if any(r.get('fresh_family') != r.get('expected_family') for r in identities if r not in weak):
            return False
        selection,*steps = report['execution_selection_events']
        if (selection.get('phase') != 'execution_selection' or selection.get('accepted') is not True
                or selection.get('attempted_slot') != slot or selection.get('click_dispatched') is not True
                or selection.get('reason') != 'target_selection_dispatched'
                or any(step.get('click_dispatched') is not False or step.get('phase') not in (
                    'hif_selected_single_plus','hif_selected_printed_cost','hif_selected_skill_support') for step in steps)):
            return False
        guard,dispatch = row['native_input_preparation'],row['native_input_dispatches'][0]
        if (guard.get('dispatch_state') != 'dispatch_attempted' or dispatch.get('kind') is not None
                or (guard.get('x'),guard.get('y')) != (dispatch.get('x'),dispatch.get('y'))):
            return False
        for endpoints in (guard['endpoints'],dispatch['endpoints']):
            _verified_endpoints(endpoints,row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'],endpoints)
        return True
    except (KeyError,ValueError,TypeError,AttributeError,IndexError):
        return False


def require_repaired_neighbor_identity(checkpoint):
    """Validate the held intent; normal fresh B owns current identity checks.

    A recovery qualification is not input evidence. Re-recognizing rejected old
    pixels here duplicated B with an incomplete context and could prevent even
    a fresh observation. Keep the original SELECT/no-submit and computation
    binding; grant no card identity or input permission from this check.
    """
    from .native_handoff_policy import STRONG, phase

    row = checkpoint['row']
    policy, state = checkpoint['policy'], checkpoint['state']
    if (not is_unsubmitted_neighbor_identity_stop(row) or checkpoint.get('selected') is not True
            or checkpoint.get('pending') or checkpoint.get('untouched_opening')
            or phase(policy) != 'pending_intent' or policy._pending is None):
        raise ValueError('HIF neighbor recovery lacks its original selected intent')
    if getattr(policy, 'profile', None) == STRONG:
        before, action = policy._pending['before'], policy._pending['action']
        expected = policy.kernel._canonical_probability_state(state)
    else:
        before, action = policy._pending
        expected = state
    failure = row['live_events'][-1]['failure_events'][-1]
    slot = failure['expected_selected_slot']
    if (before != expected or action.source_slot != slot
            or action.card_instance_id != state.deck.hand[slot].instance_id):
        raise ValueError('HIF neighbor recovery changed its original selected intent')
