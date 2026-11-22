"""Restore a held selection after an unresolved cost OCR observation."""
import math


def selected_cost_confidence_stop(row):
    """Classify the original failure; its weak cost never authorises input."""
    from ..hif_live_recovery import closed_observer_record
    from .native_card_evidence import _fresh, _frames
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    try:
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or row.get('error') or row.get('failure_event') or row.get('transaction_closed')
                or not closed_observer_record(row) or not row.get('native_inprocess_source')
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches', ())) != 1
                or row.get('native_navigation_preparation')):
            return False
        events = row['live_events']
        if len(events) == 2:
            projection = events[0]
            from .native_rule_progress_resume import PREINPUT_SCOPE

            # The existing selected-source loader independently reconstructs
            # either the hand projection or the bound rule-preinput receipt.
            hand = (projection.get('event') == 'live_hand_projection'
                and projection.get('source') == 'hif_current_turn_ordinary_plus_v1')
            preinput = (projection.get('event') == 'live_preinput_state'
                and projection.get('verification_scope') == PREINPUT_SCOPE
                and projection.get('strategy_state_bound') is True
                and bool(projection.get('checks'))
                and all(v is True for v in projection['checks'].values()))
            if (not (hand or preinput) or projection.get('accepted') is not True
                    or projection.get('executed') is not False
                    or any(projection.get(k, 0) != 0 for k in ('total_input_click_count',
                        'execution_submit_count', 'execution_selection_click_count', 'end_turn_submit_count',
                        'identity_probe_click_count', 'drinks_consumed'))):
                return False
        elif len(events) != 1:
            return False
        report = events[-1]
        reason = 'authoritative_live_displayed_effective_cost_confidence_unapproved'
        if (report.get('event') != 'live_selection_stop' or report.get('executed') is not False
                or report.get('runtime_phase') != 'authoritative_refresh_b' or report.get('reason') != reason
                or report.get('execution_selection_click_count') != 1
                or any(report.get(k, 0) != 0 for k in ('execution_submit_count', 'end_turn_submit_count',
                    'identity_probe_click_count', 'drinks_consumed')) or report.get('foreground_preparation_events')):
            return False
        first, = report['authoritative_refresh_events']
        failure, = report['failure_events']
        action = first['proposal']['candidate_traces']['high_score']['action']
        slot = action['source_slot']
        _fresh(first, slot)
        if (action['kind'] != 'PLAY_CARD' or first.get('unselected_geometry_authorized') is not True
                or failure.get('reason') != reason or failure.get('phase') != 'authoritative_refresh_b'
                or failure.get('accepted') is not False or failure.get('click_dispatched') is not False
                or failure.get('expected_selected_slot') != slot or failure.get('geometry_stable') is not True
                or failure.get('selected_identity_authorized') is not True
                or failure['select_consensus'].get('observations') != [[slot], [slot], [slot]]
                or failure['select_consensus'].get('accepted') is not True
                or _frames(first) & _frames(failure)
                or any(item.get('accepted') is not True for item in failure['identity_revalidation'])):
            return False
        resource = failure['resource_transaction']
        preview = resource.get('card_effect_preview')
        if preview is None:
            if (resource.get('displayed_effective_cost') != 0 or resource.get('card_face_cost') != 0
                    or resource.get('conservation') != dict(
                        stamina=True, shield=True, effective_cost=True, shield_first_split=True)):
                return False
        elif (preview.get('source') != 'native_hif_card_effect_preview'
                or preview.get('selected_slot') != slot
                or (preview.get('run_id'), preview.get('revision')) != (
                    row['observed_live_context']['run_id'], row['observed_live_context']['revision'])
                or resource.get('card_id') != preview.get('card_id')
                or resource.get('conservation') != dict(
                    stamina=True, shield=True, declared_cost=True, modeled_card_preview=True)
                or any(type(resource.get(key)) is not int or resource[key] < 0 for key in (
                    'displayed_effective_cost', 'card_face_cost', 'effective_payment_cost'))
                or resource['card_face_cost'] != resource['displayed_effective_cost']
                or resource['card_face_cost'] != preview.get('displayed_card_cost', preview.get('effective_cost'))
                or resource['effective_payment_cost'] != preview.get('effective_cost')
                or any(resource.get('normalized_'+key) != preview.get('baseline_'+key)
                    or resource.get('preview_'+key) != preview.get('preview_'+key) for key in ('stamina', 'shield'))):
            return False
        confidence = resource.get('displayed_effective_cost_confidence')
        threshold = failure.get('debug_displayed_effective_cost_min_confidence')
        if (resource.get('accepted') is not True or resource.get('selected_slot') != slot
                or resource.get('frame_ids') != failure['frame_ids']
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in (confidence, threshold))
                or not 0 <= confidence < threshold <= 1):
            return False
        guard, dispatch = row['native_input_preparation'], row['native_input_dispatches'][0]
        selected, *steps = report['execution_selection_events']
        if (any(selected.get(key) != value for key, value in dict(
                phase='execution_selection', accepted=True, attempted_slot=slot,
                click_dispatched=True, reason='target_selection_dispatched').items())
                or any(step.get('click_dispatched') is not False for step in steps)
                or guard.get('dispatch_state') != 'dispatch_attempted' or dispatch.get('kind') is not None
                or (guard.get('x'), guard.get('y')) != (dispatch.get('x'), dispatch.get('y'))):
            return False
        for endpoints in (guard['endpoints'], dispatch['endpoints']):
            _verified_endpoints(endpoints, row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'], endpoints)
        return True
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return False


def require_repaired_selected_cost(checkpoint):
    """Keep the verified intent; only a new complete A/B may grant submission."""
    from .native_handoff_policy import phase, pending_action

    row, policy = checkpoint['row'], checkpoint['policy']
    if (not selected_cost_confidence_stop(row) or checkpoint.get('selected') is not True
            or checkpoint.get('pending') or phase(policy) != 'pending_intent' or policy._pending is None):
        raise ValueError('HIF cost recovery lacks its original held selection')
    action = row['live_events'][-1]['authoritative_refresh_events'][0]['proposal']['candidate_traces']['high_score']['action']
    state = checkpoint['state']
    pending = pending_action(policy, state)
    if (pending.kind.value != action['kind']
            or pending.source_slot != action['source_slot'] or pending.card_instance_id != action['card_instance_id']
            or state.deck.hand[pending.source_slot].instance_id != pending.card_instance_id):
        raise ValueError('HIF cost recovery changed its retained action')
