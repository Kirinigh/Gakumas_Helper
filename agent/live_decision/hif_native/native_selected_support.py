"""Independently replay action-bound support details retained in native reports."""
import json


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def bind_recorded_selected_support(policy, report, *, require_submitted=False, source_row=None):
    """Bind evidence, not an observed poststate; the caller then runs the kernel."""
    entries = [event for event in report.get('execution_selection_events', ())
               if event.get('phase') == 'hif_selected_skill_support'
               and (event.get('receipt') or {}).get('effects_verified') is True]
    receipt = report.get('selected_skill_support')
    if receipt is None and not entries:
        return False
    if (not isinstance(receipt, dict) or len(entries) != 1
            or not _same(entries[0].get('receipt'), receipt)):
        raise ValueError('native selected support receipt and original observation differ')
    policy.bind_selected_skill_support(receipt)
    if require_submitted:
        stages = report.get('execution_selection_events', ())
        confirmations = [index for index, event in enumerate(stages)
                         if event.get('phase') == 'execution_submit' and event.get('click_dispatched') is True]
        if (report.get('execution_submit_count') != 1 or len(confirmations) != 1
                or stages.index(entries[0]) >= confirmations[0]):
            raise ValueError('native support was not verified before its one actual submission')
        refreshes = report.get('authoritative_refresh_events', ())
        if len(refreshes) > 2 or any(event.get('phase') == 'execution_submit_reobservation'
                                    for event in stages):
            from .native_card_evidence import dispatched_card_refreshes
            refreshes = dispatched_card_refreshes(report, row=source_row, bundle=policy.bundle)
        selected = [event for event in refreshes if event.get('phase') == 'authoritative_refresh_b']
        slot = receipt['selected_slot']
        if (len(selected) != 1 or selected[0].get('accepted') is not True
                or selected[0].get('expected_selected_slot') != slot
                or len(set(selected[0].get('frame_ids', ()))) != 3
                or set(selected[0]['frame_ids']) & set(receipt['frame_ids'])):
            raise ValueError('native support lacks an independent final selected B')
        from .native_card_evidence import input_evidence_is_fresh

        if not input_evidence_is_fresh(selected[0]):
            raise ValueError('native supported submission B evidence was stale')
        transaction = selected[0].get('resource_transaction', {})
        if (transaction.get('accepted') is not True or transaction.get('selected_slot') != slot
                or not _same(transaction.get('card_effect_preview'), policy.selected_resource_preview(slot))):
            raise ValueError('native support B resources differ from the independently projected effects')
    return True


def bind_observed_pending_support(policy, source_row, report, receipt, *, single_plus_estimate,
                                  baseline_event=None):
    """Read a dispatched B's complete effects without rewriting its old estimate.

    This recovery evidence grants no input. The caller still needs a fresh,
    complete and unique settlement of the independently projected outcomes.
    """
    from live_decision.high_score.support_schedule import CARD_ID, FOUR_TURN_CARD_ID

    from .native_card_evidence import _fresh, dispatched_card_refreshes

    if (not isinstance(receipt, dict) or not isinstance(single_plus_estimate, dict)
            or report.get('selected_skill_support') is not None
            or single_plus_estimate.get('definition_id') not in (CARD_ID, FOUR_TURN_CARD_ID)
            or receipt.get('definition_id') != single_plus_estimate.get('definition_id')
            or single_plus_estimate.get('plus_count') != 2
            or single_plus_estimate.get('ordinary_plus_projection') is True
            or receipt.get('raw_title') != single_plus_estimate.get('raw_title')):
        raise ValueError('pending support lacks its original known scheduled-card estimate')
    action = policy.response.action
    failures = [event for event in source_row.get('live_events', ())
                if event.get('event') == 'live_committed_state']
    estimated_rule = (len(failures) == 1
        and receipt.get('definition_id') == FOUR_TURN_CARD_ID
        and source_row.get('information_boundary_execution') is True
        and failures[0].get('verification_scope') == 'rule_maintained_information_boundary_v1'
        and failures[0].get('accepted') is True)
    if (len(failures) != 1 or not estimated_rule and failures[0].get('accepted') is not False
            or failures[0].get('executed') is not False
            or failures[0].get('execution_submit_count') != 0
            or failures[0].get('total_input_click_count') != 0
            or (failures[0].get('components') not in (None, {}) if estimated_rule
                else failures[0].get('components') != {})
            or not estimated_rule and failures[0].get('reason') != 'HIF settlement public resources do not match a legal outcome'
            or not _same(failures[0].get('single_plus_estimate'), single_plus_estimate)
            or (source_row.get('observed_live_context', {}).get('run_id'),
                source_row.get('observed_live_context', {}).get('revision'))
                != (policy.bundle.run_id, policy.bundle.revision)
            or report.get('execution_submit_count') != 1):
        raise ValueError('pending support is not one unresolved dispatched action')
    if estimated_rule:
        from .native_rule_progress_resume import verify_rule_progress

        # This was only an ordinary-plus rule projection, not an independent
        # observation of status effects. Preserve it and prove it before
        # deriving a correction from the original B's complete card body.
        verify_rule_progress(policy.bundle, policy.committed_transition(), baseline_event,
                             action, failures[0])
    first, final = dispatched_card_refreshes(report, row=source_row, bundle=policy.bundle)
    for event, phase in ((first, 'authoritative_refresh_a'), (final, 'authoritative_refresh_b')):
        _fresh(event, action.source_slot)
        if event.get('phase') != phase:
            raise ValueError('pending support source A/B phases differ')
    evidence = {key: final.get(key) for key in (
        'frame_ids', 'geometry_stable', 'select_consensus', 'select_evidence')}
    if (final.get('expected_selected_slot') != action.source_slot
            or not _same(receipt.get('frame_ids'), final.get('frame_ids'))
            or not _same(receipt.get('selection_evidence'), evidence)
            or set(first.get('frame_ids', ())) & set(final.get('frame_ids', ()))):
        raise ValueError('pending support does not use the original final selected B')
    # Recompute every raw title, cost and effect row against the current
    # complete prestate, physical instance and existing support definition.
    policy.bind_selected_skill_support(receipt)
    preview = policy.selected_resource_preview(action.source_slot)
    transaction = final.get('resource_transaction', {})
    score = final.get('visual_fields', {}).get('score', {})
    if (transaction.get('accepted') is not True
            or transaction.get('selected_slot') != action.source_slot
            or transaction.get('card_id') != receipt['card_id']
            or not transaction.get('conservation')
            or not all(value is True for value in transaction['conservation'].values())
            or not _same(transaction.get('frame_ids'), receipt['frame_ids'])
            or score.get('accepted') is not True
            or any(transaction.get(observed) != preview[expected] for observed, expected in (
                ('normalized_stamina', 'baseline_stamina'), ('normalized_shield', 'baseline_shield'),
                ('effective_payment_cost', 'effective_cost'), ('preview_stamina', 'preview_stamina'),
                ('preview_shield', 'preview_shield')))
            or score.get('value') != preview['preview_score']
            or transaction.get('card_effect_preview') is not None
                and not _same(transaction['card_effect_preview'], preview)):
        raise ValueError('pending support effects contradict the original B resources')
    return policy.selected_skill_support
