"""Recover the dispatched 919 support play from original components and fresh history."""
import copy
import json
from dataclasses import asdict

PENDING_SOURCE = '919-selected-preview-normalization-resume-step-2'
TARGET_INSTANCE = 'hif-recovered-bc0a3ee75ee343b0ac57e690e7a2494c:山札:13'


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _frames(values):
    return (isinstance(values, (list, tuple)) and len(values) == 3
            and all(isinstance(value, str) and value for value in values)
            and len(set(values)) == 3)


def _same_core(left, right):
    return all(_same(left.get(key), right.get(key)) for key in ('fields', 'families'))


def _read_only_current(row, source_row, module):
    from .native_opening_root import verify_native_round_context
    from .native_settlement_resume import _verified_endpoints

    old, current = source_row['endpoints']['game'], row['endpoints']['game']
    context = row.get('context_recovery') or {}
    if (old['pid'] != current['pid'] or old['client_size'] != current['client_size']
            or context.get('accepted') is not True or context.get('returned_to_live') is not True
            or context.get('context') != row.get('observed_live_context')
            or context.get('game_pid') != current['pid']
            or any(source_row['observed_live_context'][key] != row['observed_live_context'][key]
                   for key in ('idol', 'plan', 'max_stamina'))
            or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                   or event.get('drinks_consumed', 0) or event.get('executed') is True
                   for event in row.get('live_events', ()))):
        raise ValueError('support recovery is not a read-only continuation of the same process and Live')
    verify_native_round_context(row, module, source_row, required=True)
    dispatches = row.get('native_input_dispatches', ())
    if row.get('input_count') != len(dispatches):
        raise ValueError('support recovery navigation accounting is incomplete')
    for entry in dispatches:
        _verified_endpoints(entry.get('endpoints'), current['pid'])


def _completed_source_batch(row, bundle, submitted):
    from .native_settlement_resume import _verified_endpoints

    batches = [event for event in row['live_events'] if event.get('event') == 'live_committed_state']
    if len(batches) != 1:
        raise ValueError('support source lacks one complete original component batch')
    batch = batches[0]
    if (batch.get('accepted') is not False or batch.get('strategy_state_bound') is not False
            or batch.get('reason') != 'hif_settlement_outcome_not_unique'
            or batch.get('run_id') != bundle.run_id or batch.get('revision') != bundle.revision):
        raise ValueError('support source is not the unresolved original settlement')
    before, after = batch.get('initial_core', {}), batch.get('core', {})
    if (not _frames(before.get('frame_ids')) or not _frames(after.get('frame_ids'))
            or set(before['frame_ids']) & set(after['frame_ids']) or not _same_core(before, after)):
        raise ValueError('support source changed during complete component observation')
    components = batch.get('components', {})
    if set(components) != {'cards', 'status', 'items', 'drinks', 'turns'}:
        raise ValueError('support source lacks all five components')
    for key, component in components.items():
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or component.get('run_id') != bundle.run_id or component.get('revision') != bundle.revision):
            raise ValueError('support component is incomplete or from another epoch: ' + key)
        if key != 'drinks' and (not _frames(component.get('before_frame_ids'))
                or not _frames(component.get('after_frame_ids'))
                or set(component['before_frame_ids']) & set(component['after_frame_ids'])
                or component.get('observation_source') == 'retained_verified_component'):
            raise ValueError('support component lacks its original fresh frames: ' + key)
    dispatches = row.get('native_input_dispatches', ())
    if (row.get('input_count') != len(dispatches)
            or len(dispatches) != batch.get('total_input_click_count', -1) + 2
            or submitted.get('execution_selection_click_count') != 1
            or submitted.get('execution_submit_count') != 1):
        raise ValueError('support source is not one selection, one submission, then component navigation')
    for entry in dispatches:
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
    from .native_card_evidence import dispatched_card_refreshes
    refreshes = dispatched_card_refreshes(submitted, row=row, bundle=bundle)
    for index, refresh in enumerate(refreshes):
        box = refresh['candidate_boxes'][0][1]
        center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        if (dispatches[index].get('kind') == 'swipe'
                or (dispatches[index].get('x'), dispatches[index].get('y')) != center):
            raise ValueError('support source first audited inputs differ from its exact selected slot')
    post = [event for event in submitted.get('execution_selection_events', ())
            if event.get('phase') == 'execution_postcondition' and event.get('accepted') is True]
    if (len(post) != 1 or post[0].get('business_committed') is not True
            or not _frames(post[0].get('frame_ids'))
            or set(post[0]['frame_ids']) & set(before['frame_ids'])
            or any(post[0]['stable_post_values'].get(key) != after['fields'][index]
                   for key, index in (('turns_remaining', 0), ('stamina', 1), ('score', 3)))):
        raise ValueError('support source lacks settled business arrival before component navigation')
    return batch


def _supported_history(history_row, source_row, module, before, card_phase, outcomes):
    from .native_visible_history import visible_history_consensus

    _read_only_current(history_row, source_row, module)
    visible = visible_history_consensus(history_row['visible_history_observations'],
        latest_event=history_row['history_observation'])
    if (not _same(visible, history_row.get('visible_history'))
            or (visible['run_id'], visible['revision']) != tuple(
                history_row['observed_live_context'][key] for key in ('run_id', 'revision'))):
        raise ValueError('support visible history differs from the original three-frame observation')
    scalar = {(state.turn, state.score, state.concentration) for state in outcomes}
    if len(scalar) != 1:
        raise ValueError('support projection has ambiguous history scalar outcomes')
    turn, score, concentration = next(iter(scalar))
    title = 'スキルカード「深呼吸++」'
    groups = visible['groups']
    locations = [index for index, group in enumerate(groups) if group['title'] == title]
    if len(locations) != 1:
        raise ValueError('support history lacks one exact double-enhanced deep-breath title')
    expected = [
        dict(kind='card', title=title, lines=[title, 'スキルカードサポート',
            f'集中{before.concentration:g}→{card_phase.concentration:g}',
            f'好調{before.good_condition_turns:g}→{card_phase.good_condition_turns:g}']),
        dict(kind='system', title='持続効果「天真爛漫」', lines=['持続効果「天真爛漫」',
            f'集中{card_phase.concentration:g}→{concentration:g}']),
        dict(kind='turn-start', title=f'{turn}ターン目', lines=[f'{turn}ターン目',
            f'{round(outcomes[0].score_multiplier * 100)}%', '手札情報']),
        dict(kind='system', title='持続効果「破竹の勢い+」', lines=['持続効果「破竹の勢い+」',
            f'スコア{before.score:g}→{score:g}']),
    ]
    if (not _same(groups[locations[0]:], expected) or visible['score'] != score
            or visible['turns'] != [turn]):
        raise ValueError('support history does not match the independently projected direct and scheduled effects')
    return visible


def collect_support_source(runtime, pending_name, history_name, module):
    """Recompute 919, keep its false record intact, and verify its existing five components."""
    from live_decision.high_score.opening import verify_hif_committed_observation
    from live_decision.high_score.skill_support import (
        SUPPORT_SOURCE,
        project_supported_card_once,
        supported_deep_breath_definition,
    )

    from .native_settlement_resume import load_result, load_pending_source, decoded_dispatched_action

    if pending_name != PENDING_SOURCE:
        raise ValueError('support historical recovery is limited to the dispatched 919 action')
    bundle, _, baseline, source_row = load_pending_source(runtime, pending_name, module)
    action, submitted = decoded_dispatched_action(source_row, bundle)
    state = bundle.state
    if (action.card_instance_id != TARGET_INSTANCE or action.source_slot != 1
            or state.turn != 5 or state.turns_remaining != 5 or state.actions_remaining != 1
            or state.deck.hand[1].card_id != '57'):
        raise ValueError('support historical action differs from its complete original binding')
    original = _completed_source_batch(source_row, bundle, submitted)
    definition = supported_deep_breath_definition(bundle.kernel.cards['57'])
    phases, outcomes = project_supported_card_once(bundle, action, effective_definition=definition)
    if len(phases) != 1 or phases[0].probability != 1 or phases[0].state.actions_remaining != 0:
        raise ValueError('support historical direct play is not one deterministic final action')
    history_row = load_result(runtime, history_name)
    visible = _supported_history(history_row, source_row, module, state, phases[0].state, outcomes)
    if not _same_core(visible['after'], original['core']):
        raise ValueError('support fresh history core differs from the completed source components')
    comparisons = [verify_hif_committed_observation(bundle, candidate, core=original['core'],
        components=original['components'], baseline_components=baseline['components']) for candidate in outcomes]
    matches = [index for index, result in enumerate(comparisons) if result['accepted']]
    if len(matches) != 1:
        raise ValueError('support complete component match is not unique')
    index = matches[0]
    corrected = outcomes[index]
    receipt = dict(event='native_single_action_support_projection', source=pending_name,
        history_source=history_name, effect_source=SUPPORT_SOURCE, enhancement_count=2,
        run_id=bundle.run_id, revision=bundle.revision, action=asdict(action),
        complete_pre_state=asdict(state), effective_definition=asdict(definition),
        card_phase=asdict(phases[0].state), projected_outcome_state=asdict(corrected),
        candidate_count=len(outcomes), matched_outcome_index=index,
        visible_history_frame_ids=visible['frame_ids'], original_settlement_accepted=False,
        original_business_submit_count=1, replay_business_submit_count=0,
        retained_component_names=sorted(original['components']), submission_authorized=False)
    settlement = copy.deepcopy(original)
    settlement.update(accepted=True, strategy_state_bound=True,
        reason='hif_single_action_support_projection_verified', checks=comparisons[index]['checks'],
        expected_state=asdict(corrected), matched_outcome_count=1,
        observation_source='original_complete_batch_with_verified_support_projection',
        support_projection=receipt)
    return dict(bundle=bundle, state=corrected, settlement=settlement, source_row=source_row,
        history_row=history_row, receipt=receipt, reference=visible['latest_observation'])


def resume_support_source(source, current_row, module, current_history):
    """Publish a new epoch only through the existing fresh history continuity gate."""
    _read_only_current(current_row, source['history_row'], module)
    if set(current_history.get('frame_ids', ())) & set(source['history_row']['visible_history']['frame_ids']):
        raise ValueError('support final continuity reused the support proof frames')
    return _restore_support_history(source, module, current_history,
        module.HifLiveRunContext(**current_row['observed_live_context']), current_row['endpoints']['game']['pid'])


def _restore_support_history(source, module, history, new_run, new_pid):
    from live_decision.history import restore_verified_history

    bundle, state, event = restore_verified_history(source['bundle'], source['state'], settlement=source['settlement'],
        reference=source['reference'], current=history,
        old_run=module.HifLiveRunContext(**source['source_row']['observed_live_context']),
        new_run=new_run,
        old_pid=source['source_row']['endpoints']['game']['pid'],
        new_pid=new_pid)
    # The verified physical hand order is part of the new bound state. Keep
    # the unordered chance outcome separately in the projection receipt.
    event['expected_state'] = asdict(state)
    return bundle, state, event


def restore_native_history(runtime, module, bundle, expected, *, settlement, reference, current,
                           old_run, new_run, old_pid, new_pid):
    """Bind the current settlement and its observed history to the active battle."""
    from live_decision.history import restore_verified_history

    return restore_verified_history(bundle, expected, settlement=settlement,
        reference=reference, current=current, old_run=old_run, new_run=new_run,
        old_pid=old_pid, new_pid=new_pid)


def load_verified_support_source(runtime, name, module):
    from .native_settlement_resume import load_result

    row = load_result(runtime, name)
    if row.get('live_return') is not True:
        raise ValueError('support recovery did not complete')
    source = collect_support_source(runtime, row['native_support_source'], row['support_history_source'], module)
    bundle, state, event = resume_support_source(source, row, module, row['support_resume_history'])
    if (not _same(row.get('support_projection'), source['receipt'])
            or not _same(row.get('settlement'), event)
            or not _same(row.get('post_history'), row['support_resume_history'])):
        raise ValueError('saved support recovery differs from its independently verified sources')
    return bundle, state, event, row


def is_unsubmitted_support_stop(row):
    from ..hif_live_recovery import closed_observer_record
    from .native_upstream_selected import selected_read_without_input, is_unsubmitted_upstream_selection

    if selected_read_without_input(row) or is_unsubmitted_upstream_selection(row):
        return True

    try:
        events = row['live_events']
        if (row.get('status') != 'STOP' or row.get('error') or row.get('failure_event')
                or row.get('transaction_closed') or not closed_observer_record(row)
                or len(events) != 1 or row.get('input_count') != 1
                or len(row.get('native_input_dispatches', ())) != 1):
            return False
        event = events[0]
        steps = event['execution_selection_events']
        selected = [s for s in steps if s.get('phase') == 'execution_selection']
        support = [s for s in steps if s.get('phase') == 'hif_selected_skill_support']
        if (event.get('event') != 'live_selection_stop' or event.get('executed') is not False
                or event.get('execution_selection_click_count') != 1
                or event.get('execution_submit_count') != 0
                or any(event.get(k, 0) for k in ('end_turn_submit_count', 'drinks_consumed', 'identity_probe_click_count'))
                or len(selected) != 1 or selected[0].get('accepted') is not True
                or selected[0].get('click_dispatched') is not True or len(support) != 1
                or support[0].get('click_dispatched') is not False
                or any(s.get('click_dispatched') is True for s in steps if s is not selected[0])):
            return False
        receipt = support[0]['receipt']
        return (receipt.get('accepted') is False and receipt.get('effects_verified') is False
            and receipt.get('submission_authorized') is False
            and receipt.get('reason') == 'selected extra enhancement is outside the verified slice'
            and event.get('reason') == 'selected_skill_support_unresolved: ' + receipt['reason']
            and len(receipt.get('observations', ())) == 3
            and len(set(receipt.get('frame_ids', ()))) == 3)
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def require_repaired_support(checkpoint):
    """Full cold loader has rechecked original observations and permanent state."""
    row = checkpoint['row']
    if checkpoint.get('upstream_selected'):
        from .native_handoff_policy import phase
        from .native_upstream_selected import selected_plan

        selected_plan(row)
        if phase(checkpoint['policy']) != 'pending_intent' or checkpoint.get('pending'):
            raise ValueError('upstream selection lacks its complete original computation')
        return  # Fresh two-frame title/effects remain mandatory in the UI callback.
    journal = row.get('gray_selection_journal') or {}
    replay = journal.get('selected_skill_support_replay') or {}
    verified = journal.get('selected_skill_support') or {}
    if (not is_unsubmitted_support_stop(row) or checkpoint.get('selected') is not True
            or checkpoint.get('pending') or replay.get('source_record') != row['id']
            or replay.get('submission_authorized') is not False
            or replay.get('fresh_support_required_before_submit') is not True
            or verified.get('accepted') is not True or verified.get('effects_verified') is not True):
        raise ValueError('HIF selected support repair lacks its full original replay')


RESOURCE_PREVIEW_STOP = 'hif_full_state_revalidation_failed:HIF generated card resource preview is not independently verified'
RESOURCE_PREVIEW_STOPS = frozenset((RESOURCE_PREVIEW_STOP,
    'hif_full_state_revalidation_failed:HIF composite resource preview is not independently verified'))


def _resource_preview_read_only_prefix(event, failure):
    steps = event.get('execution_selection_events')
    if failure.get('phase') == 'authoritative_refresh_a':
        return steps == [] and failure.get('input_may_have_been_dispatched') is False
    refreshes = event.get('authoritative_refresh_events', ())
    if (failure.get('phase') != 'authoritative_refresh_b' or len(refreshes) != 1
            or not isinstance(steps, list) or len(steps) != 3
            or any(step.get('click_dispatched') is not False
                   or step.get('input_may_have_been_dispatched') is True for step in steps)):
        return False
    first = refreshes[0]
    selected, reuse, support = steps
    slot = failure.get('expected_selected_slot')
    frames = first.get('frame_ids', ())
    receipt = support.get('receipt') or {}
    action = first.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
    # The old B reader sets its may-have-input flag merely from phase=B. Here
    # the outer STOP, native empty dispatch/preparation and all known read-only
    # stages independently prove zero new input; do not rewrite that old flag.
    return (failure.get('input_may_have_been_dispatched') is True
        and first.get('phase') == 'authoritative_refresh_a' and first.get('accepted') is True
        and first.get('click_dispatched') is False and first.get('input_may_have_been_dispatched') is False
        and first.get('selected_identity_authorized') is True and first.get('expected_selected_slot') == slot
        and first.get('safe_decision', {}).get('action') == dict(kind='PLAY_CARD', card_slot=slot)
        and action.get('kind') == 'PLAY_CARD' and action.get('source_slot') == slot
        and action.get('card_instance_id') == receipt.get('card_instance_id')
        and first.get('generation') == failure.get('expected_generation') == failure.get('generation')
        and len(frames) == 3 and len(set(frames)) == 3
        and not set(frames) & set(failure.get('frame_ids', ()))
        and selected == dict(phase='execution_selection', attempted_slot=slot,
            accepted=True, click_dispatched=False, reason='target_already_selected')
        and reuse == dict(phase='hif_selected_model_evidence_reuse', selected_slot=slot,
            click_dispatched=False, source_phase='authoritative_refresh_a', frame_ids=frames)
        and support.get('phase') == 'hif_selected_skill_support'
        and receipt == event.get('selected_skill_support')
        and receipt.get('accepted') is True and receipt.get('effects_verified') is True
        and receipt.get('submission_authorized') is False and receipt.get('selected_slot') == slot
        and receipt.get('frame_ids') == frames
        and first.get('resource_transaction', {}).get('card_effect_preview')
            == failure.get('resource_transaction', {}).get('card_effect_preview'))


def _repaired_selected_baseline_geometry(row, event):
    from ..hif_live_recovery import repaired_selected_layout_frames

    if (event.get('reason') != 'gray singleton preview: ordinary selected baseline lacks independent SELECT and raised geometry'
            or event.get('authoritative_refresh_events') != [] or event.get('execution_selection_events') != []):
        return None
    retries = event.get('foreground_preparation_events', ())
    if (len(retries) != 1 or retries[0].get('phase') != 'selected_baseline_geometry_reobservation'
            or retries[0].get('input_click_count') != 0):
        return None
    groups = [retries[0]['prior'],event['failure_events'][0]]
    reference = None
    for group in groups:
        rows = group.get('candidate_boxes', ())
        complete = [boxes for boxes in rows if boxes]
        if (group.get('geometry_stable') is not False or group.get('select_consensus') is not None
                or group.get('select_evidence') != [] or len(rows) != 3 or len(complete) != 2):
            return None
        if reference is None:
            reference = complete[0]
        if not 2 <= len(reference) <= 5 or any(boxes != reference for boxes in complete):
            return None
    if any(len(b) != 4 or any(type(v) is not int for v in b) or b[2:] != [190,250] for b in reference):
        return None
    slots = [i for i,b in enumerate(reference) if all(18 <= other[1]-b[1] <= 30
        for j,other in enumerate(reference) if j != i)]
    if len(slots) != 1 or not repaired_selected_layout_frames(row,[g['frame_ids'] for g in groups],
            reference,slots[0],reference_selected=True):
        return None
    return reference,slots[0]


def selected_resource_family_stop(row):
    """A zero-input baseline rejected only because its ordinary family was absent."""
    from ..hif_live_recovery import closed_observer_record

    try:
        reason = 'gray singleton preview: ordinary selected card has no supported resource preview'
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or row.get('error') or row.get('failure_event') or row.get('transaction_closed')
                or not closed_observer_record(row) or not row.get('native_selected_source')
                or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
                or row.get('native_input_preparation') or row.get('native_navigation_preparation')):
            return False
        event, = row['live_events']
        failure, = event['failure_events']
        if (event.get('event') != 'live_selection_stop' or event.get('executed') is not False
                or event.get('input_may_have_been_dispatched') is not False
                or event.get('reason') != reason or event.get('runtime_phase') != 'resource_baseline'
                or event.get('authoritative_refresh_events') != [] or event.get('execution_selection_events') != []
                or event.get('foreground_preparation_events')
                or any(event.get(k, 0) for k in ('execution_submit_count', 'execution_selection_click_count',
                    'identity_probe_click_count', 'end_turn_submit_count', 'drinks_consumed'))):
            return False
        resource = failure['resource_transaction']
        slot, frames = resource.get('selected_slot'), failure.get('frame_ids')
        consensus = failure['select_consensus']
        confidence = resource.get('displayed_effective_cost_confidence')
        cost = resource.get('displayed_effective_cost')
        return (type(slot) is int and 0 <= slot < 5 and _frames(frames)
            and failure.get('geometry_stable') is True
            and consensus.get('accepted') is True and consensus.get('selected_slot') == slot
            and consensus.get('observations') == [[slot]] * 3
            and failure['detail_identity'].get('accepted') is True
            and resource.get('accepted') is True and resource.get('frame_ids') == frames
            and resource.get('card_effect_preview') is None
            and type(cost) is int and cost >= 0 and resource.get('card_face_cost') == cost
            and type(confidence) in (int, float) and .95 <= confidence <= 1
            and resource.get('conservation') == dict(
                effective_cost=True, stamina=True, shield=True, shield_first_split=True))
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def _raw_selected_resource_baseline(failure):
    """Recognize complete raw observations; the loader must still bind them."""
    resource = failure.get('resource_transaction') or {}
    preview = resource.get('card_effect_preview') or {}
    identity = failure.get('detail_identity') or {}
    consensus = failure.get('select_consensus') or {}
    frames, slot = failure.get('frame_ids'), resource.get('selected_slot')
    observations = identity.get('observations', ())
    return ('accepted' not in failure and 'submission_authorized' not in failure
        and failure.get('geometry_stable') is True and _frames(frames)
        and type(slot) is int and 0 <= slot < 5
        and consensus.get('accepted') is True and consensus.get('selected_slot') == slot
        and consensus.get('observations') == [[slot]] * 3
        and identity.get('accepted') is True and identity.get('card_id') == preview.get('card_id')
        and len(observations) == 3 and [r.get('frame_id') for r in observations] == frames
        and all(r.get('card_id') == identity['card_id'] for r in observations)
        and resource.get('accepted') is True and resource.get('frame_ids') == frames
        and preview.get('source') == 'native_hif_card_effect_preview'
        and preview.get('selected_slot') == slot
        and resource.get('conservation') == dict(
            stamina=True, shield=True, declared_cost=True, modeled_card_preview=True)
        and all(resource.get('normalized_'+key) == preview.get('baseline_'+key)
            and resource.get('preview_'+key) == preview.get('preview_'+key) for key in ('stamina', 'shield'))
        and resource.get('card_face_cost') == resource.get('displayed_effective_cost')
            == preview.get('displayed_card_cost', preview.get('effective_cost'))
        and resource.get('effective_payment_cost') == preview.get('effective_cost'))


def _selected_resource_model_stop(row):
    """A dispatched SELECT followed by a read-only, model-mismatched B."""
    from ..hif_live_recovery import closed_observer_record
    try:
        if (row.get('status') != 'STOP' or row.get('error') or row.get('failure_event')
                or row.get('transaction_closed') or not closed_observer_record(row)
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches', ())) != 1
                or row.get('native_navigation_preparation')):
            return False
        report, prefix = row['live_events'][-1], row['live_events'][:-1]
        if any(e.get('event') not in ('live_hand_projection', 'live_preinput_state')
                or e.get('accepted') is not True or e.get('executed') is not False
                or any(e.get(k, 0) != 0 for k in ('execution_submit_count',
                    'execution_selection_click_count', 'total_input_click_count', 'drinks_consumed')) for e in prefix):
            return False
        failures, steps = report.get('failure_events', ()), report.get('execution_selection_events', ())
        if (report.get('event') != 'live_selection_stop' or report.get('runtime_phase') != 'authoritative_refresh_b'
                or report.get('executed') is not False or report.get('execution_selection_click_count') != 1
                or any(report.get(k, 0) != 0 for k in ('execution_submit_count', 'end_turn_submit_count',
                    'identity_probe_click_count', 'drinks_consumed'))
                or any(e.get('phase') != 'resource_baseline_reobservation' or e.get('input_click_count') != 0
                    for e in report.get('foreground_preparation_events', ()))
                or len(failures) != 1 or not steps):
            return False
        failure, selection = failures[0], steps[0]
        slot, frames = failure.get('expected_selected_slot'), failure.get('frame_ids')
        resource = failure.get('resource_transaction', {})
        # The legacy B exception flag includes the known SELECT. Its explicit
        # read-only phase and the sole dispatch below prove no business input.
        if (type(slot) is not int or not 0 <= slot < 5 or not _frames(frames)
                or failure.get('phase') != 'authoritative_refresh_b' or failure.get('click_dispatched') is not False
                or failure.get('geometry_stable') is not True or failure.get('selected_identity_authorized') is not True
                or not failure.get('identity_revalidation')
                or not all(v.get('accepted') is True for v in failure['identity_revalidation'])
                or failure.get('select_consensus', {}).get('observations') != [[slot]] * 3
                or resource.get('accepted') is not False or resource.get('selected_slot') != slot
                or resource.get('frame_ids') != frames or not resource.get('card_effect_preview')
                or resource.get('conservation') != dict(stamina=True, shield=True,
                    declared_cost=True, modeled_card_preview=False)
                or any(selection.get(k) != v for k, v in dict(phase='execution_selection',
                    accepted=True, attempted_slot=slot, click_dispatched=True,
                    reason='target_selection_dispatched').items())
                or any(s.get('click_dispatched') is not False or s.get('input_may_have_been_dispatched') is True
                    or s.get('phase') not in ('hif_selected_single_plus', 'hif_selected_printed_cost') for s in steps[1:])):
            return False
        from ..hif_visual import hif_selection_click_box
        from .native_inprocess_resume import _same_input_host
        from .native_settlement_resume import _verified_endpoints

        first = report['authoritative_refresh_events'][0]
        box = hif_selection_click_box(first['candidate_boxes'][-1], slot)
        guard, dispatch = row['native_input_preparation'], row['native_input_dispatches'][0]
        if guard.get('dispatch_state') != 'dispatch_attempted':
            return False
        for target in (guard, dispatch):
            if target.get('kind') is not None or (target.get('x'), target.get('y')) != (box[0]+box[2]//2, box[1]+box[3]//2):
                return False
            _verified_endpoints(target['endpoints'], row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'], target['endpoints'])
        return True
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


def is_unsubmitted_resource_preview_stop(row):
    from ..hif_live_recovery import closed_observer_record
    from .native_selected_cost_resume import selected_cost_confidence_stop
    from .native_resource_noise_resume import selected_noise_stop

    if (selected_noise_stop(row) or selected_cost_confidence_stop(row)
            or selected_resource_family_stop(row) or _selected_resource_model_stop(row)):
        return True
    try:
        events = row['live_events']
        if (row.get('status') != 'STOP' or row.get('error') or row.get('transaction_closed')
                or not closed_observer_record(row) or not row.get('native_selected_source')
                or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
                or row.get('native_input_preparation') or row.get('native_navigation_preparation')
                or len(events) != 1):
            return False
        event = events[0]
        failures = event.get('failure_events', ())
        if (event.get('event') != 'live_selection_stop'
                or event.get('executed') is not False or event.get('input_may_have_been_dispatched') is not False
                or any(event.get(k, 0) for k in ('execution_submit_count', 'execution_selection_click_count',
                    'identity_probe_click_count', 'end_turn_submit_count', 'drinks_consumed'))
                or len(failures) != 1):
            return False
        failure = failures[0]
        if event.get('runtime_phase') == 'resource_baseline':
            from ..opening_observer import selected_baseline_noise_reobserve_allowed
            if (event.get('authoritative_refresh_events') != []
                    or event.get('execution_selection_events') != []
                    or event.get('foreground_preparation_events')):
                return False
            if _repaired_selected_baseline_geometry(row,event) is not None:
                return True
            if _raw_selected_resource_baseline(failure):
                return True
            resource = failure.get('resource_transaction') or {}
            preview = resource.get('card_effect_preview') or {}
            slot, frames = failure.get('selected_slot'), failure.get('frame_ids')
            consensus = failure.get('select_consensus') or {}
            # An observed, unsubmitted selected baseline can always be freshly
            # revalidated. Its diagnostic text and old confidence do not name
            # an authorization class; the normal loader rechecks the source.
            if (failure.get('accepted') is True and failure.get('submission_authorized') is False
                    and failure.get('geometry_stable') is True
                    and failure.get('selected_geometry_authorized') is True
                    and type(slot) is int and 0 <= slot < 5 and _frames(frames)
                    and consensus.get('accepted') is True and consensus.get('selected_slot') == slot
                    and consensus.get('observations') == [[slot]] * 3
                    and resource.get('accepted') is True and resource.get('selected_slot') == slot
                    and resource.get('frame_ids') == frames
                    and preview.get('source') == 'native_hif_card_effect_preview'
                    and preview.get('selected_slot') == slot
                    and resource.get('conservation') == dict(
                        stamina=True, shield=True, declared_cost=True, modeled_card_preview=True)
                    and all(failure.get(key) == resource.get('normalized_'+key) == preview.get('baseline_'+key)
                        and resource.get('preview_'+key) == preview.get('preview_'+key) for key in ('stamina', 'shield'))
                    and resource.get('card_face_cost') == resource.get('displayed_effective_cost')
                        == preview.get('displayed_card_cost', preview.get('effective_cost'))
                    and resource.get('effective_payment_cost') == preview.get('effective_cost')):
                return True
            return (event.get('reason') == 'gray singleton preview: ordinary selected card face, payment, or typed resource preview changed'
                and event.get('authoritative_refresh_events') == [] and event.get('execution_selection_events') == []
                and not event.get('foreground_preparation_events')
                and selected_baseline_noise_reobserve_allowed(failure,row['formal_params']))
        if event.get('reason') not in RESOURCE_PREVIEW_STOPS:
            return False
        resource = failure.get('resource_transaction') or {}
        return (_resource_preview_read_only_prefix(event, failure)
            and failure.get('reason') == event['reason'] and failure.get('accepted') is False
            and failure.get('click_dispatched') is False
            and failure.get('geometry_stable') is True and failure.get('selected_identity_authorized') is True
            and resource.get('accepted') is True and resource.get('card_effect_preview') is not None
            and resource.get('frame_ids') == failure.get('frame_ids')
            and len(set(resource.get('frame_ids', ()))) == 3
            and resource.get('selected_slot') == failure.get('expected_selected_slot')
            and resource == failure.get('visual_fields', {}).get('resource_transaction'))
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def require_repaired_resource_preview(checkpoint):
    from types import SimpleNamespace

    from .native_handoff_policy import phase, pending_action
    from ..high_score.live_policy import BoundHifOpeningPolicy
    from .native_selected_cost_resume import selected_cost_confidence_stop, require_repaired_selected_cost
    from .native_resource_noise_resume import selected_noise_stop, require_repaired_selected_noise
    row = checkpoint['row']
    if selected_cost_confidence_stop(row):
        require_repaired_selected_cost(checkpoint)
        return
    if selected_noise_stop(row):
        require_repaired_selected_noise(checkpoint)
        return
    if (not is_unsubmitted_resource_preview_stop(row) or checkpoint.get('selected') is not True
            or checkpoint.get('pending') or phase(checkpoint['policy']) != 'pending_intent'):
        raise ValueError('HIF resource preview repair lacks its zero-input selected source')
    action = pending_action(checkpoint['policy'], checkpoint['state'])
    if _selected_resource_model_stop(row):
        # Re-evaluate the saved independent observations under the repaired
        # current-play contract. Old failed observations remain untouched.
        from .native_resource_preview_replay import verify_selected_resource_model

        verify_selected_resource_model(checkpoint, action)
        return
    if row['live_events'][0].get('runtime_phase') == 'resource_baseline':
        from ..hif_gray_selection import ordinary_selected_resource_preview
        if selected_resource_family_stop(row) or _raw_selected_resource_baseline(
                row['live_events'][0]['failure_events'][0]):
            from ..selection import same_live_candidate_geometry
            from ..hif_gray_selection import ordinary_selected_zero_cost_baseline

            slot = row['live_events'][0]['failure_events'][0]['resource_transaction']['selected_slot']
            if (action.kind.value != 'PLAY_CARD' or action.source_slot != slot
                    or action.card_instance_id != checkpoint['state'].deck.hand[slot].instance_id):
                raise ValueError('HIF ordinary family repair changed its retained pending action')
            ordinary_selected_zero_cost_baseline(checkpoint['bundle'], complete_event=checkpoint['event'],
                journal=row['gray_selection_journal'], evidence=row['live_events'][0]['failure_events'][0],
                geometry_comparator=same_live_candidate_geometry)
            return  # Validates the repair only; the input path still requires fresh A/B.
        geometry = _repaired_selected_baseline_geometry(row,row['live_events'][0])
        if geometry is not None:
            reference,slot = geometry
            if (len(reference) != len(checkpoint['state'].deck.hand)
                    or action.kind.value != 'PLAY_CARD' or action.source_slot != slot
                    or action.card_instance_id != checkpoint['state'].deck.hand[slot].instance_id):
                raise ValueError('HIF baseline geometry repair changed its retained pending action')
            return  # Only authorize fresh observation; no SELECT/resource proof exists yet.
        preview = row['live_events'][0]['failure_events'][0]['resource_transaction']['card_effect_preview']
        expected = ordinary_selected_resource_preview(checkpoint['bundle'],
            complete_event=checkpoint['event'],journal=row['gray_selection_journal'])
        if not _same(preview,expected) or action.source_slot != preview['selected_slot']:
            raise ValueError('HIF baseline reobservation changed its retained state or selected preview')
        failure = row['live_events'][0]['failure_events'][0]
        if failure.get('accepted') is True:
            from ..selection import same_live_candidate_geometry
            from ..hif_gray_selection import ordinary_selected_zero_cost_baseline

            ordinary_selected_zero_cost_baseline(checkpoint['bundle'], complete_event=checkpoint['event'],
                journal=row['gray_selection_journal'], evidence=failure,
                geometry_comparator=same_live_candidate_geometry)
        # The original stop remains in history. Only the existing
        # whole-batch fresh observer may proceed; no old pixel grants input.
        return
    policy = BoundHifOpeningPolicy(checkpoint['bundle'], action_order=checkpoint['event']['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    policy.bind_selected_skill_support(row['gray_selection_journal']['selected_skill_support'])
    for first in row['live_events'][0].get('authoritative_refresh_events', ()):
        fields = first['visual_fields']
        if policy.bind_generated_card_resource_preview(fields['resource_transaction'], fields['score']) is not True:
            raise ValueError('HIF original first resource preview is still unsupported')
    failure = row['live_events'][0]['failure_events'][0]
    fields = failure['visual_fields']
    if policy.bind_generated_card_resource_preview(fields['resource_transaction'], fields['score']) is not True:
        raise ValueError('HIF original resource preview is still unsupported')
