"""Read the original A and final B from one audited card submission."""

import json
import math


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def selected_observation_window_ms(maximum_age_ms, selected_slot, *, strong_unselected=False):
    """Bound the stable-state observation separately from final input freshness."""
    if not _number(maximum_age_ms) or not 0 < maximum_age_ms <= 1000:
        raise ValueError('selected observation input age limit is invalid')
    return min(2000., 2 * maximum_age_ms) if (
        type(selected_slot) is int and selected_slot >= 0 or strong_unselected) else maximum_age_ms


def evidence_window_limit(event, maximum_age_ms=1000.):
    """Old records keep their limit; strong card A/B records carry their own."""
    recorded = event.get('observation_window_ms', maximum_age_ms)
    if (not _number(recorded) or recorded <= 0
            or recorded != maximum_age_ms and (
                event.get('phase') not in ('authoritative_refresh_a', 'authoritative_refresh_b')
                or recorded != selected_observation_window_ms(maximum_age_ms, event.get('expected_selected_slot'),
                    strong_unselected=(event.get('hif_policy_profile') == 'hif-strong-reference-trial-v1'
                                       and event.get('phase') == 'authoritative_refresh_a')))):
        raise ValueError('selected observation window is unbound')
    return recorded


def input_window_limit(event, maximum_age_ms=1000.):
    """Explicit strong card A/B records use one two-second input contract."""
    recorded = event.get('input_window_ms', maximum_age_ms)
    if (not _number(recorded) or recorded <= 0
            or recorded != maximum_age_ms and (
                event.get('hif_policy_profile') != 'hif-strong-reference-trial-v1'
                or event.get('phase') not in ('authoritative_refresh_a', 'authoritative_refresh_b')
                or recorded != selected_observation_window_ms(maximum_age_ms, event.get('expected_selected_slot'),
                    strong_unselected=event.get('phase') == 'authoritative_refresh_a')
                or recorded != evidence_window_limit(event, maximum_age_ms))):
        raise ValueError('selected input window is unbound')
    return recorded


def input_evidence_is_fresh(event, maximum_age_ms=1000.):
    """Read each record's actual limits, never reinterpret old one-second evidence."""
    limit = evidence_window_limit(event, maximum_age_ms)
    return all(_number(event.get(key)) and 0 <= event[key] <= maximum for key, maximum in (
        ('capture_age_ms', input_window_limit(event, maximum_age_ms)), ('evidence_window_age_ms', limit)))


def _frames(event):
    values = event.get('frame_ids', ())
    if (not isinstance(values, (list, tuple)) or len(values) != 3
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != 3):
        raise ValueError('card B retry lacks three independent frames')
    return set(values)


def _fresh(event, slot):
    decision = event.get('safe_decision', {})
    if (event.get('accepted') is not True or decision.get('status') != 'ALLOW'
            or decision.get('has_action_intent') is not True
            or decision.get('action') != {'kind': 'PLAY_CARD', 'card_slot': slot}):
        raise ValueError('card B retry accepted input check is incomplete')
    if not input_evidence_is_fresh(event):
        raise ValueError('card B retry accepted input evidence is stale')
    started, authorized, deadline = (event.get(key) for key in
        ('authoritative_capture_started_at', 'authorization_floor', 'input_deadline'))
    if (not all(_number(value) for value in (started, authorized, deadline))
            or not started <= authorized < deadline
            or not 0 < deadline - started <= input_window_limit(event)/1000. + 1e-9):
        raise ValueError('card B retry original input deadline is invalid')


def dispatched_card_refreshes(report, *, row=None, bundle=None):
    """Return existing events, never relabel a rejected B as accepted.

    The historical two-event route retains its consumers' existing checks.
    The three-event route additionally requires the producer's one shared
    retry budget and the host's actual final preparation/dispatch evidence.
    """
    refreshes = report.get('authoritative_refresh_events', ())
    stages = report.get('execution_selection_events', ())
    retries = [(index, event) for index, event in enumerate(stages)
               if event.get('phase') == 'execution_submit_reobservation']
    if len(refreshes) == 2 and not retries:
        return tuple(refreshes)
    if (len(refreshes) != 3 or len(retries) != 1 or row is None
            or [event.get('phase') for event in refreshes] !=
            ['authoritative_refresh_a', 'authoritative_refresh_b', 'authoritative_refresh_b']):
        raise ValueError('card input checks lack one bounded B retry')
    first, previous, final = refreshes
    retry_index, retry = retries[0]
    action = report.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
    slot = action.get('source_slot')
    if (action.get('kind') != 'PLAY_CARD' or type(slot) is not int or slot < 0
            or not isinstance(action.get('card_instance_id'), str) or not action['card_instance_id']
            or type(report.get('execution_submit_count')) is not int
            or report['execution_submit_count'] != 1
            or type(retry.get('retry_count')) is not int or retry['retry_count'] != 1
            or retry.get('attempted_slot') != slot
            or retry.get('click_dispatched') is not False
            or retry.get('input_may_have_been_dispatched') is not False):
        raise ValueError('card B retry count or target is unverified')
    if bundle is not None and (slot >= len(bundle.state.deck.hand)
            or bundle.state.deck.hand[slot].instance_id != action['card_instance_id']):
        raise ValueError('card B retry target differs from the original bound instance')
    selection_count = report.get('execution_selection_click_count')
    selections = [(index, event) for index, event in enumerate(stages)
                  if event.get('phase') == 'execution_selection']
    if (type(selection_count) is not int or selection_count not in (0, 1)
            or any(type(report.get(key)) is not int or report[key] != 0
                   for key in ('selection_click_count', 'identity_probe_click_count'))
            or any(type(report.get(key)) is not int or report[key] != selection_count + 1
                   for key in ('execution_click_count', 'total_input_click_count'))
            or sum(event.get('phase') == 'execution_selection' and event.get('click_dispatched') is True
                   for event in stages) != selection_count):
        raise ValueError('card B retry input accounting is incomplete')
    if (len(selections) != 1 or selections[0][0] >= retry_index
            or selections[0][1].get('accepted') is not True
            or selections[0][1].get('attempted_slot') != slot
            or selections[0][1].get('click_dispatched') is not bool(selection_count)
            or selections[0][1].get('reason') !=
            ('target_selection_dispatched' if selection_count else 'target_already_selected')):
        raise ValueError('card B retry original selection record is incomplete')
    generation = first.get('generation')
    if not isinstance(generation, dict) or not generation:
        raise ValueError('card B retry original A generation is absent')
    seen = set()
    for index, event in enumerate(refreshes):
        frames = _frames(event)
        if frames & seen:
            raise ValueError('card B retry reuses an earlier input frame')
        seen.update(frames)
        proposal = event.get('proposal', {})
        if (not _same(event.get('generation'), generation)
                or event.get('click_dispatched') is not False
                or not _same(proposal.get('candidate_traces', {}).get('high_score', {}).get('action'), action)
                or proposal.get('status') != 'PROPOSE'
                or proposal.get('action') != {'kind': 'PLAY_CARD', 'card_slot': slot}):
            raise ValueError('card B retry changed the original A generation or action')
        if index:
            selection = event.get('select_consensus', {})
            if (not _same(event.get('expected_generation'), generation)
                    or event.get('expected_selected_slot') != slot
                    or selection.get('accepted') is not True or selection.get('selected_slot') != slot
                    or selection.get('observations') != [[slot], [slot], [slot]]):
                raise ValueError('card B retry selected target is not independently unanimous')
    if retry.get('prior_frame_ids') != previous.get('frame_ids'):
        raise ValueError('card B retry does not identify the rejected original B')
    _fresh(first, slot)
    _fresh(final, slot)
    submits = [(index, event) for index, event in enumerate(stages)
               if event.get('phase') == 'execution_submit']
    if (not submits or submits[-1][0] <= retry_index
            or submits[-1][1].get('click_dispatched') is not True
            or submits[-1][1].get('reason') != 'submit_dispatched'
            or any(event.get('attempted_slot') != slot for _, event in submits)):
        raise ValueError('card B retry lacks one final confirmed submission')
    if retry.get('reason') in ('initial_b_stale_without_input', 'initial_b_stale_after_selection'):
        decision = previous.get('safe_decision', {})
        newly_selected = retry['reason'] == 'initial_b_stale_after_selection'
        prior_clicks = [event for event in stages[:retry_index] if event.get('click_dispatched') is True]
        if (len(submits) != 1 or selection_count != int(newly_selected)
                or (not newly_selected and not row.get('native_selected_source'))
                or previous.get('accepted') is not False or previous.get('reason') != 'stale_state'
                or previous.get('click_dispatched') is not False
                or decision.get('status') != 'STOP' or decision.get('action') is not None
                or decision.get('has_action_intent') is not False or decision.get('reason') != 'stale_state'
                or not _number(previous.get('capture_age_ms'))
                or previous['capture_age_ms'] <= input_window_limit(previous)
                or prior_clicks != ([selections[0][1]] if newly_selected else [])
                or any(event.get('input_may_have_been_dispatched') is True for event in stages[:retry_index])):
            raise ValueError('card stale B retry does not prove zero earlier business input')
    elif retry.get('reason') == 'final_preparation_expired_without_dispatch':
        _fresh(previous, slot)
        old = submits[0][1]
        if (len(submits) != 2 or submits[0][0] >= retry_index
                or old.get('reason') != 'snapshot_expired_after_input_preparation'
                or old.get('click_dispatched') is not False
                or old.get('input_may_have_been_dispatched') is not False
                or not _number(old.get('authorization_floor'))
                or not previous['authorization_floor'] <= old['authorization_floor'] < previous['input_deadline']):
            raise ValueError('card expired B retry does not prove zero earlier dispatch')
    else:
        raise ValueError('card B retry reason is unsupported')
    authorized = submits[-1][1].get('authorization_floor')
    preparation = row.get('native_input_preparation') or {}
    checked = preparation.get('authorization_checked_at')
    if (not _number(authorized) or not _number(checked)
            or not final['authorization_floor'] <= authorized <= checked < final['input_deadline']
            or preparation.get('dispatch_state') != 'dispatch_attempted'
            or preparation.get('input_deadline') != final['input_deadline']
            or preparation.get('authorization_floor') != authorized):
        raise ValueError('card B retry final dispatch exceeded its original deadline')
    # The new capture must actually follow the rejected B. Its deadlines are
    # retained verbatim; refreshing evidence never extends an old deadline.
    new_start = final.get('evidence_window_started_at')
    old_floor = previous.get('authorization_floor')
    old_start = previous.get('evidence_window_started_at')
    if (not all(_number(value) for value in (new_start, old_floor, old_start))
            or old_start < first['authorization_floor'] or new_start < old_floor
            or (retry['reason'] == 'final_preparation_expired_without_dispatch'
                and new_start < previous['input_deadline'])):
        raise ValueError('card B retry capture predates its rejected B')
    boxes = final.get('candidate_boxes', ())
    if (len(boxes) != 3 or any(len(frame) <= slot for frame in boxes)
            or any(len(frame[slot]) != 4 or any(type(value) is not int for value in frame[slot])
                   or min(frame[slot][2:]) <= 0 for frame in boxes)):
        raise ValueError('card B retry final target geometry is incomplete')
    x, y, width, height = boxes[-1][slot]
    center = x + width // 2, y + height // 2
    allowed_targets = {center}
    if len(boxes[-1]) in (4,5):
        from live_decision.hif_visual import hif_selection_click_box

        x,y,width,height = hif_selection_click_box(boxes[-1],slot)
        allowed_targets.add((x+width//2,y+height//2))
    point = preparation.get('x'),preparation.get('y')
    matches = [entry for entry in row.get('native_input_dispatches', ())
               if entry.get('kind') != 'swipe' and (entry.get('x'), entry.get('y')) == point
               and _same(entry.get('endpoints'), preparation.get('endpoints'))]
    if (len(matches) != 1 or point not in allowed_targets
            or not isinstance(preparation.get('endpoints'), dict) or not preparation['endpoints']):
        raise ValueError('card B retry lacks one actual audited final target input')
    return first, final
