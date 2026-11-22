"""Replay one D-047 title receipt against its original, audited card input."""

from .native_card_evidence import _same, _fresh, _frames, _number, dispatched_card_refreshes


def recorded_single_plus_estimate(bundle, action, report, *, source_row):
    """Call after decoded_dispatched_action; never promote a zero-submit receipt."""
    from live_decision.hif_single_plus import verify_selected_single_plus_estimate

    stages = report.get('execution_selection_events', ())
    title_steps = [(index, event) for index, event in enumerate(stages)
                   if event.get('phase') == 'hif_selected_single_plus']
    before = [(index, event) for index, event in title_steps
              if (event.get('receipt') or {}).get('single_plus_estimate') is True]
    after = [(index, event) for index, event in enumerate(stages)
             if event.get('phase') == 'hif_single_plus_submitted']
    receipt = report.get('single_plus_estimate')
    if (receipt is None and not before and not after
            and (not title_steps or report.get('selected_skill_support') is None)):
        return None
    submits = [(index, event) for index, event in enumerate(stages)
               if event.get('phase') == 'execution_submit' and event.get('click_dispatched') is True]
    superseded = receipt is None and report.get('selected_skill_support') is not None
    if (type(report.get('execution_submit_count')) is not int
            or report['execution_submit_count'] != 1
            or len(before) != 1 or len(submits) != 1
            or not before[0][0] < submits[0][0]
            or submits[0][1].get('attempted_slot') != action.source_slot
            or not superseded and (len(after) != 1 or submits[0][0] >= after[0][0]
                                   or not _same(after[0][1].get('receipt'), receipt))):
        raise ValueError('native single-plus receipt lacks its original one-card submission')
    preinput = verify_selected_single_plus_estimate(before[0][1].get('receipt'),
        bundle=bundle, action=action, require_submitted=False)
    if superseded:
        from types import SimpleNamespace

        from live_decision.high_score.live_policy import BoundHifOpeningPolicy

        from .native_selected_support import bind_recorded_selected_support

        exact = [(index, event) for index, event in enumerate(stages)
                 if event.get('phase') == 'hif_selected_skill_support']
        if (after or len(title_steps) != 1
                or len(exact) != 1 or not before[0][0] < exact[0][0] < submits[0][0]
                or before[0][1].get('click_dispatched') is not False
                or exact[0][1].get('click_dispatched') is not False
                or not _same(before[0][1].get('receipt'), preinput)):
            raise ValueError('native submitted exact support lacks its original title probe')
        policy = BoundHifOpeningPolicy(bundle, action_order=())
        policy.response = SimpleNamespace(action=action)
        if not bind_recorded_selected_support(policy, report, require_submitted=True, source_row=source_row):
            raise ValueError('native submitted exact support was not independently verified')
        verified = policy.selected_skill_support
        if (verified['frame_ids'] != preinput['frame_ids']
                or verified['raw_title'] != preinput['raw_title']
                or not _same(verified['selection_evidence'], preinput['selection_evidence'])):
            raise ValueError('native submitted exact support replaced a different title probe')
    else:
        verified = verify_selected_single_plus_estimate(receipt, bundle=bundle, action=action,
                                                        require_submitted=True)
        # This comparison verifies provenance only. The returned submitted count
        # already came from the submitted receipt and is never manufactured here.
        if not _same(dict(preinput, submit_count=verified['submit_count']), verified):
            raise ValueError('native single-plus title changed between selection and submission')

    first, final = dispatched_card_refreshes(report, row=source_row, bundle=bundle)
    slot = action.source_slot
    if (first.get('phase') != 'authoritative_refresh_a'
            or final.get('phase') != 'authoritative_refresh_b'
            or final.get('expected_selected_slot') != slot
            or _frames(first) & _frames(final)
            or set(verified['frame_ids']) & _frames(final)):
        raise ValueError('native single-plus input lacks an independent selected final B')
    _fresh(first, slot)
    _fresh(final, slot)
    authorized = submits[0][1].get('authorization_floor')
    preparation = source_row.get('native_input_preparation') or {}
    checked = preparation.get('authorization_checked_at')
    if (not _number(authorized) or not _number(checked)
            or not final['authorization_floor'] <= authorized <= checked < final['input_deadline']
            or preparation.get('dispatch_state') != 'dispatch_attempted'
            or preparation.get('input_deadline') != final['input_deadline']
            or preparation.get('authorization_floor') != authorized):
        raise ValueError('native single-plus input exceeded its original deadline')
    boxes = final.get('candidate_boxes', ())
    if (len(boxes) != 3 or any(len(frame) <= slot for frame in boxes)
            or any(len(frame[slot]) != 4 or any(type(value) is not int for value in frame[slot])
                   or min(frame[slot][2:]) <= 0 for frame in boxes)):
        raise ValueError('native single-plus final target geometry is incomplete')
    x, y, width, height = boxes[-1][slot]
    center = x + width // 2, y + height // 2
    allowed_targets = {center}
    if len(boxes[-1]) in (4,5):
        from live_decision.hif_visual import hif_selection_click_box

        x,y,width,height = hif_selection_click_box(boxes[-1],slot)
        allowed_targets.add((x+width//2,y+height//2))
    point = preparation.get('x'),preparation.get('y')
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    endpoints = preparation.get('endpoints')
    _verified_endpoints(endpoints, source_row['endpoints']['game']['pid'])
    _same_input_host(source_row['endpoints'], endpoints)
    matches = [entry for entry in source_row.get('native_input_dispatches', ())
               if entry.get('kind') != 'swipe' and (entry.get('x'), entry.get('y')) == point
               and _same(entry.get('endpoints'), endpoints)]
    if len(matches) != 1 or point not in allowed_targets:
        raise ValueError('native single-plus lacks one actual audited final target input')
    # The exact receipt owns the dispatched model. Do not turn its discarded
    # title-only probe into a submitted ordinary-plus estimate during recovery.
    return None if superseded else verified
