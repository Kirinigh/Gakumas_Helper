"""One observed retrieval selection and confirmation through the Maa input gate."""
import math
import time
from dataclasses import asdict

from .high_score.choice import hif_choice_settlements


def _target(panel, action):
    targets = [row for row in panel['candidates'] if row['instance_id'] == action.card_instance_id
               and row['slot'] == action.source_slot]
    physical = [(instance, slot) for instance, slot in panel['card_slots']
                if instance == action.card_instance_id or slot == action.source_slot]
    if (panel.get('accepted') is not True or len(targets) != 1
            or physical != [(action.card_instance_id, action.source_slot)]):
        raise ValueError('HIF choice lacks one observed physical target')
    return targets[0]


def choice_target_binding(panel, action):
    """Target instance, physical slot and variant; its current box may move."""
    return {key: value for key, value in _target(panel, action).items() if key != 'box'}


def _input_target(before, current, action, upstream):
    target = _target(current, action)
    if upstream:
        same = choice_target_binding(before, action) == choice_target_binding(current, action)
    else:
        same = before['card_slots'] == current['card_slots'] and before['candidates'] == current['candidates']
    if not same:
        raise ValueError('HIF choice candidate binding changed before input')
    return target


def _selected_instance(panel):
    count, instance = panel.get('selected_count'), panel.get('selected_instance_id')
    if ((count == 0 and panel.get('confirm_enabled') is False and instance is None)
            or (count == 1 and panel.get('confirm_enabled') is True and instance in dict(panel['card_slots']))):
        return instance
    raise ValueError('HIF choice lacks a consistent observed selection state')


def _dispatch(context, box, captured_at, record, journal, stage):
    from utils.maa_input import verify_maa_input_endpoints

    controller = context.tasker.controller
    endpoints = verify_maa_input_endpoints(controller.info,
        expected_client_size=getattr(context, 'gkh_hif_client_size', (720, 1280)))
    record.update(box=list(box), endpoints=endpoints, input_may_have_been_dispatched=True)
    journal(stage + '_dispatch_pending', {'choice': record})

    def record_no_dispatch():
        record.update(input_may_have_been_dispatched=False, choice_submit_count=0)
        journal('choice_selected' if stage == 'choice_confirm' else 'pending_ui_captured', {'choice': record})

    count = getattr(context, 'gkh_hif_upstream_dispatch_count', None)
    dispatches_before = count() if callable(count) else None
    elapsed = (time.perf_counter()-captured_at)*1000
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    if not math.isfinite(elapsed) or elapsed < 0 or (not upstream and elapsed > 1000):
        record_no_dispatch()
        raise ValueError('HIF choice input frame expired after recording')
    record['input_evidence_age_ms'] = elapsed
    x, y, width, height = box
    try:
        job = controller.post_click(x+width//2, y+height//2)
    except Exception:
        # The formal host increments its native count immediately before the SDK call.
        if dispatches_before is not None and count() == dispatches_before:
            record_no_dispatch()
        raise
    if not job.wait().succeeded:
        raise ValueError('HIF choice Maa input failed; do not repeat without observation')


def _capture_input_frame(capture_frame):
    raw = capture_frame()
    if isinstance(raw, tuple) and len(raw) == 2 and type(raw[1]) in (int, float):
        return raw
    return raw, time.perf_counter()


def select_hif_choice_card(context, projected, panel, action, *, observe_panel, capture_frame, journal):
    """Switch to the bound card, or retain an already selected target, without confirming."""
    hif_choice_settlements(projected, action, card_slots=panel['card_slots'])
    _target(panel, action)
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    previous_selection = None if upstream else _selected_instance(panel)
    image, captured_at = _capture_input_frame(capture_frame)
    fresh = observe_panel((image,), single_frame=True)
    target = _input_target(panel, fresh, action, upstream)
    selected_instance = (None if upstream and fresh.get('observed_target_slot') == action.source_slot
        and fresh.get('selected_slot') is None else _selected_instance(fresh))
    if not upstream and selected_instance != previous_selection:
        raise ValueError('HIF choice candidates changed before selection')
    record = dict(action=asdict(action), card_slots=panel['card_slots'], before_panel=panel,
        choice_submit_count=0, selection_click_count=0)
    if selected_instance == action.card_instance_id:
        record['already_selected'] = True
        journal('choice_selected', {'choice': record})
        return record
    _dispatch(context, target['box'], captured_at, record, journal, 'choice_select')
    record['selection_click_count'] = 1
    journal('choice_selected', {'choice': record})
    return record


def confirm_hif_choice_card(context, projected, panel, action, *, observe_panel, capture_frame,
                            journal, choice_record, observe_settlement, baseline):
    """Confirm once only after the selected card and enabled button are observed."""
    if (choice_record.get('choice_confirm_started') or choice_record.get('confirmation_submitted')
            or choice_record.get('choice_submit_count', 0)):
        raise ValueError('HIF choice confirmation was already attempted; recover without repeating input')
    expected = hif_choice_settlements(projected, action, card_slots=panel['card_slots'])
    target = _target(panel, action)
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    if upstream:
        _input_target(choice_record['before_panel'], panel, action, upstream)
    if (panel.get('selected_count') != 1 or panel.get('confirm_enabled') is not True
            or panel.get('selected_instance_id') != target['instance_id']):
        raise ValueError('HIF choice confirmation lacks its exact selected card')
    image, captured_at = _capture_input_frame(capture_frame)
    fresh = observe_panel((image,), single_frame=True)
    _input_target(panel, fresh, action, upstream)
    if (fresh.get('selected_count') != 1
            or fresh.get('confirm_enabled') is not True
            or fresh.get('selected_instance_id') != target['instance_id']
            or (not upstream and fresh['confirm_target'] != panel['confirm_target'])):
        raise ValueError('HIF choice selection changed before confirmation')
    confirmation = dict(action=asdict(action), card_slots=panel['card_slots'],
        before_panel=choice_record['before_panel'], selected_panel=panel,
        choice_submit_count=1, confirmation_submitted=False, input_panel=fresh['pages'][0])
    # Re-entrant calls after an uncertain dispatch must recover from the
    # journal and observed UI, not send the same confirmation again.
    choice_record['choice_confirm_started'] = True
    try:
        _dispatch(context, fresh['confirm_target'], captured_at, confirmation, journal, 'choice_confirm')
    finally:
        choice_record['choice_confirm_started'] = confirmation.get('input_may_have_been_dispatched') is True
    confirmation['confirmation_submitted'] = True
    choice_record['confirmation_submitted'] = True
    journal('choice_submitted', {'choice': confirmation})
    if upstream:
        from .hif_upstream_ui import wait_after_confirmed_move
        wait_after_confirmed_move(context)
    previous = getattr(context, 'gkh_hif_confirmed_choice', None)
    context.gkh_hif_confirmed_choice = confirmation
    try:
        settlement = observe_settlement(expected, baseline, projected.action)
    finally:
        context.gkh_hif_confirmed_choice = previous
    return dict(event='live_choice_transaction', accepted=settlement.get('accepted') is True,
        execution_submit_count=0, choice_submit_count=1, choice=confirmation,
        settlement=settlement, reason=settlement.get('reason'))
