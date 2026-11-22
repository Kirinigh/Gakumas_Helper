"""Resume an untouched accepted opening, without pretending it was a settlement."""
import json
from pathlib import Path

from route.battle_policy import FORMAL_PROFILES

from .native_opening_root import load_native_opening_root


def _metadata_only_inputs(row):
    recovery = row.get('opening_resume_context_recovery') or row.get('context_recovery', {})
    nav, dispatched = recovery.get('inputs', ()), row.get('native_input_dispatches', ())
    if row.get('input_count') == 0 and not dispatched:
        from ..hif_entry_round import verified_entry_receipt
        return verified_entry_receipt(row, recovery)
    return (recovery.get('accepted') is True and recovery.get('returned_to_live') is True
        and row.get('input_count') == len(nav) == len(dispatched) and 0 < len(nav) <= 4
        and all(item.get('succeeded') is True
            and item.get('phase') in ('live', 'menu', 'notebook', 'identity')
            and len(item.get('target', ())) == 4
            and (sent.get('x'), sent.get('y')) == (
                item['target'][0]+item['target'][2]//2, item['target'][1]+item['target'][3]//2)
            for item, sent in zip(nav, dispatched, strict=True)))


def verify_no_business_since_opening(row):
    """Only failed pre-binding observations and the reviewed preview dismissal."""
    if row.get('status') == 'ONE_INPUT_REQUIRES_PAGE_REVIEW':
        box = row.get('target_box', ())
        if (row.get('action_success') is True and row.get('input_count') == 1
                and row.get('dismiss_box') == [340, 300, 40, 40]
                and len(box) == 4 and 111 <= box[0] and 590 <= box[1]
                and box[0]+box[2] <= 557 and box[1]+box[3] <= 634
                and row.get('verified_input_processes')):
            return
        raise ValueError('opening continuity is not a verified preview cancellation')
    events = row.get('live_events', ())
    if row.get('status') == 'STOP' and not events and _metadata_only_inputs(row):
        return
    if (row.get('status') != 'STOP' or not events
            or events[-1].get('event') != 'live_opening_state'
            or events[-1].get('accepted') is not False
            or any(e.get('executed') is not False or e.get('execution_submit_count') != 0
                or e.get('end_turn_submit_count', 0) for e in events)
            or any(row.get(k) for k in ('choice_journal', 'gray_selection_journal', 'settlement',
                'native_pending_source', 'native_selected_source', 'native_resume_source'))):
        raise ValueError('opening continuity includes a possible business transaction')


def verify_unsubmitted_opening_source(source, bundle, *, allow_saved_intent=False):
    """Reject any selected, submitted or uncertain first-decision boundary."""
    from .native_opening_intent_resume import is_original_saved_opening_intent
    from .native_opening_mapping_resume import is_unsubmitted_mapping_stop

    events = source.get('live_events', ())
    initial_joint_stopped = (events and events[-1].get('event') == 'live_joint_action'
        and events[-1].get('accepted') is False and _metadata_only_inputs(source)
        and all(t.get('input_click_count') == 0 for t in events[-1].get('transactions', ())))
    if (source.get('formal_params', {}).get('hif_policy_profile') not in FORMAL_PROFILES
            or source.get('status') != 'STOP' or bundle.state.turn != 1 or bundle.state.score != 0
            or bundle.state.deck.discard or bundle.state.deck.removed
            or any(e.get('executed', False) is not False or e.get('execution_submit_count') != 0
                or e.get('end_turn_submit_count', 0) for e in events)
            or not (initial_joint_stopped or is_unsubmitted_mapping_stop(source)
                or allow_saved_intent and is_original_saved_opening_intent(source) or (
                events[-1].get('event') in ('live_selection_stop', 'live_decision_exception_stop')
                and events[-1].get('execution_selection_click_count') == 0
                and events[-1].get('input_may_have_been_dispatched') is False))
            or any(source.get(k) for k in ('choice_journal', 'gray_selection_journal'))):
        raise ValueError('opening source is not an untouched first decision')


def load_unsubmitted_opening(runtime, name, continuity_names, module, *, game, attempt_name):
    from .native_settlement_resume import load_result

    source = load_result(runtime, name)
    bundle, opening = load_native_opening_root(source, module)
    verify_unsubmitted_opening_source(source, bundle)
    if (not isinstance(continuity_names, list)
            or len(set(continuity_names)) != len(continuity_names)):
        raise ValueError('opening continuity records are incomplete')
    start = int(name.split('-', 1)[0])
    ids = [int(value.split('-', 1)[0]) for value in continuity_names]
    end = ids[-1] if ids else start
    if (ids != list(range(start+1, end+1))
            or int(attempt_name.split('-', 1)[0]) != end+1):
        raise ValueError('opening continuity has an unreviewed attempt gap')
    # The explicit receipt list must cover every completed attempt since the
    # source. A later action cannot be hidden by selecting older zero-input rows.
    files = [p for p in Path(runtime).glob('result-*.json')
        if p.stem[7:].split('-', 1)[0].isdigit()
        and int(p.stem[7:].split('-', 1)[0]) > start and p.stem[7:] != attempt_name]
    if {p.stem[7:] for p in files} != set(continuity_names):
        raise ValueError('opening continuity omits a later completed attempt')
    records = [json.loads((Path(runtime)/f'result-{value}.json').read_text(encoding='utf8'))
        for value in continuity_names]
    for row in (source, *records):
        previous = row.get('endpoints', {}).get('game', {})
        if any(previous.get(k) != game.get(k) for k in ('pid', 'client_size')):
            raise ValueError('opening continuity game process or window changed')
    for row in records:
        verify_no_business_since_opening(row)
    return bundle, opening, source
