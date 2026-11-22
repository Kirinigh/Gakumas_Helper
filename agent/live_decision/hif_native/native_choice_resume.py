"""Recover a dispatched retrieval drink from its native attempt journal."""
import json
import math
from dataclasses import asdict

from . import diagnostic_depth_reached
from .native_settlement_resume import (
    exact_state,
    load_result,
    _verified_endpoints,
    load_verified_source,
    decoded_dispatched_action,
    same_endpoint_observation,
)


def current_choice_panel(panel):
    if not isinstance(panel, dict):
        return False
    frames = panel.get('frame_ids')
    return (isinstance(frames, (list, tuple))
            and len(frames) in ((1, 2) if panel.get('selected_count') == 1 else (1,))
            and all(isinstance(frame, str) and frame for frame in frames)
            and len(set(frames)) == len(frames))


def require_current_choice_evidence(journal):
    choice = journal.get('choice') or {}
    for panel in (journal.get('choice_panel'), choice.get('before_panel'), choice.get('selected_panel')):
        if panel is not None and not current_choice_panel(panel):
            raise ValueError('retrieval resume requires a current page or its selected-title observations')
    if any(row.get('source') != 'miyu_move_title' or len(row.get('frame_ids', ())) not in (1, 2)
           for row in journal.get('detail_evidence', ())):
        raise ValueError('retrieval resume requires current selected-title evidence')


def has_unconfirmed_choice_selection(journal):
    choice = journal.get('choice') or {}
    return (journal.get('stage') == 'choice_selected' and bool(choice.get('action'))
        and choice.get('choice_submit_count') == 0 and not choice.get('choice_confirm_started')
        and not choice.get('confirmation_submitted'))


def has_confirmed_choice(journal):
    choice = journal.get('choice') or {}
    return (journal.get('stage') in ('choice_submitted', 'settled')
        and choice.get('confirmation_submitted') is True and choice.get('input_may_have_been_dispatched') is True
        and type(choice.get('choice_submit_count')) is int and choice['choice_submit_count'] == 1)


def is_confirmed_retrieval_stop(row):
    from ..hif_live_recovery import closed_observer_record

    journal = row.get('choice_journal') or {}
    try:
        require_current_choice_evidence(journal)
    except (ValueError, TypeError):
        return False
    return (row.get('status') == 'STOP' and not row.get('transaction_closed')
        and row.get('formal_params', {}).get('hif_policy_profile') == 'hif-strong-reference-trial-v1'
        and journal.get('stage') == 'choice_submitted' and has_confirmed_choice(journal)
        and closed_observer_record(row))


def is_pending_retrieval_stop(row, *, runtime=None):
    """Dispatch journal, not a failure reason, identifies this pending boundary."""
    from ..hif_live_recovery import closed_observer_record

    journal = row.get('choice_journal') or {}
    try:
        require_current_choice_evidence(journal)
        return bool(row.get('status') == 'STOP'
            and closed_observer_record(row)
            and row.get('transaction_closed') is not True
            and journal.get('stage') in ('drink_submitted', 'pending_ui_captured', 'choice_selected')
            and not journal.get('choice', {}).get('confirmation_submitted')
            and not journal.get('choice', {}).get('choice_confirm_started')
            and not journal.get('choice', {}).get('choice_submit_count')
            and (journal['stage'] != 'choice_selected' or has_unconfirmed_choice_selection(journal))
            and journal.get('action', {}).get('kind') == 'USE_DRINK'
            and journal['action'].get('drink_id') == '22'
            and journal.get('joint_event', {}).get('execution_submit_count') == 1
            and sum(t.get('use_submitted') is True for t in journal['joint_event'].get('transactions', ())) == 1)
    except (ValueError, KeyError, TypeError):
        return False


def load_choice_source(runtime, name, module, trail=(), *, allow_closed_transaction=False):
    from live_decision.high_score.choice import verify_hif_pending_choice

    from .native_support_resume import restore_native_history

    if name in trail or diagnostic_depth_reached(trail):
        raise ValueError('retrieval choice recovery chain is cyclic or unbounded')
    row = (load_result(runtime, name, allow_closed_transaction=True)
        if allow_closed_transaction else load_result(runtime, name))
    journal = row.get('choice_journal', {})
    require_current_choice_evidence(journal)
    if journal.get('stage') not in ('drink_submitted', 'pending_ui_captured', 'choice_selected', 'choice_submitted', 'settled'):
        raise ValueError('retrieval drink does not have a completed native dispatch; never resubmit')
    if row.get('native_choice_source'):
        projected, baseline, parent = load_choice_source(runtime, row['native_choice_source'], module, (*trail, name),
            allow_closed_transaction=allow_closed_transaction)
        if (row['observed_live_context'] != parent['observed_live_context']
                or row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid']
                or any(journal.get(key) != parent['choice_journal'].get(key)
                       for key in ('joint_event', 'action', 'run_id', 'revision'))):
            raise ValueError('retrieval continuation changed its original drink evidence')
        previous_choice = parent['choice_journal'].get('choice', {})
        if previous_choice.get('confirmation_submitted') is True and journal.get('choice') != previous_choice:
            raise ValueError('retrieval continuation changed an already confirmed selection')
        if any(e.get('execution_submit_count', 0) or e.get('end_turn_submit_count', 0) for e in row.get('live_events', ())):
            raise ValueError('retrieval continuation contains another business action')
        verified = verify_hif_pending_choice(projected.source_bundle, projected.action,
            exact_state(journal['pending_state']))
        return verified, baseline, row
    run = module.HifLiveRunContext(**row['observed_live_context'])
    from .native_opening_root import has_native_opening_root, load_native_opening_root, verify_native_round_context
    inventory_source = None
    if has_native_opening_root(row):
        bundle, baseline = load_native_opening_root(row, module)
    elif row.get('native_inprocess_source'):
        from .native_inprocess_resume import restore_inprocess_source
        bundle, baseline = restore_inprocess_source(runtime, row, module, trail)
    elif row.get('native_rule_resume_source'):
        from .native_rule_progress_resume import restore_rule_source
        bundle, baseline = restore_rule_source(runtime, row, module, trail)
    else:
        bundle, state, saved, parent = load_verified_source(runtime, row['native_resume_source'], module)
        inventory_source = (bundle, state, saved, parent['id'])
        verify_native_round_context(row, module, parent)
        bundle, _, baseline = restore_native_history(runtime, module, bundle, state, settlement=saved,
            reference=row['resume_reference'], current=row['resume_history'],
            old_run=module.HifLiveRunContext(**parent['observed_live_context']), new_run=run,
            old_pid=parent['endpoints']['game']['pid'], new_pid=row['endpoints']['game']['pid'])
    from .native_hand_support import bind_planning_hand_support
    from .native_preinput_history import verify_preinput_history
    bundle = bind_planning_hand_support(bundle, row, runtime=runtime, module=module,
                                        trail=trail, inventory_source=inventory_source)
    if row.get('information_boundary_execution') is True:
        from .native_rule_progress_resume import verify_rule_preinput
        baseline = verify_rule_preinput(row, bundle, baseline)
    else:
        verify_preinput_history(row, bundle, baseline)
    action, event = decoded_dispatched_action(dict(row, live_events=[journal['joint_event']]), bundle)
    if not _same(journal.get('action'), asdict(action)):
        raise ValueError('retrieval journal action differs from the native drink dispatch')
    if (journal.get('run_id'), journal.get('revision')) != (run.run_id, run.revision):
        raise ValueError('retrieval journal is not bound to its original run')
    projected = verify_hif_pending_choice(bundle, action, exact_state(journal['pending_state']))
    if event.get('accepted') is True:
        raise ValueError('retrieval source is already completely settled')
    return projected, baseline, row


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _panel(projected, panel, *, selected, action):
    from live_decision.high_score.choice import build_hif_choice_request
    from live_decision.high_score.contracts import CapabilityReport

    preplanned = panel.get('target_instance_only') is True
    target_only = (selected or preplanned) and panel.get('observed_target_slot') == action.source_slot
    other_selected = (preplanned and not selected and panel.get('selected_count') == 1
                      and panel.get('selected_slot') is None and panel.get('selected_instance_id') is None)
    if (panel.get('accepted') is not True or not current_choice_panel(panel)
            or panel.get('title') != '移動' or panel.get('selection_limit') != 1
            or panel.get('selected_count') != int(selected or other_selected)
            or panel.get('confirm_enabled') is not (selected or other_selected)):
        raise ValueError('retrieval confirmation lacks a complete observed selection panel')
    slots = tuple(tuple(pair) for pair in panel.get('card_slots', ()))
    if preplanned and slots != ((action.card_instance_id, action.source_slot),):
        raise ValueError('retrieval panel differs from its bound target')
    build_hif_choice_request(projected, card_slots=slots, target_only=preplanned,
        capabilities=CapabilityReport(False, False, False, False, False), request_id='archived-choice-panel')
    candidates = panel.get('candidates', ())
    if len(candidates) != len(slots) or {(c.get('instance_id'), c.get('slot')) for c in candidates} != set(slots):
        raise ValueError('retrieval panel candidates differ from the complete instance binding')
    refs = {ref.instance_id: ref for ref in projected.pending_state.deck.all_cards}
    zones = {ref.instance_id: name for name in ('draw_pile', 'discard')
             for ref in getattr(projected.pending_state.deck, name)}
    for candidate in candidates:
        instance = candidate['instance_id']
        if target_only and instance != action.card_instance_id:
            continue  # Retained strategy candidates were not reread during confirmation.
        ref = refs[instance]
        definition = projected.source_bundle.kernel.cards[ref.card_id]
        if (candidate.get('card_id') not in (ref.card_id, definition.canonical_card_id or ref.card_id)
                or candidate.get('source_zone') != zones.get(instance, '')):
            raise ValueError('retrieval observed candidate identity or source zone differs')
    if selected and (panel.get('selected_instance_id') != action.card_instance_id
            or type(panel.get('selected_slot')) is not int
            or panel['selected_slot'] != dict(slots).get(action.card_instance_id)
            or panel['selected_slot'] != action.source_slot):
        raise ValueError('retrieval selected panel identifies a different card or physical slot')
    if not selected and (panel.get('selected_instance_id') is not None or panel.get('selected_slot') is not None):
        raise ValueError('retrieval unselected panel contains a selected candidate')
    return slots


def _confirmed_outcomes(runtime, projected, row):
    from live_decision.high_score.choice import hif_choice_settlements
    from live_decision.hif_choice_execution import choice_target_binding
    from live_decision.high_score.contracts import ActionKind, GameAction

    journal = row['choice_journal']
    choice = journal.get('choice', {})
    if not has_confirmed_choice(journal):
        raise ValueError('retrieval choice is not confirmed; never repeat the selection blindly')
    raw = dict(choice['action'])
    raw['kind'] = ActionKind(raw['kind'])
    action = GameAction(**raw)
    before, selected = choice['before_panel'], choice['selected_panel']
    count = before.get('selected_count')
    if type(count) is not int or count not in (0, 1):
        raise ValueError('retrieval starting panel is neither unselected nor the exact selected candidate')
    identified_selection = count == 1 and (before.get('target_instance_only') is not True
                                            or before.get('selected_instance_id') is not None)
    before_action = (GameAction(ActionKind.MOVE_CARD, card_instance_id=before.get('selected_instance_id'),
                               source_slot=before.get('selected_slot')) if identified_selection else action)
    slots = _panel(projected, before, selected=identified_selection, action=before_action)
    _panel(projected, selected, selected=True, action=action)
    page = choice.get('input_panel')
    if (not isinstance(page, dict) or page.get('confirm_enabled') is not True
            or type(page.get('selected_count')) is not int or page['selected_count'] != 1
            or type(page.get('selected_slot')) is not int or page['selected_slot'] != action.source_slot):
        raise ValueError('retrieval confirmation lacks its current selected target and enabled button')
    current_button = page.get('confirm_target')
    if (not _same(choice['card_slots'], slots)
            or not _same(choice_target_binding(before, action), choice_target_binding(selected, action))):
        raise ValueError('retrieval candidate binding changed before confirmation')
    age = choice.get('input_evidence_age_ms')
    if type(age) not in (int, float) or not math.isfinite(age) or age < 0:
        raise ValueError('retrieval confirmation has no observed input age')
    box = choice.get('box', ())
    if (len(box) != 4 or any(type(value) is not int for value in box) or min(box[:2]) < 0
            or min(box[2:]) <= 0 or not _same(box, current_button)):
        raise ValueError('retrieval confirmation target was not the observed enabled button')
    game_pid = row['endpoints']['game']['pid']
    endpoints = choice.get('endpoints')
    _verified_endpoints(endpoints, game_pid)
    x, y, width, height = box
    owners, visited, current = [], set(), row
    while True:
        name = current['id']
        if name in visited or len(visited) >= 20:
            raise ValueError('retrieval confirmation ancestry is cyclic or unbounded')
        visited.add(name)
        matches = [entry for entry in current.get('native_input_dispatches', ())
                   if entry.get('kind') != 'swipe'
                   and (entry.get('x'), entry.get('y')) == (x+width//2, y+height//2)
                   and same_endpoint_observation(entry.get('endpoints'), endpoints, game_pid)]
        owners.extend(name for _ in matches)
        if not current.get('native_choice_source'):
            break
        current = load_result(runtime, current['native_choice_source'])
    if len(owners) != 1:
        raise ValueError('retrieval confirmation has no unique audited native dispatch')
    return hif_choice_settlements(projected, action, card_slots=slots)


def load_confirmed_choice_source(runtime, name, module, trail=(), *, allow_closed_transaction=False):
    """Recover only expected settlement outcomes; never authorize a new choice."""
    projected, baseline, row = load_choice_source(runtime, name, module, trail,
        allow_closed_transaction=allow_closed_transaction)
    expected = _confirmed_outcomes(runtime, projected, row)
    return projected, expected, baseline, row


def load_verified_choice_source(runtime, name, module, trail=()):
    """Reverify completed choice receipts and full components for native resume."""
    from live_decision.high_score.opening import resolve_hif_committed_observation

    projected, expected, baseline, row = load_confirmed_choice_source(runtime, name, module, trail)
    settlement = row.get('settlement', row['choice_journal'].get('settlement', {}))
    bundle = projected.source_bundle
    if (row.get('live_return') is not True or settlement.get('accepted') is not True
            or settlement.get('execution_submit_count') != 0 or settlement.get('end_turn_submit_count', 0) != 0
            or (settlement.get('run_id'), settlement.get('revision')) != (bundle.run_id, bundle.revision)):
        raise ValueError('retrieval source has no complete zero-submit settlement')
    state, comparison = resolve_hif_committed_observation(bundle, expected,
        core=settlement['core'], components=settlement['components'], baseline_components=baseline['components'],
        committed_action=projected.action)
    if (state is None or comparison.get('accepted') is not True
            or not _same(settlement.get('expected_state'), asdict(state))):
        raise ValueError('retrieval saved settlement differs from its selected outcome or complete observations')
    # As with ordinary native sources, retain the full pre-action bundle here.
    # A later fresh history resume alone installs the settled state's slots.
    return bundle, state, settlement, row
