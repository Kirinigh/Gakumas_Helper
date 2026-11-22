"""Prepare the supported HIF confirmation through normal game UI."""
import re
from copy import deepcopy

from .engine import Outcome, Decision, propose_unique, has_single_frame_id, source_frames_match
from .hif import requested_qualifier_memory
from .prestart_refill import (
    AP_REFILL_CANCELLED, AP_REFILL_CONTRACTS, AP_REFILL_INSUFFICIENT, AP_REFILL_SCENES,
    allows_ap_refill, ap_refill_prompt,
)
from .prestart_search import MEMORY_SEARCH_SCENES, MEMORY_SEARCH_CONTRACTS

MEMORY_LIST_SCENES = MEMORY_SEARCH_SCENES
PRESTART_CONTRACTS = {
    'start_open_produce': ('start_home', ('start_hif_scenario',)),
    'start_select_hif': ('start_hif_scenario', ('start_idol', 'start_confirmation',
        'start_memory_mismatch', AP_REFILL_INSUFFICIENT, 'start_refill_cancel')),
    'start_selected_idol_next': ('start_idol', ('start_confirmation', 'start_memory_mismatch')),
    **MEMORY_SEARCH_CONTRACTS,
    **AP_REFILL_CONTRACTS,
}
PRESTART_SCENES = frozenset({'start_home', 'start_hif_scenario', 'start_idol', 'start_unbound_selection',
                             'start_confirmation', *MEMORY_LIST_SCENES, *AP_REFILL_SCENES})
PRESTART_NAVIGATION = frozenset(PRESTART_CONTRACTS)


def matches_preparation_contract(pending, contract):
    return contract == (pending.source.scene, pending.expected_scenes)


def _active_caption(image, row):
    x, y, width, height = row['box']
    if float((image[y:y + height, x:x + width].min(2) > 235).mean()) < .06:
        raise ValueError('current start navigation caption is obscured or inactive')
    return tuple(row['box'])


def _scenario_main_price(rows):
    from .production import _text

    try:
        return int(_text(rows, r'\d+', (480, 910, 125, 60)))
    except ValueError:
        return None


def read_hif_scenario(image, rows, *, context=None):
    from .production import _row
    from .nonbattle_observation import _within
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('unsupported HIF preparation resolution')
    # Both upstreams navigate by the active main-stage control. The festival
    # artwork and the neighbouring qualifier's caption do not identify it.
    main = _row(rows, r'本戦', (505, 840, 145, 85), minimum=0)
    page = dict(phase='start_hif_scenario', target=_active_caption(image, main),
        start_evidence=dict(ap_cost=_scenario_main_price(rows)))
    has_event_switch = any(_within(row['box'], (240, 720, 290, 85)) and
        (row['text'] == 'イベントモード' or re.fullmatch(r'[●○]?(?:ON|OFF)', row['text'])) for row in rows)
    if not has_event_switch:
        # The scenario button is a free step. Record the season only when
        # observed; the paid confirmation still supplies the actual quote.
        winter = any(row['text'] == 'WINTER' and _within(row['box'], (270, 485, 175, 65)) for row in rows)
        return dict(page, event_mode=None, layout='winter' if winter else 'main')
    switch = _row(rows, r'[●○]?(?:ON|OFF)', (395, 720, 130, 85), minimum=0)
    if not switch['text'].endswith('OFF'):
        raise ValueError('current HIF scenario has event mode enabled')
    return dict(page, event_mode=False)


def read_selected_start_idol(image, rows):
    """Read the selected idol and P-idol card on the character page."""
    from live_decision.hif_profiles import notebook_hif_identity

    from .production import _row, _text
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('unsupported HIF preparation resolution')
    _text(rows, r'1[.]アイドル選択', (10, 15, 205, 65))
    card = _text(rows, r'\[.+\]', (355, 104, 365, 55))
    idol = _text(rows, r'.+', (500, 145, 215, 90))
    _text(rows, r'選抜試験メモリー', (470, 200, 230, 60))
    profile = notebook_hif_identity(card[1:-1], idol)
    next_row = _row(rows, r'次へ', (310, 1045, 155, 85))
    return dict(phase='start_idol', target=tuple(next_row['box']), identity=(profile.identity, profile.plan))


def read_prestart_page(context, image, rows, *, host=None):
    """None means no preparation header; a weak known header must fail closed."""
    from .new_run import start_memory_target
    from .prestart_refill import read_ap_refill_page
    from .prestart_memory import read_idol, role_page, memory_list_page, read_memory_list
    from .prestart_search import search_position
    from .nonbattle_observation import _within

    refill = read_ap_refill_page(rows, host, context)
    if refill is not None:
        return refill

    def header(pattern, region=None):
        return any(row['score'] >= .85 and re.fullmatch(pattern, row['text'])
                   and (region is None or _within(row['box'], region)) for row in rows)
    matches = [name for name, pattern in (
        ('home', r'ホーム'),
        ('idol', r'1[.]アイドル選択'), ('confirmation', r'2[.]開始確認')) if header(pattern)]
    if any(row['text'] == '本戦' and _within(row['box'], (505, 840, 145, 85)) for row in rows):
        matches.append('scenario')
    if memory_list_page(rows):
        # The list and its sort dialog can leave the confirmation header visible.
        matches = ['memory_list']
    elif role_page(rows) and 'idol' not in matches and 'confirmation' not in matches:
        matches.append('idol')
    if not matches or matches == ['home']:
        from .home import read_active_home
        try:
            page = read_active_home(image, rows, context=context, free_start=True)
        except ValueError:
            if matches:
                raise
            return None
        return dict(phase='start_home', target=page['target'], ap=page['ap'], max_ap=page['max_ap'],
            home_evidence={key: page[key] for key in ('home', 'produce', 'active_foreground_fraction')})
    if len(matches) != 1:
        raise ValueError('contradictory start preparation headers')
    if matches[0] in ('idol', 'memory_list') and host is not None and not preparation_source(host):
        return dict(phase='start_unbound_selection', target=None)
    if matches[0] == 'idol':
        return read_idol(context, image, rows)
    if matches[0] == 'memory_list':
        expected = (start_memory_target(host) or requested_qualifier_memory(context)) if host is not None else None
        position = search_position(host) if expected is not None else None
        if not position:
            raise ValueError('HIF memory list has no selected memory from this run')
        return read_memory_list(context, image, rows, expected=expected, **position)
    if matches[0] == 'scenario':
        return read_hif_scenario(image, rows, context=context)
    from .start import read_start_confirmation
    page = read_start_confirmation(context, image, rows)
    confirmation = dict(phase='start_confirmation', target=None, ap=page['ap'], remaining_ap=page['remaining_ap'],
        ap_cost=page['ap_cost'], booster_selected=page['booster_selected'], identity=(page['idol'], page['plan']),
        start_evidence=dict(price=page['price_evidence'], button=page['target'], boxes=page['booster_boxes']))
    return _selected_confirmation(host, context, image, rows, confirmation)


def _selected_confirmation(host, context, image, rows, confirmation):
    from .new_run import start_record, start_memory_target
    from .prestart_memory import read_confirm_memory, memory_change_target, confirm_memory_matches, memory_stats_match
    from .prestart_search import search_closed

    start = start_record(host.handoff) if host is not None else None
    if start is None or start.get('status') != 'available':
        return confirmation
    expected = start_memory_target(host)
    if expected is None:
        stats = read_confirm_memory(context, image, rows)
        expected = requested_qualifier_memory(context)
        if expected is None or stats is None:
            return dict(confirmation, memory_stats=stats)
        matches = memory_stats_match(expected, stats, confirm=True)
        if matches:
            return dict(confirmation, memory_stats=stats, memory_expected=stats, memory_matches=True)
    else:
        matches = confirm_memory_matches(context, image, rows, expected)
    if matches is None:
        return dict(confirmation, memory_stats=None, memory_expected=expected)
    if matches:
        return dict(confirmation, memory_expected=expected, memory_matches=True)
    if search_closed(host):
        return dict(confirmation, phase='start_memory_search_stopped', memory_expected=expected, memory_matches=False)
    return dict(confirmation, phase='start_memory_mismatch', target_kind='start_memory_change',
        target=memory_change_target(rows), memory_expected=expected, memory_matches=False)


def _scope(host):
    owner = host.ownership or {}
    if (owner.get('owner') != 'task600' or owner.get('phase') != 'recovery'
            or owner.get('game_pid') != host.game_pid or host.ledger is not None
            or (host.handoff or {}).get('live_context') is not None
            or (host.handoff or {}).get('terminal_receipt') is not None):
        raise ValueError('start preparation requires a new unbound nonbattle record')


def preparation_source(host):
    _scope(host)
    state = (host.handoff or {}).get('start_preparation', {})
    if not isinstance(state, dict):
        raise ValueError('start preparation metadata is invalid')
    source = state.get('hif_scenario', {})
    if not isinstance(source, dict):
        raise ValueError('start preparation HIF source is invalid')
    frames = source.get('frame_ids', ())
    if not isinstance(frames, (list, tuple)):
        raise ValueError('start preparation source frame identities are invalid')
    return (state.get('schema_version') == 1 and state.get('run_id') == host.run_id
        and state.get('game_pid') == host.game_pid
        and (source.get('event_mode') is False
            or source.get('event_mode') is None and source.get('layout') in ('winter', 'main')) and has_single_frame_id(frames))


def retain_preparation_page(host, page, frames):
    _scope(host)
    if not has_single_frame_id(frames):
        raise ValueError('start preparation lacks current source captures')
    state = deepcopy((host.handoff or {}).get('start_preparation', {}))
    if not isinstance(state, dict) or state and (
            state.get('schema_version') != 1 or state.get('run_id') != host.run_id
            or state.get('game_pid') != host.game_pid):
        raise ValueError('start preparation metadata belongs to another run or game')
    if page['phase'] == 'start_home':
        state = {'navigation_receipts': state['navigation_receipts']} if 'navigation_receipts' in state else {}
    state.update(schema_version=1, run_id=host.run_id, game_pid=host.game_pid,
        last_page=page['phase'], last_frame_ids=list(frames))
    if page['phase'] == 'start_hif_scenario':
        displayed_price = page.get('start_evidence', {}).get('ap_cost')
        state['hif_scenario'] = dict(frame_ids=list(frames), ap_cost=displayed_price,
            event_mode=page['event_mode'], **({'layout': page['layout']} if 'layout' in page else {}))
    if page['phase'] == AP_REFILL_INSUFFICIENT and page.get('target') is not None:
        state['ap_refill_prompt'] = dict(shortfall=page['ap_shortfall'], frame_id=frames[0])
    host.handoff = dict(host.handoff or {}, start_preparation=state)
    if page['phase'] == 'start_confirmation':
        from .new_run import retain_start_memory
        retain_start_memory(host, page, frames)


def decide_preparation(host, observation, page):
    _scope(host)
    # A normal Produce task lets the game report the shortage after selecting HIF.
    # One-off acceptance authorizations cover only their original cultivation.
    if (page['phase'] == 'start_home' and page['ap'] is not None and page['ap'] < 20
            and not allows_ap_refill(host)):
        return Decision(Outcome.PAUSE, 'hif_start_insufficient_current_ap')
    if page['phase'] in AP_REFILL_SCENES:
        from .start_transaction import _budget
        if page['phase'] == AP_REFILL_CANCELLED:
            return Decision(Outcome.STOP, 'hif_ap_refill_disabled')
        if not allows_ap_refill(host) or not preparation_source(host):
            return Decision(Outcome.STOP, 'hif_ap_refill_requires_current_hif_selection')
        if page['phase'] != 'start_refill_close' and _budget(host).get('ap_drink_request') is not None:
            return Decision(Outcome.STOP, 'hif_ap_refill_already_attempted')
        if page['phase'] != 'start_refill_cancel' and not ap_refill_prompt(host):
            return Decision(Outcome.STOP, 'hif_ap_refill_requires_game_shortage_prompt')
    if page['phase'] in ('start_unbound_selection', 'start_idol', 'start_confirmation', *MEMORY_LIST_SCENES) and not preparation_source(host):
        return Decision(Outcome.STOP, 'hif_start_scenario_provenance_required')
    if page['phase'] == 'start_confirmation':
        from .new_run import start_record
        if start_record(host.handoff) is not None:
            from .start_transaction import decide_start
            return decide_start(host, observation, page)
        return Decision(Outcome.PAUSE, 'hif_start_requires_once_only_ap_transaction')
    if page['phase'] == 'start_memory_search_stopped':
        return Decision(Outcome.STOP, 'hif_selected_memory_not_found')
    if len(observation.targets) != 1:
        return Decision(Outcome.STOP, 'target_not_uniquely_available')
    contract = PRESTART_CONTRACTS.get(observation.targets[0].kind)
    if contract is None or contract[0] != page['phase']:
        return Decision(Outcome.STOP, 'hif_start_navigation_outside_current_page')
    return propose_unique(observation, observation.targets, 'hif-prepare:' + page['phase'], contract[1])


def _returned_to_start_home(host, gate, report, seen):
    from .new_run import validate_additional_start
    from .input_receipt import original_input_report
    from .start_transaction import _budget

    target = gate.pending.target
    if (seen.scene != 'start_home' or (target.key, target.kind) != ('start_open_produce', 'start_open_produce')
            or seen.targets != (target,) or target.available is not True):
        return False
    budget = _budget(host)
    if (budget['status'] != 'available' or any(key in budget for key in ('request', 'creation_receipt', 'ap_receipt'))
            or original_input_report(host, gate) != report):
        return False
    if 'additional_start' in budget:
        validate_additional_start(budget['additional_start'], run_id=host.run_id,
            previous_run_id=budget['previous_run_id'], status='available')
    history = (host.handoff.get('start_preparation') or {}).get('navigation_receipts', ())
    return not any(row.get('disposition') == 'returned_to_start_home' for row in history)


def reconcile_preparation_navigation(host, gate, report, proof):
    from .engine import TransitionGate, validate_observation
    from .checkpoint import transition_checkpoint
    from .prestart_search import search_destination
    _scope(host)
    contracts = PRESTART_CONTRACTS.copy()
    from .first_day import FIRST_DAY_CONTRACTS, first_day_scope
    from .start_transaction import START_RESUME_CONTRACTS, START_RECEIPT_CONTRACTS, _budget
    contracts.update(START_RECEIPT_CONTRACTS)
    contracts.update(START_RESUME_CONTRACTS)
    contracts.update(FIRST_DAY_CONTRACTS)
    pending = gate.pending
    contract = contracts.get(pending.target.kind) if pending is not None else None
    if pending is not None and pending.target.kind in START_RECEIPT_CONTRACTS and _budget(host)['status'] != 'created_ap_unverified':
        raise ValueError('AP verification navigation requires the original created cultivation')
    if pending is not None and pending.target.kind in START_RESUME_CONTRACTS and _budget(host)['status'] != 'complete':
        raise ValueError('saved first-day resume requires the original observed AP receipt')
    if pending is not None and pending.target.kind in FIRST_DAY_CONTRACTS:
        first_day_scope(host)
    if (not host._active or pending is None or not matches_preparation_contract(pending, contract) or not isinstance(report, dict)
            or report.get('route', {}).get('run_id') != host.run_id
            or report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid):
        raise ValueError('preparation recovery does not match the original action')
    seen = proof.observation
    returned_home = _returned_to_start_home(host, gate, report, seen)
    matched = [row for row in report['route'].get('input_events', ())
        if row.get('target_key') == pending.target.key and row.get('target_kind') == pending.target.kind
        and row.get('source_revision') == pending.source.revision and row.get('source_scene') == pending.source.scene
        and row.get('issued_at_ms') == gate.issued_at_ms and row.get('dispatch_attempted') is True
        and row.get('success') is True and row.get('uncertain') is False]
    if (len(matched) != 1 or validate_observation(seen) is not None or seen.run_id != host.run_id
            or (seen.scene not in pending.expected_scenes and not returned_home) or seen.revision <= pending.source.revision
            or not search_destination(pending, seen, proof.start)
            or (seen.scenario, seen.idol, seen.plan) != (pending.source.scenario, pending.source.idol, pending.source.plan)
            or not all(isinstance(key, str) and key for key in proof.frame_ids)
            or not source_frames_match(seen, proof.frame_ids)
            or not 0 <= seen.capture_age_ms + host.clock.now() - seen.revision <= 1000
            or set(proof.frame_ids) & {frame.get('frame_id') for frame in report.get('frames', ())}):
        raise ValueError('preparation recovery requires independent fresh destination frames')
    previous = host.handoff
    handoff = deepcopy(previous or {})
    state = handoff.get('start_preparation', {})
    history = list(state.get('navigation_receipts', ()))
    history.append(dict(original_transition=transition_checkpoint(gate, host.clock.epoch),
        source_result=report['id'], destination=seen.scene, receipt_revision=seen.revision,
        receipt_frame_ids=list(proof.frame_ids)))
    if returned_home:
        history[-1]['disposition'] = 'returned_to_start_home'
    handoff['start_preparation'] = dict(state, navigation_receipts=history)
    host.handoff = handoff
    result = TransitionGate(timeout_ms=gate.timeout_ms)
    try:
        host._save(transition_checkpoint(result, host.clock.epoch))
    except BaseException:
        host.handoff = previous
        raise
    return result
