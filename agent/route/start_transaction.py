"""Reserve the single approved AP submission before Maa can dispatch it."""
from copy import deepcopy

from .hif import PRESET_ID, known_hif_role
from .engine import Outcome, Decision, propose_unique, has_single_frame_id
from .new_run import initial_native_start, start_record, update_additional_start

START_AP_KIND = 'start_hif_once'
START_AP_DESTINATIONS = ('start_created',)
START_RECEIPT_CONTRACTS = {'start_ap_open_menu': ('start_created', ('start_ap_menu',)),
    'start_ap_save_home': ('start_ap_menu', ('start_ap_home',))}
START_RESUME_CONTRACTS = {'start_open_current_save': ('start_resume_home', ('start_resume_confirm',)),
    'start_resume_current_save': ('start_resume_confirm', ('start_created',))}


def _budget(host):
    from .prestart import _scope
    from .battle_policy import BATTLE_PROFILES
    _scope(host)
    state = start_record(host.handoff)
    if (not isinstance(state, dict) or state.get('schema_version') != 1
            or state.get('run_id') != host.run_id or state.get('game_pid') != host.game_pid
            or state.get('ap_cost') != 20 or state.get('maximum_cultivations') != 1
            or state.get('preset') != PRESET_ID
            or state.get('policy_profile') not in BATTLE_PROFILES.values()
            or not initial_native_start(state) and (
                not isinstance(state.get('previous_run_id'), str) or state['previous_run_id'] == host.run_id)
            or state.get('status') not in ('available', 'reserved', 'created_ap_unverified', 'complete')):
        raise ValueError('current cultivation lacks its independent once-only start budget')
    return deepcopy(state)


def available_start(host, page):
    from .new_run import matches_start_memory
    from .prestart import preparation_source
    if start_record(host.handoff) is None:
        return False
    state = _budget(host)
    if state['status'] != 'available':
        return False
    role = tuple(page.get('identity', ()))
    if (not preparation_source(host) or page.get('phase') != 'start_confirmation'
            or len(role) != 2 or not known_hif_role(*role)
            or type(page.get('ap')) is not int or type(page.get('remaining_ap')) is not int
            or page['ap'] - page['remaining_ap'] != 20 or page['remaining_ap'] < 0
            or page.get('ap_cost') != 20 or tuple(page.get('booster_selected', ())) != (False, False)
            or any(type(value) is not bool for value in page['booster_selected'])):
        raise ValueError('start requires the actual supported identity, exact price and both extras OFF')
    if not matches_start_memory(host, page):
        return False
    return True


def decide_start(host, observation, page):
    if not available_start(host, page):
        return Decision(Outcome.PAUSE, 'hif_start_once_budget_not_available')
    return propose_unique(observation, tuple(target for target in observation.targets if target.kind == START_AP_KIND),
        'hif-once-authorised-new-cultivation', START_AP_DESTINATIONS)


def _created_state(host, state, pending, issued, page, events):
    request = state.get('request', {})
    matched = [event for event in events if event.get('target_kind') == START_AP_KIND
        and event.get('target_key') == pending.target.key and event.get('source_scene') == 'start_confirmation'
        and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == issued
        and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
    if (state['status'] != 'reserved' or request.get('source_revision') != pending.source.revision
            or request.get('issued_at_ms') != issued or len(matched) != 1
            or page is None or page.observation.scene != 'start_created'
            or page.observation.run_id != host.run_id
            or not isinstance(page.start, dict) or page.start.get('phase') != 'start_created'
            or page.start.get('countdown_days') != 6 or not has_single_frame_id(page.frame_ids)
            or page.observation.stable_frames != len(page.frame_ids)
            or set(page.frame_ids) & set(request['source_frame_ids'])
            or page.observation.revision <= pending.source.revision
            or not 0 <= page.observation.capture_age_ms + host.clock.now() - page.observation.revision <= 1000):
        raise ValueError('start requires its exact successful submission and fresh new cultivation evidence')
    state.update(status='created_ap_unverified', creation_receipt=dict(
        revision=page.observation.revision, frame_ids=list(page.frame_ids),
        countdown_days=6, resources=deepcopy(page.start['creation_resources'])))
    update_additional_start(state, 'consumed')
    if state.get('creation_task') is not None:
        # The formal runner has confirmed delivery of the priced input and the
        # new run. Its first-day UI shows stamina and P-points, not account AP.
        state.update(status='complete', ap_receipt=dict(revision=page.observation.revision,
            frame_ids=list(page.frame_ids), ap_before=request['ap_before'],
            quoted_ap_after=request['quoted_ap_after'], submitted_ap_cost=request['ap_cost'],
            actual_ap_after=None, balance_evidence='start_creation_balance_unobserved'))
    return state


def start_checkpoint(host, gate, *, cancel_unissued=False):
    from .prestart_refill import AP_REFILL_APPLY, AP_REFILL_CONTRACTS, allows_ap_refill, ap_refill_prompt

    pending = gate.pending
    original = host.hooks.transition_gate.pending
    if pending is not None and pending.target.kind == AP_REFILL_APPLY:
        from .prestart import matches_preparation_contract, preparation_source

        state = _budget(host)
        request = dict(revision=pending.source.revision, issued_at_ms=gate.issued_at_ms,
            prompt=ap_refill_prompt(host))
        if state.get('ap_drink_request') == request:
            return None  # The saved pending action already reserved this drink.
        page = host._page
        if (not allows_ap_refill(host) or not preparation_source(host) or not request['prompt']
                or state.get('ap_drink_request') is not None
                or not matches_preparation_contract(pending, AP_REFILL_CONTRACTS[AP_REFILL_APPLY])
                or page is None or page.observation != pending.source
                or not has_single_frame_id(page.frame_ids)):
            raise ValueError('AP drink cannot be reserved for this observation')
        state['ap_drink_request'] = request
        return state
    if pending is None and cancel_unissued and original is not None and original.target.kind == AP_REFILL_APPLY:
        state = _budget(host)
        request = state.get('ap_drink_request') or {}
        if (request.get('revision') != original.source.revision
                or request.get('issued_at_ms') != host.hooks.transition_gate.issued_at_ms):
            raise ValueError('unissued AP drink differs from its reserved input')
        del state['ap_drink_request']
        return state
    if pending is not None and pending.target.kind == START_AP_KIND:
        state = _budget(host)
        if pending.source.scene != 'start_confirmation' or pending.expected_scenes != START_AP_DESTINATIONS:
            raise ValueError('start submission has a different transition contract')
        request = state.get('request', {})
        if state['status'] == 'reserved':
            if request.get('issued_at_ms') != gate.issued_at_ms or request.get('source_revision') != pending.source.revision:
                raise ValueError('start budget already reserved by another submission')
            return state
        page = host._page
        if (state['status'] != 'available' or page is None or page.observation != pending.source
                or not isinstance(page.start, dict) or not available_start(host, page.start)
                or not has_single_frame_id(page.frame_ids) or pending.source.stable_frames != len(page.frame_ids)
                or not 0 <= pending.source.capture_age_ms + host.clock.now() - pending.source.revision <= 1000):
            raise ValueError('start budget cannot reserve a stale, unknown or repeated submission')
        state.update(status='reserved', request=dict(source_revision=pending.source.revision,
            issued_at_ms=gate.issued_at_ms, source_frame_ids=list(page.frame_ids),
            ap_before=page.start['ap'], quoted_ap_after=page.start['remaining_ap'], ap_cost=20,
            identity=list(page.start['identity']), booster_selected=list(page.start['booster_selected']),
            price_evidence=deepcopy(page.start['start_evidence']['price'])))
        update_additional_start(state, 'pending')
        return state
    if pending is None and original is not None and original.target.kind == START_AP_KIND:
        state = _budget(host)
        request = state.get('request', {})
        late = state.get('late_creation_receipt', {})
        if (state['status'] in ('created_ap_unverified', 'complete') and host._page is not None
                and request.get('source_revision') == original.source.revision
                and request.get('issued_at_ms') == host.hooks.transition_gate.issued_at_ms
                and late.get('receipt_revision') == host._page.observation.revision
                and late.get('receipt_frame_ids') == list(host._page.frame_ids)):
            return state
        if (state['status'] != 'reserved' or request.get('source_revision') != original.source.revision
                or request.get('issued_at_ms') != host.hooks.transition_gate.issued_at_ms):
            raise ValueError('start settlement differs from the reserved submission')
        if cancel_unissued:
            history = list(state.get('cancelled_unissued_requests', ()))
            history.append(state.pop('request'))
            state.update(status='available', cancelled_unissued_requests=history)
            update_additional_start(state, 'available', cancelled_unissued=True)
            return state
        return _created_state(host, state, original, host.hooks.transition_gate.issued_at_ms, host._page, host.input_events)
    return None


def reconcile_started(host, gate, report, proof):
    from .engine import TransitionGate, validate_observation
    from .checkpoint import transition_checkpoint
    state = _budget(host)
    pending = gate.pending
    if (not host._active or pending is None or pending.target.kind != START_AP_KIND
            or pending.source.scene != 'start_confirmation' or pending.expected_scenes != START_AP_DESTINATIONS
            or not isinstance(report, dict) or report.get('route', {}).get('run_id') != host.run_id
            or report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid
            or validate_observation(proof.observation) is not None
            or not has_single_frame_id(proof.frame_ids) or proof.observation.stable_frames != len(proof.frame_ids)
            or set(proof.frame_ids) & {frame.get('frame_id') for frame in report.get('frames', ())}):
        raise ValueError('new cultivation reconciliation requires the original start and independent current page')
    captured = {frame['frame_id']: frame for frame in report.get('frames', ())}
    request = state.get('request', {})
    if not has_single_frame_id(request.get('source_frame_ids')) or pending.source.stable_frames != len(request['source_frame_ids']):
        raise ValueError('new cultivation original source frames are missing')
    for key in request['source_frame_ids']:
        source = captured.get(key, {}).get('observation', {})
        if (source.get('phase') != 'start_confirmation' or source.get('ap') != request.get('ap_before')
                or source.get('remaining_ap') != request.get('quoted_ap_after')
                or source.get('identity') != request.get('identity')
                or source.get('booster_selected') != request.get('booster_selected')
                or source.get('start_evidence', {}).get('price') != request.get('price_evidence')):
            raise ValueError('new cultivation source differs from the reserved price and identity')
    state = _created_state(host, state, pending, gate.issued_at_ms, proof, report['route'].get('input_events', ()))
    state['late_creation_receipt'] = dict(original_transition=transition_checkpoint(gate, host.clock.epoch),
        source_result=report['id'], receipt_revision=proof.observation.revision, receipt_frame_ids=list(proof.frame_ids))
    previous, previous_page = host.handoff, host._page
    host.handoff = dict(previous or {}, cultivation_start=state)
    host._page = proof
    result = TransitionGate(timeout_ms=gate.timeout_ms)
    try:
        host._save(transition_checkpoint(result, host.clock.epoch))
    except BaseException:
        host.handoff, host._page = previous, previous_page
        raise
    return result


def read_start_created(rows, *, image=None, enable_ap_navigation=False):
    from .production import _text
    _text(rows, r'H[.]I[.]F本戦まで', (20, 30, 190, 65))
    _text(rows, r'6日', (40, 75, 145, 90))
    stamina = _text(rows, r'\d{1,3}/\d{1,3}', (300, 40, 170, 80))
    currency = int(_text(rows, r'\d{1,6}', (340, 100, 130, 65)))
    current, maximum = map(int, stamina.split('/'))
    if not 0 <= current <= maximum or maximum <= 0:
        raise ValueError('new cultivation resources are contradictory')
    target = None
    if enable_ap_navigation:
        from .footer import cultivation_menu_target
        target = cultivation_menu_target(image)
    return dict(phase='start_created', target=target, countdown_days=6,
        creation_resources=dict(stamina=current, max_stamina=maximum, currency=currency))


def read_start_ap_menu(rows):
    from .production import _row
    save = _row(rows, r'保存して中断', (35, 985, 175, 65))
    return dict(phase='start_ap_menu', target=tuple(save['box']))


def read_start_ap_home(image, rows, *, context=None, read_clock=False):
    from .home import read_active_home
    # AP observation grants no resume input. The original save receipt binds
    # the debit; the normal save preview still verifies role/day before resume.
    home = read_active_home(image, rows, preview_saved=True)
    result = dict(phase='start_ap_home', target=None, ap=home['ap'], max_ap=home['max_ap'], home_evidence=home)
    if read_clock:
        from .ap_readback import read_recovery_countdown
        result['recovery_evidence'] = read_recovery_countdown(rows, context=context, image=image)
    return result
