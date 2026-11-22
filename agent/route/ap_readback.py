"""Record post-start AP; optional clock evidence can explain recovery."""
import re
from copy import deepcopy

from .engine import has_single_frame_id


def bind_ap_clock_sources(producer, runtime, sources):
    """Existing local reports are evidence, never an input or new-run grant."""
    import json
    from pathlib import Path

    if (not isinstance(sources, list) or len(sources) != 2 or len(set(map(str, sources))) != 2
            or any(not isinstance(source, str) or re.fullmatch(r'task600-[a-z0-9-]{1,130}', source) is None
                for source in sources)):
        raise ValueError('AP clock sources must name two distinct original task600 reports')
    reports = [json.loads((Path(runtime)/f'result-{source}.json').read_bytes()) for source in sources]
    if any(not isinstance(report, dict) or report.get('id') != source for source, report in zip(sources, reports, strict=True)):
        raise ValueError('AP clock source report identity differs')
    producer.ap_clock_reports = reports


def _tick(key, game_pid):
    match = re.fullmatch(r'route-frame-' + str(game_pid) + r'-(\d+)', key)
    if match is None:
        raise ValueError('AP clock capture belongs to another game or clock')
    return int(match[1])


def read_recovery_countdown(rows, *, context=None, image=None):
    from .production import _row
    try:
        row = _row(rows, r'\d{2}:\d{2}', (330, 0, 130, 32))
    except ValueError:
        if context is None:
            raise
        from live_decision.hif_recovery import _field
        row = _field(context, image, 'APRecoveryCountdown', (350, 0, 100, 27), r'\d{2}:\d{2}')
    minute, second = map(int, row['text'].split(':'))
    if second >= 60 or not 0 < minute * 60 + second < 3600:
        raise ValueError('current AP recovery countdown is invalid')
    return dict(seconds=minute * 60 + second, row=deepcopy(row))


def _scenario_clock_balance(page, rows, ap):
    from .nonbattle_observation import _within

    # The scenario leaves its balance in the original screen row, sometimes joined to the plus icon.
    stored = page.get('ap'), page.get('max_ap')
    expected = (ap, 100)
    readings = [tuple(map(int, row['text'].rstrip('+').split('/'))) for row in rows
        if row['score'] >= .95 and re.fullmatch(r'\d{1,3}/\d{1,3}\+?', row['text'])
        and _within(row['box'], (290, 5, 205, 75))]
    if (any(value is not None and (type(value) is not int or value != target)
            for value, target in zip(stored, expected, strict=True))
            or any(reading != expected for reading in readings)):
        raise ValueError('AP recovery source balance contradicts the current original quote')
    return len(readings) == 1


def _clock_source(report, *, run_id, game_pid, ap, home):
    from .production import _text
    if (not isinstance(report, dict) or not isinstance(report.get('id'), str)
            or report.get('route', {}).get('run_id') != run_id
            or report.get('endpoints', {}).get('game', {}).get('pid') != game_pid
            or report.get('input_uncertain_count') != 0):
        raise ValueError('AP recovery clock source differs from this cultivation history')
    for frame in reversed(report.get('frames', ())):
        rows = frame.get('rows', ())
        try:
            page = frame.get('observation', {})
            accepted = (page.get('phase') == 'start_ap_home'
                and page.get('home_evidence', {}).get('active_foreground_fraction', 0) >= .95
                and page.get('home_evidence', {}).get('home', {}).get('confidence', 0) >= .95
                and page.get('home_evidence', {}).get('produce', {}).get('confidence', 0) >= .8) if home else (
                    page.get('phase') == 'start_hif_scenario'
                    and (page.get('event_mode') is False
                        or page.get('layout') in ('winter', 'main') and page.get('event_mode') is None))
            if accepted:
                current, maximum = page.get('ap'), page.get('max_ap')
            elif home:
                _text(rows, r'ホーム', (300, 1160, 120, 60))
                _text(rows, r'プロデュース', (225, 940, 300, 115))
            else:
                _text(rows, r'HIF', (230, 260, 355, 155))
                _text(rows, r'[『「]?HatsuboshiIDOLFESTIVAL[』」]?', (170, 595, 365, 65))
            if not accepted:
                fraction = _text(rows, r'\d{1,3}/\d{1,3}', (310, 20, 165, 55))
                current, maximum = map(int, fraction.split('/'))
            clock_row = frame.get('independent_recovery_clock')
            countdown = read_recovery_countdown([clock_row] if clock_row is not None else rows)
            tick = _tick(frame['frame_id'], game_pid)
        except (ValueError, KeyError):
            continue
        if accepted and not home:
            if not _scenario_clock_balance(page, rows, ap):
                continue
        elif current != ap or maximum != 100:
            raise ValueError('AP recovery source balance contradicts the current original quote')
        # The displayed integer countdown and its capture time form one observation.
        recovery_tick = tick + countdown['seconds'] * 1000
        return dict(source_result=report['id'], ap=ap, frame_ids=[frame['frame_id']],
            last_tick=tick, next_recovery_interval_ms=[recovery_tick - 1500, recovery_tick + 1500])
    raise ValueError('AP recovery source lacks a complete numeric observation')


def _same_run_recovery_receipt(host, state, proof, reports):
    """Measure a recovery period after the debit, including time in preparation."""
    from .checkpoint import restore_transition
    from .input_receipt import original_input_report
    request = state['request']
    game = state.get('creation_task', {}).get('game', {})
    if any(any(report.get('endpoints', {}).get('game', {}).get(key) != game.get(key)
            for key in ('pid', 'executable', 'created_at_100ns')) for report in reports):
        raise ValueError('AP recovery clocks belong to another original game instance')
    prior = _clock_source(reports[0], run_id=host.run_id, game_pid=host.game_pid, ap=request['ap_before'], home=False)
    balances = {frame.get('observation', {}).get('ap') for frame in reports[1].get('frames', ())
        if frame.get('observation', {}).get('phase') == 'start_ap_home'}
    if len(balances) != 1 or type(next(iter(balances))) is not int:
        raise ValueError('post-start clock has no unique observed balance')
    after_ap = next(iter(balances))
    after = _clock_source(reports[1], run_id=host.run_id, game_pid=host.game_pid, ap=after_ap, home=True)
    carrier = host.handoff.get('nonbattle_context', {}).get('route_input_receipt', {})
    gate = restore_transition(carrier.get('original_transition'), clock_epoch=host.clock.epoch)
    original = original_input_report(host, gate)['route']['input_events'][0]
    events = reports[1].get('route', {}).get('input_events', ())
    page = proof.start
    current = page.get('recovery_evidence', {})
    current_tick = _tick(current.get('frame_id', ''), host.game_pid)
    seconds = current.get('seconds')
    if (gate.pending.target.kind != 'start_ap_save_home' or events.count(original) != 1
            or any(event.get('success') is not True or event.get('uncertain') is not False for event in events)
            or not prior['last_tick'] < request['issued_at_ms'] < original['issued_at_ms'] < after['last_tick'] < current_tick
            or current.get('frame_id') not in proof.frame_ids or type(seconds) is not int or not 0 < seconds < 3600
            or set(proof.frame_ids) & set((*prior['frame_ids'], *after['frame_ids']))
            or type(page.get('ap')) is not int or page.get('max_ap') != 100
            or not request['quoted_ap_after'] < after_ap < page['ap'] < 100):
        raise ValueError('AP recovery needs the original debit and an independent later balance increase')
    growth = page['ap'] - after_ap
    deadline = [current_tick + seconds * 1000 - 1500, current_tick + seconds * 1000 + 1500]
    b = after['next_recovery_interval_ms']
    period = [-(-(deadline[0] - b[1]) // growth), (deadline[1] - b[0]) // growth]
    if not 60_000 <= period[0] <= period[1] <= 3_600_000 or period[1] - period[0] > 6000:
        raise ValueError('independently observed AP recovery period is unresolved')
    a = prior['next_recovery_interval_ms']
    recovered = [count for count in range(25) if max(a[0] + count * period[0], deadline[0])
        <= min(a[1] + count * period[1], deadline[1])]
    if len(recovered) != 1 or page['ap'] != request['ap_before'] - 20 + recovered[0]:
        raise ValueError('AP balance does not match the single debit and independently measured recovery')
    state.update(status='complete', actual_ap_after=page['ap'], ap_receipt=dict(
        revision=proof.observation.revision, frame_ids=list(proof.frame_ids), ap_before=request['ap_before'],
        quoted_ap_after=request['quoted_ap_after'], actual_ap_after=page['ap'], submitted_ap_cost=20,
        recovered_ap=recovered[0], net_ap_change=page['ap']-request['ap_before'], recovery_evidence=dict(
            before_start=prior, after_start=after, period_interval_ms=period, current=deepcopy(current),
            current_deadline_interval_ms=deadline, interval_covers_preparation_and_post_start=True)))
    return state


def _observed_exact_receipt(host, state, proof, report):
    """Consume an original observed debit; later balances cannot erase it."""
    from .checkpoint import restore_transition
    from .input_receipt import original_input_report
    request = state['request']
    carrier = host.handoff.get('nonbattle_context', {}).get('route_input_receipt', {})
    gate = restore_transition(carrier.get('original_transition'), clock_epoch=host.clock.epoch)
    original = original_input_report(host, gate)['route']['input_events'][0]
    game = report.get('endpoints', {}).get('game', {})
    creation = state.get('creation_task', {}).get('game', {})
    events = report.get('route', {}).get('input_events', ())
    if (gate.pending.target.kind != 'start_ap_save_home' or report.get('route', {}).get('run_id') != host.run_id
            or not isinstance(report.get('id'), str) or report.get('input_uncertain_count') != 0
            or any(game.get(key) != creation.get(key) for key in ('pid', 'executable', 'created_at_100ns'))
            or events.count(original) != 1 or any(event.get('success') is not True
                or event.get('uncertain') is not False for event in events)):
        raise ValueError('observed AP source differs from this original start and saved game')
    accepted = [event for event in report['route'].get('events', ()) if event.get('phase') == 'start_ap_home']
    if len(accepted) != 1:
        raise ValueError('observed AP source has no unique saved-home observation')
    keys = accepted[0].get('frame_ids', ())
    if not has_single_frame_id(keys) or proof is not None and set(keys) & set(proof.frame_ids):
        raise ValueError('observed AP source lacks independent original home frames')
    captured = {frame['frame_id']: frame for frame in report.get('frames', ())}
    ticks = [_tick(key, host.game_pid) for key in keys]
    revision = ticks[-1] + accepted[0]['capture_age_ms']
    latest = proof.observation.revision if proof is not None else host.clock.now()
    if (ticks != sorted(ticks) or ticks[-1] - ticks[0] > 1000
            or not original['issued_at_ms'] < ticks[0] <= revision < latest
            or proof is not None and (type(proof.start.get('ap')) is not int
                or proof.start['ap'] < request['quoted_ap_after'])):
        raise ValueError('observed AP source is not the original earlier debit observation')
    for key in keys:
        page = captured.get(key, {}).get('observation', {})
        home = page.get('home_evidence', {})
        if (page.get('phase') != 'start_ap_home' or page.get('ap') != request['quoted_ap_after']
                or page.get('max_ap') != 100 or home.get('ap') != page['ap']
                or home.get('active_foreground_fraction', 0) < .95
                or home.get('home', {}).get('confidence', 0) < .95
                or home.get('produce', {}).get('confidence', 0) < .8):
            raise ValueError('observed AP source does not confirm the original exact debit')
    state.update(status='complete', actual_ap_after=request['quoted_ap_after'], ap_receipt=dict(
        source_result=report['id'], revision=revision, frame_ids=list(keys), ap_before=request['ap_before'],
        quoted_ap_after=request['quoted_ap_after'], actual_ap_after=request['quoted_ap_after'],
        submitted_ap_cost=20, recovered_ap=0, net_ap_change=-20,
        balance_evidence='original_observed_balance_equals_quote'))
    if proof is not None:
        state['ap_receipt']['current_balance_observation'] = dict(revision=proof.observation.revision,
            frame_ids=list(proof.frame_ids), ap=proof.start['ap'], max_ap=proof.start['max_ap'])
    return state


def verified_ap_receipt(host, proof, reports, *, observed_report=None):
    from .engine import validate_observation
    from .start_transaction import _budget
    state = _budget(host)
    request, creation = state.get('request', {}), state.get('creation_receipt', {})
    if (type(request.get('ap_before')) is not int or type(request.get('quoted_ap_after')) is not int
            or request['ap_before'] - request['quoted_ap_after'] != 20 or request.get('ap_cost') != 20):
        raise ValueError('AP receipt differs from the original reserved twenty-AP quote')
    if proof is None and observed_report is not None:
        if (not host._active or state['status'] != 'created_ap_unverified'
                or host.hooks.transition_gate.pending is not None or host.hooks.transition_gate.stopped):
            raise ValueError('observed AP receipt requires its original settled save navigation')
        return _observed_exact_receipt(host, state, None, observed_report)
    page = proof.start or {}
    seen = proof.observation
    if (not host._active or host._page is not proof or state['status'] != 'created_ap_unverified'
            or host.hooks.transition_gate.pending is not None or host.hooks.transition_gate.stopped
            or seen.scene != 'start_ap_home' or page.get('phase') != 'start_ap_home'
            or seen.run_id != host.run_id or validate_observation(seen) is not None
            or not has_single_frame_id(proof.frame_ids) or seen.stable_frames != len(proof.frame_ids)
            or not 0 <= seen.capture_age_ms + host.clock.now() - seen.revision <= 1000
            or set(proof.frame_ids).intersection((*request.get('source_frame_ids', ()), *creation.get('frame_ids', ())))):
        raise ValueError('actual AP receipt requires this current saved run and independent active home')
    if observed_report is not None:
        return _observed_exact_receipt(host, state, proof, observed_report)
    if type(page.get('ap')) is int and page.get('max_ap') == 100 and page['ap'] == request['quoted_ap_after']:
        state.update(status='complete', actual_ap_after=page['ap'], ap_receipt=dict(
            revision=seen.revision, frame_ids=list(proof.frame_ids), ap_before=request['ap_before'],
            quoted_ap_after=request['quoted_ap_after'], actual_ap_after=page['ap'],
            submitted_ap_cost=20, recovered_ap=0, net_ap_change=-20,
            balance_evidence='current_balance_equals_original_quote'))
        return state
    if (reports is None and type(page.get('ap')) is int and page.get('max_ap') == 100
            and request['quoted_ap_after'] < page['ap'] <= 100):
        # The approved debit is the once-only twenty-AP submission. Upstream
        # continues after creation; increased AP does not require a timer proof.
        # Preserve the observation without inventing a recovery count.
        state.update(status='complete', actual_ap_after=page['ap'], ap_receipt=dict(
            revision=seen.revision, frame_ids=list(proof.frame_ids), ap_before=request['ap_before'],
            quoted_ap_after=request['quoted_ap_after'], actual_ap_after=page['ap'], submitted_ap_cost=20,
            recovered_ap=None, net_ap_change=page['ap'] - request['ap_before'],
            balance_increase_over_quote=page['ap'] - request['quoted_ap_after'],
            balance_evidence='current_balance_above_original_quote_recovery_unmeasured'))
        return state
    if not isinstance(reports, (tuple, list)) or len(reports) != 2:
        raise ValueError('AP recovery requires the previous and pre-start clock sources')
    if all(isinstance(report, dict) and report.get('route', {}).get('run_id') == host.run_id for report in reports):
        return _same_run_recovery_receipt(host, state, proof, reports)
    before = request['ap_before']
    old = _clock_source(reports[0], run_id=state['previous_run_id'], game_pid=host.game_pid, ap=before - 1, home=True)
    prior = _clock_source(reports[1], run_id=host.run_id, game_pid=host.game_pid, ap=before, home=False)
    a, b = old['next_recovery_interval_ms'], prior['next_recovery_interval_ms']
    period = [b[0] - a[1], b[1] - a[0]]
    current = page.get('recovery_evidence', {})
    current_tick = _tick(current.get('frame_id', ''), host.game_pid)
    seconds = current.get('seconds')
    if (not 60_000 <= period[0] <= period[1] <= 3_600_000 or period[1] - period[0] > 6000
            or not prior['last_tick'] < request['issued_at_ms'] < b[0]
            or current.get('frame_id') not in proof.frame_ids or current_tick <= request['issued_at_ms']
            or type(seconds) is not int or not 0 < seconds < 3600):
        raise ValueError('AP recovery timing does not cover this exact start and current observation')
    now_deadline = [current_tick + seconds * 1000 - 1500, current_tick + seconds * 1000 + 1500]
    recovered = [count for count in range(25) if max(b[0] + count * period[0], now_deadline[0])
        <= min(b[1] + count * period[1], now_deadline[1])]
    if (len(recovered) != 1 or type(page.get('ap')) is not int or page.get('max_ap') != 100
            or not 0 <= page['ap'] < 100 or page['ap'] != before - 20 + recovered[0]):
        raise ValueError('actual AP balance is not explained by the single start and observed recovery clock')
    state.update(status='complete', actual_ap_after=page['ap'], ap_receipt=dict(
        revision=seen.revision, frame_ids=list(proof.frame_ids), ap_before=before,
        quoted_ap_after=request['quoted_ap_after'], actual_ap_after=page['ap'],
        submitted_ap_cost=20, recovered_ap=recovered[0], net_ap_change=page['ap'] - before,
        recovery_evidence=dict(previous=old, before_start=prior, period_interval_ms=period,
            current=deepcopy(current), current_deadline_interval_ms=now_deadline)))
    return state
