"""Persist the current memory submission/result in the existing HIF envelope."""
from copy import deepcopy

from .engine import has_single_frame_id


def _base(host, terminal_report=None):
    from .ending import verified_ending_receipt
    receipt = verified_ending_receipt(host, terminal_report=(terminal_report
        if terminal_report is not None else getattr(host, '_ending_terminal_report', None)),
        round_number=(host.handoff or {}).get('round_number'))
    expected = dict(schema_version=1, route_run_id=host.run_id, game_pid=host.game_pid,
        handoff_id=host.handoff['handoff_id'], live_run_id=receipt['run_id'],
        live_revision=receipt['revision'], round_number=receipt['round_number'])
    current = host.handoff.get('closing')
    if current is not None and (not isinstance(current, dict)
            or any(type(current.get(key)) is not type(value) or current.get(key) != value for key, value in expected.items())):
        raise ValueError('closing memory state belongs to another run or terminal epoch')
    generation = current.get('generation') if isinstance(current, dict) else None
    if generation is not None and (not isinstance(generation, dict)
            or generation.get('status') not in ('prepared', 'not_issued', 'complete')):
        raise ValueError('closing memory lifecycle is unknown')
    return deepcopy(current) if current is not None else expected, receipt


def completed_memory(host, terminal_report=None):
    state, _ = _base(host, terminal_report)
    generation = state.get('generation')
    if generation is None or generation.get('status') != 'complete':
        return None
    if (not has_single_frame_id(generation.get('source_frame_ids')) or not has_single_frame_id(generation.get('receipt_frame_ids'))
            or set(generation['source_frame_ids']) & set(generation['receipt_frame_ids'])
            or type(generation.get('issued_at_ms')) is not int or generation['issued_at_ms'] < 0
            or type(generation.get('source_revision')) is not int
            or type(generation.get('receipt_revision')) is not int
            or generation['issued_at_ms'] < generation['source_revision']
            or generation['receipt_revision'] <= generation['source_revision']
            or generation['receipt_revision'] < generation['issued_at_ms']):
        raise ValueError('closing memory completion receipt is incomplete')
    return generation


def complete_memory(host, pending, issued_at_ms, proof, *, report=None, terminal_report=None):
    """Use a successful original submission and a fresh explicit completion page."""
    from .engine import validate_observation
    state, receipt = _base(host, terminal_report)
    seen = proof.observation
    if (pending.target.kind != 'end_generate_memory' or pending.source.scene != 'ending_generation'
            or seen.scene != 'ending_generated_memory' or seen.run_id != host.run_id
            or not isinstance(proof.closing, dict) or proof.closing.get('memory_brand') != 'HIF'
            or validate_observation(seen) is not None or seen.stable_frames != 1
            or seen.revision <= pending.source.revision or seen.revision - seen.capture_age_ms < issued_at_ms
            or not 0 <= seen.capture_age_ms + host.clock.now() - seen.revision <= 1000
            or not has_single_frame_id(proof.frame_ids) or set(proof.frame_ids) & set(receipt['frame_ids'])):
        raise ValueError('current memory result is not independent of its original submission')
    if report is not None:
        if (report.get('route', {}).get('run_id') != host.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid):
            raise ValueError('memory source differs from the current owned session')
        events = report['route'].get('input_events', [])
        sources = [event for event in report['route'].get('events', [])
            if event.get('phase') == 'ending_generation' and has_single_frame_id(event.get('frame_ids'))
            and type(event.get('capture_age_ms')) is int
            and int(event['frame_ids'][-1].rsplit('-', 1)[1]) + event['capture_age_ms'] == pending.source.revision]
        if len(sources) != 1 or set(proof.frame_ids) & {frame.get('frame_id') for frame in report.get('frames', [])}:
            raise ValueError('memory result reuses the original submission evidence')
        source_frames = sources[0]['frame_ids']
        captured = {frame['frame_id']: frame for frame in report.get('frames', [])}
        if any(captured.get(key, {}).get('observation', {}).get('phase') != 'ending_generation'
                or captured.get(key, {}).get('observation', {}).get('generation_idol') != host.handoff['idol']
                for key in source_frames):
            raise ValueError('memory source differs from the accepted terminal idol')
    else:
        events = host.input_events
        prepared = state.get('generation', {})
        if (prepared.get('status') != 'prepared' or prepared.get('source_revision') != pending.source.revision
                or prepared.get('issued_at_ms') != issued_at_ms or not has_single_frame_id(prepared.get('source_frame_ids'))):
            raise ValueError('memory result has no original prepared submission')
        source_frames = prepared['source_frame_ids']
    matched = [event for event in events if event.get('target_kind') == 'end_generate_memory'
        and event.get('target_key') == pending.target.key and event.get('source_scene') == 'ending_generation'
        and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == issued_at_ms
        and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
    if len(matched) != 1 or set(source_frames) & set(proof.frame_ids):
        raise ValueError('memory completion lacks its exact successful original submission')
    state['generation'] = dict(status='complete', source_revision=pending.source.revision,
        issued_at_ms=issued_at_ms, source_frame_ids=list(source_frames),
        source_result=report['id'] if report is not None else None,
        receipt_revision=seen.revision, receipt_frame_ids=list(proof.frame_ids))
    return state


def memory_checkpoint(host, gate, *, cancel_unissued=False):
    """Called by the same host's atomic write; never sends or retries input."""
    pending, original = gate.pending, host.hooks.transition_gate.pending
    if pending is not None and pending.target.kind == 'end_generate_memory':
        state, _ = _base(host)
        page = host._page
        if (page is None or page.observation != pending.source or pending.source.scene != 'ending_generation'
                or not isinstance(page.closing, dict) or page.closing.get('generation_idol') != host.handoff['idol']
                or page.observation.stable_frames != 1 or not has_single_frame_id(page.frame_ids)
                or state.get('generation', {}).get('status') == 'complete'):
            raise ValueError('memory submission is not bound to a fresh uncompleted current cover')
        state['generation'] = dict(status='prepared', source_revision=pending.source.revision,
            issued_at_ms=gate.issued_at_ms, source_frame_ids=list(page.frame_ids))
        return state
    if pending is None and original is not None and original.target.kind == 'end_generate_memory':
        if cancel_unissued:
            state, _ = _base(host)
            state['generation'] = dict(state.get('generation', {}), status='not_issued')
            return state
        state, _ = _base(host)
        already = completed_memory(host)
        if (already is not None and already['source_revision'] == original.source.revision
                and already['issued_at_ms'] == host.hooks.transition_gate.issued_at_ms
                and host._page is not None and already['receipt_revision'] == host._page.observation.revision
                and already['receipt_frame_ids'] == list(host._page.frame_ids)):
            return state
        return complete_memory(host, original, host.hooks.transition_gate.issued_at_ms, host._page)
    return None


def complete_home(host, proof, *, terminal_report=None):
    from .engine import validate_observation
    state, receipt = _base(host, terminal_report)
    generated = completed_memory(host, terminal_report)
    page = proof.closing or {}
    seen = proof.observation
    if (not host._active or host._page is not proof or generated is None
            or host.hooks.transition_gate.pending is not None or host.hooks.transition_gate.stopped
            or host.ledger is not None and host.ledger.pending is not None
            or seen.scene != 'ending_home' or page.get('phase') != 'ending_home'
            or validate_observation(seen) is not None
            or seen.run_id != host.run_id or seen.stable_frames != 1 or not has_single_frame_id(proof.frame_ids)
            or seen.revision <= generated['receipt_revision']
            or not 0 <= seen.capture_age_ms + host.clock.now() - seen.revision <= 1000
            or set(proof.frame_ids) & (set(generated['source_frame_ids']) | set(generated['receipt_frame_ids']) | set(receipt['frame_ids']))
            or type(page.get('ap')) is not int or type(page.get('max_ap')) is not int
            or not 0 <= page['ap'] <= page['max_ap'] or page['max_ap'] <= 0):
        raise ValueError('cultivation completion requires current active home and this memory receipt')
    if state.get('state') != 'complete':
        state.update(state='complete', home=dict(frame_ids=list(proof.frame_ids), revision=seen.revision,
            ap=page['ap'], max_ap=page['max_ap']))
    return state
