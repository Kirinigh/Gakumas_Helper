"""Original route dispatch evidence in the existing durable handoff.

A prepared record is ambiguous after a process death. Recovery observes the
destination of an acknowledged input, including a confirmed tap in a partial
first-lesson group. A receipt never permits the original input to be repeated.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy

from .engine import source_frames_match
from .ownership import write_hif_record
from .checkpoint import transition_checkpoint


def _same(event, gate):
    pending = gate.pending
    return (pending is not None and event.get('target_key') == pending.target.key
        and event.get('target_kind') == pending.target.kind
        and event.get('source_revision') == pending.source.revision
        and event.get('issued_at_ms') == gate.issued_at_ms)


def _transition_identity(record):
    # JSON stores expected-scene tuples as arrays. Compare the persisted
    # representation, retaining every original field except the STOP flag.
    return json.loads(json.dumps({key: value for key, value in (record or {}).items()
        if key != 'stopped'}, allow_nan=False))


def _current_source(report, gate):
    frames = report.get('frames')
    return (isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], dict)
        and source_frames_match(gate.pending.source, [frames[0].get('frame_id')]))


def persist_input_event(host, event, *, complete=False, write_record=None):
    gate, page = host.hooks.transition_gate, host._page
    if (not host._active or not _same(event, gate) or page is None
            or page.observation != gate.pending.source):
        raise ValueError('route input receipt lacks its exact original pending page')
    # Only this branch changes. Keep large, unchanged probe/budget histories
    # read-only through the same atomic full-record serialization.
    candidate = dict(host.handoff or {})
    context = dict(candidate.get('nonbattle_context') or {})
    candidate['nonbattle_context'] = context
    if complete:
        carrier = deepcopy(context.get('route_input_receipt'))
        if (not isinstance(carrier, dict) or carrier.get('dispatch_state') != 'prepared'
                or _transition_identity(carrier.get('original_transition')) != _transition_identity(transition_checkpoint(gate, host.clock.epoch))
                or len(carrier.get('report', {}).get('route', {}).get('input_events', ())) != 1
                or not _same(carrier['report']['route']['input_events'][0], gate)):
            raise ValueError('route dispatch result differs from its original prepared receipt')
        carrier['report']['route']['input_events'] = [deepcopy(event)]
        carrier['dispatch_state'] = ('cancelled_unissued' if event.get('rejected_before_dispatch') is True
            and event.get('dispatch_attempted') is False and event.get('success') is False and event.get('uncertain') is False
            else 'succeeded' if event.get('dispatch_attempted') is True
            and event.get('success') is True and event.get('uncertain') is False else 'uncertain')
        context['route_input_receipt'] = carrier
    else:
        source = host._input_source(page) if host._input_source is not None else [dict(frame_id=key) for key in page.frame_ids]
        if (not source_frames_match(page.observation, page.frame_ids)
                or [row.get('frame_id') for row in source] != list(page.frame_ids)):
            raise ValueError('route source registration does not match its original frames')
        capture = re.fullmatch(r'route-frame-' + str(host.game_pid) + r'-([0-9]+)', page.frame_ids[0])
        report_age = page.observation.revision - int(capture[1]) if capture else page.observation.capture_age_ms
        carrier = dict(schema_version=1, dispatch_state='prepared', original_transition=transition_checkpoint(gate, host.clock.epoch),
            recovery_reads=0, report=dict(id=f'task600-auto-{gate.issued_at_ms}',
                endpoints=dict(game=dict(pid=host.game_pid)), frames=deepcopy(source),
                route=dict(run_id=host.run_id, input_events=[deepcopy(event)], events=[dict(
                    phase=page.observation.scene, frame_ids=list(page.frame_ids),
                    capture_age_ms=report_age)])))
        context['route_input_receipt'] = carrier
    envelope = dict(schema_version=3, run_id=host.run_id, transition=transition_checkpoint(gate, host.clock.epoch),
        ledger=host.ledger.record() if host.ledger is not None else None,
        ownership=host.ownership, handoff=candidate)
    # The ordinary host stores its own root transition. An explicitly bound
    # navigation child can retain this exact receipt inside the same envelope
    # while leaving the original business transition untouched.
    if write_record is None:
        write_hif_record(host.path, envelope)
    else:
        write_record(envelope)
    host.handoff = candidate


def partial_first_lesson(event, pending):
    """Only consumed taps of the one first-day lesson group can support readback."""
    attempted, succeeded = event.get('clicks_attempted'), event.get('clicks_succeeded')
    return (pending is not None and pending.source.day == 1 and pending.source.scene == 'lesson'
        and pending.target.kind == 'lesson' and event.get('clicks_planned') == 3
        and type(attempted) is int and type(succeeded) is int and 1 <= succeeded <= attempted <= 3
        and succeeded < 3 and event.get('dispatch_attempted') is True
        and event.get('success') is not True and event.get('uncertain') is True)


def original_input_report(host, gate):
    carrier = (host.handoff or {}).get('nonbattle_context', {}).get('route_input_receipt')
    if (not isinstance(carrier, dict) or carrier.get('schema_version') != 1
            or _transition_identity(carrier.get('original_transition')) != _transition_identity(transition_checkpoint(gate, host.clock.epoch))):
        raise ValueError('pending route input has no persisted successful original dispatch receipt')
    report = carrier.get('report')
    events = (report or {}).get('route', {}).get('input_events', ())
    if (not isinstance(report, dict) or report.get('route', {}).get('run_id') != host.run_id
            or report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid or len(events) != 1
            or not _same(events[0], gate) or not _current_source(report, gate)
            or events[0].get('dispatch_attempted') is not True):
        raise ValueError('persisted route input receipt does not belong to this original pending action')
    if (carrier.get('dispatch_state') != 'succeeded' or events[0].get('success') is not True
            or events[0].get('uncertain') is not False):
        if carrier.get('dispatch_state') == 'uncertain' and partial_first_lesson(events[0], gate.pending):
            return deepcopy(report)
        raise ValueError('persisted route input receipt does not belong to this original pending action')
    return deepcopy(report)


def cancelled_unissued_report(host, gate):
    """Read an exact SDK rejection; never treat a prepared/unknown input as it."""
    carrier = (host.handoff or {}).get('nonbattle_context', {}).get('route_input_receipt')
    if (not isinstance(carrier, dict) or carrier.get('schema_version') != 1
            or carrier.get('dispatch_state') != 'cancelled_unissued'
            or _transition_identity(carrier.get('original_transition'))
                != _transition_identity(transition_checkpoint(gate, host.clock.epoch))):
        raise ValueError('pending route input has no exact unissued rejection receipt')
    report = carrier.get('report') or {}
    events = report.get('route', {}).get('input_events', ())
    if (report.get('route', {}).get('run_id') != host.run_id
            or report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid or len(events) != 1
            or not _same(events[0], gate) or not _current_source(report, gate) or events[0].get('rejected_before_dispatch') is not True
            or events[0].get('dispatch_attempted') is not False or events[0].get('success') is not False
            or events[0].get('uncertain') is not False
            or not isinstance(events[0].get('rejection_reason'), str) or not events[0]['rejection_reason']):
        raise ValueError('persisted unissued receipt differs from its original pending action')
    return deepcopy(report)
