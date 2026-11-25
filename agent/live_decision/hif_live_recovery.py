"""Bounded battle recovery in existing transaction reports, without UI authority.

Completed checkpoints survive a later zero-input policy STOP. Failed actions
still require a recorded timing failure or independently verified repair.
The normal checkpoint loader must
still verify the original selection/dispatch, rules and cache before observation.
"""
from __future__ import annotations

import re
import json
import math
import time
from copy import deepcopy
from functools import wraps
from contextvars import ContextVar

from .hif_native.native_selected_identity_resume import is_unsubmitted_neighbor_identity_stop

VERSION = 1
MAX_RECOVERIES = 2  # Historical receipt field, independent of retry authorization.
TERMINAL_PENDING = 'HIF result heading does not confirm its explicit round'

_AUTOMATIC_HOST = ContextVar('hif_automatic_host', default=False)


class BattleProgress:
    """miyu's battle._check_battle_progress clock, shared by normal UI polls."""

    def __init__(self):
        from .exam_loop import ExamRecovery

        self.exam = ExamRecovery()
        self.last_progress = time.monotonic()
        self.recovery_deadline = None
        self.remaining = None
        self.communication = None

    def confirmed(self, remaining=None, *, played_or_used=False):
        changed = remaining is not None and remaining != self.remaining
        if remaining is not None:
            self.remaining = remaining
        if changed or played_or_used:
            self.exam.record(no_effect=False)
            self.last_progress = time.monotonic()
            self.recovery_deadline = None

    def poll(self):
        now = time.monotonic()
        if now - self.last_progress < 120:
            return
        if self.recovery_deadline is None:
            # Normal page handling keeps running during this period. Polls and
            # clicks alone cannot extend it or authorise another submission.
            self.recovery_deadline = now + 30
        elif now >= self.recovery_deadline:
            raise ValueError('HIF battle recovery ended without confirmed progress')

    def retry_network(self, context, image, *, click):
        """Use the shared Hiro retry while HIF retains its audited input path."""
        from .inventory import _inventory_ocr
        from .hiro_maa_exam import ExamCommunication
        from .hif_capture_adapter import captured_at

        if getattr(context.tasker, 'stopping', False):
            return False
        if self.communication is None:
            controller = context.tasker.controller
            latest = None

            def capture():
                nonlocal latest
                latest = controller.post_screencap().wait().get()
                return latest

            def tap(x, y):
                click((x, y, 1, 1), captured_at(controller, latest), 'network_retry')

            self.communication = ExamCommunication(capture=capture, tap=tap, retry_only=True,
                ocr=lambda frame, **kwargs: _inventory_ocr(context, frame, 'Task275HifBattleNetworkRetry', **kwargs))
        start = time.monotonic()
        if self.communication.recover(image) is None:
            return False
        paused = time.monotonic() - start
        self.last_progress += paused
        if self.recovery_deadline is not None:
            self.recovery_deadline += paused
        return True


def automatic_battle_run(function):
    """Apply the ordinary battle budget to nested cold and hot recovery calls."""
    @wraps(function)
    def run(*args, **kwargs):
        token = _AUTOMATIC_HOST.set(kwargs.get('automatic_recovery') is True)
        try:
            return function(*args, **kwargs)
        finally:
            _AUTOMATIC_HOST.reset(token)
    return run


def automatic_host_active():
    return _AUTOMATIC_HOST.get()


def identity(session):
    if session.authorise_input() is not True:
        raise ValueError('HIF automatic recovery requires the active owner')
    record = session.record
    handoff = record['handoff']
    from .hif_battle_process_binding import accounting_identity

    current = dict(route_run_id=record['run_id'], handoff_id=handoff['handoff_id'],
        ownership_revision=record['ownership']['revision'], game_pid=session.game_pid,
        ledger_revision=handoff['ledger_revision'], policy_profile=handoff['policy_profile'])
    return accounting_identity(record, current)


def validate_parent_deadline(deadline):
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0):
        raise ValueError('HIF parent deadline must be a finite absolute timestamp')


def constrain_parent_deadline(control, deadline):
    """Add a tighter parent bound; keep the original battle budget immutable."""
    validate_control(control)
    validate_parent_deadline(deadline)
    if automatic_host_active():
        return  # Saved engineering receipts remain historical evidence.
    if deadline is not None:
        from .hif_battle_engineering_window import _completion_parent
        repair = control.get('engineering_repair') or {}
        if repair.get('version') == 1 and not _completion_parent(control) and control.get('opening_parent_window') is None:
            # Count-only repairs retain their original parent verbatim. A new
            # fixed-host parent is bound by append_followup, not by overwriting
            # the parent inside already validated historical receipts.
            return
        field = 'active_parent_deadline' if repair.get('version') == 2 or control.get('opening_parent_window') is not None or _completion_parent(control) else 'parent_deadline'
        control[field] = min(deadline, control.get(field, deadline))


def new_control(session, *, maximum_actions, timeout_seconds, now=None, parent_deadline=None):
    now = time.time() if now is None else now
    control = dict(version=VERSION, identity=identity(session), root=None, previous_record=None,
        sequence=0, windows_used=0, recoveries_used=0, maximum_actions=maximum_actions,
        timeout_seconds=timeout_seconds, maximum_recoveries=MAX_RECOVERIES,
        started_at=now, deadline=now + timeout_seconds, last_checked_at=now)
    constrain_parent_deadline(control, parent_deadline)
    return control


def require_new_opening(runtime, session):
    """An unfinished identity/opening attempt cannot reset the battle budget."""
    handoff_id = identity(session)['handoff_id']
    from .hif_recovery_reports import matching_recovery_reports

    for path in matching_recovery_reports(runtime, handoff_id):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('handoff_id') == handoff_id:
            raise ValueError('HIF incomplete opening cannot start a new automatic battle budget')


def validate_control(control):
    _validated_window_limits(control)


def _validated_window_limits(control):
    from .hif_battle_engineering_window import window_limits

    if (not isinstance(control, dict) or control.get('version') != VERSION
            or not isinstance(control.get('identity'), dict)
            or type(control.get('maximum_actions')) is not int or not 1 <= control['maximum_actions'] <= 128
            or type(control.get('timeout_seconds')) is not int or not 1 <= control['timeout_seconds'] <= 7200
            or any(type(control.get(key)) is not int or control[key] < 0
                   for key in ('sequence', 'windows_used', 'recoveries_used', 'maximum_recoveries'))
            or control['sequence'] != control['windows_used']
            or any(type(control.get(key)) not in (int, float) or not math.isfinite(control[key])
                   for key in ('started_at', 'deadline'))
            or control['deadline'] - control['started_at'] != control['timeout_seconds']):
        raise ValueError('HIF automatic recovery control is incomplete or incompatible')
    if 'parent_deadline' in control and control['parent_deadline'] is None:
        raise ValueError('HIF saved parent deadline cannot be null')
    validate_parent_deadline(control.get('parent_deadline'))
    deadline, recovery_limit = window_limits(control)
    from .hif_battle_engineering_window import _completion_parent
    if ('active_parent_deadline' in control and (control.get('engineering_repair') or {}).get('version') != 2
            and control.get('opening_parent_window') is None and not _completion_parent(control)):
        raise ValueError('HIF active parent requires its fixed engineering repair')
    checked = control.get('last_checked_at', control['started_at'])
    if type(checked) not in (int, float) or not math.isfinite(checked) or checked < control['started_at']:
        raise ValueError('HIF automatic recovery last clock observation is invalid')
    return deadline, recovery_limit


def require_deadline(control, *, now=None):
    deadline, _ = _validated_window_limits(control)
    if automatic_host_active():
        return
    now = time.time() if now is None else now
    deadline = min(deadline, control.get('active_parent_deadline', control.get('parent_deadline', deadline)))
    if (type(now) not in (int, float) or not math.isfinite(now)
            or not control.get('last_checked_at', control['started_at']) <= now < deadline):
        raise ValueError('HIF automatic battle deadline exhausted or clock moved backwards')
    control['last_checked_at'] = now


def require_budget(control, *, now=None):
    require_deadline(control, now=now)
    if not automatic_host_active() and control['windows_used'] >= control['maximum_actions']:
        raise ValueError('HIF automatic battle action limit exhausted')


def business_actions_used(rows, runtime):
    """Charge a transaction unless its SDK evidence accounts for observation alone."""
    from .hif_native.native_choice_resume import has_confirmed_choice, is_pending_retrieval_stop
    from .hif_native.native_opening_intent_resume import is_original_saved_opening_intent

    spent = 0
    for row in rows:
        events = row.get('live_events') or []
        inputs = row.get('native_input_dispatches')
        count = row.get('input_count')
        submitted = any(event.get(key) for event in events for key in (
            'execution_submit_count', 'end_turn_submit_count', 'choice_submit_count', 'drinks_consumed', 'executed'))
        journal = row.get('choice_journal') or {}
        choice = journal.get('choice') or {}
        if not row.get('native_choice_source') and (journal.get('joint_event') or {}).get('execution_submit_count'):
            submitted = True
        parent = row.get('native_choice_source')
        receipt = row.get('handoff_resume') or {}
        inherited = (parent and receipt.get('source') == parent and receipt.get('boundary_kind') == 'confirmed_retrieval'
            and has_confirmed_choice(journal))
        if (row.get('status') not in ('STOP', 'RULE_MAINTAINED_NATIVE_TRANSACTION', 'TERMINAL_VERIFIED')
                or type(count) is not int or not isinstance(inputs, list) or count != len(inputs)
                or submitted or not inherited and (choice.get('confirmation_submitted')
                    or choice.get('choice_confirm_started') or choice.get('choice_submit_count'))):
            spent += 1
            continue
        if is_original_saved_opening_intent(row) or (
                row.get('native_choice_source') and (row.get('handoff_resume') or {}).get('boundary_kind') == 'pending_retrieval'
                and is_pending_retrieval_stop(row, runtime=runtime)):
            continue
        possible = any(event.get('input_may_have_been_dispatched') for event in events)
        preparing = any(isinstance(value, dict) and value.get('dispatch_state') != 'rejected_before_dispatch'
            for key, value in row.items() if key.startswith('native_') and key.endswith('input_preparation'))
        if not count:
            navigation = row.get('native_navigation_preparation') or {}
            spent += int(possible or preparing or bool(navigation and navigation.get('dispatch_state') != 'rejected_before_dispatch'))
            continue
        opening = [event for event in events if event.get('event') == 'live_opening_state']
        # Input while opening a failed, still-unbound state belongs to reads.
        # A later decision, confirmation, or unknown post-opening attempt counts.
        if (len(opening) == 1 and opening[0].get('accepted') is False
                and all(e.get('executed') is False and e.get('execution_submit_count') == 0 for e in events)
                and not possible and not row.get('upstream_ui_journal') and not row.get('gray_selection_journal')
                and not journal and not row.get('mcts_search_cache')
                and (row.get('native_input_preparation') or {}).get('dispatch_state') in (None, 'rejected_before_dispatch')):
            continue
        if not possible and not preparing and (inherited or (row.get('native_pending_source')
                and receipt.get('boundary_kind') == 'submitted_pending' and not journal)):
            continue
        spent += 1
    return spent


def closed_observer_record(row):
    """A synchronous observer returned; this does NOT mean its action settled."""
    control = row.get('automatic_battle')
    if control is None:
        return False
    validate_control(control)
    closed = row.get('observer_returned')
    if closed is None:
        return False
    from .hif_battle_process_binding import migrated_identity
    from .hif_native.native_pending_outcome import is_unconsumed_observation

    observed_identity = migrated_identity(control['identity'], row.get('process_migrations', []))

    if (closed != dict(record=row.get('id'), sequence=control['sequence'])
            or row.get('information_boundary_execution') is not True
            or type(row.get('live_return')) is not bool
            or row.get('status') not in ('STOP', 'RULE_MAINTAINED_NATIVE_TRANSACTION', 'TERMINAL_VERIFIED')
            or row.get('frame_persistence_error')
            or row.get('error') and not (is_unconsumed_observation(row)
                and row.get('input_count') == 0 and row.get('native_input_dispatches') == [])
            or row.get('handoff_id') != control['identity'].get('handoff_id')
            or row.get('endpoints', {}).get('game', {}).get('pid') != observed_identity.get('game_pid')):
        raise ValueError('HIF observer completion receipt is inconsistent')
    return True


def reported_pending_stop(row):
    """Select the settlement loader; the submission itself still needs its SDK receipt."""
    from .hif_native.native_settlement_resume import reports_pending_submission

    return row.get('status') == 'STOP' and reports_pending_submission(row) and closed_observer_record(row)


def _unchanged_selected_stale_stop(report, refreshes, steps, stop, slot, selected):
    """Recognize the old generic stop only from the complete unchanged A/B proof."""
    if (len(refreshes) != 2 or selected != 1 or stop.get('attempted_slot') != slot
            or stop.get('reason') not in ('authoritative_redecision_changed', 'selected_input_evidence_stale')
            or stop.get('input_may_have_been_dispatched') not in (None, False)
            or any(step.get('phase') not in ('execution_selection', 'hif_selected_single_plus',
                   'hif_selected_skill_support', 'execution_redecision_stop')
                   or step.get('input_may_have_been_dispatched') is True for step in steps)):
        return False
    first, final = refreshes
    proposal = first.get('proposal', {})
    action = proposal.get('candidate_traces', {}).get('high_score', {}).get('action', {})
    generation = first.get('generation')
    selection = next((step for step in steps if step.get('phase') == 'execution_selection'), {})
    return (selection.get('accepted') is True
        and isinstance(generation, dict) and bool(generation)
        and final.get('generation') == final.get('expected_generation') == generation
        and action.get('kind') == 'PLAY_CARD' and action.get('source_slot') == slot
        and isinstance(action.get('card_instance_id'), str) and bool(action['card_instance_id'])
        and all(event.get('proposal', {}).get('status') == 'PROPOSE'
            and event['proposal'].get('action') == dict(kind='PLAY_CARD', card_slot=slot)
            and event['proposal'].get('candidate_traces', {}).get('high_score', {}).get('action') == action
            for event in (*refreshes, report))
        and first.get('select_consensus', {}).get('accepted') is True
        and first['select_consensus'].get('observations') == [[], [], []]
        and final.get('select_consensus', {}).get('accepted') is True
        and final['select_consensus'].get('selected_slot') == slot
        and final['select_consensus'].get('observations') == [[slot], [slot], [slot]])


def _selected_stale(row, reports):
    """Retain an unsubmitted intent only when every failed B expired, not identity."""
    from .hif_native.native_card_evidence import input_window_limit

    if len(reports) != 1:
        return False
    report = reports[0]
    selected = report.get('execution_selection_click_count')
    slot = report.get('current_selected_slot')
    safe = report.get('safe_decision', {})
    if (report.get('event') != 'live_decision_observe' or report.get('executed') is not False
            or report.get('execution_submit_count') != 0 or report.get('end_turn_submit_count', 0) != 0
            or type(selected) is not int or selected not in (0, 1)
            or type(slot) is not int or slot < 0
            or (selected == 0 and not row.get('native_selected_source'))
            or type(row.get('input_count')) is not int or row['input_count'] != selected
            or len(row.get('native_input_dispatches', ())) != selected
            or report.get('identity_probe_click_count') != 0
            or report.get('unknown_fields') != [] or report.get('contradictory_fields') != []
            or safe.get('status') != 'STOP' or safe.get('reason') != 'stale_state'
            or report.get('resource_baseline', {}).get('accepted') is not True):
        return False
    refreshes = report.get('authoritative_refresh_events', ())
    if len(refreshes) == 1 and selected == 0:
        return _restored_selected_a_stale(row, report, refreshes[0], slot)
    if len(refreshes) not in (2, 3) or refreshes[0].get('phase') != 'authoritative_refresh_a':
        return False
    if refreshes[0].get('accepted') is not True:
        return False
    for refresh in refreshes[1:]:
        age = refresh.get('capture_age_ms')
        identities = refresh.get('identity_revalidation', ())
        if (refresh.get('phase') != 'authoritative_refresh_b' or refresh.get('accepted') is not False
                or refresh.get('reason') != 'stale_state' or refresh.get('click_dispatched') is not False
                or refresh.get('expected_selected_slot') != slot
                or type(age) not in (int, float) or not math.isfinite(age) or age <= input_window_limit(refresh)
                or refresh.get('geometry_stable') is not True
                or refresh.get('resource_transaction', {}).get('accepted') is not True
                or not identities or any(identity.get('accepted') is not True for identity in identities)):
            return False
    steps = report.get('execution_selection_events', ())
    stops = [step for step in steps if step.get('phase') == 'execution_redecision_stop']
    selections = [step for step in steps if step.get('phase') == 'execution_selection']
    if (len(stops) != 1 or not (
            (stops[0].get('reason') == 'fresh_b_retry_binding_unresolved'
             and stops[0].get('input_may_have_been_dispatched') is False)
            or _unchanged_selected_stale_stop(report, refreshes, steps, stops[0], slot, selected))
            or len(selections) != 1 or selections[0].get('attempted_slot') != slot
            or selections[0].get('click_dispatched') is not bool(selected)
            or selections[0].get('reason') != ('target_selection_dispatched' if selected else 'target_already_selected')
            or any(step.get('click_dispatched') is not False for step in steps if step not in selections)
            or any(isinstance(step.get('receipt'), dict) and step['receipt'].get('accepted') is False
                   for step in steps)):
        return False
    # B's historical input_may_have_been_dispatched includes the prior selection;
    # only the explicit submit-stop receipt proves no business dispatch here.
    for key in ('native_input_preparation', 'native_navigation_preparation'):
        dispatch = row.get(key, {}).get('dispatch_state')
        if dispatch == 'dispatch_uncertain' or not selected and dispatch == 'dispatch_attempted':
            return False
    return True


def _restored_selected_a_stale(row, report, refresh, slot):
    """Keep a verified selected source when its first new check only expires."""
    from .hif_native.native_card_evidence import input_window_limit

    age = refresh.get('capture_age_ms')
    identities = refresh.get('identity_revalidation', ())
    proposal = report.get('proposal', {})
    action = proposal.get('candidate_traces', {}).get('high_score', {}).get('action', {})
    selected = refresh.get('select_consensus', {})
    return (not report.get('execution_selection_events')
        and all(type(report.get(key)) is int and report[key] == 0 for key in (
            'selection_click_count', 'execution_click_count', 'total_input_click_count'))
        and not row.get('native_input_preparation') and not row.get('native_navigation_preparation')
        and refresh.get('phase') == 'authoritative_refresh_a' and refresh.get('accepted') is False
        and refresh.get('reason') == 'stale_state' and refresh.get('click_dispatched') is False
        and refresh.get('input_may_have_been_dispatched') is False
        and type(age) in (int, float) and math.isfinite(age) and age > input_window_limit(refresh)
        and refresh.get('expected_selected_slot') == slot and refresh.get('geometry_stable') is True
        and refresh.get('selected_identity_authorized') is True
        and refresh.get('resource_transaction', {}).get('accepted') is True
        and identities and all(identity.get('accepted') is True for identity in identities)
        and selected.get('accepted') is True and selected.get('selected_slot') == slot
        and selected.get('observations') == [[slot], [slot], [slot]]
        and action.get('kind') == 'PLAY_CARD' and action.get('source_slot') == slot
        and isinstance(action.get('card_instance_id'), str) and bool(action['card_instance_id'])
        and all(event.get('proposal', {}).get('status') == 'PROPOSE'
            and event['proposal'].get('action') == dict(kind='PLAY_CARD', card_slot=slot)
            and event['proposal'].get('candidate_traces', {}).get('high_score', {}).get('action') == action
            for event in (refresh, report)))


def _selected_preparation_expired(row, reports):
    """An already selected intent with complete A/B but no final SDK dispatch."""
    from .hif_visual import hif_selection_click_box
    from .hif_native.native_card_evidence import _fresh, _frames
    from .hif_native.native_inprocess_resume import _same_input_host
    from .hif_native.native_settlement_resume import _verified_endpoints

    if (len(reports) != 1 or not row.get('native_selected_source')
            or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
            or row.get('native_navigation_preparation')):
        return False
    report = reports[0]
    if (report.get('event') != 'live_decision_observe' or report.get('executed') is not False
            or report.get('unknown_fields') != [] or report.get('contradictory_fields') != []
            or report.get('foreground_preparation_events') or report.get('shadow_transaction_events')
            or any(report.get(key, 0) != 0 for key in ('execution_submit_count', 'execution_selection_click_count',
                'execution_click_count', 'selection_click_count', 'total_input_click_count',
                'identity_probe_click_count', 'end_turn_submit_count', 'drinks_consumed'))):
        return False
    action = report['proposal']['candidate_traces']['high_score']['action']
    slot = action['source_slot']
    refreshes = report.get('authoritative_refresh_events', ())
    if (action.get('kind') != 'PLAY_CARD' or type(slot) is not int or slot < 0
            or not action.get('card_instance_id') or report.get('current_selected_slot') != slot
            or len(refreshes) not in (2, 3)):
        return False
    first, final = refreshes[0], refreshes[-1]
    groups = []
    for index, refresh in enumerate(refreshes):
        _fresh(refresh, slot)
        groups.append(_frames(refresh))
        if (refresh.get('phase') != ('authoritative_refresh_b' if index else 'authoritative_refresh_a')
                or refresh.get('click_dispatched') is not False
                or refresh.get('expected_selected_slot') != slot or refresh.get('geometry_stable') is not True
                or refresh.get('generation') != first.get('generation')
                or index and refresh.get('expected_generation') != first.get('generation')
                or refresh['proposal']['candidate_traces']['high_score']['action'] != action
                or refresh.get('select_consensus', {}).get('observations') != [[slot]]*3
                or refresh.get('resource_transaction', {}).get('accepted') is not True
                or not refresh.get('identity_revalidation')
                or any(item.get('accepted') is not True for item in refresh['identity_revalidation'])
                or any(refresh.get(key) != count for key,count in
                    (('health_votes',3),('playable_votes',3),('secondary_votes',0),('popup_votes',0)))):
            return False
    if len(set().union(*groups)) != 3*len(groups):
        return False
    steps = report.get('execution_selection_events', ())
    expected = [dict(phase='execution_selection',attempted_slot=slot,accepted=True,
        click_dispatched=False,reason='target_already_selected')]
    submits = [step for step in steps if step.get('phase') == 'execution_submit']
    if len(submits) != len(refreshes)-1:
        return False
    before_preparation = (len(submits) == 1 and not row.get('native_input_preparation')
        and submits[0] == dict(phase='execution_submit', attempted_slot=slot,
            click_dispatched=False, reason='snapshot_expired_before_input'))
    for index,(refresh,step) in enumerate(zip(refreshes[1:],submits,strict=True)):
        if not before_preparation and (step != dict(phase='execution_submit',attempted_slot=slot,authorization_floor=step.get('authorization_floor'),
                click_dispatched=False,input_may_have_been_dispatched=False,
                reason='snapshot_expired_after_input_preparation')
                or not refresh['authorization_floor'] <= step['authorization_floor'] < refresh['input_deadline']):
            return False
        if index:
            previous = refreshes[index]
            if refresh['evidence_window_started_at'] < previous['input_deadline']:
                return False
            expected.append(dict(phase='execution_submit_reobservation',attempted_slot=slot,
                click_dispatched=False,input_may_have_been_dispatched=False,
                reason='final_preparation_expired_without_dispatch',retry_count=1,prior_frame_ids=previous['frame_ids']))
        expected.append(step)
    auxiliary = {'hif_selected_model_evidence_reuse', 'hif_selected_skill_support', 'hif_selected_single_plus'}
    if ([step for step in steps if step.get('phase') not in auxiliary] != expected
            or any(step.get('click_dispatched') is not False
                or 'receipt' in step and step['receipt'].get('accepted') is not True
                for step in steps if step.get('phase') in auxiliary)):
        return False
    if before_preparation:
        _verified_endpoints(row['endpoints'], row['endpoints']['game']['pid'])
        return True
    guard = row.get('native_input_preparation', {})
    if (guard.get('dispatch_state') != 'rejected_before_dispatch'
            or guard.get('input_may_have_been_dispatched') is not False
            or guard.get('reason') != submits[-1]['reason'] or 'authorization_checked_at' in guard
            or guard.get('input_deadline') != final['input_deadline']
            or guard.get('authorization_floor') != submits[-1]['authorization_floor']):
        return False
    boxes = final['candidate_boxes'][-1]
    points = {(box[0]+box[2]//2,box[1]+box[3]//2) for box in (boxes[slot],hif_selection_click_box(boxes,slot))}
    _verified_endpoints(guard['endpoints'], row['endpoints']['game']['pid'])
    _same_input_host(row['endpoints'], guard['endpoints'])
    return (guard.get('x'),guard.get('y')) in points


def _selected_capture_expired(row, reports):
    """A zero-input restored selection can reacquire after observation expiry."""
    from .hif_native.native_card_evidence import _frames, evidence_window_limit
    from .hif_native.native_settlement_resume import _verified_endpoints

    if (len(reports) != 1 or not row.get('native_selected_source')
            or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
            or row.get('native_input_preparation') or row.get('native_navigation_preparation')):
        return False
    report = reports[0]
    if (report.get('event') != 'live_selection_stop' or report.get('executed') is not False
            or report.get('runtime_phase') != 'authoritative_refresh_a'
            or report.get('reason') != 'authoritative_live_evidence_window_expired'
            or report.get('input_may_have_been_dispatched') is not False
            or any(report.get(key, 0) != 0 for key in ('execution_selection_click_count',
                'execution_submit_count', 'identity_probe_click_count', 'end_turn_submit_count', 'drinks_consumed'))
            or any(report.get(key) for key in ('authoritative_refresh_events', 'execution_selection_events',
                'foreground_preparation_events', 'shadow_transaction_events'))
            or len(report.get('failure_events', ())) != 1):
        return False
    failure = report['failure_events'][0]
    slot, age = failure.get('expected_selected_slot'), failure.get('evidence_window_age_ms')
    probes = report.get('identity_probe_events', ())
    if any(probe.get('click_dispatched') is True or probe.get('input_may_have_been_dispatched') is True
            or probe.get('phase') not in ('initial_observation', 'hif_selected_detail_identity',
                'retained_complete_hand_identity', 'hif_candidate_precompute')
            or probe.get('phase') == 'hif_selected_detail_identity' and (
                probe.get('accepted') is not True or probe.get('slot') != slot) for probe in probes):
        return False
    _frames(failure)
    boxes = failure.get('candidate_boxes', ())
    if (failure.get('phase') != 'authoritative_refresh_a' or failure.get('accepted') is not False
            or failure.get('reason') != report['reason'] or failure.get('click_dispatched') is not False
            or failure.get('input_may_have_been_dispatched') is not False
            or type(slot) is not int or slot < 0 or len(boxes) != 3
            or any(len(group) <= slot or group != boxes[0] for group in boxes)
            or any(failure.get(key) != value for key,value in (
                ('health_votes',3),('playable_votes',3),('secondary_votes',0),('popup_votes',0)))
            or type(age) not in (int,float) or not math.isfinite(age)
            or age <= evidence_window_limit(failure, row['formal_params']['max_snapshot_age_ms'])):
        return False
    _verified_endpoints(row['endpoints'], row['endpoints']['game']['pid'])
    return True


def is_unsubmitted_timing_stop(row):
    """A held selection may repeat a zero-input read, never replay a submit."""
    from .hif_native.native_handoff_resume import is_cleared_selection_timing_stop
    from .hif_native.native_selected_resume import selected_reobservation_has_no_input

    if is_cleared_selection_timing_stop(row):
        return True
    try:
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or row.get('transaction_closed') or row.get('error') or row.get('failure_event')
                or not closed_observer_record(row)):
            return False
        if selected_reobservation_has_no_input(row):
            return True
        events = row['live_events']
        if (_selected_stale(row, events) or _selected_preparation_expired(row, events)
                or _selected_capture_expired(row, events)):
            return True  # Existing complete A/B timing cases retain their contract.
        if len(events) != 1:
            return False
        report = events[0]
        if (report.get('event') != 'live_selection_stop' or report.get('executed') is not False
                or report.get('reason') != 'authoritative_live_evidence_window_expired'
                or report.get('runtime_phase') != 'execute'
                or report.get('execution_submit_count') != 0
                or report.get('execution_selection_click_count') != 1
                or report.get('identity_probe_click_count') != 0
                or report.get('end_turn_submit_count', 0) != 0
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches', ())) != 1
                or not row.get('native_inprocess_source') or row.get('native_navigation_preparation')):
            return False
        first, stale = report['authoritative_refresh_events']
        retry, expired = report['failure_events']
        action = first['proposal']['candidate_traces']['high_score']['action']
        slot = action['source_slot']
        if (action['kind'] != 'PLAY_CARD' or type(slot) is not int or slot < 0
                or not action.get('card_instance_id')
                or first.get('phase') != 'authoritative_refresh_a' or first.get('accepted') is not True
                or stale.get('phase') != 'authoritative_refresh_b' or stale.get('accepted') is not False
                or stale.get('click_dispatched') is not False
                or stale.get('reason') != 'stale_state' or stale.get('expected_selected_slot') != slot
                or stale.get('generation') != stale.get('expected_generation')
                or stale.get('generation') != first.get('generation')
                or stale['proposal']['candidate_traces']['high_score']['action'] != action
                or stale.get('geometry_stable') is not True
                or stale.get('resource_transaction', {}).get('accepted') is not True
                or not stale.get('identity_revalidation')
                or any(item.get('accepted') is not True for item in stale['identity_revalidation'])
                or stale.get('select_consensus', {}).get('accepted') is not True
                or stale['select_consensus'].get('observations') != [[slot]]*3
                or stale['select_consensus'].get('selected_slot') != slot):
            return False
        if (retry != dict(phase='execution_submit_reobservation',attempted_slot=slot,
                click_dispatched=False,input_may_have_been_dispatched=False,
                reason='initial_b_stale_after_selection',retry_count=1,prior_frame_ids=stale['frame_ids'])
                or expired.get('phase') != 'authoritative_refresh_b' or expired.get('accepted') is not False
                or expired.get('reason') != 'authoritative_live_evidence_window_expired'
                or expired.get('expected_selected_slot') != slot or expired.get('click_dispatched') is not False
                or any(type(expired.get(key)) is not int or expired[key] != count for key,count in (
                    ('health_votes',3),('playable_votes',3),('secondary_votes',0),('popup_votes',0)))):
            return False
        from .hif_native.native_card_evidence import input_window_limit, evidence_window_limit

        for age, limit in ((stale.get('capture_age_ms'), input_window_limit(stale)),
                (expired.get('evidence_window_age_ms'), evidence_window_limit(expired))):
            if type(age) not in (int,float) or not math.isfinite(age) or age <= limit:
                return False
        frame_groups = [value['frame_ids'] for value in (first,stale,expired)]
        if (any(len(ids) != 3 or len(set(ids)) != 3 or any(not isinstance(v,str) or not v for v in ids)
                for ids in frame_groups) or len(set().union(*frame_groups)) != 9):
            return False
        steps = report['execution_selection_events']
        selection = steps[0]
        if (selection.get('phase') != 'execution_selection' or selection.get('accepted') is not True
                or selection.get('attempted_slot') != slot or selection.get('click_dispatched') is not True
                or selection.get('reason') != 'target_selection_dispatched'
                or any(step.get('phase') not in ('hif_selected_single_plus','hif_selected_skill_support')
                    or step.get('click_dispatched') is not False or step.get('receipt',{}).get('accepted') is not True
                    for step in steps[1:])):
            return False
        from .hif_native.native_inprocess_resume import _same_input_host
        from .hif_native.native_settlement_resume import _verified_endpoints
        guard, dispatch = row['native_input_preparation'], row['native_input_dispatches'][0]
        if (guard.get('dispatch_state') != 'dispatch_attempted' or dispatch.get('kind') is not None
                or (guard.get('x'),guard.get('y')) != (dispatch.get('x'),dispatch.get('y'))):
            return False
        for endpoints in (guard['endpoints'],dispatch['endpoints']):
            _verified_endpoints(endpoints,row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'],endpoints)
        return True
    except (KeyError,ValueError,TypeError,AttributeError,IndexError):
        return False


def _repaired_model_selection_layout(row, report, reference, slot):
    """Replay retained detector/pixel evidence when older model reads lack boxes.

    No SELECT, identity or resource observation is manufactured here. The
    normal resumed A/B reader must still establish each of those from new frames.
    """
    failure, = report['failure_events']
    selection = failure['selection_evidence']
    retry, = report['execution_selection_events'][1:]
    if (failure.get('phase') != 'hif_selected_single_plus' or failure.get('accepted') is not False
            or failure.get('click_dispatched') is not False
            or failure.get('reason') != 'selected support target lacks independent unique SELECT evidence'
            or selection.get('geometry_stable') is not False or selection.get('select_consensus') is not None
            or selection.get('select_evidence') != []
            or retry.get('phase') != 'hif_support_selection_reobservation'
            or retry.get('reason') != 'selection_transition_geometry_unstable'
            or retry.get('click_dispatched') is not False):
        return None
    groups = [retry['frame_ids'],selection['frame_ids']]
    return groups if repaired_selected_layout_frames(row,groups,reference,slot) else None


def repaired_selected_layout_frames(row, groups, reference, slot, *, reference_selected=False):
    """Check retained geometry after a repair, never supply fresh input proof."""
    import cv2
    from card_selection.model import frame_identifier, isolate_card_candidates

    from .hif_visual import hif_selected_row_proposal

    ids = [fid for group in groups for fid in group]
    frames = row['frames'][-6:]
    raw = [r['raw'] for r in row['live_raw_fields'] if r.get('node') == 'ProduceRecognitionCards'][-6:]
    if (len(groups) != 2 or any(len(g) != 3 for g in groups) or len(set(ids)) != 6
            or len(raw) != 6 or [f['frame_id'] for f in frames] != ids):
        return False
    for frame,detector in zip(frames,raw,strict=True):
        image = cv2.imread(frame['path'])
        if image is None or frame_identifier(image) != frame['frame_id'] or not detector.get('filtered'):
            return False
        proposal = hif_selected_row_proposal(image,
            isolate_card_candidates(detector['all'],include_unplayable=True),hand_count=len(reference))
        if proposal is None or proposal[1] != slot or len(proposal[0]) != len(reference):
            return False
        for index,(old,card) in enumerate(zip(reference,proposal[0],strict=True)):
            new = card.box
            if (list(new[2:]) != old[2:] or abs(new[0]-old[0]) > 2
                    or not (18 <= old[1]-new[1] <= 30 if index == slot and not reference_selected
                        else abs(new[1]-old[1]) <= 2)):
                return False
    return True


def is_unsubmitted_selected_layout_stop(row):
    """A held selection with unresolved detector frames, not a changed hand.

    This preserves a recovery boundary only. Neither empty row is accepted or
    replaced by an old row; a resumed observer must establish fresh full A/B.
    """
    try:
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or row.get('transaction_closed') or row.get('error') or row.get('failure_event')
                or not closed_observer_record(row) or len(row['live_events']) != 1
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches', ())) != 1
                or not row.get('native_inprocess_source') or row.get('native_navigation_preparation')):
            return False
        report = row['live_events'][0]
        model_layout = (report.get('runtime_phase') == 'selected_single_plus_prepare'
            and report.get('reason') == 'selected_single_plus_unresolved: selected support target lacks independent unique SELECT evidence')
        if (report.get('event') != 'live_selection_stop' or report.get('executed') is not False
                or not (model_layout or report.get('runtime_phase') == 'authoritative_refresh_b'
                    and report.get('reason') == 'authoritative_live_hand_count_changed')
                or report.get('execution_submit_count') != 0
                or report.get('execution_selection_click_count') != 1
                or report.get('identity_probe_click_count') != 0
                or report.get('end_turn_submit_count', 0) != 0 or report.get('drinks_consumed', 0) != 0
                or report.get('foreground_preparation_events') != []):
            return False
        first, = report['authoritative_refresh_events']
        failures = report['failure_events']
        action = first['proposal']['candidate_traces']['high_score']['action']
        slot = action['source_slot']
        count = first['generation']['hand_count']
        if (type(count) is not int or not 2 <= count <= 5 or type(slot) is not int or not 0 <= slot < count
                or action.get('kind') != 'PLAY_CARD' or not action.get('card_instance_id')
                or first.get('phase') != 'authoritative_refresh_a' or first.get('accepted') is not True
                or first.get('unselected_geometry_authorized') is not True
                or first.get('geometry_stable') is not True or first.get('expected_selected_slot') is not None
                or first.get('click_dispatched') is not False or first.get('input_may_have_been_dispatched') is not False
                or first['select_consensus'].get('accepted') is not True
                or first['select_consensus'].get('observations') != [[],[],[]]
                or first['safe_decision'].get('status') != 'ALLOW'
                or first['safe_decision'].get('action') != dict(kind='PLAY_CARD',card_slot=slot)
                or len(first['identity_revalidation']) != count
                or any(item.get('accepted') is not True for item in first['identity_revalidation'])
                or len(failures) != (1 if model_layout else 2)):
            return False
        reference = first['candidate_boxes'][-1]
        if len(first['candidate_boxes']) != 3 or any(len(boxes) != count for boxes in first['candidate_boxes']):
            return False
        from .hif_native.native_card_evidence import input_evidence_is_fresh

        if not input_evidence_is_fresh(first):
            return False
        if model_layout:
            observed_groups = _repaired_model_selection_layout(row,report,reference,slot)
            if observed_groups is None:
                return False
        else:
            observed_groups = [value['frame_ids'] for value in failures]
        for attempt,failure in enumerate(() if model_layout else failures):
            rows = failure['candidate_boxes']
            complete = [boxes for boxes in rows if len(boxes) == count]
            if (failure.get('phase') != 'authoritative_refresh_b' or failure.get('accepted') is not False
                    or failure.get('reason') != 'authoritative_live_hand_count_changed'
                    or failure.get('expected_selected_slot') != slot or failure.get('click_dispatched') is not False
                    or any(type(failure.get(key)) is not int or failure[key] != value for key,value in (
                        ('health_votes',3),('playable_votes',3),('secondary_votes',0),('popup_votes',0)))
                    or len(rows) != 3 or any(len(boxes) not in (0,count) for boxes in rows)
                    or len(complete) not in ((2,) if attempt == 0 else (1,2))):
                return False
            for boxes in complete:
                for index,(old,new) in enumerate(zip(reference,boxes,strict=True)):
                    if (len(old) != 4 or len(new) != 4 or any(type(v) is not int for v in (*old,*new))
                            or old[2:] != [190,250] or new[2:] != old[2:] or abs(new[0]-old[0]) > 2
                            or not (18 <= old[1]-new[1] <= 30 if index == slot else abs(new[1]-old[1]) <= 2)):
                        return False
        groups = [first['frame_ids'],*observed_groups]
        if (any(len(ids) != 3 or len(set(ids)) != 3 or any(not isinstance(v,str) or not v for v in ids)
                for ids in groups) or len(set().union(*groups)) != 9):
            return False
        steps = report['execution_selection_events']
        selection = steps[0]
        if (selection.get('phase') != 'execution_selection' or selection.get('accepted') is not True
                or selection.get('attempted_slot') != slot or selection.get('click_dispatched') is not True
                or selection.get('reason') != 'target_selection_dispatched'
                or not model_layout and any(step.get('phase') not in ('hif_selected_single_plus','hif_selected_skill_support')
                    or step.get('click_dispatched') is not False or step.get('receipt',{}).get('accepted') is not True
                    for step in steps[1:])):
            return False
        from .hif_native.native_inprocess_resume import _same_input_host
        from .hif_native.native_settlement_resume import _verified_endpoints
        guard,dispatch = row['native_input_preparation'],row['native_input_dispatches'][0]
        if (guard.get('dispatch_state') != 'dispatch_attempted' or dispatch.get('kind') is not None
                or (guard.get('x'),guard.get('y')) != (dispatch.get('x'),dispatch.get('y'))):
            return False
        for endpoints in (guard['endpoints'],dispatch['endpoints']):
            _verified_endpoints(endpoints,row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'],endpoints)
        return True
    except (KeyError,ValueError,TypeError,AttributeError,IndexError):
        return False


def recovery_candidate(row):
    """Classify a proven timing failure, not a generic STOP or unknown image."""
    if (row.get('status') != 'STOP' or row.get('live_return') is not False
            or row.get('transaction_closed') or row.get('error') or row.get('failure_event')
            or not closed_observer_record(row)):
        return None
    events = row.get('live_events', ())
    if is_unsubmitted_timing_stop(row):
        return dict(source=row['id'], kind='selected_stale_input')
    if is_unsubmitted_selected_layout_stop(row):
        return dict(source=row['id'], kind='selected_unresolved_layout')
    if is_unsubmitted_neighbor_identity_stop(row):
        return dict(source=row['id'], kind='selected_neighbor_identity')
    if any(e.get('event') in ('live_decision_exception_stop', 'live_selection_stop', 'live_decision_stop')
           for e in events):
        return None
    if row.get('upstream_ui_journal') is not None:
        from .high_score.contracts import ActionKind
        from .hif_native.native_settlement_resume import decoded_dispatched_action

        try:
            action, report = decoded_dispatched_action(row)
            failure, submitted = events
            if (action.kind is ActionKind.END_TURN and report is submitted
                    and row.get('input_count') == len(row['native_input_dispatches']) == 1
                    and failure.get('event') == 'live_committed_state' and failure.get('accepted') is False
                    and failure.get('reason') == TERMINAL_PENDING and failure.get('terminal') is True
                    and failure.get('input_click_count') == failure.get('execution_submit_count') == 0
                    and submitted.get('accepted') is False and submitted.get('settlement') == failure
                    and submitted.get('end_turn_submit_count') == submitted.get('execution_submit_count') == 1):
                return dict(source=row['id'], kind='submitted_terminal_result_not_ready')
        except (KeyError, TypeError, ValueError):
            return None
    failures = [e for e in events if e.get('event') == 'live_committed_state' and e.get('accepted') is False]
    if any(e.get('accepted') is False and e not in failures for e in events):
        return None
    if not failures:
        reports = [e for e in events if e.get('event') == 'live_decision_observe']
        if (any(e.get('accepted') is False or e.get('execution_submit_count', 0)
                or e.get('end_turn_submit_count', 0) or e.get('drinks_consumed', 0) for e in events)
                or not _selected_stale(row, reports)):
            return None
        return dict(source=row['id'], kind='selected_stale_input')
    if len(failures) != 1 or failures[0].get('execution_submit_count') != 0:
        return None
    failure = failures[0]
    if (failure.get('reason') != TERMINAL_PENDING or failure.get('terminal') is not True
            or failure.get('input_click_count') != 0):
        return None
    kind = 'submitted_terminal_result_not_ready'
    submissions = [e for e in events if e.get('execution_submit_count', 0) or e.get('end_turn_submit_count', 0)]
    if submissions:
        if (len(submissions) != 1 or row.get('native_pending_source')
                or sum(submissions[0].get(k, 0) for k in ('execution_submit_count', 'end_turn_submit_count')) != 1):
            return None
        return dict(source=row['id'], kind=kind)
    # Further result-page observations still use the original submission.
    if (row.get('native_pending_source') and row.get('input_count') == 0
            and row.get('native_input_dispatches') == []):
        return dict(source=row['native_pending_source'], kind=kind)
    return None


def is_submitted_terminal_stop(row):
    """Reuse the existing pending-terminal classification for window consumers.

    This grants no settlement: the normal checkpoint reader must reconstruct
    the original dispatch and prove that all its outcomes are terminal.
    """
    candidate = recovery_candidate(row)
    return candidate is not None and candidate['kind'] == 'submitted_terminal_result_not_ready'


def retry_source(row):
    candidate = recovery_candidate(row)
    return candidate['source'] if candidate is not None else None


def verify_recovery_checkpoint(row, checkpoint, *, readback=None, runtime=None, context=None):
    if readback is not None:
        from .hif_native.native_observed_identity_resume import (
            is_post_commit_identity_stop,
            require_repaired_post_commit_identity,
        )

        if row != checkpoint.get('row') or not is_post_commit_identity_stop(row):
            raise ValueError('HIF separate readback requires the original settled observer STOP')
        return require_repaired_post_commit_identity(checkpoint,
            readback=readback, runtime=runtime, context=context)
    from .hif_native.native_handoff_policy import phase, pending_action
    from .hif_native.native_pending_outcome import is_unconsumed_observation
    from .hif_native.native_selected_resume import selected_reobservation_has_no_input
    from .hif_native.native_inprocess_resume import stopped_policy_decision, is_settled_policy_continuation
    from .hif_native.native_opening_intent_resume import is_original_saved_opening_intent

    if is_original_saved_opening_intent(row):
        report = checkpoint.get('cache_report') or {}
        if (checkpoint.get('row') != row or checkpoint.get('untouched_opening') is not True
                or checkpoint.get('selected') or checkpoint.get('pending')
                or report.get('restored') is not True or report.get('new_search_performed') is not False):
            raise ValueError('HIF saved opening intent differs from its original completed search')
        pending_action(checkpoint['policy'], checkpoint['state'])
        return

    if selected_reobservation_has_no_input(row):
        # The loader has followed the original selection and checked the full
        # calculation cache. A failed read contributes no action or input
        # credentials, so bind the unchanged pending action to that journal.
        source = checkpoint.get('row') or {}
        journal = source.get('gray_selection_journal') or {}
        if (not is_unsubmitted_timing_stop(row) or source.get('id') != row.get('id')
                or source.get('native_selected_source') != row['native_selected_source']
                or checkpoint.get('selected') is not True or checkpoint.get('pending')
                or checkpoint.get('cache_report', {}).get('restored') is not True
                or journal.get('selection_ready') is not True
                or type(journal.get('selection_count')) is not int or journal['selection_count'] != 1
                or journal.get('business_submit_count', 0) != 0):
            raise ValueError('HIF selected reobservation lost its original selection')
        action = pending_action(checkpoint.get('policy'), checkpoint['state'])
        raw = journal['refresh_a']['proposal']['candidate_traces']['high_score']['action']
        slot = action.source_slot
        if (action.kind.value != 'PLAY_CARD' or type(slot) is not int
                or not 0 <= slot < len(checkpoint['state'].deck.hand)
                or action.card_instance_id != checkpoint['state'].deck.hand[slot].instance_id
                or any(getattr(action, key) != raw[key] for key in ('source_slot', 'card_instance_id'))
                or raw['kind'] != action.kind.value):
            raise ValueError('HIF selected reobservation changed its original pending action')
        return

    if is_settled_policy_continuation(row, checkpoint.get('row')):
        event = row['live_events'][0]
        trace = (None if event.get('event') == 'live_decision_exception_stop'
            else stopped_policy_decision(event)['trace'])
        policy = checkpoint['policy']
        if (checkpoint.get('selected') or checkpoint.get('pending') or checkpoint.get('untouched_opening')
                or phase(policy) != 'observed_successor' or policy._pending is not None
                or getattr(policy, '_window_incomplete', False)
                or trace is not None and getattr(policy, '_windows', None) != trace['planning_window_index']
                or trace is None and checkpoint.get('cache_report', {}).get('restored_complete_policy') is not True
                or checkpoint.get('cache_report', {}).get('restored') is not True):
            raise ValueError('HIF policy continuation lacks its complete saved computation')
        return

    if is_unconsumed_observation(row):
        if (checkpoint.get('reconciled_unconsumed') is not True or checkpoint.get('pending')
                or phase(checkpoint['policy']) != 'pending_intent'
                or checkpoint['row'] != row or checkpoint['policy']._pending is None):
            raise ValueError('unconsumed recovery lacks its verified unchanged intention')
        return
    from .hif_native.native_settlement_resume import (
        is_submitted_completion_stop,
        require_pending_completion_checkpoint,
    )
    if is_submitted_completion_stop(row):
        require_pending_completion_checkpoint(checkpoint)
        return
    if is_unsubmitted_neighbor_identity_stop(row):
        from .hif_native.native_selected_identity_resume import require_repaired_neighbor_identity

        require_repaired_neighbor_identity(checkpoint)
        return
    from .hif_native.native_observed_identity_resume import is_post_commit_identity_stop, require_repaired_post_commit_identity
    if is_post_commit_identity_stop(row):
        require_repaired_post_commit_identity(checkpoint)
        return
    from .hif_native.native_opening_mapping_resume import require_repaired_mapping, is_unsubmitted_mapping_stop

    if is_unsubmitted_mapping_stop(row):
        if (checkpoint.get('untouched_opening') is not True or checkpoint.get('pending')
                or checkpoint.get('selected') or checkpoint.get('row') != row):
            raise ValueError('HIF mapping recovery differs from its untouched opening')
        require_repaired_mapping(checkpoint)
        return
    from .hif_native.native_support_resume import require_repaired_support, is_unsubmitted_support_stop
    if is_unsubmitted_support_stop(row):
        require_repaired_support(checkpoint)
        return
    from .hif_native.native_support_resume import require_repaired_resource_preview, is_unsubmitted_resource_preview_stop
    if is_unsubmitted_resource_preview_stop(row):
        require_repaired_resource_preview(checkpoint)
        return
    from .hif_native.native_settlement_resume import (
        is_submitted_gallery_stop,
        is_submitted_hand_identity_stop,
        require_pending_gallery_checkpoint,
        require_pending_hand_identity_checkpoint,
    )
    if is_submitted_hand_identity_stop(row):
        require_pending_hand_identity_checkpoint(checkpoint)
        return
    if is_submitted_gallery_stop(row):
        require_pending_gallery_checkpoint(checkpoint)
        return
    candidate = recovery_candidate(row)
    if candidate is None:
        from .hif_native.native_settlement_resume import decoded_dispatched_action

        if reported_pending_stop(row):
            receipt = checkpoint.get('receipt') or {}
            if (checkpoint.get('pending') is not True or checkpoint.get('selected')
                    or checkpoint.get('expected') is None or checkpoint.get('bundle') is None
                    or receipt.get('boundary_kind') != 'submitted_pending' or receipt.get('source') != row['id']
                    or (checkpoint.get('dispatch_source') or {}).get('id') != row['id']
                    or decoded_dispatched_action(row, checkpoint['bundle'])[0] != checkpoint.get('action')):
                raise ValueError('HIF pending observation differs from its original submitted action')
            return receipt['boundary_kind']
        raise ValueError('HIF automatic recovery lacks a classified timing failure')
    if candidate['kind'] in ('selected_stale_input', 'selected_unresolved_layout'):
        valid = checkpoint.get('selected') is True and not checkpoint.get('pending')
        if _selected_capture_expired(row, row['live_events']):
            slot = row['live_events'][0]['failure_events'][0]['expected_selected_slot']
            if valid:
                try:
                    action = pending_action(checkpoint.get('policy'), checkpoint['state'])
                    valid = (action.kind.value == 'PLAY_CARD' and action.source_slot == slot
                        and action.card_instance_id == checkpoint['state'].deck.hand[slot].instance_id)
                except (ValueError, IndexError):
                    valid = False
        else:
            # Complete-but-stale selected A also carries the original action.
            # A restored flag alone cannot authorize a different cached intent.
            raw = (row['native_selection_rebind']['action'] if row.get('native_selection_rebind') else
                   row['live_events'][0]['authoritative_refresh_events'][0]['proposal']['candidate_traces']['high_score']['action'])
            if valid:
                try:
                    action = pending_action(checkpoint.get('policy'), checkpoint['state'])
                except ValueError:
                    valid = False
                else:
                    valid = (action.kind.value == raw['kind']
                        and action.source_slot == raw['source_slot']
                        and action.card_instance_id == raw['card_instance_id'])
    else:
        expected = checkpoint.get('expected')
        states = expected if isinstance(expected, (tuple, list)) else (expected,)
        terminal = candidate['kind'] == 'submitted_terminal_result_not_ready'
        valid = (checkpoint.get('pending') is True and bool(states)
                 and all(getattr(state, 'terminal', None) is terminal for state in states))
    if not valid:
        raise ValueError('HIF automatic recovery checkpoint differs from its recorded boundary')


def reserve_retry(control, row, *, previous=None):
    require_budget(control)
    candidate = recovery_candidate(row)
    if candidate is None:
        raise ValueError('HIF stop is not an automatically recoverable timing failure')
    if previous is not None and row['automatic_battle']['previous_record'] == previous['id']:
        prior = recovery_candidate(previous)
        if (prior is not None and prior['kind'] == candidate['kind']
                and (row.get('native_pending_source') or row.get('native_selected_source')) in (
                    previous['id'], prior['source'])
                and row.get('input_count') == 0 and row.get('native_input_dispatches') == []):
            raise ValueError('HIF repeated the same recovery failure without progress')
    control['recoveries_used'] += 1
    return candidate['source']


def load_automatic_chain(runtime, bound):
    """Read and validate the existing unique chain without granting a retry."""
    from .hif_recovery_reports import recovery_control_chain

    return recovery_control_chain(runtime, bound, lambda paths: _load_automatic_chain(paths, bound))


def load_completed_control_chain(runtime, session, completed):
    """Recheck the current Root anchor and every subsequent Native record.

    The already validated business completion is necessary but not sufficient:
    the current Root's retained execution proof must independently match its
    complete original Native control. No original counter or grant is renewed.
    """
    from types import SimpleNamespace

    from route.host import windows_route_clock
    from route.run_budget import validate_budget
    from route.cultivation import _budget_container
    from route.continuation import _validate_battle_execution_resume

    from .hif_recovery_reports import recovery_control_chain

    record = session.record
    bound = identity(SimpleNamespace(record=record, game_pid=session.game_pid,
        authorise_input=session.authorise_input))
    budget = (_budget_container(record['handoff']) or {}).get('automatic_cultivation') or {}
    clock = windows_route_clock()
    validate_budget(budget, run_id=record['run_id'], game_pid=session.game_pid,
        epoch=clock.epoch, now=clock.now(), max_windows=64, max_elapsed_ms=21_600_000)
    proofs = [window['battle_resume'] for window in budget.get('execution_windows', ())
        if isinstance(window.get('battle_resume'), dict)
        and window['battle_resume'].get('identity', {}).get('handoff_id') == bound['handoff_id']]
    if not proofs:
        return load_automatic_chain(runtime, bound)
    proof = proofs[-1]
    anchors = (completed['business_frontier']['pending_source'], completed.get('continuation_source'),
        *completed.get('continuation_sources', ()))
    if proof['source'] not in anchors:
        raise ValueError('completed frontier differs from the current Root control anchor')
    _validate_battle_execution_resume(proof, record['ownership'], run_id=record['run_id'])
    pattern = re.compile(r'([0-9]+)-'+re.escape(record['run_id'])+r'.*-live(?:-step-([0-9]+))?')

    def order(name):
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError('completed control anchor has an unsupported record identity')
        return int(match[1]), int(match[2] or 1)

    anchor = order(proof['source'])

    def sources():
        result, positions = [], set()
        for path in runtime.glob('result-*'+record['run_id']+'*-live*.json'):
            name = path.name[len('result-'):-len('.json')]
            position = order(name)
            if position < anchor:
                continue
            if position in positions:
                raise ValueError('completed control tail contains a forked record position')
            positions.add(position)
            result.append((position, path))
        result.sort()
        if not result or result[0][1].name != 'result-'+proof['source']+'.json':
            raise ValueError('completed control tail lost its Root anchor')
        return [path for _, path in result]

    rows = recovery_control_chain(runtime, bound,
        lambda paths: _load_automatic_chain(paths, bound, anchored=True),
        sources=sources, variant=('completed_root_anchor', proof['source']))
    # Run this against each current Root read, outside the historical data
    # reuse. A changed parent proof/owner never inherits a previous check.
    _validate_battle_execution_resume(proof, record['ownership'], source=rows[0], run_id=record['run_id'])
    return rows


def _load_automatic_chain(paths, bound, *, anchored=False):
    from .hif_recovery_reports import read_recovery_report, recovery_report_contains_handoff

    rows = {}
    for path in paths:
        if not recovery_report_contains_handoff(path, bound['handoff_id']):
            continue
        row = read_recovery_report(path)
        if row.get('handoff_id') != bound['handoff_id']:
            continue
        control = row.get('automatic_battle')
        if control is None:
            raise ValueError('HIF legacy battle needs its explicit recovery source; no automatic cold start')
        validate_control(control)
        if (control['identity'] != bound or path.name != f"result-{row.get('id')}.json"
                or re.fullmatch(r'[0-9]+-[a-z0-9-]+', row.get('id', '')) is None):
            raise ValueError('HIF automatic source ownership or identity differs')
        rows[row['id']] = row
    if not rows:
        raise ValueError('HIF started battle has no automatic recovery records')
    ordered = sorted(rows.values(), key=lambda row: row['automatic_battle']['sequence'])
    first = ordered[0]['automatic_battle']
    root = first['root'] if anchored else ordered[0]['id']
    previous = first['previous_record'] if anchored else None
    frozen = None
    last_retries = 0
    last_parent = None
    last_active_parent = None
    last_checked = None
    previous_control = None
    for sequence, row in enumerate(ordered, first['sequence'] if anchored else 1):
        control = row['automatic_battle']
        immutable = {key: control[key] for key in ('identity', 'started_at', 'deadline',
            'maximum_actions', 'timeout_seconds', 'maximum_recoveries')}
        if (control['sequence'] != sequence or control['root'] != root
                or control['previous_record'] != previous or frozen is not None and immutable != frozen
                or control['recoveries_used'] < last_retries):
            raise ValueError('HIF automatic transaction chain is forked or incomplete')
        parent = control.get('parent_deadline')
        active_parent = control.get('active_parent_deadline')
        if last_active_parent is not None and (active_parent is None or active_parent > last_active_parent):
            added = control.get('engineering_followups', [])
            old = (previous_control or {}).get('engineering_followups', [])
            from .hif_battle_engineering_window import _same_history, _completion_parent

            identity_renewal = (active_parent is not None
                and (control.get('opening_parent_window') is not None
                    and (control.get('engineering_repair') or {}).get('version') == 1
                    or _completion_parent(control))
                and len(added) == len(old)+1 and _same_history(added[:-1], old)
                and added[-1]['version'] == 2 and added[-1]['source'] == previous
                and added[-1]['grant']['parent_execution_window']['battle_resume']['boundary'] in ('unsubmitted_opening','post_commit_identity','selected_neighbor_identity','selected_input_timing','submitted_inventory_navigation','submitted_card_completion'))
            if not identity_renewal:
                raise ValueError('HIF active parent cannot be removed or extended')
        last_active_parent = active_parent
        checked = control.get('last_checked_at', control['started_at'])
        if (last_parent is not None and (parent is None or parent > last_parent)
                or last_checked is not None and checked < last_checked):
            raise ValueError('HIF automatic recovery cannot extend the parent bound or rewind its clock')
        from .hif_battle_engineering_window import verify_window_successor
        if previous_control is None and not anchored:
            if (control.get('engineering_window') is not None or control.get('engineering_repair') is not None
                    or control.get('opening_parent_window') is not None):
                raise ValueError('HIF battle engineering window has no original predecessor')
        elif previous_control is not None:
            verify_window_successor(previous_control, control, previous)
        previous_control = control
        frozen, previous, last_retries = immutable, row['id'], control['recoveries_used']
        last_parent, last_checked = parent, checked
    return ordered


def find_resume(runtime, session, *, maximum_actions, timeout_seconds, parent_deadline=None,
                engineering_resume=None, completed_checkpoint=None):
    """Choose the unique last record by its chain, never mtime or a good score."""
    bound = identity(session)
    ordered = (load_completed_control_chain(runtime, session, completed_checkpoint['completed_frontier'])
        if completed_checkpoint is not None else load_automatic_chain(runtime, bound))
    latest = ordered[-1]
    control = deepcopy(latest['automatic_battle'])
    if (control['maximum_actions'], control['timeout_seconds']) != (maximum_actions, timeout_seconds):
        raise ValueError('HIF automatic recovery cannot replace the original budget')
    constrain_parent_deadline(control, parent_deadline)
    if completed_checkpoint is not None and completed_checkpoint['completed_frontier'].get('continuation_source'):
        from .hif_native.native_pending_frontier import verify_completion_continuation

        proof = completed_checkpoint['completed_frontier']
        if latest['id'] != proof['continuation_source'] or not closed_observer_record(latest):
            raise ValueError('completed frontier continuation is not the current control boundary')
        verify_completion_continuation(latest, completed_checkpoint['row'], proof)
        from .hif_battle_engineering_window import window_limits

        limit, _ = window_limits(control)
        limit = min(limit, control.get('active_parent_deadline', control.get('parent_deadline', limit)))
        if time.time() >= limit and callable(engineering_resume):
            control = engineering_resume(latest, control)
        require_budget(control)
        return control, latest, completed_checkpoint['row']['id']
    from .hif_native.native_opening_mapping_resume import is_unsubmitted_mapping_stop
    mapping_stop = closed_observer_record(latest) and is_unsubmitted_mapping_stop(latest)
    from .hif_native.native_opening_intent_resume import is_original_saved_opening_intent
    identity_stop = is_original_saved_opening_intent(latest)
    from .hif_native.native_observed_identity_resume import is_post_commit_identity_stop
    identity_stop = identity_stop or is_post_commit_identity_stop(latest)
    from .hif_native.native_choice_resume import is_pending_retrieval_stop, is_confirmed_retrieval_stop

    retrieval_stop = is_pending_retrieval_stop(latest, runtime=runtime) or is_confirmed_retrieval_stop(latest)
    identity_stop = identity_stop or retrieval_stop
    from .hif_native.native_support_resume import is_unsubmitted_support_stop
    support_stop = is_unsubmitted_support_stop(latest)
    from .hif_native.native_support_resume import is_unsubmitted_resource_preview_stop
    preview_stop = is_unsubmitted_resource_preview_stop(latest)
    from .hif_native.native_settlement_resume import (
        is_submitted_gallery_stop,
        is_submitted_completion_stop,
        is_submitted_hand_identity_stop,
    )
    gallery_stop = (is_submitted_gallery_stop(latest) or is_submitted_hand_identity_stop(latest)
        or is_submitted_completion_stop(latest) or is_submitted_terminal_stop(latest))
    from .hif_battle_engineering_window import window_limits

    deadline, _ = window_limits(control)
    timing_stop = is_unsubmitted_timing_stop(latest)
    layout_stop = is_unsubmitted_selected_layout_stop(latest) or is_unsubmitted_neighbor_identity_stop(latest)
    needs_window = time.time() >= min(
        deadline, control.get('active_parent_deadline', control.get('parent_deadline', deadline)))
    known_boundary = (identity_stop or mapping_stop or support_stop or preview_stop
        or gallery_stop or timing_stop or layout_stop)
    if known_boundary and needs_window and callable(engineering_resume):
        control = engineering_resume(latest, control)
    require_budget(control)
    from .hif_native.native_inprocess_resume import is_settled_policy_continuation

    parent = next((row for row in ordered if row['id'] == latest.get('native_inprocess_source')), None)
    if is_settled_policy_continuation(latest, parent):
        # Keep the latest control/chain position and all spent work. Restore the
        # already committed parent, not the stopped proposal or its old inputs.
        source = parent['id']
    elif latest.get('transaction_closed') is True and latest.get('status') == 'RULE_MAINTAINED_NATIVE_TRANSACTION':
        source = latest['id']
    elif recovery_candidate(latest) is not None:
        source = reserve_retry(control, latest, previous=ordered[-2] if len(ordered) > 1 else None)
    elif known_boundary or reported_pending_stop(latest):
        # The ordinary checkpoint loader independently verifies this repaired state.
        control['recoveries_used'] += 1
        source = latest['id']
    else:
        raise ValueError('HIF stop is not an automatically recoverable timing failure')
    return control, latest, source
