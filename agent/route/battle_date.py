"""Exact formal battle STOP -> known same-save navigation -> original battle.

Source loading and completion verification grant no input. The trusted route
scheduler checks separate permissions for every capture and
navigation click; the original native owner alone reconciles its business.
"""
from __future__ import annotations

from copy import deepcopy

from .hif import HifConfig
from .engine import source_frames_match, validate_observation
from .date_recovery import ACTION, bind_date_recovery_runtime, verify_completed_date_return


def _business_scope(record):
    # Every caller owns this freshly parsed record. Copy only the branches we
    # remove; unchanged history is read-only and never needs another deep copy.
    from .cultivation import _budget_container
    handoff = dict(record['handoff'])
    navigation = handoff.get('nonbattle_context')
    if isinstance(navigation, dict):
        navigation = handoff['nonbattle_context'] = dict(navigation)
        for key in ('date_recovery', 'route_input_receipt'):
            navigation.pop(key, None)
    current = handoff
    for _ in range(20):
        if 'automatic_cultivation' in current:
            break
        nested = current.get('nonbattle_context')
        if not isinstance(nested, dict):
            break
        current['nonbattle_context'] = current = dict(nested)
    container = _budget_container(handoff)
    if container is not None:
        container.pop('automatic_cultivation', None)
    return dict(run_id=record['run_id'], ownership=record['ownership'], ledger=record['ledger'], handoff=handoff)


def pending_navigation(record):
    state = (record.get('handoff') or {}).get('nonbattle_context', {}).get('date_recovery')
    return isinstance(state, dict) and state.get('completed') is False


def recover_battle_date(runtime, request, outcome=None):
    """Return True only after supported navigation completed in the same scope.

    A first ordinary STOP with no supported current page remains that STOP.
    An existing child has no such fallback: its original pending is preserved
    and any failure ends the current invocation rather than entering battle.
    """
    from live_decision.hif_battle_entry import load_battle_navigation_source

    record = runtime._read()
    business = _business_scope(record)
    cold = pending_navigation(record)
    try:
        source = load_battle_navigation_source(runtime.runtime, request, record, outcome)
    except ValueError as error:
        if cold:
            raise
        runtime.report.setdefault('battle_date_rejections', []).append(str(error))
        return False
    preparation = runtime.authorise_navigation_preparation
    def guard():
        current = runtime._read()
        if _business_scope(current) != business:
            raise ValueError('original-save navigation business owner, ledger or handoff changed')
        if runtime.context.tasker.stopping:
            raise ValueError('original-save navigation permissions were revoked')
        if preparation is not None:
            if preparation() is not True:
                raise ValueError('original-save navigation preparation was revoked')
        else:
            if (runtime.authorise_capture() is not True or runtime.authorise_input() is not True
                    or runtime.authorise_battle() is not True):
                raise ValueError('original-save navigation permissions were revoked')
            runtime._size()
        return True
    guard()
    bound = bind_date_recovery_runtime(runtime.context, path=runtime.path, run_id=runtime.run_id,
        handoff_id=request.handoff_id, game_pid=runtime.game_pid, client_size=runtime._size(),
        source_report=source['source_report'], authorise_capture=guard, authorise_input=guard,
        evidence_sink=runtime.evidence_sink, navigation_guard=guard,
        **(dict(authorise_preparation=guard) if preparation is not None else {}))
    host, producer = bound.gkh_hif_date_recovery_host, bound.gkh_hif_date_recovery_producer
    if not cold:
        try:
            proof = producer.observe(reconciliation=True)
        except ValueError as error:
            runtime.report.setdefault('battle_date_rejections', []).append(str(error))
            return False
        seen = proof.observation
        if (seen.scene not in ACTION or validate_observation(seen) is not None
                or not source_frames_match(seen, proof.frame_ids)
                or not 0 <= seen.capture_age_ms+runtime.clock.now()-seen.revision <= 1000):
            runtime.report.setdefault('battle_date_rejections', []).append('no complete fresh supported interruption')
            return False
        # Record the concrete interruption before dispatching its navigation.
        runtime._checkpoint(opening=True)
    decision = None
    try:
        decision = host.run(bound, HifConfig(mode='advance'), max_actions=None, timeout_ms=None)
    except ValueError as error:
        rejected = [row for row in host.input_events if row.get('rejected_before_dispatch') is True
            and row.get('dispatch_attempted') is False and isinstance(row.get('rejection_reason'), str)]
        if rejected:
            primary = rejected[-1]['rejection_reason']
            runtime.report.setdefault('battle_date_followup_errors', []).append(str(error))
            raise ValueError(primary) from error
        raise
    finally:
        attempted = [row for row in host.input_events if row.get('dispatch_attempted') is True]
        if type(runtime.report['input_count']) is int:
            runtime.report['input_count'] += len(attempted)
        for key, flag in (('input_success_count', 'success'), ('input_uncertain_count', 'uncertain')):
            if type(runtime.report[key]) is int:
                runtime.report[key] += sum(row.get(flag) is True for row in attempted)
        runtime.report.setdefault('battle_date_navigation', []).append(dict(stopped_record=source['stopped_record'],
            source_result=source['source_report']['id'], cold=cold,
            decision=decision.reason if decision is not None else 'battle_date_navigation_exception',
            input_events=deepcopy(host.input_events)))
    guard()
    if decision.reason not in ('saved_live_requires_original_business_reconciliation',
            'saved_terminal_requires_original_business_reconciliation'):
        raise ValueError(decision.reason)
    receipt = verify_completed_date_return(runtime._read(), source['source_report'])
    runtime.report['battle_date_navigation'][-1]['completion'] = receipt
    runtime.bound = None
    return True
