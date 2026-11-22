"""Cultivation execution history, separate from action freshness.

Version 1 records wall time. Versions 2 and 3 exclude confirmed pauses.
Old limits and their receipts remain available for historical verification.
"""
from __future__ import annotations

import re
import json
import subprocess
from copy import deepcopy
from pathlib import Path


def _int(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_budget(budget, *, run_id, game_pid, epoch, now, max_windows=64, max_elapsed_ms=21_600_000):
    if (not isinstance(budget, dict) or type(budget.get('schema_version')) is not int
            or budget['schema_version'] not in (1, 2, 3)
            or budget.get('run_id') != run_id or budget.get('game_pid') != game_pid
            or budget.get('clock_epoch') != epoch or not _int(now) or not _int(budget.get('windows_used'))
            or not all(_int(budget.get(key)) for key in ('started_at_ms', 'last_at_ms'))
            or not budget['started_at_ms'] <= budget['last_at_ms'] <= now
            or budget.get('status') not in ('running', 'stopped', 'complete')
            or not isinstance(budget.get('process_ids'), list) or not budget['process_ids']
            or any(not _int(value, 1) for value in budget['process_ids'])):
        raise ValueError('automatic cultivation budget cannot be refreshed or reconstructed')
    if budget['schema_version'] == 3:
        if any(key in budget for key in ('max_windows', 'max_elapsed_ms', 'execution_windows', 'continuations')):
            raise ValueError('cultivation history contains an incompatible legacy allowance')
    elif budget.get('max_windows') != max_windows or budget.get('max_elapsed_ms') != max_elapsed_ms:
        raise ValueError('original cultivation budget limits were changed')
    if budget['schema_version'] == 1:
        if 'continuations' in budget or 'execution_windows' in budget:
            raise ValueError('legacy budget cannot grant its own continuation')
        return budget
    receipts = budget.get('pause_receipts')
    elapsed = budget.get('active_elapsed_ms')
    paused = budget.get('paused_at_ms')
    accounting = budget.get('accounting_state')
    if (not _int(elapsed) or not isinstance(receipts, list)
            or not _int(budget.get('last_process_id'), 1)
            or accounting not in ('active', 'paused', 'complete')
            or (accounting == 'paused') != (paused is not None)
            or budget['status'] == 'running' and accounting != 'active'
            or (budget['status'] == 'complete') != (accounting == 'complete')
            or paused is not None and (not _int(paused) or not budget['started_at_ms'] <= paused <= budget['last_at_ms'])):
        raise ValueError('automatic cultivation active-runtime accounting is incomplete')
    previous, excluded = budget['started_at_ms'], 0
    for row in receipts:
        if (not isinstance(row, dict) or not all(_int(row.get(key)) for key in ('from_ms', 'to_ms'))
                or not previous <= row['from_ms'] <= row['to_ms'] <= budget['last_at_ms']
                or not isinstance(row.get('evidence'), dict)
                or row['evidence'].get('kind') not in ('formal_runtime_stopped_checkpoint', 'audited_legacy_stopped_checkpoint')):
            raise ValueError('automatic cultivation pause receipt is not auditable')
        previous, excluded = row['to_ms'], excluded + row['to_ms'] - row['from_ms']
    if paused is not None:
        if paused < previous:
            raise ValueError('automatic cultivation open pause overlaps a closed receipt')
        excluded += budget['last_at_ms'] - paused
    if elapsed + excluded != budget['last_at_ms'] - budget['started_at_ms']:
        raise ValueError('automatic cultivation elapsed time is inconsistent with its pause receipts')
    if 'continuations' in budget or 'execution_windows' in budget:
        from .continuation import budget_limit
        budget_limit(budget, now=now)
    elif 'process_migrations' in budget:
        from .continuation import execution_process_migrations
        execution_process_migrations(budget, now=now)
    return budget


def remaining_ms(budget, now):
    """Remaining historical allowance for an old receipt or a diagnostic."""
    if not _int(now) or now < budget['last_at_ms']:
        raise ValueError('automatic cultivation clock moved backwards')
    if budget['schema_version'] == 3:
        return None
    if budget['schema_version'] == 1:
        spent = now - budget['started_at_ms']
    else:
        spent = budget['active_elapsed_ms'] + (now - budget['last_at_ms'] if budget['accounting_state'] == 'active' else 0)
    from .continuation import budget_limit
    return max(0, budget_limit(budget, now=now) - spent)


def advance_budget(budget, *, now, process_id, opening=False, reason=None, complete=False, pause_permitted=True):
    """Return a new candidate; failed persistence cannot consume/reset the old."""
    current = deepcopy(budget)
    # A stopped label/pending=None is not a lifecycle/permission pause proof.
    # Until such a complete receipt is provided, this new credit charges gaps.
    if 'continuations' in current or 'execution_windows' in current:
        pause_permitted = False
    if not _int(process_id, 1) or not _int(now) or now < current['last_at_ms']:
        raise ValueError('automatic cultivation clock or process identity is invalid')
    if current['status'] == 'complete':
        return current
    if current['schema_version'] in (2, 3):
        if current['accounting_state'] == 'active':
            current['active_elapsed_ms'] += now - current['last_at_ms']
        elif opening:
            current['pause_receipts'].append(dict(from_ms=current['paused_at_ms'], to_ms=now,
                evidence=dict(kind='formal_runtime_stopped_checkpoint', process_id=current['last_process_id'],
                    reason=current.get('last_reason'), windows_used=current['windows_used'])))
            current['paused_at_ms'] = None
            current['accounting_state'] = 'active'
        current['last_process_id'] = process_id
    if opening:
        current.update(windows_used=current['windows_used'] + 1, status='running')
    if process_id not in current['process_ids']:
        current['process_ids'].append(process_id)
    if complete:
        current['status'] = 'complete'
        if current['schema_version'] in (2, 3):
            # Completion is a terminal execution checkpoint, not an open pause.
            if current['paused_at_ms'] is not None:
                current['pause_receipts'].append(dict(from_ms=current['paused_at_ms'], to_ms=now,
                    evidence=dict(kind='formal_runtime_stopped_checkpoint', process_id=current['last_process_id'],
                        reason=current.get('last_reason'), windows_used=current['windows_used'])))
            current['paused_at_ms'] = None
            current['accounting_state'] = 'complete'
    elif reason is not None:
        current['status'] = 'stopped'
        if current['schema_version'] in (2, 3) and current['paused_at_ms'] is None and pause_permitted:
            current['paused_at_ms'] = now
            current['accounting_state'] = 'paused'
    current['last_at_ms'] = now
    if opening or reason is not None:
        current['last_reason'] = reason
    return current


def active_maa_processes():
    """Read process inventory only; never starts or controls Maa/the game."""
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(MaaPiCli|MaaGakumasu)\\.exe$' } "
        "| Select-Object Name,ProcessId | ConvertTo-Json -Compress")
    output = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', command], text=True)
    value = json.loads(output) if output.strip() else []
    return value if isinstance(value, list) else [value]


def audit_legacy_pause(*, original_record_path, current_record_path, stopped_report_path, clock=None, process_probe=None):
    """Read the actual source files and same-boot process witness for a preview."""
    from .host import windows_route_clock
    from .ownership import validate_ownership
    from .cultivation import _budget_container
    clock = clock or windows_route_clock()
    paths = [Path(path).resolve() for path in (original_record_path, current_record_path, stopped_report_path)]
    original, current, report = [json.loads(path.read_text(encoding='utf-8-sig')) for path in paths]
    if original != current:
        raise ValueError('legacy pause original host changed after its settled invocation')
    budget = (_budget_container(original.get('handoff')) or {}).get('automatic_cultivation') or {}
    cultivation = report.get('cultivation') or {}
    count = report.get('input_count')
    inputs = report.get('route', {}).get('input_events', ())
    stamps = [int(match[1]) for frame in report.get('frames', ())
        if (match := re.fullmatch(r'route-frame-' + str(budget.get('game_pid')) + r'-([0-9]+)', frame.get('frame_id', '')))]
    owner = original.get('ownership') or {}
    validate_ownership(owner)
    now = clock.now()
    active = (process_probe or active_maa_processes)()
    if (budget.get('schema_version') != 1 or budget.get('status') != 'stopped'
            or budget.get('clock_epoch') != clock.epoch or not _int(now) or now < budget.get('last_at_ms', now + 1)
            or original.get('run_id') != budget.get('run_id') or owner.get('game_pid') != budget.get('game_pid')
            or owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
            or original.get('transition', {}).get('pending') is not None
            or original.get('ledger', {}).get('pending') is not None
            or active != [] or paths[2].name != f"result-{report.get('id')}.json"
            or report.get('route', {}).get('run_id') != budget.get('run_id')
            or cultivation.get('run_id') != budget.get('run_id') or cultivation.get('status') != 'STOP'
            or cultivation.get('reason') != budget.get('last_reason')
            or not cultivation.get('windows') or cultivation['windows'][-1].get('ordinal') != budget.get('windows_used')
            or report.get('endpoints', {}).get('game', {}).get('pid') != budget.get('game_pid')
            or not _int(count) or count != len(inputs) or report.get('input_success_count') != count
            or report.get('input_uncertain_count') != 0
            or any(event.get('dispatch_attempted') is not True or event.get('success') is not True
                or event.get('uncertain') is not False or not _int(event.get('issued_at_ms'))
                or event['issued_at_ms'] > budget['last_at_ms'] for event in inputs)
            or not stamps or not 0 <= budget['last_at_ms'] - max(stamps) <= 120_000):
        raise ValueError('legacy pause lacks matching settled invocation, same-clock boundaries or an inactive route witness')
    evidence = dict(kind='audited_legacy_stopped_checkpoint', run_id=budget['run_id'], clock_epoch=clock.epoch,
        stopped_at_ms=budget['last_at_ms'], windows_used=budget['windows_used'], reason=budget.get('last_reason'),
        source_id=report['id'], source_paths=[str(path) for path in paths], last_original_capture_ms=max(stamps),
        host_snapshot=deepcopy(original),
        start_scope=dict(run_id=original['run_id'], ownership=deepcopy(owner), route_pending=None, ledger_pending=None, budget=deepcopy(budget)),
        stopped_report=dict(id=report['id'], run_id=cultivation['run_id'], status='STOP', reason=cultivation['reason'],
            input_count=count, input_success_count=count, input_uncertain_count=0),
        end_witness=dict(run_id=current['run_id'], ownership=deepcopy(owner), route_pending=None, ledger_pending=None,
            clock_epoch=clock.epoch, observed_at_ms=now, unchanged_original_record=True, active_maa_processes=[]))
    return budget, evidence, now


def legacy_migration_preview(budget, *, now, epoch, evidence=None, host_record=None):
    """Pure preview, with all pre-stop unknown time conservatively charged."""
    validate_budget(budget, run_id=budget['run_id'], game_pid=budget['game_pid'], epoch=epoch, now=now,
        max_windows=budget['max_windows'], max_elapsed_ms=budget['max_elapsed_ms'])
    if budget['schema_version'] != 1:
        raise ValueError('only an original version-one budget can be migrated once')
    snapshot = host_record if host_record is not None else (evidence or {}).get('host_snapshot')
    audit_evidence = {key: deepcopy(value) for key, value in (evidence or {}).items() if key != 'host_snapshot'}
    origin = (evidence or {}).get('start_scope') or {}
    witness = (evidence or {}).get('end_witness') or {}
    stopped = (evidence or {}).get('stopped_report') or {}
    owner = origin.get('ownership') or {}
    # A stopped label or file timestamp alone is insufficient: bind the
    # finished actual invocation, settled original scope and current absence
    # of an active route process. The preview producer keeps source paths.
    audited = (budget['status'] == 'stopped' and isinstance(evidence, dict)
        and evidence.get('kind') == 'audited_legacy_stopped_checkpoint'
        and evidence.get('run_id') == budget['run_id'] and evidence.get('clock_epoch') == epoch
        and evidence.get('stopped_at_ms') == budget['last_at_ms']
        and evidence.get('windows_used') == budget['windows_used']
        and evidence.get('reason') == budget.get('last_reason')
        and isinstance(evidence.get('source_id'), str) and bool(evidence['source_id'])
        and origin.get('run_id') == budget['run_id'] and origin.get('budget') == budget
        and origin.get('route_pending') is None and origin.get('ledger_pending') is None
        and owner.get('owner') == 'task600' and owner.get('phase') == 'nonbattle'
        and owner.get('game_pid') == budget['game_pid']
        and stopped.get('id') == evidence['source_id'] and stopped.get('run_id') == budget['run_id']
        and stopped.get('status') == 'STOP' and stopped.get('reason') == budget.get('last_reason')
        and _int(stopped.get('input_count')) and stopped.get('input_success_count') == stopped['input_count']
        and stopped.get('input_uncertain_count') == 0
        and witness.get('run_id') == budget['run_id'] and witness.get('clock_epoch') == epoch
        and witness.get('observed_at_ms') == now and witness.get('ownership') == owner
        and witness.get('route_pending') is None and witness.get('ledger_pending') is None
        and witness.get('unchanged_original_record') is True and witness.get('active_maa_processes') == [])
    current = deepcopy(budget)
    charged_until = budget['last_at_ms'] if audited else now
    current.update(schema_version=2, active_elapsed_ms=charged_until - budget['started_at_ms'],
        last_at_ms=now, last_process_id=budget['process_ids'][-1], pause_receipts=[], paused_at_ms=None,
        accounting_state='complete' if budget['status'] == 'complete' else 'active')
    if audited:
        current['pause_receipts'].append(dict(from_ms=charged_until, to_ms=now, evidence=deepcopy(audit_evidence)))
        current.update(status='stopped', paused_at_ms=now, accounting_state='paused')
    elif current['status'] == 'stopped':
        current['status'] = 'running'  # Without a pause proof, unknown time continues to count.
    current['legacy_migration'] = dict(original=deepcopy(budget), evidence=deepcopy(audit_evidence) if audited else None,
        previewed_at_ms=now, conservatively_charged_before_ms=charged_until,
        excluded_stopped_ms=now - charged_until)
    validate_budget(current, run_id=budget['run_id'], game_pid=budget['game_pid'], epoch=epoch, now=now,
        max_windows=budget['max_windows'], max_elapsed_ms=budget['max_elapsed_ms'])
    return dict(kind='legacy_active_runtime_migration_preview', budget=current,
        expected_host_record=deepcopy(snapshot),
        old_remaining_ms=remaining_ms(budget, now), proposed_remaining_ms=remaining_ms(current, now),
        wall_elapsed_ms=now - budget['started_at_ms'], active_elapsed_ms=current['active_elapsed_ms'],
        pause_exclusions_accepted=audited,
        rejection_reason=None if audited else 'no complete same-scope settled invocation and pause witnesses; unknown time charged',
        requires_review_before_application=True)
