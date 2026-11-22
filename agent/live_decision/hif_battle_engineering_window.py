"""One authorised cold engineering window, retained in the original battle chain."""
import math
from copy import deepcopy

AUTHORITY = 'fixed_native_battle_resume_20261104'
SECONDS = 7200
OPENING_SECONDS = 1800
COMPLETION_PARENT_SECONDS = 3600
OPENING_RENEWAL_MIN_SECONDS = 900
EXTRA_RECOVERIES = 2


def _same_history(left, right):
    # The factory already interns equal old source controls. Compare their
    # graph once per pair, rather than expanding that graph at every edge.
    # This retains every value check, not its result across invocations.
    from route.cultivation_factory import _same_private_json

    return _same_private_json(left, right)


def _opening_renewal_present(control):
    return (control.get('opening_parent_window') is not None
        and (control.get('engineering_repair') or {}).get('version') == 1
        and any(r.get('version') == 2 for r in control.get('engineering_followups', ())))


def _completion_parent(control):
    repair = control.get('engineering_repair') or {}
    return (repair.get('version') == 2 and repair.get('grant', {})
        .get('parent_execution_window', {}).get('battle_resume', {})
        .get('boundary') == 'submitted_card_completion'
        or repair.get('version') == 1 and any(row.get('version') == 2
            and row.get('grant', {}).get('parent_execution_window', {}).get('battle_resume', {})
                .get('boundary') == 'submitted_card_completion'
            for row in control.get('engineering_followups', ())))


def _compact_parent(control):
    """Both existing parent shapes retain their full prefix in the source file."""
    return (isinstance(control.get('opening_parent_window'), dict)
        and (control.get('engineering_repair') or {}).get('version') == 1
        or _completion_parent(control) and isinstance(control.get('engineering_window'), dict))


def window_limits(control):
    opening_time_deadline = _verify_opening_parent(control)
    window = control.get('engineering_window')
    if window is None:
        repair = control.get('engineering_repair')
        first_opening_completion = (isinstance(repair, dict) and repair.get('version') == 2
            and isinstance(control.get('opening_parent_window'), dict) and _completion_parent(control))
        if (repair is not None and (not isinstance(repair, dict)
                or repair.get('version') != 1 and not first_opening_completion)
                or 'engineering_followups' in control and repair is None):
            raise ValueError('HIF repair requires its original engineering window')
        deadline = max(control['deadline'], opening_time_deadline) if opening_time_deadline is not None else control['deadline']
        return _followup_limits(control, *_repair_limits(control, deadline, control['maximum_recoveries']))
    from .hif_live_recovery import validate_control

    fields = {'version', 'source', 'source_control', 'issued_at', 'deadline', 'seconds',
        'extra_recoveries', 'grant'}
    if (not isinstance(window, dict) or set(window) != fields or window['version'] != 1
            or window['seconds'] != SECONDS or window['extra_recoveries'] != EXTRA_RECOVERIES
            or not isinstance(window['source_control'], dict)
            or 'engineering_window' in window['source_control']
            or 'parent_deadline' not in window['source_control']):
        raise ValueError('HIF battle engineering window is incomplete')
    source, grant = window['source_control'], window['grant']
    validate_control(source)
    immutable = ('identity', 'root', 'started_at', 'deadline', 'maximum_actions',
        'timeout_seconds', 'maximum_recoveries')
    if (any(control[k] != source[k] for k in immutable)
            or control['sequence'] < source['sequence']
            or control['windows_used'] < source['windows_used']
            or control['recoveries_used'] < source['recoveries_used']
            or any(type(window[k]) not in (int, float) or not math.isfinite(window[k])
                for k in ('issued_at', 'deadline'))
            or not source.get('last_checked_at', source['started_at']) <= window['issued_at'] < window['deadline']
            or window['deadline'] != min(window['issued_at']+SECONDS, source['parent_deadline'])
            or not isinstance(grant, dict)):
        raise ValueError('HIF battle engineering window changed its original accounting')
    _verify_grant(control['identity'], window['source'], grant)
    return _followup_limits(control, *_repair_limits(control, window['deadline'], control['maximum_recoveries']+EXTRA_RECOVERIES))


def _verify_grant(bound, source_id, grant, *, parent_window=False, native_renewal=False):
    if not isinstance(grant, dict):
        raise ValueError('HIF engineering grant is missing')
    fields = {'version', 'authority', 'source', 'run_id', 'handoff_id', 'ownership', 'native_task_id', 'game_identity', 'endpoints'}
    if parent_window:
        fields |= {'parent_execution_window', 'parent_deadline'}
    if 'process_migrations' in grant:
        from .hif_battle_process_binding import migrated_identity

        fields.add('process_migrations')
        bound = migrated_identity(bound, grant['process_migrations'])
    if set(grant) != fields:
        raise ValueError('HIF engineering grant fields differ from its version')
    owner = dict(owner='task275', phase='battle', game_pid=bound['game_pid'], revision=bound['ownership_revision'])
    game, endpoints = grant.get('game_identity'), grant.get('endpoints')
    migrations = grant.get('process_migrations', [])
    if migrations and (not isinstance(game, dict) or any(game.get(key) != value
            for key, value in migrations[-1]['to_game'].items())):
        raise ValueError('HIF engineering grant changed its migrated game instance')
    if (grant.get('version') != (3 if native_renewal else 2 if parent_window else 1) or grant.get('authority') != AUTHORITY
            or grant.get('source') != source_id or not isinstance(source_id, str)
            or grant.get('run_id') != bound['route_run_id'] or grant.get('handoff_id') != bound['handoff_id']
            or grant.get('ownership') != owner or type(grant.get('native_task_id')) is not int
            or grant['native_task_id'] <= 0 or not isinstance(game, dict)
            or game.get('pid') != bound['game_pid'] or game.get('integrity_rid') != 0x3000
            or type(game.get('created_at_100ns')) is not int or game['created_at_100ns'] <= 0
            or not isinstance(game.get('executable'), str) or not game['executable']
            or not isinstance(endpoints, dict) or not {'game', 'maa', 'agent'} <= set(endpoints)
            or any(not isinstance(p, dict) or p.get('integrity_rid') != 0x3000 for p in endpoints.values())
            or any(endpoints['game'].get(k) != v for k, v in game.items())):
        raise ValueError('HIF battle engineering window lacks its fixed native grant')



def append_window(control, source, grant, *, now):
    from .hif_live_recovery import validate_control
    from .hif_native.native_support_resume import is_unsubmitted_resource_preview_stop
    from .hif_native.native_opening_mapping_resume import is_unsubmitted_mapping_stop

    validate_control(control)
    if ('engineering_window' in control or not (
            is_unsubmitted_mapping_stop(source) or is_unsubmitted_resource_preview_stop(source))
            or source['automatic_battle'].get('engineering_window') is not None
            or control['root'] != source['automatic_battle']['root']
            or 'parent_deadline' not in source['automatic_battle']
            or type(now) not in (int, float) or not math.isfinite(now)):
        raise ValueError('HIF battle engineering resume requires the original zero-submit STOP')
    candidate = deepcopy(control)
    candidate['last_checked_at'] = now
    candidate['engineering_window'] = dict(version=1, source=source['id'],
        source_control=deepcopy(source['automatic_battle']), issued_at=now,
        deadline=min(now+SECONDS, source['automatic_battle']['parent_deadline']),
        seconds=SECONDS, extra_recoveries=EXTRA_RECOVERIES, grant=deepcopy(grant))
    window_limits(candidate)
    return candidate


def verify_window_successor(previous, current, previous_id):
    """A persisted window is append-only; later restarts cannot renew it."""
    old_opening, new_opening = previous.get('opening_parent_window'), current.get('opening_parent_window')
    if old_opening is not None:
        if not _same_history(new_opening, old_opening):
            raise ValueError('HIF opening parent window was removed or renewed')
    elif new_opening is not None and (new_opening['source'] != previous_id
            or not _same_history(new_opening['source_control'], previous)):
        raise ValueError('HIF opening parent differs from its immediate original source')
    old_followups, new_followups = previous.get('engineering_followups', []), current.get('engineering_followups', [])
    if (len(new_followups) < len(old_followups) or not _same_history(new_followups[:len(old_followups)], old_followups)
            or len(new_followups) > len(old_followups)+1):
        raise ValueError('HIF engineering followup history was replaced or repeated')
    if len(new_followups) > len(old_followups):
        added = new_followups[-1]
        if added['source'] != previous_id or not _same_history(added['source_control'], _followup_source(previous)):
            raise ValueError('HIF engineering followup differs from its immediate source')
    old_repair, new_repair = previous.get('engineering_repair'), current.get('engineering_repair')
    if old_repair is not None:
        if not _same_history(new_repair, old_repair):
            raise ValueError('HIF engineering repair was removed or renewed')
    elif new_repair is not None and (new_repair['source'] != previous_id or not _same_history(new_repair['source_control'], previous)):
        raise ValueError('HIF engineering repair differs from its immediate source')
    old, new = previous.get('engineering_window'), current.get('engineering_window')
    if old is not None:
        if not _same_history(new, old):
            raise ValueError('HIF battle engineering window was removed or renewed')
    elif new is not None and (new['source'] != previous_id or not _same_history(new['source_control'], previous)):
        raise ValueError('HIF battle engineering window differs from its immediate source')


def _repair_limits(control, deadline, limit):
    repair = control.get('engineering_repair')
    if repair is None:
        return deadline, limit
    from .hif_live_recovery import validate_control
    fields = {'version', 'source', 'source_control', 'issued_at', 'extra_recoveries', 'grant'}
    version = repair.get('version') if isinstance(repair, dict) else None
    if version == 2:
        fields |= {'deadline', 'seconds', 'parent_deadline'}
    if (not isinstance(repair, dict) or set(repair) != fields or version not in (1, 2)
            or repair['extra_recoveries'] != 1 or not isinstance(repair['source_control'], dict)
            or 'engineering_repair' in repair['source_control']):
        raise ValueError('HIF engineering repair is incomplete or repeated')
    source = repair['source_control']
    validate_control(source)
    immutable = ('identity', 'root', 'started_at', 'deadline', 'maximum_actions', 'timeout_seconds',
        'maximum_recoveries', 'engineering_window', 'parent_deadline', 'opening_parent_window')
    if (any(control.get(k) != source.get(k) for k in immutable)
            or any(control[k] < source[k] for k in ('sequence', 'windows_used', 'recoveries_used'))
            or type(repair['issued_at']) not in (int, float) or not math.isfinite(repair['issued_at'])
            or repair['issued_at'] < source['last_checked_at']):
        raise ValueError('HIF engineering repair changed original accounting or deadline')
    _verify_grant(control['identity'], repair['source'], repair['grant'], parent_window=version == 2)
    if version == 2:
        boundary = repair['grant'].get('parent_execution_window', {}).get('battle_resume', {}).get('boundary')
        if boundary not in ('selected_unsubmitted', 'submitted_card_completion'):
            raise ValueError('HIF repair has an unsupported original boundary')
        _verify_parent_window(repair['grant'], repair['source'], source, boundary=boundary)
        parent = repair['parent_deadline']
        if (type(parent) not in (int, float) or not math.isfinite(parent)
                or not 0 < parent <= repair['grant']['parent_deadline']
                or repair['grant']['parent_deadline'] > repair['issued_at']+14_400
                or repair['seconds'] != SECONDS or repair['deadline'] != min(repair['issued_at']+SECONDS, parent)
                or not repair['issued_at'] < repair['deadline']
                or type(control.get('active_parent_deadline')) not in (int, float)
                or not math.isfinite(control['active_parent_deadline'])
                or (not (_completion_parent(control) and any(r.get('version') == 2
                        for r in control.get('engineering_followups', ())))
                    and not 0 < control['active_parent_deadline'] <= parent)):
            raise ValueError('HIF repair exceeds its new fixed parent window')
        deadline = repair['deadline']
    else:
        parent = source.get('active_parent_deadline', source['parent_deadline'])
        if ('active_parent_deadline' in control and not _opening_renewal_present(control) and not _completion_parent(control)
                and (source.get('opening_parent_window') is None or not 0 < control['active_parent_deadline'] <= parent)
                or not repair['issued_at'] < min(deadline, parent)):
            raise ValueError('HIF repair changed its original time bound')
    return deadline, limit+1


def append_repair(control, source, grant, *, now, parent_deadline=None):
    from .hif_live_recovery import require_budget
    from .hif_native.native_support_resume import is_unsubmitted_support_stop, is_unsubmitted_resource_preview_stop
    from .hif_native.native_settlement_resume import is_submitted_completion_stop
    from .hif_native.native_opening_intent_resume import is_saved_opening_intent_stop

    submitted = is_submitted_completion_stop(source)
    if ('engineering_repair' in control or control != source.get('automatic_battle')
            or not (submitted or is_unsubmitted_support_stop(source) or is_unsubmitted_resource_preview_stop(source)
                or is_saved_opening_intent_stop(source))):
        raise ValueError('HIF repair requires its unchanged verified source')
    candidate = deepcopy(control)
    version = 2 if isinstance(grant, dict) and grant.get('version') == 2 else 1
    if version == 1:
        require_budget(candidate, now=now)
    else:
        _verify_parent_window(grant, source['id'], control,
            boundary='submitted_card_completion' if submitted else 'selected_unsubmitted')
        if type(parent_deadline) not in (int, float) or not math.isfinite(parent_deadline) or parent_deadline <= 0:
            raise ValueError('HIF repair requires the current trusted parent deadline')
        live = dict(source['observed_live_context'], round_number=source['handoff_opening']['round_number'])
        if (grant['parent_execution_window']['battle_resume']['live_context'] != live
                or type(now) not in (int, float) or not math.isfinite(now)
                or now < control['last_checked_at'] or control['windows_used'] >= control['maximum_actions']):
            raise ValueError('HIF repair parent differs from the original Live or accounting')
        candidate['last_checked_at'] = now
        candidate['active_parent_deadline'] = min(grant['parent_deadline'], parent_deadline)
    candidate['engineering_repair'] = dict(version=version, source=source['id'],
        source_control=deepcopy(control), issued_at=now, extra_recoveries=1, grant=deepcopy(grant))
    if version == 2:
        candidate['engineering_repair'].update(seconds=SECONDS, parent_deadline=candidate['active_parent_deadline'],
            deadline=min(now+SECONDS, candidate['active_parent_deadline']))
    window_limits(candidate)
    return candidate


def append_opening_repair(control, source, grant, *, record, now):
    """One existing v1 repair receipt, with no replacement or extended clock."""
    from .hif_live_recovery import require_budget
    from .hif_navigation_recovery import verify_unstarted_inventory_preinput_source

    verify_unstarted_inventory_preinput_source(record, source)
    if (control != source.get('automatic_battle') or 'engineering_repair' in control
            or 'engineering_window' in control or control.get('opening_parent_window') is None):
        raise ValueError('HIF opening repair requires its unchanged zero-input source')
    candidate = deepcopy(control)
    require_budget(candidate, now=now)
    candidate['engineering_repair'] = dict(version=1, source=source['id'],
        source_control=deepcopy(control), issued_at=now, extra_recoveries=1, grant=deepcopy(grant))
    window_limits(candidate)
    return candidate


def parent_source_matches(binding, source_control):
    """Compare a parent proof with its complete, already validated predecessor.

    Version 2 stores the existing control projection once; the append-only
    followup prefix remains in the original control and durable source chain.
    It does not replace validation of that complete predecessor or its history.
    """
    if not isinstance(binding, dict) or not isinstance(source_control, dict):
        return False
    if binding.get('version') == 1:
        return _same_history(binding.get('source_control'), source_control)
    if (binding.get('version') != 2 or binding.get('boundary') not in ('selected_input_timing', 'selected_neighbor_identity', 'submitted_inventory_navigation', 'submitted_card_completion', 'post_commit_identity')
            or not _compact_parent(source_control)
            or not isinstance(source_control.get('engineering_followups'), list)
            or not source_control['engineering_followups']):
        return False
    return _same_history(binding.get('source_control'), {
        key: value for key, value in source_control.items() if key != 'engineering_followups'})


def _verify_parent_window(grant, source_id, source_control, *, boundary='selected_unsubmitted'):
    """Bind the parent's already verified append-only window to this failure."""
    window = grant.get('parent_execution_window')
    parent = grant.get('parent_deadline')
    from .hif_joint_time import RUN_ID, HANDOFF, BOUNDARY
    from .hif_battle_process_binding import migrated_identity

    current = migrated_identity(source_control['identity'], grant.get('process_migrations', []))
    fields = {'schema_version', 'revision', 'window_id', 'authority', 'run_id', 'clock_epoch',
        'issued_at_ms', 'max_elapsed_ms', 'max_windows', 'native_task_id', 'process_id',
        'game_identity', 'ownership', 'source_budget', 'battle_resume'}
    if boundary == BOUNDARY:
        fields.add('time_accounting')
    durations = ((OPENING_SECONDS*1000, COMPLETION_PARENT_SECONDS*1000)
        if boundary in ('submitted_card_completion', 'post_commit_identity', 'selected_neighbor_identity',
            'selected_input_timing') and _completion_parent(source_control)
        else (OPENING_SECONDS*1000,) if boundary in ('unsubmitted_opening', 'post_commit_identity',
            'selected_neighbor_identity', 'selected_input_timing', 'submitted_inventory_navigation',
            'submitted_card_completion') else (14_400_000,))
    if (not isinstance(window, dict) or set(window) != fields or window.get('schema_version') != 2
            or window.get('authority') != 'fixed_native_resume_20261104'
            or window.get('run_id') != source_control['identity']['route_run_id']
            or window.get('max_elapsed_ms') not in durations
            or type(window.get('max_windows')) is not int or window['max_windows'] != (0 if boundary == BOUNDARY else 64)
            or boundary == BOUNDARY and (current['route_run_id'] != RUN_ID or current['handoff_id'] != HANDOFF)
            or any(type(window.get(k)) is not int or window[k] <= 0
                for k in ('issued_at_ms', 'native_task_id', 'process_id'))
            or not isinstance(window.get('source_budget'), dict)
            or window['source_budget'].get('clock_epoch') != window.get('clock_epoch')
            or window['source_budget'].get('run_id') != window.get('run_id')
            or window['source_budget'].get('game_pid') != current['game_pid']
            or window.get('ownership') != grant.get('ownership')
            or window.get('game_identity') != grant.get('game_identity')
            or type(window.get('revision')) is not int or window['revision'] < 1
            or window.get('window_id') != f"{window['run_id']}:execution-window:{window['revision']}"
            or type(parent) not in (int, float) or not math.isfinite(parent) or parent <= 0):
        raise ValueError('HIF repair lacks its fixed parent execution window')
    if boundary == BOUNDARY:
        accounting = window['time_accounting']
        budget = window['source_budget']
        if (not isinstance(accounting, dict) or set(accounting) != {'previous_limit_ms', 'elapsed_at_issue_ms', 'remaining_at_issue_ms'}
                or any(type(v) is not int or v < 0 for v in accounting.values()) or budget.get('status') != 'stopped'
                or budget.get('accounting_state') != 'active'
                or accounting['elapsed_at_issue_ms'] != budget['active_elapsed_ms'] + window['issued_at_ms'] - budget['last_at_ms']
                or accounting['remaining_at_issue_ms'] != max(0, accounting['previous_limit_ms'] - accounting['elapsed_at_issue_ms'])):
            raise ValueError('time-only opening lost its elapsed time or original remaining time')
    binding = window.get('battle_resume')
    binding_fields = {'version', 'source', 'identity', 'live_context', 'source_control', 'boundary'}
    if 'process_migrations' in grant:
        binding_fields.add('process_migrations')
    if (not isinstance(binding, dict) or set(binding) !=
            binding_fields
            or binding.get('process_migrations', []) != grant.get('process_migrations', [])
            or binding['source'] != source_id
            or binding['identity'] != source_control['identity']
            or not parent_source_matches(binding, source_control)
            or binding['boundary'] != boundary
            or (binding['live_context'] is not None if boundary in ('new_round_unstarted', BOUNDARY)
                else not isinstance(binding['live_context'], dict))):
        raise ValueError('HIF parent window differs from the original selected boundary')


def _verify_opening_parent(control):
    receipt = control.get('opening_parent_window')
    if receipt is None:
        return
    from .hif_joint_time import BOUNDARY
    from .hif_live_recovery import validate_control

    fields = {'version', 'source', 'source_control', 'issued_at', 'grant'}
    if (not isinstance(receipt, dict) or set(receipt) != fields or receipt['version'] != 1
            or not isinstance(receipt['source_control'], dict)
            or 'opening_parent_window' in receipt['source_control']):
        raise ValueError('HIF opening parent receipt is incomplete or repeated')
    source = receipt['source_control']
    validate_control(source)
    grant = receipt['grant']
    _verify_grant(control['identity'], receipt['source'], grant, parent_window=True)
    parent_window = grant.get('parent_execution_window')
    binding = parent_window.get('battle_resume') if isinstance(parent_window, dict) else None
    time_only = isinstance(binding, dict) and binding.get('boundary') == BOUNDARY
    if (any(control.get(k) != source.get(k) for k in ('identity', 'root', 'started_at', 'deadline',
            'maximum_actions', 'timeout_seconds', 'maximum_recoveries', 'parent_deadline'))
            or any(control[k] < source[k] for k in ('sequence', 'windows_used', 'recoveries_used'))
            or type(receipt['issued_at']) not in (int, float) or not math.isfinite(receipt['issued_at'])
            or receipt['issued_at'] < source['last_checked_at']
            or not time_only and receipt['issued_at'] >= source['deadline']):
        raise ValueError('HIF opening parent changed original child accounting')
    _verify_parent_window(grant, receipt['source'], source, boundary=BOUNDARY if time_only else 'new_round_unstarted')
    active = control.get('active_parent_deadline')
    remaining = grant['parent_execution_window']['time_accounting']['remaining_at_issue_ms']/1000 if time_only else 0
    if (type(active) not in (int, float) or not math.isfinite(active)
            or not receipt['issued_at'] < grant['parent_deadline'] <= receipt['issued_at']+14_400+remaining
            or not 0 < active
            or not _opening_renewal_present(control) and active > grant['parent_deadline']):
        raise ValueError('HIF opening parent exceeds the trusted fixed window')
    if time_only:
        old_limit = min(source['parent_deadline'], source['deadline'])
        if active > old_limit + 14_400:
            raise ValueError('time-only opening extended beyond its original effective cutoff')
        return min(grant['parent_deadline'], old_limit + 14_400)


def append_opening_parent(control, source, grant, *, now, parent_deadline):
    """Only the parent bound changes; this does not renew time or retry count."""
    from .hif_joint_time import BOUNDARY
    from .hif_live_recovery import require_budget, validate_control, validate_parent_deadline

    validate_control(control)
    validate_parent_deadline(parent_deadline)
    if (control != source.get('automatic_battle') or 'opening_parent_window' in control
            or type(parent_deadline) not in (int, float) or not isinstance(grant, dict)):
        raise ValueError('HIF opening parent requires the original bounded source')
    candidate = deepcopy(control)
    candidate['opening_parent_window'] = dict(version=1, source=source['id'],
        source_control=deepcopy(control), issued_at=now, grant=deepcopy(grant))
    candidate['active_parent_deadline'] = min(parent_deadline, grant.get('parent_deadline', 0))
    window = grant.get('parent_execution_window')
    binding = window.get('battle_resume') if isinstance(window, dict) else None
    if isinstance(binding, dict) and binding.get('boundary') == BOUNDARY:
        candidate['active_parent_deadline'] = min(candidate['active_parent_deadline'],
            min(control['deadline'], control['parent_deadline']) + 14_400)
    require_budget(candidate, now=now)
    return candidate



def _followup_source(control):
    return {k: deepcopy(v) for k, v in control.items() if k != 'engineering_followups'}


def _followup_limits(control, deadline, limit):
    followups = control.get('engineering_followups')
    if followups is None:
        return deadline, limit
    opening = (control.get('opening_parent_window') is not None and 'engineering_window' not in control
        and (control.get('engineering_repair') or {}).get('version') == 1)
    completion = _completion_parent(control)
    completion_started = completion and not opening
    current_parent = opening or completion
    original_parent = ((control.get('engineering_repair') or {}).get('version') == 1
        and not opening and not completion and 'engineering_window' not in control)
    if (not isinstance(followups, list) or not 1 <= len(followups) <= control['maximum_actions']
            or (control.get('engineering_repair') or {}).get('version') != 2 and not (opening or completion or original_parent)):
        raise ValueError('HIF followup requires the original bounded repair window')
    mutable = {'sequence', 'windows_used', 'recoveries_used', 'last_checked_at', 'previous_record', 'active_parent_deadline'}
    previous_time, previous_sequence, names = 0, -1, set()
    previous_recoveries = control['engineering_repair']['source_control']['recoveries_used']
    renewed = False
    prior_parent = None
    prior_revision = (control['opening_parent_window']['grant']['parent_execution_window']['revision']
        if opening else control['engineering_repair']['grant'].get('parent_execution_window', {}).get('revision', 0)
        if completion else None)
    # Validation only reads these fields. The writing path still deep-copies
    # sources via _followup_source; do not allocate that history on every guard.
    current = {k: v for k, v in control.items() if k != 'engineering_followups'}
    for index, receipt in enumerate(followups):
        version = receipt.get('version') if isinstance(receipt, dict) else None
        fields = {'version', 'source', 'source_control', 'issued_at', 'extra_recoveries', 'grant'}
        if version == 2:
            fields |= {'seconds','deadline','parent_deadline'}
        if (not isinstance(receipt, dict) or set(receipt) != fields
                or version not in (1,2) or receipt['extra_recoveries'] != 1
                or not isinstance(receipt['source_control'], dict)):
            raise ValueError('HIF followup receipt is incomplete')
        source = receipt['source_control']
        source_parent = source.get('active_parent_deadline', source.get('parent_deadline'))
        active_parent = control.get('active_parent_deadline', control.get('parent_deadline'))
        if (set(source)-mutable != set(current)-mutable
                or any(not _same_history(source[k], v) for k, v in current.items() if k not in mutable)
                or any(type(source.get(k)) is not int for k in ('sequence', 'windows_used', 'recoveries_used'))
                or not previous_sequence < source['sequence'] <= control['sequence']
                or source['windows_used'] != source['sequence']
                or not previous_recoveries <= source['recoveries_used'] <= control['recoveries_used']
                or any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in (source.get('last_checked_at'), source_parent, receipt['issued_at']))
                or not previous_time <= source['last_checked_at'] <= receipt['issued_at']
                or original_parent and version != 1
                or not (current_parent and version == 2) and receipt['issued_at'] >= source_parent
                or version == 1 and receipt['issued_at'] >= deadline
                or not current_parent and not 0 < active_parent <= source_parent
                or current_parent and prior_parent is not None and not 0 < source_parent <= prior_parent
                or not isinstance(receipt['source'], str) or receipt['source'] in names):
            raise ValueError('HIF followup changed original accounting or extended its time')
        _verify_grant(control['identity'], receipt['source'], receipt['grant'],
            parent_window=version == 2, native_renewal=version == 2)
        if version == 2:
            repair = control['engineering_repair']
            grant = receipt['grant']
            # The active parent still belongs to the earlier repair source;
            # the new child grant must not rewrite that historical binding.
            if current_parent:
                original = dict(source)
                if opening or index:
                    original['engineering_followups'] = followups[:index]
                boundary = grant['parent_execution_window']['battle_resume']['boundary']
                if (boundary not in ('unsubmitted_opening','post_commit_identity','selected_neighbor_identity','selected_input_timing','submitted_inventory_navigation','submitted_card_completion')
                        or completion_started and boundary not in ('submitted_card_completion', 'post_commit_identity',
                            'selected_neighbor_identity', 'selected_input_timing')):
                    raise ValueError('HIF fixed identity window has an unknown boundary')
                _verify_parent_window(grant, receipt['source'], original, boundary=boundary)
                if boundary == 'submitted_card_completion':
                    completion_started = True
            else:
                _verify_parent_window(grant,repair['source'],repair['source_control'])
            parent = receipt['parent_deadline']
            seconds = SECONDS
            if (not current_parent and renewed or not current_parent and grant['parent_execution_window'] != repair['grant']['parent_execution_window']
                    or type(parent) not in (int,float) or not math.isfinite(parent)
                    or not 0 < parent <= grant['parent_deadline']
                    or not current_parent and (parent > source['active_parent_deadline']
                        or not 0 < control['active_parent_deadline'] <= parent)
                    or current_parent and (grant['parent_deadline'] > receipt['issued_at']+
                        (grant['parent_execution_window']['max_elapsed_ms']/1000 if completion_started else OPENING_SECONDS)
                        or grant['parent_execution_window']['revision'] <= prior_revision)
                    or receipt['seconds'] != seconds
                    or receipt['deadline'] != min(receipt['issued_at']+seconds,parent)
                    or not receipt['issued_at'] < receipt['deadline']):
                raise ValueError('HIF native renewal exceeds or replaces its existing parent window')
            deadline,renewed = receipt['deadline'],True
            if current_parent:
                prior_revision = grant['parent_execution_window']['revision']
        prior_parent = receipt['parent_deadline'] if version == 2 else source_parent
        previous_time, previous_sequence = receipt['issued_at'], source['sequence']
        previous_recoveries = source['recoveries_used']
        names.add(receipt['source'])
    if current_parent and not 0 < control['active_parent_deadline'] <= prior_parent:
        raise ValueError('HIF opening active parent lacks its latest fixed window')
    return deadline, limit+len(followups)


def append_opening_followup(control, source, grant, *, record, now):
    """One no-time v1 followup for the verified closed item reader only."""
    from .hif_live_recovery import require_budget
    from .hif_navigation_recovery import verify_unstarted_items_source

    verify_unstarted_items_source(record, source)
    if (control != source.get('automatic_battle') or 'engineering_followups' in control
            or control.get('opening_parent_window') is None or 'engineering_window' in control
            or (control.get('engineering_repair') or {}).get('version') != 1):
        raise ValueError('HIF opening followup requires its original repaired opening')
    candidate = deepcopy(control)
    require_budget(candidate, now=now)
    candidate['engineering_followups'] = [dict(version=1, source=source['id'],
        source_control=_followup_source(control), issued_at=now, extra_recoveries=1, grant=deepcopy(grant))]
    window_limits(candidate)
    return candidate


def append_followup(control, source, grant, *, now, parent_deadline=None):
    from .hif_live_recovery import require_budget, is_submitted_terminal_stop, is_unsubmitted_timing_stop, is_unsubmitted_selected_layout_stop
    from .hif_native.native_choice_resume import is_pending_retrieval_stop
    from .hif_native.native_support_resume import selected_resource_family_stop, is_unsubmitted_resource_preview_stop
    from .hif_native.native_settlement_resume import is_submitted_gallery_stop, is_submitted_completion_stop, is_submitted_hand_identity_stop
    from .hif_native.native_upstream_selected import is_unsubmitted_upstream_selection
    from .hif_native.native_selected_cost_resume import selected_cost_confidence_stop
    from .hif_native.native_submitted_navigation import is_submitted_inventory_navigation_stop
    from .hif_native.native_opening_mapping_resume import is_unsubmitted_mapping_stop
    from .hif_native.native_opening_identity_resume import is_unsubmitted_opening_identity_stop
    from .hif_native.native_observed_identity_resume import is_post_commit_identity_stop
    from .hif_native.native_selected_identity_resume import is_unsubmitted_neighbor_identity_stop
    post_commit_identity = is_post_commit_identity_stop(source)
    selected_identity = (is_unsubmitted_neighbor_identity_stop(source)
        or (is_unsubmitted_selected_layout_stop(source) or selected_cost_confidence_stop(source)
            or selected_resource_family_stop(source) or is_unsubmitted_resource_preview_stop(source))
            and _completion_parent(control)
        or (is_unsubmitted_selected_layout_stop(source) or selected_cost_confidence_stop(source)
            or selected_resource_family_stop(source) or is_unsubmitted_resource_preview_stop(source))
            and control.get('opening_parent_window') is not None
            and (control.get('engineering_repair') or {}).get('version') == 1)
    selected_timing = is_unsubmitted_timing_stop(source)
    submitted_navigation = is_submitted_inventory_navigation_stop(source)
    submitted_completion = (is_submitted_completion_stop(source) or is_submitted_terminal_stop(source)
        or is_submitted_gallery_stop(source))
    if not (is_unsubmitted_upstream_selection(source) or is_pending_retrieval_stop(source) or is_unsubmitted_resource_preview_stop(source) or is_submitted_gallery_stop(source) or is_submitted_hand_identity_stop(source)
            or selected_timing or is_unsubmitted_selected_layout_stop(source) or post_commit_identity or selected_identity or submitted_completion
            or (is_unsubmitted_mapping_stop(source) or is_unsubmitted_opening_identity_stop(source)) and control.get('opening_parent_window') is not None
                and (control.get('engineering_repair') or {}).get('version') == 1):
        raise ValueError('HIF followup requires a known repaired observer boundary')
    candidate = deepcopy(control)
    identity_renewal = isinstance(grant, dict) and grant.get('version') == 3 and (post_commit_identity or selected_identity or selected_timing or submitted_navigation or submitted_completion or is_unsubmitted_opening_identity_stop(source))
    renewal = identity_renewal or isinstance(grant,dict) and grant.get('version') == 3
    if renewal:
        if (not can_renew_native_window(control,source)
                or type(parent_deadline) not in (int,float) or not math.isfinite(parent_deadline)
                or parent_deadline <= 0):
            raise ValueError('HIF native renewal requires its repaired selected boundary and current parent')
        _verify_grant(control['identity'],source['id'],grant,parent_window=True,native_renewal=True)
        repair = control['engineering_repair']
        if identity_renewal:
            deadline, _ = window_limits(control)
            if now+OPENING_RENEWAL_MIN_SECONDS < deadline:
                raise ValueError('HIF opening renewal is not needed within the current window')
            boundary = ('submitted_card_completion' if submitted_completion else 'submitted_inventory_navigation' if submitted_navigation else 'selected_input_timing' if selected_timing else 'selected_neighbor_identity' if selected_identity
                else 'post_commit_identity' if post_commit_identity else 'unsubmitted_opening')
            _verify_parent_window(grant, source['id'], source['automatic_battle'], boundary=boundary)
            live = (dict(source['observed_live_context'],round_number=source['round_context']['round_number'])
                if post_commit_identity or selected_identity or selected_timing or submitted_navigation or submitted_completion else source['handoff_resume']['live_context'])
            if grant['parent_execution_window']['battle_resume']['live_context'] != live:
                raise ValueError('HIF opening renewal changed its bound Live epoch')
        else:
            _verify_parent_window(grant,repair['source'],repair['source_control'])
        candidate['active_parent_deadline'] = (min(grant['parent_deadline'], parent_deadline) if identity_renewal
            else min(control['active_parent_deadline'],grant['parent_deadline'],parent_deadline))
    else:
        require_budget(candidate, now=now)
    receipt = dict(version=2 if renewal else 1,source=source['id'],
        source_control=_followup_source(source['automatic_battle']),issued_at=now,extra_recoveries=1,
        grant=deepcopy(grant))
    if renewal:
        seconds = SECONDS
        receipt.update(seconds=seconds,parent_deadline=candidate['active_parent_deadline'],
            deadline=min(now+seconds,candidate['active_parent_deadline']))
    candidate.setdefault('engineering_followups', []).append(receipt)
    window_limits(candidate)
    require_budget(candidate, now=now)
    return candidate


def can_renew_native_window(control, source):
    """One fixed identity grant per closed source; selected renewal stays once."""
    from .hif_live_recovery import validate_control, is_submitted_terminal_stop, is_unsubmitted_timing_stop, is_unsubmitted_selected_layout_stop
    from .hif_native.native_support_resume import selected_resource_family_stop, is_unsubmitted_resource_preview_stop
    from .hif_native.native_settlement_resume import is_submitted_completion_stop
    from .hif_native.native_selected_cost_resume import selected_cost_confidence_stop
    from .hif_native.native_submitted_navigation import is_submitted_inventory_navigation_stop
    from .hif_native.native_opening_identity_resume import is_unsubmitted_opening_identity_stop
    from .hif_native.native_observed_identity_resume import is_post_commit_identity_stop
    from .hif_native.native_selected_identity_resume import is_unsubmitted_neighbor_identity_stop
    validate_control(control)
    from .hif_native.native_settlement_resume import is_submitted_gallery_stop

    submitted_completion = (is_submitted_completion_stop(source) or is_submitted_terminal_stop(source)
        or is_submitted_gallery_stop(source))
    if (submitted_completion and (control.get('engineering_repair') or {}).get('version') == 1
            and control == source.get('automatic_battle')):
        return not any(row.get('source') == source.get('id') for row in control.get('engineering_followups', ()))
    if (_completion_parent(control)
            and (is_submitted_completion_stop(source) or is_submitted_terminal_stop(source)
                or is_post_commit_identity_stop(source) or is_unsubmitted_neighbor_identity_stop(source)
                or is_unsubmitted_selected_layout_stop(source) or selected_cost_confidence_stop(source)
                or selected_resource_family_stop(source) or is_unsubmitted_resource_preview_stop(source)
                or is_unsubmitted_timing_stop(source))):
        return (control == source.get('automatic_battle')
            and not any(r.get('source') == source.get('id') for r in control.get('engineering_followups', ())))
    if (control.get('opening_parent_window') is not None
            and control.get('engineering_repair',{}).get('version') == 1
            and (is_unsubmitted_opening_identity_stop(source) or is_post_commit_identity_stop(source)
                or is_unsubmitted_neighbor_identity_stop(source) or is_unsubmitted_selected_layout_stop(source)
                or is_unsubmitted_timing_stop(source) or selected_cost_confidence_stop(source)
                or selected_resource_family_stop(source) or is_unsubmitted_resource_preview_stop(source)
                or is_submitted_inventory_navigation_stop(source) or is_submitted_completion_stop(source)
                or is_submitted_terminal_stop(source))):
        return (control == source.get('automatic_battle')
            and not any(r.get('source') == source.get('id') for r in control.get('engineering_followups', ())))
    return (not any(r.get('version') == 2 for r in control.get('engineering_followups',()))
        and control.get('engineering_repair',{}).get('version') == 2
              and is_unsubmitted_selected_layout_stop(source)
              and source['live_events'][0]['runtime_phase'] == 'selected_single_plus_prepare')
