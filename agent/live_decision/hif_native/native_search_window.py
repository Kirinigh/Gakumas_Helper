"""Durable strong-trial computation windows, without input authority.

A crash consumes its claimed window. Reopening an older result cannot grant
another budget, including when its complete cache was copied to a new attempt.
"""
import os
import re
import json

VERSION = 'hif-native-search-window-v1'
PROFILE = 'hif-strong-reference-trial-v1'


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', value) is None:
        raise ValueError('strong_reference_window_origin_invalid')
    return value


def _index(windows):
    if type(windows) is not int or windows < 1:
        raise ValueError('strong_reference_window_progress_invalid')
    return windows - 1


def _path(runtime, origin, index):
    return runtime / f'search-window-{_name(origin)}-{index}.json'


def _read(runtime, origin, index):
    row = json.loads(_path(runtime, origin, index).read_text(encoding='utf8'))
    if (not isinstance(row, dict) or set(row) != {
            'version', 'computation_origin', 'window_index', 'status', 'attempt', 'cache_file'}
            or row['version'] != VERSION or row['computation_origin'] != origin
            or type(row['window_index']) is not int or row['window_index'] != index
            or row['status'] not in ('started', 'complete')):
        raise ValueError('strong_reference_window_record_invalid')
    _name(row['attempt'])
    if row['status'] == 'started':
        if row['cache_file'] is not None:
            raise ValueError('strong_reference_window_record_invalid')
    elif row['cache_file'] not in (
            f"search-cache-{row['attempt']}-pending_intent.json",
            f"search-cache-{row['attempt']}-observed_successor.json"):
        raise ValueError('strong_reference_window_record_invalid')
    return row


def _write(stream, row):
    json.dump(row, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())


def begin_window(runtime, result, policy, request):
    """Called immediately before choose, after Native increments _windows.

    The request is deliberately not persisted: neither screen coordinates nor
    observation/input credentials belong to computation-window metadata.
    """
    if policy.profile != PROFILE:
        return None
    attempt = _name(result['id'])
    origin = _name(getattr(policy, '_host_computation_origin', attempt))
    index = _index(policy._windows)
    if index and _read(runtime, origin, index - 1)['status'] != 'complete':
        raise ValueError('strong_reference_previous_window_incomplete')
    row = dict(version=VERSION, computation_origin=origin, window_index=index,
               status='started', attempt=attempt, cache_file=None)
    try:
        with _path(runtime, origin, index).open('x', encoding='utf8') as stream:
            _write(stream, row)
    except FileExistsError as error:
        raise ValueError('strong_reference_window_already_started') from error
    policy._host_computation_origin = origin
    policy._host_planning_window = index
    return row


def validate_persist_window(runtime, result, policy, payload):
    """Require an actual start or an already completed restored window."""
    origin = _name(getattr(policy, '_host_computation_origin', None))
    index = _index(payload['planning_windows'])
    row = _read(runtime, origin, index)
    if policy._windows != payload['planning_windows']:
        raise ValueError('strong_reference_window_progress_invalid')
    if row['status'] != 'complete' and (
            getattr(policy, '_host_planning_window', None) != index
            or row['attempt'] != result['id']):
        raise ValueError('strong_reference_window_start_not_owned')
    if row['status'] != 'complete' and (
            payload['phase'] != 'pending_intent' or policy._pending is None
            or policy._pending['plan'].get('accepted') is not True):
        raise ValueError('strong_reference_window_has_no_accepted_intent')
    if _path(runtime, origin, index + 1).exists():
        raise ValueError('strong_reference_newer_window_already_started')
    return origin


def complete_window(runtime, result, policy, payload, cache_file):
    """Seal only after the complete computation cache is durably replaced."""
    origin = validate_persist_window(runtime, result, policy, payload)
    index = _index(payload['planning_windows'])
    row = _read(runtime, origin, index)
    if row['status'] != 'complete':
        row.update(status='complete', cache_file=cache_file)
        path = _path(runtime, origin, index)
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf8') as stream:
            _write(stream, row)
        temporary.replace(path)
    policy._host_planning_window = None


def validate_restored_window(runtime, origin, payload):
    """An incomplete or superseded cache cannot restart computation."""
    origin = _name(origin)
    index = _index(payload['planning_windows'])
    if _read(runtime, origin, index)['status'] != 'complete':
        raise ValueError('strong_reference_cached_window_incomplete')
    if _path(runtime, origin, index + 1).exists():
        raise ValueError('strong_reference_newer_window_already_started')
    return origin


def prepare_research_recompute(runtime, source, specification):
    """Explicitly select a failed attempt's saved pre-search computation.

    Business state still comes from the independently verified native source.
    This is a new development experiment, never restoration of the lost work.
    """
    from .native_settlement_resume import load_result

    if (not isinstance(specification, dict)
            or set(specification) != {'attempt', 'failed_source', 'reason'}
            or not isinstance(specification['reason'], str)
            or not specification['reason'].strip()):
        raise ValueError('strong_reference_recompute_request_invalid')
    attempt = _name(specification['attempt'])
    failed = load_result(runtime, _name(specification['failed_source']))
    source_pid = source.get('endpoints', {}).get('game', {}).get('pid')
    events = failed.get('live_events', ())
    decisions = [event for event in events if event.get('event') == 'live_decision_observe']
    if (attempt in (source['id'], failed['id'])
            or failed.get('status') != 'STOP' or failed.get('live_return') is not False
            or failed.get('native_resume_source') != source['id']
            or type(source_pid) is not int or source_pid <= 0
            or failed.get('endpoints', {}).get('game', {}).get('pid') != source_pid
            or failed.get('resume_settlement', {}).get('accepted') is not True
            or failed.get('resume_history', {}).get('accepted') is not True
            or len(decisions) != 1
            or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                   or event.get('choice_submit_count', 0) for event in events)
            or failed.get('choice_journal') or failed.get('gray_selection_journal')):
        raise ValueError('strong_reference_recompute_source_not_zero_submit_stop')
    proposal = decisions[0]['proposal']
    high_score = proposal['candidate_traces']['high_score']
    planning = high_score.get('trace', {}).get('planning', {})
    if (proposal.get('status') != 'STOP' or proposal.get('action') is not None
            or high_score.get('action') is not None
            or type(planning.get('completed_simulations')) is not int
            or planning['completed_simulations'] < 0):
        raise ValueError('strong_reference_recompute_failure_evidence_invalid')
    return failed, dict(specification, business_source=source['id'],
        failed_reason=high_score.get('reason'),
        reported_completed_simulations=planning['completed_simulations'],
        reported_elapsed_ms=planning.get('elapsed_ms'))


def validate_recompute_window(runtime, source, payload, receipt):
    """A named failed next window alone permits an explicitly new branch."""
    origin = _name(source['mcts_search_cache']['computation_origin'])
    index = _index(payload['planning_windows'])
    inherited = _read(runtime, origin, index)
    failed = _read(runtime, origin, index + 1)
    if (payload['phase'] != 'observed_successor'
            or receipt['failed_source'] != source['id']
            or inherited['status'] != 'complete' or failed['status'] != 'started'
            or failed['attempt'] != source['id']
            or failed != source.get('mcts_search_window')
            or _path(runtime, origin, index + 2).exists()
            or (runtime / f"search-cache-{source['id']}-pending_intent.json").exists()):
        raise ValueError('strong_reference_recompute_window_not_lost')
    return dict(receipt, inherited_computation_origin=origin,
        inherited_cache_record=source['id'], inherited_cache_file=source['mcts_search_cache']['file'],
        inherited_planning_windows=payload['planning_windows'],
        inherited_decision_calls=payload['decision_calls'], failed_window=failed)


def begin_research_recompute(runtime, source, payload, candidate, receipt):
    """After complete restore, retain old counters and create one inherited anchor."""
    receipt = validate_recompute_window(runtime, source, payload, receipt)
    if (candidate._windows != payload['planning_windows']
            or candidate._calls != payload['decision_calls'] or candidate._pending is not None
            or candidate._expected_after is None):
        raise ValueError('strong_reference_recompute_requires_complete_restoration')
    origin = _name(receipt['attempt'])
    index = _index(payload['planning_windows'])
    if any(runtime.glob(f'search-window-{origin}-*.json')):
        raise ValueError('strong_reference_recompute_attempt_already_used')
    # The attempt/cache identify the inherited computation, never this new run.
    inherited = dict(version=VERSION, computation_origin=origin, window_index=index,
        status='complete', attempt=source['id'], cache_file=source['mcts_search_cache']['file'])
    with _path(runtime, origin, index).open('x', encoding='utf8') as stream:
        _write(stream, inherited)
    candidate._host_computation_origin = origin
    return dict(receipt, computation_origin=origin, inherited_anchor=inherited,
        inherited_anchor_purpose='saved_pre_search_computation_only_not_new_search_completion',
        original_failed_window_modified=False, lost_computation_restored=False,
        new_development_attempt=True)
