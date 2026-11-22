"""Validate historical startup credits and their original execution windows."""
from __future__ import annotations

from copy import deepcopy

from .battle_policy import FORMAL_PROFILES

RUN_ID = 'unissued-development-route'
CONTINUATION_ID = RUN_ID + '-continuation-20261103-01'
CREDIT_MS = 3_600_000
AUTHORITY = 'fixed_native_resume_20261103'
ENGINEERING_CONTINUATION_ID = RUN_ID + '-continuation-20261103-02'
ENGINEERING_CREDIT_MS = 7_200_000
FINAL_ENGINEERING_CONTINUATION_ID = RUN_ID + '-continuation-20261103-03'
FINAL_ENGINEERING_CREDIT_MS = 14_400_000
EXECUTION_WINDOW_MS = 14_400_000
OPENING_EXECUTION_WINDOW_MS = 1_800_000
IDENTITY_EXECUTION_BOUNDARIES = frozenset({'unsubmitted_opening', 'post_commit_identity',
    'selected_neighbor_identity', 'selected_input_timing', 'submitted_inventory_navigation',
    'submitted_card_completion'})
EXECUTION_WINDOW_STARTS = 64
EXECUTION_AUTHORITY = 'fixed_native_resume_20261104'


def _execution_window_ms(proof, run_id):
    # The approved one-hour future parent reuses the same schema. Previously
    # issued short opening parents remain unchanged and retain their limits.
    if (run_id != RUN_ID and isinstance(proof, dict)
            and proof.get('boundary') in ('submitted_card_completion', 'post_commit_identity',
                'selected_neighbor_identity', 'selected_input_timing')
            and (proof.get('source_control', {}).get('engineering_repair') or {}).get('version') == 2):
        return CREDIT_MS
    return (OPENING_EXECUTION_WINDOW_MS if isinstance(proof, dict)
        and proof.get('boundary') in IDENTITY_EXECUTION_BOUNDARIES else EXECUTION_WINDOW_MS)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def accounted_probe_budget(initial, *, first, now, process_id, probe_id):
    """Keep all original consumption and close only the previously audited pause."""
    from .run_budget import advance_budget, validate_budget
    validate_budget(initial, run_id=initial['run_id'], game_pid=initial['game_pid'], epoch=initial['clock_epoch'],
        now=now, max_windows=64, max_elapsed_ms=21_600_000)
    if (initial.get('schema_version') not in (2, 3) or initial.get('accounting_state') != 'paused'
            or not _integer(first) or not initial['last_at_ms'] <= first <= now
            or initial.get('continuations')):
        raise ValueError('continuation predecessor lacks its original audited probe boundary')
    budget = deepcopy(initial)
    budget['pause_receipts'].append(dict(from_ms=budget['paused_at_ms'], to_ms=first,
        evidence=dict(kind='formal_runtime_stopped_checkpoint', process_id=budget['last_process_id'],
            reason='same_save_process_probe_started', windows_used=budget['windows_used'], probe_id=probe_id)))
    budget.update(last_at_ms=first, paused_at_ms=None, accounting_state='active', status='running')
    return advance_budget(budget, now=now, process_id=process_id, opening=False)


def _grant(grant, *, epoch, now, engineering=False, final=False):
    if engineering and final:
        raise ValueError('continuation grant cannot combine separate startup capabilities')
    identifier = FINAL_ENGINEERING_CONTINUATION_ID if final else ENGINEERING_CONTINUATION_ID if engineering else CONTINUATION_ID
    allowance = FINAL_ENGINEERING_CREDIT_MS if final else ENGINEERING_CREDIT_MS if engineering else CREDIT_MS
    fields = {'schema_version', 'continuation_id', 'run_id', 'credit_ms', 'clock_epoch',
        'issued_at_ms', 'native_task_id', 'actual_game_pid', 'authority'}
    if (not isinstance(grant, dict) or set(grant) != fields or type(grant['schema_version']) is not int or grant['schema_version'] != 1
            or grant['continuation_id'] != identifier or grant['run_id'] != RUN_ID
            or type(grant['credit_ms']) is not int or grant['credit_ms'] != allowance
            or grant['authority'] != AUTHORITY or grant['clock_epoch'] != epoch
            or not all(_integer(grant[key], 1) for key in ('native_task_id', 'actual_game_pid', 'issued_at_ms'))
            or not _integer(now) or grant['issued_at_ms'] > now):
        raise ValueError('continuation lacks the fixed native startup grant')


def closed_probe_journal(source, *, epoch, run_id=RUN_ID):
    """A null gate never closes an ambiguous prepared/uncertain input by itself."""
    from .checkpoint import restore_transition
    from .input_receipt import _same
    for carrier in source.get('inputs', ()):
        try:
            gate = restore_transition(carrier['original_transition'], clock_epoch=epoch)
            report = carrier['report']
            events = report['route']['input_events']
            event = events[0]
            if (len(events) != 1 or report['route']['run_id'] != run_id or not _same(event, gate)
                    or event.get('uncertain') is not False or gate.pending is None):
                raise ValueError('probe journal has an unresolved original dispatch')
            state = carrier['dispatch_state']
            if state == 'succeeded' and event.get('dispatch_attempted') is True and event.get('success') is True:
                continue
            cancelled = carrier['original_transition'] in source.get('cancelled_unissued', ())
            if (state == 'cancelled_unissued' and cancelled and event.get('dispatch_attempted') is False
                    and event.get('success') is False and event.get('rejected_before_dispatch') is True):
                continue
            raise ValueError('probe journal lacks a closed actual input result')
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError('probe journal closure evidence is incomplete') from error


def validate_credit(credit, *, now, epoch):
    """Verify the entire predecessor arithmetic; no host boolean grants credit."""
    from .run_budget import remaining_ms
    if (not isinstance(credit, dict) or set(credit) != {'schema_version', 'grant', 'probe_id',
            'first_started_at_ms', 'activated_process_id', 'initial_budget', 'predecessor_budget', 'source_probe'}
            or type(credit['schema_version']) is not int or credit['schema_version'] != 1):
        raise ValueError('continuation audit is incomplete')
    grant, initial, source = credit['grant'], credit['initial_budget'], credit['source_probe']
    _grant(grant, epoch=epoch, now=now)
    if (initial.get('run_id') != RUN_ID or initial.get('clock_epoch') != epoch
            or not _integer(credit['activated_process_id'], 1) or not isinstance(source, dict)
            or any(key in source for key in ('original', 'continuation', 'engineering_continuation', 'final_engineering_continuation'))
            or source.get('schema_version') != 1 or not isinstance(source.get('inputs'), list)
            or not isinstance(source.get('failures'), list) or not isinstance(source.get('observer'), dict)
            or source.get('probe_id') != credit['probe_id'] or source.get('status') != 'active'
            or source.get('first_started_at_ms') != credit['first_started_at_ms']
            or source.get('game_identity', {}).get('pid') != grant['actual_game_pid']
            or source.get('transition', {}).get('clock_epoch') != epoch
            or source.get('transition', {}).get('pending') is not None
            or source.get('transition', {}).get('stopped') is not False):
        raise ValueError('continuation changed its original probe or clock scope')
    expected = accounted_probe_budget(initial, first=credit['first_started_at_ms'],
        now=grant['issued_at_ms'], process_id=credit['activated_process_id'], probe_id=credit['probe_id'])
    if credit['predecessor_budget'] != expected or remaining_ms(expected, grant['issued_at_ms']) != 0:
        raise ValueError('continuation predecessor consumption cannot be reset or reconstructed')
    closed_probe_journal(source, epoch=epoch)
    return credit


def make_probe_credit(record, initial_budget, grant, *, now, epoch, process_id):
    """Pure candidate; only the probe's trusted startup callback can publish it."""
    from .process_resume import KEY, _without_probe
    child = record['handoff']['nonbattle_context'][KEY]
    _grant(grant, epoch=epoch, now=now)
    if (record.get('run_id') != RUN_ID or child.get('continuation') is not None
            or child.get('engineering_continuation') is not None
            or child.get('final_engineering_continuation') is not None
            or child.get('original') != _without_probe(record)
            or record['transition'].get('pending') is not None or record['ledger'].get('pending') is not None
            or record['ownership'].get('owner') != 'task600' or record['ownership'].get('phase') != 'nonbattle'
            or now-grant['issued_at_ms'] > 1000):
        raise ValueError('continuation activation requires its exact closed original source')
    source = {key: deepcopy(value) for key, value in child.items() if key != 'original'}
    credit = dict(schema_version=1, grant=deepcopy(grant), probe_id=child['probe_id'],
        first_started_at_ms=child['first_started_at_ms'], activated_process_id=process_id,
        initial_budget=deepcopy(initial_budget), source_probe=source,
        predecessor_budget=accounted_probe_budget(initial_budget, first=child['first_started_at_ms'],
            now=grant['issued_at_ms'], process_id=process_id, probe_id=child['probe_id']))
    return validate_credit(credit, now=now, epoch=epoch)


def _history_preserved(current, source):
    for key in ('inputs', 'failures', 'cancelled_unissued'):
        old, new = source.get(key, []), current.get(key, [])
        if not isinstance(old, list) or not isinstance(new, list) or new[:len(old)] != old:
            raise ValueError('engineering continuation erased original probe history')


def _engineering_predecessor(previous, *, now, process_id):
    from .run_budget import advance_budget
    predecessor = deepcopy(previous['predecessor_budget'])
    predecessor.update(game_pid=previous['grant']['actual_game_pid'], continuations=[deepcopy(previous)])
    return advance_budget(predecessor, now=now, process_id=process_id, opening=False)


def validate_engineering_credit(credit, *, now, epoch):
    """The second credit retains the complete first credit and every unknown gap."""
    fields = {'schema_version', 'grant', 'probe_id', 'first_started_at_ms', 'activated_process_id',
        'initial_budget', 'predecessor_budget', 'source_probe'}
    if (not isinstance(credit, dict) or set(credit) != fields
            or type(credit['schema_version']) is not int or credit['schema_version'] != 2):
        raise ValueError('engineering continuation audit is incomplete')
    grant, predecessor, source = credit['grant'], credit['predecessor_budget'], credit['source_probe']
    _grant(grant, epoch=epoch, now=now, engineering=True)
    chain = predecessor.get('continuations') if isinstance(predecessor, dict) else None
    if not isinstance(chain, list) or len(chain) != 1:
        raise ValueError('engineering continuation lacks its complete first credit')
    previous = validate_credit(chain[0], now=grant['issued_at_ms'], epoch=epoch)
    if (grant['issued_at_ms'] < previous['grant']['issued_at_ms'] + CREDIT_MS
            or not _integer(credit['activated_process_id'], 1)
            or credit['initial_budget'] != previous['initial_budget']
            or credit['probe_id'] != previous['probe_id']
            or credit['first_started_at_ms'] != previous['first_started_at_ms']
            or grant['actual_game_pid'] != previous['grant']['actual_game_pid']
            or not isinstance(source, dict)
            or any(key in source for key in ('original', 'continuation', 'engineering_continuation', 'final_engineering_continuation'))
            or source.get('schema_version') != 1 or source.get('status') != 'active'
            or source.get('probe_id') != credit['probe_id']
            or source.get('first_started_at_ms') != credit['first_started_at_ms']
            or source.get('game_identity') != previous['source_probe']['game_identity']
            or not isinstance(source.get('observer'), dict)
            or source.get('transition', {}).get('clock_epoch') != epoch
            or source.get('transition', {}).get('pending') is not None
            or source.get('transition', {}).get('stopped') is not False):
        raise ValueError('engineering continuation changed its original credit or probe scope')
    _history_preserved(source, previous['source_probe'])
    # Recompute every field on this read. Unlike candidate creation, validation
    # need not deepcopy the entire first credit just to compare the same bytes.
    base = previous['predecessor_budget']
    changed = {'game_pid', 'continuations', 'active_elapsed_ms', 'last_at_ms', 'last_process_id', 'process_ids'}
    process_ids = base['process_ids'] + ([] if credit['activated_process_id'] in base['process_ids']
        else [credit['activated_process_id']])
    if (set(predecessor) != set(base) | {'continuations'}
            or any(predecessor[key] != value for key, value in base.items() if key not in changed)
            or predecessor['game_pid'] != grant['actual_game_pid']
            or predecessor['last_at_ms'] != grant['issued_at_ms']
            or predecessor['last_process_id'] != credit['activated_process_id']
            or predecessor['process_ids'] != process_ids
            or predecessor['active_elapsed_ms'] != base['active_elapsed_ms']+grant['issued_at_ms']-base['last_at_ms']):
        raise ValueError('engineering continuation reset its predecessor consumption')
    closed_probe_journal(source, epoch=epoch)
    return credit


def make_engineering_credit(record, initial_budget, grant, *, now, epoch, process_id):
    """Pure second-credit candidate, published only by its separate native callback."""
    from .process_resume import KEY, _without_probe
    child = record['handoff']['nonbattle_context'][KEY]
    _grant(grant, epoch=epoch, now=now, engineering=True)
    previous = validate_credit(child.get('continuation'), now=now, epoch=epoch)
    if (record.get('run_id') != RUN_ID or child.get('engineering_continuation') is not None
            or child.get('final_engineering_continuation') is not None
            or initial_budget != previous['initial_budget']
            or child.get('original') != _without_probe(record)
            or record['transition'].get('pending') is not None or record['ledger'].get('pending') is not None
            or record['ownership'].get('owner') != 'task600' or record['ownership'].get('phase') != 'nonbattle'
            or now-grant['issued_at_ms'] > 1000):
        raise ValueError('engineering activation requires its exact closed original source')
    source = {key: deepcopy(value) for key, value in child.items()
        if key not in ('original', 'continuation', 'engineering_continuation', 'final_engineering_continuation')}
    credit = dict(schema_version=2, grant=deepcopy(grant), probe_id=child['probe_id'],
        first_started_at_ms=child['first_started_at_ms'], activated_process_id=process_id,
        initial_budget=deepcopy(initial_budget), source_probe=source,
        predecessor_budget=_engineering_predecessor(previous, now=grant['issued_at_ms'], process_id=process_id))
    return validate_engineering_credit(credit, now=now, epoch=epoch)


def validate_final_engineering_credit(credit, *, now, epoch):
    """The sole final four hours retain both earlier credits and all overrun."""
    fields = {'schema_version', 'grant', 'probe_id', 'first_started_at_ms', 'activated_process_id',
        'initial_budget', 'predecessor_budget', 'source_probe'}
    if (not isinstance(credit, dict) or set(credit) != fields
            or type(credit['schema_version']) is not int or credit['schema_version'] != 3):
        raise ValueError('final engineering continuation audit is incomplete')
    grant, predecessor, source = credit['grant'], credit['predecessor_budget'], credit['source_probe']
    _grant(grant, epoch=epoch, now=now, final=True)
    chain = predecessor.get('continuations') if isinstance(predecessor, dict) else None
    if not isinstance(chain, list) or len(chain) != 2:
        raise ValueError('final engineering continuation lacks both complete earlier credits')
    previous = validate_engineering_credit(chain[1], now=grant['issued_at_ms'], epoch=epoch)
    if (previous['predecessor_budget']['continuations'] != chain[:1]
            or grant['issued_at_ms'] < previous['grant']['issued_at_ms'] + ENGINEERING_CREDIT_MS
            or not _integer(credit['activated_process_id'], 1)
            or credit['initial_budget'] != previous['initial_budget']
            or credit['probe_id'] != previous['probe_id']
            or credit['first_started_at_ms'] != previous['first_started_at_ms']
            or grant['actual_game_pid'] != previous['grant']['actual_game_pid']
            or not isinstance(source, dict)
            or any(key in source for key in ('original', 'continuation', 'engineering_continuation', 'final_engineering_continuation'))
            or type(source.get('schema_version')) is not int or source['schema_version'] != 1
            or source.get('status') not in ('active', 'complete')
            or source.get('probe_id') != credit['probe_id']
            or source.get('first_started_at_ms') != credit['first_started_at_ms']
            or source.get('game_identity') != previous['source_probe']['game_identity']
            or not isinstance(source.get('observer'), dict)
            or not isinstance(source.get('inputs'), list) or not isinstance(source.get('failures'), list)
            or source.get('transition', {}).get('clock_epoch') != epoch
            or source.get('transition', {}).get('pending') is not None
            or source.get('transition', {}).get('stopped') is not False):
        raise ValueError('final engineering continuation changed its complete original scope')
    _history_preserved(source, previous['source_probe'])
    base = previous['predecessor_budget']
    changed = {'game_pid', 'continuations', 'active_elapsed_ms', 'last_at_ms', 'last_process_id', 'process_ids'}
    process_ids = base['process_ids'] + ([] if credit['activated_process_id'] in base['process_ids']
        else [credit['activated_process_id']])
    if (set(predecessor) != set(base)
            or any(predecessor[key] != value for key, value in base.items() if key not in changed)
            or predecessor['game_pid'] != grant['actual_game_pid']
            or predecessor['last_at_ms'] != grant['issued_at_ms']
            or predecessor['last_process_id'] != credit['activated_process_id']
            or predecessor['process_ids'] != process_ids
            or predecessor['active_elapsed_ms'] != base['active_elapsed_ms'] + grant['issued_at_ms'] - base['last_at_ms']):
        raise ValueError('final engineering continuation reset predecessor consumption or unknown time')
    closed_probe_journal(source, epoch=epoch)
    return credit


def make_final_engineering_credit(record, initial_budget, grant, *, now, epoch, process_id):
    """Pure third-credit candidate; only its independent fixed callback publishes."""
    from .run_budget import advance_budget
    from .process_resume import KEY, _without_probe
    child = record['handoff']['nonbattle_context'][KEY]
    _grant(grant, epoch=epoch, now=now, final=True)
    previous = validate_engineering_credit(child.get('engineering_continuation'), now=now, epoch=epoch)
    if (record.get('run_id') != RUN_ID or child.get('final_engineering_continuation') is not None
            or previous['predecessor_budget']['continuations'] != [child.get('continuation')]
            or initial_budget != previous['initial_budget']
            or child.get('original') != _without_probe(record)
            or record['transition'].get('pending') is not None or record['ledger'].get('pending') is not None
            or record['ownership'].get('owner') != 'task600' or record['ownership'].get('phase') != 'nonbattle'
            or now-grant['issued_at_ms'] > 1000):
        raise ValueError('final engineering activation requires its exact closed original source')
    source = {key: deepcopy(value) for key, value in child.items()
        if key not in ('original', 'continuation', 'engineering_continuation', 'final_engineering_continuation')}
    predecessor = deepcopy(previous['predecessor_budget'])
    predecessor['continuations'].append(deepcopy(previous))
    predecessor = advance_budget(predecessor, now=grant['issued_at_ms'], process_id=process_id, opening=False)
    credit = dict(schema_version=3, grant=deepcopy(grant), probe_id=child['probe_id'],
        first_started_at_ms=child['first_started_at_ms'], activated_process_id=process_id,
        initial_budget=deepcopy(initial_budget), source_probe=source, predecessor_budget=predecessor)
    return validate_final_engineering_credit(credit, now=now, epoch=epoch)


def probe_remaining(initial, state, *, now, epoch):
    if not _integer(now) or now < state['first_started_at_ms']:
        raise ValueError('probe clock moved before its immutable first start')
    credit = state.get('continuation')
    if credit is None:
        if state.get('engineering_continuation') is not None or state.get('final_engineering_continuation') is not None:
            raise ValueError('engineering continuation has no complete first credit')
        if initial.get('schema_version') == 3:
            return None
        return max(0, initial['max_elapsed_ms']-initial['active_elapsed_ms']-(now-state['first_started_at_ms']))
    validate_credit(credit, now=now, epoch=epoch)
    if (credit['initial_budget'] != initial or credit['probe_id'] != state['probe_id']
            or credit['first_started_at_ms'] != state['first_started_at_ms']
            or credit['grant']['actual_game_pid'] != state['game_identity']['pid']):
        raise ValueError('probe continuation differs from its complete original source')
    extension = state.get('engineering_continuation')
    final = state.get('final_engineering_continuation')
    if final is not None and extension is None:
        raise ValueError('final engineering continuation has no complete second credit')
    if extension is not None:
        validate_engineering_credit(extension, now=now, epoch=epoch)
        if extension['predecessor_budget']['continuations'] != [credit]:
            raise ValueError('engineering continuation replaced its original first credit')
        _history_preserved(state, extension['source_probe'])
        if final is not None:
            validate_final_engineering_credit(final, now=now, epoch=epoch)
            if final['predecessor_budget']['continuations'] != [credit, extension]:
                raise ValueError('final engineering continuation replaced its original earlier credits')
            _history_preserved(state, final['source_probe'])
            return max(0, FINAL_ENGINEERING_CREDIT_MS-(now-final['grant']['issued_at_ms']))
        return max(0, ENGINEERING_CREDIT_MS-(now-extension['grant']['issued_at_ms']))
    return max(0, CREDIT_MS-(now-credit['grant']['issued_at_ms']))


def _legacy_budget_limit(budget, *, now):
    """The original maximum stays intact; each fixed credit starts after overrun."""
    credits = budget.get('continuations')
    if credits is None:
        return budget['max_elapsed_ms']
    if not isinstance(credits, list) or len(credits) not in (1, 2, 3):
        raise ValueError('continuation chain must contain only the three approved credits')
    if len(credits) == 1:
        credit = validate_credit(credits[0], now=now, epoch=budget['clock_epoch'])
        allowance = CREDIT_MS
    elif len(credits) == 2:
        credit = validate_engineering_credit(credits[1], now=now, epoch=budget['clock_epoch'])
        allowance = ENGINEERING_CREDIT_MS
    else:
        credit = validate_final_engineering_credit(credits[2], now=now, epoch=budget['clock_epoch'])
        allowance = FINAL_ENGINEERING_CREDIT_MS
    if len(credits) > 1 and credit['predecessor_budget']['continuations'] != credits[:-1]:
        raise ValueError('engineering budget replaced its complete earlier credits')
    predecessor = credit['predecessor_budget']
    reference = budget.get('continuation_process_binding')
    same_process = budget.get('game_pid') == credit['grant']['actual_game_pid']
    rebound = (isinstance(reference, dict) and set(reference) == {
        'probe_id', 'ownership_revision', 'from_game_pid', 'to_game_pid', 'published_at_ms'}
        and reference['probe_id'] == credit['probe_id']
        and reference['from_game_pid'] == credit['grant']['actual_game_pid']
        and reference['to_game_pid'] == budget.get('game_pid')
        and reference['from_game_pid'] != reference['to_game_pid']
        and _integer(reference['ownership_revision'], 1)
        and _integer(reference['published_at_ms'])
        and credit['grant']['issued_at_ms'] <= reference['published_at_ms'] <= budget['last_at_ms'])
    immutable = ('run_id', 'clock_epoch', 'started_at_ms', 'max_elapsed_ms', 'max_windows',
        'initial_process_id', 'initial_day', 'initial_phase')
    if (budget.get('schema_version') != 2 or any(budget.get(key) != predecessor.get(key) for key in immutable)
            or not (same_process and reference is None or not same_process and rebound)
            or budget['active_elapsed_ms'] < predecessor['active_elapsed_ms']
            or budget['last_at_ms'] < predecessor['last_at_ms']
            or budget['windows_used'] < predecessor['windows_used']
            or budget['process_ids'][:len(predecessor['process_ids'])] != predecessor['process_ids']
            or budget['pause_receipts'] != predecessor['pause_receipts']
            or budget.get('paused_at_ms') is not None
            or budget.get('accounting_state') not in ('active', 'complete')):
        raise ValueError('continuation changed its predecessor budget or erased consumption')
    return predecessor['active_elapsed_ms'] + allowance


def _execution_source(budget):
    # Append-only histories remain once in the same budget. Copying the nested
    # three-credit journal into every window would inflate every route read.
    return {key: deepcopy(value) for key, value in budget.items()
        if key not in ('continuations', 'execution_windows', 'process_migrations')}


def execution_process_migrations(budget, *, now):
    """Check explicit process history; it neither advances nor grants a budget."""
    import re

    from .ownership import validate_ownership
    from .cultivation_factory import _same_private_json, validate_creation_task

    migrations = budget.get('process_migrations', [])
    windows = budget.get('execution_windows', [])
    if not isinstance(migrations, list) or not isinstance(windows, list):
        raise ValueError('execution process migration history is malformed')
    previous = None
    fields = {'schema_version', 'revision', 'run_id', 'source', 'from_game', 'to_game',
        'from_ownership', 'current_task', 'after_window_revision', 'published_at_ms', 'handoff', 'process_recovery'}
    for revision, migration in enumerate(migrations, 1):
        if (not isinstance(migration, dict) or set(migration) != fields
                or type(migration['schema_version']) is not int or migration['schema_version'] != 1
                or type(migration['revision']) is not int or migration['revision'] != revision
                or migration['run_id'] != budget.get('run_id')
                or not isinstance(migration['source'], str)
                or re.fullmatch(r'[0-9]+-[a-z0-9-]+', migration['source']) is None
                or not _integer(migration['published_at_ms']) or migration['published_at_ms'] > now
                or not _integer(migration['after_window_revision'])
                or migration['after_window_revision'] > len(windows)):
            raise ValueError('execution process migration lost its original boundary')
        old, new = migration['from_game'], migration['to_game']
        for game in (old, new):
            if (not isinstance(game, dict) or set(game) != {'pid', 'executable', 'created_at_100ns'}
                    or not _integer(game['pid'], 1) or not _integer(game['created_at_100ns'], 1)
                    or not isinstance(game['executable'], str) or not game['executable']):
                raise ValueError('execution process migration has no complete game identity')
        owner = validate_ownership(migration['from_ownership'])
        task = validate_creation_task(migration['current_task'], game_pid=new['pid'])
        handoff, proof = migration['handoff'], migration['process_recovery']
        if (old['pid'] == new['pid'] or task['game'] != new
                or (owner['owner'], owner['phase'], owner['game_pid']) != ('task275', 'battle', old['pid'])
                or not isinstance(handoff, dict)
                or set(handoff) != {'handoff_id', 'ledger_revision', 'policy_profile', 'live_context'}
                or not isinstance(handoff['handoff_id'], str) or not handoff['handoff_id']
                or not _integer(handoff['ledger_revision'], 1)
                or handoff['policy_profile'] not in FORMAL_PROFILES
                or not isinstance(handoff['live_context'], dict)
                or set(handoff['live_context']) != {'run_id', 'revision', 'idol', 'plan', 'max_stamina', 'round_number'}
                or not isinstance(proof, dict) or type(proof.get('old_pid')) is not int or proof['old_pid'] != old['pid']
                or type(proof.get('new_pid')) is not int or proof['new_pid'] != new['pid']
                or proof.get('scope') not in ('pending_card_restart_rule_history_v1', 'pending_card_restart_turn_history_v1')
                or not isinstance(proof.get('source'), str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', proof['source']) is None
                or proof.get('zero_new_business_inputs') is not True
                or not _same_private_json(proof.get('logical_context'),
                    {key: value for key, value in handoff['live_context'].items() if key != 'round_number'})):
            raise ValueError('execution process migration changed its verified scope')
        live = handoff['live_context']
        if (not isinstance(live['run_id'], str) or not live['run_id'] or not _integer(live['revision'], 1)
                or not isinstance(live['idol'], str) or not live['idol'] or live['plan'] not in ('sense', 'logic', 'anomaly')
                or not _integer(live['max_stamina'], 1) or type(live['round_number']) is not int
                or live['round_number'] not in (1, 2)):
            raise ValueError('execution process migration changed its logical context')
        boundary = migration['after_window_revision']
        if (boundary and (not isinstance(windows[boundary-1], dict)
                    or windows[boundary-1].get('issued_at_ms', now+1) > migration['published_at_ms']
                    or windows[boundary-1].get('ownership', {}).get('revision', owner['revision']+1) > owner['revision'])
                or boundary < len(windows) and windows[boundary].get('issued_at_ms', -1) < migration['published_at_ms']
                or previous is not None and (old != previous['to_game']
                    or boundary < previous['after_window_revision']
                    or migration['published_at_ms'] < previous['published_at_ms']
                    or owner['revision'] <= previous['from_ownership']['revision'])):
            raise ValueError('execution process migration replaced an earlier process boundary')
        previous = migration
    if migrations and budget.get('game_pid') != migrations[-1]['to_game']['pid']:
        raise ValueError('execution process migration does not reach the current game')
    return migrations


def _retains_execution_source(current, source):
    mutable = {'last_at_ms', 'active_elapsed_ms', 'last_process_id', 'process_ids',
        'windows_used', 'status', 'last_reason', 'accounting_state', 'terminal_source'}
    paused = isinstance(source, dict) and source.get('accounting_state') == 'paused'
    if paused:
        mutable |= {'paused_at_ms', 'pause_receipts'}
    if (not isinstance(source, dict) or source.get('schema_version') != 2
            or not paused and (source.get('accounting_state') != 'active' or source.get('paused_at_ms') is not None)
            or not paused and (current.get('accounting_state') not in ('active', 'complete') or current.get('paused_at_ms') is not None)
            or source.get('status') not in ('running', 'stopped')
            or any(not _integer(source.get(key)) for key in ('last_at_ms', 'active_elapsed_ms', 'windows_used'))
            or not _integer(source.get('last_process_id'), 1)
            or not isinstance(source.get('process_ids'), list) or not source['process_ids']
            or any(not _integer(pid, 1) for pid in source['process_ids'])
            or any(current.get(key) != value for key, value in source.items() if key not in mutable)
            or current['last_at_ms'] < source['last_at_ms']
            or not paused and current['active_elapsed_ms'] - source['active_elapsed_ms'] != current['last_at_ms'] - source['last_at_ms']
            or current['windows_used'] < source['windows_used']
            or current['process_ids'][:len(source['process_ids'])] != source['process_ids']):
        raise ValueError('execution window changed its original accounting source')
    if paused:
        original = source.get('pause_receipts')
        receipts = current.get('pause_receipts')
        start = source.get('paused_at_ms')
        if (not _integer(start) or start > source['last_at_ms'] or not isinstance(original, list)
                or not isinstance(receipts, list) or receipts[:len(original)] != original):
            raise ValueError('execution window lost its original formal pause')
        if current.get('accounting_state') == 'paused':
            if (current.get('paused_at_ms') != start or receipts != original
                    or current['active_elapsed_ms'] != source['active_elapsed_ms']):
                raise ValueError('execution window changed its retained pause')
        else:
            if (current.get('accounting_state') not in ('active', 'complete')
                    or current.get('paused_at_ms') is not None or len(receipts) != len(original)+1):
                raise ValueError('execution window lacks its one formal pause closure')
            receipt = receipts[-1]
            end = receipt.get('to_ms')
            expected = dict(from_ms=start, to_ms=end, evidence=dict(kind='formal_runtime_stopped_checkpoint',
                process_id=source['last_process_id'], reason=source.get('last_reason'), windows_used=source['windows_used']))
            if (receipt != expected or not _integer(end) or not source['last_at_ms'] <= end <= current['last_at_ms']
                    or current['active_elapsed_ms']-source['active_elapsed_ms'] != current['last_at_ms']-end):
                raise ValueError('execution window changed its original pause closure or consumption')


def _execution_limits(budget, *, now):
    import re

    from live_decision.hif_joint_time import BOUNDARY

    from .ownership import validate_ownership

    if budget.get('schema_version') == 3:
        return None, None
    elapsed_limit = _legacy_budget_limit(budget, now=now)
    starts_limit = budget['max_windows']
    migrations = execution_process_migrations(budget, now=now)
    windows = budget.get('execution_windows')
    if windows is None:
        return elapsed_limit, starts_limit
    run_id = budget.get('run_id')
    child = isinstance(run_id, str) and re.fullmatch(r'hif-[0-9a-f]{32}', run_id) is not None
    if (not isinstance(windows, list) or not windows
            or not (run_id == RUN_ID and len(budget.get('continuations', ())) == 3
                or child and 'continuations' not in budget)):
        raise ValueError('execution windows require the retained original three-credit history')
    previous = None
    fields = {'schema_version', 'revision', 'window_id', 'authority', 'run_id', 'clock_epoch',
        'issued_at_ms', 'max_elapsed_ms', 'max_windows', 'native_task_id', 'process_id',
        'game_identity', 'ownership', 'source_budget'}
    source_keys = set(budget) - {'continuations', 'execution_windows', 'process_migrations'}
    for revision, window in enumerate(windows, 1):
        battle = isinstance(window, dict) and window.get('schema_version') == 2
        time_only = battle and isinstance(window.get('battle_resume'), dict) and window['battle_resume'].get('boundary') == BOUNDARY
        expected_fields = fields | {'battle_resume'} if battle else fields
        if time_only:
            expected_fields = expected_fields | {'time_accounting'}
        if (not isinstance(window, dict) or set(window) != expected_fields
                or type(window['schema_version']) is not int or window['schema_version'] not in (1, 2)
                or type(window['revision']) is not int or window['revision'] != revision
                or window['window_id'] != f'{run_id}:execution-window:{revision}'
                or window['authority'] != EXECUTION_AUTHORITY or window['run_id'] != run_id
                or window['clock_epoch'] != budget['clock_epoch']
                or type(window['max_elapsed_ms']) is not int
                or window['max_elapsed_ms'] != _execution_window_ms(window.get('battle_resume'), run_id)
                or type(window['max_windows']) is not int or window['max_windows'] != (0 if time_only else EXECUTION_WINDOW_STARTS)
                or any(not _integer(window[key], 1) for key in ('issued_at_ms', 'native_task_id', 'process_id'))
                or not isinstance(window['source_budget'], dict) or set(window['source_budget']) != source_keys):
            raise ValueError('execution window lacks its fixed native startup record')
        source, issued = window['source_budget'], window['issued_at_ms']
        owner, game = validate_ownership(window['ownership']), window['game_identity']
        expected_game = migrations[0]['from_game'] if migrations else None
        minimum_owner_revision = 1
        for migration in migrations:
            if revision > migration['after_window_revision']:
                expected_game = migration['to_game']
                minimum_owner_revision = migration['from_ownership']['revision']+1
        expected_pid = expected_game['pid'] if expected_game is not None else budget['game_pid']
        if (battle != (owner['owner'] == 'task275') or owner['game_pid'] != expected_pid
                or not isinstance(game, dict) or game.get('pid') != expected_pid
                or game.get('integrity_rid') != 0x3000 or not _integer(game.get('created_at_100ns'), 1)
                or not isinstance(game.get('executable'), str) or not game['executable']
                or expected_game is not None and any(game.get(key) != value for key, value in expected_game.items())
                or owner['revision'] < minimum_owner_revision
                or source.get('game_pid') != expected_pid
                or window['process_id'] not in budget['process_ids']
                or not source.get('last_at_ms', now + 1) <= issued <= budget['last_at_ms'] <= now):
            raise ValueError('execution window changed its game, owner or original clock')
        if battle:
            _validate_battle_execution_resume(window['battle_resume'], owner, run_id=run_id)
            retained_migrations = window['battle_resume'].get('process_migrations')
            if retained_migrations is not None and retained_migrations != [migration for migration in migrations
                    if revision > migration['after_window_revision'] and migration['handoff']['handoff_id']
                    == window['battle_resume']['identity']['handoff_id']]:
                raise ValueError('battle execution window changed its published process history')
        # Only the validated process link changes this accounting projection.
        # The retained window and its original source are never rewritten.
        _retains_execution_source(dict(budget, game_pid=expected_pid), source)
        if previous is not None:
            _retains_execution_source(dict(source, game_pid=previous['source_budget']['game_pid']), previous['source_budget'])
            if (issued <= previous['issued_at_ms'] or source['last_at_ms'] < previous['issued_at_ms']
                    or owner['revision'] < previous['ownership']['revision']
                    or not migrations and game != previous['game_identity']):
                raise ValueError('execution window replaced an earlier startup boundary')
        spent = source['active_elapsed_ms'] + (0 if source.get('accounting_state') == 'paused'
            else issued - source['last_at_ms'])
        new_round = battle and window['battle_resume']['boundary'] == 'new_round_unstarted'
        opening = battle and window['battle_resume']['boundary'] in IDENTITY_EXECUTION_BOUNDARIES
        if time_only and (source.get('status') != 'stopped' or source['windows_used'] >= starts_limit
                or spent >= elapsed_limit + window['max_elapsed_ms']
                or window['time_accounting'] != dict(previous_limit_ms=elapsed_limit, elapsed_at_issue_ms=spent,
                    remaining_at_issue_ms=max(0, elapsed_limit-spent))
                or any((old.get('battle_resume') or {}).get('boundary') == BOUNDARY for old in windows[:revision-1])):
            raise ValueError('time-only opening changed its original clock, used time or remaining starts')
        if new_round and any((old.get('battle_resume') or {}).get('boundary') == 'new_round_unstarted'
                and old['battle_resume']['identity']['handoff_id'] == window['battle_resume']['identity']['handoff_id']
                for old in windows[:revision-1]):
            raise ValueError('new-round execution window was already issued for this handoff')
        if opening and any((old.get('battle_resume') or {}).get('boundary') in IDENTITY_EXECUTION_BOUNDARIES
                and old['battle_resume']['identity']['handoff_id'] == window['battle_resume']['identity']['handoff_id']
                and old['battle_resume']['source'] == window['battle_resume']['source']
                for old in windows[:revision-1]):
            raise ValueError('opening execution window was already issued for this source')
        if spent < elapsed_limit and source['windows_used'] < starts_limit and not (new_round or opening or time_only):
            raise ValueError('an active execution window cannot be renewed')
        if source['windows_used'] > starts_limit:
            raise ValueError('execution window erased its previous invocation limit')
        next_elapsed = (elapsed_limit if time_only else spent) + window['max_elapsed_ms']
        next_starts = source['windows_used'] + window['max_windows']
        # A factory child still owns its original approved work allowance.
        # A short repair parent must not truncate that remaining allowance.
        elapsed_limit, starts_limit = ((max(elapsed_limit, next_elapsed), max(starts_limit, next_starts))
            if child else (next_elapsed, next_starts))
        previous = window
    return elapsed_limit, starts_limit


def budget_limit(budget, *, now):
    return _execution_limits(budget, now=now)[0]


def window_limit(budget, *, now):
    return _execution_limits(budget, now=now)[1]


def _validate_battle_execution_resume(proof, ownership, *, source=None, run_id=RUN_ID):
    """Retain the verified Native boundary, without any input coordinates."""
    import re

    from live_decision.hif_joint_time import RUN_ID as joint_run_id
    from live_decision.hif_joint_time import HANDOFF, BOUNDARY
    from live_decision.hif_live_recovery import validate_control
    from live_decision.hif_battle_engineering_window import parent_source_matches

    fields = {'version', 'source', 'identity', 'live_context', 'source_control', 'boundary'}
    if isinstance(proof, dict) and 'process_migrations' in proof:
        fields.add('process_migrations')
    if (not isinstance(proof, dict) or set(proof) != fields or type(proof['version']) is not int
            or proof['version'] not in (1, 2)
            or proof['version'] == 2 and proof['boundary'] not in
                ('selected_input_timing', 'submitted_inventory_navigation', 'selected_neighbor_identity',
                 'submitted_card_completion', 'post_commit_identity')
            or proof['boundary'] not in ('selected_unsubmitted', 'new_round_unstarted', BOUNDARY, *IDENTITY_EXECUTION_BOUNDARIES) or not isinstance(proof['source'], str)
            or re.fullmatch(r'[0-9]+-[a-z0-9-]+', proof['source']) is None
            or not isinstance(proof['source_control'], dict) or not isinstance(proof['identity'], dict)):
        raise ValueError('battle execution window lacks its verified selected source')
    if proof['version'] == 1:
        validate_control(proof['source_control'])
    else:
        _validate_compact_battle_control(proof['source_control'])
    if source is not None:
        if not isinstance(source, dict) or source.get('id') != proof['source']:
            raise ValueError('battle execution window changed its original source')
        validate_control(source.get('automatic_battle'))
        if not parent_source_matches(proof, source['automatic_battle']):
            raise ValueError('battle execution window changed its original control')
    bound, live = proof['identity'], proof['live_context']
    from live_decision.hif_battle_process_binding import migrated_identity

    current = migrated_identity(bound, proof.get('process_migrations', []))
    if (proof['source_control']['identity'] != bound or bound.get('route_run_id') != run_id
            or current.get('game_pid') != ownership['game_pid']
            or current.get('ownership_revision') != ownership['revision']
            or not isinstance(bound.get('handoff_id'), str) or not bound['handoff_id']
            or type(bound.get('ledger_revision')) is not int or bound['ledger_revision'] < 1
            or bound.get('policy_profile') not in FORMAL_PROFILES):
        raise ValueError('battle execution window changed its handoff or Live epoch')
    if proof['boundary'] == BOUNDARY:
        if (live is not None or run_id != joint_run_id or bound['handoff_id'] != HANDOFF
                or bound['handoff_id'] != f'{run_id}:r1:ledger-{bound["ledger_revision"]}'
                or bound['policy_profile'] != 'hif-strong-reference-trial-v1' or proof['version'] != 1):
            raise ValueError('time-only opening lost the authorised first-round handoff')
        return
    if proof['boundary'] == 'new_round_unstarted':
        if (live is not None or bound['handoff_id'] != f'{run_id}:r2:ledger-{bound["ledger_revision"]}'):
            raise ValueError('new-round execution window is not the unopened round-two handoff')
        return
    if (not isinstance(live, dict)
            or set(live) != {'run_id', 'revision', 'idol', 'plan', 'max_stamina', 'round_number'}
            or not isinstance(live['run_id'], str) or not live['run_id']
            or not _integer(live['revision'], 1) or not _integer(live['round_number'], 1)
            or not _integer(live['max_stamina'], 1) or live['plan'] not in ('sense', 'logic', 'anomaly')):
        raise ValueError('battle execution window changed its handoff or Live epoch')
    round_number = live['round_number']
    first_round_selected = (run_id != RUN_ID
        and proof['boundary'] in ('selected_neighbor_identity', 'selected_input_timing')
        and (proof['source_control'].get('engineering_repair') or {}).get('version') == 2)
    if proof['boundary'] in IDENTITY_EXECUTION_BOUNDARIES and (
            round_number not in ((1, 2) if first_round_selected
                or proof['boundary'] in ('submitted_card_completion', 'post_commit_identity') else (2,))
            or bound['handoff_id'] != f'{run_id}:r{round_number}:ledger-{bound["ledger_revision"]}'):
        raise ValueError('identity execution window requires its bound second round')


def _validate_compact_battle_control(control):
    """Check stored shape only; issuance and input require the complete source.

    Keep the recorded recovery counters as stored. The complete source and
    its retained followups are checked before issuing an execution window.
    """
    import math

    from live_decision.hif_battle_engineering_window import _compact_parent, _completion_parent

    parent_field = 'engineering_window' if _completion_parent(control) else 'opening_parent_window'
    required = {'version', 'identity', 'root', 'previous_record', 'sequence', 'windows_used',
        'recoveries_used', 'maximum_actions', 'timeout_seconds', 'maximum_recoveries',
        'started_at', 'deadline', 'last_checked_at', 'parent_deadline',
        'active_parent_deadline', parent_field, 'engineering_repair'}
    if (set(control) != required or type(control['version']) is not int or control['version'] != 1
            or not _compact_parent(control)
            or not isinstance(control['identity'], dict)
            or not all(isinstance(control[key], str) and control[key] for key in ('root', 'previous_record'))
            or not _integer(control['maximum_actions'], 1) or control['maximum_actions'] > 128
            or not _integer(control['timeout_seconds'], 1) or control['timeout_seconds'] > 7200
            or any(not _integer(control[key]) for key in
                ('sequence', 'windows_used', 'recoveries_used', 'maximum_recoveries'))
            or control['sequence'] != control['windows_used']
            or control['windows_used'] > control['maximum_actions']
            or any(type(control[key]) not in (int, float) or not math.isfinite(control[key])
                for key in ('started_at', 'deadline', 'last_checked_at', 'parent_deadline', 'active_parent_deadline'))
            or control['deadline'] - control['started_at'] != control['timeout_seconds']
            or control['last_checked_at'] < control['started_at']
            or min(control['parent_deadline'], control['active_parent_deadline']) <= 0
            or not isinstance(control[parent_field], dict)
            or not isinstance(control['engineering_repair'], dict)):
        raise ValueError('compact battle execution source has an incompatible shape')


def append_execution_window(budget, *, now, epoch, process_id, native_task_id, ownership, game_identity,
                            battle_resume=None, battle_source=None, run_id=RUN_ID):
    """Reconstruct a candidate with the original v2 grant protocol for offline replay."""
    from live_decision.hif_joint_time import BOUNDARY

    from .run_budget import remaining_ms, advance_budget, validate_budget

    validate_budget(budget, run_id=run_id, game_pid=ownership['game_pid'], epoch=epoch,
        now=now, max_windows=64, max_elapsed_ms=21_600_000)
    if budget['schema_version'] == 3:
        return None
    if budget.get('status') == 'complete':
        raise ValueError('completed cultivation cannot start another execution window')
    new_round = isinstance(battle_resume, dict) and battle_resume.get('boundary') == 'new_round_unstarted'
    opening = isinstance(battle_resume, dict) and battle_resume.get('boundary') in IDENTITY_EXECUTION_BOUNDARIES
    time_only = isinstance(battle_resume, dict) and battle_resume.get('boundary') == BOUNDARY
    if battle_resume is not None:
        if battle_resume.get('version') == 2 and battle_source is None:
            raise ValueError('compact battle execution window requires its complete original source')
        _validate_battle_execution_resume(battle_resume, ownership, source=battle_source, run_id=run_id)
    if new_round and any((old.get('battle_resume') or {}).get('boundary') == 'new_round_unstarted'
            and old['battle_resume']['identity']['handoff_id'] == battle_resume['identity']['handoff_id']
            for old in budget.get('execution_windows', ())):
        return None
    if opening and any((old.get('battle_resume') or {}).get('boundary') in IDENTITY_EXECUTION_BOUNDARIES
            and old['battle_resume']['identity']['handoff_id'] == battle_resume['identity']['handoff_id']
            and old['battle_resume']['source'] == battle_resume['source']
            for old in budget.get('execution_windows', ())):
        return None
    if remaining_ms(budget, now) > 0 and budget['windows_used'] < window_limit(budget, now=now) and not (new_round or opening or time_only):
        return None
    if budget.get('accounting_state') == 'paused':
        # A verified startup can publish its parent while the prior formal
        # pause is still open. The normal runtime closes that exact pause.
        candidate = deepcopy(budget)
        candidate['last_at_ms'] = now
        if process_id not in candidate['process_ids']:
            candidate['process_ids'].append(process_id)
    else:
        candidate = advance_budget(budget, now=now, process_id=process_id, pause_permitted=False)
    revision = len(candidate.get('execution_windows', ())) + 1
    window = dict(schema_version=1, revision=revision, window_id=f'{run_id}:execution-window:{revision}',
        authority=EXECUTION_AUTHORITY, run_id=run_id, clock_epoch=epoch, issued_at_ms=now,
        max_elapsed_ms=_execution_window_ms(battle_resume, run_id),
        max_windows=0 if time_only else EXECUTION_WINDOW_STARTS,
        native_task_id=native_task_id, process_id=process_id, ownership=deepcopy(ownership),
        game_identity=deepcopy(game_identity), source_budget=_execution_source(budget))
    if time_only:
        window['time_accounting'] = dict(previous_limit_ms=budget_limit(budget, now=now),
            elapsed_at_issue_ms=candidate['active_elapsed_ms'], remaining_at_issue_ms=remaining_ms(budget, now))
    if battle_resume is not None:
        _validate_battle_execution_resume(battle_resume, ownership, run_id=run_id)
        window.update(schema_version=2, battle_resume=deepcopy(battle_resume))
    candidate.setdefault('execution_windows', []).append(window)
    validate_budget(candidate, run_id=run_id, game_pid=ownership['game_pid'], epoch=epoch,
        now=now, max_windows=64, max_elapsed_ms=21_600_000)
    return candidate
