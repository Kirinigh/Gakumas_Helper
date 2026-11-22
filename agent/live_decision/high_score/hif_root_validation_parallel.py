"""Four already allocated worlds price complete roots under one frozen policy."""
from __future__ import annotations

import math
import time
import pickle
from copy import deepcopy
from types import SimpleNamespace
from contextlib import ExitStack
from dataclasses import replace

from . import hif_public_graph_parallel as parallel
from .hif_mcts import _RootOption
from .hif_internal_reanalysis import Decision, _freeze
from .hif_internal_successor_graph import GraphDecision

VERSION = 'hif-parallel-root-validation-v1'
MIN_SIMULATIONS = 108
WORKERS = 4


def decline_reason(search, roots):
    if not (search.parallel_public_graph and (search.paired_adaptive_validation
            or search.uniform_validation_time_guard)):
        return 'disabled'
    if len(roots) * WORKERS < MIN_SIMULATIONS:
        return 'below_minimum_simulations'
    if search._public_turn_policy is not None or search._resource_policy is not None:
        return 'unsupported_public_or_resource_policy'
    owner = getattr(getattr(search, '_public_step_reuse_check', None), '__self__', None)
    if owner is None or getattr(owner, 'proven_tail', None) is not None:
        return 'unsupported_owner_or_proven_tail'
    if search._internal_policy is not None and type(search._internal_policy) not in (Decision, GraphDecision):
        return 'unsupported_internal_policy'
    return None


def _snapshot(candidate, request, prepared, seeds, deadline):
    from .hif_mcts_candidate import HifMctsCandidatePolicy

    parallel._guard(deadline)
    search, roots = candidate.search, prepared.roots
    if (type(candidate) is not HifMctsCandidatePolicy or decline_reason(search, roots) is not None
            or len(seeds) != WORKERS or len(set(seeds)) != WORKERS
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or len(roots) * WORKERS > search.simulations
            or any(type(root) is not _RootOption for root in roots)
            or len({root.key for root in roots}) != len(roots)
            or search._namespace != search._context(request)
            or search._new_public_step_reuse(deadline) is None):
        raise ValueError('parallel_root_validation_invalid_input')
    internal = search._internal_policy
    if internal is not None and internal.identity[1] != _freeze(search._context(request)):
        raise ValueError('parallel_root_validation_internal_context_changed')
    payload = parallel._payload(candidate, request)
    payload.update(root_validation_version=VERSION, source_namespace=search._namespace,
        internal=internal, roots=roots,
        groups=tuple(dict(group_index=index, request=request, key=root.key, root_option=root, prefix_depth=0)
                     for index, root in enumerate(roots)))
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    parallel._guard(deadline)
    return encoded, payload['groups']


def _restore(payload, deadline):
    if payload.get('root_validation_version') != VERSION:
        raise ValueError('parallel_root_validation_version_changed')
    policy = parallel._restore(payload, deadline)
    search, internal = policy.search, payload['internal']
    if internal is not None:
        if (type(internal) not in (Decision, GraphDecision)
                or internal.identity[1] != _freeze(payload['source_namespace'])):
            raise ValueError('parallel_root_validation_internal_context_changed')
        # Only runtime callback identities change between processes. All table
        # values, full states, slots and graph records remain exactly frozen.
        search._internal_policy = replace(deepcopy(internal), identity=(internal.identity[0],
            _freeze(search._context(payload['request'])), *internal.identity[2:]))
    roots = payload['roots']
    if (decline_reason(search, roots) is not None
            or any(type(root) is not _RootOption for root in roots)
            or tuple(root.key for root in roots) != tuple(group['key'] for group in payload['groups'])
            or roots != tuple(group['root_option'] for group in payload['groups'])):
        raise ValueError('parallel_root_validation_full_root_changed')
    parallel._guard(deadline)
    return policy


def _worker(connection, encoded, world_index, seed, deadline):
    started = time.perf_counter()
    record = parallel._empty_record(world_index, seed)
    policy = memo = None
    try:
        from .validation_cache import validation_session

        payload = pickle.loads(encoded)
        policy = _restore(payload, deadline)
        search = policy.search
        before = parallel._frozen_signature(search)
        memo = search._public_step_reuse = search._new_public_step_reuse(deadline)
        if memo is None:
            raise ValueError('parallel_root_validation_unqualified_policy')
        record['setup_seconds'] = time.perf_counter() - started
        with ExitStack() as stack:
            stack.enter_context(validation_session(lambda: parallel._guard(deadline)))
            policy._session = stack.enter_context(policy.endgame.search_session(deadline=deadline))
            policy._tail_session = stack.enter_context(policy.tail.search_session(deadline=deadline))
            tick = time.perf_counter()
            try:
                for group, root in zip(payload['groups'], payload['roots'], strict=True):
                    parallel._guard(deadline)
                    record['started_simulations'] += 1
                    connection.send(('started', record['started_simulations']))
                    counters = tuple(getattr(search, name) for name in parallel._COUNTERS)
                    return_tick = time.perf_counter()
                    value = search._validate_simulation(payload['request'], seed, root, deadline)
                    row = dict(group_index=group['group_index'], seed=seed, key=root.key, prefix_depth=0,
                        final_score=value[0], depth=value[2], terminal_calls=value[3], terminal_fallbacks=value[4],
                        elapsed_seconds=time.perf_counter() - return_tick,
                        return_kind=('proven_tail_expected_score' if search._validation_tail_hits > counters[0]
                            else 'terminal_evaluator_expected_score' if value[3] > value[4]
                            else 'sampled_terminal_score'),
                        counters={name: getattr(search, name) - old
                                  for name, old in zip(parallel._COUNTERS, counters, strict=True)})
                    _validate_row(row, group, seed)
                    if value[1]:
                        raise ValueError('parallel_root_validation_mutable_path')
                    # Publish a paid complete row before checking the deadline;
                    # timeout still fails the stage, but cannot erase its cost.
                    record['rows'].append(row)
                    record['completed_simulations'] += 1
                    connection.send(('row', row))
                    parallel._guard(deadline)
            finally:
                record['evaluation_seconds'] = time.perf_counter() - tick
                record['helper_summaries'] = parallel._helpers(policy)
        record['tree_rng_namespace_unchanged'] = before == parallel._frozen_signature(search)
        if not record['tree_rng_namespace_unchanged']:
            raise ValueError('parallel_root_validation_frozen_state_changed')
        parallel._guard(deadline)
        record['complete'] = True
    except Exception as error:
        record['errors'].append(dict(type=type(error).__name__, message=str(error)))
    finally:
        if memo is not None:
            record['step_reuse'] = memo.summary()
        if policy is not None:
            policy.search._public_step_reuse = None
        record['elapsed_seconds'] = time.perf_counter() - started
        record['count_incomplete'] = False
        try:
            connection.send(('finished', record))
        finally:
            connection.close()


def _validate_row(row, group, seed):
    parallel._validate_row(row, group, seed)
    if (type(row.get('elapsed_seconds')) not in (int, float)
            or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < 0
            or row.get('return_kind') not in ('sampled_terminal_score',
                'terminal_evaluator_expected_score', 'proven_tail_expected_score')):
        raise ValueError('parallel_root_validation_invalid_row')


def run_worlds(candidate, request, roots, seeds, deadline, diagnostics):
    """Keep failure rows and counts in diagnostics; never retry serially."""
    started = time.perf_counter()
    prepared = SimpleNamespace(roots=tuple(roots), deadline=deadline)
    before = parallel._frozen_signature(candidate.search)
    try:
        records = parallel._run_worlds(candidate, request, prepared, seeds, deadline, diagnostics,
            snapshot=_snapshot, worker=_worker, version=VERSION, evidence_key='parallel_root_validation')
        for record in records:
            for index, row in enumerate(record['rows']):
                _validate_row(row, dict(group_index=index, key=roots[index].key, prefix_depth=0), record['seed'])
        parallel._guard(deadline)
        return records
    except Exception as error:
        evidence = diagnostics.get('parallel_root_validation')
        if evidence is not None:
            evidence['complete'] = False
            if not evidence['errors']:
                evidence['errors'].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        evidence = diagnostics.get('parallel_root_validation')
        if evidence is not None:
            evidence.update(seeds=tuple(seeds), root_keys=tuple(root.key for root in roots),
                parent_frozen_state_unchanged=before == parallel._frozen_signature(candidate.search),
                elapsed_seconds=time.perf_counter() - started,
                deadline_remaining_seconds=deadline - started)
            evidence['finished_before_deadline'] = time.perf_counter() < deadline
            if not evidence['parent_frozen_state_unchanged']:
                evidence['complete'] = False
                raise ValueError('parallel_root_validation_parent_changed')
            if not evidence['finished_before_deadline']:
                evidence['complete'] = False
                raise TimeoutError('parallel_root_validation_deadline')
