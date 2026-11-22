"""Four isolated frozen worlds for one already prepared public graph."""
from __future__ import annotations

import math
import time
import pickle
import multiprocessing
from contextlib import ExitStack
from multiprocessing.connection import wait

VERSION = 'hif-parallel-public-graph-v3'
MIN_SIMULATIONS = 112
WORKERS = 4
_COUNTERS = ('_validation_tail_hits', '_validation_tail_misses',
             '_validation_terminal_eligible', '_validation_terminal_declined')


def _guard(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('parallel_public_graph_deadline')


def _catalogue(kernel):
    return (kernel.ruleset_id, *(tuple(sorted(getattr(kernel, name).items()))
        for name in ('cards', 'drinks', 'items', 'memory_abilities')))


def _anchor(kernel, request):
    return (kernel._canonical_probability_state(request.state), request.plan_type,
            request.decision_kind, request.utility, tuple(request.drink_slots), tuple(request.card_slots))


def _tree_signature(tree):
    # Business keys and action keys are immutable; copy only mutable statistics
    # and child sets. Preserve dictionary order because it can break mean ties.
    return tuple((key, tuple((action, tuple(values)) for action, values in node.statistics.items()),
                  node.order, frozenset(node.children)) for key, node in tree.items())


def _frozen_signature(search):
    return (_tree_signature(search._tree), search._rng.getstate(), search._adaptive_rng.getstate(),
            search._namespace, search._expected_state, frozenset(search._training_seed_history),
            search._baseline_domain_reason) + (
                (search._internal_policy,) if getattr(search, 'internal_reanalysis', False) else ())


def _snapshot(candidate, request, prepared, seeds, deadline):
    from .hif_mcts import action_key
    from .hif_mcts_cache import settings
    from .hif_mcts_candidate import HifMctsCandidatePolicy

    _guard(deadline)
    if (type(candidate) is not HifMctsCandidatePolicy
            or candidate.parallel_public_graph is not True
            or candidate.search.parallel_public_graph is not True or not prepared.ready
            or prepared.request != request or len(seeds) != WORKERS
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != WORKERS
            or prepared.required_simulations != WORKERS * len(prepared.groups)
            or not MIN_SIMULATIONS <= prepared.required_simulations <= candidate.search.simulations):
        raise ValueError('parallel_public_graph_invalid_input')
    search = candidate.search
    if search._new_public_step_reuse(deadline) is None:
        raise ValueError('parallel_public_graph_unqualified_policy')
    if search._namespace != search._context(request):
        raise ValueError('parallel_public_graph_unbound_training_tree')
    # This is an immediate, private same-window transfer, not a portable cache.
    # Pickle detaches the complete typed tree before any worker is launched.
    payload = _payload(candidate, request)
    payload['groups'] = tuple(dict(group_index=index, request=members[0]['request'],
        key=action_key(members[0]['action']),
        prefix_depth=max(member['prefix_depth'] for member in members))
        for index, members in enumerate(prepared.groups))
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _guard(deadline)
    return encoded, payload['groups']


def _payload(candidate, request):
    """Common same-window typed transport; callers validate their own domain."""
    from .hif_mcts_cache import settings

    search = candidate.search
    context = candidate._cache_context()
    core = candidate.baseline.baseline
    if core.advice.encoder.kernel is not candidate.kernel:
        raise ValueError('parallel_public_graph_model_kernel_mismatch')
    return dict(transfer_version=VERSION, kernel=candidate.kernel, advice=core.advice,
        baseline_budget_ms=core.budget_ms, endgame_budget=getattr(core.endgame, 'budget', None),
        endgame_config=core.endgame.config, configuration=dict(candidate._cache_configuration),
        context=context, request=request, anchor=_anchor(candidate.kernel, request),
        catalogue=_catalogue(candidate.kernel), settings=settings(search), tree=search._tree,
        rng=search._rng.getstate(), adaptive_rng=search._adaptive_rng.getstate(),
        training_seed_history=frozenset(search._training_seed_history),
        baseline_domain_reason=search._baseline_domain_reason)


def _restore(payload, deadline):
    from .hif_mcts import _Node
    from .hif_hybrid import HybridHifPolicy, VisibleHifCandidatePolicy
    from .hif_mcts_cache import settings
    from .hif_mcts_candidate import HifMctsCandidatePolicy

    _guard(deadline)
    if payload.get('transfer_version') != VERSION:
        raise ValueError('parallel_public_graph_transfer_version_changed')
    kernel, advice = payload['kernel'], payload['advice']
    if advice.encoder.kernel is not kernel:
        raise ValueError('parallel_public_graph_pickled_model_binding_changed')
    request = payload['request']
    request.validate()
    if (request.ruleset_id != kernel.ruleset_id or _catalogue(kernel) != payload['catalogue']
            or _anchor(kernel, request) != payload['anchor']):
        raise ValueError('parallel_public_graph_root_or_catalogue_changed')
    core = HybridHifPolicy(kernel, advice, budget_ms=payload['baseline_budget_ms'],
                           endgame_budget=payload['endgame_budget'])
    core.endgame.config = payload['endgame_config']
    policy = HifMctsCandidatePolicy(kernel, VisibleHifCandidatePolicy(kernel, core),
                                    deadline=deadline, **payload['configuration'])
    search = policy.search
    if (policy._cache_context() != payload['context'] or settings(search) != payload['settings']
            or policy.parallel_public_graph is not True or search.parallel_public_graph is not True
            or search._new_public_step_reuse(deadline) is None):
        raise ValueError('parallel_public_graph_worker_context_changed')
    history, reason = payload['training_seed_history'], payload['baseline_domain_reason']
    history_limit = (2**32 if search.quality_reference and search.fresh_adaptive_training
                     and search.internal_reanalysis else 2**31)
    if (type(history) is not frozenset
            or any(type(seed) is not int or not 0 <= seed < history_limit for seed in history)
            or reason is not None and (reason != 'joint_root_expansion_budget_ineligible'
                                      or not search.continuation_domain)):
        raise ValueError('parallel_public_graph_training_context_changed')
    tree = payload['tree']
    if (type(tree) is not dict or any(type(node) is not _Node for node in tree.values())
            or not history and any(values[0] for node in tree.values() for values in node.statistics.values())):
        raise ValueError('parallel_public_graph_invalid_typed_tree')
    # The worker owns its unpickled immutable keys. Clone mutable node contents
    # so even two restores of one local payload cannot share evolving data.
    tree = {key: _Node({action: list(values) for action, values in node.statistics.items()},
                      node.order, set(node.children)) for key, node in tree.items()}
    search._rng.setstate(payload['rng'])
    search._adaptive_rng.setstate(payload['adaptive_rng'])
    search._training_seed_history = set(history)
    search._baseline_domain_reason = reason
    search._namespace = search._context(request)
    search._expected_state = payload['anchor'][0]
    search._tree = tree
    search._reset_validation_cohort_evidence()
    search._order_deadline = deadline
    _guard(deadline)
    return policy


def _helpers(policy):
    search = policy.search
    unknown = search.unknown_node_policy
    result = dict(partial_node_timing=dict(search.last_partial_node_timing))
    if unknown is not None:
        result['unknown_node_timing'] = unknown.summary()
        for field, attribute in (('installed_score_turn_chain', 'installed_score_policy'),
                                 ('extra_action_turn_chain', 'extra_action_policy'),
                                 ('pair_order_advice', 'pair_order_policy')):
            helper = getattr(unknown, attribute, None)
            if helper is not None:
                result[field] = helper.summary()
    if search.turn_chain_continuation is not None:
        result['turn_chain_continuation'] = search.turn_chain_continuation.summary()
    if policy.double_advice:
        result['double_advice'] = {name: getattr(policy.double_ordering, name) for name in
            ('calls', 'active_calls', 'bypassed_calls', 'scope_bypassed_calls', 'resource_scope_bypassed_calls')}
    for field, session in (('terminal_cache', policy._session), ('complete_tail_cache', policy._tail_session)):
        if session is not None:
            result[field] = dict(decisions=session.decisions, memo_hits=session.memo_hits,
                                 namespaces=session.namespaces_created)
    return result


def _worker(connection, encoded, world_index, seed, deadline):
    """Importable spawn target; never chooses a seed, trains, or builds a graph."""
    started = time.perf_counter()
    record = _empty_record(world_index, seed)
    policy = memo = None
    try:
        from .validation_cache import validation_session

        _guard(deadline)
        payload = pickle.loads(encoded)
        policy = _restore(payload, deadline)
        search = policy.search
        before = _frozen_signature(search)
        memo = search._public_step_reuse = search._new_public_step_reuse(deadline)
        if memo is None or search._public_turn_policy is not None:
            raise ValueError('parallel_public_graph_worker_not_frozen')
        record['setup_seconds'] = time.perf_counter() - started
        with ExitStack() as stack:
            stack.enter_context(validation_session(lambda: _guard(deadline)))
            policy._session = stack.enter_context(policy.endgame.search_session(deadline=deadline))
            policy._tail_session = stack.enter_context(policy.tail.search_session(deadline=deadline))
            tick = time.perf_counter()
            try:
                for group in payload['groups']:
                    _guard(deadline)
                    record['started_simulations'] += 1
                    connection.send(('started', record['started_simulations']))
                    counters = tuple(getattr(search, name) for name in _COUNTERS)
                    value = search._validate_simulation(group['request'], seed, group['key'], deadline,
                                                       prefix_depth=group['prefix_depth'])
                    _guard(deadline)
                    if value[1]:
                        raise ValueError('parallel_public_graph_mutable_training_path')
                    row = dict(group_index=group['group_index'], seed=seed, key=group['key'],
                        prefix_depth=group['prefix_depth'], final_score=value[0], depth=value[2],
                        terminal_calls=value[3], terminal_fallbacks=value[4],
                        counters={name: getattr(search, name)-old for name, old in zip(_COUNTERS, counters)})
                    _validate_row(row, group, seed)
                    record['rows'].append(row)
                    record['completed_simulations'] += 1
                    connection.send(('row', row))
            finally:
                record['evaluation_seconds'] = time.perf_counter() - tick
                record['helper_summaries'] = _helpers(policy)
        record['tree_rng_namespace_unchanged'] = before == _frozen_signature(search)
        if not record['tree_rng_namespace_unchanged']:
            raise ValueError('parallel_public_graph_frozen_tree_changed')
        _guard(deadline)
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


def _empty_record(world_index, seed):
    return dict(world_index=world_index, seed=seed, complete=False, started_simulations=0,
        completed_simulations=0, count_incomplete=True, rows=[], helper_summaries={}, step_reuse=None,
        tree_rng_namespace_unchanged=False, setup_seconds=0., evaluation_seconds=0., elapsed_seconds=0., errors=[])


def _validate_row(row, group, seed):
    if (row.get('group_index') != group['group_index'] or row.get('seed') != seed
            or row.get('key') != group['key'] or row.get('prefix_depth') != group['prefix_depth']
            or type(row.get('final_score')) not in (int, float) or not math.isfinite(row['final_score'])
            or any(type(row.get(name)) is not int or row[name] < 0
                   for name in ('depth', 'terminal_calls', 'terminal_fallbacks'))
            or not group['prefix_depth'] <= row['depth'] < 256
            or row['terminal_fallbacks'] > row['terminal_calls']
            or not isinstance(row.get('counters'), dict) or set(row['counters']) != set(_COUNTERS)
            or any(type(value) is not int or value < 0 for value in row['counters'].values())):
        raise ValueError('parallel_public_graph_invalid_row')


def _receive(record, message, groups):
    kind, value = message
    if kind == 'started':
        if (type(value) is not int or value != record['started_simulations'] + 1
                or record['started_simulations'] != record['completed_simulations'] or value > len(groups)):
            raise ValueError('parallel_public_graph_invalid_progress')
        record['started_simulations'] = value
    elif kind == 'row':
        index = record['completed_simulations']
        if index >= len(groups) or record['started_simulations'] != index + 1:
            raise ValueError('parallel_public_graph_unstarted_or_duplicate_row')
        _validate_row(value, groups[index], record['seed'])
        record['rows'].append(value)
        record['completed_simulations'] += 1
    elif kind == 'finished':
        for field in ('world_index', 'seed', 'started_simulations', 'completed_simulations', 'rows'):
            if value.get(field) != record[field]:
                raise ValueError('parallel_public_graph_final_progress_mismatch')
        record.update(value)
        if (record['complete'] is not True or record['errors'] or record['count_incomplete'] is not False
                or record['tree_rng_namespace_unchanged'] is not True
                or record['completed_simulations'] != len(groups)):
            raise ValueError('parallel_public_graph_worker_incomplete')
        return True
    else:
        raise ValueError('parallel_public_graph_unknown_message')
    return False


def _cleanup(processes, receivers, *, failed, deadline):
    # Never use a pool context manager's unbounded shutdown(wait=True).
    for process in processes:
        if failed and process.is_alive():
            process.terminate()
    for process in processes:
        remaining = max(0., deadline-time.perf_counter())
        process.join(timeout=min(.05, remaining) if failed else remaining)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=.2)
    alive = any(process.is_alive() for process in processes)
    for receiver in receivers:
        receiver.close()
    for process in processes:
        if not process.is_alive():
            process.close()
    if alive:
        raise RuntimeError('parallel_public_graph_worker_cleanup_incomplete')


def run_worlds(candidate, request, prepared, seeds, deadline, diagnostics):
    """Return four complete ordered world records, or raise with failure evidence."""
    return _run_worlds(candidate, request, prepared, seeds, deadline, diagnostics,
        snapshot=_snapshot, worker=_worker, version=VERSION, evidence_key='parallel_public_graph')


def _run_worlds(candidate, request, prepared, seeds, deadline, diagnostics, *,
                snapshot, worker, version, evidence_key):
    started = time.perf_counter()
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('parallel_public_graph_finite_deadline_required')
    deadline = min(deadline, prepared.deadline)
    records = [_empty_record(index, seed) for index, seed in enumerate(seeds)]
    evidence = dict(version=version, workers=WORKERS, started_simulations=0, completed_simulations=0,
        count_incomplete=True, elapsed_seconds=0., worker_records=records, complete=False, errors=[])
    diagnostics[evidence_key] = evidence
    processes, receivers = [], []
    failed = True
    try:
        encoded, groups = snapshot(candidate, request, prepared, seeds, deadline)
        context = multiprocessing.get_context('spawn')
        waiting = {}
        for index, seed in enumerate(seeds):
            _guard(deadline)
            receiver, sender = context.Pipe(duplex=False)
            receivers.append(receiver)
            process = context.Process(target=worker, args=(sender, encoded, index, seed, deadline))
            try:
                process.start()
                processes.append(process)
            finally:
                sender.close()
            waiting[receiver] = index
        while waiting:
            _guard(deadline)
            ready = wait(tuple(waiting), timeout=max(0., deadline-time.perf_counter()))
            if not ready:
                raise TimeoutError('parallel_public_graph_deadline')
            for receiver in ready:
                index = waiting[receiver]
                if _receive(records[index], receiver.recv(), groups):
                    del waiting[receiver]
        _guard(deadline)
        failed = False
    except (TimeoutError, ValueError, RuntimeError) as error:
        evidence['errors'].append(dict(type=type(error).__name__, message=str(error)))
        raise
    except Exception as error:
        evidence['errors'].append(dict(type=type(error).__name__, message=str(error)))
        raise RuntimeError('parallel_public_graph_process_failure:' + type(error).__name__) from error
    finally:
        try:
            _cleanup(processes, receivers, failed=failed, deadline=deadline)
        except Exception as error:
            evidence['errors'].append(dict(type=type(error).__name__, message=str(error)))
            raise RuntimeError('parallel_public_graph_cleanup_failure') from error
        finally:
            evidence.update(started_simulations=sum(row['started_simulations'] for row in records),
                completed_simulations=sum(row['completed_simulations'] for row in records),
                count_incomplete=any(row['count_incomplete'] for row in records),
                elapsed_seconds=time.perf_counter()-started)
    try:
        _guard(deadline)
    except TimeoutError as error:
        evidence['errors'].append(dict(type=type(error).__name__, message=str(error)))
        raise
    evidence['complete'] = True
    return tuple(records)
