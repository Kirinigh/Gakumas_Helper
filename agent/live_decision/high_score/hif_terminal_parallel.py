"""Explicit offline execution of every exact terminal root, under one deadline."""
import math
import time
import pickle
import multiprocessing
from multiprocessing.connection import wait

from .hif import HifSenseKernel
from .planner import SearchIncomplete, _Outcome, _normalize
from .contracts import ActionKind, GameAction
from .state_copy import replace
from .hif_planner import HifActionChainPlanner
from .hif_public_graph_parallel import _cleanup

VERSION = 'hif-exact-terminal-roots-v2'
WORKERS = 4
MEMO_CHUNK_ENTRIES = 128
SERIAL_SLICE_SECONDS = .2


class _SerialSliceComplete(SearchIncomplete):
    """Only this cooperative yield may transfer unfinished roots to workers."""


def _require_supported(planner, request, deadline):
    if (type(planner.kernel) is not HifSenseKernel or request.state.hand_support
            or request.state.turns_remaining != 1 or planner.complete_horizon
            or not planner.config.require_terminal_leaf or planner.advice is not None
            or planner.use_learned_value or not math.isfinite(deadline)):
        raise SearchIncomplete('parallel_terminal_unsupported')


def adaptive_roots(planner, request, actions, deadline):
    """Finish cheap roots locally before paying process startup costs.

    The short serial slice is inside the original deadline. Only complete
    roots and complete memo entries survive it; an unfinished root is still
    evaluated by the existing exact backend with the remaining time.
    """
    _require_supported(planner, request, deadline)
    _guard(deadline)
    original_parallel = planner.parallel_terminal_roots
    original_check = planner._check_budget
    had_local_check = '_check_budget' in vars(planner)
    serial_deadline = min(deadline, time.perf_counter() + SERIAL_SLICE_SECONDS)
    known = {}

    def check_slice():
        original_check()
        _guard(deadline)
        if time.perf_counter() >= serial_deadline:
            raise _SerialSliceComplete('parallel_terminal_serial_slice')

    try:
        planner._check_budget = check_slice
        planner.parallel_terminal_roots = False
        for action in actions:
            check_slice()
            outcomes = planner._evaluate_action(request.state, action, request=request,
                                                depth=planner.config.max_decision_depth)
            check_slice()
            # Local work has already incremented the parent's counters.
            known[action] = (outcomes, planner._reached_depth, 0)
    except _SerialSliceComplete:
        pass
    finally:
        if had_local_check:
            planner._check_budget = original_check
        else:
            del planner._check_budget
        planner.parallel_terminal_roots = original_parallel
    _guard(deadline)
    missing = tuple(action for action in actions if action not in known)
    if missing:
        # The interrupted root may have completed useful children. Recheck
        # before starting workers, under the same restored outer deadline.
        cached, missing = cached_roots(planner, request, missing)
        known.update(cached)
    if missing:
        known.update(run_roots(planner, request, missing, deadline))
    _guard(deadline)
    return {action: known[action] for action in actions}


class _SingleRootPlanner(HifActionChainPlanner):
    def _initialize_search(self, request, started):
        super()._initialize_search(request, started)
        if self.seed_memo is not None:
            self._memo.update(self.seed_memo)
            self.seed_memo = None

    def _merge_equivalent_actions(self, state, actions):
        merged = super()._merge_equivalent_actions(state, actions)
        return tuple(a for a in merged if a == self.root) if state is self.root_state else merged

    def _evaluate_action(self, state, action, *, request, depth):
        result = super()._evaluate_action(state, action, request=request, depth=depth)
        if state is self.root_state and depth == self.config.max_decision_depth:
            self.root_outcomes = result
        return result


def _guard(deadline):
    if time.perf_counter() >= deadline:
        raise SearchIncomplete('planner_timeout')


def _memo_chunks(memo, seed, deadline):
    """Export complete new value entries, without repeating the input table."""
    chunk = []
    for key, outcomes in memo.items():
        _guard(deadline)
        if key not in seed:
            chunk.append((key, outcomes))
            if len(chunk) == MEMO_CHUNK_ENTRIES:
                yield tuple(chunk)
                chunk.clear()
    if chunk:
        yield tuple(chunk)


def _same_distribution(left, right):
    if left == right:
        return True
    # Outcome order is immaterial; all probabilities and resources must agree.
    values = lambda outcomes: sorted((o.score, o.probability, o.stamina, o.genki) for o in outcomes)
    return values(left) == values(right)


def _worker(connection, encoded, indices, deadline):
    try:
        kernel, request, config, actions, terminal_cache, memo = pickle.loads(encoded)
        planner = _SingleRootPlanner(kernel, config, terminal_cache=terminal_cache)
        planner.seed_memo = memo
        planner.root_state = request.state
        with planner.search_session(deadline=deadline):
            for index in indices:
                _guard(deadline)
                planner.root, planner.root_outcomes = actions[index], None
                response = planner.decide(request)
                if (planner.root_outcomes is None or not (
                        response.has_action_intent or response.reason == 'all_candidates_rejected')):
                    raise SearchIncomplete(response.reason)
                _guard(deadline)
                hits = (planner._memo_hits,
                        planner._terminal_marginals.hits - planner._terminal_hit_offsets[0],
                        planner._terminal_marginals.transition_hits - planner._terminal_hit_offsets[1])
                connection.send(('root', index, planner.root_outcomes,
                                 planner._reached_depth, planner._expanded_nodes, hits))
            # Session exit clears its tables, so serialize the complete delta
            # here. Each send and the final completion remain in the deadline.
            for chunk in _memo_chunks(planner._memo, memo, deadline):
                _guard(deadline)
                connection.send(('memo', chunk))
                _guard(deadline)
        _guard(deadline)
        connection.send(('complete',))
    except Exception as error:
        connection.send(('failed', type(error).__name__, str(error)))
    finally:
        connection.close()


def run_roots(planner, request, actions, deadline):
    """Preserve complete root distributions and order; never accept partial work."""
    _require_supported(planner, request, deadline)
    _guard(deadline)
    # These are complete normalized terminal outcome distributions from the
    # current session namespace, never actions, scalar universe-id caches or
    # input bindings. The worker uses the identical catalog/config/utility.
    parent_memo = planner._memo
    encoded = pickle.dumps((planner.kernel, request, planner.config, actions,
                            planner.terminal_cache, dict(parent_memo)))
    processes, receivers, results = [], [], {}
    memo_updates = {}
    failed = True
    try:
        context = multiprocessing.get_context('spawn')
        for worker in range(min(WORKERS, len(actions))):
            # Spread later roots away from the first heavy root, preserving
            # each group's source order. Scheduling does not affect values.
            indices = tuple(i for i in range(len(actions))
                            if (i if i < WORKERS else (i - WORKERS + 1) % WORKERS) == worker)
            receiver, sender = context.Pipe(duplex=False)
            receivers.append(receiver)
            process = context.Process(target=_worker, args=(sender, encoded, indices, deadline))
            try:
                _guard(deadline)
                process.start()
                processes.append(process)
            finally:
                sender.close()
        pending = set(receivers)
        while pending:
            _guard(deadline)
            ready = wait(tuple(pending), timeout=max(0., deadline - time.perf_counter()))
            if not ready:
                raise SearchIncomplete('planner_timeout')
            for receiver in ready:
                message = receiver.recv()
                if message[0] == 'complete':
                    pending.remove(receiver)
                elif message[0] == 'memo':
                    for key, outcomes in message[1]:
                        _guard(deadline)
                        previous = parent_memo.get(key, memo_updates.get(key))
                        if previous is not None:
                            if not _same_distribution(previous, outcomes):
                                raise SearchIncomplete('parallel_terminal_memo_conflict')
                        else:
                            memo_updates[key] = outcomes
                elif message[0] == 'root':
                    _, index, outcomes, depth, nodes, hits = message
                    if index in results or not 0 <= index < len(actions) or not outcomes:
                        raise SearchIncomplete('parallel_terminal_invalid_root')
                    if (any(not math.isfinite(o.score) or not math.isfinite(o.probability)
                            or o.probability <= 0 for o in outcomes)
                            or not math.isclose(sum(o.probability for o in outcomes), 1., abs_tol=1e-9)):
                        raise SearchIncomplete('parallel_terminal_invalid_distribution')
                    if (type(hits) is not tuple or len(hits) != 3
                            or any(type(value) is not int or value < 0 for value in hits)):
                        raise SearchIncomplete('parallel_terminal_invalid_statistics')
                    results[index] = (outcomes, depth, nodes, hits)
                else:
                    raise SearchIncomplete('parallel_terminal_worker_failed:' + repr(message[1:]))
        _guard(deadline)
        if set(results) != set(range(len(actions))):
            raise SearchIncomplete('parallel_terminal_incomplete')
        failed = False
    except SearchIncomplete:
        raise
    except Exception as error:
        raise SearchIncomplete('parallel_terminal_process_failure:' + type(error).__name__) from error
    finally:
        try:
            _cleanup(processes, receivers, failed=failed, deadline=deadline)
        except Exception as error:
            raise SearchIncomplete('parallel_terminal_cleanup_failed') from error
    _guard(deadline)
    if planner._memo is not parent_memo:
        raise SearchIncomplete('parallel_terminal_namespace_changed')
    # No worker may publish before all roots and process cleanup succeed.
    # Keep the session's original table object; imported values remain score
    # increments under its existing catalog/utility namespace.
    try:
        parent_memo.update(memo_updates)
        _guard(deadline)
    except Exception:
        for key in memo_updates:
            parent_memo.pop(key, None)
        raise
    # Each worker resets per-decision counters and reports marginal deltas.
    # Merge each completed root once, after the whole run and cleanup succeed.
    # Parent-side cached_roots hits have already accrued on these counters.
    planner._memo_hits += sum(row[3][0] for row in results.values())
    planner._terminal_marginals.hits += sum(row[3][1] for row in results.values())
    planner._terminal_marginals.transition_hits += sum(row[3][2] for row in results.values())
    return {actions[i]: results[i][:3] for i in range(len(actions))}


def cached_roots(planner, request, actions):
    """Price a root only when every child already has an exact terminal value."""
    known, missing = {}, []
    state = request.state
    for action in actions:
        planner._check_budget()
        terminal = planner.kernel.terminal_score_resources(state, action, plan_type=request.plan_type)
        if terminal is not None:
            score, stamina, genki = terminal
            known[action] = ((_Outcome(score, 1., stamina, genki),), 1, 1)
            continue
        branches = planner._transition(state, action, plan_type=request.plan_type)
        complete = True
        for branch in branches:
            child = branch.state
            if child.terminal:
                continue
            if child.actions_remaining == 0 and planner.kernel.terminal_score_resources(
                    child, GameAction(ActionKind.END_TURN), plan_type=request.plan_type) is not None:
                continue
            key = (planner._semantic_state_key(replace(child, score=0)), None)
            if key not in planner._memo:
                complete = False
                break
        if not complete:
            missing.append(action)
            continue
        outcomes = []
        for branch in branches:
            child = planner._search_state(branch.state, request=request,
                                          depth=planner.config.max_decision_depth - 1)
            outcomes.extend(replace(o, probability=o.probability * branch.probability) for o in child)
        # Child work and hits are already counted locally; only the root node
        # is deferred to the caller's normal per-root accounting.
        known[action] = (_normalize(tuple(outcomes)), planner._reached_depth, 1)
    planner._check_budget()
    return known, tuple(missing)
