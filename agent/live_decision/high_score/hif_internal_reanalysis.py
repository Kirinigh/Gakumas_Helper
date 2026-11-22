"""Opt-in building blocks for one paired internal public-state decision.

The candidate is disabled by default. Preparation sees only the current public
root and its deterministic one-step successors; an explicit quality-mode option
also completes the core's already proved root prefixes. Evaluation
reuses four TRAINING seeds under one frozen continuation; resetting a planning
sampler here is local conditional sampling, not preservation of the original
root trajectory's draw cursor. A global-route intervention is a different test.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace, dataclass

from .planner import SearchIncomplete
from .hif_mcts import action_key, _RootOption
from .contracts import PlanType, ActionKind, DecisionKind
from .hif_mcts_cache import json_value, public_request
from .hif_environment import HifEnvironment
from .hif_choice_roots import _complete_legal
from .validation_cache import validation_session
from .hif_local_continuation import LocalContinuation
from .hif_resource_reanalysis import _freeze
from .hif_public_turn_reanalysis import _StrictDeterministicKernel
from .hif_frozen_policy_continuation import deterministic_public_successor

VERSION = 'hif-internal-reanalysis-v2'
COMPLETE_ROOT_SUCCESSORS_VERSION = 'hif-internal-complete-root-successors-v1'
VALUE_SCOPE = 'LOCAL_CONDITIONAL_PAIRED_FROZEN_RETURNS'
_RETURN_KINDS = frozenset(('sampled_terminal_score', 'terminal_evaluator_expected_score',
                           'proven_tail_expected_score'))


def new_diagnostics(*, complete_root_successors=False):
    result = dict(version=VERSION, enabled=True, value_scope=VALUE_SCOPE, eligible=False,
        complete=False, status='not_run', reason='not_prepared', requested_samples=4,
        candidate_count=0, total_budget=0, minimum_training=0, minimum_validation=0,
        action_capacity=0, reserved_simulations=0, training_limit=0,
        reservation_release=dict(status='not_applicable', reason='not_prepared',
            original_training_limit=0, effective_training_limit=0, released_simulations=0,
            training_before_release=0, resource_graph_protected=False, resource_precheck=None),
        requested_simulations=0, started_simulations=0, completed_simulations=0,
        planning_seeds=(), validation_seeds=(), training_validation_disjoint=False,
        public_request=None, legal_action_keys=(), root_action=None, prior_action=None,
        frozen_action=None, node_statistics=(), selected_action=None, means=(), samples=())
    if complete_root_successors:
        result.update(complete_root_successors_enabled=True,
            complete_root_successors_version=COMPLETE_ROOT_SUCCESSORS_VERSION,
            root_continuation=(), root_expected_after_first=None)
    return result


def _prepared_evidence(prepared):
    result = dict(candidate_count=len(prepared.candidates), **{name: getattr(prepared, name) for name in (
        'total_budget', 'minimum_training', 'minimum_validation', 'action_capacity',
        'reserved_simulations', 'training_limit')})
    if prepared.complete_root_successors:
        result.update(complete_root_successors_enabled=True,
                      complete_root_successors_version=COMPLETE_ROOT_SUCCESSORS_VERSION)
    return result


def _guard(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('internal_reanalysis_finite_deadline_required')
    if time.perf_counter() >= deadline:
        raise TimeoutError('internal_reanalysis_deadline')


def _identity(request, context):
    # No score translation, instance aliasing, observation ID or input authority.
    return (VERSION, context, request.plan_type, request.decision_kind, request.utility,
            request.ruleset_id, request.protocol_version, request.state,
            tuple(request.drink_slots), tuple(request.card_slots))


@dataclass(frozen=True)
class Candidate:
    request: object
    actions: tuple
    prior_action: tuple
    root_action: tuple
    root_continuation: tuple = ()
    root_expected_after_first: object | None = None


@dataclass(frozen=True)
class Prepared:
    context: tuple
    candidates: tuple[Candidate, ...]
    total_budget: int
    minimum_training: int
    minimum_validation: int
    action_capacity: int
    reserved_simulations: int
    training_limit: int
    reason: str
    complete_root_successors: bool = False


@dataclass(frozen=True)
class Selection:
    prepared: Prepared
    candidate: Candidate
    frozen_action: tuple
    statistics: tuple


@dataclass(frozen=True)
class Sample:
    action_key: tuple
    planning_seed: int
    final_score: float
    return_kind: str
    completed_depth: int
    terminal_evaluations: int
    terminal_fallbacks: int


@dataclass(frozen=True)
class Decision:
    identity: tuple
    action_key: tuple
    frozen_action: tuple
    planning_seeds: tuple
    means: tuple
    samples: tuple[Sample, ...]
    value_scope: str = VALUE_SCOPE


def prepare(kernel, request, ordered_actions, *, prior, root_count, context,
            deadline, total_budget=256, compute_sensitivity=False, retained_policy=False,
            complete_policy=False, diagnostics=None, complete_root_successors=False,
            root_options=None):
    """Reserve before training; ordered_actions is the existing public prior order.

    root_count includes all joint roots. A complete public/resource table takes
    precedence. ``prior(public_request, legal_actions)`` returns its fixed public
    timing recommendation (or None); it receives no scenario or sampler.
    ``context`` is the caller's immutable rules/model/configuration namespace.
    ``compute_sensitivity`` is an explicit offline caller capability; only the
    quality-reference runtime grants it, for the fixed 256/512/1024 budgets.
    ``complete_root_successors`` accepts only complete core root options, with
    public action order first and the original option order within each group.
    """
    _guard(deadline)
    request.validate()
    actions = tuple(ordered_actions)
    if (type(root_count) is not int or root_count < len(actions) or not actions
            or type(compute_sensitivity) is not bool or type(total_budget) is not int
            or (total_budget not in (256, 512, 1024) if compute_sensitivity
                else not 1 <= total_budget <= 256)
            or type(retained_policy) is not bool or type(complete_policy) is not bool
            or type(complete_root_successors) is not bool
            or root_options is not None and not complete_root_successors
            or type(context) is not tuple or not callable(prior)):
        raise ValueError('internal_reanalysis_invalid_configuration')
    context = _freeze(context)
    _complete_legal(kernel, request, actions)
    if complete_root_successors:
        _complete_root_options(kernel, request, actions, root_options, root_count)
    minimum_training, minimum_validation = 4 * root_count, 4 * (root_count + retained_policy)
    capacity = (total_budget - minimum_training - minimum_validation) // 4
    if capacity < 0:
        raise ValueError('internal_reanalysis_root_coverage_budget')
    candidates, seen = [], set()
    reason = 'no_eligible_successor'
    if complete_policy:
        reason = 'complete_public_or_resource_policy'
    elif capacity < 2:
        reason = 'internal_action_coverage_budget'
    elif (request.ruleset_id != kernel.ruleset_id or request.plan_type is not PlanType.SENSE
            or request.decision_kind is not DecisionKind.LIVE or request.state.terminal
            or request.state.turns_remaining <= 2 or request.state.pending_choice is not None
            or request.state.resolution_stack or request.state.pending_after_card_ids):
        reason = 'outside_public_root_domain'
    elif complete_root_successors:
        candidates = _complete_root_candidates(kernel, request, actions, root_options,
                                               prior, capacity, deadline)
    else:
        proof = _StrictDeterministicKernel(kernel.cards, kernel.drinks, kernel.items,
                                         memory_abilities=kernel.memory_abilities)
        with validation_session(lambda: _guard(deadline), reuse_parent=True):
            for action in actions:
                _guard(deadline)
                after, _ = deterministic_public_successor(kernel, request.state, action,
                    plan_type=request.plan_type, deadline=deadline)
                if (after is None or after.terminal or after.turn != request.state.turn
                        or after.turns_remaining != request.state.turns_remaining
                        or after.actions_remaining <= 0 or not after.deck.hand
                        or after.pending_choice is not None or after.resolution_stack
                        or after.pending_after_card_ids):
                    continue
                # Also reject RNG consumed before equal branches were merged,
                # including a draw with only one possible card.
                try:
                    branches = proof.transition(request.state, replace(action, source_slot=None),
                                                plan_type=request.plan_type)
                except SearchIncomplete as error:
                    if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                        raise
                    continue
                if (len(branches) != 1 or branches[0].probability != 1.
                        or kernel._canonical_probability_state(branches[0].state) != after):
                    raise ValueError('internal_reanalysis_proof_mismatch')
                slots = tuple(request.drink_slots)
                if action.kind is ActionKind.USE_DRINK:
                    slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
                environment = HifEnvironment(kernel)
                rebound = environment.reset(after, seed=0, drink_slots=slots)
                current = replace(request, state=rebound.state, card_slots=rebound.card_slots,
                                  drink_slots=rebound.drink_slots, decision_kind=rebound.decision_kind)
                legal = tuple(environment.legal_actions())
                identity = (current.state, current.drink_slots, current.card_slots)
                if identity in seen or not 2 <= len(legal) <= capacity:
                    continue
                seen.add(identity)
                recommendation = prior(current, legal)
                _guard(deadline)
                if recommendation is None:
                    continue
                if sum(action_key(item) == recommendation for item in legal) != 1:
                    raise ValueError('internal_reanalysis_prior_not_uniquely_legal')
                candidates.append(Candidate(current, legal, recommendation, action_key(action)))
    reserve = 4 * max((len(candidate.actions) for candidate in candidates), default=0)
    _guard(deadline)
    prepared = Prepared(context, tuple(candidates), total_budget, minimum_training, minimum_validation,
                        capacity, reserve, total_budget - minimum_validation - reserve,
                        'prepared' if candidates else reason, complete_root_successors)
    if diagnostics is not None:
        diagnostics.update(new_diagnostics(complete_root_successors=complete_root_successors),
                           **_prepared_evidence(prepared))
        diagnostics.update(eligible=bool(candidates), status='prepared' if candidates else 'declined',
                           reason=prepared.reason)
    return prepared


def _complete_root_options(kernel, request, actions, options, root_count):
    """Accept only the core's complete, already proved root-option domain."""
    if (type(options) is not tuple or len(options) != root_count
            or any(type(option) is not _RootOption for option in options)
            or len({option.key for option in options}) != len(options)
            or {action_key(option.action) for option in options} != {action_key(a) for a in actions}):
        raise ValueError('internal_complete_roots_require_complete_core_options')
    for action in actions:
        group = [option for option in options if action_key(option.action) == action_key(action)]
        if any(type(option.continuation) is not tuple or len(option.continuation) > 1
               or bool(option.continuation) != (option.expected_after_first is not None) for option in group):
            raise ValueError('internal_complete_roots_invalid_prefix')
        continued = [option for option in group if option.continuation]
        if not continued:
            if len(group) != 1:
                raise ValueError('internal_complete_roots_duplicate_plain_action')
            continue
        first = continued[0]
        if first.continuation[0].kind is ActionKind.END_TURN:
            if (len(group) != 2 or len(continued) != 1
                    or action.kind is not ActionKind.PLAY_CARD):
                raise ValueError('internal_complete_roots_incomplete_optional_end')
        else:
            expected = first.expected_after_first
            if (len(continued) != len(group) or expected.pending_choice is None
                    or expected.pending_choice.action_kind is not ActionKind.MOVE_CARD
                    or any(option.expected_after_first != expected
                           or option.continuation[0].kind is not ActionKind.MOVE_CARD for option in group)):
                raise ValueError('internal_complete_roots_invalid_required_choice')
            legal = kernel.legal_actions(expected, plan_type=request.plan_type,
                                         decision_kind=DecisionKind.SECONDARY)
            if {action_key(option.continuation[0]) for option in group} != {action_key(a) for a in legal}:
                raise ValueError('internal_complete_roots_incomplete_required_choice')


def _complete_root_candidates(kernel, request, actions, options, prior, capacity, deadline):
    candidates, seen = [], set()
    proof = _StrictDeterministicKernel(kernel.cards, kernel.drinks, kernel.items,
                                     memory_abilities=kernel.memory_abilities)
    ordered = (option for action in actions for option in options
               if action_key(option.action) == action_key(action))
    with validation_session(lambda: _guard(deadline), reuse_parent=True):
        for option in ordered:
            state, slots = request.state, request.drink_slots
            for index, action in enumerate((option.action, *option.continuation)):
                _guard(deadline)
                after, _ = deterministic_public_successor(kernel, state, action,
                    plan_type=request.plan_type, deadline=deadline)
                if after is None:
                    break
                try:
                    branches = proof.transition(state, replace(action, source_slot=None), plan_type=request.plan_type)
                except SearchIncomplete as error:
                    if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                        raise
                    break
                if (len(branches) != 1 or branches[0].probability != 1.
                        or kernel._canonical_probability_state(branches[0].state) != after):
                    raise ValueError('internal_reanalysis_proof_mismatch')
                if index == 0 and option.continuation and after != option.expected_after_first:
                    raise ValueError('internal_complete_roots_first_state_mismatch')
                if action.kind is ActionKind.USE_DRINK:
                    slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
                state = after
            else:
                if (state.terminal or state.turn != request.state.turn
                        or state.turns_remaining != request.state.turns_remaining
                        or state.actions_remaining <= 0 or not state.deck.hand
                        or state.pending_choice is not None or state.resolution_stack or state.pending_after_card_ids):
                    continue
                environment = HifEnvironment(kernel)
                rebound = environment.reset(state, seed=0, drink_slots=slots)
                current = replace(request, state=rebound.state, card_slots=rebound.card_slots,
                                  drink_slots=rebound.drink_slots, decision_kind=rebound.decision_kind)
                legal = tuple(environment.legal_actions())
                identity = (current.state, current.drink_slots, current.card_slots)
                if identity in seen or not 2 <= len(legal) <= capacity:
                    continue
                seen.add(identity)
                recommendation = prior(current, legal)
                _guard(deadline)
                if recommendation is None:
                    continue
                if sum(action_key(item) == recommendation for item in legal) != 1:
                    raise ValueError('internal_reanalysis_prior_not_uniquely_legal')
                candidates.append(Candidate(current, legal, recommendation, action_key(option.action),
                    tuple(action_key(action) for action in option.continuation), option.expected_after_first))
    return candidates


def select(prepared, *, node_lookup, deadline, diagnostics=None, most_visited=False):
    """First sparse disagreement in the precommitted public order, not tree order.

    node_lookup(request, actions) supplies the physical node view. Historical
    means only nominate a node; they never count as paired evaluation evidence.
    """
    if type(most_visited) is not bool:
        raise ValueError('internal_reanalysis_selection_requires_bool')
    selected, nominations = None, []
    for index, candidate in enumerate(prepared.candidates):
        _guard(deadline)
        node = node_lookup(candidate.request, candidate.actions)
        if node is None:
            continue
        keys = tuple(action_key(action) for action in candidate.actions)
        if (set(node.statistics) != set(keys) or len(node.order) != len(keys)
                or set(node.order) != set(keys)):
            raise ValueError('internal_reanalysis_node_action_coverage')
        for visits, total in node.statistics.values():
            if (type(visits) is not int or visits < 0 or type(total) not in (int, float)
                    or not math.isfinite(total) or not visits and total != 0):
                raise ValueError('internal_reanalysis_invalid_statistics')
        if any(not row[0] for row in node.statistics.values()) or all(
                row[0] >= 4 for row in node.statistics.values()):
            continue
        incumbent = max(node.order, key=lambda key: node.statistics[key][1] / node.statistics[key][0])
        if incumbent != candidate.prior_action:
            statistics = tuple((key, *node.statistics[key]) for key in node.order)
            selection = Selection(prepared, candidate, incumbent, statistics)
            if most_visited:
                visits = sum(row[0] for row in node.statistics.values())
                nominations.append(dict(candidate_index=index, root_action=candidate.root_action,
                    total_visits=visits, node_statistics=statistics,
                    prior_action=candidate.prior_action, frozen_action=incumbent))
                if prepared.complete_root_successors:
                    nominations[-1].update(root_continuation=candidate.root_continuation,
                        root_expected_after_first=(json_value(asdict(candidate.root_expected_after_first))
                            if candidate.root_expected_after_first is not None else None))
                if selected is None or visits > selected[0]:
                    selected = visits, selection
                continue
            if diagnostics is not None:
                diagnostics.update(status='selected', reason='sparse_prior_disagreement', eligible=True,
                    **_selection_evidence(selection))
            return selection
    _guard(deadline)
    if most_visited and diagnostics is not None:
        diagnostics.update(selection_mode='most_visited_sparse_successor', nominations=tuple(nominations))
    if selected is not None:
        selection = selected[1]
        if diagnostics is not None:
            diagnostics.update(status='selected', reason='sparse_prior_disagreement', eligible=True,
                               **_selection_evidence(selection))
        return selection
    if diagnostics is not None:
        diagnostics.update(status='declined', reason='no_sparse_prior_disagreement', eligible=False)
    return None


def _selection_evidence(selection):
    candidate = selection.candidate
    result = dict(**_prepared_evidence(selection.prepared), public_request=public_request(candidate.request),
        legal_action_keys=tuple(action_key(action) for action in candidate.actions),
        root_action=candidate.root_action, prior_action=candidate.prior_action,
        frozen_action=selection.frozen_action, node_statistics=selection.statistics,
        requested_simulations=4 * len(candidate.actions))
    if selection.prepared.complete_root_successors:
        result.update(root_continuation=candidate.root_continuation,
            root_expected_after_first=(json_value(asdict(candidate.root_expected_after_first))
                                       if candidate.root_expected_after_first is not None else None))
    return result


def evaluate(selection, training_seeds, *, evaluate_return, frozen_signature, remaining_budget,
             deadline, diagnostics, validation_seeds=()):
    """Publish only one complete table; caller includes this work in the window.

    evaluate_return(public_request, training_seed, action_key, deadline) returns
    the LocalContinuation five-field complete-return tuple. The caller keeps the
    new table disabled throughout these calls and supplies a detached immutable
    frozen_signature() of the continuation tree/RNG/tables. Remaining budget is
    AFTER reserving all mandatory roots and old-policy outer returns.
    """
    candidate = selection.candidate
    keys = tuple(action_key(action) for action in candidate.actions)
    required = 4 * len(keys)
    release = diagnostics.get('reservation_release', new_diagnostics()['reservation_release'])
    diagnostics.update(new_diagnostics(complete_root_successors=selection.prepared.complete_root_successors),
                       **_selection_evidence(selection), reservation_release=release)
    diagnostics.update(eligible=True, status='evaluating', reason='paired_training_evaluation')
    local, recorded = None, []

    def complete_return(key, seed, limit):
        if frozen_signature() != before:
            raise ValueError('internal_reanalysis_frozen_policy_changed')
        result = evaluate_return(candidate.request, seed, key, limit)
        if frozen_signature() != before:
            raise ValueError('internal_reanalysis_frozen_policy_changed')
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError('internal_reanalysis_incomplete_return')
        score, kind, depth, calls, fallbacks = result
        if (type(score) not in (int, float) or not math.isfinite(score) or kind not in _RETURN_KINDS
                or any(type(value) is not int for value in (depth, calls, fallbacks))
                or not 0 <= depth < 256 or not 0 <= fallbacks <= calls):
            raise ValueError('internal_reanalysis_incomplete_return')
        recorded.append(Sample(key, seed, score, kind, depth, calls, fallbacks))
        return result

    try:
        _guard(deadline)
        seeds = tuple(training_seeds)
        outer = tuple(validation_seeds)
        if outer and (len(outer) < 4 or len(set(outer)) != len(outer)
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in outer)):
            raise ValueError('internal_reanalysis_invalid_validation_seeds')
        if (len(seeds) != 4 or len(set(seeds)) != 4
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or set(seeds).intersection(outer)):
            raise ValueError('internal_reanalysis_four_disjoint_training_seeds_required')
        diagnostics.update(planning_seeds=seeds, validation_seeds=outer,
                           training_validation_disjoint=bool(outer))
        if (candidate not in selection.prepared.candidates or len(set(keys)) != len(keys)
                or selection.frozen_action not in keys
                or required > selection.prepared.reserved_simulations
                or type(remaining_budget) is not int or required > remaining_budget):
            raise ValueError('internal_reanalysis_action_coverage_budget')
        # Put the current frozen choice first: exact mean ties preserve that policy.
        order = (selection.frozen_action, *(key for key in keys if key != selection.frozen_action))
        local = LocalContinuation(required, 0, seeds=seeds)
        before = frozen_signature()
        # This LocalContinuation lives for exactly one selected state, so no
        # runtime configuration needs to become a hashable shared-cache key.
        chosen, audit = local.choose((candidate.request.state, candidate.request.drink_slots),
                                    order, complete_return, deadline)
        samples = tuple(Sample(**row) for row in audit['samples'])
        if (len(samples) != required or {(row.action_key, row.planning_seed) for row in samples}
                != {(key, seed) for key in keys for seed in seeds}):
            raise ValueError('internal_reanalysis_incomplete_pairs')
        means = tuple((key, sum(row.final_score for row in samples if row.action_key == key) / 4)
                      for key in keys)
        _guard(deadline)
        diagnostics.update(complete=True, status='complete', reason='complete_paired_training',
                           selected_action=chosen, means=means)
        return Decision(_identity(candidate.request, selection.prepared.context), chosen,
                        selection.frozen_action, seeds, means, samples)
    except Exception as error:
        diagnostics.update(status='failed', reason=str(error), complete=False)
        raise
    finally:
        started = local.started['inner'] if local else 0
        completed = local.completed['inner'] if local else 0
        diagnostics.update(started_simulations=started, completed_simulations=completed,
                           samples=tuple(asdict(row) for row in recorded[:completed]))


def lookup(decision, request, actions, *, context, kernel=None):
    """Read-only public lookup: no seed, draw cursor, model calls or evaluation."""
    from .hif_internal_successor_graph import GraphDecision
    from .hif_internal_successor_graph import lookup as graph_lookup

    if isinstance(decision, GraphDecision):
        return graph_lookup(decision, request, actions, context=context, kernel=kernel)
    if decision is None or _identity(request, _freeze(context)) != decision.identity:
        return None
    keys = tuple(action_key(action) for action in actions)
    if len(set(keys)) != len(keys) or set(keys) != {key for key, _ in decision.means}:
        return None
    return decision.action_key
