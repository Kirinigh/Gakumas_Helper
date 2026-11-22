"""Complete public current-turn policy improvement over a frozen HIF policy."""
from __future__ import annotations

import math
import time
from dataclasses import replace, dataclass

from .kernel import KernelError
from .planner import SearchIncomplete
from .hif_mcts import action_key
from .contracts import HIF_SENSE_RULESET_ID, PlanType, ActionKind
from .hif_environment import HifEnvironment
from .hif_turn_chain_continuation import _NoDrawKernel
from .hif_frozen_policy_continuation import (
    deterministic_public_successor,
    _deterministic_public_successor,
)

VERSION = 'hif-public-turn-reanalysis-v2'
SCORE_TRANSLATION_CONTRACT = 'hif-score-translation-v1'
MAX_GRAPH_NODES = 128
MAX_GRAPH_EDGES = 512


def public_key(request):
    return request.state, tuple(request.drink_slots)


def boundary_group_key(request, action):
    # HIF score translation only: preserve every other field and physical identity.
    return replace(request.state, score=0), tuple(request.drink_slots), action_key(action)


def new_diagnostics():
    return dict(version=VERSION, enabled=True, eligible=False, complete=False,
        status='not_run', reason='training_incomplete', node_count=0, edge_count=0,
        boundary_count=0, boundary_group_count=0, required_simulations=0,
        started_simulations=0, completed_simulations=0, requested_samples=4,
        planning_seeds=(), root_options=(), boundary_groups=(),
        value_scope='PAIRED_TRAINING_BOUNDARY_MEAN_THEN_PUBLIC_MAX',
        seedwise_maximization=False, original_tree_values_used=False)


class _DomainDeclined(Exception):
    pass


class _StrictDeterministicKernel(_NoDrawKernel):
    """Reject sampling before probability branches can be merged away."""

    @staticmethod
    def _draw_candidates(deck):
        raise SearchIncomplete('public_turn_random_consumption')

    def _merge_probability_branches(self, branches):
        if len(branches) > 1:
            raise SearchIncomplete('public_turn_random_consumption')
        return super()._merge_probability_branches(branches)


def _strict_public_successor(kernel, proof, state, action, *, plan_type, deadline, guard):
    """Use the graph's fixed strict kernel once on ordinary deterministic edges."""
    try:
        successor, reason = _deterministic_public_successor(kernel, state, action,
            plan_type=plan_type, deadline=deadline, public_turn_proof=proof)
    except (SearchIncomplete, KernelError) as error:
        if isinstance(error, SearchIncomplete) and str(error) != 'public_turn_random_consumption':
            raise
        # The older probe can reject before the strict phase would be reached.
        # Preserve that reason (and any validation error), including branches
        # that become a single public result only after merging randomness.
        successor, reason = deterministic_public_successor(kernel, state, action,
            plan_type=plan_type, deadline=deadline)
        if successor is None:
            return successor, reason
        if isinstance(error, KernelError):
            raise
        guard()
        return None, 'random_consumption_boundary'
    if successor is not None:
        guard()
    return successor, reason


def backup_boundary_means(nodes, postorder, boundary_means):
    """Propagate absolute mean final scores, never maxima chosen per sample."""
    if len(postorder) != len(nodes) or set(postorder) != set(nodes):
        raise ValueError('public_turn_reanalysis_incomplete_postorder')
    if not all(type(value) in (int, float) and math.isfinite(value) for value in boundary_means.values()):
        raise ValueError('public_turn_reanalysis_nonfinite_boundary_mean')
    table, used = {}, set()
    for key in postorder:
        options = []
        for edge in nodes[key]['edges']:
            if 'child' in edge:
                if edge['child'] not in table:
                    raise ValueError('public_turn_reanalysis_incomplete_child')
                value = table[edge['child']]['mean_final_score']
            else:
                index = edge['boundary_index']
                if index not in boundary_means:
                    raise ValueError('public_turn_reanalysis_incomplete_boundary')
                used.add(index)
                value = boundary_means[index]
            options.append(dict(action_key=edge['action_key'], mean_final_score=value))
        if not options or len({row['action_key'] for row in options}) != len(options):
            raise ValueError('public_turn_reanalysis_incomplete_legal_options')
        best = max(row['mean_final_score'] for row in options)
        table[key] = dict(options=tuple(options), mean_final_score=best,
            maximizing_actions=tuple(row['action_key'] for row in options if row['mean_final_score'] == best))
    if used != set(boundary_means):
        raise ValueError('public_turn_reanalysis_unattached_boundary')
    return table


@dataclass(frozen=True)
class PublicTurnPolicy:
    root_key: tuple
    nodes: dict
    choices: dict

    def cursor(self, request):
        return PublicTurnCursor(self, self.root_key) if public_key(request) == self.root_key else None


@dataclass
class PublicTurnCursor:
    policy: PublicTurnPolicy
    expected_key: tuple
    active: bool = True

    def choose(self, request, actions):
        if not self.active or public_key(request) != self.expected_key:
            raise ValueError('public_turn_reanalysis_public_state_mismatch')
        node = self.policy.nodes[self.expected_key]
        if {action_key(action) for action in actions} != {edge['action_key'] for edge in node['edges']}:
            raise ValueError('public_turn_reanalysis_legal_actions_mismatch')
        return self.policy.choices[self.expected_key]['maximizing_actions'][0]

    def advance(self, before, action, after, sampler_before, sampler_after):
        if not self.active:
            return
        if public_key(before) != self.expected_key:
            raise ValueError('public_turn_reanalysis_public_state_mismatch')
        edges = [edge for edge in self.policy.nodes[self.expected_key]['edges']
                 if edge['action_key'] == action_key(action)]
        if len(edges) != 1:
            raise ValueError('public_turn_reanalysis_action_mismatch')
        edge = edges[0]
        if 'child' not in edge:
            self.active = False
            return
        if public_key(after) != edge['child'] or sampler_before != sampler_after:
            raise ValueError('public_turn_reanalysis_deterministic_successor_mismatch')
        self.expected_key = edge['child']


@dataclass(frozen=True)
class PreparedPublicTurnGraph:
    """Same-window topology only; no sampled returns or portable cache state.

    A declined traversal exposes a lower bound, never a purported exact cost.
    Its partial nodes are discarded so they cannot become an executable policy.
    """

    request: object
    deadline: float
    ready: bool
    required_simulations: int | None
    required_simulations_lower_bound: int
    root_key: tuple | None
    nodes: dict
    postorder: tuple
    groups: tuple
    diagnostics: dict


def _check_training_seeds(training_seeds):
    if len(training_seeds) != 4 or len(set(training_seeds)) != 4 or any(type(seed) is not int for seed in training_seeds):
        raise ValueError('public_turn_reanalysis_requires_four_original_training_seeds')


def _check_graph_parameters(rollout_budget, score_translation_contract):
    if type(rollout_budget) is not int or rollout_budget < 0:
        raise ValueError('public_turn_reanalysis_invalid_rollout_budget')
    if score_translation_contract not in (None, SCORE_TRANSLATION_CONTRACT):
        raise ValueError('public_turn_reanalysis_unknown_score_translation_contract')


def prepare_public_turn_graph(kernel, request, *, rollout_budget, deadline,
                              diagnostics, score_translation_contract=None):
    """Prove complete topology once, before training or any boundary returns."""
    _check_graph_parameters(rollout_budget, score_translation_contract)
    diagnostics['score_translation_contract'] = score_translation_contract
    nodes, boundaries, postorder, visiting, groups = {}, [], [], set(), {}
    proof = _StrictDeterministicKernel(kernel.cards, kernel.drinks, kernel.items,
                                      memory_abilities=kernel.memory_abilities)

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError('public_turn_reanalysis_deadline')

    def visit(current):
        guard()
        key = public_key(current)
        if key in visiting:
            raise _DomainDeclined('deterministic_graph_cycle')
        if key in nodes:
            return key
        if len(nodes) >= MAX_GRAPH_NODES:
            raise _DomainDeclined('graph_structure_limit')
        visiting.add(key)
        environment = HifEnvironment(kernel)
        environment.reset(current.state, seed=0, drink_slots=current.drink_slots)
        actions = tuple(environment.legal_actions())
        if not actions:
            raise ValueError('public_turn_reanalysis_no_legal_actions')
        node = nodes[key] = dict(request=current, edges=[])
        diagnostics['node_count'] = len(nodes)
        for action in actions:
            diagnostics['edge_count'] += 1
            if diagnostics['edge_count'] > MAX_GRAPH_EDGES:
                raise _DomainDeclined('graph_structure_limit')
            successor, reason = _strict_public_successor(kernel, proof, current.state, action,
                plan_type=current.plan_type, deadline=deadline, guard=guard)
            if (successor is not None and action.kind is ActionKind.USE_DRINK
                    and (successor.pending_choice is not None or successor.resolution_stack)):
                successor, reason = None, 'drink_unresolved_continuation_boundary'
            edge = dict(action_key=action_key(action), reason=reason)
            node['edges'].append(edge)
            if successor is None:
                index = len(boundaries)
                edge['boundary_index'] = index
                boundary = dict(request=current, action=action, index=index)
                boundaries.append(boundary)
                group_key = (boundary_group_key(current, action) if score_translation_contract
                             else (public_key(current), action_key(action)))
                grouped = groups.setdefault(group_key, [])
                grouped.append(boundary)
                diagnostics.update(boundary_count=len(boundaries), boundary_group_count=len(groups),
                    required_simulations=4*len(groups))
                if 4*len(groups) > rollout_budget:
                    raise _DomainDeclined('complete_graph_rollout_budget')
            else:
                slots = tuple(current.drink_slots)
                if action.kind is ActionKind.USE_DRINK:
                    consumed = (action.source_slot, action.drink_id)
                    if slots.count(consumed) != 1:
                        raise ValueError('public_turn_reanalysis_drink_slot_mismatch')
                    slots = tuple(pair for pair in slots if pair != consumed)
                # Reset only rebinds observation metadata; it does not execute
                # the already proved transition or consume a sampled event.
                rebound = environment.reset(successor, seed=0, drink_slots=slots)
                edge['child'] = visit(replace(current, state=rebound.state,
                    decision_kind=rebound.decision_kind, card_slots=rebound.card_slots,
                    drink_slots=rebound.drink_slots))
        visiting.remove(key)
        postorder.append(key)
        return key

    try:
        if (kernel.ruleset_id != HIF_SENSE_RULESET_ID or request.plan_type is not PlanType.SENSE
                or request.utility.baseline_p10 is not None
                or request.state.terminal or request.state.turns_remaining <= 2):
            raise _DomainDeclined('outside_hif_public_turn_domain')
        root = visit(request)
        # Full states can be reached by more than one prefix. Preserve the
        # original 256-step limit for every alias, including the longest one.
        prefix_lengths = {root: 0}
        for key in reversed(postorder):
            length = prefix_lengths[key]
            for edge in nodes[key]['edges']:
                if 'child' in edge:
                    child = edge['child']
                    prefix_lengths[child] = max(prefix_lengths.get(child, 0), length + 1)
                else:
                    boundaries[edge['boundary_index']]['prefix_depth'] = length
        diagnostics.update(eligible=True, status='prepared', reason='prepared')
        return PreparedPublicTurnGraph(request, deadline, True, 4*len(groups), 4*len(groups),
            root, nodes, tuple(postorder), tuple(tuple(members) for members in groups.values()), dict(diagnostics))
    except _DomainDeclined as error:
        diagnostics.update(status='declined', reason=str(error), eligible=False)
        return PreparedPublicTurnGraph(request, deadline, False, None, 4*len(groups),
            None, {}, (), (), dict(diagnostics))
    except Exception as error:
        diagnostics.update(status='failed', reason=str(error), complete=False)
        raise


def evaluate_prepared_public_turn_graph(prepared, training_seeds, *, evaluate,
                                       deadline, diagnostics):
    """Evaluate the frozen graph with all four original training worlds."""
    _check_training_seeds(training_seeds)
    diagnostics.update(prepared.diagnostics)
    diagnostics['planning_seeds'] = tuple(training_seeds)
    if not prepared.ready:
        return None
    deadline = min(deadline, prepared.deadline)

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError('public_turn_reanalysis_deadline')

    try:
        guard()
        diagnostics.update(eligible=True, status='evaluating', reason='evaluating')
        means, group_rows = {}, []
        for members in prepared.groups:
            representative = members[0]
            diagnostics['evaluation_prefix_depth'] = max(member['prefix_depth'] for member in members)
            scores = []
            for seed in training_seeds:
                guard()
                diagnostics['started_simulations'] += 1
                score = evaluate(representative['request'], seed, action_key(representative['action']), deadline)
                guard()
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError('public_turn_reanalysis_nonfinite_return')
                diagnostics['completed_simulations'] += 1
                scores.append(score)
            mean = sum(scores)/4
            for member in members:
                means[member['index']] = mean+member['request'].state.score-representative['request'].state.score
            group_rows.append(dict(planning_seeds=tuple(training_seeds), final_scores=tuple(scores),
                prefix_depth=diagnostics['evaluation_prefix_depth'],
                representative_boundary=representative['index'], boundary_indices=tuple(member['index'] for member in members)))
        table = backup_boundary_means(prepared.nodes, prepared.postorder, means)
        guard()
        diagnostics.update(status='complete', reason='complete', complete=True,
            root_options=table[prepared.root_key]['options'], boundary_groups=tuple(group_rows))
        return PublicTurnPolicy(prepared.root_key, prepared.nodes, table)
    except Exception as error:
        diagnostics.update(status='failed', reason=str(error), complete=False)
        raise


def build_public_turn_policy(kernel, request, training_seeds, *, evaluate, rollout_budget,
                             deadline, diagnostics, score_translation_contract=None):
    """Compatibility wrapper: prepare once, then evaluate the complete graph."""
    _check_training_seeds(training_seeds)
    _check_graph_parameters(rollout_budget, score_translation_contract)
    diagnostics['planning_seeds'] = tuple(training_seeds)
    prepared = prepare_public_turn_graph(kernel, request, rollout_budget=rollout_budget,
        deadline=deadline, diagnostics=diagnostics, score_translation_contract=score_translation_contract)
    return evaluate_prepared_public_turn_graph(prepared, training_seeds, evaluate=evaluate,
        deadline=deadline, diagnostics=diagnostics)
