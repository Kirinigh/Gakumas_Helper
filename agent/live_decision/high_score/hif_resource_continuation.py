"""Same-instance local resource tables across proved, real public commits.

Before a table entry is reached, normal planning remains in control. Neither
the table nor its pending proof is portable or exported with a search cache.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace, is_dataclass

from .planner import SearchIncomplete
from .hif_mcts import action_key
from .contracts import ActionKind
from .hif_mcts_cache import json_value, public_request
from .hif_choice_roots import _complete_legal
from .validation_cache import validation_session
from .hif_resource_reanalysis import VERSION as RESOURCE_VERSION
from .hif_resource_reanalysis import ResourcePolicy
from .hif_public_turn_reanalysis import VERSION as PUBLIC_VERSION
from .hif_public_turn_reanalysis import _StrictDeterministicKernel
from .hif_frozen_policy_continuation import (
    business_action,
    source_window_ms,
    source_mode_receipt,
    _source_context_matches,
    source_simulation_budget,
    deterministic_public_successor,
)

VERSION = 'hif-resource-policy-continuation-v1'
SCOPE = 'LOCAL_PUBLIC_RESOURCE_CONTINUATION'
_ZERO_RANDOM_PROOF = ('STRICT_PUBLIC_ZERO_RANDOM_PROOF',)


def _integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def source_validation(plan, action):
    """Receipt of completed source work, never new validation samples."""
    table = plan.get('resource_reanalysis') or {}
    rows = plan.get('candidate_statistics')
    elapsed = plan.get('elapsed_ms')
    limit = source_window_ms(plan)
    budget = source_simulation_budget(plan)
    seeds = table.get('planning_seeds', ())
    required = table.get('required_simulations')
    graphs = table.get('graphs')
    if (plan.get('scope') != 'SAMPLING_ESTIMATE' or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('complete_validation_coverage') is not True
            or budget is None
            or type(plan.get('requested_simulations')) is not int or plan['requested_simulations'] != budget
            or not _integer(plan.get('completed_simulations'), 1, budget)
            or limit is None or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed) or not 0 <= elapsed <= limit
            or plan.get('stop_reason') not in ('completed', 'soft_budget_reached')
            or not isinstance(rows, (tuple, list)) or not rows
            or any(not isinstance(row, dict) or any(not _integer(row.get(field), 4)
                   for field in ('training_paired_visits', 'validation_paired_visits')) for row in rows)
            or table.get('version') != RESOURCE_VERSION or table.get('complete') is not True
            or table.get('status') != 'complete' or table.get('required_simulations_exact') is not True
            or not _integer(required, 4, plan['completed_simulations'])
            or required % 4 != 0
            or any(type(table.get(name)) is not int or table[name] != required
                   for name in ('started_simulations', 'completed_simulations'))
            or not isinstance(seeds, (tuple, list)) or len(seeds) != 4
            or any(not _integer(seed, 0, 2**31-1) for seed in seeds) or len(set(seeds)) != 4
            or not isinstance(graphs, (tuple, list)) or not graphs
            or any(type(table.get(name)) is not int or table[name] != len(graphs)
                   for name in ('graph_count', 'prepared_graph_count'))):
        return None
    graph_total = 0
    for row in graphs:
        graph = row.get('public_graph', {}) if isinstance(row, dict) else {}
        count = graph.get('required_simulations')
        if (graph.get('complete') is not True or graph.get('version') != PUBLIC_VERSION
                or graph.get('requested_samples') != 4 or tuple(graph.get('planning_seeds', ())) != tuple(seeds)
                or not _integer(count, 4, budget)
                or graph.get('started_simulations') != count or graph.get('completed_simulations') != count):
            return None
        graph_total += count
    if graph_total != required:
        return None
    return dict(source_scope='SAMPLING_ESTIMATE', source_accepted=True,
        source_root_action=asdict(business_action(action)), complete_root_coverage=True,
        complete_validation_coverage=True, requested_simulations=budget,
        completed_simulations=plan['completed_simulations'],
        training_paired_visits=tuple(row['training_paired_visits'] for row in rows),
        validation_paired_visits=tuple(row['validation_paired_visits'] for row in rows),
        root_selection_scope=plan.get('root_selection_scope'),
        resource_table_complete=True, resource_table_version=RESOURCE_VERSION,
        resource_graph_count=len(graphs), resource_required_simulations=required,
        resource_completed_simulations=required, resource_planning_seeds=tuple(seeds),
        **source_mode_receipt(plan))


def _matches_policy(plan, policy):
    if not isinstance(policy, ResourcePolicy) or policy.version != RESOURCE_VERSION:
        return False
    rows = plan['resource_reanalysis']['graphs']
    if len(policy.decisions) != len(rows):
        return False
    for decision, row in zip(policy.decisions, rows, strict=True):
        graph = decision.graph
        expected_request = row.get('normalized_request')
        if is_dataclass(expected_request):
            expected_request = asdict(expected_request)
        if (decision.graph_index != row.get('graph_index')
                or json_value(decision.origin_indices) != json_value(row.get('origin_indices'))
                or json_value(asdict(decision.normalized_request)) != json_value(expected_request)
                or graph.root_key != (decision.normalized_request.state, decision.normalized_request.drink_slots)
                or set(graph.nodes) != set(graph.choices) or graph.root_key not in graph.nodes
                or any(edge['child'] not in graph.nodes for node in graph.nodes.values()
                       for edge in node['edges'] if 'child' in edge)
                or json_value(decision.options) != json_value(row['public_graph'].get('root_options'))):
            return False
    return True


class ResourcePolicyContinuation:
    """Keep one local policy only while every real step reveals no new world."""

    version = VERSION

    def __init__(self, kernel):
        self.kernel = kernel
        self.reset()

    def reset(self):
        self._pending = self._active = None

    @staticmethod
    def _guard(deadline):
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('resource_continuation_finite_deadline_required')
        if time.perf_counter() >= deadline:
            raise TimeoutError('resource_continuation_deadline')

    def prepare(self, request, context):
        saved, self._pending = self._pending, None
        self._active = None
        if saved is None:
            return
        try:
            request.validate()
            proof = saved['proof']
            if (not proof['committed'] or saved['context'] != context
                    or proof['expected_state'] != request.state
                    or proof['expected_drink_slots'] != tuple(request.drink_slots)
                    or saved['request_context'] != self._request_context(request)
                    or request.state.turn != proof['before'].turn
                    or request.state.turns_remaining != proof['before'].turns_remaining
                    or request.state.turns_remaining <= 2 or request.state.terminal):
                return
            cursor = saved.get('cursor')
            if cursor is not None:
                # This token is a strict-kernel proof, not an invented sampler
                # snapshot. The resource cursor still checks the entire child.
                cursor.advance(saved['before_request'], proof['action'], request,
                               _ZERO_RANDOM_PROOF, _ZERO_RANDOM_PROOF)
                if not cursor.active:
                    return
            self._active = saved
        except Exception:
            self.reset()
            raise

    @staticmethod
    def _request_context(request):
        return (request.plan_type, request.ruleset_id, request.utility,
                request.decision_kind, request.protocol_version)

    def choose(self, request, legal, plan, deadline):
        saved = self._active
        if saved is None:
            return None
        try:
            self._guard(deadline)
            request.validate()
            _complete_legal(self.kernel, request, legal)
            cursor = saved.get('cursor')
            if cursor is None:
                decision = saved['policy'].lookup(request, legal)
                if decision is None:
                    return None  # Preserve the table; do not force a corridor.
                cursor = saved['policy'].cursor(request, legal)
                saved.update(cursor=cursor, decision=decision)
            selected = cursor.choose(request, legal)
            matches = [action for action in legal if action_key(action) == selected]
            if len(matches) != 1:
                raise ValueError('resource_continuation_action_not_uniquely_legal')
            self._guard(deadline)
            choice = cursor.original.policy.choices[cursor.original.expected_key]
            proof = dict(saved['proof'])
            proof.update({name: asdict(proof[name]) for name in ('before', 'action', 'expected_state')})
            proof.update(source_plan_index=saved['source_plan_index'],
                         predecessor_plan_index=saved['predecessor_plan_index'])
            decision = saved['decision']
            plan.update(scope=SCOPE, requested_simulations=0, completed_simulations=0,
                candidate_statistics=[], complete_root_coverage=False,
                complete_validation_coverage=False, stop_reason='matched_committed_local_resource_policy',
                source_plan_index=saved['source_plan_index'],
                predecessor_plan_index=saved['predecessor_plan_index'],
                source_action=saved['source_validation']['source_root_action'],
                source_validation=dict(saved['source_validation']), continuation_proof=proof,
                resource_node=dict(request=public_request(request), options=choice['options'],
                    maximizing_actions=choice['maximizing_actions'], graph_index=decision.graph_index,
                    origin_indices=decision.origin_indices, score_offset=cursor.score_offset),
                resource_policy_continuation_version=VERSION)
            return matches[0]
        except Exception:
            self.reset()
            raise

    def remember(self, request, action, plan, policy, context, plan_index, deadline):
        """Register a complete new table or retain the old one over one step."""
        saved, self._active, self._pending = self._active, None, None
        try:
            self._guard(deadline)
            request.validate()
            if type(plan_index) is not int or plan_index < 0:
                raise ValueError('resource_continuation_invalid_plan_index')
            if plan.get('accepted') is not True or request.ruleset_id != self.kernel.ruleset_id:
                return False
            source = source_validation(plan, action)
            if source is not None:
                if (plan.get('compute_sensitivity_enabled') is True
                        or isinstance(context, tuple) and 'compute_sensitivity' in context
                        ) and not _source_context_matches(plan, context):
                    return False
                if not _matches_policy(plan, policy):
                    return False
                saved = dict(policy=policy, source_plan_index=plan_index, source_validation=source, cursor=None)
                # A new source may itself be a local entry. Carry its chosen
                # deterministic child too, without requiring the source action
                # to agree with the local table's recommendation.
                raw = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                               decision_kind=request.decision_kind)
                cards = dict(request.card_slots)
                legal = tuple(replace(item, source_slot=(
                    next(slot for slot, drink in request.drink_slots if drink == item.drink_id)
                    if item.kind is ActionKind.USE_DRINK else cards.get(item.card_instance_id))) for item in raw)
                decision = policy.lookup(request, legal)
                if decision is not None:
                    saved.update(cursor=policy.cursor(request, legal), decision=decision)
            elif policy is not None or (plan.get('resource_reanalysis') or {}).get('complete') is True:
                return False  # Never replace a valid table with an unproved one.
            elif saved is not None and (saved['context'] != context
                    or saved['request_context'] != self._request_context(request)
                    or saved['proof']['expected_state'] != request.state
                    or saved['proof']['expected_drink_slots'] != tuple(request.drink_slots)):
                return False
            if saved is None:
                return False
            slots = tuple(request.drink_slots)
            if action.kind is ActionKind.USE_DRINK:
                consumed = (action.source_slot, action.drink_id)
                if slots.count(consumed) != 1:
                    raise ValueError('resource_continuation_drink_slot_mismatch')
                slots = tuple(pair for pair in slots if pair != consumed)
            after, _ = deterministic_public_successor(self.kernel, request.state, action,
                plan_type=request.plan_type, deadline=deadline)
            if after is None:
                return False
            proof_kernel = _StrictDeterministicKernel(self.kernel.cards, self.kernel.drinks,
                self.kernel.items, memory_abilities=self.kernel.memory_abilities)
            with validation_session(lambda: self._guard(deadline), reuse_parent=True):
                try:
                    branches = proof_kernel.transition(request.state, replace(action, source_slot=None),
                                                       plan_type=request.plan_type)
                except SearchIncomplete as error:
                    if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                        raise
                    self._guard(deadline)
                    return False
                self._guard(deadline)
                if (len(branches) != 1 or branches[0].probability != 1.
                        or self.kernel._canonical_probability_state(branches[0].state) != after):
                    raise ValueError('resource_continuation_proof_mismatch')
            self._guard(deadline)
            self._pending = dict(saved, context=context, request_context=self._request_context(request),
                predecessor_plan_index=plan_index, before_request=request,
                proof=dict(before=request.state, before_drink_slots=tuple(request.drink_slots),
                    action=business_action(action), expected_state=after, expected_drink_slots=slots,
                    zero_random_consumption=True, proof_kind='STRICT_PUBLIC_ZERO_RANDOM_PROOF',
                    committed=False))
            return True
        except Exception:
            self.reset()
            raise

    def committed(self, before, action, after):
        saved = self._pending
        if saved is None:
            self._active = None
            return
        proof = saved['proof']
        if (not proof['committed'] and before == proof['before']
                and business_action(action) == proof['action'] and after == proof['expected_state']):
            proof['committed'] = True
        else:
            self.reset()
