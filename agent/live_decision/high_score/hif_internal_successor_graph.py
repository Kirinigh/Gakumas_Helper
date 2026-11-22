"""One selected successor's complete turn graph, priced by frozen returns.

The caller selects the successor before evaluation and owns all complete-return
accounting. Boundary worlds are conditional training samples, not replayed root
trajectories. Execution matches complete public states, including their scores.
"""
from __future__ import annotations

import math
from dataclasses import replace, dataclass

from . import hif_internal_reanalysis as internal
from .hif_mcts import action_key
from .contracts import ActionKind
from .hif_choice_roots import _complete_legal
from .hif_resource_reanalysis import _freeze, _request_records
from .hif_public_turn_reanalysis import (
    PublicTurnPolicy,
    prepare_public_turn_graph,
    evaluate_prepared_public_turn_graph,
)
from .hif_public_turn_reanalysis import new_diagnostics as graph_diagnostics

VERSION = 'hif-internal-successor-graph-v1'
VALUE_SCOPE = 'LOCAL_CONDITIONAL_PAIRED_TURN_GRAPH_FROZEN_RETURNS'


@dataclass(frozen=True)
class NodeRecord:
    identity: tuple
    legal_action_keys: tuple
    action_key: tuple


@dataclass(frozen=True)
class GraphDecision:
    identity: tuple
    graph: PublicTurnPolicy
    records: tuple[NodeRecord, ...]
    action_key: tuple
    frozen_action: tuple
    planning_seeds: tuple
    means: tuple
    value_scope: str = VALUE_SCOPE
    version: str = VERSION
    complete: bool = True


def new_diagnostics():
    return dict(successor_graph_version=VERSION, successor_graph=graph_diagnostics(),
                successor_request=None, successor_public_nodes=())


def _canonical_request(kernel, request):
    # This is the existing environment/FrozenTreePolicy normalization. In
    # particular it never removes score, history, schedules or physical slots.
    return replace(request, state=kernel._canonical_probability_state(request.state))


def _records(kernel, graph):
    records = {}
    for key, node in graph.nodes.items():
        current = _canonical_request(kernel, node['request'])
        keys = tuple(edge['action_key'] for edge in node['edges'])
        chosen = graph.choices[key]['maximizing_actions'][0]
        if len(set(keys)) != len(keys) or chosen not in keys:
            raise ValueError('internal_successor_graph_invalid_legal_choices')
        identity = internal._identity(current, ())[2:]
        record = NodeRecord(identity, keys, chosen)
        if identity in records and records[identity] != record:
            raise ValueError('internal_successor_graph_conflicting_canonical_node')
        records[identity] = record
    return tuple(records.values())


def _public_nodes(graph):
    return _request_records(tuple(dict(request=node['request'], **graph.choices[key],
        edges=tuple(dict(action_key=edge['action_key'], reason=edge['reason'],
            **({'child_request': graph.nodes[edge['child']]['request']} if 'child' in edge
               else {'boundary_index': edge['boundary_index']})) for edge in node['edges']))
        for key, node in graph.nodes.items()))


def evaluate(selection, training_seeds, *, kernel, evaluate_return, frozen_signature,
             remaining_budget, deadline, diagnostics, validation_seeds=(),
             score_translation_contract=None):
    """Prepare once; publish only after all four worlds of every group succeed.

    evaluate_return(request, seed, key, deadline) returns an absolute complete
    score. Its caller validates the complete-return contract and charges actual
    work. Before each call, successor_graph.evaluation_prefix_depth carries the
    formal maximum prefix depth. The new graph must remain disabled throughout
    evaluation; frozen_signature checks the caller's original tree/RNG/tables.
    """
    diagnostics.update(new_diagnostics(), **internal._selection_evidence(selection))
    detail = diagnostics['successor_graph']
    diagnostics.update(complete=False, eligible=True, status='preparing',
        reason='preparing_successor_graph', value_scope=VALUE_SCOPE,
        selected_action=None, means=(), samples=(), started_simulations=0,
        completed_simulations=0)
    try:
        internal._guard(deadline)
        seeds, outer = tuple(training_seeds), tuple(validation_seeds)
        if outer and (len(outer) < 4 or len(set(outer)) != len(outer)
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in outer)):
            raise ValueError('internal_successor_graph_invalid_validation_seeds')
        if (len(seeds) != 4 or len(set(seeds)) != 4
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or set(seeds).intersection(outer)):
            raise ValueError('internal_successor_graph_four_disjoint_training_seeds_required')
        if (type(remaining_budget) is not int or not 0 <= remaining_budget <= 256
                or selection.candidate not in selection.prepared.candidates
                or not callable(evaluate_return) or not callable(frozen_signature)):
            raise ValueError('internal_successor_graph_invalid_configuration')
        candidate = selection.candidate
        current = _canonical_request(kernel, candidate.request)
        _complete_legal(kernel, current, candidate.actions)
        keys = tuple(action_key(action) for action in candidate.actions)
        if len(set(keys)) != len(keys) or selection.frozen_action not in keys:
            raise ValueError('internal_successor_graph_incomplete_root_actions')
        diagnostics.update(planning_seeds=seeds, validation_seeds=outer,
                           training_validation_disjoint=bool(outer))
        detail.update(planning_seeds=seeds, rollout_budget=remaining_budget)
        before = frozen_signature()
        prepared = prepare_public_turn_graph(kernel, current,
            rollout_budget=remaining_budget, deadline=deadline, diagnostics=detail,
            score_translation_contract=score_translation_contract)
        diagnostics['requested_simulations'] = prepared.required_simulations_lower_bound
        if frozen_signature() != before:
            raise ValueError('internal_successor_graph_frozen_policy_changed')
        if not prepared.ready:
            diagnostics.update(eligible=False, status='declined', reason=detail['reason'])
            return None
        if ({edge['action_key'] for edge in prepared.nodes[prepared.root_key]['edges']} != set(keys)
                or prepared.required_simulations > remaining_budget):
            raise ValueError('internal_successor_graph_complete_coverage_mismatch')

        def counted(request, seed, key, limit):
            # detail is the same dict the formal evaluator updates before this
            # call, so the caller sees the correct prefix depth immediately.
            if frozen_signature() != before:
                raise ValueError('internal_successor_graph_frozen_policy_changed')
            value = evaluate_return(request, seed, key, limit)
            if frozen_signature() != before:
                raise ValueError('internal_successor_graph_frozen_policy_changed')
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('internal_successor_graph_incomplete_return')
            return value

        graph = evaluate_prepared_public_turn_graph(prepared, seeds, evaluate=counted,
                                                    deadline=deadline, diagnostics=detail)
        if (graph is None or not detail['complete']
                or detail['started_simulations'] != prepared.required_simulations
                or detail['completed_simulations'] != prepared.required_simulations):
            raise ValueError('internal_successor_graph_incomplete_pairs')
        frozen = PublicTurnPolicy(graph.root_key, _freeze(graph.nodes), _freeze(graph.choices))
        records = _records(kernel, frozen)
        root = frozen.choices[frozen.root_key]
        chosen = root['maximizing_actions'][0]
        means = tuple((row['action_key'], row['mean_final_score']) for row in root['options'])
        decision = GraphDecision(internal._identity(current, _freeze(selection.prepared.context)),
            frozen, records, chosen, selection.frozen_action, seeds, means)
        internal._guard(deadline)
        diagnostics.update(complete=True, status='complete', reason='complete_successor_graph',
            selected_action=chosen, means=means, successor_request=_request_records(current),
            successor_public_nodes=_public_nodes(frozen))
        return decision
    except Exception as error:
        diagnostics.update(status='failed', reason=str(error), complete=False,
                           selected_action=None, means=(), successor_request=None,
                           successor_public_nodes=())
        raise
    finally:
        diagnostics.update(started_simulations=detail['started_simulations'],
                           completed_simulations=detail['completed_simulations'])


def lookup(decision, request, actions, *, context, kernel):
    """Exact immutable public lookup; a mismatch returns to the frozen policy."""
    if (not isinstance(decision, GraphDecision) or not decision.complete
            or decision.version != VERSION or kernel is None
            or decision.identity[0] != internal.VERSION
            or decision.identity[1] != _freeze(context)):
        return None
    current = _canonical_request(kernel, request)
    identity = internal._identity(current, ())[2:]
    record = next((row for row in decision.records if row.identity == identity), None)
    if record is None:
        return None
    keys = tuple(action_key(action) for action in actions)
    if len(set(keys)) != len(keys) or set(keys) != set(record.legal_action_keys):
        return None
    slots = dict(current.card_slots)
    for action in actions:
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in current.drink_slots:
                return None
        elif action.card_instance_id is not None and (
                action.card_instance_id not in slots or action.source_slot != slots[action.card_instance_id]):
            return None
    return record.action_key
