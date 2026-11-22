"""Complete local resource graphs evaluated against one frozen continuation.

The opt-in caller supplies the established public-state key and the four
training worlds; validation worlds are not input.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace, dataclass, is_dataclass

from .hif_mcts import action_key
from .contracts import HIF_SENSE_RULESET_ID, PlanType, ActionKind, DecisionKind
from .hif_environment import HifEnvironment
from .hif_public_turn_reanalysis import (
    SCORE_TRANSLATION_CONTRACT,
    PublicTurnPolicy,
    prepare_public_turn_graph,
    evaluate_prepared_public_turn_graph,
)
from .hif_public_turn_reanalysis import (
    new_diagnostics as graph_diagnostics,
)

VERSION = 'hif-resource-reanalysis-v1'
VALUE_SCOPE = 'LOCAL_PUBLIC_RESOURCE_POLICY_IMPROVEMENT'
BUDGET_DECLINE_REUSE_VERSION = 'hif-resource-budget-decline-reuse-v1'


class _FrozenDict(dict):
    def _readonly(self, *args, **kwargs):
        raise TypeError('resource_policy_is_immutable')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _request_records(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _request_records(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_request_records(item) for item in value)
    return value


def _context(request):
    # Observation identifiers and input capabilities are never carried forward.
    return (request.plan_type, request.decision_kind, request.utility,
            request.ruleset_id, request.protocol_version)


def new_diagnostics():
    return dict(version=VERSION, enabled=True, eligible=False, complete=False,
        status='not_run', reason='not_prepared', value_scope=VALUE_SCOPE,
        requested_samples=4, planning_seeds=(), enumerated_states=0,
        eligible_states=0, graph_count=0, prepared_graph_count=0,
        node_count=0, edge_count=0, boundary_count=0, boundary_group_count=0,
        required_simulations=0, required_simulations_lower_bound=0,
        required_simulations_exact=False, started_simulations=0,
        completed_simulations=0, score_translation_contract=None, graphs=(),
        budget_decline_reuse=None)


def _guard(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('resource_reanalysis_deadline')


@dataclass(frozen=True)
class PreparedResourceGraphs:
    request: object
    deadline: float
    ready: bool
    key: object
    entries: tuple
    required_simulations: int | None
    diagnostics: dict
    budget_proof: object = None


@dataclass(frozen=True)
class _BudgetDeclineProof:
    """Window-local checked prefix, never exported as a policy or search cache."""

    kernel: object
    catalogue: tuple
    request: object
    key: object
    normalization_context: object
    score_translation_contract: str
    states: tuple
    selected: dict
    budget: int
    deadline: float
    diagnostics: dict


def _catalogue(kernel):
    return (type(kernel), kernel.ruleset_id,
            *(tuple(sorted(getattr(kernel, name).items()))
              for name in ('cards', 'drinks', 'items', 'memory_abilities')))


def _first_graph_budget_decline(diagnostics, budget):
    graphs = diagnostics['graphs']
    if (diagnostics['status'] != 'declined'
            or diagnostics['reason'] != 'resource_graph_declined:complete_graph_rollout_budget'
            or diagnostics['complete'] or diagnostics['prepared_graph_count'] != 0
            or len(graphs) != 1 or diagnostics['started_simulations'] or diagnostics['completed_simulations']):
        return False
    detail = graphs[0]['public_graph']
    return (detail['status'] == 'declined' and detail['reason'] == 'complete_graph_rollout_budget'
            and not detail['complete'] and detail['requested_samples'] == 4
            and not detail['started_simulations'] and not detail['completed_simulations']
            and diagnostics['required_simulations_lower_bound'] == diagnostics['required_simulations']
            == detail['required_simulations'] == 4 * detail['boundary_group_count'] > budget)


def _reusable_budget_proof(previous, kernel, request, states, key, budget, deadline,
                           score_translation_contract, normalization_context):
    proof = previous.budget_proof if isinstance(previous, PreparedResourceGraphs) else None
    if (not isinstance(proof, _BudgetDeclineProof) or previous.ready
            or kernel is not proof.kernel or _catalogue(kernel) != proof.catalogue
            or request != proof.request or key != proof.key
            or normalization_context != proof.normalization_context
            or score_translation_contract != proof.score_translation_contract
            or budget > proof.budget or deadline > proof.deadline
            or len(states) < len(proof.states) or states[:len(proof.states)] != proof.states
            or not _first_graph_budget_decline(proof.diagnostics, proof.budget)):
        return None
    _guard(min(deadline, proof.deadline))
    return proof


@dataclass(frozen=True)
class ResourceDecision:
    graph_index: int
    origin_indices: tuple
    origins: tuple
    normalized_request: object
    action_key: tuple
    options: tuple
    maximizing_actions: tuple
    graph: PublicTurnPolicy
    boundary_groups: tuple
    public_nodes: tuple

    def evidence(self):
        """Portable business evidence, without dictionaries keyed by states."""
        return _request_records(dict(graph_index=self.graph_index, origin_indices=self.origin_indices,
            origins=self.origins, normalized_request=self.normalized_request,
            action_key=self.action_key, options=self.options,
            maximizing_actions=self.maximizing_actions, boundary_groups=self.boundary_groups,
            public_nodes=self.public_nodes))


@dataclass(frozen=True)
class ResourcePolicy:
    version: str
    context: tuple
    key: object
    decisions: tuple

    def normalize(self, request):
        """Use only the caller's already established public-state convention."""
        state, slots = self.key(request)
        return replace(request, state=state, drink_slots=tuple(slots))

    def lookup(self, request, actions):
        request.validate()
        if _context(request) != self.context:
            return None
        current = self.normalize(request)
        keys = tuple(action_key(action) for action in actions)
        if len(set(keys)) != len(keys):
            return None
        for decision in self.decisions:
            expected = decision.normalized_request
            if (current.state == expected.state and current.drink_slots == expected.drink_slots
                    and current.card_slots == expected.card_slots
                    and set(keys) == {option['action_key'] for option in decision.options}):
                return decision
        return None

    def choose(self, request, actions):
        decision = self.lookup(request, actions)
        return None if decision is None else decision.action_key

    def cursor(self, request, actions):
        decision = self.lookup(request, actions)
        if decision is None:
            return None
        offset = request.state.score - decision.normalized_request.state.score
        normalized = self.normalize(request)
        normalized = replace(normalized, state=replace(normalized.state, score=request.state.score-offset))
        cursor = decision.graph.cursor(normalized)
        if cursor is None:
            raise ValueError('resource_reanalysis_cursor_root_mismatch')
        return ResourceCursor(self, cursor, offset)


@dataclass
class ResourceCursor:
    policy: ResourcePolicy
    original: object
    score_offset: float

    @property
    def active(self):
        return self.original.active

    def _request(self, request, expected_key=None):
        if _context(request) != self.policy.context:
            raise ValueError('resource_reanalysis_cursor_context_changed')
        normalized = self.policy.normalize(request)
        normalized = replace(normalized, state=replace(normalized.state, score=request.state.score-self.score_offset))
        expected = self.original.policy.nodes[
            self.original.expected_key if expected_key is None else expected_key]['request']
        expected_normal = self.policy.normalize(expected)
        expected_normal = replace(expected_normal, state=replace(expected_normal.state, score=expected.state.score))
        if (normalized.state == expected_normal.state and normalized.drink_slots == expected.drink_slots
                and normalized.card_slots == expected.card_slots):
            # Graph transitions retain their full state, including counters
            # that just expired. Compare with the caller's established key,
            # then bind to that graph representative without dropping fields.
            return replace(normalized, state=expected.state)
        return normalized

    def choose(self, request, actions):
        return self.original.choose(self._request(request), actions)

    def advance(self, before, action, after, sampler_before, sampler_after):
        if not self.active:
            return
        edges = self.original.policy.nodes[self.original.expected_key]['edges']
        matched = [edge for edge in edges if edge['action_key'] == action_key(action)]
        # Random/pending/cross-turn edges end this graph. Their new metadata
        # deliberately lies outside its context and must not be normalized.
        following = (after if len(matched) == 1 and 'child' not in matched[0]
                     else self._request(after, matched[0]['child'] if len(matched) == 1 else None))
        self.original.advance(self._request(before), action, following, sampler_before, sampler_after)


def prepare_resource_graphs(kernel, request, states, *, key, budget, deadline,
                            diagnostics, score_translation_contract, retain_budget_proof=False,
                            previous=None, normalization_context=None):
    """Enumerate every eligible frozen-tree key before any return is evaluated.

    ``states`` is an ordered tuple of (state, physical drink slots) keys. No
    visits or historical values are used. Duplicate normalized origins share
    one graph; overlapping graphs otherwise pay for their own full boundaries.
    """
    if (type(budget) is not int or budget < 0 or not callable(key)
            or type(retain_budget_proof) is not bool):
        raise ValueError('resource_reanalysis_invalid_parameters')
    if score_translation_contract != SCORE_TRANSLATION_CONTRACT:
        raise ValueError('resource_reanalysis_requires_score_contract')
    request.validate()
    states = tuple(states)
    diagnostics.update(new_diagnostics(), score_translation_contract=score_translation_contract)
    entries, selected = [], {}

    def declined(reason, exact=False):
        diagnostics.update(status='declined', reason=reason, complete=False,
            required_simulations_exact=exact)
        frozen = _freeze(diagnostics)
        proof = (_BudgetDeclineProof(kernel, _catalogue(kernel), request, key,
            normalization_context, score_translation_contract, states, _freeze(selected), budget,
            deadline, frozen) if retain_budget_proof and _first_graph_budget_decline(diagnostics, budget)
            else None)
        return PreparedResourceGraphs(request, deadline, False, key, (), None, frozen, proof)

    try:
        _guard(deadline)
        if (kernel.ruleset_id != HIF_SENSE_RULESET_ID or request.plan_type is not PlanType.SENSE
                or request.decision_kind is not DecisionKind.LIVE or request.utility.baseline_p10 is not None
                or request.state.terminal or request.state.turns_remaining <= 2):
            return declined('outside_resource_domain')
        proof = _reusable_budget_proof(previous, kernel, request, states, key, budget, deadline,
                                       score_translation_contract, normalization_context)
        offset = 0
        if proof is not None:
            # These are logical totals. Only the appended suffix is checked now;
            # the proof below records that actual work separately.
            offset = len(proof.states)
            diagnostics['enumerated_states'] = offset
            diagnostics['eligible_states'] = proof.diagnostics['eligible_states']
            selected = {identity: dict(request=row['request'], origins=list(row['origins']),
                                      indices=list(row['indices']))
                        for identity, row in proof.selected.items()}
        for index, (state, slots) in enumerate(states[offset:], start=offset):
            _guard(deadline)
            diagnostics['enumerated_states'] += 1
            if (state.turn != request.state.turn or state.turns_remaining != request.state.turns_remaining
                    or state.terminal or state.actions_remaining <= 0 or state.deck.hand
                    or state.pending_choice is not None or state.resolution_stack or state.pending_after_card_ids):
                continue
            environment = HifEnvironment(kernel)
            rebound = environment.reset(state, seed=0, drink_slots=tuple(slots))
            origin = replace(request, state=rebound.state, decision_kind=rebound.decision_kind,
                card_slots=rebound.card_slots, drink_slots=rebound.drink_slots)
            actions = tuple(environment.legal_actions())
            kinds = {action.kind for action in actions}
            if len(actions) < 2 or kinds != {ActionKind.END_TURN, ActionKind.USE_DRINK}:
                continue
            keys = tuple(action_key(action) for action in actions)
            if len(set(keys)) != len(keys):
                raise ValueError('resource_reanalysis_nonunique_legal_actions')
            normalized_state, normalized_slots = key(origin)
            current = replace(origin, state=normalized_state, drink_slots=tuple(normalized_slots))
            current.validate()
            if current.state.score != 0 or current.drink_slots != origin.drink_slots:
                raise ValueError('resource_reanalysis_invalid_normalized_key')
            # A caller-supplied key may prune established inert records, but
            # must never alter the legal physical action set.
            environment.reset(current.state, seed=0, drink_slots=current.drink_slots)
            if set(keys) != {action_key(action) for action in environment.legal_actions()}:
                raise ValueError('resource_reanalysis_normalization_changed_actions')
            diagnostics['eligible_states'] += 1
            identity = (current.state, current.drink_slots, current.card_slots)
            selected.setdefault(identity, dict(request=current, origins=[], indices=[]))
            selected[identity]['origins'].append(origin)
            selected[identity]['indices'].append(index)
        diagnostics['graph_count'] = len(selected)
        if not selected:
            return declined('no_eligible_resource_nodes', exact=True)
        diagnostics['eligible'] = True
        if proof is not None:
            # Appended origins may add aliases or graphs, but cannot remove the
            # first existing graph or reduce its already witnessed lower bound.
            for name in ('node_count', 'edge_count', 'boundary_count', 'boundary_group_count',
                         'required_simulations', 'required_simulations_lower_bound', 'graphs'):
                diagnostics[name] = proof.diagnostics[name]
            diagnostics['budget_decline_reuse'] = dict(version=BUDGET_DECLINE_REUSE_VERSION,
                source='internal_reservation_precheck', source_budget=proof.budget, current_budget=budget,
                source_enumerated_states=offset, current_enumerated_states=len(states),
                checked_new_states=len(states)-offset, tree_key_prefix_unchanged=True,
                context_unchanged=True, source_graph_index=0,
                required_simulations_lower_bound=proof.diagnostics['required_simulations_lower_bound'],
                graphs_built=0)
            _guard(deadline)
            return declined('resource_graph_declined:reused_budget_lower_bound')
        total = 0
        for candidate in selected.values():
            _guard(deadline)
            detail = graph_diagnostics()
            prepared = prepare_public_turn_graph(kernel, candidate['request'], rollout_budget=budget,
                deadline=deadline, diagnostics=detail, score_translation_contract=score_translation_contract)
            for name in ('node_count', 'edge_count', 'boundary_count', 'boundary_group_count'):
                diagnostics[name] += detail[name]
            total += prepared.required_simulations_lower_bound
            diagnostics.update(required_simulations=total, required_simulations_lower_bound=total)
            evidence = dict(origin_indices=tuple(candidate['indices']), origins=tuple(candidate['origins']),
                normalized_request=candidate['request'], public_graph=detail)
            diagnostics['graphs'] = (*diagnostics['graphs'], evidence)
            if not prepared.ready:
                return declined('resource_graph_declined:'+detail['reason'])
            entries.append((tuple(candidate['indices']), tuple(candidate['origins']), prepared))
            diagnostics['prepared_graph_count'] += 1
            if total > budget:
                return declined('resource_graph_total_budget', exact=len(entries) == len(selected))
        diagnostics.update(status='prepared', reason='prepared', required_simulations_exact=True)
        return PreparedResourceGraphs(request, deadline, True, key, tuple(entries), total, _freeze(diagnostics))
    except Exception as error:
        diagnostics.update(status='failed', reason=str(error), complete=False)
        raise


def evaluate_resource_graphs(prepared, training_seeds, *, evaluate, deadline, diagnostics):
    """Publish an immutable local table only after every selected graph succeeds."""
    seeds = tuple(training_seeds)
    if (len(seeds) != 4 or len(set(seeds)) != 4
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)):
        raise ValueError('resource_reanalysis_requires_four_original_training_seeds')
    diagnostics.update(prepared.diagnostics)
    diagnostics['planning_seeds'] = seeds
    if not prepared.ready:
        return None
    deadline = min(deadline, prepared.deadline)
    decisions, graph_rows = [], []
    diagnostics.update(status='evaluating', reason='evaluating')
    try:
        for index, (origin_indices, origins, public_graph) in enumerate(prepared.entries):
            _guard(deadline)
            detail = graph_diagnostics()

            def evaluate_counted(current, seed, action, limit):
                _guard(min(deadline, limit))
                diagnostics['evaluation_prefix_depth'] = detail['evaluation_prefix_depth']
                diagnostics['started_simulations'] += 1
                score = evaluate(current, seed, action, limit)
                _guard(min(deadline, limit))
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError('public_turn_reanalysis_nonfinite_return')
                diagnostics['completed_simulations'] += 1
                return score

            evidence = dict(graph_index=index, origin_indices=origin_indices, origins=origins,
                normalized_request=public_graph.request, public_graph=detail)
            graph_rows.append(evidence)
            diagnostics['graphs'] = tuple(graph_rows)
            policy = evaluate_prepared_public_turn_graph(public_graph, seeds,
                evaluate=evaluate_counted, deadline=deadline, diagnostics=detail)
            if policy is None or not detail['complete']:
                raise ValueError('resource_reanalysis_incomplete_prepared_graph')
            frozen = PublicTurnPolicy(policy.root_key, _freeze(policy.nodes), _freeze(policy.choices))
            root = frozen.choices[frozen.root_key]
            boundaries = tuple(dict(row, request=members[0]['request'],
                action_key=action_key(members[0]['action']))
                for row, members in zip(detail['boundary_groups'], public_graph.groups, strict=True))
            public_nodes = tuple(dict(request=node['request'], **policy.choices[node_key],
                edges=tuple(dict(action_key=edge['action_key'], reason=edge['reason'],
                    **({'child_request': policy.nodes[edge['child']]['request']} if 'child' in edge
                       else {'boundary_index': edge['boundary_index']})) for edge in node['edges']))
                for node_key, node in policy.nodes.items())
            evidence['public_nodes'] = public_nodes
            decisions.append(ResourceDecision(index, origin_indices, origins, public_graph.request,
                root['maximizing_actions'][0], root['options'], root['maximizing_actions'],
                frozen, _freeze(boundaries), _freeze(public_nodes)))
        _guard(deadline)
        if diagnostics['completed_simulations'] != prepared.required_simulations:
            raise ValueError('resource_reanalysis_incomplete_total')
        diagnostics.update(status='complete', reason='complete', complete=True)
        return ResourcePolicy(VERSION, _context(prepared.request), prepared.key, tuple(decisions))
    except Exception as error:
        diagnostics.update(status='failed', reason=str(error), complete=False)
        raise
