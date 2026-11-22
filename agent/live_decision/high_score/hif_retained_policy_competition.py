"""One immutable tree policy competes with a newly trained public policy.

The default domain covers tree-only sources. The explicit revealed-policy mode
also freezes complete public/resource tables for renewed comparison, without
approximating them by a tree. No observation bindings or input authority survive.
"""
from __future__ import annotations

import time
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, replace

from .hif import HifSenseKernel
from .kernel import EffectKind
from .hif_mcts import RETAINED_POLICY_COMPETITION_VERSION, REVEALED_POLICY_COMPETITION_VERSION, _Node, action_key
from .contracts import ActionKind, DecisionKind
from .hif_dominance import _plain_positive_effect
from .hif_choice_roots import _complete_legal
from .hif_public_graph_parallel import _frozen_signature

VERSION = RETAINED_POLICY_COMPETITION_VERSION
EXECUTION_VERSION = 'hif-retained-policy-execution-v1'
TREE_EXECUTION_VERSION = 'hif-retained-tree-policy-execution-v2'
REVEALED_VERSION = REVEALED_POLICY_COMPETITION_VERSION


def _complete_graph(graph):
    from .hif_public_turn_reanalysis import PublicTurnCursor, PublicTurnPolicy, public_key

    class CompleteGraph(PublicTurnPolicy):
        def cursor(self, request):
            key = public_key(request)
            return PublicTurnCursor(self, key) if key in self.nodes else None

    return CompleteGraph(graph.root_key, deepcopy(graph.nodes), deepcopy(graph.choices))


def _guard(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('retained_policy_deadline')


class FrozenTreePolicy:
    """Detached tree/helpers, shared immutable rules and exact business solver."""

    def __init__(self, owner, request, source_plan_index, deadline, *, allow_complete=False):
        from .hif_mcts_candidate import HifMctsCandidatePolicy

        _guard(deadline)
        source = owner.search
        if type(allow_complete) is not bool:
            raise ValueError('retained_policy_complete_requires_bool')
        if not allow_complete and (source._public_turn_policy is not None or source._resource_policy is not None):
            raise ValueError('retained_policy_complete_tables_require_existing_continuation')
        kwargs = dict(owner._cache_configuration)
        kwargs.pop('retained_policy_competition', None)
        kwargs.pop('retained_policy_execution', None)
        kwargs.pop('retained_tree_policy_execution', None)
        kwargs.pop('internal_reanalysis', None)
        kwargs.pop('internal_complete_root_successors', None)
        kwargs.pop('internal_successor_graph', None)
        kwargs.pop('uniform_validation_time_guard', None)
        kwargs.pop('paired_adaptive_validation', None)
        kwargs.pop('revealed_policy_competition', None)
        if source.quality_reference and source.internal_reanalysis and source.fresh_adaptive_training:
            # This evaluation-only shell does not own the full learning stack.
            # Restore the frozen search context after constructing the shell,
            # just as for the immutable internal table below.
            kwargs['fresh_adaptive_training'] = False
        self.policy = HifMctsCandidatePolicy(owner.kernel, owner.baseline,
            deadline=deadline, **kwargs)
        # Calls are sequential. These exact planners are independent of which
        # rollout policy reached the state; their complete business memo is safe
        # to share. Tree/heuristic/complete-return caches remain separate.
        self.policy.tail = owner.tail
        target = self.policy.search
        # A frozen competitor evaluates only; it cannot start this new learning
        # phase. Carry its immutable local decision along with the old tree.
        target.internal_reanalysis = source.internal_reanalysis
        target.internal_complete_root_successors = source.internal_complete_root_successors
        target.internal_successor_graph = source.internal_successor_graph
        target.uniform_validation_time_guard = source.uniform_validation_time_guard
        target.paired_adaptive_validation = source.paired_adaptive_validation
        target.fresh_adaptive_training = source.fresh_adaptive_training
        target.revealed_policy_competition = source.revealed_policy_competition
        target._tree = {key: _Node({action: list(values) for action, values in node.statistics.items()},
            node.order, set(node.children)) for key, node in source._tree.items()}
        target._rng.setstate(source._rng.getstate())
        target._adaptive_rng.setstate(source._adaptive_rng.getstate())
        target._training_seed_history = set(source._training_seed_history)
        target._baseline_domain_reason = source._baseline_domain_reason
        target._namespace = target._context(request)
        target._expected_state = source._expected_state
        if source._internal_policy is not None:
            identity = source._internal_policy.identity
            target._internal_policy = replace(source._internal_policy,
                identity=(identity[0], target._context(request), *identity[2:]))
        self.allow_complete = allow_complete
        if allow_complete:
            if source._public_turn_policy is not None:
                target._public_turn_policy = _complete_graph(source._public_turn_policy)
            if source._resource_policy is not None:
                target._resource_policy = replace(source._resource_policy,
                    key=target._resource_key, decisions=deepcopy(source._resource_policy.decisions))
        self.source_tables = dict(public_turn=target._public_turn_policy is not None,
            resource=target._resource_policy is not None, internal=target._internal_policy is not None)
        self._table_signature = deepcopy(self._tables())
        self._resource_key = target._resource_policy.key if target._resource_policy is not None else None
        self._model_context = owner._cache_context() if allow_complete else None
        self.revealed_policy_version = None
        self.validation_excluded_seeds = ()
        self.source_plan_index = source_plan_index
        self.owner = owner
        self.action = None
        self.request = None
        self._signature = _frozen_signature(target)
        _guard(deadline)

    def _tables(self):
        search = self.policy.search
        public = search._public_turn_policy
        resource = search._resource_policy
        return (None if public is None else (public.root_key, public.nodes, public.choices),
                None if resource is None else (resource.version, resource.context, resource.decisions))

    def _verify_frozen(self):
        if (self._signature != _frozen_signature(self.policy.search)
                or self.allow_complete and (self._table_signature != self._tables()
                    or (self.policy.search._resource_policy.key if self.policy.search._resource_policy else None)
                        != self._resource_key
                    or self.owner._cache_context() != self._model_context)):
            raise ValueError('retained_policy_changed_while_evaluating')

    def bind(self, request, legal, deadline):
        _guard(deadline)
        _complete_legal(self.owner.kernel, request, legal)
        # suggest() uses this same public-state normalization before evaluation.
        # The strict actual-commit proof was already checked on the input state.
        request = replace(request, state=self.owner.kernel._canonical_probability_state(request.state))
        search = self.policy.search
        search._order_deadline = deadline
        if search.unknown_node_policy is not None:
            search.unknown_node_policy.start_window()
        search._reset_partial_node_timing()
        search._public_step_reuse = search._new_public_step_reuse(deadline)
        cursor = (search._public_turn_policy.cursor(request)
                  if self.allow_complete and search._public_turn_policy is not None else None)
        if cursor is None and self.allow_complete and search._resource_policy is not None:
            cursor = search._resource_policy.cursor(request, legal)
        key = cursor.choose(request, legal) if cursor is not None else search._validation_action(request, legal)
        matches = [action for action in legal if action_key(action) == key]
        if len(matches) != 1:
            raise ValueError('retained_policy_action_not_uniquely_legal')
        self.action, self.request = matches[0], request
        if self._signature != _frozen_signature(search):
            raise ValueError('retained_policy_changed_while_binding')
        if self.allow_complete:
            self._verify_frozen()
        _guard(deadline)
        return self

    def evaluate(self, request, seed, deadline):
        _guard(deadline)
        if request != self.request or self.action is None:
            raise ValueError('retained_policy_request_changed')
        if seed in self.policy.search._training_seed_history:
            raise ValueError('retained_policy_outer_overlaps_training')
        if self.revealed_policy_version is not None and seed in self.validation_excluded_seeds:
            raise ValueError('revealed_policy_outer_reuses_old_validation')
        if self.owner._session is None or self.owner._tail_session is None:
            raise ValueError('retained_policy_outside_planning_window')
        self.policy._session = self.owner._session
        self.policy._tail_session = self.owner._tail_session
        try:
            result = self.policy.search._validate_simulation(request, seed,
                action_key(self.action), deadline)
            if result[1] or self._signature != _frozen_signature(self.policy.search):
                raise ValueError('retained_policy_mutated_during_comparison')
            if self.allow_complete:
                self._verify_frozen()
            _guard(deadline)
            return result
        finally:
            self.policy._session = self.policy._tail_session = None


class RetainedPolicyCompetition:
    """Lineage eligibility is supplied by the existing strict cohort proof."""

    version = VERSION

    def __init__(self, owner):
        self.owner = owner
        self.reset()

    def reset(self):
        self.frozen = None
        self.predecessor_plan_index = None
        self.context = None
        self.active = False
        self.reason = 'no_retained_policy'
        self.execution_source_plan_index = None
        self.graph_execution_source_plan_index = None

    def _remember_graph_execution(self, plan, plan_index):
        """A newly winning graph permits only its completely matched nodes."""
        self.graph_execution_source_plan_index = None
        if not getattr(self.owner, 'internal_successor_graph', False):
            return
        from .hif_internal_successor_graph import VERSION as GRAPH_VERSION
        from .hif_internal_successor_graph import GraphDecision

        decision = self.frozen.policy.search._internal_policy
        report = plan.get('internal_reanalysis') or {}
        graph = report.get('successor_graph') or {}
        if (all(plan.get(name) is True for name in (
                'accepted', 'complete_root_coverage', 'complete_validation_coverage',
                'internal_successor_graph_enabled'))
                and plan.get('internal_successor_graph_version') == GRAPH_VERSION
                and report.get('complete') is True and report.get('status') == 'complete'
                and report.get('successor_graph_version') == GRAPH_VERSION
                and graph.get('complete') is True and graph.get('status') == 'complete'
                and isinstance(decision, GraphDecision) and decision.complete is True
                and decision.version == GRAPH_VERSION):
            self.graph_execution_source_plan_index = plan_index

    def _graph_execution_evidence(self, request, legal):
        source = self.graph_execution_source_plan_index
        if (source is None or not getattr(self.owner, 'internal_successor_graph', False)
                or source != self.frozen.source_plan_index):
            return None
        from .hif_mcts_cache import public_request
        from .hif_internal_successor_graph import VERSION as GRAPH_VERSION
        from .hif_internal_successor_graph import lookup

        search = self.frozen.policy.search
        selected = lookup(search._internal_policy, request, legal,
                          context=search._context(request), kernel=search.kernel)
        if selected is None:
            return None
        bound = replace(request, state=self.owner.kernel._canonical_probability_state(request.state))
        return dict(version=GRAPH_VERSION, source_plan_index=source,
            public_request=public_request(bound), card_slots=tuple(bound.card_slots),
            legal_action_keys=tuple(action_key(action) for action in legal), selected_action=selected)

    def prepare(self, request, cohort, evidence):
        self.active = False
        if self.frozen is None:
            self.reason = 'no_retained_policy'
            return
        if (len(cohort) != 4 or evidence.get('reused') is not True
                or evidence.get('source_plan_index') != self.predecessor_plan_index
                or self.context != self.owner._cache_context()):
            self.reset()
            self.reason = 'deterministic_commit_or_context_changed'
            return
        # prepare() above has already compared the exact full state, slots,
        # utility and rules after a strict zero-random real commit. A retained
        # policy has no independent path around that proof.
        self.active = True
        self.reason = 'matched_deterministic_winner'

    def proposal(self, request, legal, deadline):
        return self.frozen.bind(request, legal, deadline) if self.active else None

    def choose(self, request, legal, plan, deadline, *, evaluated_complete_policy=False):
        if not self.owner.retained_policy_execution or not self.active:
            return None
        graph = None
        complete_source = False
        if self.execution_source_plan_index is None:
            graph = self._graph_execution_evidence(request, legal)
            if graph is None:
                # The native information-boundary path executes the complete
                # policy used to evaluate the winning root. A first winner has
                # not won an incumbent comparison yet; that does not erase its
                # evaluated continuation. prepare() must still prove the exact
                # deterministic commit, cohort and full business context.
                complete_source = (evaluated_complete_policy
                    and self.frozen.allow_complete
                    and getattr(self, 'revealed', True) is False
                    and (getattr(self, 'proof', None) or {}).get('committed') is True)
                if not complete_source:
                    return None
        action = self.frozen.bind(request, legal, deadline).action
        if graph is not None and action_key(action) != graph['selected_action']:
            raise ValueError('internal_successor_graph_execution_action_changed')
        reconsider = self._uncovered_action_drink(request, legal, action, deadline)
        if reconsider is not None:
            # A sparse continuation has not compared this resource exchange.
            # Keep the incumbent and original cohort for the ordinary search;
            # it may still win. This neither bans the drink nor prices stamina.
            plan['retained_policy_replanning'] = reconsider
            return None
        plan.update(scope='RETAINED_POLICY_CONTINUATION',
            stop_reason='matched_retained_policy_execution',
            retained_policy_execution_source_plan_index=(self.frozen.source_plan_index
                if complete_source else self.execution_source_plan_index
                if graph is None else graph['source_plan_index']),
            requested_simulations=0, completed_simulations=0, started_simulations=0,
            training_started_simulations=0, training_completed_simulations=0,
            validation_started_simulations=0, validation_completed_simulations=0,
            requested_validation_samples=4, complete_root_coverage=False,
            complete_validation_coverage=False, candidate_statistics=[],
            retained_policy_comparison=None, root_selection_scope=None,
            selected_continuation=[])
        if complete_source:
            plan['retained_policy_execution_origin'] = 'evaluated_complete_root_policy'
        if graph is not None:
            plan['internal_successor_graph_execution'] = graph
        return action

    def _uncovered_action_drink(self, request, legal, action, deadline):
        if (not self.owner.retained_tree_policy_execution
                or type(self.owner.kernel) is not HifSenseKernel
                or action.kind is not ActionKind.USE_DRINK):
            return None
        state = request.state
        if (not 0 < len(state.deck.hand) <= state.actions_remaining
                or state.pending_choice or state.pending_after_card_ids or state.resolution_stack):
            return None
        search = self.frozen.policy.search
        bound = self.frozen.request
        if search._internal_policy is not None:
            from .hif_internal_reanalysis import lookup

            if lookup(search._internal_policy, bound, legal,
                      context=search._context(bound), kernel=search.kernel) is not None:
                return None
        _, node = search._physical_node(bound, legal)
        if node is None:
            return None
        visited = sum(bool(row[0]) for row in node.statistics.values())
        if not 0 < visited < len(legal):
            return None
        definition = self.owner.kernel.drinks.get(action.drink_id)
        if (definition is None or definition.stamina_cost != 0 or not definition.effects
                or not all(_plain_positive_effect(effect, {EffectKind.EXTRA_ACTION})
                           for effect in definition.effects)):
            return None
        _guard(deadline)
        branches = self.owner.kernel.transition(state, replace(action, source_slot=None),
                                                plan_type=request.plan_type)
        if len(branches) != 1 or branches[0].probability != 1.0:
            return None
        after = branches[0].state
        remaining = list(state.drinks)
        remaining.remove(action.drink_id)
        if (after.drinks != tuple(remaining)
                or after.actions_remaining != state.actions_remaining + sum(
                    int(effect.amount) for effect in definition.effects)
                or replace(after, drinks=state.drinks, actions_remaining=state.actions_remaining)
                    != state):
            return None
        _guard(deadline)
        return dict(reason='uncovered_extra_action_resource_tradeoff',
            source_plan_index=self.frozen.source_plan_index,
            execution_source_plan_index=self.execution_source_plan_index,
            visited_actions=visited, legal_actions=len(legal),
            hand_count=len(state.deck.hand), actions_remaining=state.actions_remaining,
            proposed_action=action_key(action))

    def evidence(self):
        return dict(source_plan_index=self.frozen.source_plan_index if self.frozen else None,
            predecessor_plan_index=self.predecessor_plan_index, reason=self.reason,
            source_domain='complete_frozen_tree_without_public_tables')

    def remember(self, request, plan, plan_index, deadline):
        if not plan.get('validation_cohort_link_created'):
            self.reset()
            return False
        if plan['scope'] == 'SAMPLING_ESTIMATE':
            self.execution_source_plan_index = None
            self.graph_execution_source_plan_index = None
            comparison = plan.get('retained_policy_comparison') or {}
            if comparison.get('selected'):
                if not self.active or self.frozen is None or not comparison.get('complete'):
                    raise ValueError('retained_policy_winner_not_verified')
                tree_execution = self.owner.retained_tree_policy_execution and all(
                    plan.get(name) is True for name in ('accepted', 'complete_root_coverage',
                                                       'complete_validation_coverage'))
                if self.owner.retained_policy_execution and (tree_execution or any(
                        isinstance(plan.get(name), dict) and plan[name].get('complete') is True
                        for name in ('public_turn_reanalysis', 'resource_reanalysis'))):
                    self.execution_source_plan_index = plan_index
            elif (plan.get('complete_validation_coverage') and plan.get('complete_root_coverage')
                    and self.owner.search._public_turn_policy is None
                    and self.owner.search._resource_policy is None):
                self.frozen = FrozenTreePolicy(self.owner, request, plan_index, deadline)
                self._remember_graph_execution(plan, plan_index)
            else:
                self.reset()
                self.reason = 'complete_public_policy_uses_existing_continuation'
                return False
        elif plan['scope'] not in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION',
                'RETAINED_POLICY_CONTINUATION') or not self.active:
            self.reset()
            return False
        self.context = self.owner._cache_context()
        self.predecessor_plan_index = plan_index
        _guard(deadline)
        return True


class RevealedPolicyCompetition(RetainedPolicyCompetition):
    """A complete policy survives observation only as a fresh competitor."""

    def reset(self):
        super().reset()
        self.revealed = False
        self.proof = None
        self.rules_context = None
        self.validation_seeds = set()
        self._pending_request = self._pending_action = None

    def prepare(self, request, cohort, evidence):
        from .hif_mcts_cache import public_request

        self.active = self.revealed = False
        if self.frozen is None:
            self.reason = 'no_retained_policy'
            return
        proof = self.proof
        if (proof is None or not proof['committed'] or self.context != self.owner._cache_context()
                or self.rules_context != self.owner.search._context(request)
                or proof['after'] != public_request(request)
                or proof['after_card_slots'] != tuple(request.card_slots)
                or proof['ruleset_id'] != request.ruleset_id
                or proof['protocol_version'] != request.protocol_version
                or proof['predecessor_plan_index'] != self.predecessor_plan_index):
            self.reset()
            self.reason = 'revealed_commit_or_context_changed'
            return
        self.active = True
        deterministic = (len(cohort) == 4 and evidence.get('reused') is True
                         and evidence.get('source_plan_index') == self.predecessor_plan_index)
        self.revealed = not deterministic
        self.reason = 'matched_deterministic_winner' if deterministic else 'matched_revealed_public_state'
        if self.revealed:
            self.execution_source_plan_index = None
            self.graph_execution_source_plan_index = None

    def proposal(self, request, legal, deadline):
        if not self.active:
            return None
        self.frozen.revealed_policy_version = REVEALED_VERSION if self.revealed else None
        self.frozen.validation_excluded_seeds = tuple(sorted(self.validation_seeds))
        return self.frozen.bind(request, legal, deadline)

    def choose(self, request, legal, plan, deadline, *, evaluated_complete_policy=False):
        if self.revealed:
            return None
        return super().choose(request, legal, plan, deadline,
                              evaluated_complete_policy=evaluated_complete_policy)

    def evidence(self):
        return dict(super().evidence(), source_domain='complete_frozen_policy')

    def revealed_evidence(self):
        from .hif_mcts_cache import json_value

        frozen = self.frozen
        return json_value(dict(version=REVEALED_VERSION, eligible=self.active,
            mode=('revealed_recompetition' if self.revealed else 'deterministic_continuation')
                 if self.active else 'none', reason=self.reason,
            source_plan_index=frozen.source_plan_index if frozen else None,
            predecessor_plan_index=self.predecessor_plan_index,
            source_domain='complete_frozen_policy',
            source_tables=frozen.source_tables if frozen else dict(public_turn=False, resource=False, internal=False),
            source_training_seeds=sorted(frozen.policy.search._training_seed_history) if frozen else [],
            source_validation_seeds=sorted(self.validation_seeds), proof=self.proof))

    def remember(self, request, plan, plan_index, deadline):
        from .hif_mcts_cache import json_value, public_request, action_from_json

        scope = plan.get('scope')
        if scope == 'SAMPLING_ESTIMATE':
            self.graph_execution_source_plan_index = None
            if not all(plan.get(name) is True for name in (
                    'accepted', 'complete_root_coverage', 'complete_validation_coverage')):
                self.reset()
                return False
            comparison = plan.get('retained_policy_comparison') or {}
            if comparison.get('selected'):
                if not self.active or self.frozen is None or comparison.get('complete') is not True:
                    raise ValueError('revealed_policy_winner_not_verified')
                self.execution_source_plan_index = plan_index if self.owner.retained_policy_execution else None
            else:
                self.frozen = FrozenTreePolicy(self.owner, request, plan_index, deadline, allow_complete=True)
                self.validation_seeds = set()
                self.execution_source_plan_index = None
                self._remember_graph_execution(plan, plan_index)
            self.validation_seeds.update(plan['validation_cohort_evidence']['validation_seeds'])
            if drink_order := plan.get('drink_order_comparison'):
                self.validation_seeds.update(drink_order['started_seeds'])
            adaptive = plan.get('paired_adaptive_validation')
            if adaptive is not None:
                if not adaptive.get('complete'):
                    raise ValueError('revealed_policy_adaptive_validation_incomplete')
                self.validation_seeds.update(pair['seed'] for pair in adaptive['additional_pairs'])
        elif (scope not in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION', 'RETAINED_POLICY_CONTINUATION',
                           'PUBLIC_TURN_POLICY_CONTINUATION', 'LOCAL_PUBLIC_RESOURCE_CONTINUATION')
                or not self.active or self.frozen is None):
            self.reset()
            return False
        action = action_from_json(json_value(plan['action']))
        legal = self.owner.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                               decision_kind=request.decision_kind)
        if replace(action, source_slot=None) not in legal:
            raise ValueError('revealed_policy_commit_action_not_legal')
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in request.drink_slots:
                raise ValueError('revealed_policy_commit_drink_slot_changed')
        elif action.card_instance_id is not None and action.source_slot != dict(request.card_slots).get(action.card_instance_id):
            raise ValueError('revealed_policy_commit_card_slot_changed')
        self.context = self.owner._cache_context()
        self.rules_context = self.owner.search._context(request)
        self.predecessor_plan_index = plan_index
        self.proof = dict(before=public_request(request), before_card_slots=tuple(request.card_slots),
            action=asdict(action), after=None, after_card_slots=None,
            ruleset_id=request.ruleset_id, protocol_version=request.protocol_version,
            source_plan_index=self.frozen.source_plan_index, predecessor_plan_index=plan_index, committed=False)
        self._pending_request = request
        self._pending_action = action
        _guard(deadline)
        return True

    def committed(self, before, action, after):
        from .hif_mcts_cache import public_request

        proof = self.proof
        if (proof is None or proof['committed'] or before != self._pending_request.state
                or action != self._pending_action or after.terminal
                or self.context != self.owner._cache_context()
                or self.rules_context != self.owner.search._context(self._pending_request)):
            self.reset()
            return
        slots = tuple(pair for pair in self._pending_request.drink_slots
                      if action.kind is not ActionKind.USE_DRINK or pair != (action.source_slot, action.drink_id))
        if Counter(drink for _, drink in slots) != Counter(after.drinks):
            self.reset()
            return
        try:
            self.owner.kernel._validate_registered_state(after, plan_type=self._pending_request.plan_type)
            identities = (after.pending_choice.candidate_instance_ids if after.pending_choice
                          else tuple(card.instance_id for card in after.deck.hand))
            current = replace(self._pending_request, state=after, drink_slots=slots,
                card_slots=tuple((identity, index) for index, identity in enumerate(identities)),
                decision_kind=DecisionKind.SECONDARY if after.pending_choice else DecisionKind.LIVE)
            current.validate()
        except (ValueError, TypeError):
            self.reset()
            return
        proof.update(after=public_request(current), after_card_slots=tuple(current.card_slots), committed=True)
