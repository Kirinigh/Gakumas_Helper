"""Window-local current-turn action advice, never a sampled final return."""
from __future__ import annotations

import math
import time
from collections import Counter

from .hif import HifSenseKernel, _HifTurnBoundary, is_hif_turn_boundary
from .choice import _valid_branches
from .kernel import EffectKind, EffectSpec
from .planner import PlannerConfig, SearchIncomplete
from .contracts import PlanType, ActionKind, StrategyState, TimedModifier
from .state_copy import replace
from .hif_planner import HifActionChainPlanner
from .hif_forecast import sense_future_score
from .validation_cache import validation_session, check_search_budget
from .hif_card_use_forecast import installed_card_use_bonus
from .hif_end_turn_focus_projection import VERSION as END_TURN_FOCUS_VERSION
from .hif_end_turn_focus_projection import project_end_turn_focus

MAX_EXPANDED_NODES = 32
MAX_DECISION_DEPTH = 32
SCORE_ARITHMETIC_CACHE_LIMIT = 4096
FINITE_CONSUMABLES_MAX_EXPANDED_NODES = 512
FINITE_CONSUMABLES_VERSION = 'finite-consumables-turn-chain-v1'
_FINITE_CONSUMABLES_DECLINES = frozenset({
    'finite_turn_unresolved', 'finite_turn_action_domain', 'finite_turn_new_consumable',
    'finite_turn_boundary_domain', 'finite_turn_consumption_not_strict',
    'finite_turn_unbounded_end',
})
_CACHE_MISS = object()


class _NoDraw:
    def _draw_one(self, state, *, source_id):
        if len(state.deck.hand) < state.hand_limit:
            raise SearchIncomplete('turn_chain_draw_domain')
        return super()._draw_one(state, source_id=source_id)

    def _independent_draw_batch(self, state, count, *, source_id):
        # Route even optimized draws through the same full-hand check.
        return None


class _NoDrawBoundary(_NoDraw, _HifTurnBoundary):
    pass


class _NoDrawKernel(_NoDraw, HifSenseKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Complete immutable physical results only. The planner still visits
        # every node and applies its ordinary cold-search budget and utility.
        self._transitions = {}
        self._boundaries = {}
        self._concentration_percentages = {}
        self._effective_concentrations = {}
        self._score_finishes = {}
        self.reuse_counts = Counter()

    def _score_finish(self, parameter_gain, score_buff_amounts, score_multiplier):
        check_search_budget()
        # These are all operands of the inherited pure final arithmetic. Keep
        # types and buff order; unusual inputs retain the original behavior.
        values = (parameter_gain, *score_buff_amounts, score_multiplier) if type(score_buff_amounts) is tuple else ()
        key = (tuple((type(value), value) for value in values) if values and all(
            type(value) is int or type(value) is float and math.isfinite(value) for value in values) else None)
        cached = self._score_finishes.get(key, _CACHE_MISS) if key is not None else _CACHE_MISS
        if cached is not _CACHE_MISS:
            check_search_budget()
            self.reuse_counts['score_arithmetic_cache_hits'] += 1
            return cached
        result = super()._score_finish(parameter_gain, score_buff_amounts, score_multiplier)
        check_search_budget()
        if key is not None and len(self._score_finishes) < SCORE_ARITHMETIC_CACHE_LIMIT:
            self._score_finishes[key] = result
        self.reuse_counts['score_arithmetic_computations'] += 1
        return result

    def score_effect_gain(self, state, effect, source_id, *, amount=None):
        parent = super().score_effect_gain
        if (type(state) is not StrategyState or type(effect) is not EffectSpec
                or effect.kind is not EffectKind.SCORE or type(state.concentration) is not int
                or type(state.concentration_effect_buffs) is not tuple
                or not state.concentration_effect_buffs
                or any(type(buff) is not TimedModifier or type(buff.amount) not in (int, float)
                       for buff in state.concentration_effect_buffs)
                or (amount is not None and type(amount) not in (int, float))):
            return parent(state, effect, source_id, amount=amount)
        check_search_budget()
        # Resolve dynamic amounts against the untouched state, at the original
        # point before concentration arithmetic (including its error behavior).
        resolved_amount = self.effect_amount(state, effect) if amount is None else amount
        key = ((type(state.concentration), state.concentration),
               tuple((type(buff.amount), buff.amount) for buff in state.concentration_effect_buffs))
        if key in self._effective_concentrations:
            concentration = self._effective_concentrations[key]
            check_search_budget()
            self.reuse_counts['effective_concentration_cache_hits'] += 1
        else:
            concentration = math.ceil(state.concentration * (
                1 + sum(self._concentration_percentage(buff.amount)
                        for buff in state.concentration_effect_buffs)))
            check_search_budget()
            self._effective_concentrations[key] = concentration
            self.reuse_counts['effective_concentration_computations'] += 1
        # Only this pure score formula sees the temporary arithmetic view.
        # Conditions, transitions, cache state keys and callers keep state.
        scoring = replace(state, concentration=concentration, concentration_effect_buffs=())
        result = parent(scoring, effect, source_id, amount=resolved_amount)
        check_search_budget()
        return result

    def _concentration_percentage(self, amount):
        check_search_budget()
        # Cache only exact native numeric types; unusual inputs retain the
        # original conversion and exception behavior, including unhashables.
        key = (type(amount), amount) if type(amount) in (int, float) else None
        if key is not None and key in self._concentration_percentages:
            result = self._concentration_percentages[key]
            check_search_budget()
            self.reuse_counts['concentration_percentage_cache_hits'] += 1
            return result
        result = super()._concentration_percentage(amount)
        check_search_budget()
        if key is not None:
            self._concentration_percentages[key] = result
        self.reuse_counts['concentration_percentage_computations'] += 1
        return result

    def _reuse(self, table, name, state, action, plan_type, compute):
        check_search_budget()
        key = (state, action, plan_type)
        result = table.get(key, _CACHE_MISS)
        if result is not _CACHE_MISS:
            self.reuse_counts[name + '_cache_hits'] += 1
        else:
            result = compute()
            check_search_budget()
            table[key] = result
            self.reuse_counts[name + '_computations'] += 1
        check_search_budget()
        return result

    def transition(self, state, action, *, plan_type):
        parent = super().transition
        return self._reuse(self._transitions, 'transition', state, action, plan_type,
                           lambda: parent(state, action, plan_type=plan_type))

    def turn_boundary_transition(self, state, action, *, plan_type):
        # Preserve the existing unresolved next-turn boundary. Current-turn
        # end triggers still use the draw guard; next-turn refill is not run.
        if state.hand_support or action.kind is not ActionKind.END_TURN or state.turns_remaining < 2:
            return None
        return self._reuse(self._boundaries, 'boundary', state, action, plan_type,
            lambda: _NoDrawBoundary(self, state.turn + 1).transition(state, action, plan_type=plan_type))


class _FiniteConsumablesKernel(_NoDrawKernel):
    """Keep every probability branch while proving current resources decrease."""

    @staticmethod
    def _clean(state):
        if state.pending_choice or state.resolution_stack or state.pending_after_card_ids:
            raise SearchIncomplete('finite_turn_unresolved')

    def legal_actions(self, state, **kwargs):
        self._clean(state)
        actions = super().legal_actions(state, **kwargs)
        if any(action.kind not in (ActionKind.PLAY_CARD, ActionKind.USE_DRINK,
                                   ActionKind.END_TURN) for action in actions):
            raise SearchIncomplete('finite_turn_action_domain')
        return actions

    def _prove(self, before, branches, *, boundary=False):
        _valid_branches(branches)
        hand = Counter(card.instance_id for card in before.deck.hand)
        drinks = Counter(before.drinks)
        for branch in branches:
            check_search_budget()
            after = branch.state
            if (Counter(card.instance_id for card in after.deck.hand) - hand
                    or Counter(after.drinks) - drinks):
                raise SearchIncomplete('finite_turn_new_consumable')
            if boundary:
                if (after.turn != before.turn + 1
                        or after.turns_remaining != before.turns_remaining - 1
                        or not is_hif_turn_boundary(after) or after.pending_after_card_ids):
                    raise SearchIncomplete('finite_turn_boundary_domain')
            else:
                self._clean(after)
                if (after.terminal or after.turn != before.turn
                        or after.turns_remaining != before.turns_remaining
                        or len(after.deck.hand) + len(after.drinks)
                        >= len(before.deck.hand) + len(before.drinks)):
                    raise SearchIncomplete('finite_turn_consumption_not_strict')
        return branches

    def transition(self, state, action, *, plan_type):
        self._clean(state)
        if action.kind is ActionKind.END_TURN:
            raise SearchIncomplete('finite_turn_unbounded_end')
        return self._prove(state, super().transition(state, action, plan_type=plan_type))

    def turn_boundary_transition(self, state, action, *, plan_type):
        self._clean(state)
        branches = super().turn_boundary_transition(state, action, plan_type=plan_type)
        if branches is None:
            raise SearchIncomplete('finite_turn_unbounded_end')
        return self._prove(state, branches, boundary=True)


class _BoundedTurnPlanner(HifActionChainPlanner):
    def __init__(self, kernel, deadline, *, forecast_cache=None, installed_card_use_value=False,
                 finite_consumables=False, end_turn_focus_projection=False):
        self._total_deadline = deadline
        if type(installed_card_use_value) is not bool:
            raise ValueError('installed_card_use_value_requires_bool')
        if type(finite_consumables) is not bool:
            raise ValueError('finite_consumables_requires_bool')
        if type(end_turn_focus_projection) is not bool:
            raise ValueError('end_turn_focus_projection_requires_bool')
        self._max_expanded_nodes = (FINITE_CONSUMABLES_MAX_EXPANDED_NODES
                                    if finite_consumables else MAX_EXPANDED_NODES)
        self._installed_card_use_value = installed_card_use_value
        self._end_turn_focus_projection = end_turn_focus_projection
        self.focus_projection_statuses, self.focus_projection_reasons = Counter(), Counter()
        self._complete_forecasts = forecast_cache if forecast_cache is not None else {}
        self.forecast_cache_hits = self.forecast_computations = 0
        super().__init__(kernel, PlannerConfig(max_decision_depth=MAX_DECISION_DEPTH,
            timeout_ms=max(1., (deadline - time.perf_counter()) * 1000)), terminal_cache=False)

    @property
    def _expanded_nodes(self):
        return self._node_count

    @_expanded_nodes.setter
    def _expanded_nodes(self, value):
        # Count the existing expansions, rejecting the first excess node
        # before work while accepting a complete result at the fixed limit.
        if value > self._max_expanded_nodes:
            raise SearchIncomplete('turn_chain_node_limit')
        self._node_count = value

    def _initialize_search(self, request, started):
        super()._initialize_search(request, started)
        self._deadline = self._total_deadline

    def _validation_session(self):
        # Only immutable validation/transforms share the enclosing window.
        # Search memos stay per decision; both parent and child budgets apply.
        return validation_session(self._check_budget, reuse_parent=True)

    def _check_budget(self):
        # A wall-clock failure is not a public domain decision or a fallback.
        if time.perf_counter() >= self._total_deadline:
            raise TimeoutError('turn_chain_timeout')

    def _merge_equivalent_actions(self, state, actions):
        # Compare all supplied physical roots, including equivalent copies.
        return tuple(actions)

    def _future_score(self, state):
        if not self._end_turn_focus_projection:
            return super()._future_score(state)
        projection = project_end_turn_focus(self.kernel, state)
        self.focus_projection_statuses[projection.status] += 1
        self.focus_projection_reasons[projection.reason] += 1
        return sense_future_score(self.kernel, state, concentration_timeline=projection.values)

    def _search_state(self, state, *, request, depth):
        self._check_budget()
        boundary = (not state.terminal and self._boundary_turn is not None
            and state.turn >= self._boundary_turn and state.pending_choice is None
            and (not state.resolution_stack or is_hif_turn_boundary(state))
            and not (state.actions_remaining == 0 and state.turns_remaining == 1))
        key = self._semantic_state_key(replace(state, score=0)) if boundary else None
        # Keep the current decision's original memo precedence. Cross-decision
        # reuse requires the entire boundary state, including score, so even
        # subtraction/rounding matches its first cold estimate exactly.
        if boundary and key not in self._forecast_memo:
            cache_key = (True, state) if self._installed_card_use_value else state
            if self._end_turn_focus_projection:
                cache_key = (END_TURN_FOCUS_VERSION, cache_key)
            cached = self._complete_forecasts.get(cache_key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                outcomes, increment = cached
                self._forecast_memo[key] = increment
                self.forecast_cache_hits += 1
                self._check_budget()
                return outcomes
            outcomes = super()._search_state(state, request=request, depth=depth)
            self._check_budget()
            if self._installed_card_use_value:
                bonus = installed_card_use_bonus(self.kernel, state).value
                self._check_budget()
                outcomes = tuple(replace(value, score=value.score + bonus) for value in outcomes)
                self._forecast_memo[key] += bonus
            self._complete_forecasts[cache_key] = (outcomes, self._forecast_memo[key])
            self.forecast_computations += 1
            return outcomes
        return super()._search_state(state, request=request, depth=depth)


class TurnChainContinuation:
    """Deterministic bounded advice from complete current-turn comparisons.

    Node-limit and draw-domain declines depend only on the public request and
    fixed kernel. They may be cached in this window; deadlines never are.
    sense_future_score supplies the existing boundary estimate, not terminal
    truth. No future scenarios, simulator seeds or input credentials are used.
    """

    def __init__(self, kernel, *, installed_card_use_value=False, finite_consumables=False,
                 end_turn_focus_projection=False):
        if type(installed_card_use_value) is not bool:
            raise ValueError('installed_card_use_value_requires_bool')
        if type(finite_consumables) is not bool:
            raise ValueError('finite_consumables_requires_bool')
        if type(end_turn_focus_projection) is not bool:
            raise ValueError('end_turn_focus_projection_requires_bool')
        self._installed_card_use_value = installed_card_use_value
        self._end_turn_focus_projection = end_turn_focus_projection
        self._finite_consumables = finite_consumables
        self._source_kernel = kernel
        self.start_window()

    @property
    def installed_card_use_value(self):
        return self._installed_card_use_value

    @property
    def end_turn_focus_projection(self):
        return self._end_turn_focus_projection

    @property
    def finite_consumables(self):
        return self._finite_consumables

    @property
    def max_expanded_nodes(self):
        return (FINITE_CONSUMABLES_MAX_EXPANDED_NODES
                if self.finite_consumables else MAX_EXPANDED_NODES)

    def start_window(self):
        # Definitions are frozen with this window, not with the lifetime of
        # a reusable search whose next observation may register new support.
        kernel = self._source_kernel
        kernel_type = _FiniteConsumablesKernel if self.finite_consumables else _NoDrawKernel
        self.kernel = kernel_type(kernel.cards, kernel.drinks, kernel.items,
                                  memory_abilities=kernel.memory_abilities)
        if self.kernel.ruleset_id != kernel.ruleset_id:
            raise ValueError('turn_chain_ruleset_mismatch')
        self._cache = {}
        self._forecast_cache = {}
        self._forecast_cache_hits = self._forecast_computations = 0
        self._focus_projection_statuses, self._focus_projection_reasons = Counter(), Counter()
        self._calls = self._cache_hits = self._accepted = self._declined = 0
        self._expanded_nodes = 0
        self._decline_reasons = Counter()

    @staticmethod
    def _check_deadline(deadline):
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('turn_chain_invalid_deadline')
        if time.perf_counter() >= deadline:
            raise TimeoutError('turn_chain_timeout')

    def choose(self, request, actions, deadline):
        # Import lazily because MCTS owns this helper and defines its key.
        from .hif_mcts import action_key

        self._check_deadline(deadline)
        request.validate()
        actions = tuple(actions)
        if (not actions or len(set(actions)) != len(actions)
                or len({action_key(action) for action in actions}) != len(actions)):
            raise ValueError('turn_chain_actions_not_unique')
        if request.ruleset_id != self.kernel.ruleset_id or request.plan_type is not PlanType.SENSE:
            raise ValueError('turn_chain_ruleset_mismatch')
        if not request.capabilities.all_ready:
            raise ValueError('turn_chain_capability_not_ready')
        with validation_session(lambda: self._check_deadline(deadline), reuse_parent=True):
            # First bind every real root, including out-of-domain roots. The
            # finite planner then declines the whole comparison, never filters it.
            legal = (HifSenseKernel.legal_actions(self.kernel, request.state,
                        plan_type=request.plan_type, decision_kind=request.decision_kind)
                     if self.finite_consumables else self.kernel.legal_actions(
                        request.state, plan_type=request.plan_type,
                        decision_kind=request.decision_kind))
            bound = tuple(_BoundedTurnPlanner._attach_source_slot(action, request) for action in legal)
        if set(bound) != set(actions):
            raise ValueError('turn_chain_incomplete_root_actions')
        self._calls += 1
        cache_state = request.state if self.finite_consumables else replace(request.state, score=0)
        key = (cache_state, request.plan_type, request.decision_kind,
               request.utility, tuple(request.card_slots), tuple(request.drink_slots), frozenset(actions))
        if self._end_turn_focus_projection:
            key = (END_TURN_FOCUS_VERSION, key)
        selected = self._cache.get(key, _CACHE_MISS)
        if selected is not _CACHE_MISS:
            self._check_deadline(deadline)
            self._cache_hits += 1
            self._accepted += int(selected is not None)
            self._declined += int(selected is None)
            return selected
        if request.state.turns_remaining == 1:
            self._check_deadline(deadline)
            self._cache[key] = None
            self._declined += 1
            self._decline_reasons['terminal_turn'] += 1
            return None
        if request.state.terminal or request.utility.baseline_p10 is not None:
            raise ValueError('turn_chain_requires_future_estimate_boundary')

        planner = _BoundedTurnPlanner(self.kernel, deadline, forecast_cache=self._forecast_cache,
                                     installed_card_use_value=self._installed_card_use_value,
                                     finite_consumables=self.finite_consumables,
                                     end_turn_focus_projection=self._end_turn_focus_projection)
        # Keep the finite mode's complete public-state cache identity while
        # retaining the established translated arithmetic inside its solve.
        normalized = replace(request, state=replace(request.state, score=0))
        response = planner.decide(normalized)
        self._check_deadline(deadline)
        self._expanded_nodes += planner._expanded_nodes
        self._forecast_cache_hits += planner.forecast_cache_hits
        self._forecast_computations += planner.forecast_computations
        self._focus_projection_statuses.update(planner.focus_projection_statuses)
        self._focus_projection_reasons.update(planner.focus_projection_reasons)
        allowed_declines = {'search_incomplete:turn_chain_draw_domain',
                            'search_incomplete:turn_chain_node_limit'}
        if self.finite_consumables:
            allowed_declines.update('search_incomplete:' + reason
                                    for reason in _FINITE_CONSUMABLES_DECLINES)
        if not response.has_action_intent:
            if response.reason == 'search_incomplete:planner_timeout':
                raise TimeoutError('turn_chain_timeout')
            if response.reason not in allowed_declines:
                raise ValueError('turn_chain_incomplete:' + response.reason)
            self._cache[key] = None
            self._declined += 1
            self._decline_reasons[response.reason.removeprefix('search_incomplete:')] += 1
            return None
        trace = response.trace
        if (trace is None or trace.value_scope != 'TURN_CHAIN_ESTIMATE'
                or trace.expanded_nodes > self.max_expanded_nodes
                or len(trace.candidate_values) != len(actions)
                or {action_key(planner._attach_source_slot(row.action, normalized))
                    for row in trace.candidate_values}
                != {action_key(action) for action in actions}
                or any(not math.isclose(row.probability_mass, 1., rel_tol=1e-9, abs_tol=1e-9)
                       for row in trace.candidate_values)
                or response.action not in actions):
            raise ValueError('turn_chain_incomplete_root_comparison')
        selected = action_key(response.action)
        self._check_deadline(deadline)
        self._cache[key] = selected
        self._accepted += 1
        return selected

    def summary(self):
        result = dict(calls=self._calls, cache_hits=self._cache_hits, accepted=self._accepted,
            declined=self._declined, expanded_nodes=self._expanded_nodes,
            decline_reasons=dict(self._decline_reasons), future_samples=0,
            value_scope='TURN_CHAIN_ESTIMATE', max_expanded_nodes=self.max_expanded_nodes,
            max_decision_depth=MAX_DECISION_DEPTH,
            transition_computations=self.kernel.reuse_counts['transition_computations'],
            transition_cache_hits=self.kernel.reuse_counts['transition_cache_hits'],
            boundary_computations=self.kernel.reuse_counts['boundary_computations'],
            boundary_cache_hits=self.kernel.reuse_counts['boundary_cache_hits'],
            concentration_percentage_computations=self.kernel.reuse_counts['concentration_percentage_computations'],
            concentration_percentage_cache_hits=self.kernel.reuse_counts['concentration_percentage_cache_hits'],
            effective_concentration_computations=self.kernel.reuse_counts['effective_concentration_computations'],
            effective_concentration_cache_hits=self.kernel.reuse_counts['effective_concentration_cache_hits'],
            forecast_computations=self._forecast_computations,
            forecast_cache_hits=self._forecast_cache_hits)
        if self._installed_card_use_value:
            result.update(installed_card_use_value=True, installed_card_use_opportunities=1)
        if self._end_turn_focus_projection:
            result.update(end_turn_focus_projection=True,
                end_turn_focus_projection_version=END_TURN_FOCUS_VERSION,
                end_turn_focus_projection_statuses=dict(self._focus_projection_statuses),
                end_turn_focus_projection_reasons=dict(self._focus_projection_reasons))
        if self.finite_consumables:
            result.update(finite_consumables=True,
                          finite_consumables_version=FINITE_CONSUMABLES_VERSION,
                          cache_entries=len(self._cache))
        return result
