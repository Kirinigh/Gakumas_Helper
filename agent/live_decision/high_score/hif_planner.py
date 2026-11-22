"""HIF search over joint actions, with translation-invariant score caching."""
import math
import time
from contextlib import contextmanager

from .hif import is_hif_turn_boundary
from .planner import SearchIncomplete, RobustHighScorePlanner, _Outcome, _normalize
from .contracts import ActionKind, GameAction, RiskProfile
from .state_copy import replace
from .hif_forecast import sense_future_score
from .validation_cache import validation_session
from .hif_search_session import HifSearchSession
from .hif_terminal_cache import TerminalMarginalCache, terminal_schedules, fixed_terminal_turn


class HifActionChainPlanner(RobustHighScorePlanner):
    """Preserve the exact transition tree while sharing equal score increments.

    The HIF rules never use accumulated score in conditions, costs or effects;
    only additive gains read it. Card/trigger state, resources and all timers
    stay in the normal semantic key. This is not an approximate score cutoff.
    By default earlier turns end at the established future-value boundary.
    complete_horizon=True requires actual terminal leaves across all turns.
    """

    planner_name = "hif-joint-expectimax"
    planner_version = "1.0"
    policy_name = planner_name
    policy_version = planner_version

    def __init__(self, kernel, config=None, *, advice=None, use_learned_value=False,
                 terminal_cache=True, complete_horizon=False):
        super().__init__(kernel, config)
        if complete_horizon and not self.config.require_terminal_leaf:
            raise ValueError('complete_horizon requires terminal leaves')
        self.advice = advice
        self.use_learned_value = use_learned_value
        self.terminal_cache = terminal_cache
        self.complete_horizon = complete_horizon
        self._search_session = None
        self.parallel_terminal_roots = False
        self.adaptive_terminal_roots = False

    @contextmanager
    def search_session(self, *, deadline, reuse=None):
        """Bind a bounded window; explicitly supplied complete tables survive it."""
        if not math.isfinite(deadline):
            raise ValueError('HIF search session requires a finite deadline')
        if self._search_session is not None:
            raise ValueError('HIF search sessions cannot be nested')
        if reuse is not None and (
                type(reuse) is not HifSearchSession or type(self) is not HifActionChainPlanner
                or not self.complete_horizon or not self.config.require_terminal_leaf
                or self.advice is not None or self.use_learned_value):
            raise ValueError('HIF reusable session requires an exact complete-horizon planner')
        session = HifSearchSession(deadline) if reuse is None else reuse
        session._acquire(deadline)
        self._search_session = session
        try:
            yield session
        finally:
            self._search_session = None
            self._memo = {}
            self._terminal_marginals = TerminalMarginalCache(self.kernel)
            if reuse is None:
                session.clear()
            session._release()

    def _initialize_search(self, request, started):
        super()._initialize_search(request, started)
        session = self._search_session
        if session is None:
            return
        self._deadline = min(self._deadline, session.deadline)
        if ((request.state.turns_remaining == 1 or self.complete_horizon)
                and self._boundary_turn is None and self.config.require_terminal_leaf):
            self._memo, self._terminal_marginals = session.tables(self, request)
            self._terminal_hit_offsets = (self._terminal_marginals.hits,
                                          self._terminal_marginals.transition_hits)

    def _merge_equivalent_actions(self, state, actions):
        merged = super()._merge_equivalent_actions(state, actions)
        if self.advice is None or len(merged) < 2:
            return merged
        self._check_budget()
        try:
            priors = tuple(self.advice.action_priors(state, merged))
        except ValueError as error:
            raise SearchIncomplete(f'learning_advice_invalid:{error}') from error
        self._check_budget()
        if len(priors) != len(merged) or any(not math.isfinite(value) for value in priors):
            raise SearchIncomplete('learning_advice_invalid_priors')
        # Ordering never removes a candidate and provides no pruning bound.
        return tuple(action for _, action in sorted(zip(priors, merged), key=lambda row: -row[0]))

    def _validation_session(self):
        """Keep ordinary decisions independent of any enclosing search."""
        return validation_session(self._check_budget)

    def decide(self, request):
        started = time.perf_counter()
        if self.complete_horizon and not self.config.require_terminal_leaf:
            return self._stop('invalid_config:complete_horizon_requires_terminal_leaf', started)
        self._boundary_turn = (request.state.turn + 1 if request.state.turns_remaining > 1
                               and request.utility.baseline_p10 is None
                               and not self.complete_horizon else None)
        self._reached_depth = 0
        self._forecast_memo = {}
        self._terminal_marginals = TerminalMarginalCache(self.kernel)
        self._terminal_hit_offsets = (0, 0)
        self._fixed_terminal_turn = None
        self._memo_hits = 0
        self._parallel_root_values = None
        with self._validation_session():
            response = super().decide(request)
        if self.complete_horizon and response.has_action_intent and time.perf_counter() >= self._deadline:
            response = self._stop('search_incomplete:planner_timeout', started,
                                  expanded_nodes=self._expanded_nodes)
        if response.trace is not None:
            candidates = tuple(replace(candidate, ranking_utility=(
                candidate.value.p10_final_score if request.utility.risk_profile is RiskProfile.SAFE_BASELINE
                else candidate.value.expected_final_score)) for candidate in response.trace.candidate_values)
            response = replace(response, trace=replace(response.trace, completed_depth=self._reached_depth,
                candidate_values=candidates,
                reason_codes=response.trace.reason_codes + (
                    f'terminal_marginal_hits:{self._terminal_marginals.hits - self._terminal_hit_offsets[0]}',
                    f'scalar_transition_hits:{self._terminal_marginals.transition_hits - self._terminal_hit_offsets[1]}')
                + ((f'learning_order:{self.advice.model_version}',)
                    if self.advice is not None else ()),
                value_scope="TURN_CHAIN_ESTIMATE" if self._boundary_turn is not None else "FINAL_SCORE"))
        if response.has_action_intent and self._boundary_turn is not None:
            response = replace(response, reason='hif_complete_turn_chain_with_future_estimate')
        if self._search_session is not None:
            self._search_session.memo_hits += self._memo_hits
            if response.has_action_intent and time.perf_counter() >= self._search_session.deadline:
                response = self._stop('search_incomplete:planner_timeout',
                    started, expanded_nodes=self._expanded_nodes)
            if response.trace is not None:
                response = replace(response, trace=replace(response.trace,
                    reason_codes=response.trace.reason_codes + (f'session_memo_hits:{self._memo_hits}',)))
        sink = getattr(self, '_completed_action_sink', None)
        if (callable(sink) and response.has_action_intent and response.trace is not None
                and response.trace.value_scope == 'FINAL_SCORE'):
            sink(request.state, response.action, request)
        return response

    def _evaluate_action(self, state, action, *, request, depth):
        self._reached_depth = max(self._reached_depth, self.config.max_decision_depth - depth + 1)
        self._check_budget()
        if (self.parallel_terminal_roots and state is request.state
                and depth == self.config.max_decision_depth and state.turns_remaining == 1
                and '22' in state.drinks and len(state.deck.hand) >= 4
                and not state.deck.draw_pile):
            if self._parallel_root_values is None:
                from .hif_terminal_parallel import run_roots, cached_roots, adaptive_roots

                actions = self._merge_equivalent_actions(state, self.kernel.legal_actions(
                    state, plan_type=request.plan_type, decision_kind=request.decision_kind))
                known, missing = cached_roots(self, request, actions)
                if missing:
                    solve = adaptive_roots if self.adaptive_terminal_roots else run_roots
                    known.update(solve(self, request, missing, self._deadline))
                self._parallel_root_values = known
            self._check_budget()
            outcomes, reached, nodes = self._parallel_root_values[action]
            self._reached_depth = max(self._reached_depth, reached)
            self._expanded_nodes += nodes
            return outcomes
        terminal = self.kernel.terminal_score_resources(state, action, plan_type=request.plan_type)
        if terminal is not None:
            self._expanded_nodes += 1
            score, stamina, genki = terminal
            return (_Outcome(score, 1.0, stamina, genki),)
        if action.kind is ActionKind.END_TURN and self._boundary_turn == state.turn + 1:
            self._check_budget()
            projected = self.kernel.turn_boundary_transition(state, action, plan_type=request.plan_type)
            if projected is not None:
                outcomes = []
                for branch in projected:
                    child = self._search_state(branch.state, request=request, depth=depth - 1)
                    outcomes.extend(replace(value, probability=value.probability * branch.probability) for value in child)
                return _normalize(tuple(outcomes))
        key = self._terminal_marginals.key(state, action) if self.terminal_cache else None
        if key is not None and key in self._terminal_marginals.values:
            self._terminal_marginals.hits += 1
            return tuple(replace(value, score=value.score + state.score)
                         for value in self._terminal_marginals.values[key])
        result = super()._evaluate_action(state, action, request=request, depth=depth)
        if key is not None:
            self._terminal_marginals.values[key] = tuple(replace(value, score=value.score - state.score)
                                                        for value in result)
        return result

    def _transition(self, state, action, *, plan_type):
        key = self._terminal_marginals.key(state, action, transition=True) if self.terminal_cache else None
        if key is not None and key in self._terminal_marginals.transitions:
            return self._terminal_marginals.restore_transition(key, state, action)
        result = self.kernel.transition(state, action, plan_type=plan_type)
        # Search uses the kernel's canonical belief representation on both
        # cache hits and misses. Physical observation order never comes from
        # this transition; root actions retain the request's verified slots.
        result = tuple(replace(branch, state=self.kernel._canonical_probability_state(branch.state))
                       for branch in result)
        if key is not None:
            self._terminal_marginals.remember_transition(key, state, result)
        return result

    def _semantic_state_key(self, state):
        # Retain exhausted entries in the observation ledger, but their firing
        # history cannot affect future transitions of an initialized state.
        active = tuple(ref for ref in state.scheduled_effects
                       if ref.remaining_uses != 0 and ref.remaining_turns != 0)
        # A proof made at one terminal descendant does not justify removing
        # schedules from another descendant of a multi-turn complete search.
        # Keep full active schedule state in that opt-in domain.
        if (state.turns_remaining == 1 and self.terminal_cache and not self.complete_horizon
                and not state.resolution_stack and not state.pending_choice):
            if getattr(self, '_fixed_terminal_turn', None) is None:
                self._fixed_terminal_turn = fixed_terminal_turn(self.kernel, state)
            if self._fixed_terminal_turn:
                active = terminal_schedules(state)
        normalized = replace(state, scheduled_effects=active, drinks=tuple(sorted(state.drinks)))
        return RobustHighScorePlanner._semantic_state_key(normalized)

    def _search_state(self, state, *, request, depth):
        self._check_budget()
        if state.terminal:
            return (_Outcome(float(state.score), 1.0, state.stamina, state.genki),)
        if state.actions_remaining == 0 and state.turns_remaining == 1:
            # No player action is available between the last card and the
            # automatic settlement. Avoid building a decision node for it.
            terminal = self.kernel.terminal_score_resources(
                state, GameAction(ActionKind.END_TURN), plan_type=request.plan_type)
            if terminal is not None:
                score, stamina, genki = terminal
                return (_Outcome(score, 1.0, stamina, genki),)
        if (self._boundary_turn is not None and state.turn >= self._boundary_turn
                and state.pending_choice is None
                and (not state.resolution_stack or is_hif_turn_boundary(state))):
            key = self._semantic_state_key(replace(state, score=0))
            if key in self._forecast_memo:
                return (_Outcome(state.score + self._forecast_memo[key], 1.0, state.stamina, state.genki),)
            value = None
            if self.advice is not None and self.use_learned_value:
                learned = self.advice.future_score(state)
                if learned is not None:
                    if not math.isfinite(learned):
                        raise SearchIncomplete('learning_advice_invalid_value')
                    value = learned
            if value is None:
                value = self._future_score(state)
            self._check_budget()
            self._forecast_memo[key] = value - state.score
            return (_Outcome(value, 1.0, state.stamina, state.genki),)
        offset = state.score
        normalized = replace(state, score=0) if offset else state
        outcomes = super()._search_state(normalized, request=request, depth=depth)
        return tuple(replace(outcome, score=outcome.score + offset) for outcome in outcomes) if offset else outcomes

    def _future_score(self, state):
        return sense_future_score(self.kernel, state)
