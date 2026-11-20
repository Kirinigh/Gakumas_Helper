from __future__ import annotations

import math
import time
from dataclasses import replace, dataclass

from .kernel import KernelError, PythonNiaProKernel
from .contracts import (
    ActionKind,
    GameAction,
    RiskProfile,
    UtilitySpec,
    DecisionKind,
    ValueSummary,
    DecisionTrace,
    DeckKnowledge,
    StrategyState,
    CandidateValue,
    DecisionOutcome,
    DecisionRequest,
    DecisionResponse,
)
from .validation_cache import immutable_projection


class SearchIncomplete(RuntimeError):
    """Raised when every root action cannot be compared at the same complete horizon."""


@dataclass(frozen=True)
class PlannerConfig:
    max_decision_depth: int = 12
    timeout_ms: float = 500.0
    require_terminal_leaf: bool = True

    def validate(self) -> None:
        if self.max_decision_depth < 1 or self.timeout_ms <= 0:
            raise ValueError("planner depth and timeout must be positive")


@dataclass(frozen=True)
class _Outcome:
    score: float
    probability: float
    stamina: int
    genki: int


def _normalize(outcomes: tuple[_Outcome, ...]) -> tuple[_Outcome, ...]:
    merged: dict[tuple[float, int, int], float] = {}
    for outcome in outcomes:
        key = (outcome.score, outcome.stamina, outcome.genki)
        merged[key] = merged.get(key, 0.0) + outcome.probability
    compact = tuple(_Outcome(score, probability, stamina, genki) for (score, stamina, genki), probability in merged.items())
    total = sum(outcome.probability for outcome in compact)
    if total <= 0:
        raise SearchIncomplete("outcome_probability_mass_missing")
    if math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return compact
    return tuple(replace(outcome, probability=outcome.probability / total) for outcome in compact)


def _weighted_quantile(outcomes: tuple[_Outcome, ...], quantile: float) -> float:
    if not outcomes:
        raise SearchIncomplete("outcome_distribution_empty")
    ordered = sorted(outcomes, key=lambda item: item.score)
    cumulative = 0.0
    for outcome in ordered:
        cumulative += outcome.probability
        if cumulative + 1e-12 >= quantile:
            return outcome.score
    return ordered[-1].score


def _summary(outcomes: tuple[_Outcome, ...]) -> ValueSummary:
    normalized = _normalize(outcomes)
    scores = [outcome.score for outcome in normalized]
    # A score distribution must not change its mean when equal-score leaves
    # are split into different stamina/genki groups. Sum probabilities by
    # score first, then accumulate increments relative to an integer anchor.
    # This also gives offset-normalized HIF DP the same tie ranking as the
    # full reference tree, without rounding scores or adding a tie tolerance.
    grouped = {}
    for outcome in normalized:
        grouped.setdefault(outcome.score, []).append(outcome.probability)
    anchor = min(scores)
    expected = anchor + math.fsum((score - anchor) * math.fsum(probabilities)
                                  for score, probabilities in sorted(grouped.items()))
    return ValueSummary(
        expected_final_score=expected,
        p10_final_score=_weighted_quantile(normalized, 0.10),
        worst_final_score=min(scores),
        best_final_score=max(scores),
        outcome_count=len(normalized),
    )


def _action_key(action: GameAction) -> tuple[str, str, str, str]:
    return (
        action.kind.value,
        action.card_instance_id or "",
        action.drink_id or "",
        action.target_id or "",
    )


_SLOTTED_CARD_ACTIONS = {
    ActionKind.PLAY_CARD,
    ActionKind.HOLD_CARD,
    ActionKind.MOVE_CARD,
    ActionKind.FREE_PLAY,
    ActionKind.PAID_PLAY,
}


@immutable_projection
def _semantic_deck_key(deck):
    draw = tuple(card.semantic_key for card in deck.draw_pile)
    if deck.knowledge is not DeckKnowledge.KNOWN_ORDER:
        prefix = deck.known_top_count
        draw = draw[:prefix] + tuple(sorted(draw[prefix:]))
    unordered = lambda cards: tuple(sorted(card.semantic_key for card in cards))
    return (deck.knowledge, deck.known_top_count, unordered(deck.hand), draw,
            unordered(deck.discard), unordered(deck.removed), unordered(deck.held))


class RobustHighScorePlanner:
    planner_name = "bounded-expectimax"
    planner_version = "1.0"
    policy_name = planner_name
    policy_version = planner_version

    def __init__(self, kernel: PythonNiaProKernel, config: PlannerConfig | None = None):
        self.kernel = kernel
        self.config = config or PlannerConfig()
        self.config.validate()
        self._deadline = 0.0
        self._expanded_nodes = 0
        self._memo_hits = 0
        self._memo: dict[tuple[tuple[object, ...], int | None], tuple[_Outcome, ...]] = {}

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.perf_counter()
        try:
            request.validate()
        except ValueError as error:
            return self._stop(f"invalid_request:{error}", started)
        capability_stop = self._capability_response(request, started)
        if capability_stop is not None:
            return capability_stop
        if request.ruleset_id != self.kernel.ruleset_id:
            return self._stop("ruleset_mismatch", started)

        self._expanded_nodes = 0
        self._memo_hits = 0
        self._initialize_search(request, started)
        try:
            actions = self.kernel.legal_actions(
                request.state,
                plan_type=request.plan_type,
                decision_kind=request.decision_kind,
            )
            actions = self._merge_equivalent_actions(request.state, actions)
            if not actions:
                return self._stop("no_supported_legal_action", started)
            evaluated: list[tuple[GameAction, tuple[_Outcome, ...], tuple[str, ...]]] = []
            for action in actions:
                self._check_budget()
                outcomes = self._evaluate_action(
                    request.state,
                    action,
                    request=request,
                    depth=self.config.max_decision_depth,
                )
                value = _summary(outcomes)
                reasons: list[str] = []
                if any(outcome.stamina < request.utility.minimum_stamina for outcome in outcomes):
                    reasons.append("minimum_stamina_not_met")
                if any(outcome.genki < request.utility.minimum_genki for outcome in outcomes):
                    reasons.append("minimum_genki_not_met")
                if request.utility.baseline_p10 is not None and value.p10_final_score < request.utility.baseline_p10:
                    reasons.append("p10_below_approved_baseline")
                evaluated.append((action, outcomes, tuple(reasons)))
        except (KernelError, SearchIncomplete) as error:
            return self._stop(f"search_incomplete:{error}", started, expanded_nodes=self._expanded_nodes)

        eligible = [item for item in evaluated if not item[2]]
        if not eligible:
            trace = self._trace(started, evaluated, reason_codes=("all_candidates_rejected",))
            response = DecisionResponse(
                outcome=DecisionOutcome.STOP_TASK,
                reason="all_candidates_rejected",
                action=None,
                value=None,
                trace=trace,
            )
            response.validate()
            return response
        selected = self._select(eligible, request.utility)
        action, outcomes, _reasons = selected
        try:
            action = self._attach_source_slot(action, request)
        except SearchIncomplete as error:
            return self._stop(f"search_incomplete:{error}", started, expanded_nodes=self._expanded_nodes)
        value = _summary(outcomes)
        trace = self._trace(started, evaluated)
        response = DecisionResponse(
            outcome=DecisionOutcome.DECIDE,
            reason=f"{request.utility.risk_profile.value.lower()}_complete_horizon",
            action=action,
            value=value,
            trace=trace,
        )
        response.validate()
        return response

    def _initialize_search(self, request: DecisionRequest, started: float) -> None:
        """Each decision owns its memo unless a specialized planner opts in."""
        self._deadline = started + self.config.timeout_ms / 1000.0
        self._memo = {}

    def _capability_response(self, request: DecisionRequest, started: float) -> DecisionResponse | None:
        capabilities = request.capabilities
        if capabilities.all_ready:
            return None
        reasons = capabilities.reason_codes or ("capability_not_ready",)
        transient = not capabilities.observation_valid and all(reason.startswith("transient:") for reason in reasons)
        if transient and request.reobserve_attempt < 2:
            response = DecisionResponse(
                outcome=DecisionOutcome.REOBSERVE,
                reason="transient_observation_requires_reobserve",
                action=None,
                value=None,
                trace=self._empty_trace(started, reasons),
            )
            response.validate()
            return response
        return self._stop(f"capability_blocked:{','.join(reasons)}", started)

    def _evaluate_action(
        self,
        state: StrategyState,
        action: GameAction,
        *,
        request: DecisionRequest,
        depth: int,
    ) -> tuple[_Outcome, ...]:
        self._check_budget()
        self._expanded_nodes += 1
        branches = self._transition(state, action, plan_type=request.plan_type)
        outcomes: list[_Outcome] = []
        for branch in branches:
            child = self._search_state(branch.state, request=request, depth=depth - 1)
            outcomes.extend(replace(outcome, probability=outcome.probability * branch.probability) for outcome in child)
        return _normalize(tuple(outcomes))

    def _transition(self, state, action, *, plan_type):
        return self.kernel.transition(state, action, plan_type=plan_type)

    def _search_state(
        self,
        state: StrategyState,
        *,
        request: DecisionRequest,
        depth: int,
    ) -> tuple[_Outcome, ...]:
        self._check_budget()
        # A cached terminal solution is already complete regardless of how many
        # actions led to this state. Only approximate depth leaves depend on
        # the remaining horizon. Incomplete searches never populate this cache.
        cache_key = (self._semantic_state_key(state), None if self.config.require_terminal_leaf else depth)
        if cache_key in self._memo:
            self._memo_hits += 1
            return self._memo[cache_key]
        self._expanded_nodes += 1
        if state.terminal:
            outcomes = (_Outcome(float(state.score), 1.0, state.stamina, state.genki),)
            self._memo[cache_key] = outcomes
            return outcomes
        if depth <= 0:
            if self.config.require_terminal_leaf:
                raise SearchIncomplete("search_horizon_not_terminal")
            return (_Outcome(float(state.score), 1.0, state.stamina, state.genki),)
        decision_kind = (DecisionKind.SECONDARY if state.pending_choice is not None
                         else DecisionKind.LIVE if request.decision_kind is DecisionKind.LIVE
                         or request.drink_slots else DecisionKind.CARD)
        actions = self.kernel.legal_actions(state, plan_type=request.plan_type, decision_kind=decision_kind)
        actions = self._merge_equivalent_actions(state, actions)
        if not actions:
            raise SearchIncomplete("future_state_has_no_legal_action")
        candidates: list[tuple[GameAction, tuple[_Outcome, ...], tuple[str, ...]]] = []
        for action in actions:
            outcomes = self._evaluate_action(state, action, request=request, depth=depth)
            candidates.append((action, outcomes, ()))
        _action, selected, _reasons = self._select(candidates, request.utility)
        self._memo[cache_key] = selected
        sink = getattr(self, '_completed_action_sink', None)
        if callable(sink) and self.config.require_terminal_leaf:
            sink(state, _action, request)
        return selected

    @staticmethod
    def _select(
        candidates: list[tuple[GameAction, tuple[_Outcome, ...], tuple[str, ...]]],
        utility: UtilitySpec,
    ) -> tuple[GameAction, tuple[_Outcome, ...], tuple[str, ...]]:
        def key(candidate: tuple[GameAction, tuple[_Outcome, ...], tuple[str, ...]]) -> tuple[float, float, float, str]:
            action, outcomes, _reasons = candidate
            value = _summary(outcomes)
            if utility.risk_profile is RiskProfile.SAFE_BASELINE:
                metrics = (value.p10_final_score, value.worst_final_score, value.expected_final_score)
            elif utility.risk_profile is RiskProfile.AGGRESSIVE_HIGH_SCORE:
                metrics = (value.expected_final_score, value.best_final_score, value.p10_final_score)
            else:
                metrics = (value.expected_final_score, value.p10_final_score, value.worst_final_score)
            # min() on the negated metrics keeps deterministic action ordering for exact ties.
            return (-metrics[0], -metrics[1], -metrics[2], "|".join(_action_key(action)))

        return min(candidates, key=key)

    def _trace(
        self,
        started: float,
        evaluated: list[tuple[GameAction, tuple[_Outcome, ...], tuple[str, ...]]],
        *,
        reason_codes: tuple[str, ...] = (),
    ) -> DecisionTrace:
        values = tuple(
            CandidateValue(
                action=action,
                value=_summary(outcomes),
                probability_mass=sum(outcome.probability for outcome in outcomes),
                rejection_reasons=reasons,
            )
            for action, outcomes, reasons in evaluated
        )
        return DecisionTrace(
            planner=self.planner_name,
            planner_version=self.planner_version,
            ruleset_id=self.kernel.ruleset_id,
            completed_depth=self.config.max_decision_depth,
            expanded_nodes=self._expanded_nodes,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            candidate_values=values,
            reason_codes=reason_codes,
        )

    def _empty_trace(self, started: float, reasons: tuple[str, ...]) -> DecisionTrace:
        return DecisionTrace(
            planner=self.planner_name,
            planner_version=self.planner_version,
            ruleset_id=self.kernel.ruleset_id,
            completed_depth=0,
            expanded_nodes=0,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            reason_codes=reasons,
        )

    def _stop(
        self,
        reason: str,
        started: float,
        *,
        expanded_nodes: int = 0,
    ) -> DecisionResponse:
        trace = DecisionTrace(
            planner=self.planner_name,
            planner_version=self.planner_version,
            ruleset_id=self.kernel.ruleset_id,
            completed_depth=0,
            expanded_nodes=expanded_nodes,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            reason_codes=(reason,),
        )
        response = DecisionResponse(
            outcome=DecisionOutcome.STOP_TASK,
            reason=reason,
            action=None,
            value=None,
            trace=trace,
        )
        response.validate()
        return response

    def _check_budget(self) -> None:
        if time.perf_counter() > self._deadline:
            raise SearchIncomplete("planner_timeout")

    @staticmethod
    def _attach_source_slot(action: GameAction, request: DecisionRequest) -> GameAction:
        if action.kind is ActionKind.USE_DRINK and request.decision_kind is DecisionKind.LIVE:
            slots = [slot for slot, drink_id in request.drink_slots if drink_id == action.drink_id]
            if not slots:
                raise SearchIncomplete("selected_drink_has_no_verified_source_slot")
            return replace(action, source_slot=min(slots))
        if action.kind not in _SLOTTED_CARD_ACTIONS or action.card_instance_id is None:
            return action
        slots = dict(request.card_slots)
        if action.card_instance_id not in slots:
            raise SearchIncomplete("selected_card_has_no_unique_source_slot")
        return replace(action, source_slot=slots[action.card_instance_id])

    @staticmethod
    def _merge_equivalent_actions(state: StrategyState, actions: tuple[GameAction, ...]) -> tuple[GameAction, ...]:
        cards = {card.instance_id: card for card in state.deck.all_cards}
        zones = {
            card.instance_id: zone
            for zone, zone_cards in (
                ("hand", state.deck.hand),
                ("draw_pile", state.deck.draw_pile),
                ("discard", state.deck.discard),
                ("removed", state.deck.removed),
                ("held", state.deck.held),
            )
            for card in zone_cards
        }
        representatives: dict[tuple[object, ...], GameAction] = {}
        for action in actions:
            if action.kind in _SLOTTED_CARD_ACTIONS and action.card_instance_id in cards:
                card = cards[str(action.card_instance_id)]
                key = (action.kind.value, *card.semantic_key, zones[str(action.card_instance_id)])
                if state.hand_support:
                    key += (next(((fact.enhancement_count, fact.projection_kind) for fact in state.hand_support
                        if fact.instance_id == card.instance_id and fact.card_id == card.card_id
                        and fact.turn == state.turn and zones[card.instance_id] == "hand"), 0),)
            else:
                key = (action.kind.value, action.drink_id or action.target_id or "", False, "")
            representatives.setdefault(key, action)
        return tuple(representatives.values())

    @staticmethod
    def _semantic_state_key(state: StrategyState) -> tuple[object, ...]:
        if state.hand_support:
            # The usual zone multiset drops physical identity. Known support
            # belongs to one entity, including that entity's modifications.
            return ("verified_hand_support", state)
        pending = state.pending_choice
        zone_by_id: dict[str, tuple[object, ...]] = {
            card.instance_id: (zone, *card.semantic_key)
            for zone, cards in (
                ("hand", state.deck.hand),
                ("draw", state.deck.draw_pile),
                ("discard", state.deck.discard),
                ("removed", state.deck.removed),
                ("held", state.deck.held),
            )
            for card in cards
        } if pending is not None or state.pending_after_card_ids else {}
        pending_key: tuple[object, ...] | None = None
        if pending is not None:
            pending_key = (
                pending.action_kind,
                zone_by_id.get(pending.source_id, ("external", pending.source_id, False)),
                tuple(sorted(zone_by_id[item] for item in pending.candidate_instance_ids)),
                pending.continuation_index,
                pending.remaining_effect_passes,
                pending.condition_stance,
                pending.condition_full_power,
                pending.selections_remaining,
                pending.selection_optional,
            )
        return (
            state.turn,
            state.turns_remaining,
            state.score,
            state.stamina,
            state.max_stamina,
            state.genki,
            state.actions_remaining,
            state.hand_limit,
            state.score_multiplier,
            pending_key,
            tuple(
                zone_by_id.get(instance_id, ("external", instance_id, False))
                for instance_id in state.pending_after_card_ids
            ),
            state.resolution_stack,
            state.stance,
            state.previous_stance,
            state.full_power,
            state.cumulative_full_power,
            state.passion,
            state.passion_buff,
            state.passion_gain_bonus_pct,
            state.parameter_gain_bonus_pct,
            state.concentration,
            state.good_condition_turns,
            state.good_condition_fresh,
            state.excellent_condition_turns,
            state.excellent_condition_fresh,
            state.good_impression_turns,
            state.motivation,
            state.statuses,
            state.scheduled_effects,
            state.persistent_effects_initialized,
            state.double_card_effects_remaining,
            state.nullify_genki_turns,
            state.nullify_genki_fresh,
            state.half_cost_turns,
            state.half_cost_fresh,
            state.double_cost_turns,
            state.double_cost_fresh,
            state.stance_lock_turns,
            state.stance_lock_fresh,
            state.nullify_cost_cards,
            state.cost_reduction,
            state.strength_times,
            state.preservation_times,
            state.full_power_times,
            state.leisure_times,
            state.stance_changed_by_direct_effect_times,
            state.created_card_count,
            state.enthusiasm_bonus_buffs,
            state.enthusiasm_multiplier_buffs,
            state.full_power_charge_buffs,
            state.score_buffs,
            state.concentration_effect_buffs,
            state.concentration_addition_buffs,
            state.drinks,
            state.items,
            state.memory_abilities,
            state.field_effects,
            state.future_score_multipliers,
            state.turn_attributes,
            *_semantic_deck_key(state.deck),
        )
