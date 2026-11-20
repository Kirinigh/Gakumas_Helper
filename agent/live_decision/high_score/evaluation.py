from __future__ import annotations

import math
import time
import random
import statistics
from enum import Enum
from typing import Mapping, Protocol
from dataclasses import replace, dataclass

from .kernel import (
    RuleEvent,
    EffectKind,
    KernelError,
    CardDefinition,
    EffectValueSource,
    PythonNiaProKernel,
)
from .dataset import DynamicDeckSample
from .planner import PlannerConfig, RobustHighScorePlanner
from .contracts import (
    PlanType,
    ActionKind,
    GameAction,
    RiskProfile,
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


class OfflineDecisionPolicy(Protocol):
    policy_name: str
    policy_version: str

    def decide(self, request: DecisionRequest) -> DecisionResponse: ...


class LogicRoute(str, Enum):
    GOOD_IMPRESSION = "GOOD_IMPRESSION"
    MOTIVATION_GENKI = "MOTIVATION_GENKI"
    MIXED = "MIXED"


def _effect_tree(definition: CardDefinition):
    pending = list(definition.effects + definition.persistent_effects)
    while pending:
        effect = pending.pop()
        yield effect
        pending.extend(effect.scheduled_effects)


def classify_logic_route(
    state: StrategyState,
    cards: Mapping[str, CardDefinition],
) -> LogicRoute:
    good_impression_signals = 0
    motivation_genki_signals = 0
    for card in state.deck.all_cards:
        definition = cards.get(card.card_id)
        if definition is None:
            continue
        for effect in _effect_tree(definition):
            if effect.kind is EffectKind.GOOD_IMPRESSION or (
                effect.kind is EffectKind.SCORE
                and effect.value_source is EffectValueSource.GOOD_IMPRESSION
            ):
                good_impression_signals += 1
            if effect.kind in {EffectKind.MOTIVATION, EffectKind.GENKI} or (
                effect.kind is EffectKind.SCORE
                and effect.value_source
                in {EffectValueSource.MOTIVATION, EffectValueSource.GENKI}
            ):
                motivation_genki_signals += 1
    if good_impression_signals > motivation_genki_signals * 1.5:
        return LogicRoute.GOOD_IMPRESSION
    if motivation_genki_signals > good_impression_signals * 1.5:
        return LogicRoute.MOTIVATION_GENKI
    return LogicRoute.MIXED


def _resource_potential(
    state: StrategyState,
    plan_type: PlanType,
    cards: Mapping[str, CardDefinition],
) -> float:
    turns = max(1, state.turns_remaining)
    action_value = state.actions_remaining * 4.0
    if plan_type is PlanType.SENSE:
        return (
            state.concentration * turns * 0.8
            + min(state.good_condition_turns, turns) * 5.0
            + min(state.excellent_condition_turns, turns) * 3.0
            + action_value
        )
    if plan_type is PlanType.LOGIC:
        route = classify_logic_route(state, cards)
        good_impression_value = state.good_impression_turns * turns
        motivation_genki_value = (
            state.motivation * turns * 0.7 + state.genki * 0.5
        )
        if route is LogicRoute.GOOD_IMPRESSION:
            return good_impression_value + motivation_genki_value * 0.25 + action_value
        if route is LogicRoute.MOTIVATION_GENKI:
            return motivation_genki_value + good_impression_value * 0.25 + action_value
        return good_impression_value * 0.7 + motivation_genki_value * 0.7 + action_value
    return (
        state.full_power * 1.5
        + action_value
        + state.passion * 0.5
    )


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    final_score: float
    stopped: bool
    illegal_action: bool
    decision_count: int
    latency_ms: float
    reason: str


@dataclass(frozen=True)
class EpisodeSimulation:
    result: EpisodeResult
    final_state: StrategyState
    events: tuple[RuleEvent, ...]


@dataclass(frozen=True)
class PolicyMetrics:
    policy: str
    policy_version: str
    episode_count: int
    mean_final_score: float
    p10_final_score: float
    worst_final_score: float
    best_final_score: float
    stop_rate: float
    illegal_action_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass(frozen=True)
class PairedComparison:
    candidate: str
    baseline: str
    paired_episode_count: int
    mean_delta: float
    mean_delta_ci95_low: float
    mean_delta_ci95_high: float
    candidate_p10: float
    baseline_p10: float
    p10_guard_passed: bool
    mean_improvement_passed: bool
    safety_passed: bool
    promotion_passed: bool


def _p10(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.10 * len(ordered)) - 1)
    return ordered[index]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _action_without_slot(action: GameAction) -> GameAction:
    return replace(action, source_slot=None)


_SLOTTED_CARD_ACTIONS = {
    ActionKind.PLAY_CARD,
    ActionKind.HOLD_CARD,
    ActionKind.MOVE_CARD,
    ActionKind.FREE_PLAY,
    ActionKind.PAID_PLAY,
}


class ImmediateScorePolicy:
    policy_name = "legal-immediate-score"
    policy_version = "1.5"

    def __init__(self, kernel: PythonNiaProKernel, *, include_resource_potential: bool = False,
                 timeout_ms: float | None = None, same_turn_setup_followup: bool = False,
                 same_turn_score_horizon: bool = False):
        if timeout_ms is not None and (not math.isfinite(timeout_ms) or timeout_ms <= 0):
            raise ValueError("one-step decision timeout must be finite and positive")
        self.kernel = kernel
        self.timeout_ms = timeout_ms
        self.same_turn_setup_followup = same_turn_setup_followup
        self.same_turn_score_horizon = same_turn_score_horizon
        self.include_resource_potential = include_resource_potential
        if include_resource_potential:
            self.policy_name = "one-step-state-potential"

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.perf_counter()
        candidates: list[tuple[float, GameAction, ValueSummary]] = []

        def budget_stop() -> DecisionResponse | None:
            elapsed = (time.perf_counter() - started) * 1000.0
            if self.timeout_ms is None or elapsed < self.timeout_ms:
                return None
            # A slow kernel transition cannot be interrupted here. Reject its
            # result as well as every partial comparison; never submit stale
            # best-so-far intent just because this is the early-turn policy.
            return DecisionResponse(
                DecisionOutcome.STOP_TASK, "one_step_timeout", None, None,
                DecisionTrace(self.policy_name, self.policy_version, self.kernel.ruleset_id,
                              0, len(candidates), elapsed,
                              reason_codes=("decision_budget_exhausted",)),
            )

        try:
            request.validate()
            actions = self.kernel.legal_actions(
                request.state,
                plan_type=request.plan_type,
                decision_kind=request.decision_kind,
            )
            for action in actions:
                if stopped := budget_stop():
                    return stopped
                branches = None
                project=getattr(self.kernel,'resource_evaluation_transition',None)
                if (self.include_resource_potential and self.same_turn_score_horizon
                        and project is not None and request.plan_type is PlanType.SENSE
                        and action.kind is ActionKind.END_TURN):
                    branches=project(request.state,action,plan_type=request.plan_type)
                if branches is None:
                    branches = self.kernel.transition(request.state, action, plan_type=request.plan_type)
                if stopped := budget_stop():
                    return stopped
                # END_TURN also resolves next-turn schedules. Those points
                # are outside an immediate card-action horizon: crediting
                # them only to skipping makes independent delayed scores
                # crowd out useful actions in the still-active turn.
                scores = [float(request.state.score if self.same_turn_score_horizon
                                and action.kind is ActionKind.END_TURN else branch.state.score)
                          for branch in branches]
                expected = sum(branch.probability * score for branch, score in zip(branches, scores))
                if self.include_resource_potential:
                    expected += sum(
                        branch.probability
                        * _resource_potential(
                            branch.state,
                            request.plan_type,
                            self.kernel.cards,
                        )
                        for branch in branches
                    )
                if (
                    self.include_resource_potential and self.same_turn_setup_followup
                    and action.kind is ActionKind.PLAY_CARD
                    and len(branches) <= 8
                    and all(branch.state.actions_remaining >= request.state.actions_remaining
                            and branch.state.pending_choice is None for branch in branches)
                ):
                    # One free setup can draw several possible hands. Choose
                    # the next action separately in each observed branch,
                    # then weight those values; do not invent the draw order.
                    # Admission is bounded before extra transitions start.
                    continuations = [(branch, tuple(followup for followup in self.kernel.legal_actions(
                        branch.state, plan_type=request.plan_type, decision_kind=DecisionKind.CARD,
                    ) if followup.kind is ActionKind.PLAY_CARD)) for branch in branches]
                    if sum(len(followups) for _, followups in continuations) <= 32:
                        setup_value = 0.0
                        for branch, followups in continuations:
                            prepared = branch.state
                            best = prepared.score + _resource_potential(prepared, request.plan_type, self.kernel.cards)
                            for followup in followups:
                                if stopped := budget_stop():
                                    return stopped
                                continued = self.kernel.transition(prepared, followup, plan_type=request.plan_type)
                                if stopped := budget_stop():
                                    return stopped
                                if any(child.state.stamina < request.utility.minimum_stamina
                                       or child.state.genki < request.utility.minimum_genki for child in continued):
                                    continue
                                best = max(best, sum(
                                    child.probability * (child.state.score + _resource_potential(
                                        child.state, request.plan_type, self.kernel.cards,
                                    )) for child in continued
                                ))
                            setup_value += branch.probability * best
                        expected = max(expected, setup_value)
                value = ValueSummary(
                    expected_final_score=sum(branch.probability * score for branch, score in zip(branches, scores)),
                    p10_final_score=min(scores),
                    worst_final_score=min(scores),
                    best_final_score=max(scores),
                    outcome_count=len(branches),
                )
                if any(
                    branch.state.stamina < request.utility.minimum_stamina
                    or branch.state.genki < request.utility.minimum_genki
                    for branch in branches
                ):
                    continue
                candidates.append((expected, action, value))
        except (ValueError, KernelError) as error:
            return DecisionResponse(
                outcome=DecisionOutcome.STOP_TASK,
                reason=f"baseline_error:{error}",
                action=None,
                value=None,
                trace=DecisionTrace(
                    planner=self.policy_name,
                    planner_version=self.policy_version,
                    ruleset_id=self.kernel.ruleset_id,
                    completed_depth=0,
                    expanded_nodes=0,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )
        if stopped := budget_stop():
            return stopped
        if not candidates:
            reason = "baseline_all_candidates_rejected" if actions else "baseline_no_legal_action"
            return DecisionResponse(
                outcome=DecisionOutcome.STOP_TASK,
                reason=reason,
                action=None,
                value=None,
                trace=DecisionTrace(
                    planner=self.policy_name,
                    planner_version=self.policy_version,
                    ruleset_id=self.kernel.ruleset_id,
                    completed_depth=1,
                    expanded_nodes=0,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )
        def candidate_key(item: tuple[float, GameAction, ValueSummary]) -> tuple[float, float, float, str, str]:
            potential, action, value = item
            if request.utility.risk_profile is RiskProfile.SAFE_BASELINE:
                metrics = (value.p10_final_score, value.worst_final_score, potential)
            elif request.utility.risk_profile is RiskProfile.AGGRESSIVE_HIGH_SCORE:
                metrics = (potential, value.best_final_score, value.p10_final_score)
            else:
                metrics = (potential, value.p10_final_score, value.worst_final_score)
            return (-metrics[0], -metrics[1], -metrics[2], action.kind.value, action.card_instance_id or "")

        expected, action, value = min(candidates, key=candidate_key)
        _ = expected
        if action.kind in _SLOTTED_CARD_ACTIONS:
            slots = dict(request.card_slots)
            if action.card_instance_id not in slots:
                return DecisionResponse(
                    outcome=DecisionOutcome.STOP_TASK,
                    reason="baseline_source_slot_missing",
                    action=None,
                    value=None,
                    trace=None,
                )
            action = replace(action, source_slot=slots[action.card_instance_id])
        if action.kind is ActionKind.USE_DRINK and request.decision_kind is DecisionKind.LIVE:
            slots = [slot for slot, drink_id in request.drink_slots if drink_id == action.drink_id]
            if not slots:
                return DecisionResponse(DecisionOutcome.STOP_TASK, "baseline_drink_slot_missing", None, None, None)
            action = replace(action, source_slot=min(slots))
        trace = DecisionTrace(
            planner=self.policy_name,
            planner_version=self.policy_version,
            ruleset_id=self.kernel.ruleset_id,
            completed_depth=1,
            expanded_nodes=len(candidates),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            candidate_values=tuple(
                CandidateValue(action=item[1], value=item[2], probability_mass=1.0,
                               ranking_utility=item[0]) for item in candidates
            ),
            value_scope="ONE_STEP_SCORE",
        )
        response = DecisionResponse(DecisionOutcome.DECIDE, "one_step_complete", action, value, trace)
        response.validate()
        return response


class PhaseAwareHighScorePolicy:
    policy_name = "phase-aware-expectimax"
    policy_version = "1.29"

    def __init__(
        self,
        kernel: PythonNiaProKernel,
        *,
        exact_turns_remaining: int = 2,
        exact_config: PlannerConfig | None = None,
        maximum_exact_hand_size_for_multi_action: int = 3,
        same_turn_setup_followup: bool = False,
        same_turn_score_horizon: bool = False,
    ):
        if exact_turns_remaining < 1 or maximum_exact_hand_size_for_multi_action < 1:
            raise ValueError("phase-aware policy thresholds must be positive")
        self.kernel = kernel
        self.exact_turns_remaining = exact_turns_remaining
        self.maximum_exact_hand_size_for_multi_action = maximum_exact_hand_size_for_multi_action
        self.exact = RobustHighScorePlanner(kernel, exact_config)
        self.early = ImmediateScorePolicy(kernel, include_resource_potential=True,
                                          timeout_ms=self.exact.config.timeout_ms,
                                          same_turn_setup_followup=same_turn_setup_followup,
                                          same_turn_score_horizon=same_turn_score_horizon)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
            return self.exact.decide(request)
        if request.state.turns_remaining > self.exact_turns_remaining and request.utility.baseline_p10 is None:
            return self._early_decide(request, "phase_policy:pre_endgame")
        expanding_effects = {
            EffectKind.CARD_MODIFICATION,
            EffectKind.DRAW,
            EffectKind.EXTRA_ACTION,
            EffectKind.HOLD_SELECTED,
            EffectKind.SCHEDULE,
            EffectKind.TURNS_REMAINING,
            EffectKind.MOVE_RANDOM_TO_HAND,
            EffectKind.MOVE_RANDOM_TO_TOP,
            EffectKind.EXCHANGE_HAND,
            EffectKind.FREE_PLAY_SELECTED,
            EffectKind.FREE_PLAY_ALL,
            EffectKind.PAID_PLAY_SELECTED,
            EffectKind.FREE_PLAY_RANDOM,
            EffectKind.HALF_COST_TURNS,
            EffectKind.DOUBLE_COST_TURNS,
            EffectKind.ADD_CARD_TO_DECK,
            EffectKind.MOVE_SELECTED_TO_HAND,
            EffectKind.ENTHUSIASM_BONUS_BUFF,
            EffectKind.ENTHUSIASM_MULTIPLIER_BUFF,
            EffectKind.FULL_POWER_CHARGE_BUFF,
            EffectKind.STANCE_LOCK_TURNS,
            EffectKind.NULLIFY_COST_CARDS,
        }
        future_branching_effects = {
            EffectKind.DRAW,
            EffectKind.EXTRA_ACTION,
            EffectKind.HOLD_SELECTED,
            EffectKind.TURNS_REMAINING,
            EffectKind.MOVE_RANDOM_TO_HAND,
            EffectKind.MOVE_RANDOM_TO_TOP,
            EffectKind.EXCHANGE_HAND,
            EffectKind.FREE_PLAY_SELECTED,
            EffectKind.FREE_PLAY_ALL,
            EffectKind.PAID_PLAY_SELECTED,
            EffectKind.FREE_PLAY_RANDOM,
            EffectKind.ADD_CARD_TO_DECK,
            EffectKind.MOVE_SELECTED_TO_HAND,
        }

        expanding_card_present = any(
            definition is not None
            and any(effect.kind in expanding_effects for effect in _effect_tree(definition))
            for card in request.state.deck.hand
            for definition in (self.kernel.cards.get(card.card_id),)
        )
        zone_cards = {
            "HAND": request.state.deck.hand,
            "DRAW_PILE": request.state.deck.draw_pile,
            "DISCARD": request.state.deck.discard,
            "REMOVED": request.state.deck.removed,
            "HELD": request.state.deck.held,
        }
        large_secondary_reachable = any(
            sum(
                candidate.instance_id != card.instance_id
                for zone in effect.target_zones
                for candidate in zone_cards[zone.value]
            )
            > self.maximum_exact_hand_size_for_multi_action
            for card in request.state.deck.hand
            for definition in (self.kernel.cards.get(card.card_id),)
            if definition is not None
            for effect in _effect_tree(definition)
            if effect.kind
            in {
                EffectKind.HOLD_SELECTED,
                EffectKind.FREE_PLAY_SELECTED,
                EffectKind.FREE_PLAY_ALL,
                EffectKind.PAID_PLAY_SELECTED,
            }
        )
        complex_card_root = (
            request.state.pending_choice is None
            and (
                large_secondary_reachable
                or (
                    len(request.state.deck.hand) > self.maximum_exact_hand_size_for_multi_action
                    and (request.state.actions_remaining > 1 or expanding_card_present)
                )
            )
        )
        complex_secondary = (
            request.state.pending_choice is not None
            and len(request.state.pending_choice.candidate_instance_ids)
            > self.maximum_exact_hand_size_for_multi_action
        )
        refill_count = max(0, request.state.hand_limit - len(request.state.deck.hand))
        composition_refill_complexity = (
            request.state.pending_choice is None
            and request.state.turns_remaining > 1
            and request.state.deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION
            and refill_count >= 2
            and len(request.state.deck.draw_pile) >= refill_count
            and request.state.hand_limit > self.maximum_exact_hand_size_for_multi_action
        )
        wide_composition_endgame = (
            request.state.pending_choice is None
            and request.state.turns_remaining > 1
            and request.state.deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION
            and len(request.state.deck.hand)
            > self.maximum_exact_hand_size_for_multi_action
            and len(request.state.deck.draw_pile)
            > self.maximum_exact_hand_size_for_multi_action
        )
        future_secondary_draw_complexity = (
            request.state.pending_choice is None
            and request.state.turns_remaining > 1
            and request.state.deck.knowledge
            in {DeckKnowledge.KNOWN_ORDER, DeckKnowledge.KNOWN_COMPOSITION}
            and any(
                sum(
                    candidate.instance_id != card.instance_id
                    for zone in effect.target_zones
                    for candidate in zone_cards[zone.value]
                )
                > self.maximum_exact_hand_size_for_multi_action
                for card in request.state.deck.draw_pile
                for definition in (self.kernel.cards.get(card.card_id),)
                if definition is not None
                for effect in _effect_tree(definition)
                if effect.kind
                in {
                    EffectKind.HOLD_SELECTED,
                    EffectKind.FREE_PLAY_SELECTED,
                    EffectKind.FREE_PLAY_ALL,
                    EffectKind.PAID_PLAY_SELECTED,
                }
            )
        )
        future_expanding_draw_complexity = (
            request.state.pending_choice is None
            and request.state.turns_remaining > 1
            and request.state.hand_limit > self.maximum_exact_hand_size_for_multi_action
            and request.state.deck.knowledge
            in {DeckKnowledge.KNOWN_ORDER, DeckKnowledge.KNOWN_COMPOSITION}
            and any(
                any(
                    effect.kind in future_branching_effects
                    for effect in _effect_tree(definition)
                )
                for card in request.state.deck.draw_pile
                for definition in (self.kernel.cards.get(card.card_id),)
                if definition is not None
            )
        )
        # Selecting several cards for the top of the deck expands permutations
        # independently of hand width. It can also be reached after a draw/free
        # play chain in the next turn, so a three-card hand is not a bound on
        # this search. Choose the existing one-step strategy before searching;
        # an incomplete exact search still never falls back after timing out.
        reachable_cards = request.state.deck.hand
        if request.state.turns_remaining > 1:
            reachable_cards += request.state.deck.draw_pile + request.state.deck.discard
        random_top_order_complexity = any(
            effect.kind is EffectKind.MOVE_RANDOM_TO_TOP
            and effect.amount > 1
            and effect.card_selector is not None
            and sum(
                self.kernel._selector_matches(candidate, effect.card_selector)
                for zone in effect.card_selector.zones
                for candidate in zone_cards[zone.value]
            ) >= self.maximum_exact_hand_size_for_multi_action
            for card in reachable_cards
            for definition in (self.kernel.cards.get(card.card_id),)
            if definition is not None
            for effect in _effect_tree(definition)
        )
        if request.utility.baseline_p10 is None and (
            complex_card_root
            or complex_secondary
            or composition_refill_complexity
            or wide_composition_endgame
            or future_secondary_draw_complexity
            or future_expanding_draw_complexity
            or random_top_order_complexity
        ):
            return self._early_decide(request, "phase_policy:exact_complexity_guard")
        return self.exact.decide(request)

    def _early_decide(self, request: DecisionRequest, reason: str) -> DecisionResponse:
        response = self.early.decide(request)
        if response.trace is None:
            return response
        response = replace(
            response,
            trace=replace(response.trace, reason_codes=response.trace.reason_codes + (reason,)),
        )
        response.validate()
        return response


def simulate_episode(
    sample: DynamicDeckSample,
    policy: OfflineDecisionPolicy,
    kernel: PythonNiaProKernel,
    *,
    maximum_decisions: int = 64,
) -> EpisodeResult:
    return simulate_episode_with_trace(
        sample,
        policy,
        kernel,
        maximum_decisions=maximum_decisions,
    ).result


def simulate_episode_with_trace(
    sample: DynamicDeckSample,
    policy: OfflineDecisionPolicy,
    kernel: PythonNiaProKernel,
    *,
    maximum_decisions: int = 64,
) -> EpisodeSimulation:
    import time

    rng = random.Random(sample.seed)
    state = sample.request.state
    elapsed = 0.0
    events: list[RuleEvent] = []
    for decision_index in range(maximum_decisions):
        if state.terminal:
            result = EpisodeResult(
                sample.episode_id,
                float(state.score),
                False,
                False,
                decision_index,
                elapsed,
                "terminal",
            )
            return EpisodeSimulation(result, state, tuple(events))
        decision_kind = (DecisionKind.SECONDARY if state.pending_choice is not None
                         else DecisionKind.LIVE if sample.request.decision_kind is DecisionKind.LIVE
                         else DecisionKind.CARD)
        slot_ids = (
            state.pending_choice.candidate_instance_ids if state.pending_choice is not None else tuple(card.instance_id for card in state.deck.hand)
        )
        request = replace(
            sample.request,
            snapshot_id=f"{sample.episode_id}-decision-{decision_index}",
            state=state,
            decision_kind=decision_kind,
            card_slots=tuple((instance_id, index) for index, instance_id in enumerate(slot_ids)),
            drink_slots=(tuple(enumerate(state.drinks))
                         if sample.request.decision_kind is DecisionKind.LIVE else ()),
        )
        started = time.perf_counter()
        response = policy.decide(request)
        elapsed += (time.perf_counter() - started) * 1000.0
        if response.outcome is not DecisionOutcome.DECIDE or response.action is None:
            result = EpisodeResult(
                sample.episode_id, float(state.score), True, False,
                decision_index + 1, elapsed, response.reason,
            )
            return EpisodeSimulation(result, state, tuple(events))
        try:
            legal = kernel.legal_actions(state, plan_type=request.plan_type, decision_kind=decision_kind)
            action = _action_without_slot(response.action)
            if action not in legal:
                result = EpisodeResult(
                    sample.episode_id, float(state.score), True, True,
                    decision_index + 1, elapsed, "policy_returned_illegal_action",
                )
                return EpisodeSimulation(result, state, tuple(events))
            branches = kernel.transition(state, action, plan_type=request.plan_type)
        except KernelError as error:
            result = EpisodeResult(
                sample.episode_id, float(state.score), True, True,
                decision_index + 1, elapsed, f"transition_error:{error}",
            )
            return EpisodeSimulation(result, state, tuple(events))
        draw = rng.random()
        cumulative = 0.0
        selected = branches[-1]
        for branch in branches:
            cumulative += branch.probability
            if draw <= cumulative:
                selected = branch
                break
        events.extend(selected.events)
        state = selected.state
    result = EpisodeResult(
        sample.episode_id,
        float(state.score),
        True,
        False,
        maximum_decisions,
        elapsed,
        "maximum_decisions_exceeded",
    )
    return EpisodeSimulation(result, state, tuple(events))


def evaluate_policy(
    samples: tuple[DynamicDeckSample, ...],
    policy: OfflineDecisionPolicy,
    kernel: PythonNiaProKernel,
) -> tuple[PolicyMetrics, tuple[EpisodeResult, ...]]:
    results = tuple(simulate_episode(sample, policy, kernel) for sample in samples)
    return summarize_policy_results(policy, results), results


def summarize_policy_results(
    policy: OfflineDecisionPolicy,
    results: tuple[EpisodeResult, ...],
) -> PolicyMetrics:
    scores = [result.final_score for result in results]
    latencies = [result.latency_ms for result in results]
    count = len(results)
    metrics = PolicyMetrics(
        policy=policy.policy_name,
        policy_version=policy.policy_version,
        episode_count=count,
        mean_final_score=statistics.fmean(scores) if scores else 0.0,
        p10_final_score=_p10(scores),
        worst_final_score=min(scores) if scores else 0.0,
        best_final_score=max(scores) if scores else 0.0,
        stop_rate=sum(result.stopped for result in results) / count if count else 0.0,
        illegal_action_rate=sum(result.illegal_action for result in results) / count if count else 0.0,
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=_p95(latencies),
    )
    return metrics


def compare_paired_results(
    candidate_metrics: PolicyMetrics,
    candidate_results: tuple[EpisodeResult, ...],
    baseline_metrics: PolicyMetrics,
    baseline_results: tuple[EpisodeResult, ...],
    *,
    bootstrap_rounds: int = 2000,
    bootstrap_seed: int = 275,
) -> PairedComparison:
    if bootstrap_rounds < 1:
        raise ValueError("bootstrap_rounds must be positive")
    candidate_by_id = {result.episode_id: result for result in candidate_results}
    baseline_by_id = {result.episode_id: result for result in baseline_results}
    episode_ids = sorted(set(candidate_by_id) & set(baseline_by_id))
    if not episode_ids:
        raise ValueError("paired comparison requires common episodes")
    deltas = [candidate_by_id[episode_id].final_score - baseline_by_id[episode_id].final_score for episode_id in episode_ids]
    rng = random.Random(bootstrap_seed)
    means = []
    for _round in range(bootstrap_rounds):
        resampled = [deltas[rng.randrange(len(deltas))] for _index in range(len(deltas))]
        means.append(statistics.fmean(resampled))
    means.sort()
    low = means[max(0, math.floor(0.025 * (len(means) - 1)))]
    high = means[min(len(means) - 1, math.ceil(0.975 * (len(means) - 1)))]
    p10_guard = candidate_metrics.p10_final_score >= baseline_metrics.p10_final_score
    mean_pass = low > 0
    safety_pass = candidate_metrics.illegal_action_rate == 0.0 and candidate_metrics.stop_rate == 0.0
    return PairedComparison(
        candidate=f"{candidate_metrics.policy}@{candidate_metrics.policy_version}",
        baseline=f"{baseline_metrics.policy}@{baseline_metrics.policy_version}",
        paired_episode_count=len(episode_ids),
        mean_delta=statistics.fmean(deltas),
        mean_delta_ci95_low=low,
        mean_delta_ci95_high=high,
        candidate_p10=candidate_metrics.p10_final_score,
        baseline_p10=baseline_metrics.p10_final_score,
        p10_guard_passed=p10_guard,
        mean_improvement_passed=mean_pass,
        safety_passed=safety_pass,
        promotion_passed=p10_guard and mean_pass and safety_pass,
    )
