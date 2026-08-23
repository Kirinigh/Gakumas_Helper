from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from .kernel import (
    EffectKind,
    EffectSpec,
    CardCostKind,
    CardSelector,
    ConditionKind,
    CardDefinition,
    DrinkDefinition,
    EffectCondition,
    CardModification,
    EffectValueSource,
)
from .contracts import (
    NIA_PRO_RULESET_ID,
    Stance,
    CardRef,
    CardZone,
    PlanType,
    ActionKind,
    DeckBelief,
    GameAction,
    RiskProfile,
    TimedStatus,
    UtilitySpec,
    DecisionKind,
    DeckKnowledge,
    PendingChoice,
    SchedulePhase,
    StrategyState,
    TimedModifier,
    ResolutionKind,
    DecisionOutcome,
    DecisionRequest,
    CapabilityReport,
    EffectSourceKind,
    ScheduledEffectRef,
    TurnResolutionStage,
    EffectTriggerContext,
    ResolutionContinuation,
    CardEffectsContinuation,
    FreePlayAllContinuation,
    TurnResolutionContinuation,
    ScheduledEffectsContinuation,
)


@dataclass(frozen=True)
class GoldenExpectation:
    outcome: DecisionOutcome
    action: GameAction | None
    expected_final_score: float | None = None
    p10_final_score: float | None = None


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    request: DecisionRequest
    expectation: GoldenExpectation


@dataclass(frozen=True)
class GoldenSuite:
    dataset_id: str
    ruleset_id: str
    cards: dict[str, CardDefinition]
    drinks: dict[str, DrinkDefinition]
    cases: tuple[GoldenCase, ...]


def _condition(raw: dict[str, Any] | None) -> EffectCondition | None:
    if raw is None:
        return None
    return EffectCondition(
        kind=ConditionKind(str(raw["kind"])),
        value=str(raw["value"]) if raw.get("value") is not None else None,
        threshold=int(raw["threshold"]) if raw.get("threshold") is not None else None,
        all_of=tuple(
            child
            for value in raw.get("all_of", [])
            if (child := _condition(value)) is not None
        ),
        modulus=int(raw["modulus"]) if raw.get("modulus") is not None else None,
        remainder=int(raw["remainder"]) if raw.get("remainder") is not None else None,
    )


def _effect(raw: dict[str, Any]) -> EffectSpec:
    condition = _condition(raw.get("condition"))
    selector_raw = raw.get("card_selector")
    selector = (
        CardSelector(
            all_cards=bool(selector_raw.get("all_cards", False)),
            source_only=bool(selector_raw.get("source_only", False)),
            zones=tuple(CardZone(str(value)) for value in selector_raw.get("zones", [])),
            card_type=str(selector_raw["card_type"]) if selector_raw.get("card_type") is not None else None,
            card_id=str(selector_raw["card_id"]) if selector_raw.get("card_id") is not None else None,
            base_card_id=(
                str(selector_raw["base_card_id"])
                if selector_raw.get("base_card_id") is not None
                else None
            ),
            effect_tag=str(selector_raw["effect_tag"]) if selector_raw.get("effect_tag") is not None else None,
            source_type=(
                str(selector_raw["source_type"])
                if selector_raw.get("source_type") is not None
                else None
            ),
        )
        if selector_raw is not None
        else None
    )
    modification_raw = raw.get("card_modification")
    modification = (
        CardModification(
            score_bonus=int(modification_raw.get("score_bonus", 0)),
            genki_bonus=int(modification_raw.get("genki_bonus", 0)),
            stamina_cost_delta=int(modification_raw.get("stamina_cost_delta", 0)),
            typed_cost_delta=int(modification_raw.get("typed_cost_delta", 0)),
            score_repeat_bonus=int(modification_raw.get("score_repeat_bonus", 0)),
        )
        if modification_raw is not None
        else None
    )
    return EffectSpec(
        kind=EffectKind(str(raw["kind"])),
        amount=float(raw.get("amount", 0.0)),
        repeats=int(raw.get("repeats", 1)),
        stance=Stance(str(raw["stance"])) if raw.get("stance") is not None else None,
        status_id=str(raw["status_id"]) if raw.get("status_id") is not None else None,
        duration=int(raw["duration"]) if raw.get("duration") is not None else None,
        full_power_scale=float(raw.get("full_power_scale", 0.0)),
        condition=condition,
        target_zones=tuple(CardZone(str(value)) for value in raw.get("target_zones", [])),
        card_selector=selector,
        card_modification=modification,
        scheduled_effects=tuple(_effect(value) for value in raw.get("scheduled_effects", [])),
        delay_turns=int(raw.get("delay_turns", 1)),
        schedule_phase=SchedulePhase(str(raw.get("schedule_phase", SchedulePhase.TURN.value))),
        schedule_limit=(
            int(raw["schedule_limit"])
            if raw.get("schedule_limit") is not None
            else (None if "schedule_limit" in raw else 1)
        ),
        schedule_ttl=(
            int(raw["schedule_ttl"])
            if raw.get("schedule_ttl") is not None
            else None
        ),
        trigger_card_type=(
            str(raw["trigger_card_type"])
            if raw.get("trigger_card_type") is not None
            else None
        ),
        selection_count=int(raw.get("selection_count", 1)),
        selection_optional=bool(raw.get("selection_optional", False)),
        card_id=str(raw["card_id"]) if raw.get("card_id") is not None else None,
        score_enthusiasm_multiplier=float(
            raw.get("score_enthusiasm_multiplier", 1.0)
        ),
        value_source=(
            EffectValueSource(str(raw["value_source"]))
            if raw.get("value_source") is not None
            else None
        ),
        value_scale=float(raw.get("value_scale", 0.0)),
        score_concentration_multiplier=float(
            raw.get("score_concentration_multiplier", 1.0)
        ),
        score_good_condition_multiplier=float(
            raw.get("score_good_condition_multiplier", 1.0)
        ),
        genki_motivation_multiplier=float(
            raw.get("genki_motivation_multiplier", 1.0)
        ),
    )


def _card_definition(raw: dict[str, Any]) -> CardDefinition:
    condition = _condition(raw.get("condition"))
    return CardDefinition(
        card_id=str(raw["card_id"]),
        name=str(raw["name"]),
        plan=PlanType(str(raw["plan"])) if raw.get("plan") is not None else None,
        stamina_cost=int(raw.get("stamina_cost", 0)),
        full_power_cost=int(raw.get("full_power_cost", 0)),
        cost_kind=CardCostKind(str(raw.get("cost_kind", CardCostKind.NORMAL.value))),
        effects=tuple(_effect(effect) for effect in raw.get("effects", [])),
        persistent_effects=tuple(
            _effect(effect) for effect in raw.get("persistent_effects", [])
        ),
        limited=bool(raw.get("limited", False)),
        required_stance=Stance(str(raw["required_stance"])) if raw.get("required_stance") is not None else None,
        condition=condition,
        card_type=str(raw["card_type"]) if raw.get("card_type") is not None else None,
        rarity=str(raw["rarity"]) if raw.get("rarity") is not None else None,
        base_card_id=(
            str(raw["base_card_id"])
            if raw.get("base_card_id") is not None
            else None
        ),
        upgrade_card_id=(
            str(raw["upgrade_card_id"])
            if raw.get("upgrade_card_id") is not None
            else None
        ),
        source_type=(
            str(raw["source_type"])
            if raw.get("source_type") is not None
            else None
        ),
    )


def _drink_definition(raw: dict[str, Any]) -> DrinkDefinition:
    return DrinkDefinition(
        drink_id=str(raw["drink_id"]),
        name=str(raw["name"]),
        effects=tuple(_effect(effect) for effect in raw.get("effects", [])),
        stamina_cost=int(raw.get("stamina_cost", 0)),
        plan=(
            PlanType(str(raw["plan"]))
            if raw.get("plan") is not None
            else None
        ),
    )


def _card_ref(raw: dict[str, Any]) -> CardRef:
    return CardRef(
        instance_id=str(raw["instance_id"]),
        card_id=str(raw["card_id"]),
        upgraded=bool(raw.get("upgraded", False)),
        score_bonus=int(raw.get("score_bonus", 0)),
        genki_bonus=int(raw.get("genki_bonus", 0)),
        stamina_cost_delta=int(raw.get("stamina_cost_delta", 0)),
        typed_cost_delta=int(raw.get("typed_cost_delta", 0)),
        score_repeat_bonus=int(raw.get("score_repeat_bonus", 0)),
    )


def _scheduled_ref(raw: dict[str, Any]) -> ScheduledEffectRef:
    return ScheduledEffectRef(
        source_id=str(raw["source_id"]),
        card_id=str(raw["card_id"]),
        definition_effect_index=int(raw["definition_effect_index"]),
        source_kind=EffectSourceKind(
            str(raw.get("source_kind", EffectSourceKind.CARD.value))
        ),
        turns_until_fire=int(raw.get("turns_until_fire", 1)),
        trigger_phase=SchedulePhase(
            str(raw.get("trigger_phase", SchedulePhase.TURN.value))
        ),
        remaining_uses=(
            int(raw["remaining_uses"])
            if raw.get("remaining_uses") is not None
            else (None if "remaining_uses" in raw else 1)
        ),
        remaining_turns=(
            int(raw["remaining_turns"])
            if raw.get("remaining_turns") is not None
            else None
        ),
        persistent=bool(raw.get("persistent", False)),
        direct_on_trigger=bool(raw.get("direct_on_trigger", False)),
    )


def _trigger_context(raw: dict[str, Any] | None) -> EffectTriggerContext | None:
    if raw is None:
        return None
    return EffectTriggerContext(
        is_direct_effect=bool(raw.get("is_direct_effect", False)),
        used_card_instance_id=(
            str(raw["used_card_instance_id"])
            if raw.get("used_card_instance_id") is not None
            else None
        ),
        used_card_id=(
            str(raw["used_card_id"])
            if raw.get("used_card_id") is not None
            else None
        ),
        used_card_base_id=(
            str(raw["used_card_base_id"])
            if raw.get("used_card_base_id") is not None
            else None
        ),
        used_card_type=(
            str(raw["used_card_type"])
            if raw.get("used_card_type") is not None
            else None
        ),
        used_card_rarity=(
            str(raw["used_card_rarity"])
            if raw.get("used_card_rarity") is not None
            else None
        ),
        moved_card_instance_id=(
            str(raw["moved_card_instance_id"])
            if raw.get("moved_card_instance_id") is not None
            else None
        ),
        moved_card_id=(
            str(raw["moved_card_id"])
            if raw.get("moved_card_id") is not None
            else None
        ),
        moved_card_base_id=(
            str(raw["moved_card_base_id"])
            if raw.get("moved_card_base_id") is not None
            else None
        ),
        exclude_used_card_from_targets=bool(
            raw.get("exclude_used_card_from_targets", False)
        ),
    )


def _resolution_continuation(raw: dict[str, Any]) -> ResolutionContinuation:
    kind = ResolutionKind(str(raw["continuation_kind"]))
    if kind is ResolutionKind.CARD_EFFECTS:
        return CardEffectsContinuation(
            source_id=str(raw["source_id"]),
            card_id=str(raw["card_id"]),
            continuation_index=int(raw["continuation_index"]),
            remaining_effect_passes=int(raw["remaining_effect_passes"]),
            condition_stance=Stance(str(raw["condition_stance"])),
            condition_full_power=float(raw["condition_full_power"]),
        )
    if kind is ResolutionKind.SCHEDULED_EFFECTS:
        return ScheduledEffectsContinuation(
            scheduled=_scheduled_ref(raw["scheduled"]),
            next_nested_index=int(raw["next_nested_index"]),
            matched_nested_indices=tuple(
                int(value) for value in raw.get("matched_nested_indices", [])
            ),
            remaining_due=tuple(
                _scheduled_ref(value) for value in raw.get("remaining_due", [])
            ),
            remaining_matched_nested_indices=tuple(
                tuple(int(index) for index in values)
                for values in raw.get("remaining_matched_nested_indices", [])
            ),
            trigger_context=_trigger_context(raw.get("trigger_context")),
            fired=bool(raw.get("fired", False)),
        )
    if kind is ResolutionKind.FREE_PLAY_ALL:
        return FreePlayAllContinuation(
            source_id=str(raw["source_id"]),
            remaining_instance_ids=tuple(
                str(value) for value in raw.get("remaining_instance_ids", [])
            ),
        )
    if kind is ResolutionKind.TURN:
        return TurnResolutionContinuation(
            plan_type=PlanType(str(raw["plan_type"])),
            stage=TurnResolutionStage(str(raw["stage"])),
            entered_full_power=bool(raw.get("entered_full_power", False)),
            due_before_start=tuple(
                _scheduled_ref(value) for value in raw.get("due_before_start", [])
            ),
            due_start=tuple(
                _scheduled_ref(value) for value in raw.get("due_start", [])
            ),
            due_after_start=tuple(
                _scheduled_ref(value) for value in raw.get("due_after_start", [])
            ),
            due_turn=tuple(
                _scheduled_ref(value) for value in raw.get("due_turn", [])
            ),
            draws_remaining=int(raw.get("draws_remaining", 0)),
            held_return_ids=tuple(
                str(value) for value in raw.get("held_return_ids", [])
            ),
        )
    raise ValueError(f"unsupported resolution continuation: {kind.value}")


def _state(raw: dict[str, Any]) -> StrategyState:
    all_cards = tuple(_card_ref(card) for card in raw["deck"]["all_cards"])
    by_id = {card.instance_id: card for card in all_cards}

    def zone(name: str) -> tuple[CardRef, ...]:
        return tuple(by_id[str(instance_id)] for instance_id in raw["deck"].get(name, []))

    deck = DeckBelief(
        all_cards=all_cards,
        hand=zone("hand"),
        draw_pile=zone("draw_pile"),
        discard=zone("discard"),
        removed=zone("removed"),
        held=zone("held"),
        knowledge=DeckKnowledge(str(raw["deck"].get("knowledge", DeckKnowledge.KNOWN_COMPOSITION.value))),
    )
    pending_raw = raw.get("pending_choice")
    pending_choice = (
        PendingChoice(
            action_kind=ActionKind(str(pending_raw["action_kind"])),
            source_id=str(pending_raw["source_id"]),
            candidate_instance_ids=tuple(str(value) for value in pending_raw["candidate_instance_ids"]),
            continuation_index=(
                int(pending_raw["continuation_index"])
                if pending_raw.get("continuation_index") is not None
                else None
            ),
            remaining_effect_passes=int(pending_raw.get("remaining_effect_passes", 0)),
            condition_stance=(
                Stance(str(pending_raw["condition_stance"]))
                if pending_raw.get("condition_stance") is not None
                else None
            ),
            condition_full_power=(
                float(pending_raw["condition_full_power"])
                if pending_raw.get("condition_full_power") is not None
                else None
            ),
            selections_remaining=int(pending_raw.get("selections_remaining", 1)),
            selection_optional=bool(pending_raw.get("selection_optional", False)),
        )
        if pending_raw is not None
        else None
    )
    return StrategyState(
        turn=int(raw["turn"]),
        turns_remaining=int(raw["turns_remaining"]),
        score=int(raw.get("score", 0)),
        stamina=int(raw["stamina"]),
        max_stamina=int(raw["max_stamina"]),
        genki=int(raw.get("genki", 0)),
        actions_remaining=int(raw["actions_remaining"]),
        hand_limit=int(raw.get("hand_limit", 5)),
        score_multiplier=float(raw.get("score_multiplier", 1.0)),
        deck=deck,
        pending_choice=pending_choice,
        pending_after_card_ids=tuple(
            str(value) for value in raw.get("pending_after_card_ids", [])
        ),
        resolution_stack=tuple(
            _resolution_continuation(value)
            for value in raw.get("resolution_stack", [])
        ),
        stance=Stance(str(raw.get("stance", Stance.NEUTRAL.value))),
        previous_stance=Stance(str(raw.get("previous_stance", Stance.NEUTRAL.value))),
        full_power=float(raw.get("full_power", 0)),
        cumulative_full_power=float(raw.get("cumulative_full_power", raw.get("full_power", 0))),
        passion=float(raw.get("passion", 0)),
        passion_buff=int(raw.get("passion_buff", 0)),
        passion_gain_bonus_pct=float(raw.get("passion_gain_bonus_pct", 0.0)),
        parameter_gain_bonus_pct=float(raw.get("parameter_gain_bonus_pct", 0.0)),
        concentration=int(raw.get("concentration", 0)),
        good_condition_turns=int(raw.get("good_condition_turns", 0)),
        excellent_condition_turns=int(raw.get("excellent_condition_turns", 0)),
        good_impression_turns=int(raw.get("good_impression_turns", 0)),
        motivation=int(raw.get("motivation", 0)),
        statuses=tuple(TimedStatus(**status) for status in raw.get("statuses", [])),
        scheduled_effects=tuple(
            _scheduled_ref(effect) for effect in raw.get("scheduled_effects", [])
        ),
        persistent_effects_initialized=bool(raw.get("persistent_effects_initialized", False)),
        double_card_effects_remaining=int(raw.get("double_card_effects_remaining", 0)),
        half_cost_turns=int(raw.get("half_cost_turns", 0)),
        half_cost_fresh=bool(raw.get("half_cost_fresh", False)),
        double_cost_turns=int(raw.get("double_cost_turns", 0)),
        double_cost_fresh=bool(raw.get("double_cost_fresh", False)),
        stance_lock_turns=int(raw.get("stance_lock_turns", 0)),
        stance_lock_fresh=bool(raw.get("stance_lock_fresh", False)),
        nullify_cost_cards=int(raw.get("nullify_cost_cards", 0)),
        strength_times=int(raw.get("strength_times", 0)),
        preservation_times=int(raw.get("preservation_times", 0)),
        full_power_times=int(raw.get("full_power_times", 0)),
        leisure_times=int(raw.get("leisure_times", 0)),
        stance_changed_by_direct_effect_times=int(
            raw.get("stance_changed_by_direct_effect_times", 0)
        ),
        created_card_count=int(raw.get("created_card_count", 0)),
        enthusiasm_bonus_buffs=tuple(
            TimedModifier(
                amount=float(modifier["amount"]),
                remaining_turns=(
                    int(modifier["remaining_turns"])
                    if modifier.get("remaining_turns") is not None
                    else None
                ),
                fresh=bool(modifier.get("fresh", True)),
            )
            for modifier in raw.get("enthusiasm_bonus_buffs", [])
        ),
        enthusiasm_multiplier_buffs=tuple(
            TimedModifier(
                amount=float(modifier["amount"]),
                remaining_turns=(
                    int(modifier["remaining_turns"])
                    if modifier.get("remaining_turns") is not None
                    else None
                ),
                fresh=bool(modifier.get("fresh", True)),
            )
            for modifier in raw.get("enthusiasm_multiplier_buffs", [])
        ),
        full_power_charge_buffs=tuple(
            TimedModifier(
                amount=float(modifier["amount"]),
                remaining_turns=(
                    int(modifier["remaining_turns"])
                    if modifier.get("remaining_turns") is not None
                    else None
                ),
                fresh=bool(modifier.get("fresh", True)),
            )
            for modifier in raw.get("full_power_charge_buffs", [])
        ),
        score_buffs=tuple(
            TimedModifier(
                amount=float(modifier["amount"]),
                remaining_turns=(
                    int(modifier["remaining_turns"])
                    if modifier.get("remaining_turns") is not None
                    else None
                ),
                fresh=bool(modifier.get("fresh", True)),
            )
            for modifier in raw.get("score_buffs", [])
        ),
        drinks=tuple(str(value) for value in raw.get("drinks", [])),
        items=tuple(str(value) for value in raw.get("items", [])),
        field_effects=tuple(str(value) for value in raw.get("field_effects", [])),
        future_score_multipliers=tuple(float(value) for value in raw.get("future_score_multipliers", [])),
    )


def _action(raw: dict[str, Any] | None) -> GameAction | None:
    if raw is None:
        return None
    return GameAction(
        kind=ActionKind(str(raw["kind"])),
        card_instance_id=str(raw["card_instance_id"]) if raw.get("card_instance_id") is not None else None,
        drink_id=str(raw["drink_id"]) if raw.get("drink_id") is not None else None,
        target_id=str(raw["target_id"]) if raw.get("target_id") is not None else None,
        source_slot=int(raw["source_slot"]) if raw.get("source_slot") is not None else None,
    )


def _case(raw: dict[str, Any], ruleset_id: str) -> GoldenCase:
    request_raw = raw["request"]
    state = _state(request_raw["state"])
    capabilities_raw = request_raw.get("capabilities", {})
    capabilities = CapabilityReport(
        observation_valid=bool(capabilities_raw.get("observation_valid", True)),
        game_legal=bool(capabilities_raw.get("game_legal", True)),
        model_supported=bool(capabilities_raw.get("model_supported", True)),
        policy_eligible=bool(capabilities_raw.get("policy_eligible", True)),
        execution_ready=bool(capabilities_raw.get("execution_ready", True)),
        reason_codes=tuple(str(value) for value in capabilities_raw.get("reason_codes", [])),
    )
    utility_raw = request_raw.get("utility", {})
    utility = UtilitySpec(
        risk_profile=RiskProfile(str(utility_raw.get("risk_profile", RiskProfile.ROBUST_HIGH_SCORE.value))),
        gamma=float(utility_raw.get("gamma", 1.0)),
        baseline_p10=(float(utility_raw["baseline_p10"]) if utility_raw.get("baseline_p10") is not None else None),
        minimum_stamina=int(utility_raw.get("minimum_stamina", 0)),
        minimum_genki=int(utility_raw.get("minimum_genki", 0)),
    )
    slots = request_raw.get("card_slots")
    if slots is None:
        slots = [[card.instance_id, index] for index, card in enumerate(state.deck.hand)]
    request = DecisionRequest(
        snapshot_id=str(request_raw.get("snapshot_id", raw["case_id"])),
        plan_type=PlanType(str(request_raw["plan_type"])),
        decision_kind=DecisionKind(str(request_raw.get("decision_kind", DecisionKind.CARD.value))),
        state=state,
        capabilities=capabilities,
        utility=utility,
        card_slots=tuple((str(instance_id), int(slot)) for instance_id, slot in slots),
        reobserve_attempt=int(request_raw.get("reobserve_attempt", 0)),
        ruleset_id=ruleset_id,
    )
    expected = raw["expected"]
    expectation = GoldenExpectation(
        outcome=DecisionOutcome(str(expected["outcome"])),
        action=_action(expected.get("action")),
        expected_final_score=(float(expected["expected_final_score"]) if expected.get("expected_final_score") is not None else None),
        p10_final_score=float(expected["p10_final_score"]) if expected.get("p10_final_score") is not None else None,
    )
    return GoldenCase(case_id=str(raw["case_id"]), request=request, expectation=expectation)


def load_golden_suite(path: str | Path) -> GoldenSuite:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported Task-275 golden fixture schema")
    ruleset_id = str(raw.get("ruleset_id", NIA_PRO_RULESET_ID))
    cards = {_raw["card_id"]: _card_definition(_raw) for _raw in raw.get("cards", [])}
    drinks = {_raw["drink_id"]: _drink_definition(_raw) for _raw in raw.get("drinks", [])}
    return GoldenSuite(
        dataset_id=str(raw["dataset_id"]),
        ruleset_id=ruleset_id,
        cards=cards,
        drinks=drinks,
        cases=tuple(_case(case, ruleset_id) for case in raw.get("cases", [])),
    )
