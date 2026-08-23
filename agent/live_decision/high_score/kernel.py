from __future__ import annotations

import math
from enum import Enum
from typing import Mapping
from dataclasses import replace, dataclass

from .contracts import (
    NIA_PRO_RULESET_ID,
    Stance,
    CardRef,
    CardZone,
    PlanType,
    ActionKind,
    DeckBelief,
    GameAction,
    TimedStatus,
    DecisionKind,
    DeckKnowledge,
    PendingChoice,
    SchedulePhase,
    StrategyState,
    TimedModifier,
    EffectSourceKind,
    ScheduledEffectRef,
    TurnResolutionStage,
    EffectTriggerContext,
    HighScoreContractError,
    CardEffectsContinuation,
    FreePlayAllContinuation,
    TurnResolutionContinuation,
    ScheduledEffectsContinuation,
)


class KernelError(RuntimeError):
    """Raised when a transition is illegal or unsupported by the selected ruleset."""


FULL_POWER_ACTIVATION_THRESHOLD = 10
IMMEDIATE_SCHEDULE_PHASES = {
    SchedulePhase.BEFORE_CARD_USED,
    SchedulePhase.CARD_USED,
    SchedulePhase.AFTER_CARD_USED,
    SchedulePhase.END_OF_TURN,
    SchedulePhase.FULL_POWER_CHARGE_INCREASED,
    SchedulePhase.CARD_MOVED_TO_HAND,
    SchedulePhase.CARD_MOVED_TO_HELD,
    SchedulePhase.STANCE_CHANGED,
}


class EffectKind(str, Enum):
    SCORE = "SCORE"
    STAMINA = "STAMINA"
    GENKI = "GENKI"
    FULL_POWER = "FULL_POWER"
    PASSION = "PASSION"
    PASSION_BUFF = "PASSION_BUFF"
    PASSION_GAIN_BONUS_PCT = "PASSION_GAIN_BONUS_PCT"
    PARAMETER_GAIN_BONUS_PCT = "PARAMETER_GAIN_BONUS_PCT"
    STANCE = "STANCE"
    EXTRA_ACTION = "EXTRA_ACTION"
    DRAW = "DRAW"
    CONCENTRATION = "CONCENTRATION"
    GOOD_CONDITION = "GOOD_CONDITION"
    EXCELLENT_CONDITION = "EXCELLENT_CONDITION"
    GOOD_IMPRESSION = "GOOD_IMPRESSION"
    MOTIVATION = "MOTIVATION"
    STATUS = "STATUS"
    TURNS_REMAINING = "TURNS_REMAINING"
    HOLD_SELF = "HOLD_SELF"
    HOLD_SELECTED = "HOLD_SELECTED"
    CARD_MODIFICATION = "CARD_MODIFICATION"
    SCHEDULE = "SCHEDULE"
    MOVE_RANDOM_TO_HAND = "MOVE_RANDOM_TO_HAND"
    EXCHANGE_HAND = "EXCHANGE_HAND"
    DOUBLE_CARD_EFFECT = "DOUBLE_CARD_EFFECT"
    FREE_PLAY_SELECTED = "FREE_PLAY_SELECTED"
    HALF_COST_TURNS = "HALF_COST_TURNS"
    DOUBLE_COST_TURNS = "DOUBLE_COST_TURNS"
    ADD_CARD_TO_DECK = "ADD_CARD_TO_DECK"
    ENTHUSIASM_BONUS_BUFF = "ENTHUSIASM_BONUS_BUFF"
    ENTHUSIASM_MULTIPLIER_BUFF = "ENTHUSIASM_MULTIPLIER_BUFF"
    FULL_POWER_CHARGE_BUFF = "FULL_POWER_CHARGE_BUFF"
    SCORE_BUFF = "SCORE_BUFF"
    STANCE_LOCK_TURNS = "STANCE_LOCK_TURNS"
    NULLIFY_COST_CARDS = "NULLIFY_COST_CARDS"
    UPGRADE_HAND = "UPGRADE_HAND"
    FIXED_STAMINA_TO_ZERO = "FIXED_STAMINA_TO_ZERO"
    FREE_PLAY_ALL = "FREE_PLAY_ALL"
    PAID_PLAY_SELECTED = "PAID_PLAY_SELECTED"
    FREE_PLAY_RANDOM = "FREE_PLAY_RANDOM"
    MOVE_SELECTED_TO_HAND = "MOVE_SELECTED_TO_HAND"


class EffectValueSource(str, Enum):
    GOOD_CONDITION = "GOOD_CONDITION"
    GOOD_IMPRESSION = "GOOD_IMPRESSION"
    MOTIVATION = "MOTIVATION"
    GENKI = "GENKI"


class CardCostKind(str, Enum):
    NORMAL = "NORMAL"
    STAMINA = "STAMINA"
    FULL_POWER = "FULL_POWER"
    CONCENTRATION = "CONCENTRATION"
    GOOD_CONDITION = "GOOD_CONDITION"
    GOOD_IMPRESSION = "GOOD_IMPRESSION"
    MOTIVATION = "MOTIVATION"


class ConditionKind(str, Enum):
    ALL = "ALL"
    STANCE_FAMILY = "STANCE_FAMILY"
    STANCE_NOT = "STANCE_NOT"
    STANCE_IS = "STANCE_IS"
    PREVIOUS_STANCE_IS = "PREVIOUS_STANCE_IS"
    FULL_POWER_CHARGE_MIN = "FULL_POWER_CHARGE_MIN"
    FULL_POWER_CHARGE_MAX = "FULL_POWER_CHARGE_MAX"
    CUMULATIVE_FULL_POWER_MIN = "CUMULATIVE_FULL_POWER_MIN"
    TURNS_REMAINING_MAX = "TURNS_REMAINING_MAX"
    IS_DIRECT_EFFECT = "IS_DIRECT_EFFECT"
    USED_CARD_EFFECT_TAG = "USED_CARD_EFFECT_TAG"
    USED_CARD_BASE_ID = "USED_CARD_BASE_ID"
    USED_CARD_RARITY_NOT = "USED_CARD_RARITY_NOT"
    MOVED_CARD_ID = "MOVED_CARD_ID"
    MOVED_CARD_BASE_ID = "MOVED_CARD_BASE_ID"
    REMOVED_BASE_CARD_COUNT_MIN = "REMOVED_BASE_CARD_COUNT_MIN"
    STANCE_CHANGE_COUNT_MOD_EQ = "STANCE_CHANGE_COUNT_MOD_EQ"
    STANCE_CHANGE_COUNT_MIN = "STANCE_CHANGE_COUNT_MIN"
    STRENGTH_COUNT_MIN = "STRENGTH_COUNT_MIN"
    PRESERVATION_COUNT_MIN = "PRESERVATION_COUNT_MIN"
    FULL_POWER_COUNT_MIN = "FULL_POWER_COUNT_MIN"
    TROUBLE_CARD_COUNT_MIN = "TROUBLE_CARD_COUNT_MIN"
    CONCENTRATION_MIN = "CONCENTRATION_MIN"
    GOOD_CONDITION_MIN = "GOOD_CONDITION_MIN"
    EXCELLENT_CONDITION_MIN = "EXCELLENT_CONDITION_MIN"
    GOOD_IMPRESSION_MIN = "GOOD_IMPRESSION_MIN"
    MOTIVATION_MIN = "MOTIVATION_MIN"
    GENKI_MIN = "GENKI_MIN"


@dataclass(frozen=True)
class EffectCondition:
    kind: ConditionKind
    value: str | None = None
    threshold: int | None = None
    all_of: tuple[EffectCondition, ...] = ()
    modulus: int | None = None
    remainder: int | None = None

    def validate(self) -> None:
        if self.kind is ConditionKind.ALL:
            if (
                len(self.all_of) < 2
                or self.value is not None
                or self.threshold is not None
                or self.modulus is not None
                or self.remainder is not None
            ):
                raise HighScoreContractError("ALL condition requires at least two child conditions")
            for condition in self.all_of:
                condition.validate()
            return
        if self.all_of:
            raise HighScoreContractError("condition children are only valid for ALL")
        if self.kind is ConditionKind.IS_DIRECT_EFFECT:
            if any(
                value is not None
                for value in (self.value, self.threshold, self.modulus, self.remainder)
            ):
                raise HighScoreContractError("IS_DIRECT_EFFECT does not accept payload")
            return
        if self.kind is ConditionKind.STANCE_CHANGE_COUNT_MOD_EQ:
            if (
                self.modulus is None
                or self.modulus < 1
                or self.remainder is None
                or self.remainder < 0
                or self.remainder >= self.modulus
                or self.value is not None
                or self.threshold is not None
            ):
                raise HighScoreContractError("stance-change modulo condition is invalid")
            return
        if self.kind is ConditionKind.REMOVED_BASE_CARD_COUNT_MIN:
            if not self.value or self.threshold is None or self.threshold < 0:
                raise HighScoreContractError("removed base-card count condition is invalid")
            return
        if self.kind in {
            ConditionKind.STANCE_FAMILY,
            ConditionKind.STANCE_NOT,
            ConditionKind.STANCE_IS,
            ConditionKind.PREVIOUS_STANCE_IS,
            ConditionKind.USED_CARD_EFFECT_TAG,
            ConditionKind.USED_CARD_BASE_ID,
            ConditionKind.USED_CARD_RARITY_NOT,
            ConditionKind.MOVED_CARD_ID,
            ConditionKind.MOVED_CARD_BASE_ID,
        } and not self.value:
            raise HighScoreContractError(f"{self.kind.value} requires value")
        if self.kind not in {
            ConditionKind.STANCE_FAMILY,
            ConditionKind.STANCE_NOT,
            ConditionKind.STANCE_IS,
            ConditionKind.PREVIOUS_STANCE_IS,
            ConditionKind.USED_CARD_EFFECT_TAG,
            ConditionKind.USED_CARD_BASE_ID,
            ConditionKind.USED_CARD_RARITY_NOT,
            ConditionKind.MOVED_CARD_ID,
            ConditionKind.MOVED_CARD_BASE_ID,
        } and (
            self.threshold is None or self.threshold < 0
        ):
            raise HighScoreContractError(f"{self.kind.value} requires non-negative threshold")


@dataclass(frozen=True)
class CardSelector:
    all_cards: bool = False
    source_only: bool = False
    zones: tuple[CardZone, ...] = ()
    card_type: str | None = None
    card_id: str | None = None
    base_card_id: str | None = None
    effect_tag: str | None = None
    source_type: str | None = None

    def validate(self) -> None:
        if not any(
            (
                self.all_cards,
                self.source_only,
                self.zones,
                self.card_type,
                self.card_id,
                self.base_card_id,
                self.effect_tag,
                self.source_type,
            )
        ):
            raise HighScoreContractError("card selector cannot be empty")
        if (self.all_cards or self.source_only) and any(
            (
                self.all_cards and self.source_only,
                self.zones,
                self.card_type,
                self.card_id,
                self.base_card_id,
                self.effect_tag,
                self.source_type,
            )
        ):
            raise HighScoreContractError("special card selector cannot be combined with filters")
        if self.card_id is not None and self.base_card_id is not None:
            raise HighScoreContractError("card selector cannot combine exact and base ids")
        if self.card_type is not None and self.card_type not in {"active", "mental"}:
            raise HighScoreContractError("card selector type is unsupported")
        if self.effect_tag is not None and self.effect_tag not in {"strength", "fullPowerCharge"}:
            raise HighScoreContractError("card selector effect tag is unsupported")
        if self.source_type is not None and self.source_type not in {
            "default",
            "pIdol",
            "produce",
            "support",
        }:
            raise HighScoreContractError("card selector source type is unsupported")


@dataclass(frozen=True)
class CardModification:
    score_bonus: int = 0
    genki_bonus: int = 0
    stamina_cost_delta: int = 0
    typed_cost_delta: int = 0
    score_repeat_bonus: int = 0

    def validate(self) -> None:
        if not any(
            (
                self.score_bonus,
                self.genki_bonus,
                self.stamina_cost_delta,
                self.typed_cost_delta,
                self.score_repeat_bonus,
            )
        ):
            raise HighScoreContractError("card modification cannot be empty")
        if self.score_repeat_bonus < 0:
            raise HighScoreContractError("card modification repeat bonus cannot be negative")


@dataclass(frozen=True)
class EffectSpec:
    kind: EffectKind
    amount: float = 0.0
    repeats: int = 1
    stance: Stance | None = None
    status_id: str | None = None
    duration: int | None = None
    full_power_scale: float = 0.0
    condition: EffectCondition | None = None
    target_zones: tuple[CardZone, ...] = ()
    card_selector: CardSelector | None = None
    card_modification: CardModification | None = None
    scheduled_effects: tuple[EffectSpec, ...] = ()
    delay_turns: int = 1
    schedule_phase: SchedulePhase = SchedulePhase.TURN
    schedule_limit: int | None = 1
    schedule_ttl: int | None = None
    trigger_card_type: str | None = None
    selection_count: int = 1
    selection_optional: bool = False
    card_id: str | None = None
    score_enthusiasm_multiplier: float = 1.0
    value_source: EffectValueSource | None = None
    value_scale: float = 0.0
    score_concentration_multiplier: float = 1.0
    score_good_condition_multiplier: float = 1.0
    genki_motivation_multiplier: float = 1.0

    def validate(self) -> None:
        if self.repeats < 1:
            raise HighScoreContractError("effect repeats must be positive")
        if self.kind is EffectKind.STANCE and self.stance is None:
            raise HighScoreContractError("STANCE effect requires stance")
        if self.kind is EffectKind.STATUS and not self.status_id:
            raise HighScoreContractError("STATUS effect requires status_id")
        if self.kind in {
            EffectKind.HOLD_SELECTED,
            EffectKind.FREE_PLAY_SELECTED,
            EffectKind.MOVE_SELECTED_TO_HAND,
        } and not self.target_zones:
            raise HighScoreContractError(f"{self.kind.value} requires target_zones")
        if self.kind in {EffectKind.FREE_PLAY_ALL, EffectKind.PAID_PLAY_SELECTED} and not self.target_zones:
            raise HighScoreContractError(f"{self.kind.value} requires target_zones")
        if self.kind is EffectKind.FREE_PLAY_RANDOM:
            if self.card_selector is None:
                raise HighScoreContractError("FREE_PLAY_RANDOM requires a selector")
            self.card_selector.validate()
        if self.selection_count < 1:
            raise HighScoreContractError("selection count must be positive")
        if self.kind is not EffectKind.HOLD_SELECTED and (
            self.selection_count != 1 or self.selection_optional
        ):
            raise HighScoreContractError("multi-select options are only valid for HOLD_SELECTED")
        if self.kind is EffectKind.DOUBLE_CARD_EFFECT and (
            self.amount < 1 or not float(self.amount).is_integer()
        ):
            raise HighScoreContractError("DOUBLE_CARD_EFFECT requires a positive integer amount")
        if self.kind in {EffectKind.HALF_COST_TURNS, EffectKind.DOUBLE_COST_TURNS} and (
            self.amount < 1 or not float(self.amount).is_integer()
        ):
            raise HighScoreContractError(f"{self.kind.value} requires a positive integer amount")
        if self.kind is EffectKind.ADD_CARD_TO_DECK and not self.card_id:
            raise HighScoreContractError("ADD_CARD_TO_DECK requires card_id")
        if self.kind is not EffectKind.ADD_CARD_TO_DECK and self.card_id is not None:
            raise HighScoreContractError("card_id payload is only valid for ADD_CARD_TO_DECK")
        if self.score_enthusiasm_multiplier <= 0 or (
            self.kind is not EffectKind.SCORE
            and self.score_enthusiasm_multiplier != 1.0
        ):
            raise HighScoreContractError("score enthusiasm multiplier is invalid")
        if (self.value_source is None) != (self.value_scale == 0):
            raise HighScoreContractError("effect value source and scale must be declared together")
        if self.value_source is not None and self.kind not in {
            EffectKind.SCORE,
            EffectKind.CONCENTRATION,
            EffectKind.GOOD_CONDITION,
            EffectKind.EXCELLENT_CONDITION,
            EffectKind.GOOD_IMPRESSION,
            EffectKind.MOTIVATION,
            EffectKind.GENKI,
            EffectKind.STAMINA,
        }:
            raise HighScoreContractError("effect value source is invalid for this effect kind")
        if self.score_concentration_multiplier <= 0 or (
            self.kind is not EffectKind.SCORE
            and self.score_concentration_multiplier != 1.0
        ):
            raise HighScoreContractError("score concentration multiplier is invalid")
        if self.score_good_condition_multiplier <= 0 or (
            self.kind is not EffectKind.SCORE
            and self.score_good_condition_multiplier != 1.0
        ):
            raise HighScoreContractError("score good-condition multiplier is invalid")
        if self.genki_motivation_multiplier <= 0 or (
            self.kind is not EffectKind.GENKI
            and self.genki_motivation_multiplier != 1.0
        ):
            raise HighScoreContractError("genki motivation multiplier is invalid")
        if self.kind in {
            EffectKind.ENTHUSIASM_BONUS_BUFF,
            EffectKind.ENTHUSIASM_MULTIPLIER_BUFF,
            EffectKind.FULL_POWER_CHARGE_BUFF,
            EffectKind.SCORE_BUFF,
        } and (self.amount < 0 or (self.duration is not None and self.duration < 1)):
            raise HighScoreContractError("buff amount and duration are invalid")
        if self.kind in {EffectKind.STANCE_LOCK_TURNS, EffectKind.NULLIFY_COST_CARDS} and (
            self.amount < 1 or not float(self.amount).is_integer()
        ):
            raise HighScoreContractError(f"{self.kind.value} requires a positive integer amount")
        if self.kind is EffectKind.CARD_MODIFICATION:
            if self.card_selector is None or self.card_modification is None:
                raise HighScoreContractError("CARD_MODIFICATION requires selector and modification")
            self.card_selector.validate()
            self.card_modification.validate()
        if self.kind is EffectKind.MOVE_RANDOM_TO_HAND:
            if self.card_selector is None or not self.card_selector.zones:
                raise HighScoreContractError("MOVE_RANDOM_TO_HAND requires an explicitly zoned selector")
            if self.card_selector.all_cards or CardZone.HAND in self.card_selector.zones:
                raise HighScoreContractError("MOVE_RANDOM_TO_HAND source zones cannot include the hand")
            self.card_selector.validate()
        if self.kind is EffectKind.SCHEDULE:
            minimum_delay = (
                0
                if self.schedule_phase
                in {
                    SchedulePhase.CARD_USED,
                    SchedulePhase.AFTER_CARD_USED,
                    SchedulePhase.END_OF_TURN,
                    SchedulePhase.BEFORE_CARD_USED,
                    SchedulePhase.FULL_POWER_CHARGE_INCREASED,
                    SchedulePhase.CARD_MOVED_TO_HAND,
                    SchedulePhase.CARD_MOVED_TO_HELD,
                    SchedulePhase.STANCE_CHANGED,
                }
                else 1
            )
            if not self.scheduled_effects or self.delay_turns < minimum_delay:
                raise HighScoreContractError("SCHEDULE requires nested effects and a valid delay")
            if not isinstance(self.schedule_phase, SchedulePhase):
                raise HighScoreContractError("SCHEDULE requires a valid trigger phase")
            if self.schedule_limit is not None and self.schedule_limit < 1:
                raise HighScoreContractError("SCHEDULE limit must be positive")
            if self.schedule_ttl is not None and self.schedule_ttl < 1:
                raise HighScoreContractError("SCHEDULE ttl must be positive")
            if self.trigger_card_type not in {None, "active", "mental"}:
                raise HighScoreContractError("SCHEDULE card-type filter is invalid")
            if any(effect.kind is EffectKind.SCHEDULE for effect in self.scheduled_effects):
                raise HighScoreContractError("nested schedules are unsupported")
            for scheduled in self.scheduled_effects:
                scheduled.validate()
        elif (
            self.scheduled_effects
            or self.delay_turns != 1
            or self.schedule_phase is not SchedulePhase.TURN
            or self.schedule_limit != 1
            or self.schedule_ttl is not None
            or self.trigger_card_type is not None
        ):
            raise HighScoreContractError("schedule payload is only valid for SCHEDULE effects")
        if self.condition is not None:
            self.condition.validate()


@dataclass(frozen=True)
class CardDefinition:
    card_id: str
    name: str
    plan: PlanType | None
    stamina_cost: int = 0
    full_power_cost: int = 0
    cost_kind: CardCostKind = CardCostKind.NORMAL
    effects: tuple[EffectSpec, ...] = ()
    persistent_effects: tuple[EffectSpec, ...] = ()
    limited: bool = False
    required_stance: Stance | None = None
    condition: EffectCondition | None = None
    card_type: str | None = None
    rarity: str | None = None
    base_card_id: str | None = None
    upgrade_card_id: str | None = None
    source_type: str | None = None

    def validate(self) -> None:
        if not self.card_id or not self.name:
            raise HighScoreContractError("card definition requires card_id and name")
        if self.stamina_cost < 0 or self.full_power_cost < 0:
            raise HighScoreContractError("card costs cannot be negative")
        if not isinstance(self.cost_kind, CardCostKind):
            raise HighScoreContractError("card cost kind is invalid")
        if (
            self.cost_kind is CardCostKind.FULL_POWER
            and self.full_power_cost <= 0
        ):
            raise HighScoreContractError("full-power cost kind requires a full-power cost")
        if (
            self.cost_kind not in {CardCostKind.NORMAL, CardCostKind.FULL_POWER}
            and self.full_power_cost > 0
        ):
            raise HighScoreContractError("non-full-power cost kind cannot spend full power")
        if self.card_type is not None and self.card_type not in {"active", "mental", "trouble"}:
            raise HighScoreContractError("card type is unsupported")
        if self.base_card_id is not None and not self.base_card_id:
            raise HighScoreContractError("base card id cannot be empty")
        if self.upgrade_card_id == self.card_id:
            raise HighScoreContractError("card cannot upgrade to itself")
        if self.source_type is not None and self.source_type not in {
            "default",
            "pIdol",
            "produce",
            "support",
        }:
            raise HighScoreContractError("card source type is unsupported")
        for effect in self.effects:
            effect.validate()
        for effect in self.persistent_effects:
            if effect.kind is not EffectKind.SCHEDULE:
                raise HighScoreContractError("persistent card effects must be scheduled")
            effect.validate()
        if self.condition is not None:
            self.condition.validate()


@dataclass(frozen=True)
class DrinkDefinition:
    drink_id: str
    name: str
    effects: tuple[EffectSpec, ...] = ()
    stamina_cost: int = 0
    plan: PlanType | None = None

    def validate(self) -> None:
        if not self.drink_id or not self.name or self.stamina_cost < 0:
            raise HighScoreContractError("invalid drink definition")
        if self.plan is not None and not isinstance(self.plan, PlanType):
            raise HighScoreContractError("drink plan is invalid")
        for effect in self.effects:
            effect.validate()


@dataclass(frozen=True)
class ItemDefinition:
    item_id: str
    name: str
    persistent_effects: tuple[EffectSpec, ...] = ()
    plan: PlanType | None = None

    def validate(self) -> None:
        if not self.item_id or not self.name:
            raise HighScoreContractError("invalid item definition")
        if self.plan is not None and not isinstance(self.plan, PlanType):
            raise HighScoreContractError("item plan is invalid")
        for effect in self.persistent_effects:
            if effect.kind is not EffectKind.SCHEDULE:
                raise HighScoreContractError("item effects must be scheduled")
            effect.validate()


@dataclass(frozen=True)
class RuleEvent:
    event_type: str
    source_id: str
    amount: float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class TransitionBranch:
    probability: float
    state: StrategyState
    events: tuple[RuleEvent, ...]


def _stance_parameter_multiplier(stance: Stance) -> float:
    if stance is Stance.AGGRESSIVE_1:
        return 2.0
    if stance is Stance.AGGRESSIVE_2:
        return 2.5
    if stance is Stance.CONSERVE_1:
        return 0.5
    if stance is Stance.CONSERVE_2:
        return 0.25
    if stance is Stance.FULL_POWER:
        return 3.0
    if stance is Stance.LEISURE:
        return 0.0
    return 1.0


def _conserve_level(stance: Stance) -> int:
    if stance is Stance.CONSERVE_2:
        return 2
    if stance is Stance.CONSERVE_1:
        return 1
    return 0


def _same_stance_family(current: Stance, target: Stance) -> bool:
    conserve = {Stance.CONSERVE_1, Stance.CONSERVE_2}
    aggressive = {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}
    return (current in conserve and target in conserve) or (current in aggressive and target in aggressive)


def _stack_stance(current: Stance, target: Stance) -> Stance:
    if not _same_stance_family(current, target):
        return target
    if target in {Stance.CONSERVE_1, Stance.CONSERVE_2}:
        return Stance.CONSERVE_2
    if target in {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}:
        return Stance.AGGRESSIVE_2
    return target


def _is_counted_stance_change(old_stance: Stance, target: Stance) -> bool:
    family_upgrade = (
        old_stance is Stance.AGGRESSIVE_1 and target is Stance.AGGRESSIVE_2
    ) or (
        old_stance is Stance.CONSERVE_1 and target is Stance.CONSERVE_2
    )
    return target is not old_stance and not family_upgrade


def _definition_has_effect_tag(definition: CardDefinition, effect_tag: str) -> bool:
    if effect_tag == "strength":
        return any(
            effect.kind is EffectKind.STANCE
            and effect.stance in {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}
            for effect in definition.effects
        )
    if effect_tag == "fullPowerCharge":
        return any(
            effect.kind is EffectKind.FULL_POWER and effect.amount > 0
            for effect in definition.effects
        )
    return False


def _condition_matches(
    state: StrategyState,
    condition: EffectCondition | None,
    cards: Mapping[str, CardDefinition],
    trigger_context: EffectTriggerContext | None = None,
) -> bool:
    if condition is None:
        return True
    if condition.kind is ConditionKind.ALL:
        return all(
            _condition_matches(state, nested, cards, trigger_context)
            for nested in condition.all_of
        )
    if condition.kind is ConditionKind.STANCE_FAMILY:
        if condition.value == "strength":
            return state.stance in {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}
        if condition.value == "preservation":
            return state.stance in {
                Stance.CONSERVE_1,
                Stance.CONSERVE_2,
                Stance.LEISURE,
            }
        if condition.value == "fullPower":
            return state.stance is Stance.FULL_POWER
        if condition.value == "not_none":
            return state.stance is not Stance.NEUTRAL
        return False
    if condition.kind is ConditionKind.STANCE_NOT:
        if condition.value == "preservation":
            return state.stance not in {
                Stance.CONSERVE_1,
                Stance.CONSERVE_2,
                Stance.LEISURE,
            }
        if condition.value == "fullPower":
            return state.stance is not Stance.FULL_POWER
        if condition.value == "strength":
            return state.stance not in {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}
        return False
    if condition.kind is ConditionKind.STANCE_IS:
        return state.stance.value == condition.value
    if condition.kind is ConditionKind.PREVIOUS_STANCE_IS:
        return state.previous_stance.value == condition.value
    if condition.kind is ConditionKind.FULL_POWER_CHARGE_MIN:
        assert condition.threshold is not None
        return state.full_power >= condition.threshold
    if condition.kind is ConditionKind.FULL_POWER_CHARGE_MAX:
        assert condition.threshold is not None
        return state.full_power <= condition.threshold
    if condition.kind is ConditionKind.CUMULATIVE_FULL_POWER_MIN:
        assert condition.threshold is not None
        return state.cumulative_full_power >= condition.threshold
    if condition.kind is ConditionKind.TURNS_REMAINING_MAX:
        assert condition.threshold is not None
        return state.turns_remaining <= condition.threshold
    if condition.kind is ConditionKind.IS_DIRECT_EFFECT:
        return trigger_context is not None and trigger_context.is_direct_effect
    if condition.kind is ConditionKind.USED_CARD_EFFECT_TAG:
        if trigger_context is None or trigger_context.used_card_id is None:
            return False
        definition = cards.get(trigger_context.used_card_id)
        return definition is not None and _definition_has_effect_tag(
            definition,
            str(condition.value),
        )
    if condition.kind is ConditionKind.USED_CARD_BASE_ID:
        return (
            trigger_context is not None
            and trigger_context.used_card_base_id == condition.value
        )
    if condition.kind is ConditionKind.USED_CARD_RARITY_NOT:
        return (
            trigger_context is not None
            and trigger_context.used_card_rarity is not None
            and trigger_context.used_card_rarity != condition.value
        )
    if condition.kind is ConditionKind.MOVED_CARD_ID:
        return (
            trigger_context is not None
            and trigger_context.moved_card_id == condition.value
        )
    if condition.kind is ConditionKind.MOVED_CARD_BASE_ID:
        return (
            trigger_context is not None
            and trigger_context.moved_card_base_id == condition.value
        )
    if condition.kind is ConditionKind.REMOVED_BASE_CARD_COUNT_MIN:
        assert condition.threshold is not None
        return (
            sum(
                (cards[card.card_id].base_card_id or card.card_id)
                == condition.value
                for card in state.deck.removed
            )
            >= condition.threshold
        )
    if condition.kind is ConditionKind.STANCE_CHANGE_COUNT_MOD_EQ:
        assert condition.modulus is not None
        assert condition.remainder is not None
        return (
            state.stance_changed_by_direct_effect_times % condition.modulus
            == condition.remainder
        )
    if condition.kind is ConditionKind.STANCE_CHANGE_COUNT_MIN:
        assert condition.threshold is not None
        return (
            state.strength_times + state.preservation_times + state.full_power_times
            >= condition.threshold
        )
    if condition.kind is ConditionKind.STRENGTH_COUNT_MIN:
        assert condition.threshold is not None
        return state.strength_times >= condition.threshold
    if condition.kind is ConditionKind.PRESERVATION_COUNT_MIN:
        assert condition.threshold is not None
        return state.preservation_times >= condition.threshold
    if condition.kind is ConditionKind.FULL_POWER_COUNT_MIN:
        assert condition.threshold is not None
        return state.full_power_times >= condition.threshold
    if condition.kind is ConditionKind.TROUBLE_CARD_COUNT_MIN:
        assert condition.threshold is not None
        active_cards = state.deck.hand + state.deck.draw_pile + state.deck.discard + state.deck.held
        return (
            sum(cards[card.card_id].card_type == "trouble" for card in active_cards)
            >= condition.threshold
        )
    if condition.kind is ConditionKind.CONCENTRATION_MIN:
        assert condition.threshold is not None
        return state.concentration >= condition.threshold
    if condition.kind is ConditionKind.GOOD_CONDITION_MIN:
        assert condition.threshold is not None
        return state.good_condition_turns >= condition.threshold
    if condition.kind is ConditionKind.EXCELLENT_CONDITION_MIN:
        assert condition.threshold is not None
        return state.excellent_condition_turns >= condition.threshold
    if condition.kind is ConditionKind.GOOD_IMPRESSION_MIN:
        assert condition.threshold is not None
        return state.good_impression_turns >= condition.threshold
    if condition.kind is ConditionKind.MOTIVATION_MIN:
        assert condition.threshold is not None
        return state.motivation >= condition.threshold
    if condition.kind is ConditionKind.GENKI_MIN:
        assert condition.threshold is not None
        return state.genki >= condition.threshold
    return False


class PythonNiaProKernel:
    """Deterministic Task-275 probe kernel with explicit chance branches.

    This kernel is authoritative only for the frozen Task-275 probe ruleset. It does
    not claim unverified game semantics outside the registered card/effect set.
    """

    kernel_name = "python-nia-pro"
    kernel_version = "1.23"

    def __init__(
        self,
        cards: Mapping[str, CardDefinition],
        drinks: Mapping[str, DrinkDefinition] | None = None,
        items: Mapping[str, ItemDefinition] | None = None,
        *,
        ruleset_id: str = NIA_PRO_RULESET_ID,
    ):
        self.cards = dict(cards)
        self.drinks = dict(drinks or {})
        self.items = dict(items or {})
        self.ruleset_id = ruleset_id
        for definition in self.cards.values():
            definition.validate()
        for definition in self.drinks.values():
            definition.validate()
        for definition in self.items.values():
            definition.validate()

    def legal_actions(
        self,
        state: StrategyState,
        *,
        plan_type: PlanType,
        decision_kind: DecisionKind,
    ) -> tuple[GameAction, ...]:
        state.validate()
        self._validate_registered_state(state, plan_type=plan_type)
        state = self._materialize_persistent_effects(state)
        state.validate()
        if state.deck.knowledge in {DeckKnowledge.UNKNOWN, DeckKnowledge.BROKEN}:
            raise KernelError(f"unsupported_deck_knowledge:{state.deck.knowledge.value}")
        if state.terminal:
            return ()
        if state.pending_choice is not None:
            if decision_kind is not DecisionKind.SECONDARY:
                return ()
            candidate_ids = state.pending_choice.candidate_instance_ids
            if state.pending_choice.action_kind is ActionKind.FREE_PLAY:
                candidate_ids = tuple(
                    instance_id
                    for instance_id in candidate_ids
                    if self._free_play_candidate_ready(state, instance_id)
                )
            actions = tuple(
                GameAction(kind=state.pending_choice.action_kind, card_instance_id=instance_id)
                for instance_id in candidate_ids
            )
            if state.pending_choice.selection_optional:
                actions += (GameAction(kind=ActionKind.FINISH_SELECTION),)
            return actions
        if decision_kind is DecisionKind.SECONDARY:
            return ()
        actions: list[GameAction] = []
        if decision_kind is DecisionKind.CARD:
            if state.actions_remaining > 0:
                for card in state.deck.hand:
                    definition = self.cards.get(card.card_id)
                    if definition is None or (definition.plan is not None and definition.plan is not plan_type):
                        continue
                    if definition.required_stance is not None and definition.required_stance is not state.stance:
                        continue
                    if not _condition_matches(state, definition.condition, self.cards):
                        continue
                    if not self._non_full_power_cost_payable(
                        state,
                        card,
                        definition,
                    ):
                        continue
                    if self._effective_full_power_cost(card, definition) > state.full_power:
                        continue
                    if not self._card_secondary_ready(state, definition, card.instance_id):
                        continue
                    actions.append(GameAction(kind=ActionKind.PLAY_CARD, card_instance_id=card.instance_id))
            actions.append(GameAction(kind=ActionKind.END_TURN))
        elif decision_kind is DecisionKind.DRINK:
            for drink_id in state.drinks:
                definition = self.drinks.get(drink_id)
                if (
                    definition is not None
                    and definition.plan in {None, plan_type}
                    and definition.stamina_cost <= state.stamina + state.genki
                    and self._drink_secondary_ready(state, definition)
                ):
                    actions.append(GameAction(kind=ActionKind.USE_DRINK, drink_id=drink_id))
        elif decision_kind is DecisionKind.END_TURN:
            actions.append(GameAction(kind=ActionKind.END_TURN))
        return tuple(actions)

    def _card_secondary_ready(self, state: StrategyState, definition: CardDefinition, source_id: str) -> bool:
        repeats_effects = self._card_effect_passes(state, definition) > 1
        repeated_hold_has_zone_interference = repeats_effects and any(
            effect.kind is EffectKind.HOLD_SELECTED for effect in definition.effects
        ) and any(
            effect.kind
            in {
                EffectKind.DRAW,
                EffectKind.HOLD_SELF,
                EffectKind.MOVE_RANDOM_TO_HAND,
                EffectKind.EXCHANGE_HAND,
            }
            for effect in definition.effects
        )
        if repeated_hold_has_zone_interference:
            return False
        for effect in definition.effects:
            if not _condition_matches(state, effect.condition, self.cards):
                continue
            if effect.kind is EffectKind.FREE_PLAY_SELECTED and repeats_effects:
                return False
            if effect.kind in {
                EffectKind.HOLD_SELECTED,
                EffectKind.MOVE_SELECTED_TO_HAND,
            }:
                candidates = {
                    card.instance_id
                    for zone in effect.target_zones
                    for card in self._zone_cards(state, zone)
                    if card.instance_id != source_id
                }
                required = effect.selection_count * (2 if repeats_effects else 1)
                if not effect.selection_optional and len(candidates) < required:
                    return False
        return True

    def _drink_secondary_ready(
        self,
        state: StrategyState,
        definition: DrinkDefinition,
    ) -> bool:
        for effect in definition.effects:
            if not _condition_matches(state, effect.condition, self.cards):
                continue
            if effect.kind not in {
                EffectKind.HOLD_SELECTED,
                EffectKind.MOVE_SELECTED_TO_HAND,
            }:
                continue
            candidates = {
                card.instance_id
                for zone in effect.target_zones
                for card in self._zone_cards(state, zone)
            }
            if len(candidates) < effect.selection_count:
                return False
        return True

    def _free_play_candidate_ready(self, state: StrategyState, instance_id: str) -> bool:
        card = self._card_by_instance(state, instance_id)
        if card is None:
            return False
        definition = self.cards.get(card.card_id)
        if definition is None:
            return False
        condition_matches = (
            definition.required_stance in {None, state.stance}
            and _condition_matches(state, definition.condition, self.cards)
        )
        return not condition_matches or self._card_secondary_ready(state, definition, card.instance_id)

    def transition(self, state: StrategyState, action: GameAction, *, plan_type: PlanType) -> tuple[TransitionBranch, ...]:
        state.validate()
        self._validate_registered_state(state, plan_type=plan_type)
        state = self._materialize_persistent_effects(state)
        state.validate()
        action.validate()
        if action.kind is ActionKind.PLAY_CARD:
            return self._play_card(state, action, plan_type=plan_type)
        if action.kind is ActionKind.USE_DRINK:
            return self._use_drink(state, action, plan_type=plan_type)
        if action.kind is ActionKind.END_TURN:
            return self._end_turn(state, plan_type=plan_type)
        if action.kind is ActionKind.HOLD_CARD:
            return self._hold_card(state, action)
        if action.kind is ActionKind.MOVE_CARD:
            return self._move_card(state, action)
        if action.kind is ActionKind.FREE_PLAY:
            return self._free_play_card(state, action, plan_type=plan_type)
        if action.kind is ActionKind.PAID_PLAY:
            return self._paid_play_card(state, action, plan_type=plan_type)
        if action.kind is ActionKind.FINISH_SELECTION:
            return self._finish_selection(state)
        raise KernelError(f"unsupported_action:{action.kind.value}")

    def _materialize_persistent_effects(self, state: StrategyState) -> StrategyState:
        if state.persistent_effects_initialized:
            return state
        persistent = tuple(
            ScheduledEffectRef(
                source_id=card.instance_id,
                card_id=card.card_id,
                definition_effect_index=index,
                turns_until_fire=effect.delay_turns,
                trigger_phase=effect.schedule_phase,
                remaining_uses=effect.schedule_limit,
                remaining_turns=effect.schedule_ttl,
                persistent=True,
                direct_on_trigger=(
                    effect.schedule_phase is SchedulePhase.TURN
                    and effect.schedule_limit == 1
                ),
            )
            for card in state.deck.all_cards
            for index, effect in enumerate(self.cards[card.card_id].persistent_effects)
        )
        persistent += tuple(
            ScheduledEffectRef(
                source_id=f"item:{item_id}",
                card_id=item_id,
                definition_effect_index=index,
                turns_until_fire=effect.delay_turns,
                trigger_phase=effect.schedule_phase,
                remaining_uses=effect.schedule_limit,
                remaining_turns=effect.schedule_ttl,
                persistent=True,
                direct_on_trigger=(
                    effect.schedule_phase is SchedulePhase.TURN
                    and effect.schedule_limit == 1
                ),
                source_kind=EffectSourceKind.ITEM,
            )
            for item_id in state.items
            for index, effect in enumerate(
                self.items[item_id].persistent_effects
            )
        )
        return replace(
            state,
            scheduled_effects=state.scheduled_effects + persistent,
            persistent_effects_initialized=True,
        )

    def _validate_registered_state(self, state: StrategyState, *, plan_type: PlanType) -> None:
        unknown = sorted({card.card_id for card in state.deck.all_cards if card.card_id not in self.cards})
        if unknown:
            raise KernelError(f"unregistered_card_in_deck:{','.join(unknown)}")
        mismatched = sorted(
            {
                card.card_id
                for card in state.deck.all_cards
                if self.cards[card.card_id].plan not in {None, plan_type}
            }
        )
        if mismatched:
            raise KernelError(f"card_plan_mismatch_in_deck:{','.join(mismatched)}")
        unknown_drinks = sorted(
            {drink_id for drink_id in state.drinks if drink_id not in self.drinks}
        )
        if unknown_drinks:
            raise KernelError(
                f"unregistered_drink_in_inventory:{','.join(unknown_drinks)}"
            )
        mismatched_drinks = sorted(
            {
                drink_id
                for drink_id in state.drinks
                if self.drinks[drink_id].plan not in {None, plan_type}
            }
        )
        if mismatched_drinks:
            raise KernelError(
                f"drink_plan_mismatch_in_inventory:{','.join(mismatched_drinks)}"
            )
        scheduled_drinks = {
            scheduled.card_id
            for scheduled in state.scheduled_effects
            if scheduled.source_kind is EffectSourceKind.DRINK
        }
        unknown_scheduled_drinks = sorted(scheduled_drinks - set(self.drinks))
        if unknown_scheduled_drinks:
            raise KernelError(
                "unregistered_drink_schedule:"
                + ",".join(unknown_scheduled_drinks)
            )
        mismatched_scheduled_drinks = sorted(
            drink_id
            for drink_id in scheduled_drinks
            if self.drinks[drink_id].plan not in {None, plan_type}
        )
        if mismatched_scheduled_drinks:
            raise KernelError(
                "drink_plan_mismatch_in_schedule:"
                + ",".join(mismatched_scheduled_drinks)
            )
        unknown_items = sorted(
            {item_id for item_id in state.items if item_id not in self.items}
        )
        if unknown_items:
            raise KernelError(
                f"unregistered_item_in_inventory:{','.join(unknown_items)}"
            )
        mismatched_items = sorted(
            {
                item_id
                for item_id in state.items
                if self.items[item_id].plan not in {None, plan_type}
            }
        )
        if mismatched_items:
            raise KernelError(
                f"item_plan_mismatch_in_inventory:{','.join(mismatched_items)}"
            )
        if state.field_effects:
            raise KernelError("unsupported_field_effects")

    def _pay_stamina_cost(self, state: StrategyState, cost: int, source_id: str) -> tuple[StrategyState, tuple[RuleEvent, ...]]:
        if cost > state.stamina + state.genki:
            raise KernelError("insufficient_stamina_and_genki")
        absorbed = min(state.genki, cost)
        stamina_cost = cost - absorbed
        updated = replace(state, genki=state.genki - absorbed, stamina=state.stamina - stamina_cost)
        return updated, (
            RuleEvent("COST_GENKI", source_id, absorbed),
            RuleEvent("COST_STAMINA", source_id, stamina_cost),
        )

    @staticmethod
    def _declared_cost_kind(definition: CardDefinition) -> CardCostKind:
        if definition.full_power_cost > 0:
            return CardCostKind.FULL_POWER
        return definition.cost_kind

    def _non_full_power_cost_payable(
        self,
        state: StrategyState,
        card: CardRef,
        definition: CardDefinition,
    ) -> bool:
        cost = self._effective_stamina_cost(state, card, definition)
        kind = self._declared_cost_kind(definition)
        if kind in {CardCostKind.NORMAL, CardCostKind.FULL_POWER}:
            return cost <= state.stamina + state.genki
        if kind is CardCostKind.STAMINA:
            return cost <= state.stamina
        resource = {
            CardCostKind.CONCENTRATION: state.concentration,
            CardCostKind.GOOD_CONDITION: state.good_condition_turns,
            CardCostKind.GOOD_IMPRESSION: state.good_impression_turns,
            CardCostKind.MOTIVATION: state.motivation,
        }[kind]
        return cost <= resource

    def _pay_declared_non_full_power_cost(
        self,
        state: StrategyState,
        card: CardRef,
        definition: CardDefinition,
    ) -> tuple[StrategyState, tuple[RuleEvent, ...]]:
        cost = self._effective_stamina_cost(state, card, definition)
        kind = self._declared_cost_kind(definition)
        if kind in {CardCostKind.NORMAL, CardCostKind.FULL_POWER}:
            return self._pay_stamina_cost(state, cost, card.instance_id)
        if not self._non_full_power_cost_payable(state, card, definition):
            raise KernelError(f"insufficient_{kind.value.lower()}_cost_resource")
        if kind is CardCostKind.STAMINA:
            return (
                replace(state, stamina=state.stamina - cost),
                (RuleEvent("COST_STAMINA", card.instance_id, cost),),
            )
        field_name, event_type = {
            CardCostKind.CONCENTRATION: ("concentration", "COST_CONCENTRATION"),
            CardCostKind.GOOD_CONDITION: (
                "good_condition_turns",
                "COST_GOOD_CONDITION",
            ),
            CardCostKind.GOOD_IMPRESSION: (
                "good_impression_turns",
                "COST_GOOD_IMPRESSION",
            ),
            CardCostKind.MOTIVATION: ("motivation", "COST_MOTIVATION"),
        }[kind]
        return (
            replace(state, **{field_name: getattr(state, field_name) - cost}),
            (RuleEvent(event_type, card.instance_id, cost),),
        )

    def _play_card(self, state: StrategyState, action: GameAction, *, plan_type: PlanType) -> tuple[TransitionBranch, ...]:
        if state.pending_choice is not None:
            raise KernelError("secondary_choice_pending")
        if state.actions_remaining <= 0:
            raise KernelError("no_card_action_remaining")
        matches = [card for card in state.deck.hand if card.instance_id == action.card_instance_id]
        if len(matches) != 1:
            raise KernelError("card_instance_missing")
        card = matches[0]
        definition = self.cards.get(card.card_id)
        if definition is None:
            raise KernelError("card_definition_missing")
        if definition.plan is not None and definition.plan is not plan_type:
            raise KernelError("card_plan_unsupported")
        if definition.required_stance is not None and definition.required_stance is not state.stance:
            raise KernelError("card_condition_failed")
        if not _condition_matches(state, definition.condition, self.cards):
            raise KernelError("card_condition_failed")
        if not self._non_full_power_cost_payable(state, card, definition):
            raise KernelError("insufficient_declared_cost_resource")
        full_power_cost = self._effective_full_power_cost(card, definition)
        if full_power_cost > state.full_power:
            raise KernelError("insufficient_full_power")
        if not self._card_secondary_ready(state, definition, card.instance_id):
            raise KernelError("card_secondary_not_ready")
        before = self._trigger_event_phase(
            (TransitionBranch(1.0, state, ()),),
            SchedulePhase.BEFORE_CARD_USED,
            trigger_context=self._card_event_context(
                state,
                card.instance_id,
                exclude_used_card_from_targets=False,
            ),
            condition_state=state,
        )
        if len(before) != 1 or not math.isclose(before[0].probability, 1.0):
            raise KernelError("before_card_used_phase_must_be_deterministic")
        pre_use_state = before[0].state
        before_events = before[0].events
        effect_passes = self._card_effect_passes(pre_use_state, definition)

        nullifies_stamina_cost = (
            pre_use_state.nullify_cost_cards > 0
            and self._declared_cost_kind(definition)
            in {CardCostKind.NORMAL, CardCostKind.STAMINA}
        )
        working, cost_events = self._pay_declared_non_full_power_cost(
            pre_use_state,
            card,
            definition,
        )
        if nullifies_stamina_cost:
            working = replace(working, nullify_cost_cards=working.nullify_cost_cards - 1)
            cost_events += (RuleEvent("NULLIFY_CARD_COST", card.instance_id, 1),)
        working = replace(working, full_power=working.full_power - full_power_cost)
        hand = tuple(item for item in working.deck.hand if item.instance_id != card.instance_id)
        if definition.limited:
            deck = working.deck.replace_zones(hand=hand, removed=working.deck.removed + (card,))
            zone_event = RuleEvent("MOVE_CARD", card.instance_id, detail="HAND->REMOVED")
        else:
            deck = working.deck.replace_zones(hand=hand, discard=working.deck.discard + (card,))
            zone_event = RuleEvent("MOVE_CARD", card.instance_id, detail="HAND->DISCARD")
        working = replace(
            working,
            deck=deck,
            actions_remaining=working.actions_remaining - 1,
            pending_after_card_ids=working.pending_after_card_ids
            + (card.instance_id,),
        )
        events = before_events + cost_events + (
            RuleEvent("COST_FULL_POWER", card.instance_id, full_power_cost),
            zone_event,
            RuleEvent("CONSUME_ACTION", card.instance_id, 1),
        )
        if effect_passes > 1:
            working = replace(
                working,
                double_card_effects_remaining=working.double_card_effects_remaining - 1,
            )
            events += (RuleEvent("CONSUME_DOUBLE_CARD_EFFECT", card.instance_id, 1),)
        branches = (TransitionBranch(1.0, working, events),)
        branches = self._trigger_event_phase(
            branches,
            SchedulePhase.CARD_USED,
            trigger_context=self._card_event_context(
                working,
                card.instance_id,
                exclude_used_card_from_targets=True,
            ),
            condition_state=pre_use_state,
        )
        branches = self._apply_definition_effects(
            branches,
            definition,
            source_id=card.instance_id,
            condition_state=pre_use_state,
            remaining_full_passes=effect_passes - 1,
        )
        branches = self._finalize_card_resolution(branches)
        self._validate_probability_mass(branches)
        return branches

    def _hold_card(self, state: StrategyState, action: GameAction) -> tuple[TransitionBranch, ...]:
        pending = state.pending_choice
        if pending is None or pending.action_kind is not ActionKind.HOLD_CARD:
            raise KernelError("hold_choice_not_pending")
        if action.card_instance_id not in pending.candidate_instance_ids:
            raise KernelError("hold_choice_candidate_missing")
        updated, evicted_id = self._move_card_to_held(state, str(action.card_instance_id))
        moved = self._trigger_event_phase(
            (TransitionBranch(1.0, updated, ()),),
            SchedulePhase.CARD_MOVED_TO_HELD,
            trigger_context=self._moved_card_event_context(
                updated,
                str(action.card_instance_id),
            ),
        )
        if len(moved) != 1 or not math.isclose(moved[0].probability, 1.0):
            raise KernelError("card_moved_to_held_phase_must_be_deterministic")
        updated = moved[0].state
        moved_events = moved[0].events
        candidates = tuple(
            instance_id
            for instance_id in pending.candidate_instance_ids
            if instance_id != action.card_instance_id
        )
        remaining = pending.selections_remaining - 1
        if remaining > 0 and candidates:
            updated = replace(
                updated,
                pending_choice=replace(
                    pending,
                    candidate_instance_ids=candidates,
                    selections_remaining=remaining,
                ),
            )
            events = moved_events + (
                RuleEvent(
                    "RESOLVE_SECONDARY_PARTIAL",
                    pending.source_id,
                    detail=str(action.card_instance_id),
                ),
            )
            if evicted_id is not None:
                events += (RuleEvent("HOLD_LIMIT_EVICT", pending.source_id, detail=evicted_id),)
            branches = (TransitionBranch(1.0, updated, events),)
            self._validate_probability_mass(branches)
            return branches
        events = moved_events + (
            RuleEvent(
                "RESOLVE_SECONDARY",
                pending.source_id,
                detail=str(action.card_instance_id),
            ),
        )
        if evicted_id is not None:
            events += (RuleEvent("HOLD_LIMIT_EVICT", pending.source_id, detail=evicted_id),)
        return self._complete_pending_choice(updated, pending, events)

    def _move_card(self, state: StrategyState, action: GameAction) -> tuple[TransitionBranch, ...]:
        pending = state.pending_choice
        if pending is None or pending.action_kind is not ActionKind.MOVE_CARD:
            raise KernelError("move_choice_not_pending")
        if action.card_instance_id not in pending.candidate_instance_ids:
            raise KernelError("move_choice_candidate_missing")
        updated, source_zone = self._move_card_to_hand(
            state,
            str(action.card_instance_id),
        )
        moved = self._trigger_event_phase(
            (TransitionBranch(1.0, updated, ()),),
            SchedulePhase.CARD_MOVED_TO_HAND,
            trigger_context=self._moved_card_event_context(
                updated,
                str(action.card_instance_id),
            ),
        )
        if len(moved) != 1 or not math.isclose(moved[0].probability, 1.0):
            raise KernelError("card_moved_to_hand_phase_must_be_deterministic")
        return self._complete_pending_choice(
            moved[0].state,
            pending,
            moved[0].events
            + (
                RuleEvent(
                    "RESOLVE_SECONDARY",
                    pending.source_id,
                    detail=str(action.card_instance_id),
                ),
                RuleEvent(
                    "MOVE_SELECTED_TO_HAND",
                    pending.source_id,
                    detail=(
                        f"{source_zone.value}->{CardZone.HAND.value}:"
                        f"{action.card_instance_id}"
                    ),
                ),
            ),
        )

    def _finish_selection(self, state: StrategyState) -> tuple[TransitionBranch, ...]:
        pending = state.pending_choice
        if (
            pending is None
            or pending.action_kind is not ActionKind.HOLD_CARD
            or not pending.selection_optional
        ):
            raise KernelError("optional_selection_not_pending")
        return self._complete_pending_choice(
            state,
            pending,
            (RuleEvent("FINISH_SECONDARY", pending.source_id),),
        )

    def _complete_pending_choice(
        self,
        state: StrategyState,
        pending: PendingChoice,
        events: tuple[RuleEvent, ...],
    ) -> tuple[TransitionBranch, ...]:
        updated = replace(state, pending_choice=None)
        branches = (TransitionBranch(1.0, updated, events),)
        if pending.continuation_index is not None or pending.remaining_effect_passes > 0:
            if pending.source_id.startswith("drink:"):
                if pending.remaining_effect_passes:
                    raise KernelError("drink_continuation_cannot_repeat")
                drink_id = pending.source_id.removeprefix("drink:")
                definition = self.drinks.get(drink_id)
                if definition is None or pending.continuation_index is None:
                    raise KernelError("drink_continuation_missing")
                assert pending.condition_stance is not None
                assert pending.condition_full_power is not None
                condition_state = replace(
                    updated,
                    stance=pending.condition_stance,
                    full_power=pending.condition_full_power,
                )
                branches = self._apply_drink_effects(
                    branches,
                    definition,
                    source_id=pending.source_id,
                    condition_state=condition_state,
                    start_index=pending.continuation_index,
                )
                branches = self._finalize_card_resolution(branches)
                self._validate_probability_mass(branches)
                return branches
            matches = [card for card in updated.deck.all_cards if card.instance_id == pending.source_id]
            if len(matches) != 1:
                raise KernelError("secondary_source_card_missing")
            definition = self.cards.get(matches[0].card_id)
            if definition is None or (
                pending.continuation_index is not None
                and pending.continuation_index >= len(definition.effects)
            ):
                raise KernelError("secondary_continuation_missing")
            assert pending.condition_stance is not None
            assert pending.condition_full_power is not None
            condition_state = replace(
                updated,
                stance=pending.condition_stance,
                full_power=pending.condition_full_power,
            )
            branches = self._apply_definition_effects(
                branches,
                definition,
                source_id=pending.source_id,
                condition_state=condition_state,
                start_index=pending.continuation_index or len(definition.effects),
                remaining_full_passes=pending.remaining_effect_passes,
            )
        branches = self._finalize_card_resolution(branches)
        self._validate_probability_mass(branches)
        return branches

    def _push_pending_card_continuation(
        self,
        state: StrategyState,
        pending: PendingChoice,
    ) -> StrategyState:
        if pending.continuation_index is None and pending.remaining_effect_passes == 0:
            return state
        source = self._card_by_instance(state, pending.source_id)
        if source is None:
            raise KernelError("secondary_source_card_missing")
        definition = self.cards.get(source.card_id)
        if definition is None:
            raise KernelError("secondary_continuation_missing")
        if pending.condition_stance is None or pending.condition_full_power is None:
            raise KernelError("secondary_continuation_snapshot_missing")
        frame = CardEffectsContinuation(
            source_id=pending.source_id,
            card_id=definition.card_id,
            continuation_index=(
                pending.continuation_index
                if pending.continuation_index is not None
                else len(definition.effects)
            ),
            remaining_effect_passes=pending.remaining_effect_passes,
            condition_stance=pending.condition_stance,
            condition_full_power=pending.condition_full_power,
        )
        return replace(
            state,
            resolution_stack=state.resolution_stack + (frame,),
        )

    def _free_play_card(
        self,
        state: StrategyState,
        action: GameAction,
        *,
        plan_type: PlanType,
    ) -> tuple[TransitionBranch, ...]:
        pending = state.pending_choice
        if pending is None or pending.action_kind is not ActionKind.FREE_PLAY:
            raise KernelError("free_play_choice_not_pending")
        if action.card_instance_id not in pending.candidate_instance_ids:
            raise KernelError("free_play_choice_candidate_missing")
        state = self._push_pending_card_continuation(state, pending)
        card = self._card_by_instance(state, str(action.card_instance_id))
        if card is None:
            raise KernelError("free_play_card_instance_missing")
        definition = self.cards.get(card.card_id)
        if definition is None:
            raise KernelError("free_play_card_definition_missing")
        if definition.plan is not None and definition.plan is not plan_type:
            raise KernelError("free_play_card_plan_unsupported")

        original_condition_matches = (
            definition.required_stance in {None, state.stance}
            and _condition_matches(state, definition.condition, self.cards)
        )
        if original_condition_matches and not self._card_secondary_ready(
            state,
            definition,
            card.instance_id,
        ):
            raise KernelError("free_play_target_secondary_not_ready")

        before = self._trigger_event_phase(
            (TransitionBranch(1.0, replace(state, pending_choice=None), ()),),
            SchedulePhase.BEFORE_CARD_USED,
            trigger_context=self._card_event_context(
                state,
                card.instance_id,
                exclude_used_card_from_targets=False,
            ),
            condition_state=state,
        )
        if (
            len(before) != 1
            or not math.isclose(before[0].probability, 1.0)
            or before[0].state.pending_choice is not None
        ):
            raise KernelError("free_before_card_used_phase_must_be_deterministic")
        pre_use_state = before[0].state
        condition_matches = (
            definition.required_stance in {None, pre_use_state.stance}
            and _condition_matches(pre_use_state, definition.condition, self.cards)
        )

        working = replace(
            pre_use_state,
            pending_after_card_ids=pre_use_state.pending_after_card_ids
            + (card.instance_id,),
        )
        working, zone_event = self._move_card_for_use(working, card, definition)
        events = before[0].events + (
            RuleEvent("RESOLVE_SECONDARY", pending.source_id, detail=card.instance_id),
            RuleEvent("FREE_PLAY", card.instance_id, detail=pending.source_id),
            zone_event,
        )
        branches = (TransitionBranch(1.0, working, events),)
        branches = self._trigger_event_phase(
            branches,
            SchedulePhase.CARD_USED,
            trigger_context=self._card_event_context(
                working,
                card.instance_id,
                exclude_used_card_from_targets=True,
            ),
            condition_state=pre_use_state,
        )
        if not condition_matches:
            branches = tuple(
                replace(
                    branch,
                    events=branch.events
                    + (RuleEvent("FREE_PLAY_CONDITION_SKIPPED", card.instance_id),),
                )
                for branch in branches
            )
            branches = self._finalize_card_resolution(branches)
            self._validate_probability_mass(branches)
            return branches

        effect_passes = self._card_effect_passes(pre_use_state, definition)
        if effect_passes > 1:
            branches = tuple(
                replace(
                    branch,
                    state=replace(
                        branch.state,
                        double_card_effects_remaining=(
                            branch.state.double_card_effects_remaining - 1
                        ),
                    ),
                    events=branch.events
                    + (RuleEvent("CONSUME_DOUBLE_CARD_EFFECT", card.instance_id, 1),),
                )
                for branch in branches
            )
        branches = self._apply_definition_effects(
            branches,
            definition,
            source_id=card.instance_id,
            condition_state=pre_use_state,
            remaining_full_passes=effect_passes - 1,
        )
        branches = self._finalize_card_resolution(branches)
        self._validate_probability_mass(branches)
        return branches

    def _paid_play_card(
        self,
        state: StrategyState,
        action: GameAction,
        *,
        plan_type: PlanType,
    ) -> tuple[TransitionBranch, ...]:
        pending = state.pending_choice
        if pending is None or pending.action_kind is not ActionKind.PAID_PLAY:
            raise KernelError("paid_play_choice_not_pending")
        if action.card_instance_id not in pending.candidate_instance_ids:
            raise KernelError("paid_play_choice_candidate_missing")
        state = self._push_pending_card_continuation(state, pending)
        card = self._card_by_instance(state, str(action.card_instance_id))
        if card is None:
            raise KernelError("paid_play_card_instance_missing")
        definition = self.cards.get(card.card_id)
        if definition is None:
            raise KernelError("paid_play_card_definition_missing")
        if definition.plan is not None and definition.plan is not plan_type:
            raise KernelError("paid_play_card_plan_unsupported")

        usable = (
            definition.required_stance in {None, state.stance}
            and _condition_matches(state, definition.condition, self.cards)
            and self._non_full_power_cost_payable(state, card, definition)
            and self._effective_full_power_cost(card, definition) <= state.full_power
            and self._card_secondary_ready(state, definition, card.instance_id)
        )
        if not usable:
            discarded, zone_event = self._move_card_to_discard(
                replace(state, pending_choice=None),
                card,
            )
            branches = self._resume_resolution_stack(
                (
                    TransitionBranch(
                        1.0,
                        discarded,
                        (
                            RuleEvent(
                                "RESOLVE_SECONDARY",
                                pending.source_id,
                                detail=card.instance_id,
                            ),
                            RuleEvent(
                                "PAID_PLAY_UNUSABLE_DISCARD",
                                card.instance_id,
                                detail=pending.source_id,
                            ),
                            zone_event,
                        ),
                    ),
                )
            )
            self._validate_probability_mass(branches)
            return branches

        before = self._trigger_event_phase(
            (TransitionBranch(1.0, replace(state, pending_choice=None), ()),),
            SchedulePhase.BEFORE_CARD_USED,
            trigger_context=self._card_event_context(
                state,
                card.instance_id,
                exclude_used_card_from_targets=False,
            ),
            condition_state=state,
        )
        if (
            len(before) != 1
            or not math.isclose(before[0].probability, 1.0)
            or before[0].state.pending_choice is not None
        ):
            raise KernelError("paid_before_card_used_phase_must_be_deterministic")
        pre_use_state = before[0].state
        effect_passes = self._card_effect_passes(pre_use_state, definition)
        full_power_cost = self._effective_full_power_cost(card, definition)
        nullifies_stamina_cost = (
            pre_use_state.nullify_cost_cards > 0
            and self._declared_cost_kind(definition)
            in {CardCostKind.NORMAL, CardCostKind.STAMINA}
        )
        working, cost_events = self._pay_declared_non_full_power_cost(
            pre_use_state,
            card,
            definition,
        )
        if nullifies_stamina_cost:
            working = replace(
                working,
                nullify_cost_cards=working.nullify_cost_cards - 1,
            )
            cost_events += (RuleEvent("NULLIFY_CARD_COST", card.instance_id, 1),)
        working = replace(
            working,
            full_power=working.full_power - full_power_cost,
            pending_after_card_ids=working.pending_after_card_ids
            + (card.instance_id,),
        )
        working, zone_event = self._move_card_for_use(working, card, definition)
        events = before[0].events + (
            RuleEvent("RESOLVE_SECONDARY", pending.source_id, detail=card.instance_id),
            RuleEvent("PAID_PLAY", card.instance_id, detail=pending.source_id),
        ) + cost_events + (
            RuleEvent("COST_FULL_POWER", card.instance_id, full_power_cost),
            zone_event,
        )
        if effect_passes > 1:
            working = replace(
                working,
                double_card_effects_remaining=(
                    working.double_card_effects_remaining - 1
                ),
            )
            events += (RuleEvent("CONSUME_DOUBLE_CARD_EFFECT", card.instance_id, 1),)
        branches = (TransitionBranch(1.0, working, events),)
        branches = self._trigger_event_phase(
            branches,
            SchedulePhase.CARD_USED,
            trigger_context=self._card_event_context(
                working,
                card.instance_id,
                exclude_used_card_from_targets=True,
            ),
            condition_state=pre_use_state,
        )
        branches = self._apply_definition_effects(
            branches,
            definition,
            source_id=card.instance_id,
            condition_state=pre_use_state,
            remaining_full_passes=effect_passes - 1,
        )
        branches = self._finalize_card_resolution(branches)
        self._validate_probability_mass(branches)
        return branches

    def _apply_definition_effects(
        self,
        branches: tuple[TransitionBranch, ...],
        definition: CardDefinition,
        *,
        source_id: str,
        condition_state: StrategyState,
        start_index: int = 0,
        remaining_full_passes: int = 0,
    ) -> tuple[TransitionBranch, ...]:
        effect_index = start_index
        remaining = remaining_full_passes
        while True:
            for index in range(effect_index, len(definition.effects)):
                next_index = index + 1
                branches = self._apply_effect(
                    branches,
                    definition.effects[index],
                    source_id=source_id,
                    condition_state=condition_state,
                    continuation_index=next_index if next_index < len(definition.effects) else None,
                    remaining_effect_passes=remaining,
                    source_card_id=definition.card_id,
                    definition_effect_index=index,
                    direct_effect=True,
                )
                pending_count = sum(branch.state.pending_choice is not None for branch in branches)
                if pending_count:
                    suspended: list[TransitionBranch] = []
                    continuing: list[TransitionBranch] = []
                    for branch in branches:
                        pending = branch.state.pending_choice
                        if pending is None:
                            continuing.append(branch)
                            continue
                        existing_outer_continuation = any(
                            isinstance(existing, CardEffectsContinuation)
                            and existing.source_id == source_id
                            and existing.card_id == definition.card_id
                            and existing.continuation_index == next_index
                            and existing.remaining_effect_passes == remaining
                            for existing in branch.state.resolution_stack
                        )
                        needs_outer_continuation = (
                            pending.continuation_index is None
                            and pending.remaining_effect_passes == 0
                            and (next_index < len(definition.effects) or remaining > 0)
                            and not existing_outer_continuation
                        )
                        if needs_outer_continuation:
                            frame = CardEffectsContinuation(
                                source_id=source_id,
                                card_id=definition.card_id,
                                continuation_index=next_index,
                                remaining_effect_passes=remaining,
                                condition_stance=condition_state.stance,
                                condition_full_power=condition_state.full_power,
                            )
                            stack = branch.state.resolution_stack
                            insert_at = len(stack)
                            while insert_at > 0 and isinstance(
                                stack[insert_at - 1],
                                ScheduledEffectsContinuation,
                            ):
                                insert_at -= 1
                            stack = (
                                stack[:insert_at]
                                + (frame,)
                                + stack[insert_at:]
                            )
                            branch = replace(
                                branch,
                                state=replace(
                                    branch.state,
                                    resolution_stack=stack,
                                ),
                            )
                        suspended.append(branch)
                    if continuing:
                        suspended.extend(
                            self._apply_definition_effects(
                                tuple(continuing),
                                definition,
                                source_id=source_id,
                                condition_state=condition_state,
                                start_index=next_index,
                                remaining_full_passes=remaining,
                            )
                        )
                    return tuple(suspended)
            if remaining < 1:
                return branches
            remaining -= 1
            effect_index = 0

    def _use_drink(
        self,
        state: StrategyState,
        action: GameAction,
        *,
        plan_type: PlanType,
    ) -> tuple[TransitionBranch, ...]:
        if action.drink_id not in state.drinks:
            raise KernelError("drink_missing")
        definition = self.drinks.get(str(action.drink_id))
        if definition is None:
            raise KernelError("drink_definition_missing")
        if definition.plan not in {None, plan_type}:
            raise KernelError("drink_plan_unsupported")
        if not self._drink_secondary_ready(state, definition):
            raise KernelError("drink_secondary_not_ready")
        source_id = f"drink:{action.drink_id}"
        working, cost_events = self._pay_stamina_cost(
            state,
            definition.stamina_cost,
            source_id,
        )
        drinks = list(working.drinks)
        drinks.remove(str(action.drink_id))
        working = replace(working, drinks=tuple(drinks))
        branches = (
            TransitionBranch(
                1.0,
                working,
                cost_events + (RuleEvent("CONSUME_DRINK", source_id),),
            ),
        )
        branches = self._apply_drink_effects(
            branches,
            definition,
            source_id=source_id,
            condition_state=state,
        )
        branches = self._finalize_card_resolution(branches)
        self._validate_probability_mass(branches)
        return branches

    def _apply_drink_effects(
        self,
        branches: tuple[TransitionBranch, ...],
        definition: DrinkDefinition,
        *,
        source_id: str,
        condition_state: StrategyState,
        start_index: int = 0,
    ) -> tuple[TransitionBranch, ...]:
        for index in range(start_index, len(definition.effects)):
            next_index = index + 1
            branches = self._apply_effect(
                branches,
                definition.effects[index],
                source_id=source_id,
                condition_state=condition_state,
                continuation_index=(
                    next_index if next_index < len(definition.effects) else None
                ),
                source_card_id=definition.drink_id,
                definition_effect_index=index,
                source_kind=EffectSourceKind.DRINK,
                direct_effect=True,
            )
            if any(branch.state.pending_choice is not None for branch in branches):
                suspended: list[TransitionBranch] = []
                continuing: list[TransitionBranch] = []
                for branch in branches:
                    if branch.state.pending_choice is None:
                        continuing.append(branch)
                    else:
                        suspended.append(branch)
                if continuing:
                    suspended.extend(
                        self._apply_drink_effects(
                            tuple(continuing),
                            definition,
                            source_id=source_id,
                            condition_state=condition_state,
                            start_index=next_index,
                        )
                    )
                return tuple(suspended)
        return branches

    def _apply_effect(
        self,
        branches: tuple[TransitionBranch, ...],
        effect: EffectSpec,
        *,
        source_id: str,
        condition_state: StrategyState | None = None,
        continuation_index: int | None = None,
        remaining_effect_passes: int = 0,
        source_card_id: str | None = None,
        definition_effect_index: int | None = None,
        trigger_context: EffectTriggerContext | None = None,
        direct_effect: bool = False,
        source_kind: EffectSourceKind = EffectSourceKind.CARD,
    ) -> tuple[TransitionBranch, ...]:
        effect.validate()
        if effect.condition is not None:
            result: list[TransitionBranch] = []
            unconditional = replace(effect, condition=None)
            for branch in branches:
                if _condition_matches(
                    condition_state or branch.state,
                    effect.condition,
                    self.cards,
                    trigger_context,
                ):
                    result.extend(
                        self._apply_effect(
                            (branch,),
                            unconditional,
                            source_id=source_id,
                            condition_state=condition_state,
                            continuation_index=continuation_index,
                            remaining_effect_passes=remaining_effect_passes,
                            source_card_id=source_card_id,
                            definition_effect_index=definition_effect_index,
                            trigger_context=trigger_context,
                            direct_effect=direct_effect,
                            source_kind=source_kind,
                        )
                    )
                else:
                    result.append(
                        TransitionBranch(
                            branch.probability,
                            branch.state,
                            branch.events
                            + (
                                RuleEvent(
                                    "CONDITION_SKIPPED",
                                    source_id,
                                    detail=effect.condition.kind.value,
                                ),
                            ),
                        )
                    )
            return tuple(result)
        if effect.kind is EffectKind.DRAW:
            result = branches
            for _index in range(int(effect.amount)):
                expanded: list[TransitionBranch] = []
                for branch in result:
                    for draw in self._draw_one(branch.state, source_id=source_id):
                        expanded.append(
                            TransitionBranch(
                                probability=branch.probability * draw.probability,
                                state=draw.state,
                                events=branch.events + draw.events,
                            )
                        )
                result = self._merge_probability_branches(tuple(expanded))
            return result
        if effect.kind is EffectKind.MOVE_RANDOM_TO_HAND:
            assert effect.card_selector is not None
            expanded = tuple(
                TransitionBranch(
                    probability=branch.probability * moved.probability,
                    state=moved.state,
                    events=branch.events + moved.events,
                )
                for branch in branches
                for moved in self._move_random_to_hand(
                    branch.state,
                    effect.card_selector,
                    source_id=source_id,
                )
            )
            return self._merge_probability_branches(expanded)
        if effect.kind is EffectKind.EXCHANGE_HAND:
            expanded = tuple(
                TransitionBranch(
                    probability=branch.probability * exchanged.probability,
                    state=exchanged.state,
                    events=branch.events + exchanged.events,
                )
                for branch in branches
                for exchanged in self._exchange_hand(branch.state, source_id=source_id)
            )
            return self._merge_probability_branches(expanded)
        if effect.kind is EffectKind.ADD_CARD_TO_DECK:
            assert effect.card_id is not None
            expanded = tuple(
                TransitionBranch(
                    probability=branch.probability * added.probability,
                    state=added.state,
                    events=branch.events + added.events,
                )
                for branch in branches
                for added in self._add_card_to_deck(
                    branch.state,
                    effect.card_id,
                    source_id=source_id,
                )
            )
            return self._merge_probability_branches(expanded)
        if effect.kind is EffectKind.FREE_PLAY_RANDOM:
            assert effect.card_selector is not None
            expanded = tuple(
                TransitionBranch(
                    probability=branch.probability * played.probability,
                    state=played.state,
                    events=branch.events + played.events,
                )
                for branch in branches
                for played in self._free_play_random(
                    branch.state,
                    effect.card_selector,
                    source_id=source_id,
                )
            )
            return self._merge_probability_branches(expanded)

        repeats = effect.repeats
        if effect.kind is EffectKind.SCORE:
            repeat_bonuses = {
                self._card_by_instance(branch.state, source_id).score_repeat_bonus
                for branch in branches
                if self._card_by_instance(branch.state, source_id) is not None
            }
            if len(repeat_bonuses) > 1:
                raise KernelError("source_card_repeat_bonus_branch_mismatch")
            if repeat_bonuses:
                repeats += repeat_bonuses.pop()
        result = branches
        for _repeat in range(repeats):
            expanded: list[TransitionBranch] = []
            event_branched = False
            for branch in result:
                old_state = branch.state
                state, event = self._apply_non_draw_effect(
                    old_state,
                    effect,
                    source_id=source_id,
                    condition_state=condition_state,
                    continuation_index=continuation_index,
                    remaining_effect_passes=remaining_effect_passes,
                    source_card_id=source_card_id,
                    definition_effect_index=definition_effect_index,
                    trigger_context=trigger_context,
                    source_kind=source_kind,
                )
                applied = (
                    TransitionBranch(
                        branch.probability,
                        state,
                        branch.events + (event,),
                    ),
                )
                if (
                    effect.kind is EffectKind.STANCE
                    and self._is_counted_stance_change(old_state.stance, state.stance)
                ):
                    applied = self._trigger_event_phase(
                        applied,
                        SchedulePhase.STANCE_CHANGED,
                        trigger_context=EffectTriggerContext(
                            is_direct_effect=direct_effect,
                        ),
                    )
                    applied = tuple(
                        replace(
                            child,
                            state=self._increment_stance_counts(
                                child.state,
                                direct_effect=direct_effect,
                            ),
                        )
                        for child in applied
                    )
                    event_branched = event_branched or len(applied) > 1
                elif (
                    effect.kind is EffectKind.HOLD_SELF
                    and not any(
                        card.instance_id == source_id
                        for card in old_state.deck.held
                    )
                    and any(
                        card.instance_id == source_id
                        for card in state.deck.held
                    )
                ):
                    applied = self._trigger_event_phase(
                        applied,
                        SchedulePhase.CARD_MOVED_TO_HELD,
                        trigger_context=self._moved_card_event_context(
                            state,
                            source_id,
                        ),
                    )
                    event_branched = event_branched or len(applied) > 1
                elif (
                    effect.kind is EffectKind.FULL_POWER
                    and direct_effect
                    and state.full_power > old_state.full_power
                ):
                    applied = self._trigger_event_phase(
                        applied,
                        SchedulePhase.FULL_POWER_CHARGE_INCREASED,
                        trigger_context=EffectTriggerContext(
                            is_direct_effect=True,
                        ),
                    )
                    event_branched = event_branched or len(applied) > 1
                expanded.extend(applied)
            result = (
                self._merge_probability_branches(tuple(expanded))
                if event_branched
                else tuple(expanded)
            )
        return result

    def _apply_non_draw_effect(
        self,
        state: StrategyState,
        effect: EffectSpec,
        *,
        source_id: str,
        condition_state: StrategyState | None,
        continuation_index: int | None,
        remaining_effect_passes: int,
        source_card_id: str | None,
        definition_effect_index: int | None,
        trigger_context: EffectTriggerContext | None,
        source_kind: EffectSourceKind,
    ) -> tuple[StrategyState, RuleEvent]:
        amount = effect.amount
        if effect.value_source is not None:
            source_value = {
                EffectValueSource.GOOD_CONDITION: state.good_condition_turns,
                EffectValueSource.GOOD_IMPRESSION: state.good_impression_turns,
                EffectValueSource.MOTIVATION: state.motivation,
                EffectValueSource.GENKI: state.genki,
            }[effect.value_source]
            amount += source_value * effect.value_scale
        if effect.kind is EffectKind.SCORE:
            source_card = self._card_by_instance(state, source_id)
            if source_card is not None:
                amount += source_card.score_bonus
            amount += state.cumulative_full_power * effect.full_power_scale
            parameter_gain = math.ceil(
                amount
                + state.concentration * effect.score_concentration_multiplier
                + state.passion * effect.score_enthusiasm_multiplier
            )
            if state.good_condition_turns > 0:
                good_condition_bonus = 0.5
                if state.excellent_condition_turns > 0:
                    good_condition_bonus += state.good_condition_turns * 0.1
                parameter_gain *= (
                    1
                    + good_condition_bonus
                    * effect.score_good_condition_multiplier
                )
            parameter_gain *= _stance_parameter_multiplier(state.stance)
            parameter_gain *= 1.0 + state.parameter_gain_bonus_pct / 100.0
            parameter_gain *= 1.0 + sum(buff.amount for buff in state.score_buffs)
            parameter_gain = math.ceil(parameter_gain)
            score_gain = math.ceil(parameter_gain * state.score_multiplier)
            return replace(state, score=state.score + score_gain), RuleEvent("SCORE", source_id, score_gain)
        if effect.kind is EffectKind.STAMINA:
            stamina = min(state.max_stamina, max(0, state.stamina + int(amount)))
            return replace(state, stamina=stamina), RuleEvent("STAMINA", source_id, stamina - state.stamina)
        if effect.kind is EffectKind.FIXED_STAMINA_TO_ZERO:
            return (
                replace(state, stamina=0),
                RuleEvent("FIXED_STAMINA_TO_ZERO", source_id, -state.stamina),
            )
        if effect.kind is EffectKind.GENKI:
            source_card = self._card_by_instance(state, source_id)
            if source_card is not None:
                amount += source_card.genki_bonus
            amount += state.motivation * effect.genki_motivation_multiplier
            resolved = math.ceil(round(amount, 2))
            return (
                replace(state, genki=max(0, state.genki + resolved)),
                RuleEvent("GENKI", source_id, resolved),
            )
        if effect.kind is EffectKind.FULL_POWER:
            resolved = amount
            if amount > 0:
                resolved *= 1.0 + sum(buff.amount for buff in state.full_power_charge_buffs)
            return (
                replace(
                    state,
                    full_power=max(0.0, state.full_power + resolved),
                    cumulative_full_power=state.cumulative_full_power + max(0.0, resolved),
                ),
                RuleEvent("FULL_POWER", source_id, resolved),
            )
        if effect.kind is EffectKind.PASSION:
            return replace(state, passion=max(0, state.passion + int(amount))), RuleEvent("PASSION", source_id, amount)
        if effect.kind is EffectKind.PASSION_BUFF:
            return replace(state, passion_buff=max(0, state.passion_buff + int(amount))), RuleEvent("PASSION_BUFF", source_id, amount)
        if effect.kind is EffectKind.PASSION_GAIN_BONUS_PCT:
            return (
                replace(state, passion_gain_bonus_pct=state.passion_gain_bonus_pct + amount),
                RuleEvent("PASSION_GAIN_BONUS_PCT", source_id, amount),
            )
        if effect.kind is EffectKind.ENTHUSIASM_BONUS_BUFF:
            modifiers = self._add_timed_modifier(
                state.enthusiasm_bonus_buffs,
                amount=amount,
                duration=effect.duration,
            )
            return (
                replace(state, enthusiasm_bonus_buffs=modifiers),
                RuleEvent("ENTHUSIASM_BONUS_BUFF", source_id, amount, str(effect.duration)),
            )
        if effect.kind is EffectKind.ENTHUSIASM_MULTIPLIER_BUFF:
            modifiers = self._add_timed_modifier(
                state.enthusiasm_multiplier_buffs,
                amount=amount,
                duration=effect.duration,
            )
            return (
                replace(state, enthusiasm_multiplier_buffs=modifiers),
                RuleEvent("ENTHUSIASM_MULTIPLIER_BUFF", source_id, amount, str(effect.duration)),
            )
        if effect.kind is EffectKind.FULL_POWER_CHARGE_BUFF:
            modifiers = self._add_timed_modifier(
                state.full_power_charge_buffs,
                amount=amount,
                duration=effect.duration,
            )
            return (
                replace(state, full_power_charge_buffs=modifiers),
                RuleEvent("FULL_POWER_CHARGE_BUFF", source_id, amount, str(effect.duration)),
            )
        if effect.kind is EffectKind.SCORE_BUFF:
            modifiers = self._add_timed_modifier(
                state.score_buffs,
                amount=amount,
                duration=effect.duration,
            )
            return (
                replace(state, score_buffs=modifiers),
                RuleEvent("SCORE_BUFF", source_id, amount, str(effect.duration)),
            )
        if effect.kind is EffectKind.STANCE_LOCK_TURNS:
            turns = state.stance_lock_turns + int(amount)
            return (
                replace(
                    state,
                    stance_lock_turns=turns,
                    stance_lock_fresh=(
                        state.stance_lock_fresh or state.stance_lock_turns == 0
                    ),
                ),
                RuleEvent("STANCE_LOCK_TURNS", source_id, amount),
            )
        if effect.kind is EffectKind.NULLIFY_COST_CARDS:
            return (
                replace(
                    state,
                    nullify_cost_cards=state.nullify_cost_cards + int(amount),
                ),
                RuleEvent("NULLIFY_COST_CARDS", source_id, amount),
            )
        if effect.kind is EffectKind.UPGRADE_HAND:
            return self._upgrade_hand(state, source_id=source_id)
        if effect.kind is EffectKind.PARAMETER_GAIN_BONUS_PCT:
            return (
                replace(state, parameter_gain_bonus_pct=state.parameter_gain_bonus_pct + amount),
                RuleEvent("PARAMETER_GAIN_BONUS_PCT", source_id, amount),
            )
        if effect.kind is EffectKind.STANCE:
            assert effect.stance is not None
            old_stance = state.stance
            if (
                old_stance is Stance.FULL_POWER
                and effect.stance is not Stance.NEUTRAL
            ) or (
                old_stance is not Stance.FULL_POWER and state.stance_lock_turns > 0
            ) or (
                old_stance is Stance.LEISURE
                and effect.stance in {Stance.CONSERVE_1, Stance.CONSERVE_2}
            ):
                return state, RuleEvent(
                    "STANCE_BLOCKED",
                    source_id,
                    detail=f"{old_stance.value}->{effect.stance.value}",
                )
            target = _stack_stance(old_stance, effect.stance)
            updated = replace(
                state,
                stance=target,
                previous_stance=old_stance,
            )
            if _is_counted_stance_change(old_stance, target):
                updated = self._apply_stance_departure_benefits(
                    updated,
                    old_stance=old_stance,
                    target=target,
                )
            return updated, RuleEvent("STANCE", source_id, detail=f"{old_stance.value}->{target.value}")
        if effect.kind is EffectKind.EXTRA_ACTION:
            return replace(state, actions_remaining=state.actions_remaining + int(amount)), RuleEvent("EXTRA_ACTION", source_id, amount)
        if effect.kind is EffectKind.DOUBLE_CARD_EFFECT:
            return (
                replace(
                    state,
                    double_card_effects_remaining=state.double_card_effects_remaining + int(amount),
                ),
                RuleEvent("DOUBLE_CARD_EFFECT", source_id, amount),
            )
        if effect.kind is EffectKind.HALF_COST_TURNS:
            turns = state.half_cost_turns + int(amount)
            return (
                replace(
                    state,
                    half_cost_turns=turns,
                    half_cost_fresh=state.half_cost_fresh or state.half_cost_turns == 0,
                ),
                RuleEvent("HALF_COST_TURNS", source_id, amount),
            )
        if effect.kind is EffectKind.DOUBLE_COST_TURNS:
            turns = state.double_cost_turns + int(amount)
            return (
                replace(
                    state,
                    double_cost_turns=turns,
                    double_cost_fresh=(
                        state.double_cost_fresh or state.double_cost_turns == 0
                    ),
                ),
                RuleEvent("DOUBLE_COST_TURNS", source_id, amount),
            )
        if effect.kind is EffectKind.CONCENTRATION:
            return replace(state, concentration=max(0, state.concentration + int(amount))), RuleEvent("CONCENTRATION", source_id, amount)
        if effect.kind is EffectKind.GOOD_CONDITION:
            return (
                replace(state, good_condition_turns=max(0, state.good_condition_turns + int(amount))),
                RuleEvent("GOOD_CONDITION", source_id, amount),
            )
        if effect.kind is EffectKind.EXCELLENT_CONDITION:
            return (
                replace(state, excellent_condition_turns=max(0, state.excellent_condition_turns + int(amount))),
                RuleEvent("EXCELLENT_CONDITION", source_id, amount),
            )
        if effect.kind is EffectKind.GOOD_IMPRESSION:
            return (
                replace(state, good_impression_turns=max(0, state.good_impression_turns + int(amount))),
                RuleEvent("GOOD_IMPRESSION", source_id, amount),
            )
        if effect.kind is EffectKind.MOTIVATION:
            return replace(state, motivation=max(0, state.motivation + int(amount))), RuleEvent("MOTIVATION", source_id, amount)
        if effect.kind is EffectKind.STATUS:
            assert effect.status_id is not None
            status = TimedStatus(effect.status_id, int(amount), effect.duration)
            return replace(state, statuses=state.statuses + (status,)), RuleEvent("STATUS", source_id, amount, effect.status_id)
        if effect.kind is EffectKind.TURNS_REMAINING:
            turns_remaining = max(0, state.turns_remaining + int(amount))
            return (
                replace(state, turns_remaining=turns_remaining),
                RuleEvent("TURNS_REMAINING", source_id, turns_remaining - state.turns_remaining),
            )
        if effect.kind is EffectKind.HOLD_SELF:
            updated, evicted_id = self._move_card_to_held(state, source_id)
            detail = source_id if evicted_id is None else f"{source_id}|evicted:{evicted_id}"
            return updated, RuleEvent("HOLD_CARD", source_id, detail=detail)
        if effect.kind is EffectKind.HOLD_SELECTED:
            if state.pending_choice is not None:
                raise KernelError("multiple_secondary_choices_not_supported")
            candidates = tuple(
                card.instance_id
                for zone in effect.target_zones
                for card in self._zone_cards(state, zone)
                if card.instance_id != source_id
            )
            if not candidates:
                if effect.selection_optional:
                    return state, RuleEvent("OPTIONAL_HOLD_EMPTY", source_id)
                raise KernelError("secondary_choice_has_no_candidate")
            pending = PendingChoice(
                ActionKind.HOLD_CARD,
                source_id,
                tuple(dict.fromkeys(candidates)),
                continuation_index=continuation_index,
                remaining_effect_passes=remaining_effect_passes,
                condition_stance=(
                    condition_state.stance
                    if (continuation_index is not None or remaining_effect_passes > 0) and condition_state
                    else None
                ),
                condition_full_power=(
                    condition_state.full_power
                    if (continuation_index is not None or remaining_effect_passes > 0) and condition_state
                    else None
                ),
                selections_remaining=effect.selection_count,
                selection_optional=effect.selection_optional,
            )
            return (
                replace(state, pending_choice=pending),
                RuleEvent("REQUEST_SECONDARY", source_id, detail=ActionKind.HOLD_CARD.value),
            )
        if effect.kind is EffectKind.MOVE_SELECTED_TO_HAND:
            if state.pending_choice is not None:
                raise KernelError("multiple_secondary_choices_not_supported")
            candidates = tuple(
                card.instance_id
                for zone in effect.target_zones
                for card in self._zone_cards(state, zone)
                if card.instance_id != source_id
            )
            if not candidates:
                raise KernelError("secondary_choice_has_no_candidate")
            pending = PendingChoice(
                ActionKind.MOVE_CARD,
                source_id,
                tuple(dict.fromkeys(candidates)),
                continuation_index=continuation_index,
                remaining_effect_passes=remaining_effect_passes,
                condition_stance=(
                    condition_state.stance
                    if (continuation_index is not None or remaining_effect_passes > 0)
                    and condition_state
                    else None
                ),
                condition_full_power=(
                    condition_state.full_power
                    if (continuation_index is not None or remaining_effect_passes > 0)
                    and condition_state
                    else None
                ),
            )
            return (
                replace(state, pending_choice=pending),
                RuleEvent("REQUEST_SECONDARY", source_id, detail=ActionKind.MOVE_CARD.value),
            )
        if effect.kind is EffectKind.FREE_PLAY_SELECTED:
            if state.pending_choice is not None:
                raise KernelError("multiple_secondary_choices_not_supported")
            candidates = tuple(
                card.instance_id
                for zone in effect.target_zones
                for card in self._zone_cards(state, zone)
                if card.instance_id != source_id
            )
            if not candidates:
                return state, RuleEvent("FREE_PLAY_EMPTY", source_id)
            pending = PendingChoice(
                ActionKind.FREE_PLAY,
                source_id,
                tuple(dict.fromkeys(candidates)),
                continuation_index=continuation_index,
                remaining_effect_passes=remaining_effect_passes,
                condition_stance=(
                    condition_state.stance
                    if (continuation_index is not None or remaining_effect_passes > 0)
                    and condition_state
                    else None
                ),
                condition_full_power=(
                    condition_state.full_power
                    if (continuation_index is not None or remaining_effect_passes > 0)
                    and condition_state
                    else None
                ),
            )
            return (
                replace(state, pending_choice=pending),
                RuleEvent("REQUEST_SECONDARY", source_id, detail=ActionKind.FREE_PLAY.value),
            )
        if effect.kind is EffectKind.PAID_PLAY_SELECTED:
            if state.pending_choice is not None:
                raise KernelError("multiple_secondary_choices_not_supported")
            candidates = tuple(
                card.instance_id
                for zone in effect.target_zones
                for card in self._zone_cards(state, zone)
                if card.instance_id != source_id
            )
            if not candidates:
                return state, RuleEvent("PAID_PLAY_EMPTY", source_id)
            pending = PendingChoice(
                ActionKind.PAID_PLAY,
                source_id,
                tuple(dict.fromkeys(candidates)),
                continuation_index=continuation_index,
                remaining_effect_passes=remaining_effect_passes,
                condition_stance=(
                    condition_state.stance
                    if (continuation_index is not None or remaining_effect_passes > 0)
                    and condition_state
                    else None
                ),
                condition_full_power=(
                    condition_state.full_power
                    if (continuation_index is not None or remaining_effect_passes > 0)
                    and condition_state
                    else None
                ),
            )
            return (
                replace(state, pending_choice=pending),
                RuleEvent("REQUEST_SECONDARY", source_id, detail=ActionKind.PAID_PLAY.value),
            )
        if effect.kind is EffectKind.FREE_PLAY_ALL:
            if state.pending_choice is not None:
                raise KernelError("multiple_secondary_choices_not_supported")
            candidates = tuple(
                dict.fromkeys(
                    card.instance_id
                    for zone in effect.target_zones
                    for card in self._zone_cards(state, zone)
                    if card.instance_id != source_id
                )
            )
            if not candidates:
                return state, RuleEvent("FREE_PLAY_ALL_EMPTY", source_id)
            stack = state.resolution_stack
            if continuation_index is not None or remaining_effect_passes > 0:
                if source_card_id is None:
                    raise KernelError("card_continuation_definition_missing")
                if condition_state is None:
                    raise KernelError("card_continuation_snapshot_missing")
                stack += (
                    CardEffectsContinuation(
                        source_id=source_id,
                        card_id=source_card_id,
                        continuation_index=(
                            continuation_index
                            if continuation_index is not None
                            else len(self.cards[source_card_id].effects)
                        ),
                        remaining_effect_passes=remaining_effect_passes,
                        condition_stance=condition_state.stance,
                        condition_full_power=condition_state.full_power,
                    ),
                )
            if len(candidates) > 1:
                stack += (
                    FreePlayAllContinuation(
                        source_id=source_id,
                        remaining_instance_ids=candidates[1:],
                    ),
                )
            pending = PendingChoice(
                ActionKind.FREE_PLAY,
                source_id,
                (candidates[0],),
            )
            return (
                replace(
                    state,
                    pending_choice=pending,
                    resolution_stack=stack,
                ),
                RuleEvent("REQUEST_FREE_PLAY_ALL", source_id, len(candidates)),
            )
        if effect.kind is EffectKind.CARD_MODIFICATION:
            assert effect.card_selector is not None
            assert effect.card_modification is not None
            updated, target_ids = self._modify_cards(
                state,
                effect.card_selector,
                effect.card_modification,
                source_id=source_id,
                excluded_instance_id=(
                    trigger_context.used_card_instance_id
                    if trigger_context is not None
                    and trigger_context.exclude_used_card_from_targets
                    else None
                ),
            )
            return (
                updated,
                RuleEvent("CARD_MODIFICATION", source_id, detail="|".join(target_ids)),
            )
        if effect.kind is EffectKind.SCHEDULE:
            if source_card_id is None or definition_effect_index is None:
                raise KernelError("scheduled_effect_source_missing")
            scheduled = ScheduledEffectRef(
                source_id=source_id,
                card_id=source_card_id,
                definition_effect_index=definition_effect_index,
                source_kind=source_kind,
                turns_until_fire=effect.delay_turns,
                trigger_phase=effect.schedule_phase,
                remaining_uses=effect.schedule_limit,
                remaining_turns=effect.schedule_ttl,
                direct_on_trigger=(
                    effect.schedule_phase is SchedulePhase.TURN
                    and effect.schedule_limit == 1
                ),
            )
            return (
                replace(state, scheduled_effects=state.scheduled_effects + (scheduled,)),
                RuleEvent(
                    "SCHEDULE_EFFECT",
                    source_id,
                    effect.delay_turns,
                    (
                        f"{source_card_id}:{effect.schedule_phase.value}:"
                        f"{effect.schedule_limit}:{effect.schedule_ttl}"
                    ),
                ),
            )
        raise KernelError(f"unsupported_effect:{effect.kind.value}")

    def _apply_stance_departure_benefits(
        self,
        state: StrategyState,
        *,
        old_stance: Stance,
        target: Stance,
    ) -> StrategyState:
        conserve_level = _conserve_level(old_stance)
        if conserve_level and target is not Stance.LEISURE:
            base = 8 if conserve_level >= 2 else 5
            bonus = state.passion_buff + sum(
                buff.amount for buff in state.enthusiasm_bonus_buffs
            )
            multiplier = (
                1.0
                + state.passion_gain_bonus_pct / 100.0
                + sum(buff.amount for buff in state.enthusiasm_multiplier_buffs)
            )
            return replace(
                state,
                passion=state.passion + (bonus + base) * multiplier,
                genki=state.genki + (5 if conserve_level >= 2 else 0),
                actions_remaining=state.actions_remaining + 1,
            )
        if old_stance is Stance.LEISURE:
            updated = replace(
                state,
                genki=state.genki + 5,
                actions_remaining=state.actions_remaining + 1,
                passion=state.passion + (0 if target is Stance.FULL_POWER else 10),
            )
            if target is Stance.FULL_POWER:
                updated, _target_ids = self._modify_cards(
                    updated,
                    CardSelector(all_cards=True),
                    CardModification(score_bonus=10),
                )
            return updated
        return state

    @staticmethod
    def _is_counted_stance_change(old_stance: Stance, target: Stance) -> bool:
        return _is_counted_stance_change(old_stance, target)

    @staticmethod
    def _increment_stance_counts(
        state: StrategyState,
        *,
        direct_effect: bool,
    ) -> StrategyState:
        changes: dict[str, int] = {}
        if state.stance in {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}:
            changes["strength_times"] = state.strength_times + 1
        elif state.stance in {Stance.CONSERVE_1, Stance.CONSERVE_2}:
            changes["preservation_times"] = state.preservation_times + 1
        elif state.stance is Stance.FULL_POWER:
            changes["full_power_times"] = state.full_power_times + 1
        elif state.stance is Stance.LEISURE:
            changes["leisure_times"] = state.leisure_times + 1
        if direct_effect:
            changes["stance_changed_by_direct_effect_times"] = (
                state.stance_changed_by_direct_effect_times + 1
            )
        return replace(state, **changes)

    def _trigger_scalar_stance_event(
        self,
        state: StrategyState,
        events: tuple[RuleEvent, ...],
        *,
        condition_is_direct: bool,
        increment_direct: bool,
    ) -> tuple[StrategyState, tuple[RuleEvent, ...]]:
        triggered = self._trigger_event_phase(
            (TransitionBranch(1.0, state, events),),
            SchedulePhase.STANCE_CHANGED,
            trigger_context=EffectTriggerContext(
                is_direct_effect=condition_is_direct,
            ),
        )
        if len(triggered) != 1 or not math.isclose(triggered[0].probability, 1.0):
            raise KernelError("turn_start_stance_event_must_be_deterministic")
        child = triggered[0]
        return (
            self._increment_stance_counts(
                child.state,
                direct_effect=increment_direct,
            ),
            child.events,
        )

    @staticmethod
    def _effective_stamina_cost(
        state: StrategyState,
        card: CardRef,
        definition: CardDefinition,
    ) -> int:
        kind = PythonNiaProKernel._declared_cost_kind(definition)
        if state.nullify_cost_cards > 0 and kind in {
            CardCostKind.NORMAL,
            CardCostKind.STAMINA,
        }:
            return 0
        if kind is CardCostKind.FULL_POWER:
            return 0
        generic_cost_delta = (
            card.stamina_cost_delta if kind is CardCostKind.NORMAL else 0
        )
        cost = max(
            0,
            definition.stamina_cost
            + generic_cost_delta
            + card.typed_cost_delta,
        )
        if kind not in {CardCostKind.NORMAL, CardCostKind.STAMINA}:
            return cost
        if state.stance in {Stance.AGGRESSIVE_1, Stance.AGGRESSIVE_2}:
            cost *= 2
        elif state.stance is Stance.CONSERVE_1:
            cost *= 0.5
        elif state.stance is Stance.CONSERVE_2:
            cost *= 0.25
        elif state.stance is Stance.LEISURE:
            cost = 0
        if state.half_cost_turns > 0:
            cost *= 0.5
        if state.double_cost_turns > 0:
            cost *= 2
        return math.ceil(cost)

    @staticmethod
    def _effective_full_power_cost(card: CardRef, definition: CardDefinition) -> int:
        if PythonNiaProKernel._declared_cost_kind(definition) is not CardCostKind.FULL_POWER:
            return 0
        return max(0, definition.full_power_cost + card.typed_cost_delta)

    @staticmethod
    def _card_effect_passes(state: StrategyState, definition: CardDefinition) -> int:
        return 2 if state.double_card_effects_remaining > 0 and definition.rarity != "L" else 1

    @staticmethod
    def _move_card_for_use(
        state: StrategyState,
        card: CardRef,
        definition: CardDefinition,
    ) -> tuple[StrategyState, RuleEvent]:
        zone_entries = (
            (CardZone.HAND, state.deck.hand),
            (CardZone.DRAW_PILE, state.deck.draw_pile),
            (CardZone.DISCARD, state.deck.discard),
            (CardZone.REMOVED, state.deck.removed),
            (CardZone.HELD, state.deck.held),
        )
        source_zones = tuple(
            zone
            for zone, cards in zone_entries
            if any(item.instance_id == card.instance_id for item in cards)
        )
        if len(source_zones) != 1:
            raise KernelError("card_use_source_zone_invalid")
        source_zone = source_zones[0]

        def without(cards: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(item for item in cards if item.instance_id != card.instance_id)

        hand = without(state.deck.hand)
        draw_pile = without(state.deck.draw_pile)
        discard = without(state.deck.discard)
        removed = without(state.deck.removed)
        held = without(state.deck.held)
        if definition.limited:
            removed += (card,)
            destination = CardZone.REMOVED
        else:
            discard += (card,)
            destination = CardZone.DISCARD
        deck = state.deck.replace_zones(
            hand=hand,
            draw_pile=draw_pile,
            discard=discard,
            removed=removed,
            held=held,
        )
        return (
            replace(state, deck=deck),
            RuleEvent("MOVE_CARD", card.instance_id, detail=f"{source_zone.value}->{destination.value}"),
        )

    @staticmethod
    def _move_card_to_discard(
        state: StrategyState,
        card: CardRef,
    ) -> tuple[StrategyState, RuleEvent]:
        zone_entries = (
            (CardZone.HAND, state.deck.hand),
            (CardZone.DRAW_PILE, state.deck.draw_pile),
            (CardZone.DISCARD, state.deck.discard),
            (CardZone.REMOVED, state.deck.removed),
            (CardZone.HELD, state.deck.held),
        )
        source_zones = tuple(
            zone
            for zone, cards in zone_entries
            if any(item.instance_id == card.instance_id for item in cards)
        )
        if len(source_zones) != 1:
            raise KernelError("card_discard_source_zone_invalid")
        source_zone = source_zones[0]

        def without(cards: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(item for item in cards if item.instance_id != card.instance_id)

        deck = state.deck.replace_zones(
            hand=without(state.deck.hand),
            draw_pile=without(state.deck.draw_pile),
            discard=without(state.deck.discard) + (card,),
            removed=without(state.deck.removed),
            held=without(state.deck.held),
        )
        return (
            replace(state, deck=deck),
            RuleEvent(
                "MOVE_CARD",
                card.instance_id,
                detail=f"{source_zone.value}->{CardZone.DISCARD.value}",
            ),
        )

    @staticmethod
    def _card_by_instance(state: StrategyState, instance_id: str) -> CardRef | None:
        return next((card for card in state.deck.all_cards if card.instance_id == instance_id), None)

    def _card_event_context(
        self,
        state: StrategyState,
        instance_id: str,
        *,
        exclude_used_card_from_targets: bool,
    ) -> EffectTriggerContext:
        card = self._card_by_instance(state, instance_id)
        if card is None:
            raise KernelError("card_event_source_missing")
        definition = self.cards.get(card.card_id)
        if definition is None:
            raise KernelError("card_event_definition_missing")
        return EffectTriggerContext(
            used_card_instance_id=instance_id,
            used_card_id=card.card_id,
            used_card_base_id=definition.base_card_id or definition.card_id,
            used_card_type=definition.card_type,
            used_card_rarity=definition.rarity,
            exclude_used_card_from_targets=exclude_used_card_from_targets,
        )

    def _moved_card_event_context(
        self,
        state: StrategyState,
        instance_id: str,
    ) -> EffectTriggerContext:
        card = self._card_by_instance(state, instance_id)
        if card is None:
            raise KernelError("moved_card_event_source_missing")
        definition = self.cards.get(card.card_id)
        if definition is None:
            raise KernelError("moved_card_event_definition_missing")
        return EffectTriggerContext(
            is_direct_effect=True,
            moved_card_instance_id=instance_id,
            moved_card_id=card.card_id,
            moved_card_base_id=definition.base_card_id or definition.card_id,
        )

    def _complete_after_card_events(
        self,
        branches: tuple[TransitionBranch, ...],
    ) -> tuple[TransitionBranch, ...]:
        if any(branch.state.pending_choice is not None for branch in branches):
            return branches
        result = branches
        while any(branch.state.pending_after_card_ids for branch in result):
            expanded: list[TransitionBranch] = []
            for branch in result:
                if not branch.state.pending_after_card_ids:
                    expanded.append(branch)
                    continue
                source_id = branch.state.pending_after_card_ids[-1]
                state = replace(
                    branch.state,
                    pending_after_card_ids=branch.state.pending_after_card_ids[:-1],
                )
                expanded.extend(
                    self._trigger_event_phase(
                        (replace(branch, state=state),),
                        SchedulePhase.AFTER_CARD_USED,
                        trigger_context=self._card_event_context(
                            state,
                            source_id,
                            exclude_used_card_from_targets=False,
                        ),
                    )
                )
            result = tuple(expanded)
        return result

    def _complete_one_after_card_event(
        self,
        branch: TransitionBranch,
    ) -> tuple[TransitionBranch, ...]:
        if branch.state.pending_choice is not None:
            return (branch,)
        if not branch.state.pending_after_card_ids:
            return (branch,)
        source_id = branch.state.pending_after_card_ids[-1]
        state = replace(
            branch.state,
            pending_after_card_ids=branch.state.pending_after_card_ids[:-1],
        )
        return self._trigger_event_phase(
            (replace(branch, state=state),),
            SchedulePhase.AFTER_CARD_USED,
            trigger_context=self._card_event_context(
                state,
                source_id,
                exclude_used_card_from_targets=False,
            ),
        )

    def _finalize_card_resolution(
        self,
        branches: tuple[TransitionBranch, ...],
    ) -> tuple[TransitionBranch, ...]:
        resolved: list[TransitionBranch] = []
        for branch in branches:
            if branch.state.pending_choice is not None:
                resolved.append(branch)
                continue
            if branch.state.resolution_stack:
                for after in self._complete_one_after_card_event(branch):
                    if after.state.pending_choice is not None:
                        resolved.append(after)
                    else:
                        resolved.extend(self._resume_resolution_stack((after,)))
                continue
            resolved.extend(self._complete_after_card_events((branch,)))
        return tuple(resolved)

    def _resume_resolution_stack(
        self,
        branches: tuple[TransitionBranch, ...],
    ) -> tuple[TransitionBranch, ...]:
        resolved: list[TransitionBranch] = []
        for branch in branches:
            if branch.state.pending_choice is not None or not branch.state.resolution_stack:
                resolved.append(branch)
                continue
            frame = branch.state.resolution_stack[-1]
            state = replace(
                branch.state,
                resolution_stack=branch.state.resolution_stack[:-1],
            )
            current = replace(branch, state=state)
            if isinstance(frame, CardEffectsContinuation):
                definition = self.cards.get(frame.card_id)
                if definition is None:
                    raise KernelError("card_continuation_definition_missing")
                condition_state = replace(
                    state,
                    stance=frame.condition_stance,
                    full_power=frame.condition_full_power,
                )
                continued = self._apply_definition_effects(
                    (current,),
                    definition,
                    source_id=frame.source_id,
                    condition_state=condition_state,
                    start_index=frame.continuation_index,
                    remaining_full_passes=frame.remaining_effect_passes,
                )
                resolved.extend(self._finalize_card_resolution(continued))
            elif isinstance(frame, FreePlayAllContinuation):
                if not frame.remaining_instance_ids:
                    resolved.extend(self._resume_resolution_stack((current,)))
                    continue
                next_id = frame.remaining_instance_ids[0]
                stack = state.resolution_stack
                if len(frame.remaining_instance_ids) > 1:
                    stack += (
                        replace(
                            frame,
                            remaining_instance_ids=frame.remaining_instance_ids[1:],
                        ),
                    )
                pending = PendingChoice(
                    ActionKind.FREE_PLAY,
                    frame.source_id,
                    (next_id,),
                )
                resolved.append(
                    replace(
                        current,
                        state=replace(
                            state,
                            pending_choice=pending,
                            resolution_stack=stack,
                        ),
                        events=current.events
                        + (
                            RuleEvent(
                                "REQUEST_FREE_PLAY_ALL_NEXT",
                                frame.source_id,
                                detail=next_id,
                            ),
                        ),
                    )
                )
            elif isinstance(frame, ScheduledEffectsContinuation):
                continued = self._resume_scheduled_continuation(current, frame)
                for child in continued:
                    if child.state.pending_choice is not None:
                        resolved.append(child)
                    else:
                        resolved.extend(self._resume_resolution_stack((child,)))
            elif isinstance(frame, TurnResolutionContinuation):
                resolved.extend(self._advance_turn_resolution(current, frame))
            else:
                raise KernelError("unknown_resolution_continuation")
        return tuple(resolved)

    def _modify_cards(
        self,
        state: StrategyState,
        selector: CardSelector,
        modification: CardModification,
        *,
        source_id: str | None = None,
        excluded_instance_id: str | None = None,
    ) -> tuple[StrategyState, tuple[str, ...]]:
        targetable_zones = (
            CardZone.HAND,
            CardZone.DRAW_PILE,
            CardZone.DISCARD,
            CardZone.HELD,
        )
        if selector.source_only:
            target_ids = tuple(
                card.instance_id
                for card in state.deck.all_cards
                if card.instance_id == source_id
            )
        else:
            zones = targetable_zones if selector.all_cards or not selector.zones else selector.zones
            target_ids = tuple(
                card.instance_id
                for zone in zones
                for card in self._zone_cards(state, zone)
                if card.instance_id != excluded_instance_id
                and self._selector_matches(card, selector)
            )
        target_set = set(target_ids)
        if not target_set:
            return state, ()

        replacements = {
            card.instance_id: replace(
                card,
                score_bonus=card.score_bonus + modification.score_bonus,
                genki_bonus=card.genki_bonus + modification.genki_bonus,
                stamina_cost_delta=card.stamina_cost_delta + modification.stamina_cost_delta,
                typed_cost_delta=card.typed_cost_delta + modification.typed_cost_delta,
                score_repeat_bonus=card.score_repeat_bonus + modification.score_repeat_bonus,
            )
            for card in state.deck.all_cards
            if card.instance_id in target_set
        }

        def updated(zone: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(replacements.get(card.instance_id, card) for card in zone)

        deck = replace(
            state.deck,
            all_cards=updated(state.deck.all_cards),
            hand=updated(state.deck.hand),
            draw_pile=updated(state.deck.draw_pile),
            discard=updated(state.deck.discard),
            removed=updated(state.deck.removed),
            held=updated(state.deck.held),
        )
        deck.validate()
        return replace(state, deck=deck), tuple(dict.fromkeys(target_ids))

    def _upgrade_hand(
        self,
        state: StrategyState,
        *,
        source_id: str,
    ) -> tuple[StrategyState, RuleEvent]:
        replacements: dict[str, CardRef] = {}
        for card in state.deck.hand:
            definition = self.cards.get(card.card_id)
            if definition is None or definition.upgrade_card_id is None:
                continue
            if definition.upgrade_card_id not in self.cards:
                raise KernelError("upgrade_card_definition_missing")
            replacements[card.instance_id] = replace(
                card,
                card_id=definition.upgrade_card_id,
                upgraded=True,
            )
        if not replacements:
            return state, RuleEvent("UPGRADE_HAND_EMPTY", source_id)

        def updated(zone: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(replacements.get(card.instance_id, card) for card in zone)

        deck = replace(
            state.deck,
            all_cards=updated(state.deck.all_cards),
            hand=updated(state.deck.hand),
        )
        deck.validate()
        return (
            replace(state, deck=deck),
            RuleEvent(
                "UPGRADE_HAND",
                source_id,
                detail="|".join(replacements),
            ),
        )

    def _selector_matches(self, card: CardRef, selector: CardSelector) -> bool:
        definition = self.cards.get(card.card_id)
        if definition is None:
            return False
        if selector.card_type is not None and definition.card_type != selector.card_type:
            return False
        if selector.card_id is not None and card.card_id != selector.card_id:
            return False
        if selector.base_card_id is not None and (
            definition.base_card_id or definition.card_id
        ) != selector.base_card_id:
            return False
        if selector.effect_tag is not None and not self._definition_has_effect_tag(definition, selector.effect_tag):
            return False
        if selector.source_type is not None and definition.source_type != selector.source_type:
            return False
        return True

    @staticmethod
    def _definition_has_effect_tag(definition: CardDefinition, effect_tag: str) -> bool:
        return _definition_has_effect_tag(definition, effect_tag)

    @staticmethod
    def _zone_cards(state: StrategyState, zone: CardZone) -> tuple[CardRef, ...]:
        return {
            CardZone.HAND: state.deck.hand,
            CardZone.DRAW_PILE: state.deck.draw_pile,
            CardZone.DISCARD: state.deck.discard,
            CardZone.REMOVED: state.deck.removed,
            CardZone.HELD: state.deck.held,
        }[zone]

    @staticmethod
    def _move_card_to_held(state: StrategyState, instance_id: str) -> tuple[StrategyState, str | None]:
        matches = [card for card in state.deck.zones if card.instance_id == instance_id]
        if len(matches) != 1:
            raise KernelError("hold_card_instance_missing")
        card = matches[0]
        if any(item.instance_id == instance_id for item in state.deck.held):
            return state, None

        def without(zone: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(item for item in zone if item.instance_id != instance_id)

        held = state.deck.held + (card,)
        evicted = held[0] if len(held) > 2 else None
        if evicted is not None:
            held = held[1:]
        discard = without(state.deck.discard)
        if evicted is not None:
            discard += (evicted,)
        deck = state.deck.replace_zones(
            hand=without(state.deck.hand),
            draw_pile=without(state.deck.draw_pile),
            discard=discard,
            removed=without(state.deck.removed),
            held=held,
        )
        return replace(state, deck=deck), evicted.instance_id if evicted is not None else None

    @staticmethod
    def _move_card_to_hand(
        state: StrategyState,
        instance_id: str,
    ) -> tuple[StrategyState, CardZone]:
        source_entries = (
            (CardZone.DRAW_PILE, state.deck.draw_pile),
            (CardZone.DISCARD, state.deck.discard),
            (CardZone.REMOVED, state.deck.removed),
            (CardZone.HELD, state.deck.held),
        )
        matches = tuple(
            (zone, card)
            for zone, cards in source_entries
            for card in cards
            if card.instance_id == instance_id
        )
        if len(matches) != 1:
            raise KernelError("move_card_instance_missing")
        source_zone, card = matches[0]

        def without(zone: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(item for item in zone if item.instance_id != instance_id)

        deck = state.deck.replace_zones(
            hand=state.deck.hand + (card,),
            draw_pile=without(state.deck.draw_pile),
            discard=without(state.deck.discard),
            removed=without(state.deck.removed),
            held=without(state.deck.held),
        )
        return replace(state, deck=deck), source_zone

    def _move_random_to_hand(
        self,
        state: StrategyState,
        selector: CardSelector,
        *,
        source_id: str,
    ) -> tuple[TransitionBranch, ...]:
        candidates = tuple(
            (zone, card)
            for zone in selector.zones
            for card in self._zone_cards(state, zone)
            if self._selector_matches(card, selector)
        )
        if not candidates:
            return (
                TransitionBranch(
                    1.0,
                    state,
                    (RuleEvent("MOVE_RANDOM_TO_HAND_EMPTY", source_id),),
                ),
            )

        grouped: dict[tuple[CardZone, object], tuple[CardZone, CardRef, int]] = {}
        for zone, card in candidates:
            identity: object = card.instance_id
            if zone is not CardZone.DRAW_PILE or state.deck.knowledge is not DeckKnowledge.KNOWN_ORDER:
                identity = card.semantic_key
            key = (zone, identity)
            representative_zone, representative, count = grouped.get(key, (zone, card, 0))
            grouped[key] = (representative_zone, representative, count + 1)

        total = len(candidates)
        branches: list[TransitionBranch] = []
        for zone, card, count in grouped.values():
            source_zone = self._zone_cards(state, zone)
            removed = False
            remainder: list[CardRef] = []
            for item in source_zone:
                if not removed and item.instance_id == card.instance_id:
                    removed = True
                    continue
                remainder.append(item)
            replacements = {
                CardZone.DRAW_PILE: {"draw_pile": tuple(remainder)},
                CardZone.DISCARD: {"discard": tuple(remainder)},
                CardZone.REMOVED: {"removed": tuple(remainder)},
                CardZone.HELD: {"held": tuple(remainder)},
            }
            zone_update = replacements.get(zone)
            if zone_update is None:
                raise KernelError(f"move_random_source_zone_unsupported:{zone.value}")
            deck = state.deck.replace_zones(hand=state.deck.hand + (card,), **zone_update)
            moved_state = replace(state, deck=deck)
            branches.extend(
                self._trigger_event_phase(
                    (
                        TransitionBranch(
                            count / total,
                            moved_state,
                            (
                                RuleEvent(
                                    "MOVE_RANDOM_TO_HAND",
                                    source_id,
                                    detail=(
                                        f"{zone.value}->{CardZone.HAND.value}:"
                                        f"{card.instance_id}"
                                    ),
                                ),
                            ),
                        ),
                    ),
                    SchedulePhase.CARD_MOVED_TO_HAND,
                    trigger_context=self._moved_card_event_context(
                        moved_state,
                        card.instance_id,
                    ),
                )
            )
        return tuple(branches)

    def _free_play_random(
        self,
        state: StrategyState,
        selector: CardSelector,
        *,
        source_id: str,
    ) -> tuple[TransitionBranch, ...]:
        candidates = tuple(
            card
            for zone in (
                CardZone.DRAW_PILE,
                CardZone.HAND,
                CardZone.DISCARD,
                CardZone.REMOVED,
                CardZone.HELD,
            )
            for card in self._zone_cards(state, zone)
            if card.instance_id != source_id or zone is CardZone.REMOVED
            if self._selector_matches(card, selector)
        )
        if not candidates:
            return (
                TransitionBranch(
                    1.0,
                    state,
                    (RuleEvent("FREE_PLAY_RANDOM_EMPTY", source_id),),
                ),
            )
        total = len(candidates)
        branches: list[TransitionBranch] = []
        for card in candidates:
            for played in self._execute_inline_free_card(
                state,
                card,
                source_id=source_id,
            ):
                branches.append(
                    replace(
                        played,
                        probability=played.probability / total,
                    )
                )
        return self._merge_probability_branches(tuple(branches))

    def _execute_inline_free_card(
        self,
        state: StrategyState,
        card: CardRef,
        *,
        source_id: str,
    ) -> tuple[TransitionBranch, ...]:
        definition = self.cards.get(card.card_id)
        if definition is None:
            raise KernelError("inline_free_card_definition_missing")
        before = self._trigger_event_phase(
            (TransitionBranch(1.0, state, ()),),
            SchedulePhase.BEFORE_CARD_USED,
            trigger_context=self._card_event_context(
                state,
                card.instance_id,
                exclude_used_card_from_targets=False,
            ),
            condition_state=state,
        )
        prepared: list[TransitionBranch] = []
        for branch in before:
            if branch.state.pending_choice is not None:
                raise KernelError("inline_free_before_card_choice_unsupported")
            condition_matches = (
                definition.required_stance in {None, branch.state.stance}
                and _condition_matches(
                    branch.state,
                    definition.condition,
                    self.cards,
                )
            )
            working = replace(
                branch.state,
                pending_after_card_ids=branch.state.pending_after_card_ids
                + (card.instance_id,),
            )
            working, zone_event = self._move_card_for_use(
                working,
                card,
                definition,
            )
            current = TransitionBranch(
                branch.probability,
                working,
                branch.events
                + (
                    RuleEvent("FREE_PLAY_RANDOM", card.instance_id, detail=source_id),
                    zone_event,
                ),
            )
            card_used = self._trigger_event_phase(
                (current,),
                SchedulePhase.CARD_USED,
                trigger_context=self._card_event_context(
                    working,
                    card.instance_id,
                    exclude_used_card_from_targets=True,
                ),
                condition_state=branch.state,
            )
            if not condition_matches:
                prepared.extend(
                    replace(
                        child,
                        events=child.events
                        + (RuleEvent("FREE_PLAY_CONDITION_SKIPPED", card.instance_id),),
                    )
                    for child in card_used
                )
                continue
            effect_passes = self._card_effect_passes(branch.state, definition)
            if effect_passes > 1:
                card_used = tuple(
                    replace(
                        child,
                        state=replace(
                            child.state,
                            double_card_effects_remaining=(
                                child.state.double_card_effects_remaining - 1
                            ),
                        ),
                        events=child.events
                        + (RuleEvent("CONSUME_DOUBLE_CARD_EFFECT", card.instance_id, 1),),
                    )
                    for child in card_used
                )
            prepared.extend(
                self._apply_definition_effects(
                    card_used,
                    definition,
                    source_id=card.instance_id,
                    condition_state=branch.state,
                    remaining_full_passes=effect_passes - 1,
                )
            )

        resolved: list[TransitionBranch] = []
        for branch in prepared:
            if branch.state.pending_choice is not None:
                raise KernelError("inline_free_card_secondary_unsupported")
            after = self._complete_one_after_card_event(branch)
            if any(child.state.pending_choice is not None for child in after):
                raise KernelError("inline_free_after_card_choice_unsupported")
            resolved.extend(after)
        return tuple(resolved)

    def _exchange_hand(self, state: StrategyState, *, source_id: str) -> tuple[TransitionBranch, ...]:
        exchange_count = len(state.deck.hand)
        deck = state.deck.replace_zones(
            hand=(),
            discard=state.deck.discard + state.deck.hand,
        )
        branches = (
            TransitionBranch(
                1.0,
                replace(state, deck=deck),
                (RuleEvent("EXCHANGE_HAND", source_id, exchange_count),),
            ),
        )
        for _index in range(exchange_count):
            expanded: list[TransitionBranch] = []
            for branch in branches:
                for draw in self._draw_one(branch.state, source_id=source_id):
                    expanded.append(
                        TransitionBranch(
                            probability=branch.probability * draw.probability,
                            state=draw.state,
                            events=branch.events + draw.events,
                        )
                    )
            branches = self._merge_probability_branches(tuple(expanded))
        return branches

    def _add_card_to_deck(
        self,
        state: StrategyState,
        card_id: str,
        *,
        source_id: str,
    ) -> tuple[TransitionBranch, ...]:
        definition = self.cards.get(card_id)
        if definition is None:
            raise KernelError(f"generated_card_definition_missing:{card_id}")
        if state.deck.knowledge not in {DeckKnowledge.KNOWN_ORDER, DeckKnowledge.KNOWN_COMPOSITION}:
            raise KernelError(f"add_card_requires_known_deck:{state.deck.knowledge.value}")
        instance_id = f"task275-created-{state.created_card_count}"
        if any(card.instance_id == instance_id for card in state.deck.all_cards):
            raise KernelError(f"generated_card_instance_collision:{instance_id}")
        card = CardRef(instance_id=instance_id, card_id=card_id)
        all_cards = state.deck.all_cards + (card,)
        next_count = state.created_card_count + 1
        if state.deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION:
            deck = state.deck.replace_zones(
                all_cards=all_cards,
                draw_pile=state.deck.draw_pile + (card,),
            )
            return (
                TransitionBranch(
                    1.0,
                    replace(state, deck=deck, created_card_count=next_count),
                    (RuleEvent("ADD_CARD_TO_DECK", source_id, detail=f"{card_id}:{instance_id}"),),
                ),
            )

        probability = 1.0 / (len(state.deck.draw_pile) + 1)
        branches: list[TransitionBranch] = []
        for insert_position in range(len(state.deck.draw_pile) + 1):
            draw_pile = (
                state.deck.draw_pile[:insert_position]
                + (card,)
                + state.deck.draw_pile[insert_position:]
            )
            deck = state.deck.replace_zones(all_cards=all_cards, draw_pile=draw_pile)
            branches.append(
                TransitionBranch(
                    probability,
                    replace(state, deck=deck, created_card_count=next_count),
                    (
                        RuleEvent(
                            "ADD_CARD_TO_DECK",
                            source_id,
                            detail=f"{card_id}:{instance_id}:position:{insert_position}",
                        ),
                    ),
                )
            )
        return tuple(branches)

    @staticmethod
    def _add_timed_modifier(
        modifiers: tuple[TimedModifier, ...],
        *,
        amount: float,
        duration: int | None,
    ) -> tuple[TimedModifier, ...]:
        for index, modifier in enumerate(modifiers):
            if modifier.remaining_turns == duration:
                return (
                    modifiers[:index]
                    + (replace(modifier, amount=modifier.amount + amount),)
                    + modifiers[index + 1 :]
                )
        return modifiers + (TimedModifier(amount=amount, remaining_turns=duration),)

    @staticmethod
    def _decrement_timed_modifiers(
        modifiers: tuple[TimedModifier, ...],
    ) -> tuple[TimedModifier, ...]:
        result: list[TimedModifier] = []
        for modifier in modifiers:
            if modifier.fresh:
                result.append(replace(modifier, fresh=False))
            elif modifier.remaining_turns is None:
                result.append(modifier)
            elif modifier.remaining_turns > 1:
                result.append(replace(modifier, remaining_turns=modifier.remaining_turns - 1))
        return tuple(result)

    def _draw_one(self, state: StrategyState, *, source_id: str) -> tuple[TransitionBranch, ...]:
        deck = state.deck
        if not deck.draw_pile and deck.discard:
            deck = deck.replace_zones(draw_pile=deck.discard, discard=(), knowledge=DeckKnowledge.KNOWN_COMPOSITION)
            state = replace(state, deck=deck)
        if not deck.draw_pile:
            return (TransitionBranch(1.0, state, (RuleEvent("DRAW_EMPTY", source_id),)),)
        if deck.knowledge is DeckKnowledge.KNOWN_ORDER:
            candidates = ((deck.draw_pile[0], 1.0),)
        elif deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION:
            groups: dict[
                tuple[str, bool, int, int, int, int, int],
                tuple[CardRef, int],
            ] = {}
            for card in deck.draw_pile:
                key = card.semantic_key
                representative, count = groups.get(key, (card, 0))
                groups[key] = (representative, count + 1)
            total = len(deck.draw_pile)
            candidates = tuple((representative, count / total) for representative, count in groups.values())
        else:
            raise KernelError(f"draw_requires_known_deck:{deck.knowledge.value}")
        branches: list[TransitionBranch] = []
        for card, probability in candidates:
            removed = False
            next_draw: list[CardRef] = []
            for item in deck.draw_pile:
                if not removed and item.instance_id == card.instance_id:
                    removed = True
                    continue
                next_draw.append(item)
            next_deck = deck.replace_zones(hand=deck.hand + (card,), draw_pile=tuple(next_draw))
            moved_state = replace(state, deck=next_deck)
            branches.extend(
                self._trigger_event_phase(
                    (
                        TransitionBranch(
                            probability,
                            moved_state,
                            (RuleEvent("DRAW", source_id, detail=card.instance_id),),
                        ),
                    ),
                    SchedulePhase.CARD_MOVED_TO_HAND,
                    trigger_context=self._moved_card_event_context(
                        moved_state,
                        card.instance_id,
                    ),
                )
            )
        return tuple(branches)

    @staticmethod
    def _canonical_probability_state(state: StrategyState) -> StrategyState:
        def ordered(zone: tuple[CardRef, ...]) -> tuple[CardRef, ...]:
            return tuple(sorted(zone, key=lambda card: (*card.semantic_key, card.instance_id)))

        draw_pile = state.deck.draw_pile
        if state.deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION:
            draw_pile = ordered(draw_pile)
        deck = replace(
            state.deck,
            all_cards=ordered(state.deck.all_cards),
            hand=ordered(state.deck.hand),
            draw_pile=draw_pile,
            discard=ordered(state.deck.discard),
            removed=ordered(state.deck.removed),
            held=ordered(state.deck.held),
        )
        return replace(state, deck=deck)

    def _merge_probability_branches(
        self,
        branches: tuple[TransitionBranch, ...],
    ) -> tuple[TransitionBranch, ...]:
        merged: dict[StrategyState, TransitionBranch] = {}
        for branch in branches:
            key = self._canonical_probability_state(branch.state)
            previous = merged.get(key)
            if previous is None:
                merged[key] = replace(branch, state=key)
            else:
                merged[key] = replace(previous, probability=previous.probability + branch.probability)
        return tuple(merged.values())

    def _end_turn(self, state: StrategyState, *, plan_type: PlanType) -> tuple[TransitionBranch, ...]:
        if state.pending_choice is not None:
            raise KernelError("secondary_choice_pending")
        if state.terminal:
            raise KernelError("lesson_already_terminal")
        branches = (
            TransitionBranch(
                1.0,
                state,
                (RuleEvent("END_TURN", "system"),),
            ),
        )
        branches = self._trigger_event_phase(branches, SchedulePhase.END_OF_TURN)
        expanded: list[TransitionBranch] = []
        for branch in branches:
            for completed in self._complete_end_turn(branch.state, plan_type=plan_type):
                expanded.append(
                    TransitionBranch(
                        branch.probability * completed.probability,
                        completed.state,
                        branch.events + completed.events,
                    )
                )
        merged = self._merge_probability_branches(tuple(expanded))
        self._validate_probability_mass(merged)
        return merged

    def _complete_end_turn(
        self,
        state: StrategyState,
        *,
        plan_type: PlanType,
    ) -> tuple[TransitionBranch, ...]:
        score = state.score
        events: tuple[RuleEvent, ...] = ()
        if plan_type is PlanType.LOGIC and state.good_impression_turns > 0:
            gain = math.floor(state.good_impression_turns * state.score_multiplier)
            score += gain
            events += (RuleEvent("GOOD_IMPRESSION_TICK", "system", gain),)
        statuses = tuple(
            status if status.remaining_turns is None else replace(status, remaining_turns=status.remaining_turns - 1)
            for status in state.statuses
            if status.remaining_turns is None or status.remaining_turns > 1
        )
        multiplier = state.future_score_multipliers[0] if state.future_score_multipliers else state.score_multiplier
        future = state.future_score_multipliers[1:] if state.future_score_multipliers else ()
        stance_lock_turns = (
            state.stance_lock_turns
            if state.stance_lock_fresh
            else max(0, state.stance_lock_turns - 1)
        )
        turns_remaining = state.turns_remaining - 1
        next_state = replace(
            state,
            turn=state.turn + 1,
            turns_remaining=turns_remaining,
            score=score,
            actions_remaining=0 if turns_remaining == 0 else 1,
            passion=0,
            good_condition_turns=max(0, state.good_condition_turns - 1),
            excellent_condition_turns=max(0, state.excellent_condition_turns - 1),
            good_impression_turns=max(0, state.good_impression_turns - 1),
            half_cost_turns=(
                state.half_cost_turns
                if state.half_cost_fresh
                else max(0, state.half_cost_turns - 1)
            ),
            half_cost_fresh=False,
            double_cost_turns=(
                state.double_cost_turns
                if state.double_cost_fresh
                else max(0, state.double_cost_turns - 1)
            ),
            double_cost_fresh=False,
            stance_lock_turns=stance_lock_turns,
            stance_lock_fresh=False,
            enthusiasm_bonus_buffs=self._decrement_timed_modifiers(
                state.enthusiasm_bonus_buffs
            ),
            enthusiasm_multiplier_buffs=self._decrement_timed_modifiers(
                state.enthusiasm_multiplier_buffs
            ),
            full_power_charge_buffs=self._decrement_timed_modifiers(
                state.full_power_charge_buffs
            ),
            score_buffs=self._decrement_timed_modifiers(state.score_buffs),
            statuses=statuses,
            score_multiplier=multiplier,
            future_score_multipliers=future,
        )
        if next_state.terminal:
            next_state = replace(next_state, scheduled_effects=())
            branches = (TransitionBranch(1.0, next_state, events),)
            self._validate_probability_mass(branches)
            return branches
        entered_full_power = False
        next_state = replace(next_state, previous_stance=Stance.NEUTRAL)
        if next_state.stance is Stance.FULL_POWER:
            next_state = replace(
                next_state,
                stance=Stance.NEUTRAL,
                previous_stance=Stance.FULL_POWER,
            )
            events += (RuleEvent("EXIT_FULL_POWER", "system"),)
            next_state, events = self._trigger_scalar_stance_event(
                next_state,
                events,
                condition_is_direct=False,
                increment_direct=False,
            )
        if (
            next_state.stance_lock_turns == 0
            and next_state.full_power >= FULL_POWER_ACTIVATION_THRESHOLD
        ):
            old_stance = next_state.stance
            next_state = replace(
                next_state,
                stance=Stance.FULL_POWER,
                previous_stance=old_stance,
                full_power=next_state.full_power - FULL_POWER_ACTIVATION_THRESHOLD,
            )
            next_state = self._apply_stance_departure_benefits(
                next_state,
                old_stance=old_stance,
                target=Stance.FULL_POWER,
            )
            entered_full_power = True
            events += (
                RuleEvent(
                    "ENTER_FULL_POWER",
                    "system",
                    FULL_POWER_ACTIVATION_THRESHOLD,
                ),
            )
            next_state, events = self._trigger_scalar_stance_event(
                next_state,
                events,
                condition_is_direct=True,
                increment_direct=False,
            )
        active_scheduled = tuple(
            replace(
                effect,
                remaining_turns=(
                    effect.remaining_turns - 1
                    if effect.remaining_turns is not None
                    else None
                ),
            )
            for effect in next_state.scheduled_effects
            if effect.remaining_turns != 0
        )
        due = tuple(
            effect
            for effect in active_scheduled
            if effect.trigger_phase not in IMMEDIATE_SCHEDULE_PHASES
            and effect.turns_until_fire == 1
        )
        future_scheduled = tuple(
            (
                effect
                if effect.trigger_phase in IMMEDIATE_SCHEDULE_PHASES
                else replace(effect, turns_until_fire=effect.turns_until_fire - 1)
            )
            for effect in active_scheduled
            if effect.trigger_phase in IMMEDIATE_SCHEDULE_PHASES
            or effect.turns_until_fire > 1
        )
        next_state = replace(next_state, scheduled_effects=future_scheduled)
        frame = TurnResolutionContinuation(
            plan_type=plan_type,
            stage=TurnResolutionStage.BEFORE_START,
            entered_full_power=entered_full_power,
            due_before_start=tuple(
                effect
                for effect in due
                if effect.trigger_phase is SchedulePhase.BEFORE_START_OF_TURN
            ),
            due_start=tuple(
                effect
                for effect in due
                if effect.trigger_phase is SchedulePhase.START_OF_TURN
            ),
            due_after_start=tuple(
                effect
                for effect in due
                if effect.trigger_phase is SchedulePhase.AFTER_START_OF_TURN
            ),
            due_turn=tuple(
                effect
                for effect in due
                if effect.trigger_phase is SchedulePhase.TURN
            ),
        )
        next_state = replace(
            next_state,
            resolution_stack=next_state.resolution_stack + (frame,),
        )
        branches = self._resume_resolution_stack(
            (TransitionBranch(1.0, next_state, events),)
        )
        self._validate_probability_mass(branches)
        return branches

    def _advance_turn_resolution(
        self,
        branch: TransitionBranch,
        frame: TurnResolutionContinuation,
    ) -> tuple[TransitionBranch, ...]:
        if frame.stage is TurnResolutionStage.COMPLETE:
            return (branch,)
        if frame.stage is TurnResolutionStage.BEFORE_START:
            return self._run_turn_scheduled_stage(
                branch,
                frame.due_before_start,
                replace(frame, stage=TurnResolutionStage.START),
            )
        if frame.stage is TurnResolutionStage.START:
            return self._run_turn_scheduled_stage(
                branch,
                frame.due_start,
                replace(frame, stage=TurnResolutionStage.REFILL_SETUP),
            )
        if frame.stage is TurnResolutionStage.REFILL_SETUP:
            draws = max(
                0,
                branch.state.hand_limit - len(branch.state.deck.hand),
            )
            next_frame = replace(
                frame,
                stage=TurnResolutionStage.REFILL,
                draws_remaining=draws,
            )
            return self._resume_with_frame(branch, next_frame)
        if frame.stage is TurnResolutionStage.REFILL:
            if (
                frame.draws_remaining < 1
                or len(branch.state.deck.hand) >= branch.state.hand_limit
            ):
                return self._resume_with_frame(
                    branch,
                    replace(
                        frame,
                        stage=TurnResolutionStage.HELD_RETURN,
                        draws_remaining=0,
                    ),
                )
            next_frame = replace(
                frame,
                draws_remaining=frame.draws_remaining - 1,
            )
            state = replace(
                branch.state,
                resolution_stack=branch.state.resolution_stack + (next_frame,),
            )
            drawn = tuple(
                TransitionBranch(
                    branch.probability * child.probability,
                    child.state,
                    branch.events + child.events,
                )
                for child in self._draw_one(state, source_id="system")
            )
            return self._continue_resolution_after_operation(drawn)
        if frame.stage is TurnResolutionStage.HELD_RETURN:
            if not frame.entered_full_power:
                return self._resume_with_frame(
                    branch,
                    replace(frame, stage=TurnResolutionStage.AFTER_START),
                )
            if not frame.held_return_ids:
                moved = tuple(reversed(branch.state.deck.held))
                state = replace(
                    branch.state,
                    actions_remaining=branch.state.actions_remaining + 1,
                    deck=branch.state.deck.replace_zones(
                        hand=branch.state.deck.hand + moved,
                        held=(),
                    ),
                )
                prepared = replace(
                    branch,
                    state=state,
                    events=branch.events
                    + (
                        RuleEvent(
                            "FULL_POWER_READY",
                            "system",
                            len(moved),
                        ),
                    ),
                )
                if not moved:
                    return self._resume_with_frame(
                        prepared,
                        replace(
                            frame,
                            stage=TurnResolutionStage.AFTER_START,
                        ),
                    )
                return self._resume_with_frame(
                    prepared,
                    replace(frame, held_return_ids=tuple(card.instance_id for card in moved)),
                )
            moved_id = frame.held_return_ids[0]
            next_frame = (
                replace(frame, held_return_ids=frame.held_return_ids[1:])
                if len(frame.held_return_ids) > 1
                else replace(
                    frame,
                    stage=TurnResolutionStage.AFTER_START,
                    held_return_ids=(),
                )
            )
            state = replace(
                branch.state,
                resolution_stack=branch.state.resolution_stack + (next_frame,),
            )
            moved = replace(
                branch,
                state=state,
                events=branch.events
                + (
                    RuleEvent(
                        "MOVE_CARD",
                        moved_id,
                        detail=f"{CardZone.HELD.value}->{CardZone.HAND.value}",
                    ),
                ),
            )
            triggered = self._trigger_event_phase(
                (moved,),
                SchedulePhase.CARD_MOVED_TO_HAND,
                trigger_context=self._moved_card_event_context(state, moved_id),
            )
            return self._continue_resolution_after_operation(triggered)
        if frame.stage is TurnResolutionStage.AFTER_START:
            return self._run_turn_scheduled_stage(
                branch,
                frame.due_after_start,
                replace(frame, stage=TurnResolutionStage.TURN),
            )
        if frame.stage is TurnResolutionStage.TURN:
            return self._run_turn_scheduled_stage(
                branch,
                frame.due_turn,
                replace(frame, stage=TurnResolutionStage.COMPLETE),
            )
        raise KernelError("unknown_turn_resolution_stage")

    def _run_turn_scheduled_stage(
        self,
        branch: TransitionBranch,
        due: tuple[ScheduledEffectRef, ...],
        next_frame: TurnResolutionContinuation,
    ) -> tuple[TransitionBranch, ...]:
        state = replace(
            branch.state,
            resolution_stack=branch.state.resolution_stack + (next_frame,),
        )
        triggered = self._trigger_scheduled_effects(
            (replace(branch, state=state),),
            due,
        )
        return self._continue_resolution_after_operation(triggered)

    def _resume_with_frame(
        self,
        branch: TransitionBranch,
        frame: TurnResolutionContinuation,
    ) -> tuple[TransitionBranch, ...]:
        state = replace(
            branch.state,
            resolution_stack=branch.state.resolution_stack + (frame,),
        )
        return self._resume_resolution_stack((replace(branch, state=state),))

    def _continue_resolution_after_operation(
        self,
        branches: tuple[TransitionBranch, ...],
    ) -> tuple[TransitionBranch, ...]:
        resolved: list[TransitionBranch] = []
        for child in branches:
            if child.state.pending_choice is not None:
                resolved.append(child)
            else:
                resolved.extend(self._resume_resolution_stack((child,)))
        return tuple(resolved)

    def _trigger_event_phase(
        self,
        branches: tuple[TransitionBranch, ...],
        phase: SchedulePhase,
        *,
        trigger_context: EffectTriggerContext | None = None,
        condition_state: StrategyState | None = None,
    ) -> tuple[TransitionBranch, ...]:
        if phase not in IMMEDIATE_SCHEDULE_PHASES:
            raise KernelError(f"event_phase_is_not_immediate:{phase.value}")
        if not any(
            effect.trigger_phase is phase
            and effect.turns_until_fire == 0
            and effect.remaining_uses != 0
            and self._scheduled_card_filter_matches(effect, trigger_context)
            for branch in branches
            for effect in branch.state.scheduled_effects
        ):
            return branches
        expanded: list[TransitionBranch] = []
        for branch in branches:
            due = tuple(
                effect
                for effect in branch.state.scheduled_effects
                if effect.trigger_phase is phase
                and effect.turns_until_fire == 0
                and effect.remaining_uses != 0
                and self._scheduled_card_filter_matches(effect, trigger_context)
            )
            remaining = tuple(
                effect
                for effect in branch.state.scheduled_effects
                if not (
                    effect.trigger_phase is phase
                    and effect.turns_until_fire == 0
                    and effect.remaining_uses != 0
                    and self._scheduled_card_filter_matches(
                        effect,
                        trigger_context,
                    )
                )
            )
            expanded.extend(
                self._trigger_scheduled_effects(
                    (
                        replace(
                            branch,
                            state=replace(branch.state, scheduled_effects=remaining),
                        ),
                    ),
                    due,
                    trigger_context=trigger_context,
                    condition_state=condition_state or branch.state,
                )
            )
        return self._merge_probability_branches(tuple(expanded))

    def _scheduled_card_filter_matches(
        self,
        scheduled: ScheduledEffectRef,
        trigger_context: EffectTriggerContext | None,
    ) -> bool:
        definition_effects = self._scheduled_definition_effects(scheduled)
        if definition_effects is None:
            return True
        if scheduled.definition_effect_index >= len(definition_effects):
            return True
        schedule_effect = definition_effects[scheduled.definition_effect_index]
        return schedule_effect.trigger_card_type is None or (
            trigger_context is not None
            and trigger_context.used_card_type == schedule_effect.trigger_card_type
        )

    def _trigger_scheduled_effects(
        self,
        branches: tuple[TransitionBranch, ...],
        due: tuple[ScheduledEffectRef, ...],
        *,
        trigger_context: EffectTriggerContext | None = None,
        condition_state: StrategyState | None = None,
    ) -> tuple[TransitionBranch, ...]:
        expanded: list[TransitionBranch] = []
        for branch in branches:
            snapshot = condition_state or branch.state
            matched = tuple(
                self._scheduled_matched_nested_indices(
                    scheduled,
                    snapshot,
                    trigger_context,
                )
                for scheduled in due
            )
            expanded.extend(
                self._run_scheduled_sequence(
                    branch,
                    due,
                    matched,
                    trigger_context,
                )
            )
        return self._merge_probability_branches(tuple(expanded))

    def _scheduled_effect_spec(
        self,
        scheduled: ScheduledEffectRef,
    ) -> EffectSpec:
        definition_effects = self._scheduled_definition_effects(scheduled) or ()
        if (
            scheduled.definition_effect_index >= len(definition_effects)
        ):
            raise KernelError("scheduled_effect_definition_missing")
        schedule_effect = definition_effects[scheduled.definition_effect_index]
        if (
            schedule_effect.kind is not EffectKind.SCHEDULE
            or schedule_effect.schedule_phase is not scheduled.trigger_phase
        ):
            raise KernelError("scheduled_effect_reference_mismatch")
        return schedule_effect

    def _scheduled_definition_effects(
        self,
        scheduled: ScheduledEffectRef,
    ) -> tuple[EffectSpec, ...] | None:
        if scheduled.source_kind is EffectSourceKind.CARD:
            definition = self.cards.get(scheduled.card_id)
            if definition is None:
                return None
            return (
                definition.persistent_effects
                if scheduled.persistent
                else definition.effects
            )
        if scheduled.source_kind is EffectSourceKind.DRINK:
            if scheduled.persistent:
                return None
            definition = self.drinks.get(scheduled.card_id)
            return None if definition is None else definition.effects
        if scheduled.source_kind is EffectSourceKind.ITEM:
            if not scheduled.persistent:
                return None
            definition = self.items.get(scheduled.card_id)
            return None if definition is None else definition.persistent_effects
        return None

    def _scheduled_matched_nested_indices(
        self,
        scheduled: ScheduledEffectRef,
        condition_state: StrategyState,
        trigger_context: EffectTriggerContext | None,
    ) -> tuple[int, ...]:
        schedule_effect = self._scheduled_effect_spec(scheduled)
        return tuple(
            index
            for index, nested in enumerate(schedule_effect.scheduled_effects)
            if _condition_matches(
                condition_state,
                nested.condition,
                self.cards,
                trigger_context,
            )
        )

    def _run_scheduled_sequence(
        self,
        branch: TransitionBranch,
        due: tuple[ScheduledEffectRef, ...],
        matched: tuple[tuple[int, ...], ...],
        trigger_context: EffectTriggerContext | None,
    ) -> tuple[TransitionBranch, ...]:
        if not due:
            return (branch,)
        if len(due) != len(matched):
            raise KernelError("scheduled_effect_match_set_mismatch")
        scheduled = due[0]
        schedule_effect = self._scheduled_effect_spec(scheduled)
        triggered = replace(
            branch,
            events=branch.events
            + (
                RuleEvent(
                    "TRIGGER_SCHEDULED",
                    scheduled.source_id,
                    detail=(
                        f"{scheduled.card_id}:{scheduled.definition_effect_index}:"
                        f"{scheduled.trigger_phase.value}"
                    ),
                ),
            ),
        )
        return self._run_scheduled_nested(
            triggered,
            scheduled,
            schedule_effect,
            next_nested_index=0,
            matched_nested_indices=matched[0],
            remaining_due=due[1:],
            remaining_matched=matched[1:],
            trigger_context=trigger_context,
        )

    def _run_scheduled_nested(
        self,
        branch: TransitionBranch,
        scheduled: ScheduledEffectRef,
        schedule_effect: EffectSpec,
        *,
        next_nested_index: int,
        matched_nested_indices: tuple[int, ...],
        remaining_due: tuple[ScheduledEffectRef, ...],
        remaining_matched: tuple[tuple[int, ...], ...],
        trigger_context: EffectTriggerContext | None,
    ) -> tuple[TransitionBranch, ...]:
        if next_nested_index >= len(schedule_effect.scheduled_effects):
            finished = self._finish_scheduled_trigger(
                branch,
                scheduled,
                fired=bool(matched_nested_indices),
            )
            return self._run_scheduled_sequence(
                finished,
                remaining_due,
                remaining_matched,
                trigger_context,
            )

        nested = schedule_effect.scheduled_effects[next_nested_index]
        if next_nested_index not in matched_nested_indices:
            skipped = replace(
                branch,
                events=branch.events
                + (
                    RuleEvent(
                        "CONDITION_SKIPPED",
                        scheduled.source_id,
                        detail=(
                            nested.condition.kind.value
                            if nested.condition is not None
                            else "UNKNOWN"
                        ),
                    ),
                ),
            )
            return self._run_scheduled_nested(
                skipped,
                scheduled,
                schedule_effect,
                next_nested_index=next_nested_index + 1,
                matched_nested_indices=matched_nested_indices,
                remaining_due=remaining_due,
                remaining_matched=remaining_matched,
                trigger_context=trigger_context,
            )

        if nested.kind is EffectKind.HOLD_SELECTED:
            candidates = tuple(
                card.instance_id
                for zone in nested.target_zones
                for card in self._zone_cards(branch.state, zone)
                if card.instance_id != scheduled.source_id
            )
            if not candidates:
                empty = replace(
                    branch,
                    events=branch.events
                    + (RuleEvent("HOLD_SELECTED_EMPTY", scheduled.source_id),),
                )
                return self._run_scheduled_nested(
                    empty,
                    scheduled,
                    schedule_effect,
                    next_nested_index=next_nested_index + 1,
                    matched_nested_indices=matched_nested_indices,
                    remaining_due=remaining_due,
                    remaining_matched=remaining_matched,
                    trigger_context=trigger_context,
                )

        applied = self._apply_effect(
            (branch,),
            replace(nested, condition=None),
            source_id=scheduled.source_id,
            trigger_context=trigger_context,
            direct_effect=scheduled.direct_on_trigger,
        )
        resolved: list[TransitionBranch] = []
        for child in applied:
            if child.state.pending_choice is not None:
                frame = ScheduledEffectsContinuation(
                    scheduled=scheduled,
                    next_nested_index=next_nested_index + 1,
                    matched_nested_indices=matched_nested_indices,
                    remaining_due=remaining_due,
                    remaining_matched_nested_indices=remaining_matched,
                    trigger_context=trigger_context,
                    fired=bool(matched_nested_indices),
                )
                resolved.append(
                    replace(
                        child,
                        state=replace(
                            child.state,
                            resolution_stack=child.state.resolution_stack + (frame,),
                        ),
                    )
                )
            else:
                resolved.extend(
                    self._run_scheduled_nested(
                        child,
                        scheduled,
                        schedule_effect,
                        next_nested_index=next_nested_index + 1,
                        matched_nested_indices=matched_nested_indices,
                        remaining_due=remaining_due,
                        remaining_matched=remaining_matched,
                        trigger_context=trigger_context,
                    )
                )
        return tuple(resolved)

    def _resume_scheduled_continuation(
        self,
        branch: TransitionBranch,
        frame: ScheduledEffectsContinuation,
    ) -> tuple[TransitionBranch, ...]:
        schedule_effect = self._scheduled_effect_spec(frame.scheduled)
        return self._run_scheduled_nested(
            branch,
            frame.scheduled,
            schedule_effect,
            next_nested_index=frame.next_nested_index,
            matched_nested_indices=frame.matched_nested_indices,
            remaining_due=frame.remaining_due,
            remaining_matched=frame.remaining_matched_nested_indices,
            trigger_context=frame.trigger_context,
        )

    @staticmethod
    def _finish_scheduled_trigger(
        branch: TransitionBranch,
        scheduled: ScheduledEffectRef,
        *,
        fired: bool,
    ) -> TransitionBranch:
        remaining_uses = scheduled.remaining_uses
        keep = (
            scheduled.persistent
            or not fired
            or remaining_uses is None
            or remaining_uses > 1
        )
        if not keep:
            return branch
        next_uses = remaining_uses
        if fired and remaining_uses is not None:
            next_uses -= 1
        recurring = replace(
            scheduled,
            turns_until_fire=(
                0
                if scheduled.trigger_phase in IMMEDIATE_SCHEDULE_PHASES
                else 1
            ),
            remaining_uses=next_uses,
        )
        return replace(
            branch,
            state=replace(
                branch.state,
                scheduled_effects=branch.state.scheduled_effects + (recurring,),
            ),
        )

    @staticmethod
    def _validate_probability_mass(branches: tuple[TransitionBranch, ...]) -> None:
        if not branches:
            raise KernelError("transition_produced_no_branch")
        total = sum(branch.probability for branch in branches)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise KernelError(f"transition_probability_mass:{total}")
        for branch in branches:
            if branch.probability <= 0:
                raise KernelError("transition_probability_must_be_positive")
            branch.state.validate()
