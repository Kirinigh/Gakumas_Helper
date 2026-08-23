from __future__ import annotations

import re
import csv
import hashlib
from pathlib import Path
from collections import Counter
from dataclasses import replace, dataclass

from .kernel import (
    EffectKind,
    EffectSpec,
    CardCostKind,
    CardSelector,
    ConditionKind,
    CardDefinition,
    ItemDefinition,
    DrinkDefinition,
    EffectCondition,
    CardModification,
    EffectValueSource,
)
from .contracts import Stance, CardZone, PlanType, SchedulePhase

EFFECT_IR_VERSION = "task275-effect-ir@1.24"

_NUMBER = r"(-?\d+(?:\.\d+)?)"
_COST_RE = re.compile(rf"^(cost|stamina)-={_NUMBER}$")
_STATUS_COST_RE = re.compile(
    rf"^(concentration|goodConditionTurns|goodImpressionTurns|motivation)-={_NUMBER}$"
)
_FULL_POWER_COST_RE = re.compile(rf"^fullPowerCharge-={_NUMBER}$")
_SCORE_RE = re.compile(rf"^score\+={_NUMBER}$")
_SCORE_STATE_RE = re.compile(
    r"^score\+=(?:(?P<base>-?\d+(?:\.\d+)?)\+)?"
    r"(?P<source>goodConditionTurns|goodImpressionTurns|motivation|genki)"
    r"(?:\*(?P<scale>-?\d+(?:\.\d+)?))?$"
)
_SCORE_FULL_POWER_RE = re.compile(
    r"^score\+=(?:(?P<base>-?\d+(?:\.\d+)?)\+)?cumulativeFullPowerCharge"
    r"(?:\*(?P<scale>-?\d+(?:\.\d+)?))?$"
)
_GENKI_RE = re.compile(rf"^genki\+={_NUMBER}$")
_CONCENTRATION_RE = re.compile(rf"^concentration\+={_NUMBER}$")
_GOOD_CONDITION_RE = re.compile(rf"^goodConditionTurns\+={_NUMBER}$")
_EXCELLENT_CONDITION_RE = re.compile(rf"^perfectConditionTurns\+={_NUMBER}$")
_GOOD_IMPRESSION_RE = re.compile(rf"^goodImpressionTurns\+={_NUMBER}$")
_MOTIVATION_RE = re.compile(rf"^motivation\+={_NUMBER}$")
_STAMINA_GAIN_RE = re.compile(rf"^fixedStamina\+={_NUMBER}$")
_STAMINA_LOSS_RE = re.compile(rf"^fixedStamina-={_NUMBER}$")
_FIXED_STAMINA_MAX_LOSS_RE = re.compile(r"^fixedStamina-=maxStamina$")
_FULL_POWER_RE = re.compile(rf"^fullPowerCharge\+={_NUMBER}$")
_EXTRA_ACTION_RE = re.compile(r"^cardUsesRemaining\+=(\d+)$")
_DOUBLE_CARD_EFFECT_RE = re.compile(r"^doubleCardEffectCards\+=(\d+)$")
_HALF_COST_TURNS_RE = re.compile(r"^halfCostTurns\+=(\d+)$")
_DOUBLE_COST_TURNS_RE = re.compile(r"^doubleCostTurns\+=(\d+)$")
_ADD_CARD_TO_DECK_RE = re.compile(r"^addCardToDeck\((\d+)\)$")
_ENTHUSIASM_BONUS_RE = re.compile(
    r"^setEnthusiasmBonus\((\d+(?:\.\d+)?)(?:,([1-9]\d*))?\)$"
)
_ENTHUSIASM_MULTIPLIER_RE = re.compile(
    r"^setEnthusiasmBuff\((\d+(?:\.\d+)?)(?:,([1-9]\d*))?\)$"
)
_FULL_POWER_CHARGE_BUFF_RE = re.compile(
    r"^setFullPowerChargeBuff\((\d+(?:\.\d+)?)(?:,([1-9]\d*))?\)$"
)
_SCORE_BUFF_RE = re.compile(r"^setScoreBuff\((\d+(?:\.\d+)?)(?:,([1-9]\d*))?\)$")
_STANCE_LOCK_TURNS_RE = re.compile(r"^lockStanceTurns\+=([1-9]\d*)$")
_NULLIFY_COST_CARDS_RE = re.compile(r"^nullifyCostCards\+=([1-9]\d*)$")
_DRAW_RE = re.compile(r"^drawCard(?:\((\d+)\))?$")
_STANCE_RE = re.compile(r"^setStance\((strength2?|preservation2?|leisure|none)\)$")
_TURNS_REMAINING_RE = re.compile(r"^turnsRemaining\+=(\d+)$")
_IF_RE = re.compile(r"^if:(.+?)\s*\{(.*)\}$", re.DOTALL)
_HOLD_SELECTED_RE = re.compile(r"^holdSelected\[(.+)\](?:\((\d+)\))?$")
_HOLD_SELECTED_UPTO_RE = re.compile(r"^holdSelectedUpto\[(.+)\]\((\d+)\)$")
_FREE_PLAY_SELECTED_RE = re.compile(r"^useSelectedFree\[(.+)\]$")
_PAID_PLAY_SELECTED_RE = re.compile(r"^useSelected\[(.+)\]$")
_FREE_PLAY_ALL_RE = re.compile(r"^useAllFree\[(.+)\]$")
_FREE_PLAY_RANDOM_RE = re.compile(r"^useRandomCardFree\[(.+)\]$")
_ACTION_ENTHUSIASM_MULTIPLIER_RE = re.compile(
    r"^enthusiasmMultiplier=(\d+(?:\.\d+)?)$"
)
_ACTION_CONCENTRATION_MULTIPLIER_RE = re.compile(
    r"^concentrationMultiplier=(\d+(?:\.\d+)?)$"
)
_ACTION_GOOD_CONDITION_MULTIPLIER_RE = re.compile(
    r"^goodConditionTurnsMultiplier=(\d+(?:\.\d+)?)$"
)
_ACTION_MOTIVATION_MULTIPLIER_RE = re.compile(
    r"^motivationMultiplier=(\d+(?:\.\d+)?)$"
)
_TARGET_RE = re.compile(r"^target:(.+?)\s*\{(.*)\}$", re.DOTALL)
_AT_PHASE_RE = re.compile(
    r"^at:(beforeCardUsed|cardUsed|afterCardUsed|fullPowerChargeIncreased|cardMovedToHand|cardMovedToHeld|stanceChanged|endOfTurn|beforeStartOfTurn|startOfTurn|afterStartOfTurn|turn)"
    r"(?:\[(active|mental)\])?\s*\{(.*)\}$",
    re.DOTALL,
)
_MOVE_RANDOM_TO_HAND_RE = re.compile(r"^moveRandomToHand\[(.+)\]$")
_MOVE_SELECTED_TO_HAND_RE = re.compile(r"^moveSelectedToHand\[(.+)\]$")
_CARD_SCORE_RE = re.compile(r"^g\.score\+=(-?\d+)$")
_CARD_GENKI_RE = re.compile(r"^g\.genki\+=(-?\d+)$")
_CARD_COST_RE = re.compile(r"^g\.cost([+-])=(-?\d+)$")
_CARD_TYPED_COST_RE = re.compile(r"^g\.typedCost([+-])=(-?\d+)$")
_CARD_SCORE_TIMES_RE = re.compile(r"^g\.scoreTimes\+=(\d+)$")


@dataclass(frozen=True)
class UnsupportedCatalogCard:
    card_id: str
    name: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CompiledCardCatalog:
    source_path: str
    source_sha256: str
    plan_type: PlanType
    cards: dict[str, CardDefinition]
    upgrade_map: dict[str, str]
    unsupported: tuple[UnsupportedCatalogCard, ...]
    total_rows: int
    primary_card_ids: tuple[str, ...] = ()
    dependency_card_ids: tuple[str, ...] = ()

    @property
    def supported_rows(self) -> int:
        return len(self.primary_card_ids)

    @property
    def supported_card_ids(self) -> tuple[str, ...]:
        return self.primary_card_ids

    @property
    def coverage(self) -> float:
        return self.supported_rows / self.total_rows if self.total_rows else 0.0

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for card in self.unsupported:
            counts.update(card.reasons)
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class UnsupportedCatalogDrink:
    drink_id: str
    name: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CompiledDrinkCatalog:
    source_path: str
    source_sha256: str
    plan_type: PlanType
    drinks: dict[str, DrinkDefinition]
    unsupported: tuple[UnsupportedCatalogDrink, ...]
    total_rows: int

    @property
    def supported_rows(self) -> int:
        return len(self.drinks)

    @property
    def coverage(self) -> float:
        return self.supported_rows / self.total_rows if self.total_rows else 0.0

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for drink in self.unsupported:
            counts.update(drink.reasons)
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class UnsupportedCatalogItem:
    item_id: str
    name: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CompiledItemCatalog:
    source_path: str
    source_sha256: str
    plan_type: PlanType
    items: dict[str, ItemDefinition]
    unsupported: tuple[UnsupportedCatalogItem, ...]
    total_rows: int

    @property
    def supported_rows(self) -> int:
        return len(self.items)

    @property
    def coverage(self) -> float:
        return self.supported_rows / self.total_rows if self.total_rows else 0.0

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in self.unsupported:
            counts.update(item.reasons)
        return dict(sorted(counts.items()))


def _split_top_level(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _split_condition_and(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        elif char == "&" and depth == 0:
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _stance(value: str) -> Stance:
    return {
        "strength": Stance.AGGRESSIVE_1,
        "strength2": Stance.AGGRESSIVE_2,
        "preservation": Stance.CONSERVE_1,
        "preservation2": Stance.CONSERVE_2,
        "leisure": Stance.LEISURE,
        "fullPower": Stance.FULL_POWER,
        "none": Stance.NEUTRAL,
    }[value]


def _card_zones(value: str) -> tuple[CardZone, ...] | None:
    names = tuple(part.strip() for part in value.split("|"))
    mapping = {
        "hand": CardZone.HAND,
        "deck": CardZone.DRAW_PILE,
        "discarded": CardZone.DISCARD,
        "held": CardZone.HELD,
    }
    if not names or any(name not in mapping for name in names):
        return None
    return tuple(dict.fromkeys(mapping[name] for name in names))


def _compile_condition(value: str) -> EffectCondition | None:
    condition = value.strip().removeprefix("@usable ").removeprefix("if:").strip()
    conjunction = _split_condition_and(condition)
    if len(conjunction) > 1:
        compiled = tuple(_compile_condition(part) for part in conjunction)
        if any(part is None for part in compiled):
            return None
        return EffectCondition(
            ConditionKind.ALL,
            all_of=tuple(part for part in compiled if part is not None),
        )
    if condition == "isStrength":
        return EffectCondition(ConditionKind.STANCE_FAMILY, value="strength")
    if condition == "isPreservation":
        return EffectCondition(ConditionKind.STANCE_FAMILY, value="preservation")
    if condition == "isFullPower":
        return EffectCondition(ConditionKind.STANCE_FAMILY, value="fullPower")
    if condition == "isDirectEffect":
        return EffectCondition(ConditionKind.IS_DIRECT_EFFECT)
    if match := re.fullmatch(
        r"cardHasEffect\((strength|fullPowerCharge)\)",
        condition,
    ):
        return EffectCondition(
            ConditionKind.USED_CARD_EFFECT_TAG,
            value=match.group(1),
        )
    if match := re.fullmatch(r"usedCardBaseId==(\d+)", condition):
        return EffectCondition(
            ConditionKind.USED_CARD_BASE_ID,
            value=match.group(1),
        )
    if match := re.fullmatch(r"cardRarity!=(N|R|SR|SSR|L|T)", condition):
        return EffectCondition(
            ConditionKind.USED_CARD_RARITY_NOT,
            value=match.group(1),
        )
    if match := re.fullmatch(r"movedCardId==(\d+)", condition):
        return EffectCondition(
            ConditionKind.MOVED_CARD_ID,
            value=match.group(1),
        )
    if match := re.fullmatch(r"movedCardBaseId==(\d+)", condition):
        return EffectCondition(
            ConditionKind.MOVED_CARD_BASE_ID,
            value=match.group(1),
        )
    if condition == "stance!=none":
        return EffectCondition(ConditionKind.STANCE_FAMILY, value="not_none")
    if condition == "!isStrength":
        return EffectCondition(ConditionKind.STANCE_NOT, value="strength")
    if condition == "!isPreservation":
        return EffectCondition(ConditionKind.STANCE_NOT, value="preservation")
    if condition == "stance!=fullPower":
        return EffectCondition(ConditionKind.STANCE_NOT, value="fullPower")
    if match := re.fullmatch(
        r"stance==(strength2?|preservation2?|leisure|fullPower|none)",
        condition,
    ):
        return EffectCondition(ConditionKind.STANCE_IS, value=_stance(match.group(1)).value)
    if match := re.fullmatch(
        r"prevStance==(strength2?|preservation2?|leisure|fullPower|none)",
        condition,
    ):
        return EffectCondition(
            ConditionKind.PREVIOUS_STANCE_IS,
            value=_stance(match.group(1)).value,
        )
    if match := re.fullmatch(r"fullPowerCharge>=(\d+)", condition):
        return EffectCondition(ConditionKind.FULL_POWER_CHARGE_MIN, threshold=int(match.group(1)))
    if match := re.fullmatch(r"fullPowerCharge<=(\d+)", condition):
        return EffectCondition(ConditionKind.FULL_POWER_CHARGE_MAX, threshold=int(match.group(1)))
    if match := re.fullmatch(r"cumulativeFullPowerCharge>=(\d+)", condition):
        return EffectCondition(
            ConditionKind.CUMULATIVE_FULL_POWER_MIN,
            threshold=int(match.group(1)),
        )
    if match := re.fullmatch(r"turnsRemaining<=(\d+)", condition):
        return EffectCondition(ConditionKind.TURNS_REMAINING_MAX, threshold=int(match.group(1)))
    numeric_min_conditions = {
        "concentration": ConditionKind.CONCENTRATION_MIN,
        "goodConditionTurns": ConditionKind.GOOD_CONDITION_MIN,
        "perfectConditionTurns": ConditionKind.EXCELLENT_CONDITION_MIN,
        "goodImpressionTurns": ConditionKind.GOOD_IMPRESSION_MIN,
        "motivation": ConditionKind.MOTIVATION_MIN,
        "genki": ConditionKind.GENKI_MIN,
    }
    if match := re.fullmatch(
        r"(concentration|goodConditionTurns|perfectConditionTurns|goodImpressionTurns|motivation|genki)>=(\d+)",
        condition,
    ):
        return EffectCondition(
            numeric_min_conditions[match.group(1)],
            threshold=int(match.group(2)),
        )
    if match := re.fullmatch(
        r"stanceChangedByDirectEffectTimes%(\d+)==(\d+)",
        condition,
    ):
        modulus = int(match.group(1))
        remainder = int(match.group(2))
        if modulus < 1 or remainder >= modulus:
            return None
        return EffectCondition(
            ConditionKind.STANCE_CHANGE_COUNT_MOD_EQ,
            modulus=modulus,
            remainder=remainder,
        )
    count_conditions = {
        "stanceChangedTimes": ConditionKind.STANCE_CHANGE_COUNT_MIN,
        "strengthTimes": ConditionKind.STRENGTH_COUNT_MIN,
        "preservationTimes": ConditionKind.PRESERVATION_COUNT_MIN,
        "fullPowerTimes": ConditionKind.FULL_POWER_COUNT_MIN,
    }
    if match := re.fullmatch(
        r"(stanceChangedTimes|strengthTimes|preservationTimes|fullPowerTimes)>=(\d+)",
        condition,
    ):
        return EffectCondition(
            count_conditions[match.group(1)],
            threshold=int(match.group(2)),
        )
    if match := re.fullmatch(r"countCards\[!removed\s*&\s*T\]>=(\d+)", condition):
        return EffectCondition(
            ConditionKind.TROUBLE_CARD_COUNT_MIN,
            threshold=int(match.group(1)),
        )
    if match := re.fullmatch(
        r"countCards\[baseId\((\d+)\)\s*&\s*removed\]\s*>\s*0",
        condition,
    ):
        return EffectCondition(
            ConditionKind.REMOVED_BASE_CARD_COUNT_MIN,
            value=match.group(1),
            threshold=1,
        )
    return None


def _compile_card_selector(value: str) -> CardSelector | None:
    terms = tuple(part.strip() for part in value.split("&"))
    if not terms or any(not term for term in terms):
        return None
    if terms == ("all",):
        return CardSelector(all_cards=True)
    if terms == ("this",):
        return CardSelector(source_only=True)
    zone_mapping = {
        "hand": CardZone.HAND,
        "deck": CardZone.DRAW_PILE,
        "discarded": CardZone.DISCARD,
        "held": CardZone.HELD,
    }
    zones: list[CardZone] = []
    card_type: str | None = None
    card_id: str | None = None
    base_card_id: str | None = None
    effect_tag: str | None = None
    source_type: str | None = None
    for term in terms:
        normalized = term[1:-1].strip() if term.startswith("(") and term.endswith(")") else term
        alternatives = tuple(part.strip() for part in normalized.split("|"))
        if alternatives and all(alternative in zone_mapping for alternative in alternatives):
            zones.extend(zone_mapping[alternative] for alternative in alternatives)
        elif term in {"active", "mental"} and card_type is None:
            card_type = term
        elif term.isdigit() and card_id is None:
            card_id = term
        elif match := re.fullmatch(r"id\((\d+)\)", term):
            if card_id is not None:
                return None
            card_id = match.group(1)
        elif match := re.fullmatch(r"baseId\((\d+)\)", term):
            if base_card_id is not None:
                return None
            base_card_id = match.group(1)
        elif match := re.fullmatch(r"effect\((strength|fullPowerCharge)\)", term):
            if effect_tag is not None:
                return None
            effect_tag = match.group(1)
        elif term in {"default", "pIdol", "produce", "support"}:
            if source_type is not None:
                return None
            source_type = term
        else:
            return None
    selector = CardSelector(
        zones=tuple(dict.fromkeys(zones)),
        card_type=card_type,
        card_id=card_id,
        base_card_id=base_card_id,
        effect_tag=effect_tag,
        source_type=source_type,
    )
    try:
        selector.validate()
    except ValueError:
        return None
    return selector


def _compile_card_modification(value: str) -> CardModification | None:
    score_bonus = 0
    genki_bonus = 0
    stamina_cost_delta = 0
    typed_cost_delta = 0
    score_repeat_bonus = 0
    statements = _split_top_level(value)
    if not statements:
        return None
    for statement in statements:
        if match := _CARD_SCORE_RE.fullmatch(statement):
            score_bonus += int(match.group(1))
        elif match := _CARD_GENKI_RE.fullmatch(statement):
            genki_bonus += int(match.group(1))
        elif match := _CARD_COST_RE.fullmatch(statement):
            amount = int(match.group(2))
            stamina_cost_delta += amount if match.group(1) == "+" else -amount
        elif match := _CARD_TYPED_COST_RE.fullmatch(statement):
            amount = int(match.group(2))
            typed_cost_delta += amount if match.group(1) == "+" else -amount
        elif match := _CARD_SCORE_TIMES_RE.fullmatch(statement):
            score_repeat_bonus += int(match.group(1))
        else:
            return None
    modification = CardModification(
        score_bonus=score_bonus,
        genki_bonus=genki_bonus,
        stamina_cost_delta=stamina_cost_delta,
        typed_cost_delta=typed_cost_delta,
        score_repeat_bonus=score_repeat_bonus,
    )
    try:
        modification.validate()
    except ValueError:
        return None
    return modification


def _compile_action_sequence(value: str) -> tuple[EffectSpec, ...] | None:
    effects: list[EffectSpec] = []
    enthusiasm_multiplier = 1.0
    concentration_multiplier = 1.0
    good_condition_multiplier = 1.0
    motivation_multiplier = 1.0
    for statement in _split_top_level(value):
        if match := _ACTION_ENTHUSIASM_MULTIPLIER_RE.fullmatch(statement):
            enthusiasm_multiplier = float(match.group(1))
            if enthusiasm_multiplier <= 0:
                return None
            continue
        if match := _ACTION_CONCENTRATION_MULTIPLIER_RE.fullmatch(statement):
            concentration_multiplier = float(match.group(1))
            if concentration_multiplier <= 0:
                return None
            continue
        if match := _ACTION_GOOD_CONDITION_MULTIPLIER_RE.fullmatch(statement):
            good_condition_multiplier = float(match.group(1))
            if good_condition_multiplier <= 0:
                return None
            continue
        if match := _ACTION_MOTIVATION_MULTIPLIER_RE.fullmatch(statement):
            motivation_multiplier = float(match.group(1))
            if motivation_multiplier <= 0:
                return None
            continue
        compiled = _compile_action(statement)
        if compiled is None:
            return None
        effects.extend(
            replace(
                effect,
                score_enthusiasm_multiplier=enthusiasm_multiplier,
                score_concentration_multiplier=concentration_multiplier,
                score_good_condition_multiplier=good_condition_multiplier,
            )
            if effect.kind is EffectKind.SCORE
            else replace(
                effect,
                genki_motivation_multiplier=motivation_multiplier,
            )
            if effect.kind is EffectKind.GENKI
            else effect
            for effect in compiled
        )
    return tuple(effects)


def _compile_action(statement: str) -> tuple[EffectSpec, ...] | None:
    statement = statement.strip().removeprefix("@grow ").strip()
    if match := _IF_RE.fullmatch(statement):
        condition = _compile_condition(match.group(1))
        if condition is None:
            return None
        compiled = _compile_action_sequence(match.group(2))
        if compiled is None or any(effect.condition is not None for effect in compiled):
            return None
        return tuple(replace(effect, condition=condition) for effect in compiled) or None
    if match := _AT_PHASE_RE.fullmatch(statement):
        phase_name = match.group(1)
        trigger_card_type = match.group(2)
        body = match.group(3).strip()
        schedule_limit: int | None = None
        schedule_ttl: int | None = None
        trailing_lifetime = re.search(r"(?:;\s*|\s+)(limit|ttl):(\d+)\s*$", body)
        while trailing_lifetime is not None:
            lifetime_value = int(trailing_lifetime.group(2))
            if lifetime_value < 1:
                return None
            if trailing_lifetime.group(1) == "limit":
                if schedule_limit is not None:
                    return None
                schedule_limit = lifetime_value
            else:
                if schedule_ttl is not None:
                    return None
                schedule_ttl = lifetime_value
            body = body[: trailing_lifetime.start()].strip()
            trailing_lifetime = re.search(
                r"(?:;\s*|\s+)(limit|ttl):(\d+)\s*$",
                body,
            )
        if nested_condition := _IF_RE.fullmatch(body):
            nested_body = nested_condition.group(2).strip()
            nested_lifetime = re.search(
                r"(?:;\s*|\s+)(limit|ttl):(\d+)\s*$",
                nested_body,
            )
            while nested_lifetime is not None:
                lifetime_value = int(nested_lifetime.group(2))
                if lifetime_value < 1:
                    return None
                if nested_lifetime.group(1) == "limit":
                    if schedule_limit is not None:
                        return None
                    schedule_limit = lifetime_value
                else:
                    if schedule_ttl is not None:
                        return None
                    schedule_ttl = lifetime_value
                nested_body = nested_body[: nested_lifetime.start()].strip()
                nested_lifetime = re.search(
                    r"(?:;\s*|\s+)(limit|ttl):(\d+)\s*$",
                    nested_body,
                )
            body = f"if:{nested_condition.group(1)} {{ {nested_body} }}"
        if schedule_limit is None and schedule_ttl is None and phase_name == "turn":
            return None
        compiled = _compile_action_sequence(body)
        if compiled is None or any(
            effect.kind is EffectKind.SCHEDULE
            for effect in compiled
        ):
            return None
        if not compiled:
            return None
        schedule_phase = {
            "cardUsed": SchedulePhase.CARD_USED,
            "beforeCardUsed": SchedulePhase.BEFORE_CARD_USED,
            "afterCardUsed": SchedulePhase.AFTER_CARD_USED,
            "fullPowerChargeIncreased": SchedulePhase.FULL_POWER_CHARGE_INCREASED,
            "cardMovedToHand": SchedulePhase.CARD_MOVED_TO_HAND,
            "cardMovedToHeld": SchedulePhase.CARD_MOVED_TO_HELD,
            "stanceChanged": SchedulePhase.STANCE_CHANGED,
            "endOfTurn": SchedulePhase.END_OF_TURN,
            "beforeStartOfTurn": SchedulePhase.BEFORE_START_OF_TURN,
            "turn": SchedulePhase.TURN,
            "startOfTurn": SchedulePhase.START_OF_TURN,
            "afterStartOfTurn": SchedulePhase.AFTER_START_OF_TURN,
        }[phase_name]
        return (
            EffectSpec(
                EffectKind.SCHEDULE,
                scheduled_effects=compiled,
                delay_turns=(
                    0
                    if phase_name
                    in {
                        "beforeCardUsed",
                        "cardUsed",
                        "afterCardUsed",
                        "fullPowerChargeIncreased",
                        "cardMovedToHand",
                        "cardMovedToHeld",
                        "stanceChanged",
                        "endOfTurn",
                    }
                    else 1
                ),
                schedule_phase=schedule_phase,
                schedule_limit=schedule_limit,
                schedule_ttl=schedule_ttl,
                trigger_card_type=trigger_card_type,
            ),
        )
    if match := _TARGET_RE.fullmatch(statement):
        selector = _compile_card_selector(match.group(1))
        modification = _compile_card_modification(match.group(2))
        if selector is None or modification is None:
            return None
        return (
            EffectSpec(
                EffectKind.CARD_MODIFICATION,
                card_selector=selector,
                card_modification=modification,
            ),
        )
    if match := _SCORE_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.SCORE, float(match.group(1))),)
    if match := _SCORE_STATE_RE.fullmatch(statement):
        source = {
            "goodConditionTurns": EffectValueSource.GOOD_CONDITION,
            "goodImpressionTurns": EffectValueSource.GOOD_IMPRESSION,
            "motivation": EffectValueSource.MOTIVATION,
            "genki": EffectValueSource.GENKI,
        }[match.group("source")]
        return (
            EffectSpec(
                EffectKind.SCORE,
                amount=float(match.group("base") or 0),
                value_source=source,
                value_scale=float(match.group("scale") or 1),
            ),
        )
    if match := _SCORE_FULL_POWER_RE.fullmatch(statement):
        return (
            EffectSpec(
                EffectKind.SCORE,
                float(match.group("base") or 0),
                full_power_scale=float(match.group("scale") or 1),
            ),
        )
    if match := _GENKI_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.GENKI, float(match.group(1))),)
    if match := _CONCENTRATION_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.CONCENTRATION, float(match.group(1))),)
    if match := _GOOD_CONDITION_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.GOOD_CONDITION, float(match.group(1))),)
    if match := _EXCELLENT_CONDITION_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.EXCELLENT_CONDITION, float(match.group(1))),)
    if match := _GOOD_IMPRESSION_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.GOOD_IMPRESSION, float(match.group(1))),)
    if match := _MOTIVATION_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.MOTIVATION, float(match.group(1))),)
    if match := _STAMINA_GAIN_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.STAMINA, float(match.group(1))),)
    if match := _STAMINA_LOSS_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.STAMINA, -float(match.group(1))),)
    if _FIXED_STAMINA_MAX_LOSS_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.FIXED_STAMINA_TO_ZERO),)
    if match := _FULL_POWER_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.FULL_POWER, float(match.group(1))),)
    if match := _EXTRA_ACTION_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.EXTRA_ACTION, float(match.group(1))),)
    if match := _DOUBLE_CARD_EFFECT_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.DOUBLE_CARD_EFFECT, float(match.group(1))),)
    if match := _HALF_COST_TURNS_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.HALF_COST_TURNS, float(match.group(1))),)
    if match := _DOUBLE_COST_TURNS_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.DOUBLE_COST_TURNS, float(match.group(1))),)
    if match := _ADD_CARD_TO_DECK_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.ADD_CARD_TO_DECK, card_id=match.group(1)),)
    if match := _ENTHUSIASM_BONUS_RE.fullmatch(statement):
        return (
            EffectSpec(
                EffectKind.ENTHUSIASM_BONUS_BUFF,
                amount=float(match.group(1)),
                duration=int(match.group(2)) if match.group(2) is not None else None,
            ),
        )
    if match := _ENTHUSIASM_MULTIPLIER_RE.fullmatch(statement):
        return (
            EffectSpec(
                EffectKind.ENTHUSIASM_MULTIPLIER_BUFF,
                amount=float(match.group(1)),
                duration=int(match.group(2)) if match.group(2) is not None else None,
            ),
        )
    if match := _FULL_POWER_CHARGE_BUFF_RE.fullmatch(statement):
        return (
            EffectSpec(
                EffectKind.FULL_POWER_CHARGE_BUFF,
                amount=float(match.group(1)),
                duration=int(match.group(2)) if match.group(2) is not None else None,
            ),
        )
    if match := _SCORE_BUFF_RE.fullmatch(statement):
        return (
            EffectSpec(
                EffectKind.SCORE_BUFF,
                amount=float(match.group(1)),
                duration=int(match.group(2)) if match.group(2) is not None else None,
            ),
        )
    if match := _STANCE_LOCK_TURNS_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.STANCE_LOCK_TURNS, amount=float(match.group(1))),)
    if match := _NULLIFY_COST_CARDS_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.NULLIFY_COST_CARDS, amount=float(match.group(1))),)
    if match := _DRAW_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.DRAW, float(match.group(1) or 1)),)
    if match := _MOVE_RANDOM_TO_HAND_RE.fullmatch(statement):
        selector = _compile_card_selector(match.group(1))
        if selector is not None and selector.zones and CardZone.HAND not in selector.zones:
            return (EffectSpec(EffectKind.MOVE_RANDOM_TO_HAND, card_selector=selector),)
        return None
    if match := _MOVE_SELECTED_TO_HAND_RE.fullmatch(statement):
        zones = _card_zones(match.group(1))
        if zones is not None and CardZone.HAND not in zones:
            return (
                EffectSpec(
                    EffectKind.MOVE_SELECTED_TO_HAND,
                    target_zones=zones,
                ),
            )
        return None
    if statement == "exchangeHand":
        return (EffectSpec(EffectKind.EXCHANGE_HAND),)
    if statement == "upgradeHand":
        return (EffectSpec(EffectKind.UPGRADE_HAND),)
    if match := _STANCE_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.STANCE, stance=_stance(match.group(1))),)
    if match := _TURNS_REMAINING_RE.fullmatch(statement):
        return (EffectSpec(EffectKind.TURNS_REMAINING, float(match.group(1))),)
    if statement == "holdThisCard":
        return (EffectSpec(EffectKind.HOLD_SELF),)
    if match := _HOLD_SELECTED_RE.fullmatch(statement):
        zones = _card_zones(match.group(1))
        if zones is not None:
            return (
                EffectSpec(
                    EffectKind.HOLD_SELECTED,
                    target_zones=zones,
                    selection_count=int(match.group(2) or 1),
                ),
            )
    if match := _HOLD_SELECTED_UPTO_RE.fullmatch(statement):
        zones = _card_zones(match.group(1))
        if zones is not None:
            return (
                EffectSpec(
                    EffectKind.HOLD_SELECTED,
                    target_zones=zones,
                    selection_count=int(match.group(2)),
                    selection_optional=True,
                ),
            )
    if match := _FREE_PLAY_SELECTED_RE.fullmatch(statement):
        zones = _card_zones(match.group(1))
        if zones is not None:
            return (EffectSpec(EffectKind.FREE_PLAY_SELECTED, target_zones=zones),)
    if match := _PAID_PLAY_SELECTED_RE.fullmatch(statement):
        zones = _card_zones(match.group(1))
        if zones is not None:
            return (EffectSpec(EffectKind.PAID_PLAY_SELECTED, target_zones=zones),)
    if match := _FREE_PLAY_ALL_RE.fullmatch(statement):
        zones = _card_zones(match.group(1))
        if zones is not None:
            return (EffectSpec(EffectKind.FREE_PLAY_ALL, target_zones=zones),)
    if match := _FREE_PLAY_RANDOM_RE.fullmatch(statement):
        selector = _compile_card_selector(match.group(1))
        if selector is not None:
            return (EffectSpec(EffectKind.FREE_PLAY_RANDOM, card_selector=selector),)
    return None


def _compile_cost(value: str) -> tuple[int, int, CardCostKind] | None:
    if match := _COST_RE.fullmatch(value):
        number = float(match.group(2))
        if not number.is_integer() or number < 0:
            return None
        return (
            int(number),
            0,
            CardCostKind.NORMAL
            if match.group(1) == "cost"
            else CardCostKind.STAMINA,
        )
    if match := _FULL_POWER_COST_RE.fullmatch(value):
        number = float(match.group(1))
        return (
            (0, int(number), CardCostKind.FULL_POWER)
            if number.is_integer() and number >= 0
            else None
        )
    if match := _STATUS_COST_RE.fullmatch(value):
        number = float(match.group(2))
        if not number.is_integer() or number < 0:
            return None
        kind = {
            "concentration": CardCostKind.CONCENTRATION,
            "goodConditionTurns": CardCostKind.GOOD_CONDITION,
            "goodImpressionTurns": CardCostKind.GOOD_IMPRESSION,
            "motivation": CardCostKind.MOTIVATION,
        }[match.group(1)]
        return (int(number), 0, kind)
    return None


def compile_skill_card_catalog(path: str | Path, *, plan_type: PlanType) -> CompiledCardCatalog:
    source = Path(path)
    source_bytes = source.read_bytes()
    rows = list(csv.DictReader(source_bytes.decode("utf-8-sig").splitlines()))
    rows_by_id = {str(row["id"]): row for row in rows}
    selected = [row for row in rows if row.get("plan") == plan_type.value.lower()]
    source_variants: dict[str, dict[bool, str]] = {}
    for row in selected:
        source_variants.setdefault(str(row["name"]).removesuffix("+"), {})[
            str(row.get("upgraded", "")).upper() == "TRUE"
        ] = str(row["id"])
    base_card_ids = {
        card_id: variants.get(False, card_id)
        for variants in source_variants.values()
        for card_id in variants.values()
    }
    upgrade_card_ids = {
        variants[False]: variants[True]
        for variants in source_variants.values()
        if False in variants and True in variants
    }
    cards: dict[str, CardDefinition] = {}
    unsupported: list[UnsupportedCatalogCard] = []
    names: dict[str, dict[bool, str]] = {}
    for row in selected:
        card_id = str(row["id"])
        name = str(row["name"])
        reasons: list[str] = []
        raw_condition = row.get("conditions", "").strip()
        condition = _compile_condition(raw_condition) if raw_condition else None
        if raw_condition and condition is None:
            reasons.append("card_condition_not_compiled")
        cost = _compile_cost(row.get("cost", "").strip())
        if cost is None:
            reasons.append("cost_not_compiled")
        raw_actions = row.get("actions", "")
        compiled_actions = _compile_action_sequence(raw_actions)
        actions = list(compiled_actions or ())
        if compiled_actions is None:
            for statement in _split_top_level(raw_actions):
                if _compile_action_sequence(statement) is None:
                    reasons.append(f"action_not_compiled:{statement}")
        persistent_effects: list[EffectSpec] = []
        for statement in _split_top_level(row.get("effects", "")):
            effects = _compile_action(statement)
            if (
                effects is None
                or any(
                    effect.kind is not EffectKind.SCHEDULE
                    or any(
                        nested.kind
                        not in {
                            SchedulePhase.STANCE_CHANGED: {
                                EffectKind.CARD_MODIFICATION,
                            },
                            SchedulePhase.AFTER_CARD_USED: {
                                EffectKind.CARD_MODIFICATION,
                            },
                            SchedulePhase.CARD_MOVED_TO_HAND: {
                                EffectKind.DRAW,
                                EffectKind.HOLD_SELECTED,
                            },
                            SchedulePhase.CARD_MOVED_TO_HELD: {
                                EffectKind.CARD_MODIFICATION,
                            },
                            SchedulePhase.END_OF_TURN: {
                                EffectKind.FREE_PLAY_RANDOM,
                            },
                        }.get(effect.schedule_phase, set())
                        for nested in effect.scheduled_effects
                    )
                    for effect in effects
                )
            ):
                reasons.append(f"persistent_effect_not_compiled:{statement}")
            else:
                persistent_effects.extend(effects)
        secondary_indices = tuple(
            index
            for index, effect in enumerate(actions)
            if effect.kind
            in {
                EffectKind.HOLD_SELECTED,
                EffectKind.FREE_PLAY_SELECTED,
                EffectKind.MOVE_SELECTED_TO_HAND,
            }
        )
        if len(secondary_indices) > 1:
            reasons.append("multiple_secondary_choices_not_compiled")
        if not actions:
            reasons.append("no_compiled_effect")
        if reasons:
            unsupported.append(UnsupportedCatalogCard(card_id, name, tuple(dict.fromkeys(reasons))))
            continue
        assert cost is not None
        definition = CardDefinition(
            card_id=card_id,
            name=name,
            plan=plan_type,
            stamina_cost=cost[0],
            full_power_cost=cost[1],
            cost_kind=cost[2],
            effects=tuple(actions),
            persistent_effects=tuple(persistent_effects),
            limited=bool(row.get("limit", "").strip()),
            condition=condition,
            card_type=row.get("type", "").strip() or None,
            rarity=row.get("rarity", "").strip() or None,
            base_card_id=base_card_ids.get(card_id, card_id),
            upgrade_card_id=upgrade_card_ids.get(card_id),
            source_type=row.get("sourceType", "").strip() or None,
        )
        definition.validate()
        cards[card_id] = definition
        base_name = name.removesuffix("+")
        upgraded = str(row.get("upgraded", "")).upper() == "TRUE"
        names.setdefault(base_name, {})[upgraded] = card_id
    primary_card_order = tuple(cards)
    requested_dependency_card_ids = tuple(
        sorted(
            {
                str(effect.card_id)
                for definition in cards.values()
                for effect in definition.effects
                if effect.kind is EffectKind.ADD_CARD_TO_DECK and effect.card_id is not None
            }
            - set(cards),
            key=int,
        )
    )
    resolved_dependency_card_ids: list[str] = []
    for card_id in requested_dependency_card_ids:
        row = rows_by_id.get(card_id)
        if row is None:
            raise ValueError(f"generated card dependency is missing from catalog: {card_id}")
        raw_condition = row.get("conditions", "").strip()
        condition = _compile_condition(raw_condition) if raw_condition else None
        cost = _compile_cost(row.get("cost", "").strip())
        compiled_actions = _compile_action_sequence(row.get("actions", ""))
        actions = list(compiled_actions or ())
        dependency_reasons: list[str] = []
        if raw_condition and condition is None:
            dependency_reasons.append("card_condition_not_compiled")
        if cost is None:
            dependency_reasons.append("cost_not_compiled")
        if compiled_actions is None:
            for statement in _split_top_level(row.get("actions", "")):
                if _compile_action_sequence(statement) is None:
                    dependency_reasons.append(f"action_not_compiled:{statement}")
        if row.get("effects", "").strip():
            dependency_reasons.append("persistent_or_triggered_effect_not_compiled")
        if not actions and row.get("type", "").strip() != "trouble":
            dependency_reasons.append("no_compiled_effect")
        if row.get("plan", "").strip() not in {"free", plan_type.value.lower()}:
            dependency_reasons.append("dependency_plan_not_supported")
        if dependency_reasons:
            producer_ids = tuple(
                definition.card_id
                for definition in cards.values()
                if any(
                    effect.kind is EffectKind.ADD_CARD_TO_DECK
                    and effect.card_id == card_id
                    for effect in definition.effects
                )
            )
            if not producer_ids:
                raise ValueError(
                    f"generated card dependency {card_id} is not fully compiled: "
                    + ",".join(dict.fromkeys(dependency_reasons))
                )
            for producer_id in producer_ids:
                producer = cards.pop(producer_id)
                unsupported.append(
                    UnsupportedCatalogCard(
                        producer.card_id,
                        producer.name,
                        (
                            f"generated_dependency_not_compiled:{card_id}:"
                            + ",".join(dict.fromkeys(dependency_reasons)),
                        ),
                    )
                )
            continue
        assert cost is not None
        definition = CardDefinition(
            card_id=card_id,
            name=str(row["name"]),
            plan=None if row.get("plan", "").strip() == "free" else plan_type,
            stamina_cost=cost[0],
            full_power_cost=cost[1],
            cost_kind=cost[2],
            effects=tuple(actions),
            limited=bool(row.get("limit", "").strip()),
            condition=condition,
            card_type=row.get("type", "").strip() or None,
            rarity=row.get("rarity", "").strip() or None,
            base_card_id=base_card_ids.get(card_id, card_id),
            source_type=row.get("sourceType", "").strip() or None,
        )
        definition.validate()
        cards[card_id] = definition
        resolved_dependency_card_ids.append(card_id)
    primary_card_ids = tuple(
        card_id for card_id in primary_card_order if card_id in cards
    )
    dependency_card_ids = tuple(resolved_dependency_card_ids)
    upgrade_map = {
        variants[False]: variants[True]
        for variants in names.values()
        if False in variants and True in variants and variants[False] in cards and variants[True] in cards
    }
    return CompiledCardCatalog(
        source_path=str(source),
        source_sha256=hashlib.sha256(source_bytes).hexdigest().upper(),
        plan_type=plan_type,
        cards=cards,
        upgrade_map=upgrade_map,
        unsupported=tuple(unsupported),
        total_rows=len(selected),
        primary_card_ids=primary_card_ids,
        dependency_card_ids=dependency_card_ids,
    )


def compile_drink_catalog(
    path: str | Path,
    *,
    plan_type: PlanType,
) -> CompiledDrinkCatalog:
    source = Path(path)
    source_bytes = source.read_bytes()
    rows = list(csv.DictReader(source_bytes.decode("utf-8-sig").splitlines()))
    selected = [
        row
        for row in rows
        if row.get("plan") in {"free", plan_type.value.lower()}
    ]
    drinks: dict[str, DrinkDefinition] = {}
    unsupported: list[UnsupportedCatalogDrink] = []
    for row in selected:
        drink_id = str(row["id"])
        name = str(row["name"])
        raw_actions = row.get("actions", "")
        compiled_actions = _compile_action_sequence(raw_actions)
        reasons: list[str] = []
        if compiled_actions is None:
            for statement in _split_top_level(raw_actions):
                if _compile_action_sequence(statement) is None:
                    reasons.append(f"action_not_compiled:{statement}")
        elif not compiled_actions:
            reasons.append("no_compiled_effect")
        secondary_count = sum(
            effect.kind
            in {
                EffectKind.HOLD_SELECTED,
                EffectKind.FREE_PLAY_SELECTED,
                EffectKind.MOVE_SELECTED_TO_HAND,
            }
            for effect in compiled_actions or ()
        )
        if secondary_count > 1:
            reasons.append("multiple_secondary_choices_not_compiled")
        if reasons:
            unsupported.append(
                UnsupportedCatalogDrink(
                    drink_id,
                    name,
                    tuple(dict.fromkeys(reasons)),
                )
            )
            continue
        assert compiled_actions is not None
        definition = DrinkDefinition(
            drink_id=drink_id,
            name=name,
            effects=compiled_actions,
            plan=(
                None
                if row.get("plan") == "free"
                else plan_type
            ),
        )
        definition.validate()
        drinks[drink_id] = definition
    return CompiledDrinkCatalog(
        source_path=str(source),
        source_sha256=hashlib.sha256(source_bytes).hexdigest().upper(),
        plan_type=plan_type,
        drinks=drinks,
        unsupported=tuple(unsupported),
        total_rows=len(selected),
    )


def compile_item_catalog(
    path: str | Path,
    *,
    plan_type: PlanType,
) -> CompiledItemCatalog:
    source = Path(path)
    source_bytes = source.read_bytes()
    rows = list(csv.DictReader(source_bytes.decode("utf-8-sig").splitlines()))
    selected = [
        row
        for row in rows
        if row.get("mode") == "stage"
        and row.get("plan") in {"free", plan_type.value.lower()}
    ]
    items: dict[str, ItemDefinition] = {}
    unsupported: list[UnsupportedCatalogItem] = []
    for row in selected:
        item_id = str(row["id"])
        name = str(row["name"])
        raw_effects = row.get("effects", "")
        compiled_effects = _compile_action_sequence(raw_effects)
        reasons: list[str] = []
        if compiled_effects is None:
            for statement in _split_top_level(raw_effects):
                if _compile_action_sequence(statement) is None:
                    reasons.append(f"effect_not_compiled:{statement}")
        elif any(effect.kind is not EffectKind.SCHEDULE for effect in compiled_effects):
            reasons.append("nonpersistent_item_effect_not_compiled")
        if reasons:
            unsupported.append(
                UnsupportedCatalogItem(
                    item_id,
                    name,
                    tuple(dict.fromkeys(reasons)),
                )
            )
            continue
        assert compiled_effects is not None
        definition = ItemDefinition(
            item_id=item_id,
            name=name,
            persistent_effects=compiled_effects,
            plan=(None if row.get("plan") == "free" else plan_type),
        )
        definition.validate()
        items[item_id] = definition
    return CompiledItemCatalog(
        source_path=str(source),
        source_sha256=hashlib.sha256(source_bytes).hexdigest().upper(),
        plan_type=plan_type,
        items=items,
        unsupported=tuple(unsupported),
        total_rows=len(selected),
    )
