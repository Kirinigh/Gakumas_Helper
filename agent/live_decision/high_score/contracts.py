from __future__ import annotations

from enum import Enum
from typing import Any
from dataclasses import field, asdict, dataclass

HIGH_SCORE_PROTOCOL_VERSION = "2.0"
NIA_PRO_RULESET_ID = "nia-pro-task275-probe-v24"


class HighScoreContractError(ValueError):
    """Raised when a Task-275 request cannot describe a trustworthy decision."""


class PlanType(str, Enum):
    SENSE = "SENSE"
    LOGIC = "LOGIC"
    ANOMALY = "ANOMALY"


class DecisionKind(str, Enum):
    CARD = "CARD"
    DRINK = "DRINK"
    END_TURN = "END_TURN"
    SECONDARY = "SECONDARY"


class DecisionOutcome(str, Enum):
    DECIDE = "DECIDE"
    REOBSERVE = "REOBSERVE"
    STOP_TASK = "STOP_TASK"


class ActionKind(str, Enum):
    PLAY_CARD = "PLAY_CARD"
    USE_DRINK = "USE_DRINK"
    END_TURN = "END_TURN"
    HOLD_CARD = "HOLD_CARD"
    MOVE_CARD = "MOVE_CARD"
    FREE_PLAY = "FREE_PLAY"
    PAID_PLAY = "PAID_PLAY"
    CHOOSE_TARGET = "CHOOSE_TARGET"
    FINISH_SELECTION = "FINISH_SELECTION"


class RiskProfile(str, Enum):
    SAFE_BASELINE = "SAFE_BASELINE"
    ROBUST_HIGH_SCORE = "ROBUST_HIGH_SCORE"
    AGGRESSIVE_HIGH_SCORE = "AGGRESSIVE_HIGH_SCORE"


class DeckKnowledge(str, Enum):
    KNOWN_ORDER = "KNOWN_ORDER"
    KNOWN_COMPOSITION = "KNOWN_COMPOSITION"
    UNKNOWN = "UNKNOWN"
    BROKEN = "BROKEN"


class CardZone(str, Enum):
    HAND = "HAND"
    DRAW_PILE = "DRAW_PILE"
    DISCARD = "DISCARD"
    REMOVED = "REMOVED"
    HELD = "HELD"


class SchedulePhase(str, Enum):
    BEFORE_CARD_USED = "BEFORE_CARD_USED"
    CARD_USED = "CARD_USED"
    AFTER_CARD_USED = "AFTER_CARD_USED"
    FULL_POWER_CHARGE_INCREASED = "FULL_POWER_CHARGE_INCREASED"
    CARD_MOVED_TO_HAND = "CARD_MOVED_TO_HAND"
    CARD_MOVED_TO_HELD = "CARD_MOVED_TO_HELD"
    END_OF_TURN = "END_OF_TURN"
    STANCE_CHANGED = "STANCE_CHANGED"
    BEFORE_START_OF_TURN = "BEFORE_START_OF_TURN"
    START_OF_TURN = "START_OF_TURN"
    AFTER_START_OF_TURN = "AFTER_START_OF_TURN"
    TURN = "TURN"


class EffectSourceKind(str, Enum):
    CARD = "CARD"
    DRINK = "DRINK"
    ITEM = "ITEM"


class TurnResolutionStage(str, Enum):
    BEFORE_START = "BEFORE_START"
    START = "START"
    REFILL_SETUP = "REFILL_SETUP"
    REFILL = "REFILL"
    HELD_RETURN = "HELD_RETURN"
    AFTER_START = "AFTER_START"
    TURN = "TURN"
    COMPLETE = "COMPLETE"


class ResolutionKind(str, Enum):
    CARD_EFFECTS = "CARD_EFFECTS"
    SCHEDULED_EFFECTS = "SCHEDULED_EFFECTS"
    FREE_PLAY_ALL = "FREE_PLAY_ALL"
    TURN = "TURN"


class Stance(str, Enum):
    NEUTRAL = "NEUTRAL"
    CONSERVE_1 = "CONSERVE_1"
    CONSERVE_2 = "CONSERVE_2"
    AGGRESSIVE_1 = "AGGRESSIVE_1"
    AGGRESSIVE_2 = "AGGRESSIVE_2"
    LEISURE = "LEISURE"
    FULL_POWER = "FULL_POWER"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class CardRef:
    instance_id: str
    card_id: str
    upgraded: bool = False
    score_bonus: int = 0
    genki_bonus: int = 0
    stamina_cost_delta: int = 0
    typed_cost_delta: int = 0
    score_repeat_bonus: int = 0

    def validate(self) -> None:
        if not self.instance_id or not self.card_id:
            raise HighScoreContractError("card instance_id and card_id are required")
        if self.score_repeat_bonus < 0:
            raise HighScoreContractError("card score repeat bonus cannot be negative")

    @property
    def semantic_key(self) -> tuple[str, bool, int, int, int, int, int]:
        return (
            self.card_id,
            self.upgraded,
            self.score_bonus,
            self.genki_bonus,
            self.stamina_cost_delta,
            self.typed_cost_delta,
            self.score_repeat_bonus,
        )


@dataclass(frozen=True)
class DeckBelief:
    all_cards: tuple[CardRef, ...]
    hand: tuple[CardRef, ...]
    draw_pile: tuple[CardRef, ...]
    discard: tuple[CardRef, ...] = ()
    removed: tuple[CardRef, ...] = ()
    held: tuple[CardRef, ...] = ()
    knowledge: DeckKnowledge = DeckKnowledge.KNOWN_COMPOSITION

    @property
    def zones(self) -> tuple[CardRef, ...]:
        return self.hand + self.draw_pile + self.discard + self.removed + self.held

    @property
    def conservation_valid(self) -> bool:
        expected = {card.instance_id: card for card in self.all_cards}
        actual = {card.instance_id: card for card in self.zones}
        return (
            len(expected) == len(self.all_cards)
            and len(actual) == len(self.zones)
            and expected == actual
        )

    def validate(self) -> None:
        for card in self.all_cards:
            card.validate()
        for card in self.zones:
            card.validate()
        if not self.conservation_valid:
            raise HighScoreContractError("deck conservation is broken")
        if len(self.held) > 2:
            raise HighScoreContractError("held card limit is two")
        if self.knowledge is DeckKnowledge.KNOWN_ORDER and not self.draw_pile and self.all_cards:
            # An empty draw pile is valid only when every card is in another zone.
            if len(self.all_cards) != len(self.hand + self.discard + self.removed + self.held):
                raise HighScoreContractError("known draw order is incomplete")

    def replace_zones(
        self,
        *,
        all_cards: tuple[CardRef, ...] | None = None,
        hand: tuple[CardRef, ...] | None = None,
        draw_pile: tuple[CardRef, ...] | None = None,
        discard: tuple[CardRef, ...] | None = None,
        removed: tuple[CardRef, ...] | None = None,
        held: tuple[CardRef, ...] | None = None,
        knowledge: DeckKnowledge | None = None,
    ) -> DeckBelief:
        updated = DeckBelief(
            all_cards=self.all_cards if all_cards is None else all_cards,
            hand=self.hand if hand is None else hand,
            draw_pile=self.draw_pile if draw_pile is None else draw_pile,
            discard=self.discard if discard is None else discard,
            removed=self.removed if removed is None else removed,
            held=self.held if held is None else held,
            knowledge=self.knowledge if knowledge is None else knowledge,
        )
        updated.validate()
        return updated


@dataclass(frozen=True)
class TimedStatus:
    status_id: str
    stacks: int = 0
    remaining_turns: int | None = None

    def validate(self) -> None:
        if not self.status_id:
            raise HighScoreContractError("status_id is required")
        if self.stacks < 0 or (self.remaining_turns is not None and self.remaining_turns < 0):
            raise HighScoreContractError("status values cannot be negative")


@dataclass(frozen=True)
class TimedModifier:
    amount: float
    remaining_turns: int | None = None
    fresh: bool = True

    def validate(self) -> None:
        if self.amount < 0:
            raise HighScoreContractError("timed modifier amount cannot be negative")
        if self.remaining_turns is not None and self.remaining_turns < 1:
            raise HighScoreContractError("timed modifier duration must be positive")


@dataclass(frozen=True)
class ScheduledEffectRef:
    source_id: str
    card_id: str
    definition_effect_index: int
    turns_until_fire: int = 1
    trigger_phase: SchedulePhase = SchedulePhase.TURN
    remaining_uses: int | None = 1
    remaining_turns: int | None = None
    persistent: bool = False
    direct_on_trigger: bool = False
    source_kind: EffectSourceKind = EffectSourceKind.CARD

    def validate(self) -> None:
        if not self.source_id or not self.card_id:
            raise HighScoreContractError("scheduled effect requires source and card ids")
        if not isinstance(self.source_kind, EffectSourceKind):
            raise HighScoreContractError("scheduled effect source kind is invalid")
        expected_prefix = {
            EffectSourceKind.DRINK: "drink",
            EffectSourceKind.ITEM: "item",
        }.get(self.source_kind)
        if (
            expected_prefix is not None
            and self.source_id != f"{expected_prefix}:{self.card_id}"
        ):
            raise HighScoreContractError(
                f"scheduled {expected_prefix} source is invalid"
            )
        if self.source_kind is EffectSourceKind.DRINK and self.persistent:
            raise HighScoreContractError("scheduled drink effects cannot be persistent")
        if self.source_kind is EffectSourceKind.ITEM and not self.persistent:
            raise HighScoreContractError("scheduled item effects must be persistent")
        minimum_delay = (
            0
            if self.trigger_phase
            in {
                SchedulePhase.BEFORE_CARD_USED,
                SchedulePhase.CARD_USED,
                SchedulePhase.AFTER_CARD_USED,
                SchedulePhase.END_OF_TURN,
                SchedulePhase.FULL_POWER_CHARGE_INCREASED,
                SchedulePhase.CARD_MOVED_TO_HAND,
                SchedulePhase.CARD_MOVED_TO_HELD,
                SchedulePhase.STANCE_CHANGED,
            }
            else 1
        )
        if self.definition_effect_index < 0 or self.turns_until_fire < minimum_delay:
            raise HighScoreContractError("scheduled effect index and delay are invalid")
        if not isinstance(self.trigger_phase, SchedulePhase):
            raise HighScoreContractError("scheduled effect trigger phase is invalid")
        minimum_uses = 0 if self.persistent else 1
        if self.remaining_uses is not None and self.remaining_uses < minimum_uses:
            raise HighScoreContractError("scheduled effect remaining uses are invalid")
        if self.remaining_turns is not None and self.remaining_turns < 0:
            raise HighScoreContractError("scheduled effect remaining turns are invalid")


@dataclass(frozen=True)
class EffectTriggerContext:
    is_direct_effect: bool = False
    used_card_instance_id: str | None = None
    used_card_id: str | None = None
    used_card_base_id: str | None = None
    used_card_type: str | None = None
    used_card_rarity: str | None = None
    moved_card_instance_id: str | None = None
    moved_card_id: str | None = None
    moved_card_base_id: str | None = None
    exclude_used_card_from_targets: bool = False


@dataclass(frozen=True)
class CardEffectsContinuation:
    continuation_kind: ResolutionKind = field(
        default=ResolutionKind.CARD_EFFECTS,
        init=False,
    )
    source_id: str
    card_id: str
    continuation_index: int
    remaining_effect_passes: int
    condition_stance: Stance
    condition_full_power: float

    def validate(self) -> None:
        if not self.source_id or not self.card_id or self.continuation_index < 0:
            raise HighScoreContractError("card continuation reference is invalid")
        if self.remaining_effect_passes < 0 or self.condition_full_power < 0:
            raise HighScoreContractError("card continuation state is invalid")


@dataclass(frozen=True)
class ScheduledEffectsContinuation:
    continuation_kind: ResolutionKind = field(
        default=ResolutionKind.SCHEDULED_EFFECTS,
        init=False,
    )
    scheduled: ScheduledEffectRef
    next_nested_index: int
    matched_nested_indices: tuple[int, ...]
    remaining_due: tuple[ScheduledEffectRef, ...] = ()
    remaining_matched_nested_indices: tuple[tuple[int, ...], ...] = ()
    trigger_context: EffectTriggerContext | None = None
    fired: bool = False

    def validate(self) -> None:
        self.scheduled.validate()
        if self.next_nested_index < 0:
            raise HighScoreContractError("scheduled continuation index is invalid")
        if any(index < 0 for index in self.matched_nested_indices):
            raise HighScoreContractError("scheduled continuation match index is invalid")
        for scheduled in self.remaining_due:
            scheduled.validate()
        if len(self.remaining_due) != len(self.remaining_matched_nested_indices):
            raise HighScoreContractError("scheduled continuation match sets are incomplete")
        if any(
            any(index < 0 for index in indices)
            for indices in self.remaining_matched_nested_indices
        ):
            raise HighScoreContractError("scheduled continuation remaining match index is invalid")


@dataclass(frozen=True)
class FreePlayAllContinuation:
    continuation_kind: ResolutionKind = field(
        default=ResolutionKind.FREE_PLAY_ALL,
        init=False,
    )
    source_id: str
    remaining_instance_ids: tuple[str, ...]

    def validate(self) -> None:
        if not self.source_id:
            raise HighScoreContractError("free-play-all continuation requires a source")
        if len(self.remaining_instance_ids) != len(set(self.remaining_instance_ids)):
            raise HighScoreContractError("free-play-all continuation targets must be unique")


@dataclass(frozen=True)
class TurnResolutionContinuation:
    continuation_kind: ResolutionKind = field(
        default=ResolutionKind.TURN,
        init=False,
    )
    plan_type: PlanType
    stage: TurnResolutionStage
    entered_full_power: bool = False
    due_before_start: tuple[ScheduledEffectRef, ...] = ()
    due_start: tuple[ScheduledEffectRef, ...] = ()
    due_after_start: tuple[ScheduledEffectRef, ...] = ()
    due_turn: tuple[ScheduledEffectRef, ...] = ()
    draws_remaining: int = 0
    held_return_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.plan_type, PlanType) or not isinstance(
            self.stage,
            TurnResolutionStage,
        ):
            raise HighScoreContractError("turn continuation stage is invalid")
        if self.draws_remaining < 0:
            raise HighScoreContractError("turn continuation draw count is invalid")
        if len(self.held_return_ids) != len(set(self.held_return_ids)):
            raise HighScoreContractError("turn continuation held targets must be unique")
        for scheduled in (
            self.due_before_start
            + self.due_start
            + self.due_after_start
            + self.due_turn
        ):
            scheduled.validate()


ResolutionContinuation = (
    CardEffectsContinuation
    | ScheduledEffectsContinuation
    | FreePlayAllContinuation
    | TurnResolutionContinuation
)


@dataclass(frozen=True)
class PendingChoice:
    action_kind: ActionKind
    source_id: str
    candidate_instance_ids: tuple[str, ...]
    continuation_index: int | None = None
    remaining_effect_passes: int = 0
    condition_stance: Stance | None = None
    condition_full_power: float | None = None
    selections_remaining: int = 1
    selection_optional: bool = False

    def validate(self) -> None:
        if self.action_kind not in {
            ActionKind.HOLD_CARD,
            ActionKind.MOVE_CARD,
            ActionKind.FREE_PLAY,
            ActionKind.PAID_PLAY,
        }:
            raise HighScoreContractError("pending choice uses unsupported action kind")
        if not self.source_id or not self.candidate_instance_ids:
            raise HighScoreContractError("pending choice requires source and candidates")
        if len(self.candidate_instance_ids) != len(set(self.candidate_instance_ids)):
            raise HighScoreContractError("pending choice candidates must be unique")
        if self.continuation_index is not None and self.continuation_index < 1:
            raise HighScoreContractError("pending continuation index must be positive")
        if self.remaining_effect_passes < 0:
            raise HighScoreContractError("pending remaining effect passes cannot be negative")
        if self.selections_remaining < 1:
            raise HighScoreContractError("pending selections remaining must be positive")
        if self.action_kind is not ActionKind.HOLD_CARD and (
            self.selections_remaining != 1 or self.selection_optional
        ):
            raise HighScoreContractError("multi-select state is only valid for hold choices")
        if not self.selection_optional and len(self.candidate_instance_ids) < self.selections_remaining:
            raise HighScoreContractError("required selection has too few candidates")
        has_continuation = self.continuation_index is not None or self.remaining_effect_passes > 0
        has_snapshot = self.condition_stance is not None or self.condition_full_power is not None
        if has_snapshot != has_continuation:
            raise HighScoreContractError("pending continuation requires a complete condition snapshot")
        if has_continuation and (
            self.condition_stance is None or self.condition_full_power is None
        ):
            raise HighScoreContractError("pending continuation condition snapshot is incomplete")


@dataclass(frozen=True)
class StrategyState:
    turn: int
    turns_remaining: int
    score: int
    stamina: int
    max_stamina: int
    genki: int
    actions_remaining: int
    hand_limit: int
    score_multiplier: float
    deck: DeckBelief
    pending_choice: PendingChoice | None = None
    pending_after_card_ids: tuple[str, ...] = ()
    resolution_stack: tuple[ResolutionContinuation, ...] = ()
    stance: Stance = Stance.NEUTRAL
    previous_stance: Stance = Stance.NEUTRAL
    full_power: float = 0.0
    cumulative_full_power: float = 0.0
    passion: float = 0.0
    passion_buff: int = 0
    passion_gain_bonus_pct: float = 0.0
    parameter_gain_bonus_pct: float = 0.0
    concentration: int = 0
    good_condition_turns: int = 0
    excellent_condition_turns: int = 0
    good_impression_turns: int = 0
    motivation: int = 0
    statuses: tuple[TimedStatus, ...] = ()
    scheduled_effects: tuple[ScheduledEffectRef, ...] = ()
    persistent_effects_initialized: bool = False
    double_card_effects_remaining: int = 0
    half_cost_turns: int = 0
    half_cost_fresh: bool = False
    double_cost_turns: int = 0
    double_cost_fresh: bool = False
    stance_lock_turns: int = 0
    stance_lock_fresh: bool = False
    nullify_cost_cards: int = 0
    strength_times: int = 0
    preservation_times: int = 0
    full_power_times: int = 0
    leisure_times: int = 0
    stance_changed_by_direct_effect_times: int = 0
    created_card_count: int = 0
    enthusiasm_bonus_buffs: tuple[TimedModifier, ...] = ()
    enthusiasm_multiplier_buffs: tuple[TimedModifier, ...] = ()
    full_power_charge_buffs: tuple[TimedModifier, ...] = ()
    score_buffs: tuple[TimedModifier, ...] = ()
    drinks: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    field_effects: tuple[str, ...] = ()
    future_score_multipliers: tuple[float, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.turns_remaining <= 0

    def validate(self) -> None:
        if self.turn < 1 or self.turns_remaining < 0:
            raise HighScoreContractError("turn starts at one and turns_remaining cannot be negative")
        if self.score < 0 or self.stamina < 0 or self.max_stamina < 0 or self.stamina > self.max_stamina:
            raise HighScoreContractError("score or stamina is outside its valid range")
        if (
            self.genki < 0
            or self.actions_remaining < 0
            or self.hand_limit < 0
            or self.full_power < 0
            or self.cumulative_full_power < 0
            or self.passion < 0
            or self.passion_buff < 0
            or self.concentration < 0
            or self.good_condition_turns < 0
            or self.excellent_condition_turns < 0
            or self.good_impression_turns < 0
            or self.motivation < 0
            or self.double_card_effects_remaining < 0
            or self.half_cost_turns < 0
            or self.double_cost_turns < 0
            or self.stance_lock_turns < 0
            or self.nullify_cost_cards < 0
            or self.strength_times < 0
            or self.preservation_times < 0
            or self.full_power_times < 0
            or self.leisure_times < 0
            or self.stance_changed_by_direct_effect_times < 0
            or self.created_card_count < 0
        ):
            raise HighScoreContractError("resources cannot be negative")
        if not isinstance(self.stance, Stance) or not isinstance(self.previous_stance, Stance):
            raise HighScoreContractError("stance values are invalid")
        if self.half_cost_fresh and self.half_cost_turns == 0:
            raise HighScoreContractError("fresh half-cost marker requires active half-cost turns")
        if self.double_cost_fresh and self.double_cost_turns == 0:
            raise HighScoreContractError("fresh double-cost marker requires active double-cost turns")
        if self.stance_lock_fresh and self.stance_lock_turns == 0:
            raise HighScoreContractError("fresh stance-lock marker requires active stance-lock turns")
        if self.score_multiplier <= 0 or any(multiplier <= 0 for multiplier in self.future_score_multipliers):
            raise HighScoreContractError("score multipliers must be positive")
        for status in self.statuses:
            status.validate()
        for modifier in (
            self.enthusiasm_bonus_buffs
            + self.enthusiasm_multiplier_buffs
            + self.full_power_charge_buffs
            + self.score_buffs
        ):
            modifier.validate()
        for scheduled in self.scheduled_effects:
            scheduled.validate()
        for continuation in self.resolution_stack:
            continuation.validate()
        if not self.persistent_effects_initialized and any(
            scheduled.persistent for scheduled in self.scheduled_effects
        ):
            raise HighScoreContractError("persistent effects require initialized state")
        self.deck.validate()
        cards_by_instance = {card.instance_id: card for card in self.deck.all_cards}
        if any(instance_id not in cards_by_instance for instance_id in self.pending_after_card_ids):
            raise HighScoreContractError("pending after-card event references unknown card instance")
        if any(
            scheduled.source_kind is EffectSourceKind.CARD
            and (
                scheduled.source_id not in cards_by_instance
                or (
                    cards_by_instance[scheduled.source_id].card_id != scheduled.card_id
                    and not cards_by_instance[scheduled.source_id].upgraded
                )
            )
            for scheduled in self.scheduled_effects
        ):
            raise HighScoreContractError("scheduled effect source does not match the deck ledger")
        if any(
            scheduled.source_kind is EffectSourceKind.ITEM
            and scheduled.card_id not in self.items
            for scheduled in self.scheduled_effects
        ):
            raise HighScoreContractError("scheduled item source does not match the item ledger")
        if self.pending_choice is not None:
            self.pending_choice.validate()
            known_instances = {card.instance_id for card in self.deck.all_cards}
            if not set(self.pending_choice.candidate_instance_ids) <= known_instances:
                raise HighScoreContractError("pending choice references unknown card instances")
        referenced_continuation_ids = {
            continuation.source_id
            for continuation in self.resolution_stack
            if isinstance(
                continuation,
                (CardEffectsContinuation, FreePlayAllContinuation),
            )
        }
        referenced_continuation_ids.update(
            instance_id
            for continuation in self.resolution_stack
            if isinstance(continuation, FreePlayAllContinuation)
            for instance_id in continuation.remaining_instance_ids
        )
        if not referenced_continuation_ids <= set(cards_by_instance):
            raise HighScoreContractError("resolution continuation references unknown card instances")


@dataclass(frozen=True)
class CapabilityReport:
    observation_valid: bool
    game_legal: bool
    model_supported: bool
    policy_eligible: bool
    execution_ready: bool
    reason_codes: tuple[str, ...] = ()

    @property
    def all_ready(self) -> bool:
        return all(
            (
                self.observation_valid,
                self.game_legal,
                self.model_supported,
                self.policy_eligible,
                self.execution_ready,
            )
        )


@dataclass(frozen=True)
class UtilitySpec:
    risk_profile: RiskProfile = RiskProfile.ROBUST_HIGH_SCORE
    gamma: float = 1.0
    baseline_p10: float | None = None
    minimum_stamina: int = 0
    minimum_genki: int = 0

    def validate(self) -> None:
        if self.gamma != 1.0:
            raise HighScoreContractError("Task-275 final-score utility requires gamma=1")
        if self.minimum_stamina < 0 or self.minimum_genki < 0:
            raise HighScoreContractError("resource floors cannot be negative")


@dataclass(frozen=True)
class GameAction:
    kind: ActionKind
    card_instance_id: str | None = None
    drink_id: str | None = None
    target_id: str | None = None
    source_slot: int | None = None

    def validate(self) -> None:
        if self.kind in {
            ActionKind.PLAY_CARD,
            ActionKind.HOLD_CARD,
            ActionKind.MOVE_CARD,
            ActionKind.FREE_PLAY,
            ActionKind.PAID_PLAY,
        }:
            if not self.card_instance_id:
                raise HighScoreContractError(f"{self.kind.value} requires card_instance_id")
        if self.kind is ActionKind.USE_DRINK and not self.drink_id:
            raise HighScoreContractError("USE_DRINK requires drink_id")
        if self.kind is ActionKind.CHOOSE_TARGET and not self.target_id:
            raise HighScoreContractError("CHOOSE_TARGET requires target_id")


@dataclass(frozen=True)
class DecisionRequest:
    snapshot_id: str
    plan_type: PlanType
    decision_kind: DecisionKind
    state: StrategyState
    capabilities: CapabilityReport
    utility: UtilitySpec = field(default_factory=UtilitySpec)
    card_slots: tuple[tuple[str, int], ...] = ()
    reobserve_attempt: int = 0
    ruleset_id: str = NIA_PRO_RULESET_ID
    protocol_version: str = HIGH_SCORE_PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != HIGH_SCORE_PROTOCOL_VERSION:
            raise HighScoreContractError(f"unsupported high-score protocol: {self.protocol_version}")
        if not self.snapshot_id or not self.ruleset_id:
            raise HighScoreContractError("snapshot_id and ruleset_id are required")
        if self.reobserve_attempt < 0:
            raise HighScoreContractError("reobserve_attempt cannot be negative")
        self.state.validate()
        self.utility.validate()
        slot_ids = [instance_id for instance_id, _slot in self.card_slots]
        slots = [slot for _instance_id, slot in self.card_slots]
        if len(slot_ids) != len(set(slot_ids)) or len(slots) != len(set(slots)):
            raise HighScoreContractError("card slot mapping must be one-to-one")
        if self.state.pending_choice is not None:
            if self.decision_kind is not DecisionKind.SECONDARY:
                raise HighScoreContractError("pending choice requires SECONDARY decision kind")
            if set(slot_ids) != set(self.state.pending_choice.candidate_instance_ids):
                raise HighScoreContractError("secondary card slot mapping must cover every candidate")
        elif self.decision_kind is DecisionKind.SECONDARY:
            raise HighScoreContractError("SECONDARY decision kind requires pending choice")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ValueSummary:
    expected_final_score: float
    p10_final_score: float
    worst_final_score: float
    best_final_score: float
    outcome_count: int


@dataclass(frozen=True)
class CandidateValue:
    action: GameAction
    value: ValueSummary
    probability_mass: float
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionTrace:
    planner: str
    planner_version: str
    ruleset_id: str
    completed_depth: int
    expanded_nodes: int
    elapsed_ms: float
    candidate_values: tuple[CandidateValue, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionResponse:
    outcome: DecisionOutcome
    reason: str
    action: GameAction | None
    value: ValueSummary | None
    trace: DecisionTrace | None
    protocol_version: str = HIGH_SCORE_PROTOCOL_VERSION

    @property
    def has_action_intent(self) -> bool:
        return self.outcome is DecisionOutcome.DECIDE and self.action is not None

    def validate(self) -> None:
        if self.protocol_version != HIGH_SCORE_PROTOCOL_VERSION:
            raise HighScoreContractError("response protocol version mismatch")
        if self.outcome is DecisionOutcome.DECIDE:
            if self.action is None or self.value is None or self.trace is None:
                raise HighScoreContractError("DECIDE requires action, value and trace")
            self.action.validate()
        elif self.action is not None:
            raise HighScoreContractError("non-DECIDE response cannot carry an action")

    def to_dict(self) -> dict[str, Any]:
        result = _jsonable(asdict(self))
        result["has_action_intent"] = self.has_action_intent
        return result
