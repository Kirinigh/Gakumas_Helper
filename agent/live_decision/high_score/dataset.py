from __future__ import annotations

import random
import hashlib
from enum import Enum
from typing import Mapping
from dataclasses import replace, dataclass

from .kernel import CardDefinition
from .contracts import (
    Stance,
    CardRef,
    PlanType,
    DeckBelief,
    UtilitySpec,
    DecisionKind,
    DeckKnowledge,
    StrategyState,
    DecisionRequest,
    CapabilityReport,
)


class DeckMutationKind(str, Enum):
    ACQUIRE = "ACQUIRE"
    UPGRADE = "UPGRADE"
    DELETE = "DELETE"


class DatasetSplit(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


@dataclass(frozen=True)
class DeckMutationEvent:
    event_index: int
    kind: DeckMutationKind
    card_id: str
    instance_id: str
    previous_card_id: str | None = None


@dataclass(frozen=True)
class DynamicDeckSample:
    episode_id: str
    seed: int
    split: DatasetSplit
    deck_signature: str
    mutation_events: tuple[DeckMutationEvent, ...]
    request: DecisionRequest


def deck_signature(cards: tuple[CardRef, ...]) -> str:
    entries = sorted(":".join(str(value) for value in card.semantic_key) for card in cards)
    return "|".join(entries)


def split_for_deck_signature(signature: str) -> DatasetSplit:
    bucket = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return DatasetSplit.TRAIN
    if bucket < 85:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


class NiaProDynamicDeckGenerator:
    """Generate legal deck snapshots without optimizing cultivation card choices."""

    generator_name = "nia-pro-dynamic-deck"
    generator_version = "1.1"

    def __init__(
        self,
        card_definitions: Mapping[str, CardDefinition],
        *,
        plan_type: PlanType,
        base_card_ids: tuple[str, ...],
        acquisition_pool: tuple[str, ...],
        upgrade_map: Mapping[str, str] | None = None,
        minimum_deck_size: int = 5,
        mutation_count_range: tuple[int, int] = (2, 6),
        turns_remaining_range: tuple[int, int] = (1, 1),
    ):
        self.card_definitions = dict(card_definitions)
        self.plan_type = plan_type
        self.base_card_ids = base_card_ids
        self.acquisition_pool = acquisition_pool
        self.upgrade_map = dict(upgrade_map or {})
        self.minimum_deck_size = minimum_deck_size
        self.mutation_count_range = mutation_count_range
        self.turns_remaining_range = turns_remaining_range
        self._validate_config()

    def _validate_config(self) -> None:
        if self.minimum_deck_size < 1 or len(self.base_card_ids) < self.minimum_deck_size:
            raise ValueError("base deck must satisfy minimum_deck_size")
        if not self.acquisition_pool:
            raise ValueError("acquisition_pool cannot be empty")
        if self.mutation_count_range[0] < 0 or self.mutation_count_range[1] < self.mutation_count_range[0]:
            raise ValueError("invalid mutation_count_range")
        if self.turns_remaining_range[0] < 1 or self.turns_remaining_range[1] < self.turns_remaining_range[0]:
            raise ValueError("invalid turns_remaining_range")
        referenced = set(self.base_card_ids) | set(self.acquisition_pool) | set(self.upgrade_map) | set(self.upgrade_map.values())
        missing = sorted(referenced - set(self.card_definitions))
        if missing:
            raise ValueError(f"dynamic deck references unknown definitions: {missing}")
        for card_id in set(self.base_card_ids) | set(self.acquisition_pool):
            definition = self.card_definitions[card_id]
            if definition.plan not in {None, self.plan_type}:
                raise ValueError(f"card {card_id} is not legal for {self.plan_type.value}")
        if any(source == target for source, target in self.upgrade_map.items()):
            raise ValueError("upgrade_map cannot map a card to itself")

    def generate(self, *, seed: int, episode_count: int) -> tuple[DynamicDeckSample, ...]:
        if episode_count < 1:
            raise ValueError("episode_count must be positive")
        return tuple(self._episode(seed + episode_index) for episode_index in range(episode_count))

    def _episode(self, seed: int) -> DynamicDeckSample:
        rng = random.Random(seed)
        cards = [CardRef(f"{seed}-base-{index}", card_id) for index, card_id in enumerate(self.base_card_ids)]
        events: list[DeckMutationEvent] = []
        next_instance = len(cards)
        mutation_count = rng.randint(*self.mutation_count_range)
        for event_index in range(mutation_count):
            available = [DeckMutationKind.ACQUIRE]
            if any(not card.upgraded and (not self.upgrade_map or card.card_id in self.upgrade_map) for card in cards):
                available.append(DeckMutationKind.UPGRADE)
            if len(cards) > self.minimum_deck_size:
                available.append(DeckMutationKind.DELETE)
            kind = rng.choice(available)
            if kind is DeckMutationKind.ACQUIRE:
                card_id = rng.choice(self.acquisition_pool)
                instance_id = f"{seed}-acquired-{next_instance}"
                next_instance += 1
                cards.append(CardRef(instance_id, card_id))
            elif kind is DeckMutationKind.UPGRADE:
                eligible = [
                    index
                    for index, card in enumerate(cards)
                    if not card.upgraded and (not self.upgrade_map or card.card_id in self.upgrade_map)
                ]
                card_index = rng.choice(eligible)
                selected = cards[card_index]
                card_id = self.upgrade_map.get(selected.card_id, selected.card_id)
                cards[card_index] = replace(selected, card_id=card_id, upgraded=True)
                instance_id = selected.instance_id
            else:
                card_index = rng.randrange(len(cards))
                selected = cards.pop(card_index)
                card_id = selected.card_id
                instance_id = selected.instance_id
            events.append(
                DeckMutationEvent(
                    event_index,
                    kind,
                    card_id,
                    instance_id,
                    previous_card_id=selected.card_id if kind in {DeckMutationKind.UPGRADE, DeckMutationKind.DELETE} else None,
                )
            )

        rng.shuffle(cards)
        all_cards = tuple(cards)
        hand_size = min(5, len(all_cards))
        hand = all_cards[:hand_size]
        draw_pile = all_cards[hand_size:]
        signature = deck_signature(all_cards)
        deck = DeckBelief(
            all_cards=all_cards,
            hand=hand,
            draw_pile=draw_pile,
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        plan_state: dict[str, object]
        if self.plan_type is PlanType.ANOMALY:
            stance = rng.choice(
                (
                    Stance.NEUTRAL,
                    Stance.CONSERVE_1,
                    Stance.CONSERVE_2,
                    Stance.AGGRESSIVE_1,
                    Stance.LEISURE,
                    Stance.FULL_POWER,
                )
            )
            full_power = rng.randint(0, 8)
            strength_times = rng.randint(0, 4)
            preservation_times = rng.randint(0, 4)
            full_power_times = rng.randint(0, 2)
            leisure_times = rng.randint(0, 2)
            common_state = {
                "turns_remaining": rng.randint(*self.turns_remaining_range),
                "score": rng.randint(0, 500),
                "stamina": rng.randint(10, 30),
                "genki": rng.randint(0, 10),
                "actions_remaining": rng.choice((1, 2)),
                "score_multiplier": rng.choice((1.0, 1.2, 1.5)),
            }
            plan_state = {
                "stance": stance,
                "full_power": full_power,
                "cumulative_full_power": full_power + rng.randint(0, 12),
                "passion_buff": rng.randint(0, 4),
                "passion_gain_bonus_pct": rng.choice((0.0, 25.0, 50.0)),
                "parameter_gain_bonus_pct": rng.choice((0.0, 10.0, 25.0)),
                "half_cost_turns": rng.choice((0, 0, 0, 1, 2)),
                "stance_lock_turns": rng.choice((0, 0, 0, 1)),
                "nullify_cost_cards": rng.choice((0, 0, 0, 1, 2)),
                "strength_times": strength_times,
                "preservation_times": preservation_times,
                "full_power_times": full_power_times,
                "leisure_times": leisure_times,
                "stance_changed_by_direct_effect_times": rng.randint(
                    0,
                    strength_times
                    + preservation_times
                    + full_power_times
                    + leisure_times,
                ),
            }
        elif self.plan_type is PlanType.SENSE:
            plan_state = {
                "concentration": rng.randint(0, 12),
                "good_condition_turns": rng.randint(0, 5),
                "excellent_condition_turns": rng.randint(0, 3),
            }
            common_state = {
                "turns_remaining": rng.randint(*self.turns_remaining_range),
                "score": rng.randint(0, 500),
                "stamina": rng.randint(10, 30),
                "genki": rng.randint(0, 10),
                "actions_remaining": rng.choice((1, 2)),
                "score_multiplier": rng.choice((1.0, 1.2, 1.5)),
            }
        else:
            plan_state = {
                "good_impression_turns": rng.randint(0, 12),
                "motivation": rng.randint(0, 12),
            }
            common_state = {
                "turns_remaining": rng.randint(*self.turns_remaining_range),
                "score": rng.randint(0, 500),
                "stamina": rng.randint(10, 30),
                "genki": rng.randint(0, 10),
                "actions_remaining": rng.choice((1, 2)),
                "score_multiplier": rng.choice((1.0, 1.2, 1.5)),
            }
        state = StrategyState(
            turn=10,
            max_stamina=30,
            hand_limit=5,
            deck=deck,
            **common_state,
            **plan_state,
        )
        request = DecisionRequest(
            snapshot_id=f"dynamic-{seed}",
            plan_type=self.plan_type,
            decision_kind=DecisionKind.CARD,
            state=state,
            capabilities=CapabilityReport(True, True, True, True, True),
            utility=UtilitySpec(),
            card_slots=tuple((card.instance_id, index) for index, card in enumerate(hand)),
        )
        request.validate()
        return DynamicDeckSample(
            episode_id=f"nia-pro-{self.plan_type.value.lower()}-{seed}",
            seed=seed,
            split=split_for_deck_signature(signature),
            deck_signature=signature,
            mutation_events=tuple(events),
            request=request,
        )
