from __future__ import annotations

import random
import hashlib
from enum import Enum
from typing import Mapping
from dataclasses import replace, dataclass

from .kernel import CardDefinition, ItemDefinition
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


@dataclass(frozen=True)
class ItemInventoryScenario:
    scenario_id: str
    item_ids: tuple[str, ...]

    def validate(self) -> None:
        if not self.scenario_id or any(character.isspace() for character in self.scenario_id):
            raise ValueError("item inventory scenario requires a stable non-whitespace ID")
        if any(not item_id for item_id in self.item_ids):
            raise ValueError("item inventory IDs cannot be empty")
        if len(self.item_ids) != len(set(self.item_ids)):
            raise ValueError("item inventory cannot contain duplicate business IDs")


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


def build_fixed_item_inventory_scenarios(
    item_ids: tuple[str, ...],
    *,
    pairing_seed: int,
) -> tuple[ItemInventoryScenario, ...]:
    """Build deterministic empty, singleton and pair inventory scenarios."""

    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item scenario source IDs must be unique")
    if any(not item_id for item_id in item_ids):
        raise ValueError("item scenario source IDs cannot be empty")
    ordered = sorted(item_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    scenarios = [ItemInventoryScenario("empty", ())]
    scenarios.extend(ItemInventoryScenario(f"single-{item_id}", (item_id,)) for item_id in ordered)
    shuffled = list(ordered)
    random.Random(pairing_seed).shuffle(shuffled)
    if len(shuffled) > 1:
        for index in range(0, len(shuffled), 2):
            left = shuffled[index]
            right = shuffled[index + 1] if index + 1 < len(shuffled) else shuffled[0]
            pair = tuple(sorted((left, right), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)))
            scenarios.append(ItemInventoryScenario(f"pair-{'-'.join(pair)}", pair))
    for scenario in scenarios:
        scenario.validate()
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("item inventory scenario IDs must be unique")
    return tuple(scenarios)


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


class NiaProItemLifecycleGenerator:
    """Expand fixed dynamic decks into paired stage-entry item inventories.

    Every inventory scenario reuses the same deck seed, mutation history and
    synthetic turn attributes.
    The resulting state starts at turn one with no historical schedules or
    resources, so persistent effects can be initialized exactly once before
    the first policy decision instead of being injected into a later snapshot.
    """

    generator_name = "nia-pro-item-lifecycle-paired"
    generator_version = "1.1"

    def __init__(
        self,
        base_generator: NiaProDynamicDeckGenerator,
        item_definitions: Mapping[str, ItemDefinition],
        *,
        inventory_scenarios: tuple[ItemInventoryScenario, ...],
        stage_turns: int,
        max_stamina: int = 30,
    ):
        self.base_generator = base_generator
        self.item_definitions = dict(item_definitions)
        self.inventory_scenarios = inventory_scenarios
        self.stage_turns = stage_turns
        self.max_stamina = max_stamina
        self._validate_config()

    def _validate_config(self) -> None:
        if self.stage_turns < 1:
            raise ValueError("item lifecycle stage_turns must be positive")
        if self.max_stamina < 1:
            raise ValueError("item lifecycle max_stamina must be positive")
        if not self.inventory_scenarios:
            raise ValueError("item lifecycle requires inventory scenarios")
        scenario_ids: set[str] = set()
        for scenario in self.inventory_scenarios:
            scenario.validate()
            if scenario.scenario_id in scenario_ids:
                raise ValueError("item inventory scenario IDs must be unique")
            scenario_ids.add(scenario.scenario_id)
            unknown = sorted(set(scenario.item_ids) - set(self.item_definitions))
            if unknown:
                raise ValueError(f"item inventory references unknown definitions: {unknown}")
            mismatched = sorted(
                item_id for item_id in scenario.item_ids if self.item_definitions[item_id].plan not in {None, self.base_generator.plan_type}
            )
            if mismatched:
                raise ValueError(f"item inventory is not legal for {self.base_generator.plan_type.value}: {mismatched}")

    def generate(
        self,
        *,
        seed: int,
        episode_count_per_inventory: int,
    ) -> tuple[DynamicDeckSample, ...]:
        if episode_count_per_inventory < 1:
            raise ValueError("episode_count_per_inventory must be positive")
        base_samples = self.base_generator.generate(
            seed=seed,
            episode_count=episode_count_per_inventory,
        )
        attributes = ("vocal", "dance", "visual")
        samples: list[DynamicDeckSample] = []
        for scenario in self.inventory_scenarios:
            inventory_signature = ",".join(scenario.item_ids) or "empty"
            for base_sample in base_samples:
                deck = base_sample.request.state.deck
                state = StrategyState(
                    turn=1,
                    turns_remaining=self.stage_turns,
                    score=0,
                    stamina=self.max_stamina,
                    max_stamina=self.max_stamina,
                    genki=0,
                    actions_remaining=1,
                    hand_limit=5,
                    score_multiplier=1.0,
                    deck=deck,
                    items=scenario.item_ids,
                    turn_attributes=tuple(
                        attributes[(base_sample.seed + turn) % len(attributes)] for turn in range(self.stage_turns)
                    ),
                )
                request = replace(
                    base_sample.request,
                    snapshot_id=f"{base_sample.request.snapshot_id}-items-{scenario.scenario_id}",
                    state=state,
                    card_slots=tuple((card.instance_id, index) for index, card in enumerate(deck.hand)),
                )
                request.validate()
                samples.append(
                    DynamicDeckSample(
                        episode_id=f"{base_sample.episode_id}-items-{scenario.scenario_id}",
                        seed=base_sample.seed,
                        split=split_for_deck_signature(f"{base_sample.deck_signature}|items:{inventory_signature}"),
                        deck_signature=base_sample.deck_signature,
                        mutation_events=base_sample.mutation_events,
                        request=request,
                    )
                )
        return tuple(samples)
