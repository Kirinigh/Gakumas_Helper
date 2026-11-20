"""Offline HIF Sense rules; no observation, Controller or input integration."""

import re
from math import comb
from typing import Mapping, Callable
from contextvars import ContextVar
from dataclasses import dataclass

from .kernel import (
    RuleEvent,
    EffectKind,
    EffectSpec,
    KernelError,
    CardSelector,
    ConditionKind,
    CardDefinition,
    ItemDefinition,
    DrinkDefinition,
    EffectCondition,
    TransitionBranch,
    PythonNiaProKernel,
    MemoryAbilityDefinition,
)
from .contracts import (
    HIF_SENSE_RULESET_ID,
    CardZone,
    PlanType,
    ActionKind,
    DeckBelief,
    DecisionKind,
    DeckKnowledge,
    SchedulePhase,
    StrategyState,
    TurnResolutionStage,
    TurnResolutionContinuation,
)
from .state_copy import replace
from ..status_rules import (
    memory_use_counts,
    _normalize_status_text,
    status_panel_consensus,
    join_status_panel_pages,
    memory_effect_description,
)
from .validation_cache import check_search_budget
from .hif_support_planning import HifSupportPlanningMixin, clean_hand_support

HIF_POLICY_PROFILE = "hif-sense-joint-chain-v5-tiered"
_BEFORE_PLAYER_INPUT = ContextVar("hif_before_player_input", default=False)


@dataclass(frozen=True)
class ObservedHifMemory:
    definition: MemoryAbilityDefinition
    remaining_uses: int
    count_source: str


def compile_observed_hif_memories(known_pages, current_pages, card_catalog, *, known_run_id, current_run_id):
    """Compile only observed trigger/effect text and D-067 confirmed counters.

    This produces memory definitions, not a ready StrategyState. Item/source
    ordering, card schedules and other run history still need their producers.
    No card ids, effect magnitudes or remaining counts are supplied by an agent.
    """
    known_pages = tuple(tuple(group) for group in known_pages)
    current_pages = tuple(tuple(group) for group in current_pages)
    counts = memory_use_counts(known_pages,current_pages,
                              previous_run_id=known_run_id,current_run_id=current_run_id)
    pages = tuple(status_panel_consensus(group) for group in known_pages)
    rows = pages[0].rows if len(pages) == 1 else join_status_panel_pages(pages)
    definitions = {}

    def named_card(name):
        # Trigger identity uses the full fixed metadata catalogue. A memory
        # can name an unowned card whose play effects the kernel cannot compile;
        # that does not make its identity or the separate memory effect unknown.
        found = [card for card in card_catalog.entries.values()
                 if card.plan in {'sense','free'} and _normalize_status_text(card.name) == name]
        if len(found) != 1:
            raise ValueError('memory card name is not uniquely registered')
        return found[0]

    for row in rows:
        if row.title != 'メモリーアビリティ':
            continue
        description = memory_effect_description(row)
        match = re.fullmatch(r'(.+)使用後、(.+)',description)
        if not match:
            raise ValueError('unsupported observed memory trigger')
        trigger = named_card(match[1])
        base_id = (named_card(_normalize_status_text(trigger.name).removesuffix('+')).card_id
                   if trigger.upgraded else trigger.card_id)
        condition = EffectCondition(ConditionKind.USED_CARD_BASE_ID,value=base_id)
        if extra := re.fullmatch(r'スキルカード使用数追加\+(\d+)(・スキルカードを引く)?',match[2]):
            amount = int(extra[1])
            if amount < 1:
                raise ValueError('memory extra-action count is invalid')
            effects = (EffectSpec(EffectKind.EXTRA_ACTION,amount,condition=condition),)
            ability_id = f'hif-memory-{base_id}-extra-{amount}'
            if extra[2]:
                effects += (EffectSpec(EffectKind.DRAW,1,condition=condition),)
                ability_id += '-draw-1'
        elif removal := re.fullmatch(r'ランダムな山札か捨札の(.+)を除外に移動',match[2]):
            target = named_card(removal[1])
            if target.card_type != 'trouble':
                raise ValueError('observed memory removal target is outside the verified trouble-card grammar')
            effects = (EffectSpec(EffectKind.REMOVE_RANDOM,1,condition=condition,
                card_selector=CardSelector(zones=(CardZone.DRAW_PILE,CardZone.DISCARD),card_id=target.card_id)),)
            ability_id = f'hif-memory-{base_id}-remove-{target.card_id}'
        else:
            raise ValueError('unsupported observed memory effect')
        definition = MemoryAbilityDefinition(ability_id,match[1]+'使用後',(
            EffectSpec(EffectKind.SCHEDULE,delay_turns=0,schedule_phase=SchedulePhase.AFTER_CARD_USED,
                       schedule_limit=row.count,scheduled_effects=effects),),PlanType.SENSE)
        definition.validate()
        if description in definitions or any(old.ability_id == ability_id for old in definitions.values()):
            raise ValueError('observed memory identity is duplicated')
        definitions[description] = definition
    return tuple(ObservedHifMemory(definitions[row['description_key']],row['remaining_uses'],row['source'])
                 for row in counts)


class HifSenseKernel(HifSupportPlanningMixin, PythonNiaProKernel):
    """Reuse the shared rules with explicit HIF scenario completeness gates.

    Every future turn multiplier must be supplied by the HIF scene producer.
    A mid-stage snapshot must carry initialized schedules; reconstructing them
    from card identities would silently reset counters and item uses.
    """

    kernel_name = "python-hif-sense-offline"
    # Native HIF 577/581: a newly applied end-turn timer still shows two
    # turns after its first trigger, matching other fresh duration effects.
    end_turn_schedule_initial_grace = True
    # Native 646/657: newly granted excellent condition remains 6 at
    # next-turn start, then the item adds 4. Other rulesets stay unchanged.
    excellent_condition_initial_grace = True
    # Selected retrieval follows the same capacity as DRAW/random retrieval.
    selected_move_respects_hand_limit = True

    def legal_actions(self, state, *, plan_type, decision_kind):
        if (decision_kind in (DecisionKind.LIVE, DecisionKind.DRINK) and state.drinks
                and all(getattr(self.drinks.get(key), 'reserved_for_later', False) for key in state.drinks)):
            actions = super().legal_actions(state, plan_type=plan_type, decision_kind=DecisionKind.CARD)
            return () if decision_kind is DecisionKind.DRINK else actions
        return super().legal_actions(state, plan_type=plan_type, decision_kind=decision_kind)

    def _scheduled_definition_effects(self, scheduled):
        from .support_schedule import schedule_definition_effects

        derived = schedule_definition_effects(self.cards, scheduled)
        return derived if derived is not None else super()._scheduled_definition_effects(scheduled)

    def _run_scheduled_nested(self, branch, scheduled, schedule_effect, **kwargs):
        before_input = _BEFORE_PLAYER_INPUT.get() or scheduled.trigger_phase in {
            SchedulePhase.BEFORE_START_OF_TURN,
            SchedulePhase.START_OF_TURN, SchedulePhase.AFTER_START_OF_TURN,
            SchedulePhase.TURN, SchedulePhase.EVERY_TURN,
        }
        token = _BEFORE_PLAYER_INPUT.set(before_input)
        try:
            return super()._run_scheduled_nested(branch, scheduled, schedule_effect, **kwargs)
        finally:
            _BEFORE_PLAYER_INPUT.reset(token)

    def _apply_non_draw_effect_with_freshness(self, state, effect, **kwargs):
        result, event = super()._apply_non_draw_effect(state, effect, **kwargs)
        before_input = _BEFORE_PLAYER_INPUT.get() or any(
            isinstance(frame, TurnResolutionContinuation) for frame in state.resolution_stack)
        changes = {}
        if effect.kind is EffectKind.GOOD_CONDITION:
            changes['good_condition_fresh'] = bool(result.good_condition_turns and (
                state.good_condition_fresh or (not before_input and state.good_condition_turns == 0)))
        if before_input:
            # New timed effects granted before the input phase have no grace.
            # Merged rows retain their original flag, including an existing
            # after-input grant; never reset freshness merely on reapplication.
            for name in ('score_buffs', 'concentration_effect_buffs', 'concentration_addition_buffs'):
                previous = {buff.remaining_turns: buff.fresh for buff in getattr(state, name)}
                current = getattr(result, name)
                if current != getattr(state, name):
                    changes[name] = tuple(replace(buff, fresh=previous.get(buff.remaining_turns, False))
                                          for buff in current)
            for name, marker in (('excellent_condition_turns', 'excellent_condition_fresh'),
                                 ('half_cost_turns', 'half_cost_fresh'),
                                 ('double_cost_turns', 'double_cost_fresh'),
                                 ('nullify_genki_turns', 'nullify_genki_fresh')):
                if getattr(result, name) and not getattr(state, name):
                    changes[marker] = False
        return (replace(result, **changes) if changes else result), event

    def resource_evaluation_transition(self, state, action, *, plan_type):
        """Exact scalar marginal for a draw-independent END_TURN heuristic.

        This projection has no usable deck and must never enter execution or
        settlement. The one-step Sense utility reads only scalar resources.
        Any reachable card/zone dependent effect retains the full kernel.
        """
        if (state.hand_support or plan_type is not PlanType.SENSE or action.kind is not ActionKind.END_TURN
                or state.pending_choice or state.resolution_stack
                or state.turns_remaining<2 or state.stance.value!='NEUTRAL'):
            return None
        scalar_kinds={EffectKind.DRAW,EffectKind.SCORE,EffectKind.GENKI,EffectKind.STAMINA,
            EffectKind.CONCENTRATION,EffectKind.GOOD_CONDITION,EffectKind.EXCELLENT_CONDITION,
            EffectKind.EXTRA_ACTION,EffectKind.COST_REDUCTION,EffectKind.CONCENTRATION_EFFECT_BUFF,
            EffectKind.CONCENTRATION_ADDITION_BUFF}
        scalar_conditions={ConditionKind.TURN_ATTRIBUTE,ConditionKind.GOOD_CONDITION_MIN,
            ConditionKind.STAMINA_RATIO_MIN,ConditionKind.EFFECT_COUNTER_MOD_EQ}
        replay_cards = set()
        def conditions(condition):
            if condition is None: return ()
            if condition.kind is ConditionKind.ALL:
                return tuple(child for part in condition.all_of for child in conditions(part))
            return (condition,)
        def inert(effect, phase, reference):
            for condition in conditions(effect.condition):
                if (condition.kind is ConditionKind.TURN_ATTRIBUTE
                        and phase in {SchedulePhase.START_OF_TURN,SchedulePhase.AFTER_START_OF_TURN}
                        and condition.value != state.turn_attributes[1]):
                    return True
                if condition.kind is ConditionKind.REMOVED_BASE_CARD_COUNT_MIN:
                    count=sum(self.cards[c.card_id].base_card_id==condition.value
                              for c in state.deck.removed)
                    if count < condition.threshold: return True
                if (condition.kind is ConditionKind.EFFECT_COUNTER_MOD_EQ
                        and phase is SchedulePhase.EVERY_TURN and condition.modulus
                        and reference.effect_counter%condition.modulus!=condition.remainder):
                    # Matching uses the counter before its one phase update.
                    return True
            return False
        def safe_replay(effect):
            # Ignoring draw identities is valid only for a uniquely identified
            # replay already in removal, whose own effects never inspect draws.
            if effect.card_selector is None:
                return False
            cards=tuple(c for c in state.deck.all_cards if self._selector_matches(c,effect.card_selector))
            if len(cards)!=1 or cards[0] not in state.deck.removed:
                return False
            definition=self.cards[cards[0].card_id]
            if definition.condition is not None or definition.required_stance is not None:
                return False
            def safe(child):
                if child.value_source is not None or any(c.kind not in scalar_conditions for c in conditions(child.condition)):
                    return False
                if child.kind is EffectKind.SCHEDULE:
                    return child.schedule_phase is SchedulePhase.END_OF_TURN and all(safe(c) for c in child.scheduled_effects)
                return child.kind in scalar_kinds
            if not all(safe(child) for child in definition.effects):
                return False
            replay_cards.add(cards[0].card_id)
            return True
        for ref in state.scheduled_effects:
            if ref.remaining_uses == 0 or ref.remaining_turns == 0:
                continue
            if ref.trigger_phase is SchedulePhase.AFTER_CARD_USED:
                continue
            if ref.trigger_phase not in {SchedulePhase.TURN,SchedulePhase.EVERY_TURN,
                    SchedulePhase.START_OF_TURN,SchedulePhase.AFTER_START_OF_TURN,SchedulePhase.END_OF_TURN}:
                return None
            effect=self._scheduled_effect_spec(ref)
            if any(c.kind not in scalar_conditions for c in conditions(effect.condition)):
                return None
            for child in effect.scheduled_effects:
                if inert(child,ref.trigger_phase,ref): continue
                if child.kind is EffectKind.FREE_PLAY_RANDOM:
                    allowed=scalar_conditions | {ConditionKind.REMOVED_BASE_CARD_COUNT_MIN}
                    if any(c.kind not in allowed for c in conditions(child.condition)) or not safe_replay(child):
                        return None
                    continue
                if (child.kind not in scalar_kinds or child.value_source is not None
                        or any(c.kind not in scalar_conditions for c in conditions(child.condition))):
                    return None
        if replay_cards:
            for ref in state.scheduled_effects:
                if ref.remaining_uses == 0 or ref.remaining_turns == 0:
                    continue
                if ref.trigger_phase is not SchedulePhase.AFTER_CARD_USED:
                    continue
                effect=self._scheduled_effect_spec(ref)
                # Unrelated named memory triggers cannot fire on this replay.
                if any(c.kind is ConditionKind.USED_CARD_BASE_ID
                       and all(self.cards[cid].base_card_id!=c.value for cid in replay_cards)
                       for c in conditions(effect.condition)):
                    continue
                allowed=scalar_conditions | {ConditionKind.USED_CARD_BASE_ID,ConditionKind.USED_CARD_ID,
                    ConditionKind.USED_CARD_RARITY_NOT,ConditionKind.USED_CARD_SOURCE_TYPE,
                    ConditionKind.USED_CARD_EFFECT_TAG}
                children=tuple(child for child in effect.scheduled_effects if not any(
                    c.kind is ConditionKind.USED_CARD_BASE_ID
                    and all(self.cards[cid].base_card_id!=c.value for cid in replay_cards)
                    for c in conditions(child.condition)))
                if (any(c.kind not in allowed for c in conditions(effect.condition))
                        or any(child.kind not in scalar_kinds or child.value_source is not None
                               or any(c.kind not in allowed for c in conditions(child.condition))
                               for child in children)):
                    return None
        projection=_HifScalarProjection(self.cards,self.drinks,self.items,
            memory_abilities=self.memory_abilities, support_definition=self.support_definition)
        return projection.transition(state,action,plan_type=plan_type)

    def terminal_score_resources(self, state, action, *, plan_type):
        """Exact terminal utility marginal; never a settlement state.

        Sense's final turn only clears cards and timers after its triggers.
        A focus-only trigger also leaves the terminal resources
        unchanged. Keep the ordinary kernel for every other active trigger.
        """
        if (state.hand_support or plan_type is not PlanType.SENSE or action.kind is not ActionKind.END_TURN
                or state.turns_remaining != 1 or not state.persistent_effects_initialized
                or state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                or any(ref.trigger_phase is SchedulePhase.END_OF_TURN
                       and ref.remaining_uses != 0 and ref.turns_until_fire == 0
                       and not self._terminal_focus_only_schedule(ref)
                       for ref in state.scheduled_effects)):
            return None
        state.validate()
        self._validate_registered_state(state, plan_type=plan_type)
        stamina = min(state.max_stamina, state.stamina + (2 if state.actions_remaining > 0 else 0))
        return float(state.score), stamina, state.genki

    def _terminal_focus_only_schedule(self, ref):
        effect = self._scheduled_effect_spec(ref)
        return effect.schedule_counter_condition is None and all(
            nested.kind is EffectKind.CONCENTRATION and (nested.condition is None
                or nested.condition.kind is ConditionKind.CONCENTRATION_MIN)
            and nested.value_source is None for nested in effect.scheduled_effects)

    def turn_boundary_transition(self, state, action, *, plan_type):
        """Exact current-turn settlement, before next-turn start or random draw.

        The returned continuation is deliberately unresolved. It belongs to a
        future-value leaf, never to an observed or executable complete state.
        """
        if state.hand_support or action.kind is not ActionKind.END_TURN or state.turns_remaining < 2:
            return None
        projection = _HifTurnBoundary(self, state.turn + 1)
        return projection.transition(state, action, plan_type=plan_type)

    def _turn_start_draw_count(self, state: StrategyState) -> int:
        # HIF draws three before afterStartOfTurn items. Existing cards moved
        # to hand at startOfTurn consume capacity, not the basic draw count.
        return min(3, max(0, state.hand_limit - len(state.deck.hand)))

    def _turn_end_deck(self, deck: DeckBelief) -> DeckBelief:
        return deck.replace_zones(hand=(), discard=deck.discard + deck.hand)

    def _draw_one(self, state: StrategyState, *, source_id: str):
        state = clean_hand_support(state)
        if len(state.deck.hand) >= state.hand_limit:
            return (TransitionBranch(1.0, state, (RuleEvent("DRAW_HAND_FULL", source_id),)),)
        return super()._draw_one(state, source_id=source_id)

    def _apply_effect(self, branches, effect, *, source_id, **kwargs):
        check_search_budget()
        branches = tuple(replace(branch, state=clean_hand_support(branch.state))
                         if branch.state.hand_support else branch for branch in branches)
        kwargs = self._support_schedule_kwargs(effect, kwargs)
        if effect.kind is not EffectKind.DRAW or effect.condition is not None:
            return super()._apply_effect(branches, effect, source_id=source_id, **kwargs)
        effect.validate()
        expanded = []
        for branch in branches:
            check_search_budget()
            drawn = self._independent_draw_batch(branch.state, int(effect.amount), source_id=source_id)
            if drawn is None:
                expanded.extend(super()._apply_effect((branch,), effect, source_id=source_id, **kwargs))
            else:
                expanded.extend(TransitionBranch(branch.probability * child.probability, child.state,
                                                  branch.events + child.events) for child in drawn)
        return self._merge_probability_branches(tuple(expanded))

    def _move_random_to_hand(self, state, selector, *, source_id):
        if len(state.deck.hand) >= state.hand_limit:
            return (TransitionBranch(1.0, state, (RuleEvent("MOVE_TO_HAND_FULL", source_id),)),)
        return super()._move_random_to_hand(state, selector, source_id=source_id)

    def _apply_non_draw_effect(self, state, effect, *, source_id, **kwargs):
        # Fixed RIS CardManager.moveSelectedToHandByTarget returns before
        # opening a selection when full. Continue the drink/card's remaining
        # effects; do not invent a sixth card or consume an extra selection.
        if effect.kind is EffectKind.MOVE_SELECTED_TO_HAND and len(state.deck.hand) >= state.hand_limit:
            return state, RuleEvent("MOVE_TO_HAND_FULL", source_id)
        return self._apply_non_draw_effect_with_freshness(state, effect, source_id=source_id, **kwargs)

    def _independent_draw_batch(self, state: StrategyState, count: int, *, source_id: str):
        """Enumerate exact subsets when intermediate draw order has no effect.

        Prefixes and move-to-hand triggers retain the sequential kernel path.
        A depleted draw pile is certain before sampling the reshuffled discard.
        No order is invented for the game;
        each unordered subset has its exact without-replacement probability.
        """
        state = clean_hand_support(state)
        deck = state.deck
        # Match the sequential draw's discard rebuild before enumerating the
        # same unordered subsets. No move trigger, prefix or draw is skipped.
        if not deck.draw_pile and 1 < count <= min(state.hand_limit - len(deck.hand), len(deck.discard)):
            rebuilt = deck.replace_zones(draw_pile=deck.discard, discard=(),
                                         knowledge=DeckKnowledge.KNOWN_COMPOSITION)
            return self._independent_draw_batch(replace(state, deck=rebuilt), count, source_id=source_id)
        if (
            not 1 < count <= state.hand_limit - len(deck.hand)
            or deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
            or deck.known_top_count
            or any(effect.trigger_phase is SchedulePhase.CARD_MOVED_TO_HAND
                   for effect in state.scheduled_effects)
        ):
            return None
        if 0 < len(deck.draw_pile) < count <= len(deck.draw_pile) + len(deck.discard):
            # Every current draw-pile card arrives before the reshuffle, so
            # their permutations all reach the same state with probability 1.
            # Sequential draws canonicalize the discard before rebuilding it;
            # preserve that physical-copy selection for the remaining draws.
            check_search_budget()
            exhausted = self._canonical_probability_state(replace(state, deck=deck.replace_zones(
                hand=deck.hand + deck.draw_pile, draw_pile=(),
            )))
            remaining = count - len(deck.draw_pile)
            tail = (self._draw_one(exhausted, source_id=source_id) if remaining == 1 else
                    self._independent_draw_batch(exhausted, remaining, source_id=source_id))
            if tail is None:
                return None
            events = tuple(RuleEvent("DRAW", source_id, detail=card.instance_id) for card in deck.draw_pile)
            return tuple(TransitionBranch(child.probability, self._canonical_probability_state(child.state),
                                          events + child.events) for child in tail)
        if count > len(deck.draw_pile):
            return None
        groups = {}
        for card in deck.draw_pile:
            groups.setdefault(card.semantic_key, []).append(card)
        # The sequential kernel always draws a group's first physical copy,
        # then canonicalizes. Keep the fallback for noncanonical copy order.
        if any(cards != sorted(cards, key=lambda card:card.instance_id) for cards in groups.values()):
            return None
        grouped = tuple(tuple(cards) for cards in groups.values())
        denominator = comb(len(deck.draw_pile), count)

        def subsets(index, remaining, selected=(), ways=1):
            if remaining == 0:
                yield selected, ways / denominator
                return
            if index == len(grouped) or sum(map(len,grouped[index:])) < remaining:
                return
            cards = grouped[index]
            for taken in range(min(len(cards),remaining)+1):
                yield from subsets(index+1,remaining-taken,selected+cards[:taken],
                                   ways*comb(len(cards),taken))

        branches = []
        for drawn, probability in subsets(0, count):
            check_search_budget()
            ids = {card.instance_id for card in drawn}
            next_deck = deck.replace_zones(
                hand=deck.hand + drawn,
                draw_pile=tuple(card for card in deck.draw_pile if card.instance_id not in ids),
            )
            branches.append(TransitionBranch(
                probability, self._canonical_probability_state(replace(state, deck=next_deck)),
                tuple(RuleEvent("DRAW", source_id, detail=card.instance_id) for card in drawn),
            ))
        return tuple(branches)

    def _exchange_hand(self, state: StrategyState, *, source_id: str):
        if state.hand_support:
            state = replace(state, hand_support=())
        count = len(state.deck.hand)
        prepared = replace(state, deck=state.deck.replace_zones(
            hand=(), discard=state.deck.discard + state.deck.hand,
        ))
        drawn = self._independent_draw_batch(prepared, count, source_id=source_id)
        if drawn is None:
            return super()._exchange_hand(state, source_id=source_id)
        return tuple(replace(branch, events=(RuleEvent("EXCHANGE_HAND", source_id, count),) + branch.events)
                     for branch in drawn)

    def _advance_turn_resolution(self, branch, frame):
        if frame.stage is TurnResolutionStage.REFILL:
            next_frame = replace(frame, stage=TurnResolutionStage.HELD_RETURN, draws_remaining=0)
            state = replace(branch.state, resolution_stack=branch.state.resolution_stack + (next_frame,))
            drawn = self._independent_draw_batch(state, frame.draws_remaining, source_id="system")
            if drawn is not None:
                return self._continue_resolution_after_operation(tuple(
                    TransitionBranch(branch.probability * child.probability, child.state,
                                     branch.events + child.events)
                    for child in drawn
                ))
        return super()._advance_turn_resolution(branch, frame)

    def _end_turn(self, state: StrategyState, *, plan_type: PlanType):
        recovered = min(2, state.max_stamina - state.stamina) if state.actions_remaining > 0 else 0
        prepared = replace(state, stamina=state.stamina + recovered)
        return tuple(
            replace(branch, events=(RuleEvent("SKIP_RECOVERY", "system", recovered),) + branch.events)
            for branch in super()._end_turn(prepared, plan_type=plan_type)
        )

    def __init__(
        self,
        cards: Mapping[str, CardDefinition],
        drinks: Mapping[str, DrinkDefinition] | None = None,
        items: Mapping[str, ItemDefinition] | None = None,
        *,
        memory_abilities: Mapping[str, MemoryAbilityDefinition] | None = None,
        support_definition: Callable[[CardDefinition, int], CardDefinition] | None = None,
    ):
        super().__init__(cards, drinks, items, ruleset_id=HIF_SENSE_RULESET_ID, memory_abilities=memory_abilities)
        self.support_definition = support_definition

    def _validate_registered_state(self, state: StrategyState, *, plan_type: PlanType) -> None:
        if plan_type is not PlanType.SENSE:
            raise KernelError("hif_first_slice_requires_sense")
        if state.max_stamina <= 0:
            raise KernelError("hif_maximum_stamina_required")
        if len(state.future_score_multipliers) != max(0, state.turns_remaining - 1):
            raise KernelError("hif_turn_multipliers_incomplete")
        if state.turn > 1 and not state.persistent_effects_initialized:
            raise KernelError("hif_mid_stage_schedule_history_required")
        super()._validate_registered_state(state, plan_type=plan_type)


class _HifScalarProjection(HifSenseKernel):
    """Private heuristic projection; deliberately omits all draw identities."""

    def _draw_one(self, state, *, source_id):
        return (TransitionBranch(1.,state,()),)

    def _independent_draw_batch(self, state, count, *, source_id):
        return (TransitionBranch(1.,state,()),)

    def _turn_start_draw_count(self, state):
        return 0


def is_hif_turn_boundary(state):
    return (state.pending_choice is None and len(state.resolution_stack) == 1
            and isinstance(state.resolution_stack[0], TurnResolutionContinuation)
            and state.resolution_stack[0].stage is TurnResolutionStage.BEFORE_START)


class _HifTurnBoundary(HifSenseKernel):
    """Read-only facade over a validated kernel, stopping at a named phase."""

    def __init__(self, kernel, before_turn):
        self.cards, self.drinks, self.items = kernel.cards, kernel.drinks, kernel.items
        self.memory_abilities, self.ruleset_id = kernel.memory_abilities, kernel.ruleset_id
        self.support_definition = kernel.support_definition
        self._before_turn = before_turn

    def _resume_resolution_stack(self, branches):
        result = []
        for branch in branches:
            if branch.state.turn == self._before_turn and is_hif_turn_boundary(branch.state):
                result.append(branch)
            else:
                result.extend(super()._resume_resolution_stack((branch,)))
        return tuple(result)
