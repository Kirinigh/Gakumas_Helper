"""Static admission for bounded one/two-turn suffixes.

This is an offline work-allocation domain, NOT a runtime or optimality proof.
No transition, legal-action enumeration, future deck order or score is used.
Extra actions are harmless to the bound only because no card can reenter hand.
"""
import math
from collections import Counter
from dataclasses import dataclass

from .hif import HifSenseKernel
from .kernel import FULL_POWER_ACTIVATION_THRESHOLD, EffectKind, KernelError, ConditionKind
from .contracts import Stance, CardRef, CardZone, PlanType, DeckKnowledge, SchedulePhase, HighScoreContractError
from .hif_terminal_cache import _SCALAR_EFFECTS

VERSION = 'task275-fixed-next-boundary-draw-domain-v3'
FINAL_DOUBLE_VERSION = 'task275-final-direct-double-tail-domain-v1'
FINAL_OFFHAND_REMOVE_VERSION = 'task275-final-offhand-remove-tail-domain-v1'
MAX_SEQUENCE_PATHS = 4096
_CARD_PHASES = frozenset({SchedulePhase.BEFORE_CARD_USED, SchedulePhase.CARD_USED,
                         SchedulePhase.AFTER_CARD_USED})
_BOUNDARY_PHASES = frozenset({SchedulePhase.BEFORE_START_OF_TURN, SchedulePhase.START_OF_TURN,
    SchedulePhase.AFTER_START_OF_TURN, SchedulePhase.TURN, SchedulePhase.EVERY_TURN})
_SCALARS = _SCALAR_EFFECTS | {EffectKind.EXTRA_ACTION}


@dataclass(frozen=True)
class SimpleTailDomain:
    eligible: bool
    reason: str
    hand_count: int
    draw_count: int
    generated_cards_upper: int = 0
    boundary_hand_combinations_upper: int = 0
    sequence_paths_upper: int = 0
    boundary_draws_upper: int = 0
    boundary_draw_allocations_upper: int = 0


class _Unsupported(Exception):
    pass


def _possible(condition, definitions):
    """Only immutable used-card identity may prove a condition unreachable."""
    if condition is None:
        return True
    condition.validate()
    if not isinstance(condition.kind, ConditionKind):
        raise _Unsupported('unknown_condition')
    if definitions is None:
        return True
    if condition.kind is ConditionKind.ALL:
        return all(_possible(child, definitions) for child in condition.all_of)
    if condition.kind is ConditionKind.USED_CARD_ID:
        return any((card.canonical_card_id or card.card_id) == condition.value for card in definitions)
    attr = {ConditionKind.USED_CARD_BASE_ID: 'base_card_id',
            ConditionKind.USED_CARD_SOURCE_TYPE: 'source_type'}.get(condition.kind)
    if attr:
        return any((getattr(card, attr) or card.card_id) == condition.value for card in definitions)
    if condition.kind is ConditionKind.USED_CARD_RARITY_NOT:
        return any(card.rarity != condition.value for card in definitions)
    # Resource thresholds can become true after an earlier card or trigger.
    return True


def _prefixes(size):
    return sum(math.perm(size, used) for used in range(size + 1))


def _hands(counts, size):
    """Number of unordered semantic hands; only size+1 coefficients are needed."""
    coefficients = [1] + [0] * size
    for count in counts:
        coefficients = [sum(coefficients[n - k] for k in range(min(count, n) + 1))
            for n in range(size + 1)]
    return coefficients[size]


def _action_prefixes(extras, actions):
    """Count all physical-card prefixes with optimistic, fixed action gains.

    Costs and resource conditions are ignored. Each card can leave hand only
    once in this domain. Clipping above the admission limit preserves rejection.
    """
    counts = {0: 1}  # END is available at every reached prefix.
    for mask in range(1 << len(extras)):
        if mask not in counts:
            continue
        available = actions - mask.bit_count() + sum(
            extra for index, extra in enumerate(extras) if mask & (1 << index))
        if available <= 0:
            continue
        for index in range(len(extras)):
            bit = 1 << index
            if not mask & bit:
                counts[mask | bit] = min(MAX_SEQUENCE_PATHS + 1,
                    counts.get(mask | bit, 0) + counts[mask])
    return counts


class _Closure:
    def __init__(self, kernel, state, *, final=False, allow_final_double=False,
                 allow_final_offhand_remove=False):
        self.kernel, self.state = kernel, state
        self.final = final
        self.allow_final_double = allow_final_double
        self.allow_final_offhand_remove = allow_final_offhand_remove
        self.generated = Counter()
        self.final_generated = Counter()
        self.offhand_removal = None
        self.boundary_draws = []
        self.current_draws = []
        self.existing_draws = ()
        self.scheduled_actions = False
        self.current = tuple(kernel.cards[ref.card_id] for ref in state.deck.hand)
        self.future = () if final else tuple(kernel.cards[ref.card_id] for ref in state.deck.draw_pile)
        # The kernel runs at most two effect passes for a non-L card, and one
        # for L, irrespective of how many DOUBLE charges remain. No hand
        # reentry is admitted, so each physical current card is used at most once.
        self.final_double_possible = state.double_card_effects_remaining > 0 or (
            allow_final_double and any(effect.kind is EffectKind.DOUBLE_CARD_EFFECT
                for definition in self.current for effect in definition.effects))

    def effect(self, effect, definitions, *, final, direct=False, passes=1):
        effect.validate()
        if not isinstance(effect.kind, EffectKind):
            raise _Unsupported('unknown_effect')
        if not _possible(effect.condition, definitions):
            return
        if effect.kind in _SCALARS:
            return
        if (self.allow_final_double and self.final and final and direct
                and effect.kind is EffectKind.DOUBLE_CARD_EFFECT):
            # The final root counts all physical-card prefixes, irrespective
            # of action grants or doubled scalar effects. Other effects still
            # pass the unchanged closure; no hand replenishment is admitted.
            return
        if effect.kind is EffectKind.SCHEDULE:
            # A newly registered trigger is not restricted to its establishing
            # card's identity. Conservatively retain unknown future matches.
            self.schedule(effect, None, final=final, copies=effect.repeats * passes)
            return
        if effect.kind is EffectKind.ADD_CARD_TO_DECK and direct:
            target = self.kernel.cards.get(effect.card_id)
            if target is None or target.effects or target.persistent_effects:
                raise _Unsupported('generated_card_not_empty')
            target.validate()
            # The kernel adds one fixed card per effect repeat, independent
            # of amount. Conditional additions are conservatively all present.
            if not final:
                self.generated[CardRef('domain-generated', target.card_id).semantic_key] += effect.repeats * passes
            elif self.final and self.allow_final_offhand_remove:
                if type(effect.repeats) is not int or effect.repeats < 1:
                    raise _Unsupported('unbounded_final_generated_cards')
                # Direct effects retain their owning definition here. A later
                # DOUBLE can increase passes beyond today's _card_effect_passes.
                if definitions is None or len(definitions) != 1:
                    raise _Unsupported('unknown_final_generated_card_owner')
                upper_passes = (2 if self.final_double_possible
                                and definitions[0].rarity != 'L' else 1)
                self.final_generated[target.card_id] += effect.repeats * upper_passes
            return
        raise _Unsupported('unsupported_reachable_effect:' + str(effect.kind.value))

    def schedule(self, effect, definitions, *, final, delay=None, copies=1, existing=False,
                 reference=None):
        phase = effect.schedule_phase
        delay = effect.delay_turns if delay is None else delay
        if phase in _BOUNDARY_PHASES:
            if final or delay > 1:
                return
            targets = self.future
        elif phase in _CARD_PHASES:
            if delay > (0 if final else 1):
                return
            targets = definitions
            if effect.trigger_card_type and targets is not None:
                targets = tuple(card for card in targets if card.card_type == effect.trigger_card_type)
            if targets == ():
                return
        elif phase is SchedulePhase.END_OF_TURN:
            targets = definitions
        else:
            # In-hand movement triggers would invalidate unordered draw sets;
            # other phases are outside this intentionally narrow closure.
            raise _Unsupported('unsupported_schedule_phase:' + phase.value)
        if not _possible(effect.schedule_counter_condition, targets):
            return
        for nested in effect.scheduled_effects:
            if nested.kind is EffectKind.SCHEDULE:
                raise _Unsupported('nested_schedule')
            if (nested.kind is EffectKind.EXTRA_ACTION and _possible(nested.condition, targets)
                    and (not existing or phase not in _CARD_PHASES)):
                # New triggers or boundary action grants can depend on earlier
                # cards. Every permutation remains a safe fallback bound.
                self.scheduled_actions = True
            if (nested.kind is EffectKind.DRAW and phase is SchedulePhase.TURN
                    and delay == 1 and not final and _possible(nested.condition, targets)):
                nested.validate()
                if (type(nested.amount) not in (int, float) or not math.isfinite(nested.amount)
                        or nested.amount < 0 or int(nested.amount) != nested.amount
                        or nested.value_source is not None
                        or type(nested.repeats) is not int or type(copies) is not int):
                    raise _Unsupported('nonfixed_boundary_draw')
                if nested.repeats * copies > MAX_SEQUENCE_PATHS:
                    raise _Unsupported('sequence_paths_exceed_domain')
                self.boundary_draws.extend([int(nested.amount)] * (nested.repeats * copies))
                continue
            if (self.final and final and self.allow_final_offhand_remove
                    and nested.kind is EffectKind.REMOVE_RANDOM
                    and _possible(nested.condition, targets)):
                self.final_removal(effect, nested, reference if existing else None)
                continue
            self.effect(nested, targets, final=final)

    @staticmethod
    def _identity_condition(condition):
        if condition is None:
            return True
        condition.validate()
        if condition.kind is ConditionKind.ALL:
            return all(_Closure._identity_condition(child) for child in condition.all_of)
        return condition.kind in {ConditionKind.USED_CARD_ID, ConditionKind.USED_CARD_BASE_ID,
                                  ConditionKind.USED_CARD_SOURCE_TYPE, ConditionKind.USED_CARD_RARITY_NOT}

    def final_removal(self, schedule, effect, reference):
        effect.validate()
        selector = effect.card_selector
        if (reference is None or reference.trigger_phase is not SchedulePhase.AFTER_CARD_USED
                or reference.turns_until_fire != 0 or type(reference.remaining_uses) is not int
                or reference.remaining_uses != 1 or reference.remaining_turns == 0
                or type(schedule.repeats) is not int or schedule.repeats != 1
                or effect.amount != 1 or type(effect.repeats) is not int or effect.repeats != 1
                or effect.value_source is not None or selector is None or not selector.zones
                or not set(selector.zones) <= {CardZone.DRAW_PILE, CardZone.DISCARD}
                or not all(self._identity_condition(condition) for condition in
                           (schedule.condition, schedule.schedule_counter_condition, effect.condition))):
            raise _Unsupported('unsupported_final_offhand_removal')
        if self.offhand_removal is not None:
            raise _Unsupported('multiple_final_offhand_removals')
        self.offhand_removal = selector

    def removal_bound(self):
        if self.offhand_removal is None:
            return 0, 1
        selector = self.offhand_removal
        # Include all current hand matches, even limited cards: an unplayed
        # hand can be discarded at turn end. This intentionally overcounts.
        candidates = tuple(card for zone in selector.zones
                           for card in self.kernel._zone_cards(self.state, zone)) + self.state.deck.hand
        targets = sum(self.kernel._selector_matches(card, selector) for card in candidates)
        targets += sum(count for card_id, count in self.final_generated.items()
                       if self.kernel._selector_matches(CardRef('domain-generated', card_id), selector))
        return targets, max(1, targets)

    def check(self):
        for definition in self.current:
            start = len(self.boundary_draws)
            definition.validate()
            if definition.plan not in {None, PlanType.SENSE}:
                raise _Unsupported('unsupported_card_plan')
            # Definition conditions are not evaluated against today's resources.
            _possible(definition.condition, (definition,))
            passes = self.kernel._card_effect_passes(self.state, definition)
            for effect in definition.effects:
                self.effect(effect, (definition,), final=self.final, direct=True, passes=passes)
            self.current_draws.append(tuple(self.boundary_draws[start:]))
        # Fixed empty generated cards can be drawn and can match an existing
        # used-card trigger, despite having no direct effects of their own.
        generated = tuple(self.kernel.cards[key[0]] for key in self.generated)
        self.future += generated
        all_reachable = self.current + self.future
        for definition in self.future:
            definition.validate()
            if definition.plan not in {None, PlanType.SENSE}:
                raise _Unsupported('unsupported_card_plan')
            _possible(definition.condition, (definition,))
            for effect in definition.effects:
                self.effect(effect, (definition,), final=True, direct=True)
        start = len(self.boundary_draws)
        for ref in self.state.scheduled_effects:
            if ref.remaining_uses == 0:
                continue
            effect = self.kernel._scheduled_effect_spec(ref)
            # Current and final identities are a safe union. No filtering by
            # score, costs, resources, prior rank or hidden draw order occurs.
            self.schedule(effect, all_reachable, final=self.final, delay=ref.turns_until_fire,
                          existing=True, reference=ref)
        self.existing_draws = tuple(self.boundary_draws[start:])

    def action_gains(self, definitions, *, final):
        if self.scheduled_actions:
            return [1] * len(definitions)
        extras = []
        for definition in definitions:
            extra = 0
            effects = list(definition.effects)
            for ref in self.state.scheduled_effects:
                if ref.remaining_uses == 0 or ref.turns_until_fire > (1 if final else 0):
                    continue
                spec = self.kernel._scheduled_effect_spec(ref)
                if (spec.schedule_phase not in _CARD_PHASES
                        or spec.trigger_card_type and definition.card_type != spec.trigger_card_type
                        or not _possible(spec.schedule_counter_condition, (definition,))):
                    continue
                effects.extend(spec.scheduled_effects)
            for effect in effects:
                if effect.kind is not EffectKind.EXTRA_ACTION or not _possible(effect.condition, (definition,)):
                    continue
                if (type(effect.amount) not in (int, float) or not math.isfinite(effect.amount)
                        or type(effect.repeats) is not int or effect.value_source is not None):
                    raise _Unsupported('nonfixed_extra_action')
                # Two-turn DOUBLE effects remain rejected. Treat every
                # card as consuming an existing charge, even if charges run out.
                extra += (max(0, int(effect.amount)) * effect.repeats
                          * self.kernel._card_effect_passes(self.state, definition))
            extras.append(extra)
        return extras

    def boundary_paths(self, natural):
        """Count each possible draw stage allocation, not just its final union."""
        state = self.state
        next_stance_lock = (state.stance_lock_turns if state.stance_lock_fresh
                            else max(0, state.stance_lock_turns - 1))
        if next_stance_lock == 0 and state.full_power >= FULL_POWER_ACTIVATION_THRESHOLD:
            # The shared turn boundary grants another action on FULL_POWER
            # entry even with no held cards. The new action-limited bound does
            # not cover that transition; legacy all-prefix admission is intact.
            raise _Unsupported('boundary_full_power_extra_action')
        drawn = min(state.hand_limit, natural + sum(self.boundary_draws))
        if len(state.deck.draw_pile) < drawn:
            raise _Unsupported('boundary_draw_can_reshuffle')
        if drawn > 6 or state.hand_support:
            raise _Unsupported('unsupported_boundary_hand_shape')
        pool = len(state.deck.draw_pile) + sum(self.generated.values())
        # Physical subsets conservatively include duplicate semantic cards.
        # No in-hand trigger is admitted, so order within a single draw batch
        # cannot change business state. Separate batches remain distinct here.
        current = _action_prefixes(self.action_gains(self.current, final=False),
                                  max(1, state.actions_remaining) if self.scheduled_actions
                                  else state.actions_remaining)
        # The semantic closure visits each generated definition once; action
        # capacity must still include every copy that could reach the hand.
        extra_generated = tuple(self.kernel.cards[key[0]] for key, count in self.generated.items()
                                for _ in range(min(count - 1, drawn - 1)))
        gains = sorted(self.action_gains(self.future + extra_generated, final=True)
                       + [0] * drawn, reverse=True)
        future = {size: (_prefixes(size) if self.scheduled_actions else
                        sum(_action_prefixes(gains[:size], 1).values()))
                  for size in range(natural, drawn + 1)}
        paths, allocation_count = 0, 0
        for mask, prefixes in current.items():
            stages = self.existing_draws + tuple(amount
                for index, amounts in enumerate(self.current_draws) if mask & (1 << index)
                for amount in amounts)
            allocations = {natural: math.comb(pool, natural)}
            for amount in stages:
                following = Counter()
                for held, ways in allocations.items():
                    following[held] += ways  # condition may prevent the draw
                    take = min(amount, state.hand_limit - held)
                    # Ordered identities upper-bound every intermediate batch
                    # and every registration order of this prefix's cards.
                    # Products of falling factorials depend only on total
                    # draws, including capacity truncation; using combinations
                    # here could undercount another ordering of unequal batches.
                    following[held + take] += ways * math.perm(pool - held, take)
                allocations = {size: min(ways, MAX_SEQUENCE_PATHS + 1)
                               for size, ways in following.items()}
                if sum(allocations.values()) > MAX_SEQUENCE_PATHS:
                    break
            allocation_count += prefixes * sum(allocations.values())
            paths += prefixes * sum(ways * future[size] for size, ways in allocations.items())
            if paths > MAX_SEQUENCE_PATHS:
                break
        return drawn, allocation_count, paths


def assess_simple_tail_domain(kernel, state, *, allow_final_double=False, allow_final_offhand_remove=False):
    """Return a static domain decision; does not run a solver or change state.

    For two turns, the 4096 bound counts current card-order prefixes times
    possible boundary semantic hands times final card-order prefixes. For one
    turn it counts only current prefixes: the deck is never drawn again.
    Generated-card counts concern only the future natural-draw pool.
    Legacy natural-only admission ignores action limits. Fixed next-TURN draws
    use optimistic action gains and count all intermediate batch allocations.
    Scalar rule/trigger cost is not bounded by this number or by 500 ms.
    The opt-in admits direct DOUBLE only for a final-turn root; the two-turn
    closure and its action-limited bound keep their existing semantics.
    A separate final-only opt-in admits one existing single-use offhand
    removal and multiplies prefixes by its conservative physical target bound.
    """
    if type(allow_final_double) is not bool:
        raise ValueError('allow_final_double_requires_bool')
    if type(allow_final_offhand_remove) is not bool:
        raise ValueError('allow_final_offhand_remove_requires_bool')
    hand, draw = len(state.deck.hand), len(state.deck.draw_pile)
    counts = dict(hand_count=hand, draw_count=draw)
    try:
        state.validate()
        if type(kernel) is not HifSenseKernel or state.stance is not Stance.NEUTRAL:
            raise _Unsupported('unsupported_kernel_or_stance')
        if state.turns_remaining not in (1, 2) or not state.persistent_effects_initialized:
            raise _Unsupported('not_initialized_one_or_two_turn_state')
        if (state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                or state.drinks or state.field_effects or state.deck.held):
            raise _Unsupported('nonordinary_state')
        if state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION or state.deck.known_top_count:
            raise _Unsupported('requires_unknown_order_composition')
        final = state.turns_remaining == 1
        natural = 0 if final else min(3, state.hand_limit)
        if not final and draw < natural:
            raise _Unsupported('natural_draw_can_reshuffle')
        if hand > 6:
            raise _Unsupported('sequence_paths_exceed_domain')
        closure = _Closure(kernel, state, final=final, allow_final_double=allow_final_double,
                           allow_final_offhand_remove=allow_final_offhand_remove)
        closure.check()
        if final:
            paths = _prefixes(hand)
            if allow_final_offhand_remove:
                _, branches = closure.removal_bound()
                paths *= branches
            counts.update(sequence_paths_upper=paths)
            if paths > MAX_SEQUENCE_PATHS:
                raise _Unsupported('sequence_paths_exceed_domain')
            return SimpleTailDomain(True, 'zero_natural_boundary', **counts)
        if closure.boundary_draws:
            drawn, allocations, paths = closure.boundary_paths(natural)
            counts.update(generated_cards_upper=sum(closure.generated.values()),
                          boundary_draws_upper=drawn,
                          boundary_draw_allocations_upper=allocations,
                          sequence_paths_upper=paths)
            if paths > MAX_SEQUENCE_PATHS:
                raise _Unsupported('sequence_paths_exceed_domain')
            return SimpleTailDomain(True, 'fixed_next_boundary_draws', **counts)
        pool = Counter(ref.semantic_key for ref in state.deck.draw_pile) + closure.generated
        combinations = _hands(pool.values(), natural)
        paths = _prefixes(hand) * combinations * _prefixes(natural)
        counts.update(generated_cards_upper=sum(closure.generated.values()),
                      boundary_hand_combinations_upper=combinations, sequence_paths_upper=paths)
        if paths > MAX_SEQUENCE_PATHS:
            raise _Unsupported('sequence_paths_exceed_domain')
        return SimpleTailDomain(True, 'single_natural_boundary', **counts)
    except _Unsupported as error:
        return SimpleTailDomain(False, str(error), **counts)
    except (HighScoreContractError, KernelError, KeyError, IndexError, TypeError, ValueError):
        return SimpleTailDomain(False, 'invalid_or_unknown_public_definition', **counts)
