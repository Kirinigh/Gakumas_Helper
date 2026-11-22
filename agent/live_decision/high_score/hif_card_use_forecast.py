"""One public draw opportunity for already installed pure card-use scores.

This is an optional leaf feature, not a transition or a terminal return. It
retains the existing forecast's scalar resource assumptions, but does not
invent future installations, draw identities, repeated plays or extra turns.
Other CARD_USED effects are not propagated: both compared scores use the
same paid state, so interacting resource triggers remain a heuristic limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .hif import is_hif_turn_boundary
from .kernel import EffectKind, EffectSpec, KernelError, ConditionKind, TransitionBranch, _condition_matches
from .contracts import CardZone, PlanType, DeckKnowledge, SchedulePhase
from .state_copy import replace
from .hif_forecast import expected_best
from .validation_cache import check_search_budget


@dataclass(frozen=True)
class InstalledCardUseBonus:
    value: float
    reason: str
    pool_size: int = 0
    draw_count: int = 0
    schedule_count: int = 0
    ignored_start_effects: int = 0


_POOL_EFFECTS = frozenset({
    EffectKind.MOVE_RANDOM_TO_HAND, EffectKind.MOVE_RANDOM_TO_TOP,
    EffectKind.REMOVE_RANDOM, EffectKind.EXCHANGE_HAND, EffectKind.ADD_CARD_TO_DECK,
    EffectKind.HOLD_SELF, EffectKind.HOLD_SELECTED, EffectKind.CARD_MODIFICATION,
    EffectKind.UPGRADE_HAND, EffectKind.FREE_PLAY_SELECTED, EffectKind.FREE_PLAY_ALL,
    EffectKind.FREE_PLAY_RANDOM, EffectKind.PAID_PLAY_SELECTED,
    EffectKind.MOVE_SELECTED_TO_HAND, EffectKind.TURNS_REMAINING,
})


def _pure_card_score(effect):
    return (effect.kind is EffectKind.SCHEDULE
        and effect.schedule_phase is SchedulePhase.CARD_USED
        and effect.delay_turns == 0 and effect.condition is None
        and effect.schedule_counter_condition is None
        and effect.trigger_card_type in {None, 'active', 'mental'}
        and bool(effect.scheduled_effects)
        and all(child.kind is EffectKind.SCORE and child.condition is None
                and child.value_source is None and not child.full_power_scale
                for child in effect.scheduled_effects))


def _next_draw_count(kernel, state):
    """Count known draws only; never run or sample next-turn resolution."""
    count = kernel._turn_start_draw_count(state)
    ignored = 0
    frame = state.resolution_stack[0]
    for group in (frame.due_before_start, frame.due_start, frame.due_after_start,
                  frame.due_turn, frame.due_every_turn):
        for reference in group:
            check_search_budget()
            # Due references may have just reached TTL zero and still fire.
            if reference.remaining_uses == 0:
                continue
            effect = kernel._scheduled_effect_spec(reference)
            # Once a scalar start effect was left unprojected, a later
            # conditional draw/zone change no longer has a proved count.
            if ignored and any(child.condition is not None
                and child.kind in _POOL_EFFECTS | {EffectKind.DRAW}
                for child in effect.scheduled_effects):
                return None, ignored
            matched = kernel._scheduled_matched_nested_indices(reference, state, None)
            for index in matched:
                child = effect.scheduled_effects[index]
                if child.kind in _POOL_EFFECTS:
                    return None, ignored
                if child.kind is EffectKind.DRAW:
                    if (child.value_source is not None or child.repeats != 1
                            or type(child.amount) not in (int, float)
                            or not math.isfinite(child.amount) or child.amount < 0
                            or not float(child.amount).is_integer()):
                        return None, ignored
                    count += int(child.amount)
                else:
                    # As in the base leaf estimate, unexecuted scalar start
                    # gains are not future facts. Report this approximation.
                    ignored += 1
    return min(state.hand_limit, count), ignored


_START_SCALARS = frozenset({EffectKind.GENKI, EffectKind.GOOD_CONDITION,
    EffectKind.EXCELLENT_CONDITION, EffectKind.CONCENTRATION, EffectKind.EXTRA_ACTION})
_KNOWN_DRAW_CONDITIONS = frozenset({ConditionKind.TURN_ATTRIBUTE,
    ConditionKind.GOOD_CONDITION_MIN, ConditionKind.GOOD_CONDITION_MAX,
    ConditionKind.EXCELLENT_CONDITION_MIN, ConditionKind.CONCENTRATION_MIN,
    ConditionKind.GENKI_MIN, ConditionKind.STAMINA_RATIO_MIN,
    ConditionKind.EFFECT_COUNTER_MIN, ConditionKind.EFFECT_COUNTER_MOD_EQ,
    ConditionKind.REMOVED_BASE_CARD_COUNT_MIN})


def _known_draw_condition(condition):
    return (condition is None or condition.kind in _KNOWN_DRAW_CONDITIONS
        or condition.kind is ConditionKind.ALL
        and all(_known_draw_condition(child) for child in condition.all_of))


def _fixed_discard_draw(kernel, state):
    """Project scalars and one forced discard move; keep all draws symbolic.

    No sampled identity is installed in hand. A selector with no matches in
    the entire original draw pool and one discard match has the same unique
    result before or after any uniform draw. Capacity is counted in actual
    resolution order, so the fixed card can displace a later random draw.
    """
    pool, fixed, drawn = state.deck.draw_pile, (), 0
    frame = state.resolution_stack[0]
    if any(ref.trigger_phase is SchedulePhase.CARD_USED and ref.turns_until_fire == 0
           and ref.remaining_uses != 0 and ref.remaining_turns != 0
           and (ref.direct_on_trigger or not _pure_card_score(kernel._scheduled_effect_spec(ref)))
           for ref in state.scheduled_effects):
        return None
    for group in (frame.due_before_start, frame.due_start, None,
                  frame.due_after_start, frame.due_turn, frame.due_every_turn):
        check_search_budget()
        if group is None:  # Natural refill; there are no held cards in this domain.
            drawn += min(kernel._turn_start_draw_count(state), state.hand_limit - drawn - len(fixed))
            if drawn > len(pool):
                return None
            continue
        matches = []
        for ref in group:
            effect = kernel._scheduled_effect_spec(ref)
            if any(not _known_draw_condition(child.condition) for child in effect.scheduled_effects):
                return None
            # Match the whole phase against its entry snapshot, as the kernel
            # does; an earlier sibling gain cannot enable a later condition.
            matches.append(kernel._scheduled_matched_nested_indices(ref, state, None))
        for ref, matched in zip(group, matches):
            if ref.remaining_uses == 0:
                continue
            effect = kernel._scheduled_effect_spec(ref)
            for index in matched:
                child = effect.scheduled_effects[index]
                if child.kind is EffectKind.DRAW:
                    if (child != EffectSpec(EffectKind.DRAW, child.amount, condition=child.condition)
                            or not math.isfinite(child.amount) or child.amount < 0
                            or not float(child.amount).is_integer()):
                        return None
                    drawn += min(int(child.amount), state.hand_limit - drawn - len(fixed))
                    if drawn > len(pool):
                        return None
                elif child.kind is EffectKind.MOVE_RANDOM_TO_HAND:
                    selector = child.card_selector
                    if (fixed or selector is None
                            or set(selector.zones) != {CardZone.DRAW_PILE, CardZone.DISCARD}
                            or child != EffectSpec(EffectKind.MOVE_RANDOM_TO_HAND,
                                card_selector=selector, condition=child.condition)
                            or any(kernel._selector_matches(card, selector) for card in pool)):
                        return None
                    targets = tuple(card for card in state.deck.discard
                                    if kernel._selector_matches(card, selector))
                    if len(targets) != 1 or drawn + len(fixed) >= state.hand_limit:
                        return None
                    fixed = targets
                    state = replace(state, deck=state.deck.replace_zones(
                        hand=state.deck.hand + targets,
                        discard=tuple(card for card in state.deck.discard if card != targets[0])))
                elif child.kind in _START_SCALARS:
                    if child != EffectSpec(child.kind, child.amount, condition=child.condition):
                        return None
                    branches = kernel._apply_effect((TransitionBranch(1., state, ()),),
                        replace(child, condition=None), source_id=ref.source_id)
                    if len(branches) != 1 or branches[0].probability != 1.:
                        return None
                    state = branches[0].state
                else:
                    return None
    if not fixed:
        return None
    return state, pool, fixed, drawn


def _expected_best_with_fixed(values, draws, fixed_values):
    floor = max((0., *fixed_values))
    return expected_best([max(floor, value) for value in values], draws) if draws else floor


def _card_opportunity_values(kernel, state, pool, schedules):
    direct, joint = [], []
    for card in pool:
        check_search_budget()
        definition = kernel.card_definition(state, card)
        if definition is None:
            raise KernelError('card_definition_missing')
        if (definition.plan not in {None, PlanType.SENSE}
                or (definition.required_stance is not None
                    and definition.required_stance is not state.stance)
                or not _condition_matches(state, definition.condition, kernel.cards)
                or not kernel._non_full_power_cost_payable(state, card, definition)
                or kernel._effective_full_power_cost(card, definition) > state.full_power
                or not kernel._card_secondary_ready(state, definition, card.instance_id)):
            direct.append(0.)
            joint.append(0.)
            continue
        paid, _ = kernel._pay_declared_non_full_power_cost(state, card, definition)
        paid = replace(paid, full_power=paid.full_power
                       - kernel._effective_full_power_cost(card, definition))
        base = sum(kernel.score_effect_gain(paid, child, card.instance_id)
                   * (child.repeats + card.score_repeat_bonus)
                   for child in definition.effects if child.kind is EffectKind.SCORE
                   and _condition_matches(state, child.condition, kernel.cards))
        gain = 0.
        for reference, effect in schedules:
            if effect.trigger_card_type not in {None, definition.card_type}:
                continue
            source = kernel._card_by_instance(paid, reference.source_id)
            repeat_bonus = source.score_repeat_bonus if source is not None else 0
            gain += sum(kernel.score_effect_gain(paid, child, reference.source_id)
                        * (child.repeats + repeat_bonus)
                        for child in effect.scheduled_effects)
        direct.append(base)
        joint.append(base + gain)
    return direct, joint


def installed_card_use_bonus(kernel, state):
    """Marginal best-of-draw score for one payable future card, or zero.

    Finite uses need only one remaining activation. The boundary has already
    decremented TTL; no additional duration decrement belongs here. The
    installed source may be removed while its schedule remains live.
    """
    check_search_budget()
    if (state.terminal or not is_hif_turn_boundary(state)
            or state.actions_remaining <= 0):
        return InstalledCardUseBonus(0., 'not_actionable_turn_boundary')
    if (state.deck.hand or state.deck.held or state.deck.known_top_count
            or state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION):
        return InstalledCardUseBonus(0., 'nonuniform_or_held_draw_domain')
    schedules = []
    for reference in state.scheduled_effects:
        check_search_budget()
        if (reference.trigger_phase is not SchedulePhase.CARD_USED
                or reference.turns_until_fire != 0 or reference.remaining_uses == 0
                or reference.remaining_turns == 0 or reference.direct_on_trigger):
            continue
        effect = kernel._scheduled_effect_spec(reference)
        if _pure_card_score(effect):
            schedules.append((reference, effect))
    if not schedules:
        return InstalledCardUseBonus(0., 'no_installed_pure_card_score')
    # Drawing must not itself select/modify cards or alter resources through
    # an unprojected entry trigger. AFTER_CARD_USED effects happen later and
    # do not prevent this single-opportunity estimate.
    if any(reference.trigger_phase in {SchedulePhase.CARD_MOVED_TO_HAND,
                                       SchedulePhase.BEFORE_CARD_USED}
           and reference.turns_until_fire == 0 and reference.remaining_uses != 0
           and reference.remaining_turns != 0 for reference in state.scheduled_effects):
        return InstalledCardUseBonus(0., 'unprojected_entry_trigger')
    draws, ignored = _next_draw_count(kernel, state)
    pool = state.deck.draw_pile
    fixed = ()
    if draws is None or draws <= 0 or len(pool) < draws:
        projection = _fixed_discard_draw(kernel, state)
        if projection is None:
            return InstalledCardUseBonus(0., 'unresolved_or_reshuffled_draw_domain',
                                         len(pool), 0 if draws is None else draws,
                                         len(schedules), ignored)
        state, pool, fixed, draws = projection
        ignored = 0
    direct, joint = _card_opportunity_values(kernel, state, pool, schedules)
    if fixed:
        fixed_direct, fixed_joint = _card_opportunity_values(kernel, state, fixed, schedules)
        value = (_expected_best_with_fixed(joint, draws, fixed_joint)
                 - _expected_best_with_fixed(direct, draws, fixed_direct))
    else:
        value = expected_best(joint, draws) - expected_best(direct, draws)
    check_search_budget()
    if not math.isfinite(value) or value < 0:
        raise ValueError('installed_card_use_invalid_bonus')
    reason = 'estimated_fixed_discard' if fixed else 'estimated'
    return InstalledCardUseBonus(value, reason if value else 'no_matching_marginal_gain',
                                 len(pool), draws, len(schedules), ignored)
