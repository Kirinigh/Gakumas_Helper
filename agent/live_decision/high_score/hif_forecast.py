"""Explicit score-based leaf estimate, never used as a settlement prediction."""
from math import comb

from .kernel import EffectKind, _condition_matches
from .contracts import SchedulePhase, TurnResolutionContinuation
from .state_copy import replace


def expected_best(values, draws):
    """Expected maximum of an unordered draw without replacement."""
    values = sorted(values)
    count = min(draws, len(values))
    if count <= 0:
        return 0.0
    denominator = comb(len(values), count)
    return sum(value * (comb(i + 1, count) - comb(i, count)) / denominator
               for i, value in enumerate(values) if i + 1 >= count)


def score_effect_gain(kernel, state, effect, source_id):
    return kernel.score_effect_gain(state, effect, source_id)


def _decayed_modifiers(modifiers, offset):
    return tuple(replace(buff, remaining_turns=(None if buff.remaining_turns is None
                                               else buff.remaining_turns - max(0, offset - int(buff.fresh))))
                 for buff in modifiers if buff.remaining_turns is None
                 or buff.remaining_turns > max(0, offset - int(buff.fresh)))


def sense_future_score(kernel, state, *, concentration_timeline=None):
    """Estimate future conversion windows from the actual remaining card pool.

    Remaining draw identities are unknown. Use exact expected best-of-three
    immediate scores as a leaf feature, then value held score/draw drinks at
    their best remaining windows. Future buffs, depletion and trigger chains
    are approximations here; actual current-turn effects stay in the kernel.
    This value is not a bound, a P10 guarantee or a predicted game result.
    """
    if state.terminal:
        return float(state.score)
    if concentration_timeline is not None and (
            len(concentration_timeline) != state.turns_remaining
            or any(type(value) is not int or value < 0 for value in concentration_timeline)
            or concentration_timeline[0] != state.concentration):
        raise ValueError('invalid_concentration_timeline')
    removed = {card.instance_id for card in state.deck.removed}
    pool = tuple(card for card in state.deck.all_cards if card.instance_id not in removed)
    windows, unboosted, draw_gains = [], [], []
    delayed_scores = []
    for reference in state.scheduled_effects:
        if (not reference.persistent and reference.remaining_uses != 0
                and reference.trigger_phase in {SchedulePhase.TURN, SchedulePhase.BEFORE_START_OF_TURN,
                    SchedulePhase.START_OF_TURN, SchedulePhase.AFTER_START_OF_TURN}):
            delayed_scores.append((reference.turns_until_fire, reference))
    for frame in state.resolution_stack:
        if isinstance(frame, TurnResolutionContinuation):
            delayed_scores.extend((0, reference) for reference in
                (*frame.due_before_start, *frame.due_start, *frame.due_after_start, *frame.due_turn)
                if not reference.persistent and reference.remaining_uses != 0)
    multipliers = (state.score_multiplier, *state.future_score_multipliers)[:state.turns_remaining]
    for offset, multiplier in enumerate(multipliers):
        good = max(0, state.good_condition_turns - max(0, offset - int(state.good_condition_fresh)))
        excellent = max(0, state.excellent_condition_turns - max(0, offset - int(state.excellent_condition_fresh)))
        projected = replace(
            state, score=0, score_multiplier=multiplier, turn=state.turn + offset,
            turns_remaining=state.turns_remaining - offset,
            future_score_multipliers=multipliers[offset + 1:],
            turn_attributes=state.turn_attributes[offset:],
            concentration_effect_buffs=_decayed_modifiers(state.concentration_effect_buffs, offset),
            score_buffs=_decayed_modifiers(state.score_buffs, offset),
            good_condition_turns=good, good_condition_fresh=bool(good and offset == 0 and state.good_condition_fresh),
            excellent_condition_turns=excellent,
            excellent_condition_fresh=bool(excellent and offset == 0 and state.excellent_condition_fresh),
        )
        if concentration_timeline is not None:
            projected = replace(projected, concentration=concentration_timeline[offset])
        scores = []
        for card in pool:
            definition = kernel.cards[card.card_id]
            gain = 0.0
            if (_condition_matches(projected, definition.condition, kernel.cards)
                    and kernel._non_full_power_cost_payable(projected, card, definition)):
                for effect in definition.effects:
                    if effect.kind is EffectKind.SCORE and _condition_matches(projected, effect.condition, kernel.cards):
                        gain += score_effect_gain(kernel, projected, effect, card.instance_id) * (
                            effect.repeats + card.score_repeat_bonus)
            scores.append(gain)
        score = expected_best(scores, 3)
        actions = state.actions_remaining if offset == 0 else 1
        delayed = 0.
        for due_offset, reference in delayed_scores:
            if due_offset != offset:
                continue
            specification = kernel._scheduled_effect_spec(reference)
            if specification.condition is None:
                delayed += sum(score_effect_gain(kernel, projected, effect, reference.source_id) * effect.repeats
                    for effect in specification.scheduled_effects
                    if effect.kind is EffectKind.SCORE and effect.condition is None)
        windows.append(score * actions + delayed)
        unboosted.append((score * actions + delayed) / (1 + sum(buff.amount for buff in projected.score_buffs)))
        draw_gains.append((expected_best(scores, 4) - score, expected_best(scores, 5) - score))
    option_value = 0.0
    for drink_id in state.drinks:
        drink = kernel.drinks[drink_id]
        if getattr(drink, 'reserved_for_later', False):
            continue
        for effect in drink.effects:
            if effect.kind is EffectKind.SCORE_BUFF:
                width = len(windows) if effect.duration is None else effect.duration + 1
                option_value += effect.amount * max((sum(unboosted[i:i + width]) for i in range(len(windows))), default=0)
            elif effect.kind is EffectKind.EXTRA_ACTION:
                option_value += effect.amount * max(windows, default=0)
            elif effect.kind is EffectKind.DRAW:
                index = min(1, max(0, int(effect.amount) - 1))
                option_value += max((gain[index] for gain in draw_gains), default=0)
            elif effect.kind is EffectKind.SCHEDULE and all(child.kind is EffectKind.DRAW for child in effect.scheduled_effects):
                width = effect.schedule_limit or len(windows)
                option_value += max((sum(gain[0] for gain in draw_gains[i:i + width])
                                     for i in range(len(windows))), default=0)
    return state.score + sum(windows) + option_value
