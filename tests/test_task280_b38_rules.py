from dataclasses import replace

from tools.task280_b38_rules import SUMMER, SPOTLIGHT, MICROPHONE, B38RuleKernel
from tools.task280_shared_rules import summer_memory
from agent.live_decision.high_score.hif import HifSenseKernel
from agent.live_decision.high_score.kernel import EffectKind, EffectSpec, CardDefinition
from agent.live_decision.high_score.contracts import (
    CardRef,
    PlanType,
    ActionKind,
    DeckBelief,
    GameAction,
    StrategyState,
)


def state_for(hand, *, draw=(), discard=(), items=()):
    return StrategyState(turn=1, turns_remaining=3, score=0, stamina=30, max_stamina=30,
                         genki=0, actions_remaining=9, hand_limit=8, score_multiplier=1,
                         future_score_multipliers=(1, 1), good_condition_turns=6, concentration=6,
                         deck=DeckBelief(hand + draw + discard, hand, draw, discard), items=items)


def play(kernel, state, card):
    return kernel.transition(state, GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
                             plan_type=PlanType.SENSE)


def test_b38_spotlight_customized_focus_and_microphone_receipt():
    spotlight = CardRef('spotlight', SPOTLIGHT, upgraded=True)
    nonfocus = CardRef('plain', 'plain')
    focus = CardRef('focus', 'focus')
    definitions = {
        'plain': CardDefinition('plain', 'plain', PlanType.SENSE, stamina_cost=4),
        'focus': CardDefinition('focus', 'focus', PlanType.SENSE, stamina_cost=4,
                                effects=(EffectSpec(EffectKind.CONCENTRATION, amount=3),)),
    }
    kernel = B38RuleKernel(definitions, {}, {}, focus_card_ids={SPOTLIGHT, 'focus'})
    state = state_for((spotlight, nonfocus, focus), items=(MICROPHONE,))
    first, = play(kernel, state, spotlight)
    assert (first.state.concentration, first.state.good_condition_turns) == (15, 4)
    assert first.state.half_cost_turns == 3
    assert next(ref for ref in first.state.scheduled_effects if ref.card_id == MICROPHONE).remaining_uses == 2
    second, = play(kernel, first.state, nonfocus)
    assert second.state.stamina == 28
    assert next(ref for ref in second.state.scheduled_effects if ref.card_id == MICROPHONE).remaining_uses == 2
    third, = play(kernel, second.state, focus)
    assert third.state.stamina == 26
    assert third.state.half_cost_turns == 5
    assert next(ref for ref in third.state.scheduled_effects if ref.card_id == MICROPHONE).remaining_uses == 1


def test_b38_microphone_does_not_discount_triggering_cost_and_exhausts():
    focus = tuple(CardRef(f'focus-{index}', 'focus') for index in range(4))
    kernel = B38RuleKernel({
        'focus': CardDefinition('focus', 'focus', PlanType.SENSE, stamina_cost=4,
                                effects=(EffectSpec(EffectKind.CONCENTRATION, amount=3),)),
    }, {}, {}, focus_card_ids={'focus'})
    state = state_for(focus, items=(MICROPHONE,))
    for card, check in zip(focus, ((26, 3), (24, 5), (22, 7), (20, 7)), strict=True):
        branch, = play(kernel, state, card)
        state = branch.state
        assert (state.stamina, state.half_cost_turns) == check
    assert not any(ref.card_id == MICROPHONE and ref.remaining_uses for ref in state.scheduled_effects)


def test_b38_first_microphone_turn_boundary_from_referee_probe():
    focus = CardRef('focus', 'focus')
    kernel = B38RuleKernel({
        'focus': CardDefinition('focus', 'focus', PlanType.SENSE, stamina_cost=4,
                                effects=(EffectSpec(EffectKind.CONCENTRATION, amount=3),)),
    }, {}, {}, focus_card_ids={'focus'})
    start = replace(state_for((focus,), items=(MICROPHONE,)), stamina=24, max_stamina=32)
    used, = play(kernel, start, focus)
    boundary, = kernel.transition(replace(used.state, actions_remaining=0),
                                  GameAction(ActionKind.END_TURN), plan_type=PlanType.SENSE)
    assert (boundary.state.turn, boundary.state.stamina, boundary.state.concentration,
            boundary.state.half_cost_turns) == (2, 20, 9, 2)


def test_b38_spotlight_valued_as_customized_single_plus_not_referee_double_plus():
    spotlight = CardRef('spotlight', SPOTLIGHT, upgraded=True)
    ordinary = tuple(CardRef(f'plain-{index}', 'plain') for index in range(3))
    kernel = B38RuleKernel({'plain': CardDefinition('plain', 'plain', PlanType.SENSE)},
                           {}, {}, focus_card_ids=())
    state, = play(kernel, state_for((spotlight, *ordinary)), spotlight)
    state, = play(kernel, state.state, ordinary[0])
    trigger = replace(state.state, concentration=17, good_condition_turns=7,
                      excellent_condition_turns=1, score_multiplier=31.01)
    result, = play(kernel, trigger, ordinary[1])
    assert result.state.score == 5055
    assert result.state.good_condition_turns == 4
    assert result.state.scheduled_effects[0].effect_counter == 3


def test_shared_summer_only_removes_named_basic_from_draw_or_discard_and_counts():
    summer = CardRef('summer', SUMMER, upgraded=True)
    subsequent = tuple(CardRef(f'play-{index}', 'plain') for index in range(5))
    hand_basic = CardRef('in-hand', '23')
    draw_basic = CardRef('in-draw', '23')
    discard_basic = CardRef('in-discard', '23')
    other_trouble = CardRef('other-trouble', 'fake')
    definitions = {
        '23': CardDefinition('23', 'target', None, card_type='trouble', limited=True),
        'fake': CardDefinition('fake', 'other', None, card_type='trouble', limited=True),
        'plain': CardDefinition('plain', 'plain', PlanType.SENSE),
    }
    kernel = HifSenseKernel(definitions | {SUMMER: summer_memory()}, {}, {})
    start = state_for((summer, hand_basic, *subsequent), draw=(draw_basic, other_trouble), discard=(discard_basic,))
    branches = play(kernel, replace(start, good_condition_turns=0, concentration=0), summer)
    assert [branch.probability for branch in branches] == [.5, .5]
    assert {card for branch in branches for card in branch.state.deck.removed
            if card != summer} == {draw_basic, discard_basic}
    state = branches[0].state
    assert hand_basic in state.deck.hand
    assert state.scheduled_effects[0].effect_counter == 0
    for index, card in enumerate(subsequent, start=1):
        stage, = play(kernel, state, card)
        state = stage.state
        assert state.scheduled_effects[0].effect_counter == index
        assert state.score == (4 if index == 5 else 0)
