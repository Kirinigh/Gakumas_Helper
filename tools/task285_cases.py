"""Original synthetic cases for the offline paired benchmark."""

from dataclasses import dataclass

from agent.live_decision.high_score.hif import HifSenseKernel
from agent.live_decision.high_score.kernel import EffectKind, EffectSpec, CardDefinition, DrinkDefinition
from agent.live_decision.high_score.contracts import (
    CardRef,
    PlanType,
    ActionKind,
    DeckBelief,
    GameAction,
    DeckKnowledge,
    StrategyState,
)


@dataclass(frozen=True)
class Route:
    name: str
    prefix: tuple[GameAction, ...] = ()
    retain: str | None = None


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    sources: tuple[str, ...]
    kernel: HifSenseKernel
    state: StrategyState
    drink_slots: tuple[tuple[int, str], ...]
    routes: tuple[Route, ...]


def _opening(name, kernel, hand, draw_pile, *, stamina, slots=(), routes=()):
    hand = tuple(CardRef(label, label) for label in hand)
    draw_pile = tuple(CardRef(label, label) for label in draw_pile)
    state = StrategyState(
        turn=1, turns_remaining=1, score=0, stamina=stamina, max_stamina=stamina,
        genki=0, actions_remaining=1, hand_limit=5, score_multiplier=1,
        deck=DeckBelief(
            all_cards=hand + draw_pile, hand=hand, draw_pile=draw_pile,
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        ),
        drinks=tuple(drink for _, drink in slots),
    )
    state = kernel.initialize_stage_entry(state, plan_type=PlanType.SENSE)
    return Case(name, 'synthetic', (), kernel, state, slots, routes)


def load_case(name):
    if name != 'synthetic-draw':
        raise ValueError(f'unknown public benchmark case: {name}')
    effects = {
        'steady': (EffectSpec(EffectKind.SCORE, 40),),
        'draw': (EffectSpec(EffectKind.DRAW, 1), EffectSpec(EffectKind.EXTRA_ACTION, 1)),
        'small': (EffectSpec(EffectKind.SCORE, 10),),
        'big': (EffectSpec(EffectKind.SCORE, 90),),
    }
    kernel = HifSenseKernel({
        label: CardDefinition(label, label, PlanType.SENSE, effects=spec)
        for label, spec in effects.items()
    }, {})
    return _opening(name, kernel, ('steady', 'draw'), ('small', 'big'), stamina=10,
                    routes=(Route('draw', (GameAction(ActionKind.PLAY_CARD, 'draw'),)),))


def load_two_round_cases():
    """Synthetic proxy where conserving stamina unlocks the later score card."""
    drinks = {'carry': DrinkDefinition('carry', 'carry', (EffectSpec(EffectKind.STAMINA, 2),))}

    def kernel(scoring_cards):
        return HifSenseKernel({
            label: CardDefinition(label, label, PlanType.SENSE, stamina_cost=cost,
                                  effects=(EffectSpec(EffectKind.SCORE, score),))
            for label, cost, score in scoring_cards
        }, drinks)

    first = _opening('synthetic-carry', kernel((('steady', 2, 40), ('big', 5, 70))),
                     ('steady', 'big'), (), stamina=6, slots=((2, 'carry'),),
                     routes=(Route('steady', (GameAction(ActionKind.PLAY_CARD, 'steady'),)),))
    second = _opening('synthetic-carry-r2', kernel((('small', 0, 10), ('big', 5, 90))),
                      ('small', 'big'), (), stamina=6)
    return first, second
