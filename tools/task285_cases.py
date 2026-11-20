"""Original synthetic cases for the offline paired benchmark."""

from dataclasses import dataclass

from agent.live_decision.high_score.hif import HifSenseKernel
from agent.live_decision.high_score.kernel import EffectKind, EffectSpec, CardDefinition
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
    steady, draw, small, big = (CardRef(label, label) for label in effects)
    state = StrategyState(
        turn=1, turns_remaining=1, score=0, stamina=10, max_stamina=10,
        genki=0, actions_remaining=1, hand_limit=5, score_multiplier=1,
        deck=DeckBelief(
            all_cards=(steady, draw, small, big), hand=(steady, draw), draw_pile=(small, big),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        ),
    )
    state = kernel.initialize_stage_entry(state, plan_type=PlanType.SENSE)
    return Case(name, 'synthetic', (), kernel, state, (),
                (Route('draw', (GameAction(ActionKind.PLAY_CARD, 'draw'),)),))
