"""The referee seed stays outside public route selection and score summaries."""

import time
from types import SimpleNamespace

import pytest

from tools import task285_seed_bench as bench
from tools.task285_cases import Case, Route, load_case
from tools.task285_retrieval_review import compare_public, finalist_pairs, _same_retrieval
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


@pytest.fixture
def case():
    effects = {'draw': (EffectSpec(EffectKind.DRAW, 1), EffectSpec(EffectKind.EXTRA_ACTION, 1)),
               'small': (EffectSpec(EffectKind.SCORE, 10),), 'big': (EffectSpec(EffectKind.SCORE, 90),)}
    kernel = HifSenseKernel({name: CardDefinition(name, name, PlanType.SENSE, effects=spec)
                            for name, spec in effects.items()}, {})
    draw, small, big = (CardRef(name, name) for name in effects)
    state = StrategyState(turn=1, turns_remaining=1, score=0, stamina=10, max_stamina=10,
        genki=0, actions_remaining=1, hand_limit=5, score_multiplier=1,
        deck=DeckBelief(all_cards=(draw, small, big), hand=(draw,), draw_pile=(small, big),
                       knowledge=DeckKnowledge.KNOWN_COMPOSITION))
    state = kernel.initialize_stage_entry(state, plan_type=PlanType.SENSE)
    return Case('fixture', 'fixture', (), kernel, state,
                (), (Route('draw', (GameAction(ActionKind.PLAY_CARD, 'draw'),)),))


class FirstLegal:
    def choose(self, request, legal):
        return legal[0]

    def committed(self, before, action, after):
        pass


def test_public_estimates_are_invariant_to_referee_seed(case):
    options = dict(planning_count=2, seconds=5., deadline=time.perf_counter() + 10.,
                   prior_factory=lambda kernel: FirstLegal())
    left = bench.evaluate_world(case, 1, 90, **options)
    right = bench.evaluate_world(case, 2, 90, **options)
    assert left['public_rollout_seeds'] == right['public_rollout_seeds']
    assert left['arms']['B']['estimates'] == right['arms']['B']['estimates']
    assert left['arms']['B']['episode']['route'] == right['arms']['B']['episode']['route']
    assert left['arms']['C']['estimates'] != right['arms']['C']['estimates']
    for row in (left, right):
        oracle = row['arms']['C']
        assert oracle['terminal']
        assert oracle['episode']['result']['final_score'] == max(x['mean'] for x in oracle['estimates'])
    assert bench.summarize([left, right], 2)['paired']['B-A']['paired_episode_count'] == 2
    assert bench.summarize([left, right], 3)['paired']['B-A'] is None


def test_unavailable_prefix_remains_incomplete(case):
    bad = Route('missing-card', (GameAction(ActionKind.PLAY_CARD, 'not-held'),))
    result = bench._episode(case, bad, 1, FirstLegal(), time.perf_counter() + 10., 'bad')
    assert not result['terminal']
    assert result['error']['message'] == 'route_prefix_unavailable:missing-card:0'
    assert result['actions'] == []


def test_public_retrieval_review_runs_declared_legal_roots():
    case = load_case('synthetic-draw')

    def playable_roots(kernel, request, legal, *, max_options, deadline):
        assert kernel is case.kernel
        assert request.state.deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION
        assert {card.instance_id for card in request.state.deck.hand} == {'draw', 'steady'}
        assert {card.instance_id for card in request.state.deck.draw_pile} == {'small', 'big'}
        assert max_options >= len(legal)
        options = [SimpleNamespace(action=act, continuation=())
                   for act in legal if act.kind is ActionKind.PLAY_CARD]
        return options, {'root_source': 'synthetic_legal'}

    _, _, audit = compare_public(case, bench.SyntheticPrior(), bench.INCUMBENT, 20,
                                 time.perf_counter() + 5, root_builder=playable_roots)
    assert audit['root_source'] == 'synthetic_legal'
    assert audit['retained_count'] == 0
    assert len(audit['candidates']) == 2
    assert set(audit['base_seeds']).isdisjoint(audit['final_seeds'])


def test_only_uncertain_retained_target_replaces_second_finalist():
    scores = ((100, 100, 100, 100), (95, 95, 95, 95), (20, 190, 10, 150), (20, 20, 20, 20))
    assert finalist_pairs(scores, None) == ((0, 1), (0, 1))
    assert finalist_pairs(scores, 2) == ((0, 1), (0, 2))
    assert finalist_pairs(scores, 3) == ((0, 1), (0, 1))


def test_reference_requires_original_physical_bottle_and_target_instance():
    bottle = GameAction(ActionKind.USE_DRINK, drink_id='22', source_slot=1)
    target = GameAction(ActionKind.MOVE_CARD, 'target-a')
    reference = Route('reference', (bottle, target))
    assert _same_retrieval(Route('match', (bottle, target)), reference)
    assert not _same_retrieval(Route('other_target', (bottle, GameAction(ActionKind.MOVE_CARD, 'target-b'))), reference)
    assert not _same_retrieval(Route('other_bottle', (GameAction(ActionKind.USE_DRINK, drink_id='22', source_slot=2),
                                                     target)), reference)
