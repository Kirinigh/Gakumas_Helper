"""Synthetic Sense roots for the local UIObs protocol tests."""

import time
from types import SimpleNamespace
from dataclasses import replace

import pytest

from agent.live_decision.high_score.hif import HifSenseKernel
from tools.task275.gkms_continuous_policy import execute_episode
from agent.live_decision.high_score.kernel import EffectKind, EffectSpec, CardDefinition, DrinkDefinition
from agent.live_decision.high_score.contracts import (
    CardRef,
    CardZone,
    PlanType,
    ActionKind,
    DeckBelief,
    GameAction,
    TimedStatus,
    DecisionKind,
    StrategyState,
    DecisionRequest,
    CapabilityReport,
)
from agent.live_decision.high_score.hif_environment import HifEnvironment
from agent.live_decision.high_score.gkms_shadow_adapter import (
    ExamContext,
    TurnPlayHistory,
    ShadowMappingError,
    build_shadow_observation,
)
from agent.live_decision.high_score.gkms_shadow_mappings import ShadowMappings
from agent.live_decision.high_score.gkms_continuous_policy import GkmsContinuousPolicy

CARD_IDS = tuple(f'card-{index}' for index in range(11))
ATTRIBUTES = ('visual', 'dance', 'vocal', 'vocal', 'visual', 'visual', 'dance', 'vocal', 'visual')
CONTEXT = ExamContext(
    'produce-008', 'AUDITION_MID1', 2, (3, 2, 1, 1, 3, 3, 2, 1, 3),
    (1000, 2000, 3000), (10, 20, 30), 'fixture-idol', 'synthetic fixture',
    mappings=ShadowMappings(
        card_forms={**{card_id: (f'wire-{card_id}', 0, 'NORMAL') for card_id in CARD_IDS},
                    '816': ('wire-synthetic-boost', 1, 'NORMAL'),
                    '818': ('wire-synthetic-played', 1, 'NORMAL')},
        card_upgrades={}, custom_forms={}, drink_ids={'22': 'wire-retrieval'},
        schedules={}, grow_effects={}, support_deltas={'816': {}, '818': {}},
    ),
)


@pytest.fixture(scope='module')
def kernel():
    cards = {
        card_id: CardDefinition(card_id, card_id, PlanType.SENSE, stamina_cost=1,
                                effects=(EffectSpec(EffectKind.SCORE, 4),))
        for card_id in CARD_IDS
    }
    cards.update({
        card_id: CardDefinition(card_id, 'Fixture support', PlanType.SENSE, stamina_cost=3,
                                effects=(EffectSpec(EffectKind.SCORE, 4),))
        for card_id in ('816', '818')
    })
    retrieval = DrinkDefinition('22', 'Fixture retrieval', (
        EffectSpec(EffectKind.MOVE_SELECTED_TO_HAND, target_zones=(CardZone.DRAW_PILE, CardZone.DISCARD)),
        EffectSpec(EffectKind.NULLIFY_COST_CARDS, amount=2),
        EffectSpec(EffectKind.STAMINA, amount=-2),
        EffectSpec(EffectKind.DOUBLE_COST_TURNS, amount=1),
    ), plan=PlanType.SENSE)
    def support_definition(ordinary, tier):
        if (ordinary.card_id, tier) != ('816', 2):
            raise ValueError('unsupported synthetic support tier')
        return replace(ordinary, stamina_cost=2)

    return HifSenseKernel(cards, {'22': retrieval}, {}, support_definition=support_definition)


def initial():
    hand = tuple(CardRef(f'hand-{index}', card_id) for index, card_id in enumerate(CARD_IDS[:3]))
    deck = tuple(CardRef(f'deck-{index}', card_id) for index, card_id in enumerate(CARD_IDS[3:]))
    return StrategyState(
        1, 9, 0, 36, 36, 0, 1, 5, 3., DeckBelief(hand + deck, hand, deck),
        future_score_multipliers=(2., 1., 1., 3., 3., 2., 1., 3.),
        turn_attributes=ATTRIBUTES, persistent_effects_initialized=True, drinks=('22', '22'),
    )


def request(state, *, card_slots=None, drink_slots=None):
    return DecisionRequest(
        'fixture-root', PlanType.SENSE, DecisionKind.LIVE, state, CapabilityReport(True, True, True, True, False),
        card_slots=(tuple((card.instance_id, slot) for slot, card in enumerate(state.deck.hand))
                    if card_slots is None else card_slots),
        drink_slots=tuple(enumerate(state.drinks)) if drink_slots is None else drink_slots,
    )


def project(kernel, state=None, *, req=None, context=CONTEXT):
    state = initial() if state is None else state
    req = request(state) if req is None else req
    return build_shadow_observation(req, kernel, context, TurnPlayHistory(state, (), 'fixture'))


def isolated(card_id):
    state = initial()
    card = CardRef('isolated', card_id, upgraded=card_id in ('816', '818'))
    return replace(state, deck=DeckBelief((card,), (card,), ()), drinks=())


def test_projection_binds_duplicate_drinks_to_physical_slots(kernel):
    state = initial()
    req = request(state, drink_slots=((1, '22'), (3, '22')))
    shadow = build_shadow_observation(req, kernel, CONTEXT, TurnPlayHistory(state, (), 'fixture'))
    assert shadow.status == 'supported', shadow.reason_codes
    assert shadow.observation['drinks'] == ['wire-retrieval', 'wire-retrieval']
    first, second = [shadow.bind_action(('drink', i), request=req, kernel=kernel) for i in (0, 1)]
    assert (first.kind, first.source_slot, second.source_slot) == (ActionKind.USE_DRINK, 1, 3)
    with pytest.raises(ShadowMappingError, match='not_in_supplied_legal_set'):
        shadow.bind_action(('drink', 2), request=req, kernel=kernel)


def test_projection_requires_verified_explicit_context_and_history(kernel):
    state = initial()
    req = request(state)
    history = TurnPlayHistory(state, (), 'fixture')
    no_mappings = build_shadow_observation(req, kernel, replace(CONTEXT, mappings=None), history)
    assert no_mappings.status == 'unsupported'
    assert no_mappings.reason_codes == ('verified_shadow_mappings_required',)
    no_history = build_shadow_observation(req, kernel, CONTEXT, None)
    assert no_history.status == 'unsupported'
    assert no_history.reason_codes == ('played_history_missing',)


def test_anti_debuff_charges_bind_and_guard_exactly_one_effect(kernel):
    state = replace(initial(), actions_remaining=2, statuses=(TimedStatus('anti_debuff', 1),))
    projected = project(kernel, state)
    assert projected.observation['status'] == [['anti_debuff', 1, -1]]
    unverified = project(kernel, replace(state, statuses=(TimedStatus('other', 1),)))
    assert unverified.reason_codes == ('state_semantic_mapping_unverified:statuses:other',)
    assert unverified.observation is None

    card = state.deck.hand[0]
    original = kernel.cards[card.card_id]
    effects = (EffectSpec(EffectKind.DOUBLE_COST_TURNS, 1), EffectSpec(EffectKind.NULLIFY_GENKI_TURNS, 2))
    rules = HifSenseKernel(kernel.cards | {card.card_id: replace(original, effects=effects)}, kernel.drinks, {})
    played, = rules.transition(state, GameAction(ActionKind.PLAY_CARD, card.instance_id), plan_type=PlanType.SENSE)
    assert not played.state.statuses
    assert not played.state.double_cost_turns
    assert played.state.nullify_genki_turns == 2
    assert [event.detail for event in played.events if event.event_type == 'DEBUFF_NULLIFIED'] == ['DOUBLE_COST_TURNS']


@pytest.mark.parametrize(('token', 'terminal'), [(['play', 0], True), (['drink', 0], False)])
def test_continuous_policy_uses_offline_step_and_requires_local_secondary(kernel, token, terminal):
    state = replace(initial(), turn=9, turns_remaining=1, future_score_multipliers=(), turn_attributes=('visual',))
    environment = HifEnvironment(kernel)
    first = environment.reset(state, seed=1)

    def decide(observation):
        return dict(action=token, probs=[float(candidate == token) for candidate in observation['legal']])

    deadline = time.perf_counter() + 30
    policy = GkmsContinuousPolicy(
        kernel, SimpleNamespace(decide=decide), CONTEXT, first.state, source='synthetic episode',
        planning_seed=1, deadline=deadline,
    )
    result = execute_episode(environment, policy, first, episode_id='fixture', deadline=deadline)
    assert result['terminal'] is terminal
    assert result['decisions'][0]['status'] == 'committed'
    assert len(result['actions']) == 1
    if terminal:
        assert result['error'] is None and result['observed_score'] > 0
    else:
        assert result['error']['message'] == 'offline_secondary_policy_required'
