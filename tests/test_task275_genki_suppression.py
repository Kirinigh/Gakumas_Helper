"""Source-backed genki suppression, expiry and fixed-genki exceptions."""
import csv
import json
from pathlib import Path
from dataclasses import asdict, replace

import pytest

from agent.live_decision.high_score import (
    CardRef,
    PlanType,
    ActionKind,
    DeckBelief,
    EffectKind,
    EffectSpec,
    GameAction,
    StrategyState,
    CardDefinition,
    PythonNiaProKernel,
)
from agent.live_decision.high_score.golden import _state
from agent.live_decision.high_score.catalog_v2 import compile_skill_card_catalog

ROOT = Path(__file__).resolve().parents[1]


def setup_state(card_id='292'):
    catalog = compile_skill_card_catalog(ROOT/'assets/data/skill_cards.csv',
        plan_type=PlanType.SENSE, include_free_card_ids=('100', '101'))
    refs = (CardRef('source', card_id), CardRef('gain', 'gain'))
    cards = dict(catalog.cards)
    cards['gain'] = CardDefinition('gain', 'gain', PlanType.SENSE, effects=(
        EffectSpec(EffectKind.GENKI, 7), EffectSpec(EffectKind.FIXED_GENKI, 2)))
    kernel = PythonNiaProKernel(cards)
    state = StrategyState(turn=1, turns_remaining=4, score=0, stamina=40,
        max_stamina=40, genki=0, actions_remaining=2, hand_limit=0,
        score_multiplier=1, concentration=6, good_condition_turns=3,
        motivation=4, deck=DeckBelief(refs, refs, ()))
    return kernel, state


def play(kernel, state, instance):
    return kernel.transition(state, GameAction(ActionKind.PLAY_CARD,
        card_instance_id=instance), plan_type=PlanType.SENSE)[0].state


def test_all_catalog_genki_suppression_cards_compile_and_apply_in_order():
    rows = list(csv.DictReader((ROOT/'assets/data/skill_cards.csv').open(encoding='utf-8-sig')))
    affected = [r for r in rows if 'nullifyGenkiTurns' in r['actions']]
    assert {r['id'] for r in affected} == {'100', '101', '292', '293'}
    for row in affected:
        kernel, state = setup_state(row['id'])
        used = play(kernel, state, 'source')
        assert used.genki >= 15  # own gain happens before suppression, includes motivation
        assert used.nullify_genki_turns == 2 and used.nullify_genki_fresh
        after = play(kernel, used, 'gain')
        assert after.genki == used.genki + 2  # ordinary gain suppressed; fixed remains


def test_fresh_suppression_survives_application_turn_then_expires():
    kernel, state = setup_state()
    state = play(kernel, state, 'source')
    counts = []
    for _ in range(3):
        state = kernel.transition(state, GameAction(ActionKind.END_TURN),
            plan_type=PlanType.SENSE)[0].state
        counts.append(state.nullify_genki_turns)
        assert not state.nullify_genki_fresh
    assert counts == [2, 1, 0]
    state = replace(state, actions_remaining=1,
        deck=state.deck.replace_zones(hand=(state.deck.all_cards[1],),
            discard=tuple(c for c in state.deck.discard if c.instance_id!='gain')))
    used = play(kernel, state, 'gain')
    assert used.genki == state.genki + 7 + state.motivation + 2


def test_reapplication_adds_duration_without_refreshing_old_effect():
    kernel, state = setup_state()
    state = replace(state, nullify_genki_turns=1)
    used = play(kernel, state, 'source')
    assert used.genki == 0 and used.nullify_genki_turns == 3
    assert not used.nullify_genki_fresh


def test_suppression_state_roundtrip_preserves_counter_and_freshness():
    _, state = setup_state()
    state = replace(state, nullify_genki_turns=2, nullify_genki_fresh=True)
    payload = json.loads(json.dumps(asdict(state)))
    for zone in ('hand', 'draw_pile', 'discard', 'removed', 'held'):
        payload['deck'][zone] = [c['instance_id'] for c in payload['deck'][zone]]
    assert _state(payload) == state


@pytest.mark.parametrize('turns,fresh', [(-1,False),(1.5,False),(True,False),(0,True),(1,1)])
def test_invalid_suppression_state_is_rejected(turns,fresh):
    _, state = setup_state()
    with pytest.raises(ValueError):
        replace(state,nullify_genki_turns=turns,nullify_genki_fresh=fresh).validate()
