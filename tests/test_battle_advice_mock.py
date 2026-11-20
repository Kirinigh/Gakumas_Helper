import json
from dataclasses import replace

import pytest
from test_task275_gkms_shadow_adapter import kernel, initial  # noqa: F401

from tools.task275.battle_advice_mock import battle_offer, replay_advice
from agent.live_decision.high_score.contracts import CardRef, PlanType, ActionKind, DeckBelief, GameAction, DeckKnowledge, VerifiedHandSupport
from agent.live_decision.high_score.hif_environment import HifEnvironment


@pytest.fixture
def scene(kernel):  # noqa: F811
    state = initial()
    original, second = (CardRef(iid, '816', upgraded=True) for iid in ('first', 'second'))
    third = state.deck.draw_pile[0]
    state = replace(state, cost_reduction=2, deck=DeckBelief((original, second, third), (original, second), (third,)))
    env = HifEnvironment(kernel)
    request = env.reset(state, seed=4, drink_slots=((1, '22'), (3, '22')))
    request = replace(request, card_slots=(('second', 0), ('first', 1)))
    return env, request, battle_offer(request, kernel)


def unreachable(*_):
    pytest.fail('unexpected local decision')


def test_only_current_slot_binds_despite_identical_cards_and_future_plan(scene):
    _, request, offer = scene
    assert offer.payload['played'] is None and offer.payload['piles']['held'] == {}
    first, second = offer.payload['hand']
    assert first['name'] == second['name'] and (first['slot'], second['slot']) == (0, 1)
    reply = json.dumps(dict(action='play:1', plan=[dict(action='play', card=first['name'])],
                            score=999999, reason='continue live now'))
    result = replay_advice(offer, request, [reply], unreachable)
    assert result.action == GameAction(ActionKind.PLAY_CARD, 'first', source_slot=1)
    assert result.to_dict()['value'] is None and not result.to_dict()['execution_authorized']
    assert len(offer.payload['legal_actions']) == len(set(offer.payload['legal_actions']))
    assert 'continue live' not in str(result.to_dict())
    json.dumps(offer.payload, allow_nan=False)


def test_816_prompt_uses_permanent_valuation_even_with_verified_temporary_detail(scene):
    env, request, _ = scene
    fact = VerifiedHandSupport('first', '816', request.state.turn, 2)
    observed = replace(request, state=replace(request.state, hand_support=(fact,)))
    hand = battle_offer(observed, env.kernel).payload['hand']
    assert [card['printed_cost'] for card in hand] == [3, 3]
    assert [card['estimated_cost'] for card in hand] == [1, 1]
    assert observed.state.hand_support == (fact,)


def test_unknown_zone_composition_and_unknown_play_history_remain_unknown(scene):
    env, request, _ = scene
    state = replace(request.state, deck=replace(request.state.deck, knowledge=DeckKnowledge.UNKNOWN))
    view = battle_offer(replace(request, state=state), env.kernel).payload
    assert all(pile is None for pile in view['piles'].values())
    assert view['played'] is None


def test_distinct_drink_slots_and_secondary_target_remains_with_local_policy(scene):
    env, request, offer = scene
    reply = json.dumps(dict(action='drink:3', plan=[dict(action='move', card='816')]))
    first = replay_advice(offer, request, [reply], unreachable)
    assert first.action == GameAction(ActionKind.USE_DRINK, drink_id='22', source_slot=3)
    current = env.step(first.action).request
    assert current.state.pending_choice is not None
    local = env.legal_actions()[0]
    secondary = battle_offer(current, env.kernel)
    result = replay_advice(secondary, current, [], lambda _: local)
    assert (result.action, result.source, result.reply_count) == (local, 'local_policy', 0)
    assert result.action.card_instance_id in current.state.pending_choice.candidate_instance_ids


@pytest.mark.parametrize(('replies', 'reason', 'count'), [
    ([TimeoutError()], 'transport_timeout', 1), (['{'], 'invalid_json', 1),
    (['{}'], 'missing_action', 1), (['null'], 'missing_action', 1),
    ([None], 'missing_response', 1), (['{"action":"move:0"}', '{"action":"drink:0"}'], 'illegal_action', 2),
])
def test_local_fallback_with_no_score_or_skip_invented(scene, replies, reason, count):
    _, current, offer = scene
    original = dict(offer.bindings)['drink:1']
    result = replay_advice(offer, current, replies, lambda _: original)
    assert (result.source, result.action, result.reason, result.reply_count) == ('local_policy', original, reason, count)
    assert result.to_dict()['value'] is None


def test_invalid_action_can_be_corrected_once_against_same_binding(scene):
    _, current, offer = scene
    result = replay_advice(offer, current, ['{"action":"drink:0"}', '{"action":"drink:1"}'], unreachable)
    assert (result.action.source_slot, result.source, result.reply_count) == (1, 'scripted_battle_reply', 2)


def test_offline_episode_can_apply_scoreless_advice_without_a_search_cache(scene):
    env, _, _ = scene
    current = env.request()
    offer = battle_offer(current, env.kernel)
    advice = replay_advice(offer, current, ['{"action":"play:0"}'], unreachable)
    assert advice.action.card_instance_id == 'first'
    assert advice.to_dict()['value'] is None
    assert not advice.to_dict()['execution_authorized']
    step = env.step(advice.action)
    assert any(event.source_id == 'first' for event in step.events)


def test_cold_rebuild_binds_same_instance_but_old_snapshot_cannot_choose_after_reorder(scene):
    env, current, offer = scene
    reply = ['{"action":"play:1"}']
    original = replay_advice(offer, current, reply, unreachable)
    assert replay_advice(battle_offer(current, env.kernel), current, reply, unreachable) == original
    reordered = replace(current, card_slots=(('first', 0), ('second', 1)))
    fallback = GameAction(ActionKind.PLAY_CARD, 'first', source_slot=0)
    assert fallback.card_instance_id in {c.instance_id for c in reordered.state.deck.hand}
    assert env.kernel.legal_actions(reordered.state, plan_type=PlanType.SENSE, decision_kind=reordered.decision_kind)
    result = replay_advice(offer, reordered, reply, lambda r: fallback if r == reordered else unreachable())
    assert (result.reason, result.reply_count, result.action) == ('observation_changed', 0, fallback)
