"""UIObs v2 mock projection and binding on synthetic Sense fixtures."""
from copy import deepcopy
from dataclasses import replace

import pytest
from test_task275_gkms_shadow_adapter import CONTEXT, kernel, initial, project, request, isolated  # noqa: F401

from tools.task275.gkms_v2_observation import project_v2, bind_mock_output, build_v2_observation
from agent.live_decision.high_score.contracts import CardRef, ActionKind, SchedulePhase, ScheduledEffectRef, VerifiedHandSupport
from agent.live_decision.high_score.hif_environment import HifEnvironment
from agent.live_decision.high_score.gkms_shadow_adapter import TurnPlayHistory, ShadowMappingError, build_shadow_observation


def mock_pick(shadow, token):
    legal = shadow.observation['legal']
    return dict(action=token, probs=[float(candidate == token) for candidate in legal])


def test_same_name_instances_and_bottle_holes_keep_their_physical_binding(kernel):  # noqa: F811
    state = initial()
    a, b, c = state.deck.hand
    same_card = replace(b, card_id=a.card_id)
    state = replace(state, drinks=('22', '22'), deck=replace(state.deck,
        hand=(a, same_card, c), all_cards=(a, same_card, c, *state.deck.draw_pile)))
    req = request(state, card_slots=((a.instance_id, 2), (b.instance_id, 0), (c.instance_id, 1)),
        drink_slots=((3, '22'), (1, '22')))
    base = project(kernel, state, req=req)
    original = deepcopy(base.observation)
    projected = project_v2(base)
    assert projected.status == 'supported'
    obs = projected.observation
    assert obs['v'] == 'uiobs.v2' and obs['timers'] == obs['costs'] == obs['pcount'] == []
    assert obs['hand'][0] == obs['hand'][2] and obs['legal'] == original['legal']
    selected = bind_mock_output(projected, mock_pick(projected, ['play', 0]), request=req, kernel=kernel)
    assert (selected.card_instance_id, selected.source_slot) == (b.instance_id, 0)
    last = bind_mock_output(projected, mock_pick(projected, ['play', 2]), request=req, kernel=kernel)
    assert (last.card_instance_id, last.source_slot) == (a.instance_id, 2)
    bottle = bind_mock_output(projected, mock_pick(projected, ['drink', 1]), request=req, kernel=kernel)
    assert (bottle.drink_id, bottle.source_slot) == ('22', 3)
    assert bind_mock_output(projected, mock_pick(projected, ['end']), request=req, kernel=kernel).kind is ActionKind.END_TURN
    assert base.observation == original
    with pytest.raises(ShadowMappingError, match='stale_request'):
        bind_mock_output(projected, mock_pick(projected, ['play', 0]), request=replace(req, snapshot_id='later'), kernel=kernel)
    with pytest.raises(ValueError, match='argmax'):
        bind_mock_output(projected, dict(mock_pick(projected, ['play', 0]), action=['end']), request=req, kernel=kernel)
    with pytest.raises(ShadowMappingError, match='index_invalid'):
        bind_mock_output(projected, dict(mock_pick(projected, ['play', 0]), action=['play', False]), request=req, kernel=kernel)
    with pytest.raises(ValueError, match='probabilities_invalid'):
        bind_mock_output(projected, dict(action=['end'], probs=[True] * len(obs['legal'])), request=req, kernel=kernel)


@pytest.mark.parametrize('stamina', [2, 3])
def test_816_original_plus_drives_both_encoded_hand_and_legality(kernel, stamina):  # noqa: F811
    ordinary = replace(isolated('816'), stamina=stamina)
    played_card = CardRef('earlier-818', '818', upgraded=True)
    deck = replace(ordinary.deck, all_cards=(*ordinary.deck.all_cards, played_card), removed=(played_card,))
    ordinary = replace(ordinary, deck=deck)
    card, = ordinary.deck.hand
    enhanced = replace(ordinary, hand_support=(VerifiedHandSupport(card.instance_id, card.card_id, ordinary.turn, 2),))
    first, current = request(ordinary), request(enhanced)
    original_history = TurnPlayHistory(ordinary, (played_card,), 'earlier confirmed 818++', played_support=(2,))
    current_history = replace(original_history, state=enhanced)
    reference = build_v2_observation(first, kernel, CONTEXT, original_history)
    raw_shadow = build_shadow_observation(current, kernel, CONTEXT, current_history)
    assert project_v2(raw_shadow).reason_codes == ('v2_816_requires_original_plus_root',)
    result = build_v2_observation(current, kernel, CONTEXT, current_history)
    assert result.request is current and result.observation == reference.observation
    assert result.observation['hand'][0]['u'] == 1 and result.observation['played'][0]['u'] == 2
    assert result.observation['hand'][0]['p'] == int(stamina == 3)
    assert (['play', 0] in result.observation['legal']) == (stamina == 3)
    actual = kernel.legal_actions(enhanced, plan_type=current.plan_type, decision_kind=current.decision_kind)
    assert any(a.card_instance_id == card.instance_id for a in actual)
    assert enhanced.hand_support and current_history.played_support == (2,)
    if stamina == 3:
        bound = bind_mock_output(result, mock_pick(result, ['play', 0]), request=current, kernel=kernel)
        assert (bound.card_instance_id, bound.source_slot) == (card.instance_id, 0)


def test_active_timer_uses_child_id_and_remaining_delay(kernel):  # noqa: F811
    installer, child = 'fixture-install-timer', 'fixture-draw-effect'
    master_effects = {
        installer: dict(id=installer, effectType='ProduceExamEffectType_ExamEffectTimer',
                        effectValue1=2, effectValue2=0, effectCount=1, chainProduceExamEffectId=child),
        child: dict(id=child),
    }
    context = replace(CONTEXT, mappings=replace(CONTEXT.mappings, schedules={
        ('CARD', 'card-0', 0, False): (None, 'timer_not_in_protocol', 'TURN', 2, 1, None, 2, installer),
    }))
    state = isolated('card-0')
    timer = ScheduledEffectRef('isolated', 'card-0', 0, turns_until_fire=1,
        trigger_phase=SchedulePhase.TURN, remaining_uses=1, direct_on_trigger=True)
    source = project(kernel, replace(state, scheduled_effects=(timer,)), context=context)
    assert not project_v2(source).observation
    projection = project_v2(source, exam_effects=master_effects)
    assert projection.observation['timers'] == [[1, child, 0]]
    assert not any(r['local_field'] == 'scheduled_effects.timer' for r in projection.compression_evidence)
    assert any(r['local_field'] == 'scheduled_effects.timer' for r in source.compression_evidence)
    del master_effects[child]
    assert project_v2(source, exam_effects=master_effects).reason_codes == ('v2_timer_requires_verified_installer_and_child',)


@pytest.mark.parametrize(('field', 'reason'), [
    ('nullify_cost_cards', 'v2_costs_source_and_layers_unresolved'),
    ('double_card_effects_remaining', 'v2_pcount_search_and_duration_unresolved')])
def test_aggregate_count_cannot_establish_v2_scope_or_layers(kernel, field, reason):  # noqa: F811
    result = project_v2(project(kernel, replace(initial(), **{field: 1})))
    assert result.reason_codes == (reason,) and result.observation is None and not result.bindings


def test_unknown_fields_and_retrieval_are_not_ordinary_inputs(kernel):  # noqa: F811
    state = initial()
    req = request(state)
    incomplete = build_shadow_observation(req, kernel, CONTEXT, None)
    assert project_v2(incomplete) is incomplete and incomplete.reason_codes == ('played_history_missing',)
    complete = project(kernel)
    damaged = dict(complete.observation)
    del damaged['grave']
    assert project_v2(replace(complete, observation=damaged)).observation is None
    missing_plays = dict(complete.observation, num=dict(complete.observation['num'], plays=None))
    assert project_v2(replace(complete, observation=missing_plays)).observation is None
    changed = dict(complete.observation, legal=list(reversed(complete.observation['legal'])))
    assert project_v2(replace(complete, observation=changed)).reason_codes == ('v2_legal_list_differs_from_bindings',)
    env = HifEnvironment(kernel)
    req = env.reset(state, seed=17)
    history = TurnPlayHistory(req.state, (), 'known opening')
    root = project_v2(build_shadow_observation(req, kernel, CONTEXT, history))
    bottle = bind_mock_output(root, mock_pick(root, ['drink', 0]), request=req, kernel=kernel)
    step = env.step(bottle)
    history = history.advance(bottle, step.request.state, step.events)
    retrieved = project_v2(build_shadow_observation(step.request, kernel, CONTEXT, history))
    assert retrieved.status == 'local_only' and retrieved.observation is None
    with pytest.raises(ShadowMappingError, match='ordinary_state'):
        bind_mock_output(retrieved, {'action': ['end']}, request=step.request, kernel=kernel)
