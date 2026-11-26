"""UIObs v2 mock projection and binding on existing public Sense fixtures."""
from copy import deepcopy
from dataclasses import replace

import pytest
from gkms_release_fixtures import CONTEXT, kernel, initial, project, request, isolated  # noqa: F401

from tools.task275.gkms_v2_observation import bind_mock_output
from agent.live_decision.hif_gkms_runtime import uinet21_master
from agent.live_decision.high_score.contracts import CardRef, ActionKind, SchedulePhase, ScheduledEffectRef, VerifiedHandSupport
from agent.live_decision.high_score.gkms_v2_effects import COST_SEARCH, REPLAY_SEARCH, V2EffectState, effect_sources
from agent.live_decision.high_score.hif_environment import HifEnvironment
from agent.live_decision.high_score.gkms_shadow_adapter import TurnPlayHistory, ShadowMappingError, build_shadow_observation
from agent.live_decision.high_score.gkms_v2_observation import project_v2, build_v2_observation
from agent.live_decision.high_score.gkms_shadow_mappings import SCHEDULES, CARD_FORMS, _TIMER_FORMS


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
    rows = uinet21_master()['ProduceExamEffect']
    master_effects = {row['id']: row for row in rows}
    # This card's second delayed draw started two turns away and has one turn left.
    timer = ScheduledEffectRef('source', '295', 3, turns_until_fire=1,
        trigger_phase=SchedulePhase.TURN, remaining_uses=1, direct_on_trigger=True)
    source = project(kernel, isolated('295', removed=True, scheduled_effects=(timer,)))
    assert not project_v2(source).observation
    projection = project_v2(source, exam_effects=master_effects)
    assert projection.observation['timers'] == [[1, 'e_effect-exam_card_draw-0001', 0]]
    assert not any(r['local_field'] == 'scheduled_effects.timer' for r in projection.compression_evidence)
    assert any(r['local_field'] == 'scheduled_effects.timer' for r in source.compression_evidence)
    del master_effects['e_effect-exam_card_draw-0001']
    assert project_v2(source, exam_effects=master_effects).reason_codes == ('v2_timer_requires_verified_installer_and_child',)


def test_base_hif_memory_card_installs_both_enchants_after_play(kernel):  # noqa: F811
    state = isolated('816')
    card = replace(state.deck.hand[0], card_id='815', upgraded=False)
    ref = ScheduledEffectRef(card.instance_id, card.card_id, 0,
        trigger_phase=SchedulePhase.EVERY_TURN, remaining_uses=2, persistent=True)
    state = replace(state, deck=replace(state.deck, hand=(card,), all_cards=(card, *state.deck.draw_pile)),
                    scheduled_effects=(ref,))
    shadow = project(kernel, state)
    assert shadow.status == 'supported', shadow.reason_codes
    assert shadow.observation['ench'] == []
    projected = project_v2(shadow)
    assert projected.status == 'supported', projected.reason_codes
    assert projected.observation['ench'] == projected.observation['timers'] == []
    assert kernel.cards['815'].persistent_effects == kernel.cards['816'].persistent_effects
    tables = uinet21_master()
    forms = {r['upgradeCount']: r['playEffects'] for r in tables['ProduceCard']
             if r['id'] == 'p_card-01-ido-3_202'}
    assert forms[0] == forms[1]
    env = HifEnvironment(kernel)
    before = env.reset(state, seed=17)
    action = next(a for a in env.legal_actions() if a.kind is ActionKind.PLAY_CARD)
    step = env.step(action)
    history = TurnPlayHistory(before.state, (), 'base memory card transition').advance_step(action, step)
    result = build_v2_observation(step.request, kernel, CONTEXT, history)
    assert result.status == 'supported', result.reason_codes
    assert result.observation['ench'] == [
        ['enchant-p_card-01-ido-3_202-enc02', 2, -1],
        ['enchant-p_card-01-ido-3_202-enc01', -1, 2],
    ]


def test_master_matched_hif_item_keeps_the_independent_card_timer(kernel, monkeypatch):  # noqa: F811
    from agent.live_decision.high_score import gkms_shadow_adapter
    from agent.live_decision.high_score.contracts import EffectSourceKind

    mapping = ('ITEM', '442', 0, True)
    monkeypatch.setattr(gkms_shadow_adapter, 'SCHEDULES',
                        {key: value for key, value in SCHEDULES.items() if key != mapping})
    item = ScheduledEffectRef('item:442', '442', 0, turns_until_fire=0, trigger_phase=SchedulePhase.AFTER_CARD_USED,
        remaining_uses=3, persistent=True, source_kind=EffectSourceKind.ITEM)
    timer = ScheduledEffectRef('source', '295', 3, turns_until_fire=1,
        trigger_phase=SchedulePhase.TURN, remaining_uses=1, direct_on_trigger=True)
    state = isolated('295', removed=True, items=('442',), scheduled_effects=(item, timer))
    effects = {row['id']: row for row in uinet21_master()['ProduceExamEffect']}
    shadow = project(kernel, state)
    assert shadow.status == 'supported', shadow.reason_codes
    result = project_v2(shadow, exam_effects=effects)
    assert result.status == 'supported', result.reason_codes
    assert result.observation['ench'] == [['enchant-p_item_effect_01-3-313-0-enc01', 3, -1]]
    assert result.observation['timers'] == [[1, 'e_effect-exam_card_draw-0001', 0]]
    unknown = replace(state, scheduled_effects=(replace(item, remaining_uses=4), timer))
    assert project(kernel, unknown).status == 'unsupported'


@pytest.mark.parametrize('card_id,_,upgrade,kind,timers', _TIMER_FORMS)
def test_added_timers_match_master_and_actual_card_transition(kernel, card_id, _, upgrade, kind, timers):  # noqa: F811
    tables = uinet21_master()
    master_effects = {row['id']: row for row in tables['ProduceExamEffect']}
    form = CARD_FORMS[card_id]
    master_card, = [row for row in tables['ProduceCard']
                    if (row['id'], row['upgradeCount']) == form[:2]]
    installer_ids = {entry['produceExamEffectId'] for entry in master_card['playEffects']}
    assert upgrade == form[1] and kind == kernel.cards[card_id].cost_kind.value
    source = isolated(card_id)
    env = HifEnvironment(kernel)
    before = env.reset(source, seed=17)
    action = next(a for a in env.legal_actions() if a.kind is ActionKind.PLAY_CARD)
    step = env.step(action)
    history = TurnPlayHistory(before.state, (), 'current timer fixture').advance_step(action, step)
    projection = build_v2_observation(step.request, kernel, CONTEXT, history, exam_effects=master_effects)
    assert projection.status == 'supported', projection.reason_codes
    expected = []
    for index, delay, child in timers:
        assert SCHEDULES['CARD', card_id, index, False][-1] in installer_ids
        expected.append([delay, 'e_effect-exam_' + child, 0])
    assert projection.observation['timers'] == expected


@pytest.mark.parametrize('card_id', ['162', '163'])
def test_confirmed_replay_rule_uses_master_scope_and_clears_after_consumption(kernel, card_id):  # noqa: F811
    env = HifEnvironment(kernel)
    # A second eligible card remains in hand so the real rule consumes the layer.
    source = isolated(card_id)
    extra = CardRef('eligible', '23')
    source = replace(source, deck=replace(source.deck, all_cards=(*source.deck.all_cards, extra),
        hand=(*source.deck.hand, extra)))
    before = env.reset(source, seed=17)
    history = TurnPlayHistory(before.state, (), 'current replay fixture')
    sources = effect_sources(uinet21_master())
    book = V2EffectState.initial(before.state)
    action = next(a for a in env.legal_actions() if a.card_instance_id == 'source')
    step = env.step(action)
    history = history.advance_step(action, step)
    book = book.advance(before.state, action, step.request.state, step.events, sources)
    observation = build_v2_observation(step.request, kernel, CONTEXT, history, effects=book).observation
    assert observation['pcount'] == [[REPLAY_SEARCH, 1, 1, -1]]
    after = step.request
    consume = next(a for a in env.legal_actions() if a.card_instance_id == extra.instance_id)
    step = env.step(consume)
    new_book = book.advance(after.state, consume, step.request.state, step.events, sources)
    assert new_book.replay == () and step.request.state.double_card_effects_remaining == 0


def test_confirmed_vinegar_cost_lifetime_never_guesses_overlapping_layers(kernel):  # noqa: F811
    from agent.live_decision.high_score.kernel import RuleEvent
    from agent.live_decision.high_score.contracts import GameAction

    sources = effect_sources(uinet21_master())
    start = initial()
    drink = GameAction(ActionKind.USE_DRINK, drink_id='22', source_slot=1)
    events = (RuleEvent('CONSUME_DRINK', 'drink:22'), RuleEvent('NULLIFY_COST_CARDS', 'drink:22', 2))
    doubled = replace(start, nullify_cost_cards=2)
    book = V2EffectState.initial(start).advance(start, drink, doubled, events, sources)
    assert book.cost == (COST_SEARCH, 0, 2)
    overlapping = book.advance(doubled, drink, replace(doubled, nullify_cost_cards=4), events, sources)
    assert overlapping.cost is None
    play = GameAction(ActionKind.PLAY_CARD, card_instance_id=start.deck.hand[0].instance_id)
    spend = (RuleEvent('NULLIFY_CARD_COST', play.card_instance_id, 1),)
    once = replace(doubled, nullify_cost_cards=1)
    book = book.advance(doubled, play, once, spend, sources)
    assert book.cost == (COST_SEARCH, 0, 1)
    assert book.advance(once, play, start, spend, sources).cost == ()


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
