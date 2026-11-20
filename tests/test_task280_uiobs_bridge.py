from types import SimpleNamespace
from dataclasses import replace

import pytest
from test_task275_gkms_shadow_adapter import CONTEXT, kernel, initial, request  # noqa: F401

from tools import task280_uiobs_bridge as bridge
from tools.task280_gkbench_hosted import _card_key
from tools.task280_local_unclipped import summarize
from tools.task275.gkms_v2_observation import build_v2_observation
from agent.live_decision.high_score.contracts import PlanType, ActionKind, GameAction
from agent.live_decision.high_score.hif_environment import HifEnvironment
from agent.live_decision.high_score.gkms_shadow_adapter import TurnPlayHistory, build_shadow_observation


def public_observation():
    return {
        'v': 'uiobs.v2', 'exam': {'plan': 2}, 'num': {'score': 0}, 'stance': [None, 0],
        'drinks': [], 'hand': [{'id': 'first', 'u': 0}], 'played': [], 'grave': [],
        'deck': [{'id': 'one', 'u': 0}, {'id': 'two', 'u': 1}], 'hold': [], 'lost': [],
        'status': [], 'ench': [], 'timers': [], 'costs': [], 'pcount': [], 'legal': [['play', 0], ['end']],
    }


def test_public_comparison_uses_card_bags_and_rejects_unmapped_search_state():
    projection = public_observation()
    obs = projection | {'deck': list(reversed(projection['deck'])),
                        'legal': list(reversed(projection['legal']))}
    assert bridge.public_differences(obs, projection) == ()
    obs['costs'] = [[None, 0, 2]]
    assert bridge.public_differences(obs, projection) == ('costs',)
    obs.pop('timers')
    assert bridge.public_differences(obs, projection) == ('missing:timers',)


def test_high_upgrade_is_only_clamped_for_planning_identity():
    card = {'id': 'first', 'u': 3, 'c': [['custom', 1]], 'g': ['grow']}
    observed = public_observation() | {'hand': [card]}
    projected = observed | {'hand': [card | {'u': 1}]}
    assert bridge.public_differences(observed, projected) == ()
    assert observed['hand'][0]['u'] == 3
    assert bridge.public_differences(observed, projected | {'hand': [card | {'u': 1, 'c': []}]}) == ('hand',)
    assert bridge.public_differences(observed, projected | {'num': {'score': 1}}) == ('num',)


def test_public_float_status_uses_referee_f32_and_integral_counts_are_exact():
    observed = public_observation() | {'status': [['lesson_parameter_multiple', 0.10000000149011612, 3],
                                                  ['anti_debuff', 1, -1]]}
    projection = observed | {'status': [['anti_debuff', 1, -1], ['lesson_parameter_multiple', 0.1, 3]]}
    assert bridge.public_differences(observed, projection) == ()
    projection['status'][0][1] = 2
    assert bridge.public_differences(observed, projection) == ('status',)


@pytest.mark.parametrize('plan', [3, 4])
def test_non_sense_does_not_enter_sense_search(plan):
    obs = public_observation() | {'exam': {'plan': plan}}
    with pytest.raises(ValueError, match='requires_sense'):
        bridge.bind_request(obs, (), kernel=None, context=None)


def test_bind_current_receipt_and_enforce_budget_without_fallback(monkeypatch):
    obs = public_observation()
    request = SimpleNamespace(plan_type=PlanType.SENSE, state=initial())
    action = GameAction(ActionKind.PLAY_CARD, card_instance_id='first', source_slot=0)
    binding = SimpleNamespace(action=action, token=('play', 0))
    shadow = SimpleNamespace(status='supported', observation=obs, request=request, bindings=(binding,), reason_codes=())
    monkeypatch.setattr(bridge, 'build_v2_observation', lambda *_args, **_kwargs: shadow)
    candidates = [(request, object())]
    assert bridge.bind_request(obs, candidates, kernel=None, context=None) is shadow
    with pytest.raises(ValueError, match='ambiguous'):
        bridge.bind_request(obs, candidates * 2, kernel=None, context=None)
    changed = obs | {'num': {'score': 1}}
    with pytest.raises(ValueError, match='not_bound:num'):
        bridge.bind_request(changed, candidates, kernel=None, context=None)

    def factory(_kernel, *, planning_seed, deadline):
        assert (planning_seed, deadline) == (8, 5.)
        return SimpleNamespace(choose=lambda *_args: action)

    def choose(times):
        return bridge.choose_bound_action(obs, candidates, kernel=None, context=None,
                                          planning_seed=8, deadline=5., clock=iter(times).__next__,
                                          policy_factory=factory)

    assert choose([1., 4.9]) == ['play', 0]
    with pytest.raises(TimeoutError, match='after_search'):
        choose([1., 5.])
    with pytest.raises(TimeoutError, match='before_search'):
        choose([5.])
    assert bridge.choose_bound_action(
        obs, candidates, kernel=None, context=None, planning_seed=8, deadline=None,
        policy_factory=lambda *_args, **_kwargs: SimpleNamespace(choose=lambda *_: action),
    ) == ['play', 0]


def test_duplicate_referee_bottles_do_not_duplicate_strong_search_roots(monkeypatch):
    obs = public_observation() | {'drinks': ['juice', 'juice'], 'legal': [['drink', 0], ['drink', 1]]}
    request = SimpleNamespace(plan_type=PlanType.SENSE, state=initial())
    first = GameAction(ActionKind.USE_DRINK, drink_id='juice', source_slot=0)
    second = GameAction(ActionKind.USE_DRINK, drink_id='juice', source_slot=1)
    bindings = tuple(SimpleNamespace(action=action, token=('drink', index))
                     for index, action in enumerate((first, second)))
    monkeypatch.setattr(bridge, 'build_v2_observation', lambda *_args, **_kwargs:
                        SimpleNamespace(status='supported', observation=obs, request=request, bindings=bindings,
                                        reason_codes=()))

    def choose(_request, roots):
        assert roots == (first,)
        return first

    assert bridge.choose_bound_action(
        obs, [(request, None)], kernel=None, context=None, planning_seed=8, deadline=5.,
        clock=lambda: 0., policy_factory=lambda *_args, **_kwargs: SimpleNamespace(choose=choose),
    ) == ['drink', 0]


@pytest.mark.parametrize('bottle,target_index', [(0, 1), (1, 3)])
def test_successor_consumes_actual_delegated_pick_and_keeps_physical_slots(kernel, bottle, target_index):  # noqa: F811
    state = initial()
    req = request(state)
    history = TurnPlayHistory(state, (), 'complete fixture opening')
    root = build_v2_observation(req, kernel, CONTEXT, history)
    before = root.observation | {'i': 0}
    wire = ['drink', bottle]
    action = root.bind_action(wire, request=req, kernel=kernel)
    pending, = kernel.transition(state, action, plan_type=PlanType.SENSE)
    targets = pending.state.pending_choice.candidate_instance_ids
    selected = GameAction(ActionKind.MOVE_CARD, card_instance_id=targets[target_index])
    successor, = kernel.transition(pending.state, selected, plan_type=PlanType.SENSE)
    next_req = request(successor.state,
                       card_slots=(*req.card_slots, (targets[target_index], len(req.card_slots))),
                       drink_slots=tuple(pair for pair in req.drink_slots if pair[0] != bottle))
    next_history = history.advance(action, successor.state, pending.events + successor.events)
    observed = build_shadow_observation(next_req, kernel, CONTEXT, next_history).observation | {
        'i': 1, 'v': 'uiobs.v2', 'timers': [], 'pcount': [], 'costs': [['p_card_search-deck_all', 0, 2]],
    }
    reply = {'done': False, 'obs': observed, 'result': {'after': {'score': 0}, 'delta': {}}}
    expected = _card_key(before['deck'][0])
    updated, kept_history, retrieval = bridge.bind_successor(
        before, req, history, wire, reply, kernel=kernel, context=CONTEXT, expected_retrieval=expected,
    )
    assert (updated.state, updated.card_slots, updated.drink_slots, kept_history) == (
        next_req.state, next_req.card_slots, next_req.drink_slots, next_history,
    )
    actual = _card_key(observed['hand'][-1])
    assert actual != expected
    assert retrieval == {'status': 'mismatch', 'actual': actual, 'expected': expected}
    with pytest.raises(ValueError, match='stale_or_inconsistent'):
        bridge.bind_successor(before, req, history, wire, reply | {'obs': observed | {'i': 2}},
                              kernel=kernel, context=CONTEXT)
    with pytest.raises(ValueError, match='score_delta'):
        bridge.bind_successor(before, req, history, wire,
                              reply | {'result': {'after': {'score': 0}, 'delta': {'score': 1}}},
                              kernel=kernel, context=CONTEXT)
    with pytest.raises(ValueError, match='not_bound'):
        bridge.bind_successor(before, req, history, wire, reply | {'obs': observed | {'status': []}},
                              kernel=kernel, context=CONTEXT)
    with pytest.raises(ValueError, match='costs_source_and_layers_unresolved'):
        bridge.bind_successor(before, req, history, wire,
                              reply | {'obs': observed | {'costs': [['other_search', 0, 2]]}},
                              kernel=kernel, context=CONTEXT)


@pytest.mark.parametrize('last_turn', [False, True])
def test_successor_binds_play_through_automatic_turn_advance_or_terminal(kernel, last_turn):  # noqa: F811
    state = initial()
    card, draw, *_ = state.deck.draw_pile
    deck = replace(state.deck, all_cards=(card, draw), hand=(card,), draw_pile=(draw,))
    state = replace(state, deck=deck, drinks=(), stamina=5)
    if last_turn:
        state = replace(state, turn=9, turns_remaining=1, turn_attributes=state.turn_attributes[-1:],
                        score_multiplier=state.future_score_multipliers[-1], future_score_multipliers=())
    env = HifEnvironment(kernel)
    req = env.reset(state, seed=861)
    history = TurnPlayHistory(req.state, (), 'fixture with one public draw')
    root = build_v2_observation(req, kernel, CONTEXT, history)
    before = root.observation | {'i': 0}
    wire = next(token for token in before['legal'] if token[0] == 'play')
    action = root.bind_action(wire, request=req, kernel=kernel)
    step = env.step(action)
    score = step.request.state.score
    reply = {'done': step.terminated, 'result': {'after': {'score': score}, 'delta': {'score': score}}}
    if last_turn:
        assert step.terminated
        reply['final'] = {'score': score, 'decisions': 1, 'finished': True}
        assert bridge.bind_successor(before, req, history, wire, reply,
                                     kernel=kernel, context=CONTEXT) == (None, None, None)
        with pytest.raises(ValueError, match='decision_count'):
            bridge.bind_successor(before, req, history, wire, reply | {'final': reply['final'] | {'decisions': 2}},
                                  kernel=kernel, context=CONTEXT)
    else:
        expected_history = history.advance_step(action, step)
        order = bridge.ObservedHandOrder(iid for iid, _ in req.card_slots)
        order.advance(req.state, step)
        next_req = order.request(step.request)
        shadow = build_v2_observation(next_req, kernel, CONTEXT, expected_history)
        reply['obs'] = shadow.observation | {'i': 1}
        updated, kept_history, _ = bridge.bind_successor(
            before, req, history, wire, reply, kernel=kernel, context=CONTEXT,
        )
        assert (updated.state, updated.state.turn, kept_history) == (next_req.state, 2, expected_history)


@pytest.mark.parametrize('terminal,expected_score', [(True, 42), (False, None)])
def test_unclipped_report_keeps_timeouts_but_not_an_incomplete_score(terminal, expected_score):
    report = {'decision_latency_ms': [5000, 5001, 6050, 6051], 'terminal': terminal,
              'observed_score': 42, 'error': None if terminal else {'type': 'TimeoutError'},
              'result': {'decision_count': 4}, 'episode_seconds': 23}
    result = summarize(report, opening='synthetic', arm='strong', environment_seed=1, planning_seed=2)
    assert (result['over_5s'], result['over_6050ms'], result['score']) == (3, 1, expected_score)
