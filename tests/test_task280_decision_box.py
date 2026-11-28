import time
from types import SimpleNamespace
from dataclasses import replace

import pytest
from test_task275_gkms_shadow_adapter import CONTEXT, kernel, isolated  # noqa: F401

from tools import task280_decision_box as box
from agent.live_decision import hif_gkms_runtime
from agent.live_decision.high_score import gkms_continuous_policy
from agent.live_decision.high_score.contracts import ActionKind, GameAction
from agent.live_decision.high_score.hif_environment import HifEnvironment
from agent.live_decision.high_score.gkms_shadow_adapter import TurnPlayHistory


@pytest.mark.parametrize('profile', ['uinet-1.1', 'uinet-2.1', 'strong'])
def test_switch_algorithms_with_same_step_and_feedback_contract(kernel, monkeypatch, profile):  # noqa: F811
    env = HifEnvironment(kernel)
    opening = env.reset(replace(isolated('87', turn=9), actions_remaining=2), seed=645)
    history = TurnPlayHistory(opening.state, (), 'complete fixture history')
    seen = []

    def inference(obs):
        seen.append(obs['v'])
        return {'action': obs['legal'][0], 'probs': [1.] + [0.] * (len(obs['legal']) - 1)}

    def v2_loader(path, *, model_sha256, native_rules):
        assert (path, model_sha256) == ('second-checkpoint.onnx', 'checkpoint-digest')
        assert not native_rules
        return SimpleNamespace(decide=inference)

    strong = SimpleNamespace(choose=lambda _request, legal: legal[0],
                             committed=lambda _before, _action, after: seen.append(after.terminal))
    monkeypatch.setattr(hif_gkms_runtime, 'load_model', lambda: SimpleNamespace(decide=inference))
    monkeypatch.setattr(hif_gkms_runtime, 'load_uinet21', v2_loader)
    monkeypatch.setattr(hif_gkms_runtime, 'uinet21_master', lambda: {
        'ProduceCard': [], 'ProduceCardSearch': [],
        'ProduceDrink': [{'id': 'pdrink_01-3-012', 'produceDrinkEffectIds': []}],
        'ProduceDrinkEffect': [], 'ProduceExamEffect': [],
    })
    monkeypatch.setattr(gkms_continuous_policy, 'strong_policy', lambda *_args, **_kwargs: strong)
    weight = dict(model_path='second-checkpoint.onnx', model_sha256='checkpoint-digest') if profile == 'uinet-2.1' else {}
    policy = box.open_decision_box(profile, kernel=kernel, opening=opening, context=CONTEXT, history=history,
                                   planning_seed=106, deadline=time.perf_counter() + 60., **weight)
    legal = env.legal_actions()
    with pytest.raises(ValueError, match='snapshot_not_ready'):
        policy.choose(replace(opening, snapshot_id='stale'), legal)
    selected = policy.choose(opening, legal)
    assert selected == GameAction(ActionKind.PLAY_CARD, card_instance_id='source', source_slot=0)
    with pytest.raises(ValueError, match='previous_step_not_confirmed'):
        policy.choose(opening, legal)
    step = env.step(selected)
    with pytest.raises(ValueError, match='feedback_inconsistent'):
        policy.observe_transition(opening.state, selected, replace(step, reward=step.reward + 1))
    policy.observe_transition(opening.state, selected, step)
    assert not step.terminated
    next_request = step.request
    end = policy.choose(next_request, env.legal_actions())
    assert end.kind is ActionKind.END_TURN
    finished = env.step(end)
    policy.observe_transition(next_request.state, end, finished)
    assert finished.terminated
    expected = {'uinet-1.1': ['uiobs.v1'] * 2, 'uinet-2.1': ['uiobs.v2'] * 2, 'strong': [False, True]}
    assert seen == expected[profile]


def test_alternate_checkpoint_requires_its_frozen_digest(tmp_path):
    checkpoint = tmp_path / 'alternate.onnx'
    checkpoint.write_bytes(b'incomplete model')
    with pytest.raises(ValueError, match='unexpected_uinet21_weight'):
        box.load_uinet21(checkpoint)


def test_single_shared_registration_accepts_a_choose_only_strategy(kernel, monkeypatch):  # noqa: F811
    from agent.live_decision.high_score import policy_factory
    from agent.live_decision.high_score.contracts import BattleStep

    env = HifEnvironment(kernel)
    opening = env.reset(replace(isolated('87', turn=9), actions_remaining=1), seed=645)
    received = []

    def strategy(request):
        received.append(request)
        return SimpleNamespace(choose=lambda _, legal: legal[0])

    monkeypatch.setitem(policy_factory._PROVIDERS, 'test-shared', {'shared': strategy})
    session = box.open_decision_box('test-shared', kernel=kernel, opening=opening,
                                    planning_seed=1, deadline=time.perf_counter() + 10.)
    action = session.choose(opening, env.legal_actions())
    step = env.step(action)
    session.observe_transition(opening.state, action, step)
    assert received == [opening] and isinstance(step, BattleStep) and session.request == step.request
