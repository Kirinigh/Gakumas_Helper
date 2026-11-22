"""Offline protocol integration with synthetic stand-ins for all three profiles."""

import time
from types import SimpleNamespace
from dataclasses import replace

import pytest
from test_task275_gkms_shadow_adapter import CONTEXT, kernel, initial, isolated  # noqa: F401

from tools.task285_cases import Case
from tools.task280_decision_box import open_decision_box
from tools.task280_local_unclipped import run_local_case
from agent.live_decision.high_score.contracts import ActionKind
from agent.live_decision.high_score.hif_environment import HifEnvironment
from agent.live_decision.high_score.gkms_shadow_adapter import TurnPlayHistory


def _last_turn(state, *, actions=1):
    return replace(state, turn=9, turns_remaining=1, future_score_multipliers=(),
                   turn_attributes=('visual',), actions_remaining=actions)


def _choice_observation(observation, token):
    return {'action': token, 'probs': [float(action == token) for action in observation['legal']]}


@pytest.mark.parametrize('profile', ['uinet-1.1', 'uinet-2.1', 'strong'])
def test_profiles_run_the_same_completed_local_episode(kernel, profile):  # noqa: F811
    case = Case('synthetic-decision-box', 'synthetic', (), kernel, _last_turn(isolated('card-0'), actions=2), (), ())
    seen = []
    deadline = time.perf_counter() + 10.

    def inference(observation):
        seen.append(observation['v'])
        return _choice_observation(observation, observation['legal'][0])

    def strong_factory(rules, *, planning_seed, deadline):
        assert rules is kernel and planning_seed == 106
        return SimpleNamespace(choose=lambda _request, legal: legal[0],
                               committed=lambda _before, _action, after: seen.append(after.terminal))

    def factory(case, opening, *, planning_seed, deadline):
        options = ({'strong_factory': strong_factory} if profile == 'strong' else
                   {'model': SimpleNamespace(decide=inference), 'context': CONTEXT,
                    'history': TurnPlayHistory(opening.state, (), 'synthetic opening')})
        return open_decision_box(profile, kernel=case.kernel, opening=opening,
                                 planning_seed=planning_seed, deadline=deadline, **options)

    report = run_local_case(case, factory, arm=profile, environment_seed=645, planning_seed=106, deadline=deadline)
    assert report['summary']['complete'] and report['summary']['score'] == 12
    assert not report['episode']['result']['illegal_action']
    assert [row['action']['kind'] for row in report['episode']['actions']] == [ActionKind.PLAY_CARD, ActionKind.END_TURN]
    expected = {'uinet-1.1': ['uiobs.v1'] * 2, 'uinet-2.1': ['uiobs.v2'] * 2, 'strong': [False, True]}
    assert seen == expected[profile]


def test_stale_and_invalid_feedback_leave_the_pending_step_unconsumed(kernel):  # noqa: F811
    environment = HifEnvironment(kernel)
    opening = environment.reset(_last_turn(isolated('card-0'), actions=2), seed=645)
    seen = []
    options = dict(kernel=kernel, opening=opening, planning_seed=106, deadline=time.perf_counter() + 10.)
    with pytest.raises(ValueError, match='explicit_strong_factory_required'):
        open_decision_box('strong', **options)
    with pytest.raises(ValueError, match='explicit_network_model_required'):
        open_decision_box('uinet-2.1', **options)
    chooser = SimpleNamespace(choose=lambda _request, legal: legal[0],
                              committed=lambda _before, action, _after: seen.append(action))
    box = open_decision_box('strong', **options, strong_factory=lambda *_args, **_kwargs: chooser)
    legal = environment.legal_actions()
    with pytest.raises(ValueError, match='snapshot_not_ready'):
        box.choose(replace(opening, snapshot_id='stale'), legal)
    selected = box.choose(opening, legal)
    with pytest.raises(ValueError, match='previous_step_not_confirmed'):
        box.choose(opening, legal)
    step = environment.step(selected)
    with pytest.raises(ValueError, match='feedback_inconsistent'):
        box.observe_transition(opening.state, selected, replace(step, reward=step.reward + 1))
    assert not seen
    box.observe_transition(opening.state, selected, step)
    assert seen == [selected]
    with pytest.raises(ValueError, match='feedback_does_not_match_intent'):
        box.observe_transition(opening.state, selected, step)
    assert box.choose(step.request, environment.legal_actions()).kind is ActionKind.END_TURN


@pytest.mark.parametrize('with_secondary', [False, True])
def test_retrieval_uses_only_an_explicit_secondary_factory(kernel, with_secondary):  # noqa: F811
    case = Case('synthetic-retrieval-box', 'synthetic', (), kernel, _last_turn(initial()), ((1, '22'), (3, '22')), ())
    observations, secondary = [], []

    def inference(observation):
        observations.append(observation['v'])
        token = ['drink', 1] if len(observations) == 1 else ['play', 0]
        return _choice_observation(observation, token)

    def local_plan(rules, *, planning_seed, deadline):
        assert rules is kernel and planning_seed == 1000107
        return SimpleNamespace(choose=lambda _request, legal: legal[0],
                               committed=lambda _before, action, _after: secondary.append(action.kind))

    def factory(case, opening, *, planning_seed, deadline):
        return open_decision_box('uinet-1.1', kernel=case.kernel, opening=opening, planning_seed=planning_seed,
                                 deadline=deadline, model=SimpleNamespace(decide=inference), context=CONTEXT,
                                 history=TurnPlayHistory(opening.state, (), 'synthetic retrieval'),
                                 secondary_factory=local_plan if with_secondary else None)

    report = run_local_case(case, factory, arm='uinet-1.1', environment_seed=645, planning_seed=106,
                            deadline=time.perf_counter() + 10.)
    episode = report['episode']
    assert episode['actions'][0]['action']['source_slot'] == 3
    if with_secondary:
        assert episode['terminal'] and not episode['result']['illegal_action']
        assert observations == ['uiobs.v1', 'uiobs.v1']
        assert secondary == [ActionKind.MOVE_CARD]
    else:
        assert not episode['terminal']
        assert episode['error']['message'] == 'offline_secondary_policy_required'
        assert observations == ['uiobs.v1'] and not secondary
