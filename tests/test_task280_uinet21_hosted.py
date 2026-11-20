from types import SimpleNamespace

import pytest

from tools.task280_gkbench_hosted import BLACK_VINEGAR
from tools.task280_uinet21_hosted import play_slot


def test_hosted_network_continues_from_delegated_hand_even_after_slow_step():
    received = {'id': 'received', 'u': 1}
    before = {'i': 0, 'v': 'uiobs.v2', 'exam': {'plan': 2}, 'num': {'score': 0},
              'drinks': [BLACK_VINEGAR], 'hand': [], 'deck': [received], 'grave': [],
              'legal': [['drink', 0]]}
    after = before | {'i': 1, 'drinks': [], 'hand': [received], 'deck': [], 'legal': [['play', 0]]}
    replies = iter([
        {'done': False, 'obs': after, 'result': {'after': {'score': 0}, 'delta': {}}},
        {'done': True, 'result': {'after': {'score': 5}, 'delta': {'score': 5}},
         'final': {'finished': True, 'score': 5, 'decisions': 2, 'over_steps': 1}},
    ])
    observed = []
    submitted = []

    def decide(obs):
        observed.append(obs)
        return {'action': obs['legal'][0], 'probs': [1.]}

    def post(action, latency):
        submitted.append((action, latency))
        return next(replies)

    result = play_slot(before, post, SimpleNamespace(decide=decide), clock=iter([0., 7., 8., 8.5]).__next__)
    assert observed == [before, after]
    assert submitted == [(['drink', 0], 7000.), (['play', 0], 500.)]
    assert (result['over_5s'], result['server_over'], result['score']) == (1, 1, 5)
    assert result['delegated'][0]['status'] == 'unpriced'
    assert result['delegated'][0]['actual'] == ('received', 1, (), ())


def test_uinet21_cannot_use_unpriced_explicit_selection():
    obs = {'i': 0, 'v': 'uiobs.v2', 'exam': {'plan': 2}, 'num': {'score': 0}, 'legal': [['select', [0]]]}
    with pytest.raises(ValueError, match='unsupported_or_stale'):
        play_slot(obs, None, None)
