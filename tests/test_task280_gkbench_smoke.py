from types import SimpleNamespace

import pytest

from tools.task280_gkbench_smoke import verify_final, encode_action, action_bindings, synthetic_smoke
from tools.task280_gkbench_hosted import BLACK_VINEGAR, play_diagnostic, delegated_retrieval, reconcile_local_retrieval
from agent.live_decision.high_score.contracts import CardRef, ActionKind, GameAction, PendingChoice
from agent.live_decision.high_score.gkms_shadow_mappings import ShadowMappings


def test_synthetic_replays_from_new_environment():
    result = synthetic_smoke(280)
    assert result == {
        'scope': 'synthetic_offline_replay', 'score': 90,
        'actions': {'play': 2}, 'hosted_referee_verified': False,
    }


def test_wire_positions_follow_referee_order_not_local_slots():
    request = SimpleNamespace(card_slots=(('copy-a', 0), ('copy-b', 1)), drink_slots=((0, '22'), (3, '22')),
                              state=SimpleNamespace(pending_choice=None))
    play = GameAction(ActionKind.PLAY_CARD, card_instance_id='copy-b', source_slot=1)
    drink = GameAction(ActionKind.USE_DRINK, drink_id='22', source_slot=0)
    end = GameAction(ActionKind.END_TURN)
    bindings = action_bindings(request, (play, drink, end),
                               ordered_instances=('copy-b', 'copy-a'), ordered_bottle_slots=(0, 3))
    server_legal = [['play', 0], ['drink', 0], ['drink', 1], ['end']]
    assert encode_action(play, bindings, server_legal) == ['play', 0]
    assert encode_action(bindings[('drink', 1)], bindings, server_legal) == ['drink', 1]
    assert bindings[('drink', 1)].source_slot == 3
    assert encode_action(end, bindings, server_legal) == ['end']
    with pytest.raises(ValueError, match='not_advertised'):
        encode_action(play, bindings, [['play', 1]])
    with pytest.raises(ValueError, match='not_uniquely_bound'):
        action_bindings(request, (play,), ordered_instances=('copy-b', 'copy-b'), ordered_bottle_slots=(0, 3))


def test_retrieval_is_nested_select_and_requires_validated_order():
    pending = PendingChoice(ActionKind.MOVE_CARD, '22', ('deck-a', 'deck-b'))
    request = SimpleNamespace(card_slots=(('deck-a', 0), ('deck-b', 1)), drink_slots=(),
                              state=SimpleNamespace(pending_choice=pending))
    retrieve = GameAction(ActionKind.MOVE_CARD, card_instance_id='deck-a', source_slot=0)
    bindings = action_bindings(request, (retrieve,), ordered_instances=('deck-b', 'deck-a'), ordered_bottle_slots=())
    assert encode_action(retrieve, bindings, [['select', [1]]]) == ['select', [1]]
    with pytest.raises(ValueError, match='not_advertised'):
        encode_action(retrieve, bindings, [['select', 1], ['end']])


def test_hosted_v65_black_vinegar_without_selection_cannot_bind_retrieval():
    pending = PendingChoice(ActionKind.MOVE_CARD, '22', ('deck-a', 'deck-b'))
    request = SimpleNamespace(card_slots=(('deck-a', 0), ('deck-b', 1)), drink_slots=(),
                              state=SimpleNamespace(pending_choice=pending))
    retrieve = GameAction(ActionKind.MOVE_CARD, card_instance_id='deck-b', source_slot=1)
    bindings = action_bindings(request, (retrieve,), ordered_instances=('deck-a', 'deck-b'), ordered_bottle_slots=())
    advertised = [['play', 0], ['play', 1], ['play', 2], ['play', 3], ['drink', 0], ['drink', 1], ['end']]
    with pytest.raises(ValueError, match='not_advertised'):
        encode_action(retrieve, bindings, advertised)


@pytest.mark.parametrize('change,reason', [({'finished': False}, 'did_not_finish'),
                                           ({'score': 1001}, 'score_mismatch'),
                                           ({'decisions': 2}, 'decision_count_mismatch')])
def test_terminal_mismatch_is_not_a_score(change, reason):
    final = {'finished': True, 'score': 1000, 'decisions': 3}
    verify_final(final, score=1000, decisions=3)
    with pytest.raises(ValueError, match=reason):
        verify_final(final | change, score=1000, decisions=3)


def test_hosted_delegation_mismatch_follows_the_real_hand_through_terminal():
    wanted = {'id': 'want', 'u': 1}
    received = {'id': 'received', 'u': 0}
    before = {'i': 0, 'num': {'score': 0}, 'hand': [], 'deck': [wanted, received],
              'grave': [], 'drinks': [BLACK_VINEGAR], 'legal': [['drink', 0]]}
    after = before | {'i': 1, 'hand': [received], 'deck': [wanted], 'drinks': [], 'legal': [['play', 0]]}
    replies = iter([
        {'done': False, 'obs': after, 'result': {'after': {'score': 0}, 'delta': {}}},
        {'done': True, 'final': {'finished': True, 'score': 17, 'decisions': 2},
         'result': {'after': {'score': 17}, 'delta': {'score': 17}}},
    ])
    submitted = []

    def post(action, _elapsed):
        submitted.append(action)
        return next(replies)

    result = play_diagnostic(before, post, expected_pick=lambda _obs: ('want', 1, (), ()))
    assert result['retrieval'] == {'expected': ('want', 1, (), ()), 'actual': ('received', 0, (), ()),
                                   'status': 'mismatch'}
    assert result['final']['finished']
    assert submitted == [['drink', 0], ['play', 0]]


def test_delegation_does_not_guess_an_ambiguous_received_target():
    card = {'id': 'possible', 'u': 0}
    before = {'hand': [], 'deck': [card], 'grave': [], 'legal': []}
    after = {'hand': [card, card], 'deck': [], 'grave': [], 'legal': [['end']]}
    assert delegated_retrieval(before, after)['status'] == 'unverified'
    assert delegated_retrieval(before, before | {'legal': [['select', [0]]]})['status'] == 'explicit_selection'


def test_local_reconciliation_uses_the_received_instance_not_the_shadow_plan():
    wanted = CardRef('want', 'wanted-local', upgraded=True)
    received = CardRef('received', 'received-local', upgraded=True)
    pending = PendingChoice(ActionKind.MOVE_CARD, '22', ('want', 'received'))
    pending_state = SimpleNamespace(pending_choice=pending, deck=SimpleNamespace(draw_pile=(wanted, received),
                                                                                discard=()))
    kind = SimpleNamespace(probability=1.0, state=pending_state)
    settled = SimpleNamespace(pending_choice=None, deck=SimpleNamespace(hand=(received,)))
    drink = GameAction(ActionKind.USE_DRINK, drink_id='22', source_slot=0)
    expected = GameAction(ActionKind.MOVE_CARD, card_instance_id='want')
    actual = GameAction(ActionKind.MOVE_CARD, card_instance_id='received')
    actions = []

    def transition(_state, action, *, plan_type):
        actions.append(action)
        if action == drink:
            return (kind,)
        assert action == actual
        return (SimpleNamespace(probability=1.0, state=settled),)

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Request:
        state: object
        plan_type: object = None
        drink_slots: tuple = ((0, '22'), (2, '20'))
        card_slots: tuple = ()

        def validate(self):
            assert self.state is settled and self.drink_slots == ((2, '20'),)

    mappings = ShadowMappings(
        card_forms={
            'wanted-local': ('wanted-model', 1, 'NORMAL'),
            'received-local': ('received-model', 1, 'NORMAL'),
        },
        card_upgrades={}, custom_forms={}, drink_ids={}, schedules={}, grow_effects={},
    )
    receipt = {'actual': ('received-model', 2, (), ()), 'status': 'mismatch'}
    request, used, mismatch = reconcile_local_retrieval(
        SimpleNamespace(transition=transition), Request(object()), drink, expected, receipt, mappings=mappings)
    assert actions == [drink, actual]
    assert request.card_slots == (('received', 0),)
    assert (used, mismatch) == (actual, True)
    assert receipt['actual'][1] == 2
