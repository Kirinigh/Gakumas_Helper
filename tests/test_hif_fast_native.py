"""Native/offline type bridge, original intent, history and restart boundaries."""
import sys
import json
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace

import pytest
from gkms_release_fixtures import ROOT, CONTEXT, Job, Network, load_action_module
from gkms_release_fixtures import continuous_initial as initial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from route.battle_policy import FAST_PROFILE, UINET21_PROFILE  # noqa: E402
from live_decision.high_score import contracts as native  # noqa: E402
from live_decision.high_score.hif import HifSenseKernel  # noqa: E402
from live_decision.high_score.kernel import TransitionBranch  # noqa: E402
from live_decision.high_score.catalog_v2 import compile_drink_catalog, compile_skill_card_catalog  # noqa: E402
from live_decision.high_score.hif_fast_native import HifFastNativePolicy, business_copy  # noqa: E402


@pytest.fixture(scope='module')
def kernel():
    data = ROOT / 'assets/data'
    return HifSenseKernel(compile_skill_card_catalog(data / 'skill_cards.csv',
        plan_type=native.PlanType.SENSE, include_free_card_ids=('23',)).cards,
        compile_drink_catalog(data / 'p_drinks.csv', plan_type=native.PlanType.SENSE).drinks)


def setup(kernel, model, *, profile=FAST_PROFILE):
    state = business_copy(replace(initial(), actions_remaining=2, drinks=()), offline=False)
    bundle = SimpleNamespace(kernel=kernel, state=state, run_id='fast-fixture', revision=1,
        hand_instance_ids=tuple(c.instance_id for c in state.deck.hand), drink_slots=())
    policy = HifFastNativePolicy(bundle, exam_context=CONTEXT,
        source='verified untouched fixture', model=model, profile=profile)
    return bundle, policy, request(state)


def request(state):
    return native.DecisionRequest('fresh-fixture', native.PlanType.SENSE,
        native.DecisionKind.SECONDARY if state.pending_choice else native.DecisionKind.LIVE,
        state, native.CapabilityReport(True, True, True, True, True),
        card_slots=tuple((c.instance_id, i) for i, c in enumerate(state.deck.hand)),
        ruleset_id=HifSenseKernel.ruleset_id if hasattr(HifSenseKernel, 'ruleset_id') else
            'gakumas-hif-sense-v3')


def test_bridge_copies_rules_and_enums_without_executable_objects(kernel):
    from agent.live_decision.high_score.contracts import PlanType
    assert business_copy(native.PlanType.SENSE, offline=True) is PlanType.SENSE
    assert PlanType.SENSE is not native.PlanType.SENSE
    assert business_copy(business_copy(kernel.cards, offline=True), offline=False) == kernel.cards
    with pytest.raises(ValueError, match='not_supported'):
        business_copy(lambda: None, offline=True)


def test_hif_observed_history_reads_upgrades_from_its_actual_catalog(kernel):
    from agent.live_decision.high_score.kernel import RuleEvent
    from agent.live_decision.high_score.contracts import ActionKind, GameAction

    before = initial()
    card = replace(before.deck.hand[0], card_id='54', upgraded=False)
    deck = replace(before.deck, hand=(card, *before.deck.hand[1:]),
                   all_cards=(card, *before.deck.all_cards[1:]))
    state = business_copy(replace(before, deck=deck, drinks=()), offline=False)
    bundle = SimpleNamespace(kernel=kernel, state=state, run_id='fast-fixture', revision=1,
                            hand_instance_ids=tuple(c.instance_id for c in state.deck.hand), drink_slots=())
    host = HifFastNativePolicy(bundle, exam_context=CONTEXT,
                               source='observed upgrade fixture', model=Network(['end']))
    history = host.policy.history
    upgraded = replace(card, card_id='55', upgraded=True)
    after = replace(history.state, deck=replace(deck, hand=(upgraded, *deck.hand[1:]),
                    all_cards=(upgraded, *deck.all_cards[1:])))
    updated = history.advance(GameAction(ActionKind.END_TURN), after,
                              (RuleEvent('UPGRADE_HAND', 'source', detail=card.instance_id),))
    assert updated.state.deck.hand[0] == upgraded


def test_hiro_end_turn_recovery_preserves_history_until_actual_result(kernel):
    model = Network(['play', 0])
    bundle, policy, opening = setup(kernel, model)
    originally_planned = policy.decide(replace(opening, ruleset_id=kernel.ruleset_id))
    assert originally_planned.has_action_intent, originally_planned.reason
    assert originally_planned.action.kind is native.ActionKind.PLAY_CARD
    original_history = policy.policy.history
    recovered = policy.recover_end_turn(bundle.state)
    assert recovered.reason == 'hiro_forced_skip'
    assert recovered.action.kind is native.ActionKind.END_TURN
    assert policy.policy.decisions[-2]['status'] == 'observed_no_effect'
    assert policy.policy.history == original_history
    branches = kernel.transition(bundle.state, recovered.action, plan_type=native.PlanType.SENSE)
    bound = SimpleNamespace(bundle=bundle, response=recovered, committed_event_branches=branches)
    policy.retain_projected_events(bound)
    policy.confirm_observed(bound, branches[0].state, {'accepted': True}, submitted=True)
    assert policy.policy.decisions[-1]['status'] == 'committed'
    assert policy.policy.history.state.turns_remaining < original_history.state.turns_remaining


@pytest.mark.parametrize(('profile', 'version'), [(FAST_PROFILE, 'uiobs.v1'), (UINET21_PROFILE, 'uiobs.v2')])
def test_automatic_card_callback_uses_fast_decision_and_commits_its_history(kernel, monkeypatch, profile, version):
    from live_decision import hif_upstream_policy as bridge
    from live_decision.high_score import hif_fast_native

    from agent.live_decision import hif_gkms_runtime

    model = Network(['play', 0])
    bundle, _, _ = setup(kernel, model)
    monkeypatch.setattr(hif_fast_native, 'observed_exam_context', lambda *_a, **_kw: CONTEXT)
    loader = 'load_uinet21' if profile == UINET21_PROFILE else 'load_model'
    monkeypatch.setattr(hif_gkms_runtime, loader, lambda *_a, **_kw: model)
    other_loader = 'load_model' if profile == UINET21_PROFILE else 'load_uinet21'
    monkeypatch.setattr(hif_gkms_runtime, other_loader, lambda: pytest.fail('selected the wrong model'))
    dispatched, phases = [], []
    context = SimpleNamespace(tasker=SimpleNamespace(controller=SimpleNamespace(post_screencap=lambda: Job('hud'))),
        gkh_hif_information_boundary_execution=True, gkh_hif_opening_state=bundle, gkh_hif_live_context=object(),
        gkh_hif_poll_battle_progress=lambda *_: False, gkh_hif_upstream_ui_sink=lambda _: None,
        gkh_hif_upstream_dispatch_count=lambda: len(dispatched),
        gkh_hif_mcts_cache_sink=lambda candidate: phases.append(candidate._phase))
    box = ([10, 20, 30, 40], 1.)
    monkeypatch.setattr(bridge, '_stable_hand_geometry', lambda *_, **__: ('hud', [box], [0]))

    def submit(ui, *, index, hand_boxes, action):
        assert index == 0 and action['card_instance_id'] == bundle.state.deck.hand[0].instance_id
        for phase in ('select', 'confirm'):
            ui.receipt['inputs'].append(dict(phase=phase, dispatch_index=len(dispatched)))
            dispatched.append(phase)
        return True

    def settle(ctx, params, *, committed_expected, **_):
        ctx.gkh_hif_committed_state = committed_expected
        ctx.gkh_hif_committed_event = dict(accepted=True)
        return ctx.gkh_hif_committed_event

    live = load_action_module()
    opening = dict(event='live_opening_state', accepted=True, core=dict(fields=[0, 0, 0, 0, 0, ()]))
    policy = live._bind_hif_opening_policy(context, opening, profile)
    fast = policy.decision_policy
    monkeypatch.setattr(bridge.UpstreamHifCardUI, 'submit_card_once', submit)
    module = SimpleNamespace(frame_identifier=str, _publish_hif_component=lambda *_: None,
        _observe_hif_opening_state=settle, _persist_hif_mcts_computation=live._persist_hif_mcts_computation,
        _confirm_hif_mcts_observed_commit=live._confirm_hif_mcts_observed_commit)
    event = bridge.run_strong_card_step(module, context, {'hif_policy_profile': fast.profile}, policy, {})
    assert event['accepted'] and event['execution_submit_count'] == 1
    assert dispatched == ['select', 'confirm'] and len(fast.policy.model.calls) == 1
    assert fast.policy.model.calls[0]['v'] == version
    assert phases[-1] == 'observed_successor' and fast._pending is None


@pytest.mark.parametrize('submitted,accepted', [(False,False),(True,False),(True,True)])
def test_native_failure_keeps_original_reason_and_pending_intent(kernel, submitted, accepted):

    m = load_action_module()
    bundle, candidate, req = setup(kernel, Network(['play', 0]))
    candidate.decide(replace(req, ruleset_id=kernel.ruleset_id))
    pending = candidate._pending
    event = dict(accepted=accepted, reason='original_transaction_failure')
    policy = SimpleNamespace(bundle=bundle, decision_policy=candidate)
    if submitted and accepted:
        with pytest.raises(ValueError, match='fast_commit_requires_current_observed_settlement'):
            m._confirm_hif_mcts_observed_commit(SimpleNamespace(), policy, event, submitted=submitted)
    else:
        m._confirm_hif_mcts_observed_commit(SimpleNamespace(), policy, event, submitted=submitted)
        assert event['mcts_commit']['retained'] is False
    assert event['reason'] == 'original_transaction_failure' and candidate._pending == pending


@pytest.mark.parametrize('expired', [False,True])
def test_pending_intent_restores_without_model_rerun_and_rebinds_slot(kernel,expired):
    bundle, policy, req = setup(kernel, Network(['play', 0]))
    req = replace(req, ruleset_id=kernel.ruleset_id)
    decision = policy.decide(req)
    assert decision.has_action_intent, decision.reason
    saved = json.loads(json.dumps(policy.export_search_cache()))
    if expired:
        saved['compute_deadline'] = 1.
    reordered = replace(req.state, deck=replace(req.state.deck, hand=tuple(reversed(req.state.deck.hand))))
    rebound = SimpleNamespace(**{**vars(bundle), 'state': reordered,
        'hand_instance_ids': tuple(c.instance_id for c in reordered.deck.hand), 'revision': 2})
    fresh = HifFastNativePolicy.from_cache(saved, rebound, source_run_id=bundle.run_id,
        source_revision=1, accepted=True, model=Network())
    result = fresh.decide(replace(request(reordered), ruleset_id=kernel.ruleset_id))
    assert result.has_action_intent, result.reason
    assert result.action.card_instance_id == decision.action.card_instance_id
    assert result.action.source_slot == 2
    assert fresh.policy.deadline == saved['compute_deadline']
    assert fresh.policy.model.calls == []
    assert not result.to_dict()['execution_authorized']


@pytest.mark.parametrize('submitted,accepted', [(False,True),(True,False)])
def test_unconfirmed_action_never_advances_history(kernel, submitted, accepted):
    bundle, policy, req = setup(kernel, Network(['play', 0]))
    decision = policy.decide(replace(req, ruleset_id=kernel.ruleset_id))
    before = policy.policy.history
    transaction = SimpleNamespace(bundle=bundle, response=decision, committed_event_branches=())
    with pytest.raises(ValueError, match='actual_submission'):
        policy.confirm_observed(transaction, bundle.state, dict(accepted=accepted), submitted=submitted)
    assert policy.policy.history is before


def test_actual_events_advance_history_and_survive_restart(kernel):
    bundle, policy, req = setup(kernel, Network(['play', 0], ['end']))
    decision = policy.decide(replace(req, ruleset_id=kernel.ruleset_id))
    branches = kernel.transition(req.state, decision.action, plan_type=native.PlanType.SENSE)
    assert len(branches) == 1
    after = branches[0].state
    transaction = SimpleNamespace(bundle=bundle, response=decision, committed_event_branches=branches)
    policy.confirm_observed(transaction, after, dict(accepted=True), submitted=True)
    assert [c.instance_id for c in policy.policy.history.played] == [decision.action.card_instance_id]
    rebound = SimpleNamespace(**{**vars(bundle), 'state': after,
        'hand_instance_ids': tuple(c.instance_id for c in after.deck.hand)})
    saved = json.loads(json.dumps(policy.export_search_cache()))
    saved['compute_deadline'] = 1.
    restored = HifFastNativePolicy.from_cache(saved, rebound,
        source_run_id=bundle.run_id, source_revision=1, accepted=True, model=Network(['end']))
    assert restored.policy.history == policy.policy.history
    assert restored.decide(replace(request(after),ruleset_id=kernel.ruleset_id)).has_action_intent
    assert restored.policy.deadline > saved['compute_deadline']


def test_actual_settlement_without_complete_event_match_is_rejected(kernel):
    bundle, policy, req = setup(kernel, Network(['play', 0]))
    decision = policy.decide(replace(req, ruleset_id=kernel.ruleset_id))
    transaction = SimpleNamespace(bundle=bundle, response=decision,
        committed_event_branches=(TransitionBranch(1., replace(req.state, score=1), ()),))
    with pytest.raises(ValueError, match='complete_rule_events'):
        policy.confirm_observed(transaction, req.state, dict(accepted=True), submitted=True)


def test_final_turn_records_the_observed_score_without_selecting_a_modeled_branch(kernel):
    state = business_copy(replace(initial(), turn=9, turns_remaining=1, drinks=(),
        future_score_multipliers=(), turn_attributes=('visual',)), offline=False)
    bundle = SimpleNamespace(kernel=kernel, state=state, run_id='score-fixture', revision=1,
        hand_instance_ids=tuple(c.instance_id for c in state.deck.hand), drink_slots=())
    policy = HifFastNativePolicy(bundle, exam_context=CONTEXT,
        source='final turn before the first play', model=Network(['end']))
    decision = policy.decide(replace(request(state), ruleset_id=kernel.ruleset_id))
    assert decision.has_action_intent, decision.reason
    assert decision.action.kind is native.ActionKind.END_TURN
    branches = kernel.transition(state, decision.action, plan_type=native.PlanType.SENSE)
    modeled_score = max(b.state.score for b in branches)
    displayed_score = modeled_score + 1
    transaction = SimpleNamespace(bundle=bundle, response=decision, committed_event_branches=branches)
    history = policy.policy.history
    policy.confirm_terminal(transaction, dict(accepted=True, terminal=True,
        run_id=bundle.run_id, revision=1, round_number=2, frame_ids=['a', 'b', 'c'],
        score=displayed_score, expected_scores=[modeled_score]), submitted=True)
    assert policy.policy.decisions[-1]['registered_score'] == displayed_score
    assert policy.policy.history is history and policy.export_search_cache() is None


def test_pending_event_branches_restore_after_dispatch_without_replanning(kernel):
    bundle,policy,req=setup(kernel,Network(['play',0]))
    decision=policy.decide(replace(req,ruleset_id=kernel.ruleset_id))
    branches=kernel.transition(req.state,decision.action,plan_type=native.PlanType.SENSE)
    bound=SimpleNamespace(bundle=bundle,response=decision,committed_event_branches=branches)
    policy.retain_projected_events(bound)
    restored=HifFastNativePolicy.from_cache(json.loads(json.dumps(policy.export_search_cache())),bundle,
        source_run_id=bundle.run_id,source_revision=bundle.revision,accepted=True,model=Network())
    assert restored._pending_branches==branches
    restored._advance_history(req.state,decision.action,branches[0].state,restored._pending_branches)
    assert restored.policy.decisions[-1]['status']=='committed'
    assert restored.policy.model.calls==[]


@pytest.mark.parametrize('profile', [FAST_PROFILE, UINET21_PROFILE])
def test_drink_choice_uses_local_bridge_and_restores_selected_choice_without_resampling(kernel, profile):
    state=business_copy(initial(),offline=False)
    bundle=SimpleNamespace(kernel=kernel,state=state,run_id='choice',revision=1,
        hand_instance_ids=tuple(c.instance_id for c in state.deck.hand),drink_slots=((0,'22'),(1,'22')))
    policy=HifFastNativePolicy(bundle,exam_context=CONTEXT,
        source='verified untouched fixture',model=Network(['drink',1]),profile=profile)
    calls=[]
    class Bridge:
        plans=()
        def choose(self,request,legal):
            calls.append(request)
            return legal[0]
    policy.policy.secondary_factory=lambda *a,**k: Bridge()
    policy.before_secondary=lambda *args: None
    policy.after_secondary=lambda *args: None
    ordinary=replace(request(state),ruleset_id=kernel.ruleset_id,drink_slots=bundle.drink_slots)
    drink=policy.decide(ordinary)
    assert drink.has_action_intent,drink.reason
    pending=kernel.transition(state,drink.action,plan_type=native.PlanType.SENSE)[0].state
    projected=SimpleNamespace(source_bundle=bundle,action=drink.action,pending_state=pending)
    choice_request=replace(request(pending),snapshot_id='fresh-choice',ruleset_id=kernel.ruleset_id,
        card_slots=tuple((iid,i) for i,iid in enumerate(pending.pending_choice.candidate_instance_ids)),
        drink_slots=((0,'22'),))
    chosen=policy.bind_retrieval_choice(projected,choice_request,drink_submitted=True)
    assert chosen.has_action_intent,chosen.reason
    assert chosen.action.kind is native.ActionKind.MOVE_CARD and len(calls)==1
    rebound=SimpleNamespace(**{**vars(bundle),'state':pending})
    payload=json.loads(json.dumps(policy.export_search_cache()))
    different = UINET21_PROFILE if profile == FAST_PROFILE else FAST_PROFILE
    with pytest.raises(ValueError, match='fast_cache_source_or_rules_not_verified'):
        HifFastNativePolicy.from_cache(payload,rebound,source_run_id='choice',source_revision=1,
            accepted=True,model=Network(),profile=different)
    restored=HifFastNativePolicy.from_cache(payload,rebound,
        source_run_id='choice',source_revision=1,accepted=True,model=Network(),profile=profile)
    restored.policy.secondary_factory=lambda *a,**k: pytest.fail('must not resample selected choice')
    again=restored.bind_retrieval_choice(projected,choice_request,drink_submitted=True)
    assert again.action==chosen.action
    branches=kernel.transition(pending,chosen.action,plan_type=native.PlanType.SENSE)
    restored.confirm_observed(None,branches[0].state,dict(accepted=True),submitted=True)
    assert restored.policy.history.state.nullify_cost_cards==2
    assert not restored.policy.model.calls
    if profile == UINET21_PROFILE:
        settled = branches[0].state
        followup = replace(request(settled), snapshot_id='ordinary-after-retrieval',
            ruleset_id=kernel.ruleset_id, drink_slots=((0,'22'),))
        next_bundle = SimpleNamespace(**{**vars(bundle), 'state': settled})
        loaded = HifFastNativePolicy.from_cache(json.loads(json.dumps(restored.export_search_cache())),
            next_bundle, source_run_id='choice', source_revision=1, accepted=True,
            model=Network(['end']), profile=profile)
        decision = loaded.decide(followup)
        assert decision.has_action_intent, decision.reason
        assert loaded.policy.model.calls[0]['costs'] == [['p_card_search-deck_all', 0, 2]]


def test_unfinished_secondary_still_cannot_receive_new_budget(kernel,tmp_path,monkeypatch):
    from live_decision.hif_native import native_search_cache
    from live_decision.hif_native.native_fast_cache import restore_fast_cache

    bundle,policy,_ = setup(kernel,Network())
    payload = policy.export_search_cache()
    payload['compute_deadline'] = 1.
    monkeypatch.setattr(native_search_cache,'_cache_source',lambda *_: ({'id':'unfinished'},payload))
    (tmp_path/'fast-window-unfinished.json').write_text(json.dumps(dict(status='started',deadline=1.,index=1)))
    with pytest.raises(ValueError,match='incomplete_window_requires_remaining_budget_recovery'):
        restore_fast_cache(tmp_path,{},bundle,accepted=True)
    assert policy.policy.model.calls == []
