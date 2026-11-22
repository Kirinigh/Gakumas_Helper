"""Fast-policy history recovery after the existing native state verification."""
import os
import json
import time
from dataclasses import replace

from route.battle_policy import FAST_PROFILE


def restore_fast_retrieval(runtime,source,projected):
    from live_decision.high_score.hif_mcts_cache import exact_state
    from live_decision.high_score.hif_fast_native import HifFastNativePolicy

    from .native_search_cache import _cache_source
    _,payload=_cache_source(runtime,source)
    if payload is None:
        raise ValueError('fast_retrieval_original_history_missing')
    profile = source.get('formal_params', {}).get('hif_policy_profile', FAST_PROFILE)
    HifFastNativePolicy.require_cache_version(payload, profile)
    state=exact_state(payload['history']['state'])
    canonical=projected.source_bundle.kernel._canonical_probability_state
    if canonical(state) not in (canonical(projected.source_bundle.state),canonical(projected.pending_state)):
        raise ValueError('fast_retrieval_history_outside_actual_drink_boundary')
    bundle=replace(projected.source_bundle,state=state,hand_instance_ids=tuple(r.instance_id for r in state.deck.hand))
    return restore_fast_cache(runtime,source,bundle,accepted=True)


def begin_secondary(runtime, result, policy, request):
    if __package__:
        from .native_search_cache import persist_search_cache
    else:
        from .native_search_cache import persist_search_cache
    current = result.get('fast_secondary_window')
    if current and current['status'] != 'complete':
        raise ValueError('fast_secondary_prior_window_unfinished')
    started = time.perf_counter()
    result['fast_secondary_window'] = dict(status='started', index=policy.policy._secondary_calls+1,
        simulations=256, minimum_root_samples=4, started=started,
        deadline=min(started+300.,policy.policy.deadline), turn=request.state.turn)
    persist_search_cache(runtime,result,policy)
    path=runtime/f"fast-window-{result['id']}.json"
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(result['fast_secondary_window'],stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finish_secondary(runtime, result, policy):
    window=result.get('fast_secondary_window')
    if window and window['status']=='started' and policy._retrieval is not None:
        if time.perf_counter() > window['deadline']:
            raise ValueError('fast_secondary_window_budget_exceeded')
        window.update(status='complete',elapsed_ms=(time.perf_counter()-window['started'])*1000)
        path=runtime/f"fast-window-{result['id']}.json"
        temporary=path.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8') as stream:
            json.dump(window,stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def restore_fast_cache(runtime, source, bundle, *, before=None, action=None, submitted=False, accepted=False,
                       round_context=None):
    from live_decision.high_score.hif_mcts_cache import exact_state
    from live_decision.high_score.hif_fast_native import HifFastNativePolicy

    from .native_search_cache import _cache_source

    profile = source.get('formal_params', {}).get('hif_policy_profile', FAST_PROFILE)
    owner,payload=_cache_source(runtime,source)
    if payload is None:
        openings=[e for e in source.get('live_events',()) if e.get('event')=='live_opening_state'
            and e.get('accepted') is True]
        if (accepted is not True or len(openings)!=1 or bundle.state.turn!=1 or bundle.state.score!=0
                or any(e.get('execution_submit_count',0) or e.get('end_turn_submit_count',0)
                    for e in source.get('live_events',()))):
            raise ValueError('fast_native_complete_history_cache_missing')
        from types import SimpleNamespace

        from live_decision.high_score.hif_fast_native import observed_exam_context
        policy=HifFastNativePolicy(bundle,exam_context=observed_exam_context(bundle,
            SimpleNamespace(**source['observed_live_context']), round_context=round_context),
            source=source['id']+'#zero-play-opening', profile=profile)
        return policy,dict(restored=False,initialized=True,profile=profile,source=source['id'])
    HifFastNativePolicy.require_cache_version(payload, profile)
    corrected_branches = None
    correction = source.get('verified_pending_support_receipt')
    if correction is not None and payload.get('phase') == 'observed_successor':
        from types import SimpleNamespace

        from live_decision.high_score.live_policy import BoundHifOpeningPolicy
        from live_decision.high_score.support_schedule import FOUR_TURN_CARD_ID

        if (accepted is not True or submitted is not True or before is None or action is None
                or owner['id'] != source['id'] or correction.get('definition_id') != FOUR_TURN_CARD_ID):
            raise ValueError('fast_support_correction_requires_verified_original_action')
        original_bundle = replace(bundle, state=before,
            hand_instance_ids=tuple(ref.instance_id for ref in before.deck.hand))
        projection = BoundHifOpeningPolicy(original_bundle, action_order=())
        projection.response = SimpleNamespace(action=action)
        projection.bind_selected_skill_support(correction)
        expected = projection.committed_transition()
        if bundle.kernel._canonical_probability_state(expected) != bundle.kernel._canonical_probability_state(bundle.state):
            raise ValueError('fast_support_correction_differs_from_actual_settlement')
        corrected_branches = projection.committed_event_branches
        # Reuse the saved pre-submit intention and its complete history. The
        # old estimated successor remains an unchanged diagnostic artifact.
        pending_reference = dict(owner['mcts_search_cache'], phase='pending_intent',
            file=f"search-cache-{owner['id']}-pending_intent.json")
        owner,payload = _cache_source(runtime, dict(owner, mcts_search_cache=pending_reference))
        HifFastNativePolicy.require_cache_version(payload, profile)
    window=runtime/f"fast-window-{owner['id']}.json"
    if window.is_file() and json.loads(window.read_text(encoding='utf-8'))['status'] != 'complete':
        raise ValueError('fast_secondary_incomplete_window_requires_remaining_budget_recovery')
    if window.is_file() and payload['secondary_calls'] < json.loads(window.read_text(encoding='utf-8'))['index']:
        raise ValueError('fast_secondary_completed_cache_missing')
    saved=exact_state(payload['history']['state'])
    rebound=replace(bundle,state=saved,hand_instance_ids=tuple(ref.instance_id for ref in saved.deck.hand))
    policy=HifFastNativePolicy.from_cache(payload,rebound,source_run_id=payload['run_id'],
        source_revision=payload['revision'],accepted=accepted,profile=profile)
    canonical=bundle.kernel._canonical_probability_state
    if canonical(saved)!=canonical(bundle.state):
        if submitted is not True or policy._pending is None or action is None:
            raise ValueError('fast_history_advance_requires_verified_actual_action')
        if before is not None and canonical(before)!=canonical(saved):
            raise ValueError('fast_history_original_state_changed')
        policy._advance_history(saved,action,bundle.state,
            corrected_branches if corrected_branches is not None else policy._pending_branches)
        policy._pending_branches=()
    return policy,dict(restored=True,profile=profile,phase=policy._phase,
        source=owner['id'],complete_history=True,secondary_calls=policy.policy._secondary_calls,
        **({'original_pending_replayed_for_support_correction': True} if corrected_branches is not None else {}))
