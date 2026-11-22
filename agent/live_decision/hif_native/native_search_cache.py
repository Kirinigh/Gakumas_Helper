"""Optional computation sidecars after the native business source is verified.

Nothing here restores observation authority, an input deadline or an intent.
The existing native recovery loaders and fresh history remain the state source.
"""
import os
import re
import json
import time

from route.battle_policy import NETWORK_PROFILES

from . import diagnostic_depth_reached

_DEFAULT_PROFILE = 'hif-mcts-v22-trial'
_STRONG_PROFILE = 'hif-strong-reference-trial-v1'


def _configuration(profile):
    if profile == _DEFAULT_PROFILE:
        from live_decision.high_score.hif_mcts_candidate import CONFIG

        return CONFIG
    if profile == _STRONG_PROFILE:
        from live_decision.high_score.hif_strong_reference import configuration

        return configuration()
    raise ValueError('unsupported native computation profile')


def persist_search_cache(runtime, result, candidate):
    payload = candidate.export_search_cache()
    if payload is None:
        return None
    name = result['id']
    if re.fullmatch(r'[0-9]+-[a-z0-9-]+', name) is None:
        raise ValueError('invalid native cache record')
    phase = payload['phase']
    if phase not in ('pending_intent', 'observed_successor'):
        raise ValueError('invalid native cache phase')
    path = runtime / f'search-cache-{name}-{phase}.json'
    started = time.perf_counter()
    saved = dict(record_id=name, cache=payload)
    strong = payload.get('profile') == _STRONG_PROFILE
    if strong:
        from .native_search_window import validate_persist_window

        saved['computation_origin'] = validate_persist_window(runtime, result, candidate, payload)
    body = json.dumps(saved, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)
    temporary = path.with_suffix('.tmp')
    if strong or payload.get('profile') in NETWORK_PROFILES:
        with temporary.open('w', encoding='utf8') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary.write_text(body, encoding='utf8')
    temporary.replace(path)
    reference = dict(record_id=name, file=path.name, phase=phase,
                     run_id=payload['run_id'], revision=payload['revision'],
                     bytes=path.stat().st_size,
                     write_elapsed_ms=(time.perf_counter()-started)*1000)
    if strong:
        from .native_search_window import complete_window

        complete_window(runtime, result, candidate, payload, path.name)
        reference['computation_origin'] = saved['computation_origin']
    result['mcts_search_cache'] = reference
    return reference


def _cache_source(runtime, source, *, research_recompute=None, business_successor=None):
    """Only follow recovery of the same action, never skip later decisions.

    A completed newer source can validate a historical business predecessor;
    restoration never supplies that exception and still requires the latest
    completed window. No unfinished or missing window can be bypassed.
    """
    from .native_settlement_resume import load_result

    seen = set()
    while not source.get('mcts_search_cache'):
        name = source['id']
        if name in seen or diagnostic_depth_reached(seen):
            raise ValueError('native cache recovery chain is cyclic or unbounded')
        seen.add(name)
        parents = [source[key] for key in ('native_pending_source', 'native_selected_source',
                                         'native_choice_source', 'native_pending_restart_source')
                   if source.get(key)]
        if not parents:
            return source, None
        if len(parents) != 1:
            raise ValueError('native cache recovery source is ambiguous')
        source = load_result(runtime, parents[0], allow_closed_transaction=True)
    reference = source['mcts_search_cache']
    name, phase = source['id'], reference.get('phase')
    if (re.fullmatch(r'[0-9]+-[a-z0-9-]+', name) is None
            or phase not in ('pending_intent', 'observed_successor')
            or reference.get('record_id') != name
            or reference.get('file') != f'search-cache-{name}-{phase}.json'):
        raise ValueError('native cache reference is not bound to its source record')
    path = runtime / reference['file']
    if path.resolve().parent != runtime.resolve():
        raise ValueError('native cache reference leaves its runtime')
    saved = json.loads(path.read_text(encoding='utf8'))
    payload = saved['cache']
    epoch = source.get('execution_live_context', source['observed_live_context'])
    if (saved.get('record_id') != name or payload.get('phase') != phase
            or (reference.get('run_id'), reference.get('revision'))
                != (epoch['run_id'], epoch['revision'])
            or (payload.get('run_id'), payload.get('revision'))
                != (epoch['run_id'], epoch['revision'])):
        raise ValueError('native cache generation differs from verified source')
    if payload.get('profile') == _STRONG_PROFILE:
        from .native_search_window import validate_restored_window, validate_recompute_window

        if reference.get('computation_origin') != saved.get('computation_origin'):
            raise ValueError('strong_reference_computation_origin_changed')
        if business_successor is not None:
            from .native_search_window import _read, _index

            successor, latest = _cache_source(runtime, business_successor)
            first, last = _index(payload['planning_windows']), _index(latest['planning_windows'])
            if (research_recompute is not None or payload['phase'] != 'pending_intent'
                    or latest.get('profile') != _STRONG_PROFILE or first >= last
                    or (latest.get('run_id'), latest.get('revision')) != (payload['run_id'], payload['revision'])
                    or successor['mcts_search_cache']['computation_origin'] != saved['computation_origin']
                    or any(_read(runtime, saved['computation_origin'], index)['status'] != 'complete'
                           for index in range(first, last + 1))):
                raise ValueError('strong_reference_business_predecessor_not_completed')
        elif research_recompute is None:
            validate_restored_window(runtime, saved.get('computation_origin'), payload)
        else:
            validate_recompute_window(runtime, source, payload, research_recompute)
    return source, payload


def _require_current_payload(payload, profile):
    from live_decision.high_score.hif_mcts_cache import VERSION, require_candidate_tree_versions

    if not isinstance(payload, dict) or payload.get('version') != VERSION:
        raise ValueError('native_mcts_cache_source_not_verified')
    require_candidate_tree_versions(payload['candidate'], None, complete_policy=profile == _STRONG_PROFILE)


def restore_search_cache(runtime, source, bundle, *, before=None, action=None,
                         submitted=False, accepted=False, retrieval_ready=False,
                         profile=_DEFAULT_PROFILE, source_drink_slots=None, research_recompute=None,
                         information_boundary_execution=False, unsubmitted_plan=None, verified_successors=None):
    """Caller has already completed the applicable native recovery checks.

    Legacy v22 may discard a bad optional cache after business verification.
    The strong trial requires its complete original computation and stops if
    it cannot restore it. Neither path grants observation or input authority.
    """
    from live_decision.high_score.contracts import (
        PlanType,
        DecisionKind,
        DecisionRequest,
        CapabilityReport,
    )
    from live_decision.high_score.hif_mcts_native import HifMctsNativePolicy

    try:
        configuration = _configuration(profile)
        if accepted is not True:
            raise ValueError('native cache requires independently verified business state')
        cache_source, recompute = source, None
        if research_recompute is not None:
            from .native_search_window import prepare_research_recompute

            if profile != _STRONG_PROFILE:
                raise ValueError('research recompute requires strong reference')
            cache_source, recompute = prepare_research_recompute(runtime, source, research_recompute)
        owner, payload = _cache_source(runtime, cache_source, research_recompute=recompute)
        if payload is None:
            if profile == _STRONG_PROFILE:
                raise ValueError('strong reference requires complete original computation')
            return None, dict(restored=False, reason='no_search_cache')
        _require_current_payload(payload, profile)
        if payload.get('profile', _DEFAULT_PROFILE) != profile:
            raise ValueError('native cache profile differs from requested strategy')
        candidate = HifMctsNativePolicy(bundle.kernel, run_id=bundle.run_id,
            revision=bundle.revision, planning_seed=configuration['planning_seed_base'],
            deadline=time.perf_counter()+420., secondary_ready=False,
            retrieval_ready=retrieval_ready,
            **({'information_boundary_execution': True} if information_boundary_execution else {}),
            **({'profile': profile} if profile != _DEFAULT_PROFILE else {}))
        state = bundle.state
        request = DecisionRequest('native-computation-cache', PlanType.SENSE,
            DecisionKind.SECONDARY if state.pending_choice else DecisionKind.LIVE, state,
            CapabilityReport(True, True, True, True, True),
            card_slots=tuple((key, slot) for slot, key in enumerate(
                state.pending_choice.candidate_instance_ids if state.pending_choice else
                tuple(ref.instance_id for ref in state.deck.hand))),
            drink_slots=bundle.drink_slots or (), ruleset_id=bundle.kernel.ruleset_id)
        arguments = dict(request=request, source_run_id=payload['run_id'],
                         source_revision=payload['revision'], accepted=True)
        if information_boundary_execution:
            arguments['allow_information_boundary_upgrade'] = True
        if unsubmitted_plan is not None:
            arguments['unsubmitted_plan'] = unsubmitted_plan
        if verified_successors is not None:
            arguments['verified_successors'] = verified_successors
        if payload['phase'] == 'pending_intent':
            if action is None and submitted is True:
                from .native_settlement_resume import load_result, decoded_dispatched_action

                dispatched = (load_result(runtime, source['native_pending_source'],
                                          allow_closed_transaction=True)
                              if source.get('native_pending_source') else source)
                action, _ = decoded_dispatched_action(dispatched)
            arguments.update(before=before, action=action, submitted=submitted)
            if profile == _STRONG_PROFILE:
                arguments['source_drink_slots'] = source_drink_slots
        report = candidate.restore_search_cache(payload, **arguments)
        if profile == _STRONG_PROFILE:
            candidate._host_computation_origin = owner['mcts_search_cache']['computation_origin']
            if recompute is not None:
                from .native_search_window import begin_research_recompute

                report['research_recompute'] = begin_research_recompute(
                    runtime, owner, payload, candidate, recompute)
        return candidate, dict(report, source_record=owner['id'])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if profile == _STRONG_PROFILE or research_recompute is not None:
            raise ValueError(f'strong_reference_computation_restore_failed:{error}') from error
        return None, dict(restored=False, reason=str(error),
                          computation_discarded=True, fresh_search_required=True)


def restore_retrieval_cache(runtime, source, projected, *, profile=_DEFAULT_PROFILE,
                            information_boundary_execution=False):
    """Restore the same verified pending joint plan, without a predicted commit.

    The caller has already verified the original drink dispatch with
    native_choice_resume. Missing computation is an explicit stop here: a new
    search cannot silently replace the already chosen second half.
    """
    from live_decision.high_score.hif_mcts_native import HifMctsNativePolicy

    configuration = _configuration(profile)
    owner, payload = _cache_source(runtime, source)
    if payload is None or payload.get('phase') != 'pending_intent':
        raise ValueError('retrieval requires its original pending computation')
    _require_current_payload(payload, profile)
    if payload.get('profile', _DEFAULT_PROFILE) != profile:
        raise ValueError('native cache profile differs from requested strategy')
    bundle = projected.source_bundle
    origin = source['choice_journal']['joint_event']
    submitted = (origin.get('execution_submit_count') == 1
        and sum(t.get('use_submitted') is True for t in origin.get('transactions', ())) == 1)
    candidate = HifMctsNativePolicy(bundle.kernel, run_id=bundle.run_id,
        revision=bundle.revision, planning_seed=configuration['planning_seed_base'],
        deadline=time.perf_counter()+420., secondary_ready=False, retrieval_ready=True,
        **({'information_boundary_execution': True} if information_boundary_execution else {}),
        **({'profile': profile} if profile != _DEFAULT_PROFILE else {}))
    report = candidate.restore_retrieval_cache(payload, projected=projected,
        source_plan=origin['proposal']['trace']['planning'],
        source_run_id=payload['run_id'], source_revision=payload['revision'],
        drink_submitted=submitted,
        **({'allow_information_boundary_upgrade': True} if information_boundary_execution else {}))
    if profile == _STRONG_PROFILE:
        candidate._host_computation_origin = owner['mcts_search_cache']['computation_origin']
    return candidate, dict(report, source_record=owner['id'])


def verify_selected_cache(runtime, source, bundle, selected, *, business_successor=None):
    """Verify the complete unsubmitted computation before selected-card input.

    The selected-intent path still obtains fresh observation and dispatch
    authority. This restores calculation for validation only; it does not
    confirm the first action or execute/replan any continuation.
    """
    from live_decision.high_score.hif_mcts import action_key
    from live_decision.high_score.hif_mcts_cache import (
        VERSION,
        action_from_json,
        request_from_public,
        verified_drink_slot_map,
    )
    from live_decision.high_score.hif_mcts_native import HifMctsNativePolicy

    owner, payload = _cache_source(runtime, source, business_successor=business_successor)
    if (payload is None or payload.get('version') != VERSION
            or payload.get('profile') != _STRONG_PROFILE
            or payload.get('phase') != 'pending_intent'
            or (payload.get('run_id'), payload.get('revision')) != (bundle.run_id, bundle.revision)
            or selected.profile != _STRONG_PROFILE):
        raise ValueError('strong_selected_complete_computation_required')
    _require_current_payload(payload, _STRONG_PROFILE)
    for name in ('planning_windows', 'decision_calls'):
        if type(payload.get(name)) is not int or payload[name] < 0:
            raise ValueError('strong_selected_computation_progress_invalid')
    if payload['decision_calls'] < payload['planning_windows']:
        raise ValueError('strong_selected_computation_progress_invalid')
    intended = action_from_json(payload['intent'])
    configuration = _configuration(_STRONG_PROFILE)
    candidate = HifMctsNativePolicy(bundle.kernel, run_id=bundle.run_id,
        revision=bundle.revision, planning_seed=configuration['planning_seed_base'],
        deadline=time.perf_counter()+420., profile=_STRONG_PROFILE)
    anchor = request_from_public(candidate._candidate_cache_anchor(payload['candidate']), bundle.kernel)
    if anchor.state != bundle.kernel._canonical_probability_state(bundle.state):
        raise ValueError('strong_selected_computation_public_state_changed')
    verified_drink_slot_map(anchor.state, anchor.drink_slots, bundle.drink_slots or ())
    action = selected.action
    if action.kind.value == 'USE_DRINK':
        from live_decision.high_score.hif_drink_slot_binding import HifDrinkSlotBinding

        action = HifDrinkSlotBinding.create(bundle.state, anchor.drink_slots,
            bundle.drink_slots or ()).logical_action(action)
    if action_key(intended) != action_key(action):
        raise ValueError('strong_selected_computation_intent_changed')
    receipt = candidate.candidate_for_cache(payload).restore_search_cache(payload['candidate'], anchor, complete_policy=True)
    return dict(receipt, verified=True, source_record=owner['id'], input_authorized=False,
                selected_action_committed=False)
