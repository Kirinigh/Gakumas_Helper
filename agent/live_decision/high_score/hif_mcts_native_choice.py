"""Bind an evaluated retrieval target; commit only the fully observed pair.

The existing Maa transaction owns dispatch receipts, complete panel identity,
fresh input frames and settlement. A predicted pending state is never marked
as a completed observed transaction here. No search or UI input is performed.
"""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace
from dataclasses import asdict, replace

from .choice import HifRetrievalTargetRequest, hif_choice_settlements, project_hif_drink_choice
from .hif_mcts import action_key
from .contracts import ActionKind, GameAction, DecisionKind
from .hif_mcts_cache import (
    VERSION,
    same_json,
    json_value,
    exact_state,
    slots_from_json,
    action_from_json,
    same_pending_plan,
    request_from_public,
)
from .hif_choice_roots import post_card_end_state


def validate_post_card_end_plan(policy, request, action, plan, saved):
    """Prove the evaluated optional END; actual settlement still gates its use."""
    if (action.kind is not ActionKind.PLAY_CARD or request.decision_kind is not DecisionKind.LIVE
            or not policy.candidate.stop_after_card_roots or saved is None
            or saved.get('committed') is not False or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('complete_validation_coverage') is not True
            or saved.get('target') != GameAction(ActionKind.END_TURN)):
        raise ValueError('native_mcts_post_card_end_not_evaluated')
    canonical = policy.kernel._canonical_probability_state
    expected, _ = post_card_end_state(policy.kernel, canonical(request.state), action,
                                      plan_type=request.plan_type)
    raw_expected = json_value(plan.get('selected_expected_after_first'))
    if (expected is None or not isinstance(raw_expected, dict)
            or saved['before'] != canonical(request.state)
            or saved['first'] != replace(action, source_slot=None)
            or saved['expected_state'] != expected
            or canonical(exact_state(raw_expected)) != expected
            or tuple(saved['expected_drink_slots']) != tuple(request.drink_slots)
            or saved['context'] != policy.candidate.search._context(request)
            or not same_json(json_value(plan.get('action')), json_value(asdict(action)))
            or not same_json(json_value(plan.get('selected_continuation')),
                             json_value([asdict(saved['target'])]))):
        raise ValueError('native_mcts_post_card_end_evaluated_plan_changed')
    return expected


def validate_retrieval_plan(policy, before, action, plan, saved, *, projected=None):
    """Check the one supported evaluated joint root without enabling others."""
    if (not policy.retrieval_ready or action.kind is not ActionKind.USE_DRINK
            or action.drink_id != '22' or saved is None or saved.get('committed') is not False
            or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('complete_validation_coverage') is not True
            or plan.get('root_options_enabled') is not True):
        raise ValueError('native_mcts_mandatory_choice_not_connected')
    canonical = policy.kernel._canonical_probability_state
    raw_expected = json_value(plan.get('selected_expected_after_first'))
    if not isinstance(raw_expected, dict):
        raise ValueError('native_mcts_retrieval_evaluated_plan_changed')
    if (saved['before'] != canonical(before)
            or saved['first'] != replace(action, source_slot=None)
            or saved['target'].kind is not ActionKind.MOVE_CARD
            or saved['target'].source_slot is not None
            or not same_json(json_value(plan.get('action')), json_value(asdict(action)))
            or not same_json(json_value(plan.get('selected_continuation')),
                             json_value([asdict(saved['target'])]))
            or canonical(exact_state(raw_expected))
                != saved['expected_state']):
        raise ValueError('native_mcts_retrieval_evaluated_plan_changed')
    slots = tuple(saved['expected_drink_slots']) + ((action.source_slot, action.drink_id),)
    bundle = SimpleNamespace(kernel=policy.kernel, state=before, drink_slots=slots)
    expected = project_hif_drink_choice(bundle, action, canonical_input=True)
    pending = expected.pending_state.pending_choice
    candidates = pending.candidate_instance_ids
    if (pending.source_id != 'drink:22' or not candidates
            or len(set(candidates)) != len(candidates)
            or saved['target'].card_instance_id not in pending.candidate_instance_ids
            or canonical(expected.pending_state) != saved['expected_state']
            or expected.drink_slots != saved['expected_drink_slots']):
        raise ValueError('native_mcts_retrieval_outside_connected_domain')
    if projected is not None and (
            projected.source_bundle.kernel is not policy.kernel
            or canonical(projected.source_bundle.state) != canonical(before)
            or projected.action != action
            or canonical(projected.pending_state) != saved['expected_state']
            or projected.drink_slots != saved['expected_drink_slots']
            or (projected.source_bundle.run_id, projected.source_bundle.revision)
                != (policy.run_id, policy.revision)):
        raise ValueError('native_mcts_retrieval_projection_changed')
    return expected


def bind_retrieval_choice(policy, projected, request, *, drink_submitted):
    """Rebind the original business target to this complete observed panel."""
    started = time.perf_counter()
    action, plan = None, {}
    try:
        if drink_submitted is not True or policy._pending is None:
            raise ValueError('native_mcts_retrieval_drink_dispatch_not_verified')
        if policy.candidate.deadline is not None and time.perf_counter() >= policy.candidate.deadline:
            raise TimeoutError('native_mcts_host_session_expired')
        pending, saved = policy._pending, policy.candidate._choice_continuation
        validate_retrieval_plan(policy, projected.source_bundle.state, pending['action'],
            pending['plan'], saved, projected=projected)
        target_only = isinstance(request, HifRetrievalTargetRequest)
        legal = policy._legal(request, retrieval_target=saved['target'].card_instance_id if target_only else None)
        candidates = saved['expected_state'].pending_choice.candidate_instance_ids
        if (request.decision_kind is not DecisionKind.SECONDARY
                or not target_only and sorted(slot for _, slot in request.card_slots) != list(range(len(candidates)))
                or policy.kernel._canonical_probability_state(request.state) != saved['expected_state']
                or request.drink_slots != saved['expected_drink_slots']
                or request.utility != pending['utility']
                or policy.candidate.search._context(request) != saved['context']):
            raise ValueError('native_mcts_retrieval_current_panel_state_changed')
        matches = [value for value in legal if action_key(value) == action_key(saved['target'])]
        if len(matches) != 1:
            raise ValueError('native_mcts_retrieval_target_not_uniquely_bound')
        action = matches[0]
        plan = dict(accepted=True, scope='JOINT_ROOT_PENDING_SELECTION',
            complete_joint_root_coverage=True, source_plan_index=saved['source_plan_index'],
            requested_simulations=0, completed_simulations=0, candidate_statistics=[],
            action=asdict(action), new_search_performed=False, committed=False)
        policy._retrieval_choice = dict(action=action, request=request, projected=projected)
        reason = 'native_mcts_evaluated_retrieval_target_rebound'
    except (ValueError, RuntimeError, TimeoutError, KeyError, TypeError) as error:
        policy._retrieval_choice = None
        action = None
        reason = str(error)
    return policy._response(action, reason, started, plan, policy._tree_context)


def confirm_retrieval_committed(policy, *, run_id, revision, before, action,
                                pending_state, choice_action, after,
                                drink_submitted, choice_submitted, accepted):
    """Validate both actual submissions and final observation before mutation."""
    if drink_submitted is not True or choice_submitted is not True or accepted is not True:
        raise ValueError('native_mcts_retrieval_requires_two_dispatches_and_settlement')
    if not policy.matches_context(policy.kernel, run_id=run_id, revision=revision):
        raise ValueError('native_mcts_commit_context_changed')
    pending, bound = policy._pending, policy._retrieval_choice
    canonical = policy.kernel._canonical_probability_state
    if (pending is None or bound is None or pending['action'] != action
            or canonical(before) != pending['before'] or bound['action'] != choice_action
            or canonical(pending_state) != canonical(bound['request'].state)):
        raise ValueError('native_mcts_retrieval_commit_does_not_match_bound_pair')
    validate_retrieval_plan(policy, before, action, pending['plan'],
        policy.candidate._choice_continuation, projected=bound['projected'])
    expected = hif_choice_settlements(bound['projected'], choice_action,
                                     card_slots=bound['request'].card_slots)
    outcomes = expected if isinstance(expected, tuple) else (expected,)
    after.validate()
    policy.kernel._validate_registered_state(after, plan_type=bound['request'].plan_type)
    if canonical(after) not in tuple(canonical(state) for state in outcomes):
        raise ValueError('native_mcts_retrieval_final_observation_changed')
    # Both transitions are validated before either tree mutation. The pending
    # projection is only a computation waypoint in this completed transaction.
    policy.candidate.committed(before, action, pending_state)
    policy.candidate.committed(pending_state, choice_action, after)
    policy._expected_after = canonical(after)
    policy._cache_request = replace(bound['request'], state=policy._expected_after,
        decision_kind=DecisionKind.LIVE,
        card_slots=tuple((ref.instance_id, index) for index, ref in enumerate(after.deck.hand)))
    policy._pending = policy._retrieval_choice = None
    return dict(retained=True, reason='complete_observed_retrieval_pair',
                committed_actions=2, original_action=asdict(action), choice_action=asdict(choice_action))


def restore_retrieval_cache(policy, payload, *, projected, source_plan,
                            source_run_id, source_revision, drink_submitted,
                            allow_information_boundary_upgrade=False):
    """Public helper shares the native receiver's complete transaction."""
    return policy.restore_retrieval_cache(payload, projected=projected, source_plan=source_plan,
        source_run_id=source_run_id, source_revision=source_revision, drink_submitted=drink_submitted,
        allow_information_boundary_upgrade=allow_information_boundary_upgrade)


def _restore_retrieval_cache(policy, payload, *, projected, source_plan,
                             source_run_id, source_revision, drink_submitted):
    """Restore computation for a verified dispatched but unsettled retrieval.

    The host must verify the original dispatch journal and source process/run.
    It must still capture the complete current panel before binding any target.
    No cached slot, input credential, deadline or observed commit is restored.
    """
    if not policy.retrieval_ready or drink_submitted is not True:
        raise ValueError('native_mcts_retrieval_drink_dispatch_not_verified')
    if (policy._pending is not None or policy._expected_after is not None
            or policy._cache_request is not None or policy._windows or policy._calls):
        raise ValueError('native_mcts_cache_requires_new_instance')
    payload = json_value(payload)
    cached_plan = payload.pop('pending_plan', None)
    if (set(payload) != {'version', 'profile', 'run_id', 'revision', 'phase', 'intent',
                        'planning_windows', 'decision_calls', 'candidate'}
            or payload['version'] != VERSION or payload['profile'] != policy.profile
            or payload['phase'] != 'pending_intent'
            or (payload['run_id'], payload['revision']) != (source_run_id, source_revision)
            or (source_run_id, source_revision) != (policy.run_id, policy.revision)
            or not isinstance(source_run_id, str) or not source_run_id
            or type(source_revision) is not int or source_revision < 0
            or any(type(payload[key]) is not int or payload[key] < 1
                   for key in ('planning_windows', 'decision_calls'))
            or payload['decision_calls'] < payload['planning_windows']):
        raise ValueError('native_mcts_retrieval_cache_source_not_verified')
    anchor = request_from_public(policy._candidate_cache_anchor(payload['candidate']), policy.kernel)
    action = action_from_json(payload['intent'])
    source_binding = None
    from .hif_mcts_native import STRONG_PROFILE

    if policy.profile == STRONG_PROFILE:
        from .hif_drink_slot_binding import HifDrinkSlotBinding

        source_binding = HifDrinkSlotBinding.create(anchor.state, anchor.drink_slots,
                                                   projected.source_bundle.drink_slots)
        source_plan = copy.deepcopy(source_plan)
        source_plan['action'] = asdict(source_binding.logical_action(action_from_json(source_plan['action'])))
        projected = policy._logical_retrieval_projection(projected, drink_slots=anchor.drink_slots)
    if cached_plan is not None and not same_pending_plan(cached_plan, json_value(source_plan)):
        raise ValueError('native_mcts_retrieval_pending_plan_changed')
    raw = payload['candidate']['choice']
    if not isinstance(raw, dict):
        raise ValueError('native_mcts_retrieval_cache_has_no_evaluated_choice')
    saved = dict(raw, before=exact_state(raw['before']), expected_state=exact_state(raw['expected_state']),
                 first=action_from_json(raw['first']), target=action_from_json(raw['target']))
    saved['expected_drink_slots'] = slots_from_json(raw['expected_drink_slots'], saved['expected_state'])
    validate_retrieval_plan(policy, anchor.state, action, source_plan, saved, projected=projected)
    if (anchor.decision_kind is not DecisionKind.LIVE
            or anchor.drink_slots != projected.source_bundle.drink_slots
            or policy.kernel._canonical_probability_state(projected.source_bundle.state) != anchor.state):
        raise ValueError('native_mcts_retrieval_cache_anchor_changed')
    restored = policy._restore_candidate_cache(payload['candidate'], anchor)
    policy._pending = dict(before=anchor.state, action=action, utility=anchor.utility,
                           drink_slots=anchor.drink_slots, plan=copy.deepcopy(source_plan))
    if source_binding is not None:
        policy._pending['drink_binding'] = source_binding
        policy._logical_drink_slots, policy._drink_binding = source_binding.logical_slots, source_binding
    policy._cache_request = anchor
    policy._calls, policy._windows = payload['decision_calls'], payload['planning_windows']
    policy._tree_context = 'restored_evaluated_retrieval_without_observed_commit'
    return dict(restored, restored=True, observed_commit=False, source_run_id=source_run_id,
                source_revision=source_revision, reason=policy._tree_context)
