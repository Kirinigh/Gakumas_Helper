"""Verified UIObs v2 bindings for the unchanged Sense reference search."""

from __future__ import annotations

import time
from struct import pack
from collections import Counter
from dataclasses import replace

from tools.task280_gkbench_smoke import verify_final
from tools.task280_gkbench_hosted import LOCAL_BLACK_VINEGAR, planning_card_key, delegated_retrieval
from tools.task275.gkms_v2_observation import project_v2, build_v2_observation
from agent.live_decision.high_score.contracts import PlanType, ActionKind, GameAction, DecisionKind
from agent.live_decision.high_score.hif_environment import HifStep
from agent.live_decision.high_score.validation_cache import validation_session
from agent.live_decision.high_score.gkms_shadow_adapter import build_shadow_observation
from agent.live_decision.high_score.gkms_continuous_policy import ObservedHandOrder

_ORDERED_CARDS = ('hand', 'played')
_CARD_BAGS = ('deck', 'grave', 'hold', 'lost')
_EXACT_FIELDS = ('exam', 'num', 'stance', 'drinks')
_ROW_BAGS = ('status', 'ench', 'timers', 'costs', 'pcount')
_ALL_CARDS_SEARCH = 'p_card_search-deck_all'
_F32_STATUS_AMOUNTS = {'lesson_parameter_multiple', 'lesson_buff_multiple', 'lesson_buff_gain_fix'}


def _action_key(action):
    if action == ['end']:
        return ('end',)
    if (isinstance(action, list) and len(action) == 2 and action[0] in {'play', 'drink'}
            and type(action[1]) is int and action[1] >= 0):
        return tuple(action)
    raise ValueError('task280_unsupported_referee_action')


def _row_key(field, row):
    if field == 'status' and row[0] in _F32_STATUS_AMOUNTS:
        return row[0], pack('!f', row[1]), row[2]
    return tuple(row)


def public_differences(observation, projection):
    """Compare visible fields without treating an unordered zone as a draw order."""
    if observation.get('v') != 'uiobs.v2':
        return ('schema',)
    missing = tuple(field for field in (*_EXACT_FIELDS, *_ORDERED_CARDS, *_CARD_BAGS,
                                       *_ROW_BAGS, 'legal') if field not in observation)
    if missing:
        return tuple('missing:' + field for field in missing)
    differences = [field for field in _EXACT_FIELDS if observation[field] != projection[field]]
    differences += [field for field in _ORDERED_CARDS
                    if list(map(planning_card_key, observation[field]))
                    != list(map(planning_card_key, projection[field]))]
    differences += [field for field in _CARD_BAGS
                    if Counter(map(planning_card_key, observation[field]))
                    != Counter(map(planning_card_key, projection[field]))]
    differences += [field for field in _ROW_BAGS
                    if Counter(_row_key(field, row) for row in observation[field])
                    != Counter(_row_key(field, row) for row in projection[field])]
    if Counter(map(_action_key, observation['legal'])) != Counter(map(_action_key, projection['legal'])):
        differences.append('legal')
    return tuple(differences)


def bind_request(observation, candidates, *, kernel, context, exam_effects=None):
    """Bind a unique, independently derived complete local state to the receipt."""
    exam = observation.get('exam') or {}
    if type(exam.get('plan')) is not int or exam['plan'] != 2:
        raise ValueError('task280_strong_reference_requires_sense')
    matches, rejected = [], []
    for request, history in candidates:
        if request.plan_type is not PlanType.SENSE:
            raise ValueError('task280_request_plan_mismatch')
        if request.state.deck.known_top_count:
            raise ValueError('task280_public_deck_cannot_bind_known_draw_order')
        shadow = build_v2_observation(request, kernel, context, history, exam_effects=exam_effects)
        if shadow.reason_codes == ('v2_costs_source_and_layers_unresolved',):
            count = request.state.nullify_cost_cards
            # The observed single layer specifies its target and cost. The
            # independent kernel branch must already explain the uses left.
            if observation.get('costs') == [[_ALL_CARDS_SEARCH, 0, count]]:
                source = build_shadow_observation(request, kernel, context, history)
                formatted = replace(request, state=replace(request.state, nullify_cost_cards=0))
                shadow = project_v2(replace(source, request=formatted), exam_effects=exam_effects)
                if shadow.status == 'supported':
                    projection = shadow.observation | {'costs': [[_ALL_CARDS_SEARCH, 0, count]]}
                    shadow = replace(shadow, request=request, observation=projection)
        if shadow.status != 'supported':
            rejected.extend(shadow.reason_codes)
            continue
        differences = public_differences(observation, shadow.observation)
        if differences:
            rejected.extend(differences)
        else:
            matches.append(shadow)
    if len(matches) != 1:
        reason = 'ambiguous' if matches else ','.join(sorted(set(rejected))) or 'no_candidates'
        raise ValueError('task280_public_state_not_bound:' + reason)
    return matches[0]


def _resolved_branches(kernel, request, action):
    """Enumerate actual rule outcomes without sampling a hidden deal."""
    for branch in kernel.transition(request.state, action, plan_type=request.plan_type):
        choices = (branch,)
        if branch.state.pending_choice:
            choices = (
                replace(selected, events=branch.events + selected.events)
                for choice in kernel.legal_actions(branch.state, plan_type=request.plan_type,
                                                   decision_kind=DecisionKind.SECONDARY)
                for selected in kernel.transition(branch.state, choice, plan_type=request.plan_type)
            )
        for resolved in choices:
            state = resolved.state
            if state.pending_choice or state.resolution_stack or state.pending_after_card_ids:
                raise ValueError('task280_unresolved_referee_action')
            if state.terminal or state.actions_remaining:
                yield resolved
            else:
                for advanced in kernel.transition(state, GameAction(ActionKind.END_TURN),
                                                  plan_type=request.plan_type):
                    yield replace(advanced, events=resolved.events + advanced.events)


def bind_successor(before, request, history, wire_action, reply, *, kernel, context,
                   exam_effects=None, expected_retrieval=None):
    """Advance the submitted action from the actual referee receipt."""
    index = before['i']
    if type(index) is not int or index < 0 or wire_action not in before['legal']:
        raise ValueError('task280_unadvertised_or_unindexed_action')
    root_id, separator, previous_index = request.snapshot_id.rpartition('@step=')
    if (separator and previous_index != str(index)) or (not separator and index != 0):
        raise ValueError('task280_stale_local_request')
    shadow = bind_request(before, ((request, history),), kernel=kernel, context=context,
                          exam_effects=exam_effects)
    action = shadow.bind_action(wire_action, request=request, kernel=kernel)
    score = reply['result']['after']['score']
    if (type(score) is not int or type(reply['done']) is not bool
            or score - before['num']['score'] != reply['result']['delta'].get('score', 0)):
        raise ValueError('task280_referee_score_delta_mismatch')
    following = reply.get('obs')
    if reply['done']:
        verify_final(reply['final'], score=score, decisions=index + 1)
    elif (not isinstance(following, dict) or type(following.get('i')) is not int
          or following['i'] != index + 1 or following.get('num', {}).get('score') != score):
        raise ValueError('task280_stale_or_inconsistent_referee_successor')
    elif any(token[0] == 'select' for token in following['legal']):
        raise ValueError('task280_explicit_referee_selection_unbound')

    bottles = tuple(pair for pair in request.drink_slots
                    if action.kind is not ActionKind.USE_DRINK or pair[0] != action.source_slot)
    start_order = tuple(iid for iid, _ in sorted(request.card_slots, key=lambda pair: pair[1]))
    candidates = []
    with validation_session():
        for branch in _resolved_branches(kernel, request, action):
            state = branch.state
            if state.terminal != reply['done'] or state.score != score:
                continue
            if state.terminal:
                return None, None, None
            if len(state.deck.hand) != len(following['hand']):
                continue
            updated = replace(request, snapshot_id=f'{root_id or request.snapshot_id}@step={index + 1}',
                              state=state, drink_slots=bottles)
            order = ObservedHandOrder(start_order)
            order.advance(request.state, HifStep(updated, score - request.state.score, False, branch.events))
            updated = order.request(updated)
            candidate = updated, history.advance(action, state, branch.events,
                                                 card_upgrades=context.mappings.card_upgrades)
            if candidate not in candidates:
                candidates.append(candidate)
    if reply['done']:
        raise ValueError('task280_referee_terminal_not_bound')
    matched = bind_request(following, candidates, kernel=kernel, context=context, exam_effects=exam_effects)
    bound_history = next(value for current, value in candidates if current is matched.request)
    retrieval = None
    if action.kind is ActionKind.USE_DRINK and action.drink_id == LOCAL_BLACK_VINEGAR:
        retrieval = delegated_retrieval(before, following, expected=expected_retrieval)
    return matched.request, bound_history, retrieval


def choose_bound_action(observation, candidates, *, kernel, context, planning_seed, deadline=None,
                        exam_effects=None, clock=time.perf_counter, policy_factory):
    """Include state binding and policy creation in the caller's absolute budget."""
    shadow = bind_request(observation, candidates, kernel=kernel, context=context, exam_effects=exam_effects)
    if deadline is not None and clock() >= deadline:
        raise TimeoutError('task280_step_budget_expired_before_search')
    policy = policy_factory(kernel, planning_seed=planning_seed, deadline=deadline)
    roots, seen_drinks = [], set()
    for binding in shadow.bindings:
        proposed = binding.action
        if proposed.kind is ActionKind.USE_DRINK:
            if proposed.drink_id in seen_drinks:
                continue
            seen_drinks.add(proposed.drink_id)
        roots.append(proposed)
    action = policy.choose(shadow.request, tuple(roots))
    matches = [list(binding.token) for binding in shadow.bindings if binding.action == action]
    if len(matches) != 1 or matches[0] not in observation['legal']:
        raise ValueError('task280_action_not_bound_to_referee_legal')
    if deadline is not None and clock() >= deadline:
        raise TimeoutError('task280_step_budget_expired_after_search')
    return matches[0]
