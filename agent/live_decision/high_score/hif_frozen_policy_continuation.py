"""Business-only deterministic links to an already evaluated frozen policy."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from .choice import _valid_branches
from .kernel import KernelError
from .planner import SearchIncomplete
from .contracts import ActionKind
from .hif_mcts_cache import json_value, exact_state, slots_from_json, action_from_json
from .hif_choice_roots import _UNOBSERVED_EVENTS
from .validation_cache import validation_session
from .hif_turn_chain_continuation import _NoDrawKernel


def deterministic_public_successor(kernel, state, action, *, plan_type, deadline):
    """Return a complete same-turn successor, or an explicit ineligibility."""
    return _deterministic_public_successor(kernel, state, action,
        plan_type=plan_type, deadline=deadline)


def _deterministic_public_successor(kernel, state, action, *, plan_type, deadline,
                                    public_turn_proof=None):
    """Internal shared classification; only the public graph supplies its proof."""
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('frozen_policy_finite_deadline_required')

    def check_budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError('frozen_policy_deadline')

    check_budget()
    if state.terminal or state.turns_remaining <= 2 or action.kind is ActionKind.END_TURN:
        return None, 'outside_same_turn_domain'
    probe = public_turn_proof
    if probe is None:
        probe = _NoDrawKernel(kernel.cards, kernel.drinks, kernel.items,
                              memory_abilities=kernel.memory_abilities)
    if probe.ruleset_id != kernel.ruleset_id:
        return None, 'unsupported_ruleset'
    with validation_session(check_budget, reuse_parent=True):
        try:
            branches = probe.transition(state, replace(action, source_slot=None), plan_type=plan_type)
        except SearchIncomplete as error:
            if str(error) != 'turn_chain_draw_domain':
                raise
            return None, 'draw_domain'
        except KernelError as error:
            if public_turn_proof is not None:
                # The graph must distinguish an ordinary unsupported transition
                # from a failure that used to occur only in the strict proof.
                raise
            return None, 'unsupported_transition:' + str(error)
        check_budget()
        _valid_branches(branches)
        if len(branches) != 1 or branches[0].probability != 1.:
            return None, 'non_deterministic_transition'
        branch = branches[0]
        if any(event.event_type.startswith(_UNOBSERVED_EVENTS) or 'RANDOM' in event.event_type
               for event in branch.events):
            return None, 'draw_or_random_event'
        # HifEnvironment's sampler canonicalizes its final probability branch,
        # including a MOVE that appends to the hand. Reproduce that boundary;
        # later compare the actual observation verbatim, not through a key.
        after = kernel._canonical_probability_state(branch.state)
        if (after.terminal or after.turn != state.turn or after.turns_remaining != state.turns_remaining
                or (after.actions_remaining == 0 and not after.pending_choice and not after.resolution_stack)):
            return None, 'automatic_or_explicit_turn_boundary'
        # No search-key pruning, score translation, or observation normalization.
        return after, 'deterministic_same_turn_public_state'


def source_window_ms(plan):
    """Bind the source receipt to its explicit policy mode and finite watchdog."""
    fields = ('quality_reference_enabled', 'quality_reference_version',
              'quality_reference_watchdog_seconds')
    if not any(field in plan for field in fields):
        return 30000.
    from .hif_mcts import QUALITY_REFERENCE_VERSION

    seconds = plan.get('quality_reference_watchdog_seconds')
    if (plan.get('quality_reference_enabled') is not True
            or plan.get('quality_reference_version') != QUALITY_REFERENCE_VERSION
            or type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 < seconds <= 300. or plan.get('stop_reason') != 'completed'):
        return None
    return seconds * 1000


def source_simulation_budget(plan):
    """Only a versioned quality sensitivity receipt may exceed the old budget."""
    fields = ('compute_sensitivity_enabled', 'compute_sensitivity_version', 'compute_sensitivity_simulations')
    if not any(field in plan for field in fields):
        return 256
    from .hif_mcts import QUALITY_REFERENCE_VERSION, COMPUTE_SENSITIVITY_VERSION

    budget = plan.get('compute_sensitivity_simulations')
    seconds = plan.get('quality_reference_watchdog_seconds')
    if (plan.get('compute_sensitivity_enabled') is not True
            or plan.get('compute_sensitivity_version') != COMPUTE_SENSITIVITY_VERSION
            or type(budget) is not int or budget not in (256, 512, 1024)
            or plan.get('quality_reference_enabled') is not True
            or plan.get('quality_reference_version') != QUALITY_REFERENCE_VERSION
            or type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 300.):
        return None
    return budget


def source_mode_receipt(plan):
    # Existing receipt bytes stay unchanged, including quality-reference v1.
    if plan.get('compute_sensitivity_enabled') is not True:
        return {}
    return {name: plan.get(name) for name in (
        'compute_sensitivity_enabled', 'compute_sensitivity_version', 'compute_sensitivity_simulations',
        'quality_reference_enabled', 'quality_reference_version', 'quality_reference_watchdog_seconds')}


def _source_context_matches(plan, context):
    from .hif_mcts import QUALITY_REFERENCE_VERSION

    marked = isinstance(context, tuple) and 'quality_reference' in context
    compute = isinstance(context, tuple) and 'compute_sensitivity' in context
    budget = source_simulation_budget(plan)
    if compute:
        from .hif_mcts import COMPUTE_SENSITIVITY_VERSION

        index = context.index('compute_sensitivity')
        if context[index:index+3] != ('compute_sensitivity', COMPUTE_SENSITIVITY_VERSION, budget):
            return False
    return (budget is not None and marked == (plan.get('quality_reference_enabled') is True)
            and (not marked or QUALITY_REFERENCE_VERSION in context)
            and compute == (plan.get('compute_sensitivity_enabled') is True))


def qualified_source(plan):
    """Only a complete paired search can establish a continuation lineage."""
    rows = plan.get('candidate_statistics')
    limit = source_window_ms(plan)
    budget = source_simulation_budget(plan)
    return (limit is not None and budget is not None
            and plan.get('scope') == 'SAMPLING_ESTIMATE' and plan.get('accepted') is True
            and plan.get('deterministic_policy_continuation') is True
            and plan.get('complete_root_coverage') is True
            and plan.get('complete_validation_coverage') is True
            and type(plan.get('requested_simulations')) is int and plan['requested_simulations'] == budget
            and type(plan.get('completed_simulations')) is int and 0 < plan['completed_simulations'] <= budget
            and type(plan.get('elapsed_ms')) in (int, float) and 0 <= plan['elapsed_ms'] <= limit
            and plan.get('stop_reason') in ('completed', 'soft_budget_reached')
            and isinstance(rows, (list, tuple)) and bool(rows)
            and all(type(row.get('training_paired_visits')) is int and row['training_paired_visits'] >= 4
                    and type(row.get('validation_paired_visits')) is int and row['validation_paired_visits'] >= 4
                    for row in rows))


def business_action(action):
    return action if action.kind is ActionKind.USE_DRINK else replace(action, source_slot=None)


def make_link(kernel, request, action, *, source_plan, source_plan_index,
              predecessor_plan_index, context, deadline):
    after, _ = deterministic_public_successor(kernel, request.state, action,
        plan_type=request.plan_type, deadline=deadline)
    if after is None:
        return None
    slots = tuple(request.drink_slots)
    expected_slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id)) \
        if action.kind is ActionKind.USE_DRINK else slots
    return dict(source_plan=source_plan, context=context, proof=dict(
        before=request.state, before_drink_slots=slots, action=business_action(action),
        expected_state=after, expected_drink_slots=expected_slots, committed=False,
        source_plan_index=source_plan_index, predecessor_plan_index=predecessor_plan_index))


def matches_link(link, request, context):
    if (link is None or link['context'] != context or not qualified_source(link['source_plan'])
            or not _source_context_matches(link['source_plan'], context)):
        return False
    proof = link['proof']
    return (proof['committed'] is True and proof['expected_state'] == request.state
            and proof['expected_drink_slots'] == tuple(request.drink_slots)
            and request.state.turn == proof['before'].turn and request.state.turns_remaining > 2)


def export_link(link):
    if link is None:
        return None
    proof = {key: asdict(value) if key in ('before', 'action', 'expected_state') else value
             for key, value in link['proof'].items()}
    return json_value(dict(source_plan=link['source_plan'], proof=proof))


def restore_link(raw, kernel, request, *, context, plan_count, deadline):
    if raw is None:
        return None
    fields = {'before', 'before_drink_slots', 'action', 'expected_state', 'expected_drink_slots',
              'committed', 'source_plan_index', 'predecessor_plan_index'}
    if (not isinstance(raw, dict) or set(raw) != {'source_plan', 'proof'}
            or not isinstance(raw['source_plan'], dict) or not qualified_source(raw['source_plan'])
            or not _source_context_matches(raw['source_plan'], context)
            or not isinstance(raw['proof'], dict) or set(raw['proof']) != fields):
        raise ValueError('frozen_policy_cache_source_invalid')
    proof = dict(raw['proof'])
    if (type(proof['committed']) is not bool
            or any(type(proof[key]) is not int for key in ('source_plan_index', 'predecessor_plan_index'))
            or not 0 <= proof['source_plan_index'] <= proof['predecessor_plan_index'] < plan_count):
        raise ValueError('frozen_policy_cache_progress_invalid')
    proof.update(before=exact_state(proof['before']), expected_state=exact_state(proof['expected_state']),
                 action=action_from_json(proof['action']))
    proof['before_drink_slots'] = slots_from_json(proof['before_drink_slots'], proof['before'])
    proof['expected_drink_slots'] = slots_from_json(proof['expected_drink_slots'], proof['expected_state'])
    action = proof['action']
    if action != business_action(action):
        raise ValueError('frozen_policy_cache_old_card_slot')
    slots = proof['before_drink_slots']
    if action.kind is ActionKind.USE_DRINK:
        if (action.source_slot, action.drink_id) not in slots:
            raise ValueError('frozen_policy_cache_drink_slot')
        slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
    if slots != proof['expected_drink_slots']:
        raise ValueError('frozen_policy_cache_drink_transition')
    after, _ = deterministic_public_successor(kernel, proof['before'], action,
        plan_type=request.plan_type, deadline=deadline)
    state = proof['expected_state'] if proof['committed'] else proof['before']
    current_slots = proof['expected_drink_slots'] if proof['committed'] else proof['before_drink_slots']
    if after is None or after != proof['expected_state'] or state != request.state or current_slots != request.drink_slots:
        raise ValueError('frozen_policy_cache_state_changed')
    return dict(source_plan=raw['source_plan'], proof=proof, context=context)
