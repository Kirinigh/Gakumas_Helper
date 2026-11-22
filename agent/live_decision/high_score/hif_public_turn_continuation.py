"""In-process committed edges of one completely validated public-turn table."""
from __future__ import annotations

import time
from dataclasses import asdict

from .hif_mcts import action_key
from .contracts import ActionKind
from .hif_mcts_cache import public_request
from .hif_choice_roots import _complete_legal
from .hif_public_turn_reanalysis import VERSION as PUBLIC_VERSION
from .hif_public_turn_reanalysis import PublicTurnCursor, public_key
from .hif_frozen_policy_continuation import (
    business_action,
    source_mode_receipt,
    _source_context_matches,
    source_simulation_budget,
)

VERSION = 'hif-public-turn-policy-continuation-v1'


def source_validation(plan, action):
    """Keep a receipt of existing complete samples, never invent new ones."""
    table = plan.get('public_turn_reanalysis') or {}
    rows = plan.get('candidate_statistics')
    budget = source_simulation_budget(plan)
    if (budget is None or plan.get('scope') != 'SAMPLING_ESTIMATE' or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('complete_validation_coverage') is not True
            or plan.get('requested_simulations') != budget
            or not 0 < plan.get('completed_simulations', 0) <= budget
            or plan.get('stop_reason') not in ('completed', 'soft_budget_reached')
            or plan.get('compute_sensitivity_enabled') is True and plan.get('stop_reason') != 'completed'
            or table.get('complete') is not True or table.get('version') != PUBLIC_VERSION
            or not isinstance(rows, (tuple, list)) or not rows
            or any(type(row.get(field)) is not int or row[field] < 4
                   for row in rows for field in ('training_paired_visits', 'validation_paired_visits'))):
        return None
    return dict(source_scope='SAMPLING_ESTIMATE', source_accepted=True,
        source_root_action=asdict(business_action(action)),
        complete_root_coverage=True, complete_validation_coverage=True,
        requested_simulations=budget, completed_simulations=plan['completed_simulations'],
        training_paired_visits=tuple(row['training_paired_visits'] for row in rows),
        validation_paired_visits=tuple(row['validation_paired_visits'] for row in rows),
        root_selection_scope=plan['root_selection_scope'],
        public_table_complete=True, public_table_version=PUBLIC_VERSION,
        **source_mode_receipt(plan))


class PublicTurnPolicyContinuation:
    """The graph proves zero random consumption; real commits prove arrival.

    No simulator RNG token or old input binding is fabricated. Neither a
    policy table nor a pending link is portable or saved in a search cache.
    """

    version = VERSION

    def __init__(self, kernel):
        self.kernel = kernel
        self.reset()

    def reset(self):
        self._pending = self._active = None

    def prepare(self, request, context):
        saved, self._pending = self._pending, None
        self._active = None
        if saved is None:
            return
        proof = saved['proof']
        if (proof['committed'] and saved['context'] == context
                and proof['expected_state'] == request.state
                and proof['expected_drink_slots'] == tuple(request.drink_slots)
                and saved['plan_type'] == request.plan_type
                and saved['ruleset_id'] == request.ruleset_id
                and saved['utility'] == request.utility
                and request.state.turn == proof['before'].turn
                and request.state.turns_remaining > 2 and not request.state.terminal):
            self._active = saved

    def choose(self, request, legal, plan, deadline):
        saved = self._active
        if saved is None:
            return None
        if time.perf_counter() >= deadline:
            raise TimeoutError('public_turn_continuation_deadline')
        _complete_legal(self.kernel, request, legal)
        key = public_key(request)
        node = saved['policy'].nodes.get(key)
        if node is None or {action_key(a) for a in legal} != {e['action_key'] for e in node['edges']}:
            self.reset()
            return None
        # The candidate's existing single-action, terminal and dominance
        # paths run first, matching the rollout's proven-action precedence.
        selected = PublicTurnCursor(saved['policy'], key).choose(request, legal)
        matches = [action for action in legal if action_key(action) == selected]
        if len(matches) != 1:
            raise ValueError('public_turn_continuation_action_not_uniquely_legal')
        if time.perf_counter() >= deadline:
            raise TimeoutError('public_turn_continuation_deadline')
        proof = dict(saved['proof'])
        proof.update({name: asdict(proof[name]) for name in ('before', 'action', 'expected_state')})
        proof.update(source_plan_index=saved['source_plan_index'],
                     predecessor_plan_index=saved['predecessor_plan_index'])
        plan.update(scope='PUBLIC_TURN_POLICY_CONTINUATION', requested_simulations=0,
            completed_simulations=0, candidate_statistics=[], complete_root_coverage=False,
            stop_reason='matched_committed_public_turn_policy',
            source_plan_index=saved['source_plan_index'],
            predecessor_plan_index=saved['predecessor_plan_index'],
            source_action=saved['source_validation']['source_root_action'],
            source_validation=dict(saved['source_validation']), continuation_proof=proof,
            public_turn_node=dict(request=public_request(request),
                options=saved['policy'].choices[key]['options'],
                maximizing_actions=saved['policy'].choices[key]['maximizing_actions']),
            public_turn_reanalysis_version=PUBLIC_VERSION,
            public_turn_policy_continuation_version=VERSION)
        return matches[0]

    def remember(self, request, action, plan, policy, context, plan_index):
        """Arm only a graph child; the next real commit must still match it."""
        self._pending = None
        source = source_validation(plan, action)
        if source is not None:
            compute_context = isinstance(context, tuple) and 'compute_sensitivity' in context
            if ((plan.get('compute_sensitivity_enabled') is True or compute_context)
                    and not _source_context_matches(plan, context)):
                self._active = None
                return False
            if policy is None or public_key(request) != policy.root_key:
                self._active = None
                return False
            saved = dict(policy=policy, source_plan_index=plan_index, source_validation=source)
        elif self._active is not None and plan['scope'] in (
                'PUBLIC_TURN_POLICY_CONTINUATION', 'FORCED_LEGAL_ACTION', 'PROVEN_RESOURCE_DOMINANCE'):
            saved = self._active
            policy = saved['policy']
        else:
            self._active = None
            return False
        self._active = None
        node = policy.nodes.get(public_key(request))
        edges = [edge for edge in node['edges'] if edge['action_key'] == action_key(action)] if node else []
        if len(edges) != 1 or 'child' not in edges[0]:
            return False
        after, slots = edges[0]['child']
        expected_slots = tuple(request.drink_slots)
        if action.kind is ActionKind.USE_DRINK:
            consumed = (action.source_slot, action.drink_id)
            if expected_slots.count(consumed) != 1:
                raise ValueError('public_turn_continuation_drink_slot_mismatch')
            expected_slots = tuple(pair for pair in expected_slots if pair != consumed)
        if (slots != expected_slots or after.turn != request.state.turn
                or after.turns_remaining != request.state.turns_remaining
                or after.turns_remaining <= 2 or after.terminal
                or edges[0]['child'] not in policy.nodes):
            raise ValueError('public_turn_continuation_invalid_deterministic_edge')
        self._pending = dict(saved, context=context, predecessor_plan_index=plan_index,
            plan_type=request.plan_type, ruleset_id=request.ruleset_id, utility=request.utility,
            proof=dict(before=request.state, before_drink_slots=tuple(request.drink_slots),
                action=business_action(action), expected_state=after,
                expected_drink_slots=slots, committed=False))
        return True

    def committed(self, before, action, after):
        saved = self._pending
        if saved is None:
            return
        proof = saved['proof']
        if (not proof['committed'] and before == proof['before']
                and business_action(action) == proof['action'] and after == proof['expected_state']):
            proof['committed'] = True
        else:
            self.reset()
