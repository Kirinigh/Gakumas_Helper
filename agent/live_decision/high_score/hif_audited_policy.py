"""Unchanged run-time search audit shared by fast retrieval and offline evaluation.

No batch driver, sample generation, training or command-line entry lives here.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict

from agent.live_decision.high_score.hif_mcts import (
    QUALITY_REFERENCE_VERSION,
    COMPUTE_SENSITIVITY_VERSION,
    PARTIAL_NODE_TIMING_VERSION,
    SEARCH_PHASE_TIMING_VERSION,
    LAZY_VALIDATION_SEEDS_VERSION,
    PUBLIC_TURN_BUDGET_RESERVATION_VERSION,
    PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION,
    PUBLIC_VALIDATION_PHASE_BUDGET_VERSION,
)
from agent.live_decision.high_score.hif_combo_timing import VERSION as TIMING_VERSION
from agent.live_decision.high_score.hif_mcts_candidate import CONFIG as CANDIDATE_CONFIG
from agent.live_decision.high_score.hif_mcts_candidate import ACTUAL_TAIL_SESSION_REUSE_VERSION, PUBLIC_TURN_SCORE_TRANSLATION_VERSION
from agent.live_decision.high_score.hif_pair_order_advice import VERSION as PAIR_ORDER_VERSION
from agent.live_decision.high_score.hif_public_step_reuse import VERSION as PUBLIC_STEP_REUSE_VERSION
from agent.live_decision.high_score.hif_validation_cohort import VERSION as COHORT_VERSION
from agent.live_decision.high_score.hif_simple_tail_domain import FINAL_DOUBLE_VERSION, FINAL_OFFHAND_REMOVE_VERSION
from agent.live_decision.high_score.hif_internal_reanalysis import VERSION as INTERNAL_REANALYSIS_VERSION
from agent.live_decision.high_score.hif_internal_reanalysis import COMPLETE_ROOT_SUCCESSORS_VERSION
from agent.live_decision.high_score.hif_policy_continuation import UNKNOWN_TIMING_VERSION, RetrievalTrainingOrder
from agent.live_decision.high_score.hif_resource_reanalysis import VERSION as RESOURCE_REANALYSIS_VERSION
from agent.live_decision.high_score.hif_resource_reanalysis import BUDGET_DECLINE_REUSE_VERSION
from agent.live_decision.high_score.hif_public_graph_parallel import VERSION as PARALLEL_PUBLIC_GRAPH_VERSION
from agent.live_decision.high_score.hif_public_graph_parallel import MIN_SIMULATIONS as PARALLEL_PUBLIC_GRAPH_MIN_SIMULATIONS
from agent.live_decision.high_score.hif_resource_continuation import VERSION as RESOURCE_CONTINUATION_VERSION
from agent.live_decision.high_score.hif_public_turn_reanalysis import VERSION as PUBLIC_TURN_VERSION
from agent.live_decision.high_score.hif_public_turn_reanalysis import SCORE_TRANSLATION_CONTRACT
from agent.live_decision.high_score.hif_turn_chain_continuation import FINITE_CONSUMABLES_VERSION
from agent.live_decision.high_score.hif_internal_successor_graph import VERSION as INTERNAL_SUCCESSOR_GRAPH_VERSION
from agent.live_decision.high_score.hif_public_turn_continuation import VERSION as PUBLIC_CONTINUATION_VERSION
from agent.live_decision.high_score.hif_root_validation_parallel import VERSION as PARALLEL_ROOT_VALIDATION_VERSION
from agent.live_decision.high_score.hif_root_validation_parallel import MIN_SIMULATIONS as PARALLEL_ROOT_VALIDATION_MIN_SIMULATIONS
from agent.live_decision.high_score.hif_end_turn_focus_projection import VERSION as END_TURN_FOCUS_VERSION
from agent.live_decision.high_score.hif_extra_action_continuation import VERSION as EXTRA_ACTION_VERSION
from agent.live_decision.high_score.hif_extra_action_continuation import VISITED_CARD_VERSION, REFERENCE_VISITED_CARD_VERSION
from agent.live_decision.high_score.hif_paired_adaptive_validation import VERSION as PAIRED_ADAPTIVE_VALIDATION_VERSION
from agent.live_decision.high_score.hif_retained_policy_competition import REVEALED_VERSION as REVEALED_POLICY_COMPETITION_VERSION
from agent.live_decision.high_score.hif_retained_policy_competition import TREE_EXECUTION_VERSION as RETAINED_TREE_POLICY_EXECUTION_VERSION
from agent.live_decision.high_score.hif_installed_score_continuation import helper_version as installed_score_helper_version
from agent.live_decision.high_score.hif_uniform_validation_time_guard import VERSION as UNIFORM_VALIDATION_TIME_GUARD_VERSION


def _json_value(value):
    return json.loads(json.dumps(value))


RETAINED_POLICY_COMPETITION_VERSION = 'hif-retained-policy-competition-v1'


RETAINED_POLICY_EXECUTION_VERSION = 'hif-retained-policy-execution-v1'


class _AuditedMctsPolicy:
    """Verify forced-action claims against the evaluator's actual legal actions."""

    def __init__(self, policy):
        self.policy = policy
        self._last_choice = None
        self._last_committed = None
        self._public_turn_source = None
        self._public_turn_arrival = None
        self._resource_source = self._resource_arrival = self._resource_cursor = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def choose(self, request, legal):
        started = time.perf_counter()
        actual_legal = tuple(legal)
        visited_card_version = getattr(self.policy, 'visited_card_extra_action_version', None)
        previous = len(self.policy.plans)
        controller = getattr(self.policy, '_retained_policy', None)
        frozen = getattr(controller, 'frozen', None)
        execution_source = getattr(controller, 'execution_source_plan_index', None)
        graph_execution_source = getattr(controller, 'graph_execution_source_plan_index', None)
        self._revealed_context_matches = (not getattr(self.policy, 'revealed_policy_competition', False)
            or frozen is None or getattr(controller, 'context', None) == self.policy._cache_context())
        action = self.policy.choose(request, legal)
        for plan_index, plan in enumerate(self.policy.plans[previous:], previous):
            if plan.get('scope') == 'SAMPLING_ESTIMATE':
                if not _extra_action_turn_chain_evidence(plan,
                        expected=getattr(self.policy, 'extra_action_turn_chain', False),
                        expected_visited_card=getattr(self.policy, 'visited_card_extra_action', False),
                        expected_visited_card_version=visited_card_version):
                    raise ValueError('mcts_extra_action_turn_chain_evidence_invalid')
                valid = _root_selection_eligibility_evidence(plan,
                    expected=getattr(self.policy, 'visited_card_extra_action', False), verified=False,
                    expected_visited_card_version=visited_card_version)
                if valid and plan.get('root_selection_eligibility') is not None:
                    plan['root_selection_eligibility_verified'] = False
                    valid = _verify_root_selection_eligibility(self.policy.kernel, request, actual_legal, plan,
                        deadline=min(self.policy.deadline, started + self.policy.window_seconds))
                    plan['root_selection_eligibility_verified'] = valid
                if not valid:
                    raise ValueError('mcts_root_selection_eligibility_evidence_invalid')
            quality = getattr(self.policy, 'quality_reference', False)
            if not _quality_reference_evidence(plan, expected=quality,
                    window_seconds=getattr(self.policy, 'window_seconds', 30.)):
                raise ValueError('mcts_quality_reference_evidence_invalid')
            if not _compute_sensitivity_evidence(plan,
                    expected=getattr(self.policy, 'compute_sensitivity', False),
                    simulations=getattr(getattr(self.policy, 'search', None), 'simulations', 256)):
                raise ValueError('mcts_compute_sensitivity_evidence_invalid')
            if (type(plan.get('fresh_adaptive_training', False)) is not bool
                    or plan.get('fresh_adaptive_training', False)
                        is not getattr(self.policy, 'fresh_adaptive_training', False)):
                raise ValueError('mcts_fresh_adaptive_training_flag_mismatch')
            if not _paired_adaptive_validation_evidence(plan,
                    expected=getattr(self.policy, 'paired_adaptive_validation', False)):
                raise ValueError('mcts_paired_adaptive_validation_evidence_invalid')
            if not _uniform_validation_time_guard_evidence(plan,
                    expected=getattr(self.policy, 'uniform_validation_time_guard', False),
                    expected_visited_card_version=visited_card_version):
                raise ValueError('mcts_uniform_validation_time_guard_evidence_invalid')
            if quality and not _quality_reference_work_evidence(plan):
                raise ValueError('mcts_quality_reference_work_incomplete')
            # The cohort audit consumes public-table coverage before the old
            # joint-root audit below. Prove the new combined domain first.
            joint_roots_checked = (plan.get('public_turn_optional_end_roots_version')
                == PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION and plan.get('root_options_enabled') is True)
            if joint_roots_checked:
                verified = _verify_joint_roots(self.policy.kernel, request, actual_legal, plan)
                plan['joint_root_options_verified'] = verified
                if not verified:
                    raise ValueError('mcts_joint_roots_not_complete')
            if not _retained_policy_comparison_evidence(plan,
                    expected=getattr(self.policy, 'retained_policy_competition', False)):
                raise ValueError('mcts_retained_policy_comparison_evidence_invalid')
            if not _revealed_policy_mode_evidence(plan,
                    expected=getattr(self.policy, 'revealed_policy_competition', False)):
                raise ValueError('mcts_revealed_policy_competition_evidence_invalid')
            if plan.get('retained_policy_competition_enabled') is True:
                valid = self._verify_retained_policy_lineage(request, plan, plan_index,
                    deadline=min(self.policy.deadline, started + self.policy.window_seconds))
                plan['retained_policy_lineage_verified'] = valid
                if not valid:
                    raise ValueError('mcts_retained_policy_lineage_not_verified')
            if not _retained_policy_execution_evidence(self.policy.plans, plan_index, plan,
                    expected=getattr(self.policy, 'retained_policy_execution', False),
                    expected_tree=getattr(self.policy, 'retained_tree_policy_execution', False),
                    plan_offset=getattr(self.policy, '_plan_offset', 0), verified=False):
                raise ValueError('mcts_retained_policy_execution_evidence_invalid')
            if plan.get('scope') == 'RETAINED_POLICY_CONTINUATION':
                try:
                    valid = self._verify_retained_policy_execution(request, actual_legal, action, plan,
                        frozen, execution_source, graph_execution_source=graph_execution_source,
                        deadline=min(self.policy.deadline, started + self.policy.window_seconds))
                except TimeoutError as exc:
                    plan.update(accepted=False, retained_policy_execution_verified=False,
                        stop_reason='retained_policy_execution_verification_timeout')
                    raise ValueError('mcts_retained_policy_execution_verification_timeout') from exc
                plan['retained_policy_execution_verified'] = valid
                if not valid:
                    raise ValueError('mcts_retained_policy_execution_not_verified')
            if not _actual_tail_session_reuse_evidence(self.policy.plans, plan_index, plan,
                    expected=getattr(self.policy, 'actual_tail_session_reuse', False),
                    plan_offset=getattr(self.policy, '_plan_offset', 0), verified=False):
                raise ValueError('mcts_actual_tail_session_reuse_evidence_invalid')
            if plan.get('actual_tail_session_reuse_enabled') is True:
                valid = self._verify_actual_tail_session_reuse(request, plan, plan_index)
                plan['actual_tail_session_reuse_verified'] = valid
                if not valid:
                    raise ValueError('mcts_actual_tail_session_reuse_not_verified')
            if not _public_turn_score_translation_evidence(plan,
                    expected=getattr(self.policy, 'public_turn_score_translation', False)):
                raise ValueError('mcts_public_turn_score_translation_evidence_invalid')
            if not _public_turn_budget_reservation_evidence(plan,
                    expected=getattr(self.policy, 'public_turn_budget_reservation', False)):
                raise ValueError('mcts_public_turn_budget_reservation_evidence_invalid')
            if not _public_validation_phase_budget_evidence(plan,
                    expected=getattr(self.policy, 'public_validation_phase_budget', False),
                    expected_soft_window_ms=getattr(getattr(self.policy, 'search', None), 'soft_timeout_ms', None)):
                raise ValueError('mcts_public_validation_phase_budget_evidence_invalid')
            if not _parallel_public_graph_evidence(plan,
                    expected=getattr(self.policy, 'parallel_public_graph', False),
                    expected_visited_card_version=visited_card_version):
                raise ValueError('mcts_parallel_public_graph_evidence_invalid')
            if not _parallel_root_validation_evidence(plan,
                    expected=(getattr(self.policy, 'parallel_public_graph', False) is True
                        and (getattr(self.policy, 'paired_adaptive_validation', False) is True
                             or getattr(self.policy, 'uniform_validation_time_guard', False) is True)),
                    expected_visited_card_version=visited_card_version):
                raise ValueError('mcts_parallel_root_validation_evidence_invalid')
            if not _resource_reanalysis_evidence(plan,
                    expected=getattr(self.policy, 'resource_reanalysis', False)):
                raise ValueError('mcts_resource_reanalysis_evidence_invalid')
            if not _internal_reanalysis_evidence(plan,
                    expected=getattr(self.policy, 'internal_reanalysis', False), verified=False,
                    expected_complete_roots=getattr(self.policy, 'internal_complete_root_successors', False),
                    expected_successor_graph=getattr(self.policy, 'internal_successor_graph', False)):
                raise ValueError('mcts_internal_reanalysis_evidence_invalid')
            if not _final_double_tail_evidence(plan,
                    expected=getattr(self.policy, 'final_double_tail_evaluation', False)):
                raise ValueError('mcts_final_double_tail_evidence_invalid')
            if not _final_offhand_remove_tail_evidence(plan,
                    expected=getattr(self.policy, 'final_offhand_remove_tail_evaluation', False)):
                raise ValueError('mcts_final_offhand_remove_tail_evidence_invalid')
            if (plan.get('internal_reanalysis') or {}).get('complete') is True:
                valid = self._verify_internal_reanalysis(request, actual_legal, plan,
                    deadline=min(self.policy.deadline, started + self.policy.window_seconds))
                plan['internal_reanalysis_verified'] = valid
                if not valid:
                    raise ValueError('mcts_internal_reanalysis_public_node_not_verified')
            if not _public_turn_continuation_enabled(plan,
                    expected=getattr(self.policy, 'public_turn_policy_continuation', False)):
                raise ValueError('mcts_public_turn_continuation_configuration_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _unknown_node_timing_evidence(
                    plan, expected=getattr(self.policy, 'unknown_node_timing', False)):
                raise ValueError('mcts_unknown_node_timing_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _retrieval_training_order_evidence(
                    plan, expected=getattr(self.policy, 'retrieval_training_order', False)):
                raise ValueError('mcts_retrieval_training_order_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _lazy_validation_seed_evidence(
                    plan, expected=getattr(self.policy, 'lazy_validation_seeds', False)):
                raise ValueError('mcts_lazy_validation_seed_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _validation_cohort_evidence(
                    plan, expected=getattr(self.policy, 'deterministic_validation_cohort', False),
                    plans=self.policy.plans, index=plan_index,
                    plan_offset=getattr(self.policy, '_plan_offset', 0)):
                raise ValueError('mcts_validation_cohort_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _partial_node_timing_evidence(
                    plan, expected=getattr(self.policy, 'partial_node_timing', False)):
                raise ValueError('mcts_partial_node_timing_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _installed_score_turn_chain_evidence(
                    plan, expected=getattr(self.policy, 'installed_score_turn_chain', False),
                    expected_multi_action=getattr(self.policy, 'installed_score_multi_action', False),
                    expected_focus_projection=getattr(self.policy, 'installed_score_focus_projection', False)):
                raise ValueError('mcts_installed_score_turn_chain_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _extra_action_turn_chain_evidence(
                    plan, expected=getattr(self.policy, 'extra_action_turn_chain', False),
                    expected_visited_card=getattr(self.policy, 'visited_card_extra_action', False),
                    expected_visited_card_version=visited_card_version):
                raise ValueError('mcts_extra_action_turn_chain_evidence_invalid')
            if plan.get('scope') == 'SAMPLING_ESTIMATE' and not _pair_order_advice_evidence(
                    plan, expected=getattr(self.policy, 'pair_order_advice', False)):
                raise ValueError('mcts_pair_order_advice_evidence_invalid')
            if plan.get('scope') == 'LOCAL_PUBLIC_RESOURCE_CONTINUATION':
                verified = self._verify_resource_continuation(request, actual_legal, action, plan, plan_index)
                plan['resource_policy_continuation_verified'] = verified
                if not verified:
                    raise ValueError('mcts_resource_policy_continuation_not_verified')
                continue
            if plan.get('scope') == 'PUBLIC_TURN_POLICY_CONTINUATION':
                verified = self._verify_public_turn_continuation(request, actual_legal, action, plan, plan_index)
                plan['public_turn_policy_continuation_verified'] = verified
                if not verified:
                    raise ValueError('mcts_public_turn_policy_continuation_not_verified')
                continue
            if plan.get('scope') == 'FROZEN_POLICY_CONTINUATION':
                verified = _verify_frozen_policy_continuation(
                    self.policy.kernel, request, actual_legal, action, plan,
                    self.policy.plans, previous, self._last_committed,
                    deadline=min(self.policy.deadline, started + self.policy.window_seconds))
                plan['frozen_policy_continuation_verified'] = verified
                if not verified:
                    raise ValueError('mcts_frozen_policy_continuation_not_verified')
                continue
            if plan.get('exact_root_turns') == 2:
                verified = _verify_exact_two_turn_root(self.policy.kernel, request, actual_legal, action, plan)
                plan['exact_root_verified'] = verified
                if not verified:
                    raise ValueError('mcts_exact_two_turn_root_not_verified')
                continue
            if plan.get('scope') == 'JOINT_ROOT_CONTINUATION':
                origin_index = plan.get('source_plan_index')
                origin = (self.policy.plans[origin_index]
                          if type(origin_index) is int and 0 <= origin_index < previous else {})
                continuation = origin.get('selected_continuation', [])
                optional_end = (plan.get('stop_reason') == 'matched_public_optional_end'
                    and origin.get('stop_after_card_roots') is True
                    and origin.get('action', {}).get('kind') == 'PLAY_CARD'
                    and len(continuation) == 1 and continuation[0].get('kind') == 'END_TURN'
                    and request.state.pending_choice is None and not request.state.resolution_stack
                    and request.state.turn == origin.get('turn') and request.state.actions_remaining > 0)
                mandatory = (plan.get('stop_reason') == 'matched_public_mandatory_choice'
                    and request.state.pending_choice is not None)
                verified = (origin.get('accepted') is True and origin.get('root_options_enabled') is True
                    and len(continuation) == 1 and (mandatory or optional_end)
                    and _json_value(asdict(self.policy.kernel._canonical_probability_state(request.state)))
                        == _json_value(origin.get('selected_expected_after_first'))
                    and _action_identity(asdict(action)) == _action_identity(continuation[0])
                    and sum(_action_identity(asdict(option)) == _action_identity(continuation[0])
                            for option in actual_legal) == 1)
                plan['joint_continuation_verified'] = verified
                if not verified:
                    raise ValueError('mcts_joint_continuation_not_verified')
                continue
            if plan.get('root_options_enabled') and not joint_roots_checked:
                verified = _verify_joint_roots(self.policy.kernel, request, actual_legal, plan)
                plan['joint_root_options_verified'] = verified
                if not verified:
                    raise ValueError('mcts_joint_roots_not_complete')
            if plan.get('scope') == 'PROVEN_RESOURCE_DOMINANCE':
                from agent.live_decision.high_score.hif_dominance import prove_forced_end_drink_dominance
                proof = next((value for option in actual_legal
                              if (value := prove_forced_end_drink_dominance(self.policy.kernel, request, option))
                              is not None), None)
                verified = (proof is not None and action == proof.dominant_action
                            and len(actual_legal) == 2 and set(actual_legal) == set(proof.legal_actions)
                            and _json_value(asdict(proof)) == _json_value(plan.get('dominance_proof')))
                plan.update(actual_legal_action_count=len(actual_legal), resource_dominance_verified=verified)
                if not verified:
                    raise ValueError('mcts_resource_dominance_not_proven')
                continue
            if plan.get('scope') != 'FORCED_LEGAL_ACTION':
                continue
            verified = len(actual_legal) == 1 and action == actual_legal[0]
            plan.update(actual_legal_action_count=len(actual_legal), forced_legal_action_verified=verified)
            if not verified:
                raise ValueError('mcts_forced_action_not_uniquely_legal')
        if getattr(self.policy, 'public_turn_policy_continuation', False):
            current = self.policy.plans[-1]
            if current.get('scope') == 'SAMPLING_ESTIMATE':
                self._public_turn_source = self._public_turn_arrival = None
                if current.get('public_turn_policy_link_created') is True:
                    self._capture_public_turn_source(request, action, len(self.policy.plans) - 1,
                                                    self.policy.search._public_turn_policy)
        if getattr(self.policy, 'resource_reanalysis', False):
            current = self.policy.plans[-1]
            retained_won = (current.get('retained_policy_comparison') or {}).get('selected') is True
            if (current.get('resource_reanalysis') or {}).get('complete') is True and not retained_won:
                self._capture_resource_source(request, action, len(self.policy.plans) - 1,
                                              self.policy.search._resource_policy)
            if current.get('resource_policy_link_created') is not True:
                self._resource_source = self._resource_arrival = self._resource_cursor = None
        self._last_choice = (request, action, len(self.policy.plans) - 1)
        return action

    def _verify_internal_reanalysis(self, request, legal, plan, *, deadline):
        """Recreate only the deterministic edge and complete legal choices; no draws."""
        if plan.get('internal_complete_root_successors_enabled') is True:
            return self._verify_internal_complete_root(request, legal, plan, deadline=deadline)
        from dataclasses import replace

        from agent.live_decision.high_score.planner import SearchIncomplete
        from agent.live_decision.high_score.hif_mcts import action_key
        from agent.live_decision.high_score.hif_mcts_cache import public_request
        from agent.live_decision.high_score.hif_environment import HifEnvironment
        from agent.live_decision.high_score.hif_public_turn_reanalysis import _StrictDeterministicKernel

        evidence = _json_value(plan['internal_reanalysis'])
        matches = [action for action in legal if _json_value(action_key(action)) == evidence['root_action']]
        if len(matches) != 1 or time.perf_counter() >= deadline:
            return False
        action = matches[0]
        kernel = self.policy.kernel
        proof = _StrictDeterministicKernel(kernel.cards, kernel.drinks, kernel.items,
                                         memory_abilities=kernel.memory_abilities)
        try:
            branches = proof.transition(request.state, replace(action, source_slot=None),
                                        plan_type=request.plan_type)
        except SearchIncomplete:
            return False
        if len(branches) != 1 or branches[0].probability != 1.:
            return False
        after = kernel._canonical_probability_state(branches[0].state)
        if (after.terminal or after.turn != request.state.turn
                or after.turns_remaining != request.state.turns_remaining or after.actions_remaining <= 0
                or not after.deck.hand or after.pending_choice is not None or after.resolution_stack
                or after.pending_after_card_ids):
            return False
        slots = tuple(request.drink_slots)
        if action.kind.value == 'USE_DRINK':
            slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
        environment = HifEnvironment(kernel)
        rebound = environment.reset(after, seed=0, drink_slots=slots)
        current = replace(request, state=rebound.state, card_slots=rebound.card_slots,
                          drink_slots=rebound.drink_slots, decision_kind=rebound.decision_kind)
        keys = _json_value(tuple(action_key(item) for item in environment.legal_actions()))
        valid = (public_request(current) == evidence['public_request']
            and keys == evidence['legal_action_keys'] and time.perf_counter() < deadline)
        if valid and (evidence.get('successor_graph') or {}).get('complete') is True:
            return self._verify_internal_successor_nodes(current, plan, deadline=deadline)
        return valid

    def _verify_internal_successor_nodes(self, request, plan, *, deadline):
        """Bind saved mean evidence to the actual immutable table and full legal sets."""
        from agent.live_decision.high_score.hif_mcts import action_key
        from agent.live_decision.high_score.hif_mcts_cache import public_request
        from agent.live_decision.high_score.hif_environment import HifEnvironment
        from agent.live_decision.high_score.hif_choice_roots import _complete_legal
        from agent.live_decision.high_score.hif_internal_successor_graph import GraphDecision, _public_nodes

        decision = getattr(getattr(self.policy, 'search', None), '_internal_policy', None)
        evidence = plan['internal_reanalysis']
        if (not isinstance(decision, GraphDecision) or not decision.complete
                or decision.version != INTERNAL_SUCCESSOR_GRAPH_VERSION
                or _json_value(_public_nodes(decision.graph)) != _json_value(evidence['successor_public_nodes'])
                or _json_value(decision.action_key) != _json_value(evidence['selected_action'])
                or _json_value(decision.means) != _json_value(evidence['means'])
                or _json_value(decision.planning_seeds) != _json_value(evidence['planning_seeds'])
                or public_request(decision.graph.nodes[decision.graph.root_key]['request']) != public_request(request)):
            return False
        try:
            for node in decision.graph.nodes.values():
                if time.perf_counter() >= deadline:
                    return False
                current = node['request']
                environment = HifEnvironment(self.policy.kernel)
                environment.reset(current.state, seed=0, drink_slots=current.drink_slots)
                choices = tuple(environment.legal_actions())
                _complete_legal(self.policy.kernel, current, choices)
                if {action_key(action) for action in choices} != {edge['action_key'] for edge in node['edges']}:
                    return False
            return time.perf_counter() < deadline
        except (KeyError, TypeError, ValueError):
            return False

    def _verify_internal_complete_root(self, request, legal, plan, *, deadline):
        """Independently prove every edge of the recorded existing root option."""
        from dataclasses import replace

        from agent.live_decision.high_score.kernel import KernelError
        from agent.live_decision.high_score.planner import SearchIncomplete
        from agent.live_decision.high_score.hif_mcts import action_key
        from agent.live_decision.high_score.hif_mcts_cache import public_request
        from agent.live_decision.high_score.hif_environment import HifEnvironment
        from agent.live_decision.high_score.hif_public_turn_reanalysis import _StrictDeterministicKernel
        from agent.live_decision.high_score.hif_frozen_policy_continuation import deterministic_public_successor

        if not _internal_complete_root_successors_evidence(plan, expected=True):
            return False
        evidence = _json_value(plan['internal_reanalysis'])
        kernel = self.policy.kernel
        proof = _StrictDeterministicKernel(kernel.cards, kernel.drinks, kernel.items,
                                         memory_abilities=kernel.memory_abilities)
        current, choices = request, tuple(legal)
        prefix = (evidence['root_action'], *evidence['root_continuation'])
        try:
            for index, key in enumerate(prefix):
                matches = [action for action in choices if _json_value(action_key(action)) == key]
                if len(matches) != 1 or time.perf_counter() >= deadline:
                    return False
                action = matches[0]
                after, _ = deterministic_public_successor(kernel, current.state, action,
                    plan_type=current.plan_type, deadline=deadline)
                if after is None:
                    return False
                branches = proof.transition(current.state, replace(action, source_slot=None),
                                            plan_type=current.plan_type)
                if (len(branches) != 1 or branches[0].probability != 1.
                        or kernel._canonical_probability_state(branches[0].state) != after
                        or index == 0 and len(prefix) > 1
                            and _json_value(asdict(after)) != evidence['root_expected_after_first']):
                    return False
                slots = tuple(current.drink_slots)
                if action.kind.value == 'USE_DRINK':
                    slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
                environment = HifEnvironment(kernel)
                rebound = environment.reset(after, seed=0, drink_slots=slots)
                current = replace(request, state=rebound.state, card_slots=rebound.card_slots,
                    drink_slots=rebound.drink_slots, decision_kind=rebound.decision_kind)
                choices = tuple(environment.legal_actions())
        except (SearchIncomplete, KernelError, ValueError):
            return False
        after = current.state
        valid = (not after.terminal and after.turn == request.state.turn
            and after.turns_remaining == request.state.turns_remaining and after.actions_remaining > 0
            and bool(after.deck.hand) and after.pending_choice is None and not after.resolution_stack
            and not after.pending_after_card_ids and public_request(current) == evidence['public_request']
            and _json_value(tuple(action_key(action) for action in choices)) == evidence['legal_action_keys']
            and time.perf_counter() < deadline)
        if valid and (evidence.get('successor_graph') or {}).get('complete') is True:
            return self._verify_internal_successor_nodes(current, plan, deadline=deadline)
        return valid

    def _verify_retained_policy_lineage(self, request, plan, index, *, deadline):
        from agent.live_decision.high_score.hif_validation_cohort import DeterministicValidationCohort

        offset = getattr(self.policy, '_plan_offset', 0)
        plan['retained_policy_commit_proof'] = None
        if not _retained_policy_lineage_evidence(self.policy.plans, index, plan,
                plan_offset=offset, verified=False):
            return False
        if plan.get('revealed_policy_competition_enabled') is True:
            return self._verify_revealed_policy_commit(request, plan, index, deadline=deadline)
        lineage = plan['retained_policy_lineage']
        if lineage['source_plan_index'] is None:
            return True
        if self._last_committed is None:
            return False
        before, action, after, previous_index = self._last_committed
        if (previous_index != index - 1 or after != request.state
                or lineage['predecessor_plan_index'] != offset + previous_index
                or _action_identity(asdict(action))
                    != _action_identity(self.policy.plans[previous_index].get('action', {}))):
            return False
        seeds = plan['validation_cohort_link']['seeds']
        # This second helper sees the evaluator's actual committed request and
        # transition, independently of the candidate controller's saved link.
        proof = DeterministicValidationCohort(self.policy.kernel)
        context = (before.ruleset_id, before.plan_type, before.utility)
        if not proof.remember(before, action, seeds, context, deadline, offset + previous_index):
            return False
        proof.committed(before.state, action, after)
        actual = proof.prepare(request, (request.ruleset_id, request.plan_type, request.utility))
        if tuple(actual) != tuple(seeds):
            return False
        plan['retained_policy_commit_proof'] = dict(before=asdict(before.state),
            action=asdict(action), after=asdict(after), before_drink_slots=before.drink_slots,
            after_drink_slots=request.drink_slots, source_plan_index=lineage['source_plan_index'],
            predecessor_plan_index=lineage['predecessor_plan_index'])
        return True

    def _verify_revealed_policy_commit(self, request, plan, index, *, deadline):
        from agent.live_decision.high_score.hif_mcts_cache import public_request
        from agent.live_decision.high_score.hif_validation_cohort import DeterministicValidationCohort

        evidence = plan['revealed_policy_competition']
        if evidence['mode'] == 'none':
            return True
        if self._last_committed is None or not self._revealed_context_matches:
            return False
        before, action, after, previous_index = self._last_committed
        offset = getattr(self.policy, '_plan_offset', 0)
        if (previous_index != index - 1 or after != request.state
                or evidence['predecessor_plan_index'] != offset + previous_index
                or before.ruleset_id != request.ruleset_id or request.ruleset_id != self.policy.kernel.ruleset_id
                or before.protocol_version != request.protocol_version
                or before.plan_type != request.plan_type or before.utility != request.utility
                or asdict(action) != self.policy.plans[previous_index].get('action')):
            return False
        actual = dict(before=public_request(before), before_card_slots=before.card_slots,
            action=asdict(action), after=public_request(request), after_card_slots=request.card_slots,
            source_plan_index=evidence['source_plan_index'], predecessor_plan_index=offset + previous_index,
            committed=True, ruleset_id=request.ruleset_id, protocol_version=request.protocol_version)
        if _json_value(actual) != _json_value(evidence['proof']):
            return False
        if evidence['mode'] == 'deterministic_continuation':
            seeds = plan['validation_cohort_link']['seeds']
            proof = DeterministicValidationCohort(self.policy.kernel)
            if not proof.remember(before, action, seeds,
                    (before.ruleset_id, before.plan_type, before.utility), deadline, offset + previous_index):
                return False
            proof.committed(before.state, action, after)
            if tuple(proof.prepare(request, (request.ruleset_id, request.plan_type, request.utility))) != tuple(seeds):
                return False
        if time.perf_counter() >= deadline:
            return False
        plan['retained_policy_commit_proof'] = actual
        return True

    def _verify_retained_policy_execution(self, request, legal, action, plan,
                                          frozen, execution_source, *, deadline, graph_execution_source=None):
        # Use the pre-choose frozen object: the just-selected action may make
        # its next successor random and legitimately clear the controller.
        if 'internal_successor_graph_execution' in plan:
            from dataclasses import replace

            from agent.live_decision.high_score.hif_mcts import action_key
            from agent.live_decision.high_score.hif_mcts_cache import public_request
            from agent.live_decision.high_score.hif_internal_successor_graph import lookup

            proof = plan['internal_successor_graph_execution']
            if frozen is None or graph_execution_source != proof['source_plan_index']:
                return False
            search = frozen.policy.search
            current = replace(request, state=self.policy.kernel._canonical_probability_state(request.state))
            if (public_request(current) != proof['public_request']
                    or _json_value(current.card_slots) != _json_value(proof['card_slots'])
                    or _json_value(tuple(action_key(item) for item in legal)) != _json_value(proof['legal_action_keys'])
                    or lookup(search._internal_policy, current, legal,
                        context=search._context(current), kernel=search.kernel) != action_key(action)):
                return False
            execution_source = graph_execution_source
        return (frozen is not None
            and frozen.source_plan_index == plan['retained_policy_lineage']['source_plan_index']
            and execution_source == plan['retained_policy_execution_source_plan_index']
            and action in legal and asdict(action) == plan.get('action')
            and frozen.bind(request, legal, deadline).action == action)

    def _verify_actual_tail_session_reuse(self, request, plan, index):
        evidence = plan['actual_tail_session_reuse']
        if request.state.turns_remaining != 2:
            return (not evidence['reused'] and evidence['entries_before'] == 0
                    and evidence['entries_after'] == 0)
        if not evidence['reused']:
            return True
        if self._last_committed is None:
            return False
        before, action, after, previous_index = self._last_committed
        canonical = self.policy.kernel._canonical_probability_state
        slots = tuple(before.drink_slots)
        if action.kind.value == 'USE_DRINK':
            consumed = (action.source_slot, action.drink_id)
            if slots.count(consumed) != 1:
                return False
            slots = tuple(pair for pair in slots if pair != consumed)
        return (previous_index == index - 1
            and before.state.turn == after.turn == request.state.turn == plan.get('turn')
            and before.state.turns_remaining == after.turns_remaining == 2
            and canonical(after) == canonical(request.state)
            and before.plan_type == request.plan_type and before.utility == request.utility
            and before.ruleset_id == request.ruleset_id and before.decision_kind == request.decision_kind
            and slots == tuple(request.drink_slots)
            and _action_identity(asdict(action))
                == _action_identity(self.policy.plans[index - 1].get('action', {})))

    def _capture_public_turn_source(self, request, action, plan_index, table):
        """Keep independent accepted evidence, including when replaying a saved source."""
        from agent.live_decision.high_score.hif_mcts import action_key
        from agent.live_decision.high_score.hif_public_turn_reanalysis import public_key

        plan = self.policy.plans[plan_index]
        roots = plan.get('candidate_statistics', [])
        if (not getattr(self.policy, 'public_turn_policy_continuation', False)
                or not _qualified_public_turn_source(plan) or table is None
                or table.root_key != public_key(request)
                or table.root_key not in table.nodes
                or _action_identity(plan.get('action', {})) != action_key(action)
                or {edge['action_key'] for edge in table.nodes[table.root_key]['edges']}
                    != {_action_identity(row['action']) for row in roots}
                or _json_value(table.choices[table.root_key]['options'])
                    != _json_value(plan['public_turn_reanalysis']['root_options'])):
            raise ValueError('mcts_public_turn_source_not_verified')
        self._public_turn_source = (plan_index, copy.deepcopy(table))
        self._public_turn_arrival = (table.root_key, plan_index - 1)

    def _capture_resource_source(self, request, action, plan_index, table):
        from agent.live_decision.high_score.hif_environment import HifEnvironment
        from agent.live_decision.high_score.hif_choice_roots import _complete_legal
        from agent.live_decision.high_score.hif_resource_reanalysis import ResourcePolicy

        plan = self.policy.plans[plan_index]
        if (not _qualified_resource_source(plan) or type(table) is not ResourcePolicy
                or table.version != RESOURCE_REANALYSIS_VERSION
                or _action_identity(plan.get('action', {})) != _action_identity(asdict(action))
                or len(table.decisions) != plan['resource_reanalysis']['graph_count']):
            raise ValueError('mcts_resource_source_not_verified')
        for decision, row in zip(table.decisions, plan['resource_reanalysis']['graphs'], strict=True):
            if (decision.graph_index != row['graph_index']
                    or _json_value(decision.evidence()['public_nodes']) != _json_value(row['public_nodes'])
                    or _json_value(asdict(decision.normalized_request)) != _json_value(row['normalized_request'])
                    or _json_value(decision.origin_indices) != _json_value(row['origin_indices'])):
                raise ValueError('mcts_resource_source_not_verified')
            for key, node in decision.graph.nodes.items():
                environment = HifEnvironment(self.policy.kernel)
                environment.reset(node['request'].state, seed=0, drink_slots=node['request'].drink_slots)
                legal = tuple(environment.legal_actions())
                _complete_legal(self.policy.kernel, node['request'], legal)
                choices = decision.graph.choices[key]
                if ({_action_identity(asdict(option)) for option in legal}
                        != {edge['action_key'] for edge in node['edges']}
                        or _json_value(choices) != _json_value(next(
                            {name: recorded[name] for name in ('options', 'mean_final_score', 'maximizing_actions')}
                            for recorded in row['public_nodes']
                            if _resource_request_key(recorded['request']) == _resource_request_key(asdict(node['request']))))):
                    raise ValueError('mcts_resource_source_not_verified')
        # Tables are immutable. Never deepcopy their key callable, which is
        # bound to the live search object and would copy its entire tree.
        self._resource_source = (plan_index, table)
        self._resource_arrival = (request.state, tuple(request.drink_slots), plan_index - 1)
        self._resource_cursor = None

    def _verify_resource_continuation(self, request, legal, action, plan, index):
        from agent.live_decision.high_score.hif_mcts_cache import public_request
        from agent.live_decision.high_score.hif_choice_roots import _complete_legal

        offset = getattr(self.policy, '_plan_offset', 0)
        if (not _resource_continuation_evidence(self.policy.plans, index, plan,
                verified=False, plan_offset=offset) or self._resource_source is None
                or self._last_committed is None):
            return False
        source, table = self._resource_source
        before, previous_action, after, previous_index = self._last_committed
        proof = plan['continuation_proof']
        if (source + offset != plan['source_plan_index'] or previous_index != index - 1
                or self._resource_arrival != (request.state, tuple(request.drink_slots), index - 1)
                or after != request.state or before.utility != request.utility
                or before.plan_type != request.plan_type or before.ruleset_id != request.ruleset_id
                or _json_value(proof['before']) != _json_value(asdict(before.state))
                or _json_value(proof['before_drink_slots']) != _json_value(before.drink_slots)
                or _action_identity(proof['action']) != _action_identity(asdict(previous_action))
                or _json_value(plan['resource_node']['request']) != public_request(request)):
            return False
        try:
            _complete_legal(self.policy.kernel, request, legal)
            if self._resource_cursor is None:
                self._resource_cursor = table.cursor(request, legal)
            cursor = self._resource_cursor
            if cursor is None or not cursor.active:
                return False
            key = cursor.choose(request, legal)
            choice = cursor.original.policy.choices[cursor.original.expected_key]
            node = plan['resource_node']
            decision = table.decisions[node['graph_index']]
            return (cursor.original.policy is decision.graph
                and node['score_offset'] == cursor.score_offset
                and _json_value(node['origin_indices']) == _json_value(decision.origin_indices)
                and _json_value(node['options']) == _json_value(choice['options'])
                and _json_value(node['maximizing_actions']) == _json_value(choice['maximizing_actions'])
                and action in legal and asdict(action) == plan.get('action')
                and key == _action_identity(asdict(action)))
        except (ValueError, IndexError, KeyError):
            return False

    def _commit_resource(self, request, action, after, index):
        from dataclasses import replace

        from agent.live_decision.high_score.planner import SearchIncomplete
        from agent.live_decision.high_score.hif_environment import HifEnvironment
        from agent.live_decision.high_score.hif_public_turn_reanalysis import _StrictDeterministicKernel

        if self._resource_source is None:
            return
        valid = self._resource_arrival == (request.state, tuple(request.drink_slots), index - 1)
        slots = tuple(request.drink_slots)
        if action.kind.value == 'USE_DRINK':
            consumed = (action.source_slot, action.drink_id)
            valid &= slots.count(consumed) == 1
            slots = tuple(pair for pair in slots if pair != consumed)
        kernel = self.policy.kernel
        if valid:
            proof = _StrictDeterministicKernel(kernel.cards, kernel.drinks, kernel.items,
                                               memory_abilities=kernel.memory_abilities)
            try:
                branches = proof.transition(request.state, replace(action, source_slot=None),
                                            plan_type=request.plan_type)
            except SearchIncomplete as error:
                if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                    raise
                valid = False
            else:
                valid = (len(branches) == 1 and branches[0].probability == 1.
                    and kernel._canonical_probability_state(branches[0].state) == after
                    and after.turn == request.state.turn and after.turns_remaining > 2)
        if valid and self._resource_cursor is not None:
            rebound = HifEnvironment(kernel).reset(after, seed=0, drink_slots=slots)
            current = replace(request, state=after, card_slots=rebound.card_slots,
                              drink_slots=slots, decision_kind=rebound.decision_kind)
            self._resource_cursor.advance(request, action, current, (), ())
            valid = self._resource_cursor.active
        if valid:
            self._resource_arrival = (after, slots, index)
        else:
            self._resource_source = self._resource_arrival = self._resource_cursor = None

    def _verify_public_turn_continuation(self, request, legal, action, plan, index):
        from agent.live_decision.high_score.hif_mcts import action_key
        from agent.live_decision.high_score.hif_mcts_cache import public_request
        from agent.live_decision.high_score.hif_choice_roots import _complete_legal
        from agent.live_decision.high_score.hif_public_turn_reanalysis import public_key

        offset = getattr(self.policy, '_plan_offset', 0)
        if (not _public_turn_continuation_evidence(self.policy.plans, index, plan,
                                                 verified=False, plan_offset=offset)
                or self._public_turn_source is None or self._last_committed is None):
            return False
        source_index, table = self._public_turn_source
        key = public_key(request)
        if (source_index + offset != plan['source_plan_index'] or key not in table.nodes
                or self._public_turn_arrival != (key, index - 1)):
            return False
        before, previous_action, after, previous_index = self._last_committed
        proof = plan['continuation_proof']
        if (previous_index != index - 1 or after != request.state
                or _json_value(proof['before']) != _json_value(asdict(before.state))
                or _json_value(proof['before_drink_slots']) != _json_value(before.drink_slots)
                or _action_identity(proof['action']) != action_key(previous_action)
                or request.ruleset_id != before.ruleset_id or request.plan_type != before.plan_type
                or request.utility != before.utility
                or _json_value(plan['public_turn_node']['request']) != public_request(request)
                or _json_value(plan['public_turn_node']['options']) != _json_value(table.choices[key]['options'])
                or _json_value(plan['public_turn_node']['maximizing_actions'])
                    != _json_value(table.choices[key]['maximizing_actions'])):
            return False
        try:
            _complete_legal(self.policy.kernel, request, legal)
        except ValueError:
            return False
        return (action in legal and asdict(action) == plan.get('action')
            and {action_key(option) for option in legal} == {edge['action_key'] for edge in table.nodes[key]['edges']}
            and action_key(action) == table.choices[key]['maximizing_actions'][0])

    def committed(self, before, action, after):
        # Independent actual episode evidence, not a candidate's self-report.
        self._last_committed = None
        if self._last_choice is not None:
            request, selected, index = self._last_choice
            canonical_match = (getattr(self.policy, 'actual_tail_session_reuse', False)
                and self.policy.kernel._canonical_probability_state(request.state)
                    == self.policy.kernel._canonical_probability_state(before))
            if (request.state == before or canonical_match) and selected == action:
                self._last_committed = (request, action, after, index)
                self._commit_resource(request, action, after, index)
                if self._public_turn_source is not None:
                    from agent.live_decision.high_score.hif_mcts import action_key
                    from agent.live_decision.high_score.hif_public_turn_reanalysis import public_key

                    _, table = self._public_turn_source
                    key = public_key(request)
                    node = table.nodes.get(key, {})
                    edges = [edge for edge in node.get('edges', ()) if edge['action_key'] == action_key(action)]
                    if (self._public_turn_arrival == (key, index - 1) and len(edges) == 1
                            and 'child' in edges[0] and edges[0]['child'][0] == after):
                        self._public_turn_arrival = (edges[0]['child'], index)
                    else:
                        self._public_turn_source = self._public_turn_arrival = None
            else:
                self._public_turn_source = self._public_turn_arrival = None
                self._resource_source = self._resource_arrival = self._resource_cursor = None
        self._last_choice = None
        return self.policy.committed(before, action, after)


def _action_identity(action):
    return (action.get('kind'), action.get('card_instance_id'), action.get('drink_id'),
            action.get('target_id'), action.get('source_slot') if action.get('kind') == 'USE_DRINK' else None)


def _public_turn_continuation_enabled(plan, *, expected=None):
    enabled = plan.get('public_turn_policy_continuation_enabled', False)
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    version = plan.get('public_turn_policy_continuation_version')
    return version == PUBLIC_CONTINUATION_VERSION if enabled else version is None


def _resource_request_key(request):
    return json.dumps(_json_value({name: request.get(name) for name in (
        'state', 'drink_slots', 'card_slots')}), sort_keys=True)


def _resource_public_nodes_evidence(row):
    """Recompute the local table from complete paired means, not its argmax claim."""
    nodes, graph = row.get('public_nodes'), row['public_graph']
    if not isinstance(nodes, (tuple, list)) or len(nodes) != graph['node_count']:
        return False
    try:
        lookup = {_resource_request_key(node['request']): node for node in nodes}
        if len(lookup) != len(nodes):
            return False
        boundaries = {}
        for node in nodes:
            for edge in node['edges']:
                if ('child_request' in edge) == ('boundary_index' in edge):
                    return False
                if 'boundary_index' in edge:
                    index = edge['boundary_index']
                    if type(index) is not int or index in boundaries:
                        return False
                    boundaries[index] = node['request']['state']['score']
        means = {}
        for group in graph['boundary_groups']:
            base = sum(group['final_scores']) / 4
            representative = boundaries[group['representative_boundary']]
            for index in group['boundary_indices']:
                means[index] = base + boundaries[index] - representative
        if set(means) != set(boundaries) or len(means) != graph['boundary_count']:
            return False
        values, active = {}, set()

        def visit(key):
            if key in active:
                raise ValueError('cycle')
            if key in values:
                return values[key]
            active.add(key)
            node = lookup[key]
            options = []
            for edge in node['edges']:
                value = (visit(_resource_request_key(edge['child_request']))
                         if 'child_request' in edge else means[edge['boundary_index']])
                options.append(dict(action_key=edge['action_key'], mean_final_score=value))
            keys = [tuple(option['action_key']) for option in options]
            if not keys or len(set(keys)) != len(keys):
                raise ValueError('legal_options')
            best = max(option['mean_final_score'] for option in options)
            maximum = [option['action_key'] for option in options if option['mean_final_score'] == best]
            if (_json_value(options) != _json_value(node['options'])
                    or _json_value(maximum) != _json_value(node['maximizing_actions'])
                    or node['mean_final_score'] != best):
                raise ValueError('table_value')
            active.remove(key)
            values[key] = best
            return best

        root = _resource_request_key(row['normalized_request'])
        visit(root)
        return (len(values) == len(nodes)
            and sum(len(node['edges']) for node in nodes) == graph['edge_count']
            and _json_value(lookup[root]['options']) == _json_value(graph['root_options']))
    except (KeyError, TypeError, ValueError, RecursionError):
        return False


def _internal_reservation_release_evidence(plan):
    """Only a recorded unused reserve may restore the original training limit."""
    evidence = plan.get('internal_reanalysis')
    release = evidence.get('reservation_release') if isinstance(evidence, dict) else None
    fields = {'status', 'reason', 'original_training_limit', 'effective_training_limit',
              'released_simulations', 'training_before_release', 'resource_graph_protected',
              'resource_precheck'}
    if (plan.get('internal_reanalysis_enabled') is not True
            or plan.get('internal_reanalysis_version') != INTERNAL_REANALYSIS_VERSION
            or not isinstance(evidence, dict) or evidence.get('version') != INTERNAL_REANALYSIS_VERSION
            or not isinstance(release, dict) or set(release) != fields
            or any(type(release.get(name)) is not int or release[name] < 0 for name in (
                'original_training_limit', 'effective_training_limit', 'released_simulations',
                'training_before_release'))
            or type(release.get('resource_graph_protected')) is not bool):
        return False
    if evidence.get('total_budget') == 0:
        return release == dict(status='not_applicable', reason='not_prepared',
            original_training_limit=0, effective_training_limit=0, released_simulations=0,
            training_before_release=0, resource_graph_protected=False, resource_precheck=None)
    original = release['original_training_limit']
    limit = evidence.get('training_limit')
    training = plan.get('training_completed_simulations')
    if (evidence.get('total_budget') != _plan_simulation_budget(plan) or type(limit) is not int or limit < 0
            or type(training) is not int or training < 0
            or original != (plan.get('public_turn_budget_reservation') or {}).get('training_budget_after')
            or type(evidence.get('minimum_training')) is not int
            or type(evidence.get('minimum_validation')) is not int
            or not evidence['minimum_training'] <= original <= _plan_simulation_budget(plan)-evidence['minimum_validation']):
        return False
    initial = min(original, limit)
    if release['training_before_release'] > min(initial, training):
        return False
    if release['status'] == 'released':
        precheck = release['resource_precheck']
        return (release['reason'] == 'no_sparse_prior_disagreement'
            and release['released_simulations'] == original-initial > 0
            and release['effective_training_limit'] == original
            and release['training_before_release'] == initial
            and release['resource_graph_protected'] is False
            and evidence.get('status') == 'declined' and evidence.get('reason') == release['reason']
            and evidence.get('eligible') is False and evidence.get('complete') is False
            and all(evidence.get(name) == 0 for name in (
                'requested_simulations', 'started_simulations', 'completed_simulations'))
            and all(not evidence.get(name) for name in (
                'samples', 'selected_action', 'means', 'public_request', 'node_statistics',
                'legal_action_keys', 'planning_seeds'))
            and isinstance(plan.get('public_turn_reanalysis'), dict)
            and plan['public_turn_reanalysis'].get('complete') is False
            and plan['public_turn_reanalysis'].get('started_simulations')
                == plan['public_turn_reanalysis'].get('completed_simulations') == 0
            and isinstance(precheck, dict) and precheck.get('status') == 'declined'
            and precheck.get('reason') != 'complete_public_graph'
            and _resource_reanalysis_evidence(dict(plan, resource_reanalysis=precheck), expected=True))
    if release['released_simulations'] != 0 or release['effective_training_limit'] != initial:
        return False
    if release['status'] == 'not_applicable':
        return (original == initial and release['reason'] == 'training_limit_not_reduced'
            and release['resource_graph_protected'] is False and release['resource_precheck'] is None)
    if release['status'] != 'held' or original <= initial:
        return False
    if release['reason'] == 'complete_resource_graph':
        precheck = release['resource_precheck']
        return (release['resource_graph_protected'] is True
            and (plan.get('resource_reanalysis') or {}).get('complete') is True
            and evidence.get('complete') is False and evidence.get('status') == 'declined'
            and evidence.get('started_simulations') == evidence.get('completed_simulations') == 0
            and isinstance(precheck, dict) and precheck.get('version') == RESOURCE_REANALYSIS_VERSION
            and precheck.get('status') == precheck.get('reason') == 'prepared'
            and precheck.get('complete') is False and precheck.get('required_simulations_exact') is True
            and precheck.get('started_simulations') == precheck.get('completed_simulations') == 0
            and type(precheck.get('required_simulations')) is int
            and 0 < precheck['required_simulations'] <= _plan_simulation_budget(plan)-initial-evidence['minimum_validation'])
    if release['resource_graph_protected'] is not False or release['resource_precheck'] is not None:
        return False
    if release['reason'] == 'internal_selected':
        return (evidence.get('complete') is True and evidence.get('status') == 'complete'
            or evidence.get('complete') is False and evidence.get('status') == 'declined'
                and evidence.get('reason') == 'existing_public_or_resource_policy'
                and evidence.get('started_simulations') == evidence.get('completed_simulations') == 0
                and any((plan.get(name) or {}).get('complete') is True
                    for name in ('public_turn_reanalysis', 'resource_reanalysis')))
    return (release['reason'] in ('training_soft_deadline', 'training_failed')
        and evidence.get('complete') is False
        and evidence.get('started_simulations') == evidence.get('completed_simulations') == 0)


def _final_double_tail_evidence(plan, *, expected=None):
    """Bind the optional final-turn domain to the same complete internal parent."""
    enabled = plan.get('final_double_tail_evaluation_enabled', False)
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    if not enabled:
        return plan.get('final_double_tail_domain_version') is None
    return (plan.get('final_double_tail_domain_version') == FINAL_DOUBLE_VERSION
        and plan.get('internal_reanalysis_enabled') is True
        and plan.get('internal_reanalysis_version') == INTERNAL_REANALYSIS_VERSION)


def _final_offhand_remove_tail_evidence(plan, *, expected=None):
    """Bind random offhand removal admission to its explicit final-double parent."""
    enabled = plan.get('final_offhand_remove_tail_evaluation_enabled', False)
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    if not enabled:
        return plan.get('final_offhand_remove_tail_domain_version') is None
    return (plan.get('final_offhand_remove_tail_domain_version') == FINAL_OFFHAND_REMOVE_VERSION
        and _final_double_tail_evidence(plan, expected=True))


def _internal_complete_root_successors_evidence(plan, *, expected=None):
    """Bind the new prefix proof to its explicit mode and one complete root identity."""
    enabled = plan.get('internal_complete_root_successors_enabled', False)
    evidence = plan.get('internal_reanalysis')
    version = plan.get('internal_complete_root_successors_version')
    fields = {'complete_root_successors_enabled', 'complete_root_successors_version',
              'root_continuation', 'root_expected_after_first'}
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return version is None and (not isinstance(evidence, dict) or not fields.intersection(evidence))
    if (version != COMPLETE_ROOT_SUCCESSORS_VERSION
            or not _quality_reference_evidence(plan, expected=True)
            or plan.get('internal_reanalysis_enabled') is not True):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None
    if (not isinstance(evidence, dict) or not fields <= set(evidence)
            or evidence['complete_root_successors_enabled'] is not True
            or evidence['complete_root_successors_version'] != COMPLETE_ROOT_SUCCESSORS_VERSION):
        return False
    if evidence.get('public_request') is None:
        return evidence['root_continuation'] in ((), []) and evidence['root_expected_after_first'] is None
    try:
        continuation = evidence['root_continuation']
        if (not isinstance(continuation, (tuple, list)) or len(continuation) > 1
                or any(not isinstance(key, (tuple, list)) or len(key) != 5 for key in continuation)
                or bool(continuation) != isinstance(evidence['root_expected_after_first'], dict)
                or not continuation and evidence['root_expected_after_first'] is not None):
            return False
        root_key = (tuple(evidence['root_action']), tuple(tuple(key) for key in continuation))
        roots = [root for root in plan['candidate_statistics']
            if (_action_identity(root['action']), tuple(_action_identity(action)
                for action in root.get('continuation', ()))) == root_key]
        return (len(roots) == 1 and (not plan.get('root_options_enabled', False)
            or _json_value(roots[0].get('root_option_key')) == _json_value(root_key)))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _internal_successor_graph_mode(plan, *, expected=None):
    """A new graph cannot be smuggled into a legacy single-node receipt."""
    enabled = plan.get('internal_successor_graph_enabled', False)
    evidence = plan.get('internal_reanalysis')
    fields = {'successor_graph_version', 'successor_graph', 'successor_public_nodes', 'successor_graph_fallback',
              'successor_request', 'selection_mode', 'nominations'}
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return (plan.get('internal_successor_graph_version') is None
            and (not isinstance(evidence, dict) or not fields.intersection(evidence)))
    if (plan.get('internal_successor_graph_version') != INTERNAL_SUCCESSOR_GRAPH_VERSION
            or not _quality_reference_evidence(plan, expected=True)
            or plan.get('internal_reanalysis_enabled') is not True):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None
    if not isinstance(evidence, dict):
        return False
    graph = evidence.get('successor_graph')
    if graph is None:
        return not any(evidence.get(name) for name in (
            'successor_graph_version', 'successor_public_nodes', 'successor_request'))
    if (evidence.get('successor_graph_version') != INTERNAL_SUCCESSOR_GRAPH_VERSION
            or evidence.get('selection_mode') != 'most_visited_sparse_successor'
            or not isinstance(graph, dict)):
        return False
    try:
        nominations = evidence['nominations']
        if not isinstance(nominations, (list, tuple)) or not nominations:
            return False
        indices = []
        selection_fields = ('root_action', 'node_statistics', 'prior_action', 'frozen_action')
        complete_roots = plan.get('internal_complete_root_successors_enabled') is True
        if complete_roots:
            selection_fields += ('root_continuation', 'root_expected_after_first')
        root_keys = {_action_identity(row['action']) for row in plan['candidate_statistics']}
        for row in nominations:
            if complete_roots and (any(name not in row for name in selection_fields)
                    or not _internal_complete_root_successors_evidence(dict(plan,
                        internal_reanalysis=dict(evidence, **{name: row[name] for name in (
                            'root_action', 'root_continuation', 'root_expected_after_first')})), expected=True)):
                return False
            statistics = [(tuple(key), visits, total) for key, visits, total in row['node_statistics']]
            keys = [item[0] for item in statistics]
            index = row['candidate_index']
            if (type(index) is not int or not 0 <= index < evidence['candidate_count']
                    or tuple(row['root_action']) not in root_keys
                    or len(keys) < 2 or len(set(keys)) != len(keys) or any(len(key) != 5 for key in keys)
                    or any(type(visits) is not int or visits < 1 or type(total) not in (int, float)
                        or not math.isfinite(total) for _, visits, total in statistics)
                    or all(visits >= 4 for _, visits, _ in statistics)
                    or type(row['total_visits']) is not int
                    or row['total_visits'] != sum(item[1] for item in statistics)
                    or tuple(row['prior_action']) not in keys
                    or tuple(row['frozen_action']) != max(statistics, key=lambda item: item[2]/item[1])[0]
                    or row['frozen_action'] == row['prior_action']):
                return False
            indices.append(index)
        if indices != sorted(set(indices)):
            return False
        winner = max(nominations, key=lambda row: row['total_visits'])
        if graph.get('complete') is True:
            return evidence.get('successor_graph_fallback', False) is False and all(
                _json_value(evidence.get(name)) == _json_value(winner[name]) for name in selection_fields)
        # A refused complete graph pays no returns and falls back to the old
        # first eligible successor. Its ordinary receipt is checked unchanged.
        return (graph.get('status') == 'declined' and graph.get('complete') is False
            and evidence.get('successor_graph_fallback') is True
            and graph.get('started_simulations') == graph.get('completed_simulations') == 0
            and not evidence.get('successor_public_nodes') and evidence.get('successor_request') is None
            and all(_json_value(evidence.get(name)) == _json_value(nominations[0][name]) for name in selection_fields)
            and _public_turn_reanalysis_evidence(dict(plan, public_turn_reanalysis=graph), []))
    except (KeyError, TypeError, ValueError):
        return False


def _internal_successor_graph_evidence(plan, evidence, *, verified):
    """Reuse the existing complete public-graph and mean-backup proof."""
    graph = evidence['successor_graph']
    roots = plan.get('candidate_statistics', ())
    try:
        keys = [tuple(key) for key in evidence['legal_action_keys']]
        options = graph['root_options']
        means = [(tuple(key), mean) for key, mean in evidence['means']]
        seeds = tuple(evidence['planning_seeds'])
        outer = {sample['planning_seed'] for root in roots for sample in root['validation_samples']}
        outer.update(sample['planning_seed'] for sample in
            (plan.get('retained_policy_comparison') or {}).get('validation_samples', ()))
        validation = tuple(evidence['validation_seeds'])
        request = evidence['successor_request']
        if (evidence.get('value_scope') != 'LOCAL_CONDITIONAL_PAIRED_TURN_GRAPH_FROZEN_RETURNS'
                or evidence.get('reason') != 'complete_successor_graph'
                or evidence.get('status') != 'complete' or evidence.get('complete') is not True
                or evidence.get('eligible') is not True
                or verified and plan.get('internal_reanalysis_verified') is not True
                or not roots or len(keys) < 2 or len(keys) != len(set(keys))
                or {tuple(row['action_key']) for row in options} != set(keys)
                or means != [(tuple(row['action_key']), row['mean_final_score']) for row in options]
                or tuple(evidence['selected_action']) != max(means, key=lambda row: row[1])[0]
                or evidence.get('samples') not in ((), [])
                or len(seeds) != 4 or seeds != tuple(graph['planning_seeds'])
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or not evidence.get('training_validation_disjoint') or set(seeds) & outer
                or len(validation) != len(set(validation)) or any(type(seed) is not int or seed not in outer
                    for seed in validation) or set(validation) & set(seeds)
                or not isinstance(request, dict)
                or _json_value({name: request[name] for name in ('state', 'drink_slots', 'plan_type', 'utility')})
                    != _json_value(evidence['public_request'])
                or any(evidence.get(name) != graph['completed_simulations'] for name in (
                    'requested_simulations', 'started_simulations', 'completed_simulations'))
                or graph['completed_simulations'] > _plan_simulation_budget(plan)
                    - plan['training_completed_simulations'] - evidence['minimum_validation']
                or any(not isinstance(plan.get(name), dict) or plan[name].get('complete') is not False
                    or plan[name].get('started_simulations') != 0 or plan[name].get('completed_simulations') != 0
                    for name in ('public_turn_reanalysis', 'resource_reanalysis'))):
            return False
        local_roots = [dict(validation_samples=roots[0]['validation_samples']) for _ in options]
        return (_public_turn_reanalysis_evidence(dict(plan, public_turn_reanalysis=graph), local_roots)
            and _resource_public_nodes_evidence(dict(public_graph=graph,
                normalized_request=request, public_nodes=evidence['successor_public_nodes'])))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _internal_reanalysis_evidence(plan, *, expected=None, verified=True, expected_complete_roots=None,
                                  expected_successor_graph=None):
    """Audit all local conditional pairs without treating them as outer quality evidence."""
    enabled = plan.get('internal_reanalysis_enabled', False)
    evidence = plan.get('internal_reanalysis')
    if (not _internal_complete_root_successors_evidence(plan, expected=expected_complete_roots)
            or not _internal_successor_graph_mode(plan, expected=expected_successor_graph)
            or type(enabled) is not bool or expected is not None and enabled is not expected):
        return False
    if not enabled:
        return plan.get('internal_reanalysis_version') is None and evidence is None
    if (plan.get('internal_reanalysis_version') != INTERNAL_REANALYSIS_VERSION
            or any(plan.get(name) is not True for name in (
                'retained_tree_policy_execution_enabled', 'retained_policy_execution_enabled',
                'retained_policy_competition_enabled', 'resource_reanalysis_enabled',
                'public_turn_reanalysis_enabled', 'deterministic_validation_cohort_enabled'))):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None
    names = ('candidate_count', 'total_budget', 'minimum_training', 'minimum_validation',
             'action_capacity', 'reserved_simulations', 'training_limit',
             'requested_simulations', 'started_simulations', 'completed_simulations')
    if (not isinstance(evidence, dict) or evidence.get('version') != INTERNAL_REANALYSIS_VERSION
            or evidence.get('enabled') is not True or evidence.get('requested_samples') != 4
            or evidence.get('value_scope') != ('LOCAL_CONDITIONAL_PAIRED_TURN_GRAPH_FROZEN_RETURNS'
                if (evidence.get('successor_graph') or {}).get('complete') is True
                else 'LOCAL_CONDITIONAL_PAIRED_FROZEN_RETURNS')
            or any(type(evidence.get(name)) is not int or evidence[name] < 0 for name in names)
            or any(type(evidence.get(name)) is not bool for name in (
                'eligible', 'complete', 'training_validation_disjoint'))
            or evidence['completed_simulations'] > evidence['started_simulations']
            or not isinstance(evidence.get('reason'), str) or not evidence['reason']):
        return False
    evidence = _json_value(evidence)
    if not _internal_reservation_release_evidence(plan):
        return False
    roots = plan.get('candidate_statistics')
    if evidence['total_budget']:
        if not isinstance(roots, (list, tuple)) or not roots:
            return False
        minimum_training = 4 * len(roots)
        minimum_validation = 4 * (len(roots) + int(_retained_policy_competition_active(plan)))
        if (evidence['total_budget'] != _plan_simulation_budget(plan) or evidence['minimum_training'] != minimum_training
                or evidence['minimum_validation'] != minimum_validation
                or evidence['action_capacity'] != (_plan_simulation_budget(plan)-minimum_training-minimum_validation)//4
                or not 0 <= evidence['reserved_simulations'] <= 4*evidence['action_capacity']
                or evidence['reserved_simulations'] % 4
                or evidence['candidate_count'] == 0 and evidence['reserved_simulations'] != 0
                or evidence['candidate_count'] > 0 and evidence['reserved_simulations'] < 8
                or evidence['training_limit'] != _plan_simulation_budget(plan)-minimum_validation-evidence['reserved_simulations']
                or any(type(plan.get(name)) is not int
                    or not minimum_training <= plan[name]
                        <= evidence['reservation_release']['effective_training_limit']
                    for name in ('training_started_simulations', 'training_completed_simulations'))):
            return False
    if (evidence.get('successor_graph') or {}).get('complete') is True:
        return _internal_successor_graph_evidence(plan, evidence, verified=verified)
    if not evidence['complete']:
        discarded = (evidence.get('reason') == 'existing_public_or_resource_policy'
            and evidence['reservation_release']['status'] in ('held', 'not_applicable')
            and any((plan.get(name) or {}).get('complete') is True
                for name in ('public_turn_reanalysis', 'resource_reanalysis')))
        if discarded:
            try:
                keys = tuple(tuple(key) for key in evidence['legal_action_keys'])
                statistics = tuple((tuple(key), visits, total)
                    for key, visits, total in evidence['node_statistics'])
                frozen, prior = tuple(evidence['frozen_action']), tuple(evidence['prior_action'])
                discarded = (len(keys) >= 2 and len(set(keys)) == len(keys)
                    and all(len(key) == 5 for key in keys)
                    and evidence['requested_simulations'] == 4*len(keys)
                    and len(statistics) == len(keys) and {row[0] for row in statistics} == set(keys)
                    and all(type(visits) is int and visits > 0 and type(total) in (int, float)
                        and math.isfinite(total) for _, visits, total in statistics)
                    and any(visits < 4 for _, visits, _ in statistics)
                    and max(statistics, key=lambda row: row[2]/row[1])[0] == frozen
                    and frozen in keys and prior in keys and frozen != prior
                    and tuple(evidence['root_action']) in {_action_identity(root['action']) for root in roots}
                    and isinstance(evidence['public_request'], dict)
                    and set(evidence['public_request']) == {'state', 'drink_slots', 'plan_type', 'utility'})
            except (KeyError, TypeError, ValueError):
                discarded = False
        return (evidence.get('status') in ('not_run', 'declined') and not evidence['eligible']
            and evidence['started_simulations'] == evidence['completed_simulations'] == 0
            and (discarded or evidence['requested_simulations'] == 0
                and evidence.get('public_request') is None and not evidence.get('node_statistics')
                and not evidence.get('legal_action_keys'))
            and not evidence.get('samples')
            and evidence.get('selected_action') is None and not evidence.get('means')
            and not evidence.get('planning_seeds'))
    if (evidence.get('status') != 'complete' or evidence.get('reason') != 'complete_paired_training'
            or not evidence['eligible'] or not isinstance(roots, (list, tuple)) or not roots
            or verified and plan.get('internal_reanalysis_verified') is not True
            or any(not isinstance(plan.get(name), dict) or plan[name].get('complete') is not False
                or plan[name].get('started_simulations') != 0 or plan[name].get('completed_simulations') != 0
                for name in ('public_turn_reanalysis', 'resource_reanalysis'))
            or not isinstance(evidence.get('public_request'), dict)
            or set(evidence['public_request']) != {'state', 'drink_slots', 'plan_type', 'utility'}):
        return False
    try:
        keys = tuple(tuple(key) for key in evidence['legal_action_keys'])
        seeds = tuple(evidence['planning_seeds'])
        validation = tuple(evidence['validation_seeds'])
        frozen, prior = tuple(evidence['frozen_action']), tuple(evidence['prior_action'])
        selected = tuple(evidence['selected_action'])
        statistics = tuple((tuple(key), visits, total) for key, visits, total in evidence['node_statistics'])
        means = tuple((tuple(key), mean) for key, mean in evidence['means'])
        samples = evidence['samples']
        outer = {sample['planning_seed'] for root in roots for sample in root['validation_samples']}
        retained = plan.get('retained_policy_comparison') or {}
        outer.update(row['planning_seed'] for row in retained.get('validation_samples', ()))
        minimum_training = 4 * len(roots)
        minimum_validation = 4 * (len(roots) + int(_retained_policy_competition_active(plan)))
        if (len(keys) < 2 or len(set(keys)) != len(keys) or any(len(key) != 5 for key in keys)
                or frozen not in keys or prior not in keys or prior == frozen or selected not in keys
                or tuple(evidence['root_action']) not in {_action_identity(root['action']) for root in roots}
                or len(seeds) != 4 or len(set(seeds)) != 4
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or seeds != tuple(plan.get('validation_cohort_evidence', {}).get('training_seeds', ()))
                or set(seeds) & outer or set(seeds) & set(validation)
                or any(type(seed) is not int or seed not in outer for seed in validation)
                or len(set(validation)) != len(validation) or not evidence['training_validation_disjoint']
                or evidence['candidate_count'] < 1 or evidence['total_budget'] != _plan_simulation_budget(plan)
                or evidence['minimum_training'] != minimum_training
                or evidence['minimum_validation'] != minimum_validation
                or evidence['action_capacity'] != (_plan_simulation_budget(plan)-minimum_training-minimum_validation)//4
                or len(keys) > evidence['action_capacity']
                or not 4*len(keys) <= evidence['reserved_simulations'] <= 4*evidence['action_capacity']
                or evidence['reserved_simulations'] % 4
                or evidence['training_limit'] != _plan_simulation_budget(plan)-minimum_validation-evidence['reserved_simulations']
                or not minimum_training <= plan.get('training_completed_simulations', -1)
                    <= evidence['reservation_release']['effective_training_limit']
                or evidence['requested_simulations'] != 4*len(keys)
                or evidence['started_simulations'] != evidence['completed_simulations']
                or evidence['completed_simulations'] != evidence['requested_simulations']
                or len(statistics) != len(keys) or {row[0] for row in statistics} != set(keys)
                or any(type(visits) is not int or visits < 1 or type(total) not in (int, float)
                    or not math.isfinite(total) for _, visits, total in statistics)
                or all(visits >= 4 for _, visits, _ in statistics)
                or max(statistics, key=lambda row: row[2]/row[1])[0] != frozen
                or len(means) != len(keys) or {row[0] for row in means} != set(keys)
                or not isinstance(samples, list) or len(samples) != 4*len(keys)):
            return False
        pairs, sums = set(), dict.fromkeys(keys, 0.)
        for row in samples:
            key, seed = tuple(row['action_key']), row['planning_seed']
            score, depth = row['final_score'], row['completed_depth']
            calls, fallbacks = row['terminal_evaluations'], row['terminal_fallbacks']
            if (key not in keys or seed not in seeds or (key, seed) in pairs
                    or type(seed) is not int or type(score) not in (int, float) or not math.isfinite(score)
                    or row['return_kind'] not in ('sampled_terminal_score', 'terminal_evaluator_expected_score',
                                                  'proven_tail_expected_score')
                    or any(type(value) is not int for value in (depth, calls, fallbacks))
                    or not 0 <= depth < 256 or not 0 <= fallbacks <= calls):
                return False
            pairs.add((key, seed))
            sums[key] += score
        if any(type(mean) not in (int, float) or not math.isfinite(mean)
                or not math.isclose(mean, sums[key]/4, rel_tol=1e-12, abs_tol=1e-7)
                for key, mean in means):
            return False
        values = dict(means)
        order = (frozen, *(key for key in keys if key != frozen))
        return selected == max(order, key=values.__getitem__)
    except (KeyError, TypeError, ValueError):
        return False


def _resource_budget_decline_reuse_evidence(plan, evidence):
    """Reuse only the first precheck's witnessed bound, without new graph work."""
    proof = evidence.get('budget_decline_reuse')
    fields = {'version', 'source', 'source_budget', 'current_budget',
        'source_enumerated_states', 'current_enumerated_states', 'checked_new_states',
        'tree_key_prefix_unchanged', 'context_unchanged', 'source_graph_index',
        'required_simulations_lower_bound', 'graphs_built'}
    integers = fields - {'version', 'source', 'tree_key_prefix_unchanged', 'context_unchanged'}
    internal = plan.get('internal_reanalysis')
    release = internal.get('reservation_release') if isinstance(internal, dict) else None
    source = release.get('resource_precheck') if isinstance(release, dict) else None
    if (not isinstance(proof, dict) or set(proof) != fields
            or proof['version'] != BUDGET_DECLINE_REUSE_VERSION
            or proof['source'] != 'internal_reservation_precheck'
            or any(type(proof[name]) is not int or proof[name] < 0 for name in integers)
            or proof['tree_key_prefix_unchanged'] is not True or proof['context_unchanged'] is not True
            or proof['source_graph_index'] != 0 or proof['graphs_built'] != 0
            or not isinstance(source, dict) or source.get('budget_decline_reuse') is not None
            or source.get('reason') != 'resource_graph_declined:complete_graph_rollout_budget'
            or source.get('eligible') is not True or source.get('required_simulations_exact') is not False
            or source.get('prepared_graph_count') != 0
            or not _internal_reservation_release_evidence(plan)):
        return False
    roots = plan.get('candidate_statistics')
    public = plan.get('public_turn_reanalysis')
    if (type(plan.get('requested_simulations')) is not int or plan['requested_simulations'] != _plan_simulation_budget(plan)
            or not isinstance(roots, (list, tuple)) or not roots
            or internal.get('minimum_validation') != 4 * (len(roots) + int(_retained_policy_competition_active(plan)))
            or not isinstance(public, dict) or public.get('status') != 'declined'
            or any(type(public.get(name)) is not int or public[name] != 0
                   for name in ('started_simulations', 'completed_simulations'))
            or release['status'] != 'released'
            or proof['source_budget'] != _plan_simulation_budget(plan)-release['training_before_release']-internal['minimum_validation']
            or proof['current_budget'] != _plan_simulation_budget(plan)-plan['training_completed_simulations']
                -public['completed_simulations']-internal['minimum_validation']
            or proof['current_budget'] > proof['source_budget']
            or proof['source_enumerated_states'] != source['enumerated_states']
            or proof['current_enumerated_states'] != evidence['enumerated_states']
            or proof['checked_new_states'] != evidence['enumerated_states']-source['enumerated_states']
            or not 0 <= evidence['eligible_states']-source['eligible_states'] <= proof['checked_new_states']
            or not 0 <= evidence['graph_count']-source['graph_count']
                <= evidence['eligible_states']-source['eligible_states']
            or not 0 < source['graph_count'] <= source['eligible_states']
            or evidence['graph_count'] > evidence['eligible_states']
            or plan.get('resource_policy_link_created', False) is not False
            or evidence['eligible'] is not True or evidence['complete'] is not False
            or evidence['required_simulations_exact'] is not False
            or any(evidence[name] != 0 for name in (
                'prepared_graph_count', 'started_simulations', 'completed_simulations'))
            or source.get('score_translation_contract') != SCORE_TRANSLATION_CONTRACT
            or evidence.get('score_translation_contract') != SCORE_TRANSLATION_CONTRACT):
        return False
    graphs = source['graphs']
    if len(graphs) != 1 or not isinstance(graphs[0], dict):
        return False
    row = graphs[0]
    graph = row.get('public_graph')
    indices = row.get('origin_indices')
    if (not isinstance(graph, dict) or graph.get('version') != PUBLIC_TURN_VERSION
            or graph.get('enabled') is not True or graph.get('requested_samples') != 4
            or graph.get('status') != 'declined' or graph.get('reason') != 'complete_graph_rollout_budget'
            or graph.get('eligible') is not False or graph.get('complete') is not False
            or graph.get('score_translation_contract') != SCORE_TRANSLATION_CONTRACT
            or any(type(graph.get(name)) is not int or graph[name] < 0 for name in (
                'node_count', 'edge_count', 'boundary_count', 'boundary_group_count',
                'required_simulations', 'started_simulations', 'completed_simulations'))
            or graph['started_simulations'] != 0 or graph['completed_simulations'] != 0
            or any(graph.get(name) for name in ('planning_seeds', 'root_options', 'boundary_groups'))
            or graph['boundary_group_count'] <= 0
            or graph['boundary_group_count'] > graph['boundary_count']
            or graph['required_simulations'] != 4 * graph['boundary_group_count']
            or proof['required_simulations_lower_bound'] != graph['required_simulations']
            or proof['required_simulations_lower_bound'] <= proof['source_budget']
            or not isinstance(indices, (list, tuple)) or not indices
            or any(type(i) is not int or not 0 <= i < source['enumerated_states'] for i in indices)
            or list(indices) != sorted(set(indices))
            or not isinstance(row.get('origins'), (list, tuple)) or len(row['origins']) != len(indices)
            or len(indices) > source['eligible_states']
            or not isinstance(row.get('normalized_request'), dict)
            or not isinstance(row['normalized_request'].get('state'), dict)
            or row['normalized_request']['state'].get('score') != 0):
        return False
    structural = ('node_count', 'edge_count', 'boundary_count', 'boundary_group_count', 'required_simulations')
    return (all(source[name] == graph[name] for name in structural)
        and source['required_simulations_lower_bound'] == proof['required_simulations_lower_bound']
        and all(evidence[name] == source[name] for name in (*structural, 'required_simulations_lower_bound'))
        and _json_value(evidence['graphs']) == _json_value(graphs))


def _resource_reanalysis_evidence(plan, *, expected=None):
    """Local training tables are separate from full graphs and root samples."""
    enabled = plan.get('resource_reanalysis_enabled', False)
    evidence = plan.get('resource_reanalysis')
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    versions = (plan.get('resource_reanalysis_version'), plan.get('resource_policy_continuation_version'))
    if not enabled:
        return versions == (None, None) and evidence is None
    if (versions != (RESOURCE_REANALYSIS_VERSION, RESOURCE_CONTINUATION_VERSION)
            or any(plan.get(name) is not True for name in (
                'public_turn_reanalysis_enabled', 'public_turn_policy_continuation_enabled',
                'public_turn_score_translation_enabled', 'installed_score_turn_chain_enabled',
                'unknown_node_timing_enabled', 'partial_node_timing_enabled'))):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None
    names = ('enumerated_states', 'eligible_states', 'graph_count', 'prepared_graph_count',
             'node_count', 'edge_count', 'boundary_count', 'boundary_group_count',
             'required_simulations', 'required_simulations_lower_bound',
             'started_simulations', 'completed_simulations')
    if (not isinstance(evidence, dict) or evidence.get('version') != RESOURCE_REANALYSIS_VERSION
            or evidence.get('enabled') is not True or evidence.get('requested_samples') != 4
            or evidence.get('value_scope') != 'LOCAL_PUBLIC_RESOURCE_POLICY_IMPROVEMENT'
            or any(type(evidence.get(name)) is not int or evidence[name] < 0 for name in names)
            or any(type(evidence.get(name)) is not bool for name in (
                'eligible', 'complete', 'required_simulations_exact'))
            or evidence['eligible_states'] > evidence['enumerated_states']
            or evidence['prepared_graph_count'] > evidence['graph_count']
            or evidence['completed_simulations'] > evidence['started_simulations']
            or not isinstance(evidence.get('graphs'), (list, tuple))):
        return False
    reuse_reason = 'resource_graph_declined:reused_budget_lower_bound'
    if evidence.get('reason') == reuse_reason:
        return (evidence.get('status') == 'declined'
            and _resource_budget_decline_reuse_evidence(plan, evidence))
    if (evidence.get('budget_decline_reuse') is not None
            or isinstance(evidence.get('reason'), str)
                and evidence['reason'].startswith('resource_graph_declined:reused_')):
        return False
    if evidence.get('status') == 'declined':
        reason = evidence.get('reason')
        return (evidence['complete'] is False
            and evidence['started_simulations'] == evidence['completed_simulations'] == 0
            and isinstance(reason, str)
            and (reason in ('complete_public_graph', 'outside_resource_domain',
                            'no_eligible_resource_nodes', 'resource_graph_total_budget')
                 or reason.startswith('resource_graph_declined:'))
            and (reason != 'complete_public_graph'
                 or (plan.get('public_turn_reanalysis') or {}).get('complete') is True))
    seeds = evidence.get('planning_seeds')
    cohort = plan.get('validation_cohort_evidence', {})
    roots = plan.get('candidate_statistics', ())
    if (evidence.get('status') != 'complete' or evidence.get('reason') != 'complete'
            or not evidence['eligible'] or not evidence['complete'] or not evidence['required_simulations_exact']
            or evidence.get('score_translation_contract') != SCORE_TRANSLATION_CONTRACT
            or not isinstance(seeds, (list, tuple)) or len(seeds) != 4
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != 4 or tuple(seeds) != tuple(cohort.get('training_seeds', ()))
            or set(seeds).intersection(sample.get('planning_seed') for root in roots
                for sample in root.get('validation_samples', ()))
            or evidence['graph_count'] <= 0
            or evidence['graph_count'] != evidence['prepared_graph_count']
            or len(evidence['graphs']) != evidence['graph_count']
            or not roots or any(type(root.get('training_paired_visits')) is not int
                               or root['training_paired_visits'] < 4 for root in roots)
            or evidence['required_simulations'] != evidence['required_simulations_lower_bound']
            or evidence['required_simulations'] != 4 * evidence['boundary_group_count']
            or evidence['started_simulations'] != evidence['completed_simulations']
            or evidence['completed_simulations'] != evidence['required_simulations']
            or plan.get('public_turn_reanalysis', {}).get('complete') is True):
        return False
    totals = dict.fromkeys(('node_count', 'edge_count', 'boundary_count', 'boundary_group_count',
                           'started_simulations', 'completed_simulations'), 0)
    origins = []
    for index, row in enumerate(evidence['graphs']):
        indices = row.get('origin_indices') if isinstance(row, dict) else None
        graph = row.get('public_graph') if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get('graph_index') != index
                or not isinstance(indices, (list, tuple)) or not indices
                or any(type(value) is not int or not 0 <= value < evidence['enumerated_states'] for value in indices)
                or list(indices) != sorted(set(indices))
                or not isinstance(row.get('origins'), (list, tuple)) or len(row['origins']) != len(indices)
                or not isinstance(row.get('normalized_request'), dict)
                or row['normalized_request'].get('state', {}).get('score') != 0
                or not isinstance(graph, dict) or tuple(graph.get('planning_seeds', ())) != tuple(seeds)):
            return False
        options = graph.get('root_options')
        if not isinstance(options, (list, tuple)) or len(options) < 2:
            return False
        local_roots = [dict(validation_samples=roots[0].get('validation_samples', ())) for _ in options] if roots else []
        if (not _public_turn_reanalysis_evidence(dict(plan, public_turn_reanalysis=graph), local_roots)
                or not _resource_public_nodes_evidence(row)):
            return False
        for name in totals:
            totals[name] += graph[name]
        origins.extend(indices)
    return (len(origins) == evidence['eligible_states'] and len(set(origins)) == len(origins)
        and all(evidence[name] == total for name, total in totals.items()))


def _public_turn_score_translation_evidence(plan, *, expected=None):
    """Only the declared installed stack can reuse score-translated boundaries."""
    enabled = plan.get('public_turn_score_translation_enabled', False)
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    version = plan.get('public_turn_score_translation_version')
    graph = plan.get('public_turn_reanalysis')
    if not enabled:
        return version is None and not (
            plan.get('installed_score_turn_chain_enabled') is True
            and isinstance(graph, dict) and graph.get('complete') is True
            and ('score_translation_contract' not in graph or graph['score_translation_contract'] is not None))
    if (version != PUBLIC_TURN_SCORE_TRANSLATION_VERSION
            or not _public_turn_continuation_enabled(plan, expected=True)
            or any(plan.get(field) is not True for field in (
                'public_turn_reanalysis_enabled', 'pair_order_advice_enabled',
                'unknown_node_timing_enabled', 'retrieval_training_order_enabled',
                'lazy_validation_seeds_enabled', 'deterministic_validation_cohort_enabled',
                'partial_node_timing_enabled', 'installed_score_turn_chain_enabled'))):
        return False
    # Exact, forced and retained-table paths need no newly computed graph.
    if graph is None:
        return plan.get('scope') != 'SAMPLING_ESTIMATE'
    if not isinstance(graph, dict):
        return False
    if 'score_translation_contract' in graph:
        return graph['score_translation_contract'] == SCORE_TRANSLATION_CONTRACT
    # This existing admission rejection occurs before the graph builder is called.
    return (graph.get('status') == 'declined' and graph.get('reason') == 'unsupported_root_or_sampler'
        and graph.get('eligible') is False and graph.get('complete') is False
        and all(type(graph.get(name)) is int and graph[name] == 0 for name in (
            'node_count', 'edge_count', 'boundary_count', 'boundary_group_count',
            'required_simulations', 'started_simulations', 'completed_simulations'))
        and all(isinstance(graph.get(name), (list, tuple)) and not graph[name]
                for name in ('planning_seeds', 'root_options', 'boundary_groups')))


def _compute_sensitivity_evidence(plan, *, expected=None, simulations=None):
    enabled = plan.get('compute_sensitivity_enabled', False)
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    budget = plan.get('compute_sensitivity_simulations')
    if not enabled:
        return (plan.get('compute_sensitivity_version') is None
            and plan.get('compute_sensitivity_simulations') is None
            and (simulations is None or type(simulations) is int and simulations == 256))
    return (plan.get('compute_sensitivity_version') == COMPUTE_SENSITIVITY_VERSION
        and type(budget) is int and budget in (256, 512, 1024)
        and _quality_reference_evidence(plan, expected=True)
        and (simulations is None or type(simulations) is int and simulations == budget)
        and (plan.get('scope') != 'SAMPLING_ESTIMATE'
            or type(plan.get('requested_simulations')) is int and plan['requested_simulations'] == budget))


def _plan_simulation_budget(plan):
    return (plan['compute_sensitivity_simulations']
        if plan.get('compute_sensitivity_enabled') is True and _compute_sensitivity_evidence(plan) else 256)


def _source_compute_receipt(plan):
    if plan.get('compute_sensitivity_enabled') is not True:
        return {}
    return {name: plan.get(name) for name in (
        'compute_sensitivity_enabled', 'compute_sensitivity_version', 'compute_sensitivity_simulations',
        'quality_reference_enabled', 'quality_reference_version', 'quality_reference_watchdog_seconds')}


def _source_compute_mode_matches(origin, current):
    if any(plan.get('uniform_validation_time_guard_enabled') is True for plan in (origin, current)):
        fields = ('uniform_validation_time_guard_enabled', 'uniform_validation_time_guard_version')
        if (not _uniform_validation_time_guard_mode(origin) or not _uniform_validation_time_guard_mode(current)
                or tuple(origin.get(name) for name in fields) != tuple(current.get(name) for name in fields)):
            return False
    if any(plan.get('parallel_root_validation_enabled') is True for plan in (origin, current)):
        fields = ('parallel_root_validation_enabled', 'parallel_root_validation_version')
        if (not _parallel_root_validation_mode(origin) or not _parallel_root_validation_mode(current)
                or tuple(origin.get(name) for name in fields) != tuple(current.get(name) for name in fields)):
            return False
    if any(plan.get('paired_adaptive_validation_enabled') is True for plan in (origin, current)):
        fields = ('paired_adaptive_validation_enabled', 'paired_adaptive_validation_version')
        if (not _paired_adaptive_validation_mode(origin) or not _paired_adaptive_validation_mode(current)
                or tuple(origin.get(name) for name in fields) != tuple(current.get(name) for name in fields)):
            return False
    if not any(plan.get('compute_sensitivity_enabled') is True for plan in (origin, current)):
        return True
    return (_compute_sensitivity_evidence(origin) and _compute_sensitivity_evidence(current)
        and _source_compute_receipt(origin) == _source_compute_receipt(current))


def _quality_reference_evidence(plan, *, expected=None, window_seconds=None):
    enabled = plan.get('quality_reference_enabled', False)
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if window_seconds is not None and (
            type(window_seconds) not in (int, float) or not math.isfinite(window_seconds)
            or window_seconds != (300. if enabled else 30.)):
        return False
    version = plan.get('quality_reference_version')
    watchdog = plan.get('quality_reference_watchdog_seconds')
    if not enabled:
        return version is None and watchdog is None
    return (version == QUALITY_REFERENCE_VERSION and type(watchdog) in (int, float)
        and math.isfinite(watchdog) and watchdog == 300.)


def _plan_window_ms(plan):
    return (300000. if plan.get('quality_reference_enabled') is True
        and _quality_reference_evidence(plan) else 30000.)


def _paired_adaptive_validation_mode(plan, *, expected=None):
    """Only unspecified read audits accept v2; execution binds the current version."""
    enabled = plan.get('paired_adaptive_validation_enabled', False)
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    version, evidence = (plan.get(name) for name in (
        'paired_adaptive_validation_version', 'paired_adaptive_validation'))
    if not enabled:
        return version is None and evidence is None
    versions = ((PAIRED_ADAPTIVE_VALIDATION_VERSION, 'hif-paired-adaptive-validation-v2')
                if expected is None else (PAIRED_ADAPTIVE_VALIDATION_VERSION,))
    if (version not in versions
            or not _quality_reference_evidence(plan, expected=True)
            or not _compute_sensitivity_evidence(plan, expected=False, simulations=256)):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None
    return (plan.get('adaptive_validation') is True
        and plan.get('lazy_validation_seeds_enabled') is True
        and plan.get('deterministic_validation_cohort_enabled') is True
        and plan.get('paired_finalist_validation', False) is False
        and type(plan.get('requested_simulations')) is int and plan['requested_simulations'] == 256
        and type(plan.get('requested_validation_samples')) is int and plan['requested_validation_samples'] == 4
        and plan.get('validation_sample_limit') == 4 and isinstance(evidence, dict))


def _paired_adaptive_completed(plan):
    evidence = plan.get('paired_adaptive_validation')
    count = evidence.get('completed_simulations') if isinstance(evidence, dict) else None
    return (count if plan.get('paired_adaptive_validation_enabled') is True
        and type(count) is int and count >= 0 else 0)


def _paired_adaptive_seeds(plan):
    evidence = plan.get('paired_adaptive_validation')
    if plan.get('paired_adaptive_validation_enabled') is not True or not isinstance(evidence, dict):
        return set()
    return {row['seed'] for row in evidence.get('additional_pairs', ())
        if isinstance(row, dict) and type(row.get('seed')) is int}


def _paired_adaptive_time_evidence(plan, evidence, samples):
    """Recompute observed durations and admission checks without trusting maxima."""
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def close(left, right):
        return number(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-7)

    try:
        timing = evidence['time_guard']
        if (not isinstance(timing, dict) or set(timing) != {
                'rule', 'delivery_reserve_seconds', 'base_observed_returns', 'additional_observed_returns',
                'maximum_observed_return_seconds', 'last_check', 'stopped_before_next_pair'}
                or timing['rule'] != 'REMAINING_GT_TWO_MAX_OBSERVED_RETURNS_PLUS_DELIVERY'
                or not number(timing['delivery_reserve_seconds']) or timing['delivery_reserve_seconds'] != 1.
                or type(timing['stopped_before_next_pair']) is not bool):
            return False
        durations = [sample['elapsed_seconds'] for row in samples for sample in row]
        if not durations or any(not number(value) or value < 0 for value in durations):
            return False
        maximum = max(durations)
        total_seconds = sum(durations)
        parallel = plan.get('parallel_root_validation')
        if plan.get('parallel_root_validation_enabled') is True and isinstance(parallel, dict) and (
                parallel.get('status') == 'complete'):
            if not _parallel_root_validation_evidence(plan):
                return False
            # Worker return durations still determine maximum. Only their
            # concurrent contribution to the parent window uses measured wall.
            total_seconds = parallel['elapsed_seconds'] + sum(
                sample['elapsed_seconds'] for row in samples[len(plan['candidate_statistics']):] for sample in row)
        window_seconds = _plan_window_ms(plan) / 1000
        previous_remaining = window_seconds - total_seconds
        last_check = None

        def valid_check(check):
            return (isinstance(check, dict) and set(check) == {
                    'remaining_seconds', 'maximum_observed_return_seconds', 'estimated_pair_seconds', 'required_seconds'}
                and number(check['remaining_seconds']) and 0 < check['remaining_seconds'] <= window_seconds
                and check['remaining_seconds'] <= previous_remaining + 1e-7
                and close(check['maximum_observed_return_seconds'], maximum)
                and close(check['estimated_pair_seconds'], 2 * maximum)
                and close(check['required_seconds'], 2 * maximum + 1.))

        for pair in evidence['additional_pairs']:
            check = pair['time_guard']
            if not valid_check(check) or check['remaining_seconds'] <= check['required_seconds']:
                return False
            rows = pair['return_timings']
            if not isinstance(rows, list) or len(rows) != 2:
                return False
            pair_seconds = 0.
            for row, sample, index in zip(rows, pair['samples'], (pair['leader_index'], pair['opponent_index']), strict=True):
                if (not isinstance(row, dict) or set(row) != {'candidate_index', 'elapsed_seconds', 'callback_returned'}
                        or type(row['candidate_index']) is not int or row['candidate_index'] != index
                        or row['callback_returned'] is not True
                        or not number(row['elapsed_seconds']) or row['elapsed_seconds'] < 0
                        or not close(sample['elapsed_seconds'], row['elapsed_seconds'])):
                    return False
                maximum = max(maximum, row['elapsed_seconds'])
                pair_seconds += row['elapsed_seconds']
            if pair_seconds >= check['remaining_seconds']:
                return False
            total_seconds += pair_seconds
            previous_remaining = check['remaining_seconds'] - pair_seconds
            last_check = check
        stopped = timing['stopped_before_next_pair']
        if stopped:
            if (evidence['reason'] != 'time_reserve_reached'
                    or not valid_check(timing['last_check'])
                    or timing['last_check']['remaining_seconds'] > timing['last_check']['required_seconds']):
                return False
        elif evidence['reason'] == 'time_reserve_reached' or timing['last_check'] != last_check:
            return False
        return (type(timing['base_observed_returns']) is int and timing['base_observed_returns'] == len(durations)
            and type(timing['additional_observed_returns']) is int
            and timing['additional_observed_returns'] == 2 * len(evidence['additional_pairs'])
            and close(timing['maximum_observed_return_seconds'], maximum)
            and number(plan.get('elapsed_ms')) and total_seconds <= plan['elapsed_ms'] / 1000 + 1e-7)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _paired_adaptive_validation_evidence(plan, *, expected=None):
    """Rebuild pair allocation and selection from the unchanged four-world table."""
    if not _paired_adaptive_validation_mode(plan, expected=expected):
        return False
    if plan.get('paired_adaptive_validation_enabled') is not True or plan.get('scope') != 'SAMPLING_ESTIMATE':
        return True
    evidence = _json_value(plan['paired_adaptive_validation'])
    fields = {'version', 'enabled', 'value_scope', 'decision_rule', 'reward_scale', 'sample_cap',
        'complete', 'status', 'reason', 'errors', 'candidate_keys', 'base_samples', 'base_seeds',
        'excluded_seeds', 'additional_pairs', 'requested_simulations', 'started_simulations',
        'completed_simulations', 'remaining_budget', 'candidate_sample_counts', 'pair_statistics',
        'paired_regrets', 'selected_index', 'frozen_signature_unchanged', 'time_guard'}
    guarded = plan['paired_adaptive_validation_version'] == PAIRED_ADAPTIVE_VALIDATION_VERSION
    if guarded:
        fields.update(('minimax_selected_index', 'retained_index', 'retained_pair_count',
                       'retained_mean_difference', 'retained_guard_reason'))
    decision_rule = ('MINIMAX_PAIRED_MEAN_REGRET_WITH_DIRECT_RETAINED_GUARD' if guarded
                     else 'MINIMAX_PAIRED_MEAN_REGRET')
    if (set(evidence) != fields or evidence['version'] != plan['paired_adaptive_validation_version']
            or evidence['enabled'] is not True or evidence['complete'] is not True
            or evidence['status'] != 'complete' or evidence['errors'] != []
            or evidence['frozen_signature_unchanged'] is not True
            or evidence['value_scope'] != 'PAIRED_COMPLETE_POLICY_ESTIMATED_REGRET'
            or evidence['decision_rule'] != decision_rule
            or not isinstance(evidence['time_guard'], dict)
            or type(evidence['sample_cap']) is not int or evidence['sample_cap'] != 32
            or type(evidence['reward_scale']) not in (int, float)
            or not math.isfinite(evidence['reward_scale'])
            or evidence['reward_scale'] != CANDIDATE_CONFIG['reward_scale']):
        return False

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def close(left, right):
        return number(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-7)

    def seed(value, upper=2**31):
        return type(value) is int and 0 <= value < upper

    def valid_sample(row, world):
        return (isinstance(row, dict) and number(row.get('final_score'))
            and number(row.get('elapsed_seconds')) and row['elapsed_seconds'] >= 0
            and row.get('return_kind') in ('sampled_terminal_score', 'terminal_evaluator_expected_score',
                                         'proven_tail_expected_score')
            and type(row.get('completed_depth')) is int and 0 <= row['completed_depth'] < 256
            and type(row.get('terminal_evaluations')) is int and row['terminal_evaluations'] >= 0
            and type(row.get('terminal_fallbacks')) is int and row['terminal_fallbacks'] == 0
            and seed(row.get('planning_seed')) and row['planning_seed'] == world
            and ('complete' not in row or row['complete'] is True))

    try:
        roots = plan['candidate_statistics']
        if not isinstance(roots, (tuple, list)) or not roots:
            return False
        keys, samples = [], []
        for root in roots:
            if (not isinstance(root, dict) or not isinstance(root.get('action'), dict)
                    or type(root.get('validation_paired_visits')) is not int
                    or root['validation_paired_visits'] != 4):
                return False
            keys.append(root['root_option_key'] if plan.get('root_options_enabled') is True
                        else _action_identity(root['action']))
            samples.append(root['validation_samples'])
        retained = plan.get('retained_policy_comparison') or {}
        has_retained = _retained_policy_competition_active(plan)
        if has_retained:
            source = retained.get('source_plan_index', (plan.get('retained_policy_lineage') or {}).get('source_plan_index'))
            if type(source) is not int or source < 0:
                return False
            keys.append(('RETAINED_POLICY', source))
            samples.append(retained['validation_samples'])
        keys, samples = _json_value(keys), _json_value(samples)
        if (len({json.dumps(key, sort_keys=True) for key in keys}) != len(keys)
                or evidence['candidate_keys'] != keys or evidence['base_samples'] != samples
                or any(not isinstance(row, list) or len(row) != 4 for row in samples)):
            return False
        base_seeds = [row['planning_seed'] for row in samples[0]]
        if (any(not seed(value) for value in base_seeds) or len(set(base_seeds)) != 4
                or evidence['base_seeds'] != base_seeds):
            return False
        for row in samples:
            if any(not valid_sample(sample, world) or type(sample.get('scenario_index')) is not int
                   or sample['scenario_index'] != index
                   for index, (sample, world) in enumerate(zip(row, base_seeds, strict=True))):
                return False
        for root, row in zip(roots, samples, strict=False):
            if not close(root.get('validation_mean_final_score'), sum(item['final_score'] for item in row) / 4):
                return False
        if has_retained and (retained.get('complete') is not True
                or retained.get('started_simulations') != 4 or retained.get('completed_simulations') != 4
                or not close(retained.get('mean_final_score'), sum(item['final_score'] for item in samples[-1]) / 4)):
            return False
        cohort = plan['validation_cohort_evidence']
        training_excluded = set(cohort['training_seed_history']) | set(cohort['training_seeds'])
        required_excluded = set(training_excluded)
        for name in ('excluded_training_seeds', 'excluded_validation_seeds'):
            required_excluded.update(retained.get(name, ()))
        lineage = plan.get('revealed_policy_competition') or {}
        training_excluded.update(retained.get('excluded_training_seeds', ()))
        training_excluded.update(lineage.get('source_training_seeds', ()))
        for name in ('source_training_seeds', 'source_validation_seeds'):
            required_excluded.update(lineage.get(name, ()))
        excluded = evidence['excluded_seeds']
        if (not isinstance(excluded, list) or any(not seed(value, 2**32) for value in excluded)
                or not required_excluded.issubset(excluded)
                or set(base_seeds).intersection(training_excluded)):
            return False
        base_work = plan['training_completed_simulations'] + 4 * len(keys)
        if type(plan['training_completed_simulations']) is not int or plan['training_completed_simulations'] < 4 * len(roots):
            return False
        for name in ('public_turn_reanalysis', 'resource_reanalysis', 'internal_reanalysis'):
            row = plan.get(name) or {}
            count = row.get('completed_simulations', 0)
            if type(count) is not int or count < 0:
                return False
            base_work += count
        budget = 256 - base_work
        if (type(evidence['requested_simulations']) is not int or evidence['requested_simulations'] != budget
                or budget < 0 or plan.get('inner_completed_simulations', 0) != 0
                or plan.get('continuation_samples', 0) != 0):
            return False
        differences = {(left, right): [samples[right][world]['final_score'] - samples[left][world]['final_score']
                       for world in range(4)] for left in range(len(keys)) for right in range(left + 1, len(keys))}
        counts = [4] * len(keys)
        used = set(excluded) | set(base_seeds)

        def pair_mean(left, right):
            values = differences[min(left, right), max(left, right)]
            value = sum(values) / len(values)
            return value if left < right else -value

        def regrets():
            return [max((pair_mean(index, other) for other in range(len(keys)) if other != index), default=0.)
                    for index in range(len(keys))]

        pairs = evidence['additional_pairs']
        if not isinstance(pairs, list):
            return False
        for position, pair in enumerate(pairs):
            if (not isinstance(pair, dict) or set(pair) != {'leader_index', 'opponent_index', 'candidate_keys',
                    'counts_before', 'paired_regrets_before', 'opponent_scores', 'seed', 'samples', 'complete',
                    'time_guard', 'return_timings'}
                    or pair['complete'] is not True or budget - 2 * position < 2):
                return False
            eligible = [index for index, count in enumerate(counts) if count < 32]
            current_regrets = regrets()
            if len(eligible) < 2:
                return False
            leader = min(eligible, key=lambda index: (current_regrets[index], index))
            scores = []
            for other in eligible:
                if other == leader:
                    continue
                count = len(differences[min(leader, other), max(leader, other)])
                mean = pair_mean(leader, other)
                exploration = evidence['reward_scale'] * math.sqrt(2 * math.log(sum(counts) + 1) / count)
                scores.append(dict(candidate_index=other, pair_count=count, paired_mean_difference=mean,
                                   exploration=exploration, score=mean + exploration))
            opponent = max(scores, key=lambda row: (row['score'], -row['candidate_index']))['candidate_index']
            if (type(pair['leader_index']) is not int or type(pair['opponent_index']) is not int
                    or pair['leader_index'] != leader or pair['opponent_index'] != opponent
                    or pair['candidate_keys'] != [keys[leader], keys[opponent]]
                    or pair['counts_before'] != counts
                    or any(type(value) is not int for value in pair['counts_before'])
                    or len(pair['paired_regrets_before']) != len(current_regrets)
                    or any(not close(left, right) for left, right in zip(pair['paired_regrets_before'], current_regrets))
                    or not isinstance(pair['opponent_scores'], list) or len(pair['opponent_scores']) != len(scores)):
                return False
            for actual, expected_score in zip(pair['opponent_scores'], scores, strict=True):
                if (not isinstance(actual, dict) or set(actual) != set(expected_score)
                        or any(type(actual[name]) is not int or actual[name] != expected_score[name]
                               for name in ('candidate_index', 'pair_count'))
                        or any(not close(actual[name], expected_score[name]) for name in (
                            'paired_mean_difference', 'exploration', 'score'))):
                    return False
            world, rows = pair['seed'], pair['samples']
            if (not seed(world) or world in used or not isinstance(rows, list) or len(rows) != 2
                    or any(not valid_sample(row, world) or type(row.get('candidate_index')) is not int
                        or row['candidate_index'] != index for row, index in zip(rows, (leader, opponent)))):
                return False
            used.add(world)
            delta = rows[1]['final_score'] - rows[0]['final_score']
            differences[min(leader, opponent), max(leader, opponent)].append(delta if leader < opponent else -delta)
            counts[leader] += 1
            counts[opponent] += 1
        remaining = budget - 2 * len(pairs)
        eligible = [index for index, count in enumerate(counts) if count < 32]
        reason = ('fewer_than_two_candidates_below_cap' if len(eligible) < 2 else
                  'remaining_budget_below_pair' if remaining < 2 else
                  'time_reserve_reached' if (evidence.get('time_guard') or {}).get('stopped_before_next_pair') is True else None)
        expected_statistics = [dict(lower_index=left, upper_index=right, count=len(values),
            difference_sum=sum(values), mean_difference=sum(values) / len(values))
            for (left, right), values in differences.items()]
        statistics = evidence['pair_statistics']
        if (reason is None or evidence['reason'] != reason
                or not _paired_adaptive_time_evidence(plan, evidence, samples)
                or type(evidence['remaining_budget']) is not int or evidence['remaining_budget'] != remaining
                or any(type(evidence[name]) is not int or evidence[name] != 2 * len(pairs)
                    for name in ('started_simulations', 'completed_simulations'))
                or evidence['candidate_sample_counts'] != counts
                or any(type(value) is not int for value in evidence['candidate_sample_counts'])
                or not isinstance(statistics, list) or len(statistics) != len(expected_statistics)):
            return False
        for actual, expected_statistic in zip(statistics, expected_statistics, strict=True):
            if (not isinstance(actual, dict) or set(actual) != set(expected_statistic)
                    or any(type(actual[name]) is not int or actual[name] != expected_statistic[name]
                        for name in ('lower_index', 'upper_index', 'count'))
                    or any(not close(actual[name], expected_statistic[name])
                        for name in ('difference_sum', 'mean_difference'))):
                return False
        final_regrets = regrets()
        selected = min(range(len(keys)), key=lambda index: (final_regrets[index], index))
        if guarded:
            retained_index = len(roots) if has_retained else None
            pair_count, direct_mean = 0, None
            if retained_index is None:
                guard_reason = 'no_retained_candidate'
            elif selected == retained_index:
                guard_reason = 'minimax_selected_retained'
            else:
                pair_count = len(differences[min(selected, retained_index), max(selected, retained_index)])
                direct_mean = -pair_mean(selected, retained_index)
                guard_reason = ('direct_paired_mean_positive' if direct_mean > 0
                                else 'direct_paired_mean_nonpositive')
            if (type(evidence['minimax_selected_index']) is not int
                    or evidence['minimax_selected_index'] != selected
                    or (evidence['retained_index'] is not None if retained_index is None else
                        type(evidence['retained_index']) is not int or evidence['retained_index'] != retained_index)
                    or type(evidence['retained_pair_count']) is not int or evidence['retained_pair_count'] != pair_count
                    or (evidence['retained_mean_difference'] is not None if direct_mean is None else
                        not close(evidence['retained_mean_difference'], direct_mean))
                    or evidence['retained_guard_reason'] != guard_reason):
                return False
            if guard_reason == 'direct_paired_mean_nonpositive':
                selected = retained_index
        if (not isinstance(evidence['paired_regrets'], list) or len(evidence['paired_regrets']) != len(keys)
                or any(not close(left, right) for left, right in zip(evidence['paired_regrets'], final_regrets))
                or type(evidence['selected_index']) is not int or evidence['selected_index'] != selected):
            return False
        expected_row = retained if selected == len(roots) else roots[selected]
        continuation = [] if selected == len(roots) else expected_row.get('continuation', [])
        return (_action_identity(plan.get('action', {})) == _action_identity(expected_row['action'])
            and _json_value(plan.get('selected_continuation', [])) == _json_value(continuation)
            and (not has_retained or type(retained.get('selected')) is bool
                 and retained['selected'] is (selected == len(roots))))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _uniform_validation_time_guard_mode(plan, *, expected=None):
    """Current execution binds v2; unspecified read audits retain sealed v1."""
    enabled = plan.get('uniform_validation_time_guard_enabled', False)
    version, evidence = (plan.get(name) for name in (
        'uniform_validation_time_guard_version', 'uniform_validation_time_guard'))
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return version is None and evidence is None
    versions = ((UNIFORM_VALIDATION_TIME_GUARD_VERSION, 'hif-uniform-validation-time-guard-v1')
        if expected is None else (UNIFORM_VALIDATION_TIME_GUARD_VERSION,))
    return (version in versions
        and _quality_reference_evidence(plan, expected=True)
        and _compute_sensitivity_evidence(plan, expected=False, simulations=256)
        and plan.get('paired_adaptive_validation_enabled', False) is False
        and plan.get('paired_adaptive_validation_version') is None
        and plan.get('paired_adaptive_validation') is None
        and plan.get('paired_finalist_validation', False) is False
        and (evidence is None if plan.get('scope') != 'SAMPLING_ESTIMATE' else
            isinstance(evidence, dict) and plan.get('adaptive_validation') is True
            and plan.get('scenario_reuse') is True
            and plan.get('lazy_validation_seeds_enabled') is True
            and plan.get('deterministic_validation_cohort_enabled') is True
            and plan.get('requested_simulations') == 256 and plan.get('requested_validation_samples') == 4
            and plan.get('continuation_samples', 0) == 0 and plan.get('inner_completed_simulations', 0) == 0))


def _uniform_validation_time_guard_evidence(plan, *, expected=None, expected_visited_card_version=None):
    """Rebuild complete common rounds, admission before seeds, and mean selection."""
    if not _uniform_validation_time_guard_mode(plan, expected=expected):
        return False
    if plan.get('uniform_validation_time_guard_enabled') is not True:
        return True

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def close(left, right):
        return number(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-7)

    version = plan['uniform_validation_time_guard_version']
    # v1 capped actual work at 30 despite its legacy quality codec being 300.
    # v2 uses the declared quality window; receipt and plan must bind that same
    # value, not a self-reported larger window or the remaining parent time.
    window = (30. if version == 'hif-uniform-validation-time-guard-v1'
        else plan['quality_reference_watchdog_seconds'])
    if not number(plan.get('elapsed_ms')) or not 0 <= plan['elapsed_ms'] <= window * 1000:
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return True
    try:
        evidence = plan['uniform_validation_time_guard']
        fields = {'version', 'minimum_rounds', 'delivery_reserve_seconds', 'window_limit_seconds',
            'capacity', 'root_count', 'retained_present', 'rounds', 'prestart_checks', 'completed_rounds',
            'max_completed_round_seconds', 'minimum_coverage', 'complete', 'stop_reason', 'elapsed_seconds',
            'deadline_remaining_seconds', 'finished_before_deadline', 'timing_scope'}
        roots = plan['candidate_statistics']
        retained = plan.get('retained_policy_comparison') or {}
        has_retained = _retained_policy_competition_active(plan)
        if (set(evidence) != fields or evidence['version'] != version
                or not isinstance(roots, (list, tuple)) or not roots
                or type(evidence['root_count']) is not int or evidence['root_count'] != len(roots)
                or type(evidence['retained_present']) is not bool or evidence['retained_present'] is not has_retained
                or type(evidence['minimum_rounds']) is not int or evidence['minimum_rounds'] != 4
                or not close(evidence['delivery_reserve_seconds'], 1.) or not close(evidence['window_limit_seconds'], window)
                or evidence['complete'] is not True or evidence['minimum_coverage'] is not True
                or evidence['finished_before_deadline'] is not True
                or not number(evidence['elapsed_seconds']) or not 0 < evidence['elapsed_seconds'] <= plan['elapsed_ms']/1000 + 1e-7
                or not number(evidence['deadline_remaining_seconds']) or evidence['deadline_remaining_seconds'] <= 0
                or evidence['elapsed_seconds'] + evidence['deadline_remaining_seconds'] > window + 1e-7):
            return False
        rounds, capacity = evidence['completed_rounds'], evidence['capacity']
        base = plan['training_completed_simulations'] + sum(
            (plan.get(name) or {}).get('completed_simulations', 0)
            for name in ('public_turn_reanalysis', 'resource_reanalysis', 'internal_reanalysis'))
        width = len(roots) + int(has_retained)
        if (any(type(value) is not int for value in (rounds, capacity, base)) or not 4 <= rounds <= capacity <= 32
                or capacity != min(32, (256 - base)//width) or plan['validation_sample_limit'] != capacity
                or plan.get('complete_validation_coverage') is not True
                or any(type(plan.get(name)) is not int or plan[name] != rounds * len(roots)
                    for name in ('validation_started_simulations', 'validation_completed_simulations'))
                or any(type(plan.get(name)) is not int or plan[name] != base + rounds * width
                    for name in ('started_simulations', 'completed_simulations'))
                or plan.get('unused_simulation_budget') != 256 - base - rounds * width):
            return False
        samples = [root['validation_samples'] for root in roots]
        if has_retained:
            if (retained.get('complete') is not True or any(type(retained.get(name)) is not int
                    or retained[name] != rounds for name in ('started_simulations', 'completed_simulations'))):
                return False
            samples.append(retained['validation_samples'])
        seeds = [sample['planning_seed'] for sample in samples[0]]
        if (len(seeds) != rounds or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or len(set(seeds)) != rounds
                or seeds != list(plan['validation_cohort_evidence']['validation_seeds'])):
            return False
        means = []
        for row in samples:
            if not isinstance(row, (tuple, list)) or len(row) != rounds:
                return False
            for index, sample in enumerate(row):
                if (not isinstance(sample, dict) or type(sample.get('scenario_index')) is not int
                        or sample['scenario_index'] != index or type(sample.get('planning_seed')) is not int
                        or sample['planning_seed'] != seeds[index] or not number(sample.get('final_score'))
                        or sample.get('return_kind') not in ('sampled_terminal_score',
                            'terminal_evaluator_expected_score', 'proven_tail_expected_score')
                        or type(sample.get('completed_depth')) is not int or not 0 <= sample['completed_depth'] <= 256
                        or type(sample.get('terminal_evaluations')) is not int or sample['terminal_evaluations'] < 0
                        or type(sample.get('terminal_fallbacks')) is not int or sample['terminal_fallbacks'] != 0
                        or 'complete' in sample and sample['complete'] is not True):
                    return False
            means.append(sum(sample['final_score'] for sample in row) / rounds)
        if (any(type(root.get('validation_paired_visits')) is not int or root['validation_paired_visits'] != rounds
                or not close(root.get('validation_mean_final_score'), mean)
                for root, mean in zip(roots, means, strict=False))
                or plan.get('validation_paired_visits') not in (list([rounds] * len(roots)), tuple([rounds] * len(roots)))
                or has_retained and not close(retained.get('mean_final_score'), means[-1])):
            return False
        allocation = plan['validation_seed_allocation']
        if (type(allocation.get('capacity')) is not int or allocation['capacity'] != capacity
                or allocation.get('mandatory') != 4 or any(type(allocation.get(name)) is not int
                    or allocation[name] != rounds for name in ('generated', 'started', 'complete_rounds'))):
            return False
        rows = evidence['rounds']
        row_fields = {'index', 'planning_seed', 'elapsed_seconds', 'root_started', 'root_completed',
                      'retained_started', 'retained_completed', 'complete'}
        if not isinstance(rows, list) or len(rows) != rounds:
            return False
        for index, row in enumerate(rows):
            expected_counts = dict(index=index, planning_seed=seeds[index], root_started=len(roots),
                root_completed=len(roots), retained_started=int(has_retained), retained_completed=int(has_retained))
            if (not isinstance(row, dict) or set(row) != row_fields or row['complete'] is not True
                    or any(type(row[key]) is not int or row[key] != value for key, value in expected_counts.items())):
                return False
        checks = evidence['prestart_checks']
        if not isinstance(checks, list):
            return False
        parallel = plan.get('parallel_root_validation') or {}
        if evidence['timing_scope'] == 'PARALLEL_BATCH_NO_OPTIONAL_ROUNDS':
            if (capacity != 4 or rounds != 4 or evidence['stop_reason'] != 'completed' or checks
                    or evidence['max_completed_round_seconds'] is not None
                    or any(row['elapsed_seconds'] is not None for row in rows)
                    or parallel.get('status') != 'complete' or not _parallel_root_validation_evidence(plan,
                        expected=None if expected is None else True,
                        expected_visited_card_version=expected_visited_card_version)):
                return False
        elif evidence['timing_scope'] == 'SERIAL_WHOLE_ROUND':
            if (parallel.get('status') == 'complete'
                    or any(not number(row['elapsed_seconds']) or row['elapsed_seconds'] <= 0 for row in rows)
                    or sum(row['elapsed_seconds'] for row in rows) > evidence['elapsed_seconds'] + 1e-7
                    or not close(evidence['max_completed_round_seconds'], max(row['elapsed_seconds'] for row in rows))):
                return False
            stopped = rounds < capacity
            if (evidence['stop_reason'] != ('time_reserve_reached' if stopped else 'completed')
                    or len(checks) != rounds - 4 + int(stopped)):
                return False
            previous_remaining = (evidence['elapsed_seconds'] + evidence['deadline_remaining_seconds']
                - sum(row['elapsed_seconds'] for row in rows[:4]))
            for index, check in enumerate(checks, start=4):
                maximum = max(row['elapsed_seconds'] for row in rows[:index])
                allowed = index < rounds
                if (not isinstance(check, dict) or set(check) != {'next_round_index', 'completed_rounds',
                        'remaining_seconds', 'max_completed_round_seconds', 'required_seconds', 'allowed'}
                        or any(type(check[name]) is not int or check[name] != index
                            for name in ('next_round_index', 'completed_rounds'))
                        or type(check['allowed']) is not bool or check['allowed'] is not allowed
                        or not close(check['max_completed_round_seconds'], maximum)
                        or not close(check['required_seconds'], maximum + 1.)
                        or not number(check['remaining_seconds']) or not 0 < check['remaining_seconds'] <= previous_remaining + 1e-7
                        or (check['remaining_seconds'] > maximum + 1.) is not allowed):
                    return False
                previous_remaining = check['remaining_seconds'] - (rows[index]['elapsed_seconds'] if allowed else 0.)
            if checks and evidence['deadline_remaining_seconds'] > previous_remaining + 1e-7:
                return False
        else:
            return False
        timing = plan['search_phase_timing']
        phases = timing['phases']
        root_seconds = phases['root_validation']['wall_ms']/1000
        retained_seconds = phases['retained_validation']['wall_ms']/1000
        if (timing.get('version') != SEARCH_PHASE_TIMING_VERSION
                or timing.get('scope') != 'DISJOINT_SUGGEST_CALLS_EXCLUDING_CANDIDATE_DELIVERY'
                or not number(timing.get('elapsed_ms')) or not 0 <= timing['elapsed_ms'] <= plan['elapsed_ms']
                or not number(timing.get('other_ms')) or timing['other_ms'] < 0
                or any(not isinstance(phase, dict) or type(phase.get('executed')) is not bool
                    or not number(phase.get('wall_ms')) or phase['wall_ms'] < 0
                    or phase['executed'] is False and phase['wall_ms'] != 0 for phase in phases.values())
                or not close(sum(phase['wall_ms'] for phase in phases.values()) + timing['other_ms'], timing['elapsed_ms'])
                or not number(root_seconds) or root_seconds <= 0
                or not number(retained_seconds) or retained_seconds < 0
                or phases['root_validation']['executed'] is not True
                or type(phases['retained_validation']['executed']) is not bool
                or phases['retained_validation']['executed'] is not has_retained
                or has_retained and retained_seconds <= 0
                or root_seconds + retained_seconds > evidence['elapsed_seconds'] + 1e-7
                or evidence['timing_scope'] == 'SERIAL_WHOLE_ROUND'
                    and root_seconds + retained_seconds > sum(row['elapsed_seconds'] for row in rows) + 1e-7):
            return False
        if not _root_selection_eligibility_evidence(plan):
            return False
        eligible, retained_eligible = _root_selection_domain(plan, len(roots), has_retained)
        winner = max(eligible, key=lambda index: means[index])
        retained_wins = retained_eligible and means[-1] >= means[winner]
        if has_retained and retained.get('selected') is not retained_wins:
            return False
        selected = retained if retained_wins else roots[winner]
        return (_action_identity(plan['action']) == _action_identity(selected['action'])
            and _json_value(plan.get('selected_continuation', [])) == _json_value(
                [] if retained_wins else selected.get('continuation', [])))
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def _quality_reference_work_evidence(plan):
    """Quality reference completes allocated work, allowing whole-round leftovers."""
    elapsed = plan.get('elapsed_ms')
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or not 0 <= elapsed <= _plan_window_ms(plan)):
        return False
    probe = plan.get('actual_two_turn_probe')
    if probe is not None:
        if (not isinstance(probe, dict) or probe.get('complete') is not True
                or probe.get('partial_values_used') is not False
                or plan.get('scope') != 'FINAL_SCORE' or plan.get('exact_root_turns') != 2
                or any(type(probe.get(name)) not in (int, float)
                    or not math.isfinite(probe[name]) or not 0 <= probe[name] <= _plan_window_ms(plan)
                    for name in ('budget_ms', 'elapsed_ms'))
                or probe['budget_ms'] <= 0 or probe['elapsed_ms'] > min(probe['budget_ms'], elapsed) + 1e-6):
            return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return True
    roots = plan.get('candidate_statistics')
    training = plan.get('training_simulation_budget')
    if (not isinstance(roots, (list, tuple)) or not roots
            or type(training) is not int or not 4*len(roots) <= training <= _plan_simulation_budget(plan)
            or plan.get('requested_simulations') != _plan_simulation_budget(plan)
            or plan.get('requested_validation_samples') != 4
            or plan.get('adaptive_validation') is not True
            or plan.get('terminal_fallbacks') != 0
            or any(plan.get(name) != 'completed' for name in (
                'stop_reason', 'training_stop_reason', 'validation_stop_reason'))
            or any(type(plan.get(name)) is not int or plan[name] != training for name in (
                'training_started_simulations', 'training_completed_simulations'))):
        return False
    base = training
    for name in ('public_turn_reanalysis', 'resource_reanalysis', 'internal_reanalysis'):
        row = plan.get(name)
        if row is None:
            continue
        if (not isinstance(row, dict) or type(row.get('started_simulations')) is not int
                or row['started_simulations'] < 0
                or type(row.get('completed_simulations')) is not int
                or row['completed_simulations'] != row['started_simulations']):
            return False
        base += row['completed_simulations']
    retained = _retained_policy_competition_active(plan)
    width = len(roots) + int(retained)
    paired = plan.get('paired_adaptive_validation_enabled') is True
    if paired and not _paired_adaptive_validation_evidence(plan):
        return False
    capacity = 4 if paired else min(32, (_plan_simulation_budget(plan)-base)//width)
    uniform = plan.get('uniform_validation_time_guard_enabled') is True
    if uniform and not _uniform_validation_time_guard_evidence(plan):
        return False
    rounds = plan['uniform_validation_time_guard']['completed_rounds'] if uniform else capacity
    if (rounds < 4 or type(plan.get('validation_sample_limit')) is not int
            or plan['validation_sample_limit'] != capacity
            or any(not isinstance(row, dict)
                or type(row.get('training_paired_visits')) is not int or row['training_paired_visits'] < 4
                or type(row.get('validation_paired_visits')) is not int
                or row['validation_paired_visits'] != rounds for row in roots)
            or any(type(plan.get(name)) is not int or plan[name] != rounds*len(roots) for name in (
                'validation_started_simulations', 'validation_completed_simulations'))):
        return False
    if retained and any(type(plan['retained_policy_comparison'].get(name)) is not int
            or plan['retained_policy_comparison'][name] != rounds
            for name in ('started_simulations', 'completed_simulations')):
        return False
    total = base + rounds*width + _paired_adaptive_completed(plan)
    return (all(type(plan.get(name)) is int and plan[name] == total
                for name in ('started_simulations', 'completed_simulations'))
        and type(plan.get('unused_simulation_budget')) is int
        and plan['unused_simulation_budget'] == _plan_simulation_budget(plan)-total)


def _public_turn_budget_reservation_evidence(plan, *, expected=None):
    """Prove complete graph cost was reserved without borrowing root coverage."""
    enabled = plan.get('public_turn_budget_reservation_enabled', False)
    version = plan.get('public_turn_budget_reservation_version')
    evidence = plan.get('public_turn_budget_reservation')
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    if not enabled:
        return version is None and evidence is None
    fields = {'version', 'eligible', 'complete_graph', 'required_simulations',
        'required_simulations_lower_bound', 'training_budget_before', 'training_budget_after',
        'minimum_training', 'minimum_validation', 'graph_rollout_budget',
        'structure_prepared_once', 'reason', 'prepare_elapsed_ms'}
    if (version != PUBLIC_TURN_BUDGET_RESERVATION_VERSION
            or not isinstance(evidence, dict) or set(evidence) != fields
            or evidence['version'] != version
            or plan.get('actual_tail_session_reuse_enabled') is not True
            or not _public_turn_score_translation_evidence(plan, expected=True)):
        return False
    integers = ('required_simulations_lower_bound', 'training_budget_before',
        'training_budget_after', 'minimum_training', 'minimum_validation', 'graph_rollout_budget')
    elapsed = plan.get('elapsed_ms')
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= _plan_window_ms(plan)
            or any(type(evidence[name]) is not int or evidence[name] < 0 for name in integers)
            or any(type(evidence[name]) is not bool for name in
                ('eligible', 'complete_graph', 'structure_prepared_once'))
            or evidence['eligible'] != evidence['complete_graph']
            or type(evidence['prepare_elapsed_ms']) not in (int, float)
            or not math.isfinite(evidence['prepare_elapsed_ms'])
            or not 0 <= evidence['prepare_elapsed_ms'] <= elapsed
            or not isinstance(evidence['reason'], str)):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return (evidence['reason'] == 'not_run' and evidence['required_simulations'] is None
            and not evidence['eligible'] and not evidence['structure_prepared_once']
            and evidence['prepare_elapsed_ms'] == 0 and all(evidence[name] == 0 for name in integers))
    roots = plan.get('candidate_statistics')
    if not isinstance(roots, (list, tuple)) or not roots:
        return False
    minimum = 4*len(roots)
    minimum_validation = minimum + (4 if _retained_policy_competition_active(plan) else 0)
    before, after = evidence['training_budget_before'], evidence['training_budget_after']
    effective_after = after
    internal = plan.get('internal_reanalysis')
    if (plan.get('internal_reanalysis_enabled') is True and isinstance(internal, dict)
            and internal.get('total_budget') == _plan_simulation_budget(plan) and type(internal.get('training_limit')) is int):
        if not _internal_reservation_release_evidence(plan):
            return False
        effective_after = min(after, internal['reservation_release']['effective_training_limit'])
    if (plan.get('requested_simulations') != _plan_simulation_budget(plan)
            or evidence['minimum_training'] != minimum or evidence['minimum_validation'] != minimum_validation
            or evidence['graph_rollout_budget'] != _plan_simulation_budget(plan)-minimum-minimum_validation
            or not minimum <= after <= before <= _plan_simulation_budget(plan)-minimum_validation
            or plan.get('training_simulation_budget') != effective_after
            or any(type(plan.get(name)) is not int or not minimum <= plan[name] <= effective_after
                   for name in ('training_started_simulations', 'training_completed_simulations'))):
        return False
    required = evidence['required_simulations']
    graph = plan.get('public_turn_reanalysis')
    if not isinstance(graph, dict):
        return False
    if evidence['complete_graph']:
        return (type(required) is int and required > 0
            and evidence['structure_prepared_once'] and evidence['reason'] == 'reserved'
            and evidence['required_simulations_lower_bound'] == required
            and required <= evidence['graph_rollout_budget']
            and after == min(before, _plan_simulation_budget(plan)-required-minimum_validation)
            and graph.get('complete') is True and graph.get('required_simulations') == required
            and graph.get('completed_simulations') == required
            and graph.get('started_simulations') == required)
    return (required is None and after == before and graph.get('status') == 'declined'
        and graph.get('reason') == evidence['reason']
        and graph.get('required_simulations') == evidence['required_simulations_lower_bound']
        and graph.get('started_simulations') == graph.get('completed_simulations') == 0
        and (evidence['structure_prepared_once'] or evidence['reason'] == 'unsupported_root_or_sampler'))


def _public_validation_phase_budget_evidence(plan, *, expected=None, expected_soft_window_ms=None):
    """Audit a complete graph's separate outer clock, without treating it as new hard budget."""
    enabled = plan.get('public_validation_phase_budget_enabled', False)
    version = plan.get('public_validation_phase_budget_version')
    evidence = plan.get('public_validation_phase_budget')
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    if not enabled:
        return version is None and evidence is None
    times = ('soft_window_ms', 'training_elapsed_ms', 'graph_elapsed_ms',
        'validation_start_elapsed_ms', 'soft_deadline_elapsed_ms', 'validation_elapsed_ms')
    fields = {'version', 'eligible', 'reason', 'complete_rounds', *times}
    quality = plan.get('quality_reference_enabled') is True and _quality_reference_evidence(plan)
    if quality:
        fields.update(('allocation_mode', 'soft_disabled'))
    if (version != PUBLIC_VALIDATION_PHASE_BUDGET_VERSION
            or not isinstance(evidence, dict) or set(evidence) != fields
            or evidence['version'] != version
            or plan.get('public_turn_budget_reservation_enabled') is not True
            or plan.get('public_turn_budget_reservation_version') != PUBLIC_TURN_BUDGET_RESERVATION_VERSION
            or type(evidence['eligible']) is not bool
            or type(evidence['complete_rounds']) is not int or evidence['complete_rounds'] < 0
            or any(type(evidence[name]) not in (int, float) or not math.isfinite(evidence[name])
                   or evidence[name] < 0 for name in times)
            or quality and (evidence.get('allocation_mode') != 'complete_planned_work'
                or evidence.get('soft_disabled') is not True)):
        return False
    elapsed = plan.get('elapsed_ms')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= _plan_window_ms(plan):
        return False
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return (not evidence['eligible'] and evidence['reason'] == 'not_run'
            and evidence['complete_rounds'] == 0 and all(evidence[name] == 0 for name in times))
    graph = plan.get('public_turn_reanalysis')
    if not isinstance(graph, dict):
        return False
    complete_graph = graph.get('complete') is True and graph.get('status') == 'complete'
    if not evidence['eligible']:
        return (not complete_graph and evidence['reason'] == 'public_graph_incomplete'
            and evidence['complete_rounds'] == 0 and all(evidence[name] == 0 for name in times))
    if not complete_graph or evidence['reason'] != 'complete_public_graph':
        return False
    soft = evidence['soft_window_ms']
    start = evidence['validation_start_elapsed_ms']
    deadline = evidence['soft_deadline_elapsed_ms']
    # Completed paired rounds may overrun the soft target. The existing whole
    # window audit still requires the real hard limit and complete coverage.
    if (quality and soft != 0 or not quality and (soft <= 0 or expected_soft_window_ms is not None and (
            type(expected_soft_window_ms) not in (int, float)
            or not math.isfinite(expected_soft_window_ms) or soft != expected_soft_window_ms))
            or not math.isclose(start, evidence['training_elapsed_ms'] + evidence['graph_elapsed_ms'],
                                rel_tol=1e-9, abs_tol=1e-6)
            or not start <= deadline <= (min(start + soft, _plan_window_ms(plan))
                                         if not quality else _plan_window_ms(plan)) + 1e-6
            or start + evidence['validation_elapsed_ms'] > elapsed + 1e-6):
        return False
    roots = plan.get('candidate_statistics')
    if (not isinstance(roots, (tuple, list)) or not roots
            or any(not isinstance(row, dict) or type(row.get('validation_paired_visits')) is not int
                   or row['validation_paired_visits'] < 0 for row in roots)):
        return False
    rounds = [row['validation_paired_visits'] for row in roots]
    if _retained_policy_competition_active(plan):
        retained_rounds = plan['retained_policy_comparison'].get('completed_simulations')
        if type(retained_rounds) is not int or retained_rounds < 0:
            return False
        rounds.append(retained_rounds)
    return evidence['complete_rounds'] == min(rounds)


def _parallel_root_validation_mode(plan, *, expected=None):
    enabled = plan.get('parallel_root_validation_enabled', False)
    version, evidence = (plan.get(name) for name in (
        'parallel_root_validation_version', 'parallel_root_validation'))
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return version is None and evidence is None
    return (version == PARALLEL_ROOT_VALIDATION_VERSION
        and plan.get('parallel_public_graph_enabled') is True
        and plan.get('parallel_public_graph_version') == PARALLEL_PUBLIC_GRAPH_VERSION
        and (plan.get('paired_adaptive_validation_enabled') is True
        and plan.get('paired_adaptive_validation_version') in (
            (PAIRED_ADAPTIVE_VALIDATION_VERSION, 'hif-paired-adaptive-validation-v2')
            if expected is None else (PAIRED_ADAPTIVE_VALIDATION_VERSION,))
        or plan.get('uniform_validation_time_guard_enabled') is True
        and _uniform_validation_time_guard_mode(plan, expected=None if expected is None else True)))


def _parallel_root_validation_evidence(plan, *, expected=None, expected_visited_card_version=None):
    """Bind four worker worlds to every complete root and its measured return.

    Missing mode fields are accepted only for explicitly old/unspecified read
    audits. Current execution binds both opt-ins to the new version separately.
    Incomplete worker output remains paid evidence, never an accepted table.
    """
    if not _parallel_root_validation_mode(plan, expected=expected):
        return False
    if plan.get('parallel_root_validation_enabled') is not True:
        return True
    evidence = plan.get('parallel_root_validation')
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def close(left, right):
        return number(left) and number(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-7)

    try:
        roots = plan['candidate_statistics']
        if not isinstance(roots, (tuple, list)) or not roots or not isinstance(evidence, dict):
            return False
        keys = [(_action_identity(root['action']), tuple(map(_action_identity, root.get('continuation', ()))))
                for root in roots]
        if any('root_option_key' in root and _json_value(root['root_option_key']) != _json_value(key)
               for root, key in zip(roots, keys, strict=True)):
            return False
        seeds = evidence.get('seeds')
        if (evidence.get('version') != PARALLEL_ROOT_VALIDATION_VERSION
                or type(evidence.get('workers')) is not int or evidence['workers'] != 4
                or _json_value(evidence.get('root_keys')) != _json_value(keys)
                or len({json.dumps(_json_value(key)) for key in keys}) != len(keys)
                or not isinstance(seeds, (tuple, list)) or len(seeds) != 4
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or len(set(seeds)) != 4):
            return False
        cohort = plan.get('validation_cohort_evidence')
        nonfour = (plan.get('uniform_validation_time_guard_enabled') is True
            and type(plan.get('validation_sample_limit')) is int and plan['validation_sample_limit'] > 4)
        if (not isinstance(cohort, dict) or not isinstance(cohort.get('validation_seeds'), (tuple, list))
                or _json_value(cohort['validation_seeds'][:4] if nonfour else cohort['validation_seeds']) != _json_value(seeds)):
            return False
        required = 4 * len(roots)
        if evidence.get('status') == 'serial':
            reason = evidence.get('reason')
            if nonfour:
                domain = reason == 'validation_limit_not_four'
            elif reason == 'below_minimum_simulations':
                domain = required < PARALLEL_ROOT_VALIDATION_MIN_SIMULATIONS
            else:
                domain = required >= PARALLEL_ROOT_VALIDATION_MIN_SIMULATIONS and reason in (
                    'unsupported_public_or_resource_policy', 'unsupported_owner_or_proven_tail',
                    'unsupported_internal_policy')
            return (domain and evidence.get('complete') is False
                and evidence.get('count_incomplete') is False and evidence.get('worker_records') == []
                and evidence.get('errors', []) == [] and number(evidence.get('elapsed_seconds'))
                and evidence['elapsed_seconds'] == 0.
                and all(type(evidence.get(name)) is int and evidence[name] == 0
                        for name in ('started_simulations', 'completed_simulations')))
        seconds, remaining = evidence.get('elapsed_seconds'), evidence.get('deadline_remaining_seconds')
        records = evidence.get('worker_records')
        if (evidence.get('status') != 'complete' or evidence.get('reason') != 'four_worlds_complete'
                or type(plan.get('validation_sample_limit')) is not int or plan['validation_sample_limit'] != 4
                or not PARALLEL_ROOT_VALIDATION_MIN_SIMULATIONS <= required <= 256
                or evidence.get('complete') is not True or evidence.get('count_incomplete') is not False
                or evidence.get('parent_frozen_state_unchanged') is not True or evidence.get('errors') != []
                or evidence.get('finished_before_deadline') is not True
                or not number(seconds) or not number(remaining) or not 0 < seconds < remaining <= _plan_window_ms(plan)/1000
                or not isinstance(records, (tuple, list)) or len(records) != 4
                or any(type(evidence.get(name)) is not int or evidence[name] != required
                       for name in ('started_simulations', 'completed_simulations'))
                or any(type(plan.get(name)) is not int or plan[name] != required
                       for name in ('validation_started_simulations', 'validation_completed_simulations'))):
            return False
        if (not isinstance(plan.get('validation_paired_visits'), (tuple, list))
                or any(type(value) is not int for value in plan['validation_paired_visits'])
                or tuple(plan['validation_paired_visits']) != (4,) * len(roots)):
            return False
        counter_names = {'_validation_tail_hits', '_validation_tail_misses',
                         '_validation_terminal_eligible', '_validation_terminal_declined'}
        helper_checks = (
            ('unknown_node_timing', _unknown_node_timing_evidence),
            ('partial_node_timing', _partial_node_timing_evidence),
            ('installed_score_turn_chain', _installed_score_turn_chain_evidence),
            ('extra_action_turn_chain', _extra_action_turn_chain_evidence),
            ('pair_order_advice', _pair_order_advice_evidence),
        )
        for world, record in enumerate(records):
            if (not isinstance(record, dict) or type(record.get('world_index')) is not int
                    or record['world_index'] != world or type(record.get('seed')) is not int
                    or record['seed'] != seeds[world] or record.get('complete') is not True
                    or record.get('count_incomplete') is not False or record.get('errors') != []
                    or record.get('tree_rng_namespace_unchanged') is not True
                    or any(type(record.get(name)) is not int or record[name] != len(roots)
                           for name in ('started_simulations', 'completed_simulations'))
                    or any(not number(record.get(name)) or not 0 <= record[name] <= seconds
                           for name in ('setup_seconds', 'evaluation_seconds', 'elapsed_seconds'))
                    or record['setup_seconds'] + record['evaluation_seconds'] > record['elapsed_seconds'] + 1e-7):
                return False
            rows = record.get('rows')
            if not isinstance(rows, (tuple, list)) or len(rows) != len(roots):
                return False
            for index, (row, root, key) in enumerate(zip(rows, roots, keys, strict=True)):
                if (not isinstance(row, dict) or type(row.get('group_index')) is not int
                        or row['group_index'] != index or type(row.get('seed')) is not int
                        or row['seed'] != seeds[world] or _json_value(row.get('key')) != _json_value(key)
                        or type(row.get('prefix_depth')) is not int or row['prefix_depth'] != 0
                        or not number(row.get('final_score')) or not number(row.get('elapsed_seconds'))
                        or not 0 < row['elapsed_seconds'] <= record['evaluation_seconds']
                        or any(type(row.get(name)) is not int or row[name] < 0
                               for name in ('depth', 'terminal_calls', 'terminal_fallbacks'))
                        or row['depth'] >= 256 or row['terminal_fallbacks'] != 0):
                    return False
                counters = row.get('counters')
                if (not isinstance(counters, dict) or set(counters) != counter_names
                        or any(type(value) is not int or value < 0 for value in counters.values())):
                    return False
                kind = ('proven_tail_expected_score' if counters['_validation_tail_hits'] > 0
                    else 'terminal_evaluator_expected_score' if row['terminal_calls'] > row['terminal_fallbacks']
                    else 'sampled_terminal_score')
                samples = root.get('validation_samples')
                if (not isinstance(samples, (tuple, list)) or len(samples) != 4
                        or type(root.get('validation_paired_visits')) is not int
                        or root['validation_paired_visits'] != 4 or row.get('return_kind') != kind):
                    return False
                sample = samples[world]
                expected_sample = dict(scenario_index=world, planning_seed=seeds[world],
                    final_score=row['final_score'], return_kind=kind, completed_depth=row['depth'],
                    terminal_evaluations=row['terminal_calls'], terminal_fallbacks=row['terminal_fallbacks'],
                    elapsed_seconds=row['elapsed_seconds'])
                if sample != expected_sample:
                    return False
            if sum(row['elapsed_seconds'] for row in rows) > record['evaluation_seconds'] + 1e-7:
                return False
            summaries = record.get('helper_summaries')
            if not isinstance(summaries, dict) or any(name not in summaries for name, _ in helper_checks):
                return False
            worker_plan = {**plan, **summaries}
            if plan.get('visited_card_extra_action_enabled', False) and (
                    not isinstance(summaries['extra_action_turn_chain'], dict)
                    or not isinstance(plan.get('extra_action_turn_chain'), dict)
                    or summaries['extra_action_turn_chain'].get('version') != plan['extra_action_turn_chain'].get('version')):
                return False
            if any(not check(worker_plan, expected=None if name == 'extra_action_turn_chain' and expected is None else True, **(
                    {'expected_visited_card': plan.get('visited_card_extra_action_enabled', False),
                     'expected_visited_card_version': expected_visited_card_version}
                    if name == 'extra_action_turn_chain' else {})) for name, check in helper_checks):
                return False
            for name in ('terminal_cache', 'complete_tail_cache'):
                cache = summaries.get(name)
                if (not isinstance(cache, dict) or set(cache) != {'decisions', 'memo_hits', 'namespaces'}
                        or any(type(value) is not int or value < 0 for value in cache.values())):
                    return False
            double = summaries.get('double_advice')
            if (not isinstance(double, dict) or set(double) != {'calls', 'active_calls', 'bypassed_calls',
                    'scope_bypassed_calls', 'resource_scope_bypassed_calls'}
                    or any(type(value) is not int or value < 0 for value in double.values())):
                return False
            reuse = record.get('step_reuse')
            if (not isinstance(reuse, dict) or reuse.get('version') != PUBLIC_STEP_REUSE_VERSION
                    or reuse.get('complete_returns_enabled') is not True
                    or any(type(value) is not int or value < 0 for key, value in reuse.items()
                           if key not in ('version', 'complete_returns_enabled'))
                    or not {'action_computations', 'action_hits', 'action_entries', 'step_computations',
                        'step_hits', 'step_entries', 'step_bypasses', 'return_entries', 'return_hits',
                        'return_steps_saved', 'return_tail_calls_saved'} <= set(reuse)):
                return False
        timing = plan.get('search_phase_timing')
        phases = timing.get('phases') if isinstance(timing, dict) else None
        if (not isinstance(timing, dict) or timing.get('version') != SEARCH_PHASE_TIMING_VERSION
                or timing.get('scope') != 'DISJOINT_SUGGEST_CALLS_EXCLUDING_CANDIDATE_DELIVERY'
                or not number(timing.get('elapsed_ms')) or not number(plan.get('elapsed_ms'))
                or not 0 < timing['elapsed_ms'] <= plan['elapsed_ms'] <= _plan_window_ms(plan)
                or not number(timing.get('other_ms')) or timing['other_ms'] < 0 or not isinstance(phases, dict)
                or any(not isinstance(phase, dict) or type(phase.get('executed')) is not bool
                       or not number(phase.get('wall_ms')) or phase['wall_ms'] < 0
                       or phase['executed'] is False and phase['wall_ms'] != 0 for phase in phases.values())
                or not close(sum(phase['wall_ms'] for phase in phases.values()) + timing['other_ms'], timing['elapsed_ms'])
                or phases.get('root_validation', {}).get('executed') is not True
                or not seconds * 1000 <= phases['root_validation']['wall_ms'] <= timing['elapsed_ms']):
            return False
        retained = plan.get('retained_policy_comparison') or {}
        serial_seconds = (phases.get('retained_validation', {}).get('wall_ms', 0.)/1000
            if plan.get('uniform_validation_time_guard_enabled') is True else
            sum(row['elapsed_seconds'] for row in retained.get('validation_samples', ())))
        if not number(serial_seconds) or serial_seconds < 0 or seconds + serial_seconds > plan['elapsed_ms']/1000 + 1e-7:
            return False
        phase = plan.get('public_validation_phase_budget') or {}
        if phase.get('eligible') is True and (not number(phase.get('validation_elapsed_ms'))
                or (seconds + serial_seconds) * 1000 > phase['validation_elapsed_ms'] + 1e-7):
            return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def _parallel_public_graph_evidence(plan, *, expected=None, expected_visited_card_version=None):
    """Four worker worlds supply the same graph table, never extra samples.

    Existing plan helper summaries describe the parent process only. Worker
    summaries remain separately auditable and are never added to those totals.
    Only the unspecified read-only mode accepts historical v2/128 evidence.
    """
    enabled = plan.get('parallel_public_graph_enabled', False)
    version = plan.get('parallel_public_graph_version')
    graph = plan.get('public_turn_reanalysis')
    evidence = graph.get('parallel_public_graph') if isinstance(graph, dict) else None
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return version is None and evidence is None
    minimum = (128 if expected is None and version == 'hif-parallel-public-graph-v2'
               else PARALLEL_PUBLIC_GRAPH_MIN_SIMULATIONS if version == PARALLEL_PUBLIC_GRAPH_VERSION else None)
    if (minimum is None
            or any(plan.get(name) is not True for name in (
                'extra_action_turn_chain_enabled', 'public_validation_phase_budget_enabled'))):
        return False
    parallel_required = (isinstance(graph, dict) and graph.get('complete') is True
        and type(graph.get('required_simulations')) is int and graph['required_simulations'] >= minimum)
    if evidence is None:
        return not parallel_required
    if (not parallel_required or not isinstance(evidence, dict)
            or evidence.get('version') != version
            or type(evidence.get('workers')) is not int or evidence['workers'] != 4
            or evidence.get('complete') is not True
            or evidence.get('count_incomplete') is not False):
        return False
    required = graph['required_simulations']
    groups, seeds = graph.get('boundary_groups'), graph.get('planning_seeds')
    records = evidence.get('worker_records')
    seconds = evidence.get('elapsed_seconds')
    if (not isinstance(groups, (tuple, list)) or len(groups) * 4 != required
            or not isinstance(seeds, (tuple, list)) or len(seeds) != 4
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 4
            or not isinstance(records, (tuple, list)) or len(records) != 4
            or type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 <= seconds <= _plan_window_ms(plan)/1000.
            or any(type(evidence.get(name)) is not int or evidence[name] != required
                   or graph.get(name) != required
                   for name in ('started_simulations', 'completed_simulations'))):
        return False
    counter_names = ('_validation_tail_hits', '_validation_tail_misses',
                     '_validation_terminal_eligible', '_validation_terminal_declined')
    for world, record in enumerate(records):
        if (not isinstance(record, dict) or type(record.get('world_index')) is not int
                or record['world_index'] != world or record.get('seed') != seeds[world]
                or type(record.get('seed')) is not int or record.get('complete') is not True
                or record.get('count_incomplete') is not False
                or record.get('tree_rng_namespace_unchanged') is not True or record.get('errors') != []
                or any(type(record.get(name)) is not int or record[name] != len(groups)
                       for name in ('started_simulations', 'completed_simulations'))):
            return False
        for name in ('setup_seconds', 'evaluation_seconds', 'elapsed_seconds'):
            value = record.get(name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= _plan_window_ms(plan)/1000.):
                return False
        rows = record.get('rows')
        if not isinstance(rows, (tuple, list)) or len(rows) != len(groups):
            return False
        for index, (row, group) in enumerate(zip(rows, groups)):
            if not isinstance(row, dict) or not isinstance(group, dict):
                return False
            scores, group_seeds = group.get('final_scores'), group.get('planning_seeds')
            if (not isinstance(scores, (tuple, list)) or len(scores) != 4
                    or not isinstance(group_seeds, (tuple, list)) or tuple(group_seeds) != tuple(seeds)
                    or type(row.get('group_index')) is not int or row['group_index'] != index
                    or type(row.get('seed')) is not int or row['seed'] != seeds[world]
                    or type(row.get('prefix_depth')) is not int or row['prefix_depth'] < 0
                    or row['prefix_depth'] != group.get('prefix_depth')
                    or not isinstance(row.get('key'), (tuple, list)) or len(row['key']) != 5
                    or type(row.get('final_score')) not in (int, float)
                    or not math.isfinite(row['final_score']) or row['final_score'] != scores[world]
                    or any(type(row.get(name)) is not int or row[name] < 0
                           for name in ('depth', 'terminal_calls', 'terminal_fallbacks'))
                    or not row['prefix_depth'] <= row['depth'] < 256
                    or row['terminal_fallbacks'] > row['terminal_calls']):
                return False
            counters = row.get('counters')
            if (not isinstance(counters, dict) or set(counters) != set(counter_names)
                    or any(type(value) is not int or value < 0 for value in counters.values())):
                return False
        # All workers evaluated the same ordered boundaries; only the seed and
        # continuation returns differ. No policy-count aggregation is inferred.
        if world and any(tuple(row['key']) != tuple(records[0]['rows'][index]['key'])
                         for index, row in enumerate(rows)):
            return False
        summaries = record.get('helper_summaries')
        helper_checks = (
            ('unknown_node_timing', _unknown_node_timing_evidence),
            ('partial_node_timing', _partial_node_timing_evidence),
            ('installed_score_turn_chain', _installed_score_turn_chain_evidence),
            ('extra_action_turn_chain', _extra_action_turn_chain_evidence),
            ('pair_order_advice', _pair_order_advice_evidence),
        )
        if not isinstance(summaries, dict) or any(name not in summaries for name, _ in helper_checks):
            return False
        worker_plan = {**plan, **summaries}
        if plan.get('visited_card_extra_action_enabled', False) and (
                not isinstance(summaries['extra_action_turn_chain'], dict)
                or not isinstance(plan.get('extra_action_turn_chain'), dict)
                or summaries['extra_action_turn_chain'].get('version') != plan['extra_action_turn_chain'].get('version')):
            return False
        if any(not check(worker_plan, expected=None if name == 'extra_action_turn_chain' and expected is None else True, **(
                {'expected_visited_card': plan.get('visited_card_extra_action_enabled', False),
                 'expected_visited_card_version': expected_visited_card_version}
                if name == 'extra_action_turn_chain' else {})) for name, check in helper_checks):
            return False
        for name in ('terminal_cache', 'complete_tail_cache'):
            cache = summaries.get(name)
            if (not isinstance(cache, dict) or set(cache) != {'decisions', 'memo_hits', 'namespaces'}
                    or any(type(value) is not int or value < 0 for value in cache.values())):
                return False
        double = summaries.get('double_advice')
        if (not isinstance(double, dict)
                or set(double) != {'calls', 'active_calls', 'bypassed_calls',
                                   'scope_bypassed_calls', 'resource_scope_bypassed_calls'}
                or any(type(value) is not int or value < 0 for value in double.values())):
            return False
        reuse = record.get('step_reuse')
        if (not isinstance(reuse, dict) or reuse.get('version') != PUBLIC_STEP_REUSE_VERSION
                or reuse.get('complete_returns_enabled') is not True
                or any(type(value) is not int or value < 0 for key, value in reuse.items()
                       if key not in ('version', 'complete_returns_enabled'))
                or not {'action_computations', 'action_hits', 'action_entries',
                        'step_computations', 'step_hits', 'step_entries', 'step_bypasses',
                        'return_entries', 'return_hits', 'return_steps_saved',
                        'return_tail_calls_saved'} <= set(reuse)):
            return False
    for name in counter_names:
        graph_name = name.removeprefix('_validation_')
        total = sum(row['counters'][name] for record in records for row in record['rows'])
        if type(graph.get(graph_name)) is not int or graph[graph_name] != total:
            return False
    return True


def _actual_tail_session_reuse_evidence(plans, index, plan, *, expected=None,
                                      plan_offset=0, actions=None, verified=True):
    """Audit saved lineage; utility/rules/slots are independently checked at runtime."""
    enabled = plan.get('actual_tail_session_reuse_enabled', False)
    version = plan.get('actual_tail_session_reuse_version')
    evidence = plan.get('actual_tail_session_reuse')
    if type(enabled) is not bool or expected is not None and enabled is not expected:
        return False
    if not enabled:
        return version is None and evidence is None and 'actual_tail_session_reuse_verified' not in plan
    if (version != ACTUAL_TAIL_SESSION_REUSE_VERSION
            or not _public_turn_score_translation_evidence(plan, expected=True)
            or not isinstance(evidence, dict)
            or set(evidence) != {'reused', 'source_plan_index', 'entries_before', 'entries_after'}
            or type(evidence.get('reused')) is not bool
            or any(type(evidence.get(name)) is not int or evidence[name] < 0
                   for name in ('entries_before', 'entries_after'))):
        return False
    source, reused = evidence['source_plan_index'], evidence['reused']
    if not reused:
        if source is not None or evidence['entries_before'] != 0:
            return False
    else:
        if (type(plan_offset) is not int or plan_offset < 0 or type(source) is not int
                or index < 1 or source != plan_offset + index - 1
                or not 0 < evidence['entries_before'] <= 32768
                or verified and plan.get('actual_tail_session_reuse_verified') is not True):
            return False
        previous = plans[source - plan_offset]
        previous_summary = previous.get('actual_tail_session_reuse')
        cache = previous.get('complete_tail_cache')
        if (previous.get('accepted') is not True
                or previous.get('actual_tail_session_reuse_enabled') is not True
                or previous.get('actual_tail_session_reuse_version') != ACTUAL_TAIL_SESSION_REUSE_VERSION
                or previous.get('turn') != plan.get('turn')
                or not isinstance(previous_summary, dict)
                or type(previous_summary.get('entries_after')) is not int
                or previous_summary['entries_after'] != evidence['entries_before']
                or not isinstance(cache, dict) or any(type(cache.get(name)) is not int
                    or cache[name] < 0 for name in ('decisions', 'memo_hits', 'namespaces'))):
            return False
    if actions is not None:
        if not isinstance(actions, (tuple, list)) or index >= len(actions):
            return False
        current = actions[index].get('before')
        if not isinstance(current, dict):
            return False
        if current.get('turns_remaining') != 2:
            return not reused and evidence['entries_after'] == 0
        if reused:
            from agent.live_decision.high_score.kernel import PythonNiaProKernel
            from agent.live_decision.high_score.hif_mcts_cache import exact_state

            previous_action = actions[index - 1]
            before, after = previous_action.get('before'), previous_action.get('after')
            if (not isinstance(before, dict) or not isinstance(after, dict)
                    or before.get('turns_remaining') != 2 or after.get('turns_remaining') != 2
                    or not before.get('turn') == after.get('turn') == current.get('turn') == plan.get('turn')
                    or not isinstance(previous_action.get('action'), dict)
                    or _action_identity(previous_action['action']) != _action_identity(plans[index - 1].get('action', {}))):
                return False
            try:
                canonical = PythonNiaProKernel._canonical_probability_state
                # Episode actions are native asdict values until report writing;
                # normalize enum/tuple containers before the strict JSON codec.
                if canonical(exact_state(_json_value(after))) != canonical(exact_state(_json_value(current))):
                    return False
            except (TypeError, ValueError, KeyError):
                return False
    return True


def _qualified_public_turn_source(plan):
    roots = plan.get('candidate_statistics')
    table = plan.get('public_turn_reanalysis')
    if _retained_policy_competition_active(plan) and (
            not _retained_policy_comparison_evidence(plan)
            or plan['retained_policy_comparison']['selected']):
        return False
    if (not _compute_sensitivity_evidence(plan) or not _paired_adaptive_validation_evidence(plan)
            or plan.get('scope') != 'SAMPLING_ESTIMATE' or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('complete_validation_coverage') is not True
            or plan.get('requested_simulations') != _plan_simulation_budget(plan) or plan.get('requested_validation_samples') != 4
            or type(plan.get('completed_simulations')) is not int
            or not 0 < plan['completed_simulations'] <= _plan_simulation_budget(plan)
            or plan.get('stop_reason') not in ('completed', 'soft_budget_reached')
            or not isinstance(roots, (list, tuple)) or not roots or not isinstance(table, dict)
            or table.get('complete') is not True or not _public_turn_reanalysis_evidence(plan, roots)):
        return False
    cohorts = []
    for row in roots:
        if (any(type(row.get(field)) is not int or row[field] < 4 for field in (
                'training_paired_visits', 'validation_paired_visits'))
                or not isinstance(row.get('validation_samples'), (list, tuple))
                or len(row['validation_samples']) != row['validation_paired_visits']):
            return False
        seeds = [sample.get('planning_seed') for sample in row['validation_samples']]
        if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            return False
        cohorts.append(seeds)
    return all(seeds == cohorts[0] for seeds in cohorts)


def _public_turn_source_receipt(origin):
    # Independently reconstructed from the accepted report, not helper claims.
    action = dict(origin['action'])
    if action.get('kind') != 'USE_DRINK':
        action['source_slot'] = None
    roots = origin['candidate_statistics']
    return dict(source_scope='SAMPLING_ESTIMATE', source_accepted=True,
        source_root_action=action, complete_root_coverage=True, complete_validation_coverage=True,
        requested_simulations=_plan_simulation_budget(origin), completed_simulations=origin['completed_simulations'],
        training_paired_visits=tuple(row['training_paired_visits'] for row in roots),
        validation_paired_visits=tuple(row['validation_paired_visits'] for row in roots),
        root_selection_scope=origin['root_selection_scope'],
        public_table_complete=True, public_table_version=PUBLIC_TURN_VERSION, **_source_compute_receipt(origin))


def _public_turn_continuation_evidence(plans, index, plan, *, verified=True, plan_offset=0):
    source, predecessor = plan.get('source_plan_index'), plan.get('predecessor_plan_index')
    proof, node = plan.get('continuation_proof'), plan.get('public_turn_node')
    keys = {'before', 'action', 'expected_state', 'before_drink_slots', 'expected_drink_slots',
            'committed', 'source_plan_index', 'predecessor_plan_index'}
    if (type(plan_offset) is not int or plan_offset < 0
            or not _public_turn_continuation_enabled(plan, expected=True)
            or type(source) is not int or type(predecessor) is not int
            or not plan_offset <= source <= predecessor == plan_offset + index - 1
            or not _qualified_public_turn_source(plans[source - plan_offset])
            or not isinstance(proof, dict) or set(proof) != keys or proof.get('committed') is not True
            or not isinstance(node, dict) or set(node) != {'request', 'options', 'maximizing_actions'}
            or not isinstance(node.get('request'), dict)
            or not isinstance(proof.get('before'), dict) or not isinstance(proof.get('expected_state'), dict)
            or not isinstance(proof.get('action'), dict) or not isinstance(plan.get('action'), dict)
            or plan.get('public_turn_reanalysis_version') != PUBLIC_TURN_VERSION
            or verified and plan.get('public_turn_policy_continuation_verified') is not True
            or plan.get('complete_root_coverage') is not False or plan.get('optimality_proven') is True
            or plan.get('complete_validation_coverage') is True
            or plan.get('stop_reason') != 'matched_committed_public_turn_policy'
            or any(type(plan.get(field)) is not int or plan[field] != 0
                   for field in ('requested_simulations', 'completed_simulations'))
            or plan.get('candidate_statistics') != []
            or any(plan.get(field, 0) != 0 for field in (
                'training_started_simulations', 'training_completed_simulations',
                'validation_started_simulations', 'validation_completed_simulations'))
            or plan.get('root_selection_scope') is not None
            or proof['source_plan_index'] != source or proof['predecessor_plan_index'] != predecessor):
        return False
    origin, previous = plans[source - plan_offset], plans[predecessor - plan_offset]
    if (not _source_compute_mode_matches(origin, plan)
            or _json_value(plan.get('source_validation')) != _json_value(_public_turn_source_receipt(origin))
            or _json_value(plan.get('source_action')) != _json_value(_public_turn_source_receipt(origin)['source_root_action'])
            or plan.get('turn') != origin.get('turn')
            or any(row.get('accepted') is not True or row.get('turn') != origin.get('turn')
                   or row.get('scope') not in ('PUBLIC_TURN_POLICY_CONTINUATION',
                       'FORCED_LEGAL_ACTION', 'PROVEN_RESOURCE_DOMINANCE')
                   or row.get('scope') == 'PUBLIC_TURN_POLICY_CONTINUATION' and row.get('source_plan_index') != source
                   for row in plans[source - plan_offset + 1:index])
            or _action_identity(proof['action']) != _action_identity(previous.get('action', {}))
            or proof['before'].get('turn') != plan.get('turn')
            or proof['expected_state'].get('turn') != plan.get('turn')
            or type(proof['expected_state'].get('turns_remaining')) is not int
            or proof['expected_state']['turns_remaining'] <= 2
            or _json_value(node['request'].get('state')) != _json_value(proof['expected_state'])
            or _json_value(node['request'].get('drink_slots')) != _json_value(proof['expected_drink_slots'])):
        return False
    slots = proof['before_drink_slots']
    if not isinstance(slots, (list, tuple)):
        return False
    slots = _json_value(slots)
    if proof['action'].get('kind') == 'USE_DRINK':
        consumed = [proof['action'].get('source_slot'), proof['action'].get('drink_id')]
        if slots.count(consumed) != 1:
            return False
        slots.remove(consumed)
    if slots != _json_value(proof['expected_drink_slots']):
        return False
    options, maximum = node['options'], node['maximizing_actions']
    if (not isinstance(options, (list, tuple)) or not options
            or not isinstance(maximum, (list, tuple)) or not maximum
            or any(not isinstance(row, dict) or not isinstance(row.get('action_key'), (tuple, list))
                   or len(row['action_key']) != 5 or type(row.get('mean_final_score')) not in (int, float)
                   or not math.isfinite(row['mean_final_score']) for row in options)):
        return False
    action_keys = [tuple(row['action_key']) for row in options]
    if len(set(action_keys)) != len(action_keys):
        return False
    best = max(row['mean_final_score'] for row in options)
    expected = [row['action_key'] for row in options if row['mean_final_score'] == best]
    return (_json_value(maximum) == _json_value(expected)
            and _action_identity(plan['action']) == tuple(expected[0]))


def _qualified_resource_source(plan):
    roots = plan.get('candidate_statistics')
    evidence = plan.get('resource_reanalysis')
    elapsed = plan.get('elapsed_ms')
    if (not _compute_sensitivity_evidence(plan) or not _paired_adaptive_validation_evidence(plan)
            or plan.get('scope') != 'SAMPLING_ESTIMATE' or plan.get('accepted') is not True
            or plan.get('complete_root_coverage') is not True
            or plan.get('complete_validation_coverage') is not True
            or plan.get('requested_simulations') != _plan_simulation_budget(plan) or plan.get('requested_validation_samples') != 4
            or type(plan.get('completed_simulations')) is not int
            or not 0 < plan['completed_simulations'] <= _plan_simulation_budget(plan)
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or not 0 <= elapsed <= _plan_window_ms(plan)
            or plan.get('stop_reason') not in ('completed', 'soft_budget_reached')
            or not isinstance(roots, (list, tuple)) or not roots
            or not isinstance(evidence, dict) or evidence.get('complete') is not True
            or not _resource_reanalysis_evidence(plan, expected=True)):
        return False
    cohorts = []
    for row in roots:
        if (any(type(row.get(field)) is not int or row[field] < 4 for field in (
                'training_paired_visits', 'validation_paired_visits'))
                or not isinstance(row.get('validation_samples'), (list, tuple))
                or len(row['validation_samples']) != row['validation_paired_visits']):
            return False
        seeds = [sample.get('planning_seed') for sample in row['validation_samples']]
        if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            return False
        cohorts.append(seeds)
    expected_scope = ('training_disjoint_reused_resource_policy_validation'
        if plan.get('validation_cohort_evidence', {}).get('supplied_seeds') else
        'independent_resource_policy_validation')
    retained_completed = 0
    if _retained_policy_competition_active(plan):
        comparison = plan['retained_policy_comparison']
        if not _retained_policy_comparison_evidence(plan) or comparison['selected']:
            return False  # A retained winner cannot publish this window's new table.
        expected_scope = 'paired_retained_policy_competition'
        retained_completed = comparison['completed_simulations']
    return (all(seeds == cohorts[0] for seeds in cohorts)
        and plan.get('root_selection_scope') == expected_scope
        and plan['completed_simulations'] == sum(plan.get(name, 0) for name in (
            'training_completed_simulations', 'validation_completed_simulations',
            'inner_completed_simulations')) + evidence['completed_simulations'] + retained_completed
            + _paired_adaptive_completed(plan))


def _resource_source_receipt(origin):
    receipt = _public_turn_source_receipt(origin)
    receipt.pop('public_table_complete')
    receipt.pop('public_table_version')
    table = origin['resource_reanalysis']
    receipt.update(resource_table_complete=True,
        resource_table_version=RESOURCE_REANALYSIS_VERSION, resource_graph_count=table['graph_count'],
        resource_required_simulations=table['required_simulations'],
        resource_completed_simulations=table['completed_simulations'],
        resource_planning_seeds=tuple(table['planning_seeds']))
    return receipt


def _resource_normalized_public_key(raw, offset=0):
    from dataclasses import replace

    from agent.live_decision.high_score.kernel import PythonNiaProKernel
    from agent.live_decision.high_score.hif_mcts_cache import exact_state

    state = exact_state(_json_value(raw['state']))
    if state.persistent_effects_initialized:
        state = PythonNiaProKernel._prune_exhausted_schedules(state)
    return _resource_request_key(dict(raw, state=asdict(replace(state, score=state.score-offset)),
        card_slots=raw.get('card_slots', ())))


def _resource_continuation_evidence(plans, index, plan, *, verified=True, plan_offset=0):
    source, predecessor = plan.get('source_plan_index'), plan.get('predecessor_plan_index')
    proof, node = plan.get('continuation_proof'), plan.get('resource_node')
    keys = {'before', 'action', 'expected_state', 'before_drink_slots', 'expected_drink_slots',
            'committed', 'source_plan_index', 'predecessor_plan_index',
            'zero_random_consumption', 'proof_kind'}
    if (type(plan_offset) is not int or plan_offset < 0
            or not _resource_reanalysis_evidence(plan, expected=True)
            or type(source) is not int or type(predecessor) is not int
            or not plan_offset <= source <= predecessor == plan_offset+index-1
            or not 0 <= source-plan_offset < len(plans)
            or not _qualified_resource_source(plans[source-plan_offset])
            or not isinstance(proof, dict) or set(proof) != keys
            or proof.get('committed') is not True or proof.get('zero_random_consumption') is not True
            or proof.get('proof_kind') != 'STRICT_PUBLIC_ZERO_RANDOM_PROOF'
            or not isinstance(node, dict) or set(node) != {
                'request', 'options', 'maximizing_actions', 'graph_index', 'origin_indices', 'score_offset'}
            or not isinstance(node.get('request'), dict)
            or any(not isinstance(proof.get(name), dict) for name in ('before', 'action', 'expected_state'))
            or verified and plan.get('resource_policy_continuation_verified') is not True
            or plan.get('complete_root_coverage') is not False or plan.get('optimality_proven') is True
            or plan.get('complete_validation_coverage') is not False
            or plan.get('stop_reason') != 'matched_committed_local_resource_policy'
            or any(type(plan.get(field)) is not int or plan[field] != 0
                   for field in ('requested_simulations', 'completed_simulations'))
            or plan.get('candidate_statistics') != [] or plan.get('root_selection_scope') is not None
            or any(plan.get(field, 0) != 0 for field in (
                'training_started_simulations', 'training_completed_simulations',
                'validation_started_simulations', 'validation_completed_simulations',
                'inner_started_simulations', 'inner_completed_simulations'))
            or proof['source_plan_index'] != source or proof['predecessor_plan_index'] != predecessor):
        return False
    origin, previous = plans[source-plan_offset], plans[predecessor-plan_offset]
    receipt = _resource_source_receipt(origin)
    if (not _source_compute_mode_matches(origin, plan)
            or _json_value(plan.get('source_validation')) != _json_value(receipt)
            or _json_value(plan.get('source_action')) != _json_value(receipt['source_root_action'])
            or any(row.get('accepted') is not True or row.get('turn') != origin.get('turn')
                   or row.get('resource_policy_link_created') is not True
                   for row in plans[source-plan_offset:index])
            or _action_identity(proof['action']) != _action_identity(previous.get('action', {}))
            or proof['before'].get('turn') != plan.get('turn') or plan.get('turn') != origin.get('turn')
            or proof['expected_state'].get('turn') != plan.get('turn')
            or type(proof['expected_state'].get('turns_remaining')) is not int
            or proof['expected_state']['turns_remaining'] <= 2
            or _json_value(node['request'].get('state')) != _json_value(proof['expected_state'])
            or _json_value(node['request'].get('drink_slots')) != _json_value(proof['expected_drink_slots'])):
        return False
    slots = _json_value(proof['before_drink_slots'])
    if not isinstance(slots, list):
        return False
    if proof['action'].get('kind') == 'USE_DRINK':
        consumed = [proof['action'].get('source_slot'), proof['action'].get('drink_id')]
        if slots.count(consumed) != 1:
            return False
        slots.remove(consumed)
    if slots != _json_value(proof['expected_drink_slots']):
        return False
    graphs = origin['resource_reanalysis']['graphs']
    graph_index, offset = node['graph_index'], node['score_offset']
    if (type(graph_index) is not int or not 0 <= graph_index < len(graphs)
            or type(offset) not in (int, float) or not math.isfinite(offset)
            or _json_value(node['origin_indices']) != _json_value(graphs[graph_index]['origin_indices'])):
        return False
    try:
        current_key = _resource_normalized_public_key(node['request'], offset)
        matched = [row for row in graphs[graph_index]['public_nodes']
                   if _resource_normalized_public_key(row['request']) == current_key]
    except (KeyError, TypeError, ValueError):
        return False
    return (len(matched) == 1
        and _json_value(node['options']) == _json_value(matched[0]['options'])
        and _json_value(node['maximizing_actions']) == _json_value(matched[0]['maximizing_actions'])
        and _action_identity(plan.get('action', {})) == tuple(matched[0]['maximizing_actions'][0]))


def _frozen_policy_lineage(plans, index, plan):
    """Every no-new-search step must still name the same fully valued policy."""
    from agent.live_decision.high_score.hif_frozen_policy_continuation import qualified_source

    source = plan.get('source_plan_index')
    predecessor = plan.get('predecessor_plan_index')
    if (type(source) is not int or type(predecessor) is not int
            or not 0 <= source <= predecessor == index - 1):
        return False
    origin, previous = plans[source], plans[predecessor]
    if (not qualified_source(origin) or not _source_compute_mode_matches(origin, plan)
            or origin.get('requested_validation_samples') != 4
            or plan.get('turn') != origin.get('turn') or previous.get('accepted') is not True):
        return False
    if predecessor == source:
        return True
    return (previous.get('scope') in {'JOINT_ROOT_CONTINUATION', 'FROZEN_POLICY_CONTINUATION',
                                     'FORCED_LEGAL_ACTION', 'PROVEN_RESOURCE_DOMINANCE'}
            and previous.get('source_plan_index') == source)


def _verify_frozen_policy_continuation(kernel, request, legal, action, plan, plans, index,
                                     committed, *, deadline):
    """Recompute the physical predecessor and compare complete business state."""
    from agent.live_decision.high_score.hif_frozen_policy_continuation import deterministic_public_successor

    proof = plan.get('continuation_proof')
    keys = {'before', 'action', 'expected_state', 'before_drink_slots', 'expected_drink_slots',
            'committed', 'source_plan_index', 'predecessor_plan_index'}
    if (not _frozen_policy_lineage(plans, index, plan) or committed is None
            or not isinstance(proof, dict) or set(proof) != keys or proof['committed'] is not True
            or not isinstance(proof['action'], dict)
            or proof['source_plan_index'] != plan['source_plan_index']
            or proof['predecessor_plan_index'] != plan['predecessor_plan_index']):
        return False
    before_request, previous_action, after, previous_index = committed
    slots = tuple(before_request.drink_slots)
    if previous_action.kind.value == 'USE_DRINK':
        slots = tuple(pair for pair in slots if pair != (previous_action.source_slot, previous_action.drink_id))
    if (previous_index != index - 1 or after != request.state
            or _json_value(proof['before']) != _json_value(asdict(before_request.state))
            or _json_value(proof['expected_state']) != _json_value(asdict(request.state))
            or _action_identity(proof['action']) != _action_identity(asdict(previous_action))
            or _json_value(proof['before_drink_slots']) != _json_value(before_request.drink_slots)
            or _json_value(proof['expected_drink_slots']) != _json_value(slots)
            or tuple(request.drink_slots) != slots
            or request.ruleset_id != before_request.ruleset_id
            or request.plan_type != before_request.plan_type
            or request.state.turns_remaining <= 2):
        return False
    complete = kernel.legal_actions(request.state, plan_type=request.plan_type,
                                    decision_kind=request.decision_kind)
    expected = {_action_identity(asdict(value))[:4] for value in complete}
    supplied = [_action_identity(asdict(value))[:4] for value in legal]
    if (len(supplied) != len(set(supplied)) or set(supplied) != expected or action not in legal):
        return False
    proven, _ = deterministic_public_successor(kernel, before_request.state, previous_action,
        plan_type=request.plan_type, deadline=deadline)
    return proven is not None and proven == request.state


def _verify_exact_two_turn_root(kernel, request, legal, action, plan):
    """Check exact evidence against this request, including merged physical copies."""
    from agent.live_decision.high_score.planner import RobustHighScorePlanner

    complete = kernel.legal_actions(request.state, plan_type=request.plan_type,
                                   decision_kind=request.decision_kind)
    def identity(value):
        return _action_identity(value)[:4]
    expected = {identity(asdict(value)) for value in
                RobustHighScorePlanner._merge_equivalent_actions(request.state, complete)}
    if (request.state.turns_remaining != 2 or action not in legal
            or {identity(asdict(value)) for value in legal} != {identity(asdict(value)) for value in complete}
            or plan.get('scope') != 'FINAL_SCORE' or plan.get('exact_root_value_scope') != 'FINAL_SCORE'):
        return False
    rows = plan.get('exact_root_candidates')
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    found, eligible = set(), []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('action'), dict):
            return False
        key, value, mass = identity(row['action']), row.get('value'), row.get('probability_mass')
        if key in found or key not in expected or not isinstance(value, dict):
            return False
        found.add(key)
        if (type(mass) not in (int, float) or not math.isfinite(mass) or abs(mass - 1.) > 1e-9
                or any(type(value.get(field)) not in (int, float) or not math.isfinite(value[field])
                       for field in ('expected_final_score', 'p10_final_score', 'worst_final_score', 'best_final_score'))
                or type(value.get('outcome_count')) is not int or value['outcome_count'] < 1):
            return False
        if not row.get('rejection_reasons'):
            eligible.append((key, value))
    selected = [value for key, value in eligible if key == identity(asdict(action))]
    if len(selected) != 1 or selected[0] != plan.get('exact_root_value'):
        return False
    # This evaluator's fixed trials use expected-score selection. Probability
    # integration itself remains the complete solver's job, not a second solver.
    return selected[0]['expected_final_score'] >= max(value['expected_final_score'] for _, value in eligible)


def _verify_joint_roots(kernel, request, legal, plan):
    """Independently check complete physical roots and each proposed continuation."""
    from dataclasses import replace

    from agent.live_decision.high_score.contracts import ActionKind, DecisionKind

    rows = plan.get('candidate_statistics', [])
    groups = {}
    for row in rows:
        groups.setdefault(_action_identity(row['action']), []).append(row)
    if (set(groups) != {_action_identity(asdict(action)) for action in legal}
            or plan.get('physical_root_count') != len(legal)
            or plan.get('root_option_count') != len(rows)):
        return False
    for action in legal:
        group = groups[_action_identity(asdict(action))]
        if all(not row.get('continuation') for row in group):
            if len(group) != 1:
                return False
            continue
        branches = kernel.transition(request.state, replace(action, source_slot=None), plan_type=request.plan_type)
        if (len(branches) != 1 or branches[0].probability != 1.
                or any(event.event_type.startswith('DRAW') for event in branches[0].events)):
            return False
        plain = [row for row in group if not row.get('continuation')]
        if plain:
            after = branches[0].state
            continued = [row for row in group if row.get('continuation')]
            if (plan.get('stop_after_card_roots') is not True or action.kind is not ActionKind.PLAY_CARD
                    or len(plain) != 1 or len(continued) != 1 or len(group) != 2
                    or len(continued[0]['continuation']) != 1
                    or continued[0]['continuation'][0].get('kind') != 'END_TURN'
                    or after.terminal or after.turn != request.state.turn or after.actions_remaining < 1
                    or after.pending_choice is not None or after.resolution_stack
                    or any(event.event_type.startswith(('DRAW', 'MOVE_RANDOM', 'REMOVE_RANDOM',
                        'FREE_PLAY_RANDOM', 'EXCHANGE_HAND')) for event in branches[0].events)):
                return False
            next_legal = kernel.legal_actions(after, plan_type=request.plan_type,
                                              decision_kind=DecisionKind.LIVE)
            if sum(_action_identity(asdict(option)) == _action_identity(continued[0]['continuation'][0])
                   for option in next_legal) != 1:
                return False
            continue
        pending = branches[0].state.pending_choice
        if pending is None or pending.selections_remaining != 1 or pending.selection_optional:
            return False
        targets = kernel.legal_actions(branches[0].state, plan_type=request.plan_type,
                                       decision_kind=DecisionKind.SECONDARY)
        if not targets or any(target.kind is not ActionKind.MOVE_CARD for target in targets):
            return False
        if any(len(row.get('continuation', [])) != 1 for row in group):
            return False
        actual = [_action_identity(row['continuation'][0]) for row in group]
        if len(set(actual)) != len(actual) or set(actual) != {_action_identity(asdict(t)) for t in targets}:
            return False
    return True


def _public_turn_root_coverage(plan, roots, options):
    """Only proven optional END pairs share one physical public-graph edge."""
    if not isinstance(options, (list, tuple)):
        return False
    # Local resource graphs pass sample-only placeholders, not the outer roots.
    # Their physical-node coverage is checked separately by their caller.
    if (plan.get('public_turn_optional_end_roots_version') != PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION
            or plan.get('root_options_enabled') is not True
            or all(isinstance(row, dict) and 'action' not in row for row in roots)):
        return len(options) == len(roots)
    if (plan.get('stop_after_card_roots') is not True
            or plan.get('joint_root_options_verified') is not True
            or plan.get('root_option_count') != len(roots)):
        return False
    groups = {}
    try:
        for row in roots:
            if not isinstance(row, dict) or not isinstance(row.get('action'), dict):
                return False
            continuation = row.get('continuation', [])
            if not isinstance(continuation, (list, tuple)):
                return False
            groups.setdefault(_action_identity(row['action']), []).append(continuation)
        if plan.get('physical_root_count') != len(groups):
            return False
        continued = 0
        for first, variants in groups.items():
            if sum(not variant for variant in variants) != 1 or len(variants) > 2:
                return False
            for variant in variants:
                if not variant:
                    continue
                if (first[0] != 'PLAY_CARD' or len(variant) != 1 or not isinstance(variant[0], dict)
                        or _action_identity(variant[0]) != ('END_TURN', None, None, None, None)
                        or variant[0].get('source_slot') is not None):
                    return False
                continued += 1
        keys = []
        for option in options:
            key = option.get('action_key') if isinstance(option, dict) else None
            if not isinstance(key, (list, tuple)) or len(key) != 5:
                return False
            keys.append(tuple(key))
        return continued > 0 and len(keys) == len(set(keys)) == len(groups) and set(keys) == set(groups)
    except TypeError:
        return False


def _public_turn_reanalysis_evidence(plan, roots):
    """A complete training-only table never substitutes for outer validation."""
    evidence = plan.get('public_turn_reanalysis')
    if not _public_turn_score_translation_evidence(plan):
        return False
    if 'public_turn_reanalysis_enabled' not in plan:
        return evidence is None
    if (plan['public_turn_reanalysis_enabled'] is not True or not isinstance(evidence, dict)
            or evidence.get('version') != PUBLIC_TURN_VERSION
            or evidence.get('enabled') is not True
            or type(evidence.get('eligible')) is not bool or type(evidence.get('complete')) is not bool
            or evidence.get('requested_samples') != 4
            or evidence.get('value_scope') != 'PAIRED_TRAINING_BOUNDARY_MEAN_THEN_PUBLIC_MAX'
            or evidence.get('seedwise_maximization') is not False
            or evidence.get('original_tree_values_used') is not False
            or plan.get('installed_score_turn_chain_enabled') is True and evidence.get('complete') is True
                and ('score_translation_contract' not in evidence
                    or evidence['score_translation_contract'] != (SCORE_TRANSLATION_CONTRACT
                        if plan.get('public_turn_score_translation_enabled') is True else None))
            or plan.get('paired_finalist_validation', False)
            or plan.get('continuation_samples', 0) != 0):
        return False
    names = ('node_count', 'edge_count', 'boundary_count', 'boundary_group_count',
             'required_simulations', 'started_simulations', 'completed_simulations')
    if any(type(evidence.get(name)) is not int or evidence[name] < 0 for name in names):
        return False
    if evidence.get('status') == 'declined':
        return (evidence['eligible'] is False and evidence['complete'] is False
                and evidence['started_simulations'] == evidence['completed_simulations'] == 0
                and not evidence.get('root_options') and not evidence.get('boundary_groups')
                and isinstance(evidence.get('reason'), str) and bool(evidence['reason']))
    seeds = evidence.get('planning_seeds')
    groups = evidence.get('boundary_groups')
    options = evidence.get('root_options')
    if (evidence.get('status') != 'complete' or evidence.get('reason') != 'complete'
            or not evidence['eligible'] or not evidence['complete']
            or evidence['required_simulations'] != 4 * evidence['boundary_group_count']
            or evidence['started_simulations'] != evidence['required_simulations']
            or evidence['completed_simulations'] != evidence['required_simulations']
            or not isinstance(seeds, (list, tuple)) or len(seeds) != 4
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 4
            or not isinstance(groups, (list, tuple)) or len(groups) != evidence['boundary_group_count']
            or not groups or not _public_turn_root_coverage(plan, roots, options)):
        return False
    if plan.get('deterministic_validation_cohort_enabled') is True:
        cohort = plan.get('validation_cohort_evidence')
        if (not isinstance(cohort, dict) or not isinstance(cohort.get('training_seeds'), (tuple, list))
                or tuple(seeds) != tuple(cohort['training_seeds'])):
            return False
    boundary_indices = []
    for group in groups:
        scores = group.get('final_scores') if isinstance(group, dict) else None
        indices = group.get('boundary_indices') if isinstance(group, dict) else None
        if (not isinstance(group, dict) or group.get('planning_seeds') != seeds
                or not isinstance(scores, (list, tuple)) or len(scores) != 4
                or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)
                or not isinstance(indices, (list, tuple)) or not indices
                or any(type(index) is not int for index in indices)
                or group.get('representative_boundary') != indices[0]):
            return False
        boundary_indices.extend(indices)
    if sorted(boundary_indices) != list(range(evidence['boundary_count'])):
        return False
    for option in options:
        if (not isinstance(option, dict) or not isinstance(option.get('action_key'), (list, tuple))
                or type(option.get('mean_final_score')) not in (int, float)
                or not math.isfinite(option['mean_final_score'])):
            return False
    # Public table data use the training cohort, not the independent outer seeds.
    outer_seeds = {sample.get('planning_seed') for root in roots
                   for sample in root.get('validation_samples', ()) if isinstance(sample, dict)}
    return not set(seeds).intersection(outer_seeds)


def _unknown_node_timing_evidence(plan, *, expected=None):
    """A default-off report stays unchanged; enabled sampling must carry its proof."""
    if expected is not None and type(expected) is not bool:
        return False
    present = 'unknown_node_timing_enabled' in plan or 'unknown_node_timing' in plan
    if not present:
        return expected is not True
    evidence = plan.get('unknown_node_timing')
    if (expected is False or plan.get('unknown_node_timing_enabled') is not True
            or not isinstance(evidence, dict)
            or evidence.get('version') != UNKNOWN_TIMING_VERSION
            or evidence.get('timing_version') != TIMING_VERSION
            or plan.get('requested_validation_samples') != 4):
        return False
    counts = tuple(evidence.get(name) for name in
        ('calls', 'chosen_calls', 'scope_declines', 'ambiguous_pair_declines'))
    return (all(type(value) is int and value >= 0 for value in counts)
            and counts[0] == sum(counts[1:]))


def _retrieval_training_order_evidence(plan, *, expected=None):
    """Training-order evidence cannot substitute for independent root samples."""
    if expected is not None and type(expected) is not bool:
        return False
    present = 'retrieval_training_order_enabled' in plan or 'retrieval_training_order' in plan
    if not present:
        return expected is not True
    evidence = plan.get('retrieval_training_order')
    if (expected is False or plan.get('retrieval_training_order_enabled') is not True
            or not isinstance(evidence, dict)
            or evidence.get('version') != RetrievalTrainingOrder.version
            or evidence.get('unknown_version') != UNKNOWN_TIMING_VERSION
            or evidence.get('timing_version') != TIMING_VERSION
            or plan.get('unknown_node_timing_enabled') is not True
            or plan.get('scenario_reuse') is not True or plan.get('adaptive_validation') is not True
            or plan.get('requested_validation_samples') != 4):
        return False
    counts = tuple(evidence.get(name) for name in (
        'calls', 'eligible_calls', 'reordered_calls', 'unchanged_calls', 'scope_declines'))
    return (all(type(value) is int and value >= 0 for value in counts)
            and counts[0] == counts[1] + counts[4]
            and counts[1] == counts[2] + counts[3])


def _partial_node_timing_evidence(plan, *, expected=None):
    """A mixed-visit suggestion still requires a complete paired root evaluation."""
    if expected is not None and type(expected) is not bool:
        return False
    present = 'partial_node_timing_enabled' in plan or 'partial_node_timing' in plan
    if not present:
        return expected is not True
    evidence = plan.get('partial_node_timing')
    if (expected is False or plan.get('partial_node_timing_enabled') is not True
            or not isinstance(evidence, dict)
            or evidence.get('version') != PARTIAL_NODE_TIMING_VERSION
            or plan.get('unknown_node_timing_enabled') is not True
            or plan.get('deterministic_validation_cohort_enabled') is not True
            or plan.get('requested_validation_samples') != 4):
        return False
    counts = tuple(evidence.get(name) for name in ('calls', 'chosen', 'changed', 'declined'))
    return (all(type(value) is int and value >= 0 for value in counts)
            and counts[0] == counts[1] + counts[3] and counts[2] <= counts[1])


def _installed_score_turn_chain_evidence(plan, *, expected=None, expected_multi_action=None,
                                       expected_focus_projection=None):
    """A bounded current-turn suggestion does not add samples or a leaf bonus."""
    if expected is not None and type(expected) is not bool:
        return False
    if expected_multi_action is not None and type(expected_multi_action) is not bool:
        return False
    if expected_focus_projection is not None and type(expected_focus_projection) is not bool:
        return False
    multi_action = plan.get('installed_score_multi_action_enabled', False)
    focus_projection = plan.get('installed_score_focus_projection_enabled', False)
    if (type(multi_action) is not bool
            or expected_multi_action is not None and multi_action is not expected_multi_action
            or type(focus_projection) is not bool
            or expected_focus_projection is not None and focus_projection is not expected_focus_projection):
        return False
    present = ('installed_score_turn_chain_enabled' in plan or 'installed_score_turn_chain' in plan
               or 'installed_score_multi_action_enabled' in plan
               or 'installed_score_focus_projection_enabled' in plan)
    if not present:
        return expected is not True
    evidence = plan.get('installed_score_turn_chain')
    version = installed_score_helper_version(multi_action=multi_action, focus_projection=focus_projection)
    if (expected is False or plan.get('installed_score_turn_chain_enabled') is not True
            or not isinstance(evidence, dict)
            or evidence.get('version') != version
            or plan.get('partial_node_timing_enabled') is not True
            or plan.get('unknown_node_timing_enabled') is not True
            or plan.get('requested_validation_samples') != 4
            or plan.get('installed_card_use_value', False) is not False):
        return False
    counts = tuple(evidence.get(name) for name in
                   ('calls', 'scope_hits', 'scope_declines', 'chain_accepted', 'chain_declined'))
    if (not all(type(value) is int and value >= 0 for value in counts)
            or counts[0] != counts[1] + counts[2] or counts[1] != counts[3] + counts[4]):
        return False
    chain = evidence.get('turn_chain')
    if not isinstance(chain, dict):
        return False
    projection_fields = ('end_turn_focus_projection', 'end_turn_focus_projection_version',
                         'end_turn_focus_projection_statuses', 'end_turn_focus_projection_reasons')
    if focus_projection:
        statuses = chain.get('end_turn_focus_projection_statuses')
        reasons = chain.get('end_turn_focus_projection_reasons')
        if (chain.get('end_turn_focus_projection') is not True
                or chain.get('end_turn_focus_projection_version') != END_TURN_FOCUS_VERSION
                or any(not isinstance(counts, dict) or any(
                    not isinstance(key, str) or not key or type(value) is not int or value < 0
                    for key, value in counts.items()) for counts in (statuses, reasons))
                or sum(statuses.values()) != sum(reasons.values())):
            return False
    elif any(field in chain for field in projection_fields):
        return False
    names = ('calls', 'cache_hits', 'accepted', 'declined', 'expanded_nodes',
             'transition_computations', 'transition_cache_hits', 'boundary_computations',
             'boundary_cache_hits', 'concentration_percentage_computations',
             'concentration_percentage_cache_hits', 'effective_concentration_computations',
             'effective_concentration_cache_hits', 'forecast_computations', 'forecast_cache_hits')
    inner = tuple(chain.get(name) for name in names)
    limits = (('future_samples', 0), ('max_expanded_nodes', 32), ('max_decision_depth', 32))
    reasons = chain.get('decline_reasons')
    return (all(type(value) is int and value >= 0 for value in inner)
            and all(type(chain.get(name)) is int and chain[name] == value for name, value in limits)
            and inner[0] == counts[1] and inner[2:4] == counts[3:5]
            and inner[1] <= inner[0] and inner[4] <= 32 * (inner[0] - inner[1])
            and chain.get('value_scope') == 'TURN_CHAIN_ESTIMATE'
            and chain.get('installed_card_use_value', False) is False
            and 'installed_card_use_opportunities' not in chain
            and isinstance(reasons, dict)
            and all(name in {'turn_chain_draw_domain', 'turn_chain_node_limit', 'terminal_turn'}
                    and type(value) is int and value >= 0 for name, value in reasons.items())
            and sum(reasons.values()) <= inner[3])


def _extra_action_turn_chain_evidence(plan, *, expected=None, expected_visited_card=None,
                                    expected_visited_card_version=None):
    """A proved extra action admits a bounded suggestion, never new futures."""
    if expected is not None and type(expected) is not bool:
        return False
    visited_card = plan.get('visited_card_extra_action_enabled', False)
    if (type(visited_card) is not bool
            or expected_visited_card is not None and (
                type(expected_visited_card) is not bool or visited_card is not expected_visited_card)
            or expected_visited_card_version is not None and (
                not visited_card or expected_visited_card_version not in (
                    REFERENCE_VISITED_CARD_VERSION, VISITED_CARD_VERSION))):
        return False
    present = ('extra_action_turn_chain_enabled' in plan or 'extra_action_turn_chain' in plan
               or 'visited_card_extra_action_enabled' in plan)
    if not present:
        return expected is not True
    evidence = plan.get('extra_action_turn_chain')
    # Explicit policy declarations bind one supported runtime version. Only
    # unspecified historical reads retain the wider v1/v2/v3 inspection path.
    versions = ((expected_visited_card_version,) if expected_visited_card_version is not None else
        (VISITED_CARD_VERSION, 'extra-action-last-action-chain-visited-card-v2', REFERENCE_VISITED_CARD_VERSION)
        if visited_card and expected is None else (VISITED_CARD_VERSION,) if visited_card else (EXTRA_ACTION_VERSION,))
    unused_action = (visited_card and isinstance(evidence, dict)
        and evidence.get('version') in (VISITED_CARD_VERSION, 'extra-action-last-action-chain-visited-card-v2'))
    names = ('calls', 'scope_hits', 'scope_declines', 'chain_accepted', 'chain_declined')
    if (expected is False or plan.get('extra_action_turn_chain_enabled') is not True
            or not isinstance(evidence, dict)
            or set(evidence) != {*names, 'version', 'turn_chain', 'drink_turn_chain', 'strict_proof',
                                 'old_action_proof', 'scope_decline_reasons',
                                 *({'unused_action_drink'} if unused_action else set())}
            or evidence.get('version') not in versions
            or any(plan.get(name) is not True for name in (
                'unknown_node_timing_enabled', 'partial_node_timing_enabled',
                'installed_score_turn_chain_enabled', 'public_validation_phase_budget_enabled'))
            or plan.get('requested_validation_samples') != 4
            or plan.get('installed_card_use_value', False) is not False):
        return False
    if unused_action:
        guard = evidence['unused_action_drink']
        counters = ('calls', 'cache_hits', 'proven', 'declined', 'cache_entries', 'paired_nodes', 'paired_edges')
        if (not isinstance(guard, dict) or set(guard) != {'version', *counters}
                or guard.get('version') != 'unused-action-turn-proof-v1'
                or any(type(guard[name]) is not int or guard[name] < 0 for name in counters)
                or guard['calls'] != guard['proven'] + guard['declined']
                or guard['cache_hits'] > guard['calls']
                or guard['cache_entries'] > guard['calls'] - guard['cache_hits']):
            return False
    calls, hits, declines, accepted, rejected = counts = tuple(evidence[name] for name in names)
    if (not all(type(value) is int and value >= 0 for value in counts)
            or calls != hits + declines or hits != accepted + rejected):
        return False
    reasons = evidence['scope_decline_reasons']
    if (not isinstance(reasons, dict)
            or not all(name in {'outside_scope', 'no_positive_direct_extra_action',
                                'no_strict_same_turn_extra_action', 'old_suggestion_unknown',
                                'old_action_random_or_draw', 'old_action_not_deterministic',
                                'old_action_unobserved_event', 'old_action_unresolved_or_changed_boundary',
                                'old_action_proven_keeps_ap', 'old_action_ap_not_proven'}
                and type(value) is int and value >= 0 for name, value in reasons.items())
            or sum(reasons.values()) != declines):
        return False
    old_proof = evidence['old_action_proof']
    if (not isinstance(old_proof, dict)
            or set(old_proof) != {'started', 'completed', 'explicit_end'}
            or not all(type(value) is int and value >= 0 for value in old_proof.values())
            or old_proof['completed'] > old_proof['started']
            or old_proof['started'] + old_proof['explicit_end'] > calls
            or hits > old_proof['completed'] + old_proof['explicit_end']):
        return False
    proof = evidence['strict_proof']
    if (not isinstance(proof, dict)
            or not {'version', 'cache_entries'} <= set(proof)
            or set(proof) - {'version', 'cache_entries', 'transitions_started', 'transitions_completed'}
            or proof.get('version') != PAIR_ORDER_VERSION
            or type(proof.get('cache_entries')) is not int or proof['cache_entries'] != 0):
        return False
    started, completed = (proof.get(name, 0) for name in ('transitions_started', 'transitions_completed'))
    if (not all(type(value) is int and value >= 0 for value in (started, completed))
            or not hits <= completed <= started):
        return False
    names = ('calls', 'cache_hits', 'accepted', 'declined', 'expanded_nodes',
             'transition_computations', 'transition_cache_hits', 'boundary_computations',
             'boundary_cache_hits', 'concentration_percentage_computations',
             'concentration_percentage_cache_hits', 'effective_concentration_computations',
             'effective_concentration_cache_hits', 'forecast_computations', 'forecast_cache_hits')
    totals = [0, 0, 0]
    for field, maximum in (('turn_chain', 32), ('drink_turn_chain', 512)):
        chain = evidence[field]
        finite = field == 'drink_turn_chain'
        extra_fields = {'finite_consumables', 'finite_consumables_version', 'cache_entries'} if finite else set()
        limits = (('future_samples', 0), ('max_expanded_nodes', maximum), ('max_decision_depth', 32))
        if (not isinstance(chain, dict)
                or set(chain) != {*names, *(name for name, _ in limits), 'value_scope', 'decline_reasons', *extra_fields}):
            return False
        inner = tuple(chain[name] for name in names)
        reasons = chain['decline_reasons']
        allowed = {'turn_chain_draw_domain', 'turn_chain_node_limit', 'terminal_turn'}
        if finite:
            allowed.update({'finite_turn_unresolved', 'finite_turn_action_domain', 'finite_turn_new_consumable',
                            'finite_turn_boundary_domain', 'finite_turn_consumption_not_strict',
                            'finite_turn_unbounded_end'})
            if (chain['finite_consumables'] is not True
                    or chain['finite_consumables_version'] != FINITE_CONSUMABLES_VERSION
                    or type(chain['cache_entries']) is not int or chain['cache_entries'] < 0):
                return False
        if (not all(type(value) is int and value >= 0 for value in inner)
                or not all(type(chain[name]) is int and chain[name] == value for name, value in limits)
                or inner[0] != inner[2] + inner[3] or inner[1] > inner[0]
                or inner[4] > maximum * (inner[0] - inner[1])
                or finite and chain['cache_entries'] != inner[0] - inner[1]
                or chain['value_scope'] != 'TURN_CHAIN_ESTIMATE'
                or not isinstance(reasons, dict)
                or not all(name in allowed and type(value) is int and value >= 0 for name, value in reasons.items())
                or sum(reasons.values()) > inner[3]):
            return False
        for index, value in enumerate((inner[0], inner[2], inner[3])):
            totals[index] += value
    return tuple(totals) == (hits, accepted, rejected)


def _pair_order_advice_evidence(plan, *, expected=None):
    """Audit completed local proofs; these counters never add sampled worlds."""
    if expected is not None and type(expected) is not bool:
        return False
    present = 'pair_order_advice_enabled' in plan or 'pair_order_advice' in plan
    if not present:
        return expected is not True
    evidence = plan.get('pair_order_advice')
    if (expected is False or plan.get('pair_order_advice_enabled') is not True
            or not isinstance(evidence, dict) or evidence.get('version') != PAIR_ORDER_VERSION
            or plan.get('public_turn_policy_continuation_enabled') is not True
            or plan.get('unknown_node_timing_enabled') is not True
            or plan.get('partial_node_timing_enabled') is not True
            or plan.get('requested_validation_samples') != 4):
        return False
    names = ('calls', 'scope_declines', 'unchanged_calls', 'improved_calls',
             'pair_proofs', 'cache_hits', 'transitions_started', 'transitions_completed')
    counts = tuple(evidence.get(name, 0) for name in names)
    cache_entries = evidence.get('cache_entries')
    reasons = {name: value for name, value in evidence.items() if name not in (*names, 'version', 'cache_entries')}
    if (not all(type(value) is int and value >= 0 for value in (*counts, cache_entries))
            or not all(isinstance(name, str) and name.startswith('pair_declines:')
                and bool(name.removeprefix('pair_declines:'))
                and type(value) is int and value >= 0 for name, value in reasons.items())):
        return False
    calls, scope, unchanged, improved, proofs, hits, started, completed = counts
    return (calls == scope + unchanged + improved
            and proofs + hits == improved + sum(reasons.values())
            and cache_entries == proofs and completed <= started <= 4 * proofs)


def _retained_policy_competition_active(plan):
    evidence = plan.get('retained_policy_comparison')
    return (plan.get('retained_policy_competition_enabled') is True
        and isinstance(evidence, dict) and evidence.get('eligible') is True)


def _internal_successor_execution_evidence(plans, index, plan, *, plan_offset, verified):
    """A fresh graph winner may execute only its recorded deterministic nodes."""
    proof = plan.get('internal_successor_graph_execution')
    fields = {'version', 'source_plan_index', 'public_request', 'card_slots',
              'legal_action_keys', 'selected_action'}
    if (not isinstance(proof, dict) or set(proof) != fields
            or proof['version'] != INTERNAL_SUCCESSOR_GRAPH_VERSION
            or not _internal_successor_graph_mode(plan, expected=True)):
        return False
    source = plan['retained_policy_execution_source_plan_index']
    origin = plans[source - plan_offset]
    if (not isinstance(origin, dict)
            or proof['source_plan_index'] != source or type(proof['source_plan_index']) is not int
            or origin.get('scope') != 'SAMPLING_ESTIMATE' or origin.get('accepted') is not True
            or origin.get('complete_root_coverage') is not True or origin.get('complete_validation_coverage') is not True
            or not _source_compute_mode_matches(origin, plan)
            or not _internal_reanalysis_evidence(origin, expected=True, expected_successor_graph=True)
            or not (origin['internal_reanalysis'].get('successor_graph') or {}).get('complete')
            or (origin.get('retained_policy_comparison') or {}).get('selected') is True
            or not _quality_reference_work_evidence(origin)):
        return False
    try:
        for position in range(source - plan_offset + 1, index + 1):
            row, previous = plans[position], plans[position-1]
            lineage = row['retained_policy_lineage']
            cohort = row['validation_cohort_link']
            prior_seeds = ((previous.get('validation_cohort_evidence') or {}).get('validation_seeds')
                if previous.get('scope') == 'SAMPLING_ESTIMATE'
                else (previous.get('validation_cohort_link') or {}).get('seeds'))
            if (row.get('accepted') is not True or row.get('turn') != origin.get('turn')
                    or row.get('scope') not in ('RETAINED_POLICY_CONTINUATION', 'FORCED_LEGAL_ACTION', 'JOINT_ROOT_CONTINUATION')
                    or lineage.get('source_plan_index') != source
                    or lineage.get('predecessor_plan_index') != plan_offset + position - 1
                    or lineage.get('reason') != 'matched_deterministic_winner'
                    or verified and row.get('retained_policy_lineage_verified') is not True
                    or position < index and (row.get('retained_policy_link_created') is not True
                        or row.get('validation_cohort_link_created') is not True)
                    or cohort.get('version') != COHORT_VERSION or cohort.get('reused') is not True
                    or cohort.get('source_plan_index') != plan_offset + position - 1
                    or cohort.get('reason') != 'reused_deterministic_committed_cohort'
                    or not isinstance(prior_seeds, (list, tuple)) or len(prior_seeds) < 4
                    or tuple(cohort.get('seeds', ())) != tuple(prior_seeds[:4])):
                return False
            if row.get('scope') != 'RETAINED_POLICY_CONTINUATION':
                continue
            current = row.get('internal_successor_graph_execution')
            if (not isinstance(current, dict) or set(current) != fields
                    or current['version'] != INTERNAL_SUCCESSOR_GRAPH_VERSION
                    or current['source_plan_index'] != source
                    or row.get('retained_policy_execution_source_plan_index') != source):
                return False
            matches = [node for node in origin['internal_reanalysis']['successor_public_nodes']
                if _json_value({name: node['request'][name] for name in ('state', 'drink_slots', 'plan_type', 'utility')})
                    == _json_value(current['public_request'])
                and _json_value(node['request']['card_slots']) == _json_value(current['card_slots'])]
            keys = [tuple(key) for key in current['legal_action_keys']]
            if (len(matches) != 1 or len(keys) != len(set(keys))
                    or set(keys) != {tuple(edge['action_key']) for edge in matches[0]['edges']}
                    or tuple(current['selected_action']) != tuple(matches[0]['maximizing_actions'][0])
                    or tuple(current['selected_action']) != _action_identity(row['action'])):
                return False
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _retained_policy_execution_evidence(plans, index, plan, *, expected=None, expected_tree=None,
                                        plan_offset=0, verified=True):
    """The execution source is the winning comparison, not the older tree origin."""
    enabled = plan.get('retained_policy_execution_enabled', False)
    tree_enabled = plan.get('retained_tree_policy_execution_enabled', False)
    tree_version = plan.get('retained_tree_policy_execution_version')
    scope = plan.get('scope')
    if (type(enabled) is not bool or expected is not None and enabled is not expected):
        return False
    if (type(tree_enabled) is not bool or expected_tree is not None and tree_enabled is not expected_tree
            or tree_enabled and (not enabled or tree_version != RETAINED_TREE_POLICY_EXECUTION_VERSION)
            or not tree_enabled and tree_version is not None):
        return False
    if not enabled:
        return (plan.get('retained_policy_execution_version') is None
            and plan.get('retained_policy_execution_source_plan_index') is None
            and 'retained_policy_execution_verified' not in plan
            and 'internal_successor_graph_execution' not in plan
            and scope != 'RETAINED_POLICY_CONTINUATION')
    if (plan.get('retained_policy_execution_version') != RETAINED_POLICY_EXECUTION_VERSION
            or plan.get('retained_policy_competition_enabled') is not True
            or plan.get('retained_policy_competition_version') != RETAINED_POLICY_COMPETITION_VERSION):
        return False
    if scope != 'RETAINED_POLICY_CONTINUATION':
        return ('retained_policy_execution_verified' not in plan
            and 'internal_successor_graph_execution' not in plan)
    source = plan.get('retained_policy_execution_source_plan_index')
    if (not isinstance(plans, list) or type(index) is not int or not 0 <= index < len(plans)
            or type(plan_offset) is not int or plan_offset < 0 or type(source) is not int
            or not plan_offset <= source < plan_offset + index
            or plan.get('accepted') is not True
            or plan.get('stop_reason') != 'matched_retained_policy_execution'
            or plan.get('requested_validation_samples') != 4
            or any(type(plan.get(name)) is not int or plan[name] != 0 for name in (
                'requested_simulations', 'completed_simulations', 'started_simulations',
                'training_started_simulations', 'training_completed_simulations',
                'validation_started_simulations', 'validation_completed_simulations'))
            or any(plan.get(name, 0) != 0 for name in ('inner_started_simulations', 'inner_completed_simulations'))
            or plan.get('candidate_statistics') != [] or plan.get('complete_root_coverage') is not False
            or plan.get('complete_validation_coverage') is not False or plan.get('optimality_proven') is True
            or plan.get('retained_policy_comparison') is not None or plan.get('root_selection_scope') is not None
            or plan.get('selected_continuation', []) not in ((), [])
            or plan.get('validation_paired_visits', []) not in ((), [])
            or verified and plan.get('retained_policy_execution_verified') is not True):
        return False
    cohort = plan.get('validation_cohort_evidence') or {}
    allocation = plan.get('validation_seed_allocation') or {}
    if (not isinstance(cohort, dict) or not isinstance(allocation, dict)
            or cohort.get('validation_seeds', []) not in ((), [])
            or any(allocation.get(name, 0) != 0 for name in ('generated', 'started', 'complete_rounds'))):
        return False
    if 'internal_successor_graph_execution' in plan:
        return _internal_successor_execution_evidence(plans, index, plan,
            plan_offset=plan_offset, verified=verified)
    origin = plans[source - plan_offset]
    lineage = plan.get('retained_policy_lineage')
    original_lineage = origin.get('retained_policy_lineage') if isinstance(origin, dict) else None
    if (not isinstance(origin, dict) or origin.get('scope') != 'SAMPLING_ESTIMATE'
            or not _source_compute_mode_matches(origin, plan)
            or origin.get('accepted') is not True
            or origin.get('complete_root_coverage') is not True
            or origin.get('complete_validation_coverage') is not True
            or origin.get('requested_simulations') != _plan_simulation_budget(origin)
            or origin.get('retained_policy_execution_enabled') is not True
            or origin.get('retained_policy_execution_version') != RETAINED_POLICY_EXECUTION_VERSION
            or origin.get('retained_tree_policy_execution_enabled', False) is not tree_enabled
            or origin.get('retained_tree_policy_execution_version') != tree_version
            or not _retained_policy_comparison_evidence(origin)
            or not _retained_policy_competition_active(origin)
            or origin['retained_policy_comparison']['selected'] is not True
            or not tree_enabled and not any(isinstance(origin.get(name), dict) and origin[name].get('complete') is True
                for name in ('public_turn_reanalysis', 'resource_reanalysis'))
            or any(origin.get(name) is not False for name in (
                'public_turn_policy_link_created', 'resource_policy_link_created'))
            or not isinstance(lineage, dict) or not isinstance(original_lineage, dict)
            or type(lineage.get('source_plan_index')) is not int
            or not plan_offset <= lineage['source_plan_index'] < source
            or lineage['source_plan_index'] != original_lineage.get('source_plan_index')):
        return False
    for position in range(source - plan_offset, index + 1):
        row = plans[position]
        current_lineage = row.get('retained_policy_lineage') if isinstance(row, dict) else None
        revealed_origin = (position == source - plan_offset
            and row.get('revealed_policy_competition_enabled') is True
            and (row.get('revealed_policy_competition') or {}).get('mode') == 'revealed_recompetition'
            and _revealed_policy_lineage_evidence(plans, position, row,
                plan_offset=plan_offset, verified=False)
            and (not verified or row.get('retained_policy_lineage_verified') is True
                and _json_value(row.get('retained_policy_commit_proof'))
                    == _json_value(row['revealed_policy_competition']['proof'])))
        if (not isinstance(row, dict) or row.get('accepted') is not True
                or row.get('retained_policy_execution_enabled') is not True
                or row.get('retained_policy_execution_version') != RETAINED_POLICY_EXECUTION_VERSION
                or row.get('retained_tree_policy_execution_enabled', False) is not tree_enabled
                or row.get('retained_tree_policy_execution_version') != tree_version
                or row.get('turn') != origin.get('turn')
                or not isinstance(current_lineage, dict)
                or current_lineage.get('source_plan_index') != lineage['source_plan_index']
                or type(current_lineage.get('predecessor_plan_index')) is not int
                or current_lineage['predecessor_plan_index'] != plan_offset + position - 1
                or current_lineage.get('reason') != 'matched_deterministic_winner' and not revealed_origin
                or position < index and (row.get('retained_policy_link_created') is not True
                    or row.get('validation_cohort_link_created') is not True)):
            return False
        if position == source - plan_offset:
            continue
        if (row.get('scope') not in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION',
                'RETAINED_POLICY_CONTINUATION')
                or row.get('scope') == 'RETAINED_POLICY_CONTINUATION'
                    and row.get('retained_policy_execution_source_plan_index') != source):
            return False
        link = row.get('validation_cohort_link')
        previous = plans[position - 1]
        prior_seeds = ((previous.get('validation_cohort_evidence') or {}).get('validation_seeds')
            if previous.get('scope') == 'SAMPLING_ESTIMATE'
            else (previous.get('validation_cohort_link') or {}).get('seeds'))
        if (not isinstance(link, dict) or link.get('version') != COHORT_VERSION
                or link.get('reused') is not True or link.get('source_plan_index') != plan_offset + position - 1
                or link.get('reason') != 'reused_deterministic_committed_cohort'
                or not isinstance(prior_seeds, (list, tuple)) or len(prior_seeds) < 4
                or not isinstance(link.get('seeds'), (list, tuple))
                or tuple(link['seeds']) != tuple(prior_seeds[:4])):
            return False
    return True


def _revealed_policy_mode_evidence(plan, *, expected=None):
    """The new comparison authority never follows from an old lineage flag."""
    enabled = plan.get('revealed_policy_competition_enabled', False)
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return not any(name in plan for name in ('revealed_policy_competition_version',
            'revealed_policy_competition', 'revealed_policy_link_created'))
    evidence = plan.get('revealed_policy_competition')
    return (plan.get('revealed_policy_competition_version') == REVEALED_POLICY_COMPETITION_VERSION
        and _quality_reference_evidence(plan, expected=True, window_seconds=300.)
        and _compute_sensitivity_evidence(plan, expected=False, simulations=256)
        and plan.get('retained_policy_competition_enabled') is True
        and type(plan.get('revealed_policy_link_created')) is bool
        and isinstance(evidence, dict) and set(evidence) == {
            'version', 'eligible', 'mode', 'reason', 'source_plan_index', 'predecessor_plan_index',
            'source_domain', 'source_tables', 'source_training_seeds', 'source_validation_seeds', 'proof'}
        and evidence['version'] == REVEALED_POLICY_COMPETITION_VERSION
        and evidence['source_domain'] == 'complete_frozen_policy'
        and type(evidence['eligible']) is bool
        and evidence['mode'] in ('none', 'deterministic_continuation', 'revealed_recompetition')
        and isinstance(evidence['reason'], str) and bool(evidence['reason'])
        and isinstance(evidence['source_tables'], dict)
        and set(evidence['source_tables']) == {'public_turn', 'resource', 'internal'}
        and all(type(value) is bool for value in evidence['source_tables'].values()))


def _revealed_policy_commit_evidence(proof, *, source, previous, plan, actions, index):
    """Check the recorded actual transition, without enumerating random futures."""
    from agent.live_decision.high_score.contracts import StrategyState

    if (not isinstance(proof, dict) or set(proof) != {'before', 'before_card_slots', 'action',
            'after', 'after_card_slots', 'source_plan_index', 'predecessor_plan_index', 'committed',
            'ruleset_id', 'protocol_version'}
            or type(proof['source_plan_index']) is not int or type(proof['predecessor_plan_index']) is not int
            or proof['source_plan_index'] != source or proof['predecessor_plan_index'] != previous
            or proof['committed'] is not True
            or not all(isinstance(proof[name], str) and proof[name] for name in ('ruleset_id', 'protocol_version'))):
        return False
    for name in ('before', 'after'):
        request = proof[name]
        slots = proof[name + '_card_slots']
        if (not isinstance(request, dict) or set(request) != {'state', 'drink_slots', 'plan_type', 'utility'}
                or not isinstance(request['state'], dict) or not isinstance(request['utility'], dict)
                or not isinstance(request['drink_slots'], (tuple, list))
                or not isinstance(slots, (tuple, list))):
            return False
        state = request['state']
        drinks = request['drink_slots']
        if (set(state) != set(StrategyState.__dataclass_fields__)
                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or type(pair[0]) is not int or pair[0] < 0 or not isinstance(pair[1], str) for pair in drinks)
                or len({pair[0] for pair in drinks}) != len(drinks)
                or sorted(pair[1] for pair in drinks) != sorted(state['drinks'])):
            return False
        deck = state.get('deck')
        if not isinstance(deck, dict) or not isinstance(deck.get('hand'), (tuple, list)):
            return False
        pending = state.get('pending_choice')
        ids = (pending.get('candidate_instance_ids') if isinstance(pending, dict)
               else [card.get('instance_id') for card in deck['hand'] if isinstance(card, dict)])
        if (not isinstance(ids, (tuple, list)) or len(ids) != len(slots)
                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not isinstance(pair[0], str) or type(pair[1]) is not int or pair[1] < 0 for pair in slots)
                or len({pair[0] for pair in slots}) != len(slots)
                or len({pair[1] for pair in slots}) != len(slots)
                or set(ids) != {pair[0] for pair in slots}):
            return False
    before, after = proof['before'], proof['after']
    if (before['plan_type'] != after['plan_type'] or before['utility'] != after['utility']
            or after['state'].get('turn') != plan.get('turn')
            or not isinstance(proof['action'], dict)):
        return False
    slots = _json_value(before['drink_slots'])
    action = proof['action']
    if action.get('kind') == 'USE_DRINK':
        consumed = [action.get('source_slot'), action.get('drink_id')]
        if slots.count(consumed) != 1:
            return False
        slots.remove(consumed)
    if slots != _json_value(after['drink_slots']):
        return False
    if actions is not None:
        if (not isinstance(actions, (list, tuple)) or not 0 < index < len(actions)
                or not isinstance(actions[index - 1], dict) or not isinstance(actions[index], dict)
                or _json_value(before['state']) != _json_value(actions[index - 1].get('before'))
                or _json_value(action) != _json_value(actions[index - 1].get('action'))
                or _json_value(after['state']) != _json_value(actions[index - 1].get('after'))
                or _json_value(after['state']) != _json_value(actions[index].get('before'))):
            return False
    return True


def _revealed_policy_lineage_evidence(plans, index, plan, *, plan_offset=0,
                                      verified=True, actions=None):
    if (not _revealed_policy_mode_evidence(plan, expected=True)
            or not isinstance(plans, list) or type(index) is not int or not 0 <= index < len(plans)
            or type(plan_offset) is not int or plan_offset < 0
            or not _retained_policy_comparison_evidence(plan)
            or type(plan.get('retained_policy_link_created')) is not bool
            or plan['retained_policy_link_created'] is not plan['revealed_policy_link_created']
            or verified and plan.get('retained_policy_lineage_verified') is not True):
        return False
    evidence = plan['revealed_policy_competition']
    source, previous = evidence['source_plan_index'], evidence['predecessor_plan_index']
    lineage = plan.get('retained_policy_lineage')
    if lineage != {name: evidence[name] for name in (
            'source_plan_index', 'predecessor_plan_index', 'reason', 'source_domain')}:
        return False

    def seeds_valid(values, upper):
        return (isinstance(values, (tuple, list))
            and all(type(value) is int and 0 <= value < upper for value in values)
            and len(set(values)) == len(values))

    training, old_outer = evidence['source_training_seeds'], evidence['source_validation_seeds']
    if not seeds_valid(training, 2**32) or not seeds_valid(old_outer, 2**31):
        return False

    def complete_source(row, position):
        roots = row.get('candidate_statistics', ())
        elapsed = row.get('elapsed_ms')
        return (_revealed_policy_mode_evidence(row, expected=True)
            and row.get('scope') == 'SAMPLING_ESTIMATE' and row.get('accepted') is True
            and row.get('complete_root_coverage') is True and row.get('complete_validation_coverage') is True
            and row.get('requested_simulations') == 256 and row.get('requested_validation_samples') == 4
            and type(row.get('completed_simulations')) is int and 0 < row['completed_simulations'] <= 256
            and type(row.get('started_simulations')) is int
            and row['started_simulations'] == row['completed_simulations']
            and type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= _plan_window_ms(row)
            and row.get('stop_reason') == 'completed'
            and (row.get('retained_policy_comparison') or {}).get('selected') is not True
            and isinstance(roots, (tuple, list)) and bool(roots)
            and all(isinstance(root, dict) and all(type(root.get(name)) is int and root[name] >= 4
                for name in ('training_paired_visits', 'validation_paired_visits')) for root in roots)
            and _retained_policy_comparison_evidence(row)
            and _validation_cohort_evidence(row, expected=True, plans=plans, index=position, plan_offset=plan_offset))

    active = evidence['mode'] != 'none'
    if not active:
        if (evidence['eligible'] or source is not None or previous is not None or evidence['proof'] is not None
                or training or old_outer or any(evidence['source_tables'].values())
                or evidence['reason'] not in ('no_retained_policy', 'revealed_commit_or_context_changed')
                or _retained_policy_competition_active(plan)
                or verified and plan.get('retained_policy_commit_proof') is not None):
            return False
    else:
        if (evidence['eligible'] is not True or type(source) is not int or type(previous) is not int
                or not plan_offset <= source <= previous == plan_offset + index - 1
                or not complete_source(plans[source - plan_offset], source - plan_offset)):
            return False
        origin = plans[source - plan_offset]
        source_cohort = origin['validation_cohort_evidence']
        expected_tables = {name: (origin.get(name + '_reanalysis') or {}).get('complete') is True
                           for name in ('public_turn', 'resource', 'internal')}
        if (evidence['source_tables'] != expected_tables
                or tuple(training) != tuple(source_cohort['training_seed_history'])):
            return False
        expected_outer = set(source_cohort['validation_seeds']) | _paired_adaptive_seeds(origin)
        for position in range(source - plan_offset, index):
            row = plans[position]
            if (row.get('accepted') is not True or not _revealed_policy_mode_evidence(row, expected=True)
                    or row.get('retained_policy_link_created') is not True
                    or row.get('revealed_policy_link_created') is not True
                    or verified and row.get('retained_policy_lineage_verified') is not True):
                return False
            if position == source - plan_offset:
                continue
            prior = row['revealed_policy_competition']
            if (prior['source_plan_index'] != source or prior['predecessor_plan_index'] != plan_offset + position - 1
                    or prior['source_tables'] != expected_tables or tuple(prior['source_training_seeds']) != tuple(training)
                    or set(prior['source_validation_seeds']) != expected_outer):
                return False
            if row.get('scope') == 'SAMPLING_ESTIMATE':
                if not _retained_policy_comparison_evidence(row) or not (row.get('retained_policy_comparison') or {}).get('selected'):
                    return False
                expected_outer.update(sample['planning_seed'] for sample in row['retained_policy_comparison']['validation_samples'])
                expected_outer.update(_paired_adaptive_seeds(row))
            elif row.get('scope') not in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION',
                    'RETAINED_POLICY_CONTINUATION', 'PUBLIC_TURN_POLICY_CONTINUATION', 'LOCAL_PUBLIC_RESOURCE_CONTINUATION'):
                return False
        if set(old_outer) != expected_outer:
            return False
        proof = evidence['proof']
        if (not _revealed_policy_commit_evidence(proof, source=source, previous=previous,
                plan=plan, actions=actions, index=index)
                or _json_value(proof['action']) != _json_value(plans[index - 1].get('action'))
                or verified and (actions is None or _json_value(plan.get('retained_policy_commit_proof')) != _json_value(proof))):
            return False
        fresh = evidence['mode'] == 'revealed_recompetition'
        if evidence['reason'] != ('matched_revealed_public_state' if fresh else 'matched_deterministic_winner'):
            return False
        link = plan.get('validation_cohort_link') or {}
        if fresh:
            if (link.get('reused') is not False or link.get('seeds') not in ((), [])
                    or plan.get('retained_policy_execution_source_plan_index') is not None
                    or plan.get('scope') == 'RETAINED_POLICY_CONTINUATION'):
                return False
        elif (link.get('reused') is not True or link.get('source_plan_index') != previous
                or link.get('reason') != 'reused_deterministic_committed_cohort'):
            return False
        if plan.get('scope') == 'SAMPLING_ESTIMATE':
            if not _retained_policy_competition_active(plan) or not _validation_cohort_evidence(
                    plan, expected=True, plans=plans, index=index, plan_offset=plan_offset):
                return False
            comparison = plan['retained_policy_comparison']
            fields = {'revealed_policy_competition_version', 'source_plan_index',
                      'excluded_training_seeds', 'excluded_validation_seeds', 'fresh_validation_cohort'}
            if fresh:
                cohort = plan['validation_cohort_evidence']
                validation = cohort['validation_seeds']
                if (comparison.get('revealed_policy_competition_version') != REVEALED_POLICY_COMPETITION_VERSION
                        or comparison.get('source_plan_index') != source
                        or comparison.get('fresh_validation_cohort') is not True
                        or _json_value(comparison.get('excluded_training_seeds')) != _json_value(training)
                        or _json_value(comparison.get('excluded_validation_seeds')) != _json_value(old_outer)
                        or cohort['supplied_seeds'] not in ((), []) or len(validation) < 4
                        or set(validation).intersection((*training, *old_outer, *cohort['training_seed_history'], *cohort['training_seeds']))
                        or list(validation) != [sample['planning_seed'] for sample in comparison['validation_samples']]):
                    return False
            elif any(name in comparison for name in fields - {'source_plan_index'}):
                return False
    old_won = _retained_policy_competition_active(plan) and plan['retained_policy_comparison']['selected']
    if old_won and any(plan.get(name) is not False for name in ('public_turn_policy_link_created', 'resource_policy_link_created')):
        return False
    if plan['revealed_policy_link_created']:
        if plan.get('accepted') is not True:
            return False
        if plan.get('scope') == 'SAMPLING_ESTIMATE':
            return old_won or complete_source(plan, index)
        return active and plan.get('scope') in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION',
            'RETAINED_POLICY_CONTINUATION', 'PUBLIC_TURN_POLICY_CONTINUATION', 'LOCAL_PUBLIC_RESOURCE_CONTINUATION')
    return True


def _retained_policy_lineage_evidence(plans, index, plan, *, plan_offset=0,
                                      verified=True, actions=None):
    """Bind retained identity to accepted history and the evaluator's commit proof."""
    if plan.get('revealed_policy_competition_enabled') is True:
        return _revealed_policy_lineage_evidence(plans, index, plan, plan_offset=plan_offset,
            verified=verified, actions=actions)
    if not _revealed_policy_mode_evidence(plan, expected=False):
        return False
    enabled = plan.get('retained_policy_competition_enabled', False)
    if enabled is False:
        return not any(name in plan for name in ('retained_policy_lineage',
            'retained_policy_link_created', 'retained_policy_lineage_verified',
            'retained_policy_commit_proof'))
    if (enabled is not True or not isinstance(plans, list)
            or type(index) is not int or not 0 <= index < len(plans)
            or type(plan_offset) is not int or plan_offset < 0
            or not _retained_policy_comparison_evidence(plan)
            or type(plan.get('retained_policy_link_created')) is not bool
            or verified and plan.get('retained_policy_lineage_verified') is not True):
        return False
    lineage = plan.get('retained_policy_lineage')
    if (not isinstance(lineage, dict) or set(lineage) != {
            'source_plan_index', 'predecessor_plan_index', 'reason', 'source_domain'}
            or lineage['source_domain'] != 'complete_frozen_tree_without_public_tables'):
        return False
    source, previous = lineage['source_plan_index'], lineage['predecessor_plan_index']
    active = source is not None

    def new_source(row, row_index):
        if not isinstance(row, dict):
            return False
        roots = row.get('candidate_statistics')
        elapsed = row.get('elapsed_ms')
        return (_compute_sensitivity_evidence(row) and _source_compute_mode_matches(row, plan)
            and row.get('scope') == 'SAMPLING_ESTIMATE' and row.get('accepted') is True
            and row.get('complete_root_coverage') is True
            and row.get('complete_validation_coverage') is True
            and row.get('requested_simulations') == _plan_simulation_budget(row)
            and row.get('requested_validation_samples') == 4
            and type(row.get('completed_simulations')) is int
            and 0 < row['completed_simulations'] <= _plan_simulation_budget(row)
            and type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= _plan_window_ms(row)
            and row.get('stop_reason') in ('completed', 'soft_budget_reached')
            and (row.get('retained_policy_comparison') or {}).get('selected') is not True
            and all(isinstance(row.get(name), dict) and row[name].get('complete') is False
                    for name in ('public_turn_reanalysis', 'resource_reanalysis'))
            and isinstance(roots, (list, tuple)) and bool(roots)
            and all(isinstance(root, dict) and all(type(root.get(name)) is int and root[name] >= 4
                    for name in ('training_paired_visits', 'validation_paired_visits')) for root in roots)
            and _retained_policy_comparison_evidence(row)
            and _validation_cohort_evidence(row, expected=True, plans=plans,
                index=row_index, plan_offset=plan_offset))

    def reused_link(row, predecessor):
        link = row.get('validation_cohort_link')
        if (not isinstance(link, dict) or link.get('version') != COHORT_VERSION
                or link.get('reused') is not True or link.get('source_plan_index') != predecessor
                or type(link.get('source_plan_index')) is not int
                or link.get('reason') != 'reused_deterministic_committed_cohort'):
            return False
        seeds = link.get('seeds')
        if (not isinstance(seeds, (list, tuple)) or len(seeds) != 4
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or len(set(seeds)) != 4):
            return False
        predecessor_row = plans[predecessor - plan_offset]
        origin_seeds = ((predecessor_row.get('validation_cohort_evidence') or {}).get('validation_seeds')
            if predecessor_row.get('scope') == 'SAMPLING_ESTIMATE'
            else (predecessor_row.get('validation_cohort_link') or {}).get('seeds'))
        return isinstance(origin_seeds, (list, tuple)) and tuple(seeds) == tuple(origin_seeds[:4])

    if not active:
        if (previous is not None or lineage['reason'] not in (
                'no_retained_policy', 'deterministic_commit_or_context_changed')
                or _retained_policy_competition_active(plan)
                or verified and plan.get('retained_policy_commit_proof') is not None):
            return False
    else:
        if (type(source) is not int or type(previous) is not int
                or not plan_offset <= source <= previous == plan_offset + index - 1
                or lineage['reason'] != 'matched_deterministic_winner'
                or not reused_link(plan, previous)
                or not new_source(plans[source - plan_offset], source - plan_offset)):
            return False
        for position in range(source - plan_offset, index):
            row = plans[position]
            if (row.get('accepted') is not True or row.get('retained_policy_competition_enabled') is not True
                    or row.get('retained_policy_link_created') is not True
                    or row.get('validation_cohort_link_created') is not True
                    or row.get('turn') != plan.get('turn')
                    or verified and row.get('retained_policy_lineage_verified') is not True):
                return False
            if position == source - plan_offset:
                continue
            prior_lineage = row.get('retained_policy_lineage')
            if (not isinstance(prior_lineage, dict) or prior_lineage != dict(
                    source_plan_index=source, predecessor_plan_index=plan_offset + position - 1,
                    reason='matched_deterministic_winner',
                    source_domain='complete_frozen_tree_without_public_tables')
                    or not reused_link(row, plan_offset + position - 1)):
                return False
            if row.get('scope') == 'SAMPLING_ESTIMATE':
                if (not _retained_policy_comparison_evidence(row)
                        or not _retained_policy_competition_active(row)
                        or row['retained_policy_comparison']['selected'] is not True
                        or any(row.get(name) is not False for name in (
                            'public_turn_policy_link_created', 'resource_policy_link_created'))
                        or not _validation_cohort_evidence(row, expected=True, plans=plans,
                            index=position, plan_offset=plan_offset)):
                    return False
            elif row.get('scope') == 'RETAINED_POLICY_CONTINUATION':
                if not _retained_policy_execution_evidence(plans, position, row,
                        plan_offset=plan_offset, verified=verified):
                    return False
            elif row.get('scope') not in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION'):
                return False
        if plan.get('scope') == 'SAMPLING_ESTIMATE' and (
                not _retained_policy_competition_active(plan)
                or not _validation_cohort_evidence(plan, expected=True, plans=plans,
                    index=index, plan_offset=plan_offset)):
            return False
        if verified:
            proof = plan.get('retained_policy_commit_proof')
            if (not isinstance(proof, dict) or set(proof) != {'before', 'action', 'after',
                    'before_drink_slots', 'after_drink_slots', 'source_plan_index', 'predecessor_plan_index'}
                    or proof['source_plan_index'] != source or proof['predecessor_plan_index'] != previous
                    or not all(isinstance(proof[name], dict) for name in ('before', 'action', 'after'))
                    or proof['before'].get('turn') != plan.get('turn')
                    or proof['after'].get('turn') != plan.get('turn')
                    or _action_identity(proof['action'])
                        != _action_identity(plans[index - 1].get('action', {}))
                    or not isinstance(proof['before_drink_slots'], (list, tuple))
                    or not isinstance(proof['after_drink_slots'], (list, tuple))):
                return False
            slots = _json_value(proof['before_drink_slots'])
            if proof['action'].get('kind') == 'USE_DRINK':
                consumed = [proof['action'].get('source_slot'), proof['action'].get('drink_id')]
                if slots.count(consumed) != 1:
                    return False
                slots.remove(consumed)
            if slots != _json_value(proof['after_drink_slots']):
                return False
            if actions is not None and (not isinstance(actions, (list, tuple)) or index >= len(actions)
                    or any(_json_value(proof[key]) != _json_value(actions[index - 1].get(key))
                        for key in ('before', 'action', 'after'))
                    or _json_value(proof['after']) != _json_value(actions[index].get('before'))):
                return False
    old_won = _retained_policy_competition_active(plan) and plan['retained_policy_comparison']['selected']
    if old_won and any(plan.get(name) is not False for name in (
            'public_turn_policy_link_created', 'resource_policy_link_created')):
        return False
    if plan.get('scope') == 'RETAINED_POLICY_CONTINUATION' and not _retained_policy_execution_evidence(
            plans, index, plan, plan_offset=plan_offset, verified=verified):
        return False
    if plan['retained_policy_link_created']:
        if plan.get('accepted') is not True or plan.get('validation_cohort_link_created') is not True:
            return False
        if plan.get('scope') == 'SAMPLING_ESTIMATE':
            return old_won or new_source(plan, index)
        return active and plan.get('scope') in ('JOINT_ROOT_CONTINUATION', 'FORCED_LEGAL_ACTION',
            'RETAINED_POLICY_CONTINUATION')
    return True


def _root_selection_domain(plan, count, retained_present):
    evidence = plan.get('root_selection_eligibility')
    if isinstance(evidence, dict) and evidence.get('applicable') is True:
        return tuple(evidence['eligible_root_indices']), evidence['retained_eligible'] is True
    return tuple(range(count)), retained_present


def _root_selection_eligibility_evidence(plan, *, expected=None, verified=True,
                                       expected_visited_card_version=None):
    """Audit selection eligibility without discarding any paid root statistics."""
    evidence = plan.get('root_selection_eligibility')
    helper = plan.get('extra_action_turn_chain')
    current = (isinstance(helper, dict) and helper.get('version') == VISITED_CARD_VERSION
        and plan.get('visited_card_extra_action_enabled') is True)
    if plan.get('scope') != 'SAMPLING_ESTIMATE':
        return evidence is None and 'root_selection_eligibility_verified' not in plan
    if expected_visited_card_version is not None:
        if (expected_visited_card_version not in (REFERENCE_VISITED_CARD_VERSION, VISITED_CARD_VERSION)
                or plan.get('visited_card_extra_action_enabled') is not True or expected is False
                or not isinstance(helper, dict) or helper.get('version') != expected_visited_card_version):
            return False
        if expected_visited_card_version == REFERENCE_VISITED_CARD_VERSION:
            return evidence is None and 'root_selection_eligibility_verified' not in plan
    if not current:
        return (evidence is None and 'root_selection_eligibility_verified' not in plan
            and not (expected is True and plan.get('visited_card_extra_action_enabled') is True))
    if expected is False:
        return False
    try:
        from agent.live_decision.high_score.hif_mcts_cache import action_key_from_json
        from agent.live_decision.high_score.hif_unused_action_turn import VERSION as PROOF_VERSION

        fields = {'version', 'scope', 'applicable', 'complete', 'reason', 'selection_field',
            'public_request', 'original_legal_action_keys', 'eligible_root_indices',
            'retained_present', 'retained_eligible', 'checks', 'root_winner_index', 'selected_retained'}
        roots = plan['candidate_statistics']
        retained = plan.get('retained_policy_comparison')
        has_retained = _retained_policy_competition_active(plan)
        field = ('validation_mean_final_score' if plan.get('requested_validation_samples') else
                 'scenario_mean_final_score' if plan.get('scenario_reuse') else 'mean_final_score')
        if (not isinstance(evidence, dict) or set(evidence) != fields
                or evidence['version'] != 'hif-root-selection-eligibility-v1'
                or evidence['scope'] != 'PROVEN_UNUSED_ACTION_DRINK_MEAN_SELECTION'
                or not isinstance(roots, (list, tuple)) or not roots
                or any(type(evidence[name]) is not bool for name in
                    ('applicable', 'complete', 'retained_present', 'selected_retained'))
                or evidence['retained_present'] is not has_retained
                or (type(evidence['retained_eligible']) is not bool if has_retained
                    else evidence['retained_eligible'] is not None)
                or evidence['selection_field'] != field
                or not isinstance(evidence['public_request'], dict)
                or set(evidence['public_request']) != {'state', 'drink_slots', 'plan_type', 'utility'}
                or verified and plan.get('root_selection_eligibility_verified') is not True):
            return False
        legal = tuple(action_key_from_json(_json_value(key)) for key in evidence['original_legal_action_keys'])
        root_keys = tuple(_action_identity(root['action']) for root in roots)
        if (not legal or len(set(legal)) != len(legal) or set(root_keys) != set(legal)
                or has_retained and _action_identity(retained['action']) not in legal):
            return False
        accepted = plan.get('accepted') is True
        reason = ('search_not_accepted' if not accepted else
                  'paired_adaptive_selection_unsupported' if plan.get('paired_adaptive_validation_enabled') else
                  'finalist_selection_unsupported' if plan.get('paired_finalist_validation') else 'completed')
        if (evidence['reason'] != reason or evidence['complete'] is not accepted
                or evidence['applicable'] is not (reason == 'completed')
                or not isinstance(evidence['checks'], (list, tuple))):
            return False
        excluded = set()
        checks = evidence['checks']
        drink_keys = tuple(key for key in legal if key[0] == 'USE_DRINK') if evidence['applicable'] else ()
        if len(checks) != len(drink_keys):
            return False
        for check, key in zip(checks, drink_keys, strict=True):
            indices = tuple(i for i, root_key in enumerate(root_keys) if root_key == key)
            is_retained = has_retained and _action_identity(retained['action']) == key
            if (not isinstance(check, dict) or set(check) != {'action_key', 'root_indices', 'retained_candidate', 'proof'}
                    or tuple(check['action_key']) != key or tuple(check['root_indices']) != indices
                    or any(type(i) is not int for i in check['root_indices'])
                    or check['retained_candidate'] is not is_retained):
                return False
            proof = check['proof']
            if proof is None:
                continue
            transitions = proof.get('transition_checks') if isinstance(proof, dict) else None
            counts = ('paired_nodes', 'paired_edges', 'deterministic_steps', 'maximum_depth', 'minimum_retained_actions')
            flags = ('all_non_end_paths_paired', 'all_business_fields_preserved',
                     'events_identical', 'end_erases_positive_ap_at_every_node')
            if (not isinstance(proof, dict) or set(proof) != {'version', 'complete', 'excluded_action_key',
                    'original_legal_action_keys', 'retained_legal_action_keys', 'actions_before', 'actions_after_a',
                    'transition_checks', 'deletes_root_actions', 'forces_end', 'new_final_returns'}
                    or proof['version'] != PROOF_VERSION or proof['complete'] is not True
                    or tuple(proof['excluded_action_key']) != key
                    or {tuple(k) for k in proof['original_legal_action_keys']} != set(legal)
                    or {tuple(k) for k in proof['retained_legal_action_keys']} != set(legal) - {key}
                    or type(proof['actions_before']) is not int
                    or proof['actions_before'] != evidence['public_request']['state']['actions_remaining']
                    or type(proof['actions_after_a']) is not int or proof['actions_after_a'] <= proof['actions_before']
                    or proof['deletes_root_actions'] is not False or proof['forces_end'] is not False
                    or type(proof['new_final_returns']) is not int or proof['new_final_returns'] != 0
                    or not isinstance(transitions, dict) or set(transitions) != {*counts, *flags}
                    or any(transitions[name] is not True for name in flags)
                    or any(type(transitions[name]) is not int or transitions[name] < 0 for name in counts)
                    or not 1 <= transitions['paired_nodes'] <= 128 or transitions['paired_edges'] > 512
                    or transitions['minimum_retained_actions'] <= 0):
                return False
            excluded.add(key)
        eligible = tuple(i for i, key in enumerate(root_keys) if key not in excluded)
        retained_allowed = has_retained and _action_identity(retained['action']) not in excluded
        if (not eligible or tuple(evidence['eligible_root_indices']) != eligible
                or any(type(i) is not int for i in evidence['eligible_root_indices'])
                or evidence['retained_eligible'] != (retained_allowed if has_retained else None)):
            return False
        if not accepted:
            return evidence['root_winner_index'] is None and evidence['selected_retained'] is False
        means = []
        for root in roots:
            mean = root[field]
            if type(mean) not in (int, float) or not math.isfinite(mean):
                return False
            if field == 'validation_mean_final_score':
                samples = root['validation_samples']
                values = [sample['final_score'] for sample in samples]
                if (not values or len(values) != root['validation_paired_visits']
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                        or not math.isclose(mean, sum(values)/len(values), rel_tol=1e-12, abs_tol=1e-7)):
                    return False
            means.append(mean)
        if has_retained:
            values = [sample['final_score'] for sample in retained['validation_samples']]
            retained_mean = retained['mean_final_score']
            if (not values or len(values) != retained['completed_simulations']
                    or type(retained_mean) not in (int, float) or not math.isfinite(retained_mean)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                    or not math.isclose(retained_mean, sum(values)/len(values), rel_tol=1e-12, abs_tol=1e-7)):
                return False
        winner = max(eligible, key=lambda i: means[i])
        selected_retained = retained_allowed and retained['mean_final_score'] >= means[winner]
        if plan.get('paired_adaptive_validation_enabled'):
            chosen = plan['paired_adaptive_validation']['selected_index']
            winner = chosen if chosen < len(roots) else winner
            selected_retained = chosen == len(roots)
        elif plan.get('paired_finalist_validation') and (plan.get('finalist_validation') or {}).get('eligible'):
            winner = plan['finalist_validation']['selected_root_index']
            selected_retained = retained_allowed and retained['mean_final_score'] >= means[winner]
        selected = retained if selected_retained else roots[winner]
        return (type(evidence['root_winner_index']) is int and evidence['root_winner_index'] == winner
            and evidence['selected_retained'] is selected_retained
            and _action_identity(plan['action']) == _action_identity(selected['action'])
            and _json_value(plan.get('selected_continuation', [])) == _json_value(
                [] if selected_retained else selected.get('continuation', [])))
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def _verify_root_selection_eligibility(kernel, request, legal, plan, *, deadline):
    """Re-prove exclusions with the actual kernel and exact public decision."""
    from agent.live_decision.high_score.hif_mcts import action_key
    from agent.live_decision.high_score.hif_mcts_cache import public_request
    from agent.live_decision.high_score.validation_cache import validation_session
    from agent.live_decision.high_score.hif_unused_action_turn import prove_unused_action_drink_in_turn

    def check_budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_root_selection_eligibility_verification_timeout')

    check_budget()
    evidence = plan['root_selection_eligibility']
    if (_json_value(evidence['public_request']) != public_request(request)
            or _json_value(evidence['original_legal_action_keys']) != _json_value([action_key(a) for a in legal])):
        return False
    actions = {action_key(action): action for action in legal}
    with validation_session(check_budget, reuse_parent=True):
        for check in evidence['checks']:
            if check['proof'] is not None:
                actual = prove_unused_action_drink_in_turn(kernel, request, actions[tuple(check['action_key'])])
                check_budget()
                if _json_value(actual) != _json_value(check['proof']):
                    return False
    check_budget()
    return True


def _retained_policy_comparison_evidence(plan, *, expected=None):
    """Recompute one extra policy's paired returns without adding a legal root.

    This checks numerical evidence and winner identity. The controller's
    committed-source lineage is an additional, independent verification gate.
    """
    enabled = plan.get('retained_policy_competition_enabled', False)
    version = plan.get('retained_policy_competition_version')
    evidence = plan.get('retained_policy_comparison')
    scope = plan.get('root_selection_scope')
    if (type(enabled) is not bool or expected is not None and (
            type(expected) is not bool or enabled is not expected)):
        return False
    if not enabled:
        return (version is None and evidence is None
            and scope != 'paired_retained_policy_competition')
    if version != RETAINED_POLICY_COMPETITION_VERSION:
        return False
    if evidence is None:
        return scope != 'paired_retained_policy_competition'
    if (not isinstance(evidence, dict)
            or evidence.get('version') != RETAINED_POLICY_COMPETITION_VERSION
            or any(type(evidence.get(name)) is not bool for name in ('eligible', 'complete', 'selected'))
            or any(type(evidence.get(name)) is not int or evidence[name] < 0
                   for name in ('started_simulations', 'completed_simulations'))
            or not isinstance(evidence.get('validation_samples'), (tuple, list))):
        return False
    samples = evidence['validation_samples']
    if not evidence['eligible']:
        return (scope != 'paired_retained_policy_competition'
            and evidence['started_simulations'] == evidence['completed_simulations'] == 0
            and not samples and not evidence['complete'] and not evidence['selected']
            and evidence.get('mean_final_score') is None)
    roots = plan.get('candidate_statistics')
    count = evidence['completed_simulations']
    if (plan.get('scope') != 'SAMPLING_ESTIMATE'
            or scope != 'paired_retained_policy_competition'
            or plan.get('requested_validation_samples') != 4
            or plan.get('complete_validation_coverage') is not True
            or not evidence['complete'] or not 4 <= count <= 32
            or evidence['started_simulations'] != count or len(samples) != count
            or not isinstance(roots, (tuple, list)) or not roots
            or not isinstance(evidence.get('action'), dict)
            or not isinstance(plan.get('action'), dict)):
        return False

    def sample_values(values):
        if not isinstance(values, (tuple, list)) or len(values) != count:
            return None
        seeds, scores = [], []
        for index, sample in enumerate(values):
            if (not isinstance(sample, dict)
                    or type(sample.get('scenario_index')) is not int or sample['scenario_index'] != index
                    or type(sample.get('planning_seed')) is not int
                    or not 0 <= sample['planning_seed'] < 2**31
                    or type(sample.get('final_score')) not in (int, float)
                    or not math.isfinite(sample['final_score'])
                    or sample.get('return_kind') not in ('proven_tail_expected_score',
                        'terminal_evaluator_expected_score', 'sampled_terminal_score')):
                return None
            seeds.append(sample['planning_seed'])
            scores.append(sample['final_score'])
        if len(set(seeds)) != count:
            return None
        return seeds, sum(scores)/count

    values = sample_values(samples)
    if values is None:
        return False
    seeds, retained_mean = values
    reported_mean = evidence.get('mean_final_score')
    if (type(reported_mean) not in (int, float) or not math.isfinite(reported_mean)
            or not math.isclose(reported_mean, retained_mean, rel_tol=1e-12, abs_tol=1e-7)):
        return False
    means = []
    for root in roots:
        if (not isinstance(root, dict) or not isinstance(root.get('action'), dict)
                or type(root.get('validation_paired_visits')) is not int
                or root['validation_paired_visits'] != count):
            return False
        values = sample_values(root.get('validation_samples'))
        reported_mean = root.get('validation_mean_final_score')
        if (values is None or values[0] != seeds
                or type(reported_mean) not in (int, float) or not math.isfinite(reported_mean)
                or not math.isclose(reported_mean, values[1], rel_tol=1e-12, abs_tol=1e-7)):
            return False
        means.append(values[1])
    if _action_identity(evidence['action']) not in {_action_identity(root['action']) for root in roots}:
        return False
    if plan.get('paired_adaptive_validation_enabled') is True:
        if not _paired_adaptive_validation_evidence(plan):
            return False
        selected = plan['paired_adaptive_validation']['selected_index']
        winner = selected if selected < len(roots) else 0
        retained_wins = selected == len(roots)
    else:
        if not _root_selection_eligibility_evidence(plan):
            return False
        eligible, retained_eligible = _root_selection_domain(plan, len(roots), True)
        winner = max(eligible, key=lambda index: means[index])
        retained_wins = retained_eligible and retained_mean >= means[winner]
    if evidence['selected'] is not retained_wins:
        return False
    expected_action = evidence['action'] if retained_wins else roots[winner]['action']
    expected_continuation = [] if retained_wins else roots[winner].get('continuation', [])
    return (_action_identity(plan['action']) == _action_identity(expected_action)
        and _json_value(plan.get('selected_continuation', [])) == _json_value(expected_continuation))


def _lazy_validation_seed_evidence(plan, *, expected=None):
    """Seed generation counts are separate from attempted and complete rounds."""
    if expected is not None and type(expected) is not bool:
        return False
    present = 'lazy_validation_seeds_enabled' in plan or 'validation_seed_allocation' in plan
    if not present:
        return expected is not True
    evidence = plan.get('validation_seed_allocation')
    if (expected is False or plan.get('lazy_validation_seeds_enabled') is not True
            or not isinstance(evidence, dict)
            or evidence.get('version') != LAZY_VALIDATION_SEEDS_VERSION
            or plan.get('retrieval_training_order_enabled') is not True
            or plan.get('requested_validation_samples') != 4):
        return False
    counts = tuple(evidence.get(name) for name in
                   ('capacity', 'mandatory', 'generated', 'started', 'complete_rounds'))
    if (not all(type(value) is int and value >= 0 for value in counts)
            or counts[1] != 4 or not counts[0] >= counts[2] >= counts[3] >= counts[4]
            or type(plan.get('validation_sample_limit')) is not int
            or counts[0] != plan['validation_sample_limit']):
        return False
    if plan.get('complete_validation_coverage') is True:
        visits = plan.get('validation_paired_visits')
        if (not counts[2] == counts[3] == counts[4]
                or not isinstance(visits, (list, tuple)) or not visits
                or any(type(value) is not int or value != counts[4] for value in visits)):
            return False
    return True


def _validation_cohort_evidence(plan, *, expected=None, plans=None, index=None, plan_offset=0):
    """Reused worlds remain training-disjoint, not newly independent samples."""
    if expected is not None and type(expected) is not bool:
        return False
    fields = ('deterministic_validation_cohort_enabled', 'validation_cohort_evidence',
              'validation_cohort_link', 'validation_cohort_link_created')
    if not any(field in plan for field in fields):
        return expected is not True
    if type(plan_offset) is not int or plan_offset < 0:
        return False
    evidence, link = plan.get('validation_cohort_evidence'), plan.get('validation_cohort_link')
    if (expected is False or plan.get('deterministic_validation_cohort_enabled') is not True
            or plan.get('lazy_validation_seeds_enabled') is not True
            or plan.get('retrieval_training_order_enabled') is not True
            or plan.get('requested_validation_samples') != 4
            or not isinstance(evidence, dict) or evidence.get('version') != COHORT_VERSION
            or not isinstance(link, dict) or link.get('version') != COHORT_VERSION
            or type(link.get('reused')) is not bool
            or type(plan.get('validation_cohort_link_created')) is not bool):
        return False

    fresh_quality = (plan.get('fresh_adaptive_training') is True
        and _quality_reference_evidence(plan, expected=True, window_seconds=300.)
        and _compute_sensitivity_evidence(plan, expected=False, simulations=256)
        and type(plan.get('requested_simulations')) is int and plan['requested_simulations'] == 256
        and plan.get('scenario_reuse') is True
        and all(plan.get(name + '_enabled') is True for name in (
            'internal_reanalysis', 'resource_reanalysis', 'public_turn_reanalysis',
            'public_turn_policy_continuation', 'retained_tree_policy_execution',
            'retained_policy_execution', 'retained_policy_competition')))
    if plan.get('fresh_adaptive_training', False) is not False and not fresh_quality:
        return False

    def seeds_valid(seeds, *, upper=2**31):
        return (isinstance(seeds, (tuple, list))
                and all(type(seed) is int and 0 <= seed < upper for seed in seeds)
                and len(set(seeds)) == len(seeds))

    supplied, training, history, validation = (evidence.get(name) for name in
        ('supplied_seeds', 'training_seeds', 'training_seed_history', 'validation_seeds'))
    if (not all(seeds_valid(seeds) for seeds in (supplied, training, validation))
            or not seeds_valid(history, upper=2**32 if fresh_quality else 2**31)
            or len(supplied) not in (0, 4) or len(training) not in (0, 4)
            or len(validation) > 32 or not seeds_valid(link.get('seeds'))):
        return False
    reused, new = evidence.get('reused_count'), evidence.get('new_seed_count')
    reuse = bool(supplied)
    expected_scope = 'TRAINING_DISJOINT_REUSED_SCENARIOS' if reuse else 'FRESH_TRAINING_DISJOINT_SCENARIOS'
    if (type(reused) is not int or type(new) is not int or reused < 0 or new < 0
            or reused + new != len(validation) or reused != min(len(validation), len(supplied))
            or tuple(validation[:reused]) != tuple(supplied[:reused])
            or set(validation).intersection((*training, *history))
            or set(supplied).intersection((*training, *history))
            or evidence.get('scope') != expected_scope or link['reused'] is not reuse
            or tuple(link['seeds']) != tuple(supplied)):
        return False
    source = link.get('source_plan_index')
    reason = link.get('reason')
    if reuse:
        if (reason != 'reused_deterministic_committed_cohort' or type(source) is not int
                or source < plan_offset or type(index) is not int or source >= plan_offset + index
                or not isinstance(plans, list) or source - plan_offset >= len(plans)):
            return False
        origin = plans[source - plan_offset]
        if (not isinstance(origin, dict) or origin.get('accepted') is not True
                or origin.get('deterministic_validation_cohort_enabled') is not True
                or origin.get('validation_cohort_link_created') is not True):
            return False
        if origin.get('scope') == 'SAMPLING_ESTIMATE':
            origin_evidence = origin.get('validation_cohort_evidence')
            if (origin.get('complete_validation_coverage') is not True
                    or not isinstance(origin_evidence, dict)
                    or origin_evidence.get('version') != COHORT_VERSION):
                return False
            origin_seeds = origin_evidence.get('validation_seeds')
        else:
            origin_link = origin.get('validation_cohort_link')
            if (not isinstance(origin_link, dict) or origin_link.get('version') != COHORT_VERSION
                    or origin_link.get('reused') is not True):
                return False
            origin_seeds = origin_link.get('seeds')
        if not seeds_valid(origin_seeds) or tuple(origin_seeds[:4]) != tuple(supplied):
            return False
    elif (reason not in ('no_saved_cohort', 'reset', 'not_committed', 'context_changed',
                         'public_state_changed', 'drink_slots_changed')
            or source is not None and (type(source) is not int or source < 0
                or type(index) is not int or source >= plan_offset + index)
            or reason in ('no_saved_cohort', 'reset') and source is not None
            or reason not in ('no_saved_cohort', 'reset') and source is None):
        return False
    allocation = plan.get('validation_seed_allocation')
    if (not isinstance(allocation, dict) or allocation.get('started') != len(validation)
            or type(allocation.get('started')) is not int):
        return False
    complete = plan.get('complete_validation_coverage') is True
    roots = plan.get('candidate_statistics')
    if not isinstance(roots, (tuple, list)) or not roots:
        return not complete and not validation
    for root in roots:
        if not isinstance(root, dict):
            return False
        samples = root.get('validation_samples')
        if not isinstance(samples, (tuple, list)) or len(samples) > len(validation):
            return False
        if complete and len(samples) != len(validation):
            return False
        for scenario, sample in enumerate(samples):
            if (not isinstance(sample, dict) or type(sample.get('scenario_index')) is not int
                    or sample['scenario_index'] != scenario
                    or type(sample.get('planning_seed')) is not int
                    or sample['planning_seed'] != validation[scenario]
                    or type(sample.get('final_score')) not in (int, float)
                    or not math.isfinite(sample['final_score'])):
                return False
        if type(root.get('validation_paired_visits')) is not int or root['validation_paired_visits'] != len(samples):
            return False
        if complete:
            mean = root.get('validation_mean_final_score')
            if (not samples or type(mean) not in (int, float) or not math.isfinite(mean)
                    or not math.isclose(mean, sum(row['final_score'] for row in samples) / len(samples),
                                        rel_tol=1e-12, abs_tol=1e-7)):
                return False
    selection_scope = ('training_disjoint_reused_validation' if reuse
                       else 'independent_frozen_policy_validation')
    public_turn = plan.get('public_turn_reanalysis')
    if (isinstance(public_turn, dict) and public_turn.get('complete') is True
            and _public_turn_reanalysis_evidence(plan, roots)):
        selection_scope = ('training_disjoint_reused_public_turn_validation' if reuse
                           else 'independent_public_turn_reanalysis_validation')
    resource = plan.get('resource_reanalysis')
    if (isinstance(resource, dict) and resource.get('complete') is True
            and _resource_reanalysis_evidence(plan)):
        selection_scope = ('training_disjoint_reused_resource_policy_validation' if reuse
                           else 'independent_resource_policy_validation')
    if _retained_policy_competition_active(plan):
        selection_scope = 'paired_retained_policy_competition'
    if complete and (len(training) != 4 or not set(training).issubset(history)
            or len(validation) < 4 or reuse and reused != 4
            or allocation.get('complete_rounds') != len(validation)
            or plan.get('root_selection_scope') != selection_scope):
        return False
    return True
