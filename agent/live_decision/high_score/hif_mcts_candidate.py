"""Opt-in offline dynamic HIF candidate; never a native/default loader."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, replace

from .planner import PlannerConfig
from .hif_mcts import (
    QUALITY_REFERENCE_VERSION,
    COMPUTE_SENSITIVITY_VERSION,
    PARTIAL_NODE_TIMING_VERSION,
    POLICY_CONTINUATION_TRIGGER,
    LAZY_VALIDATION_SEEDS_VERSION,
    PUBLIC_TURN_BUDGET_RESERVATION_VERSION,
    PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION,
    PUBLIC_VALIDATION_PHASE_BUDGET_VERSION,
    OfflineHifMcts,
    action_key,
)
from .contracts import ActionKind, GameAction
from .hif_planner import HifActionChainPlanner
from .hif_dominance import prove_forced_end_drink_dominance
from .hif_proven_tail import VERSION as PROVEN_TAIL_VERSION
from .hif_proven_tail import HifProvenTailTable
from .hif_choice_roots import (
    _complete_legal,
    post_card_end_state,
    build_choice_root_options,
    build_stop_after_card_root_options,
)
from .hif_combo_timing import SenseComboTimingPolicy
from .hif_drink_advice import VERSION as DRINK_ADVICE_VERSION
from .hif_drink_advice import HifDrinkAdvice
from .validation_cache import validation_session
from .hif_double_advice import VERSION as DOUBLE_ADVICE_VERSION
from .hif_double_advice import HifDoubleAdvice
from .hif_search_session import HifSearchSession
from .hif_chain_candidate import PairedPlanningKernel
from .hif_temporal_advice import VERSION as TEMPORAL_ADVICE_VERSION
from .hif_temporal_advice import HifTemporalAdvice
from .hif_simple_tail_domain import VERSION as SIMPLE_TAIL_VERSION
from .hif_simple_tail_domain import (
    MAX_SEQUENCE_PATHS,
    FINAL_DOUBLE_VERSION,
    FINAL_OFFHAND_REMOVE_VERSION,
    assess_simple_tail_domain,
)
from .hif_exact_tail_ordering import VERSION as EXACT_TAIL_VERSION
from .hif_exact_tail_ordering import HifExactTailOrdering
from .hif_internal_reanalysis import VERSION as INTERNAL_REANALYSIS_VERSION
from .hif_internal_reanalysis import COMPLETE_ROOT_SUCCESSORS_VERSION
from .hif_policy_continuation import RetrievalTrainingOrder, UnknownNodeTimingPolicy
from .hif_retrieval_combo_advice import VERSION as RETRIEVAL_COMBO_VERSION
from .hif_retrieval_combo_advice import HifRetrievalComboAdvice
from .hif_frozen_policy_continuation import (
    make_link,
    export_link,
    matches_link,
    restore_link,
    business_action,
    qualified_source,
)
from .hif_retained_policy_competition import REVEALED_VERSION, TREE_EXECUTION_VERSION

VERSION = 'hif-public-state-mcts-trial-v22'
PUBLIC_TURN_SCORE_TRANSLATION_VERSION = 'hif-installed-score-translation-v1'
ACTUAL_TAIL_SESSION_REUSE_VERSION = 'hif-actual-tail-session-reuse-v1'
_MAX_RETAINED_TAIL_ENTRIES = 32768
CONFIG = dict(version=VERSION, batch_seconds=900., planning_seed_base=603000000,
              policy_kwargs=dict(simulations=256, min_root_samples=4, window_seconds=30.,
                                 soft_window_seconds=3., terminal_attempt_ms=500., scenario_reuse=True,
                                 validation_samples=4, resource_dominance=True, double_advice=True,
                                 adaptive_validation=True, temporal_advice=False, drink_advice=False,
                                 progressive_widening=False, retrieval_combo_advice=False,
                                 exact_tail_ordering=False, complete_tail_evaluation=False,
                                 simple_tail_evaluation=True, simple_final_turn_evaluation=True,
                                 joint_choice_roots=True),
              reward_scale=100000., planning_sampler='common-card-exponential-race-v1',
              value_scope='FROZEN_PUBLIC_TREE_FRESH_SCENARIO_TERMINAL_ESTIMATES',
              root_selection='independent_validation_scenarios',
              validation_allocation='half_training_then_paired_rounds_within_soft_window_max32',
              joint_training_allocation='complete_existing_training_quota_within_hard_deadline',
              resource_dominance='deterministic_forced_end_drink_v1',
              double_advice_version=DOUBLE_ADVICE_VERSION,
              two_turn_domain=SIMPLE_TAIL_VERSION, two_turn_sequence_paths=MAX_SEQUENCE_PATHS,
              retrieval_combo_version=RETRIEVAL_COMBO_VERSION,
              exact_tail_ordering_version=EXACT_TAIL_VERSION,
              terminal_horizon=2,
              validation_terminal_budget='fixed_public_domain_exact_else_sample_to_terminal',
              hypothetical_final_turn='exact_in_public_domain_else_frozen_tree_complete_return',
              actual_final_turn='existing_complete_solver',
              mandatory_choice='deterministic_public_target_joint_root_then_fresh_binding',
              repeated_simulations_are_independent_samples=False,
              future_random_support_modeled=False)


class HifMctsCandidatePolicy:
    version = VERSION

    def __init__(self, kernel, baseline, *, planning_seed, deadline,
                 simulations=256, min_root_samples=4, window_seconds=30.,
                 soft_window_seconds=3., terminal_attempt_ms=500., scenario_reuse=True,
                 validation_samples=4, resource_dominance=True, double_advice=False,
                 adaptive_validation=False, temporal_advice=False, drink_advice=False,
                 progressive_widening=False, exact_tail_ordering=False, retrieval_combo_advice=False,
                 complete_tail_evaluation=False, double_return_weights=None,
                 double_return_horizon=None, frozen_tail_evaluation=False, simple_tail_evaluation=False,
                 simple_final_turn_evaluation=False, joint_choice_roots=False, continuation_samples=0,
                 continuation_domain=False, actual_two_turn_evaluation=False,
                 continuation_incomplete_actions=False, fresh_adaptive_training=False,
                 continuation_portfolio=False, bounded_actual_two_turn_evaluation=False,
                 sparse_turn_chain=False, stop_after_card_roots=False, installed_card_use_value=False,
                 root_coverage_first=False, guard_unpriced_schedules=False,
                 single_node_expansion=False, deterministic_policy_continuation=False,
                 equivalent_instance_reuse=False, actual_two_turn_probe_ms=0.,
                 guard_unpriced_resources=False, paired_finalist_validation=False,
                 public_turn_reanalysis=False, direct_double_marginal=False,
                 unknown_node_timing=False, retrieval_training_order=False,
                 lazy_validation_seeds=False, deterministic_validation_cohort=False,
                 partial_node_timing=False, installed_score_turn_chain=False, installed_score_multi_action=False,
                 installed_score_focus_projection=False,
                 public_turn_policy_continuation=False, pair_order_advice=False,
                 public_turn_score_translation=False, actual_tail_session_reuse=False,
                 public_turn_budget_reservation=False, public_validation_phase_budget=False,
                 extra_action_turn_chain=False, visited_card_extra_action=False,
                 visited_card_extra_action_version=None,
                 parallel_public_graph=False, resource_reanalysis=False,
                 retained_policy_competition=False, retained_policy_execution=False,
                 retained_tree_policy_execution=False, internal_reanalysis=False,
                 final_double_tail_evaluation=False, final_offhand_remove_tail_evaluation=False,
                 quality_reference=False, compute_sensitivity=False,
                 internal_complete_root_successors=False, revealed_policy_competition=False,
                 internal_successor_graph=False, paired_adaptive_validation=False,
                 uniform_validation_time_guard=False):
        from .hif_extra_action_continuation import VISITED_CARD_VERSION, REFERENCE_VISITED_CARD_VERSION

        if (visited_card_extra_action_version is not None and (
                not visited_card_extra_action or visited_card_extra_action_version not in (
                    REFERENCE_VISITED_CARD_VERSION, VISITED_CARD_VERSION))):
            raise ValueError('visited_card_extra_action_version_not_supported')
        # The current version remains implicit so old/default configuration
        # and cache identities do not change. The reference is explicit.
        if visited_card_extra_action_version == VISITED_CARD_VERSION:
            visited_card_extra_action_version = None
        if (type(uniform_validation_time_guard) is not bool or uniform_validation_time_guard and (
                not quality_reference or not adaptive_validation or not lazy_validation_seeds
                or not deterministic_validation_cohort or not scenario_reuse or simulations != 256
                or min_root_samples != 4 or validation_samples != 4 or compute_sensitivity
                or paired_adaptive_validation or paired_finalist_validation or continuation_samples)):
            raise ValueError('mcts_uniform_validation_time_guard_requires_quality_four_rounds')
        if (type(paired_adaptive_validation) is not bool or paired_adaptive_validation and (
                not quality_reference or not adaptive_validation or not lazy_validation_seeds
                or not deterministic_validation_cohort or not scenario_reuse or simulations != 256
                or min_root_samples != 4 or validation_samples != 4 or compute_sensitivity
                or paired_finalist_validation or continuation_samples)):
            raise ValueError('mcts_paired_adaptive_validation_requires_quality_four_paired_search')
        if (type(revealed_policy_competition) is not bool or revealed_policy_competition and (
                not quality_reference or not retained_policy_competition or compute_sensitivity
                or simulations != 256)):
            raise ValueError('mcts_revealed_policy_requires_quality_competition')
        if type(quality_reference) is not bool:
            raise ValueError('mcts_quality_reference_requires_bool')
        if type(compute_sensitivity) is not bool:
            raise ValueError('mcts_compute_sensitivity_requires_bool')
        if type(fresh_adaptive_training) is not bool:
            raise ValueError('mcts_fresh_adaptive_training_requires_scenario_reuse')
        if compute_sensitivity and (not quality_reference or type(simulations) is not int
                or simulations not in (256, 512, 1024)):
            raise ValueError('mcts_compute_sensitivity_requires_quality_budget')
        if (type(final_offhand_remove_tail_evaluation) is not bool
                or final_offhand_remove_tail_evaluation and not (
                    simple_tail_evaluation and simple_final_turn_evaluation and final_double_tail_evaluation)):
            raise ValueError('mcts_final_offhand_remove_tail_requires_final_double_domain')
        if (type(final_double_tail_evaluation) is not bool
                or final_double_tail_evaluation and not (
                    simple_tail_evaluation and simple_final_turn_evaluation)):
            raise ValueError('mcts_final_double_tail_requires_simple_final_domain')
        if (type(internal_reanalysis) is not bool
                or internal_reanalysis and not retained_tree_policy_execution):
            raise ValueError('mcts_internal_reanalysis_requires_retained_tree_stack')
        if (type(internal_complete_root_successors) is not bool
                or internal_complete_root_successors and not (quality_reference and internal_reanalysis)):
            raise ValueError('mcts_internal_complete_roots_require_quality_internal_stack')
        if (type(internal_successor_graph) is not bool or internal_successor_graph and (
                not quality_reference or not internal_reanalysis
                or compute_sensitivity or simulations != 256)):
            raise ValueError('mcts_internal_successor_graph_requires_quality_internal_stack')
        public_turn_installed_stack = (public_turn_reanalysis and unknown_node_timing
            and partial_node_timing and retrieval_training_order and lazy_validation_seeds
            and deterministic_validation_cohort and installed_score_turn_chain)
        quality_fresh_stack = (quality_reference and not compute_sensitivity and simulations == 256
            and public_turn_installed_stack and resource_reanalysis and internal_reanalysis
            and public_validation_phase_budget)
        if (type(retained_policy_execution) is not bool
                or retained_policy_execution and not retained_policy_competition):
            raise ValueError('mcts_retained_execution_requires_competition')
        if (type(retained_tree_policy_execution) is not bool
                or retained_tree_policy_execution and not retained_policy_execution):
            raise ValueError('mcts_retained_tree_execution_requires_execution')
        if (type(retained_policy_competition) is not bool or retained_policy_competition and (
                not resource_reanalysis or simulations != 256 and not compute_sensitivity
                or min_root_samples != 4
                or validation_samples != 4 or window_seconds > (300. if quality_reference else 30.)
                or deterministic_policy_continuation or continuation_samples
                or paired_finalist_validation or equivalent_instance_reuse)):
            raise ValueError('mcts_retained_policy_requires_resource_stack')
        if (type(resource_reanalysis) is not bool or resource_reanalysis and (
                not public_turn_installed_stack or not public_turn_policy_continuation
                or not public_turn_score_translation or not public_turn_budget_reservation
                or equivalent_instance_reuse or min_root_samples != 4 or validation_samples != 4)):
            raise ValueError('mcts_resource_reanalysis_requires_complete_public_stack')
        if (deadline is None and not quality_reference or deadline is not None and not math.isfinite(deadline)
                or not math.isfinite(window_seconds) or window_seconds <= 0
                or quality_reference and window_seconds > 300.
                or not math.isfinite(terminal_attempt_ms) or terminal_attempt_ms <= 0
                or type(resource_dominance) is not bool or type(double_advice) is not bool
                or type(temporal_advice) is not bool or temporal_advice and not double_advice
                or type(drink_advice) is not bool or drink_advice and temporal_advice
                or type(exact_tail_ordering) is not bool or type(retrieval_combo_advice) is not bool
                or type(complete_tail_evaluation) is not bool
                or type(frozen_tail_evaluation) is not bool
                or type(simple_tail_evaluation) is not bool
                or type(simple_final_turn_evaluation) is not bool
                or type(actual_two_turn_evaluation) is not bool
                or type(actual_two_turn_probe_ms) not in (int, float)
                or not math.isfinite(actual_two_turn_probe_ms)
                or not 0 <= actual_two_turn_probe_ms <= 3000
                or actual_two_turn_probe_ms and (
                    not simple_tail_evaluation or actual_two_turn_evaluation)
                or type(bounded_actual_two_turn_evaluation) is not bool
                or bounded_actual_two_turn_evaluation and not (
                    actual_two_turn_evaluation and simple_tail_evaluation)
                or type(continuation_portfolio) is not bool
                or type(sparse_turn_chain) is not bool
                or type(installed_card_use_value) is not bool
                or installed_card_use_value and not sparse_turn_chain
                or type(root_coverage_first) is not bool
                or type(single_node_expansion) is not bool
                or type(equivalent_instance_reuse) is not bool
                or type(deterministic_policy_continuation) is not bool
                or deterministic_policy_continuation and (
                    not scenario_reuse or validation_samples < 4 or continuation_samples
                    or continuation_domain or continuation_incomplete_actions or continuation_portfolio
                    or frozen_tail_evaluation)
                or root_coverage_first and (not validation_samples or continuation_samples)
                or type(guard_unpriced_schedules) is not bool
                or guard_unpriced_schedules and (
                    not double_advice or temporal_advice or double_return_weights is not None)
                or type(guard_unpriced_resources) is not bool
                or guard_unpriced_resources and (
                    not double_advice or temporal_advice or double_return_weights is not None)
                or type(paired_finalist_validation) is not bool
                or type(public_turn_reanalysis) is not bool
                or type(public_turn_policy_continuation) is not bool
                or public_turn_policy_continuation and not public_turn_installed_stack
                or type(pair_order_advice) is not bool
                or pair_order_advice and not public_turn_policy_continuation
                or type(public_turn_score_translation) is not bool
                or type(actual_tail_session_reuse) is not bool
                or type(public_turn_budget_reservation) is not bool
                or public_turn_budget_reservation and not actual_tail_session_reuse
                or type(public_validation_phase_budget) is not bool
                or public_validation_phase_budget and not public_turn_budget_reservation
                or actual_tail_session_reuse and not (
                    public_turn_score_translation and actual_two_turn_probe_ms > 0
                    and simple_tail_evaluation)
                or public_turn_score_translation and not (
                    public_turn_installed_stack and public_turn_policy_continuation and pair_order_advice
                    and simple_tail_evaluation and simple_final_turn_evaluation)
                or public_turn_reanalysis and not public_turn_installed_stack and (
                    unknown_node_timing or partial_node_timing or retrieval_training_order
                    or lazy_validation_seeds or deterministic_validation_cohort or installed_score_turn_chain)
                or type(direct_double_marginal) is not bool
                or direct_double_marginal and (
                    not double_advice or temporal_advice or double_return_weights is not None)
                or type(unknown_node_timing) is not bool
                or unknown_node_timing and (exact_tail_ordering or sparse_turn_chain
                    or continuation_samples or public_turn_reanalysis and not public_turn_installed_stack)
                or type(partial_node_timing) is not bool
                or partial_node_timing and (not unknown_node_timing or exact_tail_ordering
                    or sparse_turn_chain or continuation_samples
                    or public_turn_reanalysis and not public_turn_installed_stack
                    or continuation_portfolio or deterministic_policy_continuation)
                or type(installed_score_turn_chain) is not bool
                or installed_score_turn_chain and (not partial_node_timing
                    or not deterministic_validation_cohort)
                or type(installed_score_multi_action) is not bool
                or installed_score_multi_action and not installed_score_turn_chain
                or type(installed_score_focus_projection) is not bool
                or installed_score_focus_projection and not installed_score_turn_chain
                or type(extra_action_turn_chain) is not bool
                or extra_action_turn_chain and not public_validation_phase_budget
                or type(visited_card_extra_action) is not bool
                or visited_card_extra_action and not extra_action_turn_chain
                or type(parallel_public_graph) is not bool
                or parallel_public_graph and not (extra_action_turn_chain and public_validation_phase_budget)
                or type(retrieval_training_order) is not bool
                or retrieval_training_order and (
                    not unknown_node_timing or not joint_choice_roots or not scenario_reuse
                    or not adaptive_validation or simulations != 256 and not compute_sensitivity
                    or min_root_samples != 4
                    or validation_samples != 4 or window_seconds > (300. if quality_reference else 30.)
                    or exact_tail_ordering or sparse_turn_chain or continuation_samples
                    or public_turn_reanalysis and not public_turn_installed_stack
                    or deterministic_policy_continuation
                    or single_node_expansion)
                or type(lazy_validation_seeds) is not bool
                or lazy_validation_seeds and (not retrieval_training_order
                    or fresh_adaptive_training and not quality_fresh_stack
                    or paired_finalist_validation or continuation_portfolio)
                or type(deterministic_validation_cohort) is not bool
                or deterministic_validation_cohort and not lazy_validation_seeds
                or type(stop_after_card_roots) is not bool
                or stop_after_card_roots and (simulations != 256 and not compute_sensitivity
                    or min_root_samples != 4
                    or validation_samples != 4 or continuation_samples)
                or sparse_turn_chain and (not validation_samples or continuation_samples)
                or continuation_portfolio and (continuation_samples != 4 or continuation_incomplete_actions)
                or actual_two_turn_evaluation and not (
                    complete_tail_evaluation or frozen_tail_evaluation or simple_tail_evaluation)
                or type(joint_choice_roots) is not bool
                or joint_choice_roots and validation_samples < 1
                or simple_final_turn_evaluation and not simple_tail_evaluation
                or sum((frozen_tail_evaluation, complete_tail_evaluation, simple_tail_evaluation)) > 1
                or double_return_horizon is not None and double_return_weights is None
                or double_return_weights is not None and (not double_advice or temporal_advice)
                or retrieval_combo_advice and (temporal_advice or drink_advice)):
            raise ValueError('invalid_mcts_candidate_budget')
        # Constructor values are the portable policy boundary; the current
        # host deadline and live objects are intentionally excluded.
        del public_turn_installed_stack, quality_fresh_stack, REFERENCE_VISITED_CARD_VERSION, VISITED_CARD_VERSION
        self._cache_configuration = {name: value for name, value in locals().items()
                                     if name not in ('self', 'kernel', 'baseline', 'deadline')}
        if not quality_reference:
            self._cache_configuration.pop('quality_reference')
        if not revealed_policy_competition:
            self._cache_configuration.pop('revealed_policy_competition')
        if not compute_sensitivity:
            self._cache_configuration.pop('compute_sensitivity')
        if not sparse_turn_chain:
            # The default/native v22 cache contract remains byte-compatible.
            self._cache_configuration.pop('sparse_turn_chain')
        if not stop_after_card_roots:
            self._cache_configuration.pop('stop_after_card_roots')
        if not installed_card_use_value:
            self._cache_configuration.pop('installed_card_use_value')
        if not root_coverage_first:
            self._cache_configuration.pop('root_coverage_first')
        if not guard_unpriced_schedules:
            self._cache_configuration.pop('guard_unpriced_schedules')
        if not guard_unpriced_resources:
            self._cache_configuration.pop('guard_unpriced_resources')
        if not paired_finalist_validation:
            self._cache_configuration.pop('paired_finalist_validation')
        if not uniform_validation_time_guard:
            self._cache_configuration.pop('uniform_validation_time_guard')
        if not paired_adaptive_validation:
            self._cache_configuration.pop('paired_adaptive_validation')
        if not public_turn_reanalysis:
            self._cache_configuration.pop('public_turn_reanalysis')
        if not public_turn_policy_continuation:
            self._cache_configuration.pop('public_turn_policy_continuation')
        if not pair_order_advice:
            self._cache_configuration.pop('pair_order_advice')
        if not public_turn_score_translation:
            self._cache_configuration.pop('public_turn_score_translation')
        if not actual_tail_session_reuse:
            self._cache_configuration.pop('actual_tail_session_reuse')
        if not public_turn_budget_reservation:
            self._cache_configuration.pop('public_turn_budget_reservation')
        if not public_validation_phase_budget:
            self._cache_configuration.pop('public_validation_phase_budget')
        if not direct_double_marginal:
            self._cache_configuration.pop('direct_double_marginal')
        if not unknown_node_timing:
            self._cache_configuration.pop('unknown_node_timing')
        if not retrieval_training_order:
            self._cache_configuration.pop('retrieval_training_order')
        if not lazy_validation_seeds:
            self._cache_configuration.pop('lazy_validation_seeds')
        if not deterministic_validation_cohort:
            self._cache_configuration.pop('deterministic_validation_cohort')
        if not partial_node_timing:
            self._cache_configuration.pop('partial_node_timing')
        if not installed_score_turn_chain:
            self._cache_configuration.pop('installed_score_turn_chain')
        if not installed_score_multi_action:
            self._cache_configuration.pop('installed_score_multi_action')
        if not installed_score_focus_projection:
            self._cache_configuration.pop('installed_score_focus_projection')
        if not extra_action_turn_chain:
            self._cache_configuration.pop('extra_action_turn_chain')
        if not visited_card_extra_action:
            self._cache_configuration.pop('visited_card_extra_action')
        if visited_card_extra_action_version is None:
            self._cache_configuration.pop('visited_card_extra_action_version')
        if not parallel_public_graph:
            self._cache_configuration.pop('parallel_public_graph')
        if not resource_reanalysis:
            self._cache_configuration.pop('resource_reanalysis')
        if not retained_policy_competition:
            self._cache_configuration.pop('retained_policy_competition')
        if not retained_policy_execution:
            self._cache_configuration.pop('retained_policy_execution')
        if not retained_tree_policy_execution:
            self._cache_configuration.pop('retained_tree_policy_execution')
        if not internal_reanalysis:
            self._cache_configuration.pop('internal_reanalysis')
        if not internal_successor_graph:
            self._cache_configuration.pop('internal_successor_graph')
        if not internal_complete_root_successors:
            self._cache_configuration.pop('internal_complete_root_successors')
        if not final_double_tail_evaluation:
            self._cache_configuration.pop('final_double_tail_evaluation')
        if not final_offhand_remove_tail_evaluation:
            self._cache_configuration.pop('final_offhand_remove_tail_evaluation')
        if not single_node_expansion:
            self._cache_configuration.pop('single_node_expansion')
        if not deterministic_policy_continuation:
            self._cache_configuration.pop('deterministic_policy_continuation')
        if not equivalent_instance_reuse:
            self._cache_configuration.pop('equivalent_instance_reuse')
        if not actual_two_turn_probe_ms:
            self._cache_configuration.pop('actual_two_turn_probe_ms')
        self.kernel, self.baseline, self.deadline = kernel, baseline, deadline
        self.quality_reference = quality_reference
        self.compute_sensitivity = compute_sensitivity
        self.actual_tail_session_reuse = actual_tail_session_reuse
        self.public_turn_budget_reservation = public_turn_budget_reservation
        self.public_validation_phase_budget = public_validation_phase_budget
        self.parallel_public_graph = parallel_public_graph
        self.resource_reanalysis = resource_reanalysis
        self.retained_policy_competition = retained_policy_competition
        self.revealed_policy_competition = revealed_policy_competition
        self.retained_policy_execution = retained_policy_execution
        self.retained_tree_policy_execution = retained_tree_policy_execution
        self.internal_reanalysis = internal_reanalysis
        self.internal_successor_graph = internal_successor_graph
        self.uniform_validation_time_guard = uniform_validation_time_guard
        self.paired_adaptive_validation = paired_adaptive_validation
        self.internal_complete_root_successors = internal_complete_root_successors
        self.final_double_tail_evaluation = final_double_tail_evaluation
        self.final_offhand_remove_tail_evaluation = final_offhand_remove_tail_evaluation
        self._retained_policy = None
        if retained_policy_competition:
            from .hif_retained_policy_competition import RetainedPolicyCompetition, RevealedPolicyCompetition

            self._retained_policy = (RevealedPolicyCompetition if revealed_policy_competition
                                     else RetainedPolicyCompetition)(self)
        self._resource_continuation = None
        if resource_reanalysis:
            from .hif_resource_continuation import ResourcePolicyContinuation

            self._resource_continuation = ResourcePolicyContinuation(kernel)
        self._actual_tail_link = None
        self.window_seconds = window_seconds
        self.terminal_attempt_ms = terminal_attempt_ms
        self.resource_dominance = resource_dominance
        self.double_advice = double_advice
        self.temporal_advice = temporal_advice
        self.drink_advice = drink_advice
        self.retrieval_combo_advice = retrieval_combo_advice
        self.double_return_advice = double_return_weights is not None
        self.simple_tail_evaluation = simple_tail_evaluation
        self.actual_two_turn_evaluation = actual_two_turn_evaluation
        self.actual_two_turn_probe_ms = actual_two_turn_probe_ms
        self.bounded_actual_two_turn_evaluation = bounded_actual_two_turn_evaluation
        self.joint_choice_roots = joint_choice_roots
        self.continuation_samples = continuation_samples
        self.continuation_domain = continuation_domain
        self.continuation_incomplete_actions = continuation_incomplete_actions
        self.fresh_adaptive_training = fresh_adaptive_training
        self.continuation_portfolio = continuation_portfolio
        self.sparse_turn_chain = sparse_turn_chain
        self.stop_after_card_roots = stop_after_card_roots
        self.installed_card_use_value = installed_card_use_value
        self.root_coverage_first = root_coverage_first
        self.guard_unpriced_schedules = guard_unpriced_schedules
        self.guard_unpriced_resources = guard_unpriced_resources
        self.paired_finalist_validation = paired_finalist_validation
        self.public_turn_reanalysis = public_turn_reanalysis
        self.public_turn_policy_continuation = public_turn_policy_continuation
        self.pair_order_advice = pair_order_advice
        self.public_turn_score_translation = public_turn_score_translation
        self._public_turn_continuation = None
        if public_turn_policy_continuation:
            from .hif_public_turn_continuation import PublicTurnPolicyContinuation
            self._public_turn_continuation = PublicTurnPolicyContinuation(kernel)
        self.direct_double_marginal = direct_double_marginal
        self.unknown_node_timing = unknown_node_timing
        self.partial_node_timing = partial_node_timing
        self.installed_score_turn_chain = installed_score_turn_chain
        self.installed_score_multi_action = installed_score_multi_action
        self.installed_score_focus_projection = installed_score_focus_projection
        self.extra_action_turn_chain = extra_action_turn_chain
        self.visited_card_extra_action = visited_card_extra_action
        self.visited_card_extra_action_version = visited_card_extra_action_version
        self.retrieval_training_order = retrieval_training_order
        self.lazy_validation_seeds = lazy_validation_seeds
        self.deterministic_validation_cohort = deterministic_validation_cohort
        self._validation_cohort = None
        if deterministic_validation_cohort:
            from .hif_validation_cohort import DeterministicValidationCohort
            self._validation_cohort = DeterministicValidationCohort(kernel)
        self.single_node_expansion = single_node_expansion
        self.deterministic_policy_continuation = deterministic_policy_continuation
        core = getattr(baseline, 'baseline', baseline)
        self.endgame = core.endgame
        self.tail = (HifActionChainPlanner(kernel, PlannerConfig(
            max_decision_depth=64, timeout_ms=terminal_attempt_ms), complete_horizon=True)
            if complete_tail_evaluation or frozen_tail_evaluation or simple_tail_evaluation else None)
        self.proven_tail = HifProvenTailTable(kernel, self.tail) if frozen_tail_evaluation else None
        advice = core.advice
        if temporal_advice:
            advice = HifTemporalAdvice(kernel, advice)
        elif double_advice:
            if self.double_return_advice:
                from .hif_double_return_advice import HifDoubleReturnAdvice
                advice = HifDoubleReturnAdvice(kernel, HifDoubleAdvice(kernel, advice),
                    weights=double_return_weights, max_turns_remaining=double_return_horizon)
            else:
                advice = HifDoubleAdvice(kernel, advice,
                    guard_unpriced_schedules=guard_unpriced_schedules,
                    guard_unpriced_resources=guard_unpriced_resources)
        if direct_double_marginal:
            from .hif_direct_double_advice import VERSION as DIRECT_DOUBLE_VERSION
            from .hif_direct_double_advice import HifDirectDoubleAdvice
            advice = HifDirectDoubleAdvice(advice)
            self.direct_double_marginal_version = DIRECT_DOUBLE_VERSION
        self.double_ordering = advice.double if temporal_advice else advice
        if drink_advice:
            advice = HifDrinkAdvice(kernel, advice)
        if retrieval_combo_advice:
            advice = HifRetrievalComboAdvice(kernel, advice)
        turn_chain = None
        if sparse_turn_chain:
            from .hif_turn_chain_continuation import TurnChainContinuation
            turn_chain = TurnChainContinuation(kernel, installed_card_use_value=installed_card_use_value)
        score_translation_contract = None
        if public_turn_reanalysis and not installed_score_turn_chain:
            from .hif_learning import HifFeatureEncoder, HifLearningAdvice
            # Only this audited advice/tail combination is independent of
            # accumulated score. Unknown advice retains distinct boundary work.
            if (type(core.advice) is HifLearningAdvice
                    and type(core.advice.encoder) is HifFeatureEncoder
                    and simple_tail_evaluation and simple_final_turn_evaluation
                    and not any((temporal_advice, drink_advice, retrieval_combo_advice,
                                 self.double_return_advice, frozen_tail_evaluation))):
                score_translation_contract = 'hif-score-translation-v1'
        self.search = OfflineHifMcts(
            kernel, advice=advice, simulations=simulations,
            timeout_ms=window_seconds * 1000, seed=planning_seed,
            min_root_samples=min_root_samples, reward_scale=CONFIG['reward_scale'],
            terminal_evaluator=self._terminal, sampler_factory=PairedPlanningKernel,
            soft_timeout_ms=soft_window_seconds * 1000, scenario_reuse=scenario_reuse,
            quality_reference=quality_reference,
            revealed_policy_competition=False,
            compute_sensitivity=compute_sensitivity,
            validation_samples=validation_samples, dominance_pruning=resource_dominance,
            adaptive_validation=adaptive_validation, progressive_widening=progressive_widening,
            complete_orderer=HifExactTailOrdering(kernel) if exact_tail_ordering else None,
            terminal_horizon=2 if self.tail is not None else 1,
            validation_terminal_evaluator=(self._validated_terminal
                if complete_tail_evaluation or simple_tail_evaluation else None),
            validation_tail_lookup=self.proven_tail.lookup if self.proven_tail is not None else None,
            freeze_callback=self.proven_tail.freeze if self.proven_tail is not None else None,
            terminal_eligible=self._tail_eligible if simple_tail_evaluation else None,
            terminal_eligibility_horizons=(1, 2) if simple_final_turn_evaluation else (2,),
            continuation_samples=continuation_samples, continuation_domain=continuation_domain,
            continuation_incomplete_actions=continuation_incomplete_actions,
            fresh_adaptive_training=(fresh_adaptive_training
                and not (quality_reference and internal_reanalysis)),
            continuation_policy_factory=(lambda: SenseComboTimingPolicy(kernel, baseline))
                if continuation_portfolio else None,
            turn_chain_continuation=turn_chain, stop_after_card_roots=stop_after_card_roots,
            root_coverage_first=root_coverage_first, single_node_expansion=single_node_expansion,
            equivalent_instance_reuse=equivalent_instance_reuse,
            paired_finalist_validation=paired_finalist_validation,
            paired_adaptive_validation=paired_adaptive_validation,
            uniform_validation_time_guard=uniform_validation_time_guard,
            public_turn_reanalysis=public_turn_reanalysis,
            resource_reanalysis=False,
            public_turn_budget_reservation=public_turn_budget_reservation,
            public_validation_phase_budget=public_validation_phase_budget,
            parallel_public_graph=parallel_public_graph,
            unknown_node_policy=UnknownNodeTimingPolicy(kernel, baseline,
                installed_score_turn_chain=installed_score_turn_chain,
                installed_score_multi_action=installed_score_multi_action,
                installed_score_focus_projection=installed_score_focus_projection,
                extra_action_turn_chain=extra_action_turn_chain,
                visited_card_extra_action=visited_card_extra_action,
                visited_card_extra_action_version=visited_card_extra_action_version,
                pair_order_advice=pair_order_advice) if unknown_node_timing else None,
            retrieval_training_orderer=(RetrievalTrainingOrder(kernel, baseline)
                if retrieval_training_order else None),
            lazy_validation_seeds=lazy_validation_seeds,
            deterministic_validation_cohort=deterministic_validation_cohort,
            partial_node_timing=partial_node_timing,
            score_translation_contract=score_translation_contract)
        if direct_double_marginal:
            # The raw search cache must also isolate this advice contract when
            # its caller supplies an otherwise identical external context.
            self.search.direct_double_marginal_version = self.direct_double_marginal_version
        if self.public_turn_score_translation:
            # Raw search constructors cannot claim this expanded contract.
            # Grant it only after the concrete candidate chain is constructed.
            self.search.score_translation_contract = self._public_score_translation_contract()
            self.search._public_step_reuse_check = self._public_score_translation_contract
        # Local resource tables require this candidate's verified score
        # contract, which depends on the constructed search callbacks.
        self.search.resource_reanalysis = resource_reanalysis
        self.search.revealed_policy_competition = revealed_policy_competition
        self.search.internal_reanalysis = internal_reanalysis
        self.search.internal_successor_graph = internal_successor_graph
        self.search.internal_complete_root_successors = internal_complete_root_successors
        # The quality/fresh combination has the same complete-stack boundary;
        # enable it only after the bound callback contract above is verified.
        self.search.fresh_adaptive_training = fresh_adaptive_training
        self.reset()

    def _public_score_translation_contract(self):
        """Whitelist the current candidate, not arbitrary score-aware adapters."""
        from .hif import HifSenseKernel
        from .hif_budget import HifSearchBudget, TieredHifPlanner
        from .hif_hybrid import HybridHifPolicy, VisibleHifCandidatePolicy
        from .hif_learning import HifFeatureEncoder, HifLearningAdvice
        from .hif_teachers import VisibleHifContinuationWrapper
        from .hif_double_advice import SCOPE_VERSION, COMBINED_SCOPE_VERSION, RESOURCE_SCOPE_VERSION
        from .hif_public_turn_reanalysis import SCORE_TRANSLATION_CONTRACT

        def bound(instance, name, owner):
            method = getattr(instance, name, None)
            return (getattr(method, '__self__', None) is instance
                    and getattr(method, '__func__', None) is getattr(owner, name))

        baseline = self.baseline
        if (type(self) is not HifMctsCandidatePolicy or type(self.kernel) is not HifSenseKernel
                or type(baseline) is not VisibleHifCandidatePolicy
                or baseline.wrapper_type is not VisibleHifContinuationWrapper
                or baseline.wrapper_type.version != 'visible-hif-continuation-dev-v1'
                or not bound(baseline, 'decide', VisibleHifCandidatePolicy)):
            raise ValueError('mcts_public_score_translation_unknown_baseline')
        core = baseline.baseline
        if (type(core) is not HybridHifPolicy or not bound(core, 'decide', HybridHifPolicy)
                or baseline.kernel is not self.kernel or core.kernel is not self.kernel):
            raise ValueError('mcts_public_score_translation_unknown_baseline')
        advice = core.advice
        if (type(advice) is not HifLearningAdvice or type(advice.encoder) is not HifFeatureEncoder
                or advice.model_version != 'learning-r1-ppo-27502-v1'
                or advice.encoder.encoder_version != '1.1' or advice.encoder.kernel is not self.kernel
                or not bound(advice, 'action_priors', HifLearningAdvice)
                or not bound(advice.encoder, 'state_features', HifFeatureEncoder)
                or not bound(advice.encoder, 'action_features', HifFeatureEncoder)):
            raise ValueError('mcts_public_score_translation_unknown_advice')
        ordering = self.search.advice
        expected_version = (COMBINED_SCOPE_VERSION if self.guard_unpriced_schedules and self.guard_unpriced_resources
                            else SCOPE_VERSION if self.guard_unpriced_schedules
                            else RESOURCE_SCOPE_VERSION if self.guard_unpriced_resources else DOUBLE_ADVICE_VERSION)
        if (type(ordering) is not HifDoubleAdvice or ordering is not self.double_ordering
                or ordering.old_advice is not advice or ordering.kernel is not self.kernel
                or ordering.model_version != expected_version
                or not bound(ordering, 'action_priors', HifDoubleAdvice)):
            raise ValueError('mcts_public_score_translation_unknown_advice')
        endgame = core.endgame
        if (self.endgame is not endgame or type(endgame) not in (HifActionChainPlanner, TieredHifPlanner)
                or not bound(endgame, 'decide', type(endgame))
                or type(endgame) is TieredHifPlanner and type(endgame.budget) is not HifSearchBudget
                or type(self.tail) is not HifActionChainPlanner
                or not bound(self.tail, 'decide', HifActionChainPlanner)
                or not self.tail.complete_horizon
                or any(planner.kernel is not self.kernel or planner.advice is not None
                       or planner.use_learned_value or planner.planner_version != '1.0'
                       for planner in (endgame, self.tail))
                or not all(bound(self, name, HifMctsCandidatePolicy)
                           for name in ('_terminal', '_validated_terminal', '_tail_eligible'))
                or self.search.terminal_evaluator != self._terminal
                or self.search.validation_terminal_evaluator != self._validated_terminal
                or self.search.terminal_eligible != self._tail_eligible
                or self.search.validation_tail_lookup is not None
                or self.search.sampler_factory is not PairedPlanningKernel):
            raise ValueError('mcts_public_score_translation_unknown_terminal')
        return SCORE_TRANSLATION_CONTRACT

    def _tail_eligible(self, request):
        return assess_simple_tail_domain(self.kernel, request.state,
            allow_final_double=self.final_double_tail_evaluation,
            allow_final_offhand_remove=self.final_offhand_remove_tail_evaluation).eligible

    def reset(self):
        self._clear_actual_tail_link()
        if self._retained_policy is not None:
            self._retained_policy.reset()
        self.search.reset()
        if self._resource_continuation is not None:
            self._resource_continuation.reset()
        if self._validation_cohort is not None:
            self._validation_cohort.reset()
        if self._public_turn_continuation is not None:
            self._public_turn_continuation.reset()
        self.plans, self.interventions = [], []
        self._plan_offset = 0
        self._strong_work_totals = {}
        self._strong_last_window_work = None
        self._choice_continuation = None
        self._frozen_policy_continuation = None
        self._session = None
        self._tail_session = None
        if self.proven_tail is not None:
            self.proven_tail.reset_window()
        if self.direct_double_marginal:
            self.double_ordering.reset_window()

    def window_deadline(self, started):
        stop = started + self.window_seconds
        return stop if self.deadline is None else min(stop, self.deadline)

    def _cache_context(self):
        from .hif_mcts_cache import json_value

        if self.public_turn_score_translation:
            if self.search.score_translation_contract != self._public_score_translation_contract():
                raise ValueError('mcts_public_score_translation_contract_changed')
        core = getattr(self.baseline, 'baseline', self.baseline)
        advice = core.advice
        encoder = getattr(advice, 'encoder', None)
        if (not isinstance(getattr(advice, 'model_version', None), str)
                or encoder is None or not hasattr(advice, 'weights')):
            raise ValueError('mcts_cache_model_definition_missing')
        weights = getattr(advice, 'value_weights', None)
        return json_value(dict(strategy_version=VERSION, configuration=self._cache_configuration,
            model=dict(model_version=advice.model_version,
                encoder_name=encoder.encoder_name, encoder_version=encoder.encoder_version,
                feature_names=encoder.feature_names, effective_weights=advice.weights.tolist()),
            value_weights=None if weights is None else weights.tolist(),
            baseline_type=type(self.baseline).__qualname__,
            wrapper_version=getattr(getattr(self.baseline, 'wrapper_type', None), 'version', None),
            advice_versions=({**CONFIG, 'double_advice_version': self.double_ordering.model_version}
                             if self.guard_unpriced_schedules or self.guard_unpriced_resources
                             or self.direct_double_marginal else CONFIG),
            **({'unknown_node_timing': dict(version=self.search.unknown_node_policy.version,
                timing_version=self.search.unknown_node_policy.timing_version)}
                if self.unknown_node_timing else {}),
            **({'partial_node_timing_version': PARTIAL_NODE_TIMING_VERSION}
                if self.partial_node_timing else {}),
            **({'public_turn_reanalysis_version': self.search._public_turn_reanalysis_version}
                if self.public_turn_reanalysis else {}),
            **({'public_turn_optional_end_roots_version': PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION}
                if self.public_turn_reanalysis and self.stop_after_card_roots else {}),
            **({'resource_reanalysis_version': self.search._resource_reanalysis_version,
                'resource_policy_continuation_version': self._resource_continuation.version}
                if self.resource_reanalysis else {}),
            **({'retained_policy_competition_version': self._retained_policy.version}
                if self.retained_policy_competition else {}),
            **({'revealed_policy_competition_version': REVEALED_VERSION}
                if self.revealed_policy_competition else {}),
            **({'retained_policy_execution_version': 'hif-retained-policy-execution-v1'}
                if self.retained_policy_execution else {}),
            **({'retained_tree_policy_execution_version': TREE_EXECUTION_VERSION}
                if self.retained_tree_policy_execution else {}),
            **({'internal_successor_graph_version': self.search._internal_successor_graph_version}
                if self.internal_successor_graph else {}),
            **({'uniform_validation_time_guard_version': self.search._uniform_validation_time_guard_version}
                if self.uniform_validation_time_guard else {}),
            **({'paired_adaptive_validation_version': self.search._paired_adaptive_validation_version}
                if self.paired_adaptive_validation else {}),
            **({'internal_reanalysis_version': INTERNAL_REANALYSIS_VERSION}
                if self.internal_reanalysis else {}),
            **({'internal_complete_root_successors_version': COMPLETE_ROOT_SUCCESSORS_VERSION}
                if self.internal_complete_root_successors else {}),
            **({'quality_reference_version': QUALITY_REFERENCE_VERSION}
                if self.quality_reference else {}),
            **({'compute_sensitivity_version': COMPUTE_SENSITIVITY_VERSION}
                if self.compute_sensitivity else {}),
            **({'final_double_tail_domain_version': FINAL_DOUBLE_VERSION}
                if self.final_double_tail_evaluation else {}),
            **({'final_offhand_remove_tail_domain_version': FINAL_OFFHAND_REMOVE_VERSION}
                if self.final_offhand_remove_tail_evaluation else {}),
            **({'public_turn_policy_continuation_version': self._public_turn_continuation.version}
                if self.public_turn_policy_continuation else {}),
            **({'installed_score_turn_chain_version': self.search.unknown_node_policy.installed_score_policy.version}
                if self.installed_score_turn_chain else {}),
            **({'extra_action_turn_chain_version': self.search.unknown_node_policy.extra_action_policy.version}
                if self.extra_action_turn_chain else {}),
            **({'pair_order_advice_version': self.search.unknown_node_policy.pair_order_policy.version}
                if self.pair_order_advice else {}),
            **({'public_turn_score_translation_version': PUBLIC_TURN_SCORE_TRANSLATION_VERSION}
                if self.public_turn_score_translation else {}),
            **({'actual_tail_session_reuse_version': ACTUAL_TAIL_SESSION_REUSE_VERSION}
                if self.actual_tail_session_reuse else {}),
            **({'public_turn_budget_reservation_version': PUBLIC_TURN_BUDGET_RESERVATION_VERSION}
                if self.public_turn_budget_reservation else {}),
            **({'public_validation_phase_budget_version': PUBLIC_VALIDATION_PHASE_BUDGET_VERSION}
                if self.public_validation_phase_budget else {}),
            **({'parallel_root_validation_version': self.search._parallel_root_validation_version}
                if self.parallel_public_graph and (self.paired_adaptive_validation or self.uniform_validation_time_guard) else {}),
            **({'parallel_public_graph_version': self.search._parallel_public_graph_version}
                if self.parallel_public_graph else {}),
            **({'retrieval_training_order': dict(version=self.search.retrieval_training_orderer.version,
                unknown_version=self.search.retrieval_training_orderer.unknown_version,
                timing_version=self.search.retrieval_training_orderer.timing_version)}
                if self.retrieval_training_order else {}),
            **({'lazy_validation_seeds_version': LAZY_VALIDATION_SEEDS_VERSION}
                if self.lazy_validation_seeds else {})))

    def _clear_actual_tail_link(self):
        saved, self._actual_tail_link = self._actual_tail_link, None
        if saved is not None:
            saved['session'].clear()

    def _actual_tail_context(self, request):
        # Slot order may change for cards; only values are retained and roots
        # are rebound. Bottle availability, utility and rule context must agree.
        return (self._cache_context(), request.plan_type, request.decision_kind,
                request.utility, request.ruleset_id, tuple(request.drink_slots))

    def _prepare_actual_tail_session(self, request, plan, deadline):
        if not self.actual_tail_session_reuse:
            return None, None
        plan.update(actual_tail_session_reuse_enabled=True,
                    actual_tail_session_reuse_version=ACTUAL_TAIL_SESSION_REUSE_VERSION)
        evidence = dict(reused=False, source_plan_index=None, entries_before=0, entries_after=0)
        plan['actual_tail_session_reuse'] = evidence
        saved, self._actual_tail_link = self._actual_tail_link, None
        if request.state.turns_remaining != 2:
            if saved is not None:
                saved['session'].clear()
            return None, None
        try:
            context = self._actual_tail_context(request)
        except Exception:
            if saved is not None:
                saved['session'].clear()
            raise
        if (saved is not None and saved['committed']
                and saved['source_plan_index'] == self._plan_offset + len(self.plans) - 2
                and saved['context'] == context
                and saved['after'] == self.kernel._canonical_probability_state(request.state)
                and saved['session'].entry_count):
            evidence.update(reused=True, source_plan_index=saved['source_plan_index'],
                            entries_before=saved['session'].entry_count)
            return saved['session'], context
        if saved is not None:
            saved['session'].clear()
        return HifSearchSession(deadline), context

    def _export_cache_choice(self, request):
        choice = self._choice_continuation
        if choice is not None:
            if choice['context'] != self.search._context(request):
                raise ValueError('mcts_cache_choice_context_changed')
            choice = {key: value for key, value in choice.items() if key != 'context'}
            choice = {key: asdict(value) if key in ('before', 'expected_state', 'first', 'target')
                      else value for key, value in choice.items()}
        return choice

    def _restore_cache_choice(self, choice, request, plan_count):
        from .hif_mcts_cache import exact_state, slots_from_json, action_from_json

        if choice is not None:
            if (set(choice) != {'before', 'first', 'expected_state', 'expected_drink_slots',
                    'target', 'source_plan_index', 'committed'}
                    or type(choice['committed']) is not bool
                    or type(choice['source_plan_index']) is not int
                    or not 0 <= choice['source_plan_index'] < plan_count):
                raise ValueError('mcts_cache_choice_invalid')
            choice = dict(choice, before=exact_state(choice['before']),
                          expected_state=exact_state(choice['expected_state']),
                          first=action_from_json(choice['first']), target=action_from_json(choice['target']))
            choice['expected_drink_slots'] = slots_from_json(
                choice['expected_drink_slots'], choice['expected_state'])
            expected = choice['expected_state'] if choice['committed'] else choice['before']
            optional_end = choice['target'].kind.value == 'END_TURN'
            if (expected != self.kernel._canonical_probability_state(request.state)
                    or (choice['expected_state'].pending_choice is None and not optional_end)
                    or choice['first'].source_slot is not None or choice['target'].source_slot is not None):
                raise ValueError('mcts_cache_choice_state_changed')
            if optional_end:
                deadline = self.window_deadline(time.perf_counter()) if self.deadline is None else self.deadline
                def check_budget():
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('mcts_window_timeout')
                with validation_session(check_budget):
                    check_budget()
                    proven, _ = post_card_end_state(self.kernel, choice['before'], choice['first'],
                                                    plan_type=request.plan_type)
                    check_budget()
                if (not self.stop_after_card_roots or proven is None
                        or proven != choice['expected_state']
                        or choice['target'] != GameAction(ActionKind.END_TURN)):
                    raise ValueError('mcts_cache_optional_end_not_verified')
        if choice is not None:
            choice['context'] = self.search._context(request)
        return choice

    def export_search_cache(self, request, *, complete_policy=False):
        from .hif_mcts_cache import json_value

        if complete_policy:
            from .hif_complete_candidate_cache import export_complete

            return export_complete(self, request)
        if self._retained_policy is not None and self._retained_policy.frozen is not None:
            raise ValueError('retained_policy_in_process_cache_only')
        if self._session is not None or self._tail_session is not None:
            raise ValueError('mcts_cache_window_still_running')
        payload = dict(search=self.search.export_search_cache(
            request, version_context=self._cache_context()), choice=self._export_cache_choice(request),
            plan_count=self._plan_offset+len(self.plans))
        if self.deterministic_policy_continuation:
            payload['frozen_policy_continuation'] = export_link(self._frozen_policy_continuation)
        return json_value(payload)

    def restore_search_cache(self, payload, request, *, complete_policy=False):
        try:
            from .hif_mcts_cache import require_candidate_tree_versions

            require_candidate_tree_versions(payload, self.search, complete_policy=complete_policy)
            if complete_policy:
                from .hif_complete_candidate_cache import restore_complete

                return restore_complete(self, payload, request)
            return self._restore_search_cache(payload, request)
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            raise ValueError('mcts_cache_candidate_malformed') from error

    def _restore_search_cache(self, payload, request):
        from .hif_mcts_cache import json_value

        payload = json_value(payload)
        fields = {'search', 'choice', 'plan_count'}
        if self.deterministic_policy_continuation:
            fields.add('frozen_policy_continuation')
        if (set(payload) != fields
                or type(payload['plan_count']) is not int or payload['plan_count'] < 0
                or self.plans or self._plan_offset or self._choice_continuation is not None
                or self._frozen_policy_continuation is not None):
            raise ValueError('mcts_cache_candidate_progress_invalid')
        choice = self._restore_cache_choice(payload['choice'], request, payload['plan_count'])
        frozen = (restore_link(payload['frozen_policy_continuation'], self.kernel, request,
            context=self.search._context(request), plan_count=payload['plan_count'],
            deadline=self.window_deadline(time.perf_counter()) if self.deadline is None else self.deadline)
            if self.deterministic_policy_continuation else None)
        result = self.search.restore_search_cache(payload['search'], request,
                                                  version_context=self._cache_context())
        self._choice_continuation = choice
        self._frozen_policy_continuation = frozen
        self._plan_offset = payload['plan_count']
        return result

    def _resume_frozen_policy(self, request, legal, plan, deadline):
        saved = self._frozen_policy_continuation
        _complete_legal(self.kernel, request, legal)
        self.search._order_deadline = deadline
        key = self.search._validation_action(request, legal)
        matches = [action for action in legal if action_key(action) == key]
        if len(matches) != 1:
            raise ValueError('mcts_frozen_policy_action_not_uniquely_legal')
        proof = saved['proof']
        plan.update(scope='FROZEN_POLICY_CONTINUATION', stop_reason='matched_deterministic_frozen_policy',
            requested_simulations=0, completed_simulations=0, candidate_statistics=[],
            complete_root_coverage=False, source_plan_index=proof['source_plan_index'],
            predecessor_plan_index=proof['predecessor_plan_index'], continuation_proof=export_link(saved)['proof'])
        return matches[0]

    def _remember_frozen_policy(self, request, action, plan, deadline):
        index = self._plan_offset + len(self.plans) - 1
        source = dict(plan, accepted=True)
        if qualified_source(source):
            source_index = index
        elif self._frozen_policy_continuation is not None:
            source = self._frozen_policy_continuation['source_plan']
            source_index = self._frozen_policy_continuation['proof']['source_plan_index']
            plan['source_plan_index'] = source_index
        else:
            return
        self._frozen_policy_continuation = make_link(self.kernel, request, action,
            source_plan=source, source_plan_index=source_index, predecessor_plan_index=index,
            context=self.search._context(request), deadline=deadline)

    def _resume_choice(self, request, legal, plan):
        saved, self._choice_continuation = self._choice_continuation, None
        if saved is None:
            return None
        optional_end = saved['target'].kind.value == 'END_TURN'
        if (not saved['committed']
                or (request.state.pending_choice is None and not optional_end)
                or (optional_end and (not self.stop_after_card_roots
                    or request.state.pending_choice is not None or request.state.resolution_stack
                    or request.state.actions_remaining <= 0 or request.state.terminal))
                or self.kernel._canonical_probability_state(request.state) != saved['expected_state']
                or tuple(request.drink_slots) != saved['expected_drink_slots']
                or self.search._context(request) != saved['context']):
            plan['joint_continuation_discarded'] = 'public_state_or_context_changed'
            return None
        matches = [action for action in legal if action_key(action) == action_key(saved['target'])]
        if len(matches) != 1:
            plan['joint_continuation_discarded'] = 'target_not_uniquely_legal'
            return None
        # This only reuses a business choice. The returned action is supplied
        # by this request's complete legal bindings, never an old UI slot.
        plan.update(scope='JOINT_ROOT_CONTINUATION', complete_root_coverage=False,
            complete_joint_root_coverage=True, source_plan_index=saved['source_plan_index'],
            stop_reason=('matched_public_optional_end' if optional_end else 'matched_public_mandatory_choice'),
            requested_simulations=0,
            completed_simulations=0, candidate_statistics=[])
        return matches[0]

    def _remember_choice(self, request, action, result):
        if not result.selected_continuation:
            return
        expected = result.selected_expected_after_first
        optional_end = (len(result.selected_continuation) == 1
                        and result.selected_continuation[0].kind.value == 'END_TURN')
        if (len(result.selected_continuation) != 1 or expected is None
                or (expected.pending_choice is None and not optional_end)
                or (optional_end and (not self.stop_after_card_roots or expected.pending_choice is not None
                    or expected.resolution_stack or expected.actions_remaining <= 0 or expected.terminal
                    or action.kind.value != 'PLAY_CARD' or expected.turn != request.state.turn))):
            raise ValueError('mcts_joint_root_continuation_invalid')
        slots = tuple(request.drink_slots)
        if action.kind.value == 'USE_DRINK':
            slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
        self._choice_continuation = dict(
            before=self.kernel._canonical_probability_state(request.state),
            first=replace(action, source_slot=None),
            expected_state=self.kernel._canonical_probability_state(expected),
            expected_drink_slots=slots, target=replace(result.selected_continuation[0], source_slot=None),
            context=self.search._context(request),
            source_plan_index=self._plan_offset+len(self.plans)-1, committed=False)

    def _terminal(self, request, deadline):
        # The surrounding session propagates this outer deadline into all
        # nested exact decisions and shares only complete terminal subtrees.
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_window_timeout')
        # Tiered planners reset their local timeout on each call. The shared
        # session's deadline also bounds those planners without losing its
        # completed subtree tables or mutating their normal configuration.
        planner, session = ((self.tail, self._tail_session) if request.state.turns_remaining > 1
                            else (self.endgame, self._session))
        if self.quality_reference:
            return self._quality_terminal(request, deadline, planner, session)
        original = session.deadline
        session.deadline = min(original, deadline, time.perf_counter() + self.terminal_attempt_ms / 1000)
        try:
            response = planner.decide(request)
            if (self.proven_tail is not None and request.state.turns_remaining == 2
                    and time.perf_counter() < session.deadline):
                self.proven_tail.remember(request, response)
            return response
        finally:
            session.deadline = original

    def _quality_terminal(self, request, deadline, planner, session):
        """Complete the unchanged exact solver under the window protection only."""
        original_deadline, original_config = session.deadline, planner.config
        original_budget = getattr(planner, 'budget', None)
        limit = min(original_deadline, deadline)
        remaining = (limit - time.perf_counter()) * 1000
        if remaining <= 0:
            raise TimeoutError('mcts_quality_reference_terminal_timeout')
        session.deadline = limit
        planner.config = replace(original_config, timeout_ms=remaining)
        if original_budget is not None:
            # TieredHifPlanner resets config in decide, so changing config alone
            # would silently keep its normal/final 3/15-second cutoffs.
            planner.budget = replace(original_budget,
                target_ms=min(original_budget.target_ms, remaining),
                normal_ms=remaining, slow_ms=remaining)
        try:
            response = planner.decide(request)
            if (time.perf_counter() >= limit
                    or response.reason == 'search_incomplete:planner_timeout'):
                raise TimeoutError('mcts_quality_reference_terminal_timeout')
            if self.proven_tail is not None and request.state.turns_remaining == 2:
                self.proven_tail.remember(request, response)
            return response
        finally:
            session.deadline = original_deadline
            planner.config = original_config
            if original_budget is not None:
                planner.budget = original_budget

    def _validated_terminal(self, request, deadline):
        if self.quality_reference:
            return self._terminal(request, deadline)
        if request.state.turns_remaining == 1:
            if self.simple_tail_evaluation:
                # A mandatory complete validation suffix must not inherit
                # the 500 ms exploratory probe ceiling. The existing session
                # still enforces the window; a tiered planner keeps its own
                # (potentially smaller) final-turn limit as well.
                limit = min(deadline, self._session.deadline)
                remaining = (limit - time.perf_counter()) * 1000
                if remaining <= 0:
                    raise TimeoutError('mcts_window_timeout')
                original = self.endgame.config
                self.endgame.config = replace(original, timeout_ms=remaining)
                try:
                    response = self.endgame.decide(request)
                    self._record_terminal_failure(request, response)
                    return response
                finally:
                    self.endgame.config = original
            return self._terminal(request, deadline)
        # Training probes remain bounded by terminal_attempt_ms. Mandatory
        # validation can finish its exact proof using the already allocated
        # window; it cannot extend that window, resample, or change policy.
        started = time.perf_counter()
        limit = min(deadline, self._tail_session.deadline)
        if started >= limit:
            raise TimeoutError('mcts_window_timeout')
        original = self.tail.config
        self.tail.config = replace(original, timeout_ms=(limit - started) * 1000)
        try:
            response = self.tail.decide(request)
            self._record_terminal_failure(request, response)
            return response
        finally:
            self.tail.config = original

    def _record_terminal_failure(self, request, response):
        if not response.has_action_intent and self.plans:
            # Preserve the actual failing public state for diagnosis, instead
            # of trying to reconstruct a time-bounded tree in another run.
            self.plans[-1]['terminal_validation_failure'] = dict(
                request=asdict(request), reason=response.reason,
                value_scope=response.trace.value_scope if response.trace else None)

    def _actual_two_turn(self, request, legal, deadline, plan):
        """Prove the actual root with the existing complete-horizon planner.

        Hypothetical tail admission and exploratory 500 ms probes are unchanged.
        Exact representative roots use the planner's established equivalence;
        they are not claimed to be separately sampled physical duplicates.
        """
        plan.update(scope='FINAL_SCORE', exact_root_turns=2,
                    requested_simulations=0, completed_simulations=0, candidate_statistics=[])
        _complete_legal(self.kernel, request, legal)
        response = self._validated_terminal(request, deadline)
        return self._accept_actual_two_turn(request, legal, deadline, plan, response)

    def _accept_actual_two_turn(self, request, legal, deadline, plan, response):
        plan['stop_reason'] = response.reason
        if (not response.has_action_intent or response.value is None or response.trace is None
                or response.trace.value_scope != 'FINAL_SCORE'):
            raise ValueError('mcts_actual_two_turn_search_incomplete:' + response.reason)
        raw_legal = tuple(replace(action, source_slot=None) for action in legal)
        expected = self.tail._merge_equivalent_actions(request.state, raw_legal)
        candidates = response.trace.candidate_values
        if (len(candidates) != len(expected)
                or {row.action for row in candidates} != set(expected)
                or any(not math.isfinite(row.probability_mass)
                       or not math.isclose(row.probability_mass, 1., rel_tol=1e-9, abs_tol=1e-9)
                       for row in candidates)
                or any(not all(math.isfinite(getattr(value, name)) for name in (
                    'expected_final_score', 'p10_final_score', 'worst_final_score', 'best_final_score'))
                    or value.outcome_count < 1 for value in (response.value, *(row.value for row in candidates)))):
            raise ValueError('mcts_actual_two_turn_root_values_incomplete')
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_window_timeout')
        plan.update(complete_root_coverage=True, exact_root_value_scope=response.trace.value_scope,
                    exact_root_candidates=[asdict(value) for value in candidates],
                    exact_root_value=asdict(response.value), exact_root_trace=asdict(response.trace))
        return response.action

    def _probe_actual_two_turn(self, request, legal, deadline, plan):
        """Try a complete root proof before sampling, without ranking partials.

        Unlike mandatory exact admission, this opt-in work allocation reserves
        at least 75% of the remaining hard window for the existing search. Only
        a computational timeout may return to sampling; invalid proofs stop.
        Completed session subtrees can still be reused by later tail queries.
        """
        if not self.actual_two_turn_probe_ms or request.state.turns_remaining != 2:
            return None
        started = time.perf_counter()
        if started >= deadline:
            raise TimeoutError('mcts_window_timeout')
        _complete_legal(self.kernel, request, legal)
        budget_ms = ((deadline - started) * 1000 if self.quality_reference else
                     min(self.actual_two_turn_probe_ms, (deadline - started) * 250.))
        original = self.tail.config
        self.tail.config = replace(original, timeout_ms=budget_ms)
        try:
            response = self.tail.decide(request)
        finally:
            self.tail.config = original
        plan['actual_two_turn_probe'] = dict(budget_ms=budget_ms,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            reason=response.reason, complete=response.has_action_intent,
            partial_values_used=False,
            expanded_nodes=response.trace.expanded_nodes if response.trace else 0)
        if (not response.has_action_intent
                and response.reason == 'search_incomplete:planner_timeout'):
            if self.quality_reference or time.perf_counter() >= deadline:
                raise TimeoutError('mcts_window_timeout')
            return None
        plan.update(scope='FINAL_SCORE', exact_root_turns=2,
                    requested_simulations=0, completed_simulations=0, candidate_statistics=[])
        return self._accept_actual_two_turn(request, legal, deadline, plan, response)

    def _actual_two_turn_eligible(self, request, plan):
        if not self.actual_two_turn_evaluation or request.state.turns_remaining != 2:
            return False
        if not self.bounded_actual_two_turn_evaluation:
            return True
        domain = assess_simple_tail_domain(self.kernel, request.state)
        plan['actual_two_turn_domain'] = asdict(domain)
        plan['actual_two_turn_domain_version'] = SIMPLE_TAIL_VERSION
        # Choose the bounded work path before starting any exact proof. Once
        # admitted, an incomplete proof still stops in _actual_two_turn.
        return domain.eligible

    def choose(self, request, legal):
        if self.quality_reference:
            self.search.last_search_phase_timing = None
            self.search._search_phase_started = None
        try:
            request.validate()
            if self.public_turn_score_translation:
                if (request.utility.baseline_p10 is not None
                        or self.search.score_translation_contract != self._public_score_translation_contract()):
                    raise ValueError('mcts_public_score_translation_request_outside_contract')
            if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
                raise ValueError('mcts_candidate_capability_or_ruleset_not_ready')
        except Exception:
            self._clear_actual_tail_link()
            if self._retained_policy is not None:
                self._retained_policy.reset()
            if self._validation_cohort is not None:
                self._validation_cohort.reset()
            if self._public_turn_continuation is not None:
                self._public_turn_continuation.reset()
            if self._resource_continuation is not None:
                self._resource_continuation.reset()
            raise
        started = time.perf_counter()
        deadline = self.window_deadline(started)
        search_deadline = deadline - min(.1, self.window_seconds * .05)
        plan = dict(turn=request.state.turn, accepted=False, complete_root_coverage=False,
                    scope='SAMPLING_ESTIMATE', optimality_proven=False,
                    execution_authorized=False)
        if self.quality_reference:
            plan.update(quality_reference_enabled=True,
                        quality_reference_version=QUALITY_REFERENCE_VERSION,
                        quality_reference_watchdog_seconds=self.window_seconds)
        if self.compute_sensitivity:
            plan.update(compute_sensitivity_enabled=True,
                        compute_sensitivity_version=COMPUTE_SENSITIVITY_VERSION,
                        compute_sensitivity_simulations=self.search.simulations)
        if self.sparse_turn_chain:
            plan['sparse_turn_chain'] = True
        if self.stop_after_card_roots:
            plan['stop_after_card_roots'] = True
        if self.installed_card_use_value:
            plan['installed_card_use_value'] = True
        if self.root_coverage_first:
            plan['root_coverage_first'] = True
        if self.guard_unpriced_schedules:
            plan['guard_unpriced_schedules'] = True
        if self.guard_unpriced_resources:
            plan['guard_unpriced_resources'] = True
        if self.paired_finalist_validation:
            plan['paired_finalist_validation'] = True
        if self.uniform_validation_time_guard:
            plan['uniform_validation_time_guard_enabled'] = True
            plan['uniform_validation_time_guard_version'] = self.search._uniform_validation_time_guard_version
            self.search.last_uniform_validation_time_guard = None
        if self.paired_adaptive_validation:
            plan['paired_adaptive_validation_enabled'] = True
            plan['paired_adaptive_validation_version'] = self.search._paired_adaptive_validation_version
        if self.public_turn_reanalysis:
            plan['public_turn_reanalysis_enabled'] = True
            if self.stop_after_card_roots:
                plan['public_turn_optional_end_roots_version'] = PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION
        if self.resource_reanalysis:
            plan['resource_reanalysis_enabled'] = True
            plan['resource_reanalysis_version'] = self.search._resource_reanalysis_version
            plan['resource_policy_continuation_version'] = self._resource_continuation.version
            self._resource_continuation.prepare(request, self.search._context(request))
        if self.internal_reanalysis:
            plan['internal_reanalysis_enabled'] = True
            plan['internal_reanalysis_version'] = INTERNAL_REANALYSIS_VERSION
        if self.internal_successor_graph:
            plan['internal_successor_graph_enabled'] = True
            plan['internal_successor_graph_version'] = self.search._internal_successor_graph_version
        if self.internal_complete_root_successors:
            plan['internal_complete_root_successors_enabled'] = True
            plan['internal_complete_root_successors_version'] = COMPLETE_ROOT_SUCCESSORS_VERSION
        if self.final_double_tail_evaluation:
            plan['final_double_tail_evaluation_enabled'] = True
            plan['final_double_tail_domain_version'] = FINAL_DOUBLE_VERSION
        if self.final_offhand_remove_tail_evaluation:
            plan['final_offhand_remove_tail_evaluation_enabled'] = True
            plan['final_offhand_remove_tail_domain_version'] = FINAL_OFFHAND_REMOVE_VERSION
        if self.public_turn_budget_reservation:
            plan['public_turn_budget_reservation_enabled'] = True
            plan['public_turn_budget_reservation_version'] = PUBLIC_TURN_BUDGET_RESERVATION_VERSION
            self.search._reset_public_turn_budget_reservation()
        if self.public_validation_phase_budget:
            plan['public_validation_phase_budget_enabled'] = True
            plan['public_validation_phase_budget_version'] = PUBLIC_VALIDATION_PHASE_BUDGET_VERSION
        if self.parallel_public_graph and (self.paired_adaptive_validation or self.uniform_validation_time_guard):
            plan['parallel_root_validation_enabled'] = True
            plan['parallel_root_validation_version'] = self.search._parallel_root_validation_version
            self.search.last_parallel_root_validation = None
        if self.parallel_public_graph:
            plan['parallel_public_graph_enabled'] = True
            plan['parallel_public_graph_version'] = self.search._parallel_public_graph_version
        if self.public_turn_score_translation:
            plan['public_turn_score_translation_enabled'] = True
            plan['public_turn_score_translation_version'] = PUBLIC_TURN_SCORE_TRANSLATION_VERSION
        if self.public_turn_policy_continuation:
            plan['public_turn_policy_continuation_enabled'] = True
            plan['public_turn_policy_continuation_version'] = self._public_turn_continuation.version
            self._public_turn_continuation.prepare(request, self.search._context(request))
        if self.direct_double_marginal:
            plan['direct_double_marginal'] = True
        if self.unknown_node_timing:
            plan['unknown_node_timing_enabled'] = True
            self.search.unknown_node_policy.start_window()
        if self.partial_node_timing:
            plan['partial_node_timing_enabled'] = True
            self.search._reset_partial_node_timing()
        if self.installed_score_turn_chain:
            plan['installed_score_turn_chain_enabled'] = True
        if self.installed_score_multi_action:
            plan['installed_score_multi_action_enabled'] = True
        if self.installed_score_focus_projection:
            plan['installed_score_focus_projection_enabled'] = True
        if self.extra_action_turn_chain:
            plan['extra_action_turn_chain_enabled'] = True
        if self.visited_card_extra_action:
            plan['visited_card_extra_action_enabled'] = True
        if self.pair_order_advice:
            plan['pair_order_advice_enabled'] = True
        if self.retrieval_training_order:
            plan['retrieval_training_order_enabled'] = True
            self.search.retrieval_training_orderer.start_window()
        if self.lazy_validation_seeds:
            plan['lazy_validation_seeds_enabled'] = True
            self.search._reset_validation_seed_allocation()
        held_cohort = ()
        if self.deterministic_validation_cohort:
            plan['deterministic_validation_cohort_enabled'] = True
            self.search._reset_validation_cohort_evidence()
            held_cohort = self._validation_cohort.prepare(request, self.search._context(request))
            plan['validation_cohort_link'] = self._validation_cohort.summary()
            plan['validation_cohort_link_created'] = False
        if self._retained_policy is not None:
            from .hif_retained_policy_competition import VERSION as RETAINED_POLICY_VERSION

            plan['retained_policy_competition_enabled'] = True
            plan['retained_policy_competition_version'] = RETAINED_POLICY_VERSION
            self._retained_policy.prepare(request, held_cohort, plan['validation_cohort_link'])
            plan['retained_policy_lineage'] = self._retained_policy.evidence()
            plan['retained_policy_link_created'] = False
            if self.revealed_policy_competition:
                plan.update(revealed_policy_competition_enabled=True,
                    revealed_policy_competition_version=REVEALED_VERSION,
                    revealed_policy_competition=self._retained_policy.revealed_evidence(),
                    revealed_policy_link_created=False)
                if self._retained_policy.revealed:
                    # Actual revelation establishes comparison eligibility only.
                    # It never transports a target commitment or execution link.
                    held_cohort = ()
                    self._choice_continuation = self._frozen_policy_continuation = None
                    self._public_turn_continuation.reset()
                    self._resource_continuation.reset()
            if self.retained_policy_execution:
                from .hif_retained_policy_competition import EXECUTION_VERSION

                plan['retained_policy_execution_enabled'] = True
                plan['retained_policy_execution_version'] = EXECUTION_VERSION
            if self.retained_tree_policy_execution:
                plan['retained_tree_policy_execution_enabled'] = True
                plan['retained_tree_policy_execution_version'] = TREE_EXECUTION_VERSION
        if self.single_node_expansion:
            plan['single_node_expansion'] = True
        if self.deterministic_policy_continuation:
            plan['deterministic_policy_continuation'] = True
            if not matches_link(self._frozen_policy_continuation, request, self.search._context(request)):
                self._frozen_policy_continuation = None
        if self.continuation_samples:
            plan['continuation_samples'] = self.continuation_samples
        if self.continuation_incomplete_actions:
            plan['continuation_incomplete_actions'] = True
        if self.fresh_adaptive_training:
            plan['fresh_adaptive_training'] = True
        if self.continuation_portfolio:
            plan['continuation_portfolio'] = True
            plan['continuation_trigger'] = POLICY_CONTINUATION_TRIGGER
        self.plans.append(plan)
        if self.proven_tail is not None:
            self.proven_tail.begin_window()
        advice_before = ({name: getattr(self.double_ordering, name) for name in
                          ('calls', 'active_calls', 'bypassed_calls')} if self.double_advice else {})
        if self.guard_unpriced_schedules:
            advice_before['scope_bypassed_calls'] = self.double_ordering.scope_bypassed_calls
        if self.guard_unpriced_resources:
            advice_before['resource_scope_bypassed_calls'] = self.double_ordering.resource_scope_bypassed_calls
        direct_before = dict(self.double_ordering.direct_stats) if self.direct_double_marginal else {}
        temporal_before = ({name: getattr(self.search.advice, name) for name in
                            ('calls', 'active_calls', 'bypassed_calls', 'elapsed_ms')}
                           if self.temporal_advice else {})
        drink_before = ({name: getattr(self.search.advice, name) for name in
                         ('calls', 'active_calls', 'bypassed_calls', 'elapsed_ms')}
                        if self.drink_advice else {})
        if self.drink_advice:
            self.search.advice.reset_cache()
        retrieval_before = ({name: getattr(self.search.advice, name) for name in
                            ('calls', 'active_calls', 'bypassed_calls', 'elapsed_ms')}
                           if self.retrieval_combo_advice else {})
        if self.retrieval_combo_advice:
            self.search.advice.reset_cache()
        if self.double_return_advice or self.direct_double_marginal:
            self.double_ordering.reset_window()
        def check_budget():
            if time.perf_counter() >= search_deadline:
                raise TimeoutError('mcts_window_timeout')

        reusable_tail = None
        try:
            reusable_tail, tail_context = self._prepare_actual_tail_session(request, plan, search_deadline)
            with (validation_session(check_budget) if (self.temporal_advice or self.drink_advice
                                                       or self.retrieval_combo_advice
                                                       or self.double_return_advice
                                                       or self.direct_double_marginal
                                                       or self.unknown_node_timing
                                                       or self.continuation_portfolio) else nullcontext(),
                  self.endgame.search_session(deadline=search_deadline) as session,
                  (self.tail.search_session(deadline=search_deadline,
                       **({'reuse': reusable_tail} if reusable_tail is not None else {}))
                   if self.tail is not None else nullcontext()) as tail_session):
                self._session = session
                self._tail_session = tail_session
                resumed = self._resume_choice(request, legal, plan)
                if resumed is not None:
                    action = resumed
                elif request.state.turns_remaining == 1:
                    # Actual final-turn decisions retain the full exact
                    # solver budget and must still complete or stop.
                    response = (self._terminal(request, search_deadline) if self.quality_reference
                                else self.endgame.decide(request))
                    plan.update(scope='FINAL_SCORE', stop_reason=response.reason,
                                complete_root_coverage=response.has_action_intent,
                                requested_simulations=0, completed_simulations=0,
                                candidate_statistics=[])
                    if not response.has_action_intent:
                        raise ValueError('mcts_terminal_search_incomplete:' + response.reason)
                    action = response.action
                elif len(legal) == 1:
                    action = legal[0]
                    complete_legal = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                                               decision_kind=request.decision_kind)
                    if (len(complete_legal) != 1
                            or replace(action, source_slot=None) != complete_legal[0]):
                        raise ValueError('mcts_forced_action_list_incomplete')
                    plan.update(scope='FORCED_LEGAL_ACTION', stop_reason='single_legal_action',
                                complete_root_coverage=True, requested_simulations=0,
                                completed_simulations=0, candidate_statistics=[], legal_action_count=1)
                else:
                    proof = None
                    if self.resource_dominance and len(legal) == 2:
                        for alternative in legal:
                            proof = prove_forced_end_drink_dominance(self.kernel, request, alternative)
                            if proof is not None:
                                break
                    if proof is not None:
                        action = proof.dominant_action
                        plan.update(scope='PROVEN_RESOURCE_DOMINANCE',
                            stop_reason='deterministic_forced_end_drink_dominated',
                            complete_dominance_proof=True, dominance_proof=asdict(proof),
                            requested_simulations=0, completed_simulations=0,
                            candidate_statistics=[], legal_action_count=len(legal))
                    elif self._actual_two_turn_eligible(request, plan):
                        action = self._actual_two_turn(request, legal, search_deadline, plan)
                    elif (probe_action := self._probe_actual_two_turn(
                            request, legal, search_deadline, plan)) is not None:
                        action = probe_action
                    elif (self._public_turn_continuation is not None and
                            (public_action := self._public_turn_continuation.choose(
                                request, legal, plan, search_deadline)) is not None):
                        action = public_action
                    elif (self._resource_continuation is not None and
                            (resource_action := self._resource_continuation.choose(
                                request, legal, plan, search_deadline)) is not None):
                        action = resource_action
                    elif self._frozen_policy_continuation is not None:
                        action = self._resume_frozen_policy(request, legal, plan, search_deadline)
                    elif (self._retained_policy is not None and
                            (retained_action := self._retained_policy.choose(
                                request, legal, plan, search_deadline)) is not None):
                        action = retained_action
                    else:
                        self._frozen_policy_continuation = None
                        options = None
                        baseline_domain_reason = None
                        if self.joint_choice_roots:
                            expanded, audit = build_choice_root_options(self.kernel, request, legal,
                                max_options=self.search.simulations // (
                                    self.search.min_root_samples + self.search.validation_samples),
                                deadline=search_deadline)
                            plan['joint_root_expansion'] = audit
                            if self.continuation_samples and audit['reason'] == 'budget_ineligible':
                                if self.continuation_domain:
                                    # The original helper's whole-window
                                    # ordinary path is selected before search.
                                    # Its reduced root count cannot re-enable
                                    # local comparisons for the omitted joint
                                    # domain; retain the original audit.
                                    baseline_domain_reason = 'joint_root_expansion_budget_ineligible'
                                else:
                                    plan['stop_reason'] = 'mcts_continuation_joint_root_budget_incomplete'
                                    raise ValueError(plan['stop_reason'])
                            if any(option.continuation for option in expanded):
                                options = expanded
                        if self.stop_after_card_roots:
                            expanded, audit = build_stop_after_card_root_options(
                                self.kernel, request, legal, base_options=options,
                                max_options=self.search.simulations // (
                                    self.search.min_root_samples + self.search.validation_samples),
                                deadline=search_deadline)
                            plan['stop_root_expansion'] = audit
                            if any(option.continuation for option in expanded):
                                options = expanded
                        try:
                            result = self.search.suggest(request, deadline=search_deadline,
                                **({'retained_policy': self._retained_policy.proposal(request, legal, search_deadline)}
                                   if self._retained_policy is not None else {}),
                                **({'validation_cohort': held_cohort}
                                   if self.deterministic_validation_cohort else {}),
                                **({'root_options': options} if options is not None else {}),
                                **({'baseline_domain_reason': baseline_domain_reason}
                                   if baseline_domain_reason is not None else {}))
                        finally:
                            if self.quality_reference:
                                self.search._finish_search_phase_timing()
                        if getattr(self, 'compare_drink_order', False):
                            from .hif_drink_order import compare_drink_order

                            result, comparison = compare_drink_order(self.search, request, legal, result, search_deadline)
                            if comparison is not None:
                                plan['drink_order_comparison'] = comparison
                        plan.update(asdict(result))
                        if not self.search.unknown_node_policy or not getattr(
                                self.search.unknown_node_policy.extra_action_policy, 'unused_drink_enabled', False):
                            plan.pop('root_selection_eligibility', None)
                        if self._retained_policy is None:
                            plan.pop('retained_policy_comparison', None)
                        if not self.paired_finalist_validation:
                            plan.pop('finalist_validation', None)
                        if not self.uniform_validation_time_guard:
                            plan.pop('uniform_validation_time_guard', None)
                        if not self.paired_adaptive_validation:
                            plan.pop('paired_adaptive_validation', None)
                        if not (self.parallel_public_graph and (self.paired_adaptive_validation or self.uniform_validation_time_guard)):
                            plan.pop('parallel_root_validation', None)
                        if not self.public_turn_reanalysis:
                            plan.pop('public_turn_reanalysis', None)
                        if not self.resource_reanalysis:
                            plan.pop('resource_reanalysis', None)
                        if not self.internal_reanalysis:
                            plan.pop('internal_reanalysis', None)
                        if not self.single_node_expansion:
                            plan.pop('single_node_expansion', None)
                            plan.pop('training_expansion', None)
                        if result.action is None or not result.complete_root_coverage:
                            raise ValueError('mcts_planning_incomplete:' + result.stop_reason)
                        action = result.action
                        if options is not None:
                            self._remember_choice(request, action, result)
                matches = [candidate for candidate in legal if action_key(candidate) == action_key(action)]
                if len(matches) != 1:
                    raise ValueError('mcts_candidate_action_not_uniquely_legal')
                if self.deterministic_policy_continuation:
                    plan['elapsed_ms'] = (time.perf_counter() - started) * 1000
                    self._remember_frozen_policy(request, matches[0], plan, search_deadline)
                if self.deterministic_validation_cohort:
                    next_cohort = held_cohort
                    if plan['scope'] == 'SAMPLING_ESTIMATE':
                        next_cohort = ()
                        if plan.get('complete_validation_coverage'):
                            next_cohort = tuple(self.search.last_validation_cohort_evidence[
                                'validation_seeds'][:4])
                    if next_cohort:
                        plan['validation_cohort_link_created'] = self._validation_cohort.remember(
                            request, matches[0], next_cohort, self.search._context(request),
                            search_deadline, self._plan_offset + len(self.plans) - 1)
                # Session diagnostics are deliberately not observation or
                # authorization data. The action still binds to this request.
                plan['terminal_cache'] = dict(decisions=session.decisions, memo_hits=session.memo_hits,
                                              namespaces=session.namespaces_created)
                if tail_session is not None:
                    plan['complete_tail_cache'] = dict(decisions=tail_session.decisions,
                        memo_hits=tail_session.memo_hits, namespaces=tail_session.namespaces_created)
                retained_won = bool((plan.get('retained_policy_comparison') or {}).get('selected'))
                if retained_won:
                    # These tables belong to the losing challenger. Publishing
                    # them would silently replace the retained winner next turn.
                    self._public_turn_continuation.reset()
                    self._resource_continuation.reset()
                    plan['public_turn_policy_link_created'] = False
                    plan['resource_policy_link_created'] = False
                if self._public_turn_continuation is not None and not retained_won:
                    plan['public_turn_policy_link_created'] = self._public_turn_continuation.remember(
                        request, matches[0], dict(plan, accepted=True), self.search._public_turn_policy,
                        self.search._context(request), self._plan_offset + len(self.plans) - 1)
                if self._resource_continuation is not None and not retained_won:
                    plan['resource_policy_link_created'] = self._resource_continuation.remember(
                        request, matches[0], dict(plan, accepted=True,
                            elapsed_ms=(time.perf_counter() - started) * 1000),
                        self.search._resource_policy if plan['scope'] == 'SAMPLING_ESTIMATE' else None,
                        self.search._context(request), self._plan_offset + len(self.plans) - 1,
                        search_deadline)
                if self._retained_policy is not None:
                    plan['retained_policy_link_created'] = self._retained_policy.remember(
                        request, dict(plan, accepted=True, **({'action': asdict(matches[0])}
                            if self.revealed_policy_competition else {}),
                            **({'validation_cohort_evidence': dict(self.search.last_validation_cohort_evidence)}
                               if self.revealed_policy_competition and plan['scope'] == 'SAMPLING_ESTIMATE' else {})),
                        self._plan_offset + len(self.plans) - 1, search_deadline)
                    if self.revealed_policy_competition:
                        plan['revealed_policy_link_created'] = plan['retained_policy_link_created']
            if time.perf_counter() >= deadline:
                raise ValueError('mcts_delivery_window_exhausted')
            plan.update(accepted=True, action=asdict(matches[0]))
            if reusable_tail is not None:
                count = reusable_tail.entry_count
                plan['actual_tail_session_reuse']['entries_after'] = count
                if 0 < count <= _MAX_RETAINED_TAIL_ENTRIES:
                    self._actual_tail_link = dict(session=reusable_tail, context=tail_context,
                        before=self.kernel._canonical_probability_state(request.state),
                        action=matches[0], after=None, committed=False,
                        source_plan_index=self._plan_offset + len(self.plans) - 1)
                else:
                    reusable_tail.clear()
            return matches[0]
        except TimeoutError as error:
            if self._retained_policy is None:
                raise
            # The episode runner records ValueError as a stopped decision.
            # A deadline in snapshot/bind must not discard the partial episode.
            plan['stop_reason'] = str(error)
            raise ValueError('mcts_planning_incomplete:' + str(error)) from error
        finally:
            if self._retained_policy is not None and plan.get('accepted') is not True:
                self._retained_policy.reset()
                plan['retained_policy_link_created'] = False
                if self.revealed_policy_competition:
                    plan['revealed_policy_link_created'] = False
            if reusable_tail is not None and (plan.get('accepted') is not True
                                              or self._actual_tail_link is None):
                reusable_tail.clear()
            if self._public_turn_continuation is not None and plan.get('accepted') is not True:
                self._public_turn_continuation.reset()
                plan['public_turn_policy_link_created'] = False
            if self._resource_continuation is not None and plan.get('accepted') is not True:
                self._resource_continuation.reset()
                plan['resource_policy_link_created'] = False
            if self.deterministic_validation_cohort:
                if plan['scope'] == 'SAMPLING_ESTIMATE':
                    plan['validation_cohort_evidence'] = dict(self.search.last_validation_cohort_evidence)
                if plan.get('accepted') is not True:
                    self._validation_cohort.reset()
                    plan['validation_cohort_link_created'] = False
            if self.deterministic_policy_continuation and plan.get('accepted') is not True:
                self._frozen_policy_continuation = None
            self._session = None
            self._tail_session = None
            if self.proven_tail is not None:
                plan['proven_tail'] = dict(version=PROVEN_TAIL_VERSION,
                    entries=len(self.proven_tail), frozen=self.proven_tail.frozen,
                    scope='COMPLETE_ROOT_EXPECTED_VALUES_ONLY')
                self.proven_tail.reset_window()
            # Also retain measured call time when suggest raises. Successful
            # searches already close their timer before returning the estimate.
            if self.uniform_validation_time_guard and self.search.last_uniform_validation_time_guard is not None:
                plan['uniform_validation_time_guard'] = self.search.last_uniform_validation_time_guard
            if (self.parallel_public_graph and (self.paired_adaptive_validation or self.uniform_validation_time_guard)
                    and self.search.last_parallel_root_validation is not None):
                plan['parallel_root_validation'] = self.search.last_parallel_root_validation
            if self.quality_reference and self.search.last_search_phase_timing is not None:
                self.search._finish_search_phase_timing()
                timing = self.search.last_search_phase_timing
                plan['search_phase_timing'] = dict(timing,
                    phases={name: dict(row) for name, row in timing['phases'].items()})
            plan['elapsed_ms'] = (time.perf_counter() - started) * 1000
            if self.public_turn_budget_reservation:
                plan['public_turn_budget_reservation'] = dict(self.search.last_public_turn_budget_reservation)
            if self.public_validation_phase_budget:
                plan['public_validation_phase_budget'] = dict(self.search.last_public_validation_phase_budget)
            if self.unknown_node_timing:
                plan['unknown_node_timing'] = self.search.unknown_node_policy.summary()
            if self.partial_node_timing:
                plan['partial_node_timing'] = dict(self.search.last_partial_node_timing)
            if self.installed_score_turn_chain:
                plan['installed_score_turn_chain'] = self.search.unknown_node_policy.installed_score_policy.summary()
            if self.extra_action_turn_chain:
                plan['extra_action_turn_chain'] = self.search.unknown_node_policy.extra_action_policy.summary()
            if self.pair_order_advice:
                plan['pair_order_advice'] = self.search.unknown_node_policy.pair_order_policy.summary()
            if self.retrieval_training_order:
                plan['retrieval_training_order'] = self.search.retrieval_training_orderer.summary()
            if self.lazy_validation_seeds:
                plan['validation_seed_allocation'] = dict(self.search.last_validation_seed_allocation)
            if self.double_advice:
                plan['double_advice'] = dict(version=self.double_ordering.model_version,
                    **{name: getattr(self.double_ordering, name) - value for name, value in advice_before.items()})
                if self.direct_double_marginal:
                    plan['double_advice']['direct_marginal'] = dict(
                        version=self.direct_double_marginal_version,
                        counts={name: value-direct_before.get(name, 0)
                                for name, value in self.double_ordering.direct_stats.items()})
                if self.double_return_advice or self.direct_double_marginal:
                    self.double_ordering.reset_window()
            if self.temporal_advice:
                plan['temporal_advice'] = dict(version=TEMPORAL_ADVICE_VERSION,
                    **{name: getattr(self.search.advice, name) - value for name, value in temporal_before.items()})
            if self.drink_advice:
                plan['drink_advice'] = dict(version=DRINK_ADVICE_VERSION,
                    **{name: getattr(self.search.advice, name) - value for name, value in drink_before.items()})
                self.search.advice.reset_cache()
            if self.retrieval_combo_advice:
                plan['retrieval_combo_advice'] = dict(version=RETRIEVAL_COMBO_VERSION,
                    **{name: getattr(self.search.advice, name) - value for name, value in retrieval_before.items()})
                self.search.advice.reset_cache()
            link = self._frozen_policy_continuation
            if link is not None and link['proof']['source_plan_index'] == self._plan_offset + len(self.plans) - 1:
                link['source_plan'] = dict(plan)
            if self.uniform_validation_time_guard:
                finished = time.perf_counter()
                plan['elapsed_ms'] = (finished - started) * 1000
                if plan.get('accepted') is True and finished >= deadline:
                    plan.update(accepted=False, stop_reason='mcts_delivery_window_exhausted')
                    for controller, field in ((self._retained_policy, 'retained_policy'),
                            (self._validation_cohort, 'validation_cohort'),
                            (self._public_turn_continuation, 'public_turn_policy'),
                            (self._resource_continuation, 'resource_policy')):
                        if controller is not None:
                            controller.reset()
                            plan[field + '_link_created'] = False
                    if self.revealed_policy_competition:
                        plan['revealed_policy_link_created'] = False
                    self._frozen_policy_continuation = None
                    self._choice_continuation = None
                    self._clear_actual_tail_link()
                    raise ValueError('mcts_delivery_window_exhausted')

    def committed(self, before, action, after):
        if self.revealed_policy_competition:
            self._retained_policy.committed(before, action, after)
        saved = self._actual_tail_link
        if saved is not None:
            if (not saved['committed'] and before.turn == after.turn
                    and before.turns_remaining == after.turns_remaining == 2
                    and self.kernel._canonical_probability_state(before) == saved['before']
                    and action == saved['action']):
                saved['after'] = self.kernel._canonical_probability_state(after)
                saved['committed'] = True
                if action.kind is ActionKind.USE_DRINK:
                    slots = tuple(pair for pair in saved['context'][-1]
                                  if pair != (action.source_slot, action.drink_id))
                    saved['context'] = (*saved['context'][:-1], slots)
            else:
                self._clear_actual_tail_link()
        self.search.committed(before, action, after)
        if self._resource_continuation is not None:
            self._resource_continuation.committed(before, action, after)
        if self._public_turn_continuation is not None:
            self._public_turn_continuation.committed(before, action, after)
        if self._validation_cohort is not None:
            self._validation_cohort.committed(before, action, after)
        link = self._frozen_policy_continuation
        if link is not None:
            proof = link['proof']
            if (not proof['committed'] and before == proof['before']
                    and business_action(action) == proof['action'] and after == proof['expected_state']):
                proof['committed'] = True
            else:
                self._frozen_policy_continuation = None
        saved = self._choice_continuation
        if saved is not None:
            if (not saved['committed']
                    and self.kernel._canonical_probability_state(before) == saved['before']
                    and replace(action, source_slot=None) == saved['first']
                    and self.kernel._canonical_probability_state(after) == saved['expected_state']):
                saved['committed'] = True
            else:
                self._choice_continuation = None
