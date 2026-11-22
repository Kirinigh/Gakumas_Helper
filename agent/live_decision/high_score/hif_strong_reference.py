"""Explicit strong-reference trial, not an approved production strategy.

The 017 completion-v3 configuration is preserved with the original v1
visited-card semantics. Its independent quality gate failed; neither this
loader nor an integration test promotes it or grants game-input authority.
No archived source is loaded at runtime, and current v3/default v22 stay intact.
"""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

PROFILE = VERSION = 'hif-strong-reference-trial-v1'
REFERENCE_COMMIT = '93e7d75819c032feba125616361e99af708f4fb9'
REFERENCE_CONFIGURATION_VERSION = 'hif-final-offhand-remove-tail-trial-v1'
REFERENCE_MODEL_SHA256 = 'd40921dc0e9935c2889bb0233636ab0f5e1701bb836ae4d594de2453d50ad700'
MODEL_RELATIVE_PATH = Path('assets/data/hif_policy_candidates/ppo-27502-v1.json')
INDEPENDENT_QUALITY_PASSED = False

# Frozen 017 configuration; only profile identity and the explicit v1 semantic
# selector are added. Call configuration() before supplying mutable options.
CONFIG = {'version': 'hif-strong-reference-trial-v1',
 'batch_seconds': 900.0,
 'planning_seed_base': 603000000,
 'policy_kwargs': {'simulations': 256,
                   'min_root_samples': 4,
                   'window_seconds': 300.0,
                   'soft_window_seconds': 3.0,
                   'terminal_attempt_ms': 500.0,
                   'scenario_reuse': True,
                   'validation_samples': 4,
                   'resource_dominance': True,
                   'double_advice': True,
                   'adaptive_validation': True,
                   'temporal_advice': False,
                   'drink_advice': False,
                   'progressive_widening': False,
                   'retrieval_combo_advice': False,
                   'exact_tail_ordering': False,
                   'complete_tail_evaluation': False,
                   'simple_tail_evaluation': True,
                   'simple_final_turn_evaluation': True,
                   'joint_choice_roots': True,
                   'actual_two_turn_probe_ms': 3000.0,
                   'guard_unpriced_resources': True,
                   'unknown_node_timing': True,
                   'retrieval_training_order': True,
                   'lazy_validation_seeds': True,
                   'deterministic_validation_cohort': True,
                   'partial_node_timing': True,
                   'installed_score_turn_chain': True,
                   'public_turn_reanalysis': True,
                   'public_turn_policy_continuation': True,
                   'pair_order_advice': True,
                   'public_turn_score_translation': True,
                   'actual_tail_session_reuse': True,
                   'public_turn_budget_reservation': True,
                   'public_validation_phase_budget': True,
                   'extra_action_turn_chain': True,
                   'parallel_public_graph': True,
                   'resource_reanalysis': True,
                   'retained_policy_competition': True,
                   'retained_policy_execution': True,
                   'stop_after_card_roots': True,
                   'retained_tree_policy_execution': True,
                   'internal_reanalysis': True,
                   'final_double_tail_evaluation': True,
                   'final_offhand_remove_tail_evaluation': True,
                   'quality_reference': True,
                   'fresh_adaptive_training': True,
                   'internal_complete_root_successors': True,
                   'revealed_policy_competition': True,
                   'installed_score_multi_action': True,
                   'installed_score_focus_projection': True,
                   'visited_card_extra_action': True,
                   'internal_successor_graph': True,
                   'uniform_validation_time_guard': True,
                   'visited_card_extra_action_version': 'extra-action-last-action-chain-visited-card-v1'},
 'reward_scale': 100000.0,
 'planning_sampler': 'common-card-exponential-race-v1',
 'value_scope': 'TRAINING_DISJOINT_COMPLETE_RETURNS_SAME_WORLDS_AFTER_PROVEN_DETERMINISTIC_COMMIT',
 'root_selection': 'fresh_or_reused_training_disjoint_paired_scenarios',
 'validation_allocation': 'half_training_then_paired_rounds_within_soft_window_max32',
 'joint_training_allocation': 'complete_existing_training_quota_within_hard_deadline',
 'resource_dominance': 'deterministic_forced_end_drink_v1',
 'double_advice_version': 'task275-double-semantic-v6-resource-scope-20261010',
 'two_turn_domain': 'task275-fixed-next-boundary-draw-domain-v3',
 'two_turn_sequence_paths': 4096,
 'retrieval_combo_version': 'task275-visible-retrieval-combo-v3',
 'exact_tail_ordering_version': 'task275-exact-tail-ordering-v1-20261002',
 'terminal_horizon': 2,
 'validation_terminal_budget': 'fixed_public_domain_exact_else_sample_to_terminal',
 'hypothetical_final_turn': 'exact_in_public_domain_else_frozen_tree_complete_return',
 'actual_final_turn': 'existing_complete_solver',
 'mandatory_choice': 'deterministic_public_target_joint_root_then_fresh_binding',
 'repeated_simulations_are_independent_samples': False,
 'future_random_support_modeled': False,
 'actual_two_turn': 'complete_horizon_probe_3000ms_max_quarter_remaining_window',
 'resource_scope': 'preserve_original_prior_when_six_descriptors_omit_positive_future_conversion',
 'unknown_node_timing_version': 'unknown-node-sense-timing-v3',
 'unknown_node_timing': 'clean_nonfinal_absent_validation_nodes_and_verified_black_vinegar_pending_only_no_nested_rollouts',
 'retrieval_training_order_version': 'retrieval-training-order-v1',
 'retrieval_training_order': 'new_unforced_internal_training_nodes_prior_first_drink22_public_timing_reorders_all_edges_retained',
 'lazy_validation_seeds_version': 'lazy-validation-seeds-v1',
 'deterministic_validation_cohort_version': 'deterministic-validation-cohort-v1',
 'validation_cohort_reuse': 'strict_no_draw_no_random_same_turn_committed_state; not fresh independent samples',
 'validation_seed_allocation': 'four_mandatory_then_only_started_optional_rounds',
 'partial_node_timing_version': 'partial-node-sense-timing-v1',
 'partial_node_continuation': 'existing_single_step_timing_only_for_mixed_visited_unvisited_nodes',
 'installed_score_turn_chain_version': 'installed-score-multi-action-focus-projection-chain-v1',
 'public_turn_reanalysis_version': 'hif-public-turn-reanalysis-v2',
 'public_turn_policy_continuation_version': 'hif-public-turn-policy-continuation-v1',
 'pair_order_advice_version': 'deterministic-pair-order-advice-v1',
 'public_turn_score_translation_version': 'hif-installed-score-translation-v1',
 'actual_tail_session_reuse_version': 'hif-actual-tail-session-reuse-v1',
 'public_turn_budget_reservation_version': 'hif-public-turn-budget-reservation-v1',
 'public_validation_phase_budget_version': 'hif-public-validation-phase-budget-v1',
 'extra_action_turn_chain_version': 'extra-action-last-action-chain-visited-card-v1',
 'parallel_public_graph_version': 'hif-parallel-public-graph-v3',
 'resource_reanalysis_version': 'hif-resource-reanalysis-v1',
 'resource_continuation_version': 'hif-resource-policy-continuation-v1',
 'retained_policy_competition_version': 'hif-retained-policy-competition-v1',
 'retained_policy_execution_version': 'hif-retained-policy-execution-v1',
 'public_turn_optional_end_roots_version': 'hif-public-turn-optional-end-roots-v1',
 'retained_tree_policy_execution_version': 'hif-retained-tree-policy-execution-v2',
 'internal_reanalysis_version': 'hif-internal-reanalysis-v2',
 'final_double_tail_domain_version': 'task275-final-direct-double-tail-domain-v1',
 'final_offhand_remove_tail_domain_version': 'task275-final-offhand-remove-tail-domain-v1',
 'quality_reference_version': 'hif-quality-reference-v1',
 'internal_complete_root_successors_version': 'hif-internal-complete-root-successors-v1',
 'revealed_policy_competition_version': 'hif-revealed-policy-competition-v1',
 'internal_successor_graph_version': 'hif-internal-successor-graph-v1',
 'uniform_validation_time_guard_version': 'hif-uniform-validation-time-guard-v2',
 'parallel_root_validation_version': 'hif-parallel-root-validation-v1'}


def configuration():
    """Return one private configuration; never mutate the shared reference."""
    return deepcopy(CONFIG)


def verified_model_path(model_path=None):
    """Check the frozen model without creating a policy or a search window."""
    if __package__.startswith('agent.'):
        from agent.utils.runtime_paths import resolve_runtime_path
    else:
        from utils.runtime_paths import resolve_runtime_path
    path = (resolve_runtime_path(MODEL_RELATIVE_PATH, Path('data/hif_policy_candidates/ppo-27502-v1.json'))
            if model_path is None else Path(model_path))
    payload = json.loads(path.read_text(encoding='utf8'))
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf8')
    if hashlib.sha256(encoded).hexdigest() != REFERENCE_MODEL_SHA256:
        raise ValueError('strong_reference_frozen_model_changed')
    return path


def load_baseline(kernel, model_path=None):
    """Require the complete frozen model, including weights and provenance."""
    from .hif_hybrid import load_visible_hif_candidate

    return load_visible_hif_candidate(kernel, verified_model_path(model_path))
