"""Offline public-state tree search; sampled values never authorize input."""
from __future__ import annotations

import math
import time
import random
from copy import deepcopy
from dataclasses import field, replace, dataclass

from .kernel import PythonNiaProKernel
from .planner import RobustHighScorePlanner
from .contracts import ActionKind, GameAction, RiskProfile, DecisionKind
from .hif_dominance import prove_forced_end_drink_dominance
from .hif_environment import HifEnvironment
from .hif_local_continuation import LocalContinuation
from .hif_policy_continuation import (
    POLICY_IDS,
    PolicyContinuation,
    RetrievalTrainingOrder,
    UnknownNodeTimingPolicy,
)
from .hif_installed_score_continuation import InstalledScoreTurnChain

POLICY_CONTINUATION_TRIGGER = 'first_free_post_root_choice'
FINALIST_VALIDATION_VERSION = 'hif-residual-finalist-validation-v1'
LAZY_VALIDATION_SEEDS_VERSION = 'lazy-validation-seeds-v1'
DETERMINISTIC_VALIDATION_COHORT_VERSION = 'deterministic-validation-cohort-v1'
QUALITY_REFERENCE_VERSION = 'hif-quality-reference-v1'
COMPUTE_SENSITIVITY_VERSION = 'hif-compute-sensitivity-v1'
PARTIAL_NODE_TIMING_VERSION = 'partial-node-sense-timing-v1'
PUBLIC_TURN_BUDGET_RESERVATION_VERSION = 'hif-public-turn-budget-reservation-v1'
PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION = 'hif-public-turn-optional-end-roots-v1'
PUBLIC_VALIDATION_PHASE_BUDGET_VERSION = 'hif-public-validation-phase-budget-v1'
RETAINED_POLICY_COMPETITION_VERSION = 'hif-retained-policy-competition-v1'
REVEALED_POLICY_COMPETITION_VERSION = 'hif-revealed-policy-competition-v1'
SEARCH_PHASE_TIMING_VERSION = 'hif-search-phase-timing-v1'
ROOT_SELECTION_ELIGIBILITY_VERSION = 'hif-root-selection-eligibility-v1'


def action_key(action):
    return (action.kind.value, action.card_instance_id, action.drink_id,
            action.target_id, action.source_slot if action.kind is ActionKind.USE_DRINK else None)


@dataclass(frozen=True)
class MctsEstimate:
    action: object | None
    completed_simulations: int
    candidate_statistics: tuple[dict, ...]
    elapsed_ms: float
    value_scope: str = 'SAMPLING_ESTIMATE'
    optimality_proven: bool = False
    execution_authorized: bool = False
    requested_simulations: int = 0
    stop_reason: str = 'completed'
    complete_root_coverage: bool = False
    root_paired_visits: tuple[int, ...] = ()
    retained_nodes: int = 0
    expanded_nodes: int = 0
    maximum_depth: int = 0
    terminal_evaluations: int = 0
    terminal_fallbacks: int = 0
    scenario_reuse: bool = False
    root_selection_scope: str = 'retained_complete_returns'
    selection_changed_from_retained: bool = False
    tree_multi_action_nodes: int = 0
    tree_revisited_multi_action_nodes: int = 0
    training_completed_simulations: int = 0
    validation_completed_simulations: int = 0
    requested_validation_samples: int = 0
    complete_validation_coverage: bool = False
    validation_paired_visits: tuple[int, ...] = ()
    training_stop_reason: str = 'completed'
    validation_stop_reason: str = 'disabled'
    adaptive_validation: bool = False
    validation_sample_limit: int = 0
    progressive_widening: bool = False
    complete_ordering: dict | None = None
    terminal_horizon: int = 1
    validation_tail_lookup_enabled: bool = False
    validation_tail_hits: int = 0
    validation_tail_misses: int = 0
    terminal_eligibility_enabled: bool = False
    terminal_eligibility_horizons: tuple[int, ...] = (2,)
    training_terminal_eligible: int = 0
    training_terminal_declined: int = 0
    validation_terminal_eligible: int = 0
    validation_terminal_declined: int = 0
    root_options_enabled: bool = False
    selected_continuation: tuple = ()
    selected_expected_after_first: object | None = None
    physical_root_count: int = 0
    root_option_count: int = 0
    joint_training_uses_hard_deadline: bool = False
    training_simulation_budget: int = 0
    adaptive_training_completed_simulations: int = 0
    continuation_samples: int = 0
    requested_continuation_samples: int = 0
    continuation_domain: bool = False
    continuation_incomplete_actions: bool = False
    continuation_portfolio: bool = False
    portfolio_training_allocation: dict | None = None
    continuation_trigger: str | None = None
    partial_node_continuation_calls: int = 0
    partial_node_continuation_cache_hits: int = 0
    partial_node_new_actions_evaluated: int = 0
    fresh_adaptive_training: bool = False
    fresh_adaptive_training_simulations: int = 0
    fresh_adaptive_training_distinct_scenarios: int = 0
    training_distinct_scenarios: int = 0
    domain_limit: int | None = None
    unimproved_count: int = 0
    continuation_mode: str = 'disabled'
    baseline_domain_reason: str | None = None
    started_simulations: int = 0
    training_started_simulations: int = 0
    validation_started_simulations: int = 0
    inner_started_simulations: int = 0
    inner_completed_simulations: int = 0
    unused_simulation_budget: int = 0
    local_continuation_calls: int = 0
    local_continuation_cache_hits: int = 0
    inner_terminal_evaluations: int = 0
    inner_terminal_fallbacks: int = 0
    inner_tail_hits: int = 0
    inner_tail_misses: int = 0
    inner_terminal_eligible: int = 0
    inner_terminal_declined: int = 0
    turn_chain_continuation: dict | None = None
    single_node_expansion: bool = False
    training_expansion: dict | None = None
    finalist_validation: dict | None = None
    paired_adaptive_validation: dict | None = None
    uniform_validation_time_guard: dict | None = None
    parallel_root_validation: dict | None = None
    public_turn_reanalysis: dict | None = None
    resource_reanalysis: dict | None = None
    internal_reanalysis: dict | None = None
    retained_policy_comparison: dict | None = None
    root_selection_eligibility: dict | None = None


@dataclass(frozen=True)
class _RootOption:
    action: object
    continuation: tuple
    expected_after_first: object | None

    @property
    def key(self):
        return action_key(self.action), tuple(action_key(action) for action in self.continuation)


@dataclass
class _Node:
    # Statistics contain only complete future score increments, never slots.
    statistics: dict = field(default_factory=dict)
    order: tuple = ()
    children: set = field(default_factory=set)


class OfflineHifMcts:
    """Reusable observed-state UCT tree with paired root coverage.

    Training uses UCT throughout the public-state tree. Random draws are
    revealed only after choosing an action. A model orders first visits; it
    cannot remove actions or supply the return value. Optional independent
    validation follows the frozen tree's completed means without learning.
    """

    def __init__(self, kernel, *, advice=None, simulations=256, timeout_ms=5000,
                 seed=275, min_root_samples=4, reward_scale=100000.,
                 terminal_evaluator=None, sampler_factory=None, soft_timeout_ms=None,
                 scenario_reuse=False, validation_samples=0, dominance_pruning=False,
                 adaptive_validation=False, progressive_widening=False,
                 complete_orderer=None, terminal_horizon=1, validation_terminal_evaluator=None,
                 validation_tail_lookup=None, freeze_callback=None, terminal_eligible=None,
                 terminal_eligibility_horizons=(2,), continuation_samples=0, continuation_domain=False,
                 continuation_incomplete_actions=False, fresh_adaptive_training=False,
                 continuation_policy_factory=None, turn_chain_continuation=None,
                 stop_after_card_roots=False, root_coverage_first=False,
                 single_node_expansion=False, equivalent_instance_reuse=False,
                 paired_finalist_validation=False, public_turn_reanalysis=False,
                 score_translation_contract=None, unknown_node_policy=None,
                 retrieval_training_orderer=None, lazy_validation_seeds=False,
                 deterministic_validation_cohort=False, partial_node_timing=False,
                 public_turn_budget_reservation=False, public_validation_phase_budget=False,
                 parallel_public_graph=False, resource_reanalysis=False, internal_reanalysis=False,
                 quality_reference=False, compute_sensitivity=False,
                 internal_complete_root_successors=False, revealed_policy_competition=False,
                 internal_successor_graph=False, paired_adaptive_validation=False,
                 uniform_validation_time_guard=False):
        if type(quality_reference) is not bool:
            raise ValueError('mcts_quality_reference_requires_bool')
        if type(compute_sensitivity) is not bool:
            raise ValueError('mcts_compute_sensitivity_requires_bool')
        if compute_sensitivity and (not quality_reference or type(simulations) is not int
                or simulations not in (256, 512, 1024)):
            raise ValueError('mcts_compute_sensitivity_requires_quality_budget')
        self.quality_reference = quality_reference
        self.last_search_phase_timing = None
        self.last_parallel_root_validation = None
        self._search_phase_started = None
        self.compute_sensitivity = compute_sensitivity
        if (type(revealed_policy_competition) is not bool or revealed_policy_competition and (
                not quality_reference or compute_sensitivity or simulations != 256
                or not resource_reanalysis)):
            raise ValueError('mcts_revealed_policy_requires_quality_resource_budget')
        self.revealed_policy_competition = revealed_policy_competition
        if type(internal_reanalysis) is not bool or internal_reanalysis and not resource_reanalysis:
            raise ValueError('mcts_internal_reanalysis_requires_resource_stack')
        self.internal_reanalysis = internal_reanalysis
        if (type(internal_complete_root_successors) is not bool
                or internal_complete_root_successors and not (quality_reference and internal_reanalysis)):
            raise ValueError('mcts_internal_complete_roots_require_quality_internal_stack')
        self.internal_complete_root_successors = internal_complete_root_successors
        if (type(internal_successor_graph) is not bool or internal_successor_graph and (
                not quality_reference or not internal_reanalysis
                or compute_sensitivity or simulations != 256)):
            raise ValueError('mcts_internal_successor_graph_requires_quality_internal_stack')
        self.internal_successor_graph = internal_successor_graph
        if (type(paired_adaptive_validation) is not bool or paired_adaptive_validation and (
                not quality_reference or not adaptive_validation or not lazy_validation_seeds
                or not deterministic_validation_cohort or not scenario_reuse or simulations != 256
                or min_root_samples != 4 or validation_samples != 4 or compute_sensitivity
                or paired_finalist_validation or continuation_samples)):
            raise ValueError('mcts_paired_adaptive_validation_requires_quality_four_paired_search')
        self.paired_adaptive_validation = paired_adaptive_validation
        if (type(uniform_validation_time_guard) is not bool or uniform_validation_time_guard and (
                not quality_reference or not adaptive_validation or not lazy_validation_seeds
                or not deterministic_validation_cohort or not scenario_reuse or simulations != 256
                or min_root_samples != 4 or validation_samples != 4 or compute_sensitivity
                or paired_adaptive_validation or paired_finalist_validation or continuation_samples)):
            raise ValueError('mcts_uniform_validation_time_guard_requires_quality_four_rounds')
        self.uniform_validation_time_guard = uniform_validation_time_guard
        self.last_uniform_validation_time_guard = None
        public_turn_installed_stack = (public_turn_reanalysis
            and type(unknown_node_policy) is UnknownNodeTimingPolicy
            and type(unknown_node_policy.installed_score_policy) is InstalledScoreTurnChain
            and type(retrieval_training_orderer) is RetrievalTrainingOrder
            and partial_node_timing and lazy_validation_seeds and deterministic_validation_cohort)
        if (type(resource_reanalysis) is not bool or resource_reanalysis and (
                not public_turn_installed_stack or not public_turn_budget_reservation
                or score_translation_contract != 'hif-score-translation-v1'
                or equivalent_instance_reuse or min_root_samples != 4 or validation_samples != 4)):
            raise ValueError('mcts_resource_reanalysis_requires_complete_public_stack')
        if (public_turn_reanalysis and not public_turn_installed_stack and (
                unknown_node_policy is not None or retrieval_training_orderer is not None
                or partial_node_timing or lazy_validation_seeds or deterministic_validation_cohort)):
            raise ValueError('mcts_public_turn_reanalysis_requires_complete_installed_stack')
        if (type(partial_node_timing) is not bool or partial_node_timing and (
                unknown_node_policy is None or complete_orderer is not None
                or turn_chain_continuation is not None or continuation_samples
                or continuation_policy_factory is not None
                or public_turn_reanalysis and not public_turn_installed_stack)):
            raise ValueError('mcts_partial_timing_incompatible_continuation')
        if (type(deterministic_validation_cohort) is not bool
                or deterministic_validation_cohort and (not lazy_validation_seeds or min_root_samples != 4)):
            raise ValueError('mcts_deterministic_validation_cohort_incompatible_configuration')
        if (type(lazy_validation_seeds) is not bool or lazy_validation_seeds and (
                not scenario_reuse or not adaptive_validation or validation_samples != 4
                or fresh_adaptive_training or continuation_samples
                or continuation_policy_factory is not None or turn_chain_continuation is not None
                or public_turn_reanalysis and not public_turn_installed_stack
                or paired_finalist_validation)):
            raise ValueError('mcts_lazy_validation_seeds_incompatible_configuration')
        if unknown_node_policy is not None and (
                complete_orderer is not None or turn_chain_continuation is not None
                or continuation_samples or public_turn_reanalysis and not public_turn_installed_stack):
            raise ValueError('mcts_unknown_timing_incompatible_continuation')
        if (type(simulations) is not int or simulations < 1
                or type(min_root_samples) is not int or min_root_samples < 1
                or not math.isfinite(timeout_ms) or timeout_ms <= 0
                or quality_reference and timeout_ms > 300_000
                or not math.isfinite(reward_scale) or reward_scale <= 0
                or (soft_timeout_ms is not None
                    and (not math.isfinite(soft_timeout_ms) or soft_timeout_ms <= 0))
                or type(seed) is not int or type(scenario_reuse) is not bool
                or type(validation_samples) is not int or validation_samples < 0
                or type(dominance_pruning) is not bool or type(adaptive_validation) is not bool
                or type(progressive_widening) is not bool
                or adaptive_validation and not validation_samples
                or type(terminal_horizon) is not int or terminal_horizon not in (1, 2)
                or (validation_tail_lookup is None) != (freeze_callback is None)
                or validation_tail_lookup is not None and (
                    not callable(validation_tail_lookup) or not callable(freeze_callback)
                    or not validation_samples)):
            raise ValueError('MCTS budgets must be positive')
        if (type(continuation_samples) is not int or continuation_samples not in (0, 1, 4)
                or continuation_samples and (min_root_samples != 4 or validation_samples != 4)):
            raise ValueError('mcts_continuation_requires_four_training_and_outer_samples')
        if type(continuation_domain) is not bool or continuation_domain and continuation_samples not in (1, 4):
            raise ValueError('mcts_continuation_domain_requires_one_or_four_inner')
        if (type(continuation_incomplete_actions) is not bool
                or continuation_incomplete_actions and (not continuation_domain or continuation_samples not in (1, 4))):
            raise ValueError('mcts_incomplete_actions_requires_public_continuation_domain')
        if type(fresh_adaptive_training) is not bool or fresh_adaptive_training and not scenario_reuse:
            raise ValueError('mcts_fresh_adaptive_training_requires_scenario_reuse')
        if continuation_policy_factory is not None and (
                not callable(continuation_policy_factory) or continuation_samples != 4
                or continuation_incomplete_actions):
            raise ValueError('mcts_policy_continuation_requires_four_inner_without_partial_actions')
        if turn_chain_continuation is not None and (not validation_samples or continuation_samples):
            raise ValueError('mcts_turn_chain_requires_independent_validation_without_inner_sampling')
        if (type(root_coverage_first) is not bool or root_coverage_first and (
                not validation_samples or continuation_samples)):
            raise ValueError('mcts_root_coverage_first_requires_independent_validation_without_inner')
        if (type(stop_after_card_roots) is not bool or stop_after_card_roots and (
                simulations != 256 and not compute_sensitivity
                or min_root_samples != 4 or validation_samples != 4
                or continuation_samples)):
            raise ValueError('mcts_stop_roots_require_256_four_paired_without_inner')
        if type(single_node_expansion) is not bool:
            raise ValueError('mcts_single_node_expansion_requires_bool')
        if (type(equivalent_instance_reuse) is not bool or equivalent_instance_reuse
                and (min_root_samples < 4 or validation_samples < 4)):
            raise ValueError('mcts_instance_reuse_requires_four_paired_samples')
        if (type(paired_finalist_validation) is not bool or paired_finalist_validation and (
                simulations != 256 or min_root_samples != 4 or validation_samples != 4
                or not scenario_reuse or not adaptive_validation
                or timeout_ms > (300_000 if quality_reference else 30_000)
                or fresh_adaptive_training or continuation_samples or continuation_domain
                or continuation_incomplete_actions or continuation_policy_factory is not None
                or turn_chain_continuation is not None or complete_orderer is not None
                or validation_tail_lookup is not None or single_node_expansion
                or stop_after_card_roots or root_coverage_first or progressive_widening
                or equivalent_instance_reuse)):
            raise ValueError('mcts_finalist_validation_requires_original_256_four_paired_search')
        if (type(public_turn_reanalysis) is not bool or public_turn_reanalysis and (
                simulations != 256 and not compute_sensitivity
                or min_root_samples != 4 or validation_samples != 4
                or not scenario_reuse or not adaptive_validation
                or timeout_ms > (300_000 if quality_reference else 30_000)
                or fresh_adaptive_training or continuation_samples or continuation_domain
                or continuation_incomplete_actions or continuation_policy_factory is not None
                or turn_chain_continuation is not None or complete_orderer is not None
                or validation_tail_lookup is not None or single_node_expansion
                or root_coverage_first or progressive_widening
                or equivalent_instance_reuse or paired_finalist_validation)):
            raise ValueError('mcts_public_turn_reanalysis_requires_original_256_four_paired_search')
        if (score_translation_contract not in (None, 'hif-score-translation-v1')
                or score_translation_contract is not None and (not public_turn_reanalysis
                    or public_turn_installed_stack)):
            raise ValueError('mcts_public_turn_reanalysis_invalid_score_translation_contract')
        if (type(public_turn_budget_reservation) is not bool
                or public_turn_budget_reservation and not public_turn_reanalysis):
            raise ValueError('mcts_public_turn_budget_reservation_requires_public_turn_reanalysis')
        if (type(public_validation_phase_budget) is not bool
                or public_validation_phase_budget and (
                    not public_turn_budget_reservation or not adaptive_validation
                    or not lazy_validation_seeds or min_root_samples != 4
                    or validation_samples != 4 or soft_timeout_ms is None
                    or type(soft_timeout_ms) not in (int, float))):
            raise ValueError('mcts_public_validation_phase_requires_reserved_paired_validation')
        if (getattr(unknown_node_policy, 'extra_action_policy', None) is not None
                and not public_validation_phase_budget):
            raise ValueError('mcts_extra_action_chain_requires_public_validation_phase')
        if (type(parallel_public_graph) is not bool or parallel_public_graph and (
                not public_validation_phase_budget
                or getattr(unknown_node_policy, 'extra_action_policy', None) is None)):
            raise ValueError('mcts_parallel_graph_requires_extra_action_validation_phase')
        if terminal_eligible is not None and (
                not callable(terminal_eligible) or validation_tail_lookup is not None):
            raise ValueError('terminal_eligible must be callable and cannot use frozen tail lookup')
        if (type(terminal_eligibility_horizons) is not tuple or not terminal_eligibility_horizons
                or any(type(horizon) is not int or horizon not in (1, 2)
                       for horizon in terminal_eligibility_horizons)
                or len(set(terminal_eligibility_horizons)) != len(terminal_eligibility_horizons)
                or terminal_eligibility_horizons != (2,) and (
                    terminal_eligible is None or max(terminal_eligibility_horizons) > terminal_horizon)):
            raise ValueError('invalid_terminal_eligibility_horizons')
        self.kernel, self.advice = kernel, advice
        self.simulations, self.timeout_ms, self.seed = simulations, timeout_ms, seed
        self.min_root_samples, self.reward_scale = min_root_samples, reward_scale
        self.terminal_evaluator, self.sampler_factory = terminal_evaluator, sampler_factory
        self.validation_terminal_evaluator = validation_terminal_evaluator
        self.validation_tail_lookup, self.freeze_callback = validation_tail_lookup, freeze_callback
        # Public structure only: the callback receives no seed, deadline or
        # solver cache, and must make the same decision during validation.
        self.terminal_eligible = terminal_eligible
        # The legacy (2,) default stays inactive for terminal_horizon=1.
        self.terminal_eligibility_horizons = terminal_eligibility_horizons
        self.soft_timeout_ms = soft_timeout_ms
        self.validation_samples = validation_samples
        self.adaptive_validation = adaptive_validation
        self.scenario_reuse = scenario_reuse or bool(validation_samples)
        self.dominance_pruning = dominance_pruning
        self.progressive_widening = progressive_widening
        self.complete_orderer = complete_orderer
        self.terminal_horizon = terminal_horizon
        self.continuation_samples = continuation_samples
        self.continuation_domain = continuation_domain
        self.continuation_incomplete_actions = continuation_incomplete_actions
        self.fresh_adaptive_training = fresh_adaptive_training
        self.continuation_policy_factory = continuation_policy_factory
        self.continuation_portfolio = continuation_policy_factory is not None
        self.turn_chain_continuation = turn_chain_continuation
        self.stop_after_card_roots = stop_after_card_roots
        self.root_coverage_first = root_coverage_first
        self.single_node_expansion = single_node_expansion
        self.equivalent_instance_reuse = equivalent_instance_reuse
        self.paired_finalist_validation = paired_finalist_validation
        self.public_turn_reanalysis = public_turn_reanalysis
        self.public_turn_budget_reservation = public_turn_budget_reservation
        self.public_validation_phase_budget = public_validation_phase_budget
        self.parallel_public_graph = parallel_public_graph
        self.resource_reanalysis = resource_reanalysis
        self.score_translation_contract = score_translation_contract
        self.unknown_node_policy = unknown_node_policy
        self.retrieval_training_orderer = retrieval_training_orderer
        self.lazy_validation_seeds = lazy_validation_seeds
        self.deterministic_validation_cohort = deterministic_validation_cohort
        self.partial_node_timing = partial_node_timing
        self.reset()

    def reset(self):
        self._internal_policy = None
        self._tree = {}
        self._namespace = None
        self._rng = random.Random(self.seed)
        # This stream cannot shift the existing paired/outer/inner cohorts.
        self._adaptive_rng = random.Random(f'hif-adaptive-training:{self.seed}')
        self._expected_state = None
        self._order_deadline = 0.
        self._validation_tail_hits = self._validation_tail_misses = 0
        self._training_terminal_eligible = self._training_terminal_declined = 0
        self._validation_terminal_eligible = self._validation_terminal_declined = 0
        self._continuation = None
        self._outer_continuation = None
        self._public_turn_policy = None
        self._resource_policy = None
        self._public_step_reuse = None
        self._effective_continuation_samples = self.continuation_samples
        self._domain_limit = None
        self._baseline_domain_reason = None
        self._unimproved_count = self._outer_unimproved_count = 0
        self._partial_node_calls = self._partial_node_cache_hits = self._partial_node_new_actions = 0
        self._reset_training_expansion()
        self._reset_validation_seed_allocation()
        if self.deterministic_validation_cohort:
            self._training_seed_history = set()
        self._reset_validation_cohort_evidence()
        self._reset_partial_node_timing()
        self._reset_public_turn_budget_reservation()
        if self.complete_orderer is not None:
            self.complete_orderer.reset()
        if self.turn_chain_continuation is not None:
            self.turn_chain_continuation.start_window()
        if self.unknown_node_policy is not None:
            self.unknown_node_policy.reset()
        if self.retrieval_training_orderer is not None:
            self.retrieval_training_orderer.reset()

    def _reset_public_turn_budget_reservation(self):
        self._reset_public_validation_phase_budget()
        if self.public_turn_budget_reservation:
            self.last_public_turn_budget_reservation = dict(
                version=PUBLIC_TURN_BUDGET_RESERVATION_VERSION,
                eligible=False, complete_graph=False, required_simulations=None,
                required_simulations_lower_bound=0, training_budget_before=0,
                training_budget_after=0, minimum_training=0, minimum_validation=0,
                graph_rollout_budget=0, structure_prepared_once=False,
                reason='not_run', prepare_elapsed_ms=0.)

    def _reset_public_validation_phase_budget(self):
        if self.public_validation_phase_budget:
            self.last_public_validation_phase_budget = dict(
                version=PUBLIC_VALIDATION_PHASE_BUDGET_VERSION, eligible=False, reason='not_run',
                soft_window_ms=0., training_elapsed_ms=0., graph_elapsed_ms=0.,
                validation_start_elapsed_ms=0., soft_deadline_elapsed_ms=0.,
                validation_elapsed_ms=0., complete_rounds=0)
            if self.quality_reference:
                self.last_public_validation_phase_budget.update(
                    allocation_mode='complete_planned_work', soft_disabled=True)

    def _reset_partial_node_timing(self):
        if self.partial_node_timing:
            self.last_partial_node_timing = dict(version=PARTIAL_NODE_TIMING_VERSION,
                calls=0, chosen=0, changed=0, declined=0)

    def _reset_validation_seed_allocation(self):
        if self.lazy_validation_seeds:
            self.last_validation_seed_allocation = dict(version=LAZY_VALIDATION_SEEDS_VERSION,
                capacity=0, mandatory=4, generated=0, started=0, complete_rounds=0)

    def _reset_validation_cohort_evidence(self, supplied_seeds=()):
        if self.deterministic_validation_cohort:
            self.last_validation_cohort_evidence = dict(
                version=DETERMINISTIC_VALIDATION_COHORT_VERSION,
                supplied_seeds=list(supplied_seeds), training_seeds=[],
                training_seed_history=sorted(self._training_seed_history), validation_seeds=[],
                reused_count=0, new_seed_count=0,
                scope=('TRAINING_DISJOINT_REUSED_SCENARIOS' if supplied_seeds else
                       'FRESH_TRAINING_DISJOINT_SCENARIOS'))

    def _reset_training_expansion(self):
        self._training_expansion = dict(started_training_rollouts=0, completed_training_rollouts=0,
            started_free_expansions=0, completed_free_expansions=0,
            started_prior_route_steps=0, completed_prior_route_steps=0,
            max_free_expansions_per_rollout=0)
        self._last_training_expansion = [0, 0]
        self._validation_started = 0

    def _fresh_seeds(self, count, excluded=()):
        used, seeds = set(excluded), []
        while len(seeds) < count:
            seed = self._rng.randrange(2**31)
            if seed not in used:
                used.add(seed)
                seeds.append(seed)
        return tuple(seeds)

    def export_search_cache(self, request, *, version_context):
        from .hif_mcts_cache import export_tree

        request = replace(request, state=self.kernel._canonical_probability_state(request.state))
        return export_tree(self, request, version_context)

    def restore_search_cache(self, payload, request, *, version_context):
        from .hif_mcts_cache import restore_tree

        try:
            return restore_tree(self, payload, request, version_context)
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            raise ValueError('mcts_cache_malformed') from error

    def _context(self, request):
        return (self.kernel.ruleset_id, request.plan_type, request.utility,
                tuple(sorted(self.kernel.cards.items())), tuple(sorted(self.kernel.drinks.items())),
                tuple(sorted(self.kernel.items.items())), tuple(sorted(self.kernel.memory_abilities.items())),
                id(self.advice), self.reward_scale, id(self.terminal_evaluator), self.scenario_reuse,
                self.validation_samples, self.dominance_pruning, self.adaptive_validation,
                self.progressive_widening, id(self.complete_orderer), self.terminal_horizon,
                id(self.validation_terminal_evaluator), id(self.validation_tail_lookup), id(self.freeze_callback),
                id(self.terminal_eligible), self.terminal_eligibility_horizons, self.continuation_samples,
                self.continuation_domain, self._baseline_domain_reason) + (
                    ('quality_reference', QUALITY_REFERENCE_VERSION) if self.quality_reference else ()) + (
                    ('revealed_policy_competition', REVEALED_POLICY_COMPETITION_VERSION)
                    if self.revealed_policy_competition else ()) + (
                    ('compute_sensitivity', COMPUTE_SENSITIVITY_VERSION, self.simulations)
                    if self.compute_sensitivity else ()) + (
                    ('continuation_incomplete_actions', True) if self.continuation_incomplete_actions else ()) + (
                    ('fresh_adaptive_training', True) if self.fresh_adaptive_training else ()) + (
                    ('continuation_portfolio', id(self.continuation_policy_factory))
                    if self.continuation_portfolio else ()) + (
                    ('turn_chain_continuation', id(self.turn_chain_continuation))
                    if self.turn_chain_continuation is not None else ()) + (
                    ('stop_after_card_roots', True) if self.stop_after_card_roots else ()) + (
                    ('single_node_expansion', True) if self.single_node_expansion else ()) + (
                    ('equivalent_instance_reuse', self._instance_view_version)
                    if self.equivalent_instance_reuse else ()) + (
                    ('paired_finalist_validation', True) if self.paired_finalist_validation else ()) + (
                    ('uniform_validation_time_guard', self._uniform_validation_time_guard_version)
                    if self.uniform_validation_time_guard else ()) + (
                    ('paired_adaptive_validation', self._paired_adaptive_validation_version)
                    if self.paired_adaptive_validation else ()) + (
                    ('public_turn_reanalysis', True, self._public_turn_reanalysis_version,
                     self.score_translation_contract)
                    if self.public_turn_reanalysis else ()) + (
                    ('public_turn_optional_end_roots', PUBLIC_TURN_OPTIONAL_END_ROOTS_VERSION)
                    if self.public_turn_reanalysis and self.stop_after_card_roots else ()) + (
                    ('public_turn_budget_reservation', PUBLIC_TURN_BUDGET_RESERVATION_VERSION)
                    if self.public_turn_budget_reservation else ()) + (
                    ('public_validation_phase_budget', PUBLIC_VALIDATION_PHASE_BUDGET_VERSION)
                    if self.public_validation_phase_budget else ()) + (
                    ('parallel_public_graph', self._parallel_public_graph_version)
                    if self.parallel_public_graph else ()) + (
                    ('parallel_root_validation', self._parallel_root_validation_version)
                    if self.parallel_public_graph and (self.paired_adaptive_validation or self.uniform_validation_time_guard) else ()) + (
                    ('resource_reanalysis', self._resource_reanalysis_version)
                    if self.resource_reanalysis else ()) + (
                    ('internal_reanalysis', self._internal_reanalysis_version)
                    if self.internal_reanalysis else ()) + (
                    ('internal_successor_graph', self._internal_successor_graph_version)
                    if self.internal_successor_graph else ()) + (
                    ('internal_complete_root_successors', self._internal_complete_root_successors_version)
                    if self.internal_complete_root_successors else ()) + (
                    ('unknown_node_timing', self.unknown_node_policy.version,
                     self.unknown_node_policy.timing_version)
                    if self.unknown_node_policy is not None else ()) + (
                    ('retrieval_training_order', self.retrieval_training_orderer.version,
                     self.retrieval_training_orderer.unknown_version,
                     self.retrieval_training_orderer.timing_version)
                    if self.retrieval_training_orderer is not None else ()) + (
                    ('lazy_validation_seeds', LAZY_VALIDATION_SEEDS_VERSION)
                    if self.lazy_validation_seeds else ()) + (
                    ('deterministic_validation_cohort', DETERMINISTIC_VALIDATION_COHORT_VERSION)
                    if self.deterministic_validation_cohort else ()) + (
                    ('partial_node_timing', PARTIAL_NODE_TIMING_VERSION)
                    if self.partial_node_timing else ()) + (
                    ('installed_score_turn_chain', self.unknown_node_policy.installed_score_policy.version)
                    if getattr(self.unknown_node_policy, 'installed_score_policy', None) is not None else ()) + (
                    ('extra_action_turn_chain', self.unknown_node_policy.extra_action_policy.version)
                    if getattr(self.unknown_node_policy, 'extra_action_policy', None) is not None else ()) + (
                    ('pair_order_advice', self.unknown_node_policy.pair_order_policy.version)
                    if getattr(self.unknown_node_policy, 'pair_order_policy', None) is not None else ())

    @property
    def _parallel_root_validation_version(self):
        from .hif_root_validation_parallel import VERSION

        return VERSION

    @property
    def _parallel_public_graph_version(self):
        from .hif_public_graph_parallel import VERSION

        return VERSION

    @property
    def _resource_reanalysis_version(self):
        from .hif_resource_reanalysis import VERSION

        return VERSION

    @property
    def _internal_reanalysis_version(self):
        from .hif_internal_reanalysis import VERSION

        return VERSION

    @property
    def _uniform_validation_time_guard_version(self):
        from .hif_uniform_validation_time_guard import VERSION

        return VERSION

    @property
    def _paired_adaptive_validation_version(self):
        from .hif_paired_adaptive_validation import VERSION

        return VERSION

    @property
    def _internal_successor_graph_version(self):
        from .hif_internal_successor_graph import VERSION

        return VERSION

    @property
    def _internal_complete_root_successors_version(self):
        from .hif_internal_reanalysis import COMPLETE_ROOT_SUCCESSORS_VERSION

        return COMPLETE_ROOT_SUCCESSORS_VERSION

    @property
    def _public_turn_reanalysis_version(self):
        from .hif_public_turn_reanalysis import VERSION

        return VERSION

    @property
    def _instance_view_version(self):
        from .hif_instance_view import VERSION
        return VERSION

    def _state_key(self, state, slots):
        if self.equivalent_instance_reuse:
            from .hif_instance_view import instance_view
            # Preserve all source/history references in this opt-in view.
            return replace(instance_view(state)[0], score=0), tuple(slots)
        # Retain all physical identities, timers, choices and card zones.
        # Complete observations retain exhausted counters. Once persistent
        # effects are initialized, legal_actions/transition prune these same
        # inert records before every action. Normalize only the search key;
        # the complete observation and expected business state stay intact.
        if state.persistent_effects_initialized:
            state = PythonNiaProKernel._prune_exhausted_schedules(state)
        # Score translation is valid for HIF.
        return replace(state, score=0), tuple(slots)

    def _retain(self, root):
        pending, retained = [root], {}
        while pending:
            key = pending.pop()
            if key in retained or key not in self._tree:
                continue
            node = self._tree[key]
            retained[key] = node
            pending.extend(node.children)
        self._tree = retained

    def committed(self, before, action, after):
        # The next request supplies freshly bound physical slots. Only its
        # observed successor determines which old subtree can survive.
        before = self.kernel._canonical_probability_state(before)
        if self._expected_state is not None and before != self._expected_state:
            self._tree.clear()
        self._expected_state = self.kernel._canonical_probability_state(after)

    def _action_order(self, request, actions):
        keys = tuple(action_key(action) for action in actions)
        if self.complete_orderer is not None:
            complete = self.complete_orderer.complete_order(
                request, actions, deadline=self._order_deadline)
            if complete is not None:
                if len(complete) != len(keys) or set(complete) != set(keys):
                    raise ValueError('mcts_complete_order_action_mismatch')
                return complete
        order = list(range(len(actions)))
        if self.advice is not None:
            priors = tuple(self.advice.action_priors(request.state, actions))
            if len(priors) != len(actions) or not all(map(math.isfinite, priors)):
                raise ValueError('mcts_invalid_action_priors')
            order.sort(key=lambda index: -priors[index])
        return tuple(keys[index] for index in order)

    def _physical_node(self, request, actions, *, create=False, training_order_allowed=False):
        """Expose the shared statistics through this state's real legal IDs."""
        key = self._state_key(request.state, request.drink_slots)
        node = self._tree.get(key)
        keys = tuple(action_key(action) for action in actions)
        mapping = {}
        if self.equivalent_instance_reuse:
            from .hif_instance_view import action_view, instance_view
            mapping = instance_view(request.state)[1]
        canonical = tuple(action_view(value, mapping) for value in keys) if mapping else keys
        if len(set(canonical)) != len(keys):
            raise ValueError('mcts_instance_action_mapping_not_bijective')
        forward = dict(zip(keys, canonical, strict=True))
        if node is None and create:
            order = self._action_order(request, actions)
            if training_order_allowed and self.retrieval_training_orderer is not None:
                order = self.retrieval_training_orderer.reorder(
                    request, actions, order, self._order_deadline)
                if len(order) != len(keys) or set(order) != set(keys):
                    raise ValueError('mcts_retrieval_training_order_action_mismatch')
            order = tuple(forward[value] for value in order)
            node = _Node({value: [0, 0.] for value in canonical}, order)
            self._tree[key] = node
        elif node is not None and set(canonical) != set(node.statistics):
            raise ValueError('mcts_public_state_action_mismatch')
        if node is not None and mapping:
            reverse = dict(zip(canonical, keys, strict=True))
            # The lists and child set are shared, not copied evaluations.
            node = _Node({reverse[value]: stats for value, stats in node.statistics.items()},
                         tuple(reverse[value] for value in node.order), node.children)
        return key, node

    def _node(self, request, actions, *, training_order_allowed=False):
        if training_order_allowed:
            return self._physical_node(request, actions, create=True, training_order_allowed=True)
        return self._physical_node(request, actions, create=True)

    def _select(self, node, *, progressive=False):
        total = sum(row[0] for row in node.statistics.values())
        order = node.order
        if progressive:
            # Keep every legal action; only delay opening low-prior actions
            # at deep nodes until completed visits support a wider search.
            width = min(len(order), max(2, math.ceil(math.sqrt(total + 1))))
            order = order[:width]
        for key in order:
            if node.statistics[key][0] == 0:
                return key
        return max(order, key=lambda key:
                   node.statistics[key][1] / node.statistics[key][0] / self.reward_scale
                   + math.sqrt(2 * math.log(total + 1) / node.statistics[key][0]))

    def _validation_action(self, request, actions):
        # Read only the frozen public-state tree. Neither the scenario seed
        # nor unobserved draws are available to this action selector.
        if self._internal_policy is not None:
            from .hif_internal_reanalysis import lookup

            selected = lookup(self._internal_policy, request, actions, context=self._context(request), kernel=self.kernel)
            if selected is not None:
                return selected
        _, node = self._physical_node(request, actions)
        keys = {action_key(action) for action in actions}
        if node is not None and keys != set(node.statistics):
            raise ValueError('mcts_public_state_action_mismatch')
        if (self.turn_chain_continuation is not None and len(actions) > 1
                and (node is None or any(not row[0] for row in node.statistics.values()))):
            # Sparse complete-return statistics cannot compare every action.
            # A bounded public-state turn plan can propose the continuation;
            # only the outer complete rollout supplies this root's score.
            selected = self.turn_chain_continuation.choose(
                request, actions, deadline=self._order_deadline)
            if selected is not None:
                if selected not in keys:
                    raise ValueError('mcts_turn_chain_action_not_uniquely_legal')
                return selected
        if node is None:
            if self.unknown_node_policy is not None:
                selected = self.unknown_node_policy.choose(request, actions, self._order_deadline)
                if selected is not None:
                    if sum(action_key(action) == selected for action in actions) != 1:
                        raise ValueError('mcts_unknown_timing_action_not_uniquely_legal')
                    return selected
            return self._action_order(request, actions)[0]
        visited = [key for key in node.order if node.statistics[key][0]]
        if not visited:
            return node.order[0]
        original = max(visited, key=lambda key: node.statistics[key][1] / node.statistics[key][0])
        if self.partial_node_timing and len(visited) < len(actions):
            selected = self.unknown_node_policy.choose(request, actions, self._order_deadline)
            if selected is not None and sum(action_key(action) == selected for action in actions) != 1:
                raise ValueError('mcts_partial_timing_action_not_uniquely_legal')
            # Completed helper calls only; failures propagate and cannot be
            # counted as a successful suggestion or an explicit scope decline.
            self.last_partial_node_timing['calls'] += 1
            self.last_partial_node_timing['chosen' if selected is not None else 'declined'] += 1
            if selected is not None:
                self.last_partial_node_timing['changed'] += int(selected != original)
                return selected
        extra = getattr(self.unknown_node_policy, 'extra_action_policy', None)
        if (len(visited) == len(actions) and original[0] == ActionKind.USE_DRINK.value
                and getattr(extra, 'unused_drink_enabled', False)):
            proposed = next(action for action in actions if action_key(action) == original)
            proof = extra.unused_drink_proof(request, proposed, self._order_deadline)
            if proof is not None:
                # Keep the frozen ordering among the other fully covered
                # choices. This does not mutate the tree or root coverage.
                retained = set(proof['retained_legal_action_keys'])
                if retained != keys - {original}:
                    raise ValueError('mcts_unused_drink_proof_action_mismatch')
                return max((key for key in visited if key in retained),
                           key=lambda key: node.statistics[key][1] / node.statistics[key][0])
        if len(visited) == len(actions) and (original[0] == ActionKind.END_TURN.value
                or original[0] == ActionKind.PLAY_CARD.value
                and getattr(extra, 'visited_card_extra_action', False)):
            # Visiting every edge does not make its unpaired returns a complete
            # comparison of continuing this turn. Reuse the opt-in bounded
            # comparison when it can prove an action-preserving alternative.
            # The separate card opt-in retains the helper's last-AP proof.
            # END remains a candidate; a domain decline keeps the tree choice.
            if extra is not None:
                selected = extra.choose(request, actions, self._order_deadline, proposed_key=original)
                if selected is not None:
                    if sum(action_key(action) == selected for action in actions) != 1:
                        raise ValueError('mcts_extra_action_chain_action_not_uniquely_legal')
                    return selected
        return original

    def _prepare_root_options(self, request, actions, options, deadline):
        if self.min_root_samples < 4 or self.validation_samples < 4:
            raise ValueError('mcts_root_options_require_paired_validation')
        legal = {action_key(action): action for action in actions}
        result, grouped = [], {}
        for option in options:
            continuation = option.continuation
            if (type(continuation) is not tuple or len(continuation) > 1
                    or bool(continuation) != (option.expected_after_first is not None)
                    or action_key(option.action) not in legal):
                raise ValueError('mcts_invalid_root_option')
            expected = (self.kernel._canonical_probability_state(option.expected_after_first)
                        if continuation else None)
            row = _RootOption(legal[action_key(option.action)], continuation, expected)
            result.append(row)
            grouped.setdefault(action_key(row.action), []).append(row)
        if (set(grouped) != set(legal) or len({option.key for option in result}) != len(result)):
            raise ValueError('mcts_root_options_coverage_mismatch')
        for key, group in grouped.items():
            if time.perf_counter() >= deadline:
                raise TimeoutError('mcts_window_timeout')
            first = group[0]
            optional_end = [option for option in group if option.continuation
                            and option.continuation[0].kind is ActionKind.END_TURN]
            if optional_end:
                from .hif_choice_roots import post_card_end_state
                from .validation_cache import validation_session

                if (not self.stop_after_card_roots or len(group) != 2 or len(optional_end) != 1
                        or sum(not option.continuation for option in group) != 1):
                    raise ValueError('mcts_optional_end_requires_plain_root')
                def check_budget():
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('mcts_window_timeout')
                with validation_session(check_budget, reuse_parent=True):
                    expected, _ = post_card_end_state(self.kernel, request.state, legal[key],
                                                     plan_type=request.plan_type)
                if (expected is None or optional_end[0].expected_after_first != expected
                        or optional_end[0].continuation[0] != GameAction(ActionKind.END_TURN)):
                    raise ValueError('mcts_optional_end_not_deterministic_public_state')
                continue
            if not first.continuation:
                if len(group) != 1:
                    raise ValueError('mcts_root_options_mixed_prefix')
                continue
            expected = first.expected_after_first
            if (expected.pending_choice is None or expected.terminal
                    or expected.pending_choice.action_kind is not ActionKind.MOVE_CARD
                    or expected.pending_choice.selections_remaining != 1
                    or expected.pending_choice.selection_optional
                    or any(not option.continuation or option.expected_after_first != expected for option in group)):
                raise ValueError('mcts_root_option_not_mandatory_choice')
            # A supplied target cannot encode the result of a hidden draw.
            # Prove the public intermediate state without a sampled kernel.
            branches = self.kernel.transition(request.state, replace(legal[key], source_slot=None),
                                              plan_type=request.plan_type)
            if (len(branches) != 1 or branches[0].probability != 1
                    or self.kernel._canonical_probability_state(branches[0].state) != expected):
                raise ValueError('mcts_root_option_not_deterministic')
            secondary = self.kernel.legal_actions(expected, plan_type=request.plan_type,
                                                  decision_kind=DecisionKind.SECONDARY)
            if {action_key(option.continuation[0]) for option in group} != {
                    action_key(action) for action in secondary}:
                raise ValueError('mcts_root_options_secondary_coverage_mismatch')
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_window_timeout')
        return tuple(result)

    def _simulate(self, request, seed, forced_root, deadline):
        return self._rollout(request, seed, forced_root, deadline, validation=False)

    def _validate_simulation(self, request, seed, forced_root, deadline, *, prefix_depth=0, public_prefix=None):
        if public_prefix is not None:
            if self._effective_continuation_samples or prefix_depth:
                raise ValueError('mcts_public_prefix_requires_nonrecursive_full_root_return')
            return self._rollout(request, seed, forced_root, deadline, validation=True, public_prefix=public_prefix)
        if self._effective_continuation_samples:
            return self._rollout(request, seed, forced_root, deadline, validation=True, continuation=True)
        return self._rollout(request, seed, forced_root, deadline, validation=True, prefix_depth=prefix_depth)

    def _public_turn_action(self, cursor, request, actions):
        return cursor.choose(request, actions)

    def _public_turn_supports_root_options(self, options):
        # suggest has already checked complete coverage and proved every
        # optional END prefix in _prepare_root_options. Required MOVE choices
        # remain outside this combination's public-graph domain.
        return options is None or self.stop_after_card_roots and all(
            isinstance(option, _RootOption)
            and (not option.continuation or option.continuation == (GameAction(ActionKind.END_TURN),))
            for option in options)

    def _prepare_public_turn_budget(self, request, *, root_options, training_budget,
                                    minimum_training, minimum_validation, deadline):
        from .hif_chain_candidate import PairedPlanningKernel
        from .hif_public_turn_reanalysis import new_diagnostics, prepare_public_turn_graph

        evidence = self.last_public_turn_budget_reservation
        graph_budget = max(0, self.simulations - minimum_training - minimum_validation)
        evidence.update(training_budget_before=training_budget, training_budget_after=training_budget,
                        minimum_training=minimum_training, minimum_validation=minimum_validation,
                        graph_rollout_budget=graph_budget)
        diagnostics = new_diagnostics()
        if not self._public_turn_supports_root_options(root_options) or self.sampler_factory not in (None, PairedPlanningKernel):
            diagnostics.update(status='declined', reason='unsupported_root_or_sampler')
            evidence['reason'] = diagnostics['reason']
            return None, diagnostics, training_budget, None
        started = time.perf_counter()
        evidence['structure_prepared_once'] = True
        try:
            prepared = prepare_public_turn_graph(self.kernel, request, rollout_budget=graph_budget,
                deadline=deadline, diagnostics=diagnostics,
                score_translation_contract=self.score_translation_contract)
            if time.perf_counter() >= deadline:
                raise TimeoutError('mcts_public_turn_prepare_timeout')
        except (TimeoutError, ValueError, RuntimeError) as error:
            diagnostics.update(status='failed', complete=False, reason=str(error))
            evidence.update(reason=str(error), required_simulations_lower_bound=diagnostics['required_simulations'])
            return None, diagnostics, training_budget, 'mcts_public_turn_reanalysis_failed:' + str(error)
        finally:
            evidence['prepare_elapsed_ms'] = (time.perf_counter() - started) * 1000
        evidence.update(eligible=prepared.ready, complete_graph=prepared.ready,
                        required_simulations=prepared.required_simulations,
                        required_simulations_lower_bound=prepared.required_simulations_lower_bound,
                        reason='reserved' if prepared.ready else diagnostics['reason'])
        if prepared.ready:
            training_budget = min(training_budget,
                                  self.simulations - prepared.required_simulations - minimum_validation)
            if training_budget < minimum_training:
                raise ValueError('mcts_public_turn_reserved_training_coverage_incomplete')
            evidence['training_budget_after'] = training_budget
        return prepared, diagnostics, training_budget, None

    def _new_public_step_reuse(self, deadline):
        """Only the known stateless helpers may share a frozen graph's work."""
        from .hif import HifSenseKernel
        from .hif_combo_timing import SenseComboTimingPolicy
        from .hif_mcts_candidate import HifMctsCandidatePolicy
        from .hif_chain_candidate import PairedPlanningKernel
        from .hif_pair_order_advice import PairOrderAdvice
        from .hif_public_step_reuse import PublicStepReuse
        from .hif_turn_chain_continuation import TurnChainContinuation
        from .hif_extra_action_continuation import ExtraActionTurnChain

        def original(instance, owner, *names):
            return type(instance) is owner and all(
                getattr(getattr(instance, name, None), '__self__', None) is instance
                and getattr(getattr(instance, name, None), '__func__', None) is getattr(owner, name)
                for name in names)

        if (type(self.kernel) is not HifSenseKernel
                or self.sampler_factory is not PairedPlanningKernel
                or self.score_translation_contract != 'hif-score-translation-v1'
                or self._public_turn_policy is not None or self.complete_orderer is not None
                or not original(self, OfflineHifMcts, '_validation_action', '_physical_node', '_action_order')):
            return None
        qualified = getattr(self, '_public_step_reuse_check', None)
        owner = getattr(qualified, '__self__', None)
        if (type(owner) is not HifMctsCandidatePolicy or owner.search is not self
                or getattr(qualified, '__func__', None) is not HifMctsCandidatePolicy._public_score_translation_contract):
            return None
        # Score translation alone does not make arbitrary raw-search advice
        # stateless. Recheck the concrete candidate's full advice stack.
        if qualified() != 'hif-score-translation-v1':
            return None
        unknown = self.unknown_node_policy
        if unknown is not None:
            if (not original(unknown, UnknownNodeTimingPolicy, 'choose', '_choose')
                    or not original(unknown.timing, SenseComboTimingPolicy, 'choose', 'reset')):
                return None
            for helper, owner, method in (
                    (unknown.installed_score_policy, InstalledScoreTurnChain, 'choose'),
                    (unknown.extra_action_policy, ExtraActionTurnChain, 'choose'),
                    (unknown.pair_order_policy, PairOrderAdvice, 'improve')):
                if helper is not None and not original(helper, owner, method):
                    return None
                for name in ('turn_chain', 'drink_turn_chain'):
                    chain = getattr(helper, name, None)
                    if chain is not None and not original(chain, TurnChainContinuation, 'choose'):
                        return None
        if (self.turn_chain_continuation is not None
                and not original(self.turn_chain_continuation, TurnChainContinuation, 'choose')):
            return None
        return PublicStepReuse(self.kernel, deadline, complete_returns=True)

    def _reanalyse_public_turn(self, request, seeds, *, root_options, budget, deadline,
                              prepared=None, diagnostics=None):
        from .hif_chain_candidate import PairedPlanningKernel
        from .hif_public_turn_reanalysis import (
            new_diagnostics,
            build_public_turn_policy,
            evaluate_prepared_public_turn_graph,
        )

        if diagnostics is None:
            diagnostics = new_diagnostics()
        maximum_depth = terminal_calls = terminal_fallbacks = 0
        if not self._public_turn_supports_root_options(root_options) or self.sampler_factory not in (None, PairedPlanningKernel):
            diagnostics.update(status='declined', reason='unsupported_root_or_sampler')
            return diagnostics, maximum_depth, terminal_calls, terminal_fallbacks

        parallel_rows = None
        consumed = 0

        def evaluate(current, seed, key, limit):
            nonlocal maximum_depth, terminal_calls, terminal_fallbacks, consumed
            names = ('_validation_tail_hits', '_validation_tail_misses',
                     '_validation_terminal_eligible', '_validation_terminal_declined')
            if parallel_rows is not None:
                group, world = divmod(consumed, 4)
                row = parallel_rows[world]['rows'][group]
                representative = prepared.groups[group][0]
                if (current != representative['request'] or seed != seeds[world]
                        or row['group_index'] != group or row['seed'] != seed
                        or row['key'] != key
                        or row['prefix_depth'] != diagnostics['evaluation_prefix_depth']):
                    raise ValueError('mcts_parallel_graph_result_binding_mismatch')
                maximum_depth = max(maximum_depth, row['depth'])
                terminal_calls += row['terminal_calls']
                terminal_fallbacks += row['terminal_fallbacks']
                for name in names:
                    field_name = name.removeprefix('_validation_')
                    diagnostics[field_name] = diagnostics.get(field_name, 0) + row['counters'][name]
                consumed += 1
                return row['final_score']
            before = tuple(getattr(self, name) for name in names)
            try:
                score, path, depth, calls, fallbacks = self._validate_simulation(
                    current, seed, key, limit, prefix_depth=diagnostics['evaluation_prefix_depth'])
                if path:
                    raise ValueError('mcts_public_turn_reanalysis_mutable_path')
                maximum_depth = max(maximum_depth, depth)
                terminal_calls += calls
                terminal_fallbacks += fallbacks
                return score
            finally:
                for name, original in zip(names, before):
                    field_name = name.removeprefix('_validation_')
                    diagnostics[field_name] = diagnostics.get(field_name, 0) + getattr(self, name) - original
                    setattr(self, name, original)

        # Training is already frozen here. Discard the memo before outer
        # validation (whose public cursor selects a different continuation).
        reuse = self._new_public_step_reuse(deadline)
        self._public_step_reuse = reuse
        try:
            if self.parallel_public_graph and prepared is not None and prepared.ready:
                from .hif_public_graph_parallel import MIN_SIMULATIONS, run_worlds

                if prepared.required_simulations >= MIN_SIMULATIONS:
                    if reuse is None:
                        raise ValueError('mcts_parallel_graph_unqualified_policy')
                    owner = self._public_step_reuse_check.__self__
                    # Each worker has its own complete memo and tail sessions.
                    # Parent helper summaries continue to count parent work only.
                    self._public_step_reuse = reuse = None
                    parallel_rows = run_worlds(owner, request, prepared, seeds, deadline, diagnostics)
            if prepared is None:
                self._public_turn_policy = build_public_turn_policy(
                    self.kernel, request, seeds, evaluate=evaluate, rollout_budget=budget,
                    deadline=deadline, diagnostics=diagnostics,
                    score_translation_contract=self.score_translation_contract)
            else:
                self._public_turn_policy = evaluate_prepared_public_turn_graph(
                    prepared, seeds, evaluate=evaluate, deadline=deadline, diagnostics=diagnostics)
            if parallel_rows is not None and consumed != prepared.required_simulations:
                raise ValueError('mcts_parallel_graph_unconsumed_results')
        except (TimeoutError, ValueError, RuntimeError, OSError) as error:
            diagnostics.update(status='failed', complete=False, reason=str(error))
            parallel = diagnostics.get('parallel_public_graph')
            if parallel is not None:
                # A failed worker batch cannot be retried sequentially. Preserve
                # actual reported work even when no graph values were accepted.
                diagnostics['started_simulations'] = parallel['started_simulations']
                diagnostics['completed_simulations'] = parallel['completed_simulations']
            self._public_turn_policy = None
        finally:
            self._public_step_reuse = None
            if reuse is not None:
                diagnostics['step_reuse'] = reuse.summary()
        return diagnostics, maximum_depth, terminal_calls, terminal_fallbacks

    def _resource_key(self, request):
        return self._state_key(request.state, request.drink_slots)

    def _prepare_resources(self, request, *, budget, deadline, diagnostics,
                           retain_budget_proof=False, previous=None):
        from .hif_resource_reanalysis import prepare_resource_graphs

        return prepare_resource_graphs(self.kernel, request, states=tuple(self._tree),
            key=self._resource_key,
            budget=budget, deadline=deadline, diagnostics=diagnostics,
            score_translation_contract=self.score_translation_contract,
            retain_budget_proof=retain_budget_proof, previous=previous,
            normalization_context=self._context(request))

    def _reanalyse_resources(self, request, seeds, *, budget, deadline, prepared=None,
                             diagnostics=None, previous=None):
        from .hif_resource_reanalysis import (
            new_diagnostics,
            evaluate_resource_graphs,
        )

        diagnostics = new_diagnostics() if diagnostics is None else diagnostics
        maximum_depth = terminal_calls = terminal_fallbacks = 0

        def evaluate(current, seed, key, limit):
            nonlocal maximum_depth, terminal_calls, terminal_fallbacks
            names = ('_validation_tail_hits', '_validation_tail_misses',
                     '_validation_terminal_eligible', '_validation_terminal_declined')
            before = tuple(getattr(self, name) for name in names)
            try:
                score, path, depth, calls, fallbacks = self._validate_simulation(
                    current, seed, key, limit,
                    prefix_depth=diagnostics['evaluation_prefix_depth'])
                if path:
                    raise ValueError('mcts_resource_reanalysis_mutable_path')
                maximum_depth = max(maximum_depth, depth)
                terminal_calls += calls
                terminal_fallbacks += fallbacks
                return score
            finally:
                for name, original in zip(names, before):
                    field = name.removeprefix('_validation_')
                    diagnostics[field] = diagnostics.get(field, 0) + getattr(self, name) - original
                    setattr(self, name, original)

        try:
            if prepared is None:
                prepared = self._prepare_resources(request, budget=budget, deadline=deadline,
                    diagnostics=diagnostics, **(dict(previous=previous) if previous is not None else {}))
            self._resource_policy = evaluate_resource_graphs(prepared, seeds, evaluate=evaluate,
                deadline=deadline, diagnostics=diagnostics)
        except (TimeoutError, ValueError, RuntimeError) as error:
            diagnostics.update(status='failed', complete=False, reason=str(error))
            self._resource_policy = None
        return diagnostics, maximum_depth, terminal_calls, terminal_fallbacks

    def _reanalyse_internal(self, selection, seeds, *, budget, deadline, diagnostics,
                           validation_seeds=()):
        from . import hif_internal_reanalysis as internal
        from .hif_public_graph_parallel import _frozen_signature

        maximum_depth = terminal_calls = terminal_fallbacks = 0

        def evaluate(current, seed, key, limit):
            nonlocal maximum_depth, terminal_calls, terminal_fallbacks
            names = ('_validation_tail_hits', '_validation_tail_misses',
                     '_validation_terminal_eligible', '_validation_terminal_declined')
            before = tuple(getattr(self, name) for name in names)
            try:
                prefix = (dict(prefix_depth=diagnostics['successor_graph']['evaluation_prefix_depth'])
                          if graph_evaluation else {})
                score, path, depth, calls, fallbacks = self._validate_simulation(current, seed, key, limit, **prefix)
                if path:
                    raise ValueError('mcts_internal_reanalysis_mutable_path')
                maximum_depth = max(maximum_depth, depth)
                terminal_calls += calls
                terminal_fallbacks += fallbacks
                kind = ('proven_tail_expected_score' if self._validation_tail_hits > before[0] else
                        'terminal_evaluator_expected_score' if calls > fallbacks else 'sampled_terminal_score')
                return score, kind, depth, calls, fallbacks
            finally:
                for name, original in zip(names, before):
                    field = name.removeprefix('_validation_')
                    diagnostics[field] = diagnostics.get(field, 0) + getattr(self, name) - original
                    setattr(self, name, original)

        graph_evaluation = self.internal_successor_graph
        try:
            # Conditional samples evaluate the old frozen selector throughout.
            # No action or suffix cached for that selector may cross publication.
            self._public_step_reuse = self._new_public_step_reuse(deadline)
            policy = None
            if graph_evaluation:
                from .hif_internal_successor_graph import evaluate as evaluate_graph

                policy = evaluate_graph(selection, seeds, kernel=self.kernel,
                    evaluate_return=lambda *args: evaluate(*args)[0],
                    frozen_signature=lambda: _frozen_signature(self), remaining_budget=budget,
                    deadline=deadline, diagnostics=diagnostics, validation_seeds=validation_seeds,
                    score_translation_contract=self.score_translation_contract)
                if policy is None:
                    # An unbuilt graph spends no returns. Preserve the old
                    # single-node path and its original nomination order.
                    graph_evaluation = False
                    graph_evidence = {key: value for key, value in diagnostics.items()
                                      if key.startswith('successor_') or key in ('selection_mode', 'nominations')}
                    selection = internal.select(selection.prepared,
                        node_lookup=lambda current, legal: self._physical_node(current, legal)[1],
                        deadline=deadline, diagnostics=diagnostics)
                    if selection is None:
                        raise ValueError('mcts_internal_graph_fallback_selection_missing')
            if policy is None:
                policy = internal.evaluate(selection, seeds, evaluate_return=evaluate,
                    frozen_signature=lambda: _frozen_signature(self), remaining_budget=budget,
                    deadline=deadline, diagnostics=diagnostics, validation_seeds=validation_seeds)
                if self.internal_successor_graph:
                    diagnostics.update(graph_evidence, successor_graph_fallback=True)
            self._public_step_reuse = self._new_public_step_reuse(deadline)
            self._internal_policy = policy
        except (TimeoutError, ValueError, RuntimeError) as error:
            diagnostics.update(status='failed', complete=False, reason=str(error))
            self._internal_policy = None
        return diagnostics, maximum_depth, terminal_calls, terminal_fallbacks

    def _local_continuation_action(self, request, actions, deadline, depth):
        local = self._continuation
        if local is None:
            raise ValueError('mcts_continuation_window_not_started')
        keys = self._action_order(request, actions)
        # Preserve absolute score because the cached values are absolute final
        # returns; do not reuse the training tree's score-zero key here.
        cache_key = (self._namespace, request.state, tuple(request.drink_slots), keys)

        def evaluate(key, seed, limit):
            # Inner exact diagnostics must not turn a sampled OUTER return
            # into an exact expectation in validation_samples.
            names = ('_validation_tail_hits', '_validation_tail_misses',
                     '_validation_terminal_eligible', '_validation_terminal_declined')
            before = tuple(getattr(self, name) for name in names)
            try:
                score, path, visited, calls, fallbacks = self._rollout(
                    request, seed, key, limit, validation=True, continuation=False)
                if path:
                    raise ValueError('mcts_continuation_mutable_path')
                kind = ('proven_tail_expected_score' if self._validation_tail_hits > before[0]
                        else 'terminal_evaluator_expected_score' if calls > fallbacks
                        else 'sampled_terminal_score')
                return score, kind, visited, calls, fallbacks
            finally:
                for name, original, field_name in zip(names, before,
                        ('tail_hits', 'tail_misses', 'terminal_eligible', 'terminal_declined')):
                    setattr(local, field_name, getattr(local, field_name) + getattr(self, name) - original)
                    setattr(self, name, original)

        selected, audit = local.choose(cache_key, keys, evaluate, deadline)
        self._outer_continuation = dict(audit, depth=depth, turn=request.state.turn,
                                        before_score=request.state.score, selected_action_key=selected)
        return selected

    def _policy_continuation_action(self, request, actions, deadline, depth):
        local = self._continuation
        if local is None or len(local.seeds) != 4:
            raise ValueError('mcts_policy_continuation_requires_four_frozen_inner_seeds')
        cache_key = (self._namespace, 'complete_policy_portfolio', request.state,
                     tuple(request.drink_slots), POLICY_IDS)

        def evaluate(policy_id, seed, limit):
            names = ('_validation_tail_hits', '_validation_tail_misses',
                     '_validation_terminal_eligible', '_validation_terminal_declined')
            before = tuple(getattr(self, name) for name in names)
            try:
                policy = PolicyContinuation(policy_id, self.continuation_policy_factory,
                                            self._validation_action, action_key)
                first = policy.choose(request, actions, limit)
                score, path, visited, calls, fallbacks = self._rollout(
                    request, seed, first, limit, validation=True, continuation=False,
                    policy_continuation=policy)
                if path:
                    raise ValueError('mcts_continuation_mutable_path')
                kind = ('proven_tail_expected_score' if self._validation_tail_hits > before[0]
                        else 'terminal_evaluator_expected_score' if calls > fallbacks
                        else 'sampled_terminal_score')
                return score, kind, visited, calls, fallbacks
            finally:
                for name, original, field_name in zip(names, before,
                        ('tail_hits', 'tail_misses', 'terminal_eligible', 'terminal_declined')):
                    setattr(local, field_name, getattr(local, field_name) + getattr(self, name) - original)
                    setattr(self, name, original)

        selected, audit = local.choose(cache_key, POLICY_IDS, evaluate, deadline)
        self._outer_continuation = dict(audit, depth=depth, turn=request.state.turn,
            trigger_reason=POLICY_CONTINUATION_TRIGGER,
            before_score=request.state.score, selected_policy=selected,
            samples=tuple(dict(policy_id=row['action_key'], **{
                key: value for key, value in row.items() if key != 'action_key'}) for row in audit['samples']))
        policy = PolicyContinuation(selected, self.continuation_policy_factory,
                                    self._validation_action, action_key, audit=self._outer_continuation)
        first = policy.choose(request, actions, deadline)
        self._outer_continuation['selected_action_key'] = first
        return first, policy

    def _rollout(self, request, seed, forced_root, deadline, *, validation, continuation=False,
                 policy_continuation=None, prefix_depth=0, public_prefix=None):
        from .hif_chain_candidate import PairedPlanningKernel

        if type(prefix_depth) is not int or not 0 <= prefix_depth < 256:
            raise ValueError('mcts_simulation_step_limit')
        if policy_continuation is not None and (not validation or continuation):
            raise ValueError('mcts_continuation_policy_requires_nonrecursive_validation')
        if public_prefix is not None:
            from .hif_public_turn_reanalysis import PublicTurnCursor

            if (not isinstance(public_prefix, PublicTurnCursor) or not public_prefix.active
                    or not validation or continuation or policy_continuation is not None or prefix_depth):
                raise ValueError('mcts_public_prefix_requires_fresh_validation_cursor')
        expansion = [0, 0]
        if self.single_node_expansion and not validation:
            self._last_training_expansion = expansion
            self._training_expansion['started_training_rollouts'] += 1
        option = forced_root if isinstance(forced_root, _RootOption) else None
        forced_root = action_key(option.action) if option is not None else forced_root
        environment = HifEnvironment(self.kernel)
        # Only this known factory preserves the default catalogue validation.
        # Arbitrary factories and subclasses retain their original reset path.
        paired = self.sampler_factory is PairedPlanningKernel
        reset_options = {'sampler_factory': self.sampler_factory} if paired else {}
        current = replace(environment.reset(request.state, seed=seed, drink_slots=request.drink_slots,
                                            **reset_options),
                          utility=request.utility)
        if self.sampler_factory is not None and not paired:
            environment._sampler = self.sampler_factory(self.kernel, seed)
        cursor = (self._public_turn_policy.cursor(current)
                  if validation and self._public_turn_policy is not None else None)
        step_reuse = (self._public_step_reuse
                      if validation and cursor is None and not continuation
                      and policy_continuation is None and public_prefix is None else None)
        path, terminal_calls, terminal_fallbacks = [], 0, 0
        return_checkpoints = []

        def finished(score, depth, *, reused_tail_calls=0, cacheable=True):
            if public_prefix is not None and public_prefix.active:
                raise ValueError('mcts_public_prefix_not_completed')
            if step_reuse is not None and not terminal_fallbacks and cacheable:
                step_reuse.remember_return(return_checkpoints, score=score, depth=depth,
                    tail_calls=terminal_calls + reused_tail_calls)
            # Reusing a proven suffix does not execute its tail evaluator.
            return score, path, depth + prefix_depth, terminal_calls, terminal_fallbacks

        continuation_used = False
        failed_horizons = set()
        frozen_tail_validation = validation and self.validation_tail_lookup is not None
        terminal_horizon = 1 if frozen_tail_validation else self.terminal_horizon
        for depth in range(256 - prefix_depth):
            if time.perf_counter() >= deadline:
                raise TimeoutError('mcts_window_timeout')
            state = current.state
            forced_continuation = None
            if depth == 1 and option is not None and option.continuation:
                if (self.kernel._canonical_probability_state(state) != option.expected_after_first
                        or state.terminal
                        or (state.pending_choice is None
                            and option.continuation[0].kind is not ActionKind.END_TURN)
                        or (option.continuation[0].kind is ActionKind.END_TURN
                            and (state.pending_choice is not None or state.resolution_stack
                                 or state.actions_remaining <= 0 or state.turn != request.state.turn))):
                    raise ValueError('mcts_root_option_public_state_mismatch')
                forced_continuation = action_key(option.continuation[0])
            if (step_reuse is not None and depth and forced_continuation is None
                    and not failed_horizons):
                return_key = step_reuse.return_key(environment, current)
                cached = step_reuse.completed_return(return_key,
                    remaining_depth=256 - prefix_depth - depth,
                    remaining_steps=512 - environment._steps)
                if cached is not None:
                    score, remaining_depth, reused_tail_calls = cached
                    return finished(score, depth + remaining_depth, reused_tail_calls=reused_tail_calls)
                if return_key is not None:
                    return_checkpoints.append((return_key, depth, terminal_calls))
            if state.terminal:
                if state.stamina < request.utility.minimum_stamina or state.genki < request.utility.minimum_genki:
                    raise ValueError('mcts_terminal_resource_floor')
                return finished(float(state.score), depth)
            if (depth and forced_continuation is None and frozen_tail_validation and state.turns_remaining == 2
                    and not (public_prefix is not None and public_prefix.active)):
                # The provider was frozen before any validation scenario. A
                # miss neither starts a proof nor changes the frozen policy.
                value = self.validation_tail_lookup(current)
                if time.perf_counter() >= deadline:
                    raise TimeoutError('mcts_window_timeout')
                if value is not None:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError('mcts_invalid_proven_tail_value')
                    self._validation_tail_hits += 1
                    return finished(float(value), depth)
                self._validation_tail_misses += 1
            # Keep separate root evaluations. Exact terminal expectations
            # integrate unknown draws without revealing them to the policy.
            evaluate_terminal = (depth and forced_continuation is None and state.turns_remaining <= terminal_horizon
                                 and self.terminal_evaluator is not None
                                 and state.turns_remaining not in failed_horizons
                                 and not (public_prefix is not None and public_prefix.active))
            domain_terminal = (self.terminal_eligible is not None
                               and state.turns_remaining in self.terminal_eligibility_horizons)
            if evaluate_terminal and domain_terminal:
                eligible = self.terminal_eligible(current)
                if time.perf_counter() >= deadline:
                    raise TimeoutError('mcts_window_timeout')
                if type(eligible) is not bool:
                    raise ValueError('mcts_terminal_eligibility_not_bool')
                if validation:
                    self._validation_terminal_eligible += int(eligible)
                    self._validation_terminal_declined += int(not eligible)
                else:
                    self._training_terminal_eligible += int(eligible)
                    self._training_terminal_declined += int(not eligible)
                # Declining this state does not fail the horizon: another
                # action may enter the domain later within this same turn.
                evaluate_terminal = eligible
            if evaluate_terminal:
                evaluator = (self.validation_terminal_evaluator if validation
                             and self.validation_terminal_evaluator is not None else self.terminal_evaluator)
                response = evaluator(current, deadline)
                terminal_calls += 1
                if time.perf_counter() >= deadline:
                    raise TimeoutError('mcts_window_timeout')
                if response.has_action_intent and response.value is not None:
                    if (self.terminal_horizon == 2 or frozen_tail_validation or domain_terminal
                            or self._effective_continuation_samples) and (
                            response.trace is None or response.trace.value_scope != 'FINAL_SCORE'):
                        raise ValueError('mcts_terminal_value_not_complete')
                    return finished(float(response.value.expected_final_score), depth,
                        cacheable=response.trace is not None and response.trace.value_scope == 'FINAL_SCORE')
                if response.reason != 'search_incomplete:planner_timeout':
                    raise ValueError('mcts_terminal_search_incomplete:' + response.reason)
                if self.quality_reference:
                    # A protection timeout pauses the whole fixed-work window;
                    # it must not change its continuation to an approximate one.
                    raise TimeoutError('mcts_quality_reference_terminal_timeout')
                if validation and (self.terminal_horizon == 2 or frozen_tail_validation or domain_terminal):
                    # Validation evaluates one fixed public-state policy:
                    # exact optimal suffixes, independent of warm-cache luck.
                    # A timeout cannot silently substitute a different policy.
                    raise ValueError('mcts_validation_terminal_search_incomplete:' + response.reason)
                # A costly hypothetical terminal tree is still explored by
                # the same public-state search. Only a completed game may
                # provide its return; an unfinished exact value is ignored.
                terminal_fallbacks += 1
                failed_horizons.add(state.turns_remaining)
            actions = environment.legal_actions()
            if not actions:
                raise ValueError('mcts_no_legal_action')
            if (validation and self._resource_policy is not None
                    and (cursor is None or not cursor.active)):
                cursor = self._resource_policy.cursor(current, actions)
            node = None
            back_up_node = True
            proven_action = None
            if depth and self.dominance_pruning and len(actions) == 2:
                for option in actions:
                    proof = prove_forced_end_drink_dominance(self.kernel, current, option)
                    if proof is not None:
                        proven_action = action_key(proof.dominant_action)
                        break
            if validation:
                partial_node = False
                if (continuation and not continuation_used and depth and forced_continuation is None
                        and proven_action is None and len(actions) > 1):
                    _, entry = self._physical_node(current, actions)
                    if entry is not None and set(entry.statistics) != {action_key(action) for action in actions}:
                        raise ValueError('mcts_public_state_action_mismatch')
                    # Compare complete policies at the first free choice after
                    # the root (including any mandatory joint-root target).
                    # Visited nodes still guide the frozen-tree candidate;
                    # they no longer delay the portfolio's comparison point.
                    compare_here = (self.continuation_portfolio or entry is None
                                    or not any(statistic[0] for statistic in entry.statistics.values()))
                    if self.continuation_incomplete_actions:
                        visited_actions = (sum(statistic[0] > 0 for statistic in entry.statistics.values())
                                           if entry is not None else 0)
                        partial_node = 0 < visited_actions < len(actions)
                        compare_here = compare_here or partial_node
                else:
                    compare_here = False
                alternatives = len(POLICY_IDS) if self.continuation_portfolio else len(actions)
                if compare_here and self.continuation_domain and alternatives > self._domain_limit:
                    # This public structural decision is fixed before any
                    # training/outer seed. Skipping a wide node does not use
                    # this outer's one local comparison; a later narrow node
                    # may qualify. Neither time, cache nor remaining effort
                    # enters this eligibility test.
                    self._unimproved_count += 1
                    self._outer_unimproved_count += 1
                    compare_here = False
                if public_prefix is not None and public_prefix.active:
                    selected_key = public_prefix.choose(current, actions)
                    if depth == 0 and selected_key != forced_root:
                        raise ValueError('mcts_public_prefix_root_mismatch')
                elif compare_here:
                    continuation_used = True
                    if self.continuation_incomplete_actions:
                        trigger = dict(trigger_reason='incomplete_actions' if partial_node else 'unvisited_node',
                                       visited_action_count=visited_actions,
                                       unvisited_action_count=len(actions) - visited_actions)
                        self._outer_continuation = trigger
                        self._partial_node_calls += int(partial_node)
                    if self.continuation_portfolio:
                        selected_key, policy_continuation = self._policy_continuation_action(
                            current, actions, deadline, depth)
                    else:
                        selected_key = self._local_continuation_action(current, actions, deadline, depth)
                    if self.continuation_incomplete_actions:
                        self._outer_continuation.update(trigger)
                        if partial_node:
                            hit = self._outer_continuation['cache_hit']
                            self._partial_node_cache_hits += int(hit)
                            if not hit:
                                self._partial_node_new_actions += len(actions) - visited_actions
                else:
                    selected_key = (forced_root if depth == 0 else
                                    forced_continuation or proven_action or
                                    (self._public_turn_action(cursor, current, actions)
                                     if cursor is not None and cursor.active else None) or
                                    (policy_continuation.choose(current, actions, deadline)
                                     if policy_continuation is not None else
                                     step_reuse.choose(current, actions, self._validation_action)
                                     if step_reuse is not None else self._validation_action(current, actions)))
            else:
                if (self.retrieval_training_orderer is not None and depth > 0
                        and forced_continuation is None and proven_action is None and len(actions) > 1):
                    _, node = self._node(current, actions, training_order_allowed=True)
                else:
                    _, node = self._node(current, actions)
                free_choice = (self.single_node_expansion and depth > 0 and forced_continuation is None
                               and proven_action is None and len(actions) > 1)
                untrained = free_choice and not any(statistic[0] for statistic in node.statistics.values())
                if untrained and expansion[0]:
                    # Keep a zero-visit routing node and its observed child edge
                    # so subtree retention can cross this state. It is not a
                    # trained node: later trajectories may still expand it.
                    selected_key = node.order[0]
                    back_up_node = False
                    expansion[1] += 1
                    self._training_expansion['started_prior_route_steps'] += 1
                else:
                    if untrained:
                        expansion[0] += 1
                        self._training_expansion['started_free_expansions'] += 1
                        self._training_expansion['max_free_expansions_per_rollout'] = max(
                            self._training_expansion['max_free_expansions_per_rollout'], expansion[0])
                    # The quota never disables UCT at a previously trained node,
                    # even when reached after one or more zero-visit route nodes.
                    selected_key = (forced_root if depth == 0 else forced_continuation or proven_action or
                                    self._select(node, progressive=self.progressive_widening))
            matches = [action for action in actions if action_key(action) == selected_key]
            if len(matches) != 1:
                raise ValueError('mcts_action_not_uniquely_legal')
            if node is not None and back_up_node:
                path.append((node.statistics[selected_key], state.score))
            before = current
            prefix_active = public_prefix is not None and public_prefix.active
            sampler_before = ((getattr(environment._sampler, 'draw_index', None),
                               environment._sampler._rng.getstate())
                              if cursor is not None and cursor.active or prefix_active else None)
            current = (step_reuse.step(environment, matches[0]) if step_reuse is not None
                       else environment.step(matches[0])).request
            current = replace(current, utility=request.utility)
            if cursor is not None and cursor.active:
                cursor.advance(before, matches[0], current, sampler_before,
                    (getattr(environment._sampler, 'draw_index', None), environment._sampler._rng.getstate()))
            if prefix_active:
                public_prefix.advance(before, matches[0], current, sampler_before,
                    (getattr(environment._sampler, 'draw_index', None), environment._sampler._rng.getstate()))
            if policy_continuation is not None:
                policy_continuation.committed(state, matches[0], current.state)
            if node is not None:
                node.children.add(self._state_key(current.state, current.drink_slots))
        raise ValueError('mcts_simulation_step_limit')

    def _validate_finalists(self, request, rows, keys, forced_roots, *, base_completed,
                            base_complete, validation_limit, deadline):
        """Spend a residual allowance on one fixed, paired screened cohort.

        Screening is a sampling decision, not proof that excluded roots lose.
        The ordinary validation table and its random stream remain unchanged.
        """
        residual = self.simulations - base_completed
        report = dict(version=FINALIST_VALIDATION_VERSION, enabled=True, eligible=False,
            reason='base_validation_incomplete', residual_simulation_budget=residual,
            candidate_indices=(), requested_samples=4, planning_seeds=(), started_simulations=0,
            completed_simulations=0, complete=False, rows=(), selected_root_index=None)
        maximum_depth = terminal_calls = terminal_fallbacks = 0
        if not base_complete:
            return report, maximum_depth, terminal_calls, terminal_fallbacks
        if validation_limit >= 32:
            report['reason'] = 'validation_sample_cap_reached'
        elif residual < 8:
            report['reason'] = 'insufficient_residual_budget'
        elif residual >= len(keys):
            report['reason'] = 'residual_can_cover_full_round'
        else:
            report['eligible'] = True
        if not report['eligible']:
            return report, maximum_depth, terminal_calls, terminal_fallbacks
        indices = tuple(sorted(range(len(rows)),
            key=lambda index: -rows[index]['validation_mean_final_score'])[:min(residual, 24) // 4])
        report['candidate_indices'] = indices
        result_rows = [dict(root_index=index, samples=[], mean_final_score=None,
                            combined_mean_final_score=None) for index in indices]
        report['rows'] = result_rows
        seeds = []
        # All original cohorts are in [0, 2**31). The separate stream also
        # keeps optional work from shifting the next window's ordinary RNG.
        try:
            while len(seeds) < 4:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('mcts_window_timeout')
                seed = self._adaptive_rng.randrange(2**31, 2**32)
                if seed not in seeds:
                    seeds.append(seed)
            report['planning_seeds'] = tuple(seeds)
            for scenario, seed in enumerate(seeds):
                for row in result_rows:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('mcts_window_timeout')
                    index = row['root_index']
                    tail_hits_before = self._validation_tail_hits
                    report['started_simulations'] += 1
                    score, path, depth, calls, fallbacks = self._validate_simulation(
                        request, seed, forced_roots[keys[index]], deadline)
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('mcts_window_timeout')
                    if path:
                        raise ValueError('mcts_finalist_validation_mutable_path')
                    if type(score) not in (int, float) or not math.isfinite(score):
                        raise ValueError('mcts_finalist_validation_invalid_score')
                    report['completed_simulations'] += 1
                    maximum_depth = max(maximum_depth, depth)
                    terminal_calls += calls
                    terminal_fallbacks += fallbacks
                    row['samples'].append(dict(scenario_index=scenario, planning_seed=seed,
                        final_score=score,
                        return_kind=('proven_tail_expected_score'
                            if self._validation_tail_hits > tail_hits_before else
                            'terminal_evaluator_expected_score' if calls > fallbacks else
                            'sampled_terminal_score'), completed_depth=depth,
                        terminal_evaluations=calls, terminal_fallbacks=fallbacks))
            report['complete'] = all(len(row['samples']) == 4 for row in result_rows)
            report['reason'] = 'completed' if report['complete'] else 'finalist_validation_incomplete'
        except TimeoutError:
            report['reason'] = 'timeout'
        except Exception as error:
            report['reason'] = str(error) or type(error).__name__
        for row in result_rows:
            samples = row['samples']
            total = sum(sample['final_score'] for sample in samples)
            row['mean_final_score'] = total / len(samples) if samples else None
            if len(samples) == 4:
                original = rows[row['root_index']]
                count = original['validation_paired_visits']
                original_total = sum(sample['final_score'] for sample in original['validation_samples'])
                row['combined_mean_final_score'] = (original_total + total) / (count + 4)
        if report['complete']:
            report['selected_root_index'] = max(result_rows,
                key=lambda row: row['combined_mean_final_score'])['root_index']
        return report, maximum_depth, terminal_calls, terminal_fallbacks

    def _validate_parallel_roots(self, request, roots, seeds, deadline):
        from . import hif_root_validation_parallel as parallel

        reason = parallel.decline_reason(self, roots)
        self.last_parallel_root_validation = dict(version=parallel.VERSION,
            status='serial', reason=reason, complete=False, workers=4,
            started_simulations=0, completed_simulations=0, count_incomplete=False,
            elapsed_seconds=0., worker_records=[], seeds=tuple(seeds),
            root_keys=tuple(root.key for root in roots))
        if reason is not None:
            return None
        owner = self._public_step_reuse_check.__self__
        diagnostics = {}
        try:
            self._timed_search_call('root_validation', parallel.run_worlds,
                                    owner, request, roots, seeds, deadline, diagnostics)
        finally:
            if 'parallel_root_validation' in diagnostics:
                self.last_parallel_root_validation = diagnostics['parallel_root_validation']
                self.last_parallel_root_validation.update(
                    status='complete' if self.last_parallel_root_validation['complete'] else 'failed',
                    reason='four_worlds_complete' if self.last_parallel_root_validation['complete']
                    else 'parallel_incomplete')
        return self.last_parallel_root_validation

    def _validate_adaptive_pairs(self, request, rows, keys, forced_roots, retained_policy,
                                 retained_comparison, *, budget, deadline, excluded_seeds):
        from . import hif_paired_adaptive_validation as adaptive
        from .hif_public_graph_parallel import _frozen_signature

        maximum_depth = terminal_calls = terminal_fallbacks = 0
        candidate_keys = tuple(keys)
        base_samples = tuple(row['validation_samples'] for row in rows)
        if retained_comparison is not None:
            candidate_keys += (('RETAINED_POLICY', retained_policy.source_plan_index),)
            base_samples += (tuple(retained_comparison['validation_samples']),)
        # Allocate lazily without mutating the frozen evaluator's RNG during
        # this phase. Adopt only the stream actually consumed, including failure.
        allocation_rng = random.Random()
        allocation_rng.setstate(self._rng.getstate())

        def next_seed(excluded):
            while True:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('mcts_window_timeout')
                seed = allocation_rng.randrange(2**31)
                if seed not in excluded:
                    return seed

        def evaluate(index, seed, limit):
            nonlocal maximum_depth, terminal_calls, terminal_fallbacks
            tail_before = self._validation_tail_hits
            if index < len(keys):
                score, path, depth, calls, fallbacks = self._validate_simulation(
                    request, seed, forced_roots[keys[index]], limit)
            else:
                score, path, depth, calls, fallbacks = retained_policy.evaluate(request, seed, limit)
            if path:
                raise ValueError('mcts_adaptive_validation_mutable_path')
            maximum_depth = max(maximum_depth, depth)
            terminal_calls += calls
            terminal_fallbacks += fallbacks
            return dict(final_score=score,
                return_kind=('proven_tail_expected_score' if self._validation_tail_hits > tail_before else
                    'terminal_evaluator_expected_score' if calls > fallbacks else 'sampled_terminal_score'),
                completed_depth=depth, terminal_evaluations=calls, terminal_fallbacks=fallbacks)

        try:
            report = adaptive.evaluate(candidate_keys, base_samples, remaining_budget=budget,
                reward_scale=self.reward_scale, next_seed=next_seed, evaluate_return=evaluate,
                frozen_signature=lambda: _frozen_signature(self), deadline=deadline,
                excluded_seeds=tuple(excluded_seeds),
                retained_index=len(keys) if retained_comparison is not None else None)
        finally:
            self._rng.setstate(allocation_rng.getstate())
        return report, maximum_depth, terminal_calls, terminal_fallbacks

    def _timed_search_call(self, phase, function, *args, **kwargs):
        """Measure disjoint call sites, never recursively timed rollout internals."""
        timing = self.last_search_phase_timing
        if self._search_phase_started is None:
            return function(*args, **kwargs)
        row = timing['phases'][phase]
        row['executed'] = True
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            row['wall_ms'] += (time.perf_counter() - started) * 1000

    def _finish_search_phase_timing(self):
        if self._search_phase_started is None:
            return
        timing = self.last_search_phase_timing
        timing['elapsed_ms'] = (time.perf_counter() - self._search_phase_started) * 1000
        timing['other_ms'] = max(0., timing['elapsed_ms']
            - sum(row['wall_ms'] for row in timing['phases'].values()))
        self._search_phase_started = None

    def _select_completed_root(self, request, rows, physical_actions, *, accepted,
                               selection_field, deadline, finalist=None, adaptive_pairs=None,
                               retained_comparison=None):
        """Select from paid values, excluding only a fully proved wasted drink.

        Eligibility is separate from legality and sampling. Adaptive/finalist
        selectors retain their existing contract; their evidence says so.
        """
        extra = getattr(self.unknown_node_policy, 'extra_action_policy', None)
        enabled = getattr(extra, 'unused_drink_enabled', False)
        eligible = tuple(range(len(rows)))
        retained_allowed = retained_comparison is not None
        evidence = None
        if enabled:
            from .hif_mcts_cache import public_request
            from .hif_unused_action_turn import VERSION as PROOF_VERSION

            reason = ('search_not_accepted' if not accepted else
                      'paired_adaptive_selection_unsupported' if self.paired_adaptive_validation else
                      'finalist_selection_unsupported' if self.paired_finalist_validation else 'completed')
            applicable = reason == 'completed'
            legal_keys = tuple(action_key(action) for action in physical_actions)
            evidence = dict(version=ROOT_SELECTION_ELIGIBILITY_VERSION,
                scope='PROVEN_UNUSED_ACTION_DRINK_MEAN_SELECTION', applicable=applicable,
                complete=bool(accepted), reason=reason, selection_field=selection_field,
                public_request=public_request(request), original_legal_action_keys=legal_keys,
                eligible_root_indices=eligible, retained_present=retained_allowed,
                retained_eligible=retained_allowed if retained_allowed else None,
                checks=(), root_winner_index=None, selected_retained=False)
            if applicable:
                legal_set = set(legal_keys)
                if (len(legal_set) != len(legal_keys) or not rows
                        or {action_key(row['action']) for row in rows} != legal_set
                        or retained_allowed and action_key(retained_comparison['action']) not in legal_set):
                    raise ValueError('mcts_root_selection_legal_coverage_changed')
                excluded, checks = set(), []
                for action in physical_actions:
                    if action.kind is not ActionKind.USE_DRINK:
                        continue
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('mcts_root_selection_eligibility_timeout')
                    key = action_key(action)
                    indices = tuple(index for index, row in enumerate(rows) if action_key(row['action']) == key)
                    retained_candidate = retained_allowed and action_key(retained_comparison['action']) == key
                    proof = extra.unused_drink_proof(request, action, deadline)
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('mcts_root_selection_eligibility_timeout')
                    checks.append(dict(action_key=key, root_indices=indices,
                        retained_candidate=retained_candidate, proof=deepcopy(proof)))
                    if proof is None:
                        continue
                    transitions = proof.get('transition_checks') if isinstance(proof, dict) else None
                    if (not isinstance(proof, dict) or proof.get('version') != PROOF_VERSION
                            or proof.get('complete') is not True or proof.get('excluded_action_key') != key
                            or set(proof.get('original_legal_action_keys', ())) != legal_set
                            or set(proof.get('retained_legal_action_keys', ())) != legal_set - {key}
                            or proof.get('actions_before') != request.state.actions_remaining
                            or proof.get('deletes_root_actions') is not False or proof.get('forces_end') is not False
                            or proof.get('new_final_returns') != 0 or not isinstance(transitions, dict)
                            or any(transitions.get(name) is not True for name in (
                                'all_non_end_paths_paired', 'all_business_fields_preserved',
                                'events_identical', 'end_erases_positive_ap_at_every_node'))
                            or type(transitions.get('minimum_retained_actions')) is not int
                            or transitions['minimum_retained_actions'] <= 0):
                        raise ValueError('mcts_root_selection_proof_invalid')
                    excluded.add(key)
                eligible = tuple(index for index, row in enumerate(rows) if action_key(row['action']) not in excluded)
                if not eligible:
                    raise ValueError('mcts_root_selection_no_eligible_root')
                retained_allowed = (retained_allowed
                    and action_key(retained_comparison['action']) not in excluded)
                evidence.update(eligible_root_indices=eligible,
                    retained_eligible=retained_allowed if retained_comparison is not None else None,
                    checks=tuple(checks))
        winner = None
        retained_wins = False
        if accepted:
            winner = (finalist['selected_root_index'] if finalist is not None and finalist['eligible']
                      else max(eligible, key=lambda index: rows[index][selection_field]))
            if adaptive_pairs is not None and adaptive_pairs['selected_index'] < len(rows):
                winner = adaptive_pairs['selected_index']
            if retained_comparison is not None:
                retained_wins = (adaptive_pairs['selected_index'] == len(rows) if adaptive_pairs is not None
                    else retained_allowed and retained_comparison['mean_final_score'] >= rows[winner][selection_field])
            if evidence is not None:
                evidence.update(root_winner_index=winner, selected_retained=retained_wins)
        return winner, retained_wins, evidence

    def suggest(self, request, *, deadline=None, root_options=None, baseline_domain_reason=None,
                validation_cohort=(), retained_policy=None):
        self.last_search_phase_timing = None
        self.last_uniform_validation_time_guard = None
        self._search_phase_started = None
        self._reset_validation_seed_allocation()
        self._reset_partial_node_timing()
        self._reset_validation_cohort_evidence()
        self._reset_public_turn_budget_reservation()
        if (not isinstance(validation_cohort, (tuple, list))
                or validation_cohort and (not self.deterministic_validation_cohort
                    or len(validation_cohort) != 4
                    or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in validation_cohort)
                    or len(set(validation_cohort)) != 4)):
            raise ValueError('mcts_validation_cohort_invalid')
        validation_cohort = tuple(validation_cohort)
        revealed = getattr(retained_policy, 'revealed_policy_version', None)
        if revealed is not None and (revealed != REVEALED_POLICY_COMPETITION_VERSION
                or not self.revealed_policy_competition or validation_cohort):
            raise ValueError('mcts_revealed_policy_requires_fresh_comparison')
        if retained_policy is not None and (
                (len(validation_cohort) != 4 and revealed is None) or not self.public_turn_reanalysis
                or self.continuation_samples or self.continuation_portfolio or self.paired_finalist_validation
                or not isinstance(getattr(retained_policy, 'action', None), GameAction)
                or not callable(getattr(retained_policy, 'evaluate', None))):
            raise ValueError('mcts_retained_policy_incompatible_configuration')
        if validation_cohort and self._training_seed_history.intersection(validation_cohort):
            raise ValueError('mcts_validation_cohort_overlaps_training')
        self._reset_validation_cohort_evidence(validation_cohort)
        request.validate()
        if baseline_domain_reason is not None and (
                not self.continuation_domain
                or baseline_domain_reason != 'joint_root_expansion_budget_ineligible'):
            raise ValueError('mcts_invalid_baseline_domain_reason')
        if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
            raise ValueError('mcts_capability_or_ruleset_not_ready')
        if request.utility.risk_profile is not RiskProfile.ROBUST_HIGH_SCORE or request.utility.baseline_p10 is not None:
            raise ValueError('mcts_sampling_does_not_prove_risk_floor')
        started = time.perf_counter()
        if self.quality_reference:
            self._search_phase_started = started
            self.last_search_phase_timing = dict(version=SEARCH_PHASE_TIMING_VERSION,
                scope='DISJOINT_SUGGEST_CALLS_EXCLUDING_CANDIDATE_DELIVERY',
                elapsed_ms=None, other_ms=None,
                phases={name: dict(executed=False, wall_ms=0.) for name in (
                    'preparation', 'training', 'freeze', 'public_graph', 'resource',
                    'internal', 'root_validation', 'retained_validation', 'finalist_validation')})
            if self.paired_adaptive_validation:
                self.last_search_phase_timing['phases']['paired_adaptive_validation'] = dict(
                    executed=False, wall_ms=0.)
        self._validation_tail_hits = self._validation_tail_misses = 0
        self._training_terminal_eligible = self._training_terminal_declined = 0
        self._validation_terminal_eligible = self._validation_terminal_declined = 0
        self._continuation = None
        self._outer_continuation = None
        self._public_turn_policy = None
        self._resource_policy = None
        self._internal_policy = None
        self.last_parallel_root_validation = None
        self._effective_continuation_samples = self.continuation_samples
        self._domain_limit = None
        self._baseline_domain_reason = baseline_domain_reason
        self._unimproved_count = self._outer_unimproved_count = 0
        self._partial_node_calls = self._partial_node_cache_hits = self._partial_node_new_actions = 0
        self._reset_training_expansion()
        deadline = min(deadline if deadline is not None else math.inf, started + self.timeout_ms / 1000)
        self._order_deadline = deadline
        if self.complete_orderer is not None:
            self.complete_orderer.start_window()
        if self.turn_chain_continuation is not None:
            self.turn_chain_continuation.start_window()
        if self.unknown_node_policy is not None:
            self.unknown_node_policy.start_window()
        if self.retrieval_training_orderer is not None:
            self.retrieval_training_orderer.start_window()
        soft_deadline = (started + self.soft_timeout_ms / 1000
                         if self.soft_timeout_ms is not None else math.inf)
        request = replace(request, state=self.kernel._canonical_probability_state(request.state))
        namespace = self._context(request)
        if self._namespace != namespace or (self._expected_state is not None and request.state != self._expected_state):
            self._tree.clear()
        self._namespace = namespace
        root = self._state_key(request.state, request.drink_slots)
        self._retain(root)
        retained = len(self._tree)
        environment = HifEnvironment(self.kernel)
        environment.reset(request.state, seed=self.seed, drink_slots=request.drink_slots)
        actions = environment.legal_actions()
        if request.card_slots:
            # Canonical hand order is internal to the simulator. Root proofs
            # must retain the caller's independently observed physical slots.
            actions = tuple(RobustHighScorePlanner._attach_source_slot(action, request)
                            for action in actions)
        if not actions:
            raise ValueError('mcts_no_root_actions')
        retained_comparison = None
        if retained_policy is not None:
            retained_matches = [action for action in actions
                                if action_key(action) == action_key(retained_policy.action)]
            if len(retained_matches) != 1:
                raise ValueError('mcts_retained_policy_action_not_uniquely_legal')
            retained_comparison = dict(version=RETAINED_POLICY_COMPETITION_VERSION, eligible=True,
                action=retained_matches[0], validation_samples=[], started_simulations=0,
                completed_simulations=0, complete=False, selected=False, mean_final_score=None)
            if revealed is not None:
                retained_comparison.update(revealed_policy_competition_version=revealed,
                    source_plan_index=retained_policy.source_plan_index,
                    excluded_training_seeds=sorted(retained_policy.policy.search._training_seed_history),
                    excluded_validation_seeds=sorted(retained_policy.validation_excluded_seeds),
                    fresh_validation_cohort=True)
        _, node = self._node(request, actions)
        physical_actions = actions
        physical_root_count = len(actions)
        options = (self._prepare_root_options(request, actions, root_options, deadline)
                   if root_options is not None else None)
        if options is not None:
            # Joint returns belong to this window, not to the aggregate
            # physical first-action statistic in the reusable public tree.
            keys = tuple(option.key for option in options)
            forced_roots = dict(zip(keys, options))
            actions = tuple(option.action for option in options)
            order = tuple(option.key for physical in node.order for option in options
                          if action_key(option.action) == physical)
            node = _Node({key: [0, 0.] for key in keys}, order)
        else:
            keys = tuple(action_key(action) for action in actions)
            forced_roots = dict(zip(keys, keys))
        current_counts = {key: 0 for key in keys}
        paired_counts = {key: 0 for key in keys}
        current_totals = {key: 0. for key in keys}
        paired_totals = {key: 0. for key in keys}
        scenario_visits = {key: {} for key in keys}
        scenario_totals = {key: {} for key in keys}
        inherited_visits = {key: node.statistics[key][0] for key in keys}
        validation_width = len(keys) + int(retained_policy is not None)
        minimum_validation = validation_width * self.validation_samples
        if self.continuation_domain:
            if baseline_domain_reason is None:
                outer_required = len(keys) * self.validation_samples
                self._domain_limit = (self.simulations - len(keys) * self.min_root_samples
                                      - outer_required) // (outer_required * self.continuation_samples)
            # Select the complete legacy path before drawing any scenario
            # seed or allocating minimum-only training. There is no runtime
            # fallback after learning or a partially evaluated outer.
            if baseline_domain_reason is not None or self._domain_limit < 2:
                self._effective_continuation_samples = 0
        if self.continuation_samples:
            self._continuation = LocalContinuation(self.simulations, len(keys) * self.validation_samples)
        seeds = (self._fresh_seeds(self.min_root_samples, validation_cohort) if validation_cohort else
                 self._fresh_seeds(self.min_root_samples) if self.validation_samples else
                 tuple(self._rng.randrange(2**31) for _ in range(self.min_root_samples)))
        if self.deterministic_validation_cohort:
            self.last_validation_cohort_evidence['training_seeds'] = list(seeds)
        coverage = [(key, seed) for seed in seeds for key in keys]
        portfolio_training_allocation = None
        training_budget = max(0, self.simulations - minimum_validation)
        if self.adaptive_validation:
            # Spend at most half the traversal allowance learning, except
            # that every legal root still needs its full paired coverage.
            training_budget = min(training_budget, max(len(coverage), self.simulations // 2))
        if self._effective_continuation_samples:
            if self.continuation_portfolio and self.fresh_adaptive_training:
                # Each outer can compare the fixed policies once, without
                # recursive inner comparisons. Reserve that entire worst case
                # before training; cache hits never enlarge this allocation.
                outer_required = len(keys) * self.validation_samples
                reserved = outer_required * (1 + len(POLICY_IDS) * self._effective_continuation_samples)
                available = self.simulations - reserved
                training_budget = available if available >= len(coverage) else 0
                portfolio_training_allocation = dict(mode='static_complete_policy_reserve_v1',
                                                     reserved_simulations=reserved)
            else:
                # Other local candidates retain their minimum-only training.
                training_budget = len(coverage) if len(coverage) * 2 <= self.simulations else 0
        prepared_graph = prepared_diagnostics = preparation_failure = None
        if retained_policy is not None and len(coverage) + minimum_validation > self.simulations:
            preparation_failure = 'mcts_retained_policy_coverage_budget_incomplete'
        elif self.public_turn_budget_reservation:
            prepared_graph, prepared_diagnostics, training_budget, preparation_failure = self._timed_search_call(
                'preparation', self._prepare_public_turn_budget, request, root_options=options, training_budget=training_budget,
                minimum_training=len(coverage), minimum_validation=minimum_validation,
                deadline=deadline)
        original_training_budget = training_budget
        internal_prepared = internal_report = internal_selection = None
        prepared_resource = prepared_resource_diagnostics = None
        declined_resource_precheck = None
        if self.internal_reanalysis:
            from . import hif_internal_reanalysis as internal

            internal_report = internal.new_diagnostics(
                complete_root_successors=self.internal_complete_root_successors)
            if preparation_failure is None:
                try:
                    def prior(current, legal):
                        selected = self.unknown_node_policy.choose(current, legal, deadline)
                        return self._action_order(current, legal)[0] if selected is None else selected

                    physical = {action_key(action): action for action in physical_actions}
                    internal_prepared = self._timed_search_call(
                        'preparation', internal.prepare, self.kernel, request,
                        tuple(physical[key] for key in self._action_order(request, physical_actions)),
                        prior=prior, root_count=len(keys), context=self._context(request), deadline=deadline,
                        total_budget=self.simulations, compute_sensitivity=self.compute_sensitivity,
                        retained_policy=retained_policy is not None,
                        complete_policy=prepared_graph is not None and prepared_graph.ready,
                        diagnostics=internal_report,
                        **(dict(complete_root_successors=True, root_options=options if options is not None else
                            tuple(_RootOption(action, (), None) for action in physical_actions))
                           if self.internal_complete_root_successors else {}))
                    training_budget = min(training_budget, internal_prepared.training_limit)
                    reduced = training_budget < original_training_budget
                    internal_report['reservation_release'].update(
                        status='held' if reduced else 'not_applicable',
                        reason='pending_selection' if reduced else 'training_limit_not_reduced',
                        original_training_limit=original_training_budget,
                        effective_training_limit=training_budget)
                except (TimeoutError, ValueError, RuntimeError) as error:
                    internal_report.update(status='failed', complete=False, reason=str(error))
                    preparation_failure = 'mcts_internal_reanalysis_failed:' + str(error)
        # Expanding a required choice can consume the soft target during
        # coverage alone. Finish its existing training allocation so the
        # adaptive phase can learn, without borrowing validation traversals.
        # Coverage-first candidates instead use the ordinary soft deadline
        # after completing every root's mandatory paired training cohort.
        joint_training_hard = (not self.root_coverage_first
                               and not self._effective_continuation_samples and options is not None
                               and any(option.continuation
                                   and option.continuation[0].kind is ActionKind.MOVE_CARD for option in options))
        completed = maximum_depth = terminal_calls = terminal_fallbacks = 0
        ordinary_training_started = ordinary_validation_started = 0
        adaptive_seeds, completed_adaptive_seeds = set(), set()
        failure = preparation_failure or ('mcts_continuation_root_budget_incomplete'
                   if self._effective_continuation_samples and not training_budget else None)
        # The second pass can only reclaim a declined internal reservation.
        # It resumes the same indices, statistics and seed stream before freezing.
        for training_pass in range(2):
            for simulation in range(completed, training_budget if preparation_failure is None else 0):
                if (not self.quality_reference and not joint_training_hard
                        and simulation >= len(coverage) and time.perf_counter() >= soft_deadline):
                    failure = 'soft_budget_reached'
                    break
                paired = simulation < len(coverage)
                if paired:
                    key, seed = coverage[simulation]
                else:
                    key = self._select(node)
                    if self.fresh_adaptive_training:
                        # The disjoint range keeps these scenarios independent
                        # of paired coverage and validation without consuming
                        # their RNG stream. Neither seed enters the public key.
                        seed = self._adaptive_rng.randrange(2**31, 2**32)
                        while (seed in adaptive_seeds or self.deterministic_validation_cohort
                                and seed in self._training_seed_history):
                            seed = self._adaptive_rng.randrange(2**31, 2**32)
                        adaptive_seeds.add(seed)
                    else:
                        # Repeated traversals improve downstream choices but
                        # are search effort, not additional independent samples.
                        seed = (seeds[current_counts[key] % len(seeds)] if self.scenario_reuse
                                else self._rng.randrange(2**31))
                try:
                    if self._continuation is not None:
                        self._continuation.start('training')
                    ordinary_training_started += 1
                    if self.deterministic_validation_cohort and seed not in self._training_seed_history:
                        # Failed training can still create routing nodes. Keep its
                        # world excluded from later validation until an explicit reset.
                        self._training_seed_history.add(seed)
                        self.last_validation_cohort_evidence['training_seed_history'] = sorted(
                            self._training_seed_history)
                    score, path, depth, calls, fallbacks = self._timed_search_call(
                        'training', self._simulate, request, seed, forced_roots[key], deadline)
                except TimeoutError:
                    failure = 'timeout'
                    break
                except (ValueError, RuntimeError) as error:
                    failure = str(error)
                    break
                if time.perf_counter() >= deadline:
                    failure = 'timeout'
                    break
                completed += 1
                if self.single_node_expansion:
                    self._training_expansion['completed_training_rollouts'] += 1
                    self._training_expansion['completed_free_expansions'] += self._last_training_expansion[0]
                    self._training_expansion['completed_prior_route_steps'] += self._last_training_expansion[1]
                if self.fresh_adaptive_training and not paired:
                    completed_adaptive_seeds.add(seed)
                if self._continuation is not None:
                    self._continuation.completed['training'] += 1
                maximum_depth = max(maximum_depth, depth)
                terminal_calls += calls
                terminal_fallbacks += fallbacks
                for statistic, before_score in path:
                    statistic[0] += 1
                    statistic[1] += score - before_score
                if options is not None:
                    node.statistics[key][0] += 1
                    node.statistics[key][1] += score - request.state.score
                current_counts[key] += 1
                current_totals[key] += score - request.state.score
                counts = scenario_visits[key]
                counts[seed] = counts.get(seed, 0) + 1
                totals = scenario_totals[key]
                totals[seed] = totals.get(seed, 0.) + score - request.state.score
                if paired:
                    paired_counts[key] += 1
                    paired_totals[key] += score - request.state.score
            if training_pass or internal_prepared is None:
                break
            release = internal_report['reservation_release']
            release['training_before_release'] = completed
            reduced = training_budget < original_training_budget
            if (not all(count >= self.min_root_samples for count in paired_counts.values())
                    or failure not in (None, 'soft_budget_reached')):
                if reduced:
                    release.update(status='held', reason='training_failed')
                break
            try:
                # Select once on the trained tree, before either freeze callback.
                # Later training cannot reopen this internal decision window.
                internal_selection = self._timed_search_call(
                    'preparation', internal.select, internal_prepared,
                    node_lookup=lambda current, legal: self._physical_node(current, legal)[1],
                    deadline=deadline, diagnostics=internal_report,
                    **(dict(most_visited=True) if self.internal_successor_graph else {}))
                if not reduced:
                    break
                if internal_selection is not None:
                    release.update(status='held', reason='internal_selected')
                    break
                if failure == 'soft_budget_reached':
                    release.update(status='held', reason='training_soft_deadline')
                    break
                from .hif_resource_reanalysis import new_diagnostics as resource_diagnostics

                prepared_resource_diagnostics = resource_diagnostics()
                prepared_resource = self._timed_search_call(
                    'preparation', self._prepare_resources, request,
                    budget=self.simulations - completed - minimum_validation, deadline=deadline,
                    diagnostics=prepared_resource_diagnostics, retain_budget_proof=True)
                release['resource_precheck'] = deepcopy(prepared_resource_diagnostics)
                if prepared_resource.ready:
                    # This complete structure keeps precedence and its existing
                    # budget. Evaluate precisely this prepared graph after freeze.
                    release.update(status='held', reason='complete_resource_graph',
                                   resource_graph_protected=True)
                    break
                training_budget = original_training_budget
                release.update(status='released', reason='no_sparse_prior_disagreement',
                    effective_training_limit=training_budget,
                    released_simulations=training_budget - completed)
                # A first-graph budget proof may reuse its checked key prefix;
                # appended states still undergo the original validation path.
                declined_resource_precheck = prepared_resource
                prepared_resource = prepared_resource_diagnostics = None
            except (TimeoutError, ValueError, RuntimeError) as error:
                internal_report.update(status='failed', complete=False, reason=str(error))
                if reduced:
                    release.update(status='held', reason='training_failed')
                failure = 'mcts_internal_reanalysis_failed:' + str(error)
                break
        complete = all(count >= self.min_root_samples for count in paired_counts.values())
        training_completed = completed
        training_finished = time.perf_counter() if self.public_validation_phase_budget else None
        if self.complete_orderer is not None:
            # Independent validation reads a fixed policy. A new public state
            # cannot gain an exact ordering based on earlier validation seeds
            # consuming (or leaving) computation budget.
            self._timed_search_call('freeze', self.complete_orderer.freeze)
        if self.freeze_callback is not None:
            self._timed_search_call('freeze', self.freeze_callback)
        training_failure = failure or ('completed' if complete else 'root_coverage_incomplete')
        reanalysis = None
        reanalysis_completed = 0
        resource_report = None
        resource_completed = 0
        internal_completed = 0
        if self.public_turn_reanalysis:
            from .hif_public_turn_reanalysis import new_diagnostics

            reanalysis = prepared_diagnostics if prepared_diagnostics is not None else new_diagnostics()
            if complete and failure in (None, 'soft_budget_reached'):
                arguments = (dict(prepared=prepared_graph, diagnostics=reanalysis)
                             if self.public_turn_budget_reservation else {})
                reanalysis, reanalysis_depth, reanalysis_calls, reanalysis_fallbacks = self._timed_search_call(
                    'public_graph', self._reanalyse_public_turn, request, seeds, root_options=options,
                    budget=self.simulations - completed - minimum_validation,
                    deadline=deadline, **arguments)
                reanalysis_completed = reanalysis['completed_simulations']
                maximum_depth = max(maximum_depth, reanalysis_depth)
                terminal_calls += reanalysis_calls
                terminal_fallbacks += reanalysis_fallbacks
                if reanalysis['status'] == 'failed':
                    failure = 'mcts_public_turn_reanalysis_failed:' + reanalysis['reason']
        if self.resource_reanalysis:
            from .hif_resource_reanalysis import new_diagnostics as resource_diagnostics

            resource_report = resource_diagnostics()
            if (complete and failure in (None, 'soft_budget_reached')
                    and reanalysis is not None and reanalysis['status'] == 'declined'):
                resource_report, local_depth, local_calls, local_fallbacks = self._timed_search_call(
                    'resource', self._reanalyse_resources, request, seeds, budget=self.simulations - completed - reanalysis_completed
                    - minimum_validation, deadline=deadline,
                    **(dict(prepared=prepared_resource, diagnostics=prepared_resource_diagnostics)
                       if prepared_resource is not None else
                       dict(previous=declined_resource_precheck) if declined_resource_precheck is not None else {}))
                resource_completed = resource_report['completed_simulations']
                maximum_depth = max(maximum_depth, local_depth)
                terminal_calls += local_calls
                terminal_fallbacks += local_fallbacks
                if resource_report['status'] == 'failed':
                    failure = 'mcts_resource_reanalysis_failed:' + resource_report['reason']
            elif reanalysis is not None and reanalysis['complete']:
                resource_report.update(status='declined', reason='complete_public_graph')
        if internal_selection is not None and complete and failure in (None, 'soft_budget_reached'):
            if (reanalysis is not None and reanalysis['status'] == 'declined'
                    and resource_report is not None and resource_report['status'] == 'declined'
                    and not reanalysis['started_simulations'] and not resource_report['started_simulations']):
                internal_report, local_depth, local_calls, local_fallbacks = self._timed_search_call(
                    'internal', self._reanalyse_internal, internal_selection, seeds, budget=self.simulations - completed - minimum_validation,
                    deadline=deadline, diagnostics=internal_report, validation_seeds=validation_cohort)
                internal_completed = internal_report['completed_simulations']
                maximum_depth = max(maximum_depth, local_depth)
                terminal_calls += local_calls
                terminal_fallbacks += local_fallbacks
                if internal_report['status'] == 'failed':
                    failure = 'mcts_internal_reanalysis_failed:' + internal_report['reason']
            else:
                internal_report.update(status='declined', eligible=False,
                                       reason='existing_public_or_resource_policy')
        validation_counts = {key: 0 for key in keys}
        validation_samples = {key: [] for key in keys}
        validation_failure = 'not_run' if self.validation_samples else 'disabled'
        validation_limit = self.validation_samples
        if (self.adaptive_validation and not self._effective_continuation_samples
                and not self.paired_adaptive_validation):
            validation_limit = max(self.validation_samples, min(32,
                (self.simulations - completed - reanalysis_completed - resource_completed
                 - internal_completed) // validation_width))
        uniform_guard = None
        if self.uniform_validation_time_guard:
            from .hif_uniform_validation_time_guard import UniformValidationTimeGuard

            uniform_guard = UniformValidationTimeGuard(capacity=validation_limit,
                root_count=len(keys), retained_present=retained_comparison is not None, deadline=deadline,
                window_limit_seconds=self.timeout_ms / 1000)
            self.last_uniform_validation_time_guard = uniform_guard.report
        if self.lazy_validation_seeds:
            self.last_validation_seed_allocation['capacity'] = validation_limit
        if self.validation_samples and complete and failure in (None, 'soft_budget_reached'):
            validation_soft_deadline = soft_deadline
            if self.public_validation_phase_budget:
                phase = self.last_public_validation_phase_budget
                phase['reason'] = 'public_graph_incomplete'
                if reanalysis is not None and reanalysis['complete']:
                    validation_started = time.perf_counter()
                    validation_soft_deadline = (deadline if self.quality_reference else
                        min(deadline, validation_started + self.soft_timeout_ms / 1000))
                    phase.update(eligible=True, reason='complete_public_graph',
                        soft_window_ms=0. if self.quality_reference else self.soft_timeout_ms,
                        training_elapsed_ms=(training_finished - started) * 1000,
                        graph_elapsed_ms=(validation_started - training_finished) * 1000,
                        validation_start_elapsed_ms=(validation_started - started) * 1000,
                        soft_deadline_elapsed_ms=(validation_soft_deadline - started) * 1000)
            # No more learning after this point. Each root receives the same
            # new scenarios under the same frozen continuation policy. These
            # returns are never written to the reusable training tree.
            # The optional lazy path also supports unchanged four-scenario
            # training: unused round capacity must not advance the next
            # window's RNG. Keep mandatory seeds before any inner cohort.
            lazy_outer = self.fresh_adaptive_training or self.lazy_validation_seeds
            validation_excluded = (*seeds, *self._training_seed_history) if self.deterministic_validation_cohort else seeds
            if revealed is not None:
                validation_excluded = (*validation_excluded,
                    *retained_policy.policy.search._training_seed_history,
                    *retained_policy.validation_excluded_seeds)
            validation_seeds = validation_cohort or self._fresh_seeds(
                self.validation_samples if lazy_outer else validation_limit, validation_excluded)
            if self.lazy_validation_seeds:
                self.last_validation_seed_allocation['generated'] = len(validation_seeds)
            inner_seeds = ()
            if self._effective_continuation_samples:
                # The fixed independent cohort defines one public local
                # selector for every outer. Single-sample RNG is unchanged.
                inner_seeds = self._fresh_seeds(self._effective_continuation_samples,
                                                (*seeds, *validation_seeds))
                if self._effective_continuation_samples == 1:
                    self._continuation.seed = inner_seeds[0]
                else:
                    self._continuation.seeds = inner_seeds
            validation_failure = None
            parallel_validation = None
            if self.parallel_public_graph and self.uniform_validation_time_guard and validation_limit != 4:
                self.last_parallel_root_validation = dict(version=self._parallel_root_validation_version,
                    status='serial', reason='validation_limit_not_four', complete=False, workers=4,
                    started_simulations=0, completed_simulations=0, count_incomplete=False,
                    elapsed_seconds=0., worker_records=[], seeds=tuple(validation_seeds),
                    root_keys=tuple(forced_roots[key].key if isinstance(forced_roots[key], _RootOption)
                        else (key, ()) for key in keys))
            if self.parallel_public_graph and (self.paired_adaptive_validation
                    or self.uniform_validation_time_guard and validation_limit == 4):
                complete_roots = tuple(forced_roots[key] if isinstance(forced_roots[key], _RootOption)
                    else _RootOption(action, (), None) for action, key in zip(actions, keys))
                try:
                    parallel_validation = self._validate_parallel_roots(
                        request, complete_roots, validation_seeds, deadline)
                except (TimeoutError, ValueError, RuntimeError) as error:
                    parallel_validation = self.last_parallel_root_validation
                    validation_failure = failure = ('timeout' if isinstance(error, TimeoutError) else str(error))
                if parallel_validation is not None and parallel_validation['status'] != 'serial':
                    if uniform_guard is not None:
                        uniform_guard.report.update(timing_scope='PARALLEL_BATCH_NO_OPTIONAL_ROUNDS',
                            max_completed_round_seconds=None)
                    ordinary_validation_started += parallel_validation['started_simulations']
                    for record in parallel_validation['worker_records']:
                        scenario, seed = record['world_index'], record['seed']
                        if record['started_simulations']:
                            if self.lazy_validation_seeds:
                                self.last_validation_seed_allocation['started'] += 1
                            if self.deterministic_validation_cohort:
                                self.last_validation_cohort_evidence['validation_seeds'].append(seed)
                                count = 'reused_count' if scenario < len(validation_cohort) else 'new_seed_count'
                                self.last_validation_cohort_evidence[count] += 1
                        for row in record['rows']:
                            key = keys[row['group_index']]
                            if row['key'] != complete_roots[row['group_index']].key:
                                raise ValueError('mcts_parallel_root_result_binding_mismatch')
                            completed += 1
                            maximum_depth = max(maximum_depth, row['depth'])
                            terminal_calls += row['terminal_calls']
                            terminal_fallbacks += row['terminal_fallbacks']
                            validation_counts[key] += 1
                            for name, value in row['counters'].items():
                                setattr(self, name, getattr(self, name) + value)
                            validation_samples[key].append(dict(scenario_index=scenario,
                                planning_seed=seed, final_score=row['final_score'],
                                return_kind=row['return_kind'], completed_depth=row['depth'],
                                terminal_evaluations=row['terminal_calls'],
                                terminal_fallbacks=row['terminal_fallbacks'],
                                elapsed_seconds=row['elapsed_seconds']))
                else:
                    parallel_validation = None
            for scenario in range(validation_limit):
                if validation_failure is not None:
                    break
                if (self.public_validation_phase_budget
                        and self.last_public_validation_phase_budget['eligible']
                        and time.perf_counter() >= deadline):
                    validation_failure = failure = 'timeout'
                    break
                # Only complete public graphs may start a separate soft
                # phase. It includes the mandatory rounds and is never reset
                # per round; hard time and total traversal limits stay shared.
                if (not self.quality_reference and self.adaptive_validation and scenario >= self.validation_samples
                        and time.perf_counter() >= validation_soft_deadline):
                    validation_failure = 'soft_budget_reached'
                    break
                if uniform_guard is not None:
                    try:
                        if not uniform_guard.can_start(scenario):
                            break
                    except TimeoutError:
                        validation_failure = failure = 'timeout'
                        break
                if lazy_outer and scenario >= len(validation_seeds):
                    if time.perf_counter() >= deadline:
                        validation_failure = failure = 'timeout'
                        break
                    validation_seeds += self._fresh_seeds(
                        1, (*validation_excluded, *adaptive_seeds, *validation_seeds, *inner_seeds))
                    if self.lazy_validation_seeds:
                        self.last_validation_seed_allocation['generated'] = len(validation_seeds)
                seed = validation_seeds[scenario]
                if uniform_guard is not None and time.perf_counter() >= deadline:
                    validation_failure = failure = 'timeout'
                    break
                if self.lazy_validation_seeds and parallel_validation is None:
                    self.last_validation_seed_allocation['started'] += 1
                if self.deterministic_validation_cohort and parallel_validation is None:
                    self.last_validation_cohort_evidence['validation_seeds'].append(seed)
                    count = 'reused_count' if scenario < len(validation_cohort) else 'new_seed_count'
                    self.last_validation_cohort_evidence[count] += 1
                if uniform_guard is not None:
                    round_started = time.perf_counter()
                    root_started_before = ordinary_validation_started
                    root_completed_before = sum(validation_counts.values())
                    old_started_before = retained_comparison['started_simulations'] if retained_comparison else 0
                    old_completed_before = retained_comparison['completed_simulations'] if retained_comparison else 0
                try:
                    for key in (() if parallel_validation is not None else keys):
                        tail_hits_before = self._validation_tail_hits
                        self._outer_continuation = None
                        self._outer_unimproved_count = 0
                        try:
                            if self.single_node_expansion:
                                self._validation_started += 1
                            if self._continuation is not None:
                                self._continuation.start('outer')
                            ordinary_validation_started += 1
                            if self.paired_adaptive_validation:
                                return_started = time.perf_counter()
                            score, path, depth, calls, fallbacks = self._timed_search_call(
                                'root_validation', self._validate_simulation, request, seed, forced_roots[key], deadline)
                            if self.paired_adaptive_validation:
                                return_seconds = time.perf_counter() - return_started
                            if path:
                                raise ValueError('mcts_validation_mutable_path')
                        except TimeoutError:
                            validation_failure = failure = 'timeout'
                            break
                        except (ValueError, RuntimeError) as error:
                            validation_failure = failure = str(error)
                            break
                        if uniform_guard is None and time.perf_counter() >= deadline:
                            validation_failure = failure = 'timeout'
                            break
                        completed += 1
                        if self._continuation is not None:
                            self._continuation.completed['outer'] += 1
                        maximum_depth = max(maximum_depth, depth)
                        terminal_calls += calls
                        terminal_fallbacks += fallbacks
                        validation_counts[key] += 1
                        # Diagnostic-only values from this completed traversal.
                        # Seeds belong to planning scenarios, never the live RNG.
                        # Validation has no recorded action path; do not invent
                        # one or replay a scenario merely to produce a trace.
                        validation_samples[key].append(dict(
                            scenario_index=scenario, planning_seed=seed, final_score=score,
                            return_kind=('proven_tail_expected_score' if self._validation_tail_hits > tail_hits_before
                                         else 'terminal_evaluator_expected_score' if calls > fallbacks
                                         else 'sampled_terminal_score'),
                            completed_depth=depth, terminal_evaluations=calls, terminal_fallbacks=fallbacks))
                        if self.paired_adaptive_validation:
                            validation_samples[key][-1]['elapsed_seconds'] = return_seconds
                        if self._effective_continuation_samples:
                            validation_samples[key][-1]['local_continuation'] = self._outer_continuation
                            if self.continuation_domain:
                                validation_samples[key][-1]['unimproved_public_domain'] = self._outer_unimproved_count
                        if uniform_guard is not None and time.perf_counter() >= deadline:
                            validation_failure = failure = 'timeout'
                            break
                    if validation_failure is not None:
                        break
                    if retained_comparison is not None:
                        try:
                            if time.perf_counter() >= deadline:
                                raise TimeoutError('mcts_window_timeout')
                            retained_comparison['started_simulations'] += 1
                            if self.paired_adaptive_validation:
                                return_started = time.perf_counter()
                            score, path, depth, calls, fallbacks = self._timed_search_call(
                                'retained_validation', retained_policy.evaluate, request, seed, deadline)
                            if self.paired_adaptive_validation:
                                return_seconds = time.perf_counter() - return_started
                            if (path or type(score) not in (int, float) or not math.isfinite(score)
                                    or type(depth) is not int or not 0 <= depth <= 256
                                    or type(calls) is not int or type(fallbacks) is not int
                                    or not 0 <= fallbacks <= calls):
                                raise ValueError('mcts_retained_policy_invalid_return')
                            if uniform_guard is None and time.perf_counter() >= deadline:
                                raise TimeoutError('mcts_window_timeout')
                        except TimeoutError:
                            validation_failure = failure = 'timeout'
                            break
                        except (ValueError, RuntimeError) as error:
                            validation_failure = failure = str(error)
                            break
                        completed += 1
                        retained_comparison['completed_simulations'] += 1
                        maximum_depth = max(maximum_depth, depth)
                        terminal_calls += calls
                        terminal_fallbacks += fallbacks
                        retained_comparison['validation_samples'].append(dict(
                            scenario_index=scenario, planning_seed=seed, final_score=score,
                            return_kind=('terminal_evaluator_expected_score' if calls > fallbacks
                                         else 'sampled_terminal_score'),
                            completed_depth=depth, terminal_evaluations=calls, terminal_fallbacks=fallbacks))
                        if self.paired_adaptive_validation:
                            retained_comparison['validation_samples'][-1]['elapsed_seconds'] = return_seconds
                        if uniform_guard is not None and time.perf_counter() >= deadline:
                            validation_failure = failure = 'timeout'
                            break
                    if self.lazy_validation_seeds:
                        self.last_validation_seed_allocation['complete_rounds'] += 1
                finally:
                    if uniform_guard is not None:
                        uniform_guard.record_round(index=scenario, seed=seed, started=round_started,
                            root_started=(len(keys) if parallel_validation is not None else
                                ordinary_validation_started - root_started_before),
                            root_completed=(len(keys) if parallel_validation is not None else
                                sum(validation_counts.values()) - root_completed_before),
                            retained_started=(retained_comparison['started_simulations'] - old_started_before
                                if retained_comparison else 0),
                            retained_completed=(retained_comparison['completed_simulations'] - old_completed_before
                                if retained_comparison else 0), complete=validation_failure is None)
        complete_round_counts = list(validation_counts.values())
        if retained_comparison is not None:
            complete_round_counts.append(retained_comparison['completed_simulations'])
        if self.public_validation_phase_budget and self.last_public_validation_phase_budget['eligible']:
            self.last_public_validation_phase_budget.update(
                validation_elapsed_ms=(time.perf_counter() - validation_started) * 1000,
                complete_rounds=min(complete_round_counts, default=0))
        validation_complete = (bool(self.validation_samples)
            and validation_failure in (None, 'soft_budget_reached')
            and len(set(complete_round_counts)) == 1
            and all(count >= self.validation_samples for count in complete_round_counts)
            and (uniform_guard is None or uniform_guard.report['completed_rounds']
                 == min(complete_round_counts, default=0)))
        if retained_comparison is not None:
            retained_comparison['complete'] = validation_complete
            samples = retained_comparison['validation_samples']
            retained_comparison['mean_final_score'] = (
                sum(sample['final_score'] for sample in samples) / len(samples) if samples else None)
        training_means = {key: (request.state.score + sum(
            scenario_totals[key][seed] / count for seed, count in scenario_visits[key].items())
            / len(scenario_visits[key]) if scenario_visits[key] else None) for key in keys}
        rows = tuple(dict(action=action, visits=current_counts[key],
                          root_paired_visits=paired_counts[key], retained_visits=node.statistics[key][0],
                          inherited_visits=inherited_visits[key],
                          mean_final_score=(request.state.score + node.statistics[key][1] / node.statistics[key][0]
                                            if node.statistics[key][0] else None),
                          window_mean_final_score=(request.state.score + current_totals[key] / current_counts[key]
                                                   if current_counts[key] else None),
                          paired_mean_final_score=(request.state.score + paired_totals[key] / paired_counts[key]
                                                   if paired_counts[key] else None),
                          scenario_mean_final_score=training_means[key],
                          independent_scenarios=len(scenario_visits[key]),
                          scenario_visits=tuple(scenario_visits[key].values()),
                          training_mean_final_score=(training_means[key] if self.scenario_reuse else
                              request.state.score + current_totals[key] / current_counts[key]
                              if current_counts[key] else None),
                          training_visits=current_counts[key], training_paired_visits=paired_counts[key],
                          validation_paired_visits=validation_counts[key],
                          validation_samples=tuple(validation_samples[key]),
                          validation_mean_final_score=(sum(sample['final_score']
                              for sample in validation_samples[key]) / validation_counts[key]
                                                       if validation_counts[key] else None))
                     for action, key in zip(actions, keys))
        if options is not None:
            rows = tuple(dict(row, continuation=option.continuation, root_option_key=option.key)
                         for row, option in zip(rows, options))
        base_completed = completed
        finalist = None
        if self.paired_finalist_validation:
            finalist, finalist_depth, finalist_calls, finalist_fallbacks = self._timed_search_call(
                'finalist_validation', self._validate_finalists, request, rows, keys, forced_roots, base_completed=base_completed,
                base_complete=complete and validation_complete,
                validation_limit=validation_limit, deadline=deadline)
            completed += finalist['completed_simulations']
            maximum_depth = max(maximum_depth, finalist_depth)
            terminal_calls += finalist_calls
            terminal_fallbacks += finalist_fallbacks
            if finalist['eligible'] and not finalist['complete']:
                failure = finalist['reason']
        adaptive_pairs = None
        if self.paired_adaptive_validation and complete and validation_complete and failure is None:
            paired_excluded = (*validation_excluded, *validation_seeds, *adaptive_seeds)
            if retained_policy is not None:
                paired_excluded += (*retained_policy.policy.search._training_seed_history,
                                    *retained_policy.validation_excluded_seeds)
            adaptive_pairs, paired_depth, paired_calls, paired_fallbacks = self._timed_search_call(
                'paired_adaptive_validation', self._validate_adaptive_pairs,
                request, rows, keys, forced_roots, retained_policy, retained_comparison,
                budget=self.simulations - completed - reanalysis_completed - resource_completed - internal_completed,
                deadline=deadline, excluded_seeds=paired_excluded)
            completed += adaptive_pairs['completed_simulations']
            maximum_depth = max(maximum_depth, paired_depth)
            terminal_calls += paired_calls
            terminal_fallbacks += paired_fallbacks
            if not adaptive_pairs['complete']:
                failure = adaptive_pairs['reason']
        accepted = complete and failure in (None, 'timeout', 'soft_budget_reached')
        if self.quality_reference:
            accepted = (accepted and failure is None and training_completed == training_budget
                        and (not self.validation_samples or validation_failure is None
                             and (uniform_guard is not None and validation_complete
                                  or all(count == validation_limit for count in complete_round_counts))))
        if self.validation_samples:
            accepted = accepted and validation_complete
        if finalist is not None and finalist['eligible']:
            accepted = accepted and finalist['complete']
        if self.paired_adaptive_validation:
            accepted = accepted and adaptive_pairs is not None and adaptive_pairs['complete']
        if uniform_guard is not None:
            accepted = uniform_guard.finish(complete=accepted,
                reason=failure or validation_failure) and accepted
            if not uniform_guard.report['finished_before_deadline']:
                failure = validation_failure = 'timeout'
                validation_complete = False
                if retained_comparison is not None:
                    retained_comparison['complete'] = False
        selection_field = ('validation_mean_final_score' if self.validation_samples else
                           'scenario_mean_final_score' if self.scenario_reuse else 'mean_final_score')
        selected_index, retained_wins, root_eligibility = self._select_completed_root(
            request, rows, physical_actions, accepted=accepted, selection_field=selection_field,
            deadline=deadline, finalist=finalist, adaptive_pairs=adaptive_pairs,
            retained_comparison=retained_comparison)
        selected_row = rows[selected_index] if selected_index is not None else None
        retained_row = max(rows, key=lambda row: row['mean_final_score']) if accepted else None
        selected = selected_row['action'] if selected_row is not None else None
        selection_key = 'root_option_key' if options is not None else 'action'
        selection_changed = (selected_row is not None
                             and selected_row[selection_key] != retained_row[selection_key])
        selected_option = (forced_roots[selected_row['root_option_key']]
                           if options is not None and selected_row is not None else None)
        if retained_comparison is not None and accepted:
            retained_comparison['selected'] = retained_wins
            if retained_comparison['selected']:
                selected = retained_comparison['action']
                selected_option = None
                selection_changed = action_key(selected) != action_key(retained_row['action'])
        multi_visits = [sum(statistic[0] for statistic in entry.statistics.values())
                        for entry in self._tree.values() if len(entry.statistics) > 1]
        self._expected_state = request.state
        local = self._continuation
        training_started = (local.started['training'] if local else
                            self._training_expansion['started_training_rollouts']
                            if self.single_node_expansion else ordinary_training_started
                            if self.public_turn_reanalysis or self.quality_reference else 0)
        validation_started = (local.started['outer'] if local else
                              self._validation_started if self.single_node_expansion else ordinary_validation_started
                              if self.public_turn_reanalysis or self.quality_reference else 0)
        total_started = local.total_started if local else training_started + validation_started
        if retained_comparison is not None:
            total_started += retained_comparison['started_simulations']
        if adaptive_pairs is not None:
            total_started += adaptive_pairs['started_simulations']
        if reanalysis is not None:
            total_started += reanalysis['started_simulations']
        if resource_report is not None:
            total_started += resource_report['started_simulations']
        if internal_report is not None:
            total_started += internal_report['started_simulations']
            # The initial window allocates fresh outers after local training.
            # Report the actual cohort, rather than treating an empty future
            # cohort as proof that training and validation were separated.
            outer_seeds = tuple(self.last_validation_cohort_evidence['validation_seeds'])
            internal_report['validation_seeds'] = outer_seeds
            internal_report['training_validation_disjoint'] = bool(
                len(outer_seeds) >= 4 and internal_report['planning_seeds']
                and not set(outer_seeds).intersection(internal_report['planning_seeds']))
        expansion_summary = None
        if self.single_node_expansion:
            trained = sum(any(row[0] for row in entry.statistics.values()) for entry in self._tree.values())
            expansion_summary = dict(self._training_expansion, per_rollout_limit=1,
                trained_nodes=trained, zero_visit_route_nodes=len(self._tree) - trained)
        self._finish_search_phase_timing()
        estimate = MctsEstimate(selected, completed + reanalysis_completed + resource_completed + internal_completed
                            + (local.completed['inner'] if local else 0),
                            rows, (time.perf_counter() - started) * 1000,
                            single_node_expansion=self.single_node_expansion,
                            training_expansion=expansion_summary,
                            finalist_validation=finalist,
                            root_selection_eligibility=root_eligibility,
                            public_turn_reanalysis=reanalysis,
                            resource_reanalysis=resource_report,
                            internal_reanalysis=internal_report,
                            paired_adaptive_validation=adaptive_pairs,
                            uniform_validation_time_guard=self.last_uniform_validation_time_guard,
                            parallel_root_validation=self.last_parallel_root_validation,
                            retained_policy_comparison=retained_comparison,
                            continuation_samples=self._effective_continuation_samples,
                            requested_continuation_samples=self.continuation_samples,
                            continuation_domain=self.continuation_domain,
                            continuation_incomplete_actions=self.continuation_incomplete_actions,
                            continuation_portfolio=self.continuation_portfolio,
                            portfolio_training_allocation=portfolio_training_allocation,
                            continuation_trigger=(POLICY_CONTINUATION_TRIGGER
                                                  if self.continuation_portfolio else None),
                            partial_node_continuation_calls=self._partial_node_calls,
                            partial_node_continuation_cache_hits=self._partial_node_cache_hits,
                            partial_node_new_actions_evaluated=self._partial_node_new_actions,
                            fresh_adaptive_training=self.fresh_adaptive_training,
                            fresh_adaptive_training_simulations=len(completed_adaptive_seeds),
                            fresh_adaptive_training_distinct_scenarios=len(completed_adaptive_seeds),
                            training_distinct_scenarios=len({seed for visits in scenario_visits.values()
                                                            for seed in visits}),
                            domain_limit=self._domain_limit,
                            unimproved_count=self._unimproved_count,
                            continuation_mode=('conditional_public_domain' if self._effective_continuation_samples
                                               and self.continuation_domain else
                                               'strict' if self._effective_continuation_samples else
                                               'baseline_domain' if self.continuation_domain else 'disabled'),
                            baseline_domain_reason=baseline_domain_reason,
                            started_simulations=total_started,
                            training_started_simulations=training_started,
                            validation_started_simulations=validation_started,
                            inner_started_simulations=local.started['inner'] if local else 0,
                            inner_completed_simulations=local.completed['inner'] if local else 0,
                            unused_simulation_budget=(self.simulations - total_started
                                                      if local or self.single_node_expansion
                                                      or self.public_turn_reanalysis or self.quality_reference else 0),
                            local_continuation_calls=local.calls if local else 0,
                            local_continuation_cache_hits=local.cache_hits if local else 0,
                            inner_terminal_evaluations=local.terminal_evaluations if local else 0,
                            inner_terminal_fallbacks=local.terminal_fallbacks if local else 0,
                            inner_tail_hits=local.tail_hits if local else 0,
                            inner_tail_misses=local.tail_misses if local else 0,
                            inner_terminal_eligible=local.terminal_eligible if local else 0,
                            inner_terminal_declined=local.terminal_declined if local else 0,
                            requested_simulations=self.simulations,
                            root_options_enabled=options is not None,
                            selected_continuation=(selected_option.continuation if selected_option else ()),
                            selected_expected_after_first=(selected_option.expected_after_first
                                                           if selected_option else None),
                            physical_root_count=physical_root_count, root_option_count=len(keys),
                            joint_training_uses_hard_deadline=joint_training_hard,
                            training_simulation_budget=training_budget,
                            adaptive_training_completed_simulations=max(0, training_completed - len(coverage)),
                            stop_reason=(failure or ('completed' if complete else 'root_coverage_incomplete')),
                            complete_root_coverage=complete,
                            root_paired_visits=tuple(paired_counts.values()), retained_nodes=retained,
                            expanded_nodes=len(self._tree) - retained, maximum_depth=maximum_depth,
                            terminal_evaluations=terminal_calls, terminal_fallbacks=terminal_fallbacks,
                            terminal_horizon=self.terminal_horizon,
                            validation_tail_lookup_enabled=self.validation_tail_lookup is not None,
                            validation_tail_hits=self._validation_tail_hits,
                            validation_tail_misses=self._validation_tail_misses,
                            terminal_eligibility_enabled=self.terminal_eligible is not None,
                            terminal_eligibility_horizons=self.terminal_eligibility_horizons,
                            training_terminal_eligible=self._training_terminal_eligible,
                            training_terminal_declined=self._training_terminal_declined,
                            validation_terminal_eligible=self._validation_terminal_eligible,
                            validation_terminal_declined=self._validation_terminal_declined,
                            scenario_reuse=self.scenario_reuse,
                            root_selection_scope=('paired_retained_policy_competition'
                                                  if retained_comparison is not None else
                                                  'screened_finalist_frozen_policy_validation'
                                                  if finalist is not None and finalist['complete'] else
                                                  'training_disjoint_reused_resource_policy_validation'
                                                  if resource_report is not None and resource_report['complete']
                                                  and validation_cohort else
                                                  'independent_resource_policy_validation'
                                                  if resource_report is not None and resource_report['complete'] else
                                                  'training_disjoint_reused_public_turn_validation'
                                                  if reanalysis is not None and reanalysis['complete']
                                                  and validation_cohort else
                                                  'independent_public_turn_reanalysis_validation'
                                                  if reanalysis is not None and reanalysis['complete'] else
                                                  'independent_conditional_policy_validation'
                                                  if self._effective_continuation_samples else
                                                  'training_disjoint_reused_validation' if validation_cohort else
                                                  'independent_frozen_policy_validation' if self.validation_samples else
                                                  'current_equal_weight_scenarios' if self.scenario_reuse
                                                  else 'retained_complete_returns'),
                            selection_changed_from_retained=selection_changed,
                            tree_multi_action_nodes=sum(count > 0 for count in multi_visits),
                            tree_revisited_multi_action_nodes=sum(count > 1 for count in multi_visits),
                            training_completed_simulations=training_completed,
                            validation_completed_simulations=(base_completed - training_completed
                                - (retained_comparison['completed_simulations'] if retained_comparison else 0)),
                            requested_validation_samples=self.validation_samples,
                            complete_validation_coverage=validation_complete,
                            validation_paired_visits=tuple(validation_counts.values()),
                            training_stop_reason=training_failure,
                            validation_stop_reason=validation_failure or 'completed',
                            adaptive_validation=self.adaptive_validation and not self._effective_continuation_samples,
                            validation_sample_limit=validation_limit,
                            progressive_widening=self.progressive_widening,
                            turn_chain_continuation=(self.turn_chain_continuation.summary()
                                if self.turn_chain_continuation is not None else None),
                            complete_ordering=({name: getattr(self.complete_orderer, name)
                                for name in ('calls', 'cache_hits', 'completed', 'failed', 'elapsed_ms')}
                                if self.complete_orderer is not None else None))

        if uniform_guard is not None and not uniform_guard.finish(
                complete=accepted, reason=failure or validation_failure):
            if not uniform_guard.report['finished_before_deadline']:
                if retained_comparison is not None:
                    retained_comparison.update(complete=False, selected=False)
                if root_eligibility is not None:
                    root_eligibility.update(complete=False, reason='timeout', root_winner_index=None,
                        selected_retained=False)
                estimate = replace(estimate, action=None, selected_continuation=(),
                    selected_expected_after_first=None, complete_validation_coverage=False,
                    stop_reason='timeout', validation_stop_reason='timeout')
        return estimate
