"""One public-state continuation policy, committed for a single sampled suffix."""
from __future__ import annotations

import time

from .contracts import PlanType, ActionKind
from .hif_dominance import prove_unused_action_drink
from .hif_combo_timing import SenseComboTimingPolicy
from .validation_cache import validation_session, check_search_budget
from .hif_installed_score_continuation import InstalledScoreTurnChain

POLICY_IDS = ('frozen_tree', 'sense_timing')
UNKNOWN_TIMING_VERSION = 'unknown-node-sense-timing-v3'
RETRIEVAL_TRAINING_ORDER_VERSION = 'retrieval-training-order-v1'


class UnknownNodeTimingPolicy:
    """A stateless public-state suggestion, only for absent validation nodes.

    This calls deterministic single-step timing, optionally preceded by the
    installed-score current-turn comparison. Neither path uses the builder's
    stochastic search or a committed suffix. All legal actions remain available.
    """

    version = UNKNOWN_TIMING_VERSION
    timing_version = SenseComboTimingPolicy.version

    def __init__(self, kernel, baseline, *, installed_score_turn_chain=False, pair_order_advice=False,
                 extra_action_turn_chain=False, installed_score_multi_action=False,
                 installed_score_focus_projection=False, visited_card_extra_action=False,
                 visited_card_extra_action_version=None):
        if type(installed_score_turn_chain) is not bool:
            raise ValueError('installed_score_turn_chain_requires_bool')
        if (type(installed_score_multi_action) is not bool
                or installed_score_multi_action and not installed_score_turn_chain):
            raise ValueError('installed_score_multi_action_requires_installed_chain')
        if (type(installed_score_focus_projection) is not bool
                or installed_score_focus_projection and not installed_score_turn_chain):
            raise ValueError('installed_score_focus_projection_requires_installed_chain')
        if type(pair_order_advice) is not bool:
            raise ValueError('pair_order_advice_requires_bool')
        if type(extra_action_turn_chain) is not bool:
            raise ValueError('extra_action_turn_chain_requires_bool')
        if (type(visited_card_extra_action) is not bool
                or visited_card_extra_action and not extra_action_turn_chain):
            raise ValueError('visited_card_extra_action_requires_extra_action_chain')
        from .hif_extra_action_continuation import VISITED_CARD_VERSION, REFERENCE_VISITED_CARD_VERSION

        if (visited_card_extra_action_version is not None and (
                not visited_card_extra_action or visited_card_extra_action_version not in (
                    REFERENCE_VISITED_CARD_VERSION, VISITED_CARD_VERSION))):
            raise ValueError('visited_card_extra_action_version_not_supported')
        self.timing = SenseComboTimingPolicy(kernel, baseline)
        self.installed_score_policy = (InstalledScoreTurnChain(kernel,
            installed_score_multi_action=installed_score_multi_action,
            installed_score_focus_projection=installed_score_focus_projection)
            if installed_score_turn_chain else None)
        self.extra_action_policy = None
        if extra_action_turn_chain:
            from .hif_extra_action_continuation import ExtraActionTurnChain
            self.extra_action_policy = ExtraActionTurnChain(kernel,
                visited_card_extra_action=visited_card_extra_action,
                unused_drink_enabled=visited_card_extra_action_version != REFERENCE_VISITED_CARD_VERSION)
        self.pair_order_policy = None
        if pair_order_advice:
            from .hif_pair_order_advice import PairOrderAdvice
            self.pair_order_policy = PairOrderAdvice(kernel)
        self.reset()

    def reset(self):
        self.calls = self.chosen_calls = self.scope_declines = self.ambiguous_pair_declines = 0
        self.retrieval_no_pair_declines = 0
        self.timing.reset()
        if self.installed_score_policy is not None:
            self.installed_score_policy.reset()
        if self.extra_action_policy is not None:
            self.extra_action_policy.reset()
        if self.pair_order_policy is not None:
            self.pair_order_policy.reset_window()

    def start_window(self):
        self.reset()

    def summary(self):
        return dict(version=self.version, timing_version=self.timing_version,
                    calls=self.calls, chosen_calls=self.chosen_calls,
                    scope_declines=self.scope_declines,
                    ambiguous_pair_declines=self.ambiguous_pair_declines,
                    retrieval_no_pair_declines=self.retrieval_no_pair_declines)

    @staticmethod
    def _retrieval_pending(state):
        pending = state.pending_choice
        # Observed black-vinegar choices resume the drink's remaining effects
        # directly; no card/scheduled/turn continuation stack is required.
        # Other pending shapes retain the original prior-only scope.
        return (pending is not None and pending.source_id == 'drink:22'
                and pending.action_kind is ActionKind.MOVE_CARD
                and pending.continuation_index == 1
                and pending.remaining_effect_passes == 0
                and pending.selections_remaining == 1 and not pending.selection_optional)

    @staticmethod
    def _budget(deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_window_timeout')

    def choose(self, request, actions, deadline):
        installed_before = self.installed_score_policy.scope_hits if self.installed_score_policy is not None else 0
        selected = self._choose(request, actions, deadline)
        if selected is not None and self.pair_order_policy is not None:
            # A local suggestion only. The next public decision remains free;
            # training order, visited means and exact paths do not call here.
            selected = self.pair_order_policy.improve(request, actions, selected, deadline)
        installed_attempted = (self.installed_score_policy is not None
                               and self.installed_score_policy.scope_hits > installed_before)
        if self.extra_action_policy is not None and not installed_attempted:
            # Prove the final original suggestion, after its one pair-order
            # pass. A complete extra comparison must not be reranked again.
            extra = self.extra_action_policy.choose(request, actions, deadline, proposed_key=selected)
            if extra is not None:
                selected = extra
        if selected is not None:
            from .hif_mcts import action_key

            proposed = next((action for action in actions if action_key(action) == selected), None)
            if proposed is not None and proposed.kind is ActionKind.USE_DRINK:
                # This only changes a continuation suggestion, after its
                # original timing/pair/chain choice. Search legality and root
                # coverage stay intact; no second Unknown call is counted.
                with validation_session(lambda: self._budget(deadline), reuse_parent=True):
                    check_search_budget()
                    proof = prove_unused_action_drink(self.timing.kernel, request, proposed)
                    if (proof is None and self.extra_action_policy is not None
                            and self.extra_action_policy.unused_drink_enabled):
                        proof = self.extra_action_policy.unused_drink_proof(request, proposed, deadline)
                    if proof is not None:
                        self.timing.reset()
                        try:
                            replacement = self.timing.choose(request, actions, excluded_actions=(proposed,))
                            if (replacement not in actions or replacement == proposed
                                    or sum(action_key(a) == action_key(replacement) for a in actions) != 1):
                                raise ValueError('mcts_unknown_timing_unused_drink_replacement_invalid')
                            selected = action_key(replacement)
                        finally:
                            self.timing.reset()
                    check_search_budget()
        return selected

    def _choose(self, request, actions, deadline):
        # Import at use time: the search owns the action-key contract and
        # imports this module for its independent continuation adapter.
        from .hif_mcts import action_key

        self.timing.reset()
        try:
            with validation_session(lambda: self._budget(deadline), reuse_parent=True):
                check_search_budget()
                request.validate()
                if self.installed_score_policy is not None:
                    selected = self.installed_score_policy.choose(request, actions, deadline)
                    if selected is not None:
                        check_search_budget()
                        self.calls += 1
                        self.chosen_calls += 1
                        return selected
                state = request.state
                retrieval_pending = self._retrieval_pending(state)
                if (request.plan_type is not PlanType.SENSE or state.terminal
                        or state.turns_remaining <= 1
                        or state.pending_choice is not None and not retrieval_pending
                        or state.resolution_stack or state.pending_after_card_ids
                        or len(actions) <= 1):
                    check_search_budget()
                    self.calls += 1
                    self.scope_declines += 1
                    return None
                try:
                    selected = self.timing.choose(request, actions)
                except ValueError as error:
                    ambiguous = str(error) == 'combo_timing_pair_or_target_not_unique'
                    no_pair = (retrieval_pending
                               and str(error) == 'combo_timing_retrieval_has_no_verified_pair')
                    if not ambiguous and not no_pair:
                        raise
                    # Ambiguous suggestions do not remove legal alternatives;
                    # the original prior remains the unknown-node fallback.
                    check_search_budget()
                    self.calls += 1
                    if ambiguous:
                        self.ambiguous_pair_declines += 1
                    else:
                        self.scope_declines += 1
                        self.retrieval_no_pair_declines += 1
                    return None
                if selected not in actions:
                    raise ValueError('mcts_unknown_timing_action_not_legal')
                key = action_key(selected)
                if sum(action_key(action) == key for action in actions) != 1:
                    raise ValueError('mcts_unknown_timing_action_not_uniquely_legal')
                check_search_budget()
                self.calls += 1
                self.chosen_calls += 1
                return key
        finally:
            # Neither intervention logs nor committed-combo state belongs to
            # the next public decision (or another sampled future).
            self.timing.reset()


class RetrievalTrainingOrder:
    """Public timing advice orders first exploration, never removes a choice.

    The search calls this only for a new, unforced internal training node.
    Holding retrieval and ranking it first is the scope; no claim that a
    complete combination is absent is needed or made.
    """

    version = RETRIEVAL_TRAINING_ORDER_VERSION
    unknown_version = UNKNOWN_TIMING_VERSION
    timing_version = SenseComboTimingPolicy.version

    def __init__(self, kernel, baseline):
        # Training suggestions must not consume validation-helper counters or
        # any state from a previously selected hypothetical continuation.
        self.policy = UnknownNodeTimingPolicy(kernel, baseline)
        self.reset()

    def reset(self):
        self.calls = self.eligible_calls = self.reordered_calls = 0
        self.unchanged_calls = self.scope_declines = 0
        self.policy.reset()

    def start_window(self):
        self.reset()

    def summary(self):
        return dict(version=self.version, unknown_version=self.unknown_version,
                    timing_version=self.timing_version, calls=self.calls,
                    eligible_calls=self.eligible_calls, reordered_calls=self.reordered_calls,
                    unchanged_calls=self.unchanged_calls, scope_declines=self.scope_declines)

    def reorder(self, request, actions, old_order, deadline):
        from .hif_mcts import action_key

        with validation_session(lambda: self.policy._budget(deadline), reuse_parent=True):
            check_search_budget()
            request.validate()
            order = tuple(old_order)
            keys = tuple(action_key(action) for action in actions)
            if (len(set(keys)) != len(keys) or len(order) != len(keys)
                    or set(order) != set(keys)):
                raise ValueError('mcts_retrieval_training_order_action_mismatch')
            state = request.state
            first = next((action for action in actions if action_key(action) == order[0]), None) if order else None
            eligible = (request.plan_type is PlanType.SENSE and not state.terminal
                        and state.turns_remaining > 1 and state.pending_choice is None
                        and not state.resolution_stack and not state.pending_after_card_ids
                        and len(actions) > 1 and '22' in state.drinks
                        and first is not None and first.kind is ActionKind.USE_DRINK
                        and first.drink_id == '22')
            if not eligible:
                check_search_budget()
                self.calls += 1
                self.scope_declines += 1
                return order
            selected = self.policy.choose(request, actions, deadline)
            if selected is not None and selected not in keys:
                raise ValueError('mcts_retrieval_training_suggestion_not_legal')
            chosen = next((action for action in actions if action_key(action) == selected), None)
            reordered = (chosen is not None
                         and not (chosen.kind is ActionKind.USE_DRINK and chosen.drink_id == '22'))
            result = (selected, *(key for key in order if key != selected)) if reordered else order
            check_search_budget()
            self.calls += 1
            self.eligible_calls += 1
            self.reordered_calls += int(reordered)
            self.unchanged_calls += int(not reordered)
            return result


class PolicyContinuation:
    """No environment, random seed, tree mutation, or nested comparison access."""

    def __init__(self, policy_id, timing_factory, frozen_action, action_key, *, audit=None):
        if policy_id not in POLICY_IDS:
            raise ValueError('mcts_continuation_unknown_policy')
        self.policy_id = policy_id
        self.timing = timing_factory() if policy_id == 'sense_timing' else None
        self.frozen_action, self.action_key = frozen_action, action_key
        self._pending_key = None
        self.audit = audit if audit is not None else {}
        self.audit.update(committed_steps=0, policy_choose_calls=0,
                          shared_final_turn_steps=0, original_rule_forced_steps=0,
                          timing_committed_steps=0)

    @staticmethod
    def _budget(deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_window_timeout')

    def choose(self, request, actions, deadline):
        self._budget(deadline)
        if request.state.turns_remaining == 1 or self.timing is None:
            # Both alternatives retain the original final-turn tree policy
            # whenever the rollout's shared exact evaluator did not return.
            key = self.frozen_action(request, actions)
        else:
            # This continuation belongs to the same frozen planning window.
            # Retain its validation/normalization work without losing either
            # deadline, including choices that do not enter a kernel loop.
            with validation_session(lambda: self._budget(deadline), reuse_parent=True):
                check_search_budget()
                action = self.timing.choose(request, actions)
                check_search_budget()
            if action not in actions:
                raise ValueError('mcts_continuation_policy_action_not_legal')
            key = self.action_key(action)
        self._budget(deadline)
        self._pending_key = key
        self.audit['policy_choose_calls'] += 1
        return key

    def committed(self, before, action, after):
        # Called only after the original environment successfully advances.
        self.audit['committed_steps'] += 1
        self.audit['shared_final_turn_steps'] += int(before.turns_remaining == 1)
        self.audit['original_rule_forced_steps'] += int(self._pending_key != self.action_key(action))
        self._pending_key = None
        if self.timing is not None:
            self.timing.committed(before, action, after)
            self.audit['timing_committed_steps'] += 1
