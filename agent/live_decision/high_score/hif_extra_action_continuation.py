"""Complete current-turn advice after proving an available extra action."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

from .choice import _valid_branches
from .kernel import EffectKind
from .planner import SearchIncomplete
from .contracts import PlanType, ActionKind
from .hif_planner import HifActionChainPlanner
from .hif_choice_roots import _UNOBSERVED_EVENTS
from .validation_cache import validation_session, check_search_budget
from .hif_pair_order_advice import PairOrderAdvice
from .hif_turn_chain_continuation import TurnChainContinuation

# v4 also admits a fully visited frozen node's END suggestion. The bounded
# comparison and its public-state/extra-action proof remain the same.
VERSION = 'extra-action-last-action-chain-v4'
VISITED_CARD_VERSION = 'extra-action-last-action-chain-visited-card-v3'
REFERENCE_VISITED_CARD_VERSION = 'extra-action-last-action-chain-visited-card-v1'


class ExtraActionTurnChain:
    """Reuse bounded turn comparison without committing its later actions.

    The existing suggestion must first provably spend the last action (or be
    END). A positive EXTRA_ACTION descriptor is then only a candidate for a
    strict proof that an alternative retains the turn and its available action.
    Every legal action participates in the ordinary complete comparison.
    """

    version = VERSION

    def __init__(self, kernel, *, visited_card_extra_action=False, unused_drink_enabled=True):
        if type(visited_card_extra_action) is not bool:
            raise ValueError('visited_card_extra_action_requires_bool')
        if type(unused_drink_enabled) is not bool:
            raise ValueError('unused_drink_enabled_requires_bool')
        self.visited_card_extra_action = visited_card_extra_action
        self.unused_drink_enabled = visited_card_extra_action and unused_drink_enabled
        if visited_card_extra_action:
            self.version = VISITED_CARD_VERSION if self.unused_drink_enabled else REFERENCE_VISITED_CARD_VERSION
        self.kernel = kernel
        self.proof = PairOrderAdvice(kernel)
        self.turn_chain = TurnChainContinuation(kernel)
        self.drink_turn_chain = TurnChainContinuation(kernel, finite_consumables=True)
        self.reset()

    def reset(self):
        self.calls = self.scope_hits = self.scope_declines = 0
        self.chain_accepted = self.chain_declined = 0
        self.scope_decline_reasons = Counter()
        self.old_action_proof = dict(started=0, completed=0, explicit_end=0)
        self.proof.reset_window()
        self.turn_chain.start_window()
        self.drink_turn_chain.start_window()
        self._unused_drink_cache = {}
        self._unused_drink_counts = Counter()

    def start_window(self):
        self.reset()

    def summary(self):
        result = dict(version=self.version, calls=self.calls, scope_hits=self.scope_hits,
                    scope_declines=self.scope_declines, chain_accepted=self.chain_accepted,
                    chain_declined=self.chain_declined, turn_chain=self.turn_chain.summary(),
                    drink_turn_chain=self.drink_turn_chain.summary(),
                    strict_proof=self.proof.summary(),
                    old_action_proof=dict(self.old_action_proof),
                    scope_decline_reasons=dict(self.scope_decline_reasons))
        if self.unused_drink_enabled:
            from .hif_unused_action_turn import VERSION as UNUSED_DRINK_VERSION

            result['unused_action_drink'] = dict(version=UNUSED_DRINK_VERSION,
                **{name: self._unused_drink_counts[name] for name in
                   ('calls', 'cache_hits', 'proven', 'declined', 'paired_nodes', 'paired_edges')},
                cache_entries=len(self._unused_drink_cache))
        return result

    def unused_drink_proof(self, request, action, deadline):
        """Only the existing explicit high-score variant admits this proof.

        A complete deterministic graph may exclude a continuation suggestion
        or a final mean-selection candidate. Legal roots, training edges,
        samples and future draws remain intact.
        """
        if not self.unused_drink_enabled or action.kind is not ActionKind.USE_DRINK:
            return None
        from .hif_unused_action_turn import prove_unused_action_drink_in_turn

        with validation_session(lambda: self.turn_chain._check_deadline(deadline), reuse_parent=True):
            check_search_budget()
            if any(dict(getattr(self.kernel, name)) != value
                   for name, value in self.proof._registries.items()):
                raise ValueError('extra_action_chain_window_definitions_changed')
            key = (request.plan_type, request.decision_kind, request.ruleset_id,
                   request.capabilities, request.state, request.card_slots, request.drink_slots, action)
            hit = key in self._unused_drink_cache
            result = (self._unused_drink_cache[key] if hit
                      else prove_unused_action_drink_in_turn(self.kernel, request, action))
            check_search_budget()
            if not hit:
                # Bound cache storage independently of proof coverage. A full
                # cache changes only recomputation, never the selected action.
                if len(self._unused_drink_cache) < 4096:
                    self._unused_drink_cache[key] = result
                if result is not None:
                    for name in ('paired_nodes', 'paired_edges'):
                        self._unused_drink_counts[name] += result['transition_checks'][name]
            self._unused_drink_counts['calls'] += 1
            self._unused_drink_counts['cache_hits'] += int(hit)
            self._unused_drink_counts['proven' if result is not None else 'declined'] += 1
            return result

    def _old_action_gate(self, request, action):
        state = request.state
        if action.kind is ActionKind.END_TURN:
            self.old_action_proof['explicit_end'] += 1
            return None
        self.old_action_proof['started'] += 1
        try:
            branches = self.proof._proof.transition(state, replace(action, source_slot=None),
                                                     plan_type=request.plan_type)
        except SearchIncomplete as error:
            if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                raise
            return 'old_action_random_or_draw'
        check_search_budget()
        _valid_branches(branches)
        if len(branches) != 1 or branches[0].probability != 1.:
            return 'old_action_not_deterministic'
        if any(event.event_type.startswith(_UNOBSERVED_EVENTS) or 'RANDOM' in event.event_type
               for event in branches[0].events):
            return 'old_action_unobserved_event'
        after = self.kernel._canonical_probability_state(branches[0].state)
        self.old_action_proof['completed'] += 1
        # PairOrderAdvice._step intentionally rejects AP0. Inspect the raw,
        # strictly deterministic result here rather than treating its decline
        # as proof that HifEnvironment will auto-end the turn.
        if (after.terminal or after.turn != state.turn or after.turns_remaining != state.turns_remaining
                or after.pending_choice or after.resolution_stack or after.pending_after_card_ids
                or after.drinks != state.drinks):
            return 'old_action_unresolved_or_changed_boundary'
        if after.actions_remaining >= state.actions_remaining:
            return 'old_action_proven_keeps_ap'
        if after.actions_remaining == 0:
            return None
        return 'old_action_ap_not_proven'

    def _scope(self, request, actions, proposed_key):
        from .hif_mcts import action_key

        if proposed_key is None:
            return 'old_suggestion_unknown'
        state = request.state
        if (request.plan_type is not PlanType.SENSE or not self.proof._clean(state)
                or state.actions_remaining != 1 or not actions
                or any(action.kind not in (ActionKind.PLAY_CARD, ActionKind.END_TURN,
                                          ActionKind.USE_DRINK)
                       for action in actions)):
            return 'outside_scope'
        if request.ruleset_id != self.kernel.ruleset_id or not request.capabilities.all_ready:
            raise ValueError('extra_action_chain_capability_or_ruleset_not_ready')
        legal = self.kernel.legal_actions(state, plan_type=request.plan_type,
                                          decision_kind=request.decision_kind)
        bound = {HifActionChainPlanner._attach_source_slot(action, request) for action in legal}
        if set(actions) != bound:
            raise ValueError('extra_action_chain_incomplete_root_actions')
        proposed = next(action for action in actions if action_key(action) == proposed_key)
        reason = self._old_action_gate(request, proposed)
        if reason is not None:
            return reason
        candidates = [action for action in actions
                      if self.proof._has_effect(state, action, EffectKind.EXTRA_ACTION)]
        if not candidates:
            return 'no_positive_direct_extra_action'
        for action in candidates:
            check_search_budget()
            after, _ = self.proof._step(request, state, action)
            if after is not None and after.actions_remaining >= state.actions_remaining:
                return None
        return 'no_strict_same_turn_extra_action'

    def choose(self, request, actions, deadline, *, proposed_key=None):
        from .hif_mcts import action_key

        with validation_session(lambda: self.turn_chain._check_deadline(deadline), reuse_parent=True):
            check_search_budget()
            request.validate()
            if any(dict(getattr(self.kernel, name)) != value
                   for name, value in self.proof._registries.items()):
                raise ValueError('extra_action_chain_window_definitions_changed')
            keys = tuple(action_key(action) for action in actions)
            if len(set(keys)) != len(keys):
                raise ValueError('extra_action_chain_actions_not_unique')
            if proposed_key is not None and keys.count(proposed_key) != 1:
                raise ValueError('extra_action_chain_old_suggestion_not_uniquely_legal')
            reason = self._scope(request, actions, proposed_key)
            if reason is not None:
                check_search_budget()
                self.calls += 1
                self.scope_declines += 1
                self.scope_decline_reasons[reason] += 1
                return None
            chain = self.drink_turn_chain if request.state.drinks else self.turn_chain
            selected = chain.choose(request, actions, deadline)
            if selected is not None and keys.count(selected) != 1:
                raise ValueError('extra_action_chain_action_not_uniquely_legal')
            check_search_budget()
            self.calls += 1
            self.scope_hits += 1
            self.chain_accepted += int(selected is not None)
            self.chain_declined += int(selected is None)
            return selected
