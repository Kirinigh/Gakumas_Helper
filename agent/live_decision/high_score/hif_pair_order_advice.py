"""A strictly proved local preparation order, never a dominance or commitment."""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace

from .choice import _valid_branches
from .kernel import EffectKind
from .planner import SearchIncomplete
from .contracts import PlanType, ActionKind, DecisionKind
from .hif_choice_roots import _UNOBSERVED_EVENTS
from .validation_cache import validation_session, check_search_budget

VERSION = 'deterministic-pair-order-advice-v1'


class PairOrderAdvice:
    """Suggest the first legal amplifier whose two-step swap is strictly proved."""

    version = VERSION

    def __init__(self, kernel):
        self.kernel = kernel
        self.reset_window()

    def reset_window(self):
        # Runtime import: the strict proof module also imports the search.
        from .hif_public_turn_reanalysis import _StrictDeterministicKernel

        self._cache = {}
        self.counts = Counter()
        self.last_evidence = None
        self._registries = {name: dict(getattr(self.kernel, name))
                            for name in ('cards', 'drinks', 'items', 'memory_abilities')}
        self._proof = _StrictDeterministicKernel(**self._registries)

    def summary(self):
        return dict(version=self.version, **dict(self.counts), cache_entries=len(self._cache))

    @staticmethod
    def _clean(state):
        return (not state.terminal and state.turns_remaining > 2
                and not state.pending_choice and not state.resolution_stack
                and not state.pending_after_card_ids and state.actions_remaining > 0)

    def _has_effect(self, state, action, kind):
        if action.kind is not ActionKind.PLAY_CARD:
            return False
        refs = [ref for ref in state.deck.hand if ref.instance_id == action.card_instance_id]
        if len(refs) != 1:
            raise ValueError('pair_order_card_identity_not_unique')
        definition = self.kernel.card_definition(state, refs[0])
        return any(effect.kind is kind and effect.amount > 0 for effect in definition.effects)

    def _step(self, request, state, action):
        from .hif_mcts import action_key

        # Both steps must still be legal in the complete current business state.
        legal = self.kernel.legal_actions(state, plan_type=request.plan_type,
                                         decision_kind=DecisionKind.LIVE)
        matches = [candidate for candidate in legal if action_key(candidate) == action_key(action)]
        if not matches:
            return None, 'action_not_legal_after_prefix'
        if len(matches) != 1:
            raise ValueError('pair_order_action_not_unique')
        self.counts['transitions_started'] += 1
        try:
            branches = self._proof.transition(state, replace(matches[0], source_slot=None),
                                              plan_type=request.plan_type)
        except SearchIncomplete as error:
            if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                raise
            return None, 'random_or_draw_boundary'
        check_search_budget()
        _valid_branches(branches)
        if len(branches) != 1 or branches[0].probability != 1.:
            return None, 'non_deterministic_transition'
        if any(event.event_type.startswith(_UNOBSERVED_EVENTS) or 'RANDOM' in event.event_type
               for event in branches[0].events):
            return None, 'random_or_draw_event'
        after = self.kernel._canonical_probability_state(branches[0].state)
        self.counts['transitions_completed'] += 1
        if (not self._clean(after) or after.turn != request.state.turn
                or after.turns_remaining != request.state.turns_remaining):
            return None, 'outside_clean_same_turn'
        if after.drinks != request.state.drinks:
            return None, 'drink_inventory_changed'
        return after, 'complete'

    def _pair(self, request, gain, amplifier):
        finals = []
        for route in ((gain, amplifier), (amplifier, gain)):
            current = request.state
            for action in route:
                check_search_budget()
                current, reason = self._step(request, current, action)
                if current is None:
                    return reason, 0
            finals.append(current)
        first, second = finals
        delta = second.concentration-first.concentration
        if delta <= 0:
            return 'no_concentration_improvement', delta
        if replace(second, concentration=first.concentration) != first:
            return 'other_business_fields_differ', delta
        return 'only_concentration_improves', delta

    def improve(self, request, actions, proposed_key, deadline):
        from .hif_mcts import action_key

        if type(deadline) not in (float, int) or not math.isfinite(deadline):
            raise ValueError('pair_order_finite_deadline_required')

        def guard():
            if time.perf_counter() >= deadline:
                raise TimeoutError('pair_order_deadline')

        with validation_session(guard, reuse_parent=True):
            check_search_budget()
            request.validate()
            if any(dict(getattr(self.kernel, name)) != value
                   for name, value in self._registries.items()):
                raise ValueError('pair_order_window_definitions_changed')
            keys = tuple(action_key(action) for action in actions)
            if len(set(keys)) != len(keys) or keys.count(proposed_key) != 1:
                raise ValueError('pair_order_proposed_action_not_uniquely_legal')
            gain = actions[keys.index(proposed_key)]
            self.counts['calls'] += 1
            self.last_evidence = dict(proposed_key=proposed_key, selected_key=proposed_key,
                                      reason='outside_domain')
            if (request.plan_type is not PlanType.SENSE or request.ruleset_id != self.kernel.ruleset_id
                    or not self._clean(request.state) or len(request.state.deck.hand) > 5
                    or not self._has_effect(request.state, gain, EffectKind.CONCENTRATION)):
                self.counts['scope_declines'] += 1
                return proposed_key
            # No sorting by hidden information, future return or card identity.
            for amplifier in actions:
                check_search_budget()
                if amplifier == gain or not self._has_effect(
                        request.state, amplifier, EffectKind.CONCENTRATION_ADDITION_BUFF):
                    continue
                key = (request.ruleset_id, request.plan_type, request.state,
                       tuple(request.drink_slots), proposed_key, action_key(amplifier))
                if key in self._cache:
                    self.counts['cache_hits'] += 1
                    reason, delta = self._cache[key]
                else:
                    self.counts['pair_proofs'] += 1
                    reason, delta = self._pair(request, gain, amplifier)
                    check_search_budget()
                    self._cache[key] = reason, delta
                self.last_evidence = dict(proposed_key=proposed_key,
                    selected_key=action_key(amplifier) if reason == 'only_concentration_improves' else proposed_key,
                    alternative_key=action_key(amplifier), reason=reason, concentration_delta=delta)
                if reason == 'only_concentration_improves':
                    self.counts['improved_calls'] += 1
                    return action_key(amplifier)
                self.counts['pair_declines:'+reason] += 1
            self.counts['unchanged_calls'] += 1
            return proposed_key
