"""Opt-in direct double marginal within the existing six-feature ordering.

Only the raw direct-extra column changes. Shared normalization and learned
weights remain unchanged; this descriptor is not a final-score estimate.
"""
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import replace, dataclass

import numpy as np

from .kernel import EffectKind, KernelError
from .planner import SearchIncomplete
from .contracts import PlanType, ActionKind
from .validation_cache import check_search_budget
from .hif_double_advice import HifDoubleAdvice

VERSION = 'hif-direct-double-marginal-v1'
CACHE_LIMIT = 256


@dataclass(frozen=True)
class DirectMarginalDescription:
    applicable: bool
    reason: str
    direct_delta: float | None = None
    direct_scores: tuple[tuple[float, ...], ...] = ()  # D0, D1, in effect order.
    other_score_events: tuple[tuple[str, float, str | None], ...] = ()
    total_gains: tuple[float, ...] = ()  # D0, D1; diagnostic only.


class HifDirectDoubleAdvice(HifDoubleAdvice):
    """Replace only the existing-double direct column in a proven domain.

    Resource conversion still describes other cards/future opportunities. The
    immediate effect of this card's resources on its own score belongs here.
    This does not recalibrate the other approximate descriptors.
    """

    def __init__(self, base_advice):
        if type(base_advice) is not HifDoubleAdvice:
            raise TypeError('direct_double_requires_plain_HifDoubleAdvice')
        super().__init__(base_advice.kernel, base_advice.old_advice,
            guard_unpriced_schedules=base_advice.guard_unpriced_schedules,
            guard_unpriced_resources=base_advice.guard_unpriced_resources)
        self.source_base_model_version = base_advice.model_version
        self.model_version = self.source_base_model_version + '+' + VERSION
        self.direct_stats = Counter()
        self.reset_window()

    def reset_window(self):
        """Drop window computations; diagnostic counters remain cumulative."""
        self._direct_cache = OrderedDict()
        self._direct_registry = None
        self._direct_kernel_scope = None
        self._direct_proof = None

    def _sync_registry(self):
        registries = (self.kernel.cards, self.kernel.drinks, self.kernel.items,
            self.kernel.memory_abilities)
        scope = (type(self.kernel), self.kernel.ruleset_id, self.kernel.kernel_version)
        # Definitions are immutable. Compare snapshots without rebuilding and
        # sorting the full registry for each card in the unchanged window.
        if (scope != self._direct_kernel_scope or self._direct_registry is None
                or any(current != saved for current, saved in zip(registries, self._direct_registry))):
            self._direct_cache.clear()
            self._direct_proof = None
            self._direct_registry = tuple(dict(registry) for registry in registries)
            self._direct_kernel_scope = scope

    def describe_direct(self, state, action):
        """Prove one legal card's D1-D0 direct score, or decline its domain.

        Validation and real rule errors propagate. Only the proof kernel's
        explicit unsupported-randomness signals become ordinary fallbacks.
        """
        check_search_budget()
        state.validate()
        self.kernel._validate_registered_state(state, plan_type=PlanType.SENSE)
        action.validate()
        self._sync_registry()
        action = replace(action, source_slot=None)
        key = (state, action)
        if key in self._direct_cache:
            self.direct_stats['cache_hits'] += 1
            self._direct_cache.move_to_end(key)
            return self._direct_cache[key]
        try:
            result = self._describe_uncached(state, action)
        finally:
            # The outer bounded cache retains only complete D0/D1 proofs.
            # Do not accumulate single transitions or incomplete pairs in the
            # reused proof kernel's separate transition cache.
            if self._direct_proof is not None:
                self._direct_proof._transitions.clear()
        check_search_budget()
        if result.applicable:
            if len(self._direct_cache) >= CACHE_LIMIT:
                self._direct_cache.popitem(last=False)
            self._direct_cache[key] = result
            self.direct_stats['proved_pairs'] += 1
        else:
            self.direct_stats['declined:' + result.reason] += 1
        return result

    def _describe_uncached(self, state, action):
        def decline(reason):
            return DirectMarginalDescription(False, reason)

        if action.kind is not ActionKind.PLAY_CARD or state.double_card_effects_remaining != 1:
            return decline('not_one_existing_double_play')
        if (state.terminal or not state.persistent_effects_initialized or state.pending_choice
                or state.pending_after_card_ids or state.resolution_stack):
            return decline('outside_clean_initialized_card_state')
        ref = next((card for card in state.deck.hand
            if card.instance_id == action.card_instance_id), None)
        if ref is None:
            raise KernelError('direct_double_card_not_in_current_hand')
        definition = self.kernel.card_definition(state, ref)
        if self.kernel._card_effect_passes(state, definition) != 2:
            return decline('card_does_not_receive_exactly_two_passes')
        if any(scheduled.source_id == ref.instance_id for scheduled in state.scheduled_effects):
            return decline('own_existing_schedule')
        if definition.persistent_effects:
            return decline('own_persistent_effects')
        if any(effect.kind is EffectKind.SCHEDULE for effect in definition.effects):
            return decline('creates_schedule')
        if not any(effect.kind is EffectKind.SCORE for effect in definition.effects):
            return decline('no_direct_score')

        # Local import avoids coupling the base advice import path to MCTS.
        from .hif_public_turn_reanalysis import _StrictDeterministicKernel

        if self._direct_proof is None:
            self._direct_proof = _StrictDeterministicKernel(self.kernel.cards,
                self.kernel.drinks, self.kernel.items, memory_abilities=self.kernel.memory_abilities)
        proof = self._direct_proof
        if proof.ruleset_id != self.kernel.ruleset_id:
            return decline('unsupported_ruleset')
        direct, other, gains = [], [], []
        for double in (0, 1):
            current = replace(state, double_card_effects_remaining=double)
            current.validate()
            proof._validate_registered_state(current, plan_type=PlanType.SENSE)
            try:
                branches = proof.transition(current, action, plan_type=PlanType.SENSE)
            except SearchIncomplete as error:
                if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                    raise
                return decline('draw_or_intermediate_randomness')
            self.direct_stats['completed_single_card_transitions'] += 1
            if len(branches) != 1 or branches[0].probability != 1.:
                return decline('not_deterministic')
            branch = branches[0]
            after = branch.state
            after.validate()
            proof._validate_registered_state(after, plan_type=PlanType.SENSE)
            if (after.turn != state.turn or after.turns_remaining != state.turns_remaining
                    or after.terminal or after.pending_choice or after.pending_after_card_ids
                    or after.resolution_stack):
                return decline('unresolved_or_turn_boundary')
            if any('DRAW' in event.event_type or 'RANDOM' in event.event_type
                    for event in branch.events):
                return decline('draw_or_random_event')
            consumed = tuple(event for event in branch.events
                if event.event_type == 'CONSUME_DOUBLE_CARD_EFFECT')
            if (len(consumed) != double or any(event.source_id != ref.instance_id
                    or event.amount != 1 for event in consumed)
                    or after.double_card_effects_remaining != 0):
                return decline('double_consumption_not_isolated')
            # A source id alone cannot distinguish an own-instance schedule.
            if any(event.source_id == ref.instance_id and event.event_type
                    in ('TRIGGER_SCHEDULED', 'SCHEDULE_EFFECT') for event in branch.events):
                return decline('own_schedule_event')
            scores = tuple(event for event in branch.events if event.event_type == 'SCORE')
            if any(type(event.amount) not in (int, float) or not math.isfinite(event.amount)
                    for event in scores):
                raise ValueError('direct_double_invalid_score_event')
            gain = after.score - current.score
            if math.fsum(event.amount for event in scores) != gain:
                raise ValueError('direct_double_score_events_do_not_reconcile')
            gains.append(gain)
            direct.append(tuple(event.amount for event in scores if event.source_id == ref.instance_id))
            other.append(tuple((event.source_id, event.amount, event.detail)
                for event in scores if event.source_id != ref.instance_id))
        if other[0] != other[1]:
            return decline('external_score_changes_with_double')
        delta = math.fsum(direct[1]) - math.fsum(direct[0])
        if delta != gains[1] - gains[0]:
            raise ValueError('direct_double_marginal_does_not_reconcile')
        return DirectMarginalDescription(True, 'proved_direct_marginal', delta,
            tuple(direct), other[0], tuple(gains))

    def _raw_features(self, state, actions):
        actions = tuple(actions)
        original = super()._raw_features(state, actions)
        if original is None:
            return None
        result = original.copy()
        for index, action in enumerate(actions):
            if action.kind is not ActionKind.PLAY_CARD:
                continue
            description = self.describe_direct(state, action)
            if description.applicable:
                result[index, 2] = description.direct_delta / 100.
                self.direct_stats['corrected_rows'] += int(result[index, 2] != original[index, 2])
        if not np.isfinite(result).all():
            raise ValueError('direct_double_nonfinite_features')
        return result
