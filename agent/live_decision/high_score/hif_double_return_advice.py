"""Opt-in double ordering with realised score and registered START capacity.

These are exploration descriptors, not terminal returns or pruning proofs.
Weights must be supplied by the caller; this module has no calibrated default.
"""
from __future__ import annotations

import math
from dataclasses import replace, dataclass

import numpy as np

from .hif import _HifTurnBoundary, is_hif_turn_boundary
from .kernel import EffectKind, KernelError
from .contracts import PlanType, ActionKind, GameAction, DecisionKind, SchedulePhase
from .hif_forecast import _decayed_modifiers
from .validation_cache import check_search_budget
from .hif_double_advice import FEATURE_NAMES as DOUBLE_FEATURE_NAMES
from .hif_double_advice import HifDoubleAdvice

VERSION = 'task275-double-return-descriptors-v2'
FEATURE_NAMES = (*DOUBLE_FEATURE_NAMES, 'current_turn_realized_score',
                 'remaining_start_score_capacity')
_SCORE_COLUMNS = (0, 2, 3, 4, 5, 6, 7)
_START_PHASES = frozenset({SchedulePhase.BEFORE_START_OF_TURN,
    SchedulePhase.START_OF_TURN, SchedulePhase.AFTER_START_OF_TURN,
    SchedulePhase.TURN, SchedulePhase.EVERY_TURN})
_CACHE_LIMIT = 4096


class UnsupportedDoubleReturn(ValueError):
    """A descriptor would require an unresolved choice or unsupported phase."""


@dataclass(frozen=True)
class DoubleReturnDescription:
    applicable: bool
    reason: str | None = None
    raw_features: tuple[tuple[float, ...], ...] = ()
    features: tuple[tuple[float, ...], ...] = ()


class HifDoubleReturnAdvice:
    """Extend only the existing pure PLAY/END double-effect domain.

    The current score is the actual transition's expected score increment,
    including payment and already established triggers. Remaining capacity
    counts only registered unconditional pure SCORE turn-start schedules,
    with public multipliers and decaying buffs. Other future actions/resource
    gains are not forecast. Every candidate, including END, uses its actual
    successor. Unsupported windows retain the supplied old advice unchanged.
    """

    model_version = VERSION

    def __init__(self, kernel, old_advice, *, weights, max_turns_remaining=None):
        if max_turns_remaining is not None and (
                type(max_turns_remaining) is not int or max_turns_remaining < 1):
            raise ValueError('double return horizon requires a positive integer or None')
        weights = tuple(float(value) for value in weights)
        if len(weights) != len(FEATURE_NAMES) or not all(map(math.isfinite, weights)):
            raise ValueError('double return advice requires eight finite weights')
        self.kernel, self.old_advice, self.weights = kernel, old_advice, weights
        self.max_turns_remaining = max_turns_remaining
        self.double = (old_advice if isinstance(old_advice, HifDoubleAdvice)
                       else HifDoubleAdvice(kernel, old_advice))
        self.calls = self.active_calls = self.bypassed_calls = self.cache_hits = 0
        self.last_description = None
        self.reset_window()

    def reset_window(self):
        self._cache = {}
        self._definitions = None

    def future_score(self, state):
        method = getattr(self.old_advice, 'future_score', None)
        return method(state) if method is not None else None

    def _definition_key(self):
        return (VERSION, self.double.model_version, type(self.kernel), self.kernel.ruleset_id,
                *(tuple(sorted(registry.items())) for registry in
                  (self.kernel.cards, self.kernel.drinks, self.kernel.items,
                   self.kernel.memory_abilities)))

    @staticmethod
    def _complete(branches):
        if (not branches or any(not math.isfinite(branch.probability) or branch.probability <= 0
                                for branch in branches)
                or not math.isclose(math.fsum(branch.probability for branch in branches),
                                    1., rel_tol=1e-9, abs_tol=1e-9)):
            raise UnsupportedDoubleReturn('incomplete_probability_mass')
        for branch in branches:
            state = branch.state
            if state.pending_choice or state.pending_after_card_ids:
                raise UnsupportedDoubleReturn('unresolved_successor')
            if state.resolution_stack and not is_hif_turn_boundary(state):
                raise UnsupportedDoubleReturn('unsupported_successor_phase')
        return branches

    def _successors(self, state, action):
        boundary = _HifTurnBoundary(self.kernel, state.turn + 1)
        try:
            branches = self._complete(boundary.transition(state,
                replace(action, source_slot=None), plan_type=PlanType.SENSE))
            result = []
            for branch in branches:
                check_search_budget()
                after = branch.state
                if (action.kind is ActionKind.PLAY_CARD and not after.terminal
                        and after.actions_remaining == 0 and not after.pending_choice
                        and not after.resolution_stack and not after.pending_after_card_ids):
                    legal = self.kernel.legal_actions(after, plan_type=PlanType.SENSE,
                                                     decision_kind=DecisionKind.LIVE)
                    if len(legal) == 1 and legal[0].kind is ActionKind.END_TURN:
                        advanced = self._complete(boundary.transition(after,
                            GameAction(ActionKind.END_TURN), plan_type=PlanType.SENSE))
                        result.extend(replace(child, probability=branch.probability * child.probability)
                                      for child in advanced)
                        continue
                result.append(branch)
            return self._complete(tuple(result))
        except KernelError as error:
            raise UnsupportedDoubleReturn('transition:' + str(error)) from error

    @staticmethod
    def _project(state, offset):
        multipliers = (state.score_multiplier, *state.future_score_multipliers)
        if len(multipliers) != state.turns_remaining:
            raise UnsupportedDoubleReturn('incomplete_public_multipliers')
        return replace(state, turn=state.turn + offset,
            turns_remaining=state.turns_remaining - offset,
            score_multiplier=multipliers[offset], future_score_multipliers=multipliers[offset + 1:],
            turn_attributes=state.turn_attributes[offset:],
            passion=state.passion if offset == 0 else 0,
            good_condition_turns=max(0, state.good_condition_turns - max(0, offset - int(state.good_condition_fresh))),
            excellent_condition_turns=max(0, state.excellent_condition_turns - max(0, offset - int(state.excellent_condition_fresh))),
            score_buffs=_decayed_modifiers(state.score_buffs, offset),
            concentration_effect_buffs=_decayed_modifiers(state.concentration_effect_buffs, offset))

    def _start_capacity(self, state):
        if state.terminal:
            return 0.
        # A boundary already advanced its turn/multiplier and decremented TTL.
        # Its due refs fire at offset zero, before the next decision or draw.
        refs = [(ref.turns_until_fire, ref, False) for ref in state.scheduled_effects
                if ref.trigger_phase in _START_PHASES]
        if is_hif_turn_boundary(state):
            frame = state.resolution_stack[0]
            refs.extend((0, ref, True) for ref in (*frame.due_before_start,
                *frame.due_start, *frame.due_after_start, *frame.due_turn, *frame.due_every_turn))
        result = 0.
        for first, ref, due in refs:
            check_search_budget()
            if ref.remaining_uses == 0 or (ref.remaining_turns == 0 and not due):
                continue
            effect = self.kernel._scheduled_effect_spec(ref)
            # Non-score schedules contribute no score capacity. Mixed score
            # payloads/conditional triggers cannot be valued as pure SCORE.
            if not any(child.kind is EffectKind.SCORE for child in effect.scheduled_effects):
                continue
            if (effect.condition is not None or effect.trigger_card_type is not None
                    or effect.schedule_counter_condition is not None
                    or any(child.kind is not EffectKind.SCORE or child.condition is not None
                           for child in effect.scheduled_effects)):
                raise UnsupportedDoubleReturn('non_pure_start_score_schedule')
            uses = ref.remaining_uses
            for offset in range(first, state.turns_remaining):
                check_search_budget()
                if uses == 0:
                    break
                # A positive TTL is decremented before the due event, which
                # still fires at zero in the kernel; it then cannot recur.
                if (ref.remaining_turns is not None
                        and offset > ref.remaining_turns + int(ref.duration_fresh)):
                    break
                projected = self._project(state, offset)
                source = self.kernel._card_by_instance(projected, ref.source_id)
                repeat_bonus = source.score_repeat_bonus if source is not None else 0
                result += sum(self.kernel.score_effect_gain(projected, child, ref.source_id)
                              * (child.repeats + repeat_bonus) for child in effect.scheduled_effects)
                if uses is not None:
                    uses -= 1
        return result

    def describe(self, state, actions):
        """Return immutable raw and jointly normalised eight-column features."""
        # Check every simulated/public node, not only the real decision root.
        # None preserves the original unrestricted candidate for replay.
        if self.max_turns_remaining is not None and state.turns_remaining > self.max_turns_remaining:
            return DoubleReturnDescription(False, 'outside_return_horizon')
        check_search_budget()
        actions = tuple(actions)
        definitions = self._definition_key()
        if definitions != self._definitions:
            self._cache.clear()
            self._definitions = definitions
        key = (state, actions)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        raw = self.double.raw_features(state, actions)
        if raw is None:
            description = DoubleReturnDescription(False, 'outside_double_domain')
        else:
            try:
                rows = []
                for row, action in zip(raw, actions):
                    check_search_budget()
                    branches = self._successors(state, action)
                    realised = math.fsum(branch.probability * (branch.state.score - state.score)
                                         for branch in branches)
                    capacity = math.fsum(branch.probability * self._start_capacity(branch.state)
                                         for branch in branches)
                    rows.append((*row, realised / 100., capacity / 100.))
                features = np.asarray(rows, dtype=float)
                if not np.isfinite(features).all():
                    raise UnsupportedDoubleReturn('nonfinite_features')
                raw_rows = tuple(tuple(float(value) for value in row) for row in features)
                scale = max((abs(features[:, index]).max() for index in _SCORE_COLUMNS), default=0.)
                features[:, _SCORE_COLUMNS] /= scale if scale > 0 else 1.
                description = DoubleReturnDescription(True, raw_features=raw_rows,
                    features=tuple(tuple(float(value) for value in row) for row in features))
            except UnsupportedDoubleReturn as error:
                description = DoubleReturnDescription(False, str(error))
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = description
        return description

    def matrix(self, state, actions):
        """Return normalised features, or None for an unsupported window."""
        description = self.describe(state, actions)
        return np.asarray(description.features, dtype=float) if description.applicable else None

    def action_priors(self, state, actions):
        self.calls += 1
        old = tuple(self.old_advice.action_priors(state, actions))
        description = self.describe(state, actions)
        self.last_description = description
        if not description.applicable:
            self.bypassed_calls += 1
            return old
        old_order = np.argsort(np.argsort(old, kind='stable'), kind='stable') / max(1, len(old) - 1)
        scores = np.asarray(description.features) @ np.asarray(self.weights) + old_order * 1e-8
        if not np.isfinite(scores).all():
            raise ValueError('double return advice returned nonfinite priors')
        self.active_calls += 1
        return tuple(float(score) for score in scores)
