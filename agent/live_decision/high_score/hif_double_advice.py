"""Opt-in double-effect action ordering; never a value or an input authority."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .kernel import EffectKind, EffectSpec
from .contracts import ActionKind, SchedulePhase, EffectSourceKind
from .hif_forecast import expected_best, _decayed_modifiers
from .hif_learning import HifFeatureEncoderV13

VERSION = 'task275-double-semantic-v4-20261001'
SCOPE_VERSION = 'task275-double-semantic-v5-schedule-scope-20261007'
RESOURCE_SCOPE_VERSION = 'task275-double-semantic-v6-resource-scope-20261010'
COMBINED_SCOPE_VERSION = 'task275-double-semantic-v7-schedule-resource-scope-20261010'
FEATURE_NAMES = ('new_double_target_option', 'new_double_unredeemable',
                 'existing_double_direct_extra', 'existing_double_persistent_extra',
                 'existing_double_resource_conversion', 'double_consumption_future_option')
# Exact complete terminal teachers: 13 calibration states, 10 separate
# mechanism checks. No human trajectory or old-policy return labels.
# The named artifacts are provenance only: runtime never opens local files.
CALIBRATION_SOURCE = ('live-validation-20260930-strong-r1/'
                      'semantic-double-prior-v3-result.json')
WEIGHTS = (0.7370892018779344, -0.9523809523809523, 0.9081654699346274,
           1.23996214533557, 0.9663501078775434, -0.9916514063576933)
_RESOURCE_KINDS = frozenset({EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION,
    EffectKind.EXCELLENT_CONDITION, EffectKind.GENKI, EffectKind.STAMINA,
    EffectKind.COST_REDUCTION, EffectKind.SCORE_BUFF, EffectKind.CONCENTRATION_EFFECT_BUFF,
    EffectKind.CONCENTRATION_ADDITION_BUFF, EffectKind.EXTRA_ACTION})


class _UnpricedResourceOpportunity(ValueError):
    """The six descriptors omit a positive future use of prepared resources."""


def _contains_score(effect):
    return effect.kind is EffectKind.SCORE or any(
        _contains_score(child) for child in effect.scheduled_effects)


def _unpriced_score_schedule(effect):
    if effect.kind is not EffectKind.SCHEDULE or not _contains_score(effect):
        return False
    if (effect.condition is not None or effect.schedule_phase is not SchedulePhase.CARD_USED
            or effect.trigger_card_type != 'active' or effect.delay_turns != 0
            or effect.schedule_counter_condition is not None or effect.repeats != 1):
        return True
    # The six descriptors aggregate plain active-card SCORE amounts. A
    # filtered score, nested schedule or preceding resource effect is not
    # represented by that aggregate, even if another child is represented.
    return any(child.kind is not EffectKind.SCORE or child.scheduled_effects
        or child.condition is not None or child.value_source is not None
        or child.full_power_scale != 0 or child.score_enthusiasm_multiplier != 1
        or child.score_concentration_multiplier != 1 or child.score_good_condition_multiplier != 1
        for child in effect.scheduled_effects)


class HifDoubleAdvice:
    """Reuse loaded advice outside a pure PLAY_CARD/END_TURN double window.

    Six relative descriptors rank exploration and unknown-node continuations.
    Schedule capacity and future targets are approximations, not final-score
    predictions, bounds, or proof that spending a double is always better.
    """

    model_version = VERSION

    def __init__(self, kernel, old_advice, *, guard_unpriced_schedules=False,
                 guard_unpriced_resources=False):
        if type(guard_unpriced_schedules) is not bool:
            raise ValueError('guard_unpriced_schedules_requires_bool')
        if type(guard_unpriced_resources) is not bool:
            raise ValueError('guard_unpriced_resources_requires_bool')
        self.kernel, self.old_advice = kernel, old_advice
        self.guard_unpriced_schedules = guard_unpriced_schedules
        self.guard_unpriced_resources = guard_unpriced_resources
        if guard_unpriced_schedules and guard_unpriced_resources:
            self.model_version = COMBINED_SCOPE_VERSION
        elif guard_unpriced_resources:
            self.model_version = RESOURCE_SCOPE_VERSION
        elif guard_unpriced_schedules:
            self.model_version = SCOPE_VERSION
        self.calls = self.active_calls = self.bypassed_calls = 0
        self.scope_bypassed_calls = 0
        self.resource_scope_bypassed_calls = 0

    def future_score(self, state):
        # This slice does not replace the caller's value model.
        method = getattr(self.old_advice, 'future_score', None)
        return method(state) if method is not None else None

    def _has_unpriced_double_target(self, state, actions):
        if (not actions or state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                or any(action.kind not in {ActionKind.PLAY_CARD, ActionKind.END_TURN} for action in actions)):
            return False
        refs = {card.instance_id: card for card in state.deck.hand}
        entries = []
        for action in actions:
            if action.kind is ActionKind.PLAY_CARD:
                ref = refs[action.card_instance_id]
                definition = self.kernel.card_definition(state, ref)
                creates = any(effect.kind is EffectKind.DOUBLE_CARD_EFFECT
                    and effect.condition is None and effect.amount > 0 for effect in definition.effects)
                entries.append((ref, definition, creates, self.kernel._card_effect_passes(state, definition)))
        if not any(creates or passes > 1 for _, _, creates, passes in entries):
            return False
        anticipated = replace(state, double_card_effects_remaining=1)
        for ref, definition, _, passes in entries:
            potential = passes > 1 or (
                any(creates and other.instance_id != ref.instance_id for other, _, creates, _ in entries)
                and self.kernel._card_effect_passes(anticipated, definition) > 1)
            if potential and any(_unpriced_score_schedule(effect) for effect in definition.effects):
                return True
        return False

    def _direct_score(self, state, card, *, instance_repeats=True):
        definition = self.kernel.card_definition(state, card)
        if not self.kernel._non_full_power_cost_payable(state, card, definition):
            return 0.
        return sum(self.kernel.score_effect_gain(state, effect, card.instance_id)
                   * (effect.repeats + (card.score_repeat_bonus if instance_repeats else 0))
                   for effect in definition.effects if effect.kind is EffectKind.SCORE and effect.condition is None)

    @staticmethod
    def _windows(state):
        multipliers = (state.score_multiplier, *state.future_score_multipliers)[:state.turns_remaining]
        return tuple(replace(state, score_multiplier=multiplier,
            hand_support=state.hand_support if offset == 0 else (),
            good_condition_turns=max(0, state.good_condition_turns-max(0, offset-int(state.good_condition_fresh))),
            excellent_condition_turns=max(0, state.excellent_condition_turns-max(0, offset-int(state.excellent_condition_fresh))),
            score_buffs=_decayed_modifiers(state.score_buffs, offset),
            concentration_effect_buffs=_decayed_modifiers(state.concentration_effect_buffs, offset))
            for offset, multiplier in enumerate(multipliers))

    def _future_target(self, state, cache):
        if state in cache:
            return cache[state]
        anticipated = replace(state, double_card_effects_remaining=max(1, state.double_card_effects_remaining))
        pool = tuple(card for card in state.deck.all_cards if card not in state.deck.removed)
        value = max((expected_best([
            self._direct_score(projected, card)
            if self.kernel._card_effect_passes(anticipated, self.kernel.cards[card.card_id]) > 1 else 0.
            for card in pool], 3) for projected in self._windows(state)[1:]), default=0.)/100
        cache[state] = value
        return value

    def _resource_conversion(self, state, ref, definition, passes, cache, *, check_future_scope=False):
        effects = [(index, effect) for index, effect in enumerate(definition.effects)
                   if effect.kind in _RESOURCE_KINDS and effect.condition is None]
        if not any(effect.kind is not EffectKind.EXTRA_ACTION for _, effect in effects):
            return 0.
        paid, _ = self.kernel._pay_declared_non_full_power_cost(state, ref, definition)
        projected = replace(paid, actions_remaining=max(0, paid.actions_remaining-1))
        for _ in range(passes):
            for index, effect in effects:
                projected, _ = self.kernel._apply_non_draw_effect(projected, effect,
                    source_id=ref.instance_id, condition_state=state, continuation_index=None,
                    remaining_effect_passes=0, source_card_id=definition.card_id,
                    definition_effect_index=index, trigger_context=None, source_kind=EffectSourceKind.CARD)
        remaining = [card for card in state.deck.hand if card != ref]
        if projected.actions_remaining:
            before = max((self._direct_score(paid, card) for card in remaining), default=0.)
            after = max((self._direct_score(projected, card) for card in remaining), default=0.)
            current = max(0., after-before)/100
            if (self.guard_unpriced_resources and check_future_scope and current == 0.
                    and state.turns_remaining > 1
                    and self._future_target(projected, cache) > self._future_target(paid, cache)):
                # A remaining action does not erase a later scoring opportunity.
                # Do not feed this uncalibrated future value into the old weights.
                raise _UnpricedResourceOpportunity('double_unpriced_future_resource_opportunity')
            return current
        if state.turns_remaining > 1:
            return max(0., self._future_target(projected, cache)-self._future_target(paid, cache))
        return 0.

    def _persistent_capacity(self, state, ref, definition, windows):
        schedules = [(effect, HifFeatureEncoderV13._active_score_amount(effect))
                     for effect in definition.effects if effect.condition is None]
        schedules = [(effect, amount) for effect, amount in schedules if amount]
        if not schedules:
            return 0.
        paid, _ = self.kernel._pay_declared_non_full_power_cost(state, ref, definition)
        extra = sum(effect.amount for effect in definition.effects
                    if effect.kind is EffectKind.EXTRA_ACTION and effect.condition is None)
        remaining_actions = max(0, state.actions_remaining-1+int(extra))
        remaining_cards = [card for card in state.deck.hand if card != ref]
        active_now = sum(self.kernel.cards[card.card_id].card_type == 'active'
                         and self.kernel._non_full_power_cost_payable(paid, card, self.kernel.card_definition(paid, card))
                         for card in remaining_cards)
        future_active = int(any(self.kernel.cards[card.card_id].card_type == 'active'
                                for card in remaining_cards + list(state.deck.draw_pile)))
        result = 0.
        for effect, amount in schedules:
            uses = effect.schedule_limit
            for offset, projected in enumerate(windows):
                if effect.schedule_ttl is not None and offset > effect.schedule_ttl:
                    continue
                count = min(remaining_actions, active_now) if offset == 0 else future_active
                count = count if uses is None else min(count, uses)
                result += count*self.kernel.score_effect_gain(projected, EffectSpec(EffectKind.SCORE, amount), ref.instance_id)
                if uses is not None:
                    uses -= count
        return result/100

    def raw_features(self, state, actions):
        """Unnormalised six-column descriptors, or None outside their domain.

        Score descriptors use units of 100 points. The unredeemable indicator
        is dimensionless. The resource guard returns None for an omitted future
        opportunity; it never represents that omission as a zero feature. The
        separate schedule guard retains its existing action-order-only scope.
        """
        try:
            return self._raw_features(state, actions)
        except _UnpricedResourceOpportunity:
            return None

    def _raw_features(self, state, actions):
        if (not actions or state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                or any(action.kind not in {ActionKind.PLAY_CARD, ActionKind.END_TURN} for action in actions)):
            return None
        refs = {card.instance_id: card for card in state.deck.all_cards}
        entries = []
        for action in actions:
            ref = refs.get(action.card_instance_id) if action.kind is ActionKind.PLAY_CARD else None
            definition = self.kernel.card_definition(state, ref) if ref is not None else None
            effects = tuple(effect for effect in getattr(definition, 'effects', ()) if effect.condition is None)
            creates = sum(effect.amount for effect in effects if effect.kind is EffectKind.DOUBLE_CARD_EFFECT)
            passes = self.kernel._card_effect_passes(state, definition) if ref is not None else 1
            entries.append((ref, definition, effects, creates, passes))
        if not any(creates or passes > 1 for _, _, _, creates, passes in entries):
            return None
        windows, cache = self._windows(state), {}
        descriptions, action_values = [], {}
        for ref, definition, _, _, _ in entries:
            if ref is None:
                descriptions.append((0., 0.))
                continue
            direct = self._direct_score(state, ref, instance_repeats=False)/100
            persistent = self._persistent_capacity(state, ref, definition, windows)
            descriptions.append((direct, persistent))
            # This provisional value is consumed only by a legal double creator
            # targeting another card which can actually receive its double.
            checks_creation = (self.guard_unpriced_resources and any(
                creates and other is not None and other.instance_id != ref.instance_id
                for other, _, _, creates, _ in entries)
                and self.kernel._card_effect_passes(
                    replace(state, double_card_effects_remaining=1), definition) > 1)
            action_values[ref.instance_id] = direct+persistent+self._resource_conversion(
                state, ref, definition, 1, cache, check_future_scope=checks_creation)
        future, rows = self._future_target(state, cache), []
        for (ref, definition, effects, creates, passes), (direct, persistent) in zip(entries, descriptions):
            target = 0.
            if creates:
                paid, _ = self.kernel._pay_declared_non_full_power_cost(state, ref, definition)
                extra = sum(effect.amount for effect in effects if effect.kind is EffectKind.EXTRA_ACTION)
                if state.actions_remaining-1+extra > 0:
                    anticipated = replace(paid, double_card_effects_remaining=1)
                    target = max((action_values.get(card.instance_id, 0.) for card in state.deck.hand if card != ref
                        and self.kernel._card_effect_passes(anticipated, self.kernel.cards[card.card_id]) > 1
                        and self.kernel._non_full_power_cost_payable(paid, card, self.kernel.card_definition(paid, card))), default=0.)
                target = max(target, self._future_target(paid, cache))
            consumes = passes > 1
            rows.append((creates*target, float(bool(creates) and target == 0),
                direct if consumes else 0., persistent if consumes else 0.,
                self._resource_conversion(state, ref, definition, passes, cache,
                    check_future_scope=True) if consumes else 0.,
                future if consumes else 0.))
        return np.asarray(rows, dtype=float)

    def action_priors(self, state, actions):
        self.calls += 1
        old = tuple(self.old_advice.action_priors(state, actions))
        if self.guard_unpriced_schedules and self._has_unpriced_double_target(state, actions):
            self.bypassed_calls += 1
            self.scope_bypassed_calls += 1
            return old
        try:
            features = self._raw_features(state, actions)
        except _UnpricedResourceOpportunity:
            self.bypassed_calls += 1
            self.resource_scope_bypassed_calls += 1
            return old
        if features is None:
            self.bypassed_calls += 1
            return old
        columns = (0, 2, 3, 4, 5)
        scale = max((abs(features[:, index]).max() for index in columns), default=0.)
        features[:, columns] /= scale if scale > 0 else 1.
        old_order = np.argsort(np.argsort(old, kind='stable'), kind='stable')/max(1, len(old)-1)
        scores = features @ np.asarray(WEIGHTS) + old_order*1e-8
        if not np.isfinite(scores).all():
            raise ValueError('double advice returned nonfinite action priors')
        self.active_calls += 1
        return tuple(float(score) for score in scores)
