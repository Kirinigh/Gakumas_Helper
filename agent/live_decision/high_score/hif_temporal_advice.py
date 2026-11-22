"""Offline temporal action ordering using guarded scalar rule projections."""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np

from .hif import HifSenseKernel, _HifScalarProjection
from .kernel import EffectKind
from .contracts import PlanType, ActionKind, GameAction, DecisionKind, SchedulePhase
from .hif_forecast import expected_best
from .validation_cache import check_search_budget
from .hif_double_advice import HifDoubleAdvice

VERSION = 'task275-temporal-capacity-v2-20261001'
CALIBRATION_SOURCE = ('live-validation-20260930-strong-r1/'
                      'temporal-prior-projection-result.json')
NAMES = ('certain_action_score', 'current_payable_capacity', 'future_due_window_capacity')
SAFE = {EffectKind.SCORE, EffectKind.STAMINA, EffectKind.GENKI, EffectKind.FIXED_GENKI,
    EffectKind.COST_REDUCTION, EffectKind.EXTRA_ACTION, EffectKind.CONCENTRATION,
    EffectKind.GOOD_CONDITION, EffectKind.EXCELLENT_CONDITION, EffectKind.SCORE_BUFF,
    EffectKind.CONCENTRATION_EFFECT_BUFF, EffectKind.CONCENTRATION_ADDITION_BUFF,
    EffectKind.NULLIFY_COST_CARDS, EffectKind.HALF_COST_TURNS, EffectKind.DOUBLE_COST_TURNS,
    EffectKind.NULLIFY_GENKI_TURNS}
SAFE_CONDITIONS = {'ALL', 'TURN_ATTRIBUTE', 'GOOD_CONDITION_MIN', 'CONCENTRATION_MIN',
    'STAMINA_RATIO_MIN', 'TURNS_REMAINING_MAX', 'USED_CARD_SOURCE_TYPE', 'USED_CARD_ID',
    'USED_CARD_BASE_ID', 'USED_CARD_RARITY_NOT', 'USED_CARD_EFFECT_TAG', 'IS_DIRECT_EFFECT',
    'EFFECT_COUNTER_MOD_EQ'}


class UnsupportedProjection(ValueError):
    pass


class RecordingScalar(_HifScalarProjection):
    """Use real scalar rules, record draw counts but never sample card identities."""

    def __init__(self, kernel):
        super().__init__(kernel.cards, kernel.drinks, kernel.items, memory_abilities=kernel.memory_abilities)
        self.draws = []

    def _draw_guard(self, state):
        if any(ref.trigger_phase is SchedulePhase.CARD_MOVED_TO_HAND
               and ref.turns_until_fire == 0 and ref.remaining_uses != 0 and ref.remaining_turns != 0
               for ref in state.scheduled_effects):
            raise UnsupportedProjection('active_card_moved_to_hand_trigger')

    def _apply_effect(self, branches, effect, **kwargs):
        # The real kernel evaluates conditions anew at their actual phase.
        # Only effects whose conditions matched reach its recursive
        # unconditional call; inert named/attribute triggers stay inert.
        HifTemporalAdvice.condition(effect.condition)
        if effect.condition is None and effect.kind not in SAFE | {EffectKind.SCHEDULE, EffectKind.DRAW}:
            raise UnsupportedProjection('reachable_unsupported_effect:' + effect.kind.value)
        return super()._apply_effect(branches, effect, **kwargs)

    def _apply_non_draw_effect(self, state, effect, **kwargs):
        if effect.kind not in SAFE | {EffectKind.SCHEDULE}:
            raise UnsupportedProjection('reachable_unsupported_effect:' + effect.kind.value)
        return super()._apply_non_draw_effect(state, effect, **kwargs)

    def _scheduled_matched_nested_indices(self, scheduled, condition_state, trigger_context):
        effect = self._scheduled_effect_spec(scheduled)
        HifTemporalAdvice.condition(effect.schedule_counter_condition)
        for child in effect.scheduled_effects:
            HifTemporalAdvice.condition(child.condition)
        return super()._scheduled_matched_nested_indices(scheduled, condition_state, trigger_context)

    def _turn_start_draw_count(self, state):
        self._draw_guard(state)
        self.draws.append(dict(turn=state.turn, source='natural', count=HifSenseKernel._turn_start_draw_count(self, state)))
        return 0

    def _independent_draw_batch(self, state, count, *, source_id):
        if count:
            self._draw_guard(state)
        self.draws.append(dict(turn=state.turn, source=source_id, count=count))
        return super()._independent_draw_batch(state, count, source_id=source_id)

    def _draw_one(self, state, *, source_id):
        self._draw_guard(state)
        self.draws.append(dict(turn=state.turn, source=source_id, count=1))
        return super()._draw_one(state, source_id=source_id)


class HifTemporalAdvice:
    """Opt-in ordering capacities, never terminal values or execution states.

    At most the current and next turn are projected. Unknown card consumption
    across several future windows is not modeled; the same limited card must
    not be credited repeatedly. Earlier windows keep the existing advice.
    Held cards and identity-dependent paths also fall back.
    The three equal weights match 12 exact synthetic calibration states; seven
    separate mechanism checks were not fitted. This is not a learned policy.
    """

    model_version = VERSION

    def __init__(self, kernel, old_advice, weights=None):
        self.kernel, self.old = kernel, old_advice
        self.double = HifDoubleAdvice(kernel, old_advice)
        self.projection = RecordingScalar(kernel)
        self.weights = np.asarray(weights if weights is not None else (1., 1., 1.))
        self.last_audit = None
        self.calls = self.active_calls = self.bypassed_calls = 0
        self.elapsed_ms = 0.

    @staticmethod
    def condition(condition):
        if condition is None:
            return
        if condition.kind.value not in SAFE_CONDITIONS:
            raise UnsupportedProjection('identity_dependent_condition:' + condition.kind.value)
        for child in condition.all_of:
            HifTemporalAdvice.condition(child)

    def preflight(self, state, actions):
        if state.pending_choice or state.resolution_stack or state.pending_after_card_ids:
            raise UnsupportedProjection('pending_choice_or_resolution')
        if any(action.kind not in {ActionKind.PLAY_CARD, ActionKind.END_TURN} for action in actions):
            raise UnsupportedProjection('non_play_end_window')
        refs = {card.instance_id: card for card in state.deck.hand}
        if state.double_card_effects_remaining or any(
            effect.kind is EffectKind.DOUBLE_CARD_EFFECT
            for action in actions if action.card_instance_id in refs
            for effect in self.kernel.cards[refs[action.card_instance_id].card_id].effects):
            return 'double_priority'
        if state.turns_remaining > 2:
            raise UnsupportedProjection('multiple_unknown_future_windows_not_calibrated')
        if state.deck.known_top_count:
            raise UnsupportedProjection('known_draw_prefix_not_projected')
        if state.deck.held:
            raise UnsupportedProjection('held_cards_not_projected')
        # Never reject every candidate merely because an unrelated card or
        # inactive schedule has an unsupported definition. Reachability is
        # checked by RecordingScalar at each real effect application.
        return 'temporal_projection'

    def transition(self, state, action):
        check_search_budget()
        if action.card_instance_id:
            card = next(ref for ref in state.deck.all_cards if ref.instance_id == action.card_instance_id)
            self.condition(self.kernel.cards[card.card_id].condition)
        self.projection.draws = []
        branches = self.projection.transition(state, action, plan_type=PlanType.SENSE)
        if len(branches) != 1 or branches[0].probability != 1:
            raise UnsupportedProjection('projection_not_deterministic')
        after = branches[0].state
        if after.pending_choice or after.resolution_stack or after.pending_after_card_ids:
            raise UnsupportedProjection('projection_left_pending_identity')
        return after, tuple(self.projection.draws)

    def greedy_current(self, state):
        initial = state.score
        for _ in range(len(state.deck.hand)):
            options = []
            for action in self.projection.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE):
                if action.kind is ActionKind.PLAY_CARD:
                    after, draws = self.transition(state, action)
                    if draws:
                        raise UnsupportedProjection('current_greedy_needs_new_draw_identity')
                    options.append((after.score-state.score, after))
            if not options:
                break
            gain, after = max(options, key=lambda row: row[0])
            if gain <= 0:
                break
            state = after
        return state, state.score-initial

    def card_value(self, state, card):
        check_search_budget()
        # A distribution marginal over all known identities, never a choice
        # based on unknown order; these copies cannot be execution states.
        definition = self.kernel.cards[card.card_id]
        self.condition(definition.condition)
        if state.deck.hand or state.deck.held:
            raise UnsupportedProjection('future_marginal_requires_empty_known_hand_and_held')
        self.effect_tree_is_scalar(definition.effects)
        # This capacity asks only for the next card's certain score. A
        # plain mental buff with no score trigger has exactly zero direct
        # scoring capacity; its later preparation chain is outside this
        # descriptor, so do not execute unrelated zone-changing memories.
        has_score = any(effect.kind is EffectKind.SCORE or effect.kind is EffectKind.SCHEDULE
                        for effect in definition.effects)
        for reference in state.scheduled_effects:
            if reference.remaining_uses == 0 or reference.remaining_turns == 0:
                continue
            if reference.trigger_phase not in {SchedulePhase.BEFORE_CARD_USED, SchedulePhase.CARD_USED, SchedulePhase.AFTER_CARD_USED}:
                continue
            effect = self.kernel._scheduled_effect_spec(reference)
            if effect.trigger_card_type in (None, definition.card_type):
                has_score |= any(child.kind is EffectKind.SCORE for child in effect.scheduled_effects)
        if not has_score:
            return 0.
        deck = state.deck.replace_zones(hand=(card,),
            draw_pile=tuple(ref for ref in state.deck.draw_pile if ref != card),
            discard=tuple(ref for ref in state.deck.discard if ref != card),
            held=tuple(ref for ref in state.deck.held if ref != card))
        hypothetical = replace(state, deck=deck)
        action = next((action for action in self.projection.legal_actions(hypothetical, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
                       if action.kind is ActionKind.PLAY_CARD and action.card_instance_id == card.instance_id), None)
        if action is None:
            return 0.
        after, _ = self.transition(hypothetical, action)
        return max(0., after.score-state.score)

    def effect_tree_is_scalar(self, effects):
        # Unknown automatic-play effects cannot be treated as score zero.
        for effect in effects:
            if effect.kind not in SAFE | {EffectKind.SCHEDULE, EffectKind.DRAW}:
                raise UnsupportedProjection('future_capacity_unsupported_effect:' + effect.kind.value)

    def draw_capacity(self, state, count):
        values = [self.card_value(state, card) for card in state.deck.draw_pile]
        if count <= len(values):
            return expected_best(values, count)
        known_best = max(values, default=0.)
        discard = [max(known_best, self.card_value(state, card)) for card in state.deck.discard]
        return expected_best(discard, count-len(values)) if discard else known_best

    def future(self, state, receipts, *, already_advanced, audit=False):
        capacity, windows = 0., []
        current_receipts = receipts
        include_current = already_advanced
        while not state.terminal:
            check_search_budget()
            if not include_current:
                before_score = state.score
                state, current_receipts = self.transition(state, GameAction(ActionKind.END_TURN))
                capacity += state.score-before_score
                if state.terminal:
                    break
            include_current = False
            if state.turns_remaining > 1 and any(
                reference.trigger_phase in {SchedulePhase.BEFORE_CARD_USED, SchedulePhase.CARD_USED, SchedulePhase.AFTER_CARD_USED}
                and reference.remaining_uses is not None and reference.remaining_uses > 0
                and reference.remaining_turns != 0
                for reference in state.scheduled_effects):
                # Indirect score gains (e.g. BEFORE_CARD_USED score buffs)
                # also consume a finite opportunity. We cannot propagate
                # that consumption without selecting an unknown future card.
                raise UnsupportedProjection('finite_card_uses_across_unknown_future_actions')
            natural = sum(row['count'] for row in current_receipts if row['source'] == 'natural')
            extra = sum(row['count'] for row in current_receipts if row['source'] != 'natural')
            natural = min(state.hand_limit-len(state.deck.hand), natural)
            total = min(state.hand_limit-len(state.deck.hand), natural+extra)
            ordinary = self.draw_capacity(state, natural) if state.actions_remaining else 0.
            extra_value = (self.draw_capacity(state, total)-ordinary) if state.actions_remaining and total != natural else 0.
            capacity += ordinary+extra_value
            if audit:
                windows.append(dict(turn=state.turn, natural_draws=natural, extra_draws=extra,
                    capacity_draws=total, natural_capacity=ordinary, extra_draw_increment=extra_value,
                    stamina=state.stamina, concentration=state.concentration,
                    schedules=[asdict(ref) for ref in state.scheduled_effects]))
            # Reserve the hypothetical future action: do not additionally
            # credit the stamina recovery granted by skipping that action.
            state = replace(state, actions_remaining=0)
        return capacity, windows

    def matrix(self, state, actions, *, audit=True):
        check_search_budget()
        try:
            route = self.preflight(state, actions)
            if route == 'double_priority':
                return None, dict(route=route)
            rows, details = [], []
            for action in actions:
                after, receipts = self.transition(state, action)
                immediate = after.score-state.score
                advanced = after.turn != state.turn
                current = 0.
                if not advanced and not after.terminal:
                    if receipts:
                        raise UnsupportedProjection('current_action_needs_new_draw_identity')
                    after, current = self.greedy_current(after)
                future, windows = self.future(after, receipts, already_advanced=advanced, audit=audit)
                rows.append((immediate, current, future))
                if audit:
                    details.append(dict(windows=windows))
            values = np.asarray(rows, dtype=float)
            scale = max(1e-30, float(np.max(np.abs(values))))
            return values/scale, dict(route='temporal_projection', scale=scale, raw=values.tolist(), details=details)
        except UnsupportedProjection as error:
            return None, dict(route='old_prior', reason=str(error))

    def action_priors(self, state, actions):
        self.calls += 1
        started = time.perf_counter()
        try:
            values, self.last_audit = self.matrix(state, actions, audit=False)
            if self.last_audit['route'] == 'double_priority':
                self.bypassed_calls += 1
                return self.double.action_priors(state, actions)
            old = tuple(self.old.action_priors(state, actions))
            if values is None:
                self.bypassed_calls += 1
                return old
            self.active_calls += 1
            order = np.argsort(np.argsort(old, kind='stable'), kind='stable') / max(1, len(old)-1)
            return tuple(values @ self.weights + order*1e-8)
        finally:
            self.elapsed_ms += (time.perf_counter()-started)*1000
