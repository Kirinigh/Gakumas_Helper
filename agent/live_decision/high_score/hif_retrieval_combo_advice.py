"""Public same-hand double retrieval hints; never terminal values or pruning."""
from __future__ import annotations

import math
import time

from .kernel import EffectKind
from .contracts import PlanType, ActionKind, DecisionKind, SchedulePhase
from .state_copy import replace
from .hif_learning import HifFeatureEncoderV13
from .validation_cache import check_search_budget
from .hif_double_advice import HifDoubleAdvice

VERSION = 'task275-visible-retrieval-combo-v3'
_CACHE_LIMIT = 4096


class UnsupportedRetrievalCombo(ValueError):
    """The local option cannot be described without guessing a continuation."""


class HifRetrievalComboAdvice:
    """Order retrieval options using one legal preparation and one partner.

    MOVE candidates share one comparable best-combo capacity after the actual
    move resolves. A drink is promoted only by a positive increment over the
    current hand's reachable combo capacity. In mixed drink/play windows, the
    existing double advice orders PLAY/END inside their original rank slots;
    drink ranks stay fixed until the positive retrieval-increment rule runs.
    Capacity projects the schedules left by two actual partner transitions;
    it is neither a terminal return nor a proof to discard another action.
    """

    model_version = VERSION

    def __init__(self, kernel, old_advice):
        self.kernel, self.old = kernel, old_advice
        self.double = (old_advice if isinstance(old_advice, HifDoubleAdvice)
                       else HifDoubleAdvice(kernel, old_advice))
        self.calls = self.active_calls = self.bypassed_calls = 0
        self.elapsed_ms = 0.
        self.last_audit = []
        self.reset_cache()

    def reset_cache(self):
        self._combos = {}
        self._moves = {}

    def future_score(self, state):
        method = getattr(self.old, 'future_score', None)
        return method(state) if method is not None else None

    @staticmethod
    def _single(branches, stage, *, pending=False):
        if len(branches) != 1 or not math.isclose(branches[0].probability, 1., abs_tol=1e-12):
            raise UnsupportedRetrievalCombo('random_or_multiple_branches:' + stage)
        state = branches[0].state
        if (state.resolution_stack or state.pending_after_card_ids
                or state.pending_choice is not None and not pending):
            raise UnsupportedRetrievalCombo('unresolved:' + stage)
        return state

    def _transition(self, state, action, stage, *, pending=False):
        check_search_budget()
        return self._single(self.kernel.transition(state, replace(action, source_slot=None),
            plan_type=PlanType.SENSE), stage, pending=pending)

    def _legal(self, state):
        check_search_budget()
        return self.kernel.legal_actions(state, plan_type=PlanType.SENSE,
            decision_kind=DecisionKind.SECONDARY if state.pending_choice else DecisionKind.LIVE)

    def _preparations(self, state):
        # Existing double is already a legal public resource, not a fake card.
        result = [(None, state)] if state.double_card_effects_remaining else []
        refs = {card.instance_id: card for card in state.deck.hand}
        for action in self._legal(state):
            if action.kind is not ActionKind.PLAY_CARD:
                continue
            definition = self.kernel.cards[refs[action.card_instance_id].card_id]
            if not any(effect.kind is EffectKind.DOUBLE_CARD_EFFECT for effect in definition.effects):
                continue
            # Kernel payment and conditions determine whether preparation
            # really arms a double and leaves an action in this same turn.
            after = self._transition(state, action, 'preparation')
            if (not after.terminal and after.turn == state.turn and after.actions_remaining > 0
                    and after.double_card_effects_remaining > state.double_card_effects_remaining):
                result.append((action.card_instance_id, after))
        return result

    @staticmethod
    def _score_partner(definition):
        return any(effect.kind is EffectKind.SCORE or HifFeatureEncoderV13._active_score_amount(effect) > 0
                   for effect in definition.effects)

    def _remaining_schedule_capacity(self, state, target):
        """Approximate future opportunities after all synchronous effects.

        Payment, installed schedules, consumed uses and current resources are
        already resolved by the kernel. No second payment or just-fired score
        belongs here. Only the future opportunity count remains an estimate:
        affordable active cards now, then one active play per future turn when
        one exists in the public hand/draw pool; no future draw is disclosed.
        """
        if state.terminal:
            return 0.
        schedules = tuple(scheduled for scheduled in state.scheduled_effects
            if scheduled.source_id == target.instance_id and scheduled.card_id == target.card_id
            and scheduled.trigger_phase is SchedulePhase.CARD_USED
            and scheduled.turns_until_fire == 0 and scheduled.remaining_uses != 0
            and scheduled.remaining_turns != 0)
        if not schedules:
            return 0.
        refs = {card.instance_id: card for card in state.deck.hand}
        active_now = sum(action.kind is ActionKind.PLAY_CARD
            and self.kernel.cards[refs[action.card_instance_id].card_id].card_type == 'active'
            for action in self._legal(state))
        future_active = int(any(self.kernel.cards[card.card_id].card_type == 'active'
                               for card in state.deck.hand + state.deck.draw_pile))
        windows = self.double._windows(state)
        result = 0.
        for scheduled in schedules:
            effect = self.kernel._scheduled_effect_spec(scheduled)
            if not HifFeatureEncoderV13._active_score_amount(effect):
                continue
            # Mixed or conditional child effects require another actual
            # continuation; do not silently score them as unconditional.
            if any(child.kind is not EffectKind.SCORE or child.condition is not None
                   or child.value_source is not None or child.full_power_scale
                   for child in effect.scheduled_effects):
                raise UnsupportedRetrievalCombo('non_scalar_scheduled_partner')
            uses = scheduled.remaining_uses
            for offset, projected in enumerate(windows):
                if scheduled.remaining_turns is not None and offset >= scheduled.remaining_turns:
                    break
                count = min(state.actions_remaining, active_now) if offset == 0 else future_active
                if uses is not None:
                    count = min(count, uses)
                    uses -= count
                result += count*sum(self.kernel.score_effect_gain(projected, child, target.instance_id)
                                    * child.repeats for child in effect.scheduled_effects)
        return result

    def _capacity(self, state, target, action):
        definition = self.kernel.cards[target.card_id]
        passes = self.kernel._card_effect_passes(state, definition)
        if passes <= 1 or not self._score_partner(definition):
            return None
        # A legal target is still actually played through both public rules.
        # This preserves typed payment, conditional SCORE and synchronous
        # effects. Retrieving a card is never counted as having played it.
        doubled = self._transition(state, action, 'double_partner')
        single = self._transition(replace(state, double_card_effects_remaining=0), action, 'single_partner')
        if doubled.turn != state.turn or single.turn != state.turn:
            raise UnsupportedRetrievalCombo('partner_crossed_turn')
        direct = max(0., float(doubled.score-single.score))
        persistent = max(0., self._remaining_schedule_capacity(doubled, target)
                         - self._remaining_schedule_capacity(single, target))
        # The opportunity projection is an approximation. The transitions
        # prove the current chain can finish, not its future terminal payoff.
        capacity = direct + persistent
        if not math.isfinite(capacity):
            raise ValueError('retrieval_combo_nonfinite_capacity')
        return dict(capacity=capacity, extra_immediate_score=direct,
            extra_scheduled_capacity=persistent,
            target_instance_id=target.instance_id, target_card_id=target.card_id,
            passes=passes, actions_after_partner=doubled.actions_remaining,
            score_after_partner=doubled.score)

    def combo(self, state):
        """Best supported combo currently reachable in the same public hand."""
        check_search_budget()
        if state in self._combos:
            return self._combos[state]
        if state.pending_choice or state.resolution_stack or state.pending_after_card_ids:
            raise UnsupportedRetrievalCombo('unresolved_combo_state')
        options = []
        if not state.terminal and state.actions_remaining > 0:
            for setup, prepared in self._preparations(state):
                refs = {card.instance_id: card for card in prepared.deck.hand}
                for action in self._legal(prepared):
                    if action.kind is not ActionKind.PLAY_CARD:
                        continue
                    target = refs.get(action.card_instance_id)
                    if target is None or target.instance_id == setup:
                        continue
                    described = self._capacity(prepared, target, action)
                    if described is not None:
                        options.append(dict(preparation_instance_id=setup, **described))
        result = dict(capacity=max((option['capacity'] for option in options), default=0.), options=options)
        if len(self._combos) >= _CACHE_LIMIT:
            self.reset_cache()
        self._combos[state] = result
        return result

    def move(self, state, action):
        check_search_budget()
        pending = state.pending_choice
        if (not pending or pending.action_kind is not ActionKind.MOVE_CARD
                or action.kind is not ActionKind.MOVE_CARD):
            raise UnsupportedRetrievalCombo('not_one_move_choice')
        key = state, replace(action, source_slot=None)
        if key in self._moves:
            return self._moves[key]
        after = self._transition(state, action, 'move_completion')
        if (after.turn != state.turn or not any(card.instance_id == action.card_instance_id
                                              for card in after.deck.hand)):
            raise UnsupportedRetrievalCombo('retrieved_target_not_retained_same_turn')
        result = dict(supported=True, target_instance_id=action.card_instance_id,
            after_actions=after.actions_remaining, after_stamina=after.stamina,
            after_double=after.double_card_effects_remaining, **self.combo(after))
        if len(self._moves) >= _CACHE_LIMIT:
            self.reset_cache()
        self._moves[key] = result
        return result

    def drink(self, state, action):
        if (state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                or action.kind is not ActionKind.USE_DRINK):
            raise UnsupportedRetrievalCombo('not_resolved_drink_choice')
        definition = self.kernel.drinks[action.drink_id]
        if sum(effect.kind is EffectKind.MOVE_SELECTED_TO_HAND for effect in definition.effects) != 1:
            raise UnsupportedRetrievalCombo('not_single_retrieval_drink')
        before = self.combo(state)
        pending = self._transition(state, action, 'retrieval_drink', pending=True)
        moves = self._legal(pending)
        if not pending.pending_choice or not moves or any(a.kind is not ActionKind.MOVE_CARD for a in moves):
            raise UnsupportedRetrievalCombo('retrieval_did_not_open_one_move_group')
        outcomes = [self.move(pending, move) for move in moves]
        after = max(row['capacity'] for row in outcomes)
        return dict(supported=True, capacity=after, baseline_capacity=before['capacity'],
            increment=after-before['capacity'], targets=outcomes)

    @staticmethod
    def _ranked_values(order):
        result = [0.]*len(order)
        for rank, index in enumerate(order):
            result[index] = float(len(order)-rank)
        return tuple(result)

    def _play_end_order(self, state, actions, old_order):
        if (state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                or not any(action.kind is ActionKind.USE_DRINK for action in actions)):
            return old_order, False
        indices = [i for i in old_order if actions[i].kind in {ActionKind.PLAY_CARD, ActionKind.END_TURN}]
        if len(indices) < 2:
            return old_order, False
        check_search_budget()
        active_before = self.double.active_calls
        values = tuple(self.double.action_priors(state, tuple(actions[i] for i in indices)))
        if len(values) != len(indices) or not all(map(math.isfinite, values)):
            raise ValueError('retrieval_combo_invalid_play_end_priors')
        reordered = iter(indices[i] for i in sorted(range(len(indices)), key=lambda i: -values[i]))
        positions = set(indices)
        order = [next(reordered) if i in positions else i for i in old_order]
        return order, self.double.active_calls > active_before

    def action_priors(self, state, actions):
        self.calls += 1
        started = time.perf_counter()
        self.last_audit = []
        supported = False
        try:
            check_search_budget()
            old = tuple(self.old.action_priors(state, actions))
            if len(old) != len(actions) or not all(map(math.isfinite, old)):
                raise ValueError('retrieval_combo_invalid_original_priors')
            old_order = sorted(range(len(actions)), key=lambda i: -old[i])
            if actions and all(a.kind is ActionKind.MOVE_CARD for a in actions):
                try:
                    rows = [dict(index=i, **self.move(state, action)) for i, action in enumerate(actions)]
                except UnsupportedRetrievalCombo as error:
                    self.last_audit = [dict(supported=False, reason=str(error), scope='entire_move_group')]
                    return old
                self.last_audit = rows
                supported = True
                if not any(row['capacity'] > 0 for row in rows):
                    return old
                order = sorted(old_order, key=lambda i: -rows[i]['capacity'])
            else:
                order, supported = self._play_end_order(state, actions, old_order)
                gains = {}
                for index, action in enumerate(actions):
                    if action.kind is not ActionKind.USE_DRINK:
                        continue
                    check_search_budget()
                    try:
                        row = self.drink(state, action)
                        supported = True
                    except UnsupportedRetrievalCombo as error:
                        row = dict(supported=False, reason=str(error))
                    self.last_audit.append(dict(index=index, **row))
                    if row['supported'] and row['increment'] > 0:
                        gains[index] = row['increment']
                # Fixed ordinal combination: positive retrieval increments,
                # then the mixed-window order; no action is discarded.
                if gains:
                    order = sorted(order, key=lambda i: (i not in gains, -gains.get(i, 0.)))
            return old if order == old_order else self._ranked_values(order)
        finally:
            self.active_calls += int(supported)
            self.bypassed_calls += int(not supported)
            self.elapsed_ms += (time.perf_counter()-started)*1000
