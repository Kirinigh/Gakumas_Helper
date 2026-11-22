"""Opt-in drink ordering by conditional single-card opportunities.

Each public future window is an independent resource cross-section, never a
predicted state or final return. A bottle receives its best single opportunity;
limited cards, free costs and multiple windows are not summed as extra plays.
Runtime uses only the supplied kernel and already-loaded original advice.
"""
from __future__ import annotations

import time

import numpy as np

from .kernel import EffectKind, CardCostKind, TransitionBranch, _condition_matches
from .contracts import CardZone, PlanType, ActionKind, DecisionKind, SchedulePhase
from .state_copy import replace
from .hif_forecast import expected_best, _decayed_modifiers
from .validation_cache import check_search_budget

VERSION = 'task275-drink-opportunity-v1-20261001'
MECHANISM_SOURCE = 'HIF drink-temporal-mechanisms: four exact synthetic cases'
_CACHE_LIMIT = 8192
DRINK_KINDS = frozenset({EffectKind.SCORE_BUFF, EffectKind.HALF_COST_TURNS, EffectKind.STAMINA,
    EffectKind.MOVE_SELECTED_TO_HAND, EffectKind.NULLIFY_COST_CARDS, EffectKind.DOUBLE_COST_TURNS})
SAFE = frozenset({EffectKind.SCORE, EffectKind.STAMINA, EffectKind.GENKI, EffectKind.FIXED_GENKI,
    EffectKind.COST_REDUCTION, EffectKind.EXTRA_ACTION, EffectKind.CONCENTRATION,
    EffectKind.GOOD_CONDITION, EffectKind.EXCELLENT_CONDITION, EffectKind.SCORE_BUFF,
    EffectKind.CONCENTRATION_EFFECT_BUFF, EffectKind.CONCENTRATION_ADDITION_BUFF,
    EffectKind.NULLIFY_COST_CARDS, EffectKind.HALF_COST_TURNS, EffectKind.DOUBLE_COST_TURNS,
    EffectKind.NULLIFY_GENKI_TURNS})


class UnsupportedOpportunity(ValueError):
    pass


class HifDrinkAdvice:
    """Change only drink/selection exploration order; keep the supplied policy.

    Direct scores use the kernel's payment, conditions and effect application.
    Preparation chains, future consumption and established scheduled score are
    excluded. Caches contain direct scalar values only, never transitions,
    natural draw capacities or retrieval choices. Call reset_cache per search.
    """

    model_version = VERSION

    def __init__(self, kernel, old_advice):
        self.kernel, self.old = kernel, old_advice
        self.last_audit = []
        self.calls = self.active_calls = self.bypassed_calls = 0
        self.elapsed_ms = 0.
        self.shared_target_windows = 0
        self._score_ids = {key for key, definition in kernel.cards.items()
                           if any(effect.kind is EffectKind.SCORE for effect in definition.effects)}
        self.reset_cache()

    def reset_cache(self):
        """Reset all context IDs and values together at a choose boundary."""
        self._values = {}
        self._state_contexts = {}
        self._interned_contexts = {}

    def future_score(self, state):
        method = getattr(self.old, 'future_score', None)
        return method(state) if method is not None else None

    def resource_key(self,state):
        check_search_budget()
        found = self._state_contexts.get(id(state))
        if found is not None:
            assert found[0] is state
            return found[1]
        if (len(self._state_contexts) >= _CACHE_LIMIT
                or len(self._interned_contexts) >= _CACHE_LIMIT):
            self.reset_cache()
        # Cache identity only: never pass this normalized deck to the kernel.
        # All state scalars, full card refs, removed/held zones and deck metadata
        # survive. Only permutation/partition of hand, draw and discard is lost.
        canonical = self.resource_state(state)
        context = self._interned_contexts.setdefault(canonical, len(self._interned_contexts))
        self._state_contexts[id(state)] = (state, context)
        return context

    @staticmethod
    def resource_state(state):
        pool = tuple(sorted((*state.deck.hand, *state.deck.draw_pile, *state.deck.discard),
                            key=lambda card: card.instance_id))
        return replace(state, deck=replace(state.deck, hand=(), draw_pile=pool, discard=()))

    def window(self, state, offset):
        check_search_budget()
        changes = dict(turn=state.turn+offset, turns_remaining=state.turns_remaining-offset,
            score_multiplier=(state.score_multiplier, *state.future_score_multipliers)[offset],
            future_score_multipliers=state.future_score_multipliers[offset:],
            turn_attributes=state.turn_attributes[offset:])
        for value, fresh in (('good_condition_turns','good_condition_fresh'),
                ('excellent_condition_turns','excellent_condition_fresh'),
                ('half_cost_turns','half_cost_fresh'), ('double_cost_turns','double_cost_fresh'),
                ('nullify_genki_turns','nullify_genki_fresh')):
            left = max(0, getattr(state,value)-max(0,offset-int(getattr(state,fresh))))
            changes[value], changes[fresh] = left, bool(left and offset == 0 and getattr(state,fresh))
        for name in ('score_buffs','concentration_effect_buffs','concentration_addition_buffs'):
            decayed = _decayed_modifiers(getattr(state,name),offset)
            changes[name] = tuple(replace(buff,fresh=False) for buff in decayed) if offset else decayed
        return replace(state, **changes)

    @staticmethod
    def single(branches, stage, *, allow_pending=False):
        if len(branches) != 1 or abs(branches[0].probability-1) > 1e-12:
            raise UnsupportedOpportunity('random_or_multiple_branches:' + stage)
        after = branches[0].state
        if after.resolution_stack or (after.pending_choice and not allow_pending):
            raise UnsupportedOpportunity('pending_resolution:' + stage)
        return after

    def synchronous_guard(self, state, phase, context, *, future=False):
        # Inspect only actually due, identity/condition-matching event paths.
        # Other phases and unmatched memories do not reject an opportunity.
        for ref in state.scheduled_effects:
            check_search_budget()
            if (ref.trigger_phase is not phase or ref.turns_until_fire != 0
                    or ref.remaining_uses == 0 or ref.remaining_turns == 0
                    or not self.kernel._scheduled_card_filter_matches(ref,context)):
                continue
            effect = self.kernel._scheduled_effect_spec(ref)
            indices = self.kernel._scheduled_matched_nested_indices(ref,state,context)
            if (future and indices and ref.remaining_turns is not None
                    and phase in {SchedulePhase.BEFORE_CARD_USED, SchedulePhase.CARD_USED}):
                raise UnsupportedOpportunity('matched_timed_schedule_in_future:' + phase.value + ':' + ref.card_id)
            unsupported = [effect.scheduled_effects[i].kind.value for i in indices
                           if effect.scheduled_effects[i].kind not in SAFE]
            if unsupported:
                raise UnsupportedOpportunity('synchronous:' + phase.value + ':' + ref.card_id + ':' + ','.join(unsupported))

    def direct(self, state, card, *, future=False):
        """Pay actual cost, run matching scalar pre-use events, count direct SCORE only."""
        check_search_budget()
        if card.card_id not in self._score_ids:
            return 0.
        if len(self._values) >= _CACHE_LIMIT:
            self.reset_cache()
        key=(self.resource_key(state),card,future)
        if key not in self._values:
            self._values[key]=self._direct_uncached(state,card,future=future)
        return self._values[key]

    def _direct_uncached(self,state,card, *, future=False):
        # Score a prefix conditional on this entity already having reached hand.
        # Move only that entity; do not draw, shuffle or fire an unknown entry
        # event. Real drink retrieval entry events are handled by outcomes().
        if card not in state.deck.hand:
            if card not in (*state.deck.draw_pile, *state.deck.discard):
                raise UnsupportedOpportunity('direct_card_not_in_active_pool')
            state, _ = self.kernel._move_card_to_hand(state, card.instance_id)
        definition = self.kernel.cards[card.card_id]
        if not any(e.kind is EffectKind.SCORE for e in definition.effects):
            return 0.  # Explicit scope: no preparation or newly registered 301 schedule value.
        if (definition.required_stance not in (None,state.stance)
                or not _condition_matches(state,definition.condition,self.kernel.cards)
                or not self.kernel._non_full_power_cost_payable(state,card,definition)
                or self.kernel._effective_full_power_cost(card,definition) > state.full_power):
            return 0.
        context = self.kernel._card_event_context(state,card.instance_id,exclude_used_card_from_targets=False)
        self.synchronous_guard(state,SchedulePhase.BEFORE_CARD_USED,context,future=future)
        pre = self.single(self.kernel._trigger_event_phase((TransitionBranch(1.,state,()),),
            SchedulePhase.BEFORE_CARD_USED,trigger_context=context,condition_state=state),'before_card')
        passes = self.kernel._card_effect_passes(pre,definition)
        paid,_ = self.kernel._pay_declared_non_full_power_cost(pre,card,definition)
        if pre.nullify_cost_cards and self.kernel._declared_cost_kind(definition) in {CardCostKind.NORMAL,CardCostKind.STAMINA}:
            paid = replace(paid,nullify_cost_cards=paid.nullify_cost_cards-1)
        paid = replace(paid,full_power=paid.full_power-self.kernel._effective_full_power_cost(card,definition))
        paid,_ = self.kernel._move_card_for_use(paid,card,definition)
        context = self.kernel._card_event_context(paid,card.instance_id,exclude_used_card_from_targets=True)
        self.synchronous_guard(pre,SchedulePhase.CARD_USED,context,future=future)
        paid = self.single(self.kernel._trigger_event_phase((TransitionBranch(1.,paid,()),),
            SchedulePhase.CARD_USED,trigger_context=context,condition_state=pre),'card_used')
        # Score created by an established CARD_USED schedule is deliberately
        # outside this descriptor. It may change resources before direct SCORE.
        direct_total = 0.
        last_score = max(i for i,e in enumerate(definition.effects) if e.kind is EffectKind.SCORE)
        for effect_pass in range(passes):
            for index,effect in enumerate(definition.effects):
                check_search_budget()
                if effect_pass == passes-1 and index > last_score:
                    break
                if not _condition_matches(pre,effect.condition,self.kernel.cards):
                    continue
                if effect.kind is EffectKind.SCHEDULE:
                    continue  # registration has no immediate direct-score payment
                if effect.kind not in SAFE:
                    raise UnsupportedOpportunity('direct_score_prefix:' + definition.card_id + ':' + effect.kind.value)
                branches = self.kernel._apply_effect((TransitionBranch(1.,paid,()),),effect,
                    source_id=card.instance_id,condition_state=pre,source_card_id=definition.card_id,
                    definition_effect_index=index,direct_effect=True)
                paid = self.single(branches,'direct_effect')
                direct_total += sum(event.amount for event in branches[0].events
                    if event.event_type == 'SCORE' and event.source_id == card.instance_id)
        return float(direct_total)

    def current_capacity(self,state, *, exclude=None):
        if state.actions_remaining <= 0:
            return 0.
        return max((self.direct(state,c) for c in state.deck.hand if c.instance_id != exclude),default=0.)

    def natural_capacity(self,state, *, guaranteed=()):
        """Current observed zones, one natural three-card offer; no future sequence."""
        if state.deck.known_top_count or state.deck.held:
            raise UnsupportedOpportunity('natural_offer_known_prefix_or_held')
        fixed = max((self.direct(state,c,future=True) for c in guaranteed),default=0.)
        draws = min(3,max(0,state.hand_limit-len(guaranteed)))
        values = [max(fixed,self.direct(state,c,future=True)) for c in state.deck.draw_pile]
        if len(values) >= draws:
            return expected_best(values,draws) if draws else fixed
        best = max([fixed,*values])
        # Next natural refill can reshuffle the presently observed hand plus
        # discard; this is an explicit composition cross-section, not a claim
        # that prior hypothetical limited plays leave these cards available.
        guaranteed_ids = {c.instance_id for c in guaranteed}
        rest = [max(best,self.direct(state,c,future=True)) for c in (*state.deck.discard,*state.deck.hand)
                if c.instance_id not in guaranteed_ids]
        return expected_best(rest,draws-len(values)) if rest else best

    def check_drink(self,definition):
        if any(e.condition or e.kind not in DRINK_KINDS for e in definition.effects):
            raise UnsupportedOpportunity('unsupported_compound_effect')
        moves = [e for e in definition.effects if e.kind is EffectKind.MOVE_SELECTED_TO_HAND]
        if len(moves)>1 or any(e.selection_count != 1 or not set(e.target_zones)<={CardZone.DRAW_PILE,CardZone.DISCARD} for e in moves):
            raise UnsupportedOpportunity('unsupported_retrieval_target_scope')
        return bool(moves)

    def outcomes(self,state,action):
        check_search_budget()
        self.check_drink(self.kernel.drinks[action.drink_id])
        pending = self.single(self.kernel.transition(state,action,plan_type=PlanType.SENSE),'drink',allow_pending=True)
        if not pending.pending_choice:
            return [(None,pending)]
        if pending.pending_choice.action_kind is not ActionKind.MOVE_CARD:
            raise UnsupportedOpportunity('drink_secondary_is_not_move')
        moves = self.kernel.legal_actions(pending,plan_type=PlanType.SENSE,decision_kind=DecisionKind.SECONDARY)
        if not moves:
            raise UnsupportedOpportunity('no_move_targets')
        refs = {card.instance_id: card for card in state.deck.all_cards}
        shared = not any(ref.trigger_phase is SchedulePhase.CARD_MOVED_TO_HAND
                         and ref.turns_until_fire == 0 and ref.remaining_uses != 0
                         for ref in pending.scheduled_effects)
        first = (self.single(self.kernel.transition(pending,moves[0],plan_type=PlanType.SENSE),
                            'retrieval_completion') if shared else None)
        results = []
        for move in moves:
            check_search_budget()
            ref = refs[move.card_instance_id]
            moved,_ = self.kernel._move_card_to_hand(pending,ref.instance_id)
            if shared:
                # With no entry trigger, verified unconditional scalar drink
                # continuation is identical for every legal target.
                after = replace(first, deck=moved.deck)
            else:
                context = self.kernel._moved_card_event_context(moved,ref.instance_id)
                self.synchronous_guard(moved,SchedulePhase.CARD_MOVED_TO_HAND,context)
                after = self.single(self.kernel.transition(pending,move,plan_type=PlanType.SENSE),'retrieval_completion')
            results.append((ref,after))
        if shared:
            self.shared_target_windows += 1
        return results

    def describe(self,state,action):
        if state.pending_choice or state.resolution_stack or state.pending_after_card_ids:
            raise UnsupportedOpportunity('unresolved_root')
        if state.terminal or len(state.future_score_multipliers) != state.turns_remaining-1:
            raise UnsupportedOpportunity('incomplete_future_windows')
        definition = self.kernel.drinks[action.drink_id]
        retrieval = self.check_drink(definition)
        outcomes = self.outcomes(state,action)
        rows = []
        if retrieval:
            # Exactly one single-card opportunity. Two free costs do not add
            # an action, a second score, or a second future bottle use.
            for offset in range(state.turns_remaining):
                check_search_budget()
                before = self.window(state,offset)
                baseline = self.current_capacity(before) if offset==0 else self.natural_capacity(before)
                available=outcomes if offset==0 else self.outcomes(before,action)
                if len({self.resource_state(after) for _,after in available}) != 1:
                    raise UnsupportedOpportunity('target_specific_synchronous_resource_changes')
                # Equal direct opportunities use a public stable identity.
                # This avoids dependence on unordered pool iteration, without
                # claiming every target's future capacity was optimized.
                target,after=max(available,key=lambda pair:(
                    self.direct(pair[1],pair[0],future=offset>0),pair[0].instance_id))
                if target not in after.deck.hand:
                    raise UnsupportedOpportunity('retrieved_card_not_retained_in_hand')
                capacity=self.current_capacity(after) if offset==0 else self.natural_capacity(after,guaranteed=(target,))
                chosen=dict(target=target.instance_id,capacity=capacity,
                    nullify_cost_cards=after.nullify_cost_cards,stamina=after.stamina,
                    genki=after.genki,double_cost_turns=after.double_cost_turns)
                rows.append(dict(offset=offset,baseline=baseline,increment=chosen['capacity']-baseline,**chosen))
            now = rows[0]['increment']
        else:
            after = outcomes[0][1]
            # Current use may cover a later opportunity, but only the best
            # one is credited. No finite card is counted repeatedly.
            for offset in range(state.turns_remaining):
                check_search_budget()
                before = self.window(state,offset)
                baseline = self.current_capacity(before) if offset==0 else self.natural_capacity(before)
                early = self.window(after,offset)
                early_capacity = self.current_capacity(early) if offset==0 else self.natural_capacity(early)
                late = after if offset==0 else self.outcomes(before,action)[0][1]
                late_capacity = self.current_capacity(late) if offset==0 else self.natural_capacity(late)
                rows.append(dict(offset=offset,baseline=baseline,
                    increment=late_capacity-baseline,already_used_increment=early_capacity-baseline,
                    stamina_after_use=late.stamina,genki_after_use=late.genki,
                    half_cost_turns_after_use=late.half_cost_turns))
            now = max(row['already_used_increment'] for row in rows)
        later = max((row['increment'] for row in rows[1:]),default=0.)
        return dict(supported=True,scope='conditional_single_card_direct_opportunity',retrieval=retrieval,
            now_best_increment=now,later_best_increment=later,timing_margin=now-later,windows=rows,
            excluded=['future_resource_gains','preparation_chains','established_schedule_score',
                      'future_card_consumption','second_nullified_card_value'])

    def describe_move(self, state, action):
        pending = state.pending_choice
        if (not pending or pending.action_kind is not ActionKind.MOVE_CARD
                or not pending.source_id.startswith('drink:')
                or state.resolution_stack or state.pending_after_card_ids):
            raise UnsupportedOpportunity('move_not_standalone_drink_source')
        self.check_drink(self.kernel.drinks[pending.source_id[6:]])
        moved, _ = self.kernel._move_card_to_hand(state, action.card_instance_id)
        self.synchronous_guard(moved, SchedulePhase.CARD_MOVED_TO_HAND,
                               self.kernel._moved_card_event_context(moved, action.card_instance_id))
        after = self.single(self.kernel.transition(state,action,plan_type=PlanType.SENSE),'move_completion')
        # Compare the same completed resources, excluding only this candidate
        # from the available scoring choices. No card vanishes from a zone.
        baseline = self.current_capacity(after, exclude=action.card_instance_id)
        capacity = self.current_capacity(after)
        return dict(supported=True,scope='current_single_card_direct_opportunity',
            baseline=baseline,capacity=capacity,timing_margin=capacity-baseline,
            stamina=after.stamina,genki=after.genki,nullify_cost_cards=after.nullify_cost_cards,
            double_cost_turns=after.double_cost_turns)

    @staticmethod
    def _rank_adjustments(old, rows):
        result = old.copy()
        if not len(old):
            return result
        # Rank whole (margin, old prior) groups, not a numeric epsilon which
        # could overtake a distinct margin. Equal margins preserve old order;
        # zero increments and every unrelated original value remain untouched.
        for positive in (False, True):
            entries = [(row['index'], row['timing_margin']) for row in rows
                       if row['supported'] and (row['timing_margin'] > 0 if positive
                                                else row['timing_margin'] < 0)]
            groups = sorted({(margin, float(old[index])) for index, margin in entries})
            positions = {value: rank for rank, value in enumerate(groups)}
            base = float(max(old)) + 1 if positive else float(min(old)) - 2
            for index, margin in entries:
                result[index] = base + positions[(margin, float(old[index]))] / (len(groups) + 1)
        return result

    def action_priors(self,state,actions):
        self.calls += 1
        started = time.perf_counter()
        self.last_audit = []
        try:
            check_search_budget()
            old = np.asarray(self.old.action_priors(state,actions),dtype=float)
            for index,action in enumerate(actions):
                check_search_budget()
                if action.kind not in {ActionKind.USE_DRINK,ActionKind.MOVE_CARD}:
                    continue
                action_started = time.perf_counter()
                try:
                    row = (self.describe(state,action) if action.kind is ActionKind.USE_DRINK
                           else self.describe_move(state,action))
                except UnsupportedOpportunity as error:
                    row = dict(supported=False,reason=str(error))
                self.last_audit.append(dict(index=index,kind=action.kind.value,drink_id=action.drink_id,
                    old_prior=float(old[index]),elapsed_ms=(time.perf_counter()-action_started)*1000,**row))
            result = self._rank_adjustments(old,self.last_audit)
            for row in self.last_audit:
                row['new_prior'] = float(result[row['index']])
            if any(row['supported'] for row in self.last_audit):
                self.active_calls += 1
            else:
                self.bypassed_calls += 1
            return tuple(float(value) for value in result)
        finally:
            self.elapsed_ms += (time.perf_counter()-started)*1000
