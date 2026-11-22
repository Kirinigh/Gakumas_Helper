"""Stateful, offline strategy adapters; no external automation code executes."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .contracts import PlanType, ActionKind, GameAction, DecisionKind
from .hif_environment import SampledHifKernel


class HiroStructuredTeacher:
    """Transport Hiro's priority/combo/drink decisions onto verified actions.

    Exact identities replace image matching. Equal recognition confidence is
    resolved by physical hand order and is disclosed as an adapter convention.
    External drink IDs are mapped by catalog name, never by numeric equality.
    """

    def __init__(self, kernel, source: Path, version: str):
        self.kernel = kernel
        self.version = version
        self.config = json.loads((source / 'config/cards_priority.json').read_text(encoding='utf-8-sig'))
        self.drink_config = json.loads((source / 'config/drinks.json').read_text(encoding='utf-8-sig'))
        self.priority = {row['key']: row['priority'] for row in self.config['priority']}
        combo = self.config.get('combo', {})
        self.first, self.second = combo.get('first'), combo.get('second')
        self.skip_limit = self.config['skip_limit']
        by_name = {definition.name: key for key, definition in kernel.drinks.items()}
        self.drink_ids = {number: by_name[name] for name, number in self.drink_config['name_to_id'].items()
                          if name in by_name}
        self.reset()

    def reset(self):
        self.waiting = self.done = self.failed = False
        self.skips = self.max_turn_seen = 0
        self.pending_target = None

    def card_name(self, state, action):
        card = next((card for card in state.deck.all_cards if card.instance_id == action.card_instance_id), None)
        if card is None:
            return None
        definition = self.kernel.cards[card.card_id]
        canonical = self.kernel.cards.get(definition.canonical_card_id, definition)
        return canonical.name

    def choose(self, request, legal):
        state = request.state
        if not legal:
            raise ValueError('hiro_no_legal_action')
        if state.pending_choice:
            if self.pending_target:
                matches = [action for action in legal if self.card_name(state, action) == self.pending_target]
                if matches:
                    return matches[0]
            raise ValueError('hiro_secondary_choice_without_source_policy')
        self.max_turn_seen = max(self.max_turn_seen, state.turns_remaining)
        remaining = state.turns_remaining
        plans = self.drink_config['plans']
        plan = ('first_turn' if remaining == self.max_turn_seen else
                'second_turn' if remaining == self.max_turn_seen - 1 else
                'fifth_last' if remaining == 5 else 'third_last' if remaining == 3 else
                'last' if remaining == 1 else None)
        targets = {self.drink_ids[number] for number in plans.get(plan, ()) if number in self.drink_ids}
        bottles = [action for action in legal if action.kind is ActionKind.USE_DRINK and action.drink_id in targets]
        if bottles:
            return min(bottles, key=lambda action: action.source_slot)
        cards = [action for action in legal if action.kind is ActionKind.PLAY_CARD]
        names = {self.card_name(state, action) for action in cards}
        vinegar = [action for action in legal if action.kind is ActionKind.USE_DRINK
                   and action.drink_id == self.drink_ids.get(16)]
        if vinegar and self.first and self.second:
            desired = (self.second if self.waiting and self.second not in names else
                       self.first if self.second in names and self.first not in names and len(cards) < 5 else None)
            if desired:
                self.pending_target = desired
                return vinegar[0]
        if self.waiting:
            for name in (self.second, self.first):
                matches = [action for action in cards if self.card_name(state, action) == name]
                if matches:
                    return matches[0]
            return next(action for action in legal if action.kind is ActionKind.END_TURN)
        eligible = [action for action in cards if self.card_name(state, action) != self.second
                    or self.done or self.failed]
        if eligible:
            return min(eligible, key=lambda action: (
                self.priority.get(self.card_name(state, action), 11),
                self.card_name(state, action) not in self.priority, action.source_slot))
        return next(action for action in legal if action.kind is ActionKind.END_TURN)

    def committed(self, before, action, after):
        if action.kind is ActionKind.PLAY_CARD:
            name = self.card_name(before, action)
            if self.waiting and name == self.second:
                self.waiting, self.done, self.skips = False, True, 0
            elif not self.done and self.first and name == self.first:
                self.waiting, self.done, self.skips = True, False, 0
        elif action.kind is ActionKind.END_TURN and self.waiting:
            self.skips += 1
            if self.skips >= self.skip_limit:
                self.waiting, self.failed, self.skips = False, True, 0
        if after.pending_choice is None:
            self.pending_target = None


class RisHeuristicTeacher:
    """RIS 390494e common Sense weights, adapted to the shared HIF kernel.

    Three-level extra-action speculation uses independent simulation draws.
    This is an adapted teacher, not the external engine's benchmark result.
    Recommended-effect multipliers are explicit, never inferred from an idol.
    Unsupported external growth counters are outside this Sense adapter.
    """

    version = 'ris-390494e-hif-adapter-v1'

    def __init__(self, kernel, *, recommended_effect=None, seed=0):
        self.kernel = kernel
        self.recommended_effect = recommended_effect
        self.seed = seed
        self.reset()

    def reset(self):
        self.sampler = SampledHifKernel(self.kernel, random.Random(self.seed))
        self.used = 0

    def value(self, state, used):
        turns = state.turns_remaining
        if state.terminal:
            return float(state.score)
        good = 1.75 if self.recommended_effect == 'good_condition' else 1.
        focus = 3. if self.recommended_effect == 'concentration' else 1.
        weights = (len(state.deck.hand) * 3 + used * 8 + state.stamina * turns * .05
            + state.genki * math.tanh(turns / 3) * .7
            + min(state.good_condition_turns, turns) * 1.6 * good
            + min(state.excellent_condition_turns, turns) * state.good_condition_turns * good * 1.5
            + state.concentration * turns * focus
            + sum(buff.amount * (buff.remaining_turns or turns) for buff in state.score_buffs) * 8
            + min(state.half_cost_turns, turns) * 6 - min(state.double_cost_turns, turns) * 6
            + state.cost_reduction * turns * .5 + state.double_card_effects_remaining * 50
            + state.actions_remaining * 50)
        multipliers = (state.score_multiplier, *state.future_score_multipliers)
        return state.score + math.ceil(weights * sum(multipliers) / len(multipliers))

    def _evaluate(self, state, action, depth, used):
        branch = self.sampler.sample(state, action)
        child = branch.state
        used += int(action.kind in (ActionKind.PLAY_CARD, ActionKind.FREE_PLAY, ActionKind.PAID_PLAY))
        if not child.terminal and child.actions_remaining == 0 and not child.pending_choice and not child.resolution_stack:
            child = self.sampler.sample(child, GameAction(ActionKind.END_TURN)).state
        if depth < 3 and not child.terminal and child.turn == state.turn:
            legal = self.kernel.legal_actions(child, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
            return max((self._evaluate(child, candidate, depth + 1, used) for candidate in legal),
                       default=self.value(child, used))
        return self.value(child, used)

    def choose(self, request, legal):
        if not legal:
            raise ValueError('ris_no_legal_action')
        return max(legal, key=lambda action: self._evaluate(request.state, action, 1, self.used))

    def committed(self, before, action, after):
        self.used += int(action.kind in (ActionKind.PLAY_CARD, ActionKind.FREE_PLAY, ActionKind.PAID_PLAY))


class VisibleHifContinuationWrapper:
    """Offline development hypotheses around the unchanged frozen policy.

    Decisions inspect the visible hand and supplied legal actions only. The
    fixed thresholds are development assumptions, not validated game rules.
    The final-turn complete search and secondary choices remain the baseline's.
    """

    version = 'visible-hif-continuation-dev-v1'
    modes = ('early-301', 'early-35', 'late-35', 'early-301-late-35')
    minimum_301_turns_remaining = 5
    late_35_turns_remaining = 4

    def __init__(self, kernel, baseline, mode):
        if mode not in self.modes:
            raise ValueError('unknown visible HIF continuation hypothesis')
        self.kernel, self.baseline, self.mode = kernel, baseline, mode
        self.reset()

    def reset(self):
        self.interventions = []

    def _card_id(self, state, action):
        if action.kind is not ActionKind.PLAY_CARD:
            return None
        ref = next((card for card in state.deck.hand
                    if card.instance_id == action.card_instance_id), None)
        if ref is None:
            raise ValueError('visible continuation action has no hand instance')
        definition = self.kernel.cards[ref.card_id]
        return definition.canonical_card_id or definition.card_id

    def _baseline_action(self, request, legal):
        response = self.baseline.decide(request)
        if not response.has_action_intent:
            raise ValueError(response.reason)
        if response.action not in legal:
            raise ValueError('baseline continuation action is not legal')
        return response.action

    def _intervene(self, request, action, reason):
        self.interventions.append(dict(turn=request.state.turn,
            turns_remaining=request.state.turns_remaining, reason=reason,
            card_instance_id=action.card_instance_id, source_slot=action.source_slot))
        return action

    def choose(self, request, legal):
        state = request.state
        if state.pending_choice or state.resolution_stack or state.turns_remaining == 1:
            return self._baseline_action(request, legal)
        visible = {action: self._card_id(state, action) for action in legal}
        engines = [action for action in legal if visible[action] == '301']
        strikes = [action for action in legal if visible[action] == '35']
        if (self.mode in ('early-301', 'early-301-late-35') and engines
                and state.turns_remaining >= self.minimum_301_turns_remaining):
            return self._intervene(request, engines[0], 'legal_301_with_four_future_turns')
        if strikes and self.mode == 'early-35':
            return self._intervene(request, strikes[0], 'early_single_use_35_control')
        if strikes and self.mode in ('late-35', 'early-301-late-35'):
            if state.turns_remaining <= self.late_35_turns_remaining:
                return self._intervene(request, strikes[0], 'single_use_35_last_four_turns')
            alternatives = [action for action in legal
                            if action.kind is ActionKind.PLAY_CARD and visible[action] != '35']
            original = self._baseline_action(request, legal)
            if original in strikes and alternatives:
                priors = tuple(self.baseline.advice.action_priors(state, alternatives))
                if len(priors) != len(alternatives) or any(not math.isfinite(p) for p in priors):
                    raise ValueError('visible continuation alternative priors are invalid')
                selected = alternatives[max(range(len(alternatives)), key=priors.__getitem__)]
                return self._intervene(request, selected, 'defer_35_with_another_legal_card')
            return original
        return self._baseline_action(request, legal)

    def committed(self, before, action, after):
        # HybridHifPolicy has no per-episode mutable decision state.
        pass
