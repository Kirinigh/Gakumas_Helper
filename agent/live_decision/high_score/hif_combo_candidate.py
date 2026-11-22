"""Explicit offline combo challenger; no default loading or execution authority."""
from __future__ import annotations

import math
import time
import random
from dataclasses import asdict, replace

from .contracts import PlanType, ActionKind, GameAction, DecisionKind
from .hif_environment import HifEnvironment

VERSION = 'hif-visible-combo-rollout-trial-v3-prefix-legality'
SETUP_IDS = frozenset(('162', '163'))
PARTNER_IDS = frozenset(('300', '301', '761', '762'))


def semantic_card_id(kernel, ref):
    """Custom definitions retain the catalog base identity, not a title guess."""
    definition = kernel.cards[ref.card_id]
    return str(definition.base_card_id or definition.canonical_card_id or definition.card_id)


def _key(action):
    return replace(action, source_slot=None) if action.kind is not ActionKind.USE_DRINK else action


def _resolve(action, legal):
    matches = [candidate for candidate in legal if _key(candidate) == _key(action)]
    if len(matches) != 1:
        raise ValueError('combo_prefix_action_not_uniquely_legal')
    return matches[0]


class ComboContinuation:
    """One fixed prefix followed by D, with the explicit late-12 hypothesis."""

    def __init__(self, baseline, prefix=(), *, hold_12=True, deadline=math.inf):
        self.baseline, self.prefix = baseline, tuple(prefix)
        self.hold_12, self.deadline = hold_12, deadline
        self.index = 0
        self.prefix_turn = None

    def choose(self, request, legal):
        if time.perf_counter() >= self.deadline:
            raise ValueError('combo_rollout_budget_exhausted')
        if self.index < len(self.prefix):
            if self.prefix_turn is not None and request.state.turn != self.prefix_turn:
                raise ValueError('combo_prefix_crossed_turn')
            return _resolve(self.prefix[self.index], legal)
        response = self.baseline.decide(request)
        if not response.has_action_intent:
            raise ValueError(response.reason)
        action = _resolve(response.action, legal)
        if (self.hold_12 and request.state.turns_remaining > 4
                and action.kind is ActionKind.USE_DRINK and action.drink_id == '12'):
            alternatives = [a for a in legal if not (
                a.kind is ActionKind.USE_DRINK and a.drink_id == '12')]
            advice = getattr(self.baseline, 'advice', None)
            if advice is None:
                advice = self.baseline.baseline.advice
            priors = tuple(advice.action_priors(request.state, alternatives))
            if not alternatives or len(priors) != len(alternatives) or not all(map(math.isfinite, priors)):
                raise ValueError('combo_defer_12_invalid_alternatives')
            action = alternatives[max(range(len(alternatives)), key=priors.__getitem__)]
        return action

    def committed(self, before, action, after):
        if self.index < len(self.prefix):
            if _key(action) != _key(self.prefix[self.index]):
                raise ValueError('combo_prefix_commit_mismatch')
            self.prefix_turn = before.turn if self.prefix_turn is None else self.prefix_turn
            self.index += 1


class HifComboCandidate:
    """Two common planning draws at selected windows, never a model retraining.

    Scores are complete sampled continuations, not exact search bounds. The
    caller owns the overall deadline and a planning seed independent of the
    environment. Complete estimated ties retain an equally ranked D action;
    otherwise a unique frozen prior breaks the tie. Remaining ties stop.
    """

    version = VERSION

    def __init__(self, kernel, baseline, *, planning_seed, deadline, window_seconds=8., simulator=None):
        if (type(planning_seed) is not int or not math.isfinite(deadline)
                or not math.isfinite(window_seconds) or window_seconds <= 0):
            raise ValueError('combo_candidate_requires_bounded_independent_planning')
        self.kernel, self.baseline = kernel, baseline
        self.planning_seed, self.deadline = planning_seed, deadline
        self.window_seconds, self.simulator = window_seconds, simulator
        self.reset()

    def reset(self):
        self._rng = random.Random(self.planning_seed)
        self.plans = []

    def _roles(self, state, refs):
        return ([c for c in refs if semantic_card_id(self.kernel, c) in SETUP_IDS],
                [c for c in refs if semantic_card_id(self.kernel, c) in PARTNER_IDS])

    def _deterministic_prefix(self, state, sequence, *, observable_hand_limit=None):
        """Only validate visible deterministic same-turn prefixes, never draw RNG."""
        turn = state.turn
        for action in sequence:
            if (state.turn != turn or state.terminal or (observable_hand_limit is not None
                    and len(state.deck.hand) > observable_hand_limit)):
                return False
            kind = DecisionKind.SECONDARY if state.pending_choice else DecisionKind.LIVE
            legal = self.kernel.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=kind)
            raw = replace(action, source_slot=None)
            if raw not in tuple(replace(a, source_slot=None) for a in legal):
                return False
            branches = self.kernel.transition(state, raw, plan_type=PlanType.SENSE)
            if len(branches) != 1 or not math.isclose(branches[0].probability, 1.):
                return False
            state = branches[0].state
            # An adapter's supported observation range is narrower than game
            # legality. Never change the kernel's move-to-hand semantics here.
            if observable_hand_limit is not None and len(state.deck.hand) > observable_hand_limit:
                return False
        return True

    def sequences(self, request, legal, *, observable_hand_limit=None):
        """Retain every legal root; add feasible setup/retrieval pair prefixes."""
        state = request.state
        sequences = [(action,) for action in legal]
        hand = state.deck.hand

        def pairs(prefix, available):
            setups, partners = self._roles(state, available)
            if state.double_card_effects_remaining:
                for partner in partners:
                    sequences.append((*prefix, GameAction(ActionKind.PLAY_CARD, partner.instance_id)))
            for setup in setups:
                for partner in partners:
                    sequences.append((*prefix, GameAction(ActionKind.PLAY_CARD, setup.instance_id),
                                      GameAction(ActionKind.PLAY_CARD, partner.instance_id)))

        pairs((), hand)
        if state.pending_choice and state.pending_choice.source_id == 'drink:22':
            by_id = {c.instance_id: c for c in state.deck.all_cards}
            for action in legal:
                if action.kind is ActionKind.MOVE_CARD:
                    pairs((action,), (*hand, by_id[action.card_instance_id]))
        elif not state.pending_choice:
            for drink in legal:
                if drink.kind is ActionKind.USE_DRINK and drink.drink_id == '22':
                    for card in (*state.deck.draw_pile, *state.deck.discard):
                        pairs((drink, GameAction(ActionKind.MOVE_CARD, card.instance_id)), (*hand, card))
        unique = []
        for sequence in sequences:
            # Armed partners can be generated even when their typed cost is
            # unaffordable. Those are absent candidates, not ambiguous intent.
            if not any(_key(sequence[0]) == _key(action) for action in legal):
                continue
            # Root slots are physical; subsequent slots are rebound after each step.
            if len(sequence) > 1 and not self._deterministic_prefix(
                    state, sequence, observable_hand_limit=observable_hand_limit):
                continue
            sequence = (_resolve(sequence[0], legal), *sequence[1:])
            if sequence not in unique:
                unique.append(sequence)
        return tuple(unique)

    def _trigger(self, request, legal, original):
        state = request.state
        setups, partners = self._roles(state, state.deck.hand)
        if partners and (setups or state.double_card_effects_remaining):
            return 'visible_combo_or_armed_partner'
        pending = state.pending_choice
        if pending and pending.source_id == 'drink:22':
            return 'black_vinegar_choice'
        drinks = {a.drink_id for a in legal if a.kind is ActionKind.USE_DRINK}
        pool_setup, pool_partner = self._roles(state, (*state.deck.draw_pile, *state.deck.discard))
        if '22' in drinks and ((setups and pool_partner) or (partners and pool_setup)
                               or (state.double_card_effects_remaining and pool_partner)):
            return 'black_vinegar_completes_pair'
        if '20' in drinks and state.turns_remaining >= 5:
            return 'early_20_window'
        if '12' in drinks and (state.turns_remaining <= 4 or (
                original.kind is ActionKind.USE_DRINK and original.drink_id == '12')):
            return '12_now_or_delayed_window'
        return None

    def choose(self, request, legal):
        if time.perf_counter() >= self.deadline:
            raise ValueError('combo_evaluation_budget_exhausted')
        request.validate()
        if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
            raise ValueError('combo_capability_or_ruleset_not_ready')
        response = self.baseline.decide(request)
        if not response.has_action_intent:
            raise ValueError(response.reason)
        original = _resolve(response.action, legal)
        # Keep the exact final-turn contract; prior ranking never replaces it.
        if request.state.turns_remaining == 1:
            return original
        trigger = self._trigger(request, legal, original)
        if trigger is None:
            return original
        if self.simulator is None:
            from tools.hif_chain_supervision import simulate_root
            simulator = simulate_root
        else:
            simulator = self.simulator
        sequences = self.sequences(request, legal)
        seeds = tuple(self._rng.sample(range(2**31), 2))
        deadline = min(self.deadline, time.perf_counter() + self.window_seconds)
        plan = dict(trigger=trigger, seeds=seeds, rows=[], accepted=False,
                    original_action=asdict(original), sequences=[[asdict(a) for a in seq] for seq in sequences],
                    scope='OFFLINE_COMBO_SAMPLED_ESTIMATE', execution_authorized=False,
                    optimality_proven=False, production_strategy_changed=False)
        self.plans.append(plan)
        for seed in seeds:
            for index, sequence in enumerate(sequences):
                if time.perf_counter() >= deadline:
                    raise ValueError('combo_planning_incomplete')
                continuation = ComboContinuation(self.baseline, sequence, deadline=deadline)
                row = simulator(self.kernel, request.state, request.drink_slots, sequence[0],
                    seed=seed, policy_name='visible-d-late12', continuation=continuation)
                row['sequence_index'] = index
                plan['rows'].append(row)
                if (not row['complete'] or row['stopped'] or row['illegal']
                        or row['root_dispatch_count'] != 1 or continuation.index != len(sequence)):
                    raise ValueError('combo_rollout_incomplete')
                if time.perf_counter() >= deadline:
                    raise ValueError('combo_planning_incomplete')
        scores = [sum(row['final_score'] for row in plan['rows']
                      if row['sequence_index'] == i) / len(seeds) for i in range(len(sequences))]
        best = max(scores)
        leaders = [action for action in legal if any(sequences[i][0] == action and value == best
                   for i, value in enumerate(scores))]
        plan['estimated_sequence_scores'] = scores
        tie_rule = 'unique_sample_mean'
        if original in leaders:
            action, tie_rule = original, 'no_strict_improvement_keep_D'
        elif len(leaders) == 1:
            action = leaders[0]
        else:
            advice = getattr(self.baseline, 'advice', None)
            if advice is None:
                advice = self.baseline.baseline.advice
            priors = tuple(advice.action_priors(request.state, leaders))
            if len(priors) != len(leaders) or not all(map(math.isfinite, priors)):
                raise ValueError('combo_tie_priors_invalid')
            plan['tied_leader_priors'] = [dict(action=asdict(a), prior=p) for a, p in zip(leaders, priors)]
            best_prior = max(priors)
            winners = [a for a, p in zip(leaders, priors) if p == best_prior]
            if len(winners) != 1:
                raise ValueError('combo_first_action_tied')
            action, tie_rule = winners[0], 'unique_frozen_prior_among_tied_leaders'
        plan.update(accepted=True, selected_action=asdict(action), intervention=action != original,
                    decision_tie_rule=tie_rule)
        return action

    def committed(self, before, action, after):
        # Armed doubles and pending retrieval are re-evaluated from the fresh
        # complete state; the next step never blindly falls back to D.
        pass


def physical_legal_actions(kernel, request):
    """Offline evaluation keeps the supplied four bottle positions."""
    environment = HifEnvironment(kernel)
    environment.reset(request.state, seed=0, drink_slots=request.drink_slots)
    return environment.legal_actions()
