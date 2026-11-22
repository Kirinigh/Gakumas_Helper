"""Explicit Sense timing hypothesis using legal prefixes, without rollout search."""
from __future__ import annotations

import math
from dataclasses import asdict, replace

from .contracts import PlanType, ActionKind
from .hif_combo_candidate import (
    SETUP_IDS,
    PARTNER_IDS,
    HifComboCandidate,
    _resolve,
    semantic_card_id,
)

VERSION = 'sense-combo-timing-trial-v2'


class SenseComboTimingPolicy:
    """Local trial only: T1 macchiato, late boost, and a verified complete pair.

    The old candidate is reused solely as a deterministic prefix builder. Its
    stochastic choose/rollout method is never called. Unknown or ambiguous
    retrieval targets cannot acquire a commitment through this adapter.
    """

    version = VERSION

    def __init__(self, kernel, baseline):
        self.kernel, self.baseline = kernel, baseline
        self._builder = HifComboCandidate(kernel, baseline, planning_seed=0, deadline=0.)
        self.reset()

    def reset(self):
        self._armed_by_setup = False
        self.interventions = []
        self.completed_combos = []

    def _record(self, request, action, reason, prefix=()):
        self.interventions.append(dict(turn=request.state.turn, reason=reason,
                                       action=asdict(action), prefix=[asdict(a) for a in prefix]))
        return action

    def _commit_prefix(self, request, sequence):
        return self._record(request, sequence[0], 'verified_same_turn_combo_prefix', sequence)

    def _combo_prefixes(self, request, legal):
        state = request.state
        prefixes = self._builder.sequences(request, legal,
            observable_hand_limit=min(5, state.hand_limit))
        by_id = {card.instance_id: card for card in state.deck.all_cards}
        result = []
        for sequence in prefixes:
            plays = [action for action in sequence if action.kind is ActionKind.PLAY_CARD]
            if not plays:
                continue
            roles = [semantic_card_id(self.kernel, by_id[action.card_instance_id]) for action in plays]
            valid = (len(roles) == 1 and roles[0] in PARTNER_IDS if state.double_card_effects_remaining else
                     len(roles) == 2 and roles[0] in SETUP_IDS and roles[1] in PARTNER_IDS)
            if valid:
                result.append(sequence)
        return result

    def _coverage_action(self, request, legal, prefixes):
        """Use the observed 165 effect only when it covers the remaining stage.

        Check the actual deterministic transition rather than assigning fixed
        costs, extra actions, or duration. Preserve any existing complete pair.
        """
        state = request.state
        if (state.pending_choice or state.double_card_effects_remaining
                or state.excellent_condition_turns >= state.turns_remaining):
            return None
        by_id = {card.instance_id: card for card in state.deck.hand}
        candidates = []
        for action in legal:
            if action.kind is not ActionKind.PLAY_CARD:
                continue
            definition = self.kernel.cards[by_id[action.card_instance_id].card_id]
            # 165 is the upgraded card. Its base_card_id is 164, so a base-role
            # match would incorrectly include a different concrete effect.
            if (definition.canonical_card_id or definition.card_id) == '165':
                candidates.append(action)
        if len(candidates) != 1:
            return None
        action = candidates[0]
        branches = self.kernel.transition(state, replace(action, source_slot=None), plan_type=PlanType.SENSE)
        if len(branches) != 1 or not math.isclose(branches[0].probability, 1.):
            return None
        after = branches[0].state
        if (after.turn != state.turn or after.turns_remaining != state.turns_remaining
                or after.terminal or after.pending_choice or after.actions_remaining <= 0
                or after.actions_remaining < state.actions_remaining
                or after.double_card_effects_remaining != state.double_card_effects_remaining
                or after.excellent_condition_turns < state.turns_remaining):
            return None
        if any(not self._builder._deterministic_prefix(after, prefix,
                observable_hand_limit=min(5, state.hand_limit)) for prefix in prefixes):
            return None
        return action

    def choose(self, request, legal, *, excluded_actions=()):
        request.validate()
        if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
            raise ValueError('combo_timing_capability_or_ruleset_not_ready')
        state = request.state
        if state.pending_choice and len(state.deck.hand) > min(5, state.hand_limit):
            raise ValueError('combo_timing_pending_outside_observable_hand_capacity')
        # These windows do not weaken the existing exact final-turn search.
        if state.turns_remaining == 1:
            response = self.baseline.decide(request)
            if not response.has_action_intent:
                raise ValueError(response.reason)
            return _resolve(response.action, legal)
        early_20 = [action for action in legal if action.kind is ActionKind.USE_DRINK
                    and action.drink_id == '20' and state.turn == 1]
        if early_20:
            return self._record(request, early_20[0], 'legal_20_first_turn')
        prefixes = self._combo_prefixes(request, legal)
        coverage = self._coverage_action(request, legal, prefixes)
        if coverage is not None:
            return self._record(request, coverage, 'deterministic_165_covers_remaining_turns')
        if prefixes:
            # Prefer an already complete hand over spending a retrieval bottle.
            ordinary = [seq for seq in prefixes if seq[0].kind is ActionKind.PLAY_CARD]
            candidates = ordinary or prefixes
            if len(candidates) != 1:
                raise ValueError('combo_timing_pair_or_target_not_unique')
            return self._commit_prefix(request, candidates[0])
        if state.pending_choice and state.pending_choice.source_id == 'drink:22':
            raise ValueError('combo_timing_retrieval_has_no_verified_pair')
        late_12 = [action for action in legal if action.kind is ActionKind.USE_DRINK
                   and action.drink_id == '12' and state.turns_remaining == 4]
        if late_12:
            return self._record(request, late_12[0], 'legal_12_last_four_turns')
        response = self.baseline.decide(request)
        if not response.has_action_intent:
            raise ValueError(response.reason)
        original = _resolve(response.action, legal)
        allowed = [action for action in legal if not (action.kind is ActionKind.USE_DRINK and (
            action.drink_id == '22' or action.drink_id == '12' and state.turns_remaining > 4))]
        allowed = [action for action in allowed if action not in excluded_actions]
        if original in allowed:
            return original
        advice = getattr(self.baseline, 'advice', None)
        if advice is None:
            advice = self.baseline.baseline.advice
        priors = tuple(advice.action_priors(state, allowed))
        if not allowed or len(priors) != len(allowed) or not all(map(math.isfinite, priors)):
            raise ValueError('combo_timing_masked_D_priors_invalid')
        # This is the existing D masked-prior ordering, including stable order
        # for identical descriptors; it is not a sampled-score tie resolution.
        selected = allowed[max(range(len(allowed)), key=priors.__getitem__)]
        return self._record(request, selected, 'D_with_early_12_and_noncombo_22_masked')

    def committed(self, before, action, after):
        if action.kind is ActionKind.PLAY_CARD:
            ref = next(card for card in before.deck.all_cards if card.instance_id == action.card_instance_id)
            role = semantic_card_id(self.kernel, ref)
            if self._armed_by_setup and before.double_card_effects_remaining and role in PARTNER_IDS:
                self.completed_combos.append(dict(turn=before.turn, partner=ref.card_id,
                    partner_instance_id=ref.instance_id, before_score=before.score, after_score=after.score))
            self._armed_by_setup = role in SETUP_IDS and bool(after.double_card_effects_remaining)
