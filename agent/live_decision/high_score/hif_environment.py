"""Offline reset/step adapter over the production HIF rules, without Maa."""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import replace, dataclass

from .hif import HifSenseKernel, is_hif_turn_boundary
from .kernel import KernelError
from .contracts import (
    PlanType,
    ActionKind,
    GameAction,
    DecisionKind,
    DeckKnowledge,
    StrategyState,
    DecisionRequest,
    CapabilityReport,
)


class SampledHifKernel(HifSenseKernel):
    """Sample local probability mixtures; all effects remain in the same kernel.

    A mixture can carry a parent's probability rather than unit mass. Preserve
    that mass after sampling so nested triggers do not renormalize their parent.
    This class is offline-only and must never supply search probabilities.
    """

    def __init__(self, kernel, rng):
        super().__init__(kernel.cards, kernel.drinks, kernel.items,
                         memory_abilities=kernel.memory_abilities, support_definition=kernel.support_definition)
        self._rng = rng
        self._boundaries = []
        self.last_boundary = None

    def _resume_resolution_stack(self, branches):
        self._boundaries.extend(branch.state for branch in branches if is_hif_turn_boundary(branch.state))
        return super()._resume_resolution_stack(branches)

    def _independent_draw_batch(self, state, count, *, source_id):
        # Sequential draws allow sampling between draws; no hidden deck order
        # is created or made available to the policy.
        return None

    def _draw_candidates(self, deck):
        candidates = super()._draw_candidates(deck)
        if len(candidates) == 1:
            return candidates
        threshold = self._rng.random()
        for card, probability in candidates:
            threshold -= probability
            if threshold < 0:
                return ((card, 1.),)
        return ((candidates[-1][0], 1.),)

    def _merge_probability_branches(self, branches):
        if not branches:
            return ()
        mass = math.fsum(branch.probability for branch in branches)
        if mass <= 0 or not math.isfinite(mass):
            raise KernelError('sampled_transition_invalid_probability_mass')
        if len(branches) == 1:
            selected = branches[0]
        else:
            threshold = self._rng.random() * mass
            selected = branches[-1]
            for branch in branches:
                threshold -= branch.probability
                if threshold < 0:
                    selected = branch
                    break
        return (replace(selected, probability=mass,
                        state=self._canonical_probability_state(selected.state)),)

    def sample(self, state, action):
        self._boundaries = []
        self.last_boundary = None
        branches = self.transition(state, action, plan_type=PlanType.SENSE)
        if not branches or not math.isclose(math.fsum(b.probability for b in branches), 1., abs_tol=1e-9):
            raise KernelError('sampled_transition_incomplete_probability_mass')
        result = self._merge_probability_branches(branches)[0]
        unique = tuple(dict.fromkeys(self._boundaries))
        if len(unique) == 1:
            self.last_boundary = unique[0]
        return result


@dataclass(frozen=True)
class HifStep:
    request: DecisionRequest
    reward: float
    terminated: bool
    events: tuple
    boundary: StrategyState | None = None


class HifEnvironment:
    """One Sense round, undiscounted score deltas, and physical drink holes.

    The seed belongs only to the environment. reset accepts a verified initial
    state, including its effect history; it never invents a new opening binding.
    An observed opening is therefore labelled conditional-opening by callers.
    """

    def __init__(self, kernel: HifSenseKernel):
        self.kernel = kernel
        self.state = None
        self._sampler = None
        self._drink_slots = ()
        self._steps = 0

    def reset(self, state: StrategyState, *, seed: int, drink_slots=None, sampler_factory=None):
        state.validate()
        if state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION:
            raise ValueError('HIF training requires observed composition without hidden order')
        self.kernel._validate_registered_state(state, plan_type=PlanType.SENSE)
        slots = tuple(enumerate(state.drinks)) if drink_slots is None else tuple(drink_slots)
        if (len({slot for slot, _ in slots}) != len(slots)
                or any(type(slot) is not int or not 0 <= slot < 4 for slot, _ in slots)
                or Counter(drink for _, drink in slots) != Counter(state.drinks)):
            raise ValueError('HIF environment physical drink slots mismatch')
        self.state = self.kernel._canonical_probability_state(state)
        # A planning sampler validates the same catalogue at construction;
        # create it here instead of constructing a default sampler to discard.
        self._sampler = (SampledHifKernel(self.kernel, random.Random(seed))
                         if sampler_factory is None else sampler_factory(self.kernel, seed))
        self._drink_slots = slots
        self._steps = 0
        return self.request()

    def request(self):
        if self.state is None:
            raise ValueError('HIF environment is not reset')
        pending = self.state.pending_choice
        candidates = (pending.candidate_instance_ids if pending else
                      tuple(card.instance_id for card in self.state.deck.hand))
        return DecisionRequest(
            f'hif-offline-step-{self._steps}', PlanType.SENSE,
            DecisionKind.SECONDARY if pending else DecisionKind.LIVE,
            self.state, CapabilityReport(True, True, True, True, True),
            card_slots=tuple((instance_id, i) for i, instance_id in enumerate(candidates)),
            drink_slots=self._drink_slots, ruleset_id=self.kernel.ruleset_id,
        )

    def legal_actions(self):
        if self.state is None or self.state.terminal:
            return ()
        actions = self.kernel.legal_actions(self.state, plan_type=PlanType.SENSE,
                                            decision_kind=self.request().decision_kind)
        slots = dict(self.request().card_slots)
        result = []
        for action in actions:
            if action.kind is ActionKind.USE_DRINK:
                slot = next(slot for slot, drink in self._drink_slots if drink == action.drink_id)
            else:
                slot = slots.get(action.card_instance_id)
            result.append(replace(action, source_slot=slot))
        return tuple(result)

    def step(self, action: GameAction):
        if self.state is None or self.state.terminal:
            raise ValueError('HIF episode has no active state')
        legal = self.legal_actions()
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in self._drink_slots:
                raise ValueError('HIF drink action has no physical bottle')
        elif action not in legal:
            raise ValueError('HIF action physical slot does not match its legal candidate')
        if replace(action, source_slot=None) not in tuple(replace(a, source_slot=None) for a in legal):
            raise ValueError('HIF environment illegal action')
        if self._steps >= 512:
            raise ValueError('HIF episode action limit exceeded; episode is incomplete')
        before = self.state.score
        branch = self._sampler.sample(self.state, action)
        events = branch.events
        state = branch.state
        boundary = self._sampler.last_boundary
        if action.kind is ActionKind.USE_DRINK:
            self._drink_slots = tuple(pair for pair in self._drink_slots if pair[0] != action.source_slot)
        # The game advances automatically after the final card action, but a
        # pending secondary choice must be completed before that advancement.
        if not state.terminal and state.actions_remaining == 0 and not state.pending_choice and not state.resolution_stack:
            advanced = self._sampler.sample(state, GameAction(ActionKind.END_TURN))
            state, events = advanced.state, events + advanced.events
            boundary = self._sampler.last_boundary
        if Counter(drink for _, drink in self._drink_slots) != Counter(state.drinks):
            raise ValueError('HIF environment drink consumption mismatch')
        self.state = state
        self._steps += 1
        return HifStep(self.request(), float(state.score - before), state.terminal, events, boundary)
