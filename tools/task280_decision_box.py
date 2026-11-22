"""Switch caller-supplied H.I.F. policies on one offline choose/feedback contract."""

from __future__ import annotations

import math
from functools import partial

from tools.task275.gkms_v2_observation import build_v2_observation
from agent.live_decision.high_score.contracts import PlanType, ActionKind, GameAction, DecisionKind, DeckKnowledge
from agent.live_decision.high_score.gkms_continuous_policy import GkmsContinuousPolicy


class DecisionBox:
    """One episode. Advance only on a completed step supplied by its host."""

    def __init__(self, policy, opening):
        if not callable(getattr(policy, 'choose', None)) or not any(
            callable(getattr(policy, method, None)) for method in ('observe_transition', 'committed')
        ):
            raise ValueError('decision_box_policy_interface_incomplete')
        self._policy = policy
        self._request = opening
        self._pending = None
        self._failed = False

    def choose(self, request, legal):
        if self._failed or self._pending is not None:
            raise ValueError('decision_box_previous_step_not_confirmed')
        if request != self._request or request.state.terminal:
            raise ValueError('decision_box_snapshot_not_ready')
        legal = tuple(legal)
        if not legal:
            raise ValueError('decision_box_legal_actions_missing')
        self._failed = True
        action = self._policy.choose(request, legal)
        if not isinstance(action, GameAction):
            raise ValueError('decision_box_invalid_action')
        action.validate()
        if action.kind is ActionKind.USE_DRINK:
            matched = (
                (action.source_slot, action.drink_id) in request.drink_slots
                and any(candidate.kind is ActionKind.USE_DRINK and candidate.drink_id == action.drink_id
                        for candidate in legal)
            )
        else:
            matched = action in legal
        if not matched:
            raise ValueError('decision_box_action_outside_legal_set')
        self._pending = action
        self._failed = False
        return action

    def observe_transition(self, before, action, step):
        if self._failed or self._pending is None or (before, action) != (self._request.state, self._pending):
            raise ValueError('decision_box_feedback_does_not_match_intent')
        after = step.request
        after.validate()
        if (after.snapshot_id == self._request.snapshot_id or after.ruleset_id != self._request.ruleset_id
                or after.plan_type is not self._request.plan_type or after.state.terminal != step.terminated
                or step.reward != after.state.score - before.score):
            raise ValueError('decision_box_feedback_inconsistent')
        # A policy may already have advanced part of its internal history if its
        # feedback hook throws. Reconstruct a new session from verified history.
        self._failed = True
        observe = getattr(self._policy, 'observe_transition', None)
        if callable(observe):
            observe(before, action, step)
        else:
            self._policy.committed(before, action, after.state)
        self._request, self._pending, self._failed = after, None, False


def open_decision_box(profile, *, kernel, opening, planning_seed, deadline, context=None, history=None,
                      model=None, strong_factory=None, secondary_factory=None, exam_effects=None):
    """Use a supplied model or strong chooser factory, with no default model loading.

    Network profiles require an explicit examination context and bound history.
    Their optional secondary factory supplies local selections when needed.
    """
    opening.validate()
    if (opening.plan_type is not PlanType.SENSE or opening.ruleset_id != kernel.ruleset_id
            or opening.decision_kind is not DecisionKind.LIVE
            or opening.state.terminal or not opening.capabilities.all_ready
            or opening.state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
            or opening.state.deck.known_top_count):
        raise ValueError('decision_box_unsupported_opening')
    if not math.isfinite(deadline):
        raise ValueError('decision_box_invalid_deadline')
    if profile != 'uinet-2.1' and exam_effects is not None:
        raise ValueError('decision_box_v2_options_on_other_profile')
    if profile == 'strong':
        if not callable(strong_factory) or model is not None or secondary_factory is not None:
            raise ValueError('decision_box_explicit_strong_factory_required')
        policy = strong_factory(kernel, planning_seed=planning_seed, deadline=deadline)
    elif profile in {'uinet-1.1', 'uinet-2.1'}:
        if not callable(getattr(model, 'decide', None)) or strong_factory is not None:
            raise ValueError('decision_box_explicit_network_model_required')
        if secondary_factory is not None and not callable(secondary_factory):
            raise ValueError('decision_box_secondary_factory_invalid')
        if context is None or context.mappings is None or history is None:
            raise ValueError('decision_box_network_requires_exam_and_play_history')
        history.validate(opening.state)
        builder = partial(build_v2_observation, exam_effects=exam_effects) if profile == 'uinet-2.1' else None
        hand_order = tuple(iid for iid, _ in sorted(opening.card_slots, key=lambda pair: pair[1]))
        policy = GkmsContinuousPolicy(kernel, model, context, opening.state, source=history.source,
            planning_seed=planning_seed, deadline=deadline, opening_hand_order=hand_order,
            secondary_factory=secondary_factory, observation_builder=builder)
        policy.history = history
    else:
        raise ValueError('decision_box_unknown_profile')
    return DecisionBox(policy, opening)
