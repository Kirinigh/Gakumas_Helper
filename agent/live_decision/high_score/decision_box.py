"""Shared decision and actual-transition boundary for one battle policy."""

from . import policy_factory
from .contracts import ActionKind, GameAction, DecisionKind, DeckKnowledge, BattleStrategy


def open_native_decision_box(algorithm, rules, opening, loadout, observed_types, *, item_uses,
                             statuses, status_slots=(), exam=None, unknown_deck=0, model=None, rust_stage=None,
                             resumed=None):
    """Share the native receipt and policy factory across cultivation and Tower."""
    policy = policy_factory.build_battle_policy(algorithm, backend='native', model=model, exam=exam)
    observed = {'statuses': statuses}
    if status_slots:
        observed['status_slots'] = status_slots
    if resumed is None:
        session = rules.open(opening, loadout, policy.model, observed_types, item_uses=item_uses, **observed)
    else:
        saved, frame = resumed
        opening, session = rules.restore(saved, frame, loadout, policy.model, observed_types)
        unknown_deck = saved['unknown_deck']
    if policy.observation_version is not None:
        session.observation_version = policy.observation_version
    session.unread_formation_cards = unknown_deck
    if exam is not None:
        session.exam = exam
    if policy.open_search is not None:
        session.search = policy.open_search(rules, loadout, opening.state, rust_stage)
    if policy.open_strategy is None:
        return DecisionBox(session, opening), session
    return DecisionBox(policy.open_strategy(opening), opening, rules=session), session


def open_decision_box(algorithm, *, kernel, opening, planning_seed, deadline, context=None, history=None,
                      model=None, model_path=None, model_sha256=None, exam_effects=None,
                      use_observed_hand_order=True):
    """Open a battle from the host's observed opening and physical slot bindings."""
    opening.validate()
    if (opening.ruleset_id != kernel.ruleset_id or opening.decision_kind is not DecisionKind.LIVE
            or opening.state.terminal or not opening.capabilities.all_ready
            or opening.state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION):
        raise ValueError('decision_box_unsupported_opening')
    if algorithm != 'uinet-2.1' and (model_path is not None or model_sha256 is not None or exam_effects is not None):
        raise ValueError('decision_box_v2_options_on_other_profile')
    if history is not None:
        history.validate(opening.state)
    order = tuple(iid for iid, _ in sorted(opening.card_slots, key=lambda pair: pair[1]))
    policy = policy_factory.build_battle_policy(algorithm, kernel=kernel, context=context, state=opening.state,
        source=history.source if history is not None else None, planning_seed=planning_seed,
        deadline=deadline, opening_hand_order=order, use_observed_hand_order=use_observed_hand_order, model=model,
        history=history, opening=opening,
        model_path=model_path, model_sha256=model_sha256, exam_effects=exam_effects)
    return DecisionBox(policy, opening)


class DecisionBox:
    """One episode. Advance only on a completed step supplied by its host."""

    def __init__(self, policy: BattleStrategy, opening, *, pending_action=None, rules=None):
        if not callable(getattr(policy, 'choose', None)):
            raise ValueError('decision_box_policy_interface_incomplete')
        self._policy = policy
        self._rules = rules
        self._request = opening
        self._pending = pending_action
        self._failed = False
        self._final_score = None

    @property
    def request(self):
        return self._request

    @property
    def pending_action(self):
        return self._pending

    @property
    def final_score(self):
        return self._final_score

    def record_final_score(self, score, *, observed_result=False):
        """Record the game's own settlement score after the last feedback."""
        if self._failed or self._pending is not None or not (observed_result or self._request.state.terminal):
            raise ValueError('decision_box_cannot_settle_before_terminal_feedback')
        if type(score) is not int or score < 0:
            raise ValueError('decision_box_invalid_displayed_score')
        if self._final_score is not None and self._final_score != score:
            raise ValueError('decision_box_displayed_score_already_recorded')
        if self._final_score is None:
            self._settle(score)

    def finish_on_observed_result(self, before, action, score):
        """A game's ranking can close an action whose terminal state was not modeled."""
        if self._failed or self._final_score is not None or self._pending is None or (
                before, action) != (self._request.state, self._pending):
            raise ValueError('decision_box_result_does_not_match_last_action')
        if type(score) is not int or score < 0:
            raise ValueError('decision_box_invalid_displayed_score')
        self._pending = None
        self._settle(score)

    def _settle(self, score):
        self._final_score = score
        if callable(settle := getattr(self._policy, 'on_final_score', None)):
            settle(score)

    def choose(self, request, legal):
        if self._failed or self._pending is not None or self._final_score is not None:
            raise ValueError('decision_box_previous_step_not_confirmed')
        if request != self._request or request.state.terminal:
            raise ValueError('decision_box_snapshot_not_ready')
        legal = tuple(legal)
        if not legal:
            raise ValueError('decision_box_legal_actions_missing')
        self._failed = True
        action = self._policy.choose(request, legal)
        self._bind(request, action, legal)
        self._failed = False
        return action

    def bind_host_action(self, request, action, legal):
        """Record a legal recovery action selected by the execution loop."""
        if (self._failed or self._pending is not None or self._final_score is not None
                or request != self._request or request.state.terminal):
            raise ValueError('decision_box_snapshot_not_ready')
        self._bind(request, action, legal)

    def discard_unsubmitted(self, request, action):
        if self._failed or (request, action) != (self._request, self._pending) or self._pending is None:
            raise ValueError('decision_box_feedback_does_not_match_intent')
        if callable(discard := getattr(self._policy, 'discard_unsubmitted', None)):
            discard(request, action)
        self._pending = None

    def _bind(self, request, action, legal):
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

    def reobserve_unsubmitted(self, before, action, observed):
        """Replace a superseded position without adding a played-action record."""
        if self._failed or (before, action) != (self._request, self._pending) or self._pending is None:
            raise ValueError('decision_box_reobservation_does_not_match_intent')
        self._reobserve(before, observed)

    def reobserve_idle(self, observed):
        """Use the current position if the game progressed between decisions."""
        if self._failed or self._pending is not None or self._final_score is not None:
            raise ValueError('decision_box_snapshot_not_ready')
        self._reobserve(self._request, observed)

    def _reobserve(self, before, observed):
        observed.validate()
        if (observed.snapshot_id == before.snapshot_id or observed.ruleset_id != before.ruleset_id
                or observed.plan_type is not before.plan_type or observed.state.terminal):
            raise ValueError('decision_box_reobservation_inconsistent')
        self._failed = True
        if self._rules is not None:
            self._rules.reobserve(observed)
        if callable(reobserve := getattr(self._policy, 'reobserve', None)):
            reobserve(observed)
        self._request, self._pending, self._failed = observed, None, False

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
        if self._rules is not None:
            self._rules.observe_transition(before, action, step)
        if callable(observe := getattr(self._policy, 'observe_transition', None)):
            observe(before, action, step)
        elif callable(committed := getattr(self._policy, 'committed', None)):
            committed(before, action, after.state)
        self._request, self._pending, self._failed = after, None, False
