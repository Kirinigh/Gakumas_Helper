"""Shared decision and actual-transition boundary for one battle policy."""

from .contracts import ActionKind, GameAction, DecisionKind, DeckKnowledge


def network_policy(profile, kernel, context, initial, *, source, planning_seed, deadline, model=None,
                   opening_hand_order=None, use_observed_hand_order=False, model_path=None,
                   model_sha256=None, exam_effects=None):
    from functools import partial

    from .. import hif_gkms_runtime as runtime
    from .gkms_v2_effects import CatalogV2RuleSources, effect_sources
    from .gkms_v2_observation import build_v2_observation
    from .gkms_continuous_policy import GkmsContinuousPolicy

    observation_builder = None
    v2_sources = None
    options = {}
    if getattr(context, 'use_host_legality', False):
        from ..idol_road_runtime import TowerChoicePolicy

        options['secondary_factory'] = TowerChoicePolicy
    if profile == 'uinet-2.1':
        from ..idol_road_card_rules import TowerCardKernel

        if model is None:
            model = runtime.load_uinet21(model_path, model_sha256=(
                runtime.UINET21_SHA256 if model_sha256 is None else model_sha256),
                native_rules=isinstance(kernel, TowerCardKernel))
        observation_builder = (runtime.uinet21_observation_builder() if exam_effects is None
            else partial(build_v2_observation, exam_effects=exam_effects))
        tables = runtime.uinet21_master()
        v2_sources = (CatalogV2RuleSources(kernel, tables) if getattr(context, 'use_host_legality', False)
                      else effect_sources(tables))
    elif profile == 'uinet-1.1':
        if model is None:
            model = runtime.load_model()
    else:
        raise ValueError('decision_box_unknown_network_profile')
    return GkmsContinuousPolicy(kernel, model, context, initial, source=source,
        planning_seed=planning_seed, deadline=deadline, opening_hand_order=opening_hand_order,
        use_observed_hand_order=use_observed_hand_order, observation_builder=observation_builder,
        v2_sources=v2_sources, **options)


def build_battle_policy(algorithm, *, kernel, planning_seed, deadline, context=None, state=None,
                        source=None, opening_hand_order=None, use_observed_hand_order=False,
                        model=None, model_path=None, model_sha256=None, exam_effects=None):
    """Select the policy behind choose/observe, independently of the Maa route."""
    if algorithm == 'strong':
        from .gkms_continuous_policy import strong_policy

        return strong_policy(kernel, planning_seed=planning_seed, deadline=deadline)
    if algorithm == 'rust-gnrpa':
        from .tower_gnrpa import RustTowerPolicy

        if getattr(context, 'formation', None) is None or state is None or not source:
            raise ValueError('decision_box_rust_gnrpa_requires_tower_formation')
        return RustTowerPolicy(kernel, context, state, source=source)
    if algorithm not in ('uinet-1.1', 'uinet-2.1'):
        raise ValueError('decision_box_unknown_profile')
    if context is None or state is None or not source:
        raise ValueError('decision_box_network_requires_exam_and_play_history')
    return network_policy(algorithm, kernel, context, state, source=source,
        planning_seed=planning_seed, deadline=deadline, opening_hand_order=opening_hand_order,
        use_observed_hand_order=use_observed_hand_order, model=model, model_path=model_path,
        model_sha256=model_sha256, exam_effects=exam_effects)


def open_native_decision_box(algorithm, rules, opening, loadout, observed_types, *, item_uses,
                             statuses, exam=None, unknown_deck=0, model=None, rust_stage=None):
    """Share the native receipt and policy factory across cultivation and Tower."""
    from ..hif_gkms_runtime import load_model, load_uinet21

    if algorithm == 'uinet-1.1':
        model = load_model(native_rules=True) if model is None else model
    elif algorithm == 'uinet-2.1':
        model = load_uinet21(native_rules=True) if model is None else model
    elif algorithm != 'rust-gnrpa' or exam is not None:
        raise ValueError('decision_box_unknown_native_profile')
    session = rules.open(opening, loadout, model, observed_types, item_uses=item_uses, statuses=statuses)
    session.observation_version = 2 if algorithm == 'uinet-2.1' else 1
    session.unread_formation_cards = unknown_deck
    if exam is not None:
        session.exam = exam
    if algorithm == 'rust-gnrpa':
        from .tower_gnrpa import open_native_tower_search

        session.search = open_native_tower_search(rules, loadout, opening.state, rust_stage)
    return DecisionBox(session, opening), session


def open_decision_box(algorithm, *, kernel, opening, planning_seed, deadline, context=None, history=None,
                      model=None, model_path=None, model_sha256=None, exam_effects=None):
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
    policy = build_battle_policy(algorithm, kernel=kernel, context=context, state=opening.state,
        source=history.source if history is not None else None, planning_seed=planning_seed,
        deadline=deadline, opening_hand_order=order, use_observed_hand_order=True, model=model,
        model_path=model_path, model_sha256=model_sha256, exam_effects=exam_effects)
    if history is not None and algorithm in ('uinet-1.1', 'uinet-2.1'):
        policy.history = history
    return DecisionBox(policy, opening)


class DecisionBox:
    """One episode. Advance only on a completed step supplied by its host."""

    def __init__(self, policy, opening, *, pending_action=None):
        if not callable(getattr(policy, 'choose', None)) or not any(
            callable(getattr(policy, method, None)) for method in ('observe_transition', 'committed')
        ):
            raise ValueError('decision_box_policy_interface_incomplete')
        self._policy = policy
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

    def record_final_score(self, score):
        """Record the game's own settlement score after the last feedback."""
        if self._failed or self._pending is not None or not self._request.state.terminal:
            raise ValueError('decision_box_cannot_settle_before_terminal_feedback')
        if type(score) is not int or score < 0:
            raise ValueError('decision_box_invalid_displayed_score')
        if self._final_score is not None and self._final_score != score:
            raise ValueError('decision_box_displayed_score_already_recorded')
        self._final_score = score

    def finish_on_observed_result(self, before, action, score):
        """A game's ranking can close an action whose terminal state was not modeled."""
        if self._failed or self._final_score is not None or self._pending is None or (
                before, action) != (self._request.state, self._pending):
            raise ValueError('decision_box_result_does_not_match_last_action')
        if type(score) is not int or score < 0:
            raise ValueError('decision_box_invalid_displayed_score')
        self._pending = None
        self._final_score = score

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
        self._policy.reobserve(observed)
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
        observe = getattr(self._policy, 'observe_transition', None)
        if callable(observe):
            observe(before, action, step)
        else:
            self._policy.committed(before, action, after.state)
        self._request, self._pending, self._failed = after, None, False
