"""Retain choices already completed by the exact solver, without another search."""
from contextlib import contextmanager
from dataclasses import asdict, replace

from .hif_mcts import action_key
from .contracts import ActionKind
from .hif_mcts_cache import json_value, exact_state, action_from_json
from .hif_information_boundary import classify_information_boundary


def state_key(kernel, state):
    return kernel._canonical_probability_state(replace(state, score=0))


@contextmanager
def completed_choices(candidate, request, enabled):
    """Observe existing solver selections only; do not extend its budget."""
    choices, previous = {}, []
    if enabled and request.state.turns_remaining <= 2:
        def remember(state, action, source):
            if (state.turn == request.state.turn and not state.pending_choice
                    and source.utility == request.utility and source.plan_type == request.plan_type):
                choices[state_key(candidate.kernel, state)] = replace(action, source_slot=None)
        for planner in (candidate.endgame, candidate.tail):
            if planner is not None:
                previous.append((planner, getattr(planner, '_completed_action_sink', None)))
                planner._completed_action_sink = remember
    try:
        yield choices
    finally:
        for planner, sink in previous:
            planner._completed_action_sink = sink


def extract_segment(kernel, request, first_action, choices):
    rows, state, action = [], request.state, first_action
    for _ in range(64):
        boundary = classify_information_boundary(kernel, state, action, plan_type=request.plan_type)
        if boundary.kind != 'deterministic_same_turn':
            return rows
        state = boundary.successor
        action = choices.get(state_key(kernel, state))
        if action is None:
            return rows
        rows.append(dict(state=json_value(asdict(state_key(kernel, state))), action=json_value(asdict(action))))
    raise ValueError('exact_segment_exceeds_action_bound')


def lookup(kernel, request, legal, rows):
    key = state_key(kernel, request.state)
    for row in rows:
        if exact_state(row['state']) == key:
            saved = action_from_json(row['action'])
            # Physical slots are rebound by the existing native input path.
            matches = [action for action in legal if replace(action, source_slot=None) == saved]
            if len(matches) != 1:
                raise ValueError('exact_segment_action_not_unique')
            return matches[0]
    return None


def validate(kernel, rows):
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('exact_segment_invalid')
    from .contracts import PlanType, DecisionKind
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'state', 'action'}:
            raise ValueError('exact_segment_invalid')
        state, action = exact_state(row['state']), action_from_json(row['action'])
        if (state.score != 0 or state.turns_remaining > 2 or state in seen
                or state.terminal or action.source_slot is not None
                or action.kind not in (ActionKind.PLAY_CARD, ActionKind.USE_DRINK, ActionKind.END_TURN)):
            raise ValueError('exact_segment_invalid')
        legal = kernel.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
        if action_key(action) not in {action_key(item) for item in legal}:
            raise ValueError('exact_segment_illegal_action')
        seen.add(state)
    return rows
