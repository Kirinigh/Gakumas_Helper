"""Conditional EOT concentration timeline for the existing scalar leaf value.

Only already installed, pure EOT concentration effects advance. Future card
plays and other phases retain the original forecast's approximation. The
private states are never returned as executable or observed policy states.
"""
from dataclasses import dataclass

from .hif import _HifTurnBoundary, is_hif_turn_boundary
from .contracts import Stance, PlanType, SchedulePhase
from .state_copy import replace
from .validation_cache import check_search_budget

VERSION = 'end-turn-focus-projection-v1'


@dataclass(frozen=True)
class EndTurnFocusProjection:
    values: tuple[int, ...]
    status: str
    reason: str
    omitted_schedule_count: int = 0


def project_end_turn_focus(kernel, state):
    """Keep the current window and advance only future EOT focus ledgers."""
    check_search_budget()
    flat = (state.concentration,) * state.turns_remaining
    if state.terminal or state.turns_remaining <= 1:
        return EndTurnFocusProjection(flat, 'unchanged', 'no_future_window')
    if (not state.persistent_effects_initialized or state.pending_choice
            or state.pending_after_card_ids or state.stance is not Stance.NEUTRAL
            or (state.resolution_stack and not is_hif_turn_boundary(state))):
        return EndTurnFocusProjection(flat, 'declined', 'unresolved_or_non_sense_scalar_state')
    schedules, omitted = [], 0
    for schedule in kernel._prune_exhausted_schedules(state).scheduled_effects:
        check_search_budget()
        specification = kernel._scheduled_effect_spec(schedule)
        if schedule.trigger_phase is not SchedulePhase.END_OF_TURN:
            omitted += 1
            continue
        if (schedule.turns_until_fire != 0 or specification.trigger_card_type is not None
                or not specification.scheduled_effects
                or not kernel._terminal_focus_only_schedule(schedule)):
            return EndTurnFocusProjection(flat, 'declined', 'unsupported_eot_interaction', omitted)
        schedules.append(schedule)
    if not schedules:
        return EndTurnFocusProjection(flat, 'unchanged', 'no_installed_eot_focus', omitted)
    current = replace(state, scheduled_effects=tuple(schedules), resolution_stack=())
    values = [current.concentration]
    for _ in range(state.turns_remaining - 1):
        check_search_budget()
        # The public transition entry prunes TTL-zero/used-up references.
        # Reproduce that cleanup before calling the private EOT-only boundary.
        current = kernel._prune_exhausted_schedules(current)
        projection = _HifTurnBoundary(kernel, current.turn + 1)
        branches = projection._end_turn(current, plan_type=PlanType.SENSE)
        if (len(branches) != 1 or branches[0].probability != 1
                or not is_hif_turn_boundary(branches[0].state)
                or any(event.event_type.startswith('DRAW') for event in branches[0].events)):
            raise ValueError('end_turn_focus_projection_boundary_incomplete')
        values.append(branches[0].state.concentration)
        current = replace(branches[0].state, resolution_stack=())
    check_search_budget()
    return EndTurnFocusProjection(tuple(values), 'projected', 'installed_eot_focus_only', omitted)
