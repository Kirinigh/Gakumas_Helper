"""Dependency-injected Maa bridge, also usable with inert offline contexts.

The host owns identity binding and separate capture/input authorisation. They
cannot be supplied by user-editable route settings or by the expected idol name.
"""

from __future__ import annotations

import time
from typing import Callable
from dataclasses import field, dataclass

from .hif import HifConfig, decide_hif
from .engine import (
    Outcome,
    Decision,
    Observation,
    TransitionGate,
    completion_next_page,
    validate_observation,
    current_completion_next,
    ordinary_schedule_scene,
)
from .checkpoint import PROCESS_CLOCK_EPOCH, restore_transition, transition_checkpoint

MAX_INPUT_AGE_MS = 1000
MAX_NAVIGATION_INPUT_AGE_MS = 5000


class RouteInputNotDispatched(ValueError):
    """Trusted endpoint rejected before the actual SDK input was called."""


def route_input_action(kind, box, *, day=None):
    if kind == 'reward_delete_scroll':
        if tuple(box) not in ((360,890,1,160), (360,440,1,160)):
            raise ValueError('deletion scroll is outside its card-grid viewport')
        x, y, _, height = box
        top, bottom = [x,y,1,1], [x,y+height,1,1]
        begin, end = (bottom, top) if y == 890 else (top, bottom)
        return dict(action='Swipe', begin=begin, end=end, duration=1000)
    if kind == 'start_memory_scroll':
        x, y, width, height = box
        if (x, y, width, height) != (360, 620, 1, 281):
            raise ValueError('unexpected qualifier memory list swipe')
        return dict(action='Swipe', begin=[x, y + height - 1, 1, 1], end=[x, y, 1, 1], duration=1200)
    if kind.startswith('start_memory_') or kind in ('select_consult', 'confirm_consult', 'lesson_option') or (kind == 'lesson' and day == 1):
        x,y,width,height = box
        return dict(action='Click',target=[x+width//2,y+height//2,1,1])
    if kind == 'recovery_inventory_scroll':
        x, y, width, height = box
        if width != 1 or height not in (40, 80) or not 300 <= x <= 400 or not 700 <= y <= 850:
            raise ValueError('nonbattle inventory scroll is outside its verified list viewport')
        return dict(action='Swipe', begin=[x, y + height, 1, 1], end=[x, y, 1, 1], duration=500)
    return dict(action='Click', target=list(box))


@dataclass(frozen=True)
class HifRuntimeHooks:
    observe: Callable[[], tuple[Observation, dict[str, tuple[int, int, int, int]]]]
    authorise_capture: Callable[[], bool]
    authorise_input: Callable[[], bool]
    transition_gate: TransitionGate = field(default_factory=TransitionGate)
    save_transition: Callable[[dict], bool] | None = None
    clock_epoch: str = PROCESS_CLOCK_EPOCH
    clock: Callable[[], int] | None = None
    confirmation_ready: Callable[[Observation], bool] | None = None
    cancel_unissued: Callable[[], bool] | None = None
    decide: Callable[[Observation, HifConfig], Decision] | None = None
    input_event: Callable[[dict], None] | None = None
    input_result: Callable[[dict], None] | None = None
    confirmation_transient: Callable[[Observation], bool] | None = None
    proposal_timeout_ms: Callable | None = None
    max_input_age_ms: int | Callable = MAX_INPUT_AGE_MS
    schedule_home_ready: Callable[[Observation], bool] | None = None

    def __post_init__(self):
        if (not callable(self.max_input_age_ms) and (type(self.max_input_age_ms) is not int
                or not 0 < self.max_input_age_ms <= MAX_NAVIGATION_INPUT_AGE_MS)):
            raise ValueError('route input age limit must be bounded')


def _publish_transition(hooks: HifRuntimeHooks, candidate: TransitionGate) -> bool:
    """Commit data before replacing the host's last acknowledged state."""
    if hooks.save_transition is None:
        return False
    try:
        if hooks.save_transition(transition_checkpoint(candidate, hooks.clock_epoch)) is not True:
            return False
    except Exception:
        return False
    current = hooks.transition_gate
    current.pending = candidate.pending
    current.issued_at_ms = candidate.issued_at_ms
    current.stopped = candidate.stopped
    current.timeout_ms = candidate.timeout_ms
    return True


def _confirm_pending(context, hooks, clock, sleep, observation=None, *, completion_next_received=False) -> Decision:
    """An unresolved input is observed under its original deadline, not sent again."""
    while True:
        if context.tasker.stopping:
            return Decision(Outcome.STOP, "task_stopping")
        if observation is None:
            if hooks.authorise_capture() is not True:
                return Decision(Outcome.STOP, "capture_not_authorised")
            observation, _ = hooks.observe()
        current = transition_checkpoint(hooks.transition_gate, hooks.clock_epoch)
        candidate = restore_transition(current, clock_epoch=hooks.clock_epoch)
        now = clock()
        pending = candidate.pending
        transient = (not candidate.stopped and pending is not None
            and hooks.confirmation_transient is not None and hooks.confirmation_transient(observation) is True
            and 0 <= now - candidate.issued_at_ms < candidate.timeout_ms
            and observation.run_id == pending.source.run_id
            and (observation.idol, observation.plan, observation.scenario) == (
                pending.source.idol, pending.source.plan, pending.source.scenario)
            and validate_observation(observation) is None)
        result = (Decision(Outcome.WAIT, 'awaiting_recovery_destination') if transient
            else candidate.observe(observation, now, completion_next_received=completion_next_received))
        if result.reason == "transition_confirmed" and hooks.confirmation_ready is not None:
            if hooks.confirmation_ready(observation) is not True:
                result = Decision(Outcome.WAIT, "awaiting_transaction_receipt")
                candidate = restore_transition(current, clock_epoch=hooks.clock_epoch)
        if transition_checkpoint(candidate, hooks.clock_epoch) != current:
            if not _publish_transition(hooks, candidate):
                return Decision(Outcome.STOP, "transition_checkpoint_failed")
        if result.outcome != Outcome.WAIT:
            return result
        observation = None
        sleep(0.1)


def run_hif_step(context, config: HifConfig, hooks: HifRuntimeHooks | None, *, clock=None, sleep=None) -> Decision:
    """Propose once, optionally execute once, then confirm or stop.

    authorise_input revalidates the current batch permission. Product Context
    verifies actual Maa input endpoints at its final SDK dispatch boundary;
    other callers retain complete endpoint checks in their permission gate.
    A true route mode is not that authorisation.
    No missing hook or malformed result is treated as approval.
    """
    if config.mode == "off":
        return Decision(Outcome.DISABLED, "hif_disabled")
    if hooks is None:
        return Decision(Outcome.STOP, "hif_runtime_observer_not_bound")
    clock = clock or hooks.clock or (lambda: int(time.monotonic() * 1000))
    sleep = sleep or time.sleep
    if hooks.authorise_capture() is not True:
        return Decision(Outcome.STOP, "capture_not_authorised")
    observation, boxes = hooks.observe()
    if hooks.transition_gate.stopped:
        return Decision(Outcome.STOP, "transition_already_stopped")
    if hooks.transition_gate.pending is not None:
        if hooks.save_transition is None:
            return Decision(Outcome.STOP, "hif_transition_store_not_bound")
        return _confirm_pending(context, hooks, clock, sleep, observation)
    home_checked = hooks.schedule_home_ready is not None and hooks.schedule_home_ready(observation) is True
    current_button = (observation.stable_frames == 1 and observation.scenario == "HIF"
        and not ordinary_schedule_scene(observation.scene, observation.day)
        and len(observation.targets) == 1 and observation.targets[0].kind != 'start_hif_once'
        and (observation.scene != 'interval_end_confirm' or observation.targets[0].kind == 'confirm_interval_exit'))
    if home_checked or current_button:
        refreshed_at_ms = clock()
    decide = hooks.decide or decide_hif
    decision = decide(observation, config)
    if decision.outcome != Outcome.ACTION or config.mode == "observe":
        return decision
    if context.tasker.stopping:
        return Decision(Outcome.STOP, "task_stopping")
    if hooks.authorise_input() is not True:
        return Decision(Outcome.STOP, "input_not_authorised_or_endpoint_unverified")
    if home_checked or (current_button and decision.proposal.target == observation.targets[0]):
        fresh, refreshed = observation, decision
    else:
        if hooks.authorise_capture() is not True:
            return Decision(Outcome.STOP, "capture_not_authorised")
        fresh, boxes = hooks.observe()
        refreshed_at_ms = clock()
        refreshed = decide(fresh, config)
        if (_selection_changed_on_confirmation(observation, fresh) and refreshed.outcome == Outcome.ACTION
                and fresh.revision > observation.revision):
            return Decision(Outcome.WAIT, 'hif_start_memory_selection_changed')
        if refreshed.outcome != Outcome.ACTION or fresh.state_key != observation.state_key:
            return Decision(Outcome.STOP, "pre_input_state_changed")
        if fresh.run_id != observation.run_id or fresh.revision <= observation.revision:
            return Decision(Outcome.STOP, "pre_input_observation_not_fresh")
        if refreshed.proposal.target != decision.proposal.target:
            return Decision(Outcome.STOP, "pre_input_target_changed")
        if (refreshed.proposal.rule_id != decision.proposal.rule_id
                or refreshed.proposal.expected_scenes != decision.proposal.expected_scenes):
            return Decision(Outcome.STOP, "pre_input_intent_changed")
    max_input_age_ms = (hooks.max_input_age_ms(refreshed.proposal)
        if callable(hooks.max_input_age_ms) else hooks.max_input_age_ms)
    if type(max_input_age_ms) is not int or not 0 < max_input_age_ms <= MAX_NAVIGATION_INPUT_AGE_MS:
        raise ValueError('route input age limit must be bounded')
    box = boxes.get(refreshed.proposal.target.key)
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(value) is not int for value in box):
        return Decision(Outcome.STOP, "target_box_missing")
    x, y, width, height = box
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > 720 or y + height > 1280:
        return Decision(Outcome.STOP, "target_box_outside_supported_frame")
    try:
        action = route_input_action(refreshed.proposal.target.kind, box, day=fresh.day)
    except ValueError:
        return Decision(Outcome.STOP, 'unsupported_route_input_geometry')
    if hooks.save_transition is None:
        return Decision(Outcome.STOP, "hif_transition_store_not_bound")
    if context.tasker.stopping or hooks.authorise_input() is not True:
        return Decision(Outcome.STOP, "input_gate_changed")
    issued_at_ms = clock()
    if issued_at_ms < refreshed_at_ms or fresh.capture_age_ms + issued_at_ms - refreshed_at_ms > max_input_age_ms:
        return Decision(Outcome.STOP, "pre_input_observation_expired")
    try:
        timeout_ms = (hooks.proposal_timeout_ms(refreshed.proposal) if hooks.proposal_timeout_ms is not None
            else hooks.transition_gate.timeout_ms)
        if type(timeout_ms) is not int or not 0 < timeout_ms <= 120_000:
            raise ValueError('invalid action timeout contract')
    except Exception:
        return Decision(Outcome.STOP, 'transition_timeout_contract_invalid')
    gate = TransitionGate(timeout_ms=timeout_ms)
    gate.mark_issued(refreshed.proposal, issued_at_ms)
    if not _publish_transition(hooks, gate):
        return Decision(Outcome.STOP, "transition_checkpoint_failed")
    # Checkpoint I/O consumes the same evidence/deadline budget. If input has
    # certainly not been attempted, acknowledge cancellation before clearing it.
    input_allowed = not context.tasker.stopping and hooks.authorise_input() is True
    before_input_ms = clock()
    reason = None
    if before_input_ms < issued_at_ms or fresh.capture_age_ms + before_input_ms - refreshed_at_ms > max_input_age_ms:
        reason = "pre_input_observation_expired"
    elif before_input_ms - issued_at_ms >= gate.timeout_ms:
        reason = "transition_timeout"
    elif context.tasker.stopping or not input_allowed:
        reason = "input_gate_changed"
    if reason:
        if hooks.cancel_unissued is not None:
            hooks.cancel_unissued()
        else:
            _publish_transition(hooks, TransitionGate(timeout_ms=gate.timeout_ms))
        return Decision(Outcome.STOP, reason)
    event = dict(target_key=refreshed.proposal.target.key,
        target_kind=refreshed.proposal.target.kind, source_scene=fresh.scene,
        source_revision=fresh.revision, target=list(box), issued_at_ms=issued_at_ms,
        captured_age_ms=fresh.capture_age_ms + before_input_ms - refreshed_at_ms,
        dispatch_attempted=False, success=None, uncertain=False, action=action)
    first_lesson = fresh.scene == 'lesson' and fresh.day == 1 and refreshed.proposal.target.kind == 'lesson'
    if first_lesson:
        event.update(clicks_planned=3, clicks_attempted=0, clicks_succeeded=0)
    if hooks.input_event is not None:
        try:
            hooks.input_event(event)
        except Exception:
            if hooks.cancel_unissued is not None:
                hooks.cancel_unissued()
            else:
                _publish_transition(hooks, TransitionGate(timeout_ms=gate.timeout_ms))
            return Decision(Outcome.STOP, 'input_audit_not_recorded_no_dispatch')
    # Source persistence consumes the same bounded evidence window as dispatch.
    if fresh.capture_age_ms + clock() - refreshed_at_ms > max_input_age_ms:
        if hooks.cancel_unissued is not None:
            hooks.cancel_unissued()
        return Decision(Outcome.STOP, 'pre_input_observation_expired')
    try:
        for index in range(3 if first_lesson else 1):
            if index:
                sleep(.3)
                if context.tasker.stopping or hooks.authorise_input() is not True:
                    raise RouteInputNotDispatched('input_gate_changed')
            event['dispatch_attempted'] = True
            if first_lesson:
                event['clicks_attempted'] = index + 1
            try:
                result = context.run_action(
                    "ProduceHifAdvanceClick",
                    pipeline_override={"ProduceHifAdvanceClick": action},
                )
            except RouteInputNotDispatched:
                if first_lesson:
                    event['clicks_attempted'] = index
                raise
            if getattr(result, 'success', None) is not True:
                break
            if first_lesson:
                event['clicks_succeeded'] = index + 1
    except RouteInputNotDispatched as error:
        if event.get('clicks_attempted', 0):
            # An earlier tap already reached Maa. Keep its pending transition
            # so a later run observes the destination instead of resending it.
            event.update(success=False, uncertain=True, remaining_clicks_rejected_before_dispatch=True,
                rejection_reason=str(error))
            if hooks.input_result is not None:
                try:
                    hooks.input_result(event)
                except Exception:
                    return Decision(Outcome.STOP, 'input_result_not_recorded_no_retry')
            return Decision(Outcome.STOP, 'input_not_confirmed_no_retry')
        event.update(dispatch_attempted=False, success=False, uncertain=False,
            rejected_before_dispatch=True, rejection_reason=str(error))
        if hooks.input_result is not None:
            try:
                hooks.input_result(event)
            except Exception:
                return Decision(Outcome.STOP, 'unissued_input_rejection_not_recorded')
        if hooks.cancel_unissued is not None:
            hooks.cancel_unissued()
        return Decision(Outcome.STOP, 'input_rejected_before_dispatch')
    except Exception:
        event['uncertain'] = True
        if hooks.input_result is not None:
            try:
                hooks.input_result(event)
            except Exception:
                pass  # The prepared receipt remains ambiguous; never retry.
        return Decision(Outcome.STOP, "input_not_confirmed_no_retry")
    event['success'] = result is not None and getattr(result, "success", None) is True
    event['uncertain'] = event['success'] is not True
    if hooks.input_result is not None:
        try:
            hooks.input_result(event)
        except Exception:
            return Decision(Outcome.STOP, 'input_result_not_recorded_no_retry')
    if result is None or getattr(result, "success", None) is not True:
        event['uncertain'] = True
        return Decision(Outcome.STOP, "input_not_confirmed_no_retry")
    completion_next_received = completion_next_page(fresh) and refreshed.proposal.target.kind == 'end_produce_complete_next'
    if completion_next_received:
        sleep(1.2)
    return _confirm_pending(context, hooks, clock, sleep, completion_next_received=completion_next_received)


class _SequenceStopped(RuntimeError):
    pass


def _selection_changed_on_confirmation(before, after):
    return (before.run_id == after.run_id and before.scenario == after.scenario
        and (before.idol, before.plan) == (after.idol, after.plan)
        and {before.scene, after.scene} == {'start_confirmation', 'start_memory_mismatch'})


def run_hif_sequence(
    context, config: HifConfig, hooks: HifRuntimeHooks | None, *,
    max_actions: int | None = 32, timeout_ms: int | None = None, clock=None, sleep=None,
) -> Decision:
    """Advance confirmed non-battle pages until a boundary or an explicit limit.

    Every action uses the same single-step refresh/authorisation transaction.
    The next step takes another observation; it never reuses the proposal or
    approval that preceded the last action. The cultivation runtime has no
    sequence deadline. Individual clicks keep their own confirmation timeout.
    """
    if config.mode == "off" or hooks is None:
        return run_hif_step(context, config, hooks)
    if max_actions is not None and (type(max_actions) is not int or not 1 <= max_actions <= 1000):
        raise ValueError("max_actions must be between 1 and 1000")
    if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
        raise ValueError("timeout_ms must be positive")
    clock = clock or hooks.clock or (lambda: int(time.monotonic() * 1000))
    started = clock()
    if type(started) is not int or started < 0:
        return Decision(Outcome.STOP, "invalid_monotonic_time")
    last_time = started
    previous: Observation | None = None
    expected_start = None
    identity: tuple | None = None

    def checked_clock():
        nonlocal last_time
        now = clock()
        if type(now) is not int or now < last_time:
            raise _SequenceStopped("invalid_monotonic_time")
        last_time = now
        if timeout_ms is not None and now - started >= timeout_ms:
            raise _SequenceStopped("sequence_timeout")
        if context.tasker.stopping:
            raise _SequenceStopped("task_stopping")
        return now

    def observe():
        nonlocal previous, identity, expected_start
        checked_clock()
        observation, boxes = hooks.observe()
        checked_clock()
        current_identity = (observation.run_id, observation.scenario, observation.idol, observation.plan)
        if identity is not None and current_identity != identity:
            raise _SequenceStopped("sequence_identity_changed")
        identity = current_identity
        if previous is not None and observation.revision < previous.revision:
            raise _SequenceStopped("sequence_revision_regressed")
        if expected_start is not None:
            source = expected_start.source
            if observation.revision <= source.revision:
                raise _SequenceStopped("sequence_observation_not_fresh")
            late_deletion_selection = (expected_start.target.kind == 'reward_delete_select'
                and source.scene == 'reward_delete_reselect' and observation.scene == 'reward_delete_selected'
                and source.day == observation.day)
            if (observation.state_key != source.state_key and not late_deletion_selection and not _selection_changed_on_confirmation(source, observation)
                    and not current_completion_next(expected_start, observation)):
                if (hooks.schedule_home_ready is None or hooks.schedule_home_ready(observation) is not True
                        or (observation.scene, observation.day) != (source.scene, source.day)):
                    raise _SequenceStopped("sequence_state_changed_between_steps")
                current = (hooks.decide or decide_hif)(observation, config)
                if (current.outcome != Outcome.ACTION or current.proposal.target != expected_start.target
                        or current.proposal.rule_id != expected_start.rule_id
                        or current.proposal.expected_scenes != expected_start.expected_scenes):
                    raise _SequenceStopped("sequence_state_changed_between_steps")
            expected_start = None
        previous = observation
        return observation, boxes

    def authorise(callback):
        checked_clock()
        allowed = callback()
        checked_clock()
        return allowed is True

    bounded_hooks = HifRuntimeHooks(
        observe,
        lambda: authorise(hooks.authorise_capture),
        lambda: authorise(hooks.authorise_input),
        transition_gate=hooks.transition_gate,
        save_transition=hooks.save_transition,
        clock_epoch=hooks.clock_epoch,
        clock=checked_clock,
        confirmation_ready=hooks.confirmation_ready,
        cancel_unissued=hooks.cancel_unissued,
        decide=hooks.decide,
        input_event=hooks.input_event,
        input_result=hooks.input_result,
        confirmation_transient=hooks.confirmation_transient,
        proposal_timeout_ms=hooks.proposal_timeout_ms,
        max_input_age_ms=hooks.max_input_age_ms,
        schedule_home_ready=hooks.schedule_home_ready,
    )
    try:
        actions = 0
        retried_selection = False
        while max_actions is None or actions < max_actions:
            checked_clock()
            result = run_hif_step(context, config, bounded_hooks, clock=checked_clock, sleep=sleep)
            if result.reason == 'hif_start_memory_selection_changed' and not retried_selection:
                retried_selection = True
                continue
            if result.reason != "transition_confirmed":
                return result
            retried_selection = False
            actions += 1
            # An encountered battle/selection boundary ends this invocation
            # immediately, even if the final action used the entire budget.
            boundary = (hooks.decide or decide_hif)(previous, config)
            if boundary.outcome != Outcome.ACTION:
                return boundary
            expected_start = boundary.proposal
        return Decision(Outcome.PAUSE, "sequence_action_budget_reached")
    except _SequenceStopped as stopped:
        return Decision(Outcome.STOP, str(stopped))
