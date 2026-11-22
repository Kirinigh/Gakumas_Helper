"""Single-action route proposals with explicit observation and transition gates."""

from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass


class Outcome(StrEnum):
    DISABLED = "disabled"
    ACTION = "action"
    WAIT = "wait"
    PAUSE = "pause"
    STOP = "stop"


@dataclass(frozen=True)
class Target:
    """A producer-confirmed semantic control, never an inferred screen position."""

    key: str
    kind: str
    attribute: str | None = None
    special: bool | None = None
    available: bool | None = None


@dataclass(frozen=True)
class Observation:
    run_id: str
    revision: int
    scene: str
    page_id: str
    day: int | None
    idol: str | None
    plan: str | None
    capture_age_ms: int
    stable_frames: int
    scenario: str | None = None
    targets: tuple[Target, ...] = ()
    contradictory: bool = False

    @property
    def state_key(self) -> tuple[str, str, int | None]:
        # page_id is a semantic page/step identity, not a changing image hash.
        return self.scene, self.page_id, self.day


@dataclass(frozen=True)
class Proposal:
    target: Target
    source: Observation
    rule_id: str
    expected_scenes: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str
    proposal: Proposal | None = None


def ordinary_schedule_scene(scene, day):
    if day in (1, 4):
        return scene in ('lesson', 'lesson_selected', 'lesson_choice', 'lesson_choice_preview')
    if day in (2, 5):
        return scene in ('training', 'training_selected')
    if day == 3:
        return scene in ('outing', 'outing_selected', 'outing_choice', 'outing_choice_preview')
    return day == 6 and scene in ('consult_entry', 'consult_selected')


_COMPLETION_PAGE = 'recovery:ending_summary:::::produce_complete'


def completion_next_page(observation):
    return (observation.scenario == 'HIF' and observation.scene == 'ending_summary'
        and (observation.page_id == _COMPLETION_PAGE or observation.page_id.startswith(_COMPLETION_PAGE + ':')))


def current_completion_next(proposal, observation):
    # Consecutive post-generation pages can show the same Next and no readable section name.
    return (proposal.target.kind == 'end_produce_complete_next' and completion_next_page(proposal.source)
        and completion_next_page(observation) and _COMPLETION_PAGE in (proposal.source.page_id, observation.page_id)
        and observation.targets == (proposal.target,))


def has_single_frame_id(keys):
    return isinstance(keys, (list, tuple)) and len(keys) == 1 and isinstance(keys[0], str) and bool(keys[0].strip())


def source_frames_match(observation, frame_ids):
    """A routed HIF action refers to the current capture of its control."""
    return (observation.scenario == 'HIF' and type(observation.stable_frames) is int
        and observation.stable_frames == 1 and has_single_frame_id(frame_ids))


def ledger_frames_match(observation, frame_ids):
    return has_single_frame_id(frame_ids) if observation is None else source_frames_match(observation, frame_ids)


def validate_observation(observation: Observation, max_age_ms: int = 1500) -> str | None:
    if not observation.run_id or not observation.page_id:
        return "missing_run_or_page_identity"
    if type(observation.revision) is not int or observation.revision < 0:
        return "invalid_revision"
    if type(observation.capture_age_ms) is not int or not 0 <= observation.capture_age_ms <= max_age_ms:
        return "stale_observation"
    if completion_next_page(observation) and observation.page_id.removeprefix(_COMPLETION_PAGE) not in ('', ':items', ':achievements'):
        return "invalid_completion_page"
    if (type(observation.stable_frames) is not int or
            (observation.stable_frames != 1 if observation.scenario == "HIF" else observation.stable_frames < 2)):
        return "unstable_observation"
    if observation.contradictory:
        return "contradictory_observation"
    if observation.day is not None and (type(observation.day) is not int or not 1 <= observation.day <= 7):
        return "invalid_day"
    keys = [target.key for target in observation.targets]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        return "ambiguous_target_identity"
    return None


def propose_unique(
    observation: Observation,
    candidates: tuple[Target, ...],
    rule_id: str,
    expected_scenes: tuple[str, ...],
) -> Decision:
    """Unavailable, uncertain and duplicate targets cannot become a click."""
    if len(candidates) != 1 or candidates[0].available is not True:
        return Decision(Outcome.STOP, "target_not_uniquely_available")
    return Decision(Outcome.ACTION, rule_id, Proposal(candidates[0], observation, rule_id, expected_scenes))


class TransitionGate:
    """Track one externally issued action until a new semantic state confirms it.

    The caller marks an action issued only after its authorised executor accepts
    it. Polling this gate never issues, retries or duplicates any action.
    A pending transaction is not reconstructed from a screenshot after restart.
    """

    def __init__(self, timeout_ms: int = 15_000):
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.timeout_ms = timeout_ms
        self.pending: Proposal | None = None
        self.issued_at_ms: int | None = None
        self.stopped = False

    def mark_issued(self, proposal: Proposal, now_ms: int) -> None:
        if self.stopped or self.pending is not None:
            raise ValueError("an unresolved or stopped transition cannot issue another action")
        if validate_observation(proposal.source) or not proposal.expected_scenes:
            raise ValueError("invalid proposal source or expected state")
        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("now_ms must be a monotonic nonnegative integer")
        self.pending = proposal
        self.issued_at_ms = now_ms

    def observe(self, observation: Observation, now_ms: int, *, completion_next_received=False) -> Decision:
        if self.stopped:
            return Decision(Outcome.STOP, "transition_already_stopped")
        if self.pending is None:
            return Decision(Outcome.STOP, "no_pending_transition")
        proposal = self.pending
        if type(now_ms) is not int or now_ms < self.issued_at_ms:
            return self._stop("invalid_monotonic_time")
        if now_ms - self.issued_at_ms >= self.timeout_ms:
            return self._stop("transition_timeout")
        if observation.run_id != proposal.source.run_id:
            return self._stop("run_changed")
        if (observation.idol, observation.plan, observation.scenario) != (
            proposal.source.idol, proposal.source.plan, proposal.source.scenario
        ):
            return self._stop("identity_changed")
        error = validate_observation(observation)
        if error:
            # Animation/stale frames can be polled within the absolute deadline.
            if error in {"unstable_observation", "stale_observation"}:
                return Decision(Outcome.WAIT, error)
            return self._stop(error)
        if observation.revision < proposal.source.revision:
            return self._stop("revision_regressed")
        if observation.revision == proposal.source.revision:
            if observation != proposal.source:
                return self._stop("revision_reused_with_changed_observation")
            return Decision(Outcome.WAIT, "awaiting_new_observation")
        if observation.state_key == proposal.source.state_key and not (
                completion_next_received and current_completion_next(proposal, observation)):
            return Decision(Outcome.WAIT, "awaiting_semantic_transition")
        if observation.scene not in proposal.expected_scenes:
            return self._stop("unexpected_destination")
        previous_day = proposal.source.day
        if previous_day is not None and observation.day not in {previous_day, min(7, previous_day + 1)}:
            return self._stop("unconfirmed_day_transition")
        self.pending = None
        self.issued_at_ms = None
        return Decision(Outcome.PAUSE, "transition_confirmed")

    def _stop(self, reason: str) -> Decision:
        self.stopped = True
        return Decision(Outcome.STOP, reason)
