"""Screen-driven exam flow adapted from Hiro's shared ``exec/exam.py``.

The host supplies the screen reader and input. A decision is committed only
when that reader returns actual feedback, including any secondary selection.
"""

import time
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass
from collections.abc import Callable

from .hif_card_retry import wait_for_target_select
from .high_score.contracts import ActionKind
from .high_score.hif_environment import HifStep

EXAM_CHANGE_TIMEOUT = 8.0
EXAM_STALL_TIMEOUT = 180.0
LESSON_END_DELAY = 2.0


class ExamNoProgress(ValueError):
    pass


class ExamRecovery:
    """Hiro's stage animation pulse and three ineffective dispatches rule."""

    def __init__(self):
        self.last_animation = float('-inf')
        self.misses = 0

    def record(self, *, no_effect):
        self.misses = self.misses + 1 if no_effect else 0
        if self.misses == 3:
            self.misses = 0
            return True
        return False

    def advance_animation(self, context, image, *, is_animating, click):
        from .exam_hud import battle_dialog_header, battle_choice_buttons
        from .hif_card_retry import read_exam_select_markers

        now = time.monotonic()
        if (getattr(image, 'shape', ()) != (1280, 720, 3)
                or now - self.last_animation < .4 or battle_choice_buttons(image) is not None or not is_animating()
                or battle_dialog_header(image) is not None
                or read_exam_select_markers(context, image)[1]):
            return False
        click((360, 330))
        self.last_animation = now
        return True


class ExamOutcome(Enum):
    PROGRESS = auto()
    REOBSERVE = auto()
    RESULT = auto()


@dataclass(frozen=True)
class ExamCycle:
    outcome: ExamOutcome
    evidence: object


def run_exam_session(*, execute, observe, commit, recover, limit=None, expired=None, on_limit=None):
    """Schedule one live exam, committing only actual progress or a displayed result."""
    for index in count(1) if limit is None else range(1, limit + 1):
        cycle = observe(execute(index))
        if cycle.outcome is ExamOutcome.REOBSERVE:
            if recover(cycle.evidence):
                continue
            return None
        result = commit(cycle.evidence)
        if cycle.outcome is ExamOutcome.RESULT:
            return result
        if expired is not None and expired():
            return None
    if on_limit is not None:
        on_limit()
    return None


@dataclass(frozen=True)
class ExamReobserved:
    request: object
    dispatch_no_effect: bool = False


@dataclass(frozen=True)
class ExamCardConfirmation:
    box: tuple
    sent_at: float


@dataclass(frozen=True)
class ExamResult:
    rank: int | None
    score: int
    terminal_step: HifStep | None = None


@dataclass(frozen=True)
class ExamSettlement:
    rank: int | None
    score: int
    last_live_score: int

    @property
    def difference(self):
        return self.score - self.last_live_score


@dataclass(frozen=True)
class ExamFeedback:
    read: Callable
    refresh: Callable | None = None

    def wait(self, poll, purpose):
        try:
            return poll(self.read, purpose)
        except ExamNoProgress:
            if self.refresh is not None and (observed := self.refresh()) is not None:
                return observed
            raise


def select_exam_card(target, *, tap, capture, stopping, read_select, retry_target, reconcile,
                     initial_selection=None):
    """Confirm on the target's SELECT, or let a new frame reconcile the first tap."""
    selected, seen = initial_selection, initial_selection is not None
    if selected is None:
        tap(target, purpose='select-card')
        selected, seen = wait_for_target_select(capture, read_select, stopping)
    if selected is None and not seen and not stopping():
        image = capture()
        selected, seen = read_select(image)
        if selected is None and not seen and (retry := retry_target(image)) is not None:
            tap(retry, purpose='retry-select-card')
            selected, _ = wait_for_target_select(capture, read_select, stopping)
    if selected is None:
        selected = reconcile()
        if not isinstance(selected, tuple):
            return selected
    tap(selected, purpose='confirm-card')
    return None


def exam_drink_open_probe(read_actions, ready, retry, *, dialog_visible):
    """Hiro's one re-open attempt before any use has been submitted."""
    retry_after = time.monotonic() + 2.5
    opened = False
    retried = False

    def observe(frame):
        nonlocal opened, retried
        actions = read_actions(frame)
        opened = opened or bool(actions) or dialog_visible(frame)
        if target := ready(actions):
            return target
        if not opened and not retried and time.monotonic() >= retry_after:
            retried = True
            retry(frame)
        return None

    return observe


def run_exam(box, *, legal_actions, execute, feedback, poll, force_skip=False, on_forced_skip=None,
             allow_unranked=False):
    """Bind a DecisionBox to the shared live exam session."""
    previous_miss = None
    no_effect_count = 0
    pending_skip = False
    recovery = ExamRecovery()

    def dispatch(_index):
        nonlocal pending_skip
        before = box.request
        legal = tuple(legal_actions(before))
        if pending_skip:
            pending_skip = False
            skip = tuple(action for action in legal if action.kind is ActionKind.END_TURN)
            if skip:
                if on_forced_skip is not None:
                    on_forced_skip()
                legal = skip
        action = box.choose(before, legal)
        return before, action, execute(before, action)

    def observe(attempt):
        before, action, observed = attempt
        if observed is None:
            observed = feedback(before, action).wait(poll, 'action-successor:' + action.kind.value)
        if isinstance(observed, ExamReobserved):
            outcome = ExamOutcome.REOBSERVE
        elif isinstance(observed, ExamResult):
            outcome = ExamOutcome.RESULT
        else:
            outcome = ExamOutcome.PROGRESS
        return ExamCycle(outcome, (before, action, observed))

    def recover(evidence):
        nonlocal previous_miss, no_effect_count, pending_skip
        before, action, observed = evidence
        if observed.dispatch_no_effect:
            no_effect_count = no_effect_count + 1 if action == previous_miss else 1
            previous_miss = action
            if force_skip and recovery.record(no_effect=True):
                pending_skip = action.kind is not ActionKind.END_TURN
                if not pending_skip:
                    raise ExamNoProgress('idol_road_repeated_skip_no_effect')
            elif not force_skip and no_effect_count >= 3:
                kind = 'skip' if action.kind is ActionKind.END_TURN else 'card_select'
                raise ExamNoProgress(f'idol_road_repeated_{kind}_no_effect')
        else:
            previous_miss, no_effect_count = None, 0
            recovery.record(no_effect=False)
        box.reobserve_unsubmitted(before, action, observed.request)
        return True

    def commit(evidence):
        nonlocal previous_miss, no_effect_count
        before, action, observed = evidence
        if isinstance(observed, ExamResult):
            valid_rank = observed.rank is None and allow_unranked or (
                type(observed.rank) is int and 1 <= observed.rank <= 6)
            if not valid_rank or type(observed.score) is not int or observed.score < 0:
                raise ValueError('idol_road_displayed_result_unread')
            if observed.terminal_step is None:
                box.finish_on_observed_result(before.state, action, observed.score)
            else:
                box.observe_transition(before.state, action, observed.terminal_step)
                box.record_final_score(observed.score)
            return ExamSettlement(observed.rank, box.final_score, box.request.state.score)

        if observed.terminated:
            raise ValueError('idol_road_displayed_result_unread')
        box.observe_transition(before.state, action, observed)
        previous_miss, no_effect_count = None, 0
        recovery.record(no_effect=False)
        return None

    return run_exam_session(execute=dispatch, observe=observe, commit=commit, recover=recover)
