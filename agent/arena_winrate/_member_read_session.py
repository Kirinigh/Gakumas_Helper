"""One member observation and its reader-owned, non-renewable retry budgets.

The session owns attempt bookkeeping and recovery classification. The reader
supplies the complete observation operation; the backend remains responsible
for proving the preview before any subsequent read. No frame or partial
observation is retained here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict
from logging import Logger
from dataclasses import field, dataclass
from collections.abc import Mapping

from .cancellation import ArenaReadSuperseded
from ._reader_errors import ArenaReaderError

if TYPE_CHECKING:
    from .reader import TeamTarget, MemberObservation, MemberObservationScope


class _AttemptClock(Protocol):
    def perf_counter(self) -> float: ...


class MemberReadBackend(Protocol):
    """Recovery, attempt lifecycle and optional diagnostics for this session.

    Hooks may be absent or non-callable on older/minimal backends; every use
    retains the runtime getattr/callable check. Missing recovery denies retry.
    """

    def recover_member_preview(
        self, target: TeamTarget, stage_number: int, member_slot: int, *, error_code: str,
    ) -> None: ...

    def _begin_member_read_attempt(self, target: TeamTarget, stage_number: int, member_slot: int) -> None: ...

    def _end_member_read_attempt(self) -> None: ...

    def _finalize_member_read_attempt(
        self, target: TeamTarget, stage_number: int, member_slot: int, *, succeeded: bool, superseded: bool,
    ) -> None: ...

    def begin_member_read_diagnostics(self, target: TeamTarget, stage_number: int, member_slot: int) -> None: ...

    def persist_member_read_failure(self, error: Exception) -> None: ...

    def end_member_read_diagnostics(self) -> None: ...

    def runtime_sample_cursor(self) -> Mapping[str, int]: ...

    def record_member_read_attempt(
        self, target: TeamTarget, stage_number: int, member_slot: int, *,
        wall_seconds: float, succeeded: bool, error_code: str | None, reopened: bool,
        sample_cursor: Mapping[str, int] | None = None, superseded: bool = False,
    ) -> None: ...


class MemberObservationOperation(Protocol):
    def __call__(
        self, target: TeamTarget, stage_number: int, member_slot: int, *,
        scope: MemberObservationScope, include_support_bonus: bool,
    ) -> MemberObservation: ...


@dataclass
class MemberReadRecoveryState:
    """Lives for the reader, including repeated provider calls.

    UI recovery has one reader-wide opportunity; recognition has three per
    member position. Only the closing marker resets at the next attempt.
    """

    ui_reopen_used: bool = False
    recognition_reopens: dict[tuple[str, int, int], int] = field(default_factory=dict)
    attempt_closing: bool = False


class MemberDetailRecovery(TypedDict):
    team_id: str
    stage_number: int
    member_slot: int
    group_index: int
    card_slot: int
    card_id: int
    title: str


@dataclass
class MemberDetailRecoveryState:
    """One attempt's proven failed detail, independent of diagnostic recording."""

    position: tuple[str, int, int] | None = None
    detail: MemberDetailRecovery | None = None
    managed_by_session: bool = False


def member_detail_recovery_state(owner: Any) -> MemberDetailRecoveryState:
    state = getattr(owner, "_member_detail_recovery", None)
    if state is None:
        state = MemberDetailRecoveryState()
        owner._member_detail_recovery = state
    return state


def _discard_member_detail_diagnostic_view(owner: Any) -> None:
    diagnostic = getattr(owner, "_member_failure_frames", None)
    if isinstance(diagnostic, dict):
        diagnostic.pop("skill_detail_recovery", None)


def begin_member_detail_recovery(
    owner: Any, target: TeamTarget, stage_number: int, member_slot: int, *, managed_by_session: bool = False,
) -> None:
    state = member_detail_recovery_state(owner)
    state.position = (target.team_id, stage_number, member_slot)
    state.detail = None
    state.managed_by_session = managed_by_session
    _discard_member_detail_diagnostic_view(owner)


def end_member_detail_recovery(owner: Any) -> None:
    state = member_detail_recovery_state(owner)
    state.position = None
    state.detail = None
    state.managed_by_session = False
    _discard_member_detail_diagnostic_view(owner)


def consume_member_detail_recovery(owner: Any) -> MemberDetailRecovery | None:
    state = member_detail_recovery_state(owner)
    detail = state.detail
    state.detail = None
    _discard_member_detail_diagnostic_view(owner)
    return detail


def record_member_diagnostic(backend: MemberReadBackend, logger: Logger, method: str, *args: Any, **kwargs: Any) -> None:
    recorder = getattr(backend, method, None)
    if callable(recorder):
        try:
            recorder(*args, **kwargs)
        except Exception:
            logger.exception("Could not record member diagnostics: %s", method)


class MemberReadSession:
    def __init__(
        self, backend: MemberReadBackend, state: MemberReadRecoveryState,
        read_once: MemberObservationOperation, *, clock: _AttemptClock, logger: Logger,
    ):
        self.backend = backend
        self.state = state
        self.read_once = read_once
        self.clock = clock
        self.logger = logger

    def _diagnostic(self, method: str, *args: Any, **kwargs: Any) -> None:
        record_member_diagnostic(self.backend, self.logger, method, *args, **kwargs)

    def _attempt_lifecycle(self, method: str, *args: Any, **kwargs: Any) -> None:
        hook = getattr(self.backend, method, None)
        if callable(hook):
            hook(*args, **kwargs)

    def run(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        *,
        scope: MemberObservationScope,
        include_support_bonus: bool,
    ) -> MemberObservation:
        position = (target.team_id, stage_number, member_slot)
        for attempt in range(4):
            reopened = attempt > 0
            started = self.clock.perf_counter()
            self._attempt_lifecycle("_begin_member_read_attempt", target, stage_number, member_slot)
            try:
                self._diagnostic("begin_member_read_diagnostics", target, stage_number, member_slot)
                succeeded = False
                failure: Exception | None = None
                superseded = False
                sample_cursor = None
                cursor_reader = getattr(self.backend, "runtime_sample_cursor", None)
                if callable(cursor_reader):
                    try:
                        sample_cursor = cursor_reader()
                    except Exception:
                        self.logger.exception("Could not begin member read diagnostics")
                self.state.attempt_closing = False
                try:
                    observation = self.read_once(
                        target,
                        stage_number,
                        member_slot,
                        scope=scope,
                        include_support_bonus=include_support_bonus,
                    )
                    succeeded = True
                    return observation
                except ArenaReadSuperseded:
                    superseded = True
                    raise
                except Exception as error:
                    failure = error
                    self._diagnostic("persist_member_read_failure", error)
                    recovery_code = member_reopen_error_code(
                        error, during_close=self.state.attempt_closing,
                    )
                    recognition_code = member_recognition_error_code(error)
                    recognition_recovery = recognition_code is not None or (
                        recovery_code is not None
                        and isinstance(error, ArenaReaderError)
                        and error.code.startswith(("skill_card_", "p_item_"))
                    )
                    recovery_code = recovery_code or recognition_code
                    used = self.state.recognition_reopens.get(position, 0)
                    exhausted = used >= 3 if recognition_recovery else self.state.ui_reopen_used
                    recover = getattr(self.backend, "recover_member_preview", None)
                    if (
                        exhausted
                        or recovery_code is None
                        or not callable(recover)
                    ):
                        if isinstance(error, ArenaReaderError) and (
                            recovery_code is not None
                            or error.code in {
                                "skill_card_detail_ambiguous",
                                "skill_card_badge_detail_inference_ambiguous",
                                "skill_card_customization_detail_empty",
                                "skill_card_customization_total_mismatch",
                                "p_item_detail_parser_failed",
                                "p_item_detail_ambiguous",
                                "p_item_detail_open_or_title_failed",
                                "p_item_detail_budget_exceeded",
                                "p_item_detail_recovery_failed",
                            }
                        ):
                            # These detail transactions already used their local
                            # observations/retries. A new lineup read cannot renew
                            # that budget or repair an unsupported catalog meaning.
                            error.retry_whole_read = False
                        raise
                    # Spend before navigation; a failed return cannot create another
                    # opportunity. The backend must prove the preview before rereading.
                    if recognition_recovery:
                        self.state.recognition_reopens[position] = used + 1
                    else:
                        self.state.ui_reopen_used = True
                    try:
                        recover(
                            target, stage_number, member_slot,
                            error_code=recovery_code,
                        )
                    except Exception as recovery_error:
                        raise ArenaReaderError(
                            "member_reopen_recovery_failed",
                            f"member read failed: {error}; returning to its team "
                            f"preview also failed: {recovery_error}",
                            retry_whole_read=False,
                        ) from recovery_error
                    # No partial observation escaped the failed invocation. The
                    # next invocation reopens and rereads this member completely.
                finally:
                    elapsed = self.clock.perf_counter() - started
                    try:
                        self._attempt_lifecycle(
                            "_finalize_member_read_attempt", target, stage_number, member_slot,
                            succeeded=succeeded, superseded=superseded,
                        )
                    except Exception:
                        self.logger.exception(
                            "Could not finalize member read attempt for %s/stage-%s/member-%s",
                            target.team_id, stage_number, member_slot,
                        )
                    recorder = getattr(self.backend, "record_member_read_attempt", None)
                    if callable(recorder):
                        try:
                            cursor_argument = (
                                {} if sample_cursor is None
                                else {"sample_cursor": sample_cursor}
                            )
                            if superseded:
                                cursor_argument["superseded"] = True
                            recorder(
                                target, stage_number, member_slot,
                                wall_seconds=elapsed,
                                succeeded=succeeded,
                                error_code=(
                                    None if failure is None
                                    else getattr(failure, "code", type(failure).__name__)
                                ),
                                reopened=reopened,
                                **cursor_argument,
                            )
                        except Exception:
                            self.logger.exception(
                                "Could not record member read attempt for %s/stage-%s/member-%s",
                                target.team_id, stage_number, member_slot,
                            )
                    self._diagnostic("end_member_read_diagnostics")
            finally:
                self._attempt_lifecycle("_end_member_read_attempt")
        raise AssertionError("member reopen budget exhausted without a result")


def member_recognition_error_code(error: Exception) -> str | None:
    """Only known observation failures can reopen a role, never setup faults."""
    if not isinstance(error, ArenaReaderError):
        return None
    current: BaseException | None = error
    seen: set[int] = set()
    while isinstance(current, ArenaReaderError) and id(current) not in seen:
        seen.add(id(current))
        if current.code in {
            "skill_card_cost_fallback_changed",
            "skill_card_cost_fallback_contract_invalid",
            "skill_card_cost_fallback_frame_conflict",
            "skill_card_cost_fallback_location_missing",
            "skill_card_cost_fallback_roi_conflict",
            "skill_card_detail_effect_view_conflict",
            "skill_card_detail_positive_evidence_missing",
            "skill_card_reference_unavailable",
            "skill_card_reference_catalog_invalid",
            "p_item_reader_missing",
            "p_item_runtime_not_approved",
        }:
            return None
        current = current.__cause__
    if error.code in {
        "skill_card_detail_ambiguous",
        "skill_card_badge_detail_inference_ambiguous",
        "skill_card_badge_detail_identity_unknown",
        "skill_card_customization_detail_empty",
        "skill_card_customization_total_mismatch",
        "skill_card_detail_title_unresolved",
        "skill_card_detail_upgrade_mark_missing",
        "skill_card_detail_id_changed",
        "skill_card_icon_unknown",
        "skill_card_reference_identity_ambiguous",
        "skill_card_customization_count_unknown",
        "skill_card_customization_frame_incomplete",
        "skill_card_customization_frame_shifted",
        "skill_card_layout_unstable",
        "skill_card_secondary_row_missing",
        "skill_card_slot_missing",
        "skill_card_empty_slot_unstable",
        "skill_card_duplicate_marker_unstable",
        "p_item_unknown",
        "p_item_detail_parser_failed",
        "p_item_detail_ambiguous",
        "p_item_detail_open_or_title_failed",
        "p_item_detail_budget_exceeded",
        "p_item_content_generation_unstable",
        "p_item_source_generation_unstable",
    }:
        return error.code
    return None


def member_reopen_error_code(
    error: Exception, *, during_close: bool,
) -> str | None:
    """Follow explicit UI causes without treating OCR ambiguity as UI proof."""

    current: BaseException | None = error
    seen: set[int] = set()
    p_item_restore = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if not isinstance(current, ArenaReaderError):
            return None
        if current.code in {
            "skill_card_close_failed",
            "skill_card_close_left_member",
            "skill_card_detail_disappeared",
            "skill_card_detail_missing",
            "p_item_source_restore_unproven",
        }:
            return current.code
        if current.code == "member_detail_anchor_missing" and (
            during_close or p_item_restore
        ):
            return current.code
        if during_close and current.code == "stage_member_list_timeout":
            return current.code
        if current.code == "p_item_detail_recovery_failed":
            # This production wrapper is emitted only around source-page
            # restoration, including its fallback member-anchor check.
            p_item_restore = True
        elif current.code != "skill_card_badge_detail_inference_ambiguous":
            return None
        current = current.__cause__
    return None
