"""Body retry consumes the current card transaction's unused contact only.

This wrapper owns no contact counter and cannot reset one. Opening remains the
single place that spends the shared two-contact budget; the detail reader owns
its existing bounded frame observations and the backend owns source proof.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol
from collections.abc import Mapping, Sequence, MutableMapping

from ._reader_errors import ArenaReaderError

if TYPE_CHECKING:
    from .reader import TeamTarget, ClickedSkillCard


class _RetryClock(Protocol):
    def perf_counter(self) -> float: ...


class _RecoveryLogger(Protocol):
    def info(self, message: str) -> None: ...


class SkillDetailRetryBackend(Protocol):
    """Live transaction views and the actions allowed by a body retry.

    Contact counts are read when the failure occurs, never copied into the
    session. The opening operation alone spends that same transaction budget.
    """

    @property
    def _card_transaction_contact_counts(self) -> Mapping[tuple[int, int], int]: ...

    @property
    def _detail_failure_frames(self) -> MutableMapping[str, Any] | None: ...

    @property
    def _runtime_counts(self) -> Mapping[str, int]: ...

    def _read_resolved_card_detail_once(
        self, key: tuple[int, int], candidate_ids: Sequence[int], *,
        expected_customization_count: int | None, timeout_seconds: float,
        allow_zero_without_badge_count: bool, target: TeamTarget | None,
        stage_number: int | None, member_slot: int | None,
    ) -> ClickedSkillCard: ...

    def _dismiss_skill_card_detail(self) -> None: ...

    def _assert_card_group_visible(self, group_index: int, *, card_slot: int) -> None: ...

    def _open_skill_card_once(
        self, target: TeamTarget | None, stage_number: int | None, member_slot: int | None,
        group_index: int, card_slot: int, expected_customization_count: int, *, contact_start_index: int,
    ) -> None: ...

    def _increment(self, name: str) -> None: ...

    def _record_duration_sample(self, name: str, seconds: float) -> None: ...


class SkillDetailRetrySession:
    def __init__(self, backend: SkillDetailRetryBackend, *, clock: _RetryClock, logger: _RecoveryLogger):
        self.backend = backend
        self.clock = clock
        self.logger = logger

    def run(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        timeout_seconds: float = 1.50,
        allow_zero_without_badge_count: bool = False,
        target: TeamTarget | None = None,
        stage_number: int | None = None,
        member_slot: int | None = None,
    ) -> ClickedSkillCard:
        """Reopen a prematurely vanished detail using only its unused contact."""
        arguments = {
            "expected_customization_count": expected_customization_count,
            "timeout_seconds": timeout_seconds,
            "allow_zero_without_badge_count": allow_zero_without_badge_count,
            "target": target, "stage_number": stage_number, "member_slot": member_slot,
        }
        try:
            return self.backend._read_resolved_card_detail_once(key, candidate_ids, **arguments)
        except ArenaReaderError as error:
            diagnostic = getattr(self.backend, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["error"] = str(error)
            used_contacts = getattr(self.backend, "_card_transaction_contact_counts", {}).get(key)
            if error.code not in {
                "skill_card_detail_disappeared", "skill_card_detail_id_changed",
                "skill_card_detail_upgrade_mark_missing",
                "skill_card_detail_title_unresolved",
            } or used_contacts != 1:
                raise
            original_error = str(error)
            error_code = error.code
        except Exception as error:
            diagnostic = getattr(self.backend, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["error"] = str(error)
            raise
        started = self.clock.perf_counter()
        before = dict(self.backend._runtime_counts)
        succeeded = False
        self.backend._increment("skill_card_detail_reopen_attempts")
        try:
            if error_code in {"skill_card_detail_id_changed", "skill_card_detail_upgrade_mark_missing", "skill_card_detail_title_unresolved"}:
                self.backend._dismiss_skill_card_detail()
                self.backend._assert_card_group_visible(key[0], card_slot=key[1])
                self.backend._increment("skill_card_detail_source_resets")
            # A disappeared detail already proved two source frames before any
            # dismissal. Reuse the original retry contact, never a new budget.
            self.backend._open_skill_card_once(
                target, stage_number, member_slot, key[0], key[1],
                1 if expected_customization_count is None else expected_customization_count,
                contact_start_index=used_contacts,
            )
            resolved = self.backend._read_resolved_card_detail_once(key, candidate_ids, **arguments)
            succeeded = True
            self.backend._increment("skill_card_detail_reopen_successes")
            return resolved
        except Exception as error:
            diagnostic = getattr(self.backend, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["error"] = str(error)
            raise
        finally:
            elapsed = self.clock.perf_counter() - started
            self.backend._record_duration_sample("skill_card_detail_reopen", elapsed)
            self.logger.info(json.dumps({
                "event": "arena_reader_recovery", "action": "skill_card_detail_reopen",
                "team_id": None if target is None else target.team_id,
                "stage_number": stage_number, "member_slot": member_slot,
                "group_index": key[0], "card_slot": key[1],
                "reason": original_error, "succeeded": succeeded,
                "wall_seconds": round(elapsed, 6),
                "extra_counts": {
                    name: value - before.get(name, 0)
                    for name, value in self.backend._runtime_counts.items()
                    if value > before.get(name, 0)
                },
            }, ensure_ascii=False, sort_keys=True))
