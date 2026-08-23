"""Fail-closed orchestration for reading a complete live contest lineup.

The reader owns the navigation order but delegates all screen interpretation and
normal UI input to a backend.  This keeps the three-stage contract testable
without a game process and gives the Maa backend one narrow interface.  It never
contains or exposes a challenge-start operation.
"""

from __future__ import annotations

import time
from enum import Enum
from uuid import uuid4
from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Mapping, Callable, Sequence

from .schema import SCHEMA_VERSION, validate_snapshot, validate_own_snapshot
from .stages import ContestSeasonDefinition


class ArenaReaderError(RuntimeError):
    """Raised when a screen state or required visible field is ambiguous."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class ArenaPageState(str, Enum):
    """Visible contest pages and the exhausted arena-main state."""

    READY = "ready"
    TEAM_PREVIEW = "team_preview"
    OPPONENTS_UNAVAILABLE = "opponents_unavailable"
    AMBIGUOUS = "ambiguous"


def classify_arena_page(
    *,
    rehearsal_count: int,
    opponent_count: int,
    stage_label_count: int,
    stage_total_count: int,
    exhausted_notice_count: int = 0,
) -> ArenaPageState:
    """Identify the page first, then interpret zero opponents on arena main as exhausted."""

    if rehearsal_count != 1:
        return ArenaPageState.AMBIGUOUS
    if opponent_count == 3:
        return ArenaPageState.READY
    if opponent_count != 0:
        return ArenaPageState.AMBIGUOUS
    if stage_label_count >= 3 and stage_total_count >= 3:
        return ArenaPageState.TEAM_PREVIEW
    if exhausted_notice_count == 1 and stage_total_count == 0:
        return ArenaPageState.OPPONENTS_UNAVAILABLE
    return ArenaPageState.AMBIGUOUS


def arena_page_allows_team_entry(
    state: ArenaPageState,
    *,
    require_opponents: bool,
) -> bool:
    """Accept exhausted arena main only for the independent own-team route."""

    return state is ArenaPageState.READY or (
        state is ArenaPageState.OPPONENTS_UNAVAILABLE and not require_opponents
    )


@dataclass(frozen=True)
class TeamTarget:
    team_id: str
    opponent_position: int | None

    @property
    def is_own_team(self) -> bool:
        return self.opponent_position is None


@dataclass(frozen=True)
class ClickedSkillCard:
    card_id: int
    customizations: Mapping[str, int]
    resolution_source: str = "badge_constrained"
    detail_confirmation_reads: int = 1
    detail_evidence_mode: str = "negative_dependent"


class MemberObservationScope(str, Enum):
    """Fields read during one member-page generation."""

    COMPLETE = "complete"
    CARDS_ONLY = "cards_only"


class SkillCardSlotState(str, Enum):
    """Mutually exclusive physical skill-card slot states."""

    DISABLED_BY_GRADE = "disabled_by_grade"
    EMPTY = "empty"
    EXCLUDED_DUPLICATE = "excluded_duplicate"
    PRESENT = "present"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class MemberObservation:
    """One auditable member-page generation shared by production and probes."""

    observation_id: str
    team_id: str
    stage_number: int
    member_slot: int
    scope: MemberObservationScope
    support_bonus: float | None
    params: tuple[int, int, int, int] | None
    p_item_ids: tuple[int, int, int, int] | None
    skill_card_id_groups: tuple[tuple[int, ...], tuple[int, ...]]
    customization_groups: tuple[
        tuple[Mapping[str, int], ...],
        tuple[Mapping[str, int], ...],
    ]
    excluded_duplicate_groups: tuple[tuple[bool, ...], tuple[bool, ...]]
    slot_state_groups: tuple[
        tuple[SkillCardSlotState, ...],
        tuple[SkillCardSlotState, ...],
    ]
    evidence: Mapping[str, Any]

    def to_loadout(self) -> dict[str, Any]:
        """Return the production schema only for a complete observation."""

        if self.scope is not MemberObservationScope.COMPLETE:
            raise ArenaReaderError(
                "member_observation_incomplete",
                f"{self.observation_id} contains card fields only",
            )
        if self.params is None or self.p_item_ids is None:
            raise ArenaReaderError(
                "member_observation_base_fields_missing",
                f"{self.observation_id} is missing params or P-items",
            )
        return {
            "params": list(self.params),
            "pItemIds": list(self.p_item_ids),
            "skillCardIdGroups": [list(group) for group in self.skill_card_id_groups],
            "customizationGroups": [
                [dict(customizations) for customizations in group]
                for group in self.customization_groups
            ],
            "excludedDuplicateGroups": [
                list(group) for group in self.excluded_duplicate_groups
            ],
        }

    def to_card_report(self) -> dict[str, Any]:
        """Return the shared card result consumed by read-only probes."""

        return {
            "observation_id": self.observation_id,
            "team_id": self.team_id,
            "stage_number": self.stage_number,
            "member_slot": self.member_slot,
            "params": None if self.params is None else list(self.params),
            "pItemIds": None if self.p_item_ids is None else list(self.p_item_ids),
            "skillCardIdGroups": [list(group) for group in self.skill_card_id_groups],
            "customizationGroups": [
                [dict(customizations) for customizations in group]
                for group in self.customization_groups
            ],
            "excludedDuplicateGroups": [
                list(group) for group in self.excluded_duplicate_groups
            ],
            "slotStateGroups": [
                [state.value for state in group] for group in self.slot_state_groups
            ],
            "evidence": dict(self.evidence),
        }


class ArenaReaderBackend(Protocol):
    """Administrator-Maa screen operations required by :class:`ArenaLineupReader`."""

    def ensure_arena_main(self, *, require_opponents: bool = True) -> None: ...

    def enter_team(self, target: TeamTarget) -> None: ...

    def select_stage(self, target: TeamTarget, stage_number: int) -> None: ...

    def member_slots(self, target: TeamTarget, stage_number: int) -> Sequence[int]: ...

    def open_member(self, target: TeamTarget, stage_number: int, slot: int) -> None: ...

    def read_support_bonus(self, target: TeamTarget) -> float: ...

    def read_params(
        self,
        target: TeamTarget,
        stage_number: int,
        slot: int,
    ) -> Sequence[int]: ...

    def read_p_item_ids(
        self,
        target: TeamTarget,
        stage_number: int,
        slot: int,
    ) -> Sequence[int]: ...

    def prepare_skill_card_group(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> None: ...

    def read_skill_card_id_hints(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        customization_counts: Sequence[int],
        excluded_duplicate_flags: Sequence[bool] | None = None,
    ) -> Sequence[int]: ...

    def read_skill_card_excluded_duplicate_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]: ...

    def read_skill_card_empty_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]: ...

    def read_skill_card_customization_counts(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[int]: ...

    def open_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
    ) -> None: ...

    def read_clicked_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
    ) -> ClickedSkillCard: ...

    def read_clicked_skill_card_unconstrained(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> ClickedSkillCard: ...

    def close_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> None: ...

    def close_member(self, target: TeamTarget, stage_number: int) -> None: ...

    def leave_team(self, target: TeamTarget) -> None: ...

    def recover_to_arena_main(self, *, require_opponents: bool = True) -> None: ...

    def member_observation_evidence(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
    ) -> Mapping[str, Any]: ...


def _default_capture_id() -> str:
    return f"live-{uuid4().hex}"


class ArenaLineupReader:
    """Read own lineup and three visible opponents in a fixed, auditable order."""

    def __init__(
        self,
        backend: ArenaReaderBackend,
        season: ContestSeasonDefinition,
        *,
        capture_id_factory: Callable[[], str] = _default_capture_id,
    ) -> None:
        self.backend = backend
        self.season = season
        self.capture_id_factory = capture_id_factory
        self._last_observations: list[MemberObservation] = []

    def last_observation_reports(self) -> tuple[dict[str, Any], ...]:
        """Expose low-dimensional reports from the latest public read operation."""

        return tuple(observation.to_card_report() for observation in self._last_observations)

    def read(self) -> dict[str, Any]:
        self._last_observations.clear()
        targets = (
            TeamTarget("self", None),
            TeamTarget("opponent-0", 0),
            TeamTarget("opponent-1", 1),
            TeamTarget("opponent-2", 2),
        )
        teams: list[dict[str, Any]] = []
        active_target: TeamTarget | None = None
        try:
            self.backend.ensure_arena_main(require_opponents=True)
            for target in targets:
                active_target = target
                teams.append(self._read_team(target))
                self.backend.leave_team(target)
                self.backend.ensure_arena_main(require_opponents=True)
                active_target = None
        except Exception as error:
            try:
                self.backend.recover_to_arena_main(require_opponents=True)
            except Exception as recovery_error:
                raise ArenaReaderError(
                    "recovery_failed",
                    f"read failed at {active_target}; recovery also failed: {recovery_error}",
                ) from error
            if isinstance(error, ArenaReaderError):
                raise
            raise ArenaReaderError("backend_failure", str(error)) from error

        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": self.capture_id_factory(),
            "source": "live_screen",
            "season": self.season.season,
            "stageIds": list(self.season.stage_ids),
            "own_team": teams[0],
            "opponents": teams[1:],
        }
        return validate_snapshot(snapshot)

    def read_own(self) -> dict[str, Any]:
        """Read and return only the player's reusable three-stage lineup."""

        self._last_observations.clear()
        target = TeamTarget("self", None)
        active = False
        try:
            self.backend.ensure_arena_main(require_opponents=False)
            active = True
            own_team = self._read_team(target)
            self.backend.leave_team(target)
            self.backend.ensure_arena_main(require_opponents=False)
            active = False
        except Exception as error:
            try:
                self.backend.recover_to_arena_main(require_opponents=False)
            except Exception as recovery_error:
                raise ArenaReaderError(
                    "recovery_failed",
                    f"own-team read failed while active={active}; recovery also failed: {recovery_error}",
                ) from error
            if isinstance(error, ArenaReaderError):
                raise
            raise ArenaReaderError("backend_failure", str(error)) from error

        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": self.capture_id_factory(),
            "source": "live_screen",
            "season": self.season.season,
            "stageIds": list(self.season.stage_ids),
            "own_team": own_team,
        }
        return validate_own_snapshot(snapshot)

    def read_opponents(self, own_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Read only the three opponents and combine them with a validated own capture.

        The caller decides when an own capture is still reusable.  This method never
        re-enters rehearsal, and it rejects a capture from another season or stage
        triplet before sending any UI input.
        """

        self._last_observations.clear()
        cached = validate_own_snapshot(own_snapshot)
        expected_stage_ids = list(self.season.stage_ids)
        if cached["season"] != self.season.season or cached["stageIds"] != expected_stage_ids:
            raise ArenaReaderError(
                "own_snapshot_season_mismatch",
                "cached own lineup does not match the selected contest season and stage IDs",
            )

        targets = tuple(TeamTarget(f"opponent-{position}", position) for position in range(3))
        opponents: list[dict[str, Any]] = []
        active_target: TeamTarget | None = None
        try:
            self.backend.ensure_arena_main(require_opponents=True)
            for target in targets:
                active_target = target
                opponents.append(self._read_team(target))
                self.backend.leave_team(target)
                self.backend.ensure_arena_main(require_opponents=True)
                active_target = None
        except Exception as error:
            try:
                self.backend.recover_to_arena_main(require_opponents=True)
            except Exception as recovery_error:
                raise ArenaReaderError(
                    "recovery_failed",
                    f"opponent read failed at {active_target}; recovery also failed: {recovery_error}",
                ) from error
            if isinstance(error, ArenaReaderError):
                raise
            raise ArenaReaderError("backend_failure", str(error)) from error

        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": self.capture_id_factory(),
            "source": "live_screen",
            "season": self.season.season,
            "stageIds": expected_stage_ids,
            "own_capture_id": cached["capture_id"],
            "own_team": cached["own_team"],
            "opponents": opponents,
        }
        return validate_snapshot(snapshot)

    def _read_team(self, target: TeamTarget) -> dict[str, Any]:
        team_started = time.perf_counter()
        self.backend.enter_team(target)
        support_bonus: float | None = None
        stages: list[dict[str, Any]] = []
        for stage_number, stage_id in enumerate(self.season.stage_ids, start=1):
            self.backend.select_stage(target, stage_number)
            slots = tuple(self.backend.member_slots(target, stage_number))
            self._validate_slots(slots, target, stage_number)
            members: list[dict[str, Any]] = []
            for slot in slots:
                try:
                    observation = self.read_member_observation(
                        target,
                        stage_number,
                        slot,
                        scope=MemberObservationScope.COMPLETE,
                        include_support_bonus=support_bonus is None,
                    )
                except ArenaReaderError as error:
                    raise ArenaReaderError(
                        error.code,
                        f"{target.team_id}/stage-{stage_number}/member-{slot}: "
                        f"{error.detail}",
                    ) from error
                if observation.support_bonus is not None:
                    support_bonus = observation.support_bonus
                self._last_observations.append(observation)
                members.append({"slot": slot, "loadout": observation.to_loadout()})
            stages.append(
                {
                    "stage_number": stage_number,
                    "stageId": stage_id,
                    "members": members,
                }
            )
        if support_bonus is None:
            raise ArenaReaderError("support_bonus_missing", f"{target.team_id} has no readable member")
        return {
            "team_id": target.team_id,
            "supportBonus": support_bonus,
            "read_wall_seconds": round(time.perf_counter() - team_started, 6),
            "stages": stages,
        }

    def read_member_observation(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        *,
        scope: MemberObservationScope = MemberObservationScope.COMPLETE,
        include_support_bonus: bool = False,
    ) -> MemberObservation:
        """Open, observe and close one member through the shared state machine."""

        if not isinstance(scope, MemberObservationScope):
            raise ArenaReaderError(
                "member_observation_scope_invalid",
                f"unsupported member observation scope {scope!r}",
            )
        observation_id = f"member-{uuid4().hex}"
        observation_started = time.perf_counter()
        self.backend.open_member(target, stage_number, member_slot)
        support_bonus = (
            self.backend.read_support_bonus(target) if include_support_bonus else None
        )
        params: tuple[int, ...] | None = None
        p_item_ids: tuple[int, ...] | None = None
        if scope is MemberObservationScope.COMPLETE:
            params = tuple(self.backend.read_params(target, stage_number, member_slot))
            p_item_ids = tuple(
                self.backend.read_p_item_ids(target, stage_number, member_slot)
            )
            if len(params) != 4:
                raise ArenaReaderError(
                    "params_count_mismatch",
                    f"{target.team_id}/stage-{stage_number}/member-{member_slot} yielded {len(params)} parameters",
                )
            if len(p_item_ids) != 4:
                raise ArenaReaderError(
                    "p_item_count_mismatch",
                    f"{target.team_id}/stage-{stage_number}/member-{member_slot} yielded {len(p_item_ids)} P-items",
                )

        skill_groups: list[list[int]] = [[], []]
        customization_groups: list[list[dict[str, int]]] = [[], []]
        excluded_duplicate_groups: list[list[bool]] = [[], []]
        empty_groups: list[list[bool]] = [[], []]
        group_order = (1, 0)
        # Establish one post-swipe page generation before any authoritative
        # slot classification.  The pre-swipe upper-left guard is only an
        # interaction safety gate and must not replace this shared generation.
        for group_index in group_order:
            self.backend.prepare_skill_card_group(target, stage_number, member_slot, group_index)

        # Classify physical blanks and duplicate markers for both rows before
        # badge batching or any detail click. This keeps all twelve decisions
        # on the same stable three-frame generation and skips non-interactive
        # slots before identity or customization work.
        for group_index in group_order:
            empty_flags = tuple(
                self.backend.read_skill_card_empty_flags(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                )
            )
            if len(empty_flags) != 6 or any(
                not isinstance(flag, bool) for flag in empty_flags
            ):
                raise ArenaReaderError(
                    "skill_card_empty_slot_count_mismatch",
                    f"group {group_index} yielded empty flags {empty_flags!r}",
                )
            empty_groups[group_index] = list(empty_flags)
            excluded_duplicate_flags = tuple(
                self.backend.read_skill_card_excluded_duplicate_flags(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                )
            )
            if len(excluded_duplicate_flags) != 6 or any(
                not isinstance(flag, bool) for flag in excluded_duplicate_flags
            ):
                raise ArenaReaderError(
                    "skill_card_excluded_duplicate_count_mismatch",
                    f"group {group_index} yielded duplicate flags {excluded_duplicate_flags!r}",
                )
            if any(
                empty and duplicate
                for empty, duplicate in zip(
                    empty_flags,
                    excluded_duplicate_flags,
                    strict=True,
                )
            ):
                raise ArenaReaderError(
                    "skill_card_slot_state_conflict",
                    f"group {group_index} classified one slot as both empty and duplicate",
                )
            excluded_duplicate_groups[group_index] = list(excluded_duplicate_flags)

        # Index assignment below preserves the engine's group-0/group-1 order.
        for group_index in group_order:
            excluded_duplicate_flags = tuple(excluded_duplicate_groups[group_index])
            empty_flags = tuple(empty_groups[group_index])
            customization_counts = tuple(
                self.backend.read_skill_card_customization_counts(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                )
            )
            if len(customization_counts) != 6 or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in customization_counts
            ):
                raise ArenaReaderError(
                    "skill_card_customization_count_mismatch",
                    f"group {group_index} yielded customization counts {customization_counts!r}",
                )
            current_duplicate_reader = getattr(
                self.backend,
                "current_skill_card_excluded_duplicate_flags",
                None,
            )
            if callable(current_duplicate_reader):
                updated_duplicate_flags = tuple(
                    current_duplicate_reader(group_index)
                )
                if len(updated_duplicate_flags) != 6 or any(
                    not isinstance(flag, bool) for flag in updated_duplicate_flags
                ):
                    raise ArenaReaderError(
                        "skill_card_excluded_duplicate_count_mismatch",
                        f"group {group_index} yielded updated duplicate flags "
                        f"{updated_duplicate_flags!r}",
                    )
                if any(
                    original and not updated
                    for original, updated in zip(
                        excluded_duplicate_flags,
                        updated_duplicate_flags,
                        strict=True,
                    )
                ):
                    raise ArenaReaderError(
                        "skill_card_excluded_duplicate_regressed",
                        f"group {group_index} lost an already proven duplicate flag",
                    )
                excluded_duplicate_flags = updated_duplicate_flags
                excluded_duplicate_groups[group_index] = list(
                    excluded_duplicate_flags
                )
                if any(
                    empty and duplicate
                    for empty, duplicate in zip(
                        empty_flags,
                        excluded_duplicate_flags,
                        strict=True,
                    )
                ):
                    raise ArenaReaderError(
                        "skill_card_slot_state_conflict",
                        f"group {group_index} updated one slot to both empty and duplicate",
                    )
            effective_customization_counts = tuple(
                0 if excluded or empty else count
                for count, excluded, empty in zip(
                    customization_counts,
                    excluded_duplicate_flags,
                    empty_flags,
                    strict=True,
                )
            )
            cards = list(
                self.backend.read_skill_card_id_hints(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                    effective_customization_counts,
                    excluded_duplicate_flags,
                )
            )
            if len(cards) != 6:
                raise ArenaReaderError(
                    "skill_card_id_count_mismatch",
                    f"group {group_index} yielded card IDs {cards!r}",
                )
            customizations: list[dict[str, int]] = [{} for _ in range(6)]
            for card_slot, customization_count in enumerate(
                effective_customization_counts,
                start=1,
            ):
                if customization_count == 0:
                    continue
                self.backend.open_skill_card(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                    card_slot,
                    customization_count,
                )
                clicked = self.backend.read_clicked_skill_card(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                    card_slot,
                    customization_count,
                )
                self.backend.close_skill_card(
                    target,
                    stage_number,
                    member_slot,
                    group_index,
                    card_slot,
                )
                cards[card_slot - 1] = clicked.card_id
                resolved = dict(clicked.customizations)
                resolved_count = sum(resolved.values())
                if resolved_count < 1:
                    raise ArenaReaderError(
                        "skill_card_customization_detail_empty",
                        f"group {group_index}/slot {card_slot} opened from a positive "
                        "badge but detail resolved no customization",
                    )
                if resolved_count != customization_count and not (
                    clicked.resolution_source in {
                        "detail_unique_confirmed",
                        "detail_card_face_unique_confirmed",
                    }
                    and clicked.detail_confirmation_reads >= 2
                ):
                    raise ArenaReaderError(
                        "skill_card_customization_total_mismatch",
                        f"group {group_index}/slot {card_slot} showed {customization_count} "
                        f"but detail yielded {resolved!r} without a repeated unique-detail "
                        "confirmation",
                    )
                customizations[card_slot - 1] = resolved
            invalid_cards = tuple(
                index
                for index, (card_id, excluded, empty) in enumerate(
                    zip(
                        cards,
                        excluded_duplicate_flags,
                        empty_flags,
                        strict=True,
                    ),
                    start=1,
                )
                if isinstance(card_id, bool)
                or ((excluded or empty) and card_id != 0)
                or (not excluded and not empty and card_id < 1)
            )
            if invalid_cards:
                raise ArenaReaderError(
                    "skill_card_id_count_mismatch",
                    f"group {group_index} yielded invalid card IDs {cards!r} at slots {invalid_cards!r}",
                )
            skill_groups[group_index] = cards
            customization_groups[group_index] = customizations
        slot_state_groups = tuple(
            tuple(
                SkillCardSlotState.EXCLUDED_DUPLICATE
                if excluded
                else SkillCardSlotState.EMPTY
                if empty
                else SkillCardSlotState.PRESENT
                for excluded, empty in zip(
                    excluded_duplicate_group,
                    empty_group,
                    strict=True,
                )
            )
            for excluded_duplicate_group, empty_group in zip(
                excluded_duplicate_groups,
                empty_groups,
                strict=True,
            )
        )
        evidence_reader = getattr(self.backend, "member_observation_evidence", None)
        backend_evidence = (
            dict(evidence_reader(target, stage_number, member_slot))
            if callable(evidence_reader)
            else {}
        )
        self.backend.close_member(target, stage_number)
        evidence = {
            **backend_evidence,
            "page_generation": observation_id,
            "member_observation_wall_seconds": round(
                time.perf_counter() - observation_started,
                6,
            ),
        }
        observation = MemberObservation(
            observation_id=observation_id,
            team_id=target.team_id,
            stage_number=stage_number,
            member_slot=member_slot,
            scope=scope,
            support_bonus=support_bonus,
            params=tuple(params) if params is not None else None,
            p_item_ids=tuple(p_item_ids) if p_item_ids is not None else None,
            skill_card_id_groups=(tuple(skill_groups[0]), tuple(skill_groups[1])),
            customization_groups=(
                tuple(dict(value) for value in customization_groups[0]),
                tuple(dict(value) for value in customization_groups[1]),
            ),
            excluded_duplicate_groups=(
                tuple(excluded_duplicate_groups[0]),
                tuple(excluded_duplicate_groups[1]),
            ),
            slot_state_groups=(slot_state_groups[0], slot_state_groups[1]),
            evidence=evidence,
        )
        return observation

    @staticmethod
    def _validate_slots(slots: tuple[int, ...], target: TeamTarget, stage_number: int) -> None:
        if not 1 <= len(slots) <= 3:
            raise ArenaReaderError(
                "member_count_mismatch",
                f"{target.team_id}/stage-{stage_number} yielded {len(slots)} members",
            )
        if slots != tuple(sorted(set(slots))) or any(slot not in (1, 2, 3) for slot in slots):
            raise ArenaReaderError(
                "member_slot_mismatch",
                f"{target.team_id}/stage-{stage_number} yielded slots {slots!r}",
            )
