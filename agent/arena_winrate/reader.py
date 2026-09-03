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
    conservative_cost_assumption: Mapping[str, Any] | None = None


class GenericCostFallbackCatalog(Protocol):
    """Catalog boundary needed to distrust serialized fallback metadata."""

    def validate_generic_cost_fallback_pair(
        self,
        card_id: int,
        *,
        generic_customization_id: int,
        unenhanced: Mapping[str, int],
        enhanced: Mapping[str, int],
        cost_hypotheses: Sequence[int],
        observed_badge_count: int | None,
    ) -> None: ...


def _positive_customization_mapping(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if (
            type(raw_key) is not str
            or not raw_key.isdigit()
            or str(int(raw_key)) != raw_key
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
            or raw_key in normalized
        ):
            return None
        normalized[raw_key] = raw_count
    return normalized


def validate_conservative_cost_fallback(
    clicked: ClickedSkillCard,
    *,
    catalog: GenericCostFallbackCatalog,
    expected_policy: str,
    expected_count: int | None,
    expected_count_kind: str,
) -> None:
    """Validate the complete internal proof carried by one bounded assumption."""

    assumption = clicked.conservative_cost_assumption
    invalid = False
    if expected_policy not in {
        "assume_unenhanced",
        "assume_enhanced",
    } or expected_count_kind not in {"observed", "effective", "reader"}:
        invalid = True
    if (
        type(clicked.card_id) is not int
        or clicked.card_id < 1
        or clicked.resolution_source != "generic_cost_conservative_fallback_confirmed"
        or type(clicked.detail_confirmation_reads) is not int
        or clicked.detail_confirmation_reads < 2
        or clicked.detail_evidence_mode != "conservative_generic_cost_bound"
        or not isinstance(assumption, Mapping)
    ):
        invalid = True

    if invalid:
        raise ArenaReaderError(
            "skill_card_cost_fallback_contract_invalid",
            "generic-cost fallback lacks a confirmed bounded-assumption envelope",
        )
    assert isinstance(assumption, Mapping)
    generic_id = assumption.get("generic_customization_id")
    observed_badge_count = assumption.get("observed_badge_count")
    assumption_card_id = assumption.get("card_id")
    hypotheses = assumption.get("cost_hypotheses")
    candidates = assumption.get("candidate_customizations")
    unenhanced = (
        _positive_customization_mapping(candidates.get("unenhanced"))
        if isinstance(candidates, Mapping)
        else None
    )
    enhanced = (
        _positive_customization_mapping(candidates.get("enhanced"))
        if isinstance(candidates, Mapping)
        else None
    )
    non_cost = _positive_customization_mapping(
        assumption.get("non_cost_customizations")
    )
    applied = _positive_customization_mapping(assumption.get("applied_customizations"))
    resolved = _positive_customization_mapping(clicked.customizations)
    expected_count_valid = (
        expected_count_kind == "observed" and expected_count is None
    ) or (
        type(expected_count) is int
        and expected_count >= (1 if expected_count_kind == "observed" else 0)
    )
    observed_count_valid = observed_badge_count is None or (
        type(observed_badge_count) is int and observed_badge_count > 0
    )
    if (
        assumption.get("reason_code") != "skill_card_cost_evidence_inconclusive"
        or assumption.get("policy") != expected_policy
        or type(assumption_card_id) is not int
        or assumption_card_id != clicked.card_id
        or type(generic_id) is not int
        or generic_id < 1
        or not expected_count_valid
        or not observed_count_valid
        or not isinstance(hypotheses, list)
        or len(hypotheses) != 2
        or any(type(value) is not int or value < 0 for value in hypotheses)
        or len(set(hypotheses)) != 2
        or unenhanced is None
        or enhanced is None
        or non_cost is None
        or applied is None
        or resolved is None
    ):
        raise ArenaReaderError(
            "skill_card_cost_fallback_contract_invalid",
            "generic-cost fallback metadata is incomplete or inconsistent",
        )
    generic_key = str(generic_id)
    expected_enhanced = dict(unenhanced)
    expected_enhanced[generic_key] = 1
    selected = enhanced if expected_policy == "assume_enhanced" else unenhanced
    if expected_count_kind == "observed":
        count_matches = observed_badge_count == expected_count
    elif expected_count_kind == "effective":
        count_matches = sum(resolved.values()) == expected_count
    else:
        count_matches = (
            observed_badge_count == expected_count
            if observed_badge_count is not None
            else sum(resolved.values()) == expected_count
        )
    if (
        generic_key in unenhanced
        or enhanced != expected_enhanced
        or non_cost != unenhanced
        or applied != selected
        or resolved != selected
        or not count_matches
    ):
        raise ArenaReaderError(
            "skill_card_cost_fallback_contract_invalid",
            "generic-cost fallback states are not the declared one-bit bound",
        )
    try:
        catalog.validate_generic_cost_fallback_pair(
            clicked.card_id,
            generic_customization_id=generic_id,
            unenhanced=unenhanced,
            enhanced=enhanced,
            cost_hypotheses=hypotheses,
            observed_badge_count=observed_badge_count,
        )
    except (TypeError, ValueError) as error:
        raise ArenaReaderError(
            "skill_card_cost_fallback_contract_invalid",
            f"generic-cost fallback does not match the fixed catalog: {error}",
        ) from error


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

    @property
    def grade(self) -> int | None: ...

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

    def last_read_metrics_summary(self) -> dict[str, Any]:
        """Aggregate one public read into a compact, machine-owned pace report."""

        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        samples: dict[str, list[float]] = {}
        detail_title_disambiguations: list[dict[str, Any]] = []
        for observation in self._last_observations:
            evidence = observation.evidence
            raw_timings = evidence.get("timing_seconds", {})
            raw_counts = evidence.get("counts", {})
            raw_samples = evidence.get("duration_samples_seconds", {})
            if not isinstance(raw_timings, Mapping):
                raw_timings = {}
            if not isinstance(raw_counts, Mapping):
                raw_counts = {}
            if not isinstance(raw_samples, Mapping):
                raw_samples = {}
            for name, value in raw_timings.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    timings[str(name)] = timings.get(str(name), 0.0) + float(value)
            for name, value in raw_counts.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    counts[str(name)] = counts.get(str(name), 0) + value
            for name, values in raw_samples.items():
                if not isinstance(values, (list, tuple)):
                    continue
                accepted = [
                    float(value)
                    for value in values
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                if accepted:
                    samples.setdefault(str(name), []).extend(accepted)
            raw_detail_title_disambiguations = evidence.get(
                "detail_title_disambiguations",
                (),
            )
            if isinstance(raw_detail_title_disambiguations, (list, tuple)):
                for value in raw_detail_title_disambiguations:
                    if not isinstance(value, Mapping):
                        continue
                    detail_title_disambiguations.append(
                        {
                            "team_id": observation.team_id,
                            "stage_number": observation.stage_number,
                            "member_slot": observation.member_slot,
                            **dict(value),
                        }
                    )

        def percentile(values: Sequence[float], quantile: float) -> float:
            ordered = sorted(values)
            if len(ordered) == 1:
                return ordered[0]
            position = (len(ordered) - 1) * quantile
            lower = int(position)
            upper = min(len(ordered) - 1, lower + 1)
            fraction = position - lower
            return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

        duration_percentiles = {
            name: {
                "count": len(values),
                "p50": round(percentile(values, 0.50), 6),
                "p95": round(percentile(values, 0.95), 6),
                "max": round(max(values), 6),
            }
            for name, values in sorted(samples.items())
            if values
        }
        evidence_route_names = (
            "skill_card_detail_same_frame_effect_roi_recoveries",
            "skill_card_detail_enhanced_effect_roi_recoveries",
            "skill_card_detail_effect_roi_retries",
            "skill_card_source_restore_semantic_settled_fallbacks",
            "skill_card_source_restore_source_family_fallbacks",
            "skill_card_source_restore_detail_title_disambiguations",
            "p_item_detail_effect_disambiguations",
            "p_item_generation_resamples",
        )
        retry_names = (
            "skill_card_detail_click_retries",
            "skill_card_detail_contact_retries",
            "skill_card_detail_source_resets",
            "skill_card_detail_close_retries",
        )
        return {
            "member_count": len(self._last_observations),
            "timing_seconds": {
                name: round(value, 6) for name, value in sorted(timings.items())
            },
            "counts": dict(sorted(counts.items())),
            "duration_percentile_method": "linear_interpolation_(n-1)q",
            "duration_percentiles_seconds": duration_percentiles,
            "assumptive_cost_fallbacks": [
                dict(value)
                for value in self.last_cost_customization_fallbacks()
            ],
            "detail_title_disambiguations": sorted(
                detail_title_disambiguations,
                key=lambda value: (
                    str(value.get("team_id", "")),
                    int(value.get("stage_number", 0)),
                    int(value.get("member_slot", 0)),
                    int(value.get("group_index", 0)),
                    int(value.get("card_slot", 0)),
                ),
            ),
            "recognition_evidence_routes": {
                name: counts[name]
                for name in evidence_route_names
                if counts.get(name, 0) > 0
            },
            "ui_action_retries": {
                name: counts[name]
                for name in retry_names
                if counts.get(name, 0) > 0
            },
            # Source-family restoration and title disambiguation are subsets of
            # semantic-settled proofs; consumers must not add the counters as
            # independent events or report successful disambiguation as an error.
            "counter_relationships": {
                "skill_card_source_restore_source_family_fallbacks": (
                    "subset_of_skill_card_source_restore_semantic_settled_fallbacks"
                ),
                "skill_card_source_restore_detail_title_disambiguations": (
                    "subset_of_skill_card_source_restore_source_family_fallbacks"
                ),
            },
            "duration_relationships": {
                "skill_card_detail_positive_confirmation_phase": (
                    "subset_of_skill_card_detail_resolution_phase"
                ),
                "skill_card_detail_open_phase": (
                    "component_of_skill_card_detail_transaction"
                ),
                "skill_card_detail_resolution_phase": (
                    "component_of_skill_card_detail_transaction"
                ),
                "skill_card_source_restore_phase": (
                    "component_of_skill_card_detail_transaction; may repeat after a "
                    "proven close retry"
                ),
                "skill_card_detail_capture_start_span": (
                    "diagnostic_only; does_not_change_fresh_frame_decision"
                ),
                "skill_card_source_restore_capture_start_span": (
                    "diagnostic_only; does_not_change_two_frame_restore_decision"
                ),
            },
            "screenshots_persisted": False,
        }

    def _require_backend_grade(self) -> int:
        grade = getattr(self.backend, "grade", None)
        if grade is None:
            raise ArenaReaderError(
                "arena_grade_missing",
                "the arena backend has no recognized or injected Grade",
            )
        if type(grade) is not int or not 1 <= grade <= 7:
            raise ArenaReaderError(
                "arena_grade_invalid",
                f"the arena backend Grade is outside 1..7: {grade!r}",
            )
        return grade

    def last_cost_customization_fallbacks(
        self,
        *,
        side: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return accepted, location-bound generic-cost assumptions."""

        if side not in {None, "own", "opponent"}:
            raise ValueError("side must be 'own', 'opponent', or None")
        records: list[dict[str, Any]] = []
        seen_locations: set[tuple[str, int, int, int, int]] = set()
        for observation in self._last_observations:
            for record in self._validated_cost_customization_fallback_records(
                observation
            ):
                group_index = int(record["group_index"])
                card_slot = int(record["card_slot"])
                location = (
                    observation.team_id,
                    observation.stage_number,
                    observation.member_slot,
                    group_index,
                    card_slot,
                )
                if location in seen_locations:
                    raise ArenaReaderError(
                        "skill_card_cost_fallback_duplicate",
                        f"fallback location was recorded twice: {location!r}",
                    )
                seen_locations.add(location)
                if side is None or record["side"] == side:
                    records.append(record)
        return tuple(
            sorted(
                records,
                key=lambda value: (
                    str(value.get("team_id", "")),
                    int(value.get("stage_number", 0)),
                    int(value.get("member_slot", 0)),
                    int(value.get("group_index", 0)),
                    int(value.get("card_slot", 0)),
                    int(value.get("card_id", 0)),
                ),
            )
        )

    def _validated_cost_customization_fallback_records(
        self,
        observation: MemberObservation,
    ) -> tuple[dict[str, Any], ...]:
        """Bind backend fallback evidence to the final member observation."""

        raw = observation.evidence.get("cost_customization_fallbacks", ())
        if not isinstance(raw, (list, tuple)):
            raise ArenaReaderError(
                "skill_card_cost_fallback_contract_invalid",
                f"{observation.observation_id} fallback evidence is not a sequence",
            )
        expected_side = "own" if observation.team_id == "self" else "opponent"
        opponent_token = observation.team_id.removeprefix("opponent-")
        if expected_side == "opponent" and not (
            observation.team_id.startswith("opponent-")
            and opponent_token in {"0", "1", "2"}
        ):
            raise ArenaReaderError(
                "skill_card_cost_fallback_location_invalid",
                f"unknown observation team identity: {observation.team_id!r}",
            )
        expected_position = None if expected_side == "own" else int(opponent_token)
        fallback_catalog = getattr(self.backend, "catalog", None)
        if raw and fallback_catalog is None:
            raise ArenaReaderError(
                "skill_card_cost_fallback_contract_invalid",
                "fallback evidence cannot be checked without its fixed catalog",
            )
        records: list[dict[str, Any]] = []
        seen_slots: set[tuple[int, int]] = set()
        for value in raw:
            if not isinstance(value, Mapping):
                raise ArenaReaderError(
                    "skill_card_cost_fallback_contract_invalid",
                    f"{observation.observation_id} contains a non-object fallback",
                )
            record = dict(value)
            group_index = record.get("group_index")
            card_slot = record.get("card_slot")
            record_stage_number = record.get("stage_number")
            record_member_slot = record.get("member_slot")
            record_opponent_position = record.get("opponent_position")
            position_valid = (
                record_opponent_position is None
                if expected_position is None
                else type(record_opponent_position) is int
                and record_opponent_position == expected_position
            )
            location_valid = (
                type(group_index) is int
                and group_index in (0, 1)
                and type(card_slot) is int
                and 1 <= card_slot <= 6
                and type(record.get("team_id")) is str
                and record.get("team_id") == observation.team_id
                and type(record_stage_number) is int
                and record_stage_number == observation.stage_number
                and type(record_member_slot) is int
                and record_member_slot == observation.member_slot
                and type(record.get("side")) is str
                and record.get("side") == expected_side
                and position_valid
            )
            if not location_valid:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_location_invalid",
                    f"{observation.observation_id} fallback location is inconsistent",
                )
            assert isinstance(group_index, int)
            assert isinstance(card_slot, int)
            slot_key = (group_index, card_slot)
            if slot_key in seen_slots:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_duplicate",
                    f"fallback slot was recorded twice: {slot_key!r}",
                )
            seen_slots.add(slot_key)
            expected_card_id = observation.skill_card_id_groups[group_index][
                card_slot - 1
            ]
            expected_customizations = dict(
                observation.customization_groups[group_index][card_slot - 1]
            )
            record_card_id = record.get("card_id")
            if type(record_card_id) is not int or record_card_id != expected_card_id:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_card_mismatch",
                    f"fallback card does not match final slot {slot_key!r}",
                )
            resolution_source = record.get("resolution_source")
            detail_evidence_mode = record.get("detail_evidence_mode")
            if (
                type(resolution_source) is not str
                or type(detail_evidence_mode) is not str
            ):
                raise ArenaReaderError(
                    "skill_card_cost_fallback_contract_invalid",
                    f"fallback evidence modes are not strings at {slot_key!r}",
                )
            validate_conservative_cost_fallback(
                ClickedSkillCard(
                    expected_card_id,
                    expected_customizations,
                    resolution_source=resolution_source,
                    detail_confirmation_reads=record.get(
                        "detail_confirmation_reads", 0
                    ),
                    detail_evidence_mode=detail_evidence_mode,
                    conservative_cost_assumption=record,
                ),
                catalog=fallback_catalog,
                expected_policy=(
                    "assume_unenhanced" if expected_side == "own" else "assume_enhanced"
                ),
                expected_count=sum(expected_customizations.values()),
                expected_count_kind="effective",
            )
            record["observation_id"] = observation.observation_id
            records.append(record)
        return tuple(records)

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
            arena_grade = self._require_backend_grade()
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
            "arena_grade": arena_grade,
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
            arena_grade = self._require_backend_grade()
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
            "arena_grade": arena_grade,
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
        read_started = time.perf_counter()
        cached = validate_own_snapshot(own_snapshot)
        expected_stage_ids = list(self.season.stage_ids)
        if cached["season"] != self.season.season or cached["stageIds"] != expected_stage_ids:
            raise ArenaReaderError(
                "own_snapshot_season_mismatch",
                "cached own lineup does not match the selected contest season and stage IDs",
            )
        effective_grade = self._require_backend_grade()

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
            "arena_grade": effective_grade,
            "stageIds": expected_stage_ids,
            "own_capture_id": cached["capture_id"],
            "own_team": cached["own_team"],
            "opponents": opponents,
            "read_wall_seconds": round(time.perf_counter() - read_started, 6),
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
        clicked_cost_fallback_slots: set[tuple[int, int]] = set()
        group_order = (1, 0)
        # Request the lower row first so one upward swipe establishes the
        # shared post-swipe generation for both physical rows.  The subsequent
        # upper-row request reuses that same frozen twelve-card generation.
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
                fallback_claimed = (
                    clicked.conservative_cost_assumption is not None
                    or clicked.resolution_source.startswith(
                        "generic_cost_conservative_fallback"
                    )
                    or clicked.detail_evidence_mode == "conservative_generic_cost_bound"
                )
                if fallback_claimed:
                    fallback_catalog = getattr(self.backend, "catalog", None)
                    if fallback_catalog is None:
                        raise ArenaReaderError(
                            "skill_card_cost_fallback_contract_invalid",
                            "generic-cost fallback cannot be checked without its fixed catalog",
                        )
                    validate_conservative_cost_fallback(
                        clicked,
                        catalog=fallback_catalog,
                        expected_policy=(
                            "assume_unenhanced"
                            if target.is_own_team
                            else "assume_enhanced"
                        ),
                        expected_count=customization_count,
                        expected_count_kind="reader",
                    )
                    clicked_cost_fallback_slots.add((group_index, card_slot))
                conservative_cost_fallback = fallback_claimed
                if resolved_count < 1 and not conservative_cost_fallback:
                    raise ArenaReaderError(
                        "skill_card_customization_detail_empty",
                        f"group {group_index}/slot {card_slot} opened from a positive "
                        "badge but detail resolved no customization",
                    )
                if resolved_count != customization_count and not (
                    conservative_cost_fallback
                    or (
                        clicked.resolution_source
                        in {
                            "detail_unique_confirmed",
                            "detail_card_face_unique_confirmed",
                        }
                        and clicked.detail_confirmation_reads >= 2
                    )
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
        validated_fallback_records = (
            self._validated_cost_customization_fallback_records(observation)
        )
        recorded_fallback_slots = {
            (int(record["group_index"]), int(record["card_slot"]))
            for record in validated_fallback_records
        }
        missing_fallback_slots = sorted(
            clicked_cost_fallback_slots - recorded_fallback_slots
        )
        if missing_fallback_slots:
            raise ArenaReaderError(
                "skill_card_cost_fallback_location_missing",
                f"{observation.observation_id} accepted fallback cards without location-bound evidence: {missing_fallback_slots!r}",
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
