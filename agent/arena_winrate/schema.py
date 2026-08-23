"""Versioned validation for complete three-stage arena snapshots.

The live reader must provide all three contest stages for the player and all
three visible opponents. Missing fields are rejected instead of being guessed
from power ratings or upstream simulator defaults.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
from dataclasses import dataclass

SCHEMA_VERSION = "2.0"
STAGE_NUMBERS = (1, 2, 3)
OPPONENT_COUNT = 3
MIN_TEAM_SIZE = 1
MAX_TEAM_SIZE = 3
P_ITEM_COUNT = 4
SKILL_GROUP_COUNT = 2
SKILL_GROUP_SIZE = 6


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


class SnapshotValidationError(ValueError):
    """Raised when an arena observation is incomplete or inconsistent."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(details)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: object, path: str, issues: list[ValidationIssue]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(path, "must be an object"))
        return None
    return value


def _validate_int_list(
    value: object,
    path: str,
    issues: list[ValidationIssue],
    *,
    length: int,
    allow_zero: bool,
) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        issues.append(ValidationIssue(path, "must be an array"))
        return None
    if len(value) != length:
        issues.append(ValidationIssue(path, f"must contain exactly {length} values"))
    valid = True
    for index, item in enumerate(value):
        if not _is_int(item) or item < (0 if allow_zero else 1):
            qualifier = "non-negative" if allow_zero else "positive"
            issues.append(ValidationIssue(f"{path}[{index}]", f"must be a {qualifier} integer"))
            valid = False
    if len(value) != length or not valid:
        return None
    return tuple(value)


def _validate_customizations(value: object, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, list):
        issues.append(ValidationIssue(path, "must be an array"))
        return
    if len(value) != SKILL_GROUP_COUNT:
        issues.append(ValidationIssue(path, f"must contain exactly {SKILL_GROUP_COUNT} groups"))
    for group_index, group in enumerate(value):
        group_path = f"{path}[{group_index}]"
        if not isinstance(group, list):
            issues.append(ValidationIssue(group_path, "must be an array"))
            continue
        if len(group) != SKILL_GROUP_SIZE:
            issues.append(ValidationIssue(group_path, f"must contain exactly {SKILL_GROUP_SIZE} entries"))
        for card_index, entry in enumerate(group):
            entry_path = f"{group_path}[{card_index}]"
            if not isinstance(entry, Mapping):
                issues.append(ValidationIssue(entry_path, "must be an object"))
                continue
            for customization_id, count in entry.items():
                if not str(customization_id).isdigit() or not _is_int(count) or count < 1:
                    issues.append(ValidationIssue(entry_path, "must map numeric customization IDs to positive counts"))


def _validate_loadout(value: object, path: str, issues: list[ValidationIssue]) -> None:
    loadout = _require_mapping(value, path, issues)
    if loadout is None:
        return

    required = {
        "params",
        "pItemIds",
        "skillCardIdGroups",
        "customizationGroups",
    }
    for field in sorted(required - loadout.keys()):
        issues.append(ValidationIssue(f"{path}.{field}", "is required"))

    _validate_int_list(loadout.get("params"), f"{path}.params", issues, length=4, allow_zero=False)
    _validate_int_list(loadout.get("pItemIds"), f"{path}.pItemIds", issues, length=P_ITEM_COUNT, allow_zero=True)

    skill_groups = loadout.get("skillCardIdGroups")
    if not isinstance(skill_groups, list):
        issues.append(ValidationIssue(f"{path}.skillCardIdGroups", "must be an array"))
    else:
        if len(skill_groups) != SKILL_GROUP_COUNT:
            issues.append(
                ValidationIssue(f"{path}.skillCardIdGroups", f"must contain exactly {SKILL_GROUP_COUNT} groups")
            )
        for index, group in enumerate(skill_groups):
            _validate_int_list(
                group,
                f"{path}.skillCardIdGroups[{index}]",
                issues,
                length=SKILL_GROUP_SIZE,
                allow_zero=True,
            )

    _validate_customizations(loadout.get("customizationGroups"), f"{path}.customizationGroups", issues)
    duplicate_groups = loadout.get("excludedDuplicateGroups")
    if duplicate_groups is not None:
        if not isinstance(duplicate_groups, list) or len(duplicate_groups) != SKILL_GROUP_COUNT:
            issues.append(
                ValidationIssue(
                    f"{path}.excludedDuplicateGroups",
                    f"must contain exactly {SKILL_GROUP_COUNT} groups",
                )
            )
        else:
            for group_index, group in enumerate(duplicate_groups):
                group_path = f"{path}.excludedDuplicateGroups[{group_index}]"
                if not isinstance(group, list) or len(group) != SKILL_GROUP_SIZE:
                    issues.append(
                        ValidationIssue(
                            group_path,
                            f"must contain exactly {SKILL_GROUP_SIZE} boolean values",
                        )
                    )
                    continue
                for card_index, flag in enumerate(group):
                    if not isinstance(flag, bool):
                        issues.append(
                            ValidationIssue(f"{group_path}[{card_index}]", "must be a boolean")
                        )
                    elif flag and isinstance(skill_groups, list) and group_index < len(skill_groups):
                        skill_group = skill_groups[group_index]
                        customization_groups = loadout.get("customizationGroups")
                        if (
                            isinstance(skill_group, list)
                            and card_index < len(skill_group)
                            and skill_group[card_index] != 0
                        ):
                            issues.append(
                                ValidationIssue(
                                    f"{path}.skillCardIdGroups[{group_index}][{card_index}]",
                                    "must be 0 for an excluded duplicate",
                                )
                            )
                        if (
                            isinstance(customization_groups, list)
                            and group_index < len(customization_groups)
                            and isinstance(customization_groups[group_index], list)
                            and card_index < len(customization_groups[group_index])
                            and customization_groups[group_index][card_index] != {}
                        ):
                            issues.append(
                                ValidationIssue(
                                    f"{path}.customizationGroups[{group_index}][{card_index}]",
                                    "must be empty for an excluded duplicate",
                                )
                            )


def _validate_members(value: object, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, list):
        issues.append(ValidationIssue(path, "must be an array"))
        return
    if not MIN_TEAM_SIZE <= len(value) <= MAX_TEAM_SIZE:
        issues.append(
            ValidationIssue(path, f"must contain between {MIN_TEAM_SIZE} and {MAX_TEAM_SIZE} members")
        )

    seen_slots: set[int] = set()
    for index, member_value in enumerate(value):
        member_path = f"{path}[{index}]"
        member = _require_mapping(member_value, member_path, issues)
        if member is None:
            continue
        slot = member.get("slot")
        if not _is_int(slot) or not 1 <= slot <= MAX_TEAM_SIZE:
            issues.append(ValidationIssue(f"{member_path}.slot", f"must be an integer from 1 to {MAX_TEAM_SIZE}"))
        elif slot in seen_slots:
            issues.append(ValidationIssue(f"{member_path}.slot", "must be unique within the stage team"))
        else:
            seen_slots.add(slot)
        _validate_loadout(member.get("loadout"), f"{member_path}.loadout", issues)


def _validate_team(
    value: object,
    path: str,
    expected_stage_ids: tuple[int, ...] | None,
    issues: list[ValidationIssue],
) -> str | None:
    team = _require_mapping(value, path, issues)
    if team is None:
        return None

    team_id = team.get("team_id")
    if not isinstance(team_id, str) or not team_id.strip():
        issues.append(ValidationIssue(f"{path}.team_id", "must be a non-empty string"))
        normalized_team_id = None
    else:
        normalized_team_id = team_id

    support_bonus = team.get("supportBonus")
    if isinstance(support_bonus, bool) or not isinstance(support_bonus, (int, float)) or not 0 <= support_bonus <= 1:
        issues.append(ValidationIssue(f"{path}.supportBonus", "must be a number between 0 and 1"))

    stages = team.get("stages")
    if not isinstance(stages, list):
        issues.append(ValidationIssue(f"{path}.stages", "must be an array"))
        return normalized_team_id
    if len(stages) != len(STAGE_NUMBERS):
        issues.append(ValidationIssue(f"{path}.stages", "must contain exactly stages 1, 2 and 3"))

    actual_numbers: list[int] = []
    actual_ids: list[int] = []
    for index, stage_value in enumerate(stages):
        stage_path = f"{path}.stages[{index}]"
        stage = _require_mapping(stage_value, stage_path, issues)
        if stage is None:
            continue
        stage_number = stage.get("stage_number")
        stage_id = stage.get("stageId")
        if not _is_int(stage_number) or stage_number not in STAGE_NUMBERS:
            issues.append(ValidationIssue(f"{stage_path}.stage_number", "must be 1, 2 or 3"))
        else:
            actual_numbers.append(stage_number)
            if index < len(STAGE_NUMBERS) and stage_number != STAGE_NUMBERS[index]:
                issues.append(ValidationIssue(f"{stage_path}.stage_number", "must be ordered as stages 1, 2 and 3"))
        if not _is_int(stage_id) or stage_id < 1:
            issues.append(ValidationIssue(f"{stage_path}.stageId", "must be a positive integer"))
        else:
            actual_ids.append(stage_id)
            if expected_stage_ids is not None and index < len(expected_stage_ids) and stage_id != expected_stage_ids[index]:
                issues.append(
                    ValidationIssue(
                        f"{stage_path}.stageId",
                        f"must match root stageIds[{index}] ({expected_stage_ids[index]})",
                    )
                )
        _validate_members(stage.get("members"), f"{stage_path}.members", issues)

    if len(actual_numbers) != len(set(actual_numbers)):
        issues.append(ValidationIssue(f"{path}.stages", "stage_number values must be unique"))
    if len(actual_ids) != len(set(actual_ids)):
        issues.append(ValidationIssue(f"{path}.stages", "stageId values must be unique"))
    return normalized_team_id


def _validate_root_header(
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[int, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    root = _require_mapping(value, "$", issues)
    if root is None:
        raise SnapshotValidationError(issues)

    if root.get("schema_version") != SCHEMA_VERSION:
        issues.append(ValidationIssue("$.schema_version", f"must equal {SCHEMA_VERSION!r}"))
    capture_id = root.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id.strip():
        issues.append(ValidationIssue("$.capture_id", "must be a non-empty string"))
    if root.get("source") not in {"frozen_replay", "live_screen"}:
        issues.append(ValidationIssue("$.source", "must be 'frozen_replay' or 'live_screen'"))
    season = root.get("season")
    if not _is_int(season) or season < 1:
        issues.append(ValidationIssue("$.season", "must be a positive integer"))

    stage_ids = _validate_int_list(root.get("stageIds"), "$.stageIds", issues, length=3, allow_zero=False)
    if stage_ids is not None and len(set(stage_ids)) != len(stage_ids):
        issues.append(ValidationIssue("$.stageIds", "must contain three unique stage IDs"))
    return root, stage_ids, issues


def validate_own_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an independently reusable three-stage own-team capture."""

    root, stage_ids, issues = _validate_root_header(value)
    _validate_team(root.get("own_team"), "$.own_team", stage_ids, issues)
    if issues:
        raise SnapshotValidationError(issues)
    return deepcopy(dict(root))


def validate_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a defensive copy of a complete arena snapshot."""

    root, stage_ids, issues = _validate_root_header(value)

    own_team_id = _validate_team(root.get("own_team"), "$.own_team", stage_ids, issues)
    opponents = root.get("opponents")
    opponent_ids: set[str] = set()
    if not isinstance(opponents, list):
        issues.append(ValidationIssue("$.opponents", "must be an array"))
    else:
        if len(opponents) != OPPONENT_COUNT:
            issues.append(ValidationIssue("$.opponents", f"must contain exactly {OPPONENT_COUNT} visible opponents"))
        for index, opponent in enumerate(opponents):
            path = f"$.opponents[{index}]"
            opponent_id = _validate_team(opponent, path, stage_ids, issues)
            if opponent_id is not None:
                if opponent_id == own_team_id or opponent_id in opponent_ids:
                    issues.append(ValidationIssue(f"{path}.team_id", "must be unique across the contest snapshot"))
                opponent_ids.add(opponent_id)

    if issues:
        raise SnapshotValidationError(issues)
    return deepcopy(dict(root))
