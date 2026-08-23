"""Resolve the three contest stage IDs from the simulator's bundled catalog."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Iterable

CATALOG_RELATIVE_PATH = Path("node_modules") / "gakumas-data" / "json" / "stages.json"
LATEST_SEASON = "latest"
EXPECTED_STAGE_NUMBERS = (1, 2, 3)


class StageCatalogError(ValueError):
    """Raised when the bundled stage catalog cannot resolve a complete season."""


def _csv_tuple(value: object, *, field: str, cast: type[int] | type[float]) -> tuple[int | float, ...]:
    if not isinstance(value, str):
        raise StageCatalogError(f"{field} must be a CSV string")
    try:
        return tuple(cast(item) for item in value.split(","))
    except ValueError as error:
        raise StageCatalogError(f"{field} contains an invalid number") from error


@dataclass(frozen=True)
class ContestStageDefinition:
    stage_id: int
    season: int
    stage_number: int
    preview: bool
    plan: str
    criteria: tuple[float, ...]
    turn_counts: tuple[int, ...]
    first_turns: tuple[float, ...]
    effects: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContestStageDefinition":
        stage_id = value.get("id")
        season = value.get("season")
        stage_number = value.get("stage")
        preview = value.get("preview")
        plan = value.get("plan")
        effects = value.get("effects")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in (stage_id, season)):
            raise StageCatalogError("contest stage id and season must be positive integers")
        if isinstance(stage_number, bool) or not isinstance(stage_number, int):
            raise StageCatalogError("contest stage number must be an integer")
        if not isinstance(preview, bool):
            raise StageCatalogError("contest preview flag must be boolean")
        if not isinstance(plan, str) or not plan:
            raise StageCatalogError("contest plan must be a non-empty string")
        if not isinstance(effects, str):
            raise StageCatalogError("contest effects must be a string")
        criteria = _csv_tuple(value.get("criteria"), field="criteria", cast=float)
        turn_counts = _csv_tuple(value.get("turnCounts"), field="turnCounts", cast=int)
        first_turns = _csv_tuple(value.get("firstTurns"), field="firstTurns", cast=float)
        if not all(len(items) == 3 for items in (criteria, turn_counts, first_turns)):
            raise StageCatalogError("contest criteria and turn fields must contain three values")
        return cls(
            stage_id=stage_id,
            season=season,
            stage_number=stage_number,
            preview=preview,
            plan=plan,
            criteria=criteria,
            turn_counts=turn_counts,
            first_turns=first_turns,
            effects=effects,
        )


@dataclass(frozen=True)
class ContestSeasonDefinition:
    season: int
    stages: tuple[ContestStageDefinition, ...]

    @property
    def stage_ids(self) -> tuple[int, int, int]:
        return tuple(stage.stage_id for stage in self.stages)  # type: ignore[return-value]

    @property
    def preview(self) -> bool:
        return any(stage.preview for stage in self.stages)


class ContestStageCatalog:
    """Version-matched view over ``gakumas-data/json/stages.json``."""

    def __init__(self, stages: Iterable[ContestStageDefinition]) -> None:
        by_season: dict[int, dict[int, ContestStageDefinition]] = {}
        for stage in stages:
            season = by_season.setdefault(stage.season, {})
            if stage.stage_number in season:
                raise StageCatalogError(
                    f"duplicate contest season/stage: {stage.season}/{stage.stage_number}"
                )
            season[stage.stage_number] = stage
        if not by_season:
            raise StageCatalogError("contest stage catalog is empty")
        self._by_season = by_season

    @classmethod
    def from_rows(cls, rows: object) -> "ContestStageCatalog":
        if not isinstance(rows, list):
            raise StageCatalogError("stage catalog root must be an array")
        stages = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise StageCatalogError("stage catalog entries must be objects")
            if row.get("type") == "contest":
                stages.append(ContestStageDefinition.from_mapping(row))
        return cls(stages)

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> "ContestStageCatalog":
        catalog_path = Path(bundle_dir) / CATALOG_RELATIVE_PATH
        try:
            rows = json.loads(catalog_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise StageCatalogError(f"bundled stage catalog is unavailable: {catalog_path}") from error
        except json.JSONDecodeError as error:
            raise StageCatalogError(f"bundled stage catalog is invalid JSON: {catalog_path}") from error
        return cls.from_rows(rows)

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_season))

    def resolve(self, selection: str | int = LATEST_SEASON) -> ContestSeasonDefinition:
        if selection == LATEST_SEASON:
            season_number = max(self._by_season)
        elif isinstance(selection, int) and not isinstance(selection, bool) and selection >= 1:
            season_number = selection
        else:
            raise StageCatalogError("season selection must be 'latest' or a positive integer")

        stage_map = self._by_season.get(season_number)
        if stage_map is None:
            raise StageCatalogError(f"contest season is not present in the bundled engine: {season_number}")
        actual_numbers = tuple(sorted(stage_map))
        if actual_numbers != EXPECTED_STAGE_NUMBERS:
            raise StageCatalogError(
                f"contest season {season_number} must contain exactly stages 1, 2 and 3; got {actual_numbers}"
            )
        return ContestSeasonDefinition(
            season=season_number,
            stages=tuple(stage_map[number] for number in EXPECTED_STAGE_NUMBERS),
        )
