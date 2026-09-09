"""Validated runtime parameters for the arena win-rate UI mode."""

from __future__ import annotations

from typing import Any, Mapping, Protocol
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "assets" / "arena-winrate"
DEFAULT_THRESHOLD_PERCENT = 70
DEFAULT_SIMULATIONS = 2000
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_SEASON_SELECTION = "latest"
MIN_ARENA_GRADE = 1
MAX_ARENA_GRADE = 7


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ValueError(f"{field} must be an integer")


def _season_selection(value: object) -> str | int:
    if value == DEFAULT_SEASON_SELECTION:
        return DEFAULT_SEASON_SELECTION
    season = _integer(value, field="season")
    if season < 1:
        raise ValueError("season must be 'latest' or a positive integer")
    return season


def _grade_override(value: object) -> int | None:
    if value is None:
        return None
    grade = _integer(value, field="grade_override")
    if not MIN_ARENA_GRADE <= grade <= MAX_ARENA_GRADE:
        raise ValueError(
            f"grade_override must be from {MIN_ARENA_GRADE} to {MAX_ARENA_GRADE}"
        )
    return grade


class ArenaGradeState(Protocol):
    recognized_grade: int | None
    grade_override: int | None
    effective_grade: int | None
    source: str


class ArenaGradeCache(Protocol):
    def get_grade_state(self) -> ArenaGradeState: ...

    def set_grade_override(self, value: int | None) -> ArenaGradeState: ...


def resolve_cached_arena_grade(
    cache_store: ArenaGradeCache,
    grade_override: object,
) -> tuple[int, ArenaGradeState]:
    """Persist the selected override and require one effective cached Grade."""

    normalized_override = _grade_override(grade_override)
    state = cache_store.get_grade_state()
    if state.grade_override != normalized_override:
        state = cache_store.set_grade_override(normalized_override)
    grade = state.effective_grade
    if type(grade) is not int or not MIN_ARENA_GRADE <= grade <= MAX_ARENA_GRADE:
        raise ValueError(
            "cached arena Grade is unrecognized; recalculate the own lineup "
            "or select a Grade override"
        )
    return grade, state


@dataclass(frozen=True)
class ArenaRuntimeConfig:
    season: str | int = DEFAULT_SEASON_SELECTION
    grade_override: int | None = None
    threshold_percent: int = DEFAULT_THRESHOLD_PERCENT
    simulations: int = DEFAULT_SIMULATIONS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    bundle_dir: Path = DEFAULT_BUNDLE_DIR

    @property
    def threshold(self) -> float:
        return self.threshold_percent / 100

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArenaRuntimeConfig":
        season = _season_selection(value.get("season", DEFAULT_SEASON_SELECTION))
        grade_override = _grade_override(value.get("grade_override"))
        threshold_percent = _integer(
            value.get("threshold_percent", DEFAULT_THRESHOLD_PERCENT),
            field="threshold_percent",
        )
        simulations = _integer(value.get("simulations", DEFAULT_SIMULATIONS), field="simulations")
        timeout_seconds = _integer(
            value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            field="timeout_seconds",
        )
        if not 0 <= threshold_percent <= 100:
            raise ValueError("threshold_percent must be from 0 to 100")
        if simulations < 1000:
            raise ValueError("simulations must be at least 1000")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")

        bundle_value = value.get("bundle_path", DEFAULT_BUNDLE_DIR)
        if not isinstance(bundle_value, (str, Path)) or not str(bundle_value).strip():
            raise ValueError("bundle_path must be a non-empty path")
        bundle_dir = Path(bundle_value)
        if not bundle_dir.is_absolute():
            bundle_dir = PROJECT_ROOT / bundle_dir
        return cls(
            season=season,
            grade_override=grade_override,
            threshold_percent=threshold_percent,
            simulations=simulations,
            timeout_seconds=timeout_seconds,
            bundle_dir=bundle_dir.resolve(),
        )
