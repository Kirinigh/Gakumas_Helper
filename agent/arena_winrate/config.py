"""Validated runtime parameters for the arena win-rate UI mode."""

from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "assets" / "arena-winrate"
DEFAULT_THRESHOLD_PERCENT = 70
DEFAULT_SIMULATIONS = 2000
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_SEASON_SELECTION = "latest"


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


@dataclass(frozen=True)
class ArenaRuntimeConfig:
    season: str | int = DEFAULT_SEASON_SELECTION
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
            threshold_percent=threshold_percent,
            simulations=simulations,
            timeout_seconds=timeout_seconds,
            bundle_dir=bundle_dir.resolve(),
        )
