"""Stable three-stage arena opponent selection with confidence information."""

from __future__ import annotations

import math
from statistics import NormalDist
from dataclasses import dataclass

STAGE_NUMBERS = (1, 2, 3)
VISIBLE_OPPONENT_FAMILY_SIZE = 3
FAMILYWISE_DECISION_RULE = "bonferroni_one_sided_wilson_lower"
HIGHEST_WIN_RATE_FALLBACK_RULE = "highest_observed_win_rate_fallback"


@dataclass(frozen=True)
class StageEstimate:
    stage_number: int
    stage_id: int
    wins: int
    losses: int
    ties: int
    trials: int
    first_place_ties: int = 0

    def __post_init__(self) -> None:
        if self.stage_number not in STAGE_NUMBERS:
            raise ValueError("stage_number must be 1, 2 or 3")
        if self.stage_id < 1:
            raise ValueError("stage_id must be positive")
        _validate_counts(self.wins, self.losses, self.ties, self.trials)
        if not 0 <= self.first_place_ties <= self.trials:
            raise ValueError("first_place_ties must be between zero and trials")

    @property
    def win_rate(self) -> float:
        return self.wins / self.trials


@dataclass(frozen=True)
class OpponentEstimate:
    opponent_id: str
    position: int
    wins: int
    losses: int
    ties: int
    trials: int
    stages: tuple[StageEstimate, ...] = ()

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("position must be non-negative")
        _validate_counts(self.wins, self.losses, self.ties, self.trials)
        if self.stages:
            if tuple(stage.stage_number for stage in self.stages) != STAGE_NUMBERS:
                raise ValueError("stage estimates must be ordered as stages 1, 2 and 3")
            if len({stage.stage_id for stage in self.stages}) != len(STAGE_NUMBERS):
                raise ValueError("stage estimate IDs must be unique")
            if any(stage.trials != self.trials for stage in self.stages):
                raise ValueError("stage estimate trials must match match trials")

    @property
    def win_rate(self) -> float:
        # A tied match is deliberately not promoted to a win.
        return self.wins / self.trials


@dataclass(frozen=True)
class ArenaDecision:
    selected_position: int | None
    selected_opponent_id: str | None
    threshold: float
    confidence: float
    family_size: int
    familywise_confidence: float
    decision_rule: str
    estimates: tuple[dict[str, object], ...]


def _validate_counts(wins: int, losses: int, ties: int, trials: int) -> None:
    if min(wins, losses, ties) < 0 or trials < 1:
        raise ValueError("simulation counts must be non-negative and trials must be positive")
    if wins + losses + ties != trials:
        raise ValueError("wins, losses and ties must sum to trials")


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial estimate."""

    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes and trials are inconsistent")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z_squared / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def wilson_lower_bound(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    """Return a one-sided Wilson lower bound at significance ``alpha``."""

    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes and trials are inconsistent")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    z = NormalDist().inv_cdf(1 - alpha)
    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1 - proportion) + z_squared / (4 * trials))
            / trials
        )
        / denominator
    )
    return max(0.0, center - margin)


def _stage_row(estimate: StageEstimate, confidence: float) -> dict[str, float | int]:
    lower, upper = wilson_interval(estimate.wins, estimate.trials, confidence)
    return {
        "stage_number": estimate.stage_number,
        "stage_id": estimate.stage_id,
        "trials": estimate.trials,
        "wins": estimate.wins,
        "losses": estimate.losses,
        "ties": estimate.ties,
        "first_place_ties": estimate.first_place_ties,
        "win_rate": estimate.win_rate,
        "confidence_lower": lower,
        "confidence_upper": upper,
    }


def select_first_qualified(
    estimates: list[OpponentEstimate],
    *,
    threshold: float,
    confidence: float = 0.95,
    family_size: int = VISIBLE_OPPONENT_FAMILY_SIZE,
) -> ArenaDecision:
    """Prefer the first qualified opponent, otherwise the highest win rate."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if len({estimate.position for estimate in estimates}) != len(estimates):
        raise ValueError("opponent positions must be unique")
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    if len(estimates) > family_size:
        raise ValueError("estimate count exceeds the declared opponent family")

    family_alpha = 1 - confidence
    individual_alpha = family_alpha / family_size

    rows: list[dict[str, object]] = []
    selected: OpponentEstimate | None = None
    for estimate in sorted(estimates, key=lambda item: item.position):
        lower, upper = wilson_interval(estimate.wins, estimate.trials, confidence)
        qualification_lower = wilson_lower_bound(
            estimate.wins,
            estimate.trials,
            alpha=individual_alpha,
        )
        qualified = qualification_lower >= threshold
        rows.append(
            {
                "opponent_id": estimate.opponent_id,
                "position": estimate.position,
                "trials": estimate.trials,
                "wins": estimate.wins,
                "losses": estimate.losses,
                "ties": estimate.ties,
                "win_rate": estimate.win_rate,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "qualification_lower": qualification_lower,
                "qualification_alpha": individual_alpha,
                "qualified": qualified,
                "stages": tuple(_stage_row(stage, confidence) for stage in estimate.stages),
            }
        )
        if selected is None and qualified:
            selected = estimate

    decision_rule = FAMILYWISE_DECISION_RULE
    if selected is None and estimates:
        selected = max(estimates, key=lambda item: (item.win_rate, -item.position))
        decision_rule = HIGHEST_WIN_RATE_FALLBACK_RULE

    return ArenaDecision(
        selected_position=selected.position if selected else None,
        selected_opponent_id=selected.opponent_id if selected else None,
        threshold=threshold,
        confidence=confidence,
        family_size=family_size,
        familywise_confidence=confidence,
        decision_rule=decision_rule,
        estimates=tuple(rows),
    )
