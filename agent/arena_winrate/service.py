"""Fail-closed orchestration for arena observation, simulation and decision."""

from __future__ import annotations

from typing import Any, Mapping, Callable, Protocol
from pathlib import Path
from dataclasses import replace, dataclass

from .config import DEFAULT_SIMULATIONS
from .schema import SnapshotValidationError, validate_snapshot, validate_own_snapshot
from .adapter import (
    MATCH_RULE,
    PROTOCOL_VERSION,
    SCORE_AGGREGATION,
    OWN_SCORE_AGGREGATION,
    AdapterError,
    OwnScoreBatch,
    SimulationBatch,
)
from .decision import ArenaDecision, select_first_qualified
from .own_cache import OwnScoreCacheError, OwnScoreCacheStore, OwnScoreResimulationRequired
from .challenge_flow import contest_day_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_CACHE = PROJECT_ROOT / ".local" / "arena-win-rate" / "worker-calibration-v8.json"

# A forced preparation and the challenge loop are separate Maa custom actions.
# Keep the prepared capture's day through that loop so a 04:00 boundary in the
# gap or between tickets cannot silently reuse the just-written cache.
_PREPARED_CACHE_DAYS: dict[Path, tuple[str, str]] = {}


class SnapshotProvider(Protocol):
    def read(self) -> Mapping[str, Any]: ...


class OwnSnapshotProvider(Protocol):
    def read_own(self) -> Mapping[str, Any]: ...


class ArenaAdapter(Protocol):
    def simulate(self, request: Mapping[str, Any]) -> SimulationBatch: ...


class OwnScoreAdapter(Protocol):
    def simulate_own(self, request: Mapping[str, Any]) -> OwnScoreBatch: ...


@dataclass(frozen=True)
class ArenaProviderAttemptFailure:
    """Image-free diagnostic for one rejected snapshot-provider attempt."""

    attempt: int
    error_type: str
    code: str | None
    detail: str


@dataclass(frozen=True)
class ArenaEvaluation:
    status: str
    attempts: int
    safe_to_click: bool
    decision: ArenaDecision | None = None
    distribution_summary: dict[str, object] | None = None
    error: str | None = None
    provider_attempt_failures: tuple[ArenaProviderAttemptFailure, ...] = ()


@dataclass(frozen=True)
class ArenaOwnScoreEvaluation:
    status: str
    attempts: int
    stage_distributions: tuple[dict[str, object], ...] = ()
    parallelism: dict[str, object] | None = None
    cache_path: str | None = None
    error: str | None = None
    error_detail: str | None = None
    summary_warning: str | None = None
    summary_error_detail: str | None = None


class ArenaOwnScoreService:
    """Independently calculate the player's three raw team-score distributions."""

    def __init__(
        self,
        adapter: OwnScoreAdapter,
        *,
        simulations: int = DEFAULT_SIMULATIONS,
        seed: int = 400,
        calibration_cache_path: str | Path = DEFAULT_CALIBRATION_CACHE,
        cache_store: OwnScoreCacheStore | None = None,
    ) -> None:
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1000:
            raise ValueError("simulations must be an integer of at least 1000")
        self.adapter = adapter
        self.simulations = simulations
        self.seed = seed
        self.calibration_cache_path = Path(calibration_cache_path).resolve()
        self.cache_store = cache_store

    def calculate(
        self,
        provider: OwnSnapshotProvider,
        *,
        before_cache_save: Callable[[], None] | None = None,
    ) -> ArenaOwnScoreEvaluation:
        snapshot: dict[str, Any] | None = None
        failure: Exception | None = None
        for attempt in range(1, 3):
            try:
                observation = provider.read_own()
                snapshot = validate_own_snapshot(observation)
                break
            except Exception as error:
                failure = error
        if snapshot is None:
            return ArenaOwnScoreEvaluation(
                status="incomplete_input",
                attempts=2,
                error=str(failure),
            )
        request = {
            "schema_version": PROTOCOL_VERSION,
            "request_id": snapshot["capture_id"],
            "operation": "arena_own_score",
            "simulations": self.simulations,
            "seed": self.seed,
            "season": snapshot["season"],
            "stageIds": snapshot["stageIds"],
            "score_aggregation": OWN_SCORE_AGGREGATION,
            "parallelism": {
                "mode": "auto",
                "cache_path": str(self.calibration_cache_path),
            },
            "own_team": snapshot["own_team"],
        }
        try:
            batch = self.adapter.simulate_own(request)
        except AdapterError as error:
            return ArenaOwnScoreEvaluation(
                status="adapter_failure",
                attempts=attempt,
                error=str(error),
            )
        public_distributions = tuple(
            {key: value for key, value in stage.items() if key != "own_member_scores"}
            for stage in batch.stage_distributions
        )
        cache_path: str | None = None
        summary_warning: str | None = None
        summary_error_detail: str | None = None
        if self.cache_store is not None:
            try:
                if before_cache_save is not None:
                    before_cache_save()
                self.cache_store.save(
                    snapshot,
                    batch,
                    simulations=self.simulations,
                    seed=self.seed,
                )
            except OwnScoreCacheError as error:
                return ArenaOwnScoreEvaluation(
                    status="cache_failure",
                    attempts=attempt,
                    stage_distributions=public_distributions,
                    parallelism=batch.parallelism,
                    error=str(error),
                    error_detail=error.technical_detail or str(error),
                )
            cache_path = str(self.cache_store.path)
            summary_warning = self.cache_store.last_summary_error
            summary_error_detail = self.cache_store.last_summary_error_detail
        return ArenaOwnScoreEvaluation(
            status="calculated",
            attempts=attempt,
            stage_distributions=public_distributions,
            parallelism=batch.parallelism,
            cache_path=cache_path,
            summary_warning=summary_warning,
            summary_error_detail=summary_error_detail,
        )


def reset_prepared_own_score_cache(cache_store: OwnScoreCacheStore) -> None:
    """Invalidate any preparation marker left by an earlier Maa task run."""

    _PREPARED_CACHE_DAYS.pop(cache_store.path, None)


def prepare_own_score_cache(
    adapter: OwnScoreAdapter,
    provider: OwnSnapshotProvider,
    cache_store: OwnScoreCacheStore,
    *,
    season: int,
    stage_ids: tuple[int, int, int],
    simulations: int,
    seed: int,
    expected_upstream_commit: str,
    force_recalculate: bool,
    require_prepared: bool = False,
) -> tuple[
    ArenaOwnScoreEvaluation | None,
    tuple[dict[str, Any], dict[str, Any]] | None,
]:
    """Optionally replace the cache, then load the exact cache used for a match."""

    if force_recalculate and require_prepared:
        raise ValueError("force_recalculate and require_prepared cannot both be true")
    cache_path = cache_store.path
    if force_recalculate:
        reset_prepared_own_score_cache(cache_store)
    preparation_day = contest_day_key()
    evaluation: ArenaOwnScoreEvaluation | None = None

    def ensure_preparation_day() -> None:
        if contest_day_key() != preparation_day:
            raise OwnScoreCacheError(
                "竞技场日期在己方缓存落盘前跨过了每日 04:00 刷新边界；"
                "新缓存未落盘，请重新运行任务"
            )

    if force_recalculate:
        evaluation = ArenaOwnScoreService(
            adapter,
            simulations=simulations,
            seed=seed,
            cache_store=cache_store,
        ).calculate(provider, before_cache_save=ensure_preparation_day)
        if evaluation.status != "calculated":
            return evaluation, None
    try:
        cached = cache_store.load_for_match(
            season=season,
            stage_ids=stage_ids,
            simulations=simulations,
            seed=seed,
            expected_upstream_commit=expected_upstream_commit,
        )
    except OwnScoreResimulationRequired as error:
        if force_recalculate and evaluation is not None:
            return (
                replace(
                    evaluation,
                    status="cache_failure",
                    cache_path=None,
                    error=str(error),
                    error_detail=error.technical_detail or str(error),
                ),
                None,
            )
        snapshot = cache_store.load_snapshot_for_resimulation(
            season=season,
            stage_ids=stage_ids,
        )
        if snapshot is None:
            cached = None
        else:
            class CachedOwnSnapshotProvider:
                def read_own(self) -> Mapping[str, Any]:
                    return snapshot

            evaluation = ArenaOwnScoreService(
                adapter,
                simulations=simulations,
                seed=seed,
                cache_store=cache_store,
            ).calculate(
                CachedOwnSnapshotProvider(),
                before_cache_save=ensure_preparation_day,
            )
            if evaluation.status != "calculated":
                return evaluation, None
            try:
                cached = cache_store.load_for_match(
                    season=season,
                    stage_ids=stage_ids,
                    simulations=simulations,
                    seed=seed,
                    expected_upstream_commit=expected_upstream_commit,
                )
            except OwnScoreCacheError as error:
                return (
                    replace(
                        evaluation,
                        status="cache_failure",
                        cache_path=None,
                        error=str(error),
                        error_detail=error.technical_detail or str(error),
                    ),
                    None,
                )
    except OwnScoreCacheError as error:
        if not force_recalculate or evaluation is None:
            raise
        return (
            replace(
                evaluation,
                status="cache_failure",
                cache_path=None,
                error=str(error),
                error_detail=error.technical_detail or str(error),
            ),
            None,
        )
    if force_recalculate and cached is not None:
        capture_id = str(cached[0]["capture_id"])
        _PREPARED_CACHE_DAYS[cache_path] = (preparation_day, capture_id)
    elif require_prepared:
        prepared = _PREPARED_CACHE_DAYS.get(cache_path)
        if prepared is None:
            raise OwnScoreCacheError(
                "本次任务没有成功准备己方缓存；请重新运行任务"
            )
        if cached is None:
            raise OwnScoreCacheError(
                "本次任务准备的己方缓存已缺失或与当前期数不兼容；请重新运行任务"
            )
        if prepared[1] != str(cached[0].get("capture_id")):
            raise OwnScoreCacheError(
                "本次任务准备的己方缓存已被其他内容替换；请重新运行任务"
            )
        if prepared[0] != preparation_day:
            raise OwnScoreCacheError(
                "竞技场日期在自动准备己方缓存后跨过了每日 04:00 刷新边界；请重新运行任务"
            )
    return evaluation, cached


class ArenaWinRateService:
    """Evaluate at most two captures and never infer missing loadout data."""

    def __init__(
        self,
        adapter: ArenaAdapter,
        *,
        threshold: float,
        simulations: int = DEFAULT_SIMULATIONS,
        confidence: float = 0.95,
        seed: int = 400,
        calibration_cache_path: str | Path = DEFAULT_CALIBRATION_CACHE,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1000:
            raise ValueError("simulations must be an integer of at least 1000")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.adapter = adapter
        self.threshold = threshold
        self.simulations = simulations
        self.confidence = confidence
        self.seed = seed
        self.calibration_cache_path = Path(calibration_cache_path).resolve()

    def evaluate(
        self,
        provider: SnapshotProvider,
        *,
        allow_click: bool = False,
        own_score_cache: Mapping[str, Any] | None = None,
    ) -> ArenaEvaluation:
        snapshot: dict[str, Any] | None = None
        validation_error: SnapshotValidationError | None = None
        observation_error: Exception | None = None
        provider_attempt_failures: list[ArenaProviderAttemptFailure] = []
        for attempt in range(1, 3):
            try:
                observation = provider.read()
            except Exception as error:
                observation_error = error
                raw_code = getattr(error, "code", None)
                raw_detail = getattr(error, "detail", error)
                provider_attempt_failures.append(
                    ArenaProviderAttemptFailure(
                        attempt=attempt,
                        error_type=type(error).__name__,
                        code=None if raw_code is None else str(raw_code),
                        detail=str(raw_detail),
                    )
                )
                continue
            observation_error = None
            try:
                snapshot = validate_snapshot(observation)
                break
            except SnapshotValidationError as error:
                validation_error = error
        if snapshot is None:
            if observation_error is not None:
                return ArenaEvaluation(
                    status="observation_failure",
                    attempts=2,
                    safe_to_click=False,
                    error=f"snapshot provider failed: {observation_error}",
                    provider_attempt_failures=tuple(provider_attempt_failures),
                )
            return ArenaEvaluation(
                status="incomplete_input",
                attempts=2,
                safe_to_click=False,
                error=str(validation_error),
                provider_attempt_failures=tuple(provider_attempt_failures),
            )

        request = {
            "schema_version": PROTOCOL_VERSION,
            "request_id": snapshot["capture_id"],
            "operation": "arena_match_win_rate",
            "simulations": self.simulations,
            "seed": self.seed,
            "season": snapshot["season"],
            "stageIds": snapshot["stageIds"],
            "score_aggregation": SCORE_AGGREGATION,
            "match_rule": MATCH_RULE,
            "parallelism": {
                "mode": "auto",
                "cache_path": str(self.calibration_cache_path),
            },
            "own_team": snapshot["own_team"],
            "opponents": snapshot["opponents"],
        }
        if own_score_cache is not None:
            request["own_score_cache"] = dict(own_score_cache)
        try:
            batch = self.adapter.simulate(request)
        except AdapterError as error:
            return ArenaEvaluation(
                status="adapter_failure",
                attempts=attempt,
                safe_to_click=False,
                error=str(error),
                provider_attempt_failures=tuple(provider_attempt_failures),
            )

        decision = select_first_qualified(
            list(batch.estimates),
            threshold=self.threshold,
            confidence=self.confidence,
        )
        distribution_summary = {
            "method": batch.method,
            "score_aggregation": batch.score_aggregation,
            "match_rule": batch.match_rule,
            "stages": batch.stage_distributions,
            "parallelism": batch.parallelism,
        }
        if decision.selected_position is None:
            return ArenaEvaluation(
                status="no_qualified_opponent",
                attempts=attempt,
                safe_to_click=False,
                decision=decision,
                distribution_summary=distribution_summary,
                provider_attempt_failures=tuple(provider_attempt_failures),
            )
        return ArenaEvaluation(
            status="selected" if allow_click else "dry_run_selected",
            attempts=attempt,
            safe_to_click=allow_click,
            decision=decision,
            distribution_summary=distribution_summary,
            provider_attempt_failures=tuple(provider_attempt_failures),
        )
