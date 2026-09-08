import json
import time
import ctypes
import random
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from dataclasses import asdict
from collections.abc import Mapping, Callable

from utils import logger
from maa.context import Context
from maa.pipeline import JClick, JActionType
from arena_winrate import (
    DEFAULT_OWN_SCORE_CACHE,
    AdapterError,
    ArenaReaderError,
    ArenaLineupReader,
    ArenaRuntimeConfig,
    OwnScoreCacheError,
    OwnScoreCacheStore,
    ArenaComponentError,
    ArenaWinRateService,
    ArenaOwnScoreService,
    SubprocessArenaAdapter,
    ArenaChallengeFlowError,
    ContestSeasonDefinition,
    ArenaChallengeRecordStore,
    contest_day_key,
    new_challenge_id,
    prepare_own_score_cache,
    resolve_arena_component,
    reset_prepared_own_score_cache,
    own_score_cache_prepared_for_current_task,
)
from maa.custom_action import CustomAction
from arena_winrate.config import resolve_cached_arena_grade
from arena_winrate.decision import HIGHEST_WIN_RATE_FALLBACK_RULE
from maa.agent.agent_server import AgentServer
from arena_winrate.user_messages import (
    ArenaUserStatus,
    describe_arena_error,
    own_score_user_status,
    win_rate_stop_user_status,
)
from arena_winrate.cost_fallback_logging import (
    cost_customization_fallback_log_payloads,
)
from arena_winrate.maa_challenge_actions import (
    ArenaChallengeRecordResultAction,
    ArenaChallengeVerifyRefreshAction,
    ArenaChallengeRetryCurrentBattleAction,
    challenge_result_saved,
    resume_pending_challenge,
)

from .arena_reader import MaaArenaReaderBackend

_CHALLENGE_CONFIG_CARRIERS = (
    "ChallengeStrategyConfig",
    "ChallengeSeasonConfig",
    "ChallengeGradeConfig",
    "ChallengeAutoRecalculateConfig",
)


def _administrator_process() -> bool:
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def _publish_status(context: Context, status: ArenaUserStatus, *, log_detail: str | None = None) -> bool:
    message = status.message
    if log_detail:
        message = f"{message}；技术细节：{log_detail}"
    getattr(logger, status.level)(message)
    focus_content = status.message.replace("\\", "＼").replace("/", "／")
    try:
        context.run_action(
            "ChallengeStatusNotification",
            pipeline_override={
                "ChallengeStatusNotification": {
                    "focus": {
                        "Node.ActionNode.Succeeded": {
                            "content": focus_content,
                            "display": ["log", "notification"],
                        }
                    }
                }
            },
        )
    except Exception as error:
        logger.warning(f"竞技场任务状态未能显示在前台，结果仍已写入日志: {error}")
    return False


def _stop_with_error(context: Context, summary: str, error: object | None = None) -> bool:
    detail = describe_arena_error(error) if error is not None else None
    message = summary if detail is None else f"{summary}；{detail}"
    return _publish_status(
        context,
        ArenaUserStatus(message, "error"),
        log_detail=None if error is None else str(error),
    )


def _report_unexpected_errors(method: Callable[..., bool]) -> Callable[..., bool]:
    @wraps(method)
    def wrapped(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            return method(self, context, argv)
        except Exception as error:
            return _stop_with_error(
                context,
                f"竞技场任务发生未处理异常 {type(error).__name__}，已安全停止",
                error,
            )

    return wrapped


def _resolved_challenge_params(context: Context) -> dict[str, object]:
    """Resolve base action parameters and independent UI configuration carriers."""

    node = context.get_node_data("ChallengeChoose")
    if not isinstance(node, Mapping):
        raise ValueError("ChallengeChoose node data is unavailable")

    action = node.get("action")
    action_param = action.get("param") if isinstance(action, Mapping) else None
    raw_base = (
        action_param.get("custom_action_param")
        if isinstance(action_param, Mapping)
        else None
    )
    if isinstance(raw_base, str):
        raw_base = json.loads(raw_base)
    if not isinstance(raw_base, Mapping):
        raise ValueError("ChallengeChoose custom_action_param must be an object")
    if any(not isinstance(key, str) for key in raw_base):
        raise ValueError("ChallengeChoose custom_action_param keys must be strings")

    resolved = dict(raw_base)
    carrier_by_key: dict[str, str] = {}
    for carrier_name in _CHALLENGE_CONFIG_CARRIERS:
        carrier = context.get_node_data(carrier_name)
        if not isinstance(carrier, Mapping):
            raise ValueError(f"{carrier_name} node data is unavailable")
        attached = carrier.get("attach")
        if isinstance(attached, str):
            attached = json.loads(attached)
        if not isinstance(attached, Mapping):
            raise ValueError(f"{carrier_name} attach must be an object")
        if any(not isinstance(key, str) for key in attached):
            raise ValueError(f"{carrier_name} attach keys must be strings")
        for key, value in attached.items():
            previous_carrier = carrier_by_key.get(key)
            if previous_carrier is not None:
                raise ValueError(
                    f"challenge parameter {key!r} is provided by both "
                    f"{previous_carrier} and {carrier_name}"
                )
            carrier_by_key[key] = carrier_name
            resolved[key] = value
    return resolved


def _log_own_score_evaluation(
    evaluation: object,
    *,
    trigger: str,
    cost_customization_fallbacks: tuple[dict[str, object], ...] = (),
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "arena_own_score_recalculation",
                "executed": True,
                "trigger": trigger,
                "evaluation": asdict(evaluation),
                "cost_customization_fallbacks": list(cost_customization_fallbacks),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _resolved_own_score_params(context: Context) -> dict[str, object]:
    """Resolve the own-only task without reading challenge mode or Grade."""

    resolved: dict[str, object] = {}
    for name, allowed in (
        ("ArenaOwnScoreRun", {"simulations", "timeout_seconds"}),
        ("ChallengeSeasonConfig", {"season"}),
    ):
        node = context.get_node_data(name)
        attached = node.get("attach") if isinstance(node, Mapping) else None
        if isinstance(attached, str):
            attached = json.loads(attached)
        if not isinstance(attached, Mapping) or set(attached) - allowed:
            raise ValueError(f"{name} attach contains invalid own-score parameters")
        resolved.update(attached)
    return resolved


@AgentServer.custom_action("ArenaOwnScoreRecalculate")
class ArenaOwnScoreRecalculate(CustomAction):
    """Read and cache only the own lineup; never dispatch challenge selection."""

    @_report_unexpected_errors
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del argv
        try:
            config = ArenaRuntimeConfig.from_mapping(_resolved_own_score_params(context))
        except (TypeError, ValueError) as error:
            return _stop_with_error(context, "己方总分重算参数无效，已安全停止", error)
        try:
            logger.info("正在解析竞技场组件；默认目录在本进程首次使用时会检查 RIS 生产版本")
            component = resolve_arena_component(config.bundle_dir, config.season)
            season = component.season
            for warning in component.warnings:
                logger.warning(warning)
            logger.info(
                "竞技场己方独立重算组件就绪: "
                f"状态={component.status}, RIS={component.commit[:12]}, 赛季={season.season}"
            )
            if config.season == "latest":
                logger.warning(
                    f"竞技场赛季使用 latest，当前 RIS 组件解析为第 {season.season} 期；"
                    "若与游戏当期不一致请在 UI 手动选择期数"
                )
            if season.preview:
                logger.warning(
                    "RIS 目录将所选赛季标为预览；将继续运行并保留警告，用户可在 UI 手动选择其他期数"
                )
            adapter = SubprocessArenaAdapter.from_bundle(
                component.bundle_dir,
                timeout_seconds=config.timeout_seconds,
            )
            cache_store = OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE)
            reset_prepared_own_score_cache(cache_store)
            backend = MaaArenaReaderBackend(context, season, component.bundle_dir)
            reader = ArenaLineupReader(backend, season)
            evaluation = ArenaOwnScoreService(
                adapter,
                simulations=config.simulations,
                cache_store=cache_store,
            ).calculate(reader)
            logger.info(
                json.dumps(
                    {
                        "event": "arena_own_read_metrics",
                        "status": evaluation.status,
                        **reader.last_read_metrics_summary(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            own_cost_fallbacks = (
                _log_cost_customization_fallbacks(reader, side="own")
                if evaluation.status in {"calculated", "adapter_failure", "cache_failure"}
                else ()
            )
            _log_own_score_evaluation(
                evaluation,
                trigger="manual_task",
                cost_customization_fallbacks=own_cost_fallbacks,
            )
            _publish_status(
                context,
                own_score_user_status(evaluation, simulations=config.simulations),
                log_detail=(
                    evaluation.error_detail or evaluation.error
                    if evaluation.status != "calculated"
                    else evaluation.summary_error_detail
                ),
            )
            return evaluation.status == "calculated"
        except (
            ArenaComponentError,
            ArenaReaderError,
            AdapterError,
            OwnScoreCacheError,
            OSError,
            ValueError,
        ) as error:
            return _stop_with_error(context, "己方总分重算失败，已安全停止", error)


def _log_cost_customization_fallbacks(
    reader: ArenaLineupReader,
    *,
    side: str,
) -> tuple[dict[str, object], ...]:
    """Log only assumptions belonging to the accepted reader attempt."""

    records = tuple(
        dict(value) for value in reader.last_cost_customization_fallbacks(side=side)
    )
    for payload in cost_customization_fallback_log_payloads(records, side=side):
        logger.warning(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return records


def _prepare_daily_own_score_cache(
    context: Context,
    *,
    season: ContestSeasonDefinition,
    bundle_dir: str | Path,
    adapter: SubprocessArenaAdapter,
    cache_store: OwnScoreCacheStore,
    simulations: int,
    trigger: str,
) -> bool:
    """Run the one fresh own read shared by both daily preparation entry points."""

    preparation_day = contest_day_key()
    backend = MaaArenaReaderBackend(context, season, bundle_dir)
    reader = ArenaLineupReader(backend, season)
    evaluation, cached = prepare_own_score_cache(
        adapter,
        reader,
        cache_store,
        season=season.season,
        stage_ids=season.stage_ids,
        simulations=simulations,
        seed=400,
        expected_upstream_commit=adapter.expected_upstream_commit,
        force_recalculate=True,
    )
    if evaluation is None:
        raise OwnScoreCacheError("automatic recalculation did not produce an evaluation")
    own_cost_fallbacks = (
        _log_cost_customization_fallbacks(reader, side="own")
        if evaluation.status in {"calculated", "adapter_failure", "cache_failure"}
        else ()
    )
    _log_own_score_evaluation(
        evaluation,
        trigger=trigger,
        cost_customization_fallbacks=own_cost_fallbacks,
    )
    if evaluation.status != "calculated":
        return _publish_status(
            context,
            own_score_user_status(evaluation, simulations=simulations),
            log_detail=evaluation.error_detail or evaluation.error,
        )
    if cached is None:
        return _stop_with_error(
            context,
            "每日挑战自动重算完成，但新缓存仍不可复用，已安全停止",
        )
    if contest_day_key() != preparation_day:
        return _stop_with_error(
            context,
            "每日挑战自动重算跨过了每日 04:00 刷新边界，请重新运行任务",
        )

    status = own_score_user_status(evaluation, simulations=simulations)
    getattr(logger, status.level)(f"每日挑战自动重算：{status.message}")
    if evaluation.summary_error_detail:
        logger.warning(
            "每日挑战自动重算的缓存摘要技术细节："
            f"{evaluation.summary_error_detail}"
        )
    return True


def _select_challenge_index(
    context: Context,
    *,
    index: int,
    description: str,
) -> bool:
    """Select one legacy opponent and fail closed when Maa did not click it."""

    result = context.run_task(
        "ChallengeIndex",
        pipeline_override={
            "ChallengeIndex": {
                "recognition": {"param": {"index": index}},
            }
        },
    )
    if not result or not result.status.succeeded:
        return _stop_with_error(
            context,
            f"{description}未完成：对手位置识别或点击失败，已安全停止",
        )
    return True


@AgentServer.custom_action("ChallengeResetOwnScorePreparation")
class ChallengeResetOwnScorePreparation(CustomAction):
    """Clear a marker left by an earlier task before any entry fallback."""

    @_report_unexpected_errors
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        if not resume_pending_challenge(context, argv):
            return False
        reset_prepared_own_score_cache(OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE))
        logger.info("已清除上一任务可能遗留的竞技场己方缓存准备状态")
        return True


@AgentServer.custom_action("ChallengePrepareOwnScore")
class ChallengePrepareOwnScore(CustomAction):
    """Optionally refresh the own-score cache once, before the challenge loop."""

    @_report_unexpected_errors
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        del argv
        params = _resolved_challenge_params(context)
        if params.get("mode", "win_rate") != "win_rate":
            return True
        if not resume_pending_challenge(context):
            return False

        auto_recalculate = params.get("auto_recalculate_own", False)
        if not isinstance(auto_recalculate, bool):
            return _stop_with_error(
                context,
                "每日挑战的自动重算己方数据参数必须为布尔值，已安全停止",
            )
        if not auto_recalculate:
            return True

        try:
            config = ArenaRuntimeConfig.from_mapping(params)
        except ValueError as error:
            return _stop_with_error(context, "每日挑战自动重算参数无效，已安全停止", error)
        try:
            logger.info("正在解析竞技场组件；默认目录在本进程首次使用时会检查 RIS 生产版本")
            component = resolve_arena_component(config.bundle_dir, config.season)
            season = component.season
            for warning in component.warnings:
                logger.warning(warning)
            logger.info(
                "竞技场组件就绪: "
                f"状态={component.status}, RIS={component.commit[:12]}, 赛季={season.season}"
            )
        except ArenaComponentError as error:
            return _stop_with_error(
                context,
                "每日挑战自动重算无法解析竞技场期数，已安全停止",
                error,
            )

        try:
            adapter = SubprocessArenaAdapter.from_bundle(
                component.bundle_dir,
                timeout_seconds=config.timeout_seconds,
            )
            cache_store = OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE)
            return _prepare_daily_own_score_cache(
                context,
                season=season,
                bundle_dir=component.bundle_dir,
                adapter=adapter,
                cache_store=cache_store,
                simulations=config.simulations,
                trigger="daily_option",
            )
        except (
            ArenaReaderError,
            AdapterError,
            OwnScoreCacheError,
            ArenaChallengeFlowError,
            OSError,
            ValueError,
        ) as error:
            return _stop_with_error(
                context,
                "每日挑战自动重算己方数据失败，已安全停止",
                error,
            )


@AgentServer.custom_action("ChallengeAuto")
class ChallengeAuto(CustomAction):
    """
    自动运行挑战
    """

    @_report_unexpected_errors
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """执行挑战自动选择与进入流程。

        默认使用完整编成读取和本地模拟器胜率路线，同时保留 Maa
        原有的综合力、随机和固定位置选择模式。己方手动重算使用
        同一读取器和缓存协议。

        Args:
            context: MAA 任务上下文，提供控制器、识别和任务执行能力。
            argv: Maa 自定义动作运行参数；配置从本次任务解析后的节点数据读取。

        Returns:
            True 表示动作执行完毕。
        """
        del argv
        try:
            params = _resolved_challenge_params(context)
        except (TypeError, ValueError) as error:
            return _stop_with_error(context, "竞技场任务参数无效，已安全停止", error)
        if params.get("mode", "win_rate") == "win_rate" and not resume_pending_challenge(context):
            return False

        mode_allowed = [
            "fixed",
            "random",
            "auto",
            "max",
            "min",
            "win_rate",
            "win_rate_recalculate_own",
        ]
        mode = params.get("mode", "win_rate")
        if mode not in mode_allowed:
            return _stop_with_error(
                context,
                f"竞技场任务模式 {mode!r} 无效，仅允许 {mode_allowed}，已安全停止",
            )

        if mode == "fixed":
            index = params.get("index", 1)
            if isinstance(index, bool) or not isinstance(index, int) or index not in {0, 1, 2}:
                return _stop_with_error(
                    context,
                    "固定挑战位置无效，仅允许第一、第二或第三位，已安全停止",
                )
            logger.info(f"固定选择挑战第 {index + 1} 位")
            return _select_challenge_index(
                context,
                index=index,
                description=f"固定选择第 {index + 1} 位对手",
            )

        if mode == "random":
            index = random.choice([0, 1, 2])
            logger.info(f"随机选择挑战第 {index + 1} 位")
            return _select_challenge_index(
                context,
                index=index,
                description=f"随机选择第 {index + 1} 位对手",
            )

        if mode in {"auto", "max", "min"}:
            image = context.tasker.controller.post_screencap().wait().get()
            detail = context.run_recognition(
                "ChallengeRating",
                image,
                pipeline_override={
                    "ChallengeRating": {
                        "recognition": "OCR",
                        "expected": "^\\d{3,6}$",
                        "roi": [54, 634, 233, 447],
                        "order_by": "Vertical",
                    }
                },
            )
            if not detail or not detail.hit:
                return _stop_with_error(context, "三个对手的综合力识别失败，已安全停止")
            results = detail.filtered_results or detail.all_results or ()
            ratings = [result.text for result in results]
            logger.info(f"挑战评分识别结果: {ratings}")
            ratings_int: list[int] = []
            for rating in ratings:
                try:
                    ratings_int.append(int(rating))
                except ValueError:
                    ratings_int.append(-1)
            if len(ratings_int) != 3 or all(rating == -1 for rating in ratings_int):
                return _stop_with_error(context, "三个对手的综合力识别不完整，已安全停止")

            if mode == "max":
                index = ratings_int.index(max(ratings_int))
                logger.info(f"选择挑战评分最高的第 {index + 1} 位")
            elif mode == "min":
                valid = [(index, rating) for index, rating in enumerate(ratings_int) if rating >= 0]
                if not valid:
                    return _stop_with_error(context, "三个对手的综合力均不可用，已安全停止")
                index = min(valid, key=lambda item: item[1])[0]
                logger.info(f"选择挑战评分最低的第 {index + 1} 位")
            else:
                own_detail = context.run_recognition(
                    "ChallengeRating",
                    image,
                    pipeline_override={
                        "ChallengeRating": {
                            "recognition": "OCR",
                            "expected": "^\\d{3,6}$",
                            "roi": [172, 505, 195, 69],
                            "order_by": "Vertical",
                        }
                    },
                )
                if not own_detail or not own_detail.hit:
                    return _stop_with_error(context, "己方综合力识别失败，已安全停止")
                try:
                    own_rating = int(own_detail.best_result.text)
                except (AttributeError, TypeError, ValueError) as error:
                    return _stop_with_error(context, "己方综合力不是有效整数，已安全停止", error)
                differences = [
                    abs(rating - own_rating) if rating >= 0 else float("inf")
                    for rating in ratings_int
                ]
                index = differences.index(min(differences))
                logger.info(f"自动选择挑战第 {index + 1} 位")
            return _select_challenge_index(
                context,
                index=index,
                description=f"按综合力选择第 {index + 1} 位对手",
            )

        if mode in {"win_rate", "win_rate_recalculate_own"}:
            try:
                config = ArenaRuntimeConfig.from_mapping(params)
            except ValueError as error:
                return _stop_with_error(context, "竞技场胜率参数无效，已安全停止", error)
            auto_recalculate_own = False
            if mode == "win_rate":
                auto_recalculate_own = params.get("auto_recalculate_own", False)
                if not isinstance(auto_recalculate_own, bool):
                    return _stop_with_error(
                        context,
                        "每日挑战的自动重算己方数据参数必须为布尔值，已安全停止",
                    )
            logger.info(
                "竞技场胜率模式参数: "
                f"赛季={config.season}, 门槛={config.threshold_percent}%, 模拟次数={config.simulations}, "
                f"超时={config.timeout_seconds}s, Grade覆盖={config.grade_override}"
            )
            try:
                logger.info("正在解析竞技场组件；默认目录在本进程首次使用时会检查 RIS 生产版本")
                component = resolve_arena_component(config.bundle_dir, config.season)
                season = component.season
                for warning in component.warnings:
                    logger.warning(warning)
                logger.info(
                    "竞技场组件就绪: "
                    f"状态={component.status}, RIS={component.commit[:12]}, 赛季={season.season}"
                )
            except ArenaComponentError as error:
                return _stop_with_error(
                    context,
                    "竞技场组件无法安全满足所选期数，已停止",
                    error,
                )
            logger.info(f"竞技场赛季 {season.season} 对应场地 ID: {season.stage_ids}")
            if config.season == "latest":
                logger.warning(
                    f"竞技场赛季使用 latest，当前 RIS 组件解析为第 {season.season} 期；"
                    "若与游戏当期不一致请在 UI 手动选择期数"
                )
            if season.preview:
                logger.warning(
                    "RIS 目录将所选赛季标为预览；将继续运行并保留警告，用户可在 UI 手动选择其他期数"
                )
            try:
                contest_day = contest_day_key()
                adapter = SubprocessArenaAdapter.from_bundle(
                    component.bundle_dir,
                    timeout_seconds=config.timeout_seconds,
                )
                cache_store = OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE)
                if mode == "win_rate_recalculate_own":
                    backend = MaaArenaReaderBackend(context, season, component.bundle_dir)
                    reader = ArenaLineupReader(backend, season)
                    own_evaluation = ArenaOwnScoreService(
                        adapter,
                        simulations=config.simulations,
                        cache_store=cache_store,
                    ).calculate(reader)
                    own_cost_fallbacks = (
                        _log_cost_customization_fallbacks(reader, side="own")
                        if own_evaluation.status
                        in {"calculated", "adapter_failure", "cache_failure"}
                        else ()
                    )
                    _log_own_score_evaluation(
                        own_evaluation,
                        trigger="manual_task",
                        cost_customization_fallbacks=own_cost_fallbacks,
                    )
                    return _publish_status(
                        context,
                        own_score_user_status(
                            own_evaluation,
                            simulations=config.simulations,
                        ),
                        log_detail=(
                            own_evaluation.error_detail or own_evaluation.error
                            if own_evaluation.status != "calculated"
                            else own_evaluation.summary_error_detail
                        ),
                    )

                if (
                    auto_recalculate_own
                    and not own_score_cache_prepared_for_current_task(cache_store)
                ):
                    logger.info(
                        "每日挑战自动重算前置未在页面稳定前命中；"
                        "正在从已确认的竞技场主界面补做一次己方读取"
                    )
                    if not _prepare_daily_own_score_cache(
                        context,
                        season=season,
                        bundle_dir=component.bundle_dir,
                        adapter=adapter,
                        cache_store=cache_store,
                        simulations=config.simulations,
                        trigger="daily_option_stable_entry_fallback",
                    ):
                        return False

                class CachedOnlyOwnSnapshotProvider:
                    def read_own(self) -> Mapping[str, object]:
                        raise OwnScoreCacheError(
                            "daily opponent read unexpectedly requested a fresh own-team capture"
                        )

                try:
                    own_cache_evaluation, cached = prepare_own_score_cache(
                        adapter,
                        CachedOnlyOwnSnapshotProvider(),
                        cache_store,
                        season=season.season,
                        stage_ids=season.stage_ids,
                        simulations=config.simulations,
                        seed=400,
                        expected_upstream_commit=adapter.expected_upstream_commit,
                        force_recalculate=False,
                        require_prepared=auto_recalculate_own,
                    )
                except OwnScoreCacheError as error:
                    if auto_recalculate_own:
                        return _stop_with_error(
                            context,
                            "自动重算己方数据未能安全衔接到胜率计算，已停止",
                            error,
                        )
                    return _stop_with_error(
                        context,
                        "己方分数缓存不可复用；可在胜率选敌设置中临时启用"
                        "“自动重算己方数据”，或手动运行“重算竞技场己方总分”",
                        error,
                    )
                if own_cache_evaluation is not None:
                    _log_own_score_evaluation(
                        own_cache_evaluation,
                        trigger="cached_lineup_resimulation",
                    )
                    if own_cache_evaluation.status != "calculated":
                        return _publish_status(
                            context,
                            own_score_user_status(
                                own_cache_evaluation,
                                simulations=config.simulations,
                            ),
                            log_detail=(
                                own_cache_evaluation.error_detail
                                or own_cache_evaluation.error
                            ),
                        )
                if cached is None:
                    return _stop_with_error(
                        context,
                        "所选赛季没有可复用的己方分数缓存；可在每日挑战的胜率选敌设置中启用"
                        "“自动重算己方数据”，或先手动运行“重算竞技场己方总分”"
                    )
                own_snapshot, own_score_cache = cached
                effective_grade, grade_state = resolve_cached_arena_grade(
                    cache_store,
                    None if auto_recalculate_own else config.grade_override,
                )
                logger.info(
                    "竞技场 Grade 已从己方缓存复用: "
                    f"有效值={effective_grade}, 来源={grade_state.source}, "
                    f"识别值={grade_state.recognized_grade}, 覆盖值={grade_state.grade_override}"
                )
                backend = MaaArenaReaderBackend(
                    context,
                    season,
                    component.bundle_dir,
                    known_grade=effective_grade,
                )
                reader = ArenaLineupReader(backend, season)
                opponent_read_attempts: list[dict[str, object]] = []

                class OpponentSnapshotProvider:
                    snapshot: dict[str, object] | None = None

                    def read(self) -> dict[str, object]:
                        started = time.perf_counter()
                        succeeded = False
                        try:
                            self.snapshot = reader.read_opponents(own_snapshot)
                            succeeded = True
                            return self.snapshot
                        finally:
                            opponent_read_attempts.append({
                                "attempt": len(opponent_read_attempts) + 1,
                                "succeeded": succeeded,
                                "read_wall_seconds": round(time.perf_counter() - started, 6),
                            })

                provider = OpponentSnapshotProvider()

                evaluation = ArenaWinRateService(
                    adapter,
                    threshold=config.threshold,
                    simulations=config.simulations,
                ).evaluate(
                    provider,
                    allow_click=True,
                    own_score_cache=own_score_cache,
                )
            except (
                ArenaReaderError,
                AdapterError,
                OwnScoreCacheError,
                ArenaChallengeFlowError,
                OSError,
                ValueError,
            ) as error:
                return _stop_with_error(
                    context,
                    "竞技场只读编成读取或模拟初始化失败，已安全停止",
                    error,
                )
            opponent_cost_fallbacks = (
                _log_cost_customization_fallbacks(
                    reader,
                    side="opponent",
                )
                if provider.snapshot is not None
                else ()
            )
            opponent_read_summary = {
                "read_wall_seconds": (
                    None if provider.snapshot is None
                    else provider.snapshot.get("read_wall_seconds")
                ),
                "team_read_wall_seconds": [
                    opponent.get("read_wall_seconds")
                    for opponent in (
                        [] if provider.snapshot is None
                        else provider.snapshot["opponents"]
                    )
                ],
                **reader.last_read_metrics_summary(),
                "read_attempts": opponent_read_attempts,
                # Backend totals include unfinished members and earlier
                # failed attempts, which completed-member reports omit.
                "runtime_metrics_all_attempts": backend.runtime_metrics(),
            }
            logger.info(
                json.dumps(
                    {
                        "event": "arena_win_rate_evaluation",
                        "executed": False,
                        "source_capture_id": (
                            None
                            if provider.snapshot is None
                            else provider.snapshot.get("capture_id")
                        ),
                        "evaluation": asdict(evaluation),
                        "provider_attempts": evaluation.attempts,
                        "opponent_read_summary": opponent_read_summary,
                        "cost_customization_fallbacks": list(opponent_cost_fallbacks),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if (
                not evaluation.safe_to_click
                or evaluation.decision is None
                or evaluation.decision.selected_position is None
                or provider.snapshot is None
            ):
                return _publish_status(
                    context,
                    win_rate_stop_user_status(evaluation),
                    log_detail=evaluation.error,
                )
            if contest_day_key() != contest_day:
                return _stop_with_error(
                    context,
                    "竞技场读取／模拟跨过了每日 04:00 刷新边界，已停止且不点击",
                )
            if not _administrator_process():
                return _stop_with_error(
                    context,
                    "实际 Maa 输入进程不是管理员权限，已停止且不点击",
                )

            source_capture_id = str(provider.snapshot["capture_id"])
            selected_position = evaluation.decision.selected_position
            challenge_id = new_challenge_id(selected_position)
            selected_row = next(
                row
                for row in evaluation.decision.estimates
                if row["position"] == selected_position
            )
            if evaluation.decision.decision_rule == HIGHEST_WIN_RATE_FALLBACK_RULE:
                logger.info(
                    "竞技场未有对手达到 "
                    f"{evaluation.decision.threshold * 100:.0f}% 门槛，"
                    f"按最高预测胜率选择第 {selected_position + 1} 位对手"
                    f"（胜率 {float(selected_row['win_rate']) * 100:.2f}%）"
                )
            record_store = ArenaChallengeRecordStore()
            begun = False
            try:
                record_store.begin(
                    {
                        "capture_id": challenge_id,
                        "source_capture_id": source_capture_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "contest_day": contest_day,
                        "season": season.season,
                        "stageIds": list(season.stage_ids),
                        "selected_position": selected_position,
                        "selected_opponent_id": evaluation.decision.selected_opponent_id,
                        "decision_rule": evaluation.decision.decision_rule,
                        "threshold": evaluation.decision.threshold,
                        "estimate": selected_row,
                        "estimates": [
                            dict(row) for row in evaluation.decision.estimates
                        ],
                        "provider_attempts": evaluation.attempts,
                        "opponent_read_summary": opponent_read_summary,
                        "cost_customization_fallbacks": list(opponent_cost_fallbacks),
                    }
                )
                begun = True
                guard = backend.select_opponent_for_challenge(selected_position)
                if contest_day_key() != contest_day:
                    raise ArenaChallengeFlowError(
                        "contest day changed after opponent selection and before start"
                    )
            except Exception as error:
                if begun:
                    record_store.abort(challenge_id)
                try:
                    backend.recover_to_arena_main(require_opponents=True)
                except Exception as recovery_error:
                    logger.error(f"挑战点击前置失败且竞技场主界面恢复失败: {recovery_error}")
                return _stop_with_error(
                    context,
                    "竞技场挑战点击前置失败，已停止且未点击开始",
                    error,
                )
            logger.info(
                json.dumps(
                    {
                        "event": "arena_challenge_selection_committed",
                        "executed": True,
                        "challenge_id": challenge_id,
                        "source_capture_id": source_capture_id,
                        "guard": guard,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return True

        return _stop_with_error(context, "竞技场任务未进入任何受支持的执行路径，已安全停止")


@AgentServer.custom_action("ArenaChallengeRecordResult")
class ArenaChallengeRecordResult(ArenaChallengeRecordResultAction):
    pass


@AgentServer.custom_action("ArenaChallengeVerifyRefresh")
class ArenaChallengeVerifyRefresh(ArenaChallengeVerifyRefreshAction):
    pass


@AgentServer.custom_action("ArenaChallengeRetryCurrentBattle")
class ArenaChallengeRetryCurrentBattle(ArenaChallengeRetryCurrentBattleAction):
    pass


@AgentServer.custom_action("ArenaChallengeResumePending")
class ArenaChallengeResumePending(CustomAction):
    @_report_unexpected_errors
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        return resume_pending_challenge(context, argv)


def _challenge_recognized_click(context: Context, argv: CustomAction.RunArg) -> bool:
    """Keep Maa's original recognized target without recapturing or re-OCR."""
    result = context.run_action_direct(JActionType.Click, JClick(), argv.box)
    return bool(result is not None and result.success)


@AgentServer.custom_action("ArenaChallengeStartOnce")
class ArenaChallengeStartOnce(CustomAction):
    @_report_unexpected_errors
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        if not _administrator_process():
            return False
        store = ArenaChallengeRecordStore()
        pending = store.load_pending()
        if pending is not None:
            try:
                # Persist before sending: a lost response cannot rearm Start.
                store.mark_battle_started(str(pending.get("capture_id", "")))
            except (ArenaChallengeFlowError, OSError) as error:
                return _stop_with_error(context, "当前对局已发送开始或无法登记，未重复挑战", error)
        return _challenge_recognized_click(context, argv)


@AgentServer.custom_action("ArenaChallengeLeaveResult")
class ArenaChallengeLeaveResult(CustomAction):
    @_report_unexpected_errors
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        if not _administrator_process():
            return False
        store = ArenaChallengeRecordStore()
        if store.load_pending() is not None and not challenge_result_saved(store):
            return _stop_with_error(context, "竞技场结果尚未保存，保留当前页面继续结果恢复")
        if store.load_pending() is not None and getattr(argv, "node_name", "") == "ChallengeError":
            return resume_pending_challenge(context, argv, store)
        return _challenge_recognized_click(context, argv)
