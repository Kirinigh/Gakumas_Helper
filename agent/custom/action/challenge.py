import json
import ctypes
import random
from datetime import datetime, timezone
from functools import wraps
from dataclasses import asdict
from collections.abc import Mapping, Callable

from utils import logger
from maa.context import Context
from arena_winrate import (
    DEFAULT_OWN_SCORE_CACHE,
    AdapterError,
    ArenaReaderError,
    ArenaLineupReader,
    StageCatalogError,
    ArenaRuntimeConfig,
    OwnScoreCacheError,
    OwnScoreCacheStore,
    ArenaWinRateService,
    ContestStageCatalog,
    ArenaOwnScoreService,
    SubprocessArenaAdapter,
    ArenaChallengeFlowError,
    ArenaChallengeRecordStore,
    contest_day_key,
    new_challenge_id,
    prepare_own_score_cache,
    reset_prepared_own_score_cache,
)
from maa.custom_action import CustomAction
from maa.agent.agent_server import AgentServer
from arena_winrate.user_messages import (
    ArenaUserStatus,
    describe_arena_error,
    own_score_user_status,
    win_rate_stop_user_status,
)
from arena_winrate.maa_challenge_actions import (
    ArenaChallengeRecordResultAction,
    ArenaChallengeVerifyRefreshAction,
)

from .arena_reader import MaaArenaReaderBackend


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
    """Read the fully merged ChallengeChoose parameters for this task run."""

    node = context.get_node_data("ChallengeChoose")
    if not isinstance(node, Mapping):
        raise ValueError("ChallengeChoose node data is unavailable")
    raw = (
        node.get("action", {})
        .get("param", {})
        .get("custom_action_param", {})
    )
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Mapping):
        raise ValueError("ChallengeChoose custom_action_param must be an object")
    return dict(raw)


def _log_own_score_evaluation(
    evaluation: object,
    *,
    trigger: str,
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "arena_own_score_recalculation",
                "executed": True,
                "trigger": trigger,
                "evaluation": asdict(evaluation),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


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
        del context, argv
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
            season = ContestStageCatalog.from_bundle(config.bundle_dir).resolve(config.season)
        except StageCatalogError as error:
            return _stop_with_error(
                context,
                "每日挑战自动重算无法解析竞技场期数，已安全停止",
                error,
            )

        contest_day = contest_day_key()
        try:
            backend = MaaArenaReaderBackend(context, season, config.bundle_dir)
            reader = ArenaLineupReader(backend, season)
            adapter = SubprocessArenaAdapter.from_bundle(
                config.bundle_dir,
                timeout_seconds=config.timeout_seconds,
            )
            cache_store = OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE)
            evaluation, cached = prepare_own_score_cache(
                adapter,
                reader,
                cache_store,
                season=season.season,
                stage_ids=season.stage_ids,
                simulations=config.simulations,
                seed=400,
                expected_upstream_commit=adapter.expected_upstream_commit,
                force_recalculate=True,
            )
            if evaluation is None:
                raise OwnScoreCacheError("automatic recalculation did not produce an evaluation")
            _log_own_score_evaluation(evaluation, trigger="daily_option")
            if evaluation.status != "calculated":
                return _publish_status(
                    context,
                    own_score_user_status(
                        evaluation,
                        simulations=config.simulations,
                    ),
                    log_detail=evaluation.error_detail or evaluation.error,
                )

            status = own_score_user_status(
                evaluation,
                simulations=config.simulations,
            )
            getattr(logger, status.level)(f"每日挑战自动重算：{status.message}")
            if evaluation.summary_error_detail:
                logger.warning(
                    "每日挑战自动重算的缓存摘要技术细节："
                    f"{evaluation.summary_error_detail}"
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
        if cached is None:
            return _stop_with_error(
                context,
                "每日挑战自动重算完成，但新缓存仍不可复用，已安全停止",
            )
        if contest_day_key() != contest_day:
            return _stop_with_error(
                context,
                "每日挑战自动重算跨过了每日 04:00 刷新边界，请重新运行任务",
            )
        return True


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
            argv: 自定义动作运行参数，需包含 JSON 格式的 custom_action_param。

        Returns:
            True 表示动作执行完毕。
        """
        try:
            params = json.loads(argv.custom_action_param)
        except (TypeError, json.JSONDecodeError) as error:
            return _stop_with_error(context, "竞技场任务参数不是有效 JSON，已安全停止", error)
        if not isinstance(params, dict):
            return _stop_with_error(context, "竞技场任务参数必须是 JSON 对象，已安全停止")

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
                f"超时={config.timeout_seconds}s"
            )
            try:
                season = ContestStageCatalog.from_bundle(config.bundle_dir).resolve(config.season)
            except StageCatalogError as error:
                return _stop_with_error(
                    context,
                    "竞技场赛季无法由固定模拟器目录解析，已安全停止",
                    error,
                )
            logger.info(f"竞技场赛季 {season.season} 对应场地 ID: {season.stage_ids}")
            if config.season == "latest":
                logger.warning(
                    "竞技场赛季使用固定目录 latest 默认值；将继续运行，若与游戏当期不一致请在 UI 手动选择期数"
                )
            if season.preview:
                logger.warning(
                    "固定目录将所选赛季标为预览；将继续运行并保留警告，用户可在 UI 手动选择其他期数"
                )
            try:
                contest_day = contest_day_key()
                backend = MaaArenaReaderBackend(context, season, config.bundle_dir)
                reader = ArenaLineupReader(backend, season)
                adapter = SubprocessArenaAdapter.from_bundle(
                    config.bundle_dir,
                    timeout_seconds=config.timeout_seconds,
                )
                cache_store = OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE)
                if mode == "win_rate_recalculate_own":
                    own_evaluation = ArenaOwnScoreService(
                        adapter,
                        simulations=config.simulations,
                        cache_store=cache_store,
                    ).calculate(reader)
                    _log_own_score_evaluation(own_evaluation, trigger="manual_task")
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

                try:
                    _, cached = prepare_own_score_cache(
                        adapter,
                        reader,
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
                if cached is None:
                    return _stop_with_error(
                        context,
                        "所选赛季没有可复用的己方分数缓存；可在每日挑战的胜率选敌设置中启用"
                        "“自动重算己方数据”，或先手动运行“重算竞技场己方总分”"
                    )
                own_snapshot, own_score_cache = cached

                class OpponentSnapshotProvider:
                    snapshot: dict[str, object] | None = None

                    def read(self) -> dict[str, object]:
                        self.snapshot = reader.read_opponents(own_snapshot)
                        return self.snapshot

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
            logger.info(
                json.dumps(
                    {
                        "event": "arena_win_rate_evaluation",
                        "executed": False,
                        "evaluation": asdict(evaluation),
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
