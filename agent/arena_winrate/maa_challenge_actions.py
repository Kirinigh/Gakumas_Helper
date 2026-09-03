"""Reusable Maa actions for one arena result and post-finish verification."""

from __future__ import annotations

import re
import json
import time
import ctypes
from datetime import datetime, timezone

from utils import logger
from maa.context import Context
from maa.custom_action import CustomAction

from .reader import ArenaPageState, classify_arena_page
from .challenge_flow import (
    ArenaChallengeFlowError,
    ArenaChallengeRecordStore,
    contest_day_key,
    classify_post_challenge_page,
    result_observations_complete,
    serialise_result_observations,
)


def _administrator_process() -> bool:
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def _capture(context: Context):
    return context.tasker.controller.post_screencap().wait().get()


def _full_ocr(context: Context, image):
    detail = context.run_recognition(
        "ArenaReaderOCR",
        image,
        pipeline_override={
            "ArenaReaderOCR": {
                "recognition": "OCR",
                "expected": r".+",
                "order_by": "Vertical",
            }
        },
    )
    if not detail or not detail.hit:
        return ()
    return tuple(detail.all_results or detail.filtered_results or ())


class ArenaChallengeRecordResultAction(CustomAction):
    """Record low-dimensional result evidence before its button is clicked."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del argv
        try:
            if not _administrator_process():
                raise ArenaChallengeFlowError("Maa result process is not elevated")
            image = _capture(context)
            result = serialise_result_observations(_full_ocr(context, image))
            result.update(
                {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "contest_day": contest_day_key(),
                }
            )
            if not result_observations_complete(result):
                ArenaChallengeRecordStore().note_incomplete_result(result)
                raise ArenaChallengeFlowError(
                    "arena result OCR lacks a complete outcome and three stage scores"
                )
            path = ArenaChallengeRecordStore().complete(result)
            logger.info(
                json.dumps(
                    {
                        "event": "arena_challenge_result_recorded",
                        "result_path": str(path),
                        "result": result,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        except Exception as error:
            # The ticket may already be consumed.  The finish path is still
            # allowed to restore the arena main page, while the unresolved
            # pending record blocks another challenge.
            logger.error(f"竞技场结果页记录失败；将先完成返回且阻止下一次挑战: {error}")
        return True


class ArenaChallengeVerifyRefreshAction(CustomAction):
    """Close known reward pages, then verify the refreshed three-card page."""

    def __init__(self, record_store: ArenaChallengeRecordStore | None = None) -> None:
        super().__init__()
        self._record_store = (
            ArenaChallengeRecordStore() if record_store is None else record_store
        )

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del argv
        if not _administrator_process():
            logger.error("竞技场结束页 Maa 输入进程不是管理员权限，已停止")
            return False
        try:
            pending = self._record_store.load_pending()
        except ArenaChallengeFlowError as error:
            logger.error(f"竞技场待完成挑战记录不可读，已停止: {error}")
            return False
        if pending is None:
            logger.error("竞技场待完成挑战记录不存在，禁止独立补发结束页输入")
            return False
        pending_capture_id = str(pending.get("capture_id", ""))
        if not pending_capture_id:
            logger.error("竞技场待完成挑战记录缺少 capture_id，已停止")
            return False
        started = time.monotonic()
        deadline = started + 20.0
        reward_close_clicks = 0
        finish_resend_clicks = 0
        finish_proof_box: tuple[int, int, int, int] | None = None
        last_return_evidence: dict[str, object] = {}
        while time.monotonic() < deadline:
            image = _capture(context)
            detail = context.run_recognition("ArenaReaderOpponentCards", image)
            results = (
                ()
                if not detail or not detail.hit
                else tuple(detail.filtered_results or detail.all_results or ())
            )
            close_detail = context.run_recognition("CloseButton", image)
            close_results = (
                ()
                if not close_detail or not close_detail.hit
                else tuple(
                    close_detail.filtered_results or close_detail.all_results or ()
                )
            )
            finish_detail = context.run_recognition(
                "ChallengeFinish",
                image,
                pipeline_override={
                    "ChallengeFinish": {
                        "recognition": "TemplateMatch",
                        "template": "challenge_finish.png",
                        "roi": [0, 980, 720, 300],
                        "threshold": 0.95,
                        "order_by": "Score",
                    }
                },
            )
            finish_results = (
                ()
                if not finish_detail or not finish_detail.hit
                else tuple(
                    finish_detail.filtered_results
                    or finish_detail.all_results
                    or ()
                )
            )
            texts = ()
            if close_results or not (results or finish_results):
                texts = tuple(
                    str(getattr(item, "text", "")).strip()
                    for item in _full_ocr(context, image)
                )
            arena_state = ArenaPageState.AMBIGUOUS
            if not (results or close_results or finish_results):
                arena_state = classify_arena_page(
                    rehearsal_count=sum(text == "リハーサル" for text in texts),
                    opponent_count=0,
                    stage_label_count=sum(
                        re.fullmatch(r"ステージ\s*[123]", text) is not None
                        for text in texts
                    ),
                    stage_total_count=sum(text == "総合力" for text in texts),
                    exhausted_notice_count=sum(
                        text == "本日の挑戦権を消費しました" for text in texts
                    ),
                )
            page_state = classify_post_challenge_page(
                opponent_count=len(results),
                close_count=len(close_results),
                ocr_texts=texts,
                reward_close_clicks=reward_close_clicks,
                finish_count=len(finish_results),
                finish_resend_clicks=finish_resend_clicks,
                finish_resend_authorized=True,
                arena_main_exhausted=(
                    arena_state is ArenaPageState.OPPONENTS_UNAVAILABLE
                ),
            )
            last_return_evidence = {
                "opponent_count": len(results),
                "arena_page_state": (
                    ArenaPageState.OPPONENTS_UNAVAILABLE.value
                    if arena_state is ArenaPageState.OPPONENTS_UNAVAILABLE
                    else (
                        ArenaPageState.READY.value
                        if len(results) == 3
                        else ArenaPageState.AMBIGUOUS.value
                    )
                ),
                "daily_attempts_exhausted": (
                    arena_state is ArenaPageState.OPPONENTS_UNAVAILABLE
                ),
                "reward_close_clicks": reward_close_clicks,
                "finish_resend_clicks": finish_resend_clicks,
                "wall_seconds": round(time.monotonic() - started, 6),
                "screenshots_persisted": False,
            }
            if page_state == "RETURNED":
                try:
                    result_path = self._record_store.complete_return(
                        pending_capture_id,
                        evidence=last_return_evidence,
                    )
                except ArenaChallengeFlowError as error:
                    logger.error(f"竞技场已回场但挑战生命周期未能闭合，已停止: {error}")
                    return False
                logger.info(
                    json.dumps(
                        {
                            "event": "arena_challenge_return_verified",
                            "challenge_id": pending_capture_id,
                            "result_path": str(result_path),
                            **last_return_evidence,
                            "full_lineup_reread": False,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return True
            if page_state == "RESEND_FINISH":
                current_finish_box = tuple(int(value) for value in finish_results[0].box)
                if finish_proof_box is None or any(
                    abs(current - previous) > 2
                    for current, previous in zip(
                        current_finish_box,
                        finish_proof_box,
                        strict=True,
                    )
                ):
                    finish_proof_box = current_finish_box
                    time.sleep(0.12)
                    continue
                try:
                    current_pending = self._record_store.load_pending()
                except ArenaChallengeFlowError as error:
                    logger.error(f"竞技场待完成挑战记录复核失败，已停止: {error}")
                    return False
                if (
                    current_pending is None
                    or current_pending.get("capture_id") != pending_capture_id
                ):
                    logger.error("竞技场待完成挑战记录在结束页补发前已变化，已停止")
                    return False
                box = list(current_finish_box)
                click = context.tasker.controller.post_click(
                    box[0] + box[2] // 2,
                    box[1] + box[3] // 2,
                ).wait()
                finish_resend_clicks += 1
                if not click.status.succeeded:
                    logger.error("竞技场结束页补发点击未成功，已停止")
                    return False
                logger.warning(
                    json.dumps(
                        {
                            "event": "arena_challenge_finish_click_resent",
                            "capture_id": pending_capture_id,
                            "finish_resend_clicks": finish_resend_clicks,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                time.sleep(0.25)
                continue
            finish_proof_box = None
            if page_state == "AMBIGUOUS":
                try:
                    self._record_store.note_return_unverified(
                        pending_capture_id,
                        reason_code="post_challenge_page_ambiguous",
                        evidence=last_return_evidence,
                    )
                except ArenaChallengeFlowError as error:
                    logger.error(f"竞技场回场失败证据未能持久化: {error}")
                logger.error("竞技场结束后页面证据不唯一，已停止且不会继续下一轮")
                return False
            if page_state == "CLOSE_REWARD":
                box = list(close_results[0].box)
                click = context.tasker.controller.post_click(
                    box[0] + box[2] // 2,
                    box[1] + box[3] // 2,
                ).wait()
                if not click.status.succeeded:
                    logger.error("竞技场奖励页关闭点击未成功，已停止")
                    return False
                reward_close_clicks += 1
                time.sleep(0.25)
                continue
            time.sleep(0.25)
        try:
            self._record_store.note_return_unverified(
                pending_capture_id,
                reason_code="post_challenge_return_timeout",
                evidence=last_return_evidence,
            )
        except ArenaChallengeFlowError as error:
            logger.error(f"竞技场回场超时证据未能持久化: {error}")
        logger.error("竞技场对战结束后未在 20 秒内返回竞技场主界面，已停止下一轮")
        return False
