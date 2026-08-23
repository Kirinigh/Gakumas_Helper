"""Reusable Maa actions for one arena result and post-finish verification."""

from __future__ import annotations

import json
import time
import ctypes
from datetime import datetime, timezone

from utils import logger
from maa.context import Context
from maa.custom_action import CustomAction

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

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del argv
        if not _administrator_process():
            logger.error("竞技场结束页 Maa 输入进程不是管理员权限，已停止")
            return False
        started = time.monotonic()
        deadline = started + 20.0
        reward_close_clicks = 0
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
            texts = (
                tuple(
                    str(getattr(item, "text", ""))
                    for item in _full_ocr(context, image)
                )
                if close_results
                else ()
            )
            page_state = classify_post_challenge_page(
                opponent_count=len(results),
                close_count=len(close_results),
                ocr_texts=texts,
                reward_close_clicks=reward_close_clicks,
            )
            if page_state == "RETURNED":
                logger.info(
                    json.dumps(
                        {
                            "event": "arena_challenge_return_verified",
                            "opponent_count": 3,
                            "full_lineup_reread": False,
                            "reward_close_clicks": reward_close_clicks,
                            "wall_seconds": round(time.monotonic() - started, 6),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return True
            if page_state == "AMBIGUOUS":
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
        logger.error("竞技场对战结束后未在 20 秒内返回三对手主界面，已停止下一轮")
        return False
