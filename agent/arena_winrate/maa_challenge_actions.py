"""Reusable Maa actions for one arena result and post-finish verification."""

from __future__ import annotations

import re
import json
import time
import ctypes
from datetime import datetime, timezone

from utils import logger as user_logger
from maa.context import Context
from maa.custom_action import CustomAction

from .reader import ArenaPageState, classify_arena_page
from .recovery import error_retry_box
from .task_log import arena_task_log, diagnostic_logger, report_action_failure
from .cancellation import cancellation_for
from .challenge_flow import (
    CHALLENGE_LIFECYCLE_INTENT,
    ArenaChallengeFlowError,
    ArenaChallengeRecordStore,
    contest_day_key,
    battle_outcome_tap_box,
    classify_post_challenge_page,
    result_observations_complete,
    serialise_result_observations,
)

logger = diagnostic_logger(user_logger)


def _show_saved_result(context, capture_id, store, result=None, *, result_path=None):
    state = arena_task_log.current(context)
    if state is None or state.results.get(capture_id) not in (None, "UNKNOWN"):
        return
    try:
        if result is None:
            path = result_path if result_path is not None else store.recorded_result_path(capture_id)
            if path is None:
                return
            result = json.loads(path.read_text(encoding="utf-8"))["result"]
        arena_task_log.result_saved(context, capture_id, result, user_logger)
    except Exception as error:
        logger.warning(f"Could not display the saved arena result: {error}")


def _show_resumed_return(context, capture_id, store, path):
    """Reflect a completed child pipeline in its original parent task only."""
    if arena_task_log.current(context) is None:
        return
    try:
        if path is None:
            return
        record = store._load_result(path)
        store._require_result_capture_id(record, capture_id)
        if record["lifecycle"]["state"] not in {"FINISHED", "RECOVERED_FINISHED"}:
            return
        _show_saved_result(context, capture_id, store, record["result"])
        arena_task_log.returned(context, capture_id, exhausted=record["return_verification"]["evidence"].get("daily_attempts_exhausted", False))
    except Exception:
        logger.exception("竞技场已恢复对局的摘要生成失败")


def _administrator_process() -> bool:
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def _capture(context: Context):
    cancellation_for(context).check()
    image = context.tasker.controller.post_screencap().wait().get()
    cancellation_for(context).check()
    return image


def _post_click(context, *args, **kwargs):
    cancellation_for(context).check()
    return context.tasker.controller.post_click(*args, **kwargs)


def _full_ocr(context: Context, image):
    cancellation_for(context).check()
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
    cancellation_for(context).check()
    if not detail or not detail.hit:
        return ()
    return tuple(detail.all_results or detail.filtered_results or ())


def pending_challenge_recovery_state(store: ArenaChallengeRecordStore | None = None) -> str:
    """Report the only continuation without starting another challenge."""

    store = ArenaChallengeRecordStore() if store is None else store
    pending = store.load_pending()
    if pending is None:
        return "none"
    path = store.recorded_result_path(str(pending.get("capture_id", "")))
    if path is None:
        return "result_needed"
    record = store._load_result(path)
    if record["lifecycle"]["state"] in {"FINISHED", "RECOVERED_FINISHED"}:
        store._clear_completed_pending()
        return "none"
    return "return_needed"


def challenge_result_saved(store: ArenaChallengeRecordStore | None = None) -> bool:
    """Gate result-leaving input on durable, capture-bound winner evidence."""

    store = ArenaChallengeRecordStore() if store is None else store
    pending = store.load_pending()
    return bool(
        pending is not None
        and store.recorded_result_path(str(pending.get("capture_id", ""))) is not None
    )


def _communication_retry_box(rows):
    return error_retry_box(rows)


_DEFAULT_ACTION_NEXT = {
    "ChallengeError": [],
    "ChallengeResultRecord": ["ChallengeNext"],
    "ChallengeNext": ["ChallengeFinish"],
    "ChallengeFinish": ["ChallengeVerifyRefresh"],
    "ChallengeResumeNext": ["ChallengeResumeVerify"],
    "ChallengeRetryAfterStart": ["ChallengeReady", "[JumpBack]ChallengeUnformation"],
    "ChallengeRetryBeforeSkip": ["ChallengeSkip"],
    "ChallengeRetryBeforeResult": ["ChallengeResultRecord", "ChallengeSkip"],
    "ChallengeRetryBeforeFinish": ["ChallengeFinish"],
    "ChallengeRetryBeforeReturn": ["ChallengeVerifyRefresh"],
    "ChallengeRetryBeforeResumeReturn": ["ChallengeResumeVerify"],
}


def restore_challenge_action_next(context, argv):
    """A prior round's recovery route must not replace this round's normal route."""
    name = getattr(argv, "node_name", None)
    override = getattr(context, "override_next", None)
    if name in _DEFAULT_ACTION_NEXT and callable(override):
        cancellation_for(context).check()
        if override(name, list(_DEFAULT_ACTION_NEXT[name])) is False:
            raise ArenaChallengeFlowError("could not restore the challenge continuation")


def _confirm_observed_battle(context, store, pending, *, source):
    """A post-battle page verifies consumption, but does not reassign old battles."""
    if not pending.get("battle_started"):
        return
    capture_id = str(pending.get("capture_id", ""))
    if store.recorded_result_path(capture_id) is not None:
        return
    store.confirm_battle_started(capture_id, evidence={"battle_entered": True, "source": source})
    state = arena_task_log.current(context)
    if state is not None and pending.get("origin_run_id") == state.run_id:
        arena_task_log.battle_started(context, capture_id)


def recover_skipped_challenge_result(
    context, argv=None, *, store=None, image=None, rows=None, allow_result_page=False,
) -> bool:
    """Use a failed observation to recognize a skipped result and continue locally."""
    store = ArenaChallengeRecordStore() if store is None else store
    pending = store.load_pending()
    if pending is None:
        return False
    capture_id = str(pending.get("capture_id", ""))
    image = _capture(context) if image is None else image
    rows = _full_ocr(context, image) if rows is None else rows
    texts = tuple(str(getattr(row, "text", "")).strip() for row in rows)
    # Communication dialogs and the intermediate TAP page keep their own retry.
    shape = getattr(image, "shape", (1280, 720))
    if (_communication_retry_box(rows) is not None
            or any("エラー" in text or "通信中" in text for text in texts)
            or battle_outcome_tap_box(rows, frame_size=(shape[1], shape[0])) is not None):
        return False

    def matches(name, **kwargs):
        cancellation_for(context).check()
        detail = context.run_recognition(name, image, **kwargs)
        cancellation_for(context).check()
        return () if not detail or not detail.hit else tuple(detail.filtered_results or detail.all_results or ())

    opponents = matches("ArenaReaderOpponentCards")
    close = matches("CloseButton")
    finish = matches("ChallengeFinish", pipeline_override={"ChallengeFinish": {
        "recognition": "TemplateMatch", "template": "challenge_finish.png",
        "roi": [0, 980, 720, 300], "threshold": 0.95, "order_by": "Score",
    }})
    exhausted = classify_arena_page(
        rehearsal_count=sum(text == "リハーサル" for text in texts),
        opponent_count=len(opponents),
        stage_label_count=sum(re.fullmatch(r"ステージ\s*[123]", text) is not None for text in texts),
        stage_total_count=sum(text == "総合力" for text in texts),
        exhausted_notice_count=sum(text == "本日の挑戦権を消費しました" for text in texts),
    ) is ArenaPageState.OPPONENTS_UNAVAILABLE
    state = classify_post_challenge_page(
        opponent_count=len(opponents), close_count=len(close), ocr_texts=texts,
        reward_close_clicks=0, finish_count=len(finish), finish_resend_clicks=0,
        finish_resend_authorized=True, arena_main_exhausted=exhausted,
    )
    page = {"RETURNED": "arena", "RESEND_FINISH": "finish", "CLOSE_REWARD": "reward"}.get(state)
    if page is None and allow_result_page and not (opponents or close or finish):
        if matches("ArenaChallengeFormalResultPage"):
            page = "result"
    if page is None:
        return False
    evidence = {
        "result_unavailable_verified": True, "post_battle_page": page,
        "arena_return_verified": page == "arena", "opponent_count": len(opponents),
        "daily_attempts_exhausted": exhausted,
        "arena_page_state": (ArenaPageState.OPPONENTS_UNAVAILABLE.value if exhausted
                             else ArenaPageState.READY.value if page == "arena"
                             else ArenaPageState.AMBIGUOUS.value),
        "source": getattr(argv, "node_name", None) or "pending_result_probe",
    }
    cancellation_for(context).check()
    if page != "arena":
        _confirm_observed_battle(context, store, pending, source=f"post_battle_{page}")
    path = store.complete_unknown_result(capture_id, evidence=evidence)
    _show_saved_result(context, capture_id, store, result_path=path)
    if page == "arena":
        store.complete_return(capture_id, evidence=evidence)
        arena_task_log.returned(context, capture_id, exhausted=exhausted)
    name = getattr(argv, "node_name", None)
    override = getattr(context, "override_next", None)
    if name == "ChallengeError":
        # This is a JumpBack leaf. Starting another Selector loop would leave
        # its original parent on the stack and run that parent a second time.
        if not callable(override) or override(name, []) is False:
            raise ArenaChallengeFlowError("could not restore the challenge recovery leaf")
        if page in {"finish", "reward"} and not resume_pending_challenge(context, store=store):
            raise ArenaChallengeFlowError("post-battle return recovery did not finish")
    elif name and (page != "result" or name.startswith("ChallengeRetry")):
        child = "Resume" in name
        if page == "result":
            next_nodes = ["ChallengeResumeNext" if child else "ChallengeNext"]
        elif page == "arena":
            next_nodes = [] if child else ["ChallengeSelector"]
        else:
            next_nodes = ["ChallengeResumeVerify" if child else "ChallengeVerifyRefresh"]
        if not callable(override) or override(name, next_nodes) is False:
            raise ArenaChallengeFlowError("could not route the recovered challenge page")
    logger.info(json.dumps({
        "event": "arena_challenge_result_unavailable_recovered", "capture_id": capture_id,
        "result_path": str(path), **evidence,
    }, ensure_ascii=False, sort_keys=True))
    return True


def _retry_communication(context, store, capture_id: str, box) -> bool:
    if not store.reserve_communication_retry(capture_id):
        raise ArenaChallengeFlowError("communication retry budget exhausted")
    click = _post_click(context, box[0] + box[2] // 2, box[1] + box[3] // 2).wait()
    if not click.status.succeeded:
        raise ArenaChallengeFlowError("communication retry click failed")
    logger.info(json.dumps({
        "event": "arena_challenge_communication_retry", "capture_id": capture_id,
        "communication_retries": 1,
    }, ensure_ascii=False, sort_keys=True))
    return True


def _advance_battle_outcome(context, store, capture_id: str, box) -> bool:
    if not store.reserve_outcome_tap(capture_id):
        return False
    started = time.monotonic()
    succeeded = False
    try:
        click = _post_click(context, box[0] + box[2] // 2, box[1] + box[3] // 2).wait()
        succeeded = bool(click.status.succeeded)
        if not succeeded:
            raise ArenaChallengeFlowError("outcome TAP click failed")
        return True
    finally:
        logger.info(json.dumps({
            "event": "arena_challenge_outcome_tap", "capture_id": capture_id,
            "outcome_taps": 1, "tap_box": list(box), "succeeded": succeeded,
            "extra_captures": 0, "extra_ocr": 0, "extra_clicks": 1,
            "wall_seconds": round(time.monotonic() - started, 6),
        }, ensure_ascii=False, sort_keys=True))


def _await_formal_result(context, store, capture_id: str, *, entry_probe: bool = False, argv=None) -> bool:
    """Wait on the original result template before spending any winner frame."""
    started = time.monotonic()
    wait = store.begin_result_page_wait(capture_id)
    if wait["remaining_seconds"] <= 0:
        raise ArenaChallengeFlowError("result page wait budget exhausted")
    if entry_probe and store.reserve_result_page_probe(capture_id):
        image = _capture(context)
        rows = _full_ocr(context, image)
        # These raw rows belong to page navigation, never to winner recovery.
        store.note_result_page_probe(capture_id, serialise_result_observations(rows))
        logger.info(json.dumps({
            "event": "arena_challenge_result_page_probe", "capture_id": capture_id,
            "extra_captures": 1, "extra_ocr": 1, "extra_clicks": 0,
            "wall_seconds": round(time.monotonic() - started, 6),
        }, ensure_ascii=False, sort_keys=True))
        if store.begin_result_page_wait(capture_id)["remaining_seconds"] <= 0:
            raise ArenaChallengeFlowError("result page wait budget exhausted")
        shape = getattr(image, "shape", (1280, 720))
        tap = battle_outcome_tap_box(rows, frame_size=(shape[1], shape[0]))
        if tap is not None:
            _advance_battle_outcome(context, store, capture_id, tap)
        else:
            retry = _communication_retry_box(rows)
            if retry is not None:
                _retry_communication(context, store, capture_id, retry)
    wait = store.begin_result_page_wait(capture_id)
    if wait["remaining_seconds"] <= 0:
        raise ArenaChallengeFlowError("result page wait budget exhausted")
    # DirectHit entry -> TemplateMatch/DoNothing leaf. Neither node can leave
    # a result page, send Start, or run winner OCR while the game is loading.
    timeout = max(1, int(wait["remaining_seconds"] * 1000))
    cancellation_for(context).check()
    task = context.run_task("ArenaChallengeAwaitFormalResult", pipeline_override={
        "ArenaChallengeAwaitFormalResult": {"timeout": timeout},
        "ArenaChallengeFormalResultPage": {"timeout": timeout},
    })
    cancellation_for(context).check()
    succeeded = bool(task and task.status.succeeded)
    try:
        remaining = store.begin_result_page_wait(capture_id)["remaining_seconds"]
    except ArenaChallengeFlowError:
        pending = store.load_pending()
        if pending is None or store._remaining_winner_seconds(pending, datetime.now(timezone.utc)) != 0:
            raise
        remaining = 0
    logger.info(json.dumps({
        "event": "arena_challenge_result_page_wait", "capture_id": capture_id,
        "succeeded": succeeded, "remaining_seconds": remaining,
        "wall_seconds": round(time.monotonic() - started, 6),
        "winner_frames_consumed": 0,
    }, ensure_ascii=False, sort_keys=True))
    if succeeded and remaining > 0:
        return True
    # The nested template wait exposes no reusable source image. One failed-
    # wait probe handles a result skipped while that wait was in flight.
    pending = store.load_pending()
    exhausted = pending is not None and store._remaining_winner_seconds(pending, datetime.now(timezone.utc)) == 0
    return recover_skipped_challenge_result(context, argv, store=store, allow_result_page=exhausted)


class ArenaChallengeRecordResultAction(CustomAction):
    """Save independent winner evidence before its result-page button is clicked."""

    def __init__(self, record_store: ArenaChallengeRecordStore | None = None) -> None:
        super().__init__()
        self._record_store = ArenaChallengeRecordStore() if record_store is None else record_store

    def _save(self, capture_id: str, result: dict, *, context=None) -> bool:
        store = self._record_store
        _confirm_observed_battle(context, store, store._pending_for_capture(capture_id), source="complete_result")
        unrecorded_failures = 0
        while True:
            if store.recorded_result_path(capture_id) is not None:
                _show_saved_result(context, capture_id, store)
                return True
            previous_attempts = store._recovery_budget(store._pending_for_capture(capture_id))["save_attempts"]
            try:
                reserved = store.reserve_result_save(
                    capture_id, result, unrecorded_failures=unrecorded_failures,
                )
            except (ArenaChallengeFlowError, OSError) as error:
                logger.error(f"竞技场结果保存预算写入失败，复用同帧结果重试一次: {error}")
                current_attempts = store._recovery_budget(store._pending_for_capture(capture_id))["save_attempts"]
                unrecorded_failures += int(current_attempts == previous_attempts)
                if max(previous_attempts, current_attempts) + unrecorded_failures >= 2:
                    return False
                continue
            if not reserved:
                return False
            unrecorded_failures = 0
            try:
                path = store.complete_idempotent(capture_id, result)
            except (ArenaChallengeFlowError, OSError) as error:
                logger.error(f"竞技场结果保存失败，保留同帧结果并按原预算重试: {error}")
                continue
            logger.info(json.dumps({
                "event": "arena_challenge_result_recorded",
                "result_path": str(path), "result": result,
            }, ensure_ascii=False, sort_keys=True))
            _show_saved_result(context, capture_id, store, result)
            return True

    @report_action_failure("竞技场结果尚未完整保存，任务已中断", user_logger)
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            restore_challenge_action_next(context, argv)
            if not _administrator_process():
                raise ArenaChallengeFlowError("Maa result process is not elevated")
            store = self._record_store
            pending = store.load_pending()
            if pending is None:
                raise ArenaChallengeFlowError("result page has no pending challenge")
            capture_id = str(pending.get("capture_id", ""))
            if store.recorded_result_path(capture_id) is not None:
                _show_saved_result(context, capture_id, store)
                return True
            result = store.recoverable_pending_result(capture_id)
            if result is not None:
                return self._save(capture_id, result, context=context)
            retry_deadline = None
            image, rows = None, ()
            while retry_deadline is None or time.monotonic() < retry_deadline:
                if not store.reserve_result_frame(capture_id):
                    break
                try:
                    image = _capture(context)
                    rows = _full_ocr(context, image)
                    result = serialise_result_observations(rows)
                except Exception as error:
                    image, rows = None, ()
                    result = {"ocr_observations": [], "observation_error": type(error).__name__}
                result.update({
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "contest_day": pending.get("contest_day", contest_day_key()),
                })
                if result_observations_complete(result):
                    return self._save(capture_id, result, context=context)
                store.note_incomplete_result(result, capture_id=capture_id)
                if image is not None and recover_skipped_challenge_result(
                    context, argv, store=store, image=image, rows=rows,
                ):
                    return True
                if retry_deadline is None:
                    retry_deadline = time.monotonic() + 2.0
            if image is not None and recover_skipped_challenge_result(
                context, argv, store=store, image=image, rows=rows, allow_result_page=True,
            ):
                return True
            raise ArenaChallengeFlowError("result winner recovery budget exhausted")
        except Exception as error:
            logger.error(f"竞技场结果未保存；保留原结果页并停止离页及下一次挑战: {error}")
        return False


def resume_pending_challenge(context: Context, argv=None, store=None) -> bool:
    """Continue one pending result/return in the client, never start a battle."""

    store = ArenaChallengeRecordStore() if store is None else store
    try:
        pending = store.load_pending()
        if pending is None:
            return True
        if (
            pending.get("start_reservation_required") is True
            and pending.get("lifecycle_state") == CHALLENGE_LIFECYCLE_INTENT
            and not any(key in pending for key in (
                "battle_started", "incomplete_result", "result_recovery",
                "result_recorded_at", "return_recovery", "return_verification",
            ))
        ):
            # Only new intents explicitly promise a durable Start reservation
            # before input. Missing fields in legacy records prove nothing.
            capture_id = str(pending.get("capture_id", ""))
            if store.recorded_result_path(capture_id) is None:
                cancellation_for(context).check()
                # Abort revalidates the capture, lifecycle, Start and result.
                store.abort(capture_id)
                logger.info("已清除上次选敌后尚未预留开始的记录，继续竞技场入口")
                return True
        state = pending_challenge_recovery_state(store)
        if state == "none":
            return True
        if not _administrator_process():
            raise ArenaChallengeFlowError("Maa recovery process is not elevated")
        if state == "result_needed":
            pending = store.load_pending()
            capture_id = str(pending.get("capture_id", ""))
            if store.recoverable_pending_result(capture_id) is None:
                image = _capture(context)
                rows = _full_ocr(context, image)
                exhausted = store._remaining_winner_seconds(pending, datetime.now(timezone.utc)) == 0
                recovered = recover_skipped_challenge_result(
                    context, store=store, image=image, rows=rows, allow_result_page=exhausted,
                )
                if store.load_pending() is None:
                    return True
                if not recovered:
                    wait = store.begin_result_page_wait(capture_id)
                    if wait["remaining_seconds"] <= 0:
                        detail = context.run_recognition("ArenaChallengeFormalResultPage", image)
                        if not detail or not detail.hit:
                            return False
                        # A fresh formal-page match supersedes an expired navigation
                        # wait, but never refunds a spent winner observation.
                        recovered = True
                    elif store.reserve_result_page_probe(capture_id):
                        store.note_result_page_probe(capture_id, serialise_result_observations(rows))
                    shape = getattr(image, "shape", (1280, 720))
                    tap = battle_outcome_tap_box(rows, frame_size=(shape[1], shape[0]))
                    retry = _communication_retry_box(rows)
                    if tap is not None:
                        _advance_battle_outcome(context, store, capture_id, tap)
                    elif retry is not None:
                        _retry_communication(context, store, capture_id, retry)
                    if not recovered and not _await_formal_result(context, store, capture_id):
                        return False
                    if store.load_pending() is None:
                        return True
            if store.recorded_result_path(capture_id) is None:
                if not ArenaChallengeRecordResultAction(store).run(context, None):
                    return False
        pending = store.load_pending()
        if pending is None:
            return True
        verification = pending.get("return_verification", {})
        if (
            verification.get("status") == "verified"
            and verification.get("evidence_origin") == "product"
            and isinstance(verification.get("evidence"), dict)
        ):
            path = store.complete_return(pending["capture_id"], evidence=verification["evidence"])
            _show_resumed_return(context, pending["capture_id"], store, path)
            return True
        try:
            result_path = store.recorded_result_path(pending["capture_id"])
        except Exception:
            logger.exception("竞技场恢复摘要路径暂时不可读；继续原回场流程")
            result_path = None
        cancellation_for(context).check()
        task = context.run_task("ArenaChallengeResumeReturn")
        cancellation_for(context).check()
        resumed = bool(task and task.status.succeeded and store.load_pending() is None)
        if resumed:
            _show_resumed_return(context, pending["capture_id"], store, result_path)
        return resumed
    except Exception as error:
        logger.error(f"竞技场客户端待办恢复未完成，禁止新挑战: {error}")
        return False


class ArenaChallengeRetryCurrentBattleAction(CustomAction):
    """Recover a proven dialog; an outcome TAP must finish the page wait first."""

    def __init__(self, record_store: ArenaChallengeRecordStore | None = None) -> None:
        super().__init__()
        self._record_store = ArenaChallengeRecordStore() if record_store is None else record_store

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            restore_challenge_action_next(context, argv)
            if not _administrator_process():
                raise ArenaChallengeFlowError("Maa retry process is not elevated")
            pending = self._record_store.load_pending()
            if pending is None:
                raise ArenaChallengeFlowError("communication retry has no pending challenge")
            capture_id = str(pending.get("capture_id", ""))
            unresolved = self._record_store.recorded_result_path(capture_id) is None
            image = _capture(context)
            rows = _full_ocr(context, image)
            shape = getattr(image, "shape", (1280, 720))
            tap = battle_outcome_tap_box(rows, frame_size=(shape[1], shape[0]))
            box = _communication_retry_box(rows)
            if tap is None and box is None:
                if recover_skipped_challenge_result(
                    context, argv, store=self._record_store, image=image, rows=rows,
                    allow_result_page=(unresolved and self._record_store._remaining_winner_seconds(
                        pending, datetime.now(timezone.utc),
                    ) == 0),
                ):
                    return True
                cancellation_for(context).check()
                formal = context.run_recognition("ArenaChallengeFormalResultPage", image)
                cancellation_for(context).check()
                if formal and formal.hit:
                    name = getattr(argv, "node_name", None)
                    if name:
                        next_node = "ChallengeResultRecord" if unresolved else (
                            "ChallengeResumeNext" if "Resume" in name else "ChallengeNext"
                        )
                        if context.override_next(name, [next_node]) is False:
                            raise ArenaChallengeFlowError("could not continue the formal result page")
                    return True
                raise ArenaChallengeFlowError("page has no unique supported communication dialog or outcome TAP")
            if unresolved:
                if self._record_store._remaining_winner_seconds(pending, datetime.now(timezone.utc)) == 0:
                    raise ArenaChallengeFlowError("result winner recovery budget exhausted")
                if "page_wait" in pending.get("result_recovery", {}):
                    if self._record_store.begin_result_page_wait(capture_id)["remaining_seconds"] <= 0:
                        raise ArenaChallengeFlowError("result page wait budget exhausted")
            if tap is not None:
                _advance_battle_outcome(context, self._record_store, capture_id, tap)
                return _await_formal_result(context, self._record_store, capture_id, argv=argv)
            _retry_communication(context, self._record_store, capture_id, box)
            if unresolved and "page_wait" in pending.get("result_recovery", {}):
                return _await_formal_result(context, self._record_store, capture_id, argv=argv)
            return True
        except Exception as error:
            logger.error(f"竞技场当前对局页面恢复停止: {error}")
            return False


class ArenaChallengeVerifyRefreshAction(CustomAction):
    """Close known reward pages, then verify the refreshed three-card page."""

    def __init__(self, record_store: ArenaChallengeRecordStore | None = None) -> None:
        super().__init__()
        self._record_store = (
            ArenaChallengeRecordStore() if record_store is None else record_store
        )

    @report_action_failure("竞技场返回确认未完成，任务已中断", user_logger)
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
        try:
            if self._record_store.recorded_result_path(pending_capture_id) is None:
                raise ArenaChallengeFlowError("challenge result must be saved before return input")
            return_budget = self._record_store.begin_return_recovery(pending_capture_id)
        except ArenaChallengeFlowError as error:
            logger.error(f"竞技场结果尚未保存，禁止回场输入: {error}")
            return False
        started = time.monotonic()
        deadline = started + return_budget["remaining_seconds"]
        reward_close_clicks = return_budget["reward_close_clicks"]
        finish_resend_clicks = return_budget["finish_resend_clicks"]
        finish_proof_box: tuple[int, int, int, int] | None = None
        last_return_evidence: dict[str, object] = {}
        expired_return_probe = return_budget["remaining_seconds"] <= 0
        while expired_return_probe or time.monotonic() < deadline:
            expired_return_probe = False
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
            observed_at = time.monotonic()
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
                "wall_seconds": round(observed_at - started, 6),
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
                _show_saved_result(context, pending_capture_id, self._record_store, result_path=result_path)
                arena_task_log.returned(context, pending_capture_id, exhausted=last_return_evidence["daily_attempts_exhausted"])
                return True
            if observed_at >= deadline:
                break
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
                try:
                    if not self._record_store.reserve_return_click(pending_capture_id, "finish_resend_clicks"):
                        raise ArenaChallengeFlowError("finish resend budget exhausted")
                except ArenaChallengeFlowError as error:
                    logger.error(f"竞技场结束页补发未获持久化预算，已停止: {error}")
                    return False
                click = _post_click(context,
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
                try:
                    if not self._record_store.reserve_return_click(pending_capture_id, "reward_close_clicks"):
                        raise ArenaChallengeFlowError("reward close budget exhausted")
                except ArenaChallengeFlowError as error:
                    logger.error(f"竞技场奖励页关闭未获持久化预算，已停止: {error}")
                    return False
                click = _post_click(context,
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
