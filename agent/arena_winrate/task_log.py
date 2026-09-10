"""A transient presentation of the existing arena task and result events.

No game state, recovery budgets, or records are owned here. Detailed records keep
their original severity in the file sink; only concise messages reach the GUI.
"""

from __future__ import annotations

import time
from functools import wraps
from threading import RLock
from dataclasses import field, dataclass

from .decision import HIGHEST_WIN_RATE_FALLBACK_RULE


def diagnostic_logger(logger):
    bind = getattr(logger, "bind", None)
    return bind(ui_visible=False) if callable(bind) else logger


def _tasker_key(tasker):
    handle = getattr(tasker, "_handle", None)
    return getattr(handle, "value", handle)


@dataclass
class RunDisplay:
    task_id: int
    started_at: float = field(default_factory=time.monotonic)
    enabled: bool = False
    background_shown: bool = False
    season_shown: bool = False
    starts: set[str] = field(default_factory=set)
    results: dict[str, str] = field(default_factory=dict)
    returned: set[str] = field(default_factory=set)
    failure: str | None = None
    terminal: str | None = None


class ArenaTaskLog:
    def __init__(self):
        self._runs: dict[object, RunDisplay] = {}
        self._lock = RLock()

    def start(self, key, task_id):
        with self._lock:
            current = self._runs.get(key)
            if current is None or current.task_id != task_id:
                self._runs[key] = RunDisplay(task_id)

    def current(self, context):
        with self._lock:
            state = self._runs.get(_tasker_key(getattr(context, "tasker", None)))
            get_job = getattr(context, "get_task_job", None)
            if state is not None and callable(get_job):
                try:
                    if get_job().job_id != state.task_id:
                        return None
                except Exception:
                    return None
            return state

    def enable(self, context):
        with self._lock:
            state = self.current(context)
            if state is not None:
                state.enabled = True

    def background(self, context, message, logger):
        with self._lock:
            state = self.current(context)
            if state is None:
                return
            state.enabled = True
            if not state.background_shown:
                state.background_shown = True
                logger.info(message)

    def season_selected(self, context, season, selection, logger, *, standalone=False):
        """Show the resolved RIS season before reading, once in the daily task."""
        with self._lock:
            state = self.current(context)
            if state is None:
                if not standalone:
                    return
            elif state.season_shown:
                return
            else:
                state.season_shown = True
            status = "RIS预览" if season.preview else "RIS正式"
            source = "latest" if selection == "latest" else "手动选择"
            message = (
                f"本次使用第{season.season}期（{status}，{source}）。"
                "请核对游戏当期；不一致时停止任务，并在“竞技场期数”中手动选择。"
            )
            getattr(logger, "warning" if season.preview else "info")(message)

    def failed(self, context, message):
        with self._lock:
            state = self.current(context)
            if state is not None:
                state.enabled = True
                state.failure = message

    def battle_started(self, context, capture_id):
        with self._lock:
            state = self.current(context)
            if state is not None:
                state.starts.add(capture_id)

    def result_saved(self, context, capture_id, result, logger):
        with self._lock:
            state = self.current(context)
            if state is None or capture_id in state.results:
                return
            state.enabled = True
            state.results[capture_id] = result["outcome"]
            outcomes = {"WIN": "胜", "LOSS": "负", "TIE": "平"}
            winners = {"OWN": "胜", "OPPONENT": "负", "TIE": "平"}
            stages = "／".join(winners.get(row.get("winner"), "未确认") for row in result.get("stage_results", ()))
            count = len(state.results.keys() & state.starts)
            prefix = f"第 {count} 场" if capture_id in state.starts else "恢复的对局"
            logger.info(f"竞技场{prefix}：{outcomes.get(result['outcome'], '未确认')}；三个舞台：{stages}")

    def returned(self, context, capture_id, *, exhausted=False):
        with self._lock:
            state = self.current(context)
            if state is not None:
                state.returned.add(capture_id)
                if exhausted:
                    state.terminal = "今日次数已用完"

    def node_finished(self, key, name, *, task_id=None):
        with self._lock:
            state = self._runs.get(key)
            if state is None or (task_id is not None and state.task_id != task_id):
                return
            normal = {"ChallengeRunOut": "今日次数已用完", "ChallengeReadyPeriod": "当前为准备期"}
            failed = {"ChallengeWinRateStop", "ChallengeEntryStateStop"}
            if name in normal:
                state.terminal = normal[name]
            elif name in failed:
                state.failure = state.failure or "流程未能继续"

    def finish(self, key, task_id, *, succeeded, logger):
        with self._lock:
            state = self._runs.get(key)
            if state is None or state.task_id != task_id:
                return
            del self._runs[key]
            if not state.enabled:
                return
            outcomes = [value for capture_id, value in state.results.items() if capture_id in state.starts]
            wins = outcomes.count("WIN")
            losses = outcomes.count("LOSS")
            ties = outcomes.count("TIE")
            missing = len(state.starts - state.results.keys())
            unreturned = len(state.results.keys() - state.returned)
            recovered = len(state.results.keys() - state.starts)
            completed = succeeded and not state.failure and not missing and not unreturned and bool(state.terminal)
            ending = "完成" if completed else "中断"
            elapsed = max(0, int(time.monotonic() - state.started_at))
            message = f"竞技场任务{ending}：本轮挑战 {len(state.starts)} 场，{wins} 胜 {losses} 负"
            if ties:
                message += f" {ties} 平"
            if recovered:
                message += f"；另恢复此前对局 {recovered} 场"
            if missing:
                message += f"，另有 {missing} 场结果未确认"
            if unreturned:
                message += f"，{unreturned} 场回场未确认"
            message += f"；用时 {elapsed // 60} 分 {elapsed % 60} 秒"
            if completed:
                message += f"；{state.terminal}"
            getattr(logger, "info" if completed else "warning")(message)

    def read_terminal(self, tasker, task_id, logger):
        """Read only the existing terminal nodes, once at task completion.

        Normal exhausted returns already carry their ending. No context sink
        is registered: parsing every OCR event just for display adds work.
        """
        key = _tasker_key(tasker)
        with self._lock:
            state = self._runs.get(key)
            if (state is None or state.task_id != task_id or not state.enabled
                    or state.failure or state.terminal):
                return
        try:
            task = tasker.get_task_detail(task_id)
            if task is None:
                return
            positions = {node_id: index for index, node_id in enumerate(task.node_id_list)}
            endings = []
            for name in ("ChallengeRunOut", "ChallengeReadyPeriod", "ChallengeWinRateStop", "ChallengeEntryStateStop"):
                node = tasker.get_latest_node(name)
                if node is not None and node.completed and node.name == name and node.node_id in positions:
                    endings.append((positions[node.node_id], name))
            if endings:
                self.node_finished(key, max(endings)[1], task_id=task_id)
        except Exception:
            diagnostic_logger(logger).exception("竞技场结束状态日志查询失败")


arena_task_log = ArenaTaskLog()


def report_action_failure(message, logger):
    """Present a final action failure without exposing its diagnostic payload."""
    def decorate(operation):
        @wraps(operation)
        def wrapped(self, context, argv):
            result = operation(self, context, argv)
            if result is False:
                arena_task_log.failed(context, message)
                logger.error(message + "；详细原因已写入日志")
            return result
        return wrapped
    return decorate


def opponent_rates_message(decision):
    rates = "，".join(f"{int(row['position']) + 1} 号 {float(row['win_rate']) * 100:.2f}%" for row in decision.estimates)
    selected = decision.selected_position
    message = f"对手预测胜率：{rates}"
    if selected is not None:
        message += f"；选择 {selected + 1} 号"
        if decision.decision_rule == HIGHEST_WIN_RATE_FALLBACK_RULE:
            message += f"（均未达到 {decision.threshold * 100:.0f}% 门槛，选择最高胜率）"
    return message


def register_arena_log_sinks(agent_server, logger):
    from maa.tasker import TaskerEventSink
    from maa.event_sink import NotificationType

    class TaskSink(TaskerEventSink):
        def on_tasker_task(self, tasker, noti_type, detail):
            if detail.entry != "Challenge":
                return
            key = _tasker_key(tasker)
            if noti_type == NotificationType.Starting:
                arena_task_log.start(key, detail.task_id)
            elif noti_type in (NotificationType.Succeeded, NotificationType.Failed):
                if noti_type == NotificationType.Succeeded:
                    arena_task_log.read_terminal(tasker, detail.task_id, logger)
                arena_task_log.finish(key, detail.task_id, succeeded=noti_type == NotificationType.Succeeded, logger=logger)

    sinks = (TaskSink(),)
    agent_server.add_tasker_sink(sinks[0])
    return sinks
