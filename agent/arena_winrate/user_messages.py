"""Concise, actionable Maa messages for arena task outcomes."""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

from .service import ArenaEvaluation, ArenaOwnScoreEvaluation

_MAX_ERROR_DETAIL = 420
_ERROR_LABELS = (
    ("unsupported_capture_resolution", "截图分辨率不受支持"),
    ("recovery_failed", "失败后未能恢复到竞技场主界面"),
    ("p_item_", "P 道具识别不完整或结果不唯一"),
    ("skill_card_", "技能卡识别不完整或结果不唯一"),
    ("arena_grade", "竞技场 Grade 读取失败"),
    ("rehearsal", "无法确认己方演习页面"),
    ("opponent", "对手列表或对手编成读取失败"),
    ("member_", "成员栏位或成员详情读取失败"),
    ("param_", "成员表现力数值读取失败"),
    ("support_bonus", "支援加成读取失败"),
    ("maa_", "Maa 界面操作失败"),
    ("timeout", "操作超时"),
)


@dataclass(frozen=True)
class ArenaUserStatus:
    """One user-facing status and the matching log severity."""

    message: str
    level: Literal["info", "warning", "error"]


def _normalise_error(error: object | None) -> str:
    if error is None:
        return "底层模块未提供进一步信息"
    text = " ".join(str(error).split())
    return text or "底层模块未提供进一步信息"


def describe_arena_error(error: object | None) -> str:
    """Translate a low-level arena failure into a bounded Chinese explanation."""

    raw = _normalise_error(error)
    lowered = raw.lower()
    label = "竞技场处理失败"
    for token, candidate in _ERROR_LABELS:
        if token in lowered:
            label = candidate
            break

    code = ""
    head, separator, tail = raw.partition(":")
    if separator and head.replace("_", "").isalnum() and head == head.lower():
        code = head
        detail = tail.strip()
    else:
        detail = raw
    if len(detail) > _MAX_ERROR_DETAIL:
        detail = f"{detail[:_MAX_ERROR_DETAIL].rstrip()}……（完整技术细节见日志）"
    code_text = f"（错误码 {code}）" if code else ""
    return f"{label}{code_text}：{detail}"


def own_score_user_status(
    evaluation: ArenaOwnScoreEvaluation,
    *,
    simulations: int,
) -> ArenaUserStatus:
    """Describe a manual own-score recalculation result without a generic stop label."""

    if evaluation.status == "calculated":
        if evaluation.summary_warning:
            detail = _normalise_error(evaluation.summary_warning)
            if len(detail) > _MAX_ERROR_DETAIL:
                detail = f"{detail[:_MAX_ERROR_DETAIL].rstrip()}……（完整技术细节见日志）"
            return ArenaUserStatus(
                "竞技场己方总分重算完成：三舞台分数缓存已更新；"
                f"但任务页缓存摘要刷新失败：{detail}",
                "warning",
            )
        return ArenaUserStatus(
            "竞技场己方总分重算完成：三舞台分数缓存已更新"
            f"（模拟 {simulations} 次，读取尝试 {evaluation.attempts} 次）",
            "info",
        )

    if evaluation.status == "cache_failure" and "04:00" in _normalise_error(evaluation.error):
        return ArenaUserStatus(
            f"竞技场己方总分重算失败：{_normalise_error(evaluation.error)}",
            "error",
        )

    status_labels = {
        "incomplete_input": "己方编成连续两次未能完整读取",
        "adapter_failure": "本地模拟器执行失败",
        "cache_failure": "分数缓存写入失败",
    }
    reason = status_labels.get(evaluation.status, f"未知结果状态 {evaluation.status}")
    return ArenaUserStatus(
        f"竞技场己方总分重算失败：{reason}；{describe_arena_error(evaluation.error)}",
        "error",
    )


def win_rate_stop_user_status(evaluation: ArenaEvaluation) -> ArenaUserStatus:
    """Describe why an arena opponent was not selected or challenged."""

    if evaluation.status == "no_qualified_opponent" and evaluation.decision is not None:
        estimates = ", ".join(
            f"{int(row['position']) + 1}号 {float(row['win_rate']) * 100:.2f}%/"
            f"下限 {float(row['qualification_lower']) * 100:.2f}%"
            for row in evaluation.decision.estimates
        )
        return ArenaUserStatus(
            "竞技场未挑战：没有对手达到"
            f" {evaluation.decision.threshold * 100:.0f}% 安全门槛"
            f"（胜率、家族校正下限：{estimates}）",
            "warning",
        )

    status_labels = {
        "observation_failure": "对手编成读取失败",
        "incomplete_input": "竞技场编成字段不完整",
        "adapter_failure": "本地胜率模拟器执行失败",
    }
    reason = status_labels.get(evaluation.status, f"安全条件未满足（状态 {evaluation.status}）")
    return ArenaUserStatus(
        f"竞技场未挑战：{reason}；{describe_arena_error(evaluation.error)}",
        "error",
    )
