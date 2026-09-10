"""Concise, actionable Maa messages for arena task outcomes."""

from __future__ import annotations

import re
from typing import Literal
from dataclasses import dataclass
from collections.abc import Mapping

from .service import ArenaEvaluation, ArenaOwnScoreEvaluation

_SKILL_CARD_LAYOUT_ERRORS = {
    "skill_card_layout_incomplete": "技能卡布局定位失败",
    "skill_card_layout_unstable": "技能卡布局尚未稳定",
    "skill_card_geometry_unstable": "技能卡位置尚未稳定",
    "skill_card_content_generation_unstable": "技能卡画面内容尚未稳定",
}

_ERROR_LABELS = (
    ("unsupported_capture_resolution", "截图分辨率不受支持"),
    ("permissionerror", "访问被拒绝，请检查安装目录的写入权限"),
    ("access is denied", "访问被拒绝，请检查安装目录的写入权限"),
    ("permission denied", "访问被拒绝，请检查安装目录的写入权限"),
    ("访问被拒绝", "访问被拒绝，请检查安装目录的写入权限"),
    ("communication_retry_exhausted", "通信异常，自动重试后仍未恢复"),
    ("communication", "当前出现通信异常"),
    ("p_item_detail_recovery_failed", "P 道具详情关闭后未能返回成员页面"),
    ("skill_card_close", "技能卡详情未能正常关闭或返回成员页面"),
    ("member_preview_recovery", "未能返回队伍预览并重新打开成员"),
    ("support_bonus", "支援加成页面未能打开或读取完整"),
    ("clicked skill-card text disagrees", "技能卡详情与此前读取的信息不一致"),
    ("skill_card_detail_count_conflict", "技能卡强化信息存在矛盾"),
    ("skill_card_cost_evidence_conflict", "技能卡费用信息存在矛盾"),
    ("skill_card_detail_", "技能卡详情未能确认"),
    ("p_item_detail_", "P 道具详情未能确认"),
    *_SKILL_CARD_LAYOUT_ERRORS.items(),
    ("result_save", "对局结果暂时无法保存"),
    ("result_incomplete", "对局结果尚未读全"),
    ("recovery_failed", "失败后未能恢复到竞技场主界面"),
    ("p_item_", "P 道具识别不完整或结果不唯一"),
    ("skill_card_", "技能卡识别不完整或结果不唯一"),
    ("arena_grade", "竞技场 Grade 读取失败"),
    ("rehearsal", "无法确认己方演习页面"),
    ("opponent_", "对手列表或对手编成读取失败"),
    ("member_", "成员栏位或成员详情读取失败"),
    ("param_", "成员表现力数值读取失败"),
    ("ocr_empty", "当前画面的文字未能读取"),
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


def _raw_location(raw: str) -> dict[str, object]:
    """Read only explicit reader coordinates, never IDs or unscoped slot numbers."""

    members = set(re.findall(
        r"(?<![\w-])(own|self|opponent-[0-2])/stage-([1-3])/member-([1-3])(?!\d)", raw,
    ))
    if len(members) > 1:
        return {}
    location: dict[str, object] = {}
    if members:
        team, stage, member = members.pop()
        location.update(team_id=team, stage_number=int(stage), member_slot=int(member))
    cards = set(re.findall(r"\bgroup[ -]?([01])/slot[ -]?([1-6])(?!\d)", raw))
    if len(cards) == 1:
        group, slot = cards.pop()
        location.update(kind="skill_card", group_index=int(group), card_slot=int(slot))
    return location


def _index(value: object, minimum: int, maximum: int) -> int | None:
    return value if type(value) is int and minimum <= value <= maximum else None


def _confirmed_name(value: object) -> str:
    # Only the caller's confirmed title is eligible; candidate IDs are not names.
    if not isinstance(value, str):
        return "名称未确认"
    name = value.strip()
    if (
        not name or len(name) > 100 or name.isdecimal()
        or any(ord(character) < 32 for character in name)
        or "\\" in name or "://" in name or re.match(r"^[A-Za-z]:", name)
    ):
        return "名称未确认"
    return f"「{name}」"


def _location_text(location: Mapping[str, object], raw: str, *, member_only: bool = False) -> str:
    parts = []
    team = location.get("team_id")
    opponent = _index(location.get("opponent_position"), 0, 2)
    team_match = re.fullmatch(r"opponent-([0-2])", team) if isinstance(team, str) else None
    if team in ("own", "self") and opponent is None:
        parts.append("己方")
    elif team_match and opponent in (None, int(team_match[1])):
        parts.append(f"第{int(team_match[1]) + 1}位对手")
    elif team is None and opponent is not None:
        parts.append(f"第{opponent + 1}位对手")
    for key, noun in (("stage_number", "舞台"), ("member_slot", "位成员")):
        value = _index(location.get(key), 1, 3)
        if value is not None:
            parts.append(f"第{value}{noun}")

    # Layout checks can cover both rows; the requested group is not a failed card.
    if member_only:
        return "／".join(parts)

    kind = location.get("kind")
    if kind is None:
        has_skill = "skill_card_" in raw or "card_name" in location
        has_p_item = "p_item_" in raw or "p_item_name" in location
        if has_skill != has_p_item:
            kind = "skill_card" if has_skill else "p_item"
    if kind == "skill_card":
        card = "技能卡"
        group = _index(location.get("group_index"), 0, 1)
        slot = _index(location.get("card_slot"), 1, 6)
        if group is not None:
            card += ("上排", "下排")[group]
        if slot is not None:
            card += f"第{slot}槽"
        parts.append(f"{card}（{_confirmed_name(location.get('card_name'))}）")
    elif kind == "p_item":
        card = "P 道具"
        slot = _index(location.get("screen_slot"), 1, 4)
        if slot is not None:
            card += f"左起第{slot}槽"
        parts.append(f"{card}（{_confirmed_name(location.get('p_item_name'))}）")
    return "／".join(parts)


def describe_arena_error(
    error: object | None,
    *,
    location: Mapping[str, object] | None = None,
) -> str:
    """Explain a failure without exposing raw errors, paths or candidate arrays.

    ``location`` uses existing reader coordinates: opponent_position/group_index
    are zero-based; stage_number/member_slot/card_slot/screen_slot are one-based.
    card_slot includes physical empty/plus placeholders. Names must already be
    confirmed by the caller. A supplied mapping is not mixed with raw positions.
    """

    raw = _normalise_error(error)
    lowered = raw.lower()
    error_token, label = next(
        ((token, label) for token, label in _ERROR_LABELS if token in lowered),
        ("", "竞技场处理未完成"),
    )
    if "unsupported_capture_resolution" in lowered:
        sizes = set(re.findall(r"screen_size\s*=\s*\(\s*(\d{2,5})\s*,\s*(\d{2,5})\s*\)", raw))
        if len(sizes) == 1:
            width, height = sizes.pop()
            label += f"（{int(width)}×{int(height)}）"
    position = _location_text(
        location if location is not None else _raw_location(raw), raw,
        member_only=error_token in _SKILL_CARD_LAYOUT_ERRORS,
    )
    prefix = f"{position}：" if position else ""
    return f"{prefix}{label}；请查看详细日志"


def own_score_user_status(
    evaluation: ArenaOwnScoreEvaluation,
    *,
    simulations: int,
    location: Mapping[str, object] | None = None,
) -> ArenaUserStatus:
    """Describe a manual own-score recalculation result without a generic stop label."""

    if evaluation.status == "calculated":
        if evaluation.summary_warning:
            return ArenaUserStatus(
                "竞技场己方总分重算完成：三舞台分数缓存已更新；"
                f"但任务页缓存摘要刷新失败：{describe_arena_error(evaluation.summary_warning)}",
                "warning",
            )
        return ArenaUserStatus(
            "竞技场己方总分重算完成：三舞台分数缓存已更新"
            f"（模拟 {simulations} 次，读取尝试 {evaluation.attempts} 次）",
            "info",
        )

    if evaluation.status == "cache_failure" and "04:00" in _normalise_error(evaluation.error):
        return ArenaUserStatus(
            "竞技场己方总分重算失败：读取过程跨过每日 04:00 刷新边界；"
            "新缓存未落盘，请重新运行任务；请查看详细日志",
            "error",
        )

    status_labels = {
        "incomplete_input": "己方编成未能完整读取",
        "adapter_failure": "本地模拟器执行失败",
        "cache_failure": "分数缓存写入失败",
    }
    reason = status_labels.get(evaluation.status, "处理未完成")
    return ArenaUserStatus(
        f"竞技场己方总分重算失败：{reason}；{describe_arena_error(evaluation.error, location=location)}",
        "error",
    )


def win_rate_stop_user_status(
    evaluation: ArenaEvaluation,
    *,
    location: Mapping[str, object] | None = None,
) -> ArenaUserStatus:
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
    reason = status_labels.get(evaluation.status, "处理未完成")
    return ArenaUserStatus(
        f"竞技场未挑战：{reason}；{describe_arena_error(evaluation.error, location=location)}",
        "error",
    )
