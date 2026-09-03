"""Season-scoped persistence for reusable own-team arena score samples."""

from __future__ import annotations

import re
import json
import math
from uuid import uuid4
from typing import Any
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .schema import SnapshotValidationError, validate_own_snapshot
from .adapter import UPSTREAM_COMMIT, OwnScoreBatch

CACHE_SCHEMA_VERSION = 2
LEGACY_CACHE_SCHEMA_VERSION = 1
SUPPORTED_CACHE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_CACHE_SCHEMA_VERSION, CACHE_SCHEMA_VERSION}
)
SUMMARY_RENDER_VERSION = 2
MAX_SUMMARY_CACHE_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_FILE_BYTES = 1024 * 1024
DEFAULT_OWN_SCORE_CACHE = (
    Path(__file__).resolve().parents[2] / ".local" / "arena-win-rate" / "own-score-cache-v1.json"
)
DEFAULT_OWN_SCORE_SUMMARY = DEFAULT_OWN_SCORE_CACHE.with_name("own-score-cache-summary.md")
_SUMMARY_HEADER = re.compile(
    r"^<!-- arena-own-score-summary v=(\d+) cache_size=(missing|\d+) "
    r"cache_mtime_ns=(missing|\d+) -->$"
)


class _CacheSummaryError(ValueError):
    """Raised when cached display fields cannot support a truthful summary."""


@dataclass(frozen=True)
class OwnGradeState:
    """Recognized, manually overridden, and effective arena Grade state."""

    recognized_grade: int | None
    grade_override: int | None
    effective_grade: int | None
    source: str


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _CacheSummaryError(f"{field} 不是正整数")
    return value


def _cache_schema_version(record: Mapping[str, Any]) -> int:
    value = record.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int):
        raise _CacheSummaryError("缓存格式版本无效")
    if value not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
        raise _CacheSummaryError("缓存格式版本不兼容")
    return value


def _grade_value(value: object, *, field: str, allow_none: bool) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 7:
        raise _CacheSummaryError(f"{field} 必须是 1 到 7 的整数")
    return value


def _grade_state(
    record: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> OwnGradeState:
    cache_schema = _cache_schema_version(record)
    recognized = (
        _grade_value(snapshot.get("arena_grade"), field="arena_grade", allow_none=False)
        if "arena_grade" in snapshot
        else None
    )
    override = None
    if cache_schema >= CACHE_SCHEMA_VERSION and "grade_override" in record:
        override = _grade_value(
            record.get("grade_override"),
            field="grade_override",
            allow_none=True,
        )
    if override is not None:
        return OwnGradeState(recognized, override, override, "manual_override")
    if recognized is not None:
        return OwnGradeState(recognized, None, recognized, "recognized")
    return OwnGradeState(None, None, None, "unrecognized")


def _round_half_up_mean(scores: Sequence[int]) -> int:
    if not scores:
        raise _CacheSummaryError("原始分数样本为空")
    return int(
        (Decimal(sum(scores)) / Decimal(len(scores))).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _summarize_scores(scores: Sequence[int]) -> dict[str, int | float]:
    """Mirror the fixed JS engine's interpolation over authoritative raw scores."""

    if not scores:
        raise _CacheSummaryError("原始分数样本为空")
    ordered = sorted(scores)

    def quantile(ratio: float) -> int | float:
        index = (len(ordered) - 1) * ratio
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "q1": quantile(0.25),
        "median": quantile(0.5),
        "mean": sum(ordered) / len(ordered),
        "q3": quantile(0.75),
        "max": ordered[-1],
    }


def _summary_write_reason(error: OSError) -> str:
    if getattr(error, "winerror", None) in {32, 33}:
        return "文件被其他程序占用"
    if isinstance(error, PermissionError):
        return "访问被拒绝，请检查安装目录写入权限"
    if isinstance(error, FileExistsError):
        return "目标位置存在同名目录"
    return "本地文件系统写入失败"


def _summary_header(*, cache_size: int | None, cache_mtime_ns: int | None) -> str:
    size_text = "missing" if cache_size is None else str(cache_size)
    mtime_text = "missing" if cache_mtime_ns is None else str(cache_mtime_ns)
    return (
        f"<!-- arena-own-score-summary v={SUMMARY_RENDER_VERSION} "
        f"cache_size={size_text} cache_mtime_ns={mtime_text} -->"
    )


def _with_summary_header(
    content: str,
    *,
    cache_size: int | None,
    cache_mtime_ns: int | None,
) -> str:
    return f"{_summary_header(cache_size=cache_size, cache_mtime_ns=cache_mtime_ns)}\n{content}"


def _format_score(value: int | float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.1f}"


def _format_local_time(value: str) -> str | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _invalid_summary(reason: str) -> str:
    return (
        "### 当前己方缓存\n\n"
        "**状态：缓存不可用**\n\n"
        f"原因：{reason}。请选择正确期数并重新运行“重算竞技场己方总分”。\n\n"
        "> 本页由本地缓存自动生成；不会展示原始模拟样本、截图或内部路径。\n"
    )


def _unloaded_summary(reason: str) -> str:
    return (
        "### 当前己方缓存\n\n"
        "**状态：摘要未加载**\n\n"
        f"原因：{reason}。权威 JSON 缓存未被删除，实际复用仍由任务运行时校验。\n\n"
        "> 如需查看本页，请用常规模拟次数重新运行“重算竞技场己方总分”。\n"
    )


def _grade_summary_line(state: OwnGradeState) -> str:
    source_labels = {
        "manual_override": "人工覆盖",
        "recognized": "己方读取",
        "unrecognized": "未识别",
    }
    grade = "未识别" if state.effective_grade is None else str(state.effective_grade)
    return f"- 有效 Grade：{grade}（来源：{source_labels[state.source]}）"


def _empty_member_average_table() -> list[str]:
    lines = [
        "| 舞台/栏位 | 成员分数平均值 |",
        "| --- | ---: |",
    ]
    for stage_number in range(1, 4):
        for slot in range(1, 4):
            lines.append(f"| {stage_number}/{slot} | 未识别 |")
    return lines


def _missing_summary() -> str:
    lines = [
        "### 当前己方缓存",
        "",
        "**状态：尚未生成**",
        "",
        _grade_summary_line(OwnGradeState(None, None, None, "unrecognized")),
        "",
        *_empty_member_average_table(),
        "",
        "请先选择正确的竞技场期数，再运行“重算竞技场己方总分”。",
        "",
        "> 重算完成后，切换到其他任务再返回本页即可刷新摘要。",
        "",
    ]
    return "\n".join(lines)


def _render_summary(record: Mapping[str, Any], *, cache_mtime: float | None = None) -> str:
    _cache_schema_version(record)
    snapshot_value = record.get("own_snapshot")
    if not isinstance(snapshot_value, Mapping):
        raise _CacheSummaryError("己方快照缺失")
    try:
        snapshot = validate_own_snapshot(snapshot_value)
    except SnapshotValidationError as error:
        raise _CacheSummaryError("己方快照结构不完整") from error
    grade_state = _grade_state(record, snapshot)

    season = _positive_integer(record.get("season"), field="season")
    simulations = _positive_integer(record.get("simulations"), field="simulations")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise _CacheSummaryError("seed 不是整数")
    stage_ids = record.get("stageIds")
    if (
        not isinstance(stage_ids, list)
        or len(stage_ids) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in stage_ids)
        or len(set(stage_ids)) != 3
    ):
        raise _CacheSummaryError("stageIds 必须是三个不同的正整数")
    if season != snapshot.get("season") or stage_ids != snapshot.get("stageIds"):
        raise _CacheSummaryError("缓存头与己方快照的期数或舞台不一致")
    upstream_commit = record.get("upstream_commit")
    if not isinstance(upstream_commit, str) or re.fullmatch(r"[0-9a-f]{40}", upstream_commit) is None:
        raise _CacheSummaryError("模拟器版本无效")

    cache_stages = record.get("stages")
    own_team = snapshot["own_team"]
    team_stages = own_team["stages"]
    if not isinstance(cache_stages, list) or len(cache_stages) != 3:
        raise _CacheSummaryError("缓存必须包含三个舞台的分数")

    stage_rows: list[tuple[int, int, int, dict[str, int | float]]] = []
    member_rows: dict[
        tuple[int, int],
        tuple[list[int], list[int], int, int, int | None, int],
    ] = {}
    total_members = 0
    total_p_items = 0
    total_skill_cards = 0
    total_customized_cards = 0
    total_customizations = 0
    total_duplicates = 0
    duplicates_fully_recorded = True

    for index, (cache_stage, team_stage) in enumerate(zip(cache_stages, team_stages, strict=True)):
        if not isinstance(cache_stage, Mapping) or not isinstance(team_stage, Mapping):
            raise _CacheSummaryError(f"舞台 {index + 1} 的结构无效")
        stage_number = index + 1
        stage_id = stage_ids[index]
        if (
            cache_stage.get("stage_number") != stage_number
            or cache_stage.get("stage_id") != stage_id
            or team_stage.get("stage_number") != stage_number
            or team_stage.get("stageId") != stage_id
        ):
            raise _CacheSummaryError(f"舞台 {stage_number} 的编号或 ID 不一致")
        members = team_stage.get("members")
        member_scores = cache_stage.get("member_scores")
        if not all(isinstance(items, (list, tuple)) for items in (members, member_scores)):
            raise _CacheSummaryError(f"舞台 {stage_number} 的成员分数结构无效")
        if len(members) != len(member_scores):
            raise _CacheSummaryError(f"舞台 {stage_number} 的成员数量与分数数量不一致")
        total_members += len(members)
        validated_member_scores: list[list[int]] = []

        for member_index, (member, scores) in enumerate(
            zip(members, member_scores, strict=True),
            start=1,
        ):
            if not isinstance(member, Mapping) or not isinstance(scores, (list, tuple)):
                raise _CacheSummaryError(f"舞台 {stage_number} 成员 {member_index} 的结构无效")
            if len(scores) != simulations or any(
                isinstance(score, bool) or not isinstance(score, int) or score < 0 for score in scores
            ):
                raise _CacheSummaryError(f"舞台 {stage_number} 成员 {member_index} 的原始样本无效")
            validated_scores = list(scores)
            validated_member_scores.append(validated_scores)
            slot = _positive_integer(member.get("slot"), field=f"舞台 {stage_number} 成员栏位")
            if slot > 3 or (stage_number, slot) in member_rows:
                raise _CacheSummaryError(f"舞台 {stage_number} 成员栏位无效或重复")
            loadout = member.get("loadout")
            if not isinstance(loadout, Mapping):
                raise _CacheSummaryError(f"舞台 {stage_number} 栏位 {slot} 的配置缺失")
            params = loadout.get("params")
            p_item_ids = loadout.get("pItemIds")
            skill_groups = loadout.get("skillCardIdGroups")
            customization_groups = loadout.get("customizationGroups")
            duplicate_groups = loadout.get("excludedDuplicateGroups")
            if not (
                isinstance(params, list)
                and len(params) == 4
                and isinstance(p_item_ids, list)
                and isinstance(skill_groups, list)
                and isinstance(customization_groups, list)
            ):
                raise _CacheSummaryError(f"舞台 {stage_number} 栏位 {slot} 的关键字段无效")
            visible_p_items = [item for item in p_item_ids if item != 0]
            visible_skills = [item for group in skill_groups for item in group if item != 0]
            customized_cards = 0
            customization_count = 0
            for group in customization_groups:
                for entry in group:
                    count = sum(entry.values())
                    if count:
                        customized_cards += 1
                        customization_count += count
            duplicate_count = (
                sum(bool(flag) for group in duplicate_groups for flag in group)
                if isinstance(duplicate_groups, (list, tuple))
                else None
            )
            total_p_items += len(visible_p_items)
            total_skill_cards += len(visible_skills)
            total_customized_cards += customized_cards
            total_customizations += customization_count
            if duplicate_count is None:
                duplicates_fully_recorded = False
            else:
                total_duplicates += duplicate_count
            member_rows[(stage_number, slot)] = (
                list(params),
                visible_p_items,
                len(visible_skills),
                customization_count,
                duplicate_count,
                _round_half_up_mean(validated_scores),
            )

        raw_team_scores = [
            sum(sample)
            for sample in zip(*validated_member_scores, strict=True)
        ]
        stage_rows.append(
            (stage_number, stage_id, len(members), _summarize_scores(raw_team_scores))
        )

    saved_at = record.get("saved_at_utc")
    local_time = _format_local_time(saved_at) if isinstance(saved_at, str) else None
    if local_time is not None:
        time_line = f"- 缓存写入时间：{local_time}"
    elif cache_mtime is not None:
        local_time = datetime.fromtimestamp(cache_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        time_line = f"- 缓存文件时间：{local_time}（旧缓存未记录写入时间，不等同于采集时间）"
    else:
        time_line = "- 缓存写入时间：旧缓存未记录"
    source_labels = {"live_screen": "游戏画面实读", "frozen_replay": "冻结回放"}
    source = source_labels.get(str(snapshot.get("source")), str(snapshot.get("source")))
    support_bonus = float(own_team["supportBonus"]) * 100
    read_seconds = own_team.get("read_wall_seconds")

    duplicate_summary = f"{total_duplicates} 张" if duplicates_fully_recorded else "未知（缓存未记录）"
    lines = [
        "### 当前己方缓存",
        "",
        "**状态：已生成**（是否可复用仍取决于所选期数、模拟器版本、seed 与请求样本数）",
        "",
        f"- 竞技场期数：第 {season} 期",
        f"- 舞台 ID：{' / '.join(str(item) for item in stage_ids)}",
        f"- 模拟样本：{simulations:,} 次；seed：{seed}",
        f"- 模拟器版本：{upstream_commit[:12]}",
        f"- 来源：{source}；支援加成：{support_bonus:.2f}%",
        _grade_summary_line(grade_state),
        time_line,
    ]
    if isinstance(read_seconds, (int, float)) and not isinstance(read_seconds, bool):
        lines.append(f"- 己方读取耗时：{float(read_seconds):.1f} 秒")
    lines.extend(
        [
            "- 完整性："
            f"3 舞台 / {total_members} 人 / {total_p_items} 个 P 道具 / {total_skill_cards} 张有效技能卡 / "
            f"{total_customized_cards} 张带强化卡（强化 {total_customizations} 次）/ 排除重复 {duplicate_summary}",
            "",
            "| 舞台 | ID | 成员 | 总分均值 | 总分中位数 | 四分位区间 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage_number, stage_id, member_count, stage_distribution in stage_rows:
        lines.append(
            f"| {stage_number} | {stage_id} | {member_count} | "
            f"{_format_score(stage_distribution['mean'])} | {_format_score(stage_distribution['median'])} | "
            f"{_format_score(stage_distribution['q1'])}–{_format_score(stage_distribution['q3'])} |"
        )
    lines.extend(
        [
            "",
            "| 舞台/栏位 | Vo / Da / Vi / 体力 | P 道具 ID | 有效卡 | 强化次数 | 排除重复 | 成员分数平均值 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage_number in range(1, 4):
        for slot in range(1, 4):
            row = member_rows.get((stage_number, slot))
            if row is None:
                lines.append(
                    f"| {stage_number}/{slot} | 未识别 | 未识别 | 未识别 | 未识别 | 未识别 | 未识别 |"
                )
                continue
            params, p_items, card_count, customization_count, duplicates, mean = row
            p_item_text = ", ".join(str(item) for item in p_items) if p_items else "无"
            duplicate_text = str(duplicates) if duplicates is not None else "未知（缓存未记录）"
            lines.append(
                f"| {stage_number}/{slot} | {' / '.join(str(item) for item in params)} | {p_item_text} | "
                f"{card_count} | {customization_count} | {duplicate_text} | {_format_score(mean)} |"
            )
    lines.extend(
        [
            "",
            "> 本页由权威 JSON 缓存自动派生，不展示原始模拟样本、截图或内部路径。"
            "每日挑战默认不会自动重读己方；无缓存或期数、编成等变化后，可在胜率模式中"
            "临时启用“自动重算己方数据”，也可运行独立重算任务。"
            "重算后切换到其他任务再返回本页即可刷新。",
            "",
        ]
    )
    return "\n".join(lines)


class OwnScoreCacheError(RuntimeError):
    """Raised when a present cache cannot be reused without a manual recalculation."""

    def __init__(self, message: str, *, technical_detail: str | None = None) -> None:
        super().__init__(message)
        self.technical_detail = technical_detail


class OwnScoreResimulationRequired(OwnScoreCacheError):
    """Raised when a valid cached lineup only needs new simulator samples."""


class OwnScoreCacheStore:
    """Persist one own lineup and its raw member samples for the active season."""

    def __init__(self, path: str | Path, *, summary_path: str | Path | None = None) -> None:
        self.path = Path(path).resolve()
        self.summary_path = (
            Path(summary_path).resolve()
            if summary_path is not None
            else self.path.with_name(DEFAULT_OWN_SCORE_SUMMARY.name)
        )
        self.last_summary_error: str | None = None
        self.last_summary_error_detail: str | None = None

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content.encode("utf-8"))
            temporary.replace(path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_record(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise OwnScoreCacheError("own-score cache is unreadable; use manual recalculation") from error
        if not isinstance(value, Mapping):
            raise OwnScoreCacheError("own-score cache schema changed; use manual recalculation")
        try:
            _cache_schema_version(value)
        except _CacheSummaryError as error:
            raise OwnScoreCacheError("own-score cache schema changed; use manual recalculation") from error
        return dict(value)

    def get_grade_state(self) -> OwnGradeState:
        """Return recognized, overridden, and effective Grade without changing the cache."""

        if not self.path.is_file():
            return OwnGradeState(None, None, None, "unrecognized")
        record = self._read_record()
        snapshot_value = record.get("own_snapshot")
        if not isinstance(snapshot_value, Mapping):
            raise OwnScoreCacheError("cached own lineup is invalid; use manual recalculation")
        try:
            snapshot = validate_own_snapshot(snapshot_value)
            return _grade_state(record, snapshot)
        except (SnapshotValidationError, _CacheSummaryError) as error:
            raise OwnScoreCacheError("cached arena Grade is invalid; use manual recalculation") from error

    def set_grade_override(self, value: int | None) -> OwnGradeState:
        """Set or clear the manual Grade override and atomically persist the cache."""

        try:
            normalized = _grade_value(value, field="grade_override", allow_none=True)
        except _CacheSummaryError as error:
            raise ValueError(str(error)) from error
        if not self.path.is_file():
            raise OwnScoreCacheError("own-score cache is missing; use manual recalculation")
        record = self._read_record()
        snapshot_value = record.get("own_snapshot")
        if not isinstance(snapshot_value, Mapping):
            raise OwnScoreCacheError("cached own lineup is invalid; use manual recalculation")
        try:
            snapshot = validate_own_snapshot(snapshot_value)
            current_state = _grade_state(record, snapshot)
        except (SnapshotValidationError, _CacheSummaryError) as error:
            raise OwnScoreCacheError("cached arena Grade is invalid; use manual recalculation") from error
        if (
            record.get("schema_version") == CACHE_SCHEMA_VERSION
            and record.get("grade_override") == normalized
        ):
            return current_state
        updated = dict(record)
        updated["schema_version"] = CACHE_SCHEMA_VERSION
        updated["grade_override"] = normalized
        try:
            summary = _render_summary(updated)
        except _CacheSummaryError as error:
            raise OwnScoreCacheError(f"own-score cache summary data is invalid: {error}") from error
        try:
            self._atomic_write(
                self.path,
                json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True),
            )
        except OSError as error:
            raise OwnScoreCacheError(
                f"own-score cache could not be written: {_summary_write_reason(error)}",
                technical_detail=f"cache_path={self.path!s}; {error!r}",
            ) from error
        state = _grade_state(updated, snapshot)
        self.last_summary_error = None
        self.last_summary_error_detail = None
        try:
            cache_stat = self.path.stat()
            self._atomic_write(
                self.summary_path,
                _with_summary_header(
                    summary,
                    cache_size=cache_stat.st_size,
                    cache_mtime_ns=cache_stat.st_mtime_ns,
                ),
            )
        except OSError as error:
            self.last_summary_error = f"摘要文件写入失败：{_summary_write_reason(error)}；下次启动将重试"
            self.last_summary_error_detail = f"summary_path={self.summary_path!s}; {error!r}"
        return state

    def refresh_summary(self) -> str:
        """Rebuild the user-facing Markdown from the authoritative JSON cache."""

        cache_size: int | None = None
        cache_mtime_ns: int | None = None
        if not self.path.is_file():
            content = _missing_summary()
        else:
            try:
                cache_stat = self.path.stat()
            except OSError:
                content = _invalid_summary("缓存文件状态无法读取")
            else:
                cache_size = cache_stat.st_size
                cache_mtime_ns = cache_stat.st_mtime_ns
                if cache_stat.st_size > MAX_SUMMARY_CACHE_BYTES:
                    content = _unloaded_summary(
                        f"缓存文件超过 {MAX_SUMMARY_CACHE_BYTES // (1024 * 1024)} MiB"
                    )
                else:
                    try:
                        value = json.loads(self.path.read_text(encoding="utf-8-sig"))
                    except OSError:
                        content = _invalid_summary("缓存文件无法读取")
                    except json.JSONDecodeError:
                        content = _invalid_summary("缓存文件不是有效 JSON")
                    else:
                        if not isinstance(value, Mapping):
                            content = _invalid_summary("缓存根结构不是对象")
                        else:
                            try:
                                content = _render_summary(value, cache_mtime=cache_stat.st_mtime)
                            except _CacheSummaryError as error:
                                content = _invalid_summary(str(error))
        content = _with_summary_header(
            content,
            cache_size=cache_size,
            cache_mtime_ns=cache_mtime_ns,
        )
        self._atomic_write(self.summary_path, content)
        return content

    def ensure_summary(self) -> str:
        """Reuse an up-to-date derived summary, rebuilding only when needed."""

        try:
            cache_stat = self.path.stat() if self.path.is_file() else None
            summary_stat = self.summary_path.stat() if self.summary_path.is_file() else None
            if summary_stat is None or summary_stat.st_size > MAX_SUMMARY_FILE_BYTES:
                return self.refresh_summary()
            with self.summary_path.open("rb") as stream:
                raw = stream.read(MAX_SUMMARY_FILE_BYTES + 1)
            if len(raw) > MAX_SUMMARY_FILE_BYTES:
                return self.refresh_summary()
            content = raw.decode("utf-8-sig")
            first_line = content.partition("\n")[0].rstrip("\r")
            match = _SUMMARY_HEADER.fullmatch(first_line)
            expected_header = _summary_header(
                cache_size=cache_stat.st_size if cache_stat is not None else None,
                cache_mtime_ns=cache_stat.st_mtime_ns if cache_stat is not None else None,
            )
            if (
                match is not None
                and int(match.group(1)) == SUMMARY_RENDER_VERSION
                and first_line == expected_header
            ):
                return content
        except (OSError, UnicodeError):
            pass
        return self.refresh_summary()

    def save(
        self,
        own_snapshot: Mapping[str, Any],
        batch: OwnScoreBatch,
        *,
        simulations: int,
        seed: int,
    ) -> dict[str, Any]:
        snapshot = validate_own_snapshot(own_snapshot)
        grade_override = None
        if self.path.is_file():
            try:
                existing = self._read_record()
                if existing.get("schema_version") == CACHE_SCHEMA_VERSION:
                    grade_override = _grade_value(
                        existing.get("grade_override"),
                        field="grade_override",
                        allow_none=True,
                    )
            except (OwnScoreCacheError, _CacheSummaryError):
                grade_override = None
        stages: list[dict[str, Any]] = []
        for stage in batch.stage_distributions:
            raw = stage.get("own_member_scores")
            if not isinstance(raw, tuple):
                raise OwnScoreCacheError("own-score adapter did not return reusable raw member samples")
            stages.append(
                {
                    "stage_number": stage["stage_number"],
                    "stage_id": stage["stage_id"],
                    "member_scores": [list(scores) for scores in raw],
                    "member_distributions": stage["own_member_distributions"],
                    "raw_team_distribution": stage["own_raw_team_distribution"],
                }
            )
        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "grade_override": grade_override,
            "upstream_commit": batch.upstream_commit,
            "season": snapshot["season"],
            "stageIds": snapshot["stageIds"],
            "simulations": simulations,
            "seed": seed,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "own_snapshot": snapshot,
            "stages": stages,
        }
        try:
            summary = _render_summary(record)
        except _CacheSummaryError as error:
            raise OwnScoreCacheError(f"own-score cache summary data is invalid: {error}") from error
        try:
            self._atomic_write(
                self.path,
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
            )
        except OSError as error:
            raise OwnScoreCacheError(
                f"own-score cache could not be written: {_summary_write_reason(error)}",
                technical_detail=f"cache_path={self.path!s}; {error!r}",
            ) from error
        self.last_summary_error = None
        self.last_summary_error_detail = None
        try:
            cache_stat = self.path.stat()
            self._atomic_write(
                self.summary_path,
                _with_summary_header(
                    summary,
                    cache_size=cache_stat.st_size,
                    cache_mtime_ns=cache_stat.st_mtime_ns,
                ),
            )
        except OSError as error:
            self.last_summary_error = f"摘要文件写入失败：{_summary_write_reason(error)}；下次启动将重试"
            self.last_summary_error_detail = f"summary_path={self.summary_path!s}; {error!r}"
        return record

    def load_for_match(
        self,
        *,
        season: int,
        stage_ids: tuple[int, int, int],
        simulations: int,
        seed: int,
        expected_upstream_commit: str = UPSTREAM_COMMIT,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Return own snapshot and engine cache, or ``None`` only for first/next season."""

        if re.fullmatch(r"[0-9a-f]{40}", expected_upstream_commit) is None:
            raise ValueError("expected_upstream_commit must be a lowercase 40-character SHA")
        if not self.path.is_file():
            return None
        record = self._read_record()
        if record.get("season") != season or record.get("stageIds") != list(stage_ids):
            return None
        if record.get("upstream_commit") != expected_upstream_commit:
            raise OwnScoreResimulationRequired("simulator revision changed; resimulate the cached lineup")
        if record.get("seed") != seed:
            raise OwnScoreResimulationRequired("own-score seed changed; resimulate the cached lineup")
        cached_simulations = record.get("simulations")
        if isinstance(cached_simulations, bool) or not isinstance(cached_simulations, int):
            raise OwnScoreCacheError("own-score cache sample count is invalid")
        if simulations > cached_simulations:
            raise OwnScoreResimulationRequired(
                f"own-score cache has {cached_simulations} samples; resimulate the cached lineup for {simulations}"
            )
        try:
            own_snapshot = validate_own_snapshot(record["own_snapshot"])
            stages = record["stages"]
            if not isinstance(stages, list) or len(stages) != 3:
                raise TypeError
            cache_stages: list[dict[str, Any]] = []
            for index, stage in enumerate(stages):
                scores = stage["member_scores"]
                if not isinstance(scores, list) or not scores:
                    raise TypeError
                sliced = []
                for member_scores in scores:
                    if (
                        not isinstance(member_scores, list)
                        or len(member_scores) != cached_simulations
                        or any(
                            isinstance(score, bool) or not isinstance(score, int) or score < 0
                            for score in member_scores
                        )
                    ):
                        raise TypeError
                    sliced.append(member_scores[:simulations])
                cache_stages.append(
                    {
                        "stage_number": index + 1,
                        "stage_id": stage_ids[index],
                        "member_scores": sliced,
                    }
                )
        except (KeyError, TypeError, SnapshotValidationError) as error:
            raise OwnScoreCacheError("own-score cache contents are invalid; use manual recalculation") from error
        engine_cache = {
            "upstream_commit": expected_upstream_commit,
            "season": season,
            "stageIds": list(stage_ids),
            "simulations": simulations,
            "seed": seed,
            "stages": cache_stages,
        }
        return own_snapshot, engine_cache

    def load_snapshot_for_resimulation(
        self,
        *,
        season: int,
        stage_ids: tuple[int, int, int],
    ) -> dict[str, Any] | None:
        """Load only the lineup identity; old score samples remain unusable."""

        if not self.path.is_file():
            return None
        record = self._read_record()
        if record.get("season") != season or record.get("stageIds") != list(stage_ids):
            return None
        try:
            return validate_own_snapshot(record["own_snapshot"])
        except (KeyError, SnapshotValidationError) as error:
            raise OwnScoreCacheError("cached own lineup is invalid; use manual recalculation") from error
