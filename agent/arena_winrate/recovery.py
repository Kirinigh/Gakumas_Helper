"""Recognize a retry opportunity from the already observed error dialog."""

from __future__ import annotations

import re
import unicodedata


def _control_box(row) -> tuple[int, int, int, int] | None:
    try:
        box = tuple(int(value) for value in row.box)
    except (AttributeError, TypeError, ValueError):
        return None
    return box if len(box) == 4 and min(box[:2]) >= 0 and min(box[2:]) > 0 else None


def _confirmation_retry_box(rows) -> tuple[int, int, int, int] | None:
    """Bind Yes to an explicit retry question, never to a generic confirmation."""
    texts = [(row, re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(getattr(row, "text", ""))))) for row in rows]
    groups = [
        [row for row, text in texts if re.fullmatch(pattern, text)]
        for pattern in (r"通信エラ[ー一]?", r"(?:リトライ|再試行|再接続)しますか\?", r"はい", r"キャンセル")
    ]
    if any(len(group) != 1 for group in groups):
        return None
    boxes = [_control_box(group[0]) for group in groups]
    if any(box is None for box in boxes):
        return None
    title, question, yes, cancel = boxes
    if not (
        title[1] + title[3] <= question[1]
        and question[1] + question[3] <= min(yes[1], cancel[1])
        and cancel[0] + cancel[2] <= yes[0]
        and abs((yes[1] + yes[3] / 2) - (cancel[1] + cancel[3] / 2)) <= max(yes[3], cancel[3]) / 2
    ):
        return None
    return yes


def error_retry_box(rows) -> tuple[int, int, int, int] | None:
    """Require error evidence and one named Retry or a bound retry confirmation.

    Error descriptions vary (including E203), and a return-to-title button is
    optional. Neither the full error sentence nor its code owns retry eligibility.
    """
    rows = tuple(rows)
    retry = [row for row in rows if re.fullmatch(
        r"リトライ|再試行|再接続|重试|重試|Retry",
        str(getattr(row, "text", "")).strip(), re.IGNORECASE,
    )]
    has_error = any(re.search(
        r"エラ[ー一]?|失敗|通信|接続|连接失败|連線失敗|错误|錯誤|\berror\b|\bfailed\b",
        str(getattr(row, "text", "")), re.IGNORECASE,
    ) for row in rows if all(row is not control for control in retry))
    if not has_error:
        return None
    if retry:
        return _control_box(retry[0]) if len(retry) == 1 else None
    return _confirmation_retry_box(rows)
