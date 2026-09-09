"""Recognize a retry opportunity from the already observed error dialog."""

from __future__ import annotations

import re


def error_retry_box(rows) -> tuple[int, int, int, int] | None:
    """Require error evidence and exactly one named retry control.

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
    if not has_error or len(retry) != 1:
        return None
    try:
        box = tuple(int(value) for value in retry[0].box)
    except (AttributeError, TypeError, ValueError):
        return None
    if len(box) != 4 or min(box[:2]) < 0 or min(box[2:]) <= 0:
        return None
    return box
