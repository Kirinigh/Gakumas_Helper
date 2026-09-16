"""GitHub cooldowns: shared anonymous disk state or authenticated memory state.

Only a UTC resume timestamp is persisted. This is network state, independent of
the active RIS component and of any authenticated client's request budget.
"""

from __future__ import annotations

import os
import json
import math
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from email.utils import parsedate_to_datetime


class GitHubRateLimitError(RuntimeError):
    _arena_component_retry_exhausted = True

    def __init__(self, resume_at: float, *, persistent: bool = True):
        self.resume_at = resume_at
        date = datetime.fromtimestamp(resume_at, timezone.utc)
        try:
            date = date.astimezone()
        except (OverflowError, ValueError):
            pass
        label = date.strftime("%Y-%m-%d %H:%M:%S %z")
        waiting = "等待期间重启也不会重复请求" if persistent else "本进程等待期间不会重复请求"
        super().__init__(f"GitHub 请求额度受限，{label} 后可重新检查；{waiting}。")


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 253402300799 else None


def rate_limit_resume(status, headers, body: str, now: float) -> float | None:
    """Keep permission errors separate and honor the longest server deadline."""
    if status not in (403, 429):
        return None
    headers = {str(key).lower(): value for key, value in (headers or {}).items()}
    exhausted = str(headers.get("x-ratelimit-remaining", "")) == "0"
    retry = headers.get("retry-after")
    if not (status == 429 or exhausted or retry is not None or "rate limit" in body.lower()):
        return None
    resume = now + 60
    seconds = _number(retry)
    if seconds is not None and seconds > 0:
        resume = now + seconds
    elif isinstance(retry, str):
        try:
            date = parsedate_to_datetime(retry)
            if date.tzinfo is not None and date.timestamp() > now:
                resume = date.timestamp()
        except (ValueError, TypeError, OverflowError):
            pass
    reset = _number(headers.get("x-ratelimit-reset"))
    if exhausted and reset is not None:
        resume = max(resume, reset)
    return min(resume, 253402300799)


class GitHubRateLimit:
    """A None path keeps authenticated state out of the anonymous IP budget."""

    def __init__(self, path: Path | None, *, clock=time.time, on_error=None):
        self.path = path
        self.clock = clock
        self.on_error = on_error or (lambda error: None)
        self._resume_at = 0.0
        self._gate = threading.RLock()

    def _read(self) -> float:
        if self.path is None:
            return 0.0
        try:
            state = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if isinstance(state, dict) and type(state.get("schema_version")) is int and state["schema_version"] == 1:
                value = state.get("resume_at")
                if type(value) in (int, float) and _number(value) is not None:
                    return float(value)
            raise ValueError("GitHub cooldown state has an invalid schema or resume_at")
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            self.on_error(error)
        return 0.0

    @contextmanager
    def _file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Same byte range as FileStream.Lock(0, 1) in the desktop updater.
        with self.path.with_suffix(".lock").open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                lock = lambda: fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1)
                unlock = lambda: fcntl.lockf(handle.fileno(), fcntl.LOCK_UN, 1)
            for attempt in range(4):
                try:
                    handle.seek(0)
                    lock()
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                unlock()

    def check(self) -> None:
        with self._gate:
            # Atomic replacement lets readers use either complete state.
            self._resume_at = max(self._resume_at, self._read())
            if self._resume_at > self.clock():
                raise GitHubRateLimitError(self._resume_at, persistent=self.path is not None)

    def record(self, resume_at: float) -> None:
        with self._gate:
            self._resume_at = max(self._resume_at, resume_at)
            if self.path is None:
                return
            temporary = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with self._file_lock():
                    self._resume_at = max(self._resume_at, self._read())
                    temporary.write_text(
                        json.dumps({"schema_version": 1, "resume_at": self._resume_at}), encoding="utf-8",
                    )
                    os.replace(temporary, self.path)
            except OSError as error:
                # Disk failure must not lose this process's cooldown or turn
                # one rate-limit response into another immediate request.
                self.on_error(error)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    self.on_error(error)

    def reject(self, status, headers, body: str) -> None:
        resume = rate_limit_resume(status, headers, body, self.clock())
        if resume is not None:
            self.record(resume)
            raise GitHubRateLimitError(self._resume_at, persistent=self.path is not None)
