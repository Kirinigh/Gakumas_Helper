"""Cooperative cancellation shared by one arena action and its reader/adapter."""

from __future__ import annotations

from contextvars import ContextVar
from collections.abc import Callable


class ArenaTaskCancelled(BaseException):
    """Unwind business retries, like asyncio cancellation, to the action boundary.

    Cancellation is deliberately not an Exception: reader recovery handlers must
    not turn a user's Stop into another page recovery or a second lineup read.
    The registered action wrapper consumes this signal before crossing Maa's ABI.
    """

    code = "task_cancelled"


class ArenaCancellation:
    def __init__(self, stopping: Callable[[], bool]) -> None:
        self._stopping = stopping
        self.requested = False

    def check(self) -> None:
        if self.requested or self._stopping():
            self.requested = True
            raise ArenaTaskCancelled("arena task stopped by user")


active_cancellation: ContextVar[ArenaCancellation | None] = ContextVar(
    "arena_cancellation", default=None,
)


def cancellation_for(context) -> ArenaCancellation:
    active = active_cancellation.get()
    if active is not None:
        return active
    return ArenaCancellation(
        lambda: getattr(getattr(context, "tasker", None), "stopping", False) is True,
    )
