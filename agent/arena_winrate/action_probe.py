"""Bounded action lifecycle diagnostics; no parameters, images or device access."""
from __future__ import annotations

import time
from uuid import uuid4
from threading import Lock

from .recognition_probe import RecognitionProbe


class ActionProbe:
    def __init__(self, root):
        self.policy = RecognitionProbe(root)
        self.remaining = 256
        self.lock = Lock()

    def start(self, action: str, node: str, emit):
        try:
            with self.lock:
                if not self.policy.active() or self.remaining <= 0:
                    return None
                self.remaining -= 1
            entry = {"event": "arena_action_probe", "action_id": uuid4().hex,
                     "action": action, "node": node, "state": "started"}
            emit(entry)
            return entry, time.monotonic()
        except Exception:
            return None

    @staticmethod
    def finish(ticket, state: str, error_type: str | None, emit):
        if ticket is None:
            return
        try:
            entry, started = ticket
            emit({**entry, "state": state, "error_type": error_type,
                  "elapsed_seconds": round(time.monotonic() - started, 6)})
        except Exception:
            pass
