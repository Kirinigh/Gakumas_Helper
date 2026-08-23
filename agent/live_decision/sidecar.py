from __future__ import annotations

import json
import time
import subprocess
from dataclasses import dataclass

from .contracts import PROTOCOL_VERSION, LiveState, LiveAction, DecisionProposal


@dataclass(frozen=True)
class JsonSidecarProvider:
    """Explicit stdin/stdout boundary for isolated search or RL candidates."""

    command: tuple[str, ...]
    timeout_ms: float
    provider_name: str
    provider_version: str

    def decide(self, state: LiveState) -> DecisionProposal:
        started = time.perf_counter()
        request = json.dumps(
            {"protocol_version": PROTOCOL_VERSION, "operation": "decide", "state": state.to_dict()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            result = subprocess.run(
                list(self.command),
                input=request,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_ms / 1000.0,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            reason = "sidecar_timeout" if isinstance(error, subprocess.TimeoutExpired) else "sidecar_unavailable"
            return self._stop(reason, elapsed_ms)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.returncode != 0:
            return self._stop("sidecar_nonzero_exit", elapsed_ms)
        try:
            raw = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return self._stop("sidecar_invalid_json", elapsed_ms)
        if raw.get("protocol_version") != PROTOCOL_VERSION:
            return self._stop("sidecar_protocol_mismatch", elapsed_ms)
        action_raw = raw.get("action")
        action = None
        if isinstance(action_raw, dict):
            action = LiveAction(kind=str(action_raw.get("kind", "")), card_slot=action_raw.get("card_slot"))
        return DecisionProposal(
            status=str(raw.get("status", "STOP")),
            reason=str(raw.get("reason", "sidecar_reason_missing")),
            action=action,
            confidence=float(raw.get("confidence", 0.0)),
            provider=self.provider_name,
            provider_version=self.provider_version,
            elapsed_ms=elapsed_ms,
            scores={str(key): float(value) for key, value in raw.get("scores", {}).items()},
        )

    def _stop(self, reason: str, elapsed_ms: float) -> DecisionProposal:
        return DecisionProposal(
            status="STOP",
            reason=reason,
            action=None,
            confidence=0.0,
            provider=self.provider_name,
            provider_version=self.provider_version,
            elapsed_ms=elapsed_ms,
        )
