"""Data-only route checkpoints; restored proposals can confirm, never replay.

The host must supply the same verified monotonic-clock epoch across processes.
A fresh process token is the conservative default: it cannot silently renew a
pending deadline from another host or boot. Coordinates and input grants are
not part of this record.
"""

from __future__ import annotations

import os
import json
import tempfile
from uuid import uuid4
from pathlib import Path
from dataclasses import asdict, fields

from .engine import Target, Proposal, Observation, TransitionGate, validate_observation

PROCESS_CLOCK_EPOCH = str(uuid4())


def save_transition_checkpoint(path: str | Path, raw: dict, *, clock_epoch: str) -> bool:
    """Replace one host-owned checkpoint only after a complete validated write."""
    restore_transition(raw, clock_epoch=clock_epoch)
    data = json.dumps(raw, ensure_ascii=False, allow_nan=False).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=destination.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return True
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_transition_checkpoint(path: str | Path, *, clock_epoch: str) -> TransitionGate:
    """Missing/corrupt state is an error, never an empty new transaction."""
    return restore_transition(json.loads(Path(path).read_text(encoding="utf-8")), clock_epoch=clock_epoch)


def transition_checkpoint(gate: TransitionGate, clock_epoch: str) -> dict:
    if not isinstance(clock_epoch, str) or not clock_epoch:
        raise ValueError("missing route clock epoch")
    return {
        "schema_version": 1,
        "clock_epoch": clock_epoch,
        "timeout_ms": gate.timeout_ms,
        "issued_at_ms": gate.issued_at_ms,
        "stopped": gate.stopped,
        "pending": asdict(gate.pending) if gate.pending is not None else None,
    }


def _record(raw: object, cls: type) -> dict:
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(cls)}:
        raise ValueError("invalid route checkpoint record")
    return dict(raw)


def _target(raw: object) -> Target:
    row = _record(raw, Target)
    if any(not isinstance(row[key], str) or not row[key] for key in ("key", "kind")):
        raise ValueError("invalid route target identity")
    if row["attribute"] is not None and not isinstance(row["attribute"], str):
        raise ValueError("invalid route target attribute")
    if any(row[key] is not None and type(row[key]) is not bool for key in ("special", "available")):
        raise ValueError("invalid route target availability")
    return Target(**row)


def restore_transition(raw: object, *, clock_epoch: str) -> TransitionGate:
    """Reject incomplete state or a different clock instead of cold-starting."""
    keys = {"schema_version", "clock_epoch", "timeout_ms", "issued_at_ms", "stopped", "pending"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("invalid route checkpoint fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported route checkpoint version")
    if not isinstance(clock_epoch, str) or not clock_epoch or raw["clock_epoch"] != clock_epoch:
        raise ValueError("route checkpoint clock epoch mismatch")
    if type(raw["stopped"]) is not bool:
        raise ValueError("invalid route checkpoint stopped flag")
    gate = TransitionGate(timeout_ms=raw["timeout_ms"])
    gate.stopped = raw["stopped"]
    if raw["pending"] is None:
        if raw["issued_at_ms"] is not None:
            raise ValueError("route checkpoint timestamp without pending action")
        return gate
    if type(raw["issued_at_ms"]) is not int or raw["issued_at_ms"] < 0:
        raise ValueError("invalid route checkpoint issue time")
    pending = _record(raw["pending"], Proposal)
    source = _record(pending["source"], Observation)
    for key in ("run_id", "scene", "page_id", "idol", "plan", "scenario"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError("missing route checkpoint source identity")
    if type(source["contradictory"]) is not bool or not isinstance(source["targets"], (list, tuple)):
        raise ValueError("invalid route checkpoint source")
    source["targets"] = tuple(_target(target) for target in source["targets"])
    observation = Observation(**source)
    if validate_observation(observation):
        raise ValueError("invalid route checkpoint observation")
    target = _target(pending["target"])
    if target not in observation.targets or target.available is not True:
        raise ValueError("route checkpoint target not in source")
    scenes = pending["expected_scenes"]
    if not isinstance(scenes, (list, tuple)) or not scenes or any(
        not isinstance(scene, str) or not scene for scene in scenes
    ):
        raise ValueError("invalid route checkpoint destination")
    if not isinstance(pending["rule_id"], str) or not pending["rule_id"]:
        raise ValueError("missing route checkpoint rule")
    gate.pending = Proposal(target, observation, pending["rule_id"], tuple(scenes))
    gate.issued_at_ms = raw["issued_at_ms"]
    return gate
