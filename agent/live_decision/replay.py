from __future__ import annotations

import json
import statistics
from typing import Any, Iterable
from pathlib import Path
from dataclasses import asdict, dataclass

from .contracts import LiveSafetyFilter, LiveStateProvider, LiveDecisionProvider


@dataclass(frozen=True)
class ReplayMetrics:
    sample_count: int
    action_target_count: int
    known_state_count: int
    proposal_count: int
    allowed_count: int
    action_target_match_count: int
    expected_outcome_match_count: int
    timeout_count: int
    coverage: float
    legal_action_rate: float
    target_achievement_rate: float
    expected_outcome_match_rate: float
    safety_effective_rate: float
    unknown_state_rate: float
    timeout_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_replay(
    samples: Iterable[dict[str, Any]],
    state_provider: LiveStateProvider,
    decision_provider: LiveDecisionProvider,
    safety_filter: LiveSafetyFilter,
) -> tuple[ReplayMetrics, list[dict[str, Any]]]:
    samples = tuple(samples)
    records: list[dict[str, Any]] = []
    for sample in samples:
        state = state_provider.build_state(sample["state"])
        proposal = decision_provider.decide(state)
        safe = safety_filter.filter(state, proposal)
        target = sample.get("expected_action")
        actual = safe.action.to_dict() if safe.action is not None else None
        records.append(
            {
                "sample_id": sample["sample_id"],
                "state_unknown": bool(state.unknown_fields),
                "proposal": proposal.to_dict(),
                "safe_decision": safe.to_dict(),
                "target_match": actual == target,
            }
        )
    total = len(records)
    latencies = sorted(float(record["proposal"]["elapsed_ms"]) for record in records)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95_index = max(0, min(len(latencies) - 1, int(0.95 * len(latencies) + 0.999999) - 1)) if latencies else 0
    p95 = latencies[p95_index] if latencies else 0.0
    known = sum(not record["state_unknown"] for record in records)
    proposals = sum(record["proposal"]["status"] == "PROPOSE" for record in records)
    allowed = sum(record["safe_decision"]["status"] == "ALLOW" for record in records)
    action_targets = sum(sample.get("expected_action") is not None for sample in samples)
    action_target_matches = sum(
        record["target_match"] and record["safe_decision"]["action"] is not None for record in records
    )
    outcome_matches = sum(record["target_match"] for record in records)
    timeouts = sum(record["safe_decision"]["reason"] == "provider_timeout" for record in records)
    metrics = ReplayMetrics(
        sample_count=total,
        action_target_count=action_targets,
        known_state_count=known,
        proposal_count=proposals,
        allowed_count=allowed,
        action_target_match_count=action_target_matches,
        expected_outcome_match_count=outcome_matches,
        timeout_count=timeouts,
        coverage=_ratio(proposals, total),
        legal_action_rate=_ratio(allowed, proposals),
        target_achievement_rate=_ratio(action_target_matches, action_targets),
        expected_outcome_match_rate=_ratio(outcome_matches, total),
        safety_effective_rate=_ratio(allowed, total),
        unknown_state_rate=_ratio(total - known, total),
        timeout_rate=_ratio(timeouts, total),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
    )
    return metrics, records


def load_replay(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("samples"), list):
        raise ValueError("unsupported Live replay manifest")
    return raw["samples"]


def write_report(path: str | Path, metrics: ReplayMetrics, records: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        json.dumps({"schema_version": 1, "metrics": asdict(metrics), "records": records}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
