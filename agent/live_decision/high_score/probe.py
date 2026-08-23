from __future__ import annotations

import time
import statistics
from typing import Any
from dataclasses import asdict, dataclass

from .golden import GoldenSuite
from .kernel import PythonNiaProKernel
from .planner import PlannerConfig, RobustHighScorePlanner


@dataclass(frozen=True)
class ArchitectureGate:
    minimum_semantic_match: float = 1.0
    minimum_trace_match: float = 1.0
    maximum_transition_p95_ms: float = 10.0
    maximum_planning_p95_ms: float = 500.0
    maximum_tie_latency_delta_ratio: float = 0.10


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _python_candidate(suite: GoldenSuite, gate: ArchitectureGate, *, rounds: int) -> dict[str, Any]:
    kernel = PythonNiaProKernel(suite.cards, suite.drinks, ruleset_id=suite.ruleset_id)
    planner = RobustHighScorePlanner(
        kernel,
        PlannerConfig(max_decision_depth=8, timeout_ms=2000.0, require_terminal_leaf=True),
    )
    semantic_matches = 0
    trace_matches = 0
    planning_latencies: list[float] = []
    transition_latencies: list[float] = []
    for round_index in range(rounds):
        for case in suite.cases:
            started = time.perf_counter()
            response = planner.decide(case.request)
            planning_latencies.append((time.perf_counter() - started) * 1000.0)
            if round_index == 0:
                if response.outcome is case.expectation.outcome and response.action == case.expectation.action:
                    semantic_matches += 1
                expected_score = case.expectation.expected_final_score
                expected_p10 = case.expectation.p10_final_score
                if expected_score is None and expected_p10 is None:
                    trace_matches += 1
                elif response.value is not None:
                    score_matches = expected_score is None or response.value.expected_final_score == expected_score
                    p10_matches = expected_p10 is None or response.value.p10_final_score == expected_p10
                    if score_matches and p10_matches:
                        trace_matches += 1
            if response.action is not None:
                transition_started = time.perf_counter()
                kernel.transition(case.request.state, response.action, plan_type=case.request.plan_type)
                transition_latencies.append((time.perf_counter() - transition_started) * 1000.0)
    total = len(suite.cases)
    semantic_match = semantic_matches / total if total else 0.0
    trace_match = trace_matches / total if total else 0.0
    transition_p95 = _p95(transition_latencies)
    planning_p95 = _p95(planning_latencies)
    mandatory = {
        "licensed_or_self_authored": True,
        "local_offline_packaging": True,
        "arbitrary_state_injection": True,
        "controlled_chance_branches": True,
        "structured_event_trace": True,
        "semantic_match": semantic_match >= gate.minimum_semantic_match,
        "trace_match": trace_match >= gate.minimum_trace_match,
        "transition_latency": transition_p95 <= gate.maximum_transition_p95_ms,
        "planning_latency": planning_p95 <= gate.maximum_planning_p95_ms,
    }
    return {
        "candidate": "python",
        "implementation": f"{kernel.kernel_name}@{kernel.kernel_version}",
        "status": "PASS" if all(mandatory.values()) else "FAIL",
        "mandatory": mandatory,
        "metrics": {
            "semantic_match": semantic_match,
            "trace_match": trace_match,
            "transition_p95_ms": transition_p95,
            "planning_p50_ms": statistics.median(planning_latencies) if planning_latencies else 0.0,
            "planning_p95_ms": planning_p95,
            "measured_plans": len(planning_latencies),
        },
        "scope": "task275_anomaly_golden_probe_only",
    }


def build_architecture_probe_report(
    suite: GoldenSuite,
    provider_assessments: dict[str, Any],
    *,
    rounds: int = 20,
    gate: ArchitectureGate | None = None,
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("probe rounds must be positive")
    gate = gate or ArchitectureGate()
    python = _python_candidate(suite, gate, rounds=rounds)
    assessments = {str(item.get("provider")): item for item in provider_assessments.get("providers", [])}
    engine = assessments.get("gakumas-tools-engine", {})
    adapter_acquired = engine.get("acquisition") in {"ACQUIRED", "ACTIVE"}
    adapter_protocol_ready = engine.get("same_fixture_status") == "EXECUTED"
    adapter = {
        "candidate": "adapter",
        "implementation": "gakumas-tools-engine",
        "status": "FAIL",
        "mandatory": {
            "licensed_or_self_authored": engine.get("license") == "BSD-3-Clause",
            "local_offline_packaging": adapter_acquired,
            "arbitrary_state_injection": False,
            "controlled_chance_branches": False,
            "structured_event_trace": False,
            "semantic_match": adapter_protocol_ready,
            "trace_match": adapter_protocol_ready,
            "transition_latency": False,
            "planning_latency": False,
        },
        "metrics": dict(engine.get("comparison_metrics", {})),
        "failure_reason": "pinned_engine_not_acquired_or_no_task275_v2_adapter",
    }
    hybrid = {
        "candidate": "hybrid",
        "implementation": "python-orchestration+gakumas-tools-engine",
        "status": "FAIL",
        "mandatory": dict(adapter["mandatory"]),
        "metrics": dict(adapter["metrics"]),
        "failure_reason": "hybrid_requires_a_passing_external_rule_adapter",
    }
    candidates = [python, adapter, hybrid]
    passing = [candidate for candidate in candidates if candidate["status"] == "PASS"]
    if not passing:
        selected = None
        reason = "no_candidate_passed_all_mandatory_gates"
    elif len(passing) == 1:
        selected = passing[0]["candidate"]
        reason = "only_candidate_passing_all_mandatory_gates"
    else:
        # Semantic coverage wins first. Hybrid is the frozen tie-break only when
        # latency is within the approved delta and the external adapter passes.
        by_name = {candidate["candidate"]: candidate for candidate in passing}
        if "hybrid" in by_name:
            best_latency = min(float(candidate["metrics"]["planning_p95_ms"]) for candidate in passing)
            hybrid_latency = float(by_name["hybrid"]["metrics"]["planning_p95_ms"])
            if hybrid_latency <= best_latency * (1.0 + gate.maximum_tie_latency_delta_ratio):
                selected = "hybrid"
                reason = "semantic_tie_and_hybrid_latency_within_ten_percent"
            else:
                selected = min(passing, key=lambda candidate: float(candidate["metrics"]["planning_p95_ms"]))["candidate"]
                reason = "semantic_tie_resolved_by_planning_latency"
        else:
            selected = min(passing, key=lambda candidate: float(candidate["metrics"]["planning_p95_ms"]))["candidate"]
            reason = "semantic_tie_resolved_by_planning_latency"
    return {
        "schema_version": 1,
        "protocol_version": "2.0",
        "dataset_id": suite.dataset_id,
        "ruleset_id": suite.ruleset_id,
        "gate": asdict(gate),
        "candidates": candidates,
        "selection": {
            "selected": selected,
            "reason": reason,
            "scope": "Anomaly golden ruleset; rerun required after an external V2 adapter becomes executable.",
        },
    }
