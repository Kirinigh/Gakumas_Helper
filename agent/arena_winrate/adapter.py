"""Process-isolated adapter for the pinned gakumas-tools simulator."""

from __future__ import annotations

import re
import json
import math
import time
import subprocess
from typing import Any, Mapping, Sequence
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Callable

from .decision import StageEstimate, OpponentEstimate

PROTOCOL_VERSION = "2.0"
UPSTREAM_COMMIT = "5658ec13ec9978f1117ccc191dfb83c346d42f6f"
SIMULATION_METHOD = "independent_empirical_three_stage_match"
SCORE_AGGREGATION = "raw_sum_plus_stage_wide_first_place_20_percent"
MATCH_RULE = "best_of_three_strict_wins"
OWN_SCORE_METHOD = "independent_empirical_own_three_stage_scores"
OWN_SCORE_AGGREGATION = "raw_team_sum_without_cross_side_first_place_bonus"


class AdapterError(RuntimeError):
    """Raised for unavailable, timed-out, crashed or invalid adapters."""


@dataclass(frozen=True)
class SimulationBatch:
    request_id: str
    upstream_commit: str
    estimates: tuple[OpponentEstimate, ...]
    method: str = SIMULATION_METHOD
    score_aggregation: str = SCORE_AGGREGATION
    match_rule: str = MATCH_RULE
    stage_distributions: tuple[dict[str, object], ...] = ()
    parallelism: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OwnScoreBatch:
    request_id: str
    upstream_commit: str
    stage_distributions: tuple[dict[str, object], ...]
    parallelism: dict[str, object]
    method: str = OWN_SCORE_METHOD
    score_aggregation: str = OWN_SCORE_AGGREGATION


class SubprocessArenaAdapter:
    """Run the JavaScript simulator out of process with a bounded timeout."""

    def __init__(
        self,
        command: Sequence[str | Path],
        *,
        timeout_seconds: float,
        expected_upstream_commit: str = UPSTREAM_COMMIT,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if re.fullmatch(r"[0-9a-f]{40}", expected_upstream_commit) is None:
            raise ValueError("expected_upstream_commit must be a lowercase 40-character SHA")
        self.command = tuple(str(part) for part in command)
        self.timeout_seconds = timeout_seconds
        self.expected_upstream_commit = expected_upstream_commit
        self.cancel_check = cancel_check

    @classmethod
    def from_bundle(
        cls, bundle_dir: Path, *, timeout_seconds: float,
        cancel_check: Callable[[], None] | None = None,
    ) -> "SubprocessArenaAdapter":
        manifest_path = bundle_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AdapterError(f"engine bundle manifest is unavailable or invalid: {manifest_path}") from error
        commit = manifest.get("commit") if isinstance(manifest, Mapping) else None
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != 1
            or manifest.get("project") != "gakumas-tools"
            or not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        ):
            raise AdapterError("engine bundle manifest identity is invalid")
        return cls(
            (bundle_dir / "node.exe", bundle_dir / "runner.mjs"),
            timeout_seconds=timeout_seconds,
            expected_upstream_commit=commit,
            cancel_check=cancel_check,
        )

    def simulate(self, request: Mapping[str, Any]) -> SimulationBatch:
        return self._parse_response(
            self._invoke(request),
            request,
            expected_upstream_commit=self.expected_upstream_commit,
        )

    def simulate_own(self, request: Mapping[str, Any]) -> OwnScoreBatch:
        """Simulate only the reusable own-team raw score distributions."""

        return self._parse_own_response(
            self._invoke(request),
            request,
            expected_upstream_commit=self.expected_upstream_commit,
        )

    def _invoke(self, request: Mapping[str, Any]) -> object:
        executable = Path(self.command[0])
        if not executable.is_file():
            raise AdapterError(f"adapter executable is unavailable: {executable}")

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            runner = self._run_cancellable if self.cancel_check is not None else subprocess.run
            completed = runner(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
                check=False,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as error:
            raise AdapterError(f"adapter timed out after {self.timeout_seconds:g}s") from error
        except OSError as error:
            raise AdapterError(f"adapter could not start: {error}") from error

        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-1000:]
            raise AdapterError(f"adapter exited with code {completed.returncode}: {stderr}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise AdapterError("adapter returned invalid JSON") from error

    def _run_cancellable(self, command, *, input, timeout, check, **kwargs):
        """Stop the exact simulator process; its worker_threads exit with it."""
        del check
        assert self.cancel_check is not None
        self.cancel_check()
        kwargs.pop("capture_output")
        started = time.monotonic()
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, **kwargs,
        ) as process:
            try:
                payload = input
                while True:
                    self.cancel_check()
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        stdout, stderr = process.communicate(
                            input=payload, timeout=min(0.1, remaining),
                        )
                        self.cancel_check()
                        return subprocess.CompletedProcess(
                            command, process.returncode, stdout, stderr,
                        )
                    except subprocess.TimeoutExpired:
                        # communicate resumes its existing stdin/stdout state.
                        payload = None
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    @staticmethod
    def _parse_own_response(
        payload: object,
        request: Mapping[str, Any],
        *,
        expected_upstream_commit: str = UPSTREAM_COMMIT,
    ) -> OwnScoreBatch:
        if not isinstance(payload, Mapping):
            raise AdapterError("adapter response must be an object")
        if payload.get("schema_version") != PROTOCOL_VERSION:
            raise AdapterError("adapter protocol version mismatch")
        if payload.get("request_id") != request.get("request_id"):
            raise AdapterError("adapter request_id mismatch")
        engine = payload.get("engine")
        if not isinstance(engine, Mapping) or engine.get("commit") != expected_upstream_commit:
            raise AdapterError("adapter upstream commit mismatch")
        if payload.get("method") != OWN_SCORE_METHOD:
            raise AdapterError("adapter own-score method mismatch")
        if payload.get("score_aggregation") != OWN_SCORE_AGGREGATION:
            raise AdapterError("adapter own-score aggregation mismatch")

        parallelism = SubprocessArenaAdapter._parse_parallelism(payload)
        simulations = request.get("simulations")
        stage_ids = request.get("stageIds")
        own_request = request.get("own_team")
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or not isinstance(stage_ids, list)
            or len(stage_ids) != 3
            or not isinstance(own_request, Mapping)
        ):
            raise AdapterError("own-score request shape is invalid")
        own_stages = own_request.get("stages")
        stage_payloads = payload.get("stages")
        if (
            not isinstance(own_stages, list)
            or len(own_stages) != 3
            or not isinstance(stage_payloads, list)
            or len(stage_payloads) != 3
        ):
            raise AdapterError("own-score stages must contain exactly three entries")

        parsed: list[dict[str, object]] = []
        for index, candidate in enumerate(stage_payloads):
            if not isinstance(candidate, Mapping):
                raise AdapterError("own-score stage result is invalid")
            stage_number = SubprocessArenaAdapter._strict_int(candidate.get("stage_number"), "stage number")
            stage_id = SubprocessArenaAdapter._strict_int(candidate.get("stage_id"), "stage ID")
            own_stage = own_stages[index]
            if (
                stage_number != index + 1
                or stage_id != stage_ids[index]
                or not isinstance(own_stage, Mapping)
                or not isinstance(own_stage.get("members"), list)
            ):
                raise AdapterError("own-score stage order or identity mismatch")
            parsed.append(
                {
                    "stage_number": stage_number,
                    "stage_id": stage_id,
                    "own_member_distributions": SubprocessArenaAdapter._parse_distribution_list(
                        candidate.get("own_member_distributions"),
                        simulations,
                        len(own_stage["members"]),
                    ),
                    "own_member_scores": SubprocessArenaAdapter._parse_score_matrix(
                        candidate.get("own_member_scores"),
                        simulations,
                        len(own_stage["members"]),
                    ),
                    "own_raw_team_distribution": SubprocessArenaAdapter._parse_distribution(
                        candidate.get("own_raw_team_distribution"), simulations
                    ),
                }
            )
        return OwnScoreBatch(
            request_id=str(payload["request_id"]),
            upstream_commit=str(engine["commit"]),
            stage_distributions=tuple(parsed),
            parallelism=parallelism,
        )

    @staticmethod
    def _parse_response(
        payload: object,
        request: Mapping[str, Any],
        *,
        expected_upstream_commit: str = UPSTREAM_COMMIT,
    ) -> SimulationBatch:
        if not isinstance(payload, Mapping):
            raise AdapterError("adapter response must be an object")
        if payload.get("schema_version") != PROTOCOL_VERSION:
            raise AdapterError("adapter protocol version mismatch")
        if payload.get("request_id") != request.get("request_id"):
            raise AdapterError("adapter request_id mismatch")
        engine = payload.get("engine")
        if not isinstance(engine, Mapping) or engine.get("commit") != expected_upstream_commit:
            raise AdapterError("adapter upstream commit mismatch")
        if payload.get("method") != SIMULATION_METHOD:
            raise AdapterError("adapter simulation method mismatch")
        if payload.get("score_aggregation") != SCORE_AGGREGATION:
            raise AdapterError("adapter score aggregation mismatch")
        if payload.get("match_rule") != MATCH_RULE:
            raise AdapterError("adapter match rule mismatch")

        parallelism_payload = SubprocessArenaAdapter._parse_parallelism(payload)

        simulations = request.get("simulations")
        stage_ids = request.get("stageIds")
        own_request = request.get("own_team")
        opponent_requests = request.get("opponents")
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or not isinstance(stage_ids, list)
            or len(stage_ids) != 3
            or not isinstance(own_request, Mapping)
            or not isinstance(opponent_requests, list)
        ):
            raise AdapterError("simulation request shape is invalid")

        own_stage_requests = own_request.get("stages")
        if not isinstance(own_stage_requests, list) or len(own_stage_requests) != 3:
            raise AdapterError("simulation own stage request is invalid")
        stage_payloads = payload.get("stages")
        if not isinstance(stage_payloads, list) or len(stage_payloads) != 3:
            raise AdapterError("adapter stages must contain exactly three entries")

        stage_distributions: list[dict[str, object]] = []
        for index, candidate_stage in enumerate(stage_payloads):
            if not isinstance(candidate_stage, Mapping):
                raise AdapterError("adapter stage result is invalid")
            stage_number = SubprocessArenaAdapter._strict_int(candidate_stage.get("stage_number"), "stage number")
            stage_id = SubprocessArenaAdapter._strict_int(candidate_stage.get("stage_id"), "stage ID")
            if stage_number != index + 1 or stage_id != stage_ids[index]:
                raise AdapterError("adapter stage order or identity mismatch")
            own_stage_request = own_stage_requests[index]
            if not isinstance(own_stage_request, Mapping) or not isinstance(own_stage_request.get("members"), list):
                raise AdapterError("simulation own stage members are invalid")
            own_member_distributions = SubprocessArenaAdapter._parse_distribution_list(
                candidate_stage.get("own_member_distributions"),
                simulations,
                len(own_stage_request["members"]),
            )
            stage_distributions.append(
                {
                    "stage_number": stage_number,
                    "stage_id": stage_id,
                    "own_member_distributions": own_member_distributions,
                    "own_raw_team_distribution": SubprocessArenaAdapter._parse_distribution(
                        candidate_stage.get("own_raw_team_distribution"), simulations
                    ),
                    "opponents": [],
                }
            )

        candidate_payloads = payload.get("candidates")
        if not isinstance(candidate_payloads, list):
            raise AdapterError("adapter candidates must be an array")

        estimates: list[OpponentEstimate] = []
        try:
            for position, candidate in enumerate(candidate_payloads):
                if not isinstance(candidate, Mapping):
                    raise TypeError
                opponent_request = opponent_requests[position]
                if not isinstance(opponent_request, Mapping):
                    raise TypeError
                opponent_stage_requests = opponent_request.get("stages")
                if not isinstance(opponent_stage_requests, list) or len(opponent_stage_requests) != 3:
                    raise TypeError

                trials = SubprocessArenaAdapter._strict_int(candidate.get("trials"), "candidate trials")
                if trials != simulations:
                    raise ValueError
                candidate_stages = candidate.get("stages")
                if not isinstance(candidate_stages, list) or len(candidate_stages) != 3:
                    raise TypeError
                stage_estimates: list[StageEstimate] = []
                for stage_index, stage_result in enumerate(candidate_stages):
                    if not isinstance(stage_result, Mapping):
                        raise TypeError
                    stage_number = SubprocessArenaAdapter._strict_int(
                        stage_result.get("stage_number"), "candidate stage number"
                    )
                    stage_id = SubprocessArenaAdapter._strict_int(stage_result.get("stage_id"), "candidate stage ID")
                    stage_trials = SubprocessArenaAdapter._strict_int(stage_result.get("trials"), "candidate stage trials")
                    if stage_number != stage_index + 1 or stage_id != stage_ids[stage_index] or stage_trials != simulations:
                        raise ValueError
                    stage_estimates.append(
                        StageEstimate(
                            stage_number=stage_number,
                            stage_id=stage_id,
                            wins=SubprocessArenaAdapter._strict_int(stage_result.get("wins"), "stage wins"),
                            losses=SubprocessArenaAdapter._strict_int(stage_result.get("losses"), "stage losses"),
                            ties=SubprocessArenaAdapter._strict_int(stage_result.get("ties"), "stage ties"),
                            trials=stage_trials,
                            first_place_ties=SubprocessArenaAdapter._strict_int(
                                stage_result.get("first_place_ties"), "first-place ties"
                            ),
                        )
                    )
                    opponent_stage_request = opponent_stage_requests[stage_index]
                    if not isinstance(opponent_stage_request, Mapping) or not isinstance(
                        opponent_stage_request.get("members"), list
                    ):
                        raise TypeError
                    stage_distributions[stage_index]["opponents"].append(
                        {
                            "opponent_id": str(candidate["opponent_id"]),
                            "position": position,
                            "member_distributions": SubprocessArenaAdapter._parse_distribution_list(
                                stage_result.get("opponent_member_distributions"),
                                simulations,
                                len(opponent_stage_request["members"]),
                            ),
                            "own_effective_team_distribution": SubprocessArenaAdapter._parse_distribution(
                                stage_result.get("own_effective_team_distribution"), simulations
                            ),
                            "opponent_raw_team_distribution": SubprocessArenaAdapter._parse_distribution(
                                stage_result.get("opponent_raw_team_distribution"), simulations
                            ),
                            "opponent_effective_team_distribution": SubprocessArenaAdapter._parse_distribution(
                                stage_result.get("opponent_effective_team_distribution"), simulations
                            ),
                        }
                    )

                estimates.append(
                    OpponentEstimate(
                        opponent_id=str(candidate["opponent_id"]),
                        position=SubprocessArenaAdapter._strict_int(candidate.get("position"), "candidate position"),
                        wins=SubprocessArenaAdapter._strict_int(candidate.get("wins"), "candidate wins"),
                        losses=SubprocessArenaAdapter._strict_int(candidate.get("losses"), "candidate losses"),
                        ties=SubprocessArenaAdapter._strict_int(candidate.get("ties"), "candidate ties"),
                        trials=trials,
                        stages=tuple(stage_estimates),
                    )
                )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise AdapterError("adapter candidate result is invalid") from error

        expected = [(str(item["team_id"]), index) for index, item in enumerate(opponent_requests)]
        actual = [(estimate.opponent_id, estimate.position) for estimate in estimates]
        if actual != expected:
            raise AdapterError("adapter candidate order or identity mismatch")
        for stage in stage_distributions:
            stage["opponents"] = tuple(stage["opponents"])
        return SimulationBatch(
            request_id=str(payload["request_id"]),
            upstream_commit=str(engine["commit"]),
            estimates=tuple(estimates),
            stage_distributions=tuple(stage_distributions),
            parallelism=dict(parallelism_payload),
        )

    @staticmethod
    def _parse_parallelism(payload: Mapping[str, Any]) -> dict[str, object]:
        parallelism_payload = payload.get("parallelism")
        workers = payload.get("workers")
        if not isinstance(parallelism_payload, Mapping):
            raise AdapterError("adapter parallelism metadata is missing")
        available = parallelism_payload.get("available_parallelism")
        selected = parallelism_payload.get("selected_workers")
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or available < 1
            or isinstance(selected, bool)
            or not isinstance(selected, int)
            or not 1 <= selected <= available
            or workers != selected
        ):
            raise AdapterError("adapter parallelism metadata is invalid")
        return dict(parallelism_payload)

    @staticmethod
    def _strict_int(value: object, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"adapter {field} is invalid")
        return value

    @staticmethod
    def _parse_distribution_list(
        value: object,
        expected_count: int,
        expected_length: int,
    ) -> tuple[dict[str, float | int], ...]:
        if not isinstance(value, list) or len(value) != expected_length:
            raise AdapterError("adapter member distributions are invalid")
        return tuple(SubprocessArenaAdapter._parse_distribution(item, expected_count) for item in value)

    @staticmethod
    def _parse_score_matrix(
        value: object,
        expected_trials: int,
        expected_members: int,
    ) -> tuple[tuple[int, ...], ...]:
        if not isinstance(value, list) or len(value) != expected_members:
            raise AdapterError("raw member score matrix has an invalid member count")
        parsed: list[tuple[int, ...]] = []
        for scores in value:
            if (
                not isinstance(scores, list)
                or len(scores) != expected_trials
                or any(isinstance(score, bool) or not isinstance(score, int) or score < 0 for score in scores)
            ):
                raise AdapterError("raw member score matrix contains invalid scores")
            parsed.append(tuple(scores))
        return tuple(parsed)

    @staticmethod
    def _parse_distribution(value: object, expected_count: int) -> dict[str, float | int]:
        if not isinstance(value, Mapping):
            raise AdapterError("adapter distribution summary must be an object")
        fields = ("min", "q1", "median", "mean", "q3", "max")
        if value.get("count") != expected_count:
            raise AdapterError("adapter distribution sample count mismatch")
        numbers: list[float] = []
        for field_name in fields:
            field_value = value.get(field_name)
            if isinstance(field_value, bool) or not isinstance(field_value, (int, float)) or not math.isfinite(field_value):
                raise AdapterError(f"adapter distribution {field_name} is invalid")
            numbers.append(float(field_value))
        if numbers[0] > numbers[1] or numbers[1] > numbers[2] or numbers[2] > numbers[4] or numbers[4] > numbers[5]:
            raise AdapterError("adapter distribution quantiles are inconsistent")
        return {"count": expected_count, **{field_name: value[field_name] for field_name in fields}}
