"""Sequential opponent decisions with one simulator and one foreground reader."""

from __future__ import annotations

import time
from copy import copy
from uuid import uuid4
from threading import Event
from dataclasses import replace
from concurrent.futures import Future, TimeoutError, ThreadPoolExecutor

from .config import DEFAULT_LOWER_THRESHOLD_PERCENT
from .schema import validate_own_snapshot, validate_opponent_snapshot
from .adapter import MATCH_RULE, PROTOCOL_VERSION, SCORE_AGGREGATION
from .service import DEFAULT_CALIBRATION_CACHE, ArenaEvaluation, ArenaProviderAttemptFailure
from .decision import QUICK_THIRD_OPPONENT_RULE, select_first_qualified
from .cancellation import ArenaTaskCancelled, ArenaReadSuperseded


class _SimulationFailure(ArenaReadSuperseded):
    def __init__(self, error):
        self.error = error


class ProgressiveArenaService:
    """UI stays on the caller thread. Only immutable simulation input is shared."""

    def __init__(self, adapter, *, threshold, lower_threshold=DEFAULT_LOWER_THRESHOLD_PERCENT / 100, simulations=2000, seed=400,
                 confidence=0.95, cancel_check=lambda: None):
        if not 0 <= lower_threshold <= threshold <= 1:
            raise ValueError("lower threshold must be between zero and the upper threshold")
        if type(simulations) is not int or simulations < 1000:
            raise ValueError("simulations must be an integer of at least 1000")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.adapter = adapter
        self.threshold = threshold
        self.lower_threshold = lower_threshold
        self.simulations = simulations
        self.seed = seed
        self.confidence = confidence
        self.cancel_check = cancel_check
        self.snapshot = None
        self.read_attempts = []

    def evaluate(self, reader, own_snapshot, *, own_score_cache=None, allow_click=False):
        own = validate_own_snapshot(own_snapshot)
        backend = reader.backend
        previous_hook = getattr(backend, "_progressive_read_check", None)
        stopped = Event()
        adapter = copy(self.adapter)

        def worker_cancel():
            if stopped.is_set():
                raise ArenaTaskCancelled("progressive simulator stopped")

        # Never call a Maa proxy from the simulation thread.
        if hasattr(adapter, "cancel_check"):
            adapter.cancel_check = worker_cancel
        completed = []
        estimates = []
        batches = []
        failures = []
        future: Future | None = None
        decision = None
        attempts = 0
        retry_remaining = 1  # Shared by this whole selection, not renewed per opponent.
        simulation_seconds = 0.0
        started = time.perf_counter()
        timeline = {
            "clock": "perf_counter_offsets_from_evaluation_start",
            "simulations": [], "decision_ready": None, "read_superseded": None,
            "recovery_started": None, "recovery_finished": None,
        }
        self.snapshot = None
        self.read_attempts = []

        def collect(*, wait=False):
            nonlocal future, decision, simulation_seconds
            self.cancel_check()
            if future is None:
                return
            while wait and not future.done():
                self.cancel_check()
                try:
                    future.result(timeout=0.1)
                except TimeoutError:
                    continue
                except Exception as error:
                    raise _SimulationFailure(error) from error
            if not future.done():
                return
            try:
                batch, elapsed, error, run_timing = future.result()
            except Exception as error:
                raise _SimulationFailure(error) from error
            future = None
            simulation_seconds += elapsed
            timeline["simulations"].append({
                **run_timing, "collected": time.perf_counter() - started,
            })
            if error is not None:
                raise _SimulationFailure(error)
            expected_position = len(estimates)
            if (len(batch.estimates) != 1 or batch.estimates[0].position != expected_position
                    or batch.estimates[0].opponent_id != f"opponent-{expected_position}"):
                raise _SimulationFailure(ValueError("single-opponent simulation identity mismatch"))
            batches.append(batch)
            estimates.extend(batch.estimates)
            candidate = select_first_qualified(
                estimates, threshold=self.threshold, confidence=self.confidence,
                family_size=3, allow_fallback=len(estimates) == 3,
            )
            if candidate.selected_position is not None:
                decision = candidate
            elif len(estimates) == 2 and all(row["qualification_lower"] <= self.lower_threshold for row in candidate.estimates):
                decision = replace(candidate, selected_position=2, selected_opponent_id="opponent-2",
                                   decision_rule=QUICK_THIRD_OPPONENT_RULE)
            if decision is not None:
                timeline["decision_ready"] = time.perf_counter() - started

        def poll():
            collect()
            if decision is not None:
                timeline["read_superseded"] = time.perf_counter() - started
                raise ArenaReadSuperseded("an earlier opponent result selected the challenge")

        def request_for(snapshot):
            position = snapshot["opponent_position"]
            offset = sum(len(stage["members"]) for stage in own["own_team"]["stages"])
            offset += sum(len(stage["members"]) for item in completed[:position] for stage in item["opponent"]["stages"])
            request = {
                "schema_version": PROTOCOL_VERSION, "request_id": snapshot["capture_id"],
                "operation": "arena_match_win_rate", "simulations": self.simulations, "seed": self.seed,
                "season": own["season"], "stageIds": own["stageIds"], "score_aggregation": SCORE_AGGREGATION,
                "match_rule": MATCH_RULE, "own_team": own["own_team"], "opponents": [snapshot["opponent"]],
                "opponent_positions": [position], "opponent_seed_offset": offset,
                "parallelism": {"mode": "auto", "cache_path": str(DEFAULT_CALIBRATION_CACHE), "reader_concurrent": True},
            }
            if own_score_cache is not None:
                request["own_score_cache"] = own_score_cache
            return request

        def evaluation(status, error=None):
            return ArenaEvaluation(
                status=status, attempts=attempts, safe_to_click=status == "selected" and allow_click, decision=decision,
                error=None if error is None else str(error), provider_attempt_failures=tuple(failures),
                distribution_summary={"mode": "progressive", "lower_threshold": self.lower_threshold,
                                      "simulated_positions": [item.position for item in estimates],
                                      "simulation_seconds": simulation_seconds,
                                      "batches": [{"stages": batch.stage_distributions, "parallelism": batch.parallelism}
                                                  for batch in batches],
                                      "timeline_seconds": timeline,
                                      "wall_seconds": time.perf_counter() - started},
            )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arena-simulator")

        def simulate(request):
            run_started = time.perf_counter()
            batch = None
            failure = None
            try:
                if hasattr(adapter, "timeout_seconds"):
                    remaining = self.adapter.timeout_seconds - simulation_seconds
                    if remaining <= 0:
                        raise TimeoutError("本轮模拟累计超时")
                    adapter.timeout_seconds = remaining
                batch = adapter.simulate(request)
            except Exception as error:
                failure = error
            finished = time.perf_counter()
            # The worker returns immutable timing values with its result; only
            # the foreground collector updates the summary and touches Maa.
            return batch, finished - run_started, failure, {
                "position": request["opponent_positions"][0],
                "started": run_started - started, "finished": finished - started,
            }

        def return_to_arena():
            backend._progressive_read_check = previous_hook
            self.cancel_check()
            timeline["recovery_started"] = time.perf_counter() - started
            restore = getattr(backend, "finish_progressive_read", None)
            if callable(restore):
                restore()
            else:
                backend.recover_to_arena_main(require_opponents=True)
            timeline["recovery_finished"] = time.perf_counter() - started

        def record_return_failure(error):
            failures.append(ArenaProviderAttemptFailure(
                attempts, type(error).__name__, "recovery_failed",
                f"return after stopping speculative read failed: {error}", None, False,
            ))

        try:
            backend._progressive_read_check = poll
            try:
                for position in range(3):
                    while True:
                        self.cancel_check()
                        attempt_started = time.perf_counter()
                        attempts += 1
                        outcome = "failed"
                        try:
                            snapshot = validate_opponent_snapshot(reader.read_opponent(own, position))
                            if snapshot["opponent_position"] != position:
                                raise ValueError("reader returned a different opponent")
                            completed.append(snapshot)
                            outcome = "complete"
                            break
                        except ArenaReadSuperseded:
                            outcome = "superseded"
                            if timeline["read_superseded"] is None:
                                timeline["read_superseded"] = time.perf_counter() - started
                            raise
                        except Exception as error:
                            failures.append(ArenaProviderAttemptFailure(
                                attempts, type(error).__name__, getattr(error, "code", None),
                                str(getattr(error, "detail", error)), time.perf_counter() - attempt_started,
                                getattr(error, "retry_whole_read", None),
                            ))
                            # An already-running result can make this speculative
                            # read unnecessary; never turn its failure into a zero rate.
                            collect(wait=True)
                            if decision is not None:
                                raise ArenaReadSuperseded("selected despite speculative read failure") from error
                            if retry_remaining == 0 or getattr(error, "retry_whole_read", True) is False:
                                return evaluation("observation_failure", f"snapshot provider failed: {error}")
                            retry_remaining -= 1
                        finally:
                            self.read_attempts.append({"position": position, "attempt": attempts, "outcome": outcome,
                                                       "read_wall_seconds": time.perf_counter() - attempt_started})
                    # If reading finished first, wait for its predecessor before
                    # submitting this team. No second simulator competes for CPU.
                    collect(wait=True)
                    if decision is not None:
                        break
                    future = executor.submit(simulate, request_for(snapshot))
                collect(wait=True)
            except _SimulationFailure as error:
                try:
                    return_to_arena()
                except Exception as recovery_error:
                    record_return_failure(recovery_error)
                    return evaluation("adapter_failure", f"{error.error}; recovery also failed: {recovery_error}")
                return evaluation("adapter_failure", error.error)
            except ArenaReadSuperseded:
                try:
                    return_to_arena()
                except Exception as error:
                    record_return_failure(error)
                    return evaluation("observation_failure", f"recovery_failed: {error}")
            if decision is None:
                return evaluation("adapter_failure", "no completed decision")
            self.cancel_check()
            self.snapshot = {"capture_id": uuid4().hex, "source": "progressive_live_screen",
                             "opponents": [item["opponent"] for item in completed],
                             "read_positions": [item["opponent_position"] for item in completed],
                             "opponents_complete": len(completed) == 3,
                             "read_wall_seconds": sum(item["read_wall_seconds"] for item in self.read_attempts)}
            return evaluation("selected")
        finally:
            backend._progressive_read_check = previous_hook
            stopped.set()
            if future is not None:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
