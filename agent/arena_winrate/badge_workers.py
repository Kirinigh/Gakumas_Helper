"""First-run worker calibration for local arena badge processing."""

from __future__ import annotations

import os
import json
import time
import hashlib
import platform
import statistics
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BADGE_WORKER_CACHE = (
    PROJECT_ROOT / ".local" / "arena-win-rate" / "badge-worker-calibration-v3.json"
)


@dataclass(frozen=True)
class BadgeWorkerSelection:
    workers: int
    cache_hit: bool
    signature: str
    benchmarks: tuple[dict[str, float | int], ...]


class BadgeWorkerCalibrator:
    """Choose the smallest materially faster worker count and cache it."""

    def __init__(
        self,
        cache_path: str | Path = DEFAULT_BADGE_WORKER_CACHE,
        *,
        repetitions: int = 5,
        minimum_speedup: float = 0.05,
        worker_cost: float = 0.035,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.repetitions = repetitions
        self.minimum_speedup = minimum_speedup
        self.worker_cost = worker_cost

    @staticmethod
    def signature(
        *,
        frame_shape: Sequence[int],
        batch_slots: int,
        reference_version: str,
        recognition_asset_signature: str,
    ) -> str:
        try:
            import cv2

            opencv_version = cv2.__version__
        except ImportError:
            opencv_version = "missing"
        payload = {
            "cpu": platform.processor(),
            "logical_cores": os.cpu_count() or 1,
            "opencv": opencv_version,
            "frame_shape": [int(value) for value in frame_shape],
            "batch_slots": int(batch_slots),
            "reference_version": reference_version,
            "recognition_asset_signature": recognition_asset_signature,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest().upper()

    def _load(self, signature: str) -> BadgeWorkerSelection | None:
        try:
            record = json.loads(self.cache_path.read_text(encoding="utf-8-sig"))
            if record.get("schema_version") != 3 or record.get("signature") != signature:
                return None
            workers = int(record["selected_workers"])
            if workers < 1 or workers > (os.cpu_count() or 1):
                return None
            benchmarks = tuple(record.get("benchmarks", ()))
            return BadgeWorkerSelection(workers, True, signature, benchmarks)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save(
        self,
        *,
        signature: str,
        workers: int,
        benchmarks: Sequence[dict[str, float | int]],
    ) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 3,
            "signature": signature,
            "selected_workers": workers,
            "selection_policy": {
                "minimum_speedup": self.minimum_speedup,
                "worker_cost": self.worker_cost,
                "repetitions": self.repetitions,
            },
            "benchmarks": list(benchmarks),
        }
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    def select(
        self,
        *,
        signature: str,
        run_batch: Callable[[int], None],
    ) -> BadgeWorkerSelection:
        cached = self._load(signature)
        if cached is not None:
            return cached
        cores = max(1, os.cpu_count() or 1)
        workload_ceiling = min(12, cores)
        candidates = tuple(
            value for value in (1, 2, 4, 8, 12) if value <= workload_ceiling
        )
        samples: dict[int, list[float]] = {value: [] for value in candidates}
        run_batch(1)
        for _ in range(self.repetitions):
            for workers in candidates:
                started = time.perf_counter()
                run_batch(workers)
                samples[workers].append(time.perf_counter() - started)
        benchmarks = []
        for workers in candidates:
            values = sorted(samples[workers])
            p50 = statistics.median(values)
            p95 = values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))]
            latency = p50 * 0.6 + p95 * 0.4
            benchmarks.append(
                {
                    "workers": workers,
                    "p50_seconds": round(p50, 6),
                    "p95_seconds": round(p95, 6),
                    "weighted_seconds": round(
                        latency * (1.0 + self.worker_cost * (workers - 1)),
                        6,
                    ),
                }
            )
        selected = benchmarks[0]
        for candidate in benchmarks[1:]:
            speedup = 1.0 - float(candidate["p50_seconds"]) / max(
                float(selected["p50_seconds"]),
                1e-9,
            )
            if (
                speedup >= self.minimum_speedup
                and float(candidate["weighted_seconds"])
                < float(selected["weighted_seconds"])
            ):
                selected = candidate
        workers = int(selected["workers"])
        self._save(signature=signature, workers=workers, benchmarks=benchmarks)
        return BadgeWorkerSelection(workers, False, signature, tuple(benchmarks))
