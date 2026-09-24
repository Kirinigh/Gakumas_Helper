"""Pure projections of reader metrics; collection and transactions stay outside."""

from typing import Any
from collections.abc import Mapping, Sequence


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def duration_percentiles(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {
        name: {
            "count": len(values),
            "p50": round(percentile(values, 0.50), 6),
            "p95": round(percentile(values, 0.95), 6),
            "max": round(max(values), 6),
        }
        for name, values in sorted(samples.items())
        if values
    }


def member_metric_deltas(
    timings: Mapping[str, float],
    counts: Mapping[str, int],
    samples: Mapping[str, Sequence[float]],
    baseline: tuple[Mapping[str, float], Mapping[str, int], Mapping[str, int]],
) -> dict[str, Any]:
    timing_before, counts_before, sample_lengths = baseline
    return {
        "timing_seconds": {
            name: round(value - timing_before.get(name, 0.0), 6)
            for name, value in timings.items()
            if value - timing_before.get(name, 0.0) > 0
        },
        "counts": {
            name: value - counts_before.get(name, 0)
            for name, value in counts.items()
            if value - counts_before.get(name, 0) > 0
        },
        "duration_samples_seconds": {
            name: [round(value, 6) for value in values[sample_lengths.get(name, 0):]]
            for name, values in samples.items()
            if values[sample_lengths.get(name, 0):]
        },
    }
