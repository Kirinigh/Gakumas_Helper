"""Summarize a caller-supplied paired episode with the nominal 5s budget."""
from __future__ import annotations


def summarize(report, *, opening, arm, environment_seed, planning_seed):
    timing = report['decision_latency_ms']
    return dict(scope='local_diagnostic_not_hosted_or_qualified_fast', opening=opening, mode=arm,
                environment_seed=environment_seed, planning_seed=planning_seed,
                complete=report['terminal'], score=report['observed_score'] if report['terminal'] else None,
                error=report['error'], decisions=report['result']['decision_count'],
                over_5s=sum(value > 5000 for value in timing),
                over_6050ms=sum(value > 6050 for value in timing),
                max_ms=round(max(timing, default=0), 2), seconds=round(report['episode_seconds'], 2))
