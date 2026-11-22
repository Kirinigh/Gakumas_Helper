"""Run a local diagnostic episode and report nominal time-budget overruns."""
from __future__ import annotations

import json
import math
import time
import argparse
from pathlib import Path

from tools.task275.gkms_continuous_policy import execute_episode
from agent.live_decision.high_score.hif_environment import HifEnvironment


def summarize(report, *, opening, arm, environment_seed, planning_seed):
    timing = report['decision_latency_ms']
    return dict(scope='local_diagnostic_not_hosted_or_qualified_fast', opening=opening, mode=arm,
                environment_seed=environment_seed, planning_seed=planning_seed,
                complete=report['terminal'], score=report['observed_score'] if report['terminal'] else None,
                error=report['error'], decisions=report['result']['decision_count'],
                over_5s=sum(value > 5000 for value in timing),
                over_6050ms=sum(value > 6050 for value in timing),
                max_ms=round(max(timing, default=0), 2), seconds=round(report['episode_seconds'], 2))


def run_local_case(case, policy_factory, *, arm, environment_seed, planning_seed, deadline):
    """Replay a supplied rule opening and separate nominal timing from the overall deadline."""
    if not math.isfinite(deadline):
        raise ValueError('local episode requires a finite deadline')
    environment = HifEnvironment(case.kernel)
    request = environment.reset(case.state, seed=environment_seed, drink_slots=case.drink_slots)
    policy = policy_factory(case, request, planning_seed=planning_seed, deadline=deadline)
    report = execute_episode(environment, policy, request,
                             episode_id=f'{case.name}:{environment_seed}', deadline=deadline)
    return dict(summary=summarize(report, opening=case.name, arm=arm, environment_seed=environment_seed,
                                  planning_seed=planning_seed), episode=report)


def main():
    from tools.task285_cases import load_case
    from tools.task285_seed_bench import RoutePolicy, SyntheticPrior, write_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('prior', 'draw-first'), required=True,
                        help='Choose one of the two synthetic continuations')
    parser.add_argument('--environment-seed', type=int, default=280)
    parser.add_argument('--planning-seed', type=int, default=880280)
    parser.add_argument('--episode-seconds', type=float, default=450.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0 or args.output.exists():
        parser.error('a positive finite deadline and a new output path are required')

    def synthetic_policy(case, request, *, planning_seed, deadline):
        prior = SyntheticPrior()
        return RoutePolicy(prior, case.routes[0]) if args.arm == 'draw-first' else prior

    report = run_local_case(load_case('synthetic-draw'), synthetic_policy, arm=args.arm,
                            environment_seed=args.environment_seed, planning_seed=args.planning_seed,
                            deadline=time.perf_counter() + args.episode_seconds)
    write_report(args.output, report)
    print(json.dumps(report['summary']))
    return 0 if report['summary']['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
