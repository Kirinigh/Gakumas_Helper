"""Run a local HIF paired benchmark with separate policy and referee seeds."""

import gzip
import json
import math
import time
import random
import argparse
from types import SimpleNamespace
from pathlib import Path
from statistics import fmean
from dataclasses import asdict, replace

from tools.task285_cases import Route, load_case
from tools.task275.gkms_continuous_policy import execute_episode
from agent.live_decision.high_score.evaluation import EpisodeResult, compare_paired_results, summarize_policy_results
from agent.live_decision.high_score.hif_environment import HifEnvironment

COHORTS = {'synthetic': (('synthetic-draw',), 285000000, 285100000)}
INCUMBENT = Route('incumbent')
CASE_SEED_STRIDE = 10_000


class RoutePolicy:
    """Apply a declared prefix using the referee's current legal instance bindings."""

    def __init__(self, chooser, route):
        self.chooser = chooser
        self.route = route
        self.index = 0
        self.continuing = False

    def _state(self, state):
        if self.route.retain is None:
            return state
        return replace(state, drinks=tuple(drink for drink in state.drinks if drink != self.route.retain))

    def choose(self, request, legal):
        if self.index < len(self.route.prefix):
            wanted = self.route.prefix[self.index]
            canonical = replace(wanted, source_slot=None)
            matches = [action for action in legal if replace(action, source_slot=None) == canonical
                       and (wanted.source_slot is None or action.source_slot == wanted.source_slot)]
            if len(matches) != 1:
                raise ValueError(f'route_prefix_unavailable:{self.route.name}:{self.index}')
            self.index += 1
            return matches[0]
        if self.route.retain is not None:
            request = replace(request, state=self._state(request.state),
                drink_slots=tuple(pair for pair in request.drink_slots if pair[1] != self.route.retain))
            legal = tuple(action for action in legal if action.drink_id != self.route.retain)
        self.continuing = True
        return self.chooser.choose(request, legal)

    def committed(self, before, action, after):
        if self.continuing:
            self.chooser.committed(self._state(before), action, self._state(after))


class SyntheticPrior:
    """Prefer the largest immediate score in the synthetic public hand."""

    def choose(self, request, legal):
        priorities = {'big': 3, 'steady': 2, 'small': 1, 'draw': 0}
        return max(legal, key=lambda action: priorities.get(action.card_instance_id, -1))

    def committed(self, before, action, after):
        pass


def write_report(path, report):
    opener = gzip.open if path.suffix == '.gz' else open
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, 'xt', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, separators=(',', ':'))


def _episode(case, route, environment_seed, chooser, deadline, episode_id):
    env = HifEnvironment(case.kernel)
    request = env.reset(case.state, seed=environment_seed, drink_slots=case.drink_slots)
    policy = RoutePolicy(chooser, route)
    report = execute_episode(env, policy, request, episode_id=episode_id, deadline=deadline)
    if report['terminal'] and policy.index != len(route.prefix):
        report['terminal'] = False
        report['error'] = dict(type='ValueError', message='route_prefix_unfinished')
        report['result'].update(stopped=True, final_score=0., reason='route_prefix_unfinished')
    report['route'] = route.name
    report['ending_resources'] = dict(stamina=env.state.stamina, max_stamina=env.state.max_stamina,
                                      drink_slots=env.request().drink_slots)
    return report


def choose_route(case, worlds, chooser_for, deadline):
    estimates = []
    routes = (INCUMBENT, *case.routes)
    for route in routes:
        scores, errors = [], []
        for seed in worlds:
            if time.perf_counter() >= deadline:
                break
            trial = _episode(case, route, seed, chooser_for(case.kernel), deadline, f'{case.name}:planning')
            scores.append(trial['result']['final_score'] if trial['terminal'] else None)
            errors.append(trial['error'])
        estimates.append(dict(route=route.name, scores=scores, errors=errors,
                              not_run=len(worlds) - len(scores),
                              mean=fmean(scores) if len(scores) == len(worlds)
                              and all(score is not None for score in scores) else None))
    if any(row['mean'] is None for row in estimates):
        return None, estimates
    best = max(range(len(routes)), key=lambda index: estimates[index]['mean'])
    return routes[best], estimates


def _first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left['actions'], right['actions'])):
        if a['action'] != b['action']:
            return dict(index=index, turn=a['before']['turn'], left=a['action'], right=b['action'])
    return None


def evaluate_world(case, environment_seed, planning_seed, *, planning_count, seconds, deadline,
                   prior_factory=None):
    episode_id = f'{case.name}:{environment_seed}:{planning_seed}'

    def chooser_for(kernel):
        return SyntheticPrior() if prior_factory is None else prior_factory(kernel)

    planning_rng = random.Random(planning_seed)
    planning_seeds = tuple(planning_rng.sample(range(2**31, 3 * 2**30), planning_count))
    if environment_seed in planning_seeds:
        raise ValueError('evaluation seed overlaps the public planning worlds')
    configurations = (('A', ()), ('B', planning_seeds), ('C', (environment_seed,)))
    results = {}
    for label, worlds in configurations:
        cutoff = min(deadline, time.perf_counter() + seconds)
        started = time.perf_counter()
        route, estimates = (choose_route(case, worlds, chooser_for, cutoff)
                            if worlds else (INCUMBENT, []))
        planning_ms = (time.perf_counter() - started) * 1000 if worlds else 0.
        if route is None:
            results[label] = dict(terminal=False, episode=None, estimates=estimates,
                                  planning_ms=planning_ms, error='incomplete_candidate_evaluation')
            continue
        trial = _episode(case, route, environment_seed, chooser_for(case.kernel), cutoff, episode_id)
        results[label] = dict(terminal=trial['terminal'], episode=trial, estimates=estimates,
                              planning_ms=planning_ms, error=trial['error'])
    a, b, c = (results[label]['episode'] for label in ('A', 'B', 'C'))
    return dict(case=case.name, environment_seed=environment_seed, planning_seed=planning_seed,
                public_rollout_seeds=planning_seeds, arms=results,
                first_divergence_B_minus_A=_first_divergence(a, b) if a and b else None,
                first_divergence_C_minus_B=_first_divergence(b, c) if b and c else None)


def summarize(rows, requested):
    keys = {(row['case'], row['environment_seed'], row['planning_seed']) for row in rows}
    if len(keys) != len(rows) or len({row['case'] for row in rows}) > 1 or len(rows) > requested:
        raise ValueError('summary requires unique worlds for one case')
    per_arm = {}
    for label in ('A', 'B', 'C'):
        runs = [row['arms'][label] for row in rows]
        finished = tuple(EpisodeResult(**run['episode']['result']) for run in runs if run['terminal'])
        policy = SimpleNamespace(policy_name=label, policy_version='task285-v1')
        latencies = sorted(ms for run in runs if (episode := run['episode'])
                           for phase in (episode, episode.get('r2_proxy', {}))
                           for ms in phase.get('decision_latency_ms', ()))
        per_arm[label] = dict(complete=len(finished), incomplete=len(runs) - len(finished),
            not_run=requested - len(runs), planning_ms=sum(run['planning_ms'] for run in runs),
            decision_p95_ms=latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
            illegal=sum(run['episode']['result']['illegal_action'] for run in runs if run['episode']),
            scores=asdict(summarize_policy_results(policy, finished)) if finished else None,
            _finished=finished)
    paired = {}
    for left, right in (('B', 'A'), ('C', 'B')):
        a, b = per_arm[left], per_arm[right]
        if min(a['complete'], b['complete']) == requested and requested > 1:
            comparison = asdict(compare_paired_results(SimpleNamespace(**a['scores']), a['_finished'],
                                                        SimpleNamespace(**b['scores']), b['_finished']))
            paired[f'{left}-{right}'] = {key: value for key, value in comparison.items()
                                       if key != 'promotion_passed'}
        else:
            paired[f'{left}-{right}'] = None
    for arm in per_arm.values():
        del arm['_finished']
    return dict(arms=per_arm, paired=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=tuple(COHORTS), default='synthetic')
    parser.add_argument('--worlds', type=int, default=2, help='Evaluation worlds per case')
    parser.add_argument('--planning-worlds', type=int, default=4, help='Public rollouts per route')
    parser.add_argument('--episode-seconds', type=float, default=120.)
    parser.add_argument('--batch-seconds', type=float, default=900.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.worlds <= CASE_SEED_STRIDE or not 0 < args.planning_worlds <= CASE_SEED_STRIDE or any(
            not math.isfinite(value) or value <= 0 for value in (args.episode_seconds, args.batch_seconds)):
        parser.error(f'counts must be in [1, {CASE_SEED_STRIDE}]; time budgets must be positive and finite')
    if args.output.exists():
        parser.error('output exists; choose a new report path')
    names, environment_base, planning_base = COHORTS[args.cohort]
    deadline = time.perf_counter() + args.batch_seconds
    cases = {}
    for offset, name in enumerate(names):
        case = load_case(name)
        rows = []
        for index in range(args.worlds):
            if time.perf_counter() >= deadline:
                break
            rows.append(evaluate_world(case, environment_base + offset * CASE_SEED_STRIDE + index,
                planning_base + offset * CASE_SEED_STRIDE + index, planning_count=args.planning_worlds,
                seconds=args.episode_seconds, deadline=deadline))
        cases[name] = dict(family=case.family, sources=case.sources, ruleset=case.kernel.ruleset_id,
            objective='synthetic_terminal_score',
            routes=[asdict(route) for route in (INCUMBENT, *case.routes)], requested=args.worlds,
            not_run=args.worlds - len(rows), worlds=rows, summary=summarize(rows, args.worlds))
    complete = all(case['not_run'] == 0 and all(arm['terminal'] for row in case['worlds']
                   for arm in row['arms'].values()) for case in cases.values())
    report = dict(version='task285-v1', scope='synthetic_offline_conditional_diagnostic', cohort=args.cohort,
        rollout_policy='synthetic_public_score_prior', oracle_arm='C',
        referee_sampler='HifEnvironment/SampledHifKernel',
        episode_seconds=args.episode_seconds, batch_seconds=args.batch_seconds,
        planning_worlds=args.planning_worlds, complete=complete, cases=cases)
    write_report(args.output, report)
    print(json.dumps(dict(complete=complete, output=str(args.output),
                          summaries={name: record['summary'] for name, record in cases.items()})))
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
