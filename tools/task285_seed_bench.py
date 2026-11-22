"""Run a local HIF paired benchmark with separate policy and referee seeds."""

import gzip
import json
import math
import time
import random
import argparse
from types import SimpleNamespace
from pathlib import Path
from functools import partial
from statistics import fmean
from dataclasses import asdict, replace

from tools.task285_cases import Route, load_case, load_two_round_cases
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
    """Use a fixed visible action order for the synthetic cases."""

    def choose(self, request, legal):
        priorities = {'big': 4, 'steady': 3, 'carry': 2, 'small': 1, 'draw': 0}
        return max(legal, key=lambda action: priorities.get(action.card_instance_id or action.drink_id, -1))

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


def _finish_route(case, route, seed, chooser_for, deadline, episode_id, *, r2_case, r2_seed):
    first = _episode(case, route, seed, chooser_for(case.kernel), deadline, episode_id)
    if r2_case is None or not first['terminal']:
        return first
    resources = first['ending_resources']
    slots = resources['drink_slots']
    drinks = tuple(drink for _, drink in slots)
    if any(r2_case.kernel.drinks.get(drink) != case.kernel.drinks[drink] for drink in drinks):
        raise ValueError('round-two proxy lacks the same carried drink definition')
    deficit = resources['max_stamina'] - resources['stamina']
    state = replace(r2_case.state, stamina=max(0, r2_case.state.stamina - deficit), drinks=drinks)
    carried = replace(r2_case, state=state, drink_slots=slots)
    second = _episode(carried, INCUMBENT, r2_seed, chooser_for(carried.kernel), deadline, episode_id + ':r2')
    # The original proxy objective registers R1 at 120% rounded down, then adds R2 raw score.
    first['registered_r1_score'] = first['observed_score'] * 6 // 5
    first['r2_proxy'] = second
    first['r2_proxy_conditions'] = dict(source=r2_case.name, r1_stamina_deficit=deficit,
                                       r2_opening_stamina=state.stamina, r2_drink_slots=slots)
    first['terminal'] = second['terminal']
    first['error'] = second['error']
    aggregate, r2_result = first['result'], second['result']
    aggregate.update(stopped=not second['terminal'], reason=r2_result['reason'],
                     final_score=first['registered_r1_score'] + r2_result['final_score'] if second['terminal'] else 0.)
    aggregate['illegal_action'] |= r2_result['illegal_action']
    for key in ('decision_count', 'latency_ms'):
        aggregate[key] += r2_result[key]
    return first


def choose_route(case, worlds, chooser_for, deadline, *, r2_case=None, r2_worlds=None):
    if (r2_case is None) != (r2_worlds is None):
        raise ValueError('round-two proxy and its planning seeds must be supplied together')
    estimates = []
    routes = (INCUMBENT, *case.routes)
    worlds_with_r2 = tuple(zip(worlds, (None,) * len(worlds) if r2_worlds is None else r2_worlds, strict=True))
    for route in routes:
        scores, errors = [], []
        for seed, r2_seed in worlds_with_r2:
            if time.perf_counter() >= deadline:
                break
            trial = _finish_route(case, route, seed, chooser_for, deadline, f'{case.name}:planning',
                                  r2_case=r2_case, r2_seed=r2_seed)
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
                   prior_factory=None, continuation_factory=None, r2_case=None, r2_environment_seed=None):
    """Rank with the public prior, then execute under the same continuation across A/B/C.

    Both factories return fresh choosers implementing choose() and committed().
    The optional continuation also receives the planning seed and arm deadline.
    """
    episode_id = f'{case.name}:{environment_seed}:{planning_seed}'

    def chooser_for(kernel):
        return SyntheticPrior() if prior_factory is None else prior_factory(kernel)

    planning_rng = random.Random(planning_seed)
    planning_seeds = tuple(planning_rng.sample(range(2**31, 3 * 2**30), planning_count))
    r2_planning_seeds = None
    if r2_case is not None:
        if r2_case.drink_slots or r2_case.state.drinks:
            raise ValueError('round-two proxy must begin without its own drinks')
        if r2_environment_seed is None:
            r2_environment_seed = environment_seed + 1_000_000
        if type(r2_environment_seed) is not int or r2_environment_seed == environment_seed:
            raise ValueError('round two requires a distinct integer referee seed')
        r2_planning_seeds = tuple(planning_rng.sample(range(3 * 2**30, 2**32), planning_count))
    elif r2_environment_seed is not None:
        raise ValueError('round-two referee seed requires a round-two proxy')
    if {environment_seed, r2_environment_seed}.intersection((*planning_seeds, *(r2_planning_seeds or ()))):
        raise ValueError('evaluation seed overlaps the public planning worlds')
    configurations = (('A', (), None), ('B', planning_seeds, r2_planning_seeds),
                      ('C', (environment_seed,), (r2_environment_seed,) if r2_case is not None else None))
    results = {}
    for label, worlds, r2_worlds in configurations:
        cutoff = min(deadline, time.perf_counter() + seconds)
        started = time.perf_counter()
        route, estimates = (choose_route(case, worlds, chooser_for, cutoff, r2_case=r2_case, r2_worlds=r2_worlds)
                            if worlds else (INCUMBENT, []))
        planning_ms = (time.perf_counter() - started) * 1000 if worlds else 0.
        if route is None:
            results[label] = dict(terminal=False, episode=None, estimates=estimates,
                                  planning_ms=planning_ms, error='incomplete_candidate_evaluation')
            continue
        continuation = (chooser_for if continuation_factory is None else
                        partial(continuation_factory, planning_seed=planning_seed, deadline=cutoff))
        trial = _finish_route(case, route, environment_seed, continuation, cutoff, episode_id,
                              r2_case=r2_case, r2_seed=r2_environment_seed)
        results[label] = dict(terminal=trial['terminal'], episode=trial, estimates=estimates,
                              planning_ms=planning_ms, error=trial['error'])
    a, b, c = (results[label]['episode'] for label in ('A', 'B', 'C'))
    return dict(case=case.name, environment_seed=environment_seed, planning_seed=planning_seed,
                public_rollout_seeds=planning_seeds, r2_environment_seed=r2_environment_seed,
                r2_public_rollout_seeds=r2_planning_seeds, arms=results,
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
    parser.add_argument('--continuation', choices=('prior',), default='prior',
                        help='Use the synthetic public prior for complete rollout')
    parser.add_argument('--two-round', action='store_true', help='Use the synthetic stamina and bottle carryover proxy')
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
    if args.two_round:
        first, r2_case = load_two_round_cases()
        cases_to_run = (first,)
    else:
        cases_to_run = tuple(load_case(name) for name in names)
        r2_case = None
    deadline = time.perf_counter() + args.batch_seconds
    cases = {}
    for offset, case in enumerate(cases_to_run):
        rows = []
        for index in range(args.worlds):
            if time.perf_counter() >= deadline:
                break
            rows.append(evaluate_world(case, environment_base + offset * CASE_SEED_STRIDE + index,
                planning_base + offset * CASE_SEED_STRIDE + index, planning_count=args.planning_worlds,
                seconds=args.episode_seconds, deadline=deadline, r2_case=r2_case))
        cases[case.name] = dict(family=case.family, sources=case.sources, ruleset=case.kernel.ruleset_id,
            objective='r1_registered_plus_synthetic_r2_raw' if args.two_round else 'synthetic_terminal_score',
            routes=[asdict(route) for route in (INCUMBENT, *case.routes)], requested=args.worlds,
            not_run=args.worlds - len(rows), worlds=rows, summary=summarize(rows, args.worlds))
    complete = all(case['not_run'] == 0 and all(arm['terminal'] for row in case['worlds']
                   for arm in row['arms'].values()) for case in cases.values())
    report = dict(version='task285-v1', scope='synthetic_offline_conditional_diagnostic', cohort=args.cohort,
        rollout_policy='synthetic_public_score_prior', oracle_arm='C', continuation=args.continuation,
        r2_proxy_case=r2_case.name if r2_case is not None else None,
        referee_sampler='HifEnvironment/SampledHifKernel',
        episode_seconds=args.episode_seconds, batch_seconds=args.batch_seconds,
        planning_worlds=args.planning_worlds, complete=complete, cases=cases)
    write_report(args.output, report)
    print(json.dumps(dict(complete=complete, output=str(args.output),
                          summaries={name: record['summary'] for name, record in cases.items()})))
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
