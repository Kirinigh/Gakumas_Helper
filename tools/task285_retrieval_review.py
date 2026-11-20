"""Compare post-draw retrieval allocation on a caller-supplied public HIF case."""

import math
import time
from statistics import fmean, stdev
from dataclasses import asdict

from tools.task285_cases import Route
from tools.task285_seed_bench import INCUMBENT, RoutePolicy
from agent.live_decision.high_score.hif_environment import HifEnvironment

BASE_SAMPLES = 4
FINAL_SAMPLES = 4
MAX_ROOTS = 64


def _reset(case, seed):
    env = HifEnvironment(case.kernel)
    env.reset(case.state, seed=seed, drink_slots=case.drink_slots)
    return env


def _routes(case, deadline, root_builder):
    env = _reset(case, 0)
    options, audit = root_builder(case.kernel, env.request(), env.legal_actions(),
                                  max_options=MAX_ROOTS, deadline=deadline)
    return tuple(Route(str(index), (option.action, *option.continuation))
                 for index, option in enumerate(options)), audit


def _terminal(case, prior, route, seed, deadline):
    env = _reset(case, seed)
    policy = RoutePolicy(prior, route)
    request = env.request()
    while not request.state.terminal:
        if time.perf_counter() >= deadline:
            raise TimeoutError('retrieval review deadline')
        request = env.step(policy.choose(request, env.legal_actions())).request
    if policy.index != len(route.prefix):
        raise ValueError('retrieval route ended before its legal prefix completed')
    return request.state.score


def finalist_pairs(scores, retained_index):
    """Only use the retained target when the base paired gap is uncertain."""
    if len(scores) < 2 or len(scores[0]) < 2 or any(len(row) != len(scores[0]) for row in scores):
        raise ValueError('paired finalist comparison needs two routes and at least two common worlds')
    order = sorted(range(len(scores)), key=lambda index: -fmean(scores[index]))
    leader, runner_up = order[:2]
    pair = (leader, runner_up)
    if retained_index is None or retained_index in pair:
        return pair, pair
    gaps = [a - b for a, b in zip(scores[leader], scores[retained_index], strict=True)]
    se = stdev(gaps) / math.sqrt(len(gaps))
    return pair, (leader, retained_index) if fmean(gaps) <= se else pair


def _same_retrieval(route, reference):
    return len(route.prefix) == len(reference.prefix) == 2 and route.prefix == reference.prefix


def compare_public(case, prior, reference, planning_seed, deadline, *, root_builder):
    """Compare refinements from the caller's public legal-root builder."""
    routes, audit = _routes(case, deadline, root_builder)
    matching = [index for index, route in enumerate(routes) if _same_retrieval(route, reference)]
    retained = matching[0] if len(matching) == 1 else None
    if len(routes) < 2:
        return INCUMBENT, INCUMBENT, dict(reason='fewer_than_two_roots', retained_count=len(matching))
    base_seeds = range(planning_seed, planning_seed + BASE_SAMPLES)
    final_seeds = range(base_seeds.stop, base_seeds.stop + FINAL_SAMPLES)
    scores = [[_terminal(case, prior, route, seed, deadline) for seed in base_seeds] for route in routes]
    a, b = finalist_pairs(scores, retained)
    compared = {index: [_terminal(case, prior, routes[index], seed, deadline) for seed in final_seeds]
                for index in sorted(set(a + b))}
    control, candidate = (max(pair, key=lambda index: fmean(compared[index])) for pair in (a, b))
    audit.update(retained_count=len(matching), retained_index=retained,
        retained_base_rank=(sorted(range(len(scores)), key=lambda i: -fmean(scores[i])).index(retained) + 1
                            if retained is not None else None),
        candidates=[dict(prefix=[asdict(action) for action in route.prefix], base_scores=values,
                         final_scores=compared.get(index))
                    for index, (route, values) in enumerate(zip(routes, scores, strict=True))],
        control_pair=a, candidate_pair=b, selected_control=control, selected_candidate=candidate,
        planned_returns_per_arm=len(routes) * BASE_SAMPLES + 2 * FINAL_SAMPLES,
        base_seeds=list(base_seeds), final_seeds=list(final_seeds))
    return routes[control], routes[candidate], audit
