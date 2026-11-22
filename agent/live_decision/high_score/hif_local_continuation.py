"""Window-local complete continuation comparisons and shared traversal accounting."""
from __future__ import annotations

import math
import time
from dataclasses import field, dataclass


@dataclass
class LocalContinuation:
    limit: int
    outer_required: int
    seed: int | None = None
    started: dict = field(default_factory=lambda: dict(training=0, outer=0, inner=0))
    completed: dict = field(default_factory=lambda: dict(training=0, outer=0, inner=0))
    cache: dict = field(default_factory=dict)
    calls: int = 0
    cache_hits: int = 0
    terminal_evaluations: int = 0
    terminal_fallbacks: int = 0
    tail_hits: int = 0
    tail_misses: int = 0
    terminal_eligible: int = 0
    terminal_declined: int = 0
    seeds: tuple[int, ...] = ()

    @property
    def total_started(self):
        return sum(self.started.values())

    def start(self, kind):
        if self.total_started >= self.limit:
            raise ValueError('mcts_continuation_total_budget_incomplete')
        self.started[kind] += 1

    def choose(self, cache_key, action_keys, evaluate, deadline):
        """Evaluate all alternatives with the fixed independent planning cohort.

        The caller supplies only public state to evaluate. This object has no
        outer scenario seed, sampler or unseen draw access. Cache entries keep
        absolute complete scores, so their key must include the full score.
        """
        if time.perf_counter() >= deadline:
            raise TimeoutError('mcts_window_timeout')
        if self.seeds and (type(self.seeds) is not tuple or len(self.seeds) != 4
                or any(type(seed) is not int for seed in self.seeds) or len(set(self.seeds)) != 4):
            raise ValueError('mcts_continuation_invalid_independent_seeds')
        seeds = self.seeds or (self.seed,)
        seed_audit = dict(planning_seeds=self.seeds) if self.seeds else dict(planning_seed=self.seed)
        self.calls += 1
        if cache_key in self.cache:
            self.cache_hits += 1
            selected, samples = self.cache[cache_key]
            return selected, dict(cache_hit=True, **seed_audit, samples=samples)
        remaining_outer = self.outer_required - self.started['outer']
        if self.total_started + len(action_keys) * len(seeds) + remaining_outer > self.limit:
            raise ValueError('mcts_continuation_action_coverage_budget_incomplete')
        if seeds == (None,):
            raise ValueError('mcts_continuation_seed_not_frozen')
        samples = []
        for seed in seeds:
            for key in action_keys:
                self.start('inner')
                score, kind, depth, calls, fallbacks = evaluate(key, seed, deadline)
                if time.perf_counter() >= deadline:
                    raise TimeoutError('mcts_window_timeout')
                if not math.isfinite(score):
                    raise ValueError('mcts_continuation_nonfinite_complete_return')
                self.completed['inner'] += 1
                self.terminal_evaluations += calls
                self.terminal_fallbacks += fallbacks
                samples.append(dict(action_key=key, final_score=score, return_kind=kind,
                    completed_depth=depth, terminal_evaluations=calls, terminal_fallbacks=fallbacks,
                    **(dict(planning_seed=seed) if self.seeds else {})))
        # Input order is the existing deterministic action order, used only
        # to resolve a complete-value tie. Partial groups are never cached.
        selected = (max(action_keys, key=lambda key: sum(row['final_score'] for row in samples
                    if row['action_key'] == key) / len(seeds)) if self.seeds else
                    max(samples, key=lambda row: row['final_score'])['action_key'])
        samples = tuple(samples)
        self.cache[cache_key] = selected, samples
        return selected, dict(cache_hit=False, **seed_audit, samples=samples)
