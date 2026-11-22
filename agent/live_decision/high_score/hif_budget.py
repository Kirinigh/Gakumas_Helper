"""Elapsed-time tiers for one uninterrupted HIF search, separate from UI clocks."""
import math
import time
from dataclasses import dataclass

from .planner import PlannerConfig, SearchIncomplete
from .state_copy import replace
from .hif_planner import HifActionChainPlanner


@dataclass(frozen=True)
class HifSearchBudget:
    target_ms: float = 500
    normal_ms: float = 3000
    slow_ms: float = 15000
    allow_slow: bool = False

    def validate(self):
        values = (self.target_ms, self.normal_ms, self.slow_ms)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 < self.target_ms <= self.normal_ms <= self.slow_ms
                or type(self.allow_slow) is not bool):
            raise ValueError('invalid HIF search budget tiers')

    def limit_ms(self, state):
        return self.slow_ms if self.allow_slow and state.turns_remaining == 1 else self.normal_ms


class TieredHifPlanner(HifActionChainPlanner):
    """Crossing soft tiers never restarts the tree or clears its memo tables."""

    def __init__(self, kernel, budget=None):
        self.budget = budget or HifSearchBudget()
        self.budget.validate()
        super().__init__(kernel, PlannerConfig(max_decision_depth=64, timeout_ms=self.budget.normal_ms))

    def decide(self, request):
        self.config = replace(self.config, timeout_ms=self.budget.limit_ms(request.state))
        self._tier_started = time.perf_counter()
        self._target_crossed = self._slow_entered = False
        response = super().decide(request)
        elapsed = (time.perf_counter() - self._tier_started) * 1000
        self._target_crossed |= elapsed > self.budget.target_ms
        self._slow_entered |= elapsed > self.budget.normal_ms and self.config.timeout_ms > self.budget.normal_ms
        # Include trace construction/cleanup in the hard bound too. A finished
        # tree delivered after the limit is still not an actionable decision.
        if response.has_action_intent and elapsed > self.config.timeout_ms:
            response = self._stop('search_incomplete:planner_timeout', self._tier_started,
                                  expanded_nodes=self._expanded_nodes)
        tier = 'slow' if self._slow_entered else 'normal' if self._target_crossed else 'target'
        if response.trace is not None:
            response = replace(response, trace=replace(response.trace, elapsed_ms=elapsed,
                reason_codes=response.trace.reason_codes + (
                    f'budget_tier:{tier}', f'budget_limit_ms:{self.config.timeout_ms:g}',
                    f'budget_target_ms:{self.budget.target_ms:g}',
                    f'budget_normal_ms:{self.budget.normal_ms:g}', 'budget_search_runs:1')))
        return response

    def _check_budget(self):
        now = time.perf_counter()
        if now > self._deadline:
            raise SearchIncomplete('planner_timeout')
        elapsed = (now - self._tier_started) * 1000
        if elapsed > self.budget.target_ms:
            self._target_crossed = True
        if elapsed > self.budget.normal_ms and self.config.timeout_ms > self.budget.normal_ms:
            self._slow_entered = True
