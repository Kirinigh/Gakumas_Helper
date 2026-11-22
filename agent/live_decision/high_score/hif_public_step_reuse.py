"""Complete work reused within one frozen public graph, never input evidence."""
from __future__ import annotations

import math
import time

from .contracts import ActionKind
from .state_copy import replace
from .hif_environment import HifStep
from .hif_chain_candidate import PairedPlanningKernel

VERSION = 'hif-public-step-reuse-v2'
_MISS = object()


class PublicStepReuse:
    """Reuse complete public choices and sampled steps within one frozen phase.

    The owner supplies one fixed policy and catalogue for this object's lifetime.
    Root controls and depth limits stay in the rollout. Successful suffixes
    may also be shared by the qualified, fixed validation policy. Actual tail
    calls and reused tail results have separate counters. No state is exported
    or retained for another planning window.
    """

    def __init__(self, kernel, deadline, *, complete_returns=False):
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('public_step_reuse_invalid_deadline')
        self.kernel, self.deadline = kernel, deadline
        self._actions, self._steps = {}, {}
        self.complete_returns = complete_returns
        self._returns = {}
        self.return_hits = self.return_steps_saved = self.return_tail_calls_saved = 0
        self.action_computations = self.action_hits = 0
        self.step_computations = self.step_hits = self.step_bypasses = 0

    def _check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('public_step_reuse_timeout')

    def choose(self, request, actions, compute):
        from .hif_mcts import action_key

        self._check()
        actions = tuple(actions)
        key = (replace(request, snapshot_id='same-window-public-graph'), actions)
        selected = self._actions.get(key, _MISS)
        hit = selected is not _MISS
        if not hit:
            selected = compute(request, actions)
            self._check()
        if sum(action_key(action) == selected for action in actions) != 1:
            raise ValueError('public_step_reuse_action_not_uniquely_legal')
        self._check()
        if hit:
            self.action_hits += 1
        else:
            self._actions[key] = selected
            self.action_computations += 1
        return selected

    @staticmethod
    def _check_cached_step(environment, action):
        # Match HifEnvironment.step, including a non-first physical duplicate
        # bottle that is legal although legal_actions binds its first bottle.
        if environment.state is None or environment.state.terminal:
            raise ValueError('HIF episode has no active state')
        legal = environment.legal_actions()
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in environment._drink_slots:
                raise ValueError('HIF drink action has no physical bottle')
        elif action not in legal:
            raise ValueError('HIF action physical slot does not match its legal candidate')
        if replace(action, source_slot=None) not in tuple(replace(a, source_slot=None) for a in legal):
            raise ValueError('HIF environment illegal action')
        if environment._steps >= 512:
            raise ValueError('HIF episode action limit exceeded; episode is incomplete')

    def step(self, environment, action):
        self._check()
        sampler = environment._sampler
        if environment.kernel is not self.kernel or type(sampler) is not PairedPlanningKernel:
            result = environment.step(action)
            self._check()
            self.step_computations += 1
            self.step_bypasses += 1
            return result
        key = (environment.state, environment._drink_slots, action,
               sampler.planning_seed, sampler.draw_index, sampler._rng.getstate())
        cached = self._steps.get(key, _MISS)
        if cached is _MISS:
            result = environment.step(action)
            self._check()
            after = (result, environment._drink_slots, sampler.draw_index,
                     sampler._rng.getstate(), tuple(sampler._boundaries), sampler.last_boundary)
            self._check()
            self._steps[key] = after
            self.step_computations += 1
            return result
        self._check_cached_step(environment, action)
        self._check()
        previous, slots, draw_index, rng, boundaries, last_boundary = cached
        environment.state = previous.request.state
        environment._drink_slots = slots
        environment._steps += 1
        sampler.draw_index = draw_index
        sampler._rng.setstate(rng)
        sampler._boundaries = list(boundaries)
        sampler.last_boundary = last_boundary
        result = HifStep(environment.request(), previous.reward, previous.terminated,
                         previous.events, previous.boundary)
        self._check()
        self.step_hits += 1
        return result

    def return_key(self, environment, request):
        """A suffix starts after all root controls, under one immutable policy."""
        self._check()
        sampler = environment._sampler
        if (not self.complete_returns or environment.kernel is not self.kernel
                or type(sampler) is not PairedPlanningKernel):
            return None
        return (replace(request, snapshot_id='same-window-public-graph'),
                sampler.planning_seed, sampler.draw_index, sampler._rng.getstate())

    def completed_return(self, key, *, remaining_depth, remaining_steps):
        self._check()
        cached = self._returns.get(key) if key is not None else None
        if cached is None:
            return None
        score, depth, tail_calls = cached
        # The original loop must reach its return before range() is exhausted.
        # The environment's step ceiling, unlike that loop, allows equality.
        if depth >= remaining_depth or depth > remaining_steps:
            return None
        self._check()
        self.return_hits += 1
        self.return_steps_saved += depth
        self.return_tail_calls_saved += tail_calls
        return cached

    def remember_return(self, checkpoints, *, score, depth, tail_calls):
        self._check()
        if not math.isfinite(score):
            raise ValueError('public_step_reuse_invalid_complete_return')
        for key, at_depth, previous_calls in checkpoints:
            self._returns[key] = (score, depth - at_depth, tail_calls - previous_calls)
        self._check()

    def summary(self):
        return dict(version=VERSION, action_computations=self.action_computations,
                    action_hits=self.action_hits, action_entries=len(self._actions),
                    step_computations=self.step_computations, step_hits=self.step_hits,
                    step_entries=len(self._steps), step_bypasses=self.step_bypasses,
                    complete_returns_enabled=self.complete_returns,
                    return_entries=len(self._returns), return_hits=self.return_hits,
                    return_steps_saved=self.return_steps_saved,
                    return_tail_calls_saved=self.return_tail_calls_saved)
