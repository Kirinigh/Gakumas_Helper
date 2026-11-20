"""Offline episode runner over the unchanged shared fast policy."""
from __future__ import annotations

import time
from dataclasses import asdict, replace

from agent.live_decision.high_score.contracts import ActionKind


def execute_episode(environment, policy, initial_request, *, episode_id, deadline, publish=None):
    """Run the existing sampled rule environment to its actual terminal state.

    Unlike the older evaluator's redundant `action in legal` check, physical
    duplicate drink slots are checked by environment.step itself. That method
    checks both the actual bottle and the canonical semantic action.
    """
    request = initial_request
    actions, latencies = [], []
    began = time.perf_counter()
    error, illegal = None, False
    try:
        while not request.state.terminal:
            if time.perf_counter() >= deadline:
                raise TimeoutError('complete_episode_batch_timeout')
            legal = environment.legal_actions()
            if publish:
                publish(dict(phase='choose', action_count=len(actions), turn=request.state.turn))
            started = time.perf_counter()
            try:
                action = policy.choose(request, legal)
            finally:
                latencies.append((time.perf_counter() - started) * 1000)
            if time.perf_counter() >= deadline:
                raise TimeoutError('complete_episode_batch_timeout')
            if replace(action, source_slot=None) not in tuple(replace(a, source_slot=None) for a in legal):
                illegal = True
                raise ValueError('illegal_policy_action')
            if action.kind is not ActionKind.USE_DRINK and action not in legal:
                illegal = True
                raise ValueError('illegal_policy_action')
            if action.kind is ActionKind.USE_DRINK and (
                    (action.source_slot, action.drink_id) not in request.drink_slots):
                illegal = True
                raise ValueError('illegal_physical_drink_action')
            before = request.state
            step = environment.step(action)
            request = step.request
            actions.append(dict(action=asdict(action), before=asdict(before),
                after=asdict(request.state), events=[asdict(event) for event in step.events]))
            if hasattr(policy, 'observe_transition'):
                policy.observe_transition(before, action, step)
            else:
                policy.committed(before, action, request.state)
            if time.perf_counter() >= deadline:
                raise TimeoutError('complete_episode_batch_timeout')
    except Exception as failure:
        error = dict(type=type(failure).__name__, message=str(failure))
    complete = request.state.terminal and error is None
    return dict(result=dict(episode_id=str(episode_id),
        final_score=float(request.state.score) if complete else 0., stopped=not complete,
        illegal_action=illegal, decision_count=len(actions), latency_ms=sum(latencies),
        reason='terminal' if complete else error['message']),
        terminal=complete, observed_score=request.state.score, actions=actions,
        decision_latency_ms=latencies, episode_seconds=time.perf_counter() - began,
        error=error, decisions=getattr(policy, 'decisions', []),
        secondary_plans=getattr(policy, 'secondary_plans', []),
        plans=getattr(policy, 'plans', []), timing=getattr(policy, 'timing', {}),
        scoring_convention='actual_sampled_terminal_score; incomplete sentinel=0, excluded from quality aggregates')
