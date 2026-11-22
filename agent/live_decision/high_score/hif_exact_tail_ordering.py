"""Optional complete-terminal ordering for small public HIF suffixes.

This never supplies a guessed score, deletes an action, or authorizes input.
An incomplete proof leaves the caller's ordering unchanged. Frozen validation
can only reuse the window's existing results, including cached failures.
"""
from __future__ import annotations

import math
import time

from .planner import PlannerConfig, SearchIncomplete, RobustHighScorePlanner
from .contracts import ActionKind, RiskProfile
from .validation_cache import validation_session

VERSION = 'task275-exact-tail-ordering-v1-20261002'


def _action_key(action):
    # Match the search consumer without importing it (which would be cyclic).
    return (action.kind.value, action.card_instance_id, action.drink_id,
            action.target_id, action.source_slot if action.kind is ActionKind.USE_DRINK else None)


class _CompleteRootPlanner(RobustHighScorePlanner):
    @staticmethod
    def _merge_equivalent_actions(state, actions):
        # Retain physical roots. No new equivalence or slot-mapping protocol.
        return tuple(actions)


class HifExactTailOrdering:
    """At most 10 ms per proof and 100 ms of work in one search window.

    Only the base complete-horizon planner is used. HifActionChainPlanner's
    multi-turn forecast is deliberately not accepted as a terminal proof.
    The owner must call start_window() before training and freeze() before
    independent validation. Its outer deadline still governs cached reads.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.start_window()

    def start_window(self):
        self._cache = {}
        self._frozen = False
        self.calls = self.cache_hits = self.completed = self.failed = 0
        self.elapsed_ms = 0.
        self.last_reason = 'not_called'

    def reset(self):
        self.start_window()

    def freeze(self):
        self._frozen = True

    def _key(self, request, actions):
        definitions = tuple(tuple(sorted(getattr(self.kernel, name).items()))
                            for name in ('cards', 'drinks', 'items', 'memory_abilities'))
        return (type(self.kernel), self.kernel.ruleset_id, definitions,
                request.state, request.utility, request.plan_type, request.decision_kind,
                request.ruleset_id, request.protocol_version, request.capabilities,
                request.card_slots, request.drink_slots, actions)

    @staticmethod
    def _ranking(candidate, utility):
        value = candidate.value
        if utility.risk_profile is RiskProfile.SAFE_BASELINE:
            values = (value.p10_final_score, value.worst_final_score, value.expected_final_score)
        elif utility.risk_profile is RiskProfile.AGGRESSIVE_HIGH_SCORE:
            values = (value.expected_final_score, value.best_final_score, value.p10_final_score)
        else:
            values = (value.expected_final_score, value.p10_final_score, value.worst_final_score)
        return bool(candidate.rejection_reasons), *(-v for v in values)

    def complete_order(self, request, actions, *, deadline):
        self.calls += 1
        actions = tuple(actions)
        if request.state.terminal or not 1 <= request.state.turns_remaining <= 2:
            self.last_reason = 'outside_two_turn_scope'
            return None
        key = self._key(request, actions)
        if key in self._cache:
            self.cache_hits += 1
            result, self.last_reason = self._cache[key]
            return result
        if self._frozen:
            self.last_reason = 'frozen_cache_miss'
            return None

        started = time.perf_counter()
        remaining = min(10., 100. - self.elapsed_ms, (deadline - started) * 1000)
        result = None
        try:
            if not math.isfinite(deadline) or remaining <= 0:
                self.last_reason = 'proof_budget_exhausted'
                return None
            proof_deadline = started + remaining / 1000
            planner = _CompleteRootPlanner(self.kernel, PlannerConfig(
                max_decision_depth=32, timeout_ms=remaining, require_terminal_leaf=True))

            def check_budget():
                if time.perf_counter() >= proof_deadline:
                    raise SearchIncomplete('planner_timeout')

            with validation_session(check_budget):
                check_budget()
                request.validate()
                if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
                    self.last_reason = 'request_not_ready'
                    return None
                legal = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                                  decision_kind=request.decision_kind)
                expected = tuple(planner._attach_source_slot(action, request) for action in legal)
                if (not actions or len(set(actions)) != len(actions)
                        or len(expected) != len(actions) or set(expected) != set(actions)):
                    self.last_reason = 'physical_root_set_mismatch'
                    return None
                response = planner.decide(request)
                check_budget()
                if not response.has_action_intent or response.trace is None:
                    self.last_reason = response.reason
                    return None
                if response.trace.value_scope != 'FINAL_SCORE':
                    self.last_reason = 'nonterminal_value_scope'
                    return None
                candidates = response.trace.candidate_values
                bound = tuple(planner._attach_source_slot(candidate.action, request) for candidate in candidates)
                if (len(bound) != len(actions) or len(set(bound)) != len(bound)
                        or set(bound) != set(actions)
                        or any(not math.isclose(c.probability_mass, 1., abs_tol=1e-9)
                               or not all(math.isfinite(v) for v in (
                                   c.value.expected_final_score, c.value.p10_final_score,
                                   c.value.worst_final_score, c.value.best_final_score))
                               for c in candidates)):
                    self.last_reason = 'incomplete_terminal_root_coverage'
                    return None
                by_action = dict(zip(bound, candidates, strict=True))
                ordered = sorted(actions, key=lambda action: self._ranking(by_action[action], request.utility))
                check_budget()
                result = tuple(_action_key(action) for action in ordered)
                self.last_reason = 'complete_terminal_order'
                self.completed += 1
                return result
        except (ValueError, RuntimeError) as error:
            self.last_reason = str(error)
            return None
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1000
            self.failed += int(result is None)
            self._cache[key] = (result, self.last_reason)
