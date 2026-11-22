"""Window-local values from complete two-turn proofs, frozen before reuse."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .kernel import KernelError
from .planner import _action_key
from .contracts import HighScoreContractError
from .state_copy import replace

VERSION = 'task275-proven-two-turn-values-v1'


@dataclass(frozen=True)
class _ProvenValue:
    expected_increment: float
    minimum_candidate_p10_increment: float
    expected_roundoff: float
    p10_roundoff: float


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _increment_roundoff(total, score, increment):
    # The total is already a rounded float. Subtracting a larger score can
    # lose several increment ULPs, so retain the scale of both operands too.
    return math.ulp(total) + math.ulp(score) + math.ulp(increment)


def _valid_summary(value):
    if value is None or type(value.outcome_count) is not int or value.outcome_count < 1:
        return False
    values = (value.expected_final_score, value.p10_final_score,
              value.worst_final_score, value.best_final_score)
    return (all(_finite(number) for number in values)
            and value.worst_final_score <= value.expected_final_score <= value.best_final_score
            and value.worst_final_score <= value.p10_final_score <= value.best_final_score)


class HifProvenTailTable:
    """Store only explicitly completed roots; never import a planner's memo.

    Entries contain score increments and a baseline admissibility scalar.
    They retain the whole canonical public state and rule/utility namespace,
    but no response, action, physical slot, sampled seed, or simulator state.
    Lookup is available only after freeze and never invokes the planner.
    """

    def __init__(self, kernel, planner):
        if planner.kernel is not kernel:
            raise ValueError('proven tail planner and kernel must match')
        self.kernel, self.planner = kernel, planner
        self.reset_window()

    def reset_window(self):
        self._values = {}
        self.frozen = False

    def begin_window(self):
        self.reset_window()

    def freeze(self):
        self.frozen = True

    def __len__(self):
        return len(self._values)

    def _key(self, request):
        planner, kernel = self.planner, self.kernel
        if (request.state.turns_remaining != 2 or not request.capabilities.all_ready
                or planner.kernel is not kernel or request.ruleset_id != kernel.ruleset_id
                or not planner.complete_horizon or not planner.config.require_terminal_leaf):
            return None
        request.validate()
        namespace = (
            VERSION, id(kernel), type(kernel), kernel.ruleset_id,
            type(planner), planner.planner_name, planner.planner_version,
            planner.terminal_cache, planner.complete_horizon,
            request.protocol_version, request.ruleset_id,
            request.plan_type, request.utility, request.decision_kind, bool(request.drink_slots),
            *(tuple(sorted(getattr(kernel, name).items()))
              for name in ('cards', 'drinks', 'items', 'memory_abilities')),
        )
        state = kernel._canonical_probability_state(replace(request.state, score=0))
        return namespace, state

    def remember(self, request, response):
        """Accept one complete two-turn response before the table is frozen."""
        if self.frozen:
            return False
        try:
            key = self._key(request)
            response.validate()
            trace = response.trace
            if (key is None or not response.has_action_intent or not _valid_summary(response.value)
                    or response.protocol_version != request.protocol_version or trace is None
                    or trace.value_scope != 'FINAL_SCORE' or trace.ruleset_id != request.ruleset_id
                    or trace.planner != self.planner.planner_name
                    or trace.planner_version != self.planner.planner_version):
                return False
            legal = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                               decision_kind=request.decision_kind)
            legal_keys = {_action_key(action) for action in legal}
            candidates = trace.candidate_values
            if not legal_keys or len(candidates) != len(legal_keys):
                return False
            covered = set()
            selected = None
            selected_key = _action_key(response.action)
            for candidate in candidates:
                candidate.action.validate()
                action_key = _action_key(candidate.action)
                if (action_key not in legal_keys or action_key in covered
                        or candidate.rejection_reasons or not _valid_summary(candidate.value)
                        or not _finite(candidate.probability_mass)
                        or not math.isclose(candidate.probability_mass, 1., rel_tol=1e-9, abs_tol=1e-9)
                        or candidate.ranking_utility is not None and not _finite(candidate.ranking_utility)):
                    return False
                covered.add(action_key)
                if action_key == selected_key:
                    selected = candidate.value
            if covered != legal_keys or selected != response.value:
                return False
            baseline = request.utility.baseline_p10
            minimum_p10 = min(candidate.value.p10_final_score for candidate in candidates)
            if baseline is not None and (not _finite(baseline) or minimum_p10 < baseline):
                return False
            expected_increment = response.value.expected_final_score - request.state.score
            p10_increment = minimum_p10 - request.state.score
            value = _ProvenValue(expected_increment, p10_increment,
                _increment_roundoff(response.value.expected_final_score, request.state.score, expected_increment),
                _increment_roundoff(minimum_p10, request.state.score, p10_increment))
            if not _finite(value.expected_increment) or not _finite(value.minimum_candidate_p10_increment):
                return False
            if key in self._values:
                previous = self._values[key]
                if (abs(previous.expected_increment - value.expected_increment)
                        > previous.expected_roundoff + value.expected_roundoff
                        or abs(previous.minimum_candidate_p10_increment - value.minimum_candidate_p10_increment)
                        > previous.p10_roundoff + value.p10_roundoff):
                    raise ValueError('proven_tail_conflicting_complete_values')
                # Accept only representational roundoff and retain the first
                # complete value, independent of later translated requests.
                return True
            self._values[key] = value
            return True
        except (KernelError, HighScoreContractError):
            return False

    def lookup(self, request):
        """Return a frozen expected final score or None, with no solving."""
        if not self.frozen:
            raise ValueError('proven_tail_lookup_requires_frozen_window')
        try:
            key = self._key(request)
        except HighScoreContractError:
            return None
        value = self._values.get(key)
        if value is None:
            return None
        baseline = request.utility.baseline_p10
        if baseline is not None and value.minimum_candidate_p10_increment + request.state.score < baseline:
            return None
        result = value.expected_increment + request.state.score
        return float(result) if _finite(result) else None
