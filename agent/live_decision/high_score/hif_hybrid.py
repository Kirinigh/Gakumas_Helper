"""Shared model-prior/complete-endgame candidate; input remains separately gated."""
import math
import time
from types import SimpleNamespace
from dataclasses import asdict, replace, dataclass

from .planner import PlannerConfig, SearchIncomplete
from .contracts import DecisionTrace, _jsonable
from .hif_planner import HifActionChainPlanner

VISIBLE_HIF_CANDIDATE_PROFILE = 'hybrid-ppo27502-visible-d-trial-v1'


@dataclass(frozen=True)
class HifPriorTrace:
    value_scope: str
    elapsed_ms: float
    model_version: str
    ranking_priors: tuple = ()


@dataclass(frozen=True)
class HifPriorDecision:
    """An action ranking is not a final-score DecisionResponse."""
    action: object | None
    reason: str
    trace: HifPriorTrace

    @property
    def has_action_intent(self):
        return self.action is not None

    def to_dict(self):
        return dict(_jsonable(asdict(self)), protocol_version='task275-hif-prior-intent-v1',
                    has_action_intent=self.has_action_intent, value=None)


class HybridHifPolicy:
    def __init__(self, kernel, advice, *, budget_ms=500, endgame_budget=None):
        self.kernel = kernel
        self.advice = advice
        self.budget_ms = budget_ms
        self.endgame = HifActionChainPlanner(kernel, PlannerConfig(
            max_decision_depth=64, timeout_ms=budget_ms))
        if endgame_budget is not None:
            from .hif_budget import TieredHifPlanner
            self.endgame = TieredHifPlanner(kernel, endgame_budget)

    def decide(self, request):
        # Never replace an incomplete terminal tree with a model prior.
        if request.state.turns_remaining == 1:
            return self.endgame.decide(request)
        started = time.perf_counter()
        action, reason, priors = None, 'offline_model_prior', ()
        try:
            request.validate()
            if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
                raise ValueError('capability_or_ruleset_not_ready')
            legal = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                              decision_kind=request.decision_kind)
            if not legal:
                raise ValueError('no_supported_legal_action')
            priors = tuple(self.advice.action_priors(request.state, legal))
            if len(priors) != len(legal) or any(not math.isfinite(p) for p in priors):
                raise ValueError('invalid_model_priors')
            selected = legal[max(range(len(legal)), key=lambda index: priors[index])]
            action = self.endgame._attach_source_slot(selected, request)
        except (ValueError, SearchIncomplete) as error:
            reason = f'offline_model_stopped:{error}'
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed >= self.budget_ms:
            action, reason = None, 'offline_model_stopped:planner_timeout'
        return HifPriorDecision(action, reason, HifPriorTrace(
            'MODEL_ACTION_PRIOR', elapsed, self.advice.model_version, priors))


def load_frozen_hif_candidate(kernel, model_path):
    """Load the locally trained and evaluated candidate into the native adapter."""
    import json

    from .challenger import LinearBehaviorCloningModel
    from .hif_budget import HifSearchBudget
    from .hif_learning import HifLearningAdvice

    payload = json.loads(model_path.read_text(encoding='utf-8'))
    model = payload['model']
    for field in ('feature_names', 'feature_scales', 'weights'):
        model[field] = tuple(model[field])
    advice = HifLearningAdvice(kernel, LinearBehaviorCloningModel(**model),
                              value_weights=payload['value_weights'])
    if advice.model_version != 'learning-r1-ppo-27502-v1':
        raise ValueError('unapproved HIF candidate model version')
    return HybridHifPolicy(kernel, advice, endgame_budget=HifSearchBudget(allow_slow=True))


@dataclass(frozen=True)
class VisibleHifPriorTrace(HifPriorTrace):
    policy_profile: str = VISIBLE_HIF_CANDIDATE_PROFILE
    trial_scope: str = 'CURRENT_HIF_ROUND_EXPLICIT_TRIAL'
    wrapper_version: str = ''
    intervention: str | None = None


@dataclass(frozen=True)
class VisibleHifSearchTrace(DecisionTrace):
    policy_profile: str = VISIBLE_HIF_CANDIDATE_PROFILE
    trial_scope: str = 'CURRENT_HIF_ROUND_EXPLICIT_TRIAL'
    wrapper_version: str = ''
    intervention: str | None = None


class VisibleHifCandidatePolicy:
    """Explicit trial adapter for the evaluated D wrapper, never the default.

    This changes legal action ranking only. Complete endgame values and STOP
    outcomes retain their original response type and semantics; ordinary
    interventions remain prior intents with no claimed final-score value.
    """

    policy_profile = VISIBLE_HIF_CANDIDATE_PROFILE

    def __init__(self, kernel, baseline):
        from .hif_teachers import VisibleHifContinuationWrapper

        self.kernel, self.baseline = kernel, baseline
        self.wrapper_type = VisibleHifContinuationWrapper

    def _tag(self, response):
        if response.trace is None:
            return response
        trace_type = VisibleHifPriorTrace if isinstance(response, HifPriorDecision) else VisibleHifSearchTrace
        return replace(response, trace=trace_type(**vars(response.trace),
            wrapper_version=self.wrapper_type.version))

    def decide(self, request):
        state = request.state
        if state.turns_remaining == 1 or state.pending_choice or state.resolution_stack:
            return self._tag(self.baseline.decide(request))
        started = time.perf_counter()
        baseline_responses = []

        def baseline_decide(current):
            response = self.baseline.decide(current)
            baseline_responses.append(response)
            return response

        action, intervention, reason = None, None, 'visible_candidate_stopped'
        try:
            request.validate()
            if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
                raise ValueError('capability_or_ruleset_not_ready')
            raw_legal = self.kernel.legal_actions(state, plan_type=request.plan_type,
                decision_kind=request.decision_kind)
            legal = tuple(self.baseline.endgame._attach_source_slot(candidate, request)
                          for candidate in raw_legal)
            if not legal:
                raise ValueError('no_supported_legal_action')
            wrapper = self.wrapper_type(self.kernel, SimpleNamespace(
                decide=baseline_decide, advice=self.baseline.advice), 'early-301-late-35')
            action = wrapper.choose(request, legal)
            if action not in legal:
                raise ValueError('visible_candidate_action_not_legal')
            if wrapper.interventions:
                intervention = wrapper.interventions[-1]['reason']
                reason = 'visible_candidate:'+intervention
        except (ValueError, SearchIncomplete) as error:
            action, reason = None, 'visible_candidate_stopped:'+str(error)
        elapsed = (time.perf_counter()-started)*1000
        if elapsed >= self.baseline.budget_ms:
            action, reason = None, 'visible_candidate_stopped:planner_timeout'
        elif baseline_responses and (not baseline_responses[-1].has_action_intent
                or action == baseline_responses[-1].action and intervention is None):
            # Preserve the original failure or unchanged delegated decision.
            return self._tag(baseline_responses[-1])
        return HifPriorDecision(action, reason, VisibleHifPriorTrace(
            'MODEL_ACTION_PRIOR', elapsed, self.baseline.advice.model_version,
            wrapper_version=self.wrapper_type.version, intervention=intervention))


def load_visible_hif_candidate(kernel, model_path):
    """Opt in to the current-round D trial; frozen/default loaders stay unchanged."""
    return VisibleHifCandidatePolicy(kernel, load_frozen_hif_candidate(kernel, model_path))
