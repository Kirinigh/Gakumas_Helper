from __future__ import annotations

import math
import random
from typing import Iterable
from dataclasses import asdict, replace, dataclass

import numpy as np

from .kernel import EffectKind, KernelError, PythonNiaProKernel
from .dataset import DynamicDeckSample
from .contracts import (
    ActionKind,
    GameAction,
    DecisionKind,
    ValueSummary,
    DecisionTrace,
    CandidateValue,
    DecisionOutcome,
    DecisionRequest,
    DecisionResponse,
)
from .evaluation import OfflineDecisionPolicy

_SLOTTED_ACTIONS = {
    ActionKind.PLAY_CARD,
    ActionKind.HOLD_CARD,
    ActionKind.MOVE_CARD,
    ActionKind.FREE_PLAY,
    ActionKind.PAID_PLAY,
}

_EFFECT_FEATURES = (
    EffectKind.SCORE,
    EffectKind.DRAW,
    EffectKind.EXTRA_ACTION,
    EffectKind.CONCENTRATION,
    EffectKind.GOOD_CONDITION,
    EffectKind.EXCELLENT_CONDITION,
    EffectKind.GOOD_IMPRESSION,
    EffectKind.MOTIVATION,
    EffectKind.GENKI,
    EffectKind.FULL_POWER,
    EffectKind.PASSION,
    EffectKind.STANCE,
    EffectKind.SCHEDULE,
    EffectKind.CARD_MODIFICATION,
    EffectKind.HOLD_SELF,
    EffectKind.HOLD_SELECTED,
    EffectKind.MOVE_RANDOM_TO_HAND,
    EffectKind.EXCHANGE_HAND,
    EffectKind.FREE_PLAY_SELECTED,
    EffectKind.FREE_PLAY_ALL,
    EffectKind.PAID_PLAY_SELECTED,
    EffectKind.FREE_PLAY_RANDOM,
)

_DELTA_FIELDS = (
    "stamina",
    "genki",
    "concentration",
    "good_condition_turns",
    "excellent_condition_turns",
    "good_impression_turns",
    "motivation",
    "full_power",
    "cumulative_full_power",
    "passion",
    "actions_remaining",
    "turns_remaining",
)


def _without_slot(action: GameAction) -> GameAction:
    return replace(action, source_slot=None)


def _effect_tree(definition):
    pending = list(definition.effects + definition.persistent_effects)
    while pending:
        effect = pending.pop()
        yield effect
        pending.extend(effect.scheduled_effects)


@dataclass(frozen=True)
class ExpertChoice:
    request: DecisionRequest
    legal_actions: tuple[GameAction, ...]
    expert_action: GameAction


@dataclass(frozen=True)
class ExpertCollection:
    examples: tuple[ExpertChoice, ...]
    episode_count: int
    terminal_episode_count: int
    stopped_episode_count: int
    illegal_expert_action_count: int


@dataclass(frozen=True)
class LinearBehaviorCloningModel:
    model_name: str
    model_version: str
    encoder_name: str
    encoder_version: str
    ruleset_id: str
    feature_names: tuple[str, ...]
    feature_scales: tuple[float, ...]
    weights: tuple[float, ...]
    pair_count: int
    epoch_count: int
    learning_rate: float
    regularization: float
    training_choice_accuracy: float

    def validate(self) -> None:
        size = len(self.feature_names)
        if size == 0 or len(self.feature_scales) != size or len(self.weights) != size:
            raise ValueError("behavior-cloning model vectors have inconsistent sizes")
        if not self.encoder_name or not self.encoder_version or not self.ruleset_id:
            raise ValueError("behavior-cloning model provenance is incomplete")
        if self.pair_count < 1 or self.epoch_count < 1 or self.learning_rate <= 0:
            raise ValueError("behavior-cloning model requires training evidence")
        if not 0 <= self.training_choice_accuracy <= 1:
            raise ValueError("behavior-cloning accuracy is outside its valid range")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.feature_scales):
            raise ValueError("behavior-cloning feature scales must be finite and positive")
        if any(not math.isfinite(weight) for weight in self.weights):
            raise ValueError("behavior-cloning weights must be finite")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


class ActionFeatureEncoder:
    encoder_name = "task275-one-step-action-features"
    encoder_version = "1.0"

    def __init__(self, kernel: PythonNiaProKernel):
        self.kernel = kernel
        self.feature_names = (
            "score_delta_expected",
            "score_delta_worst",
            "score_delta_best",
            *tuple(f"{field}_delta_expected" for field in _DELTA_FIELDS),
            "concentration_turn_value",
            "good_condition_turn_value",
            "excellent_condition_turn_value",
            "good_impression_turn_value",
            "motivation_turn_value",
            "full_power_turn_value",
            "passion_turn_value",
            "terminal_probability",
            "chance_entropy",
            "branch_count",
            "card_limited",
            "card_stamina_cost",
            "card_full_power_cost",
            *tuple(f"action_{kind.value.lower()}" for kind in ActionKind),
            *tuple(f"effect_{kind.value.lower()}" for kind in _EFFECT_FEATURES),
        )

    def encode(
        self,
        request: DecisionRequest,
        action: GameAction,
    ) -> tuple[np.ndarray, tuple, ValueSummary]:
        branches = self.kernel.transition(
            request.state,
            action,
            plan_type=request.plan_type,
        )
        probabilities = np.asarray(
            [branch.probability for branch in branches],
            dtype=np.float64,
        )
        score_deltas = np.asarray(
            [branch.state.score - request.state.score for branch in branches],
            dtype=np.float64,
        )
        expected_deltas = {
            field: sum(
                branch.probability
                * (getattr(branch.state, field) - getattr(request.state, field))
                for branch in branches
            )
            for field in _DELTA_FIELDS
        }
        remaining_turns = max(1, request.state.turns_remaining)
        card_definition = None
        if action.card_instance_id is not None:
            card_ref = next(
                (
                    card
                    for card in request.state.deck.all_cards
                    if card.instance_id == action.card_instance_id
                ),
                None,
            )
            if card_ref is not None:
                card_definition = self.kernel.cards.get(card_ref.card_id)
        effect_kinds = (
            {effect.kind for effect in _effect_tree(card_definition)}
            if card_definition is not None
            else set()
        )
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        values = [
            float(probabilities @ score_deltas),
            float(score_deltas.min()),
            float(score_deltas.max()),
            *tuple(float(expected_deltas[field]) for field in _DELTA_FIELDS),
            float(expected_deltas["concentration"] * remaining_turns),
            float(expected_deltas["good_condition_turns"] * remaining_turns),
            float(expected_deltas["excellent_condition_turns"] * remaining_turns),
            float(expected_deltas["good_impression_turns"] * remaining_turns),
            float(expected_deltas["motivation"] * remaining_turns),
            float(expected_deltas["full_power"] * remaining_turns),
            float(expected_deltas["passion"] * remaining_turns),
            float(sum(branch.probability for branch in branches if branch.state.terminal)),
            float(entropy),
            float(len(branches)),
            float(card_definition.limited if card_definition is not None else False),
            float(card_definition.stamina_cost if card_definition is not None else 0),
            float(card_definition.full_power_cost if card_definition is not None else 0),
            *tuple(float(action.kind is kind) for kind in ActionKind),
            *tuple(float(kind in effect_kinds) for kind in _EFFECT_FEATURES),
        ]
        if len(values) != len(self.feature_names):
            raise RuntimeError("action feature vector does not match its schema")
        final_scores = [float(branch.state.score) for branch in branches]
        value = ValueSummary(
            expected_final_score=sum(
                branch.probability * branch.state.score for branch in branches
            ),
            p10_final_score=min(final_scores),
            worst_final_score=min(final_scores),
            best_final_score=max(final_scores),
            outcome_count=len(branches),
        )
        return np.asarray(values, dtype=np.float64), branches, value


def collect_expert_choices(
    samples: tuple[DynamicDeckSample, ...],
    expert: OfflineDecisionPolicy,
    kernel: PythonNiaProKernel,
    *,
    maximum_decisions: int = 64,
) -> ExpertCollection:
    examples: list[ExpertChoice] = []
    terminal_count = 0
    stopped_count = 0
    illegal_count = 0
    for sample in samples:
        rng = random.Random(sample.seed)
        state = sample.request.state
        terminal = False
        stopped = False
        for decision_index in range(maximum_decisions):
            if state.terminal:
                terminal = True
                break
            decision_kind = (
                DecisionKind.SECONDARY
                if state.pending_choice is not None
                else DecisionKind.CARD
            )
            slot_ids = (
                state.pending_choice.candidate_instance_ids
                if state.pending_choice is not None
                else tuple(card.instance_id for card in state.deck.hand)
            )
            request = replace(
                sample.request,
                snapshot_id=f"{sample.episode_id}-expert-{decision_index}",
                state=state,
                decision_kind=decision_kind,
                card_slots=tuple(
                    (instance_id, index)
                    for index, instance_id in enumerate(slot_ids)
                ),
            )
            legal_actions = kernel.legal_actions(
                state,
                plan_type=request.plan_type,
                decision_kind=decision_kind,
            )
            response = expert.decide(request)
            if response.outcome is not DecisionOutcome.DECIDE or response.action is None:
                stopped = True
                break
            expert_action = _without_slot(response.action)
            if expert_action not in legal_actions:
                illegal_count += 1
                stopped = True
                break
            if len(legal_actions) > 1:
                examples.append(
                    ExpertChoice(request, legal_actions, expert_action)
                )
            branches = kernel.transition(
                state,
                expert_action,
                plan_type=request.plan_type,
            )
            draw = rng.random()
            cumulative = 0.0
            selected = branches[-1]
            for branch in branches:
                cumulative += branch.probability
                if draw <= cumulative:
                    selected = branch
                    break
            state = selected.state
        if terminal:
            terminal_count += 1
        elif stopped or not state.terminal:
            stopped_count += 1
    return ExpertCollection(
        examples=tuple(examples),
        episode_count=len(samples),
        terminal_episode_count=terminal_count,
        stopped_episode_count=stopped_count,
        illegal_expert_action_count=illegal_count,
    )


def _choice_accuracy(
    examples: Iterable[ExpertChoice],
    encoder: ActionFeatureEncoder,
    scales: np.ndarray,
    weights: np.ndarray,
) -> float:
    correct = 0
    count = 0
    for example in examples:
        candidates = []
        for action in example.legal_actions:
            features, _branches, _value = encoder.encode(example.request, action)
            score = float((features / scales) @ weights)
            candidates.append((score, action))
        selected = min(
            candidates,
            key=lambda item: (
                -item[0],
                item[1].kind.value,
                item[1].card_instance_id or "",
            ),
        )[1]
        correct += selected == example.expert_action
        count += 1
    return correct / count if count else 0.0


def train_linear_behavior_cloning(
    examples: tuple[ExpertChoice, ...],
    encoder: ActionFeatureEncoder,
    *,
    epoch_count: int = 300,
    learning_rate: float = 0.08,
    regularization: float = 0.002,
) -> LinearBehaviorCloningModel:
    if epoch_count < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError("behavior-cloning training parameters are invalid")
    differences: list[np.ndarray] = []
    for example in examples:
        encoded = {
            action: encoder.encode(example.request, action)[0]
            for action in example.legal_actions
        }
        expert_features = encoded[example.expert_action]
        differences.extend(
            expert_features - features
            for action, features in encoded.items()
            if action != example.expert_action
        )
    if not differences:
        raise ValueError("behavior-cloning training requires competing expert choices")
    raw = np.vstack(differences)
    scales = np.std(raw, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    matrix = raw / scales
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    for _epoch in range(epoch_count):
        margins = np.clip(matrix @ weights, -40.0, 40.0)
        errors = 1.0 / (1.0 + np.exp(margins))
        gradient = -(matrix.T @ errors) / len(matrix)
        gradient += regularization * weights
        weights -= learning_rate * gradient
    model = LinearBehaviorCloningModel(
        model_name="task275-linear-behavior-cloning",
        model_version="0.1",
        encoder_name=encoder.encoder_name,
        encoder_version=encoder.encoder_version,
        ruleset_id=encoder.kernel.ruleset_id,
        feature_names=encoder.feature_names,
        feature_scales=tuple(float(value) for value in scales),
        weights=tuple(float(value) for value in weights),
        pair_count=len(matrix),
        epoch_count=epoch_count,
        learning_rate=learning_rate,
        regularization=regularization,
        training_choice_accuracy=_choice_accuracy(
            examples,
            encoder,
            scales,
            weights,
        ),
    )
    model.validate()
    return model


class LinearBehaviorCloningPolicy:
    policy_name = "isolated-linear-bc-challenger"
    policy_version = "0.1"

    def __init__(
        self,
        kernel: PythonNiaProKernel,
        model: LinearBehaviorCloningModel,
    ):
        model.validate()
        self.kernel = kernel
        self.model = model
        self.encoder = ActionFeatureEncoder(kernel)
        if model.feature_names != self.encoder.feature_names:
            raise ValueError("behavior-cloning model uses an incompatible feature schema")
        if (
            model.encoder_name != self.encoder.encoder_name
            or model.encoder_version != self.encoder.encoder_version
            or model.ruleset_id != kernel.ruleset_id
        ):
            raise ValueError("behavior-cloning model provenance is incompatible")
        self.scales = np.asarray(model.feature_scales, dtype=np.float64)
        self.weights = np.asarray(model.weights, dtype=np.float64)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if not request.capabilities.all_ready:
            return self._stop("challenger_capability_gate_failed")
        if request.ruleset_id != self.kernel.ruleset_id:
            return self._stop("challenger_ruleset_mismatch")
        try:
            request.validate()
            legal_actions = self.kernel.legal_actions(
                request.state,
                plan_type=request.plan_type,
                decision_kind=request.decision_kind,
            )
            candidates = []
            for action in legal_actions:
                features, branches, value = self.encoder.encode(request, action)
                if any(
                    branch.state.stamina < request.utility.minimum_stamina
                    or branch.state.genki < request.utility.minimum_genki
                    for branch in branches
                ):
                    continue
                learned_score = float((features / self.scales) @ self.weights)
                candidates.append((learned_score, action, value))
        except (ValueError, KernelError) as error:
            return self._stop(f"challenger_error:{error}")
        if not candidates:
            return self._stop("challenger_no_eligible_action")
        _score, action, value = min(
            candidates,
            key=lambda item: (
                -item[0],
                item[1].kind.value,
                item[1].card_instance_id or "",
            ),
        )
        trace_actions = tuple(
            CandidateValue(candidate[1], candidate[2], 1.0)
            for candidate in candidates
        )
        if action.kind in _SLOTTED_ACTIONS:
            slots = dict(request.card_slots)
            if action.card_instance_id not in slots:
                return self._stop("challenger_source_slot_missing")
            action = replace(action, source_slot=slots[action.card_instance_id])
        response = DecisionResponse(
            DecisionOutcome.DECIDE,
            "isolated_behavior_cloning_rank",
            action,
            value,
            DecisionTrace(
                planner=self.policy_name,
                planner_version=self.policy_version,
                ruleset_id=self.kernel.ruleset_id,
                completed_depth=1,
                expanded_nodes=len(candidates),
                elapsed_ms=0.0,
                candidate_values=trace_actions,
                reason_codes=("offline_challenger_no_input_authority",),
            ),
        )
        response.validate()
        return response

    def _stop(self, reason: str) -> DecisionResponse:
        return DecisionResponse(
            DecisionOutcome.STOP_TASK,
            reason,
            None,
            None,
            DecisionTrace(
                planner=self.policy_name,
                planner_version=self.policy_version,
                ruleset_id=self.kernel.ruleset_id,
                completed_depth=0,
                expanded_nodes=0,
                elapsed_ms=0.0,
                reason_codes=("offline_challenger_no_input_authority",),
            ),
        )
