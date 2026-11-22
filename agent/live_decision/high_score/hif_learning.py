"""Lightweight legal-candidate encoding and offline learned search advice."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .kernel import EffectKind
from .contracts import ActionKind, SchedulePhase
from .challenger import LinearBehaviorCloningModel, _effect_tree

_STATE_FIELDS = (
    ('turns_remaining', 12), ('stamina', 50), ('genki', 50), ('actions_remaining', 3),
    ('concentration', 30), ('good_condition_turns', 20), ('excellent_condition_turns', 10),
    ('double_card_effects_remaining', 2), ('nullify_cost_cards', 2), ('cost_reduction', 5),
    ('half_cost_turns', 5), ('score_multiplier', 40),
)
_EFFECTS = (
    (EffectKind.SCORE, 100), (EffectKind.CONCENTRATION, 10), (EffectKind.GOOD_CONDITION, 10),
    (EffectKind.EXCELLENT_CONDITION, 5), (EffectKind.GENKI, 20), (EffectKind.STAMINA, 10),
    (EffectKind.EXTRA_ACTION, 1), (EffectKind.DRAW, 3), (EffectKind.SCORE_BUFF, 1),
    (EffectKind.CONCENTRATION_EFFECT_BUFF, 1), (EffectKind.CONCENTRATION_ADDITION_BUFF, 1),
)


class HifFeatureEncoder:
    """Public counts and rule descriptors; no transition enumeration or RNG.

    Candidate/state interactions let a small linear scorer learn timing. Deck
    aggregates are invariant to the arbitrary order of an unknown draw pile.
    The encoder is an approximation, not a sufficient Markov-state claim.
    """

    encoder_name = 'task275-hif-public-candidate-features'
    encoder_version = '1.1'

    def __init__(self, kernel):
        self.kernel = kernel
        self.state_names = ('bias', *(name for name, _ in _STATE_FIELDS),
            'score_buff', 'future_multiplier_max', 'good_x_excellent', 'drinks_count',
            'draw_count', 'discard_count', 'removed_count', 'draw_score', 'draw_focus',
            'hand_count', 'known_prefix_count', 'pending_turn_phase')
        self.action_names = (*(f'action_{kind.value}' for kind in ActionKind),
            *(f'effect_{kind.value}' for kind, _ in _EFFECTS), 'stamina_cost', 'limited', 'upgraded')
        self.feature_names = tuple(f'{a}*{s}' for a in self.action_names for s in self.state_names)

    @lru_cache(maxsize=4096)
    def _definition(self, definition):
        effects = tuple(_effect_tree(definition))
        return tuple(sum(effect.amount * effect.repeats for effect in effects if effect.kind is kind) / scale
                     for kind, scale in _EFFECTS)

    def state_features(self, state):
        bag = [self._definition(self.kernel.cards[card.card_id]) for card in state.deck.draw_pile]
        values = [1., *(getattr(state, field) / scale for field, scale in _STATE_FIELDS),
            sum(buff.amount for buff in state.score_buffs),
            max(state.future_score_multipliers, default=state.score_multiplier) / 40,
            state.good_condition_turns / 20 * bool(state.excellent_condition_turns),
            len(state.drinks) / 4, len(state.deck.draw_pile) / 30,
            len(state.deck.discard) / 30, len(state.deck.removed) / 30,
            sum(row[0] for row in bag) / max(1, len(bag)),
            sum(row[1] for row in bag) / max(1, len(bag)),
            len(state.deck.hand) / 5, state.deck.known_top_count / 5, float(bool(state.resolution_stack))]
        return np.asarray(values, dtype=np.float64)

    def action_features(self, state, action):
        card = next((card for card in state.deck.all_cards if card.instance_id == action.card_instance_id), None)
        definition = (self.kernel.card_definition(state, card) if card is not None else
                      self.kernel.drinks.get(action.drink_id))
        effects = self._definition(definition) if definition is not None else (0.,) * len(_EFFECTS)
        cost = getattr(definition, 'stamina_cost', 0) / 10
        return np.asarray([*(float(action.kind is kind) for kind in ActionKind), *effects,
            cost, float(getattr(definition, 'limited', False)), float(card.upgraded if card else False)])

    def candidate_features(self, state, action):
        return np.outer(self.action_features(state, action), self.state_features(state)).reshape(-1)

    def encode(self, request, action):
        # Reuse the existing BC fitting/model contract. This encoder has no
        # one-step rollout or predicted final score to return.
        return self.candidate_features(request.state, action), (), None

    def matrix(self, state, actions):
        return np.asarray([self.candidate_features(state, action) for action in actions])


class HifFeatureEncoderV12(HifFeatureEncoder):
    """Add observed drink semantics without changing legacy feature values."""

    encoder_version = '1.2'
    extra_action_names = ('effect_MOVE_SELECTED_TO_HAND', 'effect_UPGRADE_HAND',
        'effect_NULLIFY_COST_CARDS', 'effect_DOUBLE_COST_TURNS', 'effect_HALF_COST_TURNS')
    extra_state_names = ('hand_upgradeable_count', 'hand_extra_action_total',
        'retrieval_drinks_count', 'extra_action_drinks_count', 'double_cost_turns',
        'future_multiplier_next', 'future_multiplier_second')

    def __init__(self, kernel):
        super().__init__(kernel)
        self.action_names = (*self.action_names, *self.extra_action_names)
        self.state_names = (*self.state_names, *self.extra_state_names)
        self.feature_names = tuple(f'{a}*{s}' for a in self.action_names for s in self.state_names)

    @lru_cache(maxsize=4096)
    def _drink_effects(self, definition):
        effects = tuple(_effect_tree(definition))
        return (
            sum(e.selection_count for e in effects if e.kind is EffectKind.MOVE_SELECTED_TO_HAND),
            sum(1 for e in effects if e.kind is EffectKind.UPGRADE_HAND),
            sum(e.amount for e in effects if e.kind is EffectKind.NULLIFY_COST_CARDS) / 2,
            sum(e.amount for e in effects if e.kind is EffectKind.DOUBLE_COST_TURNS) / 3,
            sum(e.amount for e in effects if e.kind is EffectKind.HALF_COST_TURNS) / 3,
        )

    def state_features(self, state):
        old = super().state_features(state)
        hand = [self.kernel.card_definition(state, card) for card in state.deck.hand]
        drinks = [self.kernel.drinks[drink] for drink in state.drinks
                  if not getattr(self.kernel.drinks[drink], 'reserved_for_later', False)]
        extra = (
            sum(definition.upgrade_card_id is not None for definition in hand) / 5,
            sum(e.amount for d in hand for e in _effect_tree(d)
                if e.kind is EffectKind.EXTRA_ACTION) / 5,
            sum(self._drink_effects(d)[0] > 0 for d in drinks) / 4,
            sum(any(e.kind is EffectKind.EXTRA_ACTION for e in _effect_tree(d)) for d in drinks) / 4,
            state.double_cost_turns / 3,
            state.future_score_multipliers[0] / 40 if state.future_score_multipliers else 0.,
            state.future_score_multipliers[1] / 40 if len(state.future_score_multipliers) > 1 else 0.,
        )
        return np.concatenate((old, extra))

    def action_features(self, state, action):
        old = super().action_features(state, action)
        card = next((c for c in state.deck.all_cards if c.instance_id == action.card_instance_id), None)
        definition = self.kernel.card_definition(state, card) if card is not None else self.kernel.drinks.get(action.drink_id)
        extra = self._drink_effects(definition) if definition is not None else (0.,) * len(self.extra_action_names)
        return np.concatenate((old, extra))


class HifFeatureEncoderV13(HifFeatureEncoderV12):
    """Expose immediate unconditional active-card score schedules, not a forecast.

    Aggregate fixed-rule amounts and public runtime limits; never encode source
    IDs or hidden draw order. Conditional/delayed schedules retain the legacy
    approximation and are outside these additional descriptors.
    """

    encoder_version = '1.3'
    schedule_fields = ('amount', 'finite_uses', 'unlimited_uses', 'finite_turns', 'unlimited_turns')

    def __init__(self, kernel):
        super().__init__(kernel)
        self.action_names = (*self.action_names, 'card_type_active', 'card_type_mental',
            *(f'creates_active_score_{name}' for name in self.schedule_fields))
        self.state_names = (*self.state_names,
            *(f'active_score_{name}' for name in self.schedule_fields))
        self.feature_names = tuple(f'{a}*{s}' for a in self.action_names for s in self.state_names)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _active_score_amount(effect):
        if (effect.kind is not EffectKind.SCHEDULE or effect.condition is not None
                or effect.schedule_phase is not SchedulePhase.CARD_USED
                or effect.trigger_card_type != 'active' or effect.delay_turns != 0
                or effect.schedule_counter_condition is not None):
            return 0.
        return sum(child.amount * child.repeats for child in effect.scheduled_effects
            if child.kind is EffectKind.SCORE and child.condition is None
            and child.value_source is None and child.full_power_scale == 0)

    @staticmethod
    def _schedule_features(amount, uses, turns):
        return np.asarray((amount / 100, (uses or 0) / 4, float(uses is None),
                           (turns or 0) / 12, float(turns is None)))

    def state_features(self, state):
        extra = np.zeros(len(self.schedule_fields))
        for ref in state.scheduled_effects:
            if (ref.trigger_phase is not SchedulePhase.CARD_USED or ref.turns_until_fire != 0
                    or ref.remaining_uses == 0 or ref.remaining_turns == 0):
                continue
            amount = self._active_score_amount(self.kernel._scheduled_effect_spec(ref))
            if amount:
                extra += self._schedule_features(amount, ref.remaining_uses, ref.remaining_turns)
        return np.concatenate((super().state_features(state), extra))

    def action_features(self, state, action):
        card = next((c for c in state.deck.all_cards if c.instance_id == action.card_instance_id), None)
        definition = self.kernel.card_definition(state, card) if card is not None else self.kernel.drinks.get(action.drink_id)
        extra = np.zeros(len(self.schedule_fields))
        for effect in getattr(definition, 'effects', ()):
            amount = self._active_score_amount(effect)
            if amount:
                extra += self._schedule_features(amount, effect.schedule_limit, effect.schedule_ttl)
        card_type = getattr(definition, 'card_type', None)
        return np.concatenate((super().action_features(state, action),
            (float(card_type == 'active'), float(card_type == 'mental')), extra))


class HifFeatureEncoderV14(HifFeatureEncoderV13):
    """Describe direct scoring at current public resources, without a rollout.

    This is not a next-state or final-score prediction: preceding effects,
    conditional effects and future triggers are deliberately not simulated.
    """

    encoder_version = '1.4'

    def __init__(self, kernel):
        super().__init__(kernel)
        self.action_names = (*self.action_names, 'direct_score_at_current_resources',
            'direct_score_focus_weight', 'direct_score_repeat_count')
        self.feature_names = tuple(f'{a}*{s}' for a in self.action_names for s in self.state_names)

    def action_features(self, state, action):
        card = next((c for c in state.deck.all_cards if c.instance_id == action.card_instance_id), None)
        definition = self.kernel.card_definition(state, card) if card is not None else self.kernel.drinks.get(action.drink_id)
        effects = tuple(effect for effect in getattr(definition, 'effects', ())
            if effect.kind is EffectKind.SCORE and effect.condition is None)
        current_gain = sum(self.kernel.score_effect_gain(state, effect, action.card_instance_id)
            * effect.repeats for effect in effects)
        return np.concatenate((super().action_features(state, action), (
            current_gain / 10000,
            sum(effect.score_concentration_multiplier * effect.repeats for effect in effects) / 4,
            sum(effect.repeats for effect in effects) / 4)))


def hif_feature_encoder(kernel, version='1.1'):
    """Model metadata selects its exact encoder; callers keep the 1.1 default."""
    encoders = {'1.1': HifFeatureEncoder, '1.2': HifFeatureEncoderV12,
        '1.3': HifFeatureEncoderV13, '1.4': HifFeatureEncoderV14}
    if version not in encoders:
        raise ValueError(f'unsupported HIF candidate encoder version: {version}')
    return encoders[version](kernel)


class HifLearningAdvice:
    """Offline model adapter. Values are estimates and never pruning bounds."""

    def __init__(self, kernel, model: LinearBehaviorCloningModel, *, value_weights=None):
        model.validate()
        self.encoder = hif_feature_encoder(kernel, model.encoder_version)
        if (model.ruleset_id != kernel.ruleset_id or model.feature_names != self.encoder.feature_names
                or model.encoder_name != self.encoder.encoder_name
                or model.encoder_version != self.encoder.encoder_version):
            raise ValueError('HIF model does not match the rules or candidate encoder')
        self.model_version = model.model_version
        self.weights = np.asarray(model.weights) / np.asarray(model.feature_scales)
        self.value_weights = None if value_weights is None else np.asarray(value_weights, dtype=np.float64)
        if self.value_weights is not None and (self.value_weights.shape != (len(self.encoder.state_names),)
                                                or not np.isfinite(self.value_weights).all()):
            raise ValueError('HIF value model has invalid dimensions or values')

    def action_priors(self, state, actions):
        # Factor the same outer-product linear model. State encoding is shared
        # across candidates; no model or candidate semantics change.
        descriptors = np.asarray([self.encoder.action_features(state, action) for action in actions])
        state_features = self.encoder.state_features(state)
        scores = descriptors @ (self.weights.reshape((-1, len(state_features))) @ state_features)
        if not np.isfinite(scores).all():
            raise ValueError('HIF model returned nonfinite action priors')
        return tuple(float(score) for score in scores)

    def future_score(self, state):
        if self.value_weights is None:
            return None
        return float(state.score + self.encoder.state_features(state) @ self.value_weights)
