"""Known current-hand support only; future/re-drawn support uses D-047.

No frame, input authority, future random support or permanent upgrade is stored
here. A definition captured by a suspended card stays on its existing frame.
"""
from __future__ import annotations

from .kernel import RuleEvent, EffectKind, PythonNiaProKernel
from .contracts import EffectSourceKind, VerifiedHandSupport
from .state_copy import replace


def clean_hand_support(state):
    if not state.hand_support:
        return state
    held = {(card.instance_id, card.card_id) for card in state.deck.hand}
    remaining = tuple(fact for fact in state.hand_support
                      if fact.turn == state.turn and not state.terminal
                      and (fact.instance_id, fact.card_id) in held)
    return state if remaining == state.hand_support else replace(state, hand_support=remaining)


def ordinary_plus_definition(cards, ordinary):
    """Fixed single-plus estimate, retaining the physical card's base identity.

    This does not prove random support magnitudes. Persistent deck abilities
    cannot be replaced by a current-hand observation. Direct secondary effects
    still use the old source-card continuation and are explicitly unsupported.
    """
    upgraded = cards.get(ordinary.upgrade_card_id) if ordinary is not None else None
    if (upgraded is None or upgraded.base_card_id != ordinary.base_card_id
            or upgraded.plan != ordinary.plan
            or ordinary.canonical_card_id not in (None, ordinary.card_id)
            or upgraded.persistent_effects != ordinary.persistent_effects):
        raise ValueError('ordinary_plus_estimate_incomplete: catalog upgrade relation or persistent effects')
    pending_kinds = {EffectKind.HOLD_SELECTED, EffectKind.MOVE_SELECTED_TO_HAND,
                     EffectKind.FREE_PLAY_SELECTED, EffectKind.PAID_PLAY_SELECTED,
                     EffectKind.FREE_PLAY_ALL}
    def requires_source_continuation(effects):
        return any(effect.kind in pending_kinds
                   or requires_source_continuation(effect.scheduled_effects) for effect in effects)
    if requires_source_continuation(upgraded.effects):
        raise ValueError('ordinary_plus_estimate_incomplete: secondary source continuation')
    return replace(upgraded, card_id=ordinary.card_id,
                   canonical_card_id=ordinary.canonical_card_id,
                   upgrade_card_id=ordinary.upgrade_card_id)


def uses_hand_support_for_strategy(fact: VerifiedHandSupport) -> bool:
    """Value the permanent 816+ even when its temporary printed cost was observed."""
    return fact.card_id != '816'


class HifSupportPlanningMixin:
    def _supported_definition(self, ordinary, tier):
        if ordinary is None or self.support_definition is None:
            raise ValueError('verified_support_catalog_not_available_offline')
        supported = self.support_definition(ordinary, tier)
        if (supported.card_id, supported.plan) != (ordinary.card_id, ordinary.plan):
            raise ValueError('verified_support_definition_identity_changed')
        supported.validate()
        return supported

    def card_definition(self, state, card):
        ordinary = self.cards.get(card.card_id)
        if not state.hand_support or card not in state.deck.hand:
            return ordinary
        for fact in state.hand_support:
            if (fact.instance_id, fact.card_id, fact.turn) == (card.instance_id, card.card_id, state.turn):
                if fact.projection_kind == 'ordinary_plus_estimate':
                    return ordinary_plus_definition(self.cards, ordinary)
                return self._supported_definition(ordinary, fact.enhancement_count)
        return ordinary

    def _card_resolution_support_tier(self, definition):
        ordinary = self.cards.get(definition.card_id)
        if definition == ordinary:
            return 0
        if self._card_resolution_projection_kind(definition) == 'ordinary_plus_estimate':
            return 1
        for tier in (1, 2, 3):
            try:
                if self._supported_definition(ordinary, tier) == definition:
                    return tier
            except ValueError:
                continue
        raise ValueError('card_resolution_definition_is_not_a_verified_support_tier')

    def _card_resolution_projection_kind(self, definition):
        ordinary = self.cards.get(definition.card_id)
        if definition != ordinary and ordinary is not None and ordinary.upgrade_card_id:
            if definition == ordinary_plus_definition(self.cards, ordinary):
                return 'ordinary_plus_estimate'
        return 'verified_effects'

    def _resolving_card_definition(self, frame):
        ordinary = self.cards.get(frame.card_id)
        if frame.projection_kind == 'ordinary_plus_estimate':
            return ordinary_plus_definition(self.cards, ordinary)
        if not frame.support_tier:
            return ordinary
        return self._supported_definition(ordinary, frame.support_tier)

    @staticmethod
    def _canonical_probability_state(state):
        return PythonNiaProKernel._canonical_probability_state(clean_hand_support(state))

    def transition(self, state, action, *, plan_type):
        original = state.hand_support
        branches = super().transition(clean_hand_support(state), action, plan_type=plan_type)
        if not original:
            return branches
        result = []
        for branch in branches:
            after = clean_hand_support(branch.state)
            ended = tuple(fact for fact in original if fact not in after.hand_support)
            events = tuple(RuleEvent('HAND_SUPPORT_BINDING_ENDED', fact.instance_id,
                detail=('turn_expired' if after.turn != fact.turn or after.terminal
                        else 'left_hand_future_use_d047_approximation')) for fact in ended)
            result.append(replace(branch, state=after, events=branch.events + events))
        return tuple(result)

    def _trigger_event_phase(self, branches, phase, **kwargs):
        # This boundary is reached after a physical card leaves the hand and
        # before CARD_USED can draw it again. condition_state stays unchanged.
        branches = tuple(replace(branch, state=clean_hand_support(branch.state))
                         if branch.state.hand_support else branch for branch in branches)
        return super()._trigger_event_phase(branches, phase, **kwargs)

    @staticmethod
    def _move_card_to_hand(state, instance_id):
        # An already absent entity cannot revive its previous hand binding.
        return PythonNiaProKernel._move_card_to_hand(clean_hand_support(state), instance_id)

    @staticmethod
    def _move_card_to_held(state, instance_id):
        moved, evicted = PythonNiaProKernel._move_card_to_held(state, instance_id)
        return clean_hand_support(moved), evicted

    @staticmethod
    def _move_card_for_use(state, card, definition):
        moved, event = PythonNiaProKernel._move_card_for_use(state, card, definition)
        return clean_hand_support(moved), event

    @staticmethod
    def _move_card_to_discard(state, card):
        moved, event = PythonNiaProKernel._move_card_to_discard(state, card)
        return clean_hand_support(moved), event

    def _support_schedule_kwargs(self, effect, kwargs):
        # Reuse the already verified 597 derived schedule, not a global card
        # replacement. Its four created triggers outlive the hand binding.
        if effect.kind is not EffectKind.SCHEDULE:
            return kwargs
        ordinary = self.cards.get(kwargs.get('source_card_id'))
        index = kwargs.get('definition_effect_index')
        if (ordinary is not None and ordinary.upgrade_card_id and type(index) is int
                and (index >= len(ordinary.effects) or ordinary.effects[index] != effect)):
            upgraded = ordinary_plus_definition(self.cards, ordinary)
            if 0 <= index < len(upgraded.effects) and upgraded.effects[index] == effect:
                # Like the existing derived-support index, this references the
                # fixed upgraded catalog effects under the same physical base.
                return dict(kwargs, definition_effect_index=len(ordinary.effects) + index)
        from .support_schedule import (
            SUPPORTED_CARD_IDS,
            ordinary_schedule_index,
            supported_schedule_card,
            supported_schedule_index,
        )

        card_id = kwargs.get('source_card_id')
        if (card_id not in SUPPORTED_CARD_IDS
                or kwargs.get('definition_effect_index') != ordinary_schedule_index(card_id)):
            return kwargs
        ordinary = self.cards.get(card_id)
        if ordinary is None:
            return kwargs
        try:
            supported = supported_schedule_card(ordinary)
        except ValueError:
            return kwargs
        if effect != supported.effects[ordinary_schedule_index(card_id)]:
            return kwargs
        return dict(kwargs, definition_effect_index=supported_schedule_index(card_id))

    def _scheduled_definition_effects(self, scheduled):
        ordinary = self.cards.get(scheduled.card_id)
        if (scheduled.source_kind is EffectSourceKind.CARD and not scheduled.persistent
                and ordinary is not None and ordinary.upgrade_card_id
                and scheduled.definition_effect_index >= len(ordinary.effects)):
            upgraded = ordinary_plus_definition(self.cards, ordinary)
            return (*ordinary.effects, *upgraded.effects)
        return super()._scheduled_definition_effects(scheduled)
