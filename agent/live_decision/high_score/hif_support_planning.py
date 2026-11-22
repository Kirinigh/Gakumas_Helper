"""Known current-hand support only; future/re-drawn support uses D-047.

No frame, input authority, future random support or permanent upgrade is stored
here. A definition captured by a suspended card stays on its existing frame.
"""
from __future__ import annotations

import json
import math

from .kernel import RuleEvent, EffectKind, PythonNiaProKernel
from .contracts import EffectSourceKind, VerifiedHandSupport
from .state_copy import replace


def replay_visible_hand_projection(bundle, event):
    """Recompute a saved current-hand estimate from complete title evidence."""
    from dataclasses import asdict

    from .contracts import _jsonable
    from ..contracts import LiveState
    from .live_policy import BoundHifOpeningPolicy
    from .hif_mcts_cache import exact_state

    def same(left, right):
        def canonical(value):
            if isinstance(value, dict) and 'bound_state' in value:
                value = dict(value, bound_state=asdict(exact_state(value['bound_state'])))
            return _jsonable(value)

        return json.dumps(canonical(left), sort_keys=True, allow_nan=False) == json.dumps(
            canonical(right), sort_keys=True, allow_nan=False)

    if (not isinstance(event, dict) or event.get('source') != 'hif_current_turn_ordinary_plus_v1'
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)
            or exact_state(event.get('bound_state')) != bundle.state
            or not same(event.get('inventory_keys'), bundle.inventory_keys)):
        raise ValueError('ordinary_plus_projection_source_changed')
    policy = BoundHifOpeningPolicy(bundle, action_order=())
    restored = policy.bind_visible_hand(LiveState.from_dict(event.get('visible_state')),
        identity_events=event.get('identity_events', ()))
    if policy.ordinary_plus_projection is None or not same(event, policy.ordinary_plus_projection):
        raise ValueError('ordinary_plus_projection_evidence_changed')
    return restored


def observed_plus_hand(bundle, live, identity_events):
    """Replay the observer's existing title/instance evidence, never its effects.

    Retained hand readers deliberately report the permanent ID. Their complete
    blue-title receipt can supply a current ordinary-plus estimate separately.
    No pixel acquisition, selection or new input authority occurs here.
    """
    relevant = {'initial_observation', 'hif_complete_title_identity',
                'retained_complete_hand_identity', 'hif_temporary_support_identity'}
    events = tuple(event for event in identity_events if event.get('phase') in relevant)
    initial = [event for event in events if event.get('phase') == 'initial_observation']
    proofs = [event for event in events if event.get('phase') == 'hif_temporary_support_identity']
    changed = [slot for slot, (observed, ref) in enumerate(zip(live.hand, bundle.state.deck.hand, strict=True))
               if observed.card_id == bundle.kernel.cards[ref.card_id].upgrade_card_id and not ref.upgraded]
    if not proofs and not changed:
        return live, events
    if len(initial) != 1:
        raise ValueError('ordinary_plus_estimate_requires_original_identity_batch')
    origin = initial[0]
    frame_ids = origin.get('frame_ids', ())
    boxes = origin.get('candidate_boxes', ())
    if (origin.get('geometry_stable') is not True or origin.get('identity_source_accepted') is not True
            or len(frame_ids) != 3 or len(set(frame_ids)) != 3
            or any(not isinstance(value, str) or not value for value in frame_ids)
            or len(boxes) != 3 or any(len(frame) != len(live.hand) for frame in boxes)):
        raise ValueError('ordinary_plus_estimate_identity_batch_incomplete')
    hand = list(live.hand)
    seen = set()
    for proof in proofs:
        slot = proof.get('slot')
        if type(slot) is not int or not 0 <= slot < len(hand) or slot in seen:
            raise ValueError('ordinary_plus_estimate_identity_slot_ambiguous')
        seen.add(slot)
        ref = bundle.state.deck.hand[slot]
        ordinary = bundle.kernel.cards[ref.card_id]
        visible = bundle.kernel.cards.get(ordinary.upgrade_card_id)
        if (proof.get('source') != 'bound_rightmost_blue_support_identity'
                or (proof.get('run_id'), proof.get('revision'), proof.get('observed_turn'))
                    != (bundle.run_id, bundle.revision, bundle.state.turn)
                or (proof.get('card_instance_id'), proof.get('card_id')) != (ref.instance_id, ref.card_id)
                or ref.upgraded or (ref.instance_id, ref.card_id, ()) not in bundle.inventory_keys
                or visible is None or proof.get('visible_card_id') != visible.card_id
                or visible.base_card_id != ordinary.base_card_id or visible.name != ordinary.name+'+'
                or proof.get('effects_verified') is not False or proof.get('submission_authorized') is not False
                or hand[slot].card_id not in (ref.card_id, visible.card_id)
                or not hand[slot].identity_accepted):
            raise ValueError('ordinary_plus_estimate_identity_source_changed')
        observations = proof.get('observations', ())
        if len(observations) != 3 or [row.get('frame_id') for row in observations] != list(frame_ids):
            raise ValueError('ordinary_plus_estimate_title_frames_changed')
        for index, row in enumerate(observations):
            title = row.get('title', {})
            box, plate = title.get('box', ()), row.get('title_plate', ())
            card_box = boxes[index][slot]
            confidence = title.get('score')
            if (row.get('slot') != slot or title.get('text') != visible.name
                    or type(confidence) not in (int, float) or not math.isfinite(confidence)
                    or not .95 <= confidence <= 1
                    or any(len(value) != 4 or any(type(number) is not int for number in value)
                           or min(value[2:]) <= 0 for value in (box, plate, card_box))
                    or not card_box[0] <= box[0] < box[0]+box[2] <= card_box[0]+card_box[2]
                    or not 1090 <= box[1] <= 1120 or not 14 <= box[3] <= 32
                    or not plate[0]-5 <= box[0] < box[0]+box[2] <= plate[0]+plate[2]+5
                    or row.get('suffix_color') != 'blue'
                    or row.get('marker', {}).get('upgraded') is not True):
                raise ValueError('ordinary_plus_estimate_complete_title_changed')
        hand[slot] = replace(hand[slot], card_id=visible.card_id, upgraded=True)
    for slot in set(changed)-seen:
        ref = bundle.state.deck.hand[slot]
        # Existing opening receipts independently retain their complete title.
        opening = [proof for proof in bundle.opening_skill_support
                   if proof.get('card_instance_id') == ref.instance_id]
        full = [event for event in events if event.get('phase') == 'hif_complete_title_identity'
                and event.get('accepted') is True and event.get('frame_ids') == list(frame_ids)
                and len(event.get('card_ids', ())) == len(hand)
                and event['card_ids'][slot] == hand[slot].card_id]
        if not opening and len(full) != 1:
            raise ValueError('ordinary_plus_estimate_requires_complete_current_title')
    return replace(live, hand=tuple(hand)), events


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


def verified_hand_support_fact(bundle, receipt):
    """Recheck the existing complete receipt for its original held instance.

    The native producer/transaction wiring is intentionally outside this slice.
    Identity-only opening receipts cannot become complete-effect facts.
    """
    from .skill_support import supported_definition
    from ..hif_skill_support import verify_selected_skill_support_observations

    if (not isinstance(receipt, dict) or receipt.get('accepted') is not True
            or receipt.get('effects_verified') is not True
            or (receipt.get('run_id'), receipt.get('revision')) != (bundle.run_id, bundle.revision)):
        raise ValueError('hand_support_requires_complete_current_effect_receipt')
    slot = receipt.get('selected_slot')
    if type(slot) is not int or not 0 <= slot < len(bundle.state.deck.hand):
        raise ValueError('hand_support_requires_one_bound_instance')
    card = bundle.state.deck.hand[slot]
    if card.instance_id != receipt.get('card_instance_id'):
        raise ValueError('hand_support_instance_changed')
    checked = verify_selected_skill_support_observations(
        receipt.get('observations', ()), bundle=bundle, selected_slot=slot,
        selection_evidence=receipt.get('selection_evidence', {}))
    if checked.get('accepted') is not True or json.dumps(checked, sort_keys=True) != json.dumps(receipt, sort_keys=True):
        raise ValueError('hand_support_complete_effect_receipt_changed')
    definition = supported_definition(bundle.kernel.cards[card.card_id],
                                      enhancement_count=checked.get('enhancement_count'))
    if (definition.stamina_cost != checked.get('printed_cost')
            or definition.cost_kind.value != checked.get('cost_kind')):
        raise ValueError('hand_support_printed_cost_changed')
    return VerifiedHandSupport(card.instance_id, card.card_id, bundle.state.turn,
                               checked['enhancement_count'])


class HifSupportPlanningMixin:
    def _supported_definition(self, ordinary, tier):
        if self.support_definition is None:
            from .skill_support import supported_definition

            return supported_definition(ordinary, enhancement_count=tier)
        if ordinary is None:
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
