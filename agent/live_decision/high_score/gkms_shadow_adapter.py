"""Offline UIObs projection for caller-supplied Sense states and mappings.

Only an already bound request and verified, caller-supplied mappings can
produce supported input. The caller retains ownership of its state and data.
"""
from __future__ import annotations

import math
from typing import Any, Sequence
from collections import Counter
from dataclasses import asdict, replace, dataclass

from .kernel import RuleEvent, KernelError
from .contracts import (
    Stance,
    CardRef,
    PlanType,
    ActionKind,
    GameAction,
    DecisionKind,
    DeckKnowledge,
    StrategyState,
    DecisionRequest,
    ScheduledEffectRef,
    HighScoreContractError,
)
from .gkms_shadow_mappings import ShadowMappings


class ShadowMappingError(ValueError):
    """A specific missing mapping or invalid/stale proposal, never an END."""


def _require(condition, reason):
    if not condition:
        raise ShadowMappingError(reason)


@dataclass(frozen=True)
class ExamContext:
    """Observed whole-round metadata; the state supplies current resources."""

    produce: str
    step: str
    plan: int
    turn_types: tuple[int, ...]
    permil: tuple[int, int, int]
    criteria: tuple[int, int, int]
    idol: str
    source: str
    mappings: ShadowMappings | None = None

    def observation(self, state: StrategyState):
        _require(self.produce in {'produce-007', 'produce-008'}
                 and self.step in {'AUDITION_MID1', 'AUDITION_MID2', 'AUDITION_FINAL'}
                 and self.plan == 2 and bool(self.source), 'exam_context_unverified')
        _require(self.turn_types and all(type(x) is int and x in (1, 2, 3) for x in self.turn_types),
                 'exam_turn_types_invalid')
        _require(len(self.permil) == len(self.criteria) == 3
                 and all(type(x) is int and x > 0 for x in self.permil + self.criteria),
                 'exam_multipliers_or_criteria_invalid')
        _require(state.turn + state.turns_remaining - 1 == len(self.turn_types),
                 'exam_limit_or_extra_turn_unmapped')
        expected = self.turn_types[state.turn - 1:]
        names = {1: 'vocal', 2: 'dance', 3: 'visual'}
        _require(tuple(names[x] for x in expected) == state.turn_attributes,
                 'exam_state_turn_attributes_mismatch')
        multipliers = tuple(self.permil[x - 1] / 1000 for x in expected)
        actual = (state.score_multiplier, *state.future_score_multipliers)
        _require(len(actual) == len(multipliers)
                 and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                         for a, b in zip(actual, multipliers)), 'exam_state_multipliers_mismatch')
        return dict(produce=self.produce, step=self.step, plan=self.plan, idol=self.idol,
                    turn=state.turn, limit=len(self.turn_types), types=list(self.turn_types),
                    permil=list(self.permil), crit=list(self.criteria))


@dataclass(frozen=True)
class TurnPlayHistory:
    """Complete ordered history tied to exactly one immutable local state.

At a saved opening construct this with played=() only after confirming the
accepted opening had zero submitted cards. For a later saved root supply the
complete accepted chain; actions_remaining is never used to infer plays.
"""

    state: StrategyState
    played: tuple[CardRef, ...]
    source: str
    played_support: tuple[int | None, ...] = ()
    expired_start_effects: tuple[ScheduledEffectRef, ...] = ()

    def validate(self, state):
        _require(bool(self.source), 'played_history_source_missing')
        _require(self.state == state, 'played_history_state_mismatch')
        owned = {card.instance_id for card in state.deck.all_cards}
        _require(all(card.instance_id in owned for card in self.played), 'played_history_unknown_instance')
        _require(not self.played_support or len(self.played_support) == len(self.played),
                 'played_history_support_length_mismatch')

    def advance(self, action: GameAction, after: StrategyState, events: Sequence[RuleEvent], *, card_upgrades=None):
        """Consume complete HifStep events, including turn-start Encore replays.

        The local kernel emits phase-qualified TRIGGER_SCHEDULED before a replay;
        a new-turn phase or system refill distinguishes it from end-turn effects.
        Created cards and explicit upgrades are reconstructed in event order.
        Growth with an intermediate use but no per-event magnitude is rejected;
        it does not occur in either frozen inventory and is never filled from
        a later card version. Multiple effect passes count one physical play.
        """
        self.validate(self.state)
        action.validate()
        after.validate()
        card_upgrades = card_upgrades if card_upgrades is not None else {}
        markers = {'CONSUME_ACTION', 'FREE_PLAY', 'PAID_PLAY', 'FREE_PLAY_RANDOM'}
        if action.kind is ActionKind.PLAY_CARD:
            _require(sum(event.event_type == 'CONSUME_ACTION'
                         and event.source_id == action.card_instance_id for event in events) == 1,
                     'played_history_action_event_missing_or_duplicate')
        _require(after.turn in {self.state.turn, self.state.turn + 1}, 'played_history_turn_gap')
        for kind, marker in ((ActionKind.FREE_PLAY, 'FREE_PLAY'), (ActionKind.PAID_PLAY, 'PAID_PLAY')):
            if action.kind is kind:
                allowed = {marker, 'PAID_PLAY_UNUSABLE_DISCARD'} if kind is ActionKind.PAID_PLAY else {marker}
                _require(sum(event.event_type in allowed and event.source_id == action.card_instance_id
                             for event in events) == 1, 'played_history_secondary_play_event_missing')
        ledger = {card.instance_id: card for card in self.state.deck.all_cards}
        after_cards = {card.instance_id: card for card in after.deck.all_cards}
        played = list(self.played)
        support = list(self.played_support or (None,) * len(played))
        active_support = {fact.instance_id: fact for fact in self.state.hand_support if fact.turn == self.state.turn}
        last_modification = {}
        for index, event in enumerate(events):
            if event.event_type == 'CARD_MODIFICATION':
                for target in (event.detail or '').split('|'):
                    last_modification[target] = index
        unknown_versions = set()
        advanced = after.turn != self.state.turn
        ended = False
        entered = False
        phase = None
        new_phases = {'BEFORE_START_OF_TURN', 'START_OF_TURN', 'AFTER_START_OF_TURN', 'TURN', 'EVERY_TURN'}
        for index, event in enumerate(events):
            if event.event_type == 'END_TURN':
                _require(not ended, 'played_history_multiple_turn_boundaries')
                ended, phase = True, None
            if event.event_type == 'TRIGGER_SCHEDULED':
                phase = (event.detail or '').rsplit(':', 1)[-1]
            start = (event.event_type == 'TRIGGER_SCHEDULED' and phase in new_phases
                     or event.event_type == 'DRAW' and event.source_id == 'system')
            if advanced and ended and not entered and start:
                played.clear()
                support.clear()
                active_support.clear()
                entered = True
            if event.event_type == 'ADD_CARD_TO_DECK':
                cid, separator, instance_id = (event.detail or '').partition(':')
                instance_id = instance_id.split(':position:', 1)[0]
                _require(separator and instance_id not in ledger and instance_id in after_cards,
                         'played_history_created_card_event_invalid')
                # This is the exact constructor in kernel._add_card_to_deck.
                ledger[instance_id] = CardRef(instance_id, cid)
            elif event.event_type == 'UPGRADE_HAND':
                for target in (event.detail or '').split('|'):
                    _require(target in ledger and ledger[target].card_id in card_upgrades,
                             'played_history_upgrade_unmapped')
                    ledger[target] = replace(ledger[target], card_id=card_upgrades[ledger[target].card_id], upgraded=True)
            elif event.event_type == 'CARD_MODIFICATION':
                for target in (event.detail or '').split('|'):
                    _require(target in ledger and target in after_cards, 'played_history_growth_instance_missing')
                    if last_modification[target] != index:
                        unknown_versions.add(target)
                    else:
                        # All numerical changes have happened by this point;
                        # retain the separately reconstructed upgrade identity.
                        values = {key: getattr(after_cards[target], key) for key in _GROWTH_FIELDS}
                        ledger[target] = replace(ledger[target], **values)
                        unknown_versions.discard(target)
            if event.event_type in markers:
                _require(not (advanced and ended and not entered and phase is None),
                         'played_history_turn_boundary_ambiguous')
                instance_id = event.source_id
                _require(instance_id in ledger and instance_id not in unknown_versions,
                         f'played_history_card_version_unverified:{instance_id}')
                card = ledger[instance_id]
                played.append(card)
                fact = active_support.pop(instance_id, None)
                support.append(fact.enhancement_count if fact and fact.card_id == card.card_id else None)
        if advanced:
            _require(ended, 'played_history_turn_event_missing')
            if not entered:
                played.clear()
                support.clear()
        _require(ledger == after_cards, 'played_history_unlogged_card_mutation')
        residues = list(self.expired_start_effects if not advanced else ())
        active = Counter((ref.source_id, ref.card_id, ref.definition_effect_index)
                         for ref in after.scheduled_effects if ref.remaining_uses != 0)
        triggered = Counter((e.source_id, e.detail) for e in events if e.event_type == 'TRIGGER_SCHEDULED')
        for ref in self.state.scheduled_effects:
            key = ref.source_id, ref.card_id, ref.definition_effect_index
            detail = f'{ref.card_id}:{ref.definition_effect_index}:START_OF_TURN'
            if (advanced and ref.trigger_phase.value == 'START_OF_TURN' and not ref.persistent
                    and ref.remaining_uses == 1 and triggered[ref.source_id, detail] and not active[key]):
                residues.append(replace(ref, remaining_uses=0, last_fired_turn=after.turn))
                triggered[ref.source_id, detail] -= 1
        return TurnPlayHistory(after, tuple(played), self.source, tuple(support), tuple(residues))

    def advance_step(self, action: GameAction, step, *, card_upgrades=None):
        """Convenience consumer of the existing HifStep; it does not re-simulate."""
        return self.advance(action, step.request.state, step.events, card_upgrades=card_upgrades)


_GROWTH_FIELDS = ('score_bonus', 'genki_bonus', 'stamina_cost_delta', 'typed_cost_delta', 'score_repeat_bonus')


def _compression(evidence, field, value, reason, source):
    if evidence is not None:
        evidence.append(dict(local_field=field, value=value, reason=reason, source=source))


def _grow_ids(kind, amount, mappings):
    """Exact numerical decomposition into real fixed-master growth records."""
    _require(type(amount) is int and amount >= 0, f'card_growth_integer_required:{kind}')
    _require(kind in mappings.grow_effects, f'card_growth_unmapped:{kind}')
    result = []
    for value, identifier in reversed(mappings.grow_effects[kind]):
        count, amount = divmod(amount, value)
        result.extend([identifier] * count)
    _require(amount == 0, f'card_growth_not_representable:{kind}:{amount}')
    return result


def _card(card: CardRef, mappings, *, support_count=None, evidence=None):
    _require(card.card_id in mappings.card_forms, f'card_form_unmapped:{card.card_id}')
    identifier, upgrade, cost_kind = mappings.card_forms[card.card_id]
    _require(type(card.upgraded) is bool and card.upgraded == bool(upgrade),
             f'card_upgrade_unverified:{card.card_id}')
    values = {name: getattr(card, name) for name in _GROWTH_FIELDS}
    extra = {}
    if support_count is not None:
        if upgrade == 0 and support_count == 1 and card.card_id in mappings.card_upgrades:
            upgraded_id = mappings.card_upgrades[card.card_id]
            _require(upgraded_id in mappings.card_forms, f'card_form_unmapped:{upgraded_id}')
            identifier, upgrade, cost_kind = mappings.card_forms[upgraded_id]
        else:
            _require(upgrade == 1 and support_count == 2 and card.card_id in mappings.support_deltas,
                     f'hand_support_tier_unmapped:{card.card_id}:{support_count}')
            upgrade = support_count
            for name, amount in mappings.support_deltas[card.card_id].items():
                if name in values:
                    values[name] += amount
                else:
                    extra[name] = amount
        _compression(evidence, 'hand_support', dict(instance_id=card.instance_id, enhancement_count=support_count),
                     'Known local support uses the explicit tier plus equivalent fixed growth records; '
                     'the public featurizer falls back to the highest available master tier.',
                     'skill_support.supported_definition; GKMS feat.py:_resolved')
    result = {'id': identifier, 'u': upgrade}
    if card.card_id in mappings.custom_forms:
        result['c'] = [list(pair) for pair in mappings.custom_forms[card.card_id]]
    grows = []
    for field, positive, negative in (('score_bonus', 'LessonAdd', 'LessonReduce'),
                                    ('genki_bonus', 'BlockAdd', 'BlockReduce'),
                                    ('score_repeat_bonus', 'LessonCountAdd', None)):
        amount = values[field]
        _require(type(amount) is int, f'card_growth_integer_required:{field}')
        if amount:
            _require(amount > 0 or negative is not None, f'card_growth_sign_invalid:{field}')
            grows += _grow_ids(positive if amount > 0 else negative, abs(amount), mappings)
    generic = values['stamina_cost_delta']
    _require(type(generic) is int, 'card_growth_integer_required:stamina_cost_delta')
    if generic and cost_kind != 'NORMAL':
        _compression(evidence, 'card.stamina_cost_delta', dict(instance_id=card.instance_id, value=generic),
                     'This generic delta is inert for the card typed cost in the local kernel.',
                     'kernel.py:_effective_stamina_cost')
        generic = 0
    typed = values['typed_cost_delta']
    _require(type(typed) is int, 'card_growth_integer_required:typed_cost_delta')
    prefix = {'NORMAL': 'Cost', 'STAMINA': 'CostPenetrate', 'CONCENTRATION': 'CostLessonBuff',
              'GOOD_CONDITION': 'CostParameterBuff', 'GOOD_IMPRESSION': 'CostReview',
              'MOTIVATION': 'CostAggressive', 'FULL_POWER': 'CostFullPowerPoint'}[cost_kind]
    for name, amount in (('Cost', generic), (prefix, typed)):
        if amount:
            grows += _grow_ids(name + ('Add' if amount > 0 else 'Reduce'), abs(amount), mappings)
    for kind, amount in extra.items():
        grows += _grow_ids(kind, amount, mappings)
    if grows:
        result['g'] = grows
        _compression(evidence, 'card.dynamic_growth', dict(instance_id=card.instance_id, deltas=values, grow_ids=grows),
                     'Numeric semantics use real master records. StrategyState stores aggregate deltas, '
                     'not original growth-event segmentation; the external ngrow feature may therefore '
                     'differ. Negative/nested growth also follows the public featurizer limitations.',
                     'kernel.py:_modify_cards; GKMS customize.py and feat.py:_card')
    return result


def _enchant(ref, state, evidence, mappings, *, kernel=None):
    key = ref.source_kind.value, ref.card_id, ref.definition_effect_index, ref.persistent
    _require(key in mappings.schedules, f'enchant_definition_unmapped:{key}')
    identifier, mode, phase, delay, uses, ttl, raw_turns, installer = mappings.schedules[key]
    _require(ref.trigger_phase.value == phase and 0 <= ref.turns_until_fire <= delay
             and ref.direct_on_trigger == (phase == 'TURN' and uses == 1),
             f'enchant_trigger_unverified:{key}')
    _require((uses is None and ref.remaining_uses is None)
             or (uses is not None and type(ref.remaining_uses) is int and 0 <= ref.remaining_uses <= uses),
             f'enchant_uses_invalid:{key}')
    _require((ttl is None and ref.remaining_turns is None)
             or (ttl is not None and type(ref.remaining_turns) is int and 0 <= ref.remaining_turns <= ttl),
             f'enchant_turns_invalid:{key}')
    if ref.persistent and ref.source_kind.value == 'CARD' and ref.card_id == '816':
        if not any(card.instance_id == ref.source_id for card in state.deck.removed):
            _compression(evidence, 'scheduled_effects.latent_encore', asdict(ref),
                         'Local initialization keeps the future Encore ledger before the card is played. '
                         'The raw once-play installer has not installed a visible enchant yet.', installer)
            return None
    if mode == 'timer_not_in_protocol':
        _compression(evidence, 'scheduled_effects.timer', asdict(ref),
                     'ExamEffectTimer has no ProduceExamStatusEnchant ID or timer field in uiobs.v1; '
                     'do not substitute the unrelated timed_gain status.', installer)
        return None
    if ref.effect_counter or ref.last_fired_turn is not None or ref.duration_fresh:
        _compression(evidence, 'scheduled_effects.counters_and_phase', asdict(ref),
                     'The enchant input contains only ID, uses and turns; source/counter/freshness '
                     'remain with the local rule engine.', 'GKMS encode.py:enchant t_ext; ' + installer)
    if (ref.card_id, ref.definition_effect_index) in {
            ('762', 3), ('597@14=1,54=1', 3), ('597@54=1', 2)}:
        value = asdict(ref)
        if kernel is not None:
            value['local_effect'] = asdict(kernel._scheduled_effect_spec(ref))
        _compression(evidence, 'scheduled_effects.support_payload', value,
                     'A supported nested score has the same raw enchant identity; UIObs has no '
                     'instance-specific nested-effect magnitude override.', 'support_schedule.py; ' + installer)
    if mode == 'remaining_uses':
        return [identifier, -1 if ref.remaining_uses is None else ref.remaining_uses, -1]
    if mode == 'remaining_turns':
        return [identifier, -1, ref.remaining_turns]
    if mode == 'end_uses_as_turns':
        # 363 installs five turns, while local rules retain the current end
        # plus five future ends. Initial grace is the sixth use, not a sixth UI turn.
        return [identifier, -1, min(raw_turns, ref.remaining_uses)]
    if mode == 'start_uses_as_turns':
        # Reuse the already verified 912->913 convention in opening.py.
        turns = ref.remaining_uses + int(ref.last_fired_turn == state.turn)
        _require(0 <= turns <= raw_turns, f'enchant_start_duration_invalid:{key}')
        return [identifier, -1, turns]
    raise ShadowMappingError(f'enchant_mode_unmapped:{mode}')


def _statuses(state):
    result = []
    for status in state.statuses:
        _require(status.status_id == 'anti_debuff' and type(status.stacks) is int
                 and status.stacks > 0 and status.remaining_turns is None,
                 f'state_semantic_mapping_unverified:statuses:{status.status_id}')
        result.append(['anti_debuff', status.stacks, -1])
    for field, kind in (('concentration', 'lesson_buff'), ('good_impression_turns', 'review'),
                        ('motivation', 'aggressive'), ('cost_reduction', 'stamina_consumption_down_fix')):
        value = getattr(state, field)
        if value:
            result.append([kind, value, -1])
    if state.good_condition_turns:
        result.append(['parameter_buff', state.good_condition_turns, state.good_condition_turns])
    for field, kind in (('excellent_condition_turns', 'parameter_buff_multiple'),
                        ('half_cost_turns', 'stamina_consumption_down'),
                        ('double_cost_turns', 'stamina_consumption_add'),
                        ('nullify_genki_turns', 'block_restriction')):
        if getattr(state, field):
            result.append([kind, 1, getattr(state, field)])
    for field, kind in (('score_buffs', 'lesson_parameter_multiple'),
                        ('concentration_effect_buffs', 'lesson_buff_multiple'),
                        ('concentration_addition_buffs', 'lesson_buff_gain_fix')):
        for modifier in getattr(state, field):
            _require(math.isfinite(modifier.amount), f'status_value_invalid:{field}')
            result.append([kind, modifier.amount, -1 if modifier.remaining_turns is None else modifier.remaining_turns])
    return result


_MODEL_COMPRESSION = (
    'UIObs has no scheduled-effect source identity, event counter, last-fired marker or freshness bit.',
    'The fixed featurizer omits some trigger conjunctions, target-card searches and nested effects.',
    'Deck zones are bags; no hidden draw order is supplied or inferred.',
)


@dataclass(frozen=True)
class ActionBinding:
    token: tuple
    action: GameAction


@dataclass(frozen=True)
class ShadowObservation:
    status: str
    request: DecisionRequest
    observation: dict[str, Any] | None = None
    bindings: tuple[ActionBinding, ...] = ()
    reason_codes: tuple[str, ...] = ()
    compression_notes: tuple[str, ...] = _MODEL_COMPRESSION
    compression_evidence: tuple[dict[str, Any], ...] = ()
    mappings: ShadowMappings | None = None

    def bind_action(self, model_action: Sequence, *, request: DecisionRequest, kernel) -> GameAction:
        """Validate a model token against this exact root and recheck local legality."""
        _require(self.status == 'supported', 'model_proposal_without_supported_observation')
        _require(request == self.request, 'model_proposal_stale_request')
        request.validate()
        _require(isinstance(model_action, (list, tuple)) and len(model_action) in (1, 2),
                 'model_action_shape_invalid')
        token = tuple(model_action)
        _require(token == ('end',) or (len(token) == 2 and token[0] in ('play', 'drink')
                 and type(token[1]) is int and token[1] >= 0), 'model_action_kind_or_index_invalid')
        matches = [binding.action for binding in self.bindings if binding.token == token]
        _require(len(matches) == 1, 'model_action_not_in_supplied_legal_set')
        action = matches[0]
        legal = kernel.legal_actions(request.state, plan_type=request.plan_type, decision_kind=request.decision_kind)
        _require(replace(action, source_slot=None) in tuple(replace(x, source_slot=None) for x in legal),
                 'model_action_no_longer_locally_legal')
        return action


def build_shadow_observation(request: DecisionRequest, kernel, context: ExamContext,
                             history: TurnPlayHistory | None) -> ShadowObservation:
    """Return supported UIObs, an explicit local-only phase, or a precise gap.

    Use the rule engine's types consistently. No partially filled UIObs is
    exposed after an error, and no unknown action ever becomes ['end'].
    """
    try:
        _require(isinstance(request, DecisionRequest), 'request_type_namespace_mismatch')
        request.validate()
        state = request.state
        if state.terminal:
            return ShadowObservation('local_only', request, reason_codes=('terminal',))
        if (request.decision_kind is DecisionKind.SECONDARY or state.pending_choice
                or state.resolution_stack or state.pending_after_card_ids):
            return ShadowObservation('local_only', request, reason_codes=('secondary_or_unfinished_resolution',))
        _require(request.plan_type is PlanType.SENSE and request.decision_kind is DecisionKind.LIVE,
                 'ordinary_sense_live_required')
        _require(state.deck.knowledge is DeckKnowledge.KNOWN_COMPOSITION, 'observed_composition_required')
        _require(state.persistent_effects_initialized, 'persistent_effect_lifecycle_missing')
        _require(history is not None, 'played_history_missing')
        history.validate(state)
        _require(state.stance is Stance.NEUTRAL and state.previous_stance is Stance.NEUTRAL,
                 'sense_stance_unmapped')
        # These fields can require actual additional UIObs statuses/enchants.
        # Their mappings have not been verified; do not relabel them as model loss.
        for field in ('full_power', 'cumulative_full_power', 'passion', 'passion_buff',
                      'passion_gain_bonus_pct', 'parameter_gain_bonus_pct', 'stance_lock_turns',
                      'strength_times', 'preservation_times', 'full_power_times', 'leisure_times',
                      'stance_changed_by_direct_effect_times', 'enthusiasm_bonus_buffs',
                      'enthusiasm_multiplier_buffs', 'full_power_charge_buffs',
                      'field_effects'):
            _require(not getattr(state, field), f'state_semantic_mapping_unverified:{field}')
        mappings = context.mappings
        _require(isinstance(mappings, ShadowMappings), 'verified_shadow_mappings_required')
        _require(set(state.items) <= mappings.item_ids, 'item_inventory_unmapped')
        for item_id in set(state.items) & mappings.observed_only_items.keys():
            from .kernel import ItemDefinition

            _require(kernel.items.get(item_id) == ItemDefinition(
                item_id, mappings.observed_only_items[item_id], (), PlanType.SENSE)
                and not any(ref.source_id == f'item:{item_id}' for ref in state.scheduled_effects),
                'stage_start_item_not_resolved')
        known_memories = {key[1] for key in mappings.schedules if key[0] == 'MEMORY'}
        _require(set(state.memory_abilities) <= known_memories, 'memory_inventory_unmapped')
        evidence = []
        for field, raw in (
                ('nullify_cost_cards', 'ExamSearchPlayCardStaminaConsumptionChange'),
                ('double_card_effects_remaining', 'ExamCardSearchEffectPlayCountBuff')):
            if getattr(state, field):
                _compression(evidence, field, getattr(state, field),
                    'The local count is known and determines legal costs/effect passes. The fixed '
                    'UIObs status vocabulary and enchant schema have no matching count input.',
                    f"ProduceExamEffectType_{raw}; GKMS obs.py:STATUS_KINDS; encode.py")
        for field in ('good_condition_fresh', 'excellent_condition_fresh', 'half_cost_fresh',
                      'double_cost_fresh', 'nullify_genki_fresh'):
            if getattr(state, field):
                _compression(evidence, field, True, 'UIObs stores remaining turns without a fresh bit.', 'GKMS encode.py:st_num')
        for field in ('score_buffs', 'concentration_effect_buffs', 'concentration_addition_buffs'):
            fresh = [asdict(value) for value in getattr(state, field) if value.fresh]
            if fresh:
                _compression(evidence, field + '.fresh', fresh,
                             'UIObs stores each known amount and lifetime without layer freshness.', 'GKMS encode.py:st_num')
        for card in state.deck.all_cards:
            _card(card, mappings)
        exam = context.observation(state)
        slots = dict(request.card_slots)
        _require(set(slots) == {card.instance_id for card in state.deck.hand}
                 and all(type(slot) is int and slot >= 0 for slot in slots.values()),
                 'hand_slot_binding_incomplete')
        hand = sorted(state.deck.hand, key=lambda card: slots[card.instance_id])
        hand_index = {card.instance_id: i for i, card in enumerate(hand)}
        drinks = sorted(request.drink_slots)
        _require(all(drink in mappings.drink_ids for _, drink in drinks), 'drink_inventory_unmapped')
        legal = kernel.legal_actions(state, plan_type=request.plan_type, decision_kind=request.decision_kind)
        bindings = []
        for action in legal:
            if action.kind is ActionKind.PLAY_CARD:
                _require(action.card_instance_id in hand_index, 'legal_play_has_no_hand_binding')
                bindings.append(ActionBinding(('play', hand_index[action.card_instance_id]),
                    replace(action, source_slot=slots[action.card_instance_id])))
            elif action.kind is ActionKind.USE_DRINK:
                matches = [(i, slot) for i, (slot, drink) in enumerate(drinks) if drink == action.drink_id]
                _require(bool(matches), 'legal_drink_has_no_bottle_binding')
                bindings.extend(ActionBinding(('drink', i), replace(action, source_slot=slot)) for i, slot in matches)
            elif action.kind is ActionKind.END_TURN:
                bindings.append(ActionBinding(('end',), action))
            else:
                raise ShadowMappingError(f'ordinary_legal_action_unmapped:{action.kind.value}')
        order = {'play': 0, 'drink': 1, 'end': 2}
        bindings.sort(key=lambda item: (order[item.token[0]], item.token[1:] or (-1,)))
        _require(bindings and len({binding.token for binding in bindings}) == len(bindings),
                 'legal_action_mapping_empty_or_ambiguous')
        playable = {binding.action.card_instance_id for binding in bindings if binding.token[0] == 'play'}
        numbers = dict(score=state.score, stamina=state.stamina, max_stamina=state.max_stamina,
                       block=state.genki, playable=state.actions_remaining, plays=len(history.played))
        _require(all(type(value) is int and value >= 0 for value in numbers.values()), 'num_integer_required')
        en = []
        refs = tuple(ref for ref in state.scheduled_effects if ref.remaining_uses != 0 and ref.remaining_turns != 0)
        for ref in refs + history.expired_start_effects:
            mapped = _enchant(ref, state, evidence, mappings, kernel=kernel)
            if mapped is not None:
                en.append(mapped)
        support = {fact.instance_id: fact.enhancement_count for fact in state.hand_support if fact.turn == state.turn}
        observation = dict(v='uiobs.v1', exam=exam, num=numbers, stance=[None, 0],
            status=_statuses(state), ench=en,
            hand=[dict(_card(card, mappings, support_count=support.get(card.instance_id), evidence=evidence),
                       p=int(card.instance_id in playable)) for card in hand],
            deck=[_card(card, mappings, evidence=evidence) for card in state.deck.draw_pile],
            grave=[_card(card, mappings, evidence=evidence) for card in state.deck.discard],
            hold=[_card(card, mappings, evidence=evidence) for card in state.deck.held],
            lost=[_card(card, mappings, evidence=evidence) for card in state.deck.removed],
            played=[_card(card, mappings, support_count=tier, evidence=evidence) for card, tier in zip(
                history.played, history.played_support or (None,) * len(history.played), strict=True)],
            drinks=[mappings.drink_ids[drink] for _, drink in drinks], legal=[list(binding.token) for binding in bindings])
        notes = _MODEL_COMPRESSION
        if state.deck.known_top_count:
            notes += ('The explicitly known deck prefix is not representable by the external bag encoder.',)
            _compression(evidence, 'deck.known_top_count', state.deck.known_top_count,
                         'The encoder treats draw-zone input as a bag.', 'GKMS encode.py:ZONES')
        return ShadowObservation('supported', request, observation, tuple(bindings), compression_notes=notes,
                                 compression_evidence=tuple(evidence), mappings=mappings)
    except (ShadowMappingError, HighScoreContractError, KernelError) as error:
        return ShadowObservation('unsupported', request, reason_codes=(str(error),))
