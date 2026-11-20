"""Offline replay of a BattleBrain-shaped reply. No transport or game entry point."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, fields, replace, dataclass, is_dataclass

from agent.live_decision.high_score.planner import RobustHighScorePlanner
from agent.live_decision.high_score.contracts import ActionKind, GameAction, DecisionKind, DeckKnowledge, DecisionRequest
from agent.live_decision.high_score.hif_support_planning import uses_hand_support_for_strategy


@dataclass(frozen=True)
class BattleOffer:
    request: DecisionRequest
    payload: dict
    bindings: tuple[tuple[str, GameAction], ...]


def _description(value):
    """Compact a typed rule using its own declared defaults, including nested effects."""
    if isinstance(value, tuple):
        return [_description(item) for item in value]
    if is_dataclass(value):
        return {field.name: _description(item) for field in fields(value)
                if (item := getattr(value, field.name)) != field.default}
    return value


def battle_offer(request, kernel, *, played=None):
    """Build a local prompt from public business state and physical slot bindings."""
    request.validate()
    state = request.state
    if (request.decision_kind is not DecisionKind.LIVE or state.terminal or state.pending_choice
            or state.resolution_stack or state.pending_after_card_ids):
        return BattleOffer(request, {}, ())
    zones = ('draw_pile', 'discard', 'held', 'removed')
    if state.deck.knowledge in (DeckKnowledge.UNKNOWN, DeckKnowledge.BROKEN):
        return BattleOffer(request, dict(piles=dict.fromkeys(zones), played=played, legal_actions=None), ())
    support = tuple(fact for fact in state.hand_support if uses_hand_support_for_strategy(fact))
    if state.hand_support != support:
        state = replace(state, hand_support=support)
    slots = dict(request.card_slots)
    if (set(slots) != {card.instance_id for card in state.deck.hand}
            or any(type(slot) is not int or slot < 0 for slot in slots.values())):
        raise ValueError('mock_hand_slots_incomplete')
    legal = kernel.legal_actions(state, plan_type=request.plan_type, decision_kind=request.decision_kind)
    bindings = {}
    for action in legal:
        if action.kind is ActionKind.PLAY_CARD:
            bound = RobustHighScorePlanner._attach_source_slot(action, request)
            bindings[f'play:{bound.source_slot}'] = bound
        elif action.kind is ActionKind.USE_DRINK:
            for slot, drink_id in request.drink_slots:
                if drink_id == action.drink_id:
                    bindings[f'drink:{slot}'] = replace(action, source_slot=slot)
        elif action.kind is ActionKind.END_TURN:
            bindings['skip'] = action
        else:
            return BattleOffer(request, {}, ())

    hand = []
    for ref in sorted(state.deck.hand, key=lambda card: slots[card.instance_id]):
        definition = kernel.card_definition(state, ref)
        cost = (definition.full_power_cost if definition.cost_kind.value == 'FULL_POWER'
                else kernel._effective_stamina_cost(state, ref, definition))
        hand.append(dict(slot=slots[ref.instance_id], card_id=ref.card_id, name=definition.name,
                         cost_kind=definition.cost_kind.value, printed_cost=definition.stamina_cost,
                         estimated_cost=cost,
                         playable=f'play:{slots[ref.instance_id]}' in bindings,
                         effects=_description(definition.effects), persistent_effects=_description(definition.persistent_effects)))
    piles = {name: dict(sorted(Counter(card.card_id for card in getattr(state.deck, name)).items())) for name in zones}
    payload = dict(turn=state.turn, remaining_turns=state.turns_remaining, score=state.score, stamina=state.stamina,
                   max_stamina=state.max_stamina, genki=state.genki, actions_remaining=state.actions_remaining,
                   concentration=state.concentration, good_condition=state.good_condition_turns,
                   score_multiplier=state.score_multiplier, half_cost_turns=state.half_cost_turns,
                   cost_reduction=state.cost_reduction, score_buffs=_description(state.score_buffs),
                   concentration_effect_buffs=_description(state.concentration_effect_buffs),
                   concentration_addition_buffs=_description(state.concentration_addition_buffs),
                   scheduled_effects=_description(state.scheduled_effects), future_score_multipliers=state.future_score_multipliers,
                   hand=hand, piles=piles, played=played, drinks=[
                       dict(slot=slot, drink_id=drink_id, name=kernel.drinks[drink_id].name,
                            stamina_cost=kernel.drinks[drink_id].stamina_cost, effects=_description(kernel.drinks[drink_id].effects))
                       for slot, drink_id in sorted(request.drink_slots)],
                   legal_actions=list(bindings))
    return BattleOffer(request, payload, tuple(bindings.items()))


@dataclass(frozen=True)
class Advice:
    action: GameAction | None
    source: str
    reason: str
    reply_count: int

    def to_dict(self):
        return dict(asdict(self), value=None, execution_authorized=False)


def replay_advice(offer, current, scripted_replies, local_decide):
    """Only the first action binds. A plan, prior reply or score text grants no later action."""
    attempts = 0

    def local(reason):
        return Advice(local_decide(current), 'local_policy', reason, attempts)

    if current != offer.request:
        return local('observation_changed')
    if not offer.bindings:
        return local('local_phase')
    legal = dict(offer.bindings)
    replies = iter(scripted_replies)
    for attempts in (1, 2):
        reply = next(replies, None)
        if isinstance(reply, TimeoutError):
            return local('transport_timeout')
        if not isinstance(reply, str):
            return local('missing_response')
        try:
            data = json.loads(reply)
        except ValueError:
            return local('invalid_json')
        if not isinstance(data, dict) or not isinstance(data.get('action'), str):
            return local('missing_action')
        action = legal.get(data['action'].strip())
        if action is not None:
            return Advice(action, 'scripted_battle_reply', 'bound_legal_action', attempts)
    return local('illegal_action')
