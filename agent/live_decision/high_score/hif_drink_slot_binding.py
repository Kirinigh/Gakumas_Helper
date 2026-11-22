"""Translate verified bottle positions without rewriting saved computation.

Logical positions name the bottles already used by the candidate's tables.
Observed positions belong to the current Native boundary. This mapping grants
no input authority and does not carry a frame, deadline or execution receipt.
"""
from __future__ import annotations

from dataclasses import field, replace, dataclass

from .contracts import ActionKind, GameAction, DecisionRequest
from .hif_mcts_cache import json_value, slots_from_json, verified_drink_slot_map


@dataclass(frozen=True)
class HifDrinkSlotBinding:
    logical_slots: tuple[tuple[int, str], ...]
    observed_slots: tuple[tuple[int, str], ...]
    _pairs: tuple[tuple[tuple[int, str], tuple[int, str]], ...] = field(repr=False)

    @classmethod
    def create(cls, state, logical_slots, observed_slots):
        logical = slots_from_json(json_value(logical_slots), state)
        observed = slots_from_json(json_value(observed_slots), state)
        mapping = verified_drink_slot_map(state, logical, observed)
        return cls(logical, observed, tuple(mapping.items()))

    def logical_request(self, request: DecisionRequest) -> DecisionRequest:
        if tuple(request.drink_slots) != self.observed_slots:
            raise ValueError('hif_drink_binding_observed_slots_changed')
        # The caller may reuse a binding after a card, but never after the
        # bottle inventory changes. Other business state is owned by Native.
        verified_drink_slot_map(request.state, self.logical_slots, self.observed_slots)
        return replace(request, drink_slots=self.logical_slots)

    def _action(self, action: GameAction, *, to_observed: bool) -> GameAction:
        if action.kind is not ActionKind.USE_DRINK:
            return action
        if type(action.source_slot) is not int:
            raise ValueError('hif_drink_binding_action_not_bound')
        source = (action.source_slot, action.drink_id)
        matches = [observed if to_observed else logical for logical, observed in self._pairs
                   if source == (logical if to_observed else observed)]
        if len(matches) != 1:
            raise ValueError('hif_drink_binding_action_not_bound')
        return replace(action, source_slot=matches[0][0])

    def logical_action(self, action: GameAction) -> GameAction:
        return self._action(action, to_observed=False)

    def observed_action(self, action: GameAction) -> GameAction:
        return self._action(action, to_observed=True)
