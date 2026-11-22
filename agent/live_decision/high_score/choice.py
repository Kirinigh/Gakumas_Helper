"""Pure model boundaries for a drink which waits for one retrieved card.

These projections never prove a native dispatch or grant input permission. The
host owns its durable dispatch record, fresh choice UI binding and final scan.
The original pre-drink bundle remains available for final settlement checks.
"""

import json
import math
from collections import Counter
from dataclasses import asdict, replace, dataclass

from .contracts import PlanType, ActionKind, DecisionKind, DecisionRequest


@dataclass(frozen=True)
class HifDrinkChoiceProjection:
    source_bundle: object
    action: object
    pending_state: object
    drink_slots: tuple


def _same_state(left, right):
    return json.dumps(asdict(left), sort_keys=True) == json.dumps(asdict(right), sort_keys=True)


def _valid_branches(branches):
    if (not branches or any(not math.isfinite(branch.probability) or branch.probability <= 0 for branch in branches)
            or not math.isclose(sum(branch.probability for branch in branches), 1.0, rel_tol=0, abs_tol=1e-9)):
        raise ValueError('HIF choice outcome probabilities are incomplete')
    for branch in branches:
        branch.state.validate()


def project_hif_drink_choice(bundle, action, *, canonical_input=False):
    """Predict one deterministic drink-origin MOVE_CARD boundary, without input."""
    if type(canonical_input) is not bool:
        raise ValueError('HIF choice canonical input flag must be boolean')
    state = bundle.state
    state.validate()
    action.validate()
    slots = tuple(bundle.drink_slots or ())
    if (action.kind is not ActionKind.USE_DRINK or type(action.source_slot) is not int
            or action.card_instance_id is not None or action.target_id is not None
            or state.pending_choice or state.resolution_stack or state.pending_after_card_ids
            or len({slot for slot, _ in slots}) != len(slots)
            or any(type(slot) is not int or not 0 <= slot < 4 for slot, _ in slots)
            or Counter(drink for _, drink in slots) != Counter(state.drinks)
            or (action.source_slot, action.drink_id) not in slots):
        raise ValueError('HIF drink choice requires the complete pre-drink state and physical slot')
    # Native search canonicalizes unknown deck order before its first action.
    # Legacy joint/hybrid journals keep their original deterministic order.
    transition_state = bundle.kernel._canonical_probability_state(state) if canonical_input else state
    branches = bundle.kernel.transition(transition_state, action, plan_type=PlanType.SENSE)
    _valid_branches(branches)
    if len(branches) != 1:
        raise ValueError('HIF drink choice requires one deterministic pending outcome')
    pending_state = branches[0].state
    pending = pending_state.pending_choice
    if (pending is None or pending.action_kind is not ActionKind.MOVE_CARD
            or pending.source_id != 'drink:' + action.drink_id
            or pending.selection_optional or pending.selections_remaining != 1
            or pending_state.resolution_stack or pending_state.pending_after_card_ids):
        raise ValueError('HIF drink outcome is not a supported retrieval boundary')
    remaining = tuple(pair for pair in slots if pair[0] != action.source_slot)
    if Counter(drink for _, drink in remaining) != Counter(pending_state.drinks):
        raise ValueError('HIF pending drink consumption differs from its physical slot')
    return HifDrinkChoiceProjection(bundle, action, pending_state, remaining)


def verify_hif_pending_choice(bundle, action, recorded_state):
    """Recompute saved pending state; native consumption proof stays with host."""
    projected = project_hif_drink_choice(bundle, action)
    recorded_state.validate()
    if not _same_state(projected.pending_state, recorded_state):
        # Both historical projections are recomputed in full; neither the
        # saved journal nor its candidate set/physical binding is rewritten.
        projected = project_hif_drink_choice(bundle, action, canonical_input=True)
        if not _same_state(projected.pending_state, recorded_state):
            raise ValueError('HIF saved pending choice differs from the dispatched model outcome')
    return projected


class HifRetrievalTargetRequest(DecisionRequest):
    """The drink plan already names one instance. Only its physical slot was read."""

    def _validate_secondary_slots(self, slot_ids):
        pending = self.state.pending_choice
        if (pending.source_id != 'drink:22' or len(slot_ids) != 1 or slot_ids[0] not in pending.candidate_instance_ids
                or not 0 <= self.card_slots[0][1] < len(pending.candidate_instance_ids)):
            raise ValueError('HIF retrieval target is outside the pending panel')


def build_hif_choice_request(projected, *, card_slots, capabilities, request_id, target_only=False):
    """Bind observed physical slots for a new choice or an already evaluated target."""
    slots = tuple(card_slots)
    if any(type(slot) is not int or slot < 0 for _, slot in slots):
        raise ValueError('HIF choice card slots are not physical nonnegative indices')
    request_type = HifRetrievalTargetRequest if target_only else DecisionRequest
    request = request_type(request_id, PlanType.SENSE, DecisionKind.SECONDARY,
        projected.pending_state, capabilities, card_slots=slots,
        drink_slots=projected.drink_slots, ruleset_id=projected.source_bundle.kernel.ruleset_id)
    request.validate()
    return request


def hif_choice_settlements(projected, choice, *, card_slots):
    """Predict every settled result of one bound selection; never pick a card."""
    from .contracts import CapabilityReport

    # Validating a model transition grants no execution capability.
    slots = tuple(tuple(pair) for pair in card_slots)
    request = build_hif_choice_request(projected, card_slots=slots,
        capabilities=CapabilityReport(False, False, False, False, False), request_id='model-choice-settlement',
        target_only=slots == ((choice.card_instance_id, choice.source_slot),))
    choice.validate()
    slots = dict(request.card_slots)
    if (choice.kind is not ActionKind.MOVE_CARD or choice.drink_id is not None or choice.target_id is not None
            or type(choice.source_slot) is not int or choice.card_instance_id not in slots
            or choice.source_slot != slots[choice.card_instance_id]):
        raise ValueError('HIF selection is not bound to its observed candidate slot')
    kernel = projected.source_bundle.kernel
    legal = kernel.legal_actions(projected.pending_state, plan_type=PlanType.SENSE,
        decision_kind=DecisionKind.SECONDARY)
    if replace(choice, source_slot=None) not in legal:
        raise ValueError('HIF selection is not a legal pending action')
    branches = kernel.transition(projected.pending_state, choice, plan_type=PlanType.SENSE)
    _valid_branches(branches)
    states = tuple(dict.fromkeys(branch.state for branch in branches))
    if any(state.pending_choice or state.resolution_stack or state.pending_after_card_ids
           or state.turn != projected.pending_state.turn
           or state.actions_remaining <= 0 for state in states):
        raise ValueError('HIF retrieval outcome still requires a choice or automatic turn continuation')
    return states[0] if len(states) == 1 else states
