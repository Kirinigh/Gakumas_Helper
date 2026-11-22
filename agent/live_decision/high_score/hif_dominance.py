"""Sufficient, exact resource dominance checks for the offline HIF kernel."""
from __future__ import annotations

import math
from dataclasses import replace, dataclass

from .hif import HifSenseKernel
from .kernel import EffectKind, EffectSpec, KernelError
from .contracts import (
    HIF_SENSE_RULESET_ID,
    PlanType,
    ActionKind,
    GameAction,
    DecisionKind,
    DeckKnowledge,
    SchedulePhase,
    DecisionRequest,
)
from .validation_cache import check_search_budget


@dataclass(frozen=True)
class HifResourceDominanceProof:
    dominant_action: GameAction
    dominated_action: GameAction
    legal_actions: tuple[GameAction, ...]
    successor_legal_actions: tuple[GameAction, ...]
    drink_slots_before: tuple[tuple[int, str], ...]
    drink_slots_after: tuple[tuple[int, str], ...]
    actions_before: int
    actions_after: int
    proof_rule: str = 'hif_pure_extra_action_before_forced_end_v1'
    scope: str = 'PROVEN_RESOURCE_DOMINANCE'
    complete_dominance_proof: bool = True
    optimality_proven: bool = False
    execution_authorized: bool = False


# These direct effects neither inspect the action count nor introduce a
# choice, card event, random operation or nested schedule before its reset.
_END_SCALAR_KINDS = frozenset({
    EffectKind.SCORE, EffectKind.GENKI, EffectKind.STAMINA,
    EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION,
    EffectKind.EXCELLENT_CONDITION, EffectKind.GOOD_IMPRESSION,
    EffectKind.MOTIVATION, EffectKind.COST_REDUCTION, EffectKind.EXTRA_ACTION,
})


def _plain_positive_effect(effect, kinds):
    return (type(effect) is EffectSpec and effect.kind in kinds
            and type(effect.amount) in (int, float) and math.isfinite(effect.amount)
            and effect.amount > 0 and int(effect.amount) == effect.amount
            and effect == EffectSpec(effect.kind, effect.amount))


def _end_effects_erase_extra_actions(kernel, state):
    """Audit the only phase executed before HIF resets remaining actions.

    HIF skip recovery reads only actions > 0. The shared kernel then runs
    END_OF_TURN and resets actions to 1 (or 0 at terminal), before next-turn
    draws and triggers. Complex END effects are outside this proof domain.
    """
    for ref in state.scheduled_effects:
        if (ref.trigger_phase is not SchedulePhase.END_OF_TURN
                or ref.turns_until_fire != 0 or ref.remaining_uses == 0
                or ref.remaining_turns == 0):
            continue
        effect = kernel._scheduled_effect_spec(ref)
        plain_schedule = EffectSpec(
            EffectKind.SCHEDULE, scheduled_effects=effect.scheduled_effects,
            delay_turns=0, schedule_phase=SchedulePhase.END_OF_TURN,
            schedule_limit=effect.schedule_limit, schedule_ttl=effect.schedule_ttl)
        if (effect != plain_schedule
                or not all(_plain_positive_effect(child, _END_SCALAR_KINDS)
                           for child in effect.scheduled_effects)):
            return False
    return True


def prove_forced_end_drink_dominance(
    kernel: HifSenseKernel, request: DecisionRequest, action: GameAction,
) -> HifResourceDominanceProof | None:
    """Prove END weakly dominates one pure-action drink followed by END.

    Return None outside the sufficient domain. Do not mutate the request,
    legal actions, kernel or inventory, and do not sample or enumerate END.
    The result is a local dominance proof, not a sampled value or a claim
    that the original legal list contained only one action.

    This relies on the exact current HIF rules: inventory has no passive or
    count condition, drinks cannot be generated, and slots only bind input.
    Retaining the bottle therefore preserves every future feasible policy.
    Unknown kernels, including sampled subclasses, are deliberately refused.
    """
    if (type(kernel) is not HifSenseKernel
            or kernel.ruleset_id != HIF_SENSE_RULESET_ID
            or request.ruleset_id != HIF_SENSE_RULESET_ID
            or request.plan_type is not PlanType.SENSE
            or request.decision_kind is not DecisionKind.LIVE
            or not request.capabilities.all_ready
            or action.kind is not ActionKind.USE_DRINK):
        return None
    try:
        request.validate()
        action.validate()
        state = request.state
        if (state.terminal or state.actions_remaining <= 0
                or not state.persistent_effects_initialized
                or state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
                or state.pending_choice or state.pending_after_card_ids or state.resolution_stack
                or type(action.source_slot) is not int
                or (action.source_slot, action.drink_id) not in request.drink_slots):
            return None
        definition = kernel.drinks.get(action.drink_id)
        if (definition is None or definition.stamina_cost != 0 or not definition.effects
                or not all(_plain_positive_effect(effect, {EffectKind.EXTRA_ACTION})
                           for effect in definition.effects)):
            return None
        unbound = replace(action, source_slot=None)
        end = GameAction(ActionKind.END_TURN)
        legal = kernel.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
        if len(legal) != 2 or set(legal) != {end, unbound}:
            return None
        if not _end_effects_erase_extra_actions(kernel, state):
            return None
        # This is the unsampled production transition, after the cheap
        # semantic filter has excluded random/compound effects.
        branches = kernel.transition(state, unbound, plan_type=PlanType.SENSE)
        if len(branches) != 1 or branches[0].probability != 1.0:
            return None
        branch = branches[0]
        after = branch.state
        remaining = list(state.drinks)
        remaining.remove(action.drink_id)
        if (after.drinks != tuple(remaining) or after.actions_remaining <= 0
                or after.actions_remaining != state.actions_remaining + sum(
                    int(effect.amount) for effect in definition.effects)
                or replace(after, drinks=state.drinks, actions_remaining=state.actions_remaining) != state):
            return None
        source = f'drink:{action.drink_id}'
        if (sum(event.event_type == 'CONSUME_DRINK' for event in branch.events) != 1
                or any(event.source_id != source or not (
                    event.event_type == 'CONSUME_DRINK'
                    or event.event_type == 'EXTRA_ACTION' and event.amount > 0
                    or event.event_type in {'COST_GENKI', 'COST_STAMINA'} and event.amount == 0)
                       for event in branch.events)):
            return None
        after_legal = kernel.legal_actions(after, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
        if after_legal != (end,):
            return None
        bound_legal = tuple(action if candidate == unbound else candidate for candidate in legal)
        return HifResourceDominanceProof(
            dominant_action=end, dominated_action=action, legal_actions=bound_legal,
            successor_legal_actions=after_legal, drink_slots_before=request.drink_slots,
            drink_slots_after=tuple(pair for pair in request.drink_slots if pair[0] != action.source_slot),
            actions_before=state.actions_remaining, actions_after=after.actions_remaining)
    except (ValueError, KernelError):
        # A proof failure never removes an action from the caller's search.
        return None


_UNUSED_DRINK_END = GameAction(ActionKind.END_TURN)


def _plain_action_preserving_drink_effect(effect):
    if (type(effect) is not EffectSpec or type(effect.amount) not in (int, float)
            or not math.isfinite(effect.amount)):
        return False
    if effect.kind is EffectKind.SCORE_BUFF:
        return (effect.amount > 0 and (effect.duration is None
                or type(effect.duration) is int and effect.duration > 0)
                and effect == EffectSpec(effect.kind, effect.amount, duration=effect.duration))
    if effect.kind is EffectKind.HALF_COST_TURNS:
        return _plain_positive_effect(effect, {EffectKind.HALF_COST_TURNS})
    return (effect.kind is EffectKind.STAMINA and int(effect.amount) == effect.amount
            and effect == EffectSpec(effect.kind, effect.amount))


def _empty_positive_action_state(state):
    return (not state.terminal and state.actions_remaining > 0 and not state.deck.hand
            and not state.pending_choice and not state.pending_after_card_ids
            and not state.resolution_stack)


def _verified_scalar_drink_transition(kernel, state, action, effects):
    check_search_budget()
    branches = kernel.transition(state, action, plan_type=PlanType.SENSE)
    check_search_budget()
    if len(branches) != 1 or branches[0].probability != 1.0:
        return None
    branch = branches[0]
    allowed = {'CONSUME_DRINK', 'COST_GENKI', 'COST_STAMINA'} | {e.kind.value for e in effects}
    if (not _empty_positive_action_state(branch.state)
            or sum(e.event_type == 'CONSUME_DRINK' for e in branch.events) != 1
            or any(e.source_id != f'drink:{action.drink_id}' or e.event_type not in allowed
                   for e in branch.events)):
        return None
    return branch


def prove_unused_action_drink(kernel, request, action):
    """Prove one extra-action suggestion unused in a strict two-drink state.

    A and B are both legal now, the hand is empty, and AP is positive.
    A only adds AP; B only changes unconditional score/cost/stamina fields.
    Verify A, B and A then B against full states and matching B events.
    Every available continuation ends with AP reset before the next draw.
    This never deletes an action from search or ranks B against END.
    The existing two-action proof remains separate and unchanged.
    """
    check_search_budget()
    if (type(kernel) is not HifSenseKernel or kernel.ruleset_id != HIF_SENSE_RULESET_ID
            or request.ruleset_id != HIF_SENSE_RULESET_ID
            or request.plan_type is not PlanType.SENSE or request.decision_kind is not DecisionKind.LIVE
            or not request.capabilities.all_ready or action.kind is not ActionKind.USE_DRINK):
        return None
    try:
        request.validate()
        action.validate()
        state = request.state
        if (not _empty_positive_action_state(state) or not state.persistent_effects_initialized
                or state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
                or len(state.drinks) != 2 or len(set(state.drinks)) != 2
                or len(request.drink_slots) != 2 or type(action.source_slot) is not int
                or (action.source_slot, action.drink_id) not in request.drink_slots):
            return None
        a = kernel.drinks.get(action.drink_id)
        b_id = next(identity for identity in state.drinks if identity != action.drink_id)
        b = kernel.drinks.get(b_id)
        if (a is None or a.stamina_cost != 0 or not a.effects
                or not all(_plain_positive_effect(e, {EffectKind.EXTRA_ACTION}) for e in a.effects)
                or b is None or not b.effects or not all(_plain_action_preserving_drink_effect(e) for e in b.effects)):
            return None
        a_key = replace(action, source_slot=None)
        b_key = GameAction(ActionKind.USE_DRINK, drink_id=b_id)
        legal = kernel.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
        if len(legal) != 3 or set(legal) != {_UNUSED_DRINK_END, a_key, b_key}:
            return None
        after_a = _verified_scalar_drink_transition(kernel, state, a_key, a.effects)
        after_b = _verified_scalar_drink_transition(kernel, state, b_key, b.effects)
        if after_a is None or after_b is None:
            return None
        sa, sb = after_a.state, after_b.state
        gain = sum(int(e.amount) for e in a.effects)
        if (sa.drinks != (b_id,) or sa.actions_remaining != state.actions_remaining + gain
                or replace(sa, drinks=state.drinks, actions_remaining=state.actions_remaining) != state
                or sb.drinks != (a.drink_id,) or sb.actions_remaining != state.actions_remaining
                or replace(sb, drinks=state.drinks, stamina=state.stamina, genki=state.genki,
                           score_buffs=state.score_buffs, half_cost_turns=state.half_cost_turns,
                           half_cost_fresh=state.half_cost_fresh) != state):
            return None
        after_ab = _verified_scalar_drink_transition(kernel, sa, b_key, b.effects)
        if after_ab is None:
            return None
        sab = after_ab.state
        if (sab.drinks or sab.actions_remaining != state.actions_remaining + gain
                or replace(sab, drinks=sb.drinks, actions_remaining=sb.actions_remaining) != sb
                or after_ab.events != after_b.events):
            return None
        for public, expected in ((state, {_UNUSED_DRINK_END, a_key, b_key}), (sa, {_UNUSED_DRINK_END, b_key}),
                                 (sb, {_UNUSED_DRINK_END, a_key}), (sab, {_UNUSED_DRINK_END})):
            if (not _end_effects_erase_extra_actions(kernel, public)
                    or set(kernel.legal_actions(public, plan_type=PlanType.SENSE,
                                                decision_kind=DecisionKind.LIVE)) != expected):
                return None
        slots = {identity: slot for slot, identity in request.drink_slots}
        bound = tuple(replace(x, source_slot=slots[x.drink_id]) if x.kind is ActionKind.USE_DRINK
                      else x for x in legal)
        check_search_budget()
        # The search imports this module; defer its action-key utility.
        from .hif_mcts import action_key

        return dict(version='hif-unused-action-drink-v1', excluded_action_key=action_key(action),
                    retained_legal_action_keys=tuple(action_key(x) for x in bound if x != action),
                    paired_drink_action_key=action_key(replace(b_key, source_slot=slots[b_id])),
                    original_legal_action_keys=tuple(action_key(x) for x in bound),
                    transition_checks=dict(a_preserves_other_state=True, b_scalar_only=True,
                        ab_equals_b_except_ap_and_inventory=True, b_events_identical=True,
                        end_erases_positive_ap_in_all_four_states=True, deterministic_steps=3),
                    actions_before=state.actions_remaining, actions_after_a=sa.actions_remaining,
                    complete=True, deletes_root_actions=False, forces_end=False)
    except (ValueError, KernelError):
        return None
