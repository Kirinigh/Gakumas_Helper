"""A bounded proof that a pure action drink cannot help this entire turn."""
from __future__ import annotations

from dataclasses import replace

from .hif import HifSenseKernel
from .kernel import EffectKind, KernelError
from .planner import SearchIncomplete
from .contracts import HIF_SENSE_RULESET_ID, PlanType, ActionKind, GameAction, DecisionKind, DeckKnowledge
from .hif_dominance import _plain_positive_effect, _end_effects_erase_extra_actions
from .hif_choice_roots import _UNOBSERVED_EVENTS
from .validation_cache import check_search_budget

VERSION = 'unused-action-turn-proof-v1'
MAX_PAIRED_NODES = 128
MAX_PAIRED_EDGES = 512
_END = GameAction(ActionKind.END_TURN)


class _Declined(Exception):
    pass


def _clean(state):
    return (not state.terminal and state.actions_remaining > 0
            and not state.hand_support and not state.pending_choice
            and not state.resolution_stack and not state.pending_after_card_ids)


def prove_unused_action_drink_in_turn(kernel, request, action):
    """Compare every deterministic non-END continuation, never sample a draw.

    The held side retains A; the spent side has only A's extra action count.
    Every non-A legal choice must exist on both sides with identical events
    and full states apart from those two fields. Both sides must retain a
    positive action count at every node, so automatic END and skip-recovery
    differences are excluded. Existing END proofs then erase that difference.
    A partial graph, random boundary, choice or exhausted budget is no proof.
    The caller may decline the proved waste; legal roots and samples stay intact.
    """
    check_search_budget()
    if (type(kernel) is not HifSenseKernel or kernel.ruleset_id != HIF_SENSE_RULESET_ID
            or request.ruleset_id != HIF_SENSE_RULESET_ID or request.plan_type is not PlanType.SENSE
            or request.decision_kind is not DecisionKind.LIVE or not request.capabilities.all_ready
            or action.kind is not ActionKind.USE_DRINK):
        return None
    try:
        request.validate()
        action.validate()
        state = request.state
        if (not _clean(state) or state.turns_remaining <= 1 or not state.persistent_effects_initialized
                or state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
                or len(set(state.drinks)) != len(state.drinks) or type(action.source_slot) is not int
                or (action.source_slot, action.drink_id) not in request.drink_slots):
            return None
        definition = kernel.drinks.get(action.drink_id)
        if (definition is None or definition.stamina_cost != 0 or not definition.effects
                or not all(_plain_positive_effect(effect, {EffectKind.EXTRA_ACTION})
                           for effect in definition.effects)):
            return None
        gain = sum(int(effect.amount) for effect in definition.effects)
        unbound = replace(action, source_slot=None)
        legal = kernel.legal_actions(state, plan_type=request.plan_type, decision_kind=request.decision_kind)
        if unbound not in legal:
            return None
        # Import lazily: the strict proof adapter imports the search itself.
        from .hif_public_turn_reanalysis import _StrictDeterministicKernel

        proof = _StrictDeterministicKernel(**{name: dict(getattr(kernel, name))
            for name in ('cards', 'drinks', 'items', 'memory_abilities')})
        nodes, visiting = set(), set()
        counts = dict(paired_nodes=0, paired_edges=0, deterministic_steps=0, maximum_depth=0,
                      minimum_retained_actions=state.actions_remaining)

        def step(current, chosen, *, original=False):
            check_search_budget()
            branches = (kernel if original else proof).transition(current, chosen, plan_type=request.plan_type)
            check_search_budget()
            if (len(branches) != 1 or branches[0].probability != 1.
                    or any(event.event_type.startswith(_UNOBSERVED_EVENTS) or 'RANDOM' in event.event_type
                           for event in branches[0].events)):
                raise _Declined('non_deterministic_step')
            counts['deterministic_steps'] += 1
            return kernel._canonical_probability_state(branches[0].state), branches[0].events

        spent, events = step(state, unbound, original=True)
        source = f'drink:{action.drink_id}'
        if (sum(event.event_type == 'CONSUME_DRINK' for event in events) != 1
                or any(event.source_id != source or not (
                    event.event_type == 'CONSUME_DRINK'
                    or event.event_type == 'EXTRA_ACTION' and event.amount > 0
                    or event.event_type in {'COST_GENKI', 'COST_STAMINA'} and event.amount == 0)
                    for event in events)):
            return None

        def visit(held, used, depth):
            check_search_budget()
            key = held, used
            if key in nodes:
                return
            if key in visiting or counts['paired_nodes'] >= MAX_PAIRED_NODES:
                raise _Declined('cycle_or_node_budget')
            if (not _clean(held) or not _clean(used)
                    or held.turn != state.turn or used.turn != state.turn
                    or held.turns_remaining != state.turns_remaining or used.turns_remaining != state.turns_remaining
                    or held.drinks.count(action.drink_id) != 1
                    or used.drinks != tuple(identity for identity in held.drinks if identity != action.drink_id)
                    or used.actions_remaining != held.actions_remaining + gain
                    or replace(used, drinks=held.drinks, actions_remaining=held.actions_remaining) != held):
                raise _Declined('paired_state_not_equivalent')
            kept_legal = set(kernel.legal_actions(held, plan_type=request.plan_type, decision_kind=request.decision_kind))
            used_legal = set(kernel.legal_actions(used, plan_type=request.plan_type, decision_kind=request.decision_kind))
            if (kept_legal - {unbound} != used_legal or unbound not in kept_legal or _END not in used_legal
                    or any(candidate.kind not in (ActionKind.PLAY_CARD, ActionKind.USE_DRINK, ActionKind.END_TURN)
                           for candidate in used_legal)
                    or not _end_effects_erase_extra_actions(kernel, held)
                    or not _end_effects_erase_extra_actions(kernel, used)):
                raise _Declined('legal_actions_or_end_not_equivalent')
            counts['paired_nodes'] += 1
            counts['maximum_depth'] = max(counts['maximum_depth'], depth)
            counts['minimum_retained_actions'] = min(counts['minimum_retained_actions'], held.actions_remaining)
            visiting.add(key)
            for candidate in sorted(used_legal - {_END}, key=repr):
                if counts['paired_edges'] >= MAX_PAIRED_EDGES:
                    raise _Declined('edge_budget')
                counts['paired_edges'] += 1
                after_held, held_events = step(held, candidate)
                after_used, used_events = step(used, candidate)
                if held_events != used_events:
                    raise _Declined('paired_events_differ')
                visit(after_held, after_used, depth + 1)
            visiting.remove(key)
            nodes.add(key)

        visit(state, spent, 0)
        check_search_budget()
        from .hif_mcts import action_key

        card_slots = dict(request.card_slots)
        drink_slots = {identity: slot for slot, identity in request.drink_slots}
        bound = tuple(replace(candidate, source_slot=drink_slots[candidate.drink_id]
            if candidate.kind is ActionKind.USE_DRINK else card_slots.get(candidate.card_instance_id))
            for candidate in legal)
        return dict(version=VERSION, complete=True, excluded_action_key=action_key(action),
            original_legal_action_keys=tuple(action_key(candidate) for candidate in bound),
            retained_legal_action_keys=tuple(action_key(candidate) for candidate in bound if candidate != action),
            actions_before=state.actions_remaining, actions_after_a=spent.actions_remaining,
            transition_checks=dict(all_non_end_paths_paired=True, all_business_fields_preserved=True,
                events_identical=True, end_erases_positive_ap_at_every_node=True, **counts),
            deletes_root_actions=False, forces_end=False, new_final_returns=0)
    except (ValueError, KernelError, SearchIncomplete, _Declined):
        return None
