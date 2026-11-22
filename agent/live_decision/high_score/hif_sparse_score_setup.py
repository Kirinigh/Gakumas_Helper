"""Public card-first advice for a one-visit draw-drink continuation."""

from dataclasses import replace

from .kernel import EffectKind
from .hif_mcts import action_key
from .contracts import ActionKind, SchedulePhase
from .hif_information_boundary import classify_information_boundary


def _draw_score_drink(definition, turns_remaining):
    if definition is None:
        return False
    effects = {effect.kind: effect for effect in definition.effects}
    required = {EffectKind.SCORE_BUFF, EffectKind.DRAW, EffectKind.SCHEDULE}
    if (len(effects) != len(definition.effects) or set(effects) - {EffectKind.STAMINA} != required
            or any(effect.condition is not None for effect in effects.values())):
        return False
    stamina = effects.get(EffectKind.STAMINA)
    if (stamina is None and definition.stamina_cost <= 0) or (stamina is not None and stamina.amount >= 0):
        return False
    buff, schedule = effects[EffectKind.SCORE_BUFF], effects[EffectKind.SCHEDULE]
    return (buff.amount > 0 and buff.duration is not None and 0 < buff.duration < turns_remaining
            and effects[EffectKind.DRAW].amount > 0 and schedule.schedule_phase is SchedulePhase.AFTER_START_OF_TURN
            and schedule.delay_turns == 1 and schedule.schedule_limit is not None and schedule.schedule_limit > 1
            and len(schedule.scheduled_effects) == 1 and schedule.scheduled_effects[0].kind is EffectKind.DRAW
            and schedule.scheduled_effects[0].amount > 0 and schedule.scheduled_effects[0].condition is None)


def _one_visit_winner(frozen, legal, drink):
    search = frozen.policy.search
    bound = frozen.request
    if search._public_turn_policy is not None and search._public_turn_policy.cursor(bound) is not None:
        return False
    if search._resource_policy is not None and search._resource_policy.cursor(bound, legal) is not None:
        return False
    if search._internal_policy is not None:
        from .hif_internal_reanalysis import lookup

        if lookup(search._internal_policy, bound, legal, context=search._context(bound), kernel=search.kernel) is not None:
            return False
    _, node = search._physical_node(bound, legal)
    return (node is not None and set(node.statistics) == {action_key(action) for action in legal}
            and all(visits == 1 for visits, _ in node.statistics.values())
            and max(node.order, key=lambda key: node.statistics[key][1]) == action_key(drink))


def select_sparse_score_setup(frozen, request, legal, drink):
    """Return a unique deterministic setup that leaves both scoring and drinking legal."""
    state = request.state
    if drink.kind is not ActionKind.USE_DRINK or state.actions_remaining < 2:
        return None
    kernel = frozen.owner.kernel
    if not _draw_score_drink(kernel.drinks.get(drink.drink_id), state.turns_remaining):
        return None
    if not _one_visit_winner(frozen, legal, drink):
        return None
    held = {ref.instance_id: ref for ref in state.deck.hand}
    setups = []
    for action in legal:
        if action.kind is ActionKind.PLAY_CARD:
            effects = kernel.card_definition(state, held[action.card_instance_id]).effects
            if ({effect.kind for effect in effects} == {EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION}
                    and all(effect.condition is None and effect.amount > 0 for effect in effects)):
                setups.append(action)
    if len(setups) != 1:
        return None
    setup, = setups
    boundary = classify_information_boundary(kernel, state, replace(setup, source_slot=None), plan_type=request.plan_type)
    after = boundary.successor
    if (boundary.kind != 'deterministic_same_turn' or after.score != state.score
            or after.concentration <= state.concentration or after.good_condition_turns <= state.good_condition_turns):
        return None
    following = kernel.legal_actions(after, plan_type=request.plan_type, decision_kind=request.decision_kind)
    if replace(drink, source_slot=None) not in following:
        return None
    for action in following:
        if (action.kind is ActionKind.PLAY_CARD
                and any(effect.kind is EffectKind.SCORE and effect.amount > 0 and effect.condition is None
                        for effect in kernel.card_definition(after, held[action.card_instance_id]).effects)):
            return setup
    return None
