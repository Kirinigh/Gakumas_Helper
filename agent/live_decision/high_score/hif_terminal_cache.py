"""Conservative dependency analysis for terminal marginals and transitions.

The first transition always uses the full shared kernel. Terminal resource
distributions are reused only when layouts cannot matter and no action remains.
Scalar transition deltas may grant another action, but preserve the actual deck
for subsequent search. No cached state is returned to execution or observation.
"""
from .kernel import IMMEDIATE_SCHEDULE_PHASES, EffectKind, ConditionKind, TransitionBranch, EffectValueSource, _condition_matches
from .contracts import ActionKind, SchedulePhase
from .state_copy import replace

_SCALAR_EFFECTS = frozenset({
    EffectKind.SCORE, EffectKind.STAMINA, EffectKind.GENKI, EffectKind.FIXED_GENKI,
    EffectKind.COST_REDUCTION, EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION,
    EffectKind.EXCELLENT_CONDITION, EffectKind.GOOD_IMPRESSION, EffectKind.MOTIVATION,
    EffectKind.NULLIFY_GENKI_TURNS, EffectKind.HALF_COST_TURNS, EffectKind.DOUBLE_COST_TURNS,
    EffectKind.SCORE_BUFF, EffectKind.CONCENTRATION_EFFECT_BUFF,
    EffectKind.CONCENTRATION_ADDITION_BUFF, EffectKind.NULLIFY_COST_CARDS,
    EffectKind.FIXED_STAMINA_TO_ZERO,
})
_SCALAR_CONDITIONS = frozenset({
    ConditionKind.STANCE_FAMILY, ConditionKind.STANCE_NOT, ConditionKind.STANCE_IS,
    ConditionKind.PREVIOUS_STANCE_IS, ConditionKind.FULL_POWER_CHARGE_MIN,
    ConditionKind.FULL_POWER_CHARGE_MAX, ConditionKind.CUMULATIVE_FULL_POWER_MIN,
    ConditionKind.TURNS_REMAINING_MAX, ConditionKind.IS_DIRECT_EFFECT,
    ConditionKind.USED_CARD_EFFECT_TAG, ConditionKind.USED_CARD_BASE_ID,
    ConditionKind.USED_CARD_ID, ConditionKind.USED_CARD_SOURCE_TYPE,
    ConditionKind.USED_CARD_RARITY_NOT, ConditionKind.TURN_ATTRIBUTE,
    ConditionKind.STANCE_CHANGE_COUNT_MOD_EQ, ConditionKind.STANCE_CHANGE_COUNT_MIN,
    ConditionKind.STRENGTH_COUNT_MIN, ConditionKind.PRESERVATION_COUNT_MIN,
    ConditionKind.FULL_POWER_COUNT_MIN, ConditionKind.CONCENTRATION_MIN,
    ConditionKind.GOOD_CONDITION_MIN, ConditionKind.GOOD_CONDITION_MAX,
    ConditionKind.STAMINA_RATIO_MIN, ConditionKind.EXCELLENT_CONDITION_MIN,
    ConditionKind.GOOD_IMPRESSION_MIN, ConditionKind.MOTIVATION_MIN,
    ConditionKind.GENKI_MIN, ConditionKind.EFFECT_COUNTER_MIN, ConditionKind.EFFECT_COUNTER_MOD_EQ,
})
_USED_CARD_CONDITIONS = frozenset({
    ConditionKind.USED_CARD_EFFECT_TAG, ConditionKind.USED_CARD_BASE_ID,
    ConditionKind.USED_CARD_ID, ConditionKind.USED_CARD_SOURCE_TYPE, ConditionKind.USED_CARD_RARITY_NOT,
})
_REACHABLE_PHASES = frozenset({SchedulePhase.BEFORE_CARD_USED, SchedulePhase.CARD_USED,
                             SchedulePhase.AFTER_CARD_USED, SchedulePhase.END_OF_TURN})


def _scalar_condition(condition):
    if condition is None:
        return True
    if condition.kind is ConditionKind.ALL:
        return all(_scalar_condition(child) for child in condition.all_of)
    return condition.kind in _SCALAR_CONDITIONS


def _scalar_effect(effect, *, allow_extra=False):
    if (not _scalar_condition(effect.condition) or effect.value_source not in {
            None, EffectValueSource.MAX_STAMINA, EffectValueSource.GOOD_CONDITION,
            EffectValueSource.GOOD_IMPRESSION, EffectValueSource.MOTIVATION, EffectValueSource.GENKI}):
        return False
    if effect.kind is EffectKind.SCHEDULE:
        if effect.schedule_phase not in _REACHABLE_PHASES:
            return True  # No later turn can be entered after this last action.
        return (_scalar_condition(effect.schedule_counter_condition)
                and all(_scalar_effect(nested, allow_extra=allow_extra) for nested in effect.scheduled_effects))
    return effect.kind in _SCALAR_EFFECTS or (allow_extra and effect.kind is EffectKind.EXTRA_ACTION)


def fixed_terminal_turn(kernel, state):
    """Prove the owned effect closure cannot add turns or introduce new cards."""
    if (state.hand_support or state.turns_remaining != 1
            or not state.persistent_effects_initialized or state.field_effects):
        return False
    definitions = [kernel.cards[card.card_id] for card in state.deck.all_cards]
    definitions += [kernel.drinks[key] for key in state.drinks]
    definitions += [kernel.items[key] for key in state.items]
    definitions += [kernel.memory_abilities[key] for key in state.memory_abilities]
    forbidden = {EffectKind.TURNS_REMAINING, EffectKind.ADD_CARD_TO_DECK,
                 EffectKind.UPGRADE_HAND, EffectKind.CARD_MODIFICATION}
    def allowed(effect):
        return effect.kind not in forbidden and all(allowed(child) for child in effect.scheduled_effects)
    groups = [getattr(definition, name, ()) for definition in definitions
              for name in ('effects', 'persistent_effects')]
    groups += [kernel._scheduled_definition_effects(ref) or () for ref in state.scheduled_effects]
    return all(allowed(effect) for group in groups for effect in group)


def terminal_schedules(state, *, scalar_card=False):
    """Key-only phase grouping. Order within each executable phase is intact."""
    return tuple(sorted((ref for ref in state.scheduled_effects
        if ref.trigger_phase in (_REACHABLE_PHASES if scalar_card else IMMEDIATE_SCHEDULE_PHASES)
        and ref.turns_until_fire == 0
        and ref.remaining_uses != 0), key=lambda ref: ref.trigger_phase.value))


class TerminalMarginalCache:
    def __init__(self, kernel):
        self.kernel = kernel
        self.universes = {}  # Strong references prevent identity reuse.
        self.eligibility = {}
        self.values = {}
        self.hits = 0
        self.transitions = {}
        self.transition_hits = 0

    def _static_false(self, condition, state, context):
        if condition is None:
            return False
        if condition.kind is ConditionKind.ALL:
            return any(self._static_false(child, state, context) for child in condition.all_of)
        return (condition.kind in _USED_CARD_CONDITIONS
                and not _condition_matches(state, condition, self.kernel.cards, context))

    def _eligible(self, state, action, *, allow_extra=False):
        if action.kind is ActionKind.USE_DRINK:
            definition = self.kernel.drinks.get(action.drink_id)
            return definition is not None and all(
                _scalar_effect(effect, allow_extra=allow_extra) for effect in definition.effects)
        card = next((c for c in state.deck.hand if c.instance_id == action.card_instance_id), None)
        if card is None:
            return False
        definition = self.kernel.cards[card.card_id]
        if not _scalar_condition(definition.condition):
            return False
        # Direct effects have their own context; reject all non-scalar kinds
        # even when the condition happens to be false in this one observation.
        if not all(_scalar_effect(effect, allow_extra=allow_extra) for effect in definition.effects):
            return False
        context = self.kernel._card_event_context(state, card.instance_id, exclude_used_card_from_targets=False)
        for ref in state.scheduled_effects:
            if (ref.remaining_uses == 0
                    or ref.turns_until_fire != 0 or ref.trigger_phase not in _REACHABLE_PHASES):
                continue
            effect = self.kernel._scheduled_effect_spec(ref)
            if not _scalar_condition(effect.schedule_counter_condition):
                return False
            event_context = None if ref.trigger_phase is SchedulePhase.END_OF_TURN else context
            for nested in effect.scheduled_effects:
                # Used-card identity remains fixed for all reachable phases;
                # neither scalar effects nor their timers can play a new card.
                if self._static_false(nested.condition, state, event_context):
                    continue
                if not _scalar_effect(nested, allow_extra=allow_extra):
                    return False
        return True

    def key(self, state, action, *, transition=False):
        if (state.hand_support or state.turns_remaining != 1 or state.actions_remaining < 1
                or (not transition and state.actions_remaining != 1) or state.field_effects
                or not state.persistent_effects_initialized or state.pending_choice
                or state.resolution_stack or state.pending_after_card_ids
                or action.kind not in ({ActionKind.PLAY_CARD, ActionKind.USE_DRINK}
                                       if transition else {ActionKind.PLAY_CARD})):
            return None
        if action.kind is ActionKind.USE_DRINK and action.drink_id not in state.drinks:
            return None
        universe = id(state.deck.all_cards)
        self.universes[universe] = state.deck.all_cards
        schedules = state.scheduled_effects if transition else terminal_schedules(state, scalar_card=True)
        identity = (action.kind, action.card_instance_id, action.drink_id)
        eligibility_key = (universe, identity, schedules, transition)
        eligible = self.eligibility.get(eligibility_key)
        if eligible is None:
            eligible = self.eligibility[eligibility_key] = self._eligible(state, action, allow_extra=transition)
        if not eligible:
            return None
        # Keep every scalar/trigger field, card customization and source id.
        # Scalar effects/conditions do not inspect drink inventory. Transition
        # replay consumes its specific bottle from the current inventory;
        # terminal marginals cannot consume another drink after the last card.
        omitted = {'deck', 'score', 'drinks'}
        return (universe, identity,
                tuple(schedules if name == 'scheduled_effects' else value
                      for name, value in state.__dict__.items() if name not in omitted))

    def remember_transition(self, key, before, branches):
        if len(branches) != 1 or branches[0].probability != 1:
            return
        branch = branches[0]
        if branch.state.pending_choice or branch.state.resolution_stack:
            return
        changes = {name: value for name, value in branch.state.__dict__.items()
                   if name not in {'deck', 'score', 'drinks'} and value != getattr(before, name)}
        self.transitions[key] = (changes, branch.state.score - before.score, branch.events)

    def restore_transition(self, key, state, action):
        changes, score_delta, events = self.transitions[key]
        deck = state.deck
        drinks = state.drinks
        if action.kind is ActionKind.PLAY_CARD:
            card = next(card for card in state.deck.hand if card.instance_id == action.card_instance_id)
            definition = self.kernel.cards[card.card_id]
            hand = tuple(other for other in state.deck.hand if other.instance_id != card.instance_id)
            zone = 'removed' if definition.limited else 'discard'
            deck = state.deck.replace_zones(hand=hand, **{zone: getattr(state.deck, zone) + (card,)})
        elif action.kind is ActionKind.USE_DRINK:
            remaining = list(drinks)
            remaining.remove(action.drink_id)
            drinks = tuple(remaining)
        updated = replace(state, deck=deck, drinks=drinks, score=state.score + score_delta, **changes)
        updated = self.kernel._canonical_probability_state(updated)
        self.transition_hits += 1
        return (TransitionBranch(1.0, updated, events),)
