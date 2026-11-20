"""Observed support schedules, separate from permanent card effects."""
from dataclasses import replace

from .kernel import EffectKind, EffectSpec, CardCostKind, ConditionKind, CardDefinition, EffectCondition
from .contracts import PlanType, SchedulePhase, EffectSourceKind

CARD_ID = '597@14=1,54=1'
FOUR_TURN_CARD_ID = '597@54=1'
COUNTED_CARD_ID = '762'
NEXT_TURN_CARD_ID = '145'
SUPPORTED_CARD_IDS = (CARD_ID, FOUR_TURN_CARD_ID, COUNTED_CARD_ID, NEXT_TURN_CARD_ID)
# Index 3 is an explicit derived schedule entry, never a playable card effect.
SCHEDULE_INDEX = 3
SOURCE = 'native-hif-1074-1075:597@14=1,54=1:++'


def ordinary_counted_score_definition():
    """The exact frozen 762 rule; its ordinary score remains six."""
    counter = EffectCondition(ConditionKind.GOOD_CONDITION_MAX, threshold=20)
    trigger = EffectCondition(ConditionKind.ALL, all_of=(counter,
        EffectCondition(ConditionKind.EFFECT_COUNTER_MIN, threshold=2),
        EffectCondition(ConditionKind.EFFECT_COUNTER_MOD_EQ, modulus=2, remainder=0)))
    return CardDefinition(COUNTED_CARD_ID, '脚光+', PlanType.SENSE, stamina_cost=2,
        cost_kind=CardCostKind.GOOD_CONDITION,
        effects=(EffectSpec(EffectKind.CONCENTRATION, amount=7),
            EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
            EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
                schedule_phase=SchedulePhase.AFTER_CARD_USED, schedule_limit=None,
                schedule_counter_condition=counter,
                scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=6,
                    score_concentration_multiplier=4, condition=trigger),
                    EffectSpec(EffectKind.GOOD_CONDITION, amount=-3, condition=trigger)))),
        limited=True, card_type='active', rarity='SSR', base_card_id='761', source_type='produce')


def ordinary_schedule_index(card_id):
    return {CARD_ID: 1, FOUR_TURN_CARD_ID: 1, COUNTED_CARD_ID: 2, NEXT_TURN_CARD_ID: 1}[card_id]


def supported_schedule_index(card_id):
    return 2 if card_id in (FOUR_TURN_CARD_ID, NEXT_TURN_CARD_ID) else SCHEDULE_INDEX


def ordinary_definition(card_id=CARD_ID):
    if card_id == NEXT_TURN_CARD_ID:
        return CardDefinition(card_id, '成就+', PlanType.SENSE, stamina_cost=10,
            effects=(EffectSpec(EffectKind.GOOD_CONDITION, amount=7),
                EffectSpec(EffectKind.SCHEDULE, delay_turns=1,
                    schedule_phase=SchedulePhase.TURN, schedule_limit=1,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=40),))),
            limited=True, card_type='active', rarity='SSR', base_card_id='144', source_type='produce')
    if card_id not in (CARD_ID, FOUR_TURN_CARD_ID):
        raise ValueError('unsupported ordinary scheduled card')
    definition = CardDefinition(CARD_ID, '破竹の勢い+', PlanType.SENSE, stamina_cost=2,
        effects=(EffectSpec(EffectKind.CONCENTRATION, amount=8),
            EffectSpec(EffectKind.SCHEDULE, delay_turns=1,
                schedule_phase=SchedulePhase.START_OF_TURN, schedule_limit=4,
                scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=5),)),
            EffectSpec(EffectKind.GOOD_CONDITION, amount=2)),
        condition=EffectCondition(ConditionKind.GOOD_CONDITION_MIN, threshold=8),
        limited=True, card_type='active', rarity='SR', base_card_id='596',
        source_type='produce', canonical_card_id='597')
    return (replace(definition, card_id=card_id, effects=definition.effects[:2])
        if card_id == FOUR_TURN_CARD_ID else definition)


def supported_schedule_card(ordinary):
    if ordinary.card_id == COUNTED_CARD_ID:
        if ordinary != ordinary_counted_score_definition():
            raise ValueError('support schedule requires the exact ordinary definition')
        schedule = ordinary.effects[2]
        return replace(ordinary, effects=(*ordinary.effects[:2], replace(schedule,
            scheduled_effects=(replace(schedule.scheduled_effects[0], amount=7),
                schedule.scheduled_effects[1]))))
    if ordinary != ordinary_definition(ordinary.card_id):
        raise ValueError('support schedule requires the exact ordinary definition')
    if ordinary.card_id == NEXT_TURN_CARD_ID:
        return replace(ordinary, stamina_cost=9, effects=(ordinary.effects[0],
            replace(ordinary.effects[1], scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=44),))))
    return replace(ordinary, effects=(replace(ordinary.effects[0], amount=10),
        replace(ordinary.effects[1], scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=7),)),
        *ordinary.effects[2:]))


def is_support_schedule(ref):
    return (ref.source_kind is EffectSourceKind.CARD and ref.card_id in SUPPORTED_CARD_IDS
        and ref.definition_effect_index == supported_schedule_index(ref.card_id) and not ref.persistent)


def schedule_definition_effects(cards, ref):
    """Resolve a saved derived reference without modifying the permanent deck."""
    if not is_support_schedule(ref):
        return None
    ordinary = cards[ref.card_id]
    effective = supported_schedule_card(ordinary)
    if ref.card_id == NEXT_TURN_CARD_ID:
        if (ref.trigger_phase is not SchedulePhase.TURN or ref.direct_on_trigger is not True
                or ref.remaining_turns is not None or ref.remaining_uses != 1
                or ref.turns_until_fire not in (0, 1) or ref.duration_fresh):
            raise ValueError('support schedule reference differs from its verified lifetime')
        return (*ordinary.effects, effective.effects[1])
    if ref.card_id == COUNTED_CARD_ID:
        if (ref.trigger_phase is not SchedulePhase.AFTER_CARD_USED or ref.direct_on_trigger
                or ref.remaining_turns is not None or ref.remaining_uses is not None
                or ref.turns_until_fire != 0 or ref.duration_fresh
                or type(ref.effect_counter) is not int or ref.effect_counter < 0):
            raise ValueError('support schedule reference differs from its verified lifetime')
        return (*ordinary.effects, effective.effects[2])
    if (ref.trigger_phase is not SchedulePhase.START_OF_TURN or ref.direct_on_trigger
            or ref.remaining_turns is not None or type(ref.remaining_uses) is not int
            or not 0 <= ref.remaining_uses <= 4):
        raise ValueError('support schedule reference differs from its verified lifetime')
    return (*ordinary.effects, effective.effects[1])


def schedule_status_name(cards, ref):
    """Name the saved effective schedule, retaining permanent card identity."""
    name = cards[ref.card_id].name
    # The complete observed ++ body is represented by this derived index.
    # Revalidate it before recognizing a temporary suffix in a status row.
    return name + '+' if schedule_definition_effects(cards, ref) is not None else name
