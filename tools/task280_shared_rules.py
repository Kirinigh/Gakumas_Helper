"""Reusable rule definitions for the isolated H.I.F. benchmark adapter."""

from agent.live_decision.high_score.kernel import (
    EffectKind,
    EffectSpec,
    CardSelector,
    ConditionKind,
    CardDefinition,
    EffectCondition,
)
from agent.live_decision.high_score.contracts import CardZone, SchedulePhase

SUMMER = '585'
SUMMER_FORM = ('p_card-00-sup-3_152', 1)
SUMMER_EFFECTS = (
    'e_effect-exam_card_move-p_card_search-deck_grave-p_card-00-acc-0_002-lost-random-1_1',
    'e_effect-exam_playable_value_add-01',
    'e_effect-exam_status_enchant-inf-enchant-p_card-00-sup-3_152-enc01',
)


def summer_memory():
    """Play the ordinary plus version with the explicit removal target."""
    selector = CardSelector(zones=(CardZone.DRAW_PILE, CardZone.DISCARD), base_card_id='23')
    counter = EffectCondition(ConditionKind.EFFECT_COUNTER_MOD_EQ, modulus=5, remainder=4)
    return CardDefinition(
        SUMMER, '夏夜に咲く思い出+', None, stamina_cost=4, limited=True,
        card_type='active', source_type='support', rarity='SSR',
        effects=(
            EffectSpec(EffectKind.REMOVE_RANDOM, amount=1, card_selector=selector),
            EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
            EffectSpec(
                EffectKind.SCHEDULE, schedule_phase=SchedulePhase.CARD_USED, delay_turns=0, schedule_limit=None,
                schedule_counter_condition=EffectCondition(ConditionKind.CONCENTRATION_MIN, threshold=0),
                scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=4, condition=counter),),
            ),
        ),
    )


def shared_card_candidates(master_cards, local_cards):
    """Recognize an ordinary rule candidate only at its verified source identity."""
    matches = [row for row in master_cards if (row['id'], row.get('upgradeCount', 0)) == SUMMER_FORM]
    local = [row for row in local_cards if row['id'] == SUMMER]
    definition = summer_memory()
    if (len(matches) != 1 or len(local) != 1
            or matches[0]['name'] != definition.name or local[0]['name'] != definition.name
            or tuple(row['produceExamEffectId'] for row in matches[0]['playEffects']) != SUMMER_EFFECTS):
        return {}
    return {SUMMER_FORM: definition}
