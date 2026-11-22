"""Project locally compiled schedules with a unique equivalent model rule."""

from __future__ import annotations

from functools import lru_cache

from .kernel import EffectKind, EffectSpec, ConditionKind, EffectCondition
from .contracts import SchedulePhase, EffectSourceKind
from ..hif_gkms_runtime import uinet21_master
from .gkms_master_forms import card_form


@lru_cache(maxsize=1)
def _tables():
    source = uinet21_master()
    return {
        name: {row['id']: row for row in source[name]}
        for name in ('ProduceCardSearch', 'ProduceExamEffect', 'ProduceExamStatusEnchant', 'ProduceExamTrigger')
    }


def _item_child_signature(child, condition):
    if child != EffectSpec(child.kind, child.amount, condition=condition):
        return None
    amount = child.amount
    if not float(amount).is_integer():
        return None
    if child.kind is EffectKind.COST_REDUCTION and amount > 0:
        return 'ExamStaminaConsumptionDownFix', amount, 0, 0, -1
    if child.kind is EffectKind.EXTRA_ACTION and amount > 0:
        return 'ExamPlayableValueAdd', 0, 0, amount, 0
    if child.kind is EffectKind.STAMINA and amount < 0:
        return 'ExamStaminaReduceFix', -amount, 0, 0, 0
    return None


def _read_item_child(row):
    values = (
        row.get('effectType', '').removeprefix('ProduceExamEffectType_'),
        row.get('effectValue1'), row.get('effectValue2'), row.get('effectCount'), row.get('effectTurn'),
    )
    for field, value in row.items():
        if field in {'id', 'effectGroupIds', 'effectType', 'effectValue1', 'effectValue2', 'effectCount', 'effectTurn'}:
            continue
        if value != 0 and not (isinstance(value, str) and value.endswith('_Unknown')):
            return None
    return values


@lru_cache(maxsize=64)
def _match_item_enchant(stamina_permil, children):
    tables = _tables()
    triggers = tables['ProduceExamTrigger']
    effects = tables['ProduceExamEffect']
    searches = tables['ProduceCardSearch']
    matches = []
    for enchant in tables['ProduceExamStatusEnchant'].values():
        trigger = triggers.get(enchant.get('produceExamTriggerId'), {})
        search = searches.get(trigger.get('produceCardSearchId'), {})
        if (set(enchant) != {'id', 'produceExamTriggerId', 'produceExamEffectIds'}
                or trigger.get('phaseTypes') != ['ProduceExamPhaseType_ExamCardPlayAfter']
                or trigger.get('phaseValues')
                or trigger.get('fieldStatusTypes') != ['ProduceExamFieldStatusType_StaminaUpMultiple']
                or trigger.get('fieldStatusValues') != [stamina_permil]
                or trigger.get('upperSearchCount') != 0 or trigger.get('lowerSearchCount') != 1
                or trigger.get('cardMovePositionType') != 'ProduceCardMovePositionType_Unknown'
                or trigger.get('lessonType') != 'ProduceStepLessonType_Unknown'
                or set(trigger) - {'id', 'phaseTypes', 'phaseValues', 'fieldStatusTypes', 'fieldStatusValues',
                    'upperSearchCount', 'lowerSearchCount', 'cardMovePositionType', 'lessonType', 'produceCardSearchId'}
                or search.get('cardPositionType') != 'ProduceCardPositionType_Target'
                or search.get('cardSearchTag') != 'idol-unique'):
            continue
        if any(value != 0 and not (isinstance(value, str) and value.endswith('_Unknown'))
                for field, value in search.items() if field not in {'id', 'cardPositionType', 'cardSearchTag'}):
            continue
        if tuple(_read_item_child(effects.get(effect_id, {}))
                 for effect_id in enchant['produceExamEffectIds']) == children:
            matches.append(enchant['id'])
    return matches[0] if len(matches) == 1 else None


def _item_enchant(ref, spec):
    if (not spec.scheduled_effects or spec.schedule_limit is None
            or spec != EffectSpec(EffectKind.SCHEDULE, scheduled_effects=spec.scheduled_effects,
                delay_turns=0, schedule_phase=SchedulePhase.AFTER_CARD_USED, schedule_limit=spec.schedule_limit)
            or ref.trigger_phase is not SchedulePhase.AFTER_CARD_USED or ref.turns_until_fire != 0
            or ref.remaining_uses is None or not 0 < ref.remaining_uses <= spec.schedule_limit
            or ref.remaining_turns is not None or ref.direct_on_trigger or ref.effect_counter):
        return None
    condition = spec.scheduled_effects[0].condition
    if condition is None or condition.kind is not ConditionKind.ALL or len(condition.all_of) != 2:
        return None
    source, stamina = condition.all_of
    if (source != EffectCondition(ConditionKind.USED_CARD_SOURCE_TYPE, value='pIdol')
            or stamina.kind is not ConditionKind.STAMINA_RATIO_MIN
            or stamina.threshold is None or not 0 <= stamina.threshold <= 1):
        return None
    stamina_permil = round(stamina.threshold * 1000)
    if stamina.threshold != stamina_permil / 1000:
        return None
    children = tuple(_item_child_signature(child, condition) for child in spec.scheduled_effects)
    if any(child is None for child in children):
        return None
    enchant_id = _match_item_enchant(stamina_permil, children)
    return [enchant_id, ref.remaining_uses, -1] if enchant_id else None


def _latent_opening_card(ref, spec, state, played, kernel):
    """The kernel tracks an opening ledger before the model's once-play install."""
    if (state.turn != 1 or ref.trigger_phase is not SchedulePhase.EVERY_TURN
            or ref.turns_until_fire != spec.delay_turns or ref.direct_on_trigger
            or ref.last_fired_turn is not None or ref.effect_counter
            or ref.remaining_uses != spec.schedule_limit or ref.remaining_turns is not None):
        return False
    gate = spec.schedule_counter_condition
    if gate is None or gate.kind is not ConditionKind.REMOVED_BASE_CARD_COUNT_MIN or gate.threshold != 1:
        return False
    deck = state.deck
    if played or deck.removed or deck.discard or deck.held:
        return False
    card = next((card for card in deck.all_cards if card.instance_id == ref.source_id), None)
    if card is None or card.card_id != ref.card_id:
        return False
    definition = kernel.cards.get(card.card_id)
    if definition is None or not gate.value or gate.value != definition.base_card_id:
        return False
    form = card_form(definition, card.upgraded)
    if form is None:
        return False
    tables = uinet21_master()
    effects = _tables()['ProduceExamEffect']
    rows = [row for row in tables['ProduceCard'] if (row['id'], row['upgradeCount']) == form[:2]]
    if len(rows) != 1:
        return False
    installs = [
        effects.get(link.get('produceExamEffectId'), {})
        for link in rows[0].get('playEffects', ()) if link.get('isOncePlayEffect') is True
    ]
    candidates = [
        effect for effect in installs
        if effect.get('effectType') == 'ProduceExamEffectType_ExamStatusEnchantEncore'
        and effect.get('effectCount') == spec.schedule_limit and effect.get('effectTurn') == -1
        and effect.get('produceExamStatusEnchantId') in _tables()['ProduceExamStatusEnchant']
    ]
    return len(candidates) == 1


def project_enchant(ref, state, history, kernel):
    """Return a model row, an inactive opening ledger, or an unresolved effect."""
    if not ref.persistent:
        return False, None
    spec = kernel._scheduled_effect_spec(ref)
    try:
        if ref.source_kind is EffectSourceKind.ITEM:
            row = _item_enchant(ref, spec)
            return row is not None, row
        if ref.source_kind is EffectSourceKind.CARD and _latent_opening_card(ref, spec, state, history.played, kernel):
            return True, None
    except FileNotFoundError:
        return False, None
    return False, None
