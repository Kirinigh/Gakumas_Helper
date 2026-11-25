"""Tower settlement for card forms backed by the frozen Rust master."""

import re
import json
import math
import zlib
import hashlib
import sqlite3
from pathlib import Path
from functools import lru_cache
from contextlib import closing
from collections import deque, defaultdict
from dataclasses import replace, dataclass

from numpy import float32

from agent.master_data import shared_master_rows
from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

from .high_score.kernel import (
    RuleEvent,
    EffectKind,
    EffectSpec,
    CardCostKind,
    CardSelector,
    ConditionKind,
    CardDefinition,
    ItemDefinition,
    EffectCondition,
    TransitionBranch,
    EffectValueSource,
    PythonNiaProKernel,
)
from .idol_road_timeline import TowerTurnHint, tower_field_rule, match_formation_effects
from .high_score.contracts import (
    Stance,
    PlanType,
    SchedulePhase,
    StrategyState,
    TimedModifier,
    TurnResolutionStage,
)
from .high_score.state_copy import register_copy_record

_HIRO = '8380622a467cc30296c91b79dba4348166ef8d12'
_MASTER = 'd31baf6608ee4553091f28446f628cd3ad66ff01'
_MASTER_SHA256 = '50141992bc9f6617290f536b291c0890b11fbdd4b1d326d111db84d2561108a6'
_FAMILIES = {
    172: ('p_card-02-men-3_004', 'produce'),
    516: ('p_card-02-act-1_037', 'produce'),
    520: ('p_card-02-men-3_040', 'produce'),
    616: ('p_card-02-sup-3_156', 'support'),
    755: ('p_card-02-ido-3_169', 'pIdol'),
    757: ('p_card-02-ido-3_190', 'pIdol'),
    871: ('p_card-02-ido-3_203', 'pIdol'),
    873: ('p_card-02-ido-3_216', 'pIdol'),
}
_WIRE_FORMS = {str(base + upgrade): (wire, upgrade) for base, (wire, _) in _FAMILIES.items()
               for upgrade in (0, 1)}
_ENCORE = 'e_effect-exam_status_enchant_encore-0001-03-inf-enchant-p_card-02-ido-3_203-enc01'
_REVIEW_REFUND = 'review_from_consumption'
_REVIEW_VALUE_MULTIPLE = 'review_value_multiple'
_REVIEW_MODIFIERS = {'review_gain_multiple': 'review_gain_buffs', 'review_multiple': 'review_score_buffs'}
# StartPlay triggers are selected together before the item can extend the round.
_ENCORE_PHASE = SchedulePhase.AFTER_START_OF_TURN
_NATIVE_PHASES = {4: SchedulePhase.START_OF_TURN, 23: SchedulePhase.CARD_USED, 27: SchedulePhase.AFTER_START_OF_TURN}
# ProduceItem and ProduceItemEffect at _MASTER, in addition to the native
# tables shipped to the runner. The first item already has a local CSV ID.
_ITEMS = {
    '404': ('咲季の完全食・改', 'enchant-pitem_02-3-309-0-enc01', 1),
    'pitem_02-3-321-0': ('頂点の輝き', 'enchant-p_item_effect_02-3-321-0-enc01', 2),
    'pitem_02-3-321-1': ('頂点の輝き+', 'enchant-p_item_effect_02-3-321-1-enc01', 2),
}
_PER_TURN_ITEM = '404'
_SUPPORT_DRAW_FORM = ('p_card-02-sup-3_178', 1)


def has_tower_card_rules(card_ids):
    return any(cid in _WIRE_FORMS for cid in card_ids)


def tower_skip_recovery():
    setting, = _native_master()['ExamSetting']
    return setting['examTurnEndRecoveryStamina']


def _tower_catalog_rows(table, pattern):
    shared = shared_master_rows('tower-runtime', (table,), revision=_MASTER, pattern=pattern)
    if shared is not None:
        return tuple(shared[table])
    relative = Path(_HIRO) / 'vendor/gakumas-player/data/master' / _MASTER / 'catalog.sqlite'
    path = resolve_runtime_path(Path('.local/task275-rust/runtime') / relative,
                                Path('data/idol_road/rust_gnrpa') / relative)
    if not path.is_file():
        return ()
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        records = db.execute(
            "SELECT payload FROM records WHERE table_name = ? AND row_id GLOB ? ORDER BY ordinal",
            (table, pattern),
        ).fetchall()
    return tuple(json.loads(zlib.decompress(payload)) for payload, in records)


@lru_cache(maxsize=40)
def _tower_stage_rows(layer):
    schedules = defaultdict(dict)
    for row in _tower_catalog_rows('ProduceExamGimmickEffectGroup', f'p_exam_gimmick-tower_001-*-stage_{layer:03d}'):
        if row['startTurn'] and not row['remainingTurn'] and not row['remainingTurnPermil']:
            schedule = schedules[row['id']]
            existing = schedule.get(row['startTurn'])
            if existing is None or row['priority'] < existing['priority']:
                schedule[row['startTurn']] = row
    return dict(schedules)


def _stage_descriptions(layer):
    return {key: {turn: re.sub(r'<[^>]+>', '', ''.join(part['text'] for part in row['produceDescriptions']))
                  for turn, row in rows.items()} for key, rows in _tower_stage_rows(layer).items()}


def tower_stage_descriptions(layer):
    return tuple(_stage_descriptions(layer).values())


def tower_stage_timeline(layer, identity):
    return {turn: TowerTurnHint(turn, '', description)
            for turn, description in _stage_descriptions(layer).get(identity, {}).items()}


def bind_tower_stage(layer, timeline):
    """An observed schedule selects Hiro's variant and the corresponding fixed items."""
    key = match_formation_effects(timeline, _stage_descriptions(layer))
    if key is None:
        return None
    return tower_stage_rules(layer, key)


@dataclass(frozen=True)
class TowerStageRules:
    identity: str
    rows: dict
    fixed_items: tuple


@lru_cache(maxsize=128)
def tower_stage_rules(layer, key):
    # Hiro tower.stage_items_for: its gimmick variant number is the fixed item slot.
    family_mechanic, variant, stage = key.removeprefix('p_exam_gimmick-').rsplit('-', 2)
    stem = f'pitem_{family_mechanic}-{int(variant)}-{stage}'
    items = tuple(row for row in _tower_catalog_rows('ProduceItem', f'{stem}*')
                  if row['id'] == stem or re.fullmatch(re.escape(stem) + r'-\d+', row['id']))
    return TowerStageRules(key, _tower_stage_rows(layer)[key], items)


_STAGE15_ITEM = 'pitem_tower_001-exam_review-1-stage_015'


@lru_cache(maxsize=1)
def _stage_native_rows():
    tables = _native_master()
    return {name: {row['id']: row for row in tables[name]} for name in (
        'ProduceExamEffect', 'ProduceExamTrigger', 'ProduceExamStatusEnchant', 'ProduceCardSearch')}


@lru_cache(maxsize=256)
def fixed_stage_item_effects(item_id):
    found, = _tower_catalog_rows('ProduceItem', item_id)
    return tuple(row for effect in found['produceItemEffectIds']
                 for row in _tower_catalog_rows('ProduceItemEffect', effect))


def fixed_stage_item_enchant(item_id):
    rows = fixed_stage_item_effects(item_id)
    if len(rows) != 1:
        return None
    effect = rows[0]
    if (effect['effectType'], effect['effectTurn'], effect.get('effectCount', 0)) != (
            'ProduceItemEffectType_ExamStatusEnchant', -1, 0):
        return None
    return effect['produceExamStatusEnchantId']


def _links(row):
    for field, table in (
        ('playEffects', 'ProduceExamEffect'), ('produceExamEffectIds', 'ProduceExamEffect'),
        ('chainProduceExamEffectIds', 'ProduceExamEffect'), ('moveProduceExamEffectIds', 'ProduceExamEffect'),
        ('produceCardGrowEffectIds', 'ProduceCardGrowEffect'), ('produceCardCustomizeIds', 'ProduceCardCustomize'),
        ('moveProduceExamTriggerIds', 'ProduceExamTrigger'),
    ):
        for link in row.get(field) or ():
            if isinstance(link, dict):
                yield 'ProduceExamTrigger', link.get('produceExamTriggerId')
                link = link['produceExamEffectId']
            yield table, link
    for field, table in (
        ('chainProduceExamEffectId', 'ProduceExamEffect'), ('produceExamStatusEnchantId', 'ProduceExamStatusEnchant'),
        ('produceExamTriggerId', 'ProduceExamTrigger'), ('playProduceExamTriggerId', 'ProduceExamTrigger'),
        ('produceCardSearchId', 'ProduceCardSearch'), ('produceCardSearchId2', 'ProduceCardSearch'),
    ):
        yield table, row.get(field)


@lru_cache(maxsize=1)
def _native_master():
    relative = Path(_HIRO) / 'native-master' / f'master-{_MASTER}.json'
    path = resolve_runtime_path(Path('.local/task275-rust/runtime') / relative,
                                Path('data/idol_road/rust_gnrpa') / relative)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _MASTER_SHA256:
        raise ValueError('idol_road_native_card_rules_revision_mismatch')
    return json.loads(data)['tables']


@lru_cache(maxsize=1)
def _native_cards():
    return {(row['id'], row['upgradeCount']): row for row in _native_master()['ProduceCard']}


@lru_cache(maxsize=1)
def tower_card_tables():
    """Extract these forms and their rule references from the frozen native master."""
    native = _native_master()
    ids = {wire for wire, _ in _WIRE_FORMS.values()}
    cards = [row for row in native['ProduceCard'] if row['id'] in ids]
    if {(row['id'], row['upgradeCount']) for row in cards} != {(wire, up) for wire in ids for up in range(4)}:
        raise ValueError('idol_road_native_card_forms_incomplete')
    rows = defaultdict(dict)
    for table, entries in native.items():
        for row in entries:
            rows[table].setdefault(row['id'], []).append(row)
    result = defaultdict(list, ProduceCard=cards)
    queue = deque(link for card in cards for link in _links(card))
    queue.extend(('ProduceExamStatusEnchant', enchant) for _, enchant, _ in _ITEMS.values())
    for layer in range(15, 28):
        for key, stage_rows in _tower_stage_rows(layer).items():
            queue.extend(('ProduceExamEffect', row['produceExamEffectId']) for row in stage_rows.values())
            stage = tower_stage_rules(layer, key)
            queue.extend(('ProduceExamStatusEnchant', effect['produceExamStatusEnchantId'])
                         for item in stage.fixed_items for effect in fixed_stage_item_effects(item['id']))
    visited = set()
    while queue:
        table, key = queue.popleft()
        if not key or (table, key) in visited:
            continue
        visited.add((table, key))
        for row in rows[table].get(key, ()):
            result[table].append(row)
            queue.extend(_links(row))
    return dict(result)


def extend_uinet_master(master):
    """Add the shared Rust rules data to a native exam policy's model master."""
    tables = _native_master()
    for row in tables['ProduceCard']:
        key = row['id'], row['upgradeCount']
        master.cards.setdefault(key, {k: v for k, v in row.items() if k != 'upgradeCount'})
    for table, name in (
        ('ProduceExamEffect', 'effects'), ('ProduceExamTrigger', 'triggers'),
        ('ProduceExamStatusEnchant', 'enchants'), ('ProduceCardSearch', 'searches'),
        ('ProduceCardGrowEffect', 'grow_effects'), ('ProduceCardCustomize', 'customizes'),
        ('ProduceDrink', 'drinks'), ('ProduceDrinkEffect', 'drink_effects'),
    ):
        target = getattr(master, name)
        for row in tables.get(table, ()):
            key = (row['id'], row['customizeCount']) if table == 'ProduceCardCustomize' else row['id']
            target.setdefault(key, row)
    return master


def tower_card_form(definition, upgraded):
    form = _WIRE_FORMS.get(definition.card_id)
    if form is None or form[1] != int(upgraded):
        return None
    row = _native_cards()[form]
    cost, amount = _card_cost(row)
    if (definition.name, definition.cost_kind, definition.stamina_cost) == (row['name'], cost, amount):
        return (*form, cost.value)
    return None


def _card_cost(row):
    kind = row['costType']
    if kind:
        return {2: CardCostKind.GOOD_IMPRESSION, 3: CardCostKind.MOTIVATION}[kind], row['costValue']
    if row['forceStamina']:
        return CardCostKind.STAMINA, row['forceStamina']
    return CardCostKind.NORMAL, row['stamina']


def tower_enchant_mapping(ref):
    if ref.source_kind.value == 'ITEM' and ref.persistent and ref.definition_effect_index == 0:
        if ref.card_id in _ITEMS:
            _, enchant, uses = _ITEMS[ref.card_id]
            phase = _item_phase(enchant)
            return enchant, 'remaining_uses', phase.value, int(phase is not SchedulePhase.CARD_USED), uses, None, -1, enchant
    if (ref.source_kind.value, ref.card_id, ref.definition_effect_index, ref.persistent) == ('CARD', '732', 2, False):
        row = _native_cards()[_SUPPORT_DRAW_FORM]
        installer = row['playEffects'][2]['produceExamEffectId']
        return None, 'timer_not_in_protocol', SchedulePhase.TURN.value, 1, 1, None, 0, installer
    if ref.source_kind.value != 'CARD' or ref.persistent or ref.card_id not in _WIRE_FORMS:
        return None
    row = _native_cards()[_WIRE_FORMS[ref.card_id]]
    if ref.definition_effect_index >= len(row['playEffects']):
        return None
    installer_id = row['playEffects'][ref.definition_effect_index]['produceExamEffectId']
    data = _rule_rows()
    installer = data['ProduceExamEffect'][installer_id]
    if installer['effectType'] not in {22, 219}:
        return None
    identifier = installer['produceExamStatusEnchantId']
    trigger = data['ProduceExamTrigger'][data['ProduceExamStatusEnchant'][identifier]['produceExamTriggerId']]
    phase, = trigger['phaseTypes']
    uses, turns = installer['effectCount'] or None, _duration(installer)
    if installer_id == _ENCORE:
        return identifier, 'remaining_uses', _ENCORE_PHASE.value, 1, uses, None, -1, installer_id
    phase = _NATIVE_PHASES[phase]
    return (identifier, 'tower_enchant', phase.value, int(phase is not SchedulePhase.CARD_USED),
            uses, turns, -1, installer_id)


@lru_cache(maxsize=1)
def _rule_rows():
    tables = tower_card_tables()
    return {name: {row['id']: row for row in tables[name]} for name in (
        'ProduceExamEffect', 'ProduceExamTrigger', 'ProduceExamStatusEnchant', 'ProduceCardSearch')}


def _duration(row):
    return row['effectTurn'] if row['effectTurn'] >= 0 else None


def _item_phase(enchant_id):
    tables = _rule_rows()
    enchant = tables['ProduceExamStatusEnchant'][enchant_id]
    phase, = tables['ProduceExamTrigger'][enchant['produceExamTriggerId']]['phaseTypes']
    return {25: SchedulePhase.CARD_USED, 27: SchedulePhase.AFTER_START_OF_TURN}[phase]


def tower_item_catalog(catalog, metadata, *, stage=None, plan=PlanType.LOGIC):
    definitions = {}
    tables = _rule_rows()
    for local, (name, enchant_id, limit) in _ITEMS.items():
        enchant = tables['ProduceExamStatusEnchant'][enchant_id]
        trigger_id = enchant['produceExamTriggerId']
        trigger = tables['ProduceExamTrigger'][trigger_id]
        phase = _item_phase(enchant_id)
        counter = None
        if phase is SchedulePhase.CARD_USED:
            interval, = trigger['phaseValues']
            condition = EffectCondition(ConditionKind.EFFECT_COUNTER_MOD_EQ, modulus=interval, remainder=interval - 1)
            counter = EffectCondition(ConditionKind.EFFECT_COUNTER_MIN, threshold=0)
        else:
            condition = _local_condition(trigger_id, tables)
        children = tuple(replace(_local_effect(tables['ProduceExamEffect'][eid], tables), condition=condition)
                         for eid in enchant['produceExamEffectIds'])
        schedule = EffectSpec(EffectKind.SCHEDULE, schedule_phase=phase, schedule_limit=limit,
                              delay_turns=int(phase is not SchedulePhase.CARD_USED), scheduled_effects=children,
                              schedule_counter_condition=counter)
        definition = ItemDefinition(local, name, (schedule,), PlanType.LOGIC)
        definition.validate()
        definitions[local] = definition
    for row in stage.fixed_items if stage else ():
        identity = row['id']
        enchant = fixed_stage_item_enchant(identity)
        if enchant is not None and _stage_enchant(enchant) is not None:
            definitions[identity] = ItemDefinition(identity, row['name'], (), plan)
    added = tuple(dict(id=key, name=value.name, mode='stage', plan='logic',
                       upgraded='TRUE' if value.name.endswith('+') else 'FALSE')
                  for key, value in definitions.items() if all(row['id'] != key for row in metadata))
    return replace(catalog, items={**catalog.items, **definitions}, total_rows=catalog.total_rows + len(added),
                   unsupported=tuple(row for row in catalog.unsupported if row.item_id not in definitions)), (*metadata, *added)


def _local_condition(trigger_id, tables):
    if not trigger_id:
        return None
    trigger = tables['ProduceExamTrigger'][trigger_id]
    condition, = trigger['fieldStatusTypes']
    threshold, = trigger['fieldStatusValues']
    return EffectCondition({34: ConditionKind.GENKI_MIN, 35: ConditionKind.GOOD_IMPRESSION_MIN,
                            41: ConditionKind.TURNS_REMAINING_MAX}[condition], threshold=threshold)


def _local_schedule(row, tables):
    enchant = tables['ProduceExamStatusEnchant'][row['produceExamStatusEnchantId']]
    trigger = tables['ProduceExamTrigger'][enchant['produceExamTriggerId']]
    phase, = trigger['phaseTypes']
    children = tuple(_local_effect(tables['ProduceExamEffect'][eid], tables) for eid in enchant['produceExamEffectIds'])
    condition = None
    if phase == 23:
        group, = tables['ProduceCardSearch'][trigger['produceCardSearchId']]['effectGroupIds']
        condition = EffectCondition(ConditionKind.USED_CARD_EFFECT_TAG, value=group)
        interval, = trigger['phaseValues']
        # The shared scheduler evaluates the nested effect before advancing its counter.
        eligible = EffectCondition(ConditionKind.EFFECT_COUNTER_MOD_EQ, modulus=interval, remainder=interval - 1)
        children = tuple(replace(child, condition=EffectCondition(ConditionKind.ALL, all_of=(condition, eligible)))
                         for child in children)
    return EffectSpec(EffectKind.SCHEDULE, schedule_phase=_NATIVE_PHASES[phase],
                      delay_turns=int(phase != 23), schedule_limit=row['effectCount'] or None,
                      schedule_ttl=_duration(row), scheduled_effects=children, schedule_counter_condition=condition)


def _local_effect(row, tables):
    kind = row['effectType']
    if kind == 1:
        return EffectSpec(EffectKind.SCORE, row['effectValue1'], repeats=row['effectCount'])
    if kind == 3:
        return EffectSpec(EffectKind.GENKI, row['effectValue1'])
    if kind == 119:
        return EffectSpec(EffectKind.FIXED_GENKI, row['effectValue1'])
    if kind == 49:
        return EffectSpec(EffectKind.FULL_POWER, row['effectValue1'])
    if kind == 4:
        return EffectSpec(EffectKind.DRAW, row['effectValue1'])
    if kind == 6 and row['targetProduceCardId'] == _FAMILIES[757][0]:
        if (row['movePositionType'], row['targetUpgradeCount'], row['pickCountMin'], row['pickCountMax']) != (2, 0, 1, 1):
            raise ValueError('idol_road_generated_card_rule_unexpected')
        return EffectSpec(EffectKind.ADD_CARD_TO_DECK, card_id='757')
    if kind == 22:
        return _local_schedule(row, tables)
    if kind == 31:
        return EffectSpec(EffectKind.GOOD_IMPRESSION, row['effectValue1'])
    if kind == 151:
        return EffectSpec(EffectKind.GOOD_IMPRESSION, -row['effectValue1'])
    if kind == 28:
        return EffectSpec(EffectKind.STAMINA, row['effectValue1'])
    if kind == 63:
        return EffectSpec(EffectKind.TURNS_REMAINING, 1)
    if kind == 42:
        return EffectSpec(EffectKind.MOTIVATION, row['effectValue1'])
    if kind == 14:
        return EffectSpec(EffectKind.EXTRA_ACTION, row['effectCount'])
    if kind == 66:
        return EffectSpec(EffectKind.ANTI_DEBUFF, row['effectCount'])
    if kind == 39:
        return EffectSpec(EffectKind.SCORE_BUFF, row['effectValue1'] / 1000, duration=_duration(row))
    if kind in {36, 177, 182}:
        status = {36: _REVIEW_VALUE_MULTIPLE, 177: 'review_gain_multiple', 182: 'review_multiple'}[kind]
        return EffectSpec(EffectKind.STATUS, row['effectValue1'], status_id=status, duration=_duration(row))
    if kind == 219 and row['id'] == _ENCORE:
        enchant = tables['ProduceExamStatusEnchant'][row['produceExamStatusEnchantId']]
        return EffectSpec(EffectKind.SCHEDULE, schedule_phase=_ENCORE_PHASE, schedule_limit=row['effectCount'],
            scheduled_effects=(EffectSpec(EffectKind.FREE_PLAY_RANDOM, card_selector=CardSelector(source_only=True),
                condition=_local_condition(enchant['produceExamTriggerId'], tables)),))
    if kind == 202:
        return EffectSpec(EffectKind.STATUS, row['effectValue1'], status_id=_REVIEW_REFUND)
    if kind in {19, 124, 125}:
        source = {19: EffectValueSource.GENKI, 124: EffectValueSource.GOOD_IMPRESSION,
                  125: EffectValueSource.MOTIVATION}[kind]
        return EffectSpec(EffectKind.SCORE, repeats=row['effectCount'], value_source=source,
                          value_scale=row['effectValue1'] / 1000)
    raise ValueError(f'idol_road_native_effect_not_supported:{row["id"]}')


_STAGE_COUNTERS = {
    'ReviewUp': 'good_impression_turns',
    'CardPlayAggressiveUp': 'motivation',
    'LessonBuffUp': 'concentration',
    'ParameterBuffUp': 'good_condition_turns',
    'BlockUp': 'genki',
    'FullPowerPointGetSumUp': 'cumulative_full_power',
    'ConcentrationChangeCountUp': 'strength_times',
    'PreservationChangeCountUp': 'preservation_times',
}


def _stage_condition(row):
    kind = row['fieldStatusType'].rsplit('_', 1)[-1]
    value = row['fieldStatusValue']
    negated = row['fieldStatusCheckType'] == 'ProduceExamTriggerCheckType_Not'
    if kind == 'Unknown':
        return lambda state: not negated
    if kind == 'ParameterBuff':
        return lambda state: (state.good_condition_turns > 0) != negated
    if kind == 'StanceChangeCountUp':
        return lambda state: ((state.strength_times + state.preservation_times + state.full_power_times) >= value) != negated
    if kind == 'PreservationUp':
        return lambda state: (state.stance is Stance.LEISURE or state.stance in (
            Stance.CONSERVE_1, Stance.CONSERVE_2) and (value < 2 or state.stance is Stance.CONSERVE_2)) != negated
    if kind in _STAGE_COUNTERS:
        return lambda state: (getattr(state, _STAGE_COUNTERS[kind]) >= value) != negated
    if kind in ('StaminaUpMultiple', 'StaminaLessMultiple'):
        if kind == 'StaminaUpMultiple':
            return lambda state: (state.stamina * 1000 >= state.max_stamina * value) != negated
        return lambda state: (state.stamina * 1000 <= state.max_stamina * value) != negated
    if kind in ('PlayCardSkill', 'PlayCardLesson'):
        expected = 'mental' if kind == 'PlayCardSkill' else 'active'
        return lambda state: (state.last_played_card_type == expected) != negated
    return None


@lru_cache(maxsize=256)
def _stage_enchant(enchant_id):
    """Compile supported trigger shapes from the same rows Hiro's engine uses."""
    tables = _stage_native_rows()
    row = tables['ProduceExamStatusEnchant'].get(enchant_id)
    if row is None:
        return None
    trigger = tables['ProduceExamTrigger'][row['produceExamTriggerId']]
    phases = tuple(trigger['phaseTypes'])
    if phases in ((5,), (27,)) and not any(trigger[k] for k in (
            'fieldStatusTypes', 'produceCardSearchId', 'phaseValues')):
        try:
            effects = tuple(_local_effect(tables['ProduceExamEffect'][key], tables)
                            for key in row['produceExamEffectIds'])
        except ValueError:
            return None
        safe = {EffectKind.STAMINA, EffectKind.GENKI, EffectKind.FIXED_GENKI, EffectKind.SCORE,
                EffectKind.GOOD_IMPRESSION, EffectKind.MOTIVATION, EffectKind.FULL_POWER,
                EffectKind.TURNS_REMAINING, EffectKind.EXTRA_ACTION, EffectKind.ANTI_DEBUFF}
        if phases == (27,):
            safe.add(EffectKind.DRAW)
        if effects and all(e.kind in safe for e in effects):
            return phases[0], effects
    elif phases == (31,) and trigger['produceCardSearchId'] == 'p_card_search-target':
        if row['produceExamEffectIds'] == ['e_effect-exam_card_create_id-p_card-00-acc-0_002-0-deck_first-1_1']:
            return 31, ()
    elif phases == (3,) and trigger['produceCardSearchId'] == 'p_card_search-target-1-p_card-02-men-3_002':
        if row['produceExamEffectIds'] == ['e_effect-exam_card_move-p_card_search-trouble-deck_all-lost-all-0_0']:
            return 3, ()
    return None


def stage_direct_effect(row):
    tables = _stage_native_rows()
    rule = tables['ProduceExamEffect'].get(row['produceExamEffectId'])
    if rule is None:
        return None
    if rule['effectType'] == 22:
        enchant = rule['produceExamStatusEnchantId']
        if _stage_enchant(enchant) and (rule['effectCount'], rule['effectTurn']) == (0, -1):
            return enchant
        return None
    if rule['effectType'] == 10:
        return EffectSpec(EffectKind.CONCENTRATION, rule['effectValue1'])
    if rule['effectType'] == 2:
        return EffectSpec(EffectKind.GOOD_CONDITION, rule['effectTurn'])
    if rule['effectType'] in {1, 3, 14, 19, 28, 31, 36, 39, 42, 49, 63, 66, 119, 124, 125, 151, 177, 182}:
        return _local_effect(rule, tables)
    return None


def stage_rule_supported(row):
    return _stage_condition(row) is not None and stage_direct_effect(row) is not None


def tower_card_definitions():
    tables = _rule_rows()
    result = {}
    for local, form in _WIRE_FORMS.items():
        row = _native_cards()[form]
        base = int(local) - form[1]
        cost, amount = _card_cost(row)
        effects = tuple(replace(_local_effect(tables['ProduceExamEffect'][link['produceExamEffectId']], tables),
                                condition=_local_condition(link['produceExamTriggerId'], tables))
                        for link in row['playEffects'])
        definition = CardDefinition(local, row['name'], PlanType.LOGIC, stamina_cost=amount, cost_kind=cost,
            effects=effects, limited=row['playMovePositionType'] == 6,
            condition=_local_condition(row['playProduceExamTriggerId'], tables),
            card_type={1: 'active', 2: 'mental'}[row['category']], rarity='SSR', base_card_id=str(base),
            upgrade_card_id=str(base + 1) if not form[1] else None, source_type=_FAMILIES[base][1],
            effect_groups=tuple(row['effectGroupIds']))
        definition.validate()
        result[local] = definition
    return result


@register_copy_record
@dataclass(frozen=True)
class TowerCardState(StrategyState):
    review_consumed: int = 0
    encore_installed: tuple[str, ...] = ()
    review_fresh: bool = False
    review_gain_buffs: tuple[TimedModifier, ...] = ()
    review_score_buffs: tuple[TimedModifier, ...] = ()
    extra_turns: int = 0
    slump_turns: int = 0
    review_at_turn_end: int = 0
    last_played_card_type: str | None = None
    blocked_card_type: str | None = None
    native_stage_enchants: tuple[str, ...] = ()

    def projected_stage_enchants(self):
        return [[key, -1, -1] for key in self.native_stage_enchants]

    def review_statuses(self):
        result = [[kind, modifier.amount / 1000, -1 if modifier.remaining_turns is None
                   else modifier.remaining_turns + int(modifier.fresh)]
                for kind, field in _REVIEW_MODIFIERS.items() for modifier in getattr(self, field)]
        if self.extra_turns:
            result.append(['extra_turn', self.extra_turns, -1])
        if self.slump_turns:
            result.append(['slump', 1, self.slump_turns])
        return result


class TowerCardKernel(PythonNiaProKernel):
    def __init__(self, cards, drinks=None, items=None):
        from .high_score.gkms_master_forms import card_form
        from .high_score.gkms_shadow_mappings import CARD_FORMS

        registered = {}
        self._end_turn_lost = set()
        self.stage_rules = {}
        self.native_stage_rules = {}
        for local, definition in cards.items():
            form = CARD_FORMS.get(local) or card_form(definition, definition.name.endswith('+'))
            row = _native_cards().get(form[:2]) if form else None
            if row is not None:
                definition = replace(definition, effect_groups=tuple(row['effectGroupIds']))
                if row['isEndTurnLost']:
                    self._end_turn_lost.add(local)
            registered[local] = definition
        super().__init__(registered, drinks, items)

    def use_stage_timeline(self, timeline):
        self.stage_rules = {turn: rule for turn, row in timeline.items()
                            if (rule := tower_field_rule(row.description)) is not None}

    def use_native_stage(self, stage):
        self.native_stage_rules = {turn: (_stage_condition(row), stage_direct_effect(row))
                                   for turn, row in stage.rows.items() if stage_rule_supported(row)}

    def initialize_stage_entry(self, state, *, plan_type):
        state = super().initialize_stage_entry(state, plan_type=plan_type)
        fixed = tuple(enchant for item in state.items if item.startswith('pitem_tower_')
                      and (enchant := fixed_stage_item_enchant(item)) is not None)
        state = replace(state, native_stage_enchants=fixed)
        # Current-turn numbers at entry already come from the opening HUD.
        rule = self.native_stage_rules.get(1)
        if rule and isinstance(rule[1], str):
            state, _ = self._apply_native_stage_rule(state, *rule)
        return state

    def _card_secondary_ready(self, state, definition, source_id):
        if state.blocked_card_type and state.blocked_card_type == definition.card_type:
            return False
        return super()._card_secondary_ready(state, definition, source_id)

    def _trigger_event_phase(self, branches, phase, *, trigger_context=None, condition_state=None):
        if phase is SchedulePhase.CARD_USED and trigger_context is not None:
            branches = tuple(replace(branch, state=replace(branch.state,
                             last_played_card_type=trigger_context.used_card_type)) for branch in branches)
        if phase is SchedulePhase.AFTER_CARD_USED and trigger_context is not None:
            branches = tuple(self._stage_card_moved(branch, trigger_context) for branch in branches)
        if phase is SchedulePhase.END_OF_TURN:
            branches = self._fire_stage_enchants(branches, 5)
        return super()._trigger_event_phase(branches, phase, trigger_context=trigger_context,
                                            condition_state=condition_state)

    def _fire_stage_enchants(self, branches, phase):
        if not branches or not branches[0].state.native_stage_enchants:
            return branches
        result = []
        for branch in branches:
            pending = (branch,)
            for enchant in branch.state.native_stage_enchants:
                trigger, children = _stage_enchant(enchant)
                if trigger == phase:
                    for child in children:
                        pending = self._apply_effect(pending, child, source_id=enchant)
            result.extend(pending)
        return tuple(result)

    def _stage_card_moved(self, branch, context):
        state = branch.state
        if not state.native_stage_enchants:
            return branch
        events = branch.events
        source = context.used_card_instance_id
        phases = {_stage_enchant(enchant)[0]: enchant for enchant in state.native_stage_enchants}
        # Rust fires the lost-zone trigger after the played card resolves,
        # before its card-play-after enchant, whose nested move does not refire it.
        if 31 in phases and any(card.instance_id == source for card in state.deck.removed):
            generated, = self._add_card_to_deck(state, '23', source_id=phases[31], on_top=True)
            state = generated.state
            events += generated.events
        if 3 in phases and context.used_card_id in {'170', '171'}:
            deck = state.deck
            def clean(cards):
                return tuple(card for card in cards if self.cards[card.card_id].card_type != 'trouble')

            trouble = tuple(card for card in (*deck.hand, *deck.draw_pile, *deck.discard, *deck.held)
                            if self.cards[card.card_id].card_type == 'trouble')
            state = replace(state, deck=deck.replace_zones(hand=clean(deck.hand), draw_pile=clean(deck.draw_pile),
                    discard=clean(deck.discard), held=clean(deck.held), removed=(*deck.removed, *trouble)))
            events += tuple(RuleEvent('TOWER_STAGE_MOVE_LOST', phases[3], detail=card.instance_id)
                            for card in trouble)
        return replace(branch, state=state, events=events)

    def _apply_native_stage_rule(self, state, condition, effect):
        if not condition(state):
            return state, ()
        if isinstance(effect, str):
            return replace(state, native_stage_enchants=(*state.native_stage_enchants, effect)), (
                RuleEvent('TOWER_STAGE_ENCHANT', 'stage', detail=effect),)
        branch, = self._apply_effect((TransitionBranch(1., state, ()),), effect, source_id='stage')
        after = branch.state
        if effect.kind is EffectKind.SCORE_BUFF:
            after = replace(after, score_buffs=tuple(replace(buff, fresh=False) for buff in after.score_buffs))
        return after, branch.events

    def _apply_stage_rule(self, state):
        if native := self.native_stage_rules.get(state.turn):
            return self._apply_native_stage_rule(state, *native)
        rule = self.stage_rules.get(state.turn)
        if rule is None:
            return state, ()
        kind, amount, condition = rule
        if kind in {'block_active', 'block_mental'} and state.last_played_card_type == condition:
            for index, status in enumerate(state.statuses):
                if status.status_id == 'anti_debuff' and status.stacks > 0:
                    remainder = (replace(status, stacks=status.stacks - 1),) if status.stacks > 1 else ()
                    statuses = (*state.statuses[:index], *remainder, *state.statuses[index + 1:])
                    return replace(state, statuses=statuses), (RuleEvent('DEBUFF_NULLIFIED', 'stage', 1, kind),)
            return replace(state, blocked_card_type=kind.removeprefix('block_')), (
                RuleEvent('TOWER_STAGE_EFFECT', 'stage', amount, kind),)
        if kind == 'review_at_end' and state.motivation >= condition:
            return replace(state, review_at_turn_end=state.review_at_turn_end + amount), (
                RuleEvent('TOWER_STAGE_EFFECT', 'stage', amount, kind),)
        if kind == 'extra_actions' and state.good_impression_turns >= condition:
            return replace(state, actions_remaining=state.actions_remaining + amount), (
                RuleEvent('EXTRA_ACTION', 'stage', amount),)
        if kind == 'extra_turns' and state.good_impression_turns >= condition:
            after, event = self._apply_non_draw_effect(
                state, EffectSpec(EffectKind.TURNS_REMAINING, amount), source_id='stage')
            return after, (event,)
        if kind == 'review_multiple' and state.good_impression_turns >= condition:
            after, event = self._update_review(state, state.good_impression_turns * (amount - 1),
                                               source_id='stage', fixed=True)
            return after, (event,)
        if ((kind == 'slump' and state.score < condition)
                or (kind == 'review_slump' and state.good_impression_turns <= condition)):
            return replace(state, slump_turns=amount), (RuleEvent('TOWER_STAGE_EFFECT', 'stage', amount, kind),)
        if kind == 'score_buff':
            threshold, duration = condition
            if state.score >= threshold:
                score_buffs = self._add_timed_modifier(state.score_buffs, amount=amount / 100, duration=duration)
                score_buffs = tuple(replace(buff, fresh=False) if buff.remaining_turns == duration else buff
                                    for buff in score_buffs)
                return replace(state, score_buffs=score_buffs), (
                    RuleEvent('SCORE_BUFF', 'stage', amount / 100, str(duration)),)
        if kind == 'review' and (condition is None or state.good_impression_turns >= condition):
            after, event = self._update_review(state, amount, source_id='stage')
            return after, (event,)
        if kind == 'review_score' and (condition is None or state.good_impression_turns >= condition):
            points = math.ceil(float(float32(state.good_impression_turns) * float32(amount / 100)))
            gain = self.score_effect_gain(state, EffectSpec(EffectKind.SCORE), 'stage', amount=points)
            return replace(state, score=state.score + gain), (RuleEvent('SCORE', 'stage', gain),)
        return state, ()

    def score_effect_gain(self, state, effect, source_id, *, amount=None):
        if state.slump_turns:
            return 0
        return super().score_effect_gain(state, effect, source_id, amount=amount)

    @staticmethod
    def _review_multiplier(modifiers):
        multiplier = float32(1)
        for modifier in modifiers:
            multiplier += float32(modifier.amount / 1000)
        return multiplier

    def _review_turn_end(self, state):
        review = state.good_impression_turns
        amount = math.ceil(float(float32(review) * self._review_multiplier(state.review_score_buffs)))
        gain = self.score_effect_gain(state, EffectSpec(EffectKind.SCORE), 'system', amount=amount)
        remaining = review - int(review > 0 and not state.review_fresh and state.turns_remaining > 1)
        return gain, remaining

    def _complete_end_turn(self, state, *, plan_type):
        events = ()
        if state.review_at_turn_end:
            state, event = self._update_review(state, state.review_at_turn_end, source_id='stage')
            events = (event,)
        if state.good_impression_turns and not state.review_fresh and state.turns_remaining > 1:
            state = replace(state, review_consumed=state.review_consumed + 1)
        # An enchant with one passing turn left expires before the next StartPlay.
        state = replace(state, scheduled_effects=tuple(ref for ref in state.scheduled_effects
                        if not (ref.card_id in _WIRE_FORMS and ref.remaining_turns == 1 and not ref.duration_fresh)))
        branches = super()._complete_end_turn(state, plan_type=plan_type)
        return tuple(replace(branch, events=events + branch.events) for branch in branches) if events else branches

    def _advance_turn_resolution(self, branch, frame):
        state = branch.state
        events = branch.events
        if frame.stage is TurnResolutionStage.BEFORE_START:
            refs = tuple(replace(ref, effect_counter=0) if ref.source_id == f'item:{_PER_TURN_ITEM}' else ref
                         for ref in state.scheduled_effects)
            state = replace(state, review_gain_buffs=self._decrement_timed_modifiers(state.review_gain_buffs),
                            review_score_buffs=self._decrement_timed_modifiers(state.review_score_buffs),
                            scheduled_effects=refs, slump_turns=max(0, state.slump_turns - 1), blocked_card_type=None)
            state, seen = self._apply_stage_rule(state)
            events += seen
        elif frame.stage is TurnResolutionStage.COMPLETE:
            refs = tuple(replace(ref, duration_fresh=False) if ref.card_id in _WIRE_FORMS else ref
                         for ref in state.scheduled_effects)
            state = replace(state, review_fresh=False, scheduled_effects=refs,
                            review_gain_buffs=tuple(replace(buff, fresh=False) for buff in state.review_gain_buffs),
                            review_score_buffs=tuple(replace(buff, fresh=False) for buff in state.review_score_buffs))
        elif frame.stage is TurnResolutionStage.AFTER_START and state.native_stage_enchants:
            return tuple(result for applied in self._fire_stage_enchants((branch,), 27)
                         for result in super(TowerCardKernel, self)._advance_turn_resolution(applied, frame))
        return super()._advance_turn_resolution(replace(branch, state=state, events=events), frame)

    def _turn_end_deck(self, deck):
        lost = tuple(card for card in deck.hand if card.card_id in self._end_turn_lost)
        discarded = tuple(card for card in deck.hand if card.card_id not in self._end_turn_lost)
        return deck.replace_zones(hand=(), removed=deck.removed + lost, discard=deck.discard + discarded)

    def _turn_start_draw_count(self, state):
        return min(3, super()._turn_start_draw_count(state))

    def _pay_declared_non_full_power_cost(self, state, card, definition):
        updated, events = super()._pay_declared_non_full_power_cost(state, card, definition)
        if definition.cost_kind is CardCostKind.GOOD_IMPRESSION:
            updated = replace(updated, review_fresh=state.review_fresh and updated.good_impression_turns > 0,
                              review_consumed=state.review_consumed +
                              state.good_impression_turns - updated.good_impression_turns)
        return updated, events

    def _update_review(self, state, amount, *, source_id, fixed=False):
        if amount > 0 and not fixed:
            amount = math.ceil(float(float32(amount) * self._review_multiplier(state.review_gain_buffs)))
        before = state.good_impression_turns
        after = max(0, before + amount)
        return replace(state, good_impression_turns=after,
                       review_fresh=after > 0 and (before == 0 or state.review_fresh),
                       review_consumed=state.review_consumed + max(0, before - after)), RuleEvent(
            'GOOD_IMPRESSION', source_id, after - before)

    def _apply_non_draw_effect(self, state, effect, *, source_id, **kwargs):
        if effect.kind is EffectKind.GOOD_IMPRESSION:
            return self._update_review(state, int(self.effect_amount(state, effect)), source_id=source_id)
        if effect.kind is EffectKind.TURNS_REMAINING:
            added = int(self.effect_amount(state, effect))
            attributes = state.turn_attributes
            future = state.future_score_multipliers
            forecast = {}
            if attributes:
                forecast['turn_attributes'] = attributes + (attributes[-1],) * added
                forecast['future_score_multipliers'] = future + ((future[-1] if future else state.score_multiplier),) * added
            return replace(state, turns_remaining=state.turns_remaining + added,
                           extra_turns=state.extra_turns + added, **forecast), RuleEvent('TURNS_REMAINING', source_id, added)
        if effect.kind is EffectKind.STATUS:
            if effect.status_id in {_REVIEW_REFUND, _REVIEW_VALUE_MULTIPLE}:
                fixed = effect.status_id == _REVIEW_VALUE_MULTIPLE
                value = state.good_impression_turns if fixed else state.review_consumed
                amount = math.ceil(float(float32(value) * float32(effect.amount / 1000)))
                return self._update_review(state, amount, source_id=source_id, fixed=fixed)
            if effect.status_id in _REVIEW_MODIFIERS:
                field = _REVIEW_MODIFIERS[effect.status_id]
                buffs = list(getattr(state, field))
                index = next((index for index in reversed(range(len(buffs)))
                              if buffs[index].remaining_turns == effect.duration), None)
                if index is None:
                    buffs.append(TimedModifier(effect.amount, effect.duration))
                else:
                    buffs[index] = replace(buffs[index], amount=buffs[index].amount + effect.amount)
                return replace(state, **{field: tuple(buffs)}), RuleEvent(
                    'STATUS', source_id, effect.amount, effect.status_id)
        if effect.kind is EffectKind.SCHEDULE and kwargs.get('source_card_id') in {'871', '872'}:
            if source_id in state.encore_installed:
                return state, RuleEvent('ONCE_PLAY_EFFECT_SKIPPED', source_id)
            state = replace(state, encore_installed=(*state.encore_installed, source_id))
        updated, event = super()._apply_non_draw_effect(state, effect, source_id=source_id, **kwargs)
        if (effect.kind is EffectKind.SCHEDULE and effect.schedule_ttl is not None
                and kwargs.get('source_card_id') in _WIRE_FORMS):
            *before, scheduled = updated.scheduled_effects
            updated = replace(updated, scheduled_effects=(*before, replace(scheduled, duration_fresh=True)))
        return updated, event

    def _add_card_to_deck(self, state, card_id, *, source_id, on_top=False):
        # The Rust create effect places a new unupgraded copy ahead of any known prefix.
        return super()._add_card_to_deck(state, card_id, source_id=source_id, on_top=on_top or card_id == '757')

    def _free_play_random(self, state, selector, *, source_id):
        if selector.source_only and source_id in state.encore_installed:
            card = next((card for card in state.deck.zones if card.instance_id == source_id), None)
            if card is None:
                return (TransitionBranch(1., state, (RuleEvent('FREE_PLAY_RANDOM_EMPTY', source_id),)),)
            return self._execute_inline_free_card(state, card, source_id=source_id)
        return super()._free_play_random(state, selector, source_id=source_id)
