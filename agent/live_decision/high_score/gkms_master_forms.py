"""Resolve ordinary compiled card and drink forms against the pinned model master."""

from functools import lru_cache
from collections import defaultdict

from ..hif_gkms_runtime import uinet21_master

PLAN_CODES = {'SENSE': 2, 'LOGIC': 3, 'ANOMALY': 4}
_MASTER_PLANS = {None: 'Common', 'SENSE': 'Plan1', 'LOGIC': 'Plan2', 'ANOMALY': 'Plan3'}
_MASTER_CATEGORIES = {'active': 'ActiveSkill', 'mental': 'MentalSkill', 'trouble': 'Trouble'}
_TYPED_COSTS = {
    'FULL_POWER': 'ExamFullPowerPoint',
    'CONCENTRATION': 'ExamLessonBuff',
    'GOOD_CONDITION': 'ExamParameterBuff',
    'GOOD_IMPRESSION': 'ExamReview',
    'MOTIVATION': 'ExamCardPlayAggressive',
}


def _plan(definition):
    key = definition.plan.value if definition.plan is not None else None
    return 'ProducePlanType_' + _MASTER_PLANS[key]


@lru_cache(maxsize=1)
def _by_name():
    master = uinet21_master()
    cards, drinks = defaultdict(list), defaultdict(list)
    for row in master['ProduceCard']:
        cards[row['name'], row['planType'], row['upgradeCount']].append(row)
    for row in master['ProduceDrink']:
        drinks[row['name'], row['planType']].append(row['id'])
    return cards, drinks


def card_form(definition, upgraded):
    if (definition is None or type(upgraded) is not bool or not definition.card_id.isdecimal()
            or definition.canonical_card_id not in (None, definition.card_id)
            or definition.card_type not in _MASTER_CATEGORIES or not definition.rarity):
        return None
    grade = int(upgraded)
    if definition.name.endswith('+') != upgraded:
        return None
    kind = definition.cost_kind.value
    stamina = definition.stamina_cost
    if kind == 'NORMAL':
        expected_cost = (stamina, 0, 'Unknown', 0)
    elif kind == 'STAMINA':
        expected_cost = (0, stamina, 'Unknown', 0)
    else:
        amount = definition.full_power_cost if kind == 'FULL_POWER' else stamina
        expected_cost = (0, 0, _TYPED_COSTS[kind], amount)
    try:
        cards, _ = _by_name()
    except FileNotFoundError:
        return None
    rarity = definition.rarity.upper()
    # The GKH catalog denotes Trouble cards as T. The model uses normal rarity
    # for that category; category, plan, form, name and costs still must match.
    if rarity == 'T' and definition.card_type == 'trouble':
        rarity = 'N'
    candidates = [row['id'] for row in cards.get((definition.name, _plan(definition), grade), ())
                  if row['category'] == 'ProduceCardCategory_' + _MASTER_CATEGORIES[definition.card_type]
                  and row['rarity'].rsplit('_', 1)[-1].upper() == rarity
                  and (row['stamina'], row['forceStamina'], row['costType'].removeprefix('ExamCostType_'),
                       row['costValue']) == expected_cost]
    if len(candidates) == 1:
        return candidates[0], grade, kind
    return None


def drink_id(definition):
    if definition is None or not definition.drink_id.isdecimal():
        return None
    try:
        _, drinks = _by_name()
    except FileNotFoundError:
        return None
    candidates = drinks.get((definition.name, _plan(definition)), ())
    return candidates[0] if len(candidates) == 1 else None
