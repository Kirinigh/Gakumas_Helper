"""Resolve ordinary compiled card and drink forms against the pinned model master."""

from functools import lru_cache
from collections import defaultdict
from unicodedata import normalize

from .kernel import CardCostKind, CardDefinition
from ..hif_gkms_runtime import uinet21_master

PLAN_CODES = {'SENSE': 2, 'LOGIC': 3, 'ANOMALY': 4}
_MASTER_PLANS = {None: 'Common', 'SENSE': 'Plan1', 'LOGIC': 'Plan2', 'ANOMALY': 'Plan3'}
_MASTER_CATEGORIES = {'active': 'ActiveSkill', 'mental': 'MentalSkill', 'trouble': 'Trouble'}
_MASTER_RARITIES = {'L': 'LEGEND', 'T': 'N'}
_CARD_TITLE_ALIASES = {
    '愛をこめて': '愛を込めて',
    '月明りに包まれて': '月明かりに包まれて',
    '夢と現境界線': '夢と現の境界線',
}
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


def _card_title(name):
    title = normalize('NFKC', name)
    base = title.removesuffix('+')
    return _CARD_TITLE_ALIASES.get(base, base) + title[len(base):]


def _same_rarity(entry, rarity):
    rarity = rarity.upper()
    return entry['rarity'].rsplit('_', 1)[-1].upper() == _MASTER_RARITIES.get(rarity, rarity)


@lru_cache(maxsize=2)
def _by_name(native=False):
    if native:
        from .tower_gnrpa import load_native_tower

        _, upstream = load_native_tower()
        catalog = upstream.master()
        master = {name: catalog.rows(name) for name in ('ProduceCard', 'ProduceDrink')}
    else:
        master = uinet21_master()
    cards, drinks = defaultdict(list), defaultdict(list)
    for row in master['ProduceCard']:
        cards[_card_title(row['name']), row['planType'], row['upgradeCount']].append(row)
    for row in master['ProduceDrink']:
        drinks[row['name'], row['planType']].append(row['id'])
    return cards, drinks


def card_form(definition, upgraded, *, native=False):
    if (definition is None or type(upgraded) is not bool
            or definition.canonical_card_id not in (None, definition.card_id)
            or definition.card_type not in _MASTER_CATEGORIES or not definition.rarity):
        return None
    direct = native and definition.card_id.startswith('p_card-')
    if not direct and not definition.card_id.isdecimal():
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
    cards, _ = _by_name(native)
    candidates = [row['id'] for row in cards.get((_card_title(definition.name), _plan(definition), grade), ())
                  if row['category'] == 'ProduceCardCategory_' + _MASTER_CATEGORIES[definition.card_type]
                  and _same_rarity(row, definition.rarity)
                  and (not direct or row['id'] == definition.card_id.removesuffix('+'))
                  and (row['stamina'], row['forceStamina'], row['costType'].removeprefix('ExamCostType_'),
                       row['costValue']) == expected_cost]
    if len(candidates) == 1:
        return candidates[0], grade, kind
    if candidates:
        return None
    from ..idol_road_card_rules import tower_card_form

    return tower_card_form(definition, upgraded)


def _native_cost(entry):
    status = entry['costType'].removeprefix('ExamCostType_')
    if status == 'Unknown':
        kind = CardCostKind.STAMINA if entry['forceStamina'] else CardCostKind.NORMAL
        return kind, entry['forceStamina'] or entry['stamina']
    return next(((CardCostKind[local], entry['costValue']) for local, wire in _TYPED_COSTS.items()
                 if wire == status), None)


def native_card_metadata(rows, plan, *, include_master_only=False):
    """Bind local card IDs and, when requested, cards directly from Hiro's master."""
    master, _ = _by_name(True)
    variants = defaultdict(dict)
    matched = {}
    for row in rows:
        if row['plan'] not in ('free', plan.value.lower()) or row['type'] not in _MASTER_CATEGORIES:
            continue
        card_plan = None if row['plan'] == 'free' else plan
        plan_key = 'ProducePlanType_' + _MASTER_PLANS[card_plan.value if card_plan else None]
        grade = int(row['upgraded'].upper() == 'TRUE')
        candidates = [entry for entry in master.get((_card_title(row['name']), plan_key, grade), ())
                      if entry['category'] == 'ProduceCardCategory_' + _MASTER_CATEGORIES[row['type']]
                      and _same_rarity(entry, row['rarity'])]
        if len(candidates) != 1:
            continue
        entry, = candidates
        cost = _native_cost(entry)
        if cost is None:
            continue
        variants[entry['id']].setdefault(grade, []).append(row['id'])
        matched[row['id']] = row, entry, card_plan, *cost
    if include_master_only:
        plans = {'ProducePlanType_Common': None, 'ProducePlanType_' + _MASTER_PLANS[plan.value]: plan}
        categories = {'ProduceCardCategory_' + value: key for key, value in _MASTER_CATEGORIES.items()}
        for (_, plan_key, grade), entries in master.items():
            if plan_key not in plans or grade > 1:
                continue
            for entry in entries:
                wire = entry['id']
                card_type = categories.get(entry['category'])
                cost = _native_cost(entry)
                if not card_type or cost is None or grade in variants[wire]:
                    continue
                card_id = wire + '+' * grade
                variants[wire][grade] = [card_id]
                row = {'type': card_type, 'rarity': entry['rarity'].rsplit('_', 1)[-1], 'sourceType': None}
                matched[card_id] = row, entry, plans[plan_key], *cost
    definitions = {}
    for card_id, (row, entry, card_plan, kind, amount) in matched.items():
        forms = variants[entry['id']]
        grade = entry['upgradeCount']
        if len(forms[grade]) != 1:
            continue
        base, upper = forms.get(0, ()), forms.get(1, ())
        definitions[card_id] = CardDefinition(
            card_id, entry['name'], card_plan,
            stamina_cost=0 if kind is CardCostKind.FULL_POWER else amount,
            full_power_cost=amount if kind is CardCostKind.FULL_POWER else 0, cost_kind=kind,
            limited=entry['playMovePositionType'] == 'ProduceCardMovePositionType_Lost',
            card_type=row['type'], rarity=row['rarity'], source_type=row['sourceType'],
            base_card_id=base[0] if len(base) == 1 else card_id,
            upgrade_card_id=upper[0] if not grade and len(upper) == 1 else None,
        )
    return definitions


def drink_id(definition, *, native=False):
    if definition is None or not definition.drink_id.isdecimal():
        return None
    _, drinks = _by_name(native)
    candidates = drinks.get((definition.name, _plan(definition)), ())
    return candidates[0] if len(candidates) == 1 else None
