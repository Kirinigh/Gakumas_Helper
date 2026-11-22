"""Current-play cost-only support tiers from the already pinned master data.

This is a semantic compiler, not a card-name allowlist. Unknown effects,
conditions, permanent customizations and changed effects remain unsupported.
"""
import re
import json
from functools import lru_cache
from dataclasses import replace

from .kernel import EffectKind, EffectSpec, CardDefinition
from .contracts import PlanType
from .gkms_master_forms import card_form

SOURCE = 'gakumasu-diff@571dbb62601e78998cddeacdbce3ea1bc672d7fc'

@lru_cache(maxsize=1)
def _master():
    from ..hif_gkms_runtime import _master_path

    root = _master_path()
    cards = json.loads((root / 'ProduceCard.json').read_text(encoding='utf8'))
    effects = json.loads((root / 'ProduceExamEffect.json').read_text(encoding='utf8'))
    return ({(r['id'], r['upgradeCount']): r for r in cards}, {r['id']: r for r in effects})


def _effect(entry, effects):
    if set(entry) != {'produceExamEffectId', 'hideIcon', 'isOncePlayEffect'} or entry['hideIcon'] or entry['isOncePlayEffect']:
        raise ValueError('support master has a conditional or hidden effect')
    raw = effects[entry['produceExamEffectId']]
    forms = {
        'ProduceExamEffectType_ExamLessonBuff': (EffectKind.CONCENTRATION, 'effectValue1', '集中+{}'),
        'ProduceExamEffectType_ExamParameterBuff': (EffectKind.GOOD_CONDITION, 'effectTurn', '好調{}ターン'),
        'ProduceExamEffectType_ExamBlock': (EffectKind.GENKI, 'effectValue1', '元気+{}'),
    }
    if raw['effectType'] not in forms:
        raise ValueError('support master effect semantics are not compiled')
    kind, value_key, line = forms[raw['effectType']]
    defaults = dict(effectValue1=0, effectValue2=0, effectCount=0, effectTurn=0,
        targetUpgradeCount=0, targetExamEffectType='ProduceExamEffectType_Unknown',
        movePositionType='ProduceCardMovePositionType_Unknown',
        pickRangeType='ProducePickRangeType_Unknown', pickCountType='ProducePickCountType_Unknown',
        pickCountMin=0, pickCountMax=0, pickRangeType2='ProducePickRangeType_Unknown',
        pickCountType2='ProducePickCountType_Unknown', pickCountMin2=0, pickCountMax2=0)
    rest = {k: v for k, v in raw.items() if k not in ('id', 'effectType', 'effectGroupIds', value_key)}
    if rest != {k: v for k, v in defaults.items() if k != value_key} or type(raw[value_key]) is not int or raw[value_key] <= 0:
        raise ValueError('support master effect has additional semantics')
    return EffectSpec(kind, amount=raw[value_key]), line.format(raw[value_key])


def cost_only_support(ordinary, enhancement_count):
    """Return the effective definition and complete-body specification together."""
    form = card_form(ordinary, True)
    if (form is None or '@' in ordinary.card_id or form[1:] != (1, 'NORMAL')
            or type(enhancement_count) is not int or enhancement_count not in (2, 3)):
        raise ValueError('support master requires an uncustomized ordinary plus instance')
    cards, effects = _master()
    raw = cards[(form[0], 1)]
    effective = cards[(form[0], enhancement_count)]
    fields = set(('id upgradeCount name assetId isCharacterAsset rarity planType category stamina '
        'forceStamina costType costValue playEffects playMovePositionType moveEffectTriggerType '
        'isEndTurnLost isInitial isRestrict libraryHidden noDeckDuplication isReward '
        'unlockProducerLevel rentalUnlockProducerLevel evaluation isInitialDeckProduceCard '
        'effectGroupIds produceCardCustomizeIds maxCustomizeCount isConversion viewStartTime isLimited order').split())
    if set(raw) != fields or set(effective) != fields:
        raise ValueError('support master card has unknown fields')
    ignored = {'name', 'upgradeCount', 'stamina', 'evaluation'}
    if ({k: v for k, v in raw.items() if k not in ignored}
            != {k: v for k, v in effective.items() if k not in ignored}
            or raw['forceStamina'] != 0 or raw['costValue'] != 0 or raw['costType'] != 'ExamCostType_Unknown'
            or raw['planType'] != 'ProducePlanType_Plan1'
            or raw['category'] not in ('ProduceCardCategory_MentalSkill', 'ProduceCardCategory_ActiveSkill')
            or raw['playMovePositionType'] not in ('ProduceCardMovePositionType_Lost', 'ProduceCardMovePositionType_Discard')
            or raw['moveEffectTriggerType'] != 'ProduceCardMoveEffectTriggerType_Unknown'
            or raw['isEndTurnLost'] or raw['isRestrict']
            or any(k in raw for k in ('playConditions', 'moveEffects', 'enchantIds', 'playEffectTriggerType'))
            or type(effective['stamina']) is not int or not 0 <= effective['stamina'] < raw['stamina']):
        raise ValueError('support master tier is not a cost-only enhancement')
    compiled = [_effect(e, effects) for e in raw['playEffects']]
    limited = raw['playMovePositionType'] == 'ProduceCardMovePositionType_Lost'
    expected = CardDefinition(ordinary.card_id, raw['name'], PlanType.SENSE,
        stamina_cost=raw['stamina'], effects=tuple(e for e, _ in compiled), limited=limited,
        card_type='mental' if raw['category'].endswith('MentalSkill') else 'active',
        rarity=raw['rarity'].removeprefix('ProduceCardRarity_').upper(),
        base_card_id=ordinary.base_card_id, upgrade_card_id=ordinary.upgrade_card_id,
        source_type=ordinary.source_type, canonical_card_id=ordinary.canonical_card_id)
    if ordinary != expected or effective['name'] != raw['name'].rstrip('+') + '+' * enhancement_count:
        raise ValueError('support master differs from the full permanent definition')
    body = ('試験・ステージ開始時手札に入る' if raw['isInitial'] else '')
    body += ''.join(line for _, line in compiled)
    body += '試験・ステージ中1回' if limited else ''
    return replace(ordinary, stamina_cost=effective['stamina']), dict(
        name=raw['name'].rstrip('+'), source_id=f'{form[0]}:{enhancement_count}', source=SOURCE,
        enhancement_count=enhancement_count, cost=effective['stamina'], cost_kind='NORMAL',
        body=re.escape(body), effects=[dict(kind=e.kind.value, amount=e.amount) for e, _ in compiled], limited=limited)


def genki_only_support(ordinary, enhancement_count):
    """Compile one changed plain GENKI effect; every other master field stays."""
    form = card_form(ordinary, True)
    if (form is None or '@' in ordinary.card_id or form[1] != 1
            or type(enhancement_count) is not int or enhancement_count not in (2, 3)
            or ordinary.persistent_effects):
        raise ValueError('support genki tier requires an ordinary plus definition')
    cards, effects = _master()
    raw, effective = cards[(form[0], 1)], cards[(form[0], enhancement_count)]
    ignored = {'name', 'upgradeCount', 'evaluation', 'playEffects'}
    ordinary_cost = raw['forceStamina'] or raw['costValue'] or raw['stamina']
    if ({k: v for k, v in raw.items() if k not in ignored}
            != {k: v for k, v in effective.items() if k not in ignored}
            or raw['name'] != ordinary.name
            or ordinary.cost_kind.value != form[2] or ordinary.stamina_cost != ordinary_cost
            or effective['name'] != raw['name'].rstrip('+') + '+' * enhancement_count
            or len(raw['playEffects']) != len(effective['playEffects'])
            or len(raw['playEffects']) != len(ordinary.effects)):
        raise ValueError('support genki tier changes other card semantics')
    projected = list(ordinary.effects)
    changed = []
    for index, (before, after) in enumerate(zip(raw['playEffects'], effective['playEffects'], strict=True)):
        if before == after:
            continue
        old, _ = _effect(before, effects)
        new, _ = _effect(after, effects)
        if old != ordinary.effects[index] or old.kind is not EffectKind.GENKI or new.kind is not EffectKind.GENKI:
            raise ValueError('support genki tier differs from its unchanged ordinary effects')
        projected[index] = new
        changed.append(index)
    if len(changed) != 1:
        raise ValueError('support genki tier must change exactly one plain genki effect')
    return replace(ordinary, effects=tuple(projected))
