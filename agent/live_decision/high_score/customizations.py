"""Compile observed permanent upgrades without changing the base registry.

The observation layer owns badge/detail evidence. This pure compiler consumes
one resolved catalog group; it never infers absence or temporary enhancements.
"""

import re
from dataclasses import replace

from .kernel import EffectKind
from .catalog_v2 import _compile_condition, _compile_action_sequence

_GROWTH = {
    'score': EffectKind.SCORE,
    'genki': EffectKind.GENKI,
    'concentration': EffectKind.CONCENTRATION,
    'goodConditionTurns': EffectKind.GOOD_CONDITION,
    'goodImpressionTurns': EffectKind.GOOD_IMPRESSION,
    'motivation': EffectKind.MOTIVATION,
}


def _level_effect(text, level):
    if not text.startswith('level:'):
        return text
    blocks = {}
    position = 0
    while position < len(text):
        match = re.match(r'\s*level:(\d+)\s*\{', text[position:])
        if not match:
            raise ValueError('customization level grammar is unsupported')
        start = position + match.end()
        stop, depth = start, 1
        while stop < len(text) and depth:
            depth += (text[stop] == '{') - (text[stop] == '}')
            stop += 1
        index = int(match[1])
        if depth or index in blocks:
            raise ValueError('customization level body is incomplete or duplicated')
        blocks[index] = text[start:stop-1].strip()
        position = stop
        if not text[position:].strip():
            break
    if level not in blocks:
        raise ValueError('customization level is not defined')
    return blocks[level]


def compile_observed_customizations(base, card_row, customization_rows, levels):
    """Create a separate definition for one exact, nonempty observed group.

    Only already upgraded cards are admitted, so an ordinary upgrade cannot
    silently drop permanent changes. Generated base cards retain the original
    registry entry. Unsupported edits are rejected atomically.
    """
    if (str(card_row['id']) != base.card_id or base.canonical_card_id is not None
            or str(card_row.get('upgraded')).upper() != 'TRUE'
            or base.upgrade_card_id is not None or not levels):
        raise ValueError('customization requires one canonical upgraded card')
    allowed = tuple(filter(None, str(card_row.get('availableCustomizations', '')).split(',')))
    rows = {str(row['id']): row for row in customization_rows}
    if len(rows) != len(customization_rows):
        raise ValueError('customization catalog has duplicate ids')
    if any(key not in allowed or key not in rows or type(value) is not int
           or not 1 <= value <= int(rows[key]['max']) for key, value in levels.items()):
        raise ValueError('observed customization group is not legal')
    effects = list(base.effects)
    condition, limited = base.condition, base.limited
    for key in allowed:
        if key not in levels:
            continue
        row, level = rows[key], levels[key]
        if row.get('forceInitialHand') or row.get('cost') or row.get('limit'):
            raise ValueError('customization structural edit is not compiled')
        if row.get('conditions'):
            updated = _compile_condition(str(row['conditions']))
            if updated is None:
                raise ValueError('customization condition is not compiled')
            condition = updated
        if row.get('effects'):
            statement = _level_effect(str(row['effects']).strip(), level)
            match = re.fullmatch(r'at:prestage\s*\{\s*target:this\s*\{\s*g\.(\w+)\+=(\d+)\s*}\s*}', statement)
            kind = _GROWTH.get(match[1]) if match else None
            if kind is None:
                raise ValueError('customization growth is not compiled')
            targets = [i for i, effect in enumerate(effects) if effect.kind is kind]
            if (len(targets) != 1 or any(
                nested.kind is kind for effect in effects + list(base.persistent_effects)
                for nested in effect.scheduled_effects
            )):
                raise ValueError('customization growth needs one direct effect carrier')
            index, = targets
            if effects[index].value_source is not None:
                raise ValueError('customization state-scaled growth is not compiled')
            effects[index] = replace(effects[index], amount=effects[index].amount + int(match[2]))
        if row.get('actions'):
            appended = _compile_action_sequence(str(row['actions']))
            if appended is None:
                raise ValueError('customization appended action is not compiled')
            effects.extend(appended)
    suffix = ','.join(f'{key}={levels[key]}' for key in sorted(levels, key=int))
    result = replace(base, card_id=f'{base.card_id}@{suffix}', canonical_card_id=base.card_id,
                     effects=tuple(effects), condition=condition, limited=limited)
    result.validate()
    return result
