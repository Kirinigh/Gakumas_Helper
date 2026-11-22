"""Read explicit HIF customization goals without changing files or granting input.

Format reference: miyu1019/MaaGakumasu fb8fd00c, shop.py:687-759.
The current profession's profiles entry takes precedence over legacy cards.
Legacy option arrays mean cumulative maxima, never additional purchases.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

PROFESSIONS = frozenset({'好调', '集中', '好印象', '元気', '全力', '強気'})


@dataclass(frozen=True)
class CardCustomizationGoal:
    card_id: str
    goals: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CustomizationSelection:
    profession: str
    enabled: bool
    source: str
    source_revision: str | None
    cards: tuple[CardCustomizationGoal, ...]


def _identifier(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or str(int(value)) != value:
        raise ValueError('customization configuration requires canonical local IDs')
    return value


def _option_goals(selected, rules):
    if isinstance(selected, list):
        if (any(type(value) is not int or str(value) not in rules for value in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('legacy customization selection contains an unknown or duplicate option')
        goals = {str(value): rules[str(value)][0] for value in selected}
    elif isinstance(selected, dict):
        goals = {_identifier(key): value for key, value in selected.items()}
        if any(key not in rules or type(value) is not int or not 0 <= value <= rules[key][0]
                for key, value in goals.items()):
            raise ValueError('customization cumulative target is unknown or outside its maximum')
    else:
        raise ValueError('customization targets require an object or legacy option array')
    return tuple((key, goals[key]) for key in rules if goals.get(key, 0) > 0)


def parse_customization_selection(raw, *, profession, enabled, catalog):
    """Build the full ordered queue; resource capacity is observed later."""
    if not isinstance(profession, str) or profession not in PROFESSIONS or type(enabled) is not bool:
        raise ValueError('customization profession or explicit enable switch is invalid')
    if not enabled:
        return CustomizationSelection(profession, False, 'disabled', None, ())
    if raw is None:
        return CustomizationSelection(profession, True, 'missing', None, ())
    if not isinstance(raw, dict):
        raise ValueError('customization selection file must contain an object')
    revision = raw.get('sourceRevision')
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError('customization source revision is invalid')
    if 'profiles' in raw:
        profiles = raw['profiles']
        if not isinstance(profiles, dict):
            raise ValueError('customization profiles must contain an object')
        selected, source = profiles.get(profession, {}), 'profiles'
    else:
        selected, source = raw.get('cards', {}), 'legacy_cards'
    if not isinstance(selected, dict):
        raise ValueError('current profession customization selection must contain an object')
    for key in selected:
        if _identifier(key) not in catalog.cards:
            raise ValueError('customization selection contains an unknown local card')
    queue = []
    for card_id in catalog.cards:
        if card_id not in selected:
            continue
        goals = _option_goals(selected[card_id], catalog.rules(card_id))
        if goals:
            queue.append(CardCustomizationGoal(card_id, goals))
    return CustomizationSelection(profession, True, source, revision, tuple(queue))


def load_customization_selection(path, *, profession, enabled, catalog):
    """The production caller supplies its fixed user path; no file is rewritten."""
    if not enabled:
        return parse_customization_selection(None, profession=profession, enabled=enabled, catalog=catalog)
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None
    return parse_customization_selection(raw, profession=profession, enabled=enabled, catalog=catalog)
