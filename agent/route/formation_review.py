"""Plan future qualifier formation changes from saved observations."""
from __future__ import annotations

from math import isfinite
from collections import Counter


def formation_changes(slots, desired):
    """Match saved card or memory identities, keeping any already-covered slot."""
    remaining = Counter(desired)
    available = []
    unknown = []
    for index, slot in enumerate(slots, 1):
        identity = slot.get('id')
        if slot.get('observed') is not True:
            unknown.append(index)
        elif identity is not None and remaining[identity] > 0:
            remaining[identity] -= 1
        else:
            available.append((index, identity))

    replacements = list(remaining.elements())
    moves = [dict(slot=index, current=identity, replacement=replacement)
        for (index, identity), replacement in zip(available, replacements)]
    return dict(moves=moves, unplaced=replacements[len(moves):], unknown_slots=unknown)


def rental_candidates(rows, *, attribute, order):
    """Rank observed compatible rentals by level or by an explicitly supplied score."""
    if attribute not in ('Vo', 'Da', 'Vi') or order not in ('level', 'score'):
        raise ValueError('choose Vo, Da, or Vi and a rental ordering')
    eligible = []
    for row in rows:
        card, level, breaks = (row.get(key) for key in ('card_id', 'level', 'breaks'))
        if (not row.get('row') or not isinstance(card, str) or not card or row.get('compatible') is not True
                or row.get('covered') is not False or type(level) is not int or level <= 0):
            continue
        if type(breaks) is not int or breaks < 0:
            breaks = None
        score = row.get('score')
        if order == 'score':
            if type(score) not in (float, int) or not isfinite(score):
                continue
        elif row.get('attribute') != attribute:
            continue
        else:
            score = None
        key = (score, level, breaks if breaks is not None else -1) if order == 'score' else (
            level, breaks if breaks is not None else -1)
        eligible.append((key, dict(row=row['row'], card_id=card, level=level, breaks=breaks, score=score)))
    if not eligible:
        return dict(candidates=[], tied_first=False)
    ranked = sorted(eligible, key=lambda item: item[0], reverse=True)
    tied_first = len(ranked) > 1 and ranked[0][0] == ranked[1][0]
    return dict(candidates=[candidate for _, candidate in ranked], tied_first=tied_first)
