"""Bounded field structure over original OCR atoms, without semantic repair.

The caller removes only decorations already proved from pixels. This module
never drops/rewrites atoms or supplies values: it checks the field units those
atoms can form, independently of their original OCR packaging and item order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ._detail_text_layout import Box, Atom, _text_connected

_OPEN = re.compile(r"(?:[+＋の]|消費|追加|値増加|コスト|スコア|コア|元気|体力|集中|やる気|好印象|好調|全力値)$")
_POSITIVE = r"[+＋][1-9][0-9]*"
_UNSIGNED = r"(?:0|[1-9][0-9]*)"
_NEGATIVE = r"[-−－][1-9][0-9]*"
_INCREASE = r"(?:増加|增加)"
_SCALAR_LABEL = "(?:スコア|元気|集中|やる気|好印象|好調|全力値)"
_MULTIPLIER_LABEL = "(?:元気|集中|やる気|好印象|全力値)"
_MULTIPLIER = _UNSIGNED + r"(?:\.[0-9]+)?倍"
_DURATION_LABEL = "(?:絶好調|好調)"
_DURATION = r"[1-9][0-9]*ターン"
_LABELS = (
    (_DURATION_LABEL, _DURATION),
    (_MULTIPLIER_LABEL, _MULTIPLIER),
    ("スキルカード使用数追加", _POSITIVE),
    ("ターン追加", _POSITIVE),
    (f"スコア上昇回数{_INCREASE}", _POSITIVE),
    (f"スコア値{_INCREASE}", _POSITIVE),
    (f"(?:体力消費)?コスト値{_INCREASE}", _POSITIVE),
    ("(?:体力消費)?コスト値減少", _NEGATIVE),
    ("体力消費", _UNSIGNED),
    (_SCALAR_LABEL, _POSITIVE),
)
_NEXT_FIELD = "(?:" + "|".join(label + r"[ \t]*" + value for label, value in _LABELS) + ")"
_END = r"(?=$|[・·、（(]|[ \t]*(?=" + _NEXT_FIELD + "))"
_FIELDS = tuple((re.compile(label), re.compile(label + r"[ \t]*" + value + _END))
                for label, value in _LABELS)
_START = re.compile(r"スキルカード|ターン|体力|コスト|スコア|元気|集中|やる気|好印象|絶好調|好調|全力値")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_PAYLOAD = re.compile(r"^[+＋\-−－0-9.%％]|^(?:pt|回)$", re.IGNORECASE)
_MAX_ITEMS, _MAX_UNIT_ITEMS, _MAX_UNIT_CHARS = 128, 4, 128


def _assignment_started(label: str, tail: str) -> bool:
    """A stat name does not imply that its unsigned text is an addition.

    Bare labels and written operators need closure. Unsigned values start a
    check only for their identified role: fixed count/cost/duration, or an
    explicit multiplier unit. Other complete descriptions belong to the
    existing effect parser, not a new body-wide scalar validator here.
    """
    if not tail:
        return True
    if re.match(r"[+＋\-−－.%％]", tail):
        return True
    if not tail[0].isdigit():
        return False
    if label == _SCALAR_LABEL:
        return False
    unit_role = {_MULTIPLIER_LABEL: _MULTIPLIER, _DURATION_LABEL: _DURATION}.get(label)
    if unit_role is not None:
        value = re.match(unit_role, tail)
        if value is None:
            return False
        remainder = tail[value.end():].lstrip(' \t')
        return (re.match(unit_role + _END, tail) is not None
                or re.match(r"[+＋\-−－0-9.%％]", remainder) is not None)
    return True


def _inside(outer: Box, inner: Box) -> bool:
    x, y, w, h = outer
    a, b, c, d = inner
    return c > 0 and d > 0 and x <= a and y <= b and a + c <= x + w and b + d <= y + h


def _same_line(left: Box, right: Box) -> bool:
    _, y, _, h = left
    _, b, _, d = right
    return (max(h, d) <= 1.5 * min(h, d)
            and min(y + h, b + d) - max(y, b) >= .5 * min(h, d)
            and abs(y + h / 2 - b - d / 2) <= .65 * max(h, d))


def _ordered(left: Box, right: Box) -> bool:
    x, _, w, _ = left
    a, _, c, _ = right
    # Both centers have independent support outside the shared padding.
    return x < a and x + w < a + c and x + w / 2 < a and a + c / 2 > x + w


def has_unfinished_operands(
    items: Sequence[Atom], panel: Box, title: Box, *, body_top: int | None = None,
) -> bool:
    """Return true if any bounded operand lacks an unambiguous typed closure.

    ``items`` are ``(original_text, original_box)`` pairs. Header consumers may
    explicitly include the title band with ``body_top=title[1]``; the actual
    title box is always excluded. The default includes only the body below it.
    """
    top = title[1] + title[3] if body_top is None else body_top
    atoms = tuple((text.strip(), tuple(box)) for text, box in items
                  if text.strip() and tuple(box) != title and _inside(panel, tuple(box)) and box[1] >= top)
    if len(atoms) > _MAX_ITEMS or sum(len(text) for text, _ in atoms) > _MAX_ITEMS * _MAX_UNIT_CHARS:
        return True
    if not atoms:
        return False
    # Duplicate or deeply superposed observations cannot manufacture separate
    # word/value anchors. Only relevant field attempts need this new proof.
    risk = any(_OPEN.search(text) or any(pattern.search(text) for pattern, _ in _FIELDS)
               for text, _ in atoms)
    if not risk:
        return False
    if len({box for _, box in atoms}) != len(atoms):
        return True

    def horizontal(index: int) -> tuple[int, ...]:
        box = atoms[index][1]
        candidates = [i for i, (_, other) in enumerate(atoms) if i != index
                      and _same_line(box, other) and other[0] > box[0]]
        if not candidates:
            return ()
        first_x = min(atoms[i][1][0] for i in candidates)
        nearest = tuple(i for i in candidates if atoms[i][1][0] == first_x)
        if len(nearest) != 1:
            return ()
        return tuple(i for i in nearest if _ordered(box, atoms[i][1])
                     and _text_connected(box, atoms[i][1], title[0]))

    def vertical(index: int) -> tuple[int, ...]:
        box = atoms[index][1]
        x, y, w, h = box
        # A small nonsemantic icon wholly inside this line is not its next
        # row. A number, operator or named effect in that band is protected
        # evidence; skipping it would choose a later, more convenient value.
        if any(i != index and (not _same_line(box, other)
                               or not (_ordered(box, other) or _ordered(other, box)))
               and y <= other[1] and other[1] + other[3] <= y + h
               and (any(character.isnumeric() for character in text)
                    or re.search(r"[+＋\-−－%％]", text) or _START.search(text) or _OPEN.search(text))
               for i, (text, other) in enumerate(atoms)):
            return ()
        candidates = [i for i, (_, other) in enumerate(atoms) if i != index
                      and not _same_line(box, other)
                      and other[1] + other[3] > y + h
                      and other[1] + other[3] / 2 > y + h / 2
                      and other[1] <= y + h + .75 * max(h, other[3], title[3])]
        if not candidates:
            return ()
        first = min(candidates, key=lambda i: (atoms[i][1][1] + atoms[i][1][3] / 2, atoms[i][1][0]))
        line = [i for i in candidates if i == first or _same_line(atoms[first][1], atoms[i][1])]
        first_x = min(atoms[i][1][0] for i in line)
        nearest = [i for i in line if atoms[i][1][0] == first_x]
        if len(nearest) != 1:
            return ()
        return tuple(i for i in nearest
                     if 0 <= atoms[i][1][1] - y - h <= .75 * max(h, atoms[i][1][3], title[3])
                     and max(h, atoms[i][1][3]) <= 1.5 * min(h, atoms[i][1][3])
                     and (abs(atoms[i][1][0] - title[0]) <= .75 * max(h, atoms[i][1][3], title[3])
                          or abs(atoms[i][1][0] - x) <= .75 * max(h, atoms[i][1][3]))
                     and _text_connected(box, atoms[i][1], title[0]))

    # Each proof contains original character intervals. A completed field
    # cannot claim another pending field merely because they share an atom.
    proofs: list[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = []
    attempted: set[tuple[int, int, int]] = set()
    for start, (text, _) in enumerate(atoms):
        for match in _START.finditer(text):
            offset = match.start()
            prefix = text[offset:]
            for label, _ in _FIELDS:
                found = label.match(prefix)
                if found:
                    tail = prefix[found.end():].lstrip(" \t")
                    if _assignment_started(label.pattern, tail):
                        attempted.add((start, offset, offset + found.end()))
            pending = [((start,), prefix, ((start, offset, len(text)),), False)]
            while pending:
                chain, joined, spans, wrapped = pending.pop()
                if len(joined) > _MAX_UNIT_CHARS:
                    continue
                for _, field in _FIELDS:
                    complete = field.match(joined)
                    if complete is None:
                        continue
                    end = complete.end()
                    consumed = []
                    position = 0
                    for i, a, b in spans:
                        length = b - a
                        if position < end:
                            consumed.append((i, a, a + min(length, end - position)))
                        position += length
                    last = chain[-1]
                    extra_numeric = any(_PAYLOAD.search(atoms[i][0]) for i in horizontal(last))
                    if not extra_numeric:
                        proofs.append((chain, tuple(consumed)))
                if len(chain) == _MAX_UNIT_ITEMS:
                    continue
                last = chain[-1]
                following = horizontal(last)
                options = [(i, False) for i in following]
                if not following and not wrapped:
                    for i in vertical(last):
                        # An unfinished field name wraps only from the right
                        # edge back to the fixed left text margin. A complete
                        # label awaiting its number uses ordinary adjacency.
                        label_complete = any(label.fullmatch(joined) for label, _ in _FIELDS)
                        box, other = atoms[last][1], atoms[i][1]
                        scale = max(box[3], other[3], title[3])
                        payload_wrap = label_complete and bool(_PAYLOAD.match(atoms[i][0]))
                        if payload_wrap:
                            value_text = atoms[i][0]
                            roles = [(label, value) for label, value in _LABELS if re.fullmatch(label, joined)]
                            if not any(re.match(value + r"(?=$|[（(])", value_text)
                                       and (label not in {"スキルカード使用数追加", "ターン追加"}
                                            or re.fullmatch(value, value_text))
                                       for label, value in roles):
                                continue
                            # The original count-value wrap owns one entire
                            # adjacent payload row, not the first value before
                            # another independently observed effect field.
                            if any(j != i and j not in chain and _same_line(other, candidate)
                                   for j, (_, candidate) in enumerate(atoms)):
                                continue
                        if payload_wrap or (not label_complete or not _PAYLOAD.match(atoms[i][0])) and (
                            0 <= panel[0] + panel[2] - box[0] - box[2] <= scale
                            and abs(other[0] - title[0]) <= .75 * scale
                        ):
                            options.append((i, True))
                for i, is_wrap in options:
                    if i in chain:
                        continue
                    next_text = atoms[i][0]
                    # Never turn two source numbers into a new multi-digit
                    # value by removing their atom boundary.
                    if joined[-1:].isdigit() and next_text[:1].isdigit():
                        continue
                    pending.append((chain + (i,), joined + next_text,
                                    spans + ((i, 0, len(next_text)),), wrapped or is_wrap))

    def covered(index: int, begin: int, end: int) -> bool:
        matches = {(chain, spans) for chain, spans in proofs
                   if any(i == index and a <= begin and b >= end for i, a, b in spans)}
        # Alternative OCR anchors are not alternative values we may choose.
        tails = {chain[chain.index(index):] for chain, _ in matches}
        shortest = {chain for chain in tails if not any(other != chain and chain[:len(other)] == other
                                                        for other in tails)}
        return len(shortest) == 1

    for index, (text, box) in enumerate(atoms):
        if any(not covered(i, a, b) for i, a, b in attempted if i == index):
            return True
        if not _OPEN.search(text):
            continue
        if covered(index, len(text) - 1, len(text)):
            continue
        if text.endswith("の"):
            neighbours = horizontal(index)
            if len(neighbours) == 1:
                other = neighbours[0]
                next_text, next_box = atoms[other]
                gap = next_box[0] - box[0] - box[2]
                if _JAPANESE.match(next_text) and gap <= .75 * max(box[3], next_box[3]):
                    continue
        return True
    return False
