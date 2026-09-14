"""Same-frame, error-only skill body repair; never choose text by card ID."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from collections.abc import Sequence

from .catalog import _normalise_text
from ._detail_text_layout import (
    NUMERIC_TOKEN_BOUNDARY,
    Box,
    Atom,
    _gaps,
    _inside_roi,
    _text_connected,
)
from ._skill_card_title_recovery import _one_edit

# These are shared game terms, independent of the current card's allowed effects.
# Time conditions and numeric operands are deliberately not fuzzy vocabulary.
_STANCE_CLAUSES = ("強気に変更", "温存に変更", "全力に変更")
_VALUE_LABELS = (
    "元気", "集中", "好調", "絶好調", "好印象", "やる気", "全力値",
    "スコア", "スコア値増加", "消費体力減少", "好印象強化",
    "スキルカード使用数追加",
)
_VALUE = re.compile(r"(?P<label>[^+\-\d%％]+)(?P<value>[+＋][0-9]+(?:[%％])?(?:[（(][0-9]+ターン[）)])?)\Z")
_DECORATION = re.compile(r"^[★☆✦◆◇・＊*+＋\s]*")


@dataclass(frozen=True, slots=True)
class EffectBodyRecovery:
    text: str
    reordered_indices: tuple[int, ...] = ()
    replacements: tuple[tuple[int, str, str], ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.reordered_indices or self.replacements)


def _unique_word(observed: str, words: Sequence[str]) -> str | None:
    if (_normalise_text(observed) in words or not observed.isalpha()
            or any(char.isnumeric() for char in observed)
            or any(token in observed for token in ("不", "無", "非", "未", "禁", "ない", "除", "以外"))):
        return None
    matches = [word for word in words if not observed.startswith(word) and _one_edit(observed, word)]
    return matches[0] if len(matches) == 1 else None


def _repair_atom(text: str, preceding: str) -> str:
    decoration = _DECORATION.match(text).group()
    body = text[len(decoration):]
    condition = ""
    if body.startswith("次のターン、"):
        condition, body = "次のターン、", body[len("次のターン、"):]
    # A circled digit directly between a complete time clause and a complete
    # stance clause is a damaged icon carrier, not a numeric effect operand.
    # Plain digits, numbers elsewhere, and an uncertain stance stay untouched.
    if (condition or _DECORATION.sub("", preceding, count=1) == "次のターン、") and (
        body.startswith("①") and body[1:] in _STANCE_CLAUSES
    ):
        return decoration + condition + body[1:]
    replacement = _unique_word(body, _STANCE_CLAUSES)
    if replacement is not None:
        return decoration + condition + replacement
    value = _VALUE.fullmatch(body)
    if value is not None:
        replacement = _unique_word(value["label"], _VALUE_LABELS)
        if replacement is not None:
            return decoration + condition + replacement + value["value"]
    return text


def recover_skill_effect_body(
    atoms: tuple[Atom, ...], title_box: Box, search_roi: Box,
    rows: Sequence[tuple[str, Box, tuple[Atom, ...]]],
) -> EffectBodyRecovery:
    """Reorder only title-connected body tokens and repair complete terms.

    The caller reuses its existing spatial row helper. Unassociated text,
    standalone integers, titles, signs and numeric values keep their original
    positions/content. This does not turn the wide search ROI into a paragraph.
    """
    lookup = {(text.strip(), box): index for index, (text, box) in enumerate(atoms)}
    if len(lookup) != len(atoms):
        return EffectBodyRecovery("\n".join(text for text, _ in atoms))
    body: list[int] = []
    components: list[list[int]] = []
    for _, _, row in rows:
        group: list[int] = []
        groups: list[list[int]] = []
        for atom in row:
            index = lookup[atom]
            text, box = atom
            if (box == title_box or not _inside_roi(box, search_roi)
                    or box[1] < title_box[1] + title_box[3] / 2
                    or sum(char.isalpha() for char in text) < 2):
                continue
            if group and _gaps(atoms[group[-1]][1], box)[0] > (
                atoms[group[-1]][1][3] + box[3]
            ):
                groups.append(group)
                group = []
            group.append(index)
        if group:
            groups.append(group)
        for group in groups:
            connected = any(
                _text_connected(atoms[previous][1], atoms[index][1], title_box[0])
                for previous in body for index in group
            ) if body else any(
                _text_connected(title_box, atoms[index][1], title_box[0], first_body=True)
                for index in group
            )
            if connected:
                body.extend(group)
                components.append(group)

    order = list(range(len(atoms)))
    values = [text for text, _ in atoms]
    moved: list[int] = []
    replacements: list[tuple[int, str, str]] = []
    for group in components:
        positions = sorted(group)
        span = list(range(positions[0], positions[-1] + 1))
        intruders = [index for index in span if index not in group]
        # A non-text fragment on another visual row can sort between two body
        # words by its top edge. Keep it verbatim behind a lexical boundary.
        # A real number, signed operand or another text clause cannot be moved.
        movable = all(
            not any(char.isalpha() for char in atoms[index][0])
            and re.fullmatch(r"[+\-−]?[0-9]+", unicodedata.normalize("NFKC", atoms[index][0]).strip()) is None
            and not any(atoms[index] in row for _, _, row in rows if atoms[group[0]] in row)
            # The preserved fragment must be beyond the complete line's right
            # edge. A detached operand to its left or between its words is not
            # licensed to move merely because its OCR height differs.
            and atoms[index][1][0] >= max(atoms[item][1][0] + atoms[item][1][2] for item in group)
            for index in intruders
        )
        if group != positions and movable:
            ordered = group + intruders
            for position, index in zip(span, ordered, strict=True):
                order[position] = index
            for index in intruders:
                values[index] = NUMERIC_TOKEN_BOUNDARY + values[index] + NUMERIC_TOKEN_BOUNDARY
            moved.extend(ordered)
        for index in group:
            position = order.index(index)
            previous = order[position - 1] if position else None
            preceding = values[previous] if previous in group else ""
            original = values[index]
            repaired = _repair_atom(original, preceding)
            if repaired != original:
                values[index] = repaired
                replacements.append((index, original, repaired))
    return EffectBodyRecovery(
        "\n".join(values[index] for index in order), tuple(moved), tuple(replacements),
    )
