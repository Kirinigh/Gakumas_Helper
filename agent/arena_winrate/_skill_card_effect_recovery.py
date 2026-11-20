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
    "スキルカード使用数追加", "熱意追加",
)
_VALUE = re.compile(r"(?P<label>[^+\-\d%％]+)(?P<value>[+＋][0-9]+(?:[%％])?(?:[（(][0-9]+ターン[）)])?)\Z")
_DECORATION = re.compile(r"^[★☆✦◆◇・＊*+＋\s]*")


@dataclass(frozen=True, slots=True)
class EffectBodyRecovery:
    text: str
    reordered_indices: tuple[int, ...] = ()
    replacements: tuple[tuple[int, str, str], ...] = ()
    excluded_indices: tuple[int, ...] = ()
    deferred_indices: tuple[int, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.reordered_indices or self.replacements or self.excluded_indices or self.deferred_indices)


def recover_overlapping_enthusiasm_operand(
    native: Sequence[Atom], derived: Sequence[Atom], title: Box, derived_title: Box,
) -> tuple[int, str] | None:
    """Reuse an intact native operand when the derived label repeats as a digit."""
    def overlaps(a: Box, b: Box) -> bool:
        return (max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
                and max(a[1], b[1]) < min(a[1] + a[3], b[1] + b[3]))

    if not overlaps(title, derived_title):
        return None
    fields = [(box, match[1]) for text, box in native
              if (match := re.fullmatch(r"熱意追加(\+[1-9][0-9]*)", _normalise_text(text)))
              and box[1] >= title[1] + title[3]]
    if len(fields) != 1:
        return None
    field, value = fields[0]
    x, y, width, height = field
    if height <= 0:
        return None
    # The full field must cover both pieces of the very same derived row.
    # OCR boxes can drift slightly between the two scales.
    margin = height / 4
    parts = [(index, _normalise_text(text), box) for index, (text, box) in enumerate(derived)
             if text.strip() and overlaps(field, box)]
    if len(parts) != 2:
        return None
    (head_index, label, head), (index, payload, operand) = sorted(parts, key=lambda part: part[2][0])
    if label != "熱意追加" or re.fullmatch(r"[0-9]" + re.escape(value), payload) is None:
        return None
    hx, hy, hw, hh = head
    ox, oy, ow, oh = operand
    overlap = hx + hw - ox
    glyph_width = min(ow / len(payload), hw / len(label))
    field_width_per_character = width / (len(label) + len(value))
    if (abs(hx - x) > margin or abs(ox + ow - x - width) > margin
            or not (y - margin <= min(hy, oy) and max(hy + hh, oy + oh) <= y + height + margin)
            or abs(hy + hh / 2 - oy - oh / 2) > margin
            or ox - hx < (len(label) - 1) * field_width_per_character
            or not .5 * glyph_width <= overlap <= 1.5 * glyph_width):
        return None

    # A fragment just outside the intact box may still extend the operand.
    # Require an isolated row in both views before reconciling the overlap.
    def adjacent(box: Box) -> bool:
        bx, by, bw, bh = box
        return (bx <= x + width + field_width_per_character
                and bx + bw >= x - field_width_per_character
                and by < y + height and by + bh > y)

    if (sum(adjacent(box) for text, box in native if text.strip()) != 1
            or any(adjacent(box) for i, (text, box) in enumerate(derived)
                   if text.strip() and i not in (head_index, index))):
        return None
    return index, value


def recover_wrapped_effect_decoration(
    atoms: tuple[Atom, ...], title_box: Box, search_roi: Box,
    rows: Sequence[tuple[str, Box, tuple[Atom, ...]]],
    *, background_parameter_left: int | None = None, corroborating_atoms: Sequence[Atom] = (),
) -> EffectBodyRecovery:
    """Recover the reading order of a complete field interrupted by an icon lane.

    Nothing supplies or edits an operand. Two nearby complete effect rows
    must establish the indented text column, and the continuation must return
    to the title's left margin. Unknown text and all numeric/sign/negation
    fragments remain evidence. A single ASCII digit moves out of the field
    only when the enlarged view reads the same complete field independently
    and its box does not touch either operand. The digit itself stays literal.
    """
    unchanged = EffectBodyRecovery("\n".join(text for text, _ in atoms))
    if len(set(atoms)) != len(atoms) or title_box not in [box for _, box in atoms]:
        return unchanged
    # Rows must come from these exact atoms, not a previous capture or crop.
    row_atoms = [atom for _, _, row in rows for atom in row]
    if len(row_atoms) != len(atoms) or set(row_atoms) != set(atoms):
        return unchanged
    physical_row = {atom: index for index, (_, _, row) in enumerate(rows) for atom in row}
    normalized = [_normalise_text(text) for text, _ in atoms]
    pairs: list[tuple[int, tuple[int, ...], bool]] = []
    for head, (label, box) in enumerate(atoms):
        if normalized[head] not in _VALUE_LABELS or not _inside_roi(box, search_roi):
            continue
        x, y, width, height = box
        if height <= 0 or y < title_box[1] + title_box[3]:
            continue
        supports = []
        for index, (text, other) in enumerate(atoms):
            match = _VALUE.fullmatch(normalized[index])
            if (match is not None and match['label'] in _VALUE_LABELS
                    and _inside_roi(other, search_roi)
                    and title_box[1] + title_box[3] <= other[1] < y
                    and y - other[1] <= 4 * height
                    and 0.7 * height <= other[3] <= 1.5 * height
                    and abs(other[0] - x) <= 0.65 * height):
                supports.append(index)
        if len({physical_row[atoms[index]] for index in supports}) < 2:
            continue
        values = []
        for index, (_, other) in enumerate(atoms):
            # Buff percentages require their entire duration in this one atom.
            # Other shared effects may have a complete simple signed value.
            grammar = (r"\+[0-9]+%\([0-9]+ターン\)" if normalized[head].endswith('強化')
                       else r"\+[0-9]+(?:%\([0-9]+ターン\))?")
            if (re.fullmatch(grammar, normalized[index]) is not None
                    and _inside_roi(other, search_roi)
                    and 0.7 * height <= other[3] <= 1.5 * height
                    and 0.8 * height <= other[1] - y <= 1.8 * height
                    and abs(other[0] - title_box[0]) <= 0.8 * height):
                values.append(index)
        if len(values) != 1 or values[0] <= head:
            continue
        value = values[0]
        value_box = atoms[value][1]
        # Inspect the entire wrapped band, including another possible header
        # or value beyond the left margin. None may silently disappear.
        band = [index for index, (_, other) in enumerate(atoms)
                if _inside_roi(other, search_roi)
                and y <= other[1] + other[3] / 2 <= value_box[1] + value_box[3]]
        background = tuple(index for index in band
                           if background_parameter_left is not None
                           and atoms[index][1][0] >= background_parameter_left
                           and (re.fullmatch(r"[0-9]+%?", normalized[index]) is not None
                                or normalized[index] in {'体力', 'ダンス', 'ビジュアル', 'ボーカル'}))
        fragments = [index for index in band if index not in (head, value, *background)]
        if len(fragments) != 1:
            continue
        icon = fragments[0]
        text, (a, b, w, h) = atoms[icon]
        glyph = text.strip()
        numeric = re.fullmatch(r"[0-9]", glyph) is not None
        if (len(glyph) != 1 or not (glyph.isalpha() or numeric)
                or (not numeric and any(char.isnumeric() for char in text))
                or text.strip() in '不無非未禁除'
                or not 0.35 * height <= w <= 0.85 * height
                or not 0.35 * height <= h <= 0.85 * height
                or not 0.6 <= w / h <= 1.6
                or abs((b + h / 2) - (y + height / 2)) > 0.35 * height
                or a < title_box[0] or a + w > x - 0.5 * height):
            continue
        if numeric and not _corroborates_wrapped_field(corroborating_atoms, atoms[head], atoms[value], atoms[icon][1]):
            continue
        pairs.append((icon, background, numeric))
    if len(pairs) != 1:
        return unchanged
    icon, background, numeric = pairs[0]
    excluded = background if numeric else tuple(sorted((icon, *background)))
    deferred = (icon,) if numeric else ()
    return EffectBodyRecovery(
        "\n".join([text for index, (text, _) in enumerate(atoms) if index not in (*excluded, *deferred)]
                  + [f"{NUMERIC_TOKEN_BOUNDARY}{atoms[index][0]}{NUMERIC_TOKEN_BOUNDARY}" for index in deferred]),
        excluded_indices=excluded,
        deferred_indices=deferred,
    )


def _corroborates_wrapped_field(atoms: Sequence[Atom], label: Atom, value: Atom, icon: Box) -> bool:
    def overlaps(a: Box, b: Box) -> bool:
        return (min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0])
                and min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1]))

    if overlaps(icon, label[1]) or overlaps(icon, value[1]) or any(overlaps(icon, box) for _, box in atoms):
        return False
    head, operand = (tuple(index for index, (text, box) in enumerate(atoms)
                           if _normalise_text(text) == _normalise_text(original) and overlaps(box, original_box))
                     for original, original_box in (label, value))
    if len(head) != 1 or len(operand) != 1:
        return False
    head_box, value_box = atoms[head[0]][1], atoms[operand[0]][1]
    top, bottom = head_box[1], value_box[1] + value_box[3]
    return (top < value_box[1] and not any(
        top <= box[1] + box[3] / 2 < bottom
        for index, (_, box) in enumerate(atoms) if index not in (head[0], operand[0])
    ))


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
