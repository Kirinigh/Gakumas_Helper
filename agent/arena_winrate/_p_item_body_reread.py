"""Pure bounds and semantic checks for a failed P-item body reread.

No pixels, OCR, backend, clock or retry state are owned here. The caller binds
the derived observation to its original image and keeps independent-frame,
cancellation and transaction budgets.
"""
from __future__ import annotations

import re
import math
import unicodedata
from collections import deque
from dataclasses import replace, dataclass
from collections.abc import Mapping, Sequence

from ._reader_visual import _spatial_ocr_rows
from ._p_item_detail_text import (
    NUMBER,
    BOUNDARY,
    PItemDetailTextResult,
    PItemSourceTextFrames,
    normalize,
    source_field,
    atom_text_view,
    _is_complete_numeric_field,
    _adjacent_operand_fragments,
)

_TRAILER = re.compile(r"\(試験・ステージ内[1-9][0-9]*回\)\Z")
_MAX_ANOMALIES = 5
# These are protected meanings, never replacement candidates. Both sides of
# an alignment protect the complete term, including gaps inside a term.
_SEMANTIC = re.compile(r"ない|以上|以下|以内|未満|以外|増加|減少|追加|消費|回復|ターン|場合|ごと|以降|最終|開始|終了|[不無无非未禁除増減超回枚倍前後]")
_DISPLAY_UNIT = re.compile(r".[\ufe0e\ufe0f]?", re.DOTALL)


@dataclass(frozen=True)
class BodyRereadPlan:
    roi: tuple[float, float, float, float]
    family_ids: tuple[int, ...]
    base_id: int
    title_text: str
    trailer_text: str
    included_indices: tuple[int, ...]
    references: tuple[tuple[int, str, tuple[str, ...]], ...]
    original_signature: tuple
    source_frames: tuple
    original_result: PItemDetailTextResult


def _canonical(text):
    return normalize(text).translate(str.maketrans({"值": "値", "增": "増", "·": "・"})).replace("体力消费", "体力消費")


def _valid_atoms(atoms, coordinate_size=None):
    try:
        for atom in atoms:
            box = atom["box"]
            if (not isinstance(atom["text"], str) or len(box) != 4
                    or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0):
                return False
            if coordinate_size and (box[0] + box[2] > coordinate_size[0] or box[1] + box[3] > coordinate_size[1]):
                return False
        return True
    except (KeyError, TypeError):
        return False


def _signature(atoms):
    return tuple((atom["text"], tuple(atom["box"])) for atom in atoms)


def _clear_result(result):
    diagnostics = result.diagnostics
    return bool(not diagnostics.get("unbound_reference_ids")
                and not diagnostics.get("layout", {}).get("uncertainty"))


def _source_background(atom, sources, height):
    return len(sources.matching_frames(atom, height)) >= 2


def build_body_reread_plan(
    atoms: Sequence[Mapping], *, title_index: int,
    source_frames: Sequence[Sequence[Mapping]], family_references: Mapping[int, Mapping],
    original_result: PItemDetailTextResult, coordinate_size=(720, 1280),
) -> BodyRereadPlan | None:
    """Include every unproved atom between an exact title and common trailer."""
    if (original_result.status != "ambiguous" or original_result.reason != "insufficient_difference_evidence"
            or not _clear_result(original_result)
            or any(original_result.diagnostics.get("conflicts", {}).get(item_id,
                       original_result.diagnostics.get("conflicts", {}).get(str(item_id), ()))
                   for item_id in original_result.ids)
            or type(title_index) is not int or not 0 <= title_index < len(atoms)
            or not _valid_atoms(atoms, coordinate_size)):
        return None
    title = _canonical(atoms[title_index]["text"])
    if not title or title.endswith("+") or sum(_canonical(atom["text"]) == title for atom in atoms) != 1:
        return None
    references = []
    for item_id, row in family_references.items():
        name, paragraphs = row.get("name"), row.get("paragraphs")
        if (type(item_id) is not int or item_id <= 0 or not isinstance(name, str)
                or not isinstance(paragraphs, (tuple, list)) or not paragraphs
                or not all(isinstance(text, str) and _canonical(text) for text in paragraphs)
                or _canonical(name).removesuffix("+") != title):
            return None
        references.append((item_id, _canonical(name), tuple(_canonical(text) for text in paragraphs)))
    references.sort()
    family = tuple(row[0] for row in references)
    bases = [item_id for item_id, name, _ in references if name == title]
    if len(family) < 2 or len(bases) != 1 or not original_result.ids or not set(original_result.ids).issubset(family):
        return None
    trailers = {paragraphs[-1] for _, _, paragraphs in references}
    if len(trailers) != 1 or not _TRAILER.fullmatch(next(iter(trailers))):
        return None
    trailer = next(iter(trailers))
    ends = [index for index, atom in enumerate(atoms) if _canonical(atom["text"]) == trailer]
    if len(ends) != 1:
        return None
    title_box, end_box = atoms[title_index]["box"], atoms[ends[0]]["box"]
    if end_box[1] < title_box[1] + title_box[3]:
        return None
    if (len(source_frames) > 3 or any(not _valid_atoms(frame) for frame in source_frames)):
        return None
    frozen_sources = tuple(tuple({"text": atom["text"], "box": tuple(atom["box"])} for atom in frame) for frame in source_frames)
    sources = PItemSourceTextFrames(list(frozen_sources))
    included = tuple(index for index, atom in enumerate(atoms)
                     if atom["box"][1] < end_box[1] + end_box[3]
                     and atom["box"][1] + atom["box"][3] > title_box[1]
                     and (index in (title_index, ends[0]) or not _source_background(atom, sources, title_box[3])))
    boxes = [atoms[index]["box"] for index in included]
    pad = min(4, max(2, title_box[3] * .1))
    left, top = max(0, min(box[0] for box in boxes) - pad), max(0, min(box[1] for box in boxes) - pad)
    right = min(coordinate_size[0], max(box[0] + box[2] for box in boxes) + pad)
    bottom = min(coordinate_size[1], max(box[1] + box[3] for box in boxes) + pad)
    return BodyRereadPlan((left, top, right - left, bottom - top), family, bases[0], title, trailer,
                          included, tuple(references), _signature(atoms), frozen_sources, original_result)


def _body_atoms(atoms, title_index, end_index, sources):
    title_box, end_box = atoms[title_index]["box"], atoms[end_index]["box"]
    return [atom for index, atom in enumerate(atoms) if index != title_index
            and atom["box"][1] < end_box[1] + end_box[3]
            and atom["box"][1] + atom["box"][3] > title_box[1] + title_box[3]
            and (index == end_index or not _source_background(atom, sources, title_box[3]))]


def _body_text(atoms):
    # A newly recognized complete symbol is semantic OCR evidence. No digit,
    # sign, Latin/CJK glyph, shape or brightness is classified as decoration.
    retained = [atom for atom in atoms if not (len(_canonical(atom["text"])) == 1
                and unicodedata.category(_canonical(atom["text"])) == "So")]
    ordered = [{"text": text, "box": box} for _, _, row in _spatial_ocr_rows(retained) for text, box in row]
    text, _ = atom_text_view(ordered, {"atoms": [list(range(len(ordered)))]})
    return text


def contradicted_body_ids(plan: BodyRereadPlan, atoms: Sequence[Mapping]) -> frozenset[int]:
    """Retain explicit title/field conflicts, including an unclosed unique label.

    Missing rows, empty observations and lone OCR fragments are not evidence
    for either variant. A trailer is not needed to retain an already complete
    conflicting field; it is still mandatory for positive fusion.
    """
    if not _valid_atoms(atoms):
        return frozenset()
    titles = [atom for atom in atoms if _canonical(atom["text"]).removesuffix("+") == plan.title_text]
    if len(titles) != 1:
        return frozenset()
    sources = PItemSourceTextFrames(list(plan.source_frames))
    title_box = titles[0]["box"]
    bottom = plan.roi[1] + plan.roi[3]
    body = [atom for atom in atoms if atom is not titles[0]
            and atom["box"][1] < bottom and atom["box"][1] + atom["box"][3] > title_box[1] + title_box[3]
            and not _source_background(atom, sources, title_box[3])]
    text = _body_text(body)
    rejected = {item_id for item_id, name, _ in plan.references
                if _canonical(titles[0]["text"]).endswith("+") and not name.endswith("+")}
    rules = {}
    for _, _, paragraphs in plan.references:
        for paragraph in paragraphs:
            literals, values, pattern = source_field(paragraph)
            if values:
                rules[literals] = re.compile(pattern)
    for literals, pattern in rules.items():
        observed = {match.groups() for match in pattern.finditer(text)}
        unique_head = bool(_is_complete_numeric_field(literals, ("0",)) and literals[0] in text)
        for item_id, _, paragraphs in plan.references:
            allowed = {match.groups() for paragraph in paragraphs for match in pattern.finditer(paragraph)}
            if ((observed and not observed.issubset(allowed))
                    or (unique_head and not any(literals[0] in paragraph for paragraph in paragraphs))):
                rejected.add(item_id)
    for atom in body:
        literals, values, pattern = source_field(_canonical(atom["text"]))
        if not _is_complete_numeric_field(literals, values):
            continue
        compiled = re.compile(pattern)
        for item_id, _, paragraphs in plan.references:
            allowed = {match.groups() for paragraph in paragraphs for match in compiled.finditer(paragraph)}
            if tuple(values) not in allowed:
                rejected.add(item_id)
    return frozenset(rejected)


def _protection(text):
    characters = {index for index, char in enumerate(text)
                  if char.isnumeric() or char in "+-−.,%" + BOUNDARY or unicodedata.category(char).startswith("P")}
    gaps = set()
    for match in _SEMANTIC.finditer(text):
        characters.update(range(match.start(), match.end()))
        gaps.update(range(match.start() + 1, match.end()))
    return characters, gaps


def _align_original_body(plan, original_atoms, original_body, expected):
    """At most five separated character edits, covering both views in order."""
    lookup = {}
    for index, atom in enumerate(original_atoms):
        key = (atom["text"].strip(), tuple(round(value) for value in atom["box"]))
        if key in lookup:
            return None
        lookup[key] = index
    rows = [[lookup[(text, tuple(box))] for text, box in row]
            for _, _, row in _spatial_ocr_rows(original_body)]
    paragraphs = {paragraph for _, _, values in plan.references for paragraph in values}
    field_atoms = {index for field in plan.original_result.diagnostics.get("observed_fields", ())
                   for hit in field.get("observations", ()) for index in hit.get("atom_indices", ())}
    tails = {item["tail_atom_index"] for item in _adjacent_operand_fragments(
        list(original_atoms), {"atoms": rows}, plan.original_result)}
    fragments, anchors, fragment_anchors = set(), set(), {}
    complete_atoms = [index for row in rows for index in row
                      if _canonical(original_atoms[index]["text"]) in paragraphs
                      and sum(char.isalpha() for char in original_atoms[index]["text"]) >= 2]
    for row in rows:
        for index in row:
            box = original_atoms[index]["box"]
            complete = [other for other in complete_atoms if other != index
                        and abs(box[1] + box[3] / 2 - original_atoms[other]["box"][1] - original_atoms[other]["box"][3] / 2)
                        <= .45 * min(box[3], original_atoms[other]["box"][3])]
            text = _canonical(original_atoms[index]["text"])
            if (len(_DISPLAY_UNIT.findall(text)) != 1 or not complete or index in complete
                    or index in field_atoms or index in tails or text in paragraphs
                    or any(char in "+-−.,%不無无非未禁除" for char in text)
                    or unicodedata.category(text[0]) == "So"):
                continue
            # A close digit immediately before a numeric paragraph is also a
            # possible operand prefix. It is not an unowned icon fragment.
            x, y, w, h = original_atoms[index]["box"]
            if any(min(x + w, original_atoms[other]["box"][0] + original_atoms[other]["box"][2])
                   > max(x, original_atoms[other]["box"][0]) for other in complete):
                continue
            numeric_edge = any(
                (_canonical(original_atoms[other]["text"])[0].isdigit()
                 and -.25 * min(h, original_atoms[other]["box"][3])
                 <= original_atoms[other]["box"][0] - x - w <= .5 * min(h, original_atoms[other]["box"][3]))
                or ((_canonical(original_atoms[other]["text"])[-1].isdigit()
                     or _canonical(original_atoms[other]["text"]).endswith("%"))
                    and -.25 * min(h, original_atoms[other]["box"][3])
                    <= x - original_atoms[other]["box"][0] - original_atoms[other]["box"][2]
                    <= .5 * min(h, original_atoms[other]["box"][3]))
                for other in complete)
            if numeric_edge:
                continue
            fragments.add(index)
            fragment_anchors[index] = set(complete)
            anchors.update(complete)
    # Each entry is (display character, original atom, source character
    # offset, forced unowned-fragment edit). Known So decorations need no edit.
    stream, raw_parts, previous, previous_atom = [], [], "", None
    for index in (index for row in rows for index in row):
        atom = original_atoms[index]
        text = _canonical(atom["text"])
        if not text or (len(text) == 1 and unicodedata.category(text) == "So"):
            continue
        if index in fragments:
            stream.append((text, index, len("".join(raw_parts)), True))
            continue
        probe = [{"text": previous, "box": previous_atom["box"]}, {"text": text, "box": atom["box"]}] if previous_atom else [{"text": text, "box": atom["box"]}]
        joined, _ = atom_text_view(probe, {"atoms": [list(range(len(probe)))]})
        separator = BOUNDARY if previous_atom and joined[len(previous):].startswith(BOUNDARY) else ""
        if separator:
            stream.append((separator, index, len("".join(raw_parts)), False))
            raw_parts.append(separator)
        offset = len("".join(raw_parts))
        stream.extend((match.group(), index, offset + match.start(), False) for match in _DISPLAY_UNIT.finditer(text))
        raw_parts.append(text)
        previous, previous_atom = text, atom
    raw = "".join(raw_parts)
    if tuple(NUMBER.findall(raw)) != tuple(NUMBER.findall(expected)):
        return None
    target = [(match.group(), match.start()) for match in _DISPLAY_UNIT.finditer(expected)]
    if max(len(stream), len(target)) > 1024 or abs(len(stream) - len(target)) > _MAX_ANOMALIES:
        return None
    original_protected, original_gaps = _protection(raw)
    expected_protected, expected_gaps = _protection(expected)
    protected = [not fragment and (atom_index in anchors
                 or len(_DISPLAY_UNIT.findall(_canonical(original_atoms[atom_index]["text"]))) == 1
                 or any(offset + i in original_protected for i in range(len(char))))
                 for char, atom_index, offset, fragment in stream]
    target_protected = [any(offset + i in expected_protected for i in range(len(char))) for char, offset in target]
    queue, visited = deque([(0, 0, (), False)]), set()
    while queue:
        i, j, edits, adjacent = queue.popleft()
        previous_proof = (edits[-1]["kind"], edits[-1]["atom_index"], tuple(edits[-1].get("anchor_atom_indices", ()))) if adjacent else None
        state = (i, j, len(edits), previous_proof)
        if state in visited:
            continue
        visited.add(state)
        if len(visited) > 32768:
            return None
        if i == len(stream) and j == len(target):
            return {"anomaly_count": len(edits), "anomalies": list(edits)}
        if i < len(stream) and not stream[i][3] and j < len(target) and stream[i][0] == target[j][0]:
            queue.appendleft((i + 1, j + 1, edits, False))
        shared_anchors = set()
        if adjacent and i < len(stream) and stream[i][3] and edits[-1]["kind"] == "independent_fragment":
            previous_index, current_index = edits[-1]["atom_index"], stream[i][1]
            px, py, pw, ph = original_atoms[previous_index]["box"]
            cx, cy, _, ch = original_atoms[current_index]["box"]
            # Distinct, clearly separated glyph atoms may share a textual
            # anchor without an intervening text character. A run of letters
            # is still a possible word, never a sequence of removable icons.
            if (previous_index != current_index and cx - px - pw >= min(ph, ch)
                    and abs(py + ph / 2 - cy - ch / 2) <= .45 * min(ph, ch)
                    and not (edits[-1]["observed"].isalpha() and stream[i][0].isalpha())):
                shared_anchors = fragment_anchors[current_index] & set(edits[-1].get("anchor_atom_indices", ()))
        if (adjacent and not shared_anchors) or len(edits) >= _MAX_ANOMALIES:
            continue
        if i < len(stream):
            char, atom_index, offset, fragment = stream[i]
            target_gap = target[j][1] if j < len(target) else len(expected)
            if fragment or (not protected[i] and target_gap not in expected_gaps):
                edit = {"kind": "independent_fragment" if fragment else "delete", "atom_index": atom_index,
                        "source_offset": offset, "observed": char, "reference": ""}
                if fragment:
                    edit["anchor_atom_indices"] = sorted(shared_anchors or fragment_anchors[atom_index])
                queue.append((i + 1, j, edits + (edit,), True))
            if fragment:
                continue
            if j < len(target) and not protected[i] and not target_protected[j]:
                queue.append((i + 1, j + 1, edits + ({"kind": "substitute", "atom_index": atom_index,
                                                   "source_offset": offset, "observed": char, "reference": target[j][0]},), True))
        original_gap = stream[i][2] if i < len(stream) else len(raw)
        if j < len(target) and not target_protected[j] and original_gap not in original_gaps:
            atom_index = stream[min(i, len(stream) - 1)][1] if stream else None
            queue.append((i, j + 1, edits + ({"kind": "insert", "atom_index": atom_index,
                                            "source_offset": original_gap, "observed": "", "reference": target[j][0]},), True))
    return None


def fuse_body_reread(
    plan: BodyRereadPlan, original_atoms: Sequence[Mapping], reread_atoms: Sequence[Mapping],
    reread_result: PItemDetailTextResult,
) -> PItemDetailTextResult | None:
    """Accept only a complete new body consistent with all original semantics."""
    if (not _valid_atoms(original_atoms) or _signature(original_atoms) != plan.original_signature
            or not _valid_atoms(reread_atoms) or not _clear_result(reread_result)
            or reread_result.status != "unique" or len(reread_result.ids) != 1):
        return None
    item_id = reread_result.ids[0]
    references = {item: (name, paragraphs) for item, name, paragraphs in plan.references}
    if (item_id not in references or item_id not in plan.original_result.ids or reread_result.diagnostics.get("conflicts", {}).get(item_id)
            or "complete_reference_positive" not in reread_result.diagnostics.get("positive_support", {}).get(item_id, ())):
        return None
    if item_id in (contradicted_body_ids(plan, original_atoms) | contradicted_body_ids(plan, reread_atoms)):
        return None
    left, top, width, height = plan.roi
    if any(not (left <= atom["box"][0] and top <= atom["box"][1]
                and atom["box"][0] + atom["box"][2] <= left + width
                and atom["box"][1] + atom["box"][3] <= top + height) for atom in reread_atoms):
        return None
    titles = [index for index, atom in enumerate(reread_atoms)
              if _canonical(atom["text"]).removesuffix("+") == plan.title_text]
    ends = [index for index, atom in enumerate(reread_atoms) if _canonical(atom["text"]) == plan.trailer_text]
    if len(titles) != 1 or len(ends) != 1:
        return None
    title_index, end_index = titles[0], ends[0]
    if _canonical(reread_atoms[title_index]["text"]).endswith("+") and not references[item_id][0].endswith("+"):
        return None
    if reread_atoms[end_index]["box"][1] < sum((reread_atoms[title_index]["box"][1], reread_atoms[title_index]["box"][3])):
        return None
    sources = PItemSourceTextFrames(list(plan.source_frames))
    expected = "".join(references[item_id][1])
    if _body_text(_body_atoms(reread_atoms, title_index, end_index, sources)) != expected:
        return None
    original_title = next(index for index, atom in enumerate(original_atoms) if _canonical(atom["text"]) == plan.title_text)
    original_end = next(index for index, atom in enumerate(original_atoms) if _canonical(atom["text"]) == plan.trailer_text)
    original_body = _body_atoms(original_atoms, original_title, original_end, sources)
    # Missing family-specific action labels are not evidence of the ordinary
    # variant. Even an unclosed observed recovery field must survive fusion.
    original_text = _body_text(original_body)
    for _, _, paragraphs in plan.references:
        for paragraph in paragraphs:
            literals, values, _ = source_field(paragraph)
            if (_is_complete_numeric_field(literals, values) and literals[0] in original_text
                    and literals[0] not in expected):
                return None
    alignment = _align_original_body(plan, original_atoms, original_body, expected)
    if alignment is None:
        return None
    if plan.original_result.diagnostics.get("conflicts", {}).get(item_id):
        return None
    return replace(reread_result, reason="complete_reference_body_reread", diagnostics={
        **reread_result.diagnostics,
        "body_reread": {"roi": plan.roi, "original_atom_indices": plan.included_indices,
                        "complete_reference_id": item_id, "original_atoms_modified": False, **alignment},
    })
