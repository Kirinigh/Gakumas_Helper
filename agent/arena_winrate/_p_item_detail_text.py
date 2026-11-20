"""Fixed P-item text evidence, independent of OCR and image-ranking candidates."""
from __future__ import annotations

import re
import json
import math
import unicodedata
from typing import Any, Literal
from pathlib import Path
from dataclasses import field, replace, dataclass
from collections.abc import Mapping, Sequence

BOUNDARY = "\u2063"
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)?%?")
OBSERVED_NUMBER = r"([+-]?[0-9]+(?:[.,][0-9]+)?%?)(?![0-9.,%])"
_DETAIL_COUNT_TRAILER = re.compile(r"\(試験・ステージ内[1-9][0-9]*回\)\Z")


@dataclass(frozen=True, slots=True)
class PItemDetailTextResult:
    status: Literal["unique", "ambiguous", "unknown"]
    ids: tuple[int, ...] = ()
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def retain_native_detail_vetoes(
    result: PItemDetailTextResult, vetoes: dict[int, list[dict]],
) -> PItemDetailTextResult:
    """Keep contradictions across interpretations of the same observation."""
    if not vetoes:
        return result
    diagnostics = result.diagnostics
    conflicts = diagnostics.get("conflicts", {})
    retained = {item_id: list(entries) for item_id, entries in diagnostics.get("native_detail_vetoes", {}).items()}
    for item_id, entries in vetoes.items():
        seen = retained.setdefault(item_id, [])
        seen.extend(entry for entry in entries if entry not in seen)
    if result.status == "unique" and any(item_id in retained for item_id in result.ids):
        result = replace(result, status="unknown", ids=(), reason="excluded_native_detail_conflict")
    return replace(result, diagnostics={
        **diagnostics,
        "native_detail_vetoes": retained,
        "conflicts": {
            **retained,
            **{item_id: entries for item_id, entries in conflicts.items() if entries or item_id not in retained},
        },
        "positive_support": {
            item_id: entries if item_id not in retained else []
            for item_id, entries in diagnostics.get("positive_support", {}).items()
        },
    })


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _normalize_detail_atom(text: str) -> str:
    return normalize(text).translate(str.maketrans({"值": "値", "增": "増", "·": "・"})).replace("体力消费", "体力消費")


def _valid_detail_atoms(atoms: Sequence[Mapping], size: tuple[int, int] | None = None) -> bool:
    try:
        for atom in atoms:
            box = atom["box"]
            if (not isinstance(atom["text"], str) or len(box) != 4
                    or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                           for value in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0):
                return False
            if size is not None and (box[0] + box[2] > size[0] or box[1] + box[3] > size[1]):
                return False
        return True
    except (KeyError, TypeError, OverflowError):
        return False


def bounds(atom: dict) -> tuple[float, float, float, float]:
    x, y, w, h = atom["box"]
    return x, y, x + w, y + h

def union(atoms: list[dict]) -> tuple[float, float, float, float]:
    boxes = [bounds(atom) for atom in atoms]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)

def same_source_position(a: dict, b: dict, height: float) -> bool:
    ax, ay, ar, ab = bounds(a)
    bx, by, br, bb = bounds(b)
    overlap = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    denominator = min((ar - ax) * (ab - ay), (br - bx) * (bb - by))
    return bool(denominator > 0 and overlap / denominator >= 0.8 and abs((ax + ar) - (bx + br)) <= height and abs((ay + ab) - (by + bb)) <= height)

class PItemSourceTextFrames:
    def __init__(self, frames: list[list[dict]]):
        if len(frames) > 3:
            raise ValueError("at most three same-transaction source frames")
        self.frames = []
        for frame in frames:
            index = {}
            for atom in frame:
                index.setdefault(normalize(atom["text"]), []).append(atom)
            self.frames.append(index)

    def matching_frames(self, atom: dict, height: float) -> list[int]:
        key = normalize(atom["text"])
        return [i for i, frame in enumerate(self.frames) if any(same_source_position(atom, other, height) for other in frame.get(key, ()))]

def component_rows(atoms: list[dict], height: float) -> list[dict]:
    rows = []
    for atom in sorted(atoms, key=lambda a: (a["box"][1] + a["box"][3] / 2, a["box"][0])):
        center = atom["box"][1] + atom["box"][3] / 2
        matching = [row for row in rows if abs(row["center"] - center) <= 0.45 * min(row["height"], atom["box"][3])]
        if matching:
            min(matching, key=lambda row: abs(row["center"] - center))["atoms"].append(atom)
        else:
            rows.append({"center": center, "height": atom["box"][3], "atoms": [atom]})
    output = []
    for row in rows:
        # Only multi-letter atoms may connect horizontal text regions.
        strong = sorted((a for a in row["atoms"] if sum(c.isalpha() for c in a["text"]) >= 2), key=lambda a: a["box"][0])
        weak = [a for a in row["atoms"] if a not in strong]
        components = []
        for atom in strong:
            if components and bounds(atom)[0] - components[-1]["core_box"][2] <= 2 * height:
                components[-1]["atoms"].append(atom)
                components[-1]["core_box"] = union(components[-1]["atoms"])
            else:
                components.append({"atoms": [atom], "core_box": bounds(atom), "strong": True})
        for atom in weak:
            left, _, right, _ = bounds(atom)
            distances = [(max(0, component["core_box"][0] - right, left - component["core_box"][2]), component) for component in components if component["strong"]]
            nearest = min(distances, key=lambda pair: pair[0]) if distances else None
            if nearest is not None and nearest[0] <= 2 * height:
                nearest[1]["atoms"].append(atom)
            else:
                components.append({"atoms": [atom], "core_box": bounds(atom), "strong": False})
        for component in components:
            component["atoms"].sort(key=lambda a: (a["box"][0], a["index"]))
            component["box"] = union(component["atoms"])
        output.append({"components": sorted(components, key=lambda c: c["box"][0]), "center": row["center"]})
    return output

def body_layout(atoms: list[dict], title_index: int, sources: PItemSourceTextFrames, panel_box: tuple[int, int, int, int] | None = None, *, allow_wrapped_gap: bool = False) -> dict:
    title = atoms[title_index]
    tx, _, _, title_bottom = bounds(title)
    height = float(title["box"][3])
    if height <= 0:
        return {"status": "unknown", "reason": "invalid_title_height", "body": "", "atoms": []}
    eligible, rejected, decorations = [], [], []
    for index, raw in enumerate(atoms):
        atom = {**raw, "index": index}
        if index == title_index:
            continue
        key = normalize(atom["text"])
        if not key:
            rejected.append({"index": index, "reason": "empty_source_text"})
            continue
        matched_sources = sources.matching_frames(atom, height)
        if len(matched_sources) >= 2:
            rejected.append({"index": index, "reason": "stable_source_background", "source_frame_indices": matched_sources})
            continue
        if panel_box is not None:
            px, py, pw, ph = panel_box
            ax, ay, ar, ab = bounds(atom)
            spill = max(px - ax, py - ay, ar - px - pw, ab - py - ph)
            if spill > 0:
                overlap_width = min(ar, px + pw) - max(ax, px)
                overlap_height = min(ab, py + ph) - max(ay, py)
                # Give detector boxes a letter-sized tolerance when most of
                # the atom is still in the observed panel. Never clip its text.
                within_tolerance = (spill <= .5 * height and overlap_width >= .75 * (ar - ax)
                                    and overlap_height >= .75 * (ab - ay))
                if not within_tolerance:
                    if px <= (ax + ar) / 2 <= px + pw and py <= (ay + ab) / 2 <= py + ph:
                        return {"status": "unknown", "reason": "panel_atom_crosses_boundary", "body": "", "atoms": []}
                    rejected.append({"index": index, "reason": "outside_confirmed_detail_panel"})
                    continue
        x, y, w, h = atom["box"]
        if y + h / 2 < title_bottom or x < tx - 0.5 * height:
            rejected.append({"index": index, "reason": "outside_title_body_origin"})
            continue
        if len(key) == 1 and unicodedata.category(key) == "So":
            decorations.append({"index": index, "reason": "complete_symbol_decoration", "text": atom["text"], "box": atom["box"]})
            continue
        eligible.append(atom)
    selected, uncertainty, wrapped_gaps = [], [], []
    last_bottom = title_bottom
    body_left, body_right = tx, tx
    started = False
    for row in component_rows(eligible, height):
        candidates = []
        for component in row["components"]:
            left, top, right, bottom = component["box"]
            anchor = tx - 0.5 * height <= component["core_box"][0] <= tx + 4 * height
            if component["strong"] and anchor:
                candidates.append(component)
            elif started and not component["strong"] and left <= body_right + 2 * height and right >= body_left - 2 * height:
                candidates.append(component)
            else:
                if started and left <= body_right + 2 * height and right >= body_left - 2 * height and top <= last_bottom + height:
                    uncertainty.append({"reason": "nearby_unassociated_component", "atom_indices": [a["index"] for a in component["atoms"]]})
                rejected.extend({"index": a["index"], "reason": "outside_connected_body_column"} for a in component["atoms"])
        if not candidates:
            continue
        top = min(component["box"][1] for component in candidates)
        if top - last_bottom > height:
            if not started:
                return {"status": "unknown", "reason": "body_not_rendered_next_to_title", "body": "", "atoms": [], "rejected": rejected, "decorations": decorations}
            # Failure-only candidate layout. A confirmed panel and a long row
            # reaching its right edge can delimit one omitted wrap row. This
            # does not itself supply any text or authorize a result.
            if (allow_wrapped_gap and panel_box is not None and not wrapped_gaps
                    and top - last_bottom <= 2 * height
                    and max(bounds(a)[2] for a in selected[-1]) >= panel_box[0] + panel_box[2] - 1.5 * height):
                wrapped_gaps.append({"after_atom_index": selected[-1][-1]["index"], "gap": top - last_bottom})
            else:
                rejected.extend({"index": a["index"], "reason": "after_body_geometric_gap"} for later in component_rows([a for a in eligible if bounds(a)[1] >= top], height) for component in later["components"] for a in component["atoms"])
                break
        if panel_box is None and len([component for component in candidates if component["strong"]]) > 1:
            uncertainty.append({"reason": "multiple_body_components_same_row", "atom_indices": [a["index"] for c in candidates for a in c["atoms"]]})
        joined_atoms = sorted((a for component in candidates for a in component["atoms"]), key=lambda a: (a["box"][0], a["index"]))
        selected.append(joined_atoms)
        last_bottom = max(bounds(atom)[3] for atom in joined_atoms)
        for component in candidates:
            if component["strong"]:
                body_left = min(body_left, component["core_box"][0])
                body_right = max(body_right, component["core_box"][2])
        started = True
    body = "\n".join("".join(atom["text"] for atom in row) for row in selected)
    return {"status": "resolved" if started and not uncertainty else "unknown", "reason": "fixed_connected_body" if started and not uncertainty else "layout_uncertain" if uncertainty else "body_not_rendered", "body": body, "atoms": [[atom["index"] for atom in row] for row in selected], "rejected": rejected, "decorations": decorations, "uncertainty": uncertainty, "title_height": height, "wrapped_gaps": wrapped_gaps}

def atom_text_view(atoms: list[dict], layout: dict) -> tuple[str, list[dict]]:
    """Reuse frozen layout, retaining atom spans and independent numeric edges."""
    parts, spans = [], []
    offset = 0
    previous = ""
    previous_atom = None
    for row in layout["atoms"]:
        for index in row:
            # Native OCR can emit the simplified glyph in this fixed cost
            # label. Keep every operand and the original atom boundary intact.
            text = _normalize_detail_atom(atoms[index]["text"])
            if not text:
                continue
            # A percentage is also a complete numeric edge. An independent
            # decoration digit must not turn +25% into the malformed +25%4.
            numeric_edge = previous and (
                previous[-1].isdigit()
                or (previous.endswith("%") and len(previous) > 1 and previous[-2].isdigit())
            )
            # A lone signed value may genuinely wrap onto its own line.
            # A leading digit beside another complete text row (e.g. the
            # trial-count footer) cannot instead finish the previous effect.
            # Keep numeric unit fragments attached to their own value.
            separate_text_row_number = (
                previous_atom is not None and index == row[0] and NUMBER.fullmatch(text)
                and any(sum(char.isalpha() for char in atoms[other]["text"]) >= 2
                        and normalize(atoms[other]["text"]) not in {"ターン", "回", "枚", "倍", "秒", "個"}
                        for other in row[1:])
            )
            if (numeric_edge and text[0].isdigit()) or separate_text_row_number:
                parts.append(BOUNDARY)
                offset += 1
            parts.append(text)
            spans.append({"start": offset, "end": offset + len(text), "atom_index": index})
            offset += len(text)
            previous = text
            previous_atom = atoms[index]
    return "".join(parts), spans

def token_span_valid(text: str, start: int, end: int, spans: list[dict]) -> bool:
    """A short label must not start/end inside a different word in one atom."""
    left = next((span for span in spans if span["start"] <= start < span["end"]), None)
    right = next((span for span in spans if span["start"] < end <= span["end"]), None)
    if left is None or right is None or BOUNDARY in text[start:end]:
        return False
    if start > left["start"] and text[start - 1] not in "、,。;:()[]【】・":
        return False
    if end < right["end"] and text[end] not in "、,。;:()[]【】・":
        return False
    return True

def source_field(text: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Alternating literal fragments/numbers; no DSL or trigger vocabulary."""
    text = normalize(text)
    literals, values = [], []
    offset = 0
    for token in NUMBER.finditer(text):
        literals.append(text[offset:token.start()])
        values.append(token.group())
        offset = token.end()
    literals.append(text[offset:])
    pattern = OBSERVED_NUMBER.join(_source_literal_pattern(part) for part in literals)
    return tuple(literals), tuple(values), pattern


def _adjacent_operand_fragments(atoms: list[dict], layout: dict, result: PItemDetailTextResult) -> list[dict]:
    """Keep close numeric OCR tails uncertain instead of consuming a prefix.

    The atom separator protects 4 + 2 from becoming 42, but it does not prove
    that the first atom is the complete operand. Only same-row, touching
    numeric tails of an actually observed field participate; distant counters
    and numbers on another row keep their original independent boundaries.
    """
    field_atoms = {index for field in result.diagnostics.get("observed_fields", ())
                   for hit in field.get("observations", ()) for index in hit.get("atom_indices", ())}
    fragments = []
    for row in layout["atoms"]:
        for left_index, right_index in zip(row, row[1:]):
            left, right = atoms[left_index], atoms[right_index]
            before, after = normalize(left["text"]), normalize(right["text"])
            if (left_index not in field_atoms or not before
                    or not (before[-1].isdigit() or before.endswith("%"))
                    or re.fullmatch(r"[+\-0-9.,%]+", after) is None):
                continue
            lx, ly, lw, lh = left["box"]
            rx, ry, _, rh = right["box"]
            height = min(lh, rh)
            gap = rx - lx - lw
            if (height > 0 and -.25 * height <= gap <= .5 * height
                    and abs(ly + lh / 2 - ry - rh / 2) <= .45 * height):
                fragments.append({"field_atom_index": left_index, "tail_atom_index": right_index,
                                  "observed_tail": right["text"], "gap": gap})
    return fragments


def _source_literal_pattern(literal: str) -> str:
    """Accept the rendered cost icon only at its reference-defined label.

    OCR may merge the heart into the following cost label. Match that known
    text marker without editing atoms or operands; every reference value and
    the original whole-body equality check keep their existing contracts.
    Explicit icons already present in a reference remain literal.
    """
    escaped = re.escape(literal)
    # Only a reference that explicitly names FixedStamina may accept the old
    # rendering without its prefix. Existing short labels keep their boundaries.
    escaped = escaped.replace("固定体力消費", r"(?:固定)?体力消費")
    if "❤" in literal or "♥" in literal:
        return escaped
    return escaped.replace("コスト値", r"(?:[❤♥][\ufe0e\ufe0f]?)?コスト値")


def _default_reference_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    installed = root / "data/p_item_detail_text.json"
    return installed if installed.is_file() else root / "assets/data/p_item_detail_text.json"


def _binding_matches(active: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    fields = ("id", "name", "mode", "sourceType", "pIdolId", "upgraded", "plan", "rarity")
    return bool(
        isinstance(active, Mapping) and isinstance(reference, Mapping)
        and all(active.get(key) == reference.get(key) for key in fields)
        and isinstance(active.get("effects"), str)
        and re.sub(r"\s+", "", active["effects"])
        == re.sub(r"\s+", "", str(reference.get("effects", "")))
    )


def _is_complete_numeric_field(literals: tuple[str, ...], values: tuple[str, ...]) -> bool:
    # Fixed source action fields, not partial trigger clauses or a keyword list.
    return bool(values and literals[0] and sum(c.isalpha() for c in literals[0]) >= 2
                and all(not part or all(c.isalpha() or c == "・" for c in part) for part in literals))


def _native_body_conflict_ids(body: list[dict], paragraphs_by_id: Mapping[int, tuple[str, ...]]) -> set[int]:
    """Check intact fields and explicit variant labels, without supplying operands."""
    from ._reader_visual import _spatial_ocr_rows

    retained = [atom for atom in body if not (len(value := _normalize_detail_atom(atom["text"])) == 1
                and unicodedata.category(value) == "So")]
    ordered = [{"text": text, "box": box} for _, _, row in _spatial_ocr_rows(retained) for text, box in row]
    text, _ = atom_text_view(ordered, {"atoms": [list(range(len(ordered)))]})
    rules = {}
    for paragraphs in paragraphs_by_id.values():
        for paragraph in paragraphs:
            literals, values, pattern = source_field(paragraph)
            if values:
                rules[literals] = re.compile(pattern)
    rejected = set()
    for literals, pattern in rules.items():
        observed = {match.groups() for match in pattern.finditer(text)}
        complete_label = _is_complete_numeric_field(literals, ("0",)) and literals[0] in text
        for item_id, paragraphs in paragraphs_by_id.items():
            allowed = {match.groups() for paragraph in paragraphs for match in pattern.finditer(paragraph)}
            if ((observed and not observed.issubset(allowed))
                    or (complete_label and not any(literals[0] in paragraph for paragraph in paragraphs))):
                rejected.add(item_id)
    for atom in body:
        literals, values, pattern = source_field(_normalize_detail_atom(atom["text"]))
        if not _is_complete_numeric_field(literals, values):
            continue
        compiled = re.compile(pattern)
        for item_id, paragraphs in paragraphs_by_id.items():
            allowed = {match.groups() for paragraph in paragraphs for match in compiled.finditer(paragraph)}
            if values not in allowed:
                rejected.add(item_id)
    return rejected


def _shared_scalar_after_bullet(
    bullet: dict, field: dict, paragraphs: set[str], shared_paragraphs: set[str], height: float,
) -> str | None:
    """Prove a separate paragraph mark before a complete shared signed field."""
    bx, by, bw, bh = bullet["box"]
    fx, fy, _, fh = field["box"]
    if not (0 < bw <= height and 0 < bh <= height and 0 < fh
            and 0 <= fx - bx - bw <= height
            and abs(by + bh / 2 - fy - fh / 2) <= .45 * min(bh, fh)):
        return None
    text = normalize(field["text"])
    # A letter attached to the front of an exact catalog field is one OCR
    # anomaly. It cannot supply a label, operand, sign, unit or condition.
    if text not in paragraphs and re.match(r"[A-Za-z]", text):
        text = text[1:]
    if text not in shared_paragraphs:
        return None
    literals, values, _ = source_field(text)
    if len(values) != 1 or not values[0].startswith("+") or literals[-1] or not _is_complete_numeric_field(literals, values):
        return None
    return text


def _detached_title_plus(
    atoms: list[dict], title_index: int, sources: PItemSourceTextFrames, size: tuple[int, int],
) -> int | None:
    """Find a separate plus on the title baseline, independent of body operands."""
    _, title_top, title_right, title_bottom = bounds(atoms[title_index])
    height = title_bottom - title_top
    suffixes = []
    for index, atom in enumerate(atoms):
        if index == title_index or normalize(atom["text"]) != "+":
            continue
        x, y, width, sign_height = atom["box"]
        if (0 < width <= height and 0 < sign_height <= height
                and -.25 * height <= x - title_right <= .75 * height
                and abs(y + sign_height / 2 - (title_top + title_bottom) / 2) <= .5 * sign_height
                and _valid_detail_atoms((atom,), size)
                and len(sources.matching_frames(atom, height)) < 2):
            suffixes.append(index)
    if len(suffixes) != 1:
        return None
    index = suffixes[0]
    sx, sy, sw, sh = atoms[index]["box"]
    for atom in atoms:
        x, y, _, atom_height = atom["box"]
        if (NUMBER.match(normalize(atom["text"])) and -.25 * sh <= x - sx - sw <= sh
                and abs(y + atom_height / 2 - sy - sh / 2) <= .45 * min(sh, atom_height)):
            return None
    return index


class PItemDetailTextIndex:
    """One catalog-owned index; unrelated catalog additions do not invalidate it."""

    def __init__(
        self,
        p_items: Sequence[Mapping[str, Any]],
        *,
        reference_data: Mapping[str, Any] | None = None,
        reference_path: str | Path | None = None,
        candidate_ids_by_plan: Mapping[str | None, frozenset[int]] | None = None,
    ) -> None:
        if reference_data is None:
            path = Path(reference_path) if reference_path is not None else _default_reference_path()
            reference_data = json.loads(path.read_text(encoding="utf-8"))
        if reference_data.get("schema_version") != 1 or not isinstance(reference_data.get("rows"), list):
            raise ValueError("P-item text reference schema is invalid")
        self._candidate_ids_by_plan = candidate_ids_by_plan
        self.revision = str(reference_data.get("revision", ""))
        self._active = {int(row["id"]): dict(row) for row in p_items}
        self._references = {int(row["id"]): row for row in reference_data["rows"]}
        self.available_ids = tuple(sorted(item_id for item_id, row in self._active.items()
                                          if item_id in self._references
                                          and _binding_matches(row, self._references[item_id]["active"])))
        available = set(self.available_ids)
        self._bodies = {item_id: normalize("\n".join(self._references[item_id]["paragraphs"])) for item_id in available}
        self._titles: dict[str, set[int]] = {}
        groups: dict[tuple[Any, ...], set[int]] = {}
        for item_id, row in self._active.items():
            self._titles.setdefault(normalize(str(row.get("name", ""))), set()).add(item_id)
            if item_id in available:
                self._titles.setdefault(normalize(self._references[item_id]["name"]), set()).add(item_id)
            family = ("pIdol", row["pIdolId"]) if row.get("sourceType") == "pIdol" and row.get("pIdolId") is not None else ("item", item_id)
            groups.setdefault(family, set()).add(item_id)
        families = {item_id: members for members in groups.values() for item_id in members}
        self._title_scopes = {title: tuple(sorted(set().union(*(families[item_id] for item_id in direct)))) for title, direct in self._titles.items() if title}
        self._tables = {}
        for ids in set(self._title_scopes.values()):
            fields = {}
            for item_id in ids:
                if item_id not in available:
                    continue
                for paragraph in self._references[item_id]["paragraphs"]:
                    literals, values, pattern = source_field(paragraph)
                    entry = fields.setdefault(literals, {"literals": literals, "pattern": re.compile(pattern), "expected": {}, "duplicate_ids": []})
                    if item_id in entry["expected"]:
                        entry["duplicate_ids"].append(item_id)
                    entry["expected"][item_id] = values
            self._tables[ids] = fields
        complete_fields = {}
        for row in self._references.values():
            for paragraph in row["paragraphs"]:
                literals, values, pattern = source_field(paragraph)
                if _is_complete_numeric_field(literals, values):
                    complete_fields[literals] = re.compile(pattern)
        self._complete_fields = complete_fields
        self._family_known_fields = {}
        for ids in self._tables:
            known = {}
            for item_id in ids:
                if item_id not in available:
                    continue
                for paragraph in self._references[item_id]["paragraphs"]:
                    text = normalize(paragraph)
                    spans = [{"start": 0, "end": len(text), "atom_index": 0}]
                    for literals, pattern in complete_fields.items():
                        for match in pattern.finditer(text):
                            if token_span_valid(text, match.start(), match.end(), spans):
                                known.setdefault(literals, {}).setdefault(item_id, set()).add(tuple(match.groups()))
            self._family_known_fields[ids] = known

    def family_ids_for_title(
        self, title: str, *, plan: str | None = None, eligible_ids: frozenset[int] | None = None,
    ) -> tuple[int, ...]:
        family = self._title_scopes.get(normalize(title), ())
        if self._candidate_ids_by_plan is not None and plan in self._candidate_ids_by_plan:
            plan_ids = self._candidate_ids_by_plan[plan]
            family = tuple(item_id for item_id in family if item_id in plan_ids)
        else:
            family = tuple(item_id for item_id in family
                           if plan in (None, "", "free") or self._active[item_id].get("plan") in (None, "", "free", plan))
        return tuple(item_id for item_id in family if eligible_ids is None or item_id in eligible_ids)

    def _upgraded(self, item_id: int) -> bool:
        row = self._active[item_id]
        value = row.get("upgraded")
        return value if isinstance(value, bool) else normalize(str(row.get("name", ""))).endswith("+")

    def retain_native_body_conflicts(
        self, atoms, *, title_index, source_frames, plan, eligible_ids=None, result,
    ):
        """Keep explicit negative evidence from title/footer-bounded native text."""
        diagnostics = result.diagnostics
        conflicts = diagnostics.get("conflicts", {})
        if (result.status != "ambiguous" or result.reason != "insufficient_difference_evidence"
                or diagnostics.get("unbound_reference_ids") or diagnostics.get("layout", {}).get("uncertainty")
                or any(conflicts.get(item_id, conflicts.get(str(item_id))) for item_id in result.ids)
                or not _valid_detail_atoms(atoms, (720, 1280))
                or len(source_frames) > 3 or any(not _valid_detail_atoms(frame) for frame in source_frames)):
            return result
        title = _normalize_detail_atom(atoms[title_index]["text"])
        if not title or title.endswith("+") or sum(_normalize_detail_atom(atom["text"]) == title for atom in atoms) != 1:
            return result
        family = self.family_ids_for_title(atoms[title_index]["text"], plan=plan, eligible_ids=eligible_ids)
        if len(family) < 2 or not set(family).issubset(self.available_ids) or not result.ids or not set(result.ids).issubset(family):
            return result
        references = {item_id: self._references[item_id] for item_id in family}
        names = [_normalize_detail_atom(row["name"]) for row in references.values()]
        if names.count(title) != 1 or any(name.removesuffix("+") != title for name in names):
            return result
        if any(not isinstance(row["paragraphs"], (tuple, list)) for row in references.values()):
            return result
        paragraphs_by_id = {item_id: tuple(_normalize_detail_atom(p) for p in row["paragraphs"])
                            for item_id, row in references.items()}
        if any(not paragraphs or not all(paragraphs) for paragraphs in paragraphs_by_id.values()):
            return result
        trailers = {paragraphs[-1] for paragraphs in paragraphs_by_id.values()}
        if len(trailers) != 1 or not _DETAIL_COUNT_TRAILER.fullmatch(next(iter(trailers))):
            return result
        trailer = next(iter(trailers))
        ends = [atom for atom in atoms if _normalize_detail_atom(atom["text"]) == trailer]
        _, title_top, _, title_bottom = bounds(atoms[title_index])
        if len(ends) != 1 or ends[0]["box"][1] < title_bottom:
            return result
        title_height = title_bottom - title_top
        trailer_bottom = bounds(ends[0])[3]
        sources = PItemSourceTextFrames(source_frames)
        non_source = [atom for index, atom in enumerate(atoms) if index != title_index
                      and len(sources.matching_frames(atom, title_height)) < 2]
        # Retain the original observation's footer allowance. It bounds only
        # negative evidence and cannot authorize an identity or a field value.
        bottom = max(trailer_bottom, max((bounds(atom)[3] for atom in non_source
                                         if atom["box"][1] < trailer_bottom and bounds(atom)[3] > title_top), default=0))
        bottom = min(1280, bottom + min(4, max(2, title_height * .1)))
        body = [atom for atom in non_source if atom["box"][1] < bottom and bounds(atom)[3] > title_bottom]
        rejected = _native_body_conflict_ids(body, paragraphs_by_id)
        if not rejected:
            return result
        return replace(result, diagnostics={
            **diagnostics,
            "conflicts": {
                **conflicts,
                **{item_id: [*conflicts.get(item_id, ()), {"reason": "native_body_field_conflict"}]
                   for item_id in rejected},
            },
        })

    def _resolve(
        self, title: str, text: str, spans: list[dict], *, plan: str | None,
        eligible_ids: frozenset[int] | None = None,
        uncertain_atoms: set[int] | None = None,
        layout_diagnostics: Mapping[str, Any] | None = None,
    ) -> PItemDetailTextResult:
        title = normalize(title)
        family = self._title_scopes.get(title, ())
        scoped = self.family_ids_for_title(title, plan=plan, eligible_ids=eligible_ids)
        if not scoped:
            return PItemDetailTextResult("unknown", reason="title_not_in_catalog", diagnostics={"title_family_ids": family})
        uncertain_atoms = uncertain_atoms or set()
        available = set(self.available_ids)
        bound = set(scoped) & available
        unbound = set(scoped) - available
        fields = self._tables.get(family, {})
        # Restore only source-owned katakana words with a long vowel between
        # kana. A numeric minus, hyphenated label or longer OCR word is never
        # rewritten; equal-length edits retain every original atom boundary.
        long_vowels = {}
        for literals in fields:
            for literal in literals:
                for word in re.findall(r"[ァ-ヶー]+", literal):
                    if re.search(r"[ァ-ヶ]ー[ァ-ヶ]", word):
                        alias = re.sub(r"(?<=[ァ-ヶ])ー(?=[ァ-ヶ])", "-", word)
                        long_vowels.setdefault(alias, set()).add(word)
        vowel_edits = []
        for alias, words in long_vowels.items():
            if len(words) != 1:
                continue
            word = next(iter(words))
            pattern = re.compile(r"(?<![ァ-ヶー-])" + re.escape(alias) + r"(?![ァ-ヶー-])")
            for match in pattern.finditer(text):
                if BOUNDARY not in match.group():
                    vowel_edits.append({"span": (match.start(), match.end()), "observed": alias, "reference": word})
            text = pattern.sub(word, text)
        conflicts = {item_id: [] for item_id in scoped}
        support = {item_id: [] for item_id in scoped}
        observations, protected = [], []
        complete_hits = []
        family_hits = []
        for literals, entry in fields.items():
            for match in entry["pattern"].finditer(text):
                atom_ids = {span["atom_index"] for span in spans if span["end"] > match.start() and span["start"] < match.end()}
                if not (atom_ids & uncertain_atoms) and token_span_valid(text, match.start(), match.end(), spans):
                    family_hits.append((literals, match.start(), match.end()))
        for literals, pattern in self._complete_fields.items():
            for match in pattern.finditer(text):
                atom_ids = {span["atom_index"] for span in spans if span["end"] > match.start() and span["start"] < match.end()}
                if not (atom_ids & uncertain_atoms) and token_span_valid(text, match.start(), match.end(), spans):
                    complete_hits.append((literals, match.start(), match.end(), tuple(match.groups())))
        for literals, start, end, values in complete_hits:
            if any(other_start <= start and other_end >= end and (other_start, other_end) != (start, end)
                   for _, other_start, other_end in family_hits):
                continue
            if any(other_start <= start and other_end >= end and (other_start, other_end) != (start, end)
                   for _, other_start, other_end, _ in complete_hits):
                continue
            expected_values = self._family_known_fields.get(family, {}).get(literals, {})
            if not expected_values and not unbound:
                observation = {"reason": "explicit_known_field_absent_from_family", "literal_fragments": literals, "values": values, "span": (start, end)}
                observations.append(observation)
                for item_id in scoped:
                    conflicts[item_id].append(observation)
            elif expected_values:
                # A recognized short action may omit a shared explanation. Its
                # values still cannot contradict every reference occurrence.
                for item_id in bound:
                    allowed = expected_values.get(item_id, set())
                    if values not in allowed:
                        conflicts[item_id].append({"reason": "explicit_complete_field_value_conflict", "literal_fragments": literals,
                                                   "values": values, "allowed_values": tuple(sorted(allowed)), "span": (start, end)})
        for literals, entry in fields.items():
            if entry["duplicate_ids"]:
                continue
            found, possible = [], []
            for match in entry["pattern"].finditer(text):
                # A complete contextual field owns its numeric subfields: an
                # immediate gain and a delayed gain must not be merged.
                if any(start <= match.start() and end >= match.end()
                       and (start, end) != (match.start(), match.end())
                       for _, start, end in family_hits):
                    continue
                atom_ids = {span["atom_index"] for span in spans if span["end"] > match.start() and span["start"] < match.end()}
                observation = {"values": tuple(match.groups()), "span": (match.start(), match.end()), "atom_indices": tuple(sorted(atom_ids))}
                if token_span_valid(text, match.start(), match.end(), spans) and not (atom_ids & uncertain_atoms):
                    found.append(observation)
                elif match.groups() and not any(start <= match.start() and end >= match.end() for _, start, end, _ in complete_hits):
                    # Keep possible atom-internal contradictions protected. They
                    # cannot supply positive evidence or be repaired by stripping.
                    possible.append(observation)
            if possible:
                protected.append({"literal_fragments": literals, "observations": possible})
                for item_id in bound:
                    expected = entry["expected"].get(item_id)
                    if any(hit["values"] != expected for hit in possible):
                        conflicts[item_id].append({"reason": "protected_field_boundary_conflict", "literal_fragments": literals, "observations": possible})
            if not found:
                continue
            observed = {hit["values"] for hit in found}
            discriminative = len({entry["expected"].get(item_id) for item_id in bound}) > 1
            observations.append({"literal_fragments": literals, "observations": found, "discriminative": discriminative})
            for item_id in bound:
                expected = entry["expected"].get(item_id)
                if len(observed) > 1:
                    conflicts[item_id].append({"reason": "multiple_values_same_field", "literal_fragments": literals, "observed": tuple(sorted(observed))})
                elif expected is None or observed != {expected}:
                    conflicts[item_id].append({"reason": "explicit_reference_field_conflict", "literal_fragments": literals, "expected": expected, "observed": tuple(observed)})
                elif discriminative:
                    support[item_id].append("reference_difference")
        if not uncertain_atoms:
            for item_id in bound:
                if text and self._bodies[item_id] == text:
                    support[item_id].append("complete_reference_positive")
        # Existing confirmed title authority must not gain a body-render wait.
        title_positive = set()
        for item_id in scoped:
            if title.endswith("+"):
                if self._upgraded(item_id):
                    title_positive.add(item_id)
                    support[item_id].append("explicit_upgraded_title")
                else:
                    conflicts[item_id].append({"reason": "explicit_plus_title_conflict"})
            elif len(scoped) == 1:
                title_positive.add(item_id)
                support[item_id].append("unique_title_and_plan")
        remaining = tuple(item_id for item_id in scoped if not conflicts[item_id])
        supported = tuple(item_id for item_id in remaining if support[item_id])
        diagnostics = {"title_family_ids": family, "plan_scoped_ids": scoped, "unbound_reference_ids": tuple(sorted(unbound)), "positive_support": support, "conflicts": conflicts, "observed_fields": observations, "protected_fields": protected}
        if vowel_edits:
            diagnostics["long_vowel_recovery"] = {"edits": vowel_edits, "numeric_value_edits": 0, "additional_ocr_calls": 0}
        if layout_diagnostics:
            diagnostics["layout"] = dict(layout_diagnostics)
        if not remaining:
            return PItemDetailTextResult("unknown", reason="explicit_or_protected_field_conflict", diagnostics=diagnostics)
        unresolved_siblings = unbound & set(remaining)
        if len(supported) == 1 and (not unresolved_siblings or supported[0] in title_positive):
            return PItemDetailTextResult("unique", supported, "title_or_reference_positive", diagnostics)
        return PItemDetailTextResult("ambiguous", supported if not unresolved_siblings and supported else remaining, "reference_unavailable" if unresolved_siblings else "insufficient_difference_evidence", diagnostics)

    def _recover_wrapped_body(
        self, atoms: list[dict], title_index: int, sources: PItemSourceTextFrames,
        panel_box: tuple[int, int, int, int], *, plan: str | None,
        eligible_ids: frozenset[int] | None,
    ) -> PItemDetailTextResult | None:
        """Exact complete-reference proof after bounded nonnumeric OCR loss.

        All family members must be bound. We may remove aligned, indented
        paragraph bullets and restore one alphabetic character at a proved
        wrap gap; no observed number, numeric sign or percent is rewritten.
        Partial-field matches and visual candidates cannot authorize recovery.
        """
        title = str(atoms[title_index]["text"])
        scoped = self.family_ids_for_title(title, plan=plan, eligible_ids=eligible_ids)
        if len(scoped) < 2 or any(item_id not in self._bodies for item_id in scoped):
            return None
        layout = body_layout(atoms, title_index, sources, panel_box, allow_wrapped_gap=True)
        if (layout["status"] != "resolved" or any(
                entry["reason"] in ("after_body_geometric_gap", "outside_connected_body_column")
                for entry in layout.get("rejected", ()))):
            return None
        height = layout["title_height"]
        title_left = atoms[title_index]["box"][0]
        paragraphs_by_item = [{normalize(paragraph) for paragraph in self._references[item_id]["paragraphs"]}
                              for item_id in scoped]
        paragraphs = set.union(*paragraphs_by_item)
        shared_paragraphs = set.intersection(*paragraphs_by_item)
        replacements = {}
        bullet_indices = []
        bullet_lefts = []
        field_prefix_edits = []
        detached_scalar_bullet = False
        for row in layout["atoms"]:
            first = row[0]
            text = normalize(atoms[first]["text"])
            left = atoms[first]["box"][0]
            if not (text.startswith("+") and title_left + .5 * height <= left <= title_left + 1.5 * height):
                continue
            if text == "+" and len(row) == 2:
                field_index = row[1]
                field = _shared_scalar_after_bullet(atoms[first], atoms[field_index], paragraphs, shared_paragraphs, height)
                if field is not None:
                    replacements[first] = ""
                    bullet_indices.append(first)
                    bullet_lefts.append(left)
                    detached_scalar_bullet = True
                    if field != normalize(atoms[field_index]["text"]):
                        replacements[field_index] = field
                        field_prefix_edits.append(field_index)
                    continue
            row_text = "".join(normalize(atoms[index]["text"]) for index in row)[1:]
            if not row_text or not row_text[0].isalpha():
                continue
            if text == "+" and (len(row) < 2 or bounds(atoms[first])[2] > atoms[row[1]]["box"][0]):
                continue
            if not any(paragraph.startswith(row_text) for paragraph in paragraphs):
                continue
            replacements[first] = text[1:]
            bullet_indices.append(first)
            bullet_lefts.append(left)
        # A lone mark needs the separate, complete shared field above. Other
        # marks must still agree with the same column on independent rows.
        if bullet_indices and ((len(bullet_indices) < 2 and not detached_scalar_bullet)
                               or max(bullet_lefts) - min(bullet_lefts) > .5 * height):
            return None
        repaired_atoms = [{**atom, "text": replacements.get(index, atom["text"])}
                          for index, atom in enumerate(atoms)]
        text, spans = atom_text_view(repaired_atoms, layout)
        gaps = layout.get("wrapped_gaps", ())
        if not replacements and not gaps:
            return None
        offset = None
        if gaps:
            prior = [span for span in spans if span["atom_index"] == gaps[0]["after_atom_index"]]
            if len(prior) != 1:
                return None
            offset = prior[0]["end"]
        candidates = []
        for item_id in scoped:
            expected = self._bodies[item_id]
            if offset is None:
                if text == expected:
                    candidates.append((item_id, ""))
            elif (len(expected) == len(text) + 1 and offset < len(expected)
                    and expected[offset].isalpha() and not expected[offset].isnumeric()
                    and expected[:offset] + expected[offset + 1:] == text):
                candidates.append((item_id, expected[offset]))
        if len(candidates) != 1:
            return None
        item_id, character = candidates[0]
        repaired_text = text if offset is None else text[:offset] + character + text[offset:]
        repaired_spans = spans
        if offset is not None:
            repaired_spans = [{**span, "start": span["start"] + (span["start"] >= offset),
                               "end": span["end"] + (span["end"] > offset)} for span in spans]
            repaired_spans.append({"start": offset, "end": offset + 1, "atom_index": -1})
        # Keep the ordinary conflict rules, including an explicit '+' title.
        result = self._resolve(title, repaired_text, repaired_spans,
                               plan=plan, eligible_ids=eligible_ids)
        if result.status != "unique" or result.ids != (item_id,):
            return None
        return PItemDetailTextResult("unique", result.ids, "complete_reference_wrapped_body_recovery", {
            **result.diagnostics,
            "layout": {"panel_box": panel_box, "status": layout["status"], "reason": "complete_reference_wrapped_body",
                       "body_atom_indices": layout["atoms"], "excluded_atom_count": len(layout.get("rejected", ())), "uncertainty": ()},
            "body_recovery": {"bullet_atom_indices": tuple(bullet_indices), "field_prefix_edits": tuple(field_prefix_edits), "wrapped_gaps": gaps,
                              "inserted_character": character, "insertion_offset": offset,
                              "observed_text": text, "complete_reference_id": item_id,
                              "numeric_edits": 0, "additional_ocr_calls": 0},
        })

    def _wrapped_body_tail(
        self, atoms: list[dict], title_index: int, sources: PItemSourceTextFrames,
        panel_box: tuple[int, int, int, int], *, plan: str | None,
        eligible_ids: frozenset[int] | None,
    ) -> tuple[dict, int, set[str]] | None:
        """Prove one bounded missing wrap before considering its tail text."""
        title = str(atoms[title_index]["text"])
        scoped = self.family_ids_for_title(title, plan=plan, eligible_ids=eligible_ids)
        if len(scoped) < 2 or any(item_id not in self._bodies for item_id in scoped):
            return None
        layout = body_layout(atoms, title_index, sources, panel_box, allow_wrapped_gap=True)
        gaps = layout.get("wrapped_gaps", ())
        if (layout["status"] != "resolved" or len(gaps) != 1 or any(
                entry["reason"] in ("after_body_geometric_gap", "outside_connected_body_column")
                for entry in layout.get("rejected", ()))):
            return None
        rows = layout["atoms"]
        gap_row = next(i for i, row in enumerate(rows) if row[-1] == gaps[0]["after_atom_index"])
        prior, _ = atom_text_view(atoms, {"atoms": [rows[gap_row]]})
        paragraphs = {normalize(paragraph) for item_id in scoped
                      for paragraph in self._references[item_id]["paragraphs"]}
        if not any(paragraph[:-1] == prior and paragraph[-1:].isalpha()
                   and not paragraph[-1:].isnumeric() for paragraph in paragraphs):
            return None
        return layout, gap_row, paragraphs

    def _recover_wrapped_numeric_paragraph(
        self, atoms: list[dict], title_index: int, sources: PItemSourceTextFrames,
        panel_box: tuple[int, int, int, int], *, plan: str | None,
        eligible_ids: frozenset[int] | None,
    ) -> PItemDetailTextResult | None:
        """Read an intact numeric paragraph beyond one proved missing wrap.

        The earlier complete-body proof remains preferred. Here unrelated
        missing trigger words/punctuation cannot hide an intact cost, but the
        wrap row still has exactly one missing final source letter. No text is
        inserted and every observed numeric token must match the whole source
        in order, including otherwise unclaimed independent digits.
        """
        candidate = self._wrapped_body_tail(
            atoms, title_index, sources, panel_box, plan=plan, eligible_ids=eligible_ids,
        )
        if candidate is None:
            return None
        layout, gap_row, paragraphs = candidate
        rows, gaps = layout["atoms"], layout["wrapped_gaps"]
        title = str(atoms[title_index]["text"])
        tail = []
        for row in rows[gap_row + 1:]:
            row_text, _ = atom_text_view(atoms, {"atoms": [row]})
            literals, values, _ = source_field(row_text)
            if row_text not in paragraphs:
                return None
            if _is_complete_numeric_field(literals, values):
                tail.append(tuple(row))
        if not tail:
            return None
        text, spans = atom_text_view(atoms, layout)
        result = self._resolve(title, text, spans, plan=plan, eligible_ids=eligible_ids,
                               layout_diagnostics={"panel_box": panel_box, "status": layout["status"],
                                                   "reason": "wrapped_numeric_paragraph",
                                                   "body_atom_indices": rows,
                                                   "excluded_atom_count": len(layout.get("rejected", ())),
                                                   "uncertainty": ()})
        if (result.status != "unique" or "reference_difference" not in
                result.diagnostics["positive_support"][result.ids[0]]):
            return None
        item_id = result.ids[0]
        observed = tuple(match.group() for match in NUMBER.finditer(text))
        expected = tuple(match.group() for match in NUMBER.finditer(self._bodies[item_id]))
        if observed != expected:
            return None
        # The independent, complete tail paragraph itself must distinguish the
        # family; no earlier partial body field can authorize this recovery.
        tail_text, tail_spans = atom_text_view(atoms, {"atoms": tail})
        tail_result = self._resolve(title, tail_text, tail_spans, plan=plan, eligible_ids=eligible_ids)
        if tail_result.status != "unique" or tail_result.ids != result.ids:
            return None
        return PItemDetailTextResult("unique", result.ids, "wrapped_numeric_paragraph_recovery", {
            **result.diagnostics,
            "numeric_paragraph_recovery": {"paragraph_atom_indices": tail, "wrapped_gaps": gaps,
                                           "observed_text": text, "observed_numbers": observed,
                                           "numeric_value_edits": 0, "inserted_characters": 0,
                                           "additional_ocr_calls": 0},
        })

    def _retain_wrapped_numeric_conflicts(
        self, atoms: list[dict], title_index: int, sources: PItemSourceTextFrames,
        panel_box: tuple[int, int, int, int], result: PItemDetailTextResult, *,
        plan: str | None, eligible_ids: frozenset[int] | None,
    ) -> PItemDetailTextResult:
        """An excluded complete field can reject an assumption, never prove it."""
        candidate = self._wrapped_body_tail(
            atoms, title_index, sources, panel_box, plan=plan, eligible_ids=eligible_ids,
        )
        if candidate is None:
            return result
        layout, gap_row, _ = candidate
        conflicts = {item_id: list(entries) for item_id, entries in result.diagnostics["conflicts"].items()}
        retained = []
        for row in layout["atoms"][gap_row + 1:]:
            text, spans = atom_text_view(atoms, {"atoms": [row]})
            literals, values, _ = source_field(text)
            if not _is_complete_numeric_field(literals, values) or literals not in self._complete_fields:
                continue
            observed = self._resolve(str(atoms[title_index]["text"]), text, spans,
                                     plan=plan, eligible_ids=eligible_ids)
            added = False
            for item_id, entries in observed.diagnostics["conflicts"].items():
                for entry in entries:
                    if entry["reason"] in {
                        "explicit_complete_field_value_conflict", "explicit_reference_field_conflict",
                        "explicit_known_field_absent_from_family",
                    }:
                        conflicts[item_id].append({**entry, "evidence": "wrapped_numeric_tail",
                                                   "atom_indices": tuple(row)})
                        added = True
            if added:
                retained.append({"text": text, "atom_indices": tuple(row)})
        if not retained:
            return result
        return PItemDetailTextResult(result.status, result.ids, result.reason, {
            **result.diagnostics,
            "conflicts": conflicts,
            "wrapped_numeric_negative_evidence": {
                "observed_fields": retained, "wrapped_gaps": layout["wrapped_gaps"],
                "numeric_value_edits": 0, "additional_ocr_calls": 0,
                "positive_confirmation": False,
            },
        })

    def _recover_delay_bullet(
        self, atoms: list[dict], title_index: int, layout: dict,
        result: PItemDetailTextResult, *, plan: str | None,
        eligible_ids: frozenset[int] | None,
    ) -> PItemDetailTextResult | None:
        """Remove one indented bullet misread before an invariant turn delay.

        The delay cannot distinguish the variants: every bound family member
        must require the same unsigned count. An independent, unchanged field
        must already support one variant, whose only conflict is this extra +.
        Actual gains, negative counts, changed digits and text-only inputs keep
        their existing conflict rules.
        """
        diagnostic = result.diagnostics
        if (layout["status"] != "resolved" or diagnostic["layout"]["panel_box"] is None
                or diagnostic["unbound_reference_ids"] or any(
                    entry["reason"] in ("after_body_geometric_gap", "outside_connected_body_column")
                    for entry in layout.get("rejected", ()))):
            return None
        scoped = diagnostic["plan_scoped_ids"]
        supported = [item_id for item_id in scoped
                     if "reference_difference" in diagnostic["positive_support"][item_id]]
        if len(scoped) < 2 or len(supported) != 1:
            return None
        item_id = supported[0]
        conflicts = diagnostic["conflicts"][item_id]
        if len(conflicts) != 1 or conflicts[0]["reason"] != "explicit_reference_field_conflict":
            return None
        conflict = conflicts[0]
        literals, expected, observed = conflict["literal_fragments"], conflict["expected"], conflict["observed"]
        if (not expected or literals[0] or not literals[1].startswith("ターン後、")
                or re.fullmatch(r"[1-9][0-9]*", expected[0]) is None
                or set(observed) != {("+" + expected[0], *expected[1:])}):
            return None
        entry = self._tables[diagnostic["title_family_ids"]][literals]
        if entry["duplicate_ids"] or any(
                not entry["expected"].get(candidate)
                or entry["expected"][candidate][0] != expected[0] for candidate in scoped):
            return None
        hits = [hit for field in diagnostic["observed_fields"]
                if field["literal_fragments"] == literals for hit in field["observations"]]
        if len(hits) != 1:
            return None
        prefix = "+" + expected[0] + "ターン後、"
        title_left = atoms[title_index]["box"][0]
        height = layout["title_height"]
        _, original_spans = atom_text_view(atoms, layout)
        candidates = [row[0] for row in layout["atoms"]
                      if normalize(atoms[row[0]]["text"]).startswith(prefix)
                      and title_left + .5 * height <= atoms[row[0]]["box"][0] <= title_left + 1.5 * height
                      and any(span["atom_index"] == row[0] and span["start"] == hits[0]["span"][0]
                              for span in original_spans)]
        if len(candidates) != 1:
            return None
        atom_index = candidates[0]
        repaired_atoms = list(atoms)
        repaired_atoms[atom_index] = {**atoms[atom_index], "text": normalize(atoms[atom_index]["text"])[1:]}
        text, spans = atom_text_view(repaired_atoms, layout)
        repaired = self._resolve(str(atoms[title_index]["text"]), text, spans,
                                 plan=plan, eligible_ids=eligible_ids,
                                 layout_diagnostics=diagnostic["layout"])
        if repaired.status != "unique" or repaired.ids != (item_id,):
            return None
        return PItemDetailTextResult("unique", repaired.ids, "reference_delay_bullet_recovery", {
            **repaired.diagnostics,
            "delay_bullet_recovery": {"atom_index": atom_index, "observed_prefix": prefix,
                                      "reference_prefix": prefix[1:], "removed_decoration": "+",
                                      "independent_support": "reference_difference",
                                      "numeric_value_edits": 0, "additional_ocr_calls": 0},
        })

    def _native_detail_vetoes(
        self, atoms: list[dict], title_index: int, sources: PItemSourceTextFrames,
        layout: dict, result: PItemDetailTextResult, size: tuple[int, int],
    ) -> dict[int, list[dict]]:
        title = normalize(atoms[title_index]["text"])
        scoped = result.diagnostics.get("plan_scoped_ids", ())
        if (title.endswith("+") or not any(self._upgraded(item_id) for item_id in scoped)
                or not _valid_detail_atoms((atoms[title_index],), size)):
            return {}
        exact_ids = self._titles.get(title, ())
        ordinary_ids = [item_id for item_id in scoped if item_id in exact_ids and not self._upgraded(item_id)]
        if len(ordinary_ids) != 1 or ordinary_ids[0] not in result.ids:
            return {}
        ordinary = ordinary_ids[0]
        contradictions = []
        if set(scoped) & self._titles.get(title + "+", set()):
            suffix = _detached_title_plus(atoms, title_index, sources, size)
            if suffix is not None:
                contradictions.append({"reason": "detached_native_title_plus", "atom_indices": (suffix,)})
        rows = layout["atoms"]
        excluded = [entry["index"] for entry in layout.get("rejected", ())
                    if entry["reason"] in {"after_body_geometric_gap", "outside_connected_body_column",
                                           "outside_title_body_origin"}]
        known = self._family_known_fields.get(self._title_scopes.get(title, ()), {})
        if not rows or not known or not excluded:
            return {ordinary: contradictions} if contradictions else {}

        # Geometry admits nearby candidates; only a complete known field can
        # contradict the ordinary result. The OCR boxes and values stay intact.
        row_bounds = [union([atoms[index] for index in row]) for row in rows]
        left = min(box[0] for box in row_bounds)
        right = max(box[2] for box in row_bounds)
        title_bottom = bounds(atoms[title_index])[3]
        height = layout["title_height"]
        eligible = []
        for index in excluded:
            atom = atoms[index]
            x, y, end_x, end_y = bounds(atom)
            numeric_fragment = re.fullmatch(r"[+\-0-9.,%]+", normalize(atom["text"])) is not None
            lane_end = right + height if numeric_fragment else min(left + 4 * height, right + height)
            if (left - height <= x <= lane_end
                    and end_x >= left and y >= title_bottom - .25 * height
                    and any(y - bottom <= 3 * height and end_y >= top - height
                            for _, top, _, bottom in row_bounds)
                    and _valid_detail_atoms((atom,), size)):
                eligible.append({**atom, "index": index})
        for row in component_rows(eligible, height):
            for component in row["components"]:
                indices = [atom["index"] for atom in component["atoms"]]
                if any(bounds(atoms[b])[0] - bounds(atoms[a])[2] > .5 * height
                       for a, b in zip(indices, indices[1:])):
                    continue
                text, _ = atom_text_view(atoms, {"atoms": [indices]})
                field, boundary, operand_tail = text.partition(BOUNDARY)
                literals, values, _ = source_field(field)
                incomplete_operand = bool(boundary and re.fullmatch(r"[+\-0-9.,%]+", operand_tail))
                if (literals in known and values
                        and (incomplete_operand or values not in known[literals].get(ordinary, ()))):
                    contradictions.append({
                        "reason": "excluded_native_body_field_conflict",
                        "atom_indices": tuple(indices), "text": text, "values": values,
                        "operand_tail": operand_tail if incomplete_operand else None,
                        "literal_fragments": literals,
                    })
        return {ordinary: contradictions} if contradictions else {}

    def match(
        self, atoms: Sequence[Mapping[str, Any]], *, title_index: int,
        source_frames: Sequence[Sequence[Mapping[str, Any]]] | PItemSourceTextFrames = (),
        plan: str | None = None,
        eligible_ids: frozenset[int] | None = None,
        panel_box: tuple[int, int, int, int] | None = None,
        detail_image: Any = None,
        detail_coordinate_size: tuple[int, int] | None = None,
    ) -> PItemDetailTextResult:
        if not isinstance(title_index, int) or isinstance(title_index, bool) or not 0 <= title_index < len(atoms):
            return PItemDetailTextResult("unknown", reason="invalid_title_index")
        atom_rows = [dict(atom) for atom in atoms]
        sources = source_frames if isinstance(source_frames, PItemSourceTextFrames) else PItemSourceTextFrames([[dict(atom) for atom in frame] for frame in source_frames])
        layout = body_layout(atom_rows, title_index, sources, panel_box)
        if layout["reason"] == "panel_atom_crosses_boundary":
            return PItemDetailTextResult("unknown", reason="panel_atom_crosses_boundary")
        text, spans = atom_text_view(atom_rows, layout)
        uncertain = {index for entry in layout.get("uncertainty", ()) for index in entry.get("atom_indices", ())}
        result = self._resolve(str(atom_rows[title_index]["text"]), text, spans, plan=plan, eligible_ids=eligible_ids, uncertain_atoms=uncertain,
                               layout_diagnostics={"panel_box": panel_box, "status": layout["status"], "reason": layout["reason"], "body_atom_indices": layout["atoms"], "excluded_atom_count": len(layout.get("rejected", ())), "uncertainty": layout.get("uncertainty", ())})
        vetoes = self._native_detail_vetoes(
            atom_rows, title_index, sources, layout, result, detail_coordinate_size or (720, 1280),
        )
        result = self._recover_native_body(
            atom_rows, title_index, sources, layout, result, plan=plan, eligible_ids=eligible_ids,
            panel_box=panel_box, detail_image=detail_image, detail_coordinate_size=detail_coordinate_size,
        )
        return retain_native_detail_vetoes(result, vetoes)

    def _recover_native_body(
        self, atom_rows: list[dict], title_index: int, sources: PItemSourceTextFrames, layout: dict,
        result: PItemDetailTextResult, *, plan: str | None, eligible_ids: frozenset[int] | None,
        panel_box: tuple[int, int, int, int] | None, detail_image: Any,
        detail_coordinate_size: tuple[int, int] | None,
    ) -> PItemDetailTextResult:
        fragments = _adjacent_operand_fragments(atom_rows, layout, result)
        if fragments:
            return PItemDetailTextResult("unknown", reason=result.reason if result.status == "unknown" else "numeric_operand_fragment_unresolved", diagnostics={
                **result.diagnostics, "numeric_operand_fragments": fragments,
            })
        if result.reason == "explicit_or_protected_field_conflict" and panel_box is not None:
            recovered = self._recover_delay_bullet(atom_rows, title_index, layout, result,
                                                    plan=plan, eligible_ids=eligible_ids)
            if recovered is not None:
                return recovered
        if result.reason == "insufficient_difference_evidence" and panel_box is not None:
            recovered = self._recover_wrapped_body(atom_rows, title_index, sources, panel_box, plan=plan, eligible_ids=eligible_ids)
            if recovered is not None:
                return recovered
            recovered = self._recover_wrapped_numeric_paragraph(
                atom_rows, title_index, sources, panel_box, plan=plan, eligible_ids=eligible_ids,
            )
            if recovered is not None:
                return recovered
        contextual_candidate = bool(
            result.status == "unique" and not normalize(str(atom_rows[title_index]["text"])).endswith("+")
            and "reference_difference" in result.diagnostics.get("positive_support", {}).get(result.ids[0], ())
        )
        if panel_box is not None and (result.status != "unique" or contextual_candidate):
            # A short base-variant field can itself be an incomplete prefix of
            # the upgraded paragraph when a separate effect icon interrupts it.
            from ._p_item_detail_decorations import (
                recover_decorated_body,
                recover_owned_effect_icons,
                recover_contextual_effect_icon,
            )

            scoped = self.family_ids_for_title(str(atom_rows[title_index]["text"]), plan=plan, eligible_ids=eligible_ids)
            if len(scoped) >= 2 and all(item_id in self._bodies for item_id in scoped):
                paragraphs = {item_id: self._references[item_id]["paragraphs"] for item_id in scoped}
                if result.status != "unique":
                    proof = recover_decorated_body(
                        atom_rows, title_index, layout, paragraphs,
                        tuple(literals[0] for literals in self._complete_fields),
                    )
                    reason = "complete_reference_decoration_recovery"
                    if proof is None and result.reason == "insufficient_difference_evidence":
                        proof = recover_owned_effect_icons(
                            atom_rows, title_index, layout, paragraphs,
                            image=detail_image, coordinate_size=detail_coordinate_size,
                        )
                        reason = "same_frame_paragraph_mark_recovery"
                    if proof is not None:
                        recovered = self._resolve(str(atom_rows[title_index]["text"]), proof["text"], proof["spans"],
                                                  plan=plan, eligible_ids=eligible_ids,
                                                  layout_diagnostics=result.diagnostics.get("layout"))
                        if recovered.status == "unique" and recovered.ids == (proof["item_id"],):
                            return PItemDetailTextResult("unique", recovered.ids, reason, {
                                **recovered.diagnostics,
                                "decoration_recovery": {"edits": proof["edits"], "complete_reference_id": proof["item_id"],
                                                        "numeric_value_edits": 0, "additional_ocr_calls": 0},
                            })
                # A separately recognized effect icon can occur without any
                # damaged delay bullets. Require the complete distinguishing
                # paragraph, then keep every other original-body conflict.
                proof = recover_contextual_effect_icon(
                    atom_rows, title_index, layout, paragraphs,
                    tuple(literals[0] for literals in self._complete_fields),
                )
                if proof is not None:
                    recovered = self._resolve(str(atom_rows[title_index]["text"]), proof["text"], proof["spans"],
                                              plan=plan, eligible_ids=eligible_ids,
                                              layout_diagnostics=result.diagnostics.get("layout"))
                    if (recovered.status == "unique" and recovered.ids == (proof["item_id"],)
                            and "reference_difference" in recovered.diagnostics["positive_support"][proof["item_id"]]):
                        return PItemDetailTextResult("unique", recovered.ids, "contextual_effect_icon_recovery", {
                            **recovered.diagnostics,
                            "decoration_recovery": {"edits": proof["edits"], "complete_reference_id": proof["item_id"],
                                                    "reference_paragraph": proof["paragraph"],
                                                    "paragraph_atom_indices": proof["atom_indices"],
                                                    "numeric_value_edits": 0, "additional_ocr_calls": 0},
                        })
                    if contextual_candidate:
                        # Once the longer complete field is proved, an older
                        # short-field conclusion cannot hide a newly observed
                        # contradiction elsewhere in the original body.
                        return recovered
        if (result.status == "ambiguous" and result.reason == "insufficient_difference_evidence"
                and panel_box is not None):
            return self._retain_wrapped_numeric_conflicts(
                atom_rows, title_index, sources, panel_box, result,
                plan=plan, eligible_ids=eligible_ids,
            )
        return result

    def match_text(self, title_text: str, detail_text: str, *, plan: str | None = None, eligible_ids: frozenset[int] | None = None) -> PItemDetailTextResult:
        """Compatibility core for old text-only mocks; production passes atoms."""
        lines = detail_text.splitlines()
        title_matches = [i for i, line in enumerate(lines) if normalize(line) == normalize(title_text)]
        if title_matches:
            lines = lines[title_matches[0] + 1:]
        atoms = [{"text": line} for line in lines]
        text, spans = atom_text_view(atoms, {"atoms": [[i] for i in range(len(atoms))]})
        return self._resolve(title_text, text, spans, plan=plan, eligible_ids=eligible_ids,
                             layout_diagnostics={"status": "text_only_compatibility", "production_geometry_claim": False})
