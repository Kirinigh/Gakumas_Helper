"""Fixed P-item text evidence, independent of OCR and image-ranking candidates."""
from __future__ import annotations

import re
import json
import unicodedata
from typing import Any, Literal
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence

BOUNDARY = "\u2063"
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)?%?")
OBSERVED_NUMBER = r"([+-]?[0-9]+(?:[.,][0-9]+)?%?)(?![0-9.,%])"


@dataclass(frozen=True, slots=True)
class PItemDetailTextResult:
    status: Literal["unique", "ambiguous", "unknown"]
    ids: tuple[int, ...] = ()
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))

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

def body_layout(atoms: list[dict], title_index: int, sources: PItemSourceTextFrames) -> dict:
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
        x, y, w, h = atom["box"]
        if y + h / 2 < title_bottom or x < tx - 0.5 * height:
            rejected.append({"index": index, "reason": "outside_title_body_origin"})
            continue
        if len(key) == 1 and unicodedata.category(key) == "So":
            decorations.append({"index": index, "reason": "complete_symbol_decoration", "text": atom["text"], "box": atom["box"]})
            continue
        eligible.append(atom)
    selected, uncertainty = [], []
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
            rejected.extend({"index": a["index"], "reason": "after_body_geometric_gap"} for later in component_rows([a for a in eligible if bounds(a)[1] >= top], height) for component in later["components"] for a in component["atoms"])
            break
        if len([component for component in candidates if component["strong"]]) > 1:
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
    return {"status": "resolved" if started and not uncertainty else "unknown", "reason": "fixed_connected_body" if started and not uncertainty else "layout_uncertain" if uncertainty else "body_not_rendered", "body": body, "atoms": [[atom["index"] for atom in row] for row in selected], "rejected": rejected, "decorations": decorations, "uncertainty": uncertainty, "title_height": height}

def atom_text_view(atoms: list[dict], layout: dict) -> tuple[str, list[dict]]:
    """Reuse frozen layout, retaining atom spans and independent numeric edges."""
    parts, spans = [], []
    offset = 0
    previous = ""
    for row in layout["atoms"]:
        for index in row:
            text = normalize(atoms[index]["text"])
            if not text:
                continue
            if previous and previous[-1].isdigit() and text[0].isdigit():
                parts.append(BOUNDARY)
                offset += 1
            parts.append(text)
            spans.append({"start": offset, "end": offset + len(text), "atom_index": index})
            offset += len(text)
            previous = text
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
    pattern = OBSERVED_NUMBER.join(re.escape(part) for part in literals)
    return tuple(literals), tuple(values), pattern


def _default_reference_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    installed = root / "data/p_item_detail_text.json"
    return installed if installed.is_file() else root / "assets/data/p_item_detail_text.json"


def _binding_matches(active: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    fields = ("mode", "sourceType", "pIdolId", "upgraded", "plan", "rarity")
    return bool(
        all(active.get(key) == reference.get(key) for key in fields)
        and isinstance(active.get("effects"), str)
        and re.sub(r"\s+", "", active["effects"])
        == re.sub(r"\s+", "", str(reference.get("effects", "")))
    )


def _is_complete_numeric_field(literals: tuple[str, ...], values: tuple[str, ...]) -> bool:
    # Fixed source action fields, not partial trigger clauses or a keyword list.
    return bool(values and literals[0] and sum(c.isalpha() for c in literals[0]) >= 2
                and all(not part or all(c.isalpha() or c == "・" for c in part) for part in literals))


class PItemDetailTextIndex:
    """One catalog-owned index; unrelated catalog additions do not invalidate it."""

    def __init__(
        self,
        p_items: Sequence[Mapping[str, Any]],
        *,
        reference_data: Mapping[str, Any] | None = None,
        reference_path: str | Path | None = None,
    ) -> None:
        if reference_data is None:
            path = Path(reference_path) if reference_path is not None else _default_reference_path()
            reference_data = json.loads(path.read_text(encoding="utf-8"))
        if reference_data.get("schema_version") != 1 or not isinstance(reference_data.get("rows"), list):
            raise ValueError("P-item text reference schema is invalid")
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

    def family_ids_for_title(self, title: str, *, plan: str | None = None) -> tuple[int, ...]:
        return tuple(item_id for item_id in self._title_scopes.get(normalize(title), ())
                     if plan in (None, "", "free") or self._active[item_id].get("plan") in (None, "", "free", plan))

    def _upgraded(self, item_id: int) -> bool:
        row = self._active[item_id]
        value = row.get("upgraded")
        return value if isinstance(value, bool) else normalize(str(row.get("name", ""))).endswith("+")

    def _resolve(
        self, title: str, text: str, spans: list[dict], *, plan: str | None,
        uncertain_atoms: set[int] | None = None,
        layout_diagnostics: Mapping[str, Any] | None = None,
    ) -> PItemDetailTextResult:
        title = normalize(title)
        family = self._title_scopes.get(title, ())
        scoped = self.family_ids_for_title(title, plan=plan)
        if not scoped:
            return PItemDetailTextResult("unknown", reason="title_not_in_catalog", diagnostics={"title_family_ids": family})
        uncertain_atoms = uncertain_atoms or set()
        available = set(self.available_ids)
        bound = set(scoped) & available
        unbound = set(scoped) - available
        fields = self._tables.get(family, {})
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
        if layout_diagnostics:
            diagnostics["layout"] = dict(layout_diagnostics)
        if not remaining:
            return PItemDetailTextResult("unknown", reason="explicit_or_protected_field_conflict", diagnostics=diagnostics)
        unresolved_siblings = unbound & set(remaining)
        if len(supported) == 1 and (not unresolved_siblings or supported[0] in title_positive):
            return PItemDetailTextResult("unique", supported, "title_or_reference_positive", diagnostics)
        return PItemDetailTextResult("ambiguous", supported if not unresolved_siblings and supported else remaining, "reference_unavailable" if unresolved_siblings else "insufficient_difference_evidence", diagnostics)

    def match(
        self, atoms: Sequence[Mapping[str, Any]], *, title_index: int,
        source_frames: Sequence[Sequence[Mapping[str, Any]]] | PItemSourceTextFrames = (),
        plan: str | None = None,
    ) -> PItemDetailTextResult:
        if not isinstance(title_index, int) or isinstance(title_index, bool) or not 0 <= title_index < len(atoms):
            return PItemDetailTextResult("unknown", reason="invalid_title_index")
        atom_rows = [dict(atom) for atom in atoms]
        sources = source_frames if isinstance(source_frames, PItemSourceTextFrames) else PItemSourceTextFrames([[dict(atom) for atom in frame] for frame in source_frames])
        layout = body_layout(atom_rows, title_index, sources)
        text, spans = atom_text_view(atom_rows, layout)
        uncertain = {index for entry in layout.get("uncertainty", ()) for index in entry.get("atom_indices", ())}
        return self._resolve(str(atom_rows[title_index]["text"]), text, spans, plan=plan, uncertain_atoms=uncertain,
                             layout_diagnostics={"status": layout["status"], "reason": layout["reason"], "body_atom_indices": layout["atoms"], "excluded_atom_count": len(layout.get("rejected", ())), "uncertainty": layout.get("uncertainty", ())})

    def match_text(self, title_text: str, detail_text: str, *, plan: str | None = None) -> PItemDetailTextResult:
        """Compatibility core for old text-only mocks; production passes atoms."""
        lines = detail_text.splitlines()
        title_matches = [i for i, line in enumerate(lines) if normalize(line) == normalize(title_text)]
        if title_matches:
            lines = lines[title_matches[0] + 1:]
        atoms = [{"text": line} for line in lines]
        text, spans = atom_text_view(atoms, {"atoms": [[i] for i in range(len(atoms))]})
        return self._resolve(title_text, text, spans, plan=plan,
                             layout_diagnostics={"status": "text_only_compatibility", "production_geometry_claim": False})
