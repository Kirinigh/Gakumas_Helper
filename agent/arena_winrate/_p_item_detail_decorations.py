"""Failure-only, full-reference proof for repeated delay bullets and icons."""
from __future__ import annotations

import re

from ._p_item_detail_text import BOUNDARY, bounds, normalize, atom_text_view, token_span_valid

DELAY = re.compile(r"[1-9][0-9]*ターン後、")


def _effect_icon_indices(atoms, rows, text_by_atom, height, effect_heads, *, bullet_left=None):
    icons = set()
    for row in rows:
        for position, i in enumerate(row[:-1]):
            text = text_by_atom[i]
            if len(text) != 1 or not (text.isalpha() and not text.isnumeric() or text == "+"):
                continue
            x, y, w, h = atoms[i]["box"]
            following = atoms[row[position + 1]]
            fx, fy, fw, fh = following["box"]
            # A numeric sign needs independent paragraph-bullet proof. Icon
            # recovery alone never removes signs, numbers or embedded text.
            if text == "+" and (bullet_left is None or position != 0 or abs(x - bullet_left) > .5 * height):
                continue
            if (0 < w <= height and 0 < h <= height and fw > 2 * height
                    and -.5 * height <= fx - (x + w) <= 2 * height
                    and abs(y + h / 2 - fy - fh / 2) <= .45 * min(h, fh)
                    and (position == 0 or x >= bounds(atoms[row[position - 1]])[2])
                    and any(head and text_by_atom[row[position + 1]].startswith(head) for head in effect_heads)):
                icons.add(i)
    return icons


def recover_contextual_effect_icon(atoms, title_index, layout, paragraphs, effect_heads):
    """Prove one separate icon using a complete variant-specific paragraph.

    This does not depend on a second OCR error in a delay bullet. The original
    unsigned delay, all effects and every numeric value must already be present.
    The whole original body is then resolved again, preserving other conflicts.
    """
    if (layout["status"] != "resolved" or not paragraphs or any(
        entry["reason"] in ("after_body_geometric_gap", "outside_connected_body_column")
        for entry in layout.get("rejected", ())
    )):
        return None
    rows = layout["atoms"]
    if sum(map(len, rows)) > 96:
        return None
    text_by_atom = {i: normalize(atoms[i]["text"]).translate(str.maketrans({"值": "値", "增": "増", "·": "・"}))
                    for row in rows for i in row}
    icons = _effect_icon_indices(atoms, rows, text_by_atom, layout["title_height"], effect_heads)
    if not icons or len(icons) > 8:
        return None
    normalized = {item_id: tuple(normalize(text) for text in texts) for item_id, texts in paragraphs.items()}
    shared_delays = set.intersection(*(
        {match.group() for text in texts if (match := DELAY.match(text))}
        for texts in normalized.values()
    ))
    ordered = [i for row in rows for i in row]
    positions = {i: offset for offset, i in enumerate(ordered)}
    accepted = []
    for icon_index in sorted(icons):
        repaired_atoms = list(atoms)
        repaired_atoms[icon_index] = {**atoms[icon_index], "text": ""}
        text, spans = atom_text_view(repaired_atoms, layout)
        for item_id, texts in normalized.items():
            for paragraph in texts:
                delay = DELAY.match(paragraph)
                if (delay is None or delay.group() not in shared_delays
                        or sum(paragraph in other for other in normalized.values()) != 1):
                    continue
                for hit in re.finditer(re.escape(paragraph), text):
                    if not token_span_valid(text, hit.start(), hit.end(), spans):
                        continue
                    covered = [span["atom_index"] for span in spans
                               if span["end"] > hit.start() and span["start"] < hit.end()]
                    if not covered or not positions[covered[0]] < positions[icon_index] < positions[covered[-1]]:
                        continue
                    # Context starts at the complete, indented delay atom;
                    # a word-internal or signed suffix cannot supply it.
                    first = atoms[covered[0]]
                    left = atoms[title_index]["box"][0]
                    height = layout["title_height"]
                    if (text_by_atom[covered[0]] != delay.group()
                            or not left + .5 * height <= first["box"][0] <= left + 1.5 * height):
                        continue
                    accepted.append({"item_id": item_id, "text": text, "spans": spans,
                                     "paragraph": paragraph, "atom_indices": covered,
                                     "edits": [{"atom_index": icon_index, "kind": "effect_icon",
                                                "observed": text_by_atom[icon_index]}]})
    return accepted[0] if len(accepted) == 1 else None


def recover_decorated_body(atoms, title_index, layout, paragraphs, effect_heads):
    """Return one complete body, never a fuzzy text or numeric-field match.

    Two aligned delay rows shared by every bound variant establish the bullet
    column. Small separate nonnumeric glyphs may occupy effect-icon positions;
    their removal still has to explain every character of exactly one complete
    reference. Nothing is learned from the image ranking or upgraded title.
    """
    if layout["status"] != "resolved" or any(
        entry["reason"] in ("after_body_geometric_gap", "outside_connected_body_column")
        for entry in layout.get("rejected", ())
    ):
        return None
    rows = layout["atoms"]
    if sum(map(len, rows)) > 96 or not paragraphs:
        return None
    height = layout["title_height"]
    left = atoms[title_index]["box"][0]
    shared_delays = set.intersection(*(
        {match.group() for text in texts if (match := DELAY.match(normalize(text)))}
        for texts in paragraphs.values()
    ))
    text_by_atom = {i: normalize(atoms[i]["text"]).translate(str.maketrans({"值": "値", "增": "増", "·": "・"}))
                    for row in rows for i in row}
    delays = {row[0] for row in rows
              if left + .5 * height <= atoms[row[0]]["box"][0] <= left + 1.5 * height
              and any(text_by_atom[row[0]].startswith("+" + delay) for delay in shared_delays)}
    if len(delays) < 2 or max(atoms[i]["box"][0] for i in delays) - min(atoms[i]["box"][0] for i in delays) > .5 * height:
        return None
    bullet_left = sum(atoms[i]["box"][0] for i in delays) / len(delays)

    def starts_effect(text):
        return any(head and text.startswith(head) for head in effect_heads)

    icons = _effect_icon_indices(atoms, rows, text_by_atom, height, effect_heads, bullet_left=bullet_left)

    accepted = []
    for item_id, texts in paragraphs.items():
        normalized = [normalize(text) for text in texts]
        expected = ""
        paragraph_starts = set()
        for text in normalized:
            paragraph_starts.add(len(expected))
            # Keep independent numbers on adjacent paragraphs independent,
            # just as the ordinary atom view does (e.g. +1 then 6 turns).
            if expected and text and text[0].isdigit() and (expected[-1].isdigit() or expected.endswith("%")):
                expected += BOUNDARY
            expected += text
        # offset, spans, edits, last nonempty row, last nonempty atom text
        states = [(0, [], [], -1, "")]
        for row_number, row in enumerate(rows):
            for i in row:
                original = text_by_atom[i]
                options = [(original, None)]
                if i in delays:
                    options.append((original[1:], "delay_bullet"))
                if i in icons:
                    options.append(("", "paragraph_bullet" if original == "+" else "effect_icon"))
                following_states = []
                for offset, spans, edits, last_row, previous in states:
                    for text, change in options:
                        if change in ("delay_bullet", "paragraph_bullet") and offset not in paragraph_starts:
                            continue
                        changes = edits + ([{"atom_index": i, "kind": change, "observed": original}] if change else [])
                        if len(changes) > 8:
                            continue
                        prefix = BOUNDARY if (text and text[0].isdigit() and previous
                                              and (previous[-1].isdigit() or previous.endswith("%"))) else ""
                        prefixes = [prefix]
                        # Only the reference's effect separator may be missing
                        # at a physical wrap. No digit, sign or word is supplied.
                        if (text and last_row >= 0 and last_row != row_number and previous
                                and previous[-1].isdigit() and starts_effect(text)):
                            prefixes.append("・")
                        for separator in prefixes:
                            part = separator + text
                            if not expected.startswith(part, offset):
                                continue
                            extra = [{"atom_index": i, "kind": "wrapped_separator", "observed": ""}] if separator == "・" else []
                            new_spans = spans + ([{"start": offset + (separator == BOUNDARY),
                                                   "end": offset + len(part), "atom_index": i}] if part else [])
                            following_states.append((offset + len(part), new_spans, changes + extra,
                                                     row_number if text else last_row, text or previous))
                states = following_states
                if len(states) > 16:
                    states = []
                if not states:
                    break
            if not states:
                break
        for offset, spans, edits, _, _ in states:
            if (offset == len(expected) and len(edits) <= 8
                    and sum(edit["kind"] == "delay_bullet" for edit in edits) >= 2):
                accepted.append({"item_id": item_id, "text": expected, "spans": spans, "edits": edits})
    return accepted[0] if len(accepted) == 1 else None
