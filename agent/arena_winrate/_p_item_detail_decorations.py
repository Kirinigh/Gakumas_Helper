"""Failure-only, full-reference proof for repeated delay bullets and icons."""
from __future__ import annotations

import re

import cv2
import numpy as np

from ._p_item_detail_text import BOUNDARY, bounds, normalize, source_field, atom_text_view, token_span_valid

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


def _separate_paragraph_mark(mark, owner, height):
    mx, my, mw, mh = mark["box"]
    ox, oy, _, oh = owner["box"]
    return (.2 * height <= ox - (mx + mw) <= 2 * height
            and 0 < mw <= height and 0 < mh <= height
            and abs(my + mh / 2 - oy - oh / 2) <= .45 * min(mh, oh))


def _same_frame_mark_match(image, observed, recognized, coordinate_size):
    """Compare two observed patches, without classifying a glyph by shape."""
    sx, sy = image.shape[1] / coordinate_size[0], image.shape[0] / coordinate_size[1]
    x, y, right, bottom = bounds(observed)
    left, top, right, bottom = round(x * sx), round(y * sy), round(right * sx), round(bottom * sy)
    width, height = right - left, bottom - top
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]
            and min(width, height) >= 4):
        return None
    sample = image[top:bottom, left:right]
    if not np.any(np.std(sample, axis=(0, 1))):
        return None
    bx, by, bw, bh = recognized["box"]
    margin = max(1, round(.15 * min(width, height)))
    rx = round((bx + bw / 2) * sx - width / 2) - margin
    ry = round((by + bh / 2) * sy - height / 2) - margin
    rw, rh = width + 2 * margin, height + 2 * margin
    if not (0 <= rx and rx + rw <= image.shape[1] and 0 <= ry and ry + rh <= image.shape[0]):
        return None
    search = image[ry:ry + rh, rx:rx + rw]
    _, similarity, _, (dx, dy) = cv2.minMaxLoc(cv2.matchTemplate(search, sample, cv2.TM_CCOEFF_NORMED))
    patch = search[dy:dy + height, dx:dx + width]
    mean_error = float(np.abs(sample.astype(np.int16) - patch).mean())
    return (similarity, mean_error) if similarity >= .985 and mean_error <= 4 else None


def recover_owned_effect_icons(atoms, title_index, layout, paragraphs, *, image, coordinate_size):
    """Recover a complete body only after confirming each detached paragraph mark.

    Require the native OCR to explicitly recognize an ornamental symbol next
    to another complete catalog-owned paragraph on this same frame. Pixel
    agreement corroborates this one existing symbol at repeated row positions.
    Nothing in this fallback classifies a digit, a word-internal glyph, a
    colored effect icon with no witnessed instance, or a numeric operator.
    """
    if (layout["status"] != "resolved" or len(paragraphs) < 2
            or any(entry["reason"] in ("after_body_geometric_gap", "outside_connected_body_column")
                   for entry in layout.get("rejected", ()))
            or not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or coordinate_size is None or len(coordinate_size) != 2 or min(coordinate_size) <= 0):
        return None
    rows = layout["atoms"]
    decorations = layout.get("decorations", ())
    if not decorations or sum(map(len, rows)) > 96:
        return None
    references = {item_id: tuple(normalize(part) for part in body) for item_id, body in paragraphs.items()}
    shared = set.intersection(*(set(body) for body in references.values()))
    fields = {part for part in shared if part and not (part[0].isnumeric() or part[0] in "+-−") and source_field(part)[1]}
    height = layout["title_height"]
    title_bottom = bounds(atoms[title_index])[3]
    owners = {row[-1] for row in rows
              if atoms[row[-1]]["box"][1] >= title_bottom and normalize(atoms[row[-1]]["text"]) in fields}
    witnesses = [
        (entry["index"], owner) for entry in decorations if entry["reason"] == "complete_symbol_decoration"
        for owner in owners if _separate_paragraph_mark(atoms[entry["index"]], atoms[owner], height)
    ]
    edits = []
    for row in rows:
        if len(row) != 2 or row[-1] not in owners:
            continue
        marker, owner = row
        observed = normalize(atoms[marker]["text"])
        if observed != "+" or not _separate_paragraph_mark(atoms[marker], atoms[owner], height):
            continue
        for witness, witness_owner in witnesses:
            if (witness_owner == owner
                    or abs(atoms[marker]["box"][0] - atoms[witness]["box"][0]) > .2 * height):
                continue
            match = _same_frame_mark_match(image, atoms[marker], atoms[witness], coordinate_size)
            if match is not None:
                similarity, mean_error = match
                edits.append({"kind": "same_frame_paragraph_mark", "atom_index": marker, "owner_atom_index": owner,
                              "reference_atom_index": witness, "observed": observed,
                              "similarity": float(similarity), "mean_pixel_error": mean_error})
                break
    if not edits:
        return None
    omitted = {edit["atom_index"] for edit in edits}
    text, spans = atom_text_view(atoms, {"atoms": [[i for i in row if i not in omitted] for row in rows]})
    matched = [item_id for item_id, body in references.items() if text == "".join(body)]
    if len(matched) != 1:
        return None
    return {"item_id": matched[0], "text": text, "spans": spans, "edits": edits}


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


def wrapped_effect_separator(atoms, layout, paragraphs):
    """Prove a missing separator from two exact rows and a witnessed wrap."""
    if layout["status"] != "resolved" or len(layout["atoms"]) < 4:
        return None
    rows = layout["atoms"]
    height = layout["title_height"]
    row_texts = [atom_text_view(atoms, {"atoms": [row]})[0] for row in rows]
    references = {item_id: tuple(map(normalize, texts)) for item_id, texts in paragraphs.items()}
    shared = set.intersection(*(set(texts) for texts in references.values()))
    witnessed = {}
    candidates = []
    for index, (left, right) in enumerate(zip(row_texts, row_texts[1:])):
        start = atoms[rows[index][0]]
        tail_index = rows[index][-1]
        end = atoms[tail_index]
        continuation = atoms[rows[index + 1][0]]
        gap = continuation["box"][1] - bounds(end)[3]
        if not (left and right and -.25 * height <= gap <= .75 * height):
            continue
        if left.endswith("・") and left + right in shared:
            witnessed[right] = (start, end, continuation)
            continue
        if not left[-1].isdigit() or right not in witnessed:
            continue
        previous_start, previous_end, previous_continuation = witnessed[right]
        if (abs(start["box"][0] - previous_start["box"][0]) > .5 * height
                or abs(bounds(end)[2] - bounds(previous_end)[2]) > height
                or abs(continuation["box"][0] - previous_continuation["box"][0]) > .5 * height):
            continue
        compound = left + "・" + right
        owners = [item_id for item_id, texts in references.items() if compound in texts]
        if len(owners) != 1 or any((left, right) in zip(texts, texts[1:]) for texts in references.values()):
            continue
        candidates.append((owners[0], tail_index, compound))
    return candidates[0] if len(candidates) == 1 else None


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
