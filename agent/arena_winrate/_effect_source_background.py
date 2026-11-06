"""Prove removable old text after a failed, title-bound effect observation.

No OCR, pixels or session state are read here. The caller must validate the
transaction and distinct source image objects before supplying their atoms.
"""

from __future__ import annotations

import math
import unicodedata
from numbers import Real
from collections import Counter
from collections.abc import Mapping, Sequence

from ._p_item_detail_text import PItemSourceTextFrames, normalize

_OPERATORS = frozenset("+-*/=^%٪‰‱×÷−·➕➖➗✕✖")


def _validated_box(value):
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 4):
        return None
    try:
        if any(isinstance(number, bool) or not isinstance(number, Real)
               or not math.isfinite(number) for number in value):
            return None
        x, y, width, height = value
        if (min(x, y) < 0 or min(width, height) <= 0
                or not math.isfinite(x + width) or not math.isfinite(y + height)):
            return None
    except (OverflowError, TypeError, ValueError):
        return None
    return tuple(value)


def _validated_atoms(items):
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return None
    result = []
    for item in items:
        text = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
        box = item.get("box") if isinstance(item, Mapping) else getattr(item, "box", None)
        box = _validated_box(box)
        if not isinstance(text, str) or box is None:
            return None
        result.append({"text": text, "box": box})
    return result


def _protected_text(text):
    return any(
        char.isnumeric() or char in _OPERATORS
        or unicodedata.category(char) in {"Sm", "Pd"}
        for char in text + unicodedata.normalize("NFKC", text)
    )


def certified_source_background_lines(
    native_items, *, effect_roi, full_text, source_frames,
) -> frozenset[str]:
    """Return exact full-text lines licensed for removal by two old sources.

    Only unique, nonnumeric atoms wholly outside the title-bound ROI qualify.
    A source match uses the existing normalized-text and same-position gate;
    an ambiguous source occurrence cannot vote. Invalid input preserves all
    text. This function neither edits atoms nor interprets effect values.
    """
    roi = _validated_box(effect_roi)
    native = _validated_atoms(native_items)
    if (roi is None or native is None or not isinstance(full_text, str)
            or not isinstance(source_frames, Sequence)
            or isinstance(source_frames, (str, bytes)) or not 2 <= len(source_frames) <= 3):
        return frozenset()
    sources = [_validated_atoms(frame) for frame in source_frames]
    if any(frame is None for frame in sources):
        return frozenset()

    native_counts = Counter(normalize(atom["text"]) for atom in native)
    signatures = Counter((atom["text"], atom["box"]) for atom in native)
    lines = full_text.splitlines()
    line_counts = Counter(normalize(line) for line in lines)
    exact_lines = {normalize(line): line for line in lines}
    # Repeated text in one old frame is ambiguous even if one position fits.
    unique_sources = []
    for frame in sources:
        counts = Counter(normalize(atom["text"]) for atom in frame)
        unique_sources.append([
            atom for atom in frame
            if counts[normalize(atom["text"])] == 1
            and len(atom["text"].splitlines()) == 1 and not _protected_text(atom["text"])
        ])
    source_text = PItemSourceTextFrames(unique_sources)
    left, top, width, height = roi
    removable = set()
    for atom in native:
        text, box = atom["text"], atom["box"]
        canonical = normalize(text)
        if (not canonical or len(text.splitlines()) != 1 or _protected_text(text)
                or native_counts[canonical] != 1 or signatures[text, box] != 1
                or line_counts[canonical] != 1 or _protected_text(exact_lines[canonical])):
            continue
        x, y, atom_width, atom_height = box
        # Closed boundaries count as intersecting: touching text is retained.
        if not (x + atom_width < left or left + width < x
                or y + atom_height < top or top + height < y):
            continue
        if len(source_text.matching_frames(atom, atom_height)) >= 2:
            removable.add(exact_lines[canonical])
    return frozenset(removable)
