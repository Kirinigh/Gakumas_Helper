"""Error-only recovery of isolated integers in skill-card detail OCR.

The caller owns title identity and coordinate conversion.  This module only
checks text proximity; it neither identifies source pixels nor selects a
layout according to a catalog result.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal
from statistics import median
from dataclasses import field, dataclass

Box = tuple[int, int, int, int]
Atom = tuple[str, Box]
NUMERIC_TOKEN_BOUNDARY = "\u2063"
_INTEGER = re.compile(r"[0-9]+\Z")
_SIGNED_INTEGER = re.compile(r"[+-][0-9]+\Z")


@dataclass(frozen=True, slots=True)
class NumberDiagnostic:
    index: int
    reason: str
    body_index: int | None = None
    horizontal_gap: float | None = None
    vertical_gap: float | None = None
    body_height: float | None = None


@dataclass(frozen=True, slots=True)
class _LayoutInspection:
    status: Literal["clean", "suspect", "inconclusive"]
    far_indices: tuple[int, ...] = ()
    ambiguous_indices: tuple[int, ...] = ()
    diagnostics: tuple[NumberDiagnostic, ...] = ()
    _atoms: tuple[Atom, ...] = field(default=(), repr=False)
    _body_indices: tuple[int, ...] = field(default=(), repr=False)
    _body_left: int = field(default=0, repr=False)


def _valid_box(box: Box) -> bool:
    return bool(
        len(box) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in box)
        and box[2] > 0 and box[3] > 0
    )



def _inside_roi(box: Box, roi: Box) -> bool:
    x, y, width, height = box
    left, top, roi_width, roi_height = roi
    return bool(left <= x + width / 2 <= left + roi_width
                and top <= y + height / 2 <= top + roi_height)


def _text_token(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _gaps(left: Box, right: Box) -> tuple[int, int]:
    x, y, width, height = left
    a, b, other_width, other_height = right
    return (
        max(0, x - a - other_width, a - x - width),
        max(0, y - b - other_height, b - y - height),
    )


def _text_connected(
    previous: Box, current: Box, body_left: int, *, first_body: bool = False,
) -> bool:
    """Connect nearby text, with a fixed return indent for wrapped lines."""

    x, y, width, height = previous
    a, b, other_width, other_height = current
    scale = min(height, other_height)
    if max(height, other_height) > 1.5 * scale:
        return False
    gap_x, gap_y = _gaps(previous, current)
    overlap_y = min(y + height, b + other_height) - max(y, b)
    if overlap_y >= 0.5 * scale and gap_x <= 2 * scale:
        return True
    return bool(
        b + other_height / 2 > y + height / 2
        and gap_y <= (2 if first_body else 1) * scale
        and (
            min(x + width, a + other_width) > max(x, a)
            or abs(a - body_left) <= scale
        )
    )


def _inspect_distant_numbers(
    atoms: tuple[Atom, ...],
    title_box: Box | None,
    search_roi: Box,
) -> _LayoutInspection:
    """Report isolated unsigned integers strictly more than two text heights away.

    ``search_roi`` limits both the body backbone and eligible integers.
    Outside atoms remain unchanged; this is never a normal-path cleanliness gate.
    A trusted title box may be transformed from the same captured image when
    an existing derived recognition omitted the title.  The caller must reject
    conflicting titles and bind that transform to this exact OCR view.
    """

    tokens = tuple(_text_token(text) for text, _ in atoms)
    numbers = tuple(index for index, token in enumerate(tokens) if _INTEGER.fullmatch(token))
    if not numbers:
        return _LayoutInspection("clean", _atoms=atoms)

    def inconclusive(reason: str) -> _LayoutInspection:
        return _LayoutInspection(
            "inconclusive", ambiguous_indices=numbers,
            diagnostics=tuple(NumberDiagnostic(index, reason) for index in numbers),
            _atoms=atoms,
        )

    if title_box is None or not _valid_box(title_box):
        return inconclusive("trusted_title_missing")
    if not _valid_box(search_roi) or any(not _valid_box(box) for _, box in atoms):
        return inconclusive("geometry_invalid")
    left, top, width, height = search_roi
    numbers = tuple(index for index in numbers if _inside_roi(atoms[index][1], search_roi))
    if not numbers:
        return _LayoutInspection("clean", _atoms=atoms)
    title_x, title_y, _, title_height = title_box
    candidates = []
    for index, (text, box) in enumerate(atoms):
        x, y, item_width, item_height = box
        if not (
            left <= x + item_width / 2 <= left + width
            and top <= y + item_height / 2 <= top + height
            and y >= title_y + title_height / 2
        ):
            continue
        # One-letter icons, punctuation and integers cannot extend the body.
        # All original atoms are still retained by the serializer.
        if sum(character.isalpha() for character in text) >= 2:
            candidates.append(index)
    candidates.sort(key=lambda index: (
        atoms[index][1][1] + atoms[index][1][3] / 2, atoms[index][1][0],
    ))
    body: list[int] = []
    body_left = title_x
    for index in candidates:
        box = atoms[index][1]
        connected = any(
            _text_connected(atoms[other][1], box, body_left)
            for other in reversed(body)
        ) if body else _text_connected(title_box, box, body_left, first_body=True)
        if not connected:
            continue
        if not body:
            body_left = min(title_x, box[0])
        body.append(index)
    if not body:
        return inconclusive("connected_body_missing")

    body_boxes = tuple(atoms[index][1] for index in body)
    min_x = min(box[0] for box in body_boxes)
    min_y = min(box[1] for box in body_boxes)
    max_right = max(box[0] + box[2] for box in body_boxes)
    max_bottom = max(box[1] + box[3] for box in body_boxes)
    max_height = max(box[3] for box in body_boxes)
    envelope = (min_x, min_y, max_right - min_x, max_bottom - min_y)
    far = []
    ambiguous = []
    diagnostics = []
    for index in numbers:
        box = atoms[index][1]
        x, y, item_width, item_height = box
        if not (left <= x and top <= y and x + item_width <= left + width
                and y + item_height <= top + height):
            ambiguous.append(index)
            diagnostics.append(NumberDiagnostic(index, "integer_crosses_search_edge"))
            continue
        # A nearby detached text block is not proven to belong to this title,
        # but its number must not be removed merely because it is not in body.
        detached = next((other for other in candidates if other not in body
                         and max(_gaps(box, atoms[other][1])) <= 2 * atoms[other][1][3]), None)
        if detached is not None:
            ambiguous.append(index)
            gx, gy = _gaps(box, atoms[detached][1])
            diagnostics.append(NumberDiagnostic(
                index, "near_unassociated_text", detached, gx, gy, atoms[detached][1][3],
            ))
            continue
        gap_x, gap_y = _gaps(box, envelope)
        # A distance from the union envelope is a conservative lower bound
        # for distance from every constituent text box, not a panel boundary.
        if max(gap_x, gap_y) > 2 * max_height:
            far.append(index)
            diagnostics.append(NumberDiagnostic(
                index, "far_from_body_envelope", None, gap_x, gap_y, max_height,
            ))
            continue
        nearest = None
        near = False
        uncertain = False
        for body_index, body_box in zip(body, body_boxes, strict=True):
            gap_x, gap_y = _gaps(box, body_box)
            ratio = max(gap_x, gap_y) / body_box[3]
            candidate = (ratio, body_index, gap_x, gap_y, body_box[3])
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
            if gap_x <= 2 * body_box[3] and gap_y <= body_box[3]:
                bx, by, bw, bh = body_box
                nx, ny, nw, nh = box
                aligned = (
                    min(by + bh, ny + nh) > max(by, ny)
                    or min(bx + bw, nx + nw) > max(bx, nx)
                    or abs(nx - body_left) <= bh
                )
                if aligned:
                    near = True
                    nearest = candidate
                    break
                uncertain = True
            elif ratio <= 2:
                uncertain = True
            if near:
                break
        assert nearest is not None
        ratio, body_index, gap_x, gap_y, body_height = nearest
        if not near and uncertain:
            ambiguous.append(index)
        elif not near:
            far.append(index)
        diagnostics.append(NumberDiagnostic(
            index, "near_body" if near else "proximity_ambiguous" if uncertain
            else "far_from_body",
            body_index, gap_x, gap_y, body_height,
        ))
    return _LayoutInspection(
        "inconclusive" if ambiguous else "suspect" if far else "clean",
        tuple(far), tuple(ambiguous), tuple(diagnostics),
        atoms, tuple(body), body_left,
    )



@dataclass(frozen=True, slots=True)
class ErrorNumericLayoutRecovery:
    status: Literal["unchanged", "recovered", "unrecoverable"]
    far_indices: tuple[int, ...]
    ambiguous_indices: tuple[int, ...]
    protected_atoms: tuple[Atom, ...]
    original_text: str
    protected_text: str
    diagnostics: tuple[NumberDiagnostic, ...]


def _raw_text(atoms: tuple[Atom, ...]) -> str:
    return "\n".join(text for text, _ in atoms if text.strip())


@dataclass(frozen=True, slots=True)
class ErrorWrappedSignedRecovery:
    status: Literal["unchanged", "recovered", "unrecoverable"]
    original_atoms: tuple[Atom, ...]
    atom_order: tuple[int, ...]
    reordered_indices: tuple[int, ...]
    protected_text: str
    reason: str


def recover_wrapped_signed_text(
    atoms: tuple[Atom, ...], title_box: Box | None, search_roi: Box,
) -> ErrorWrappedSignedRecovery:
    """Restore one inverted clause row before an intact wrapped signed value.

    Only failed skill details call this helper. It has no catalog access and
    retains every original atom. A comma-ended clause and its sole terminal
    label must share a title-connected row; the signed value must return to
    the body left on the immediately following line. Other layouts are left
    alone. Small icons cannot connect body regions or determine line height.
    """
    original_order = tuple(range(len(atoms)))

    def result(status, reason, order=original_order, changed=(), text=None):
        return ErrorWrappedSignedRecovery(
            status, atoms, order, changed,
            _raw_text(atoms) if text is None else text, reason,
        )

    if (title_box is None or not _valid_box(title_box)
            or not _valid_box(search_roi)
            or any(not _valid_box(box) for _, box in atoms)):
        return result("unrecoverable", "trusted_geometry_missing")
    signed = tuple(i for i, (text, box) in enumerate(atoms)
                   if _SIGNED_INTEGER.fullmatch(_text_token(text))
                   and _inside_roi(box, search_roi))
    if not signed:
        return result("unchanged", "no_signed_wrap")
    core = [i for i, (text, box) in enumerate(atoms)
            if box != title_box and _inside_roi(box, search_roi)
            and box[1] >= title_box[1] + title_box[3] / 2
            and sum(character.isalpha() for character in text) >= 2]
    core.sort(key=lambda i: (atoms[i][1][1] + atoms[i][1][3] / 2, atoms[i][1][0]))

    def same_row(a, b):
        _, ay, _, ah = atoms[a][1]
        _, by, _, bh = atoms[b][1]
        return (max(ah, bh) <= 1.5 * min(ah, bh)
                and min(ay + ah, by + bh) - max(ay, by) >= 0.5 * min(ah, bh))

    rows: list[list[int]] = []
    for i in core:
        if rows and all(same_row(i, other) for other in rows[-1]):
            rows[-1].append(i)
        else:
            rows.append([i])
    body: list[int] = []
    body_left = title_box[0]
    candidates = []
    for row in rows:
        row.sort(key=lambda i: atoms[i][1][0])
        components: list[list[int]] = []
        for i in row:
            # Two text heights are the pair's median-based 2H threshold.
            if (components and _gaps(atoms[components[-1][-1]][1], atoms[i][1])[0]
                    <= atoms[components[-1][-1]][1][3] + atoms[i][1][3]):
                components[-1].append(i)
            else:
                components.append([i])
        for component in components:
            connected = any(
                _text_connected(atoms[previous][1], atoms[i][1], body_left)
                for previous in body for i in component
            ) if body else any(
                _text_connected(title_box, atoms[i][1], body_left, first_body=True)
                for i in component
            )
            if not connected:
                continue
            body.extend(component)
            if (len(component) != 2 or component == sorted(component)
                    or not _text_token(atoms[component[-2]][0]).endswith(("、", ","))
                    or not _text_token(atoms[component[-1]][0]).isalpha()):
                continue
            if any(atoms[left][1][0] + atoms[left][1][2] > atoms[right][1][0]
                   for left, right in zip(component, component[1:], strict=False)):
                continue
            scale = float(median(atoms[i][1][3] for i in component))
            bottom = max(atoms[i][1][1] + atoms[i][1][3] for i in component)
            wraps = [i for i in signed
                     if i > max(component)
                     and 0 <= atoms[i][1][1] - bottom <= scale
                     and abs(atoms[i][1][0] - body_left) <= scale
                     and (2 / 3) * scale <= atoms[i][1][3] <= 1.5 * scale]
            if len(wraps) != 1:
                continue
            stop = wraps[0]
            start = min(component)
            span = list(range(start, stop))
            top = min(atoms[i][1][1] for i in component)
            # Every intervening atom must be on this exact row. An unrelated
            # line, second recipient or overlapping text cannot be moved out
            # of the way to manufacture an association with the signed value.
            if any(
                atoms[i][1][3] > 1.5 * scale
                or min(atoms[i][1][1] + atoms[i][1][3], bottom)
                - max(atoms[i][1][1], top) < 0.5 * min(atoms[i][1][3], scale)
                for i in span
            ):
                continue
            ordered = sorted(span, key=lambda i: (atoms[i][1][0], i))
            if ordered[-1] != component[-1]:
                continue
            candidates.append((start, stop, ordered, component, scale))
    if len(candidates) != 1:
        return result("unrecoverable" if candidates else "unchanged",
                      "multiple_row_repairs" if candidates else "no_unique_inverted_clause_wrap")
    start, stop, ordered, component, scale = candidates[0]
    order = original_order[:start] + tuple(ordered) + original_order[stop:]
    left = min(atoms[i][1][0] for i in component)
    right = max(atoms[i][1][0] + atoms[i][1][2] for i in component)
    protected = []
    for i in order:
        text, box = atoms[i]
        # Keep far row fragments verbatim, with non-collapsible boundaries.
        # They cannot supply a numeric operand to either neighbouring line.
        far = start <= i < stop and max(left - box[0] - box[2], box[0] - right) > 2 * scale
        protected.append(f"{NUMERIC_TOKEN_BOUNDARY}{text}{NUMERIC_TOKEN_BOUNDARY}" if far else text)
    return result("recovered", "same_row_clause_order_with_signed_wrap", order,
                  tuple(ordered), "\n".join(protected))


def recover_distant_number_text(
    atoms: tuple[Atom, ...], title_box: Box | None, search_roi: Box,
) -> ErrorNumericLayoutRecovery:
    """Recover only a failed detail observation, without catalog access or OCR.

    The caller proves unique title identity and same-view coordinates. This
    function does not authorize accepting a card, a badge count, or a new frame.
    Only whole unsigned integers wholly within the search ROI can be isolated.
    Ambiguous atoms stay intact. If an affected text connection is uncertain,
    ``unrecoverable`` is returned; protected text is then diagnostic only.
    """

    inspection = _inspect_distant_numbers(atoms, title_box, search_roi)
    original_text = _raw_text(atoms)
    far = frozenset(inspection.far_indices)
    body = frozenset(inspection._body_indices)
    ambiguous = frozenset(inspection.ambiguous_indices)
    protected = list(atoms)
    diagnostics = list(inspection.diagnostics)
    unsafe_connection = False
    for index in inspection.far_indices:
        previous = next((i for i in range(index - 1, -1, -1)
                         if i not in far and atoms[i][0].strip()), None)
        following = next((i for i in range(index + 1, len(atoms))
                          if i not in far and atoms[i][0].strip()), None)
        replacement = NUMERIC_TOKEN_BOUNDARY
        if previous in ambiguous or following in ambiguous:
            unsafe_connection = True
            diagnostics.append(NumberDiagnostic(index, "adjacent_ambiguous_atom"))
        elif previous in body and following in body:
            assert previous is not None and following is not None
            left_text, right_text = atoms[previous][0].rstrip(), atoms[following][0].lstrip()
            if (left_text[-1].isalpha() and right_text[0].isalpha()
                    and _text_connected(atoms[previous][1], atoms[following][1], inspection._body_left)):
                replacement = ""
            else:
                # Do not silently join two different effect labels, supply a
                # missing operand, or bridge an unobserved paragraph gap.
                unsafe_connection = True
                diagnostics.append(NumberDiagnostic(index, "body_connection_unproven"))
        elif previous in body or following in body:
            # A boundary can safely isolate a trailing external number. It
            # cannot turn an incomplete numeric operator into recovered text.
            adjacent = previous if previous in body else following
            assert adjacent is not None
            token = _text_token(atoms[adjacent][0])
            if token.endswith(("+", "-", "−")) or token.startswith(("+", "-", "−")):
                unsafe_connection = True
                diagnostics.append(NumberDiagnostic(index, "numeric_operand_unproven"))
            other = following if previous in body else previous
            if other is not None and sum(character.isalpha() for character in atoms[other][0]) >= 2:
                unsafe_connection = True
                diagnostics.append(NumberDiagnostic(index, "detached_text_connection"))
        protected[index] = (replacement, atoms[index][1])
    protected_atoms = tuple(protected) if far else atoms
    if unsafe_connection or (inspection.status == "inconclusive" and not inspection._body_indices):
        status = "unrecoverable"
    else:
        status = "recovered" if protected_atoms != atoms else "unchanged"
    return ErrorNumericLayoutRecovery(
        status, inspection.far_indices, inspection.ambiguous_indices,
        protected_atoms, original_text, _raw_text(protected_atoms), tuple(diagnostics),
    )
