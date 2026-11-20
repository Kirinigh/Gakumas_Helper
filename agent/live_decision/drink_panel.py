"""Read HIF drink evidence from normalized frames and explicit observations."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .layout import calibrated_live_client_size

_SLOT_X = (17, 102, 187, 273)


@dataclass(frozen=True)
class DrinkSlotObservation:
    slot: int
    state: str
    drink_id: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class DrinkDetailEvidence:
    """Read-only, scoped detail and verified return supplied by the caller."""

    slot: int
    drink_id: str
    confidence: float
    before_frame_ids: tuple[str, ...]
    return_frame_ids: tuple[str, ...]
    returned_unchanged: bool


def _frame_ids_are_independent(frame_ids):
    return len(frame_ids) == 3 and all(isinstance(value, str) and value for value in frame_ids) and len(set(frame_ids)) == 3


def _slot_evidence(observations):
    if len(observations) != 3 or any(len(row) != len(_SLOT_X) for row in observations):
        raise ValueError("drink occupancy requires four slots in each of three frames")
    for row in observations:
        for box, left in zip(row, _SLOT_X, strict=True):
            if box is None:
                continue
            if len(box) != 4 or any(type(value) is not int for value in box):
                raise ValueError("drink occupancy box is invalid")
            x, y, width, height = box
            if not (width > 0 and height > 0 and left <= x < x + width <= left + 90 and 1154 <= y < y + height <= 1244):
                raise ValueError("drink occupancy box exceeds its physical slot")
    return tuple(zip(*observations, strict=True))


def _stable_box(boxes):
    return (
        all(box is not None for box in boxes)
        and all(box[2:] == boxes[0][2:] for box in boxes)
        and all(max(box[axis] for box in boxes) - min(box[axis] for box in boxes) <= 2 for axis in (0, 1))
    )


def observe_drink_inventory(
    frames, occupied, empty, *, client_size, frame_ids, catalog_ids, recognizer=None, authorized_slot=None, read_detail=None
):
    """Resolve positively observed slots. A caller may read only one authorized slot.

    The optional callback supplies scoped detail evidence and a verified return.
    This function owns no controller or input endpoint.
    """
    frame_ids = tuple(frame_ids)
    if (
        not _frame_ids_are_independent(frame_ids)
        or len(frames) != 3
        or any(not isinstance(frame, np.ndarray) or frame.shape != (1280, 720, 3) or frame.dtype != np.uint8 for frame in frames)
        or calibrated_live_client_size(client_size) is None
    ):
        raise ValueError("drink inventory requires three distinct normalized frame observations")
    if authorized_slot is not None and (type(authorized_slot) is not int or not 0 <= authorized_slot < 4):
        raise ValueError("authorized drink slot is invalid")

    occupied_slots = _slot_evidence(occupied)
    empty_slots = _slot_evidence(empty)
    states = []
    for filled, vacant in zip(occupied_slots, empty_slots, strict=True):
        if _stable_box(filled) and all(box is None for box in vacant):
            states.append("OCCUPIED")
        elif _stable_box(vacant) and all(box is None for box in filled):
            states.append("EMPTY")
        else:
            states.append("UNKNOWN")

    catalog_ids = frozenset(catalog_ids)
    physical_slots = [slot for slot, state in enumerate(states) if state == "OCCUPIED"]
    icons = recognizer(frames, physical_slots, client_size=client_size) if recognizer and physical_slots else {}
    result = []
    for slot, state in enumerate(states):
        icon = icons.get(slot, {})
        drink_id, confidence = icon.get("drink_id"), icon.get("confidence")
        if (
            state == "OCCUPIED"
            and drink_id in catalog_ids
            and type(confidence) in (int, float)
            and math.isfinite(confidence)
            and 0.5 <= confidence <= 1
        ):
            result.append(DrinkSlotObservation(slot, state, drink_id, confidence))
        else:
            result.append(DrinkSlotObservation(slot, state))

    if (
        read_detail is None
        or authorized_slot is None
        or "UNKNOWN" in states
        or states[authorized_slot] != "OCCUPIED"
        or result[authorized_slot].drink_id is not None
    ):
        return tuple(result)
    detail = read_detail(slot=authorized_slot, frame_ids=frame_ids)
    if (
        isinstance(detail, DrinkDetailEvidence)
        and detail.slot == authorized_slot
        and detail.drink_id in catalog_ids
        and type(detail.confidence) in (int, float)
        and math.isfinite(detail.confidence)
        and 0.95 <= detail.confidence <= 1
        and detail.before_frame_ids == frame_ids
        and detail.returned_unchanged is True
        and _frame_ids_are_independent(detail.return_frame_ids)
        and not set(frame_ids).intersection(detail.return_frame_ids)
    ):
        result[authorized_slot] = DrinkSlotObservation(authorized_slot, "OCCUPIED", detail.drink_id, detail.confidence)
    return tuple(result)


def empty_drink_slot_targets(image, template, *, client_size=None, render_profile="standard"):
    """Positive empty-cup evidence, independent of occupied-anchor misses."""
    if getattr(image, "shape", ()) != (1280, 720, 3) or getattr(template, "shape", ()) != (45, 36):
        raise ValueError("empty drink glyph layout is uncalibrated")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    reference = template.astype(np.float32)
    compact = (
        isinstance(client_size, tuple)
        and len(client_size) == 2
        and all(type(value) is int for value in client_size)
        and (calibrated_live_client_size(client_size) == (480, 853) or client_size in ((540, 960), (541, 961), (541, 962), (542, 962)))
    )
    threshold = 0.90 if compact else 0.95
    if (
        render_profile not in ("standard", "compact_route_opacity_binomial3")
        or render_profile != "standard"
        and calibrated_live_client_size(client_size) != (480, 853)
    ):
        raise ValueError("empty drink render profile is uncalibrated")
    route_raster = render_profile == "compact_route_opacity_binomial3"
    # The normalized 480px client gives adjacent slots different horizontal
    # half-pixel sampling phases (884 slot 2 versus the slot 3 reference).
    # Keep the same full cup; merge bounded phase responses before applying
    # the profile threshold and unchanged unique-target check.
    phases = (
        reference,
        *(
            cv2.warpAffine(
                reference,
                np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32),
                (36, 45),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for dx in (-0.5, 0.5)
        ),
    )
    matched_phases = tuple(cv2.GaussianBlur(phase, (3, 3), 0) for phase in phases) if route_raster else phases
    slots = []
    for x, anchor_x in zip((17, 102, 187, 273), (44, 129, 214, 300), strict=True):
        crop = image[1154:1244, x : x + 90]
        # A translucent white stroke obeys the same alpha equation in the
        # minimum RGB channel. It retains contrast on saturated PV lighting
        # and skin where grayscale compression obscures an intact cup (940).
        target = None
        gray_score = None
        contrasts = (
            (("minimum_channel_opacity_binomial3", crop.min(axis=2)),)
            if route_raster
            else (("grayscale", cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), ("minimum_channel", crop.min(axis=2)))
        )
        for basis, contrast in contrasts:
            glyphs = cv2.morphologyEx(contrast, cv2.MORPH_TOPHAT, kernel)
            # The empty cup is translucent white: foreground contrast equals
            # alpha*(255-background). Remove that background dependence (865)
            # while retaining the full glyph and raw contrast check.
            background = contrast.astype(np.float32) - glyphs
            opacity = glyphs.astype(np.float32) / np.maximum(255 - background, 1)
            measured_opacity = cv2.GaussianBlur(opacity, (3, 3), 0) if route_raster else opacity
            responses = [cv2.matchTemplate(measured_opacity, phase, cv2.TM_CCOEFF_NORMED) for phase in matched_phases]
            response = np.maximum.reduce(responses)
            _, score, _, (dx, dy) = cv2.minMaxLoc(response)
            if basis == "grayscale":
                gray_score = float(score)
            mask = (np.isfinite(response) & (response >= threshold)).astype(np.uint8)
            count, _ = cv2.connectedComponents(mask)
            if count > 2:
                raise ValueError("empty drink glyph has multiple possible targets")
            if count == 1 or float(glyphs[dy : dy + 45, dx : dx + 36].std()) < 5:
                continue
            if compact and (abs(x + dx - anchor_x) > 2 or abs(1154 + dy - 1177) > 2):
                continue
            proof = {}
            if route_raster:
                phase_index = int(np.argmax([value[dy, dx] for value in responses]))
                observed = measured_opacity[dy : dy + 45, dx : dx + 36]
                chosen = matched_phases[phase_index]
                parts = [
                    float(cv2.matchTemplate(observed[top:bottom], chosen[top:bottom], cv2.TM_CCOEFF_NORMED)[0, 0])
                    for top, bottom in ((0, 15), (15, 30), (30, 45))
                ]
                if not np.isfinite(parts).all() or min(parts) < 0.8:
                    continue
                proof = dict(
                    normalization_profile=render_profile,
                    normalization_kernel="binomial_3x3",
                    sampling_phase_x=(0, -0.5, 0.5)[phase_index],
                    component_correlations=parts,
                    raw_confidence=float(
                        cv2.matchTemplate(opacity[dy : dy + 45, dx : dx + 36], phases[phase_index], cv2.TM_CCOEFF_NORMED)[0, 0]
                    ),
                )
            target = dict(
                box=(x + dx, 1154 + dy, 36, 45),
                confidence=float(score),
                grayscale_confidence=gray_score,
                contrast_basis=basis,
                acceptance_threshold=threshold,
                client_size=client_size if compact else None,
                **proof,
            )
            break  # Preserve a passed grayscale proof; no pixels are combined.
        slots.append(target)
    return tuple(slots)
