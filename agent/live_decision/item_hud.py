"""Single-frame item sidebar observation and reconciliation with the detail panel."""

import re
import json
import math
import time
import logging
from functools import lru_cache
from dataclasses import dataclass

import cv2
import numpy as np

from .inventory import _inventory_ocr
from .item_panel import resolve_item_descriptions, observe_item_description_transaction
from .status_panel import StatusInspectionError, _normalize_status_text
from .hif_capture_adapter import captured_at as frame_captured_at


@dataclass(frozen=True)
class ArchivedLayout:
    slots: int
    left: int
    icon_size: int
    pitch: float


REGULAR = ArchivedLayout(4, 634, 66, 69.33)
COMPACT = ArchivedLayout(5, 647, 52, 55.8)


class ItemHudCatalogCoverageError(ValueError):
    """The icon index does not cover the requested item catalog."""


def highpass(pixels):
    image = pixels.astype(np.float32)
    return image - cv2.GaussianBlur(image, (0, 0), 3)


def rank_icons(image, references, left, top, size):
    pad = 4
    search = highpass(image[top-pad:top+size+pad, left-pad:left+size+pad])
    scores = []
    for asset, template, mask in references:
        values = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        values[~np.isfinite(values)] = -1
        _, score, _, (x, y) = cv2.minMaxLoc(values)
        scores.append((float(score), asset, (left-pad+x, top-pad+y, size, size)))
    return sorted(scores, reverse=True)[:2]


def badge_edge_support(gray, x, y):
    """Look for a disc edge with one contrast direction around its rim.

    The translucent badge can be lighter than a saturated dark backdrop.
    """
    angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    contrasts = []
    for angle in angles:
        if .32 < angle < 2.55:
            continue
        dx, dy = math.cos(angle), math.sin(angle)
        inner = [gray[round(y + r*dy), round(x + r*dx)] for r in (8, 9, 10)]
        outer = [gray[round(y + r*dy), round(x + r*dx)] for r in (13, 14, 15)]
        contrasts.append(np.median(outer) - np.median(inner))
    return max(sum(float(value) >= 9 for value in contrasts),
               sum(float(value) <= -9 for value in contrasts))


def read_badge(context, frame, gray, y):
    edge_support = badge_edge_support(gray, 697, y)
    if edge_support < 6:
        return dict(value=None, edge_support=edge_support, reads=[])
    reads = []
    for left, right, radius, light in ((686, 709, 12, 220), (691, 704, 10, 210), (690, 705, 10, 185)):
        crop = frame[y-radius:y+radius+1, left:right]
        glyph = np.where(crop.min(axis=2) >= light, 0, 255).astype(np.uint8)
        if cv2.countNonZero(255 - glyph) < 8:
            continue
        glyph = cv2.copyMakeBorder(glyph, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        glyph = cv2.cvtColor(cv2.resize(glyph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        rows = _inventory_ocr(context, glyph, "Task086HudDigit", only_rec=True)
        if len(rows) == 1:
            reads.append((rows[0].text, rows[0].score))
    # Same-frame rereads only. Multi-character/ambiguous/weak data stay unknown.
    strong = [int(text) for text, score in reads if len(text) == 1 and text in "0123456789" and .95 <= score <= 1]
    value = strong[0] if strong and len(set(strong)) == 1 else None
    return dict(value=value, edge_support=edge_support, reads=reads)


def observe_frame(context, frame, layout, references, crosswalk):
    if frame.shape != (1280, 720, 3):
        raise ValueError("unsupported archived HUD shape")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = []
    for ordinal in range(layout.slots):
        shift = round(layout.pitch * ordinal)
        ranked = rank_icons(frame, references[layout.icon_size], layout.left, 244 + shift, layout.icon_size)
        (score, asset, box), (second, _, _) = ranked
        anchor = score >= .50 and score - second >= .15
        ids = crosswalk.get(asset, ())
        item_id = ids[0] if anchor and len(ids) == 1 else None
        badge = read_badge(context, frame, gray, 246 + shift)
        rows.append(dict(slot=ordinal + 1, slot_evidence=anchor or badge["value"] is not None,
            icon_box=box, top_assets=ranked, candidates=ids, accepted_asset=asset if anchor else None,
            p_item_id=item_id, remaining_uses=badge["value"], badge=badge))
    return rows


def read_bound_item_uses(context, frame, directory, limited):
    """Read numeric badges only on artwork already bound to the loadout."""
    if not limited or getattr(frame, 'shape', ()) != (1280, 720, 3):
        return {}
    references, crosswalk = load_hud_references(directory, frozenset())
    assets = {asset: keys for asset, ids in crosswalk.items() if len(keys := limited.intersection(ids)) == 1}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    counts = {}
    for layout in (REGULAR, COMPACT):
        templates = [row for row in references[layout.icon_size] if row[0] in assets]
        if not templates:
            continue
        for ordinal in range(layout.slots):
            shift = round(layout.pitch * ordinal)
            ranked = rank_icons(frame, templates, layout.left, 244 + shift, layout.icon_size)
            score, asset, _ = ranked[0]
            if score < .50 or (len(ranked) > 1 and score - ranked[1][0] < .15):
                continue
            value = read_badge(context, frame, gray, 246 + shift)['value']
            if value is not None:
                item, = assets[asset]
                counts.setdefault(item, set()).add(value)
    return {item: values.pop() for item, values in counts.items() if len(values) == 1}


@lru_cache(maxsize=2)
def load_hud_references(directory, required_ids, sizes=(REGULAR.icon_size, COMPACT.icon_size)):
    """Read the Live icon source, including produce items and shared art."""
    crosswalk = {}
    mapped_ids = set()
    entries = json.loads((directory / "crosswalk.json").read_bytes())
    if not isinstance(entries, list):
        raise ValueError("invalid Live item crosswalk")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid Live item crosswalk entry")
        item_id, asset = str(entry.get("business_id")), entry.get("asset_name")
        if (not re.fullmatch(r"(?:[0-9]+|pitem_[0-9]{2}-[0-3]-[0-9]{3}-[0-9](?:-[0-9]{3})?)", item_id)
                or item_id in mapped_ids
                or not isinstance(asset, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", asset)):
            raise ValueError("ambiguous Live item crosswalk entry")
        mapped_ids.add(item_id)
        crosswalk.setdefault(asset, []).append(item_id)
    if not required_ids <= mapped_ids:
        raise ItemHudCatalogCoverageError("Live item crosswalk does not cover the current stage catalog")

    references = {size: [] for size in sizes}
    for path in sorted((directory / "official-art").glob("*.webp")):
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("Live item source must contain transparent art")
        for size, templates in references.items():
            resized = cv2.resize(pixels, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.erode((resized[:, :, 3] > 180).astype(np.uint8), np.ones((3, 3), np.uint8))
            if mask.sum() >= 20:
                templates.append((path.stem, highpass(resized[:, :, :3]), mask))
    if any(len(templates) < 2 or not crosswalk.keys() <= {row[0] for row in templates}
           for templates in references.values()):
        raise ValueError("Live item icon source is incomplete")
    return references, crosswalk


def observe_default_item_slots(context, frame, *, directory, metadata):
    """Identify the layout by the different downstream icon positions."""
    if getattr(frame, "shape", ()) != (1280, 720, 3) or not (directory / "crosswalk.json").is_file():
        return None
    try:
        required_ids = frozenset(row["id"] for row in metadata if row["mode"] == "stage")
        references, crosswalk = load_hud_references(directory, required_ids)
    except (OSError, ValueError, cv2.error) as error:
        logging.getLogger(__name__).warning("Live item HUD references unavailable: %s", error)
        return None

    candidates = []
    for layout in (REGULAR, COMPACT):
        rows = observe_frame(context, frame, layout, references, crosswalk)
        # The first slot starts at the same height in both layouts. Use
        # positively matched icons in distinct later positions to bind slots.
        if sum(bool(row["accepted_asset"]) for row in rows[1:]) >= 2:
            candidates.append(rows)
    return candidates[0] if len(candidates) == 1 else None


def resolve_item_inventory(observed, metadata, compiled_catalog, *, plan, confirm):
    """Preserve sufficient HUD rows and confirm the ordered inventory at most once.

    The caller supplies a calibrated, complete slot layout and a same-state
    confirmation path. The confirmation returns the full native panel result.
    """
    slots = tuple(observed)
    if not slots or any(type(row["slot"]) is not int or row["slot"] != index
                        for index, row in enumerate(slots, 1)):
        raise ValueError("HUD item slots are not a complete ordered observation")
    available = {}
    for row in metadata:
        if row.get("mode") == "stage" and row.get("plan") in ("free", plan):
            available.setdefault(row["id"], []).append(row)

    direct = []
    for row in slots:
        item_id, remaining = row.get("p_item_id"), row.get("remaining_uses")
        matches = available.get(item_id, ())
        definition = compiled_catalog.items.get(item_id)
        effects = definition.persistent_effects if definition is not None else ()
        if (row.get("slot_evidence") is not True or not row.get("accepted_asset")
                or list(row.get("candidates", ())) != [item_id]
                or len(matches) != 1 or type(remaining) is not int
                or len(effects) != 1 or type(effects[0].schedule_limit) is not int
                or not 0 <= remaining <= effects[0].schedule_limit
                or matches[0]["upgraded"] not in ("TRUE", "FALSE")):
            direct.append(None)
            continue
        badge_confidence = [score for text, score in row["badge"]["reads"]
                            if text == str(remaining) and .95 <= score <= 1]
        tops = row["top_assets"]
        if not badge_confidence or len(tops) != 2:
            direct.append(None)
            continue
        score, asset, _ = tops[0]
        if (asset != row["accepted_asset"] or not .50 <= score <= 1
                or not -1 <= tops[1][0] <= 1 or score - tops[1][0] < .15):
            direct.append(None)
            continue
        direct.append(dict(item_id=item_id, name=_normalize_status_text(matches[0]["name"]),
                           upgraded=matches[0]["upgraded"] == "TRUE", remaining_uses=remaining,
                           use_limit=effects[0].schedule_limit, confidence=min(score, max(badge_confidence)),
                           definition_effect_index=0))

    if all(row is not None for row in direct):
        _require_unique_inventory(direct)
        return tuple(direct), ["hud"] * len(direct), None

    confirmed, events = confirm()
    if (len(confirmed) != len(slots) or not events or events[-1].get("accepted") is not True
            or events[-1].get("complete_list") is not True or events[-1].get("returned_unchanged") is not True):
        raise ValueError("item detail did not confirm the full HUD layout")
    _require_unique_inventory(confirmed)
    for hud, strong, detail in zip(slots, direct, confirmed, strict=True):
        item_id = hud.get("p_item_id")
        aliases = hud.get("candidates")
        remaining = hud.get("remaining_uses")
        if (item_id is not None and item_id != detail["item_id"]
                or hud.get("accepted_asset") and aliases and detail["item_id"] not in aliases
                or remaining is not None and remaining != detail["remaining_uses"]
                or strong is not None and strong["upgraded"] != detail["upgraded"]):
            raise ValueError(f"HUD and item detail disagree at physical slot {hud['slot']}")
    return tuple(strong if strong is not None else detail
                 for strong, detail in zip(direct, confirmed, strict=True)), [
        "detail" if row is None else "hud" for row in direct], events


def _require_unique_inventory(rows):
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("item inventory does not have unique identities")


def observe_hud_first_item_inventory(context, *, hud_reader, capture_frames, checkpoint,
                                     frame_identifier, maximum_age_ms, metadata, compiled_catalog, plan):
    """Read one Live frame and use the original panel at most once."""
    status = dict(phase="item_hud_inventory", accepted=False, complete_list=False,
                  returned_unchanged=False, input_click_count=0, strategy_state_bound=False)
    detail_events = []
    initial = None

    def confirm():
        pages, events = observe_item_description_transaction(
            context, capture_frames=capture_frames, checkpoint=checkpoint,
            frame_identifier=frame_identifier, maximum_age_ms=maximum_age_ms,
            complete_list=True, expected_live=initial)
        detail_events.extend(events)
        return resolve_item_descriptions(pages, metadata, compiled_catalog, plan=plan), events

    def result_events(events):
        if not events:
            return [status]
        events[-1]["item_hud_observation"] = dict(frame_ids=status["before_frame_ids"],
                                                 slots=status.get("hud_slots", ()))
        return events

    try:
        if type(maximum_age_ms) not in (int, float) or not 0 < maximum_age_ms <= 1000:
            raise ValueError("item HUD freshness gate is invalid")
        controller = context.tasker.controller
        frame = controller.post_screencap().wait().get()
        upstream = getattr(context, "gkh_hif_upstream_card_ui", False) is True
        captured = None if upstream else frame_captured_at(controller, frame)
        frame_id = frame_identifier(frame)
        status["before_frame_ids"] = [frame_id]
        detected = hud_reader(context, frame)
        observed = () if detected is None else tuple(detected)
        status["hud_slots"] = observed
        if observed:
            initial = checkpoint((frame,))
            if tuple(initial["frame_ids"]) != (frame_id,):
                raise ValueError("item HUD checkpoint did not certify the observed frame")
            items, sources, _ = resolve_item_inventory(
                observed, metadata, compiled_catalog, plan=plan, confirm=confirm)
        else:
            items, events = confirm()
            if (not events or events[-1].get("accepted") is not True
                    or events[-1].get("complete_list") is not True
                    or events[-1].get("returned_unchanged") is not True):
                raise ValueError("item detail did not confirm the inventory")
            sources = ["detail"] * len(items)
        if not detail_events:
            if captured is not None and not 0 <= (time.perf_counter() - captured) * 1000 <= maximum_age_ms:
                raise ValueError("item HUD observation is stale")
            status.update(accepted=True, complete_list=True, returned_unchanged=True, after_frame_ids=[frame_id])
        events = result_events(detail_events)
        events[-1].update(item_slot_sources=sources, item_hud_frame_id=frame_id)
        return items, events
    except StatusInspectionError as error:
        raise StatusInspectionError(str(error), result_events(error.events)) from error
    except ValueError as error:
        events = result_events(detail_events)
        events[-1].update(accepted=False, identity_count_resolution_error=str(error))
        raise StatusInspectionError(str(error), events) from error
