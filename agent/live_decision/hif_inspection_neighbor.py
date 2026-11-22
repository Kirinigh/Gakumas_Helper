"""Navigation-only continuity of an unmoved inspected card's right neighbor.

The host establishes the unique inspection input and its source transaction.
This local reference only proves a family from original strong observations
and identical current artwork at the same screen coordinates. It supplies no
upgrade, payment, business action, or input-freshness evidence.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import replace

import numpy as np

try:
    from agent.card_selection.model import frame_identifier
    from agent.card_selection.types import CandidateBox
except ModuleNotFoundError:
    from card_selection.model import frame_identifier
    from card_selection.types import CandidateBox

from .selection import live_selection_geometry_evidence
from .masked_art import FrozenMaskedArt
from .partial_card import build_live_card_views

SCOPE = 'inspection_cancel_unmoved_neighbor_exact_screen_art_v1'
RECOGNIZER = 'card_art_embedding_top5_hif_masked_v1'


def _view(frame_shape, candidates, slot, selected_slot):
    profile = 'hif-four-card-art-v1' if len(candidates) == 4 else 'hif-five-card-art-v2'
    view = build_live_card_views(frame_shape, candidates, layout_profile=profile,
                                 selected_slot=selected_slot)[slot]
    if len(candidates) == 4 and selected_slot is not None and slot == selected_slot+1:
        # Four-card inspection 1549: the selected card's + badge/white outline
        # extends beyond its frame. Add six fixed pixels to the layout's two-
        # pixel guard. Only this navigation comparison changes; model crops and
        # its raw predictions retain their original geometry. mask_key still
        # enforces eight complete grid cells after the inset.
        x, y, width, height = view.visible_box
        ox, oy = view.canvas_offset
        view = replace(view, visible_box=(x+6, y, width-6, height),
                       canvas_offset=(ox+6, oy), visible_ratio=(width-6)/view.canvas_size[0])
    return view


def _model(prediction):
    return (prediction.recognizer, prediction.model_sha256,
            prediction.classes_sha256, prediction.gallery_sha256)


def _common_region(views):
    keys = [FrozenMaskedArt.mask_key(view) for view in views]
    key = max(k[0] for k in keys), min(k[1] for k in keys)
    if key[1]-key[0] < 8 or len({v.canvas_size for v in views}) != 1:
        raise ValueError('inspection neighbor common artwork is too small')
    common = [FrozenMaskedArt.common_view(view, key) for view in views]
    left = max(v.visible_box[0] for v in common)
    top = max(v.visible_box[1] for v in common)
    right = min(v.visible_box[0]+v.visible_box[2] for v in common)
    bottom = min(v.visible_box[1]+v.visible_box[3] for v in common)
    size = views[-1].canvas_size[0]
    # mask_key already requires eight complete grid cells. Their inward
    # integer raster can be 45 pixels at size 181, not ceil(181*8/32)=46.
    minimum_width = math.floor((key[0]+8)*size/32)-math.ceil(key[0]*size/32)
    if right-left < minimum_width or bottom-top != size:
        raise ValueError('inspection neighbor common artwork lacks its full height')
    yy, xx = np.ogrid[top:bottom, left:right]
    inside = np.ones((bottom-top, right-left), dtype=bool)
    center = (size-1)/2
    for view in common:
        origin_x = view.visible_box[0]-view.canvas_offset[0]
        inside &= ((xx-origin_x-center)**2+(yy-view.visible_box[1]-center)**2 <= (center-2)**2)
    if not inside.any():
        raise ValueError('inspection neighbor common artwork has no circle interior')
    return (left, top, right-left, bottom-top), inside


def _source_title_valid(proof, prediction, *, frame_id, family, catalog=None):
    """A complete same-frame title supplements only weak confidence, not margin."""
    if (not isinstance(proof, dict) or proof.get('accepted') is not True
            or proof.get('reason') != 'complete_base_title_and_visual_family_agree'
            or proof.get('frame_id') != frame_id or proof.get('slot') != prediction.slot
            or tuple(proof.get('family', ())) != family or proof.get('upgrade_verified') is not False
            or type(proof.get('confidence')) not in (int, float) or not .95 <= proof['confidence'] <= 1
            or proof.get('visual_confidence') != prediction.confidence
            or not isinstance(proof.get('text'), str) or not proof['text'].strip()):
        return False
    roi, box = proof.get('roi', ()), prediction.box
    if (len(roi) != 4 or len(box) != 4 or any(type(v) is not int for v in (*roi, *box))
            or min(roi[2:]) <= 0 or min(box[2:]) <= 0
            or roi[0] < box[0] or roi[1] < box[1]
            or roi[0]+roi[2] > box[0]+box[2] or roi[1]+roi[3] > box[1]+box[3]):
        return False
    if catalog is not None:
        from .upgrade import normalize_upgrade_title

        normalize = lambda value: normalize_upgrade_title(''.join(value.split())).rstrip('+')
        matching = {key for key, entry in catalog.entries.items() if normalize(entry.name) == normalize(proof['text'])}
        if matching != set(family):
            return False
    return True


def _source_identity_valid(prediction, *, frame_id, family, title_proof=None, catalog=None):
    if (prediction.recognizer != RECOGNIZER or not family
            or not prediction.top_k_card_ids or tuple(prediction.top_k_card_ids[0]) != family
            or not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1
            or not math.isfinite(prediction.margin) or not .05 <= prediction.margin <= 1):
        return False
    if prediction.accepted is True and prediction.reason == 'accepted_visual_family' and prediction.confidence >= .85:
        return title_proof is None
    return (prediction.accepted is False and prediction.reason == 'below_acceptance_threshold'
            and prediction.confidence < .85
            and _source_title_valid(title_proof, prediction, frame_id=frame_id, family=family, catalog=catalog))


def build_reference(*, run_id, revision, selected_slot, frames, candidates, predictions, frame_ids,
                    title_proofs=None, catalog=None):
    """Retain three independently proved pre-inspection neighbor crops.

    The returned dict is private runtime data, rebuilt by the host on restore;
    its read-only pixel arrays are not an input snapshot or a persisted receipt.
    """
    if (not run_id or type(revision) is not int or type(selected_slot) is not int
            or len(frames) != 3 or len(candidates) != 3 or len(predictions) != 3
            or len(frame_ids) != 3 or len(set(frame_ids)) != 3):
        raise ValueError('inspection neighbor source requires three independent observations')
    count, slot = len(candidates[0]), selected_slot + 1
    boxes = tuple(tuple(c.box) for c in candidates[0])
    if count not in (4, 5) or not 0 <= selected_slot < slot < count:
        raise ValueError('inspection neighbor source has no overlapping right neighbor')
    titles = (None,)*3 if title_proofs is None else title_proofs
    if len(titles) != 3 or any(proof is not None for proof in titles) and catalog is None:
        raise ValueError('inspection neighbor title source requires three catalog-checked observations')
    sources, family, model = [], None, None
    for frame, row, batch, fid, title in zip(frames, candidates, predictions, frame_ids, titles, strict=True):
        geometry = live_selection_geometry_evidence(row)
        if (frame.shape != (1280, 720, 3) or frame.dtype != np.uint8
                or frame_identifier(frame) != fid or len(batch) != count
                or tuple(c.slot for c in row) != tuple(range(count))
                or tuple(tuple(c.box) for c in row) != boxes
                or not geometry['available'] or geometry['ambiguous'] or geometry['raised_slots']
                or any(p.slot != c.slot or tuple(p.box) != tuple(c.box) or p.frame_id != fid
                       for p, c in zip(batch, row, strict=True))):
            raise ValueError('inspection neighbor source geometry or frame identity changed')
        p = batch[slot]
        current_family = tuple(p.top_k_card_ids[0]) if p.top_k_card_ids else ()
        if (not _source_identity_valid(p, frame_id=fid, family=current_family, title_proof=title, catalog=catalog)
                or family is not None and current_family != family
                or model is not None and _model(p) != model):
            raise ValueError('inspection neighbor source family is not independently strong')
        family, model = current_family, _model(p)
        view = _view(frame.shape, row, slot, None)
        FrozenMaskedArt.mask_key(view)  # Preserve the existing minimum visible area.
        x, y, w, h = view.visible_box
        pixels = frame[y:y+h, x:x+w].copy()
        pixels.flags.writeable = False
        sources.append(dict(view=view, pixels=pixels, prediction=p.to_dict(), title_proof=deepcopy(title)))
    return dict(scope=SCOPE, run_id=run_id, revision=revision, selected_slot=selected_slot,
                slot=slot, frame_ids=tuple(frame_ids), boxes=boxes, family=family,
                model=model, sources=tuple(sources))


def corroborate(context, frame, candidates, predictions, *, selected_slot, evidence):
    """Return a separate family proof; never promote the weak art prediction."""
    run = getattr(context, 'gkh_hif_live_context', None)
    ref = getattr(context, 'gkh_hif_inspection_neighbor_reference', None)
    if (run is None or not isinstance(ref, dict) or ref.get('scope') != SCOPE
            or (ref.get('run_id'), ref.get('revision')) != (run.run_id, run.revision)
            or selected_slot != ref.get('selected_slot') or selected_slot is None
            or frame.shape != (1280, 720, 3) or frame.dtype != np.uint8
            or len(candidates) != len(ref['boxes']) or len(predictions) != len(candidates)):
        return None
    slot, fid = ref['slot'], frame_identifier(frame)
    if (fid in ref['frame_ids'] or tuple(c.slot for c in candidates) != tuple(range(len(candidates)))
            or any(p.slot != c.slot or tuple(p.box) != tuple(c.box) or p.frame_id != fid
                   for p, c in zip(predictions, candidates, strict=True))):
        return None
    geometry = live_selection_geometry_evidence(candidates)
    if (not geometry['available'] or geometry['ambiguous']
            or tuple(geometry['raised_slots']) != (selected_slot,)):
        return None
    for i, (old, current) in enumerate(zip(ref['boxes'], candidates, strict=True)):
        box = tuple(current.box)
        if i != selected_slot:
            if box != old:
                return None
        elif (box[2:] != old[2:] or abs(box[0]-old[0]) > 8 or not 18 <= old[1]-box[1] <= 34):
            return None
    p = predictions[slot]
    if (p.accepted or p.reason not in ('below_acceptance_threshold', 'below_margin_threshold')
            or _model(p) != ref['model'] or not p.top_k_card_ids
            or tuple(p.top_k_card_ids[0]) != ref['family']
            or not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1
            or not math.isfinite(p.margin) or not 0 <= p.margin <= 1):
        return None
    try:
        fresh_view = _view(frame.shape, candidates, slot, selected_slot)
        views = [source['view'] for source in ref['sources']] + [fresh_view]
        screen_box, inside = _common_region(views)
    except ValueError:
        return None
    left, top, width, height = screen_box
    right, bottom = left+width, top+height
    pixels = frame[top:bottom, left:right][inside]
    for source in ref['sources']:
        x, y, _, _ = source['view'].visible_box
        old = source['pixels'][top-y:bottom-y, left-x:right-x]
        if old.shape[:2] != inside.shape or not np.array_equal(old[inside], pixels):
            return None
    proof = dict(accepted=True, scope=SCOPE, frame_id=fid, slot=slot, family=ref['family'],
                 selected_slot=selected_slot, reference_frame_ids=ref['frame_ids'],
                 run_id=ref['run_id'], revision=ref['revision'],
                 source_predictions=tuple(source['prediction'] for source in ref['sources']),
                 source_candidate_boxes=ref['boxes'], source_model=ref['model'],
                 pixel_comparison='exact_all_source_circle_interiors', common_screen_box=screen_box,
                 interior_pixel_count=int(inside.sum()), upgrade_verified=False,
                 original_prediction=p.to_dict())
    if any(source.get('title_proof') is not None for source in ref['sources']):
        proof['source_title_proofs'] = tuple(deepcopy(source['title_proof']) for source in ref['sources'])
    evidence.append(proof)
    return proof


def valid_proof(proof, *, prediction, candidates, frame_id, run_id, revision, selected_slot, family, catalog=None):
    """Recheck a recorded pixel proof's identity and complete geometry envelope.

    The accepted exact comparison is a recorded observation, like a retained
    OCR observation; this validator does not claim to re-read missing pixels.
    Legacy proofs without the complete source envelope cannot authorize a new
    independently inspected reference.
    """
    try:
        raw = prediction if isinstance(prediction, dict) else prediction.to_dict()
        family = tuple(family)
        ids = proof['reference_frame_ids']
        slot = proof['slot']
        if (proof.get('accepted') is not True or proof.get('scope') != SCOPE
                or proof.get('upgrade_verified') is not False
                or proof.get('pixel_comparison') != 'exact_all_source_circle_interiors'
                or not run_id or type(revision) is not int
                or (proof.get('run_id'), proof.get('revision')) != (run_id, revision)
                or type(selected_slot) is not int or proof.get('selected_slot') != selected_slot
                or type(slot) is not int or slot != selected_slot+1
                or not family or tuple(proof.get('family', ())) != family
                or len(ids) != 3 or any(not isinstance(fid, str) or not fid for fid in ids)
                or len(set(ids)) != 3 or frame_id in ids or not isinstance(frame_id, str) or not frame_id
                or proof.get('frame_id') != frame_id
                or json.dumps(raw, sort_keys=True, allow_nan=False) != json.dumps(
                    proof.get('original_prediction'), sort_keys=True, allow_nan=False)):
            return False
        boxes = tuple(tuple(box) for box in proof['source_candidate_boxes'])
        rows = tuple(CandidateBox(slot=c['slot'], box=tuple(c['box']))
                     if isinstance(c, dict) else c for c in candidates)
        if (len(boxes) not in (4, 5) or len(rows) != len(boxes)
                or not 0 <= selected_slot < slot < len(boxes)
                or any(type(c.slot) is not int for c in rows)
                or tuple(c.slot for c in rows) != tuple(range(len(boxes)))
                or any(len(box) != 4 or any(type(v) is not int for v in box)
                       or min(box[2:]) <= 0 for box in (*boxes, *(tuple(c.box) for c in rows)))):
            return False
        old_rows = tuple(CandidateBox(slot=i, box=box) for i, box in enumerate(boxes))
        old_geometry = live_selection_geometry_evidence(old_rows)
        geometry = live_selection_geometry_evidence(rows)
        if (not old_geometry['available'] or old_geometry['ambiguous'] or old_geometry['raised_slots']
                or not geometry['available'] or geometry['ambiguous']
                or tuple(geometry['raised_slots']) != (selected_slot,)):
            return False
        for i, (old, current) in enumerate(zip(boxes, rows, strict=True)):
            box = tuple(current.box)
            if i != selected_slot and box != old:
                return False
            if i == selected_slot and (box[2:] != old[2:] or abs(box[0]-old[0]) > 8
                                       or not 18 <= old[1]-box[1] <= 34):
                return False
        current = SimpleNamespace(**raw)
        model = tuple(proof['source_model'])
        if (len(model) != 4 or any(not isinstance(value, str) or not value for value in model)
                or model[0] != RECOGNIZER or _model(current) != model
                or current.slot != slot or tuple(current.box) != tuple(rows[slot].box)
                or current.frame_id != frame_id or current.accepted is not False
                or current.reason not in ('below_acceptance_threshold', 'below_margin_threshold')
                or not current.top_k_card_ids or tuple(current.top_k_card_ids[0]) != family
                or not math.isfinite(current.confidence) or not 0 <= current.confidence <= 1
                or not math.isfinite(current.margin) or not 0 <= current.margin <= 1):
            return False
        sources = proof['source_predictions']
        titles = proof.get('source_title_proofs', (None,)*3)
        if len(sources) != 3 or len(titles) != 3:
            return False
        for fid, source, title in zip(ids, sources, titles, strict=True):
            p = SimpleNamespace(**source)
            if (p.slot != slot or tuple(p.box) != boxes[slot] or p.frame_id != fid
                    or _model(p) != model
                    or not _source_identity_valid(p, frame_id=fid, family=family, title_proof=title, catalog=catalog)):
                return False
        views = [_view((1280, 720, 3), old_rows, slot, None),
                 _view((1280, 720, 3), rows, slot, selected_slot)]
        screen_box, inside = _common_region(views)
        return (tuple(proof['common_screen_box']) == screen_box
                and type(proof['interior_pixel_count']) is int
                and proof['interior_pixel_count'] == int(inside.sum()))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
