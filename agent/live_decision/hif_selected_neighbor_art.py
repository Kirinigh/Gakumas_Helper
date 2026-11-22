"""Current-transaction A/B artwork continuity, separate from raw recognition.

The caller verifies the accepted A, one selection, complete B/SELECT, business
instance binding and input deadline. This module only proves the neighbor's
family from fresh A pixels and an independent inspection receipt or A's own
complete title on its authoritative last frame.
References are private runtime objects, never restored from report JSON.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np

from .selection import live_selection_geometry_evidence
from .hif_inspected_art import visible_art, independent_detail_reference_valid
from .hif_inspection_neighbor import RECOGNIZER, _view, _model, _common_region, _source_identity_valid

try:
    from agent.card_selection.model import frame_identifier
except ModuleNotFoundError:
    from card_selection.model import frame_identifier

SCOPE = 'current_selected_neighbor_inspected_a_exact_screen_art_v1'


def build_strong_hand_reference(*, run_id, revision, selected_slot, frames, candidates,
                                predictions, generation, hand_instance_ids):
    """Keep independently strong A artwork for unchanged, nonselected instances."""
    count = len(hand_instance_ids)
    ids = tuple(frame_identifier(frame) for frame in frames)
    boxes = tuple(tuple(card.box) for card in candidates[-1])
    if (count not in (4, 5) or not 0 <= selected_slot < count
            or len(frames) != 3 or len(candidates) != 3 or len(set(ids)) != 3
            or len(set(hand_instance_ids)) != count or len(predictions) != count
            or not generation or generation.get('hand_count') != count
            or any(len(row) != count or any(tuple(card.box[2:]) != old[2:]
                       or any(abs(card.box[axis]-old[axis]) > 1 for axis in (0,1))
                       for card,old in zip(row,boxes,strict=True))
                   or live_selection_geometry_evidence(row).get('raised_slots') != ()
                   for row in candidates)):
        raise ValueError('strong selection reference lacks the complete unselected A')
    sources = {}
    for slot, prediction in enumerate(predictions):
        if (slot == selected_slot or prediction.slot != slot or prediction.frame_id != ids[-1]
                or tuple(prediction.box) != boxes[slot] or not prediction.accepted
                or prediction.reason != 'accepted_visual_family' or prediction.recognizer != RECOGNIZER
                or not prediction.top_k_card_ids
                or not all(math.isfinite(value) and .85 <= value <= 1
                           for value in (prediction.confidence, prediction.raw_score))
                or not math.isfinite(prediction.margin) or not .05 <= prediction.margin <= 1):
            continue
        view = _view(frames[-1].shape, candidates[-1], slot, None)
        x, y, w, h = view.visible_box
        pixels = frames[-1][y:y+h, x:x+w].copy()
        pixels.flags.writeable = False
        sources[slot] = dict(view=view, pixels=pixels, prediction=deepcopy(prediction.to_dict()))
    return dict(run_id=run_id, revision=revision, selected_slot=selected_slot, boxes=boxes,
        frame_ids=ids, generation=deepcopy(generation), hand_instance_ids=tuple(hand_instance_ids), sources=sources)


def corroborate_strong_hand(reference, *, run_id, revision, hand_instance_ids, generation,
                            selected_slot, frames, candidates, predictions):
    """Reuse a current strong family or exact common artwork, without conflict.

    The caller dates input evidence from the matching frame, not the last frame.
    A weak ambiguous crop cannot overrule a qualified matching current frame.
    """
    try:
        ref = reference
        ids = tuple(frame_identifier(frame) for frame in frames)
        count = len(hand_instance_ids)
        if (not isinstance(ref, dict) or (run_id, revision, selected_slot) !=
                (ref['run_id'], ref['revision'], ref['selected_slot'])
                or generation != ref['generation'] or tuple(hand_instance_ids) != ref['hand_instance_ids']
                or len(frames) != 3 or len(candidates) != 3 or len(predictions) != 3
                or len(set(ids)) != 3 or set(ids) & set(ref['frame_ids'])):
            return []
        for frame, row, batch in zip(frames, candidates, predictions, strict=True):
            geometry = live_selection_geometry_evidence(row)
            if (frame.shape != (1280,720,3) or frame.dtype != np.uint8 or len(row) != count
                    or len(batch) != count or not geometry['available'] or geometry['ambiguous']
                    or geometry['raised_slots'] != (selected_slot,)):
                return []
            for slot, (old, card) in enumerate(zip(ref['boxes'], row, strict=True)):
                if slot == selected_slot:
                    if card.box[0] != old[0] or card.box[2:] != old[2:] or not 18 <= old[1]-card.box[1] <= 34:
                        return []
                elif (tuple(card.box[2:]) != old[2:]
                        or any(abs(card.box[axis]-old[axis]) > 1 for axis in (0,1))):
                    # Fitting the whole overlapped row may move any unselected
                    # box by one pixel on either axis. Artwork is compared at identical
                    # screen coordinates, never shifted to force a match.
                    return []
        proofs = []
        for slot, source in ref['sources'].items():
            old = source['prediction']
            family = tuple(old['top_k_card_ids'][0])
            batch = [items[slot] for items in predictions]
            from .hif_bound_title import in_candidate_band

            if any(p.slot != slot or p.frame_id != fid or tuple(p.box) != tuple(row[slot].box)
                   or not p.top_k_card_ids
                   or (tuple(p.top_k_card_ids[0]) != family and not in_candidate_band(p,family,.05))
                   or _model(p) != tuple(old[key] for key in ('recognizer','model_sha256','classes_sha256','gallery_sha256'))
                   or p.reason not in ('accepted_visual_family','below_acceptance_threshold','below_margin_threshold')
                   for p, fid, row in zip(batch, ids, candidates, strict=True)):
                continue
            views = [source['view']] + [_view(frame.shape,row,slot,selected_slot)
                for frame,row in zip(frames,candidates,strict=True)]
            box, inside = _common_region(views)
            x,y,w,h = box
            sx,sy,_,_ = source['view'].visible_box
            expected = source['pixels'][y-sy:y-sy+h,x-sx:x-sx+w][inside]
            matched = [fid for fid,frame in zip(ids,frames,strict=True)
                       if np.array_equal(frame[y:y+h,x:x+w][inside],expected)]
            current_identity = None
            independent_family = False
            if not matched and all(tuple(p.top_k_card_ids[0]) == family for p in batch):
                current_identity = next((p for p in reversed(batch) if p.accepted
                    and all(math.isfinite(v) and .85 <= v <= 1 for v in (p.confidence,p.raw_score))
                    and math.isfinite(p.margin) and .05 <= p.margin <= 1), None)
                if current_identity is not None:
                    # This frame independently reads the already bound family.
                    # Do not also require historical pixels to remain identical.
                    independent_family = True
                    matched = [current_identity.frame_id]
            if any(tuple(p.top_k_card_ids[0]) != family for p in batch):
                current_identity = next((p for p in reversed(batch)
                    if p.frame_id in matched and p.accepted
                    and tuple(p.top_k_card_ids[0]) == family
                    and all(math.isfinite(v) and .85 <= v <= 1 for v in (p.confidence,p.raw_score))
                    and math.isfinite(p.margin) and .05 <= p.margin <= 1), None)
                if current_identity is None:
                    continue
                # Use this actual current strong prediction, never relabel the
                # ambiguous last frame or borrow A's score for that frame.
                matched = [current_identity.frame_id]
            if matched:
                proofs.append(dict(accepted=True,scope='bound_selection_strong_common_art_v1',slot=slot,
                    selected_slot=selected_slot,run_id=run_id,revision=revision,
                    card_instance_id=hand_instance_ids[slot],family=family,
                    reference_frame_id=ref['frame_ids'][-1],frame_ids=ids,matching_frame_ids=matched,
                    source_prediction=deepcopy(old),fresh_predictions=[p.to_dict() for p in batch],
                    **(dict(matched_prediction_frame_id=current_identity.frame_id) if current_identity else {}),
                    common_screen_box=box,interior_pixel_count=int(inside.sum()),
                    pixel_comparison=('independent_current_exact_family' if independent_family
                        else 'exact_current_common_circle'),upgrade_verified=False,submission_authorized=False))
        return proofs
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return []


def _weak(prediction):
    return (prediction.accepted is False
        and prediction.reason in ('below_acceptance_threshold', 'below_margin_threshold')
        and prediction.recognizer == RECOGNIZER and bool(prediction.top_k_card_ids)
        and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in (prediction.confidence, prediction.margin))
        and (prediction.confidence < .85 or prediction.margin < .05))


def build_reference(*, run_id, revision, selected_slot, card_instance_id,
                    frames, candidates, prediction, inspection_reference=None,
                    generation, hand_instance_ids, title_reference=None, catalog=None):
    """Bind three current unselected A images; classify no extra frame.

    ``prediction`` is the actual raw neighbor prediction from A's last
    authoritative frame. The other two A images independently prove the
    receipt's fixed artwork. A title source belongs only to the authoritative
    last frame; all A/B crops must later agree in the same physical artwork
    region. Neither its OCR nor its prediction is copied to another frame.
    """
    receipt = inspection_reference
    if (not isinstance(run_id, str) or not run_id or type(revision) is not int
            or not isinstance(card_instance_id, str) or not card_instance_id
            or not isinstance(generation, dict) or not generation
            or type(selected_slot) is not int or len(frames) != 3 or len(candidates) != 3):
        raise ValueError('selected neighbor A lacks its current independent identity proof')
    title_source = title_reference is not None
    complete_full_source = False
    if title_source:
        title = title_reference
        family = tuple(title.get('family', ())) if isinstance(title, dict) else ()
        from .hif_bound_title import full_view_art_evidence_valid
        from .hif_inspection_neighbor import _source_title_valid

        complete_full_source = (isinstance(title, dict) and catalog is not None
            and full_view_art_evidence_valid(title.get('complete_full_art'), prediction, family, .05)
            and _source_title_valid(title, prediction, frame_id=frame_identifier(frames[-1]),
                                    family=family, catalog=catalog))
        if (catalog is None or not isinstance(title, dict)
                or (title.get('run_id'), title.get('revision'), title.get('card_instance_id')) !=
                   (run_id, revision, card_instance_id)
                or not (complete_full_source or _source_identity_valid(prediction,
                    frame_id=frame_identifier(frames[-1]), family=family, title_proof=title, catalog=catalog))
                or json.dumps(title.get('original_prediction'), sort_keys=True, allow_nan=False) !=
                   json.dumps(prediction.to_dict(), sort_keys=True, allow_nan=False)):
            raise ValueError('selected neighbor A lacks its authoritative complete-title proof')
    else:
        if (not isinstance(receipt, dict)
                or (receipt.get('run_id'), receipt.get('revision')) != (run_id, revision)
                or receipt.get('slot') != selected_slot+1
                or not independent_detail_reference_valid(receipt)):
            raise ValueError('selected neighbor A lacks its current independent inspection proof')
        family = tuple(receipt['family'])
    count, slot = len(candidates[0]), selected_slot+1
    ids = tuple(frame_identifier(frame) for frame in frames)
    boxes = tuple(tuple(card.box) for card in candidates[0])
    if (count not in (4, 5) or not 0 <= selected_slot < slot < count
            or len(set(ids)) != 3 or len(hand_instance_ids) != count
            or len(set(hand_instance_ids)) != count
            or any(not isinstance(value, str) or not value for value in hand_instance_ids)
            or hand_instance_ids[slot] != card_instance_id):
        raise ValueError('selected neighbor A requires one complete three-frame hand')
    sources = []
    for frame, row in zip(frames, candidates, strict=True):
        geometry = live_selection_geometry_evidence(row)
        row_boxes = tuple(tuple(card.box) for card in row)
        same_boxes = row_boxes == boxes
        if title_source and len(row_boxes) == count:
            # Accepted A1555 has one vertical detector pixel of jitter in an
            # unrelated gray slot. Keep the two proof slots exact, all x/size
            # values exact, and never translate any retained artwork pixels.
            same_boxes = all(current == old if i in (selected_slot, slot) else
                current[0] == old[0] and current[2:] == old[2:] and abs(current[1]-old[1]) <= 1
                for i, (old, current) in enumerate(zip(boxes, row_boxes, strict=True)))
        if (frame.shape != (1280, 720, 3) or frame.dtype != np.uint8
                or tuple(card.slot for card in row) != tuple(range(count))
                or not same_boxes
                or not geometry['available'] or geometry['ambiguous'] or geometry['raised_slots']
                or not title_source and visible_art(frame, row, slot,
                    visibility_profile=receipt['art'].get('visibility_profile')) != receipt['art']):
            raise ValueError('selected neighbor A geometry or independent current pixels differ')
        view = _view(frame.shape, row, slot, None)
        x, y, w, h = view.visible_box
        pixels = frame[y:y+h, x:x+w].copy()
        pixels.flags.writeable = False
        sources.append(dict(view=view, pixels=pixels))
    source = prediction
    model = _model(source)
    if (source.frame_id != ids[-1] or source.slot != slot
            or tuple(source.box) != boxes[slot] or not _weak(source)
            or not complete_full_source and tuple(source.top_k_card_ids[0]) != family
            or any(not isinstance(value, str) or not value for value in model)):
        raise ValueError('selected neighbor A raw prediction is not from its proved frame')
    reference = dict(scope=SCOPE, run_id=run_id, revision=revision, selected_slot=selected_slot,
        slot=slot, card_instance_id=card_instance_id, family=family, model=model,
        frame_ids=ids, boxes=boxes, sources=tuple(sources),
        generation=deepcopy(generation), hand_instance_ids=tuple(hand_instance_ids),
        source_prediction=deepcopy(source.to_dict()))
    if title_source:
        reference.update(title_source_frame_id=ids[-1], title_source_proof=deepcopy(title_reference))
    else:
        reference.update(inspection_selected_frame_ids=tuple(receipt['selected_frame_ids']),
            inspection_reference_frame_ids=tuple(receipt['reference_frame_ids']))
    return reference


def corroborate(reference, *, frames, candidates, prediction, run_id, revision,
                selected_slot, card_instance_id, generation, hand_instance_ids):
    """Return independent family evidence, leaving the raw ranking untouched."""
    ref = reference
    try:
        if (not isinstance(ref, dict) or ref.get('scope') != SCOPE
                or (run_id, revision, selected_slot, card_instance_id) !=
                   (ref['run_id'], ref['revision'], ref['selected_slot'], ref['card_instance_id'])
                or generation != ref['generation']
                or tuple(hand_instance_ids) != ref['hand_instance_ids']
                or len(frames) != 3 or len(candidates) != 3):
            return None
        slot = ref['slot']
        ids = tuple(frame_identifier(frame) for frame in frames)
        if (len(set(ids)) != 3 or set(ids) & set(ref['frame_ids'])
                or prediction.frame_id != ids[-1] or prediction.slot != slot
                or len(candidates[-1]) != len(ref['boxes'])
                or tuple(prediction.box) != tuple(candidates[-1][slot].box)
                or not _weak(prediction) or _model(prediction) != ref['model']
                or prediction.view_mode != 'left_occluded'):
            return None
        fresh_views = []
        for frame, row in zip(frames, candidates, strict=True):
            geometry = live_selection_geometry_evidence(row)
            if (frame.shape != (1280, 720, 3) or frame.dtype != np.uint8
                    or len(row) != len(ref['boxes'])
                    or tuple(card.slot for card in row) != tuple(range(len(row)))
                    or tuple(tuple(card.box) for card in row) != tuple(tuple(card.box) for card in candidates[-1])
                    or not geometry['available'] or geometry['ambiguous']
                    or tuple(geometry['raised_slots']) != (selected_slot,)):
                return None
            for i, (old, card) in enumerate(zip(ref['boxes'], row, strict=True)):
                box = tuple(card.box)
                unrelated_y_tolerance = 1 if 'title_source_proof' in ref and i not in (selected_slot, slot) else 0
                if (box[2:] != old[2:] or abs(box[0]-old[0]) > 8
                        or (not 18 <= old[1]-box[1] <= 34 if i == selected_slot
                            else abs(box[1]-old[1]) > unrelated_y_tolerance)):
                    return None
            fresh_views.append(_view(frame.shape, row, slot, selected_slot))
        views = [source['view'] for source in ref['sources']] + fresh_views
        # Keep the same physical artwork origin. No pixel alignment, resize,
        # translation search or crop selected from the observed differences.
        origins = {(v.visible_box[0]-v.canvas_offset[0], v.visible_box[1]) for v in views}
        if len(origins) != 1:
            # 1992: the actual neighbor body stays at the exact same box,
            # while fitting the other cards shifts its reconstructed origin
            # by two pixels. Compare only the fixed screen intersection; do
            # not move, align, resize, or search any pixels to obtain a match.
            fixed_complete_body = (ref.get('title_source_proof', {}).get('complete_full_art') is not None
                and all(tuple(row[slot].box) == ref['boxes'][slot] for row in candidates)
                and len({y for _, y in origins}) == 1
                and max(x for x, _ in origins)-min(x for x, _ in origins) <= 2)
            if not fixed_complete_body:
                return None
        box, inside = _common_region(views)
        left, top, width, height = box
        for frame in frames:
            current = frame[top:top+height, left:left+width][inside]
            for source in ref['sources']:
                x, y, _, _ = source['view'].visible_box
                old = source['pixels'][top-y:top-y+height, left-x:left-x+width]
                if old.shape[:2] != inside.shape or not np.array_equal(old[inside], current):
                    return None
        proof = dict(accepted=True, scope=SCOPE, run_id=run_id, revision=revision,
            selected_slot=selected_slot, slot=slot, card_instance_id=card_instance_id,
            family=ref['family'], frame_id=ids[-1], frame_ids=ids,
            reference_frame_ids=ref['frame_ids'], generation=deepcopy(ref['generation']),
            hand_instance_ids=ref['hand_instance_ids'],
            source_prediction=deepcopy(ref['source_prediction']),
            original_prediction=prediction.to_dict(), common_screen_box=box,
            interior_pixel_count=int(inside.sum()),
            pixel_comparison='exact_all_current_a_circle_interiors',
            upgrade_verified=False, input_authorized=False, submission_authorized=False)
        if 'title_source_proof' in ref:
            proof.update(title_source_frame_id=ref['title_source_frame_id'],
                title_source_proof=deepcopy(ref['title_source_proof']))
        else:
            proof.update(inspection_selected_frame_ids=ref['inspection_selected_frame_ids'],
                inspection_reference_frame_ids=ref['inspection_reference_frame_ids'])
        return proof
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
